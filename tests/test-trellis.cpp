#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// nari fork: numeric tests for the trellis types TRL1..TRL6, GGML_OP_TRELLIS and GGML_OP_TRELLIS_GET_ROWS.
// Every coded product is compared against a float64 matmul with the dense weight that ggml_trellis_dequantize
// produces (the host reference), and --dump writes the same inputs so tests/trellis-reference-check.py can
// compare the op against nari's shipped reference decoder on a real GGUF.

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static bool verbose = false;

// per-type worst absolute error over all cases, and the worst tolerance-relative error
static double max_abs[6];
static double max_rel[6];

// The reference is a float64 product of the dense weight, so the CPU path matches it to fp32 rounding. A tensor-core
// backend multiplies f16 operands (the codes decode to f16, the activations convert to f16), and a coded linear sums K
// terms that largely cancel, so its worst relative error over the ~10k checked values reaches a few 1e-3: 2e-2 here.
// A wrong tile, rotation, scale or stage is off by O(1), which is how the sm_75 race this test caught showed up.
static double tolerance = 2e-4;

// A coded embedding is read on the host: the GPU backends decline GGML_OP_TRELLIS_GET_ROWS (their coded product is a
// linear only), so the scheduler leaves the lookup on the CPU, and the embedding checks always run there.
static ggml_backend_t host_backend = nullptr;
static int skipped = 0;
static int mismatches = 0;

static void check_close(double actual, double expected, int bits, const char * what) {
    const double abs_err = std::abs(actual - expected);
    const double rel_err = abs_err / (1.0 + std::abs(expected));
    if (abs_err > max_abs[bits - 1]) {
        max_abs[bits - 1] = abs_err;
    }
    if (rel_err > max_rel[bits - 1]) {
        max_rel[bits - 1] = rel_err;
    }
    if (!std::isfinite(actual) || rel_err >= tolerance) {
        ++mismatches;
        if (mismatches == 1) {
            std::fprintf(stderr, "TRL%d %s: got %.9g expected %.9g (abs %.3g)\n", bits, what, actual, expected, abs_err);
        }
    }
}

struct case_desc {
    int k, n;
    int32_t desc[9];
};

// (K, N) and the descriptor of each case: identity, blockwise Sylvester, whole-slice H_m for m = 1, 12, 20, 68
static case_desc make_case(int mode) {
    case_desc c{};
    switch (mode) {
        case 0: c.k = 256; c.n = 128; { const int32_t d[] = {0, 0, 1, 0, 0, 1, 0}; std::memcpy(c.desc, d, sizeof(d)); } break;
        case 1: c.k = 256; c.n = 128; { const int32_t d[] = {1, 128, 1, 1, 128, 1, 0}; std::memcpy(c.desc, d, sizeof(d)); } break;
        case 2: c.k = 256; c.n = 128; { const int32_t d[] = {1, 128, 1, 1, 128, 1, 0}; std::memcpy(c.desc, d, sizeof(d)); } break;
        case 3: c.k = 256; c.n = 128; { const int32_t d[] = {2, 0, 1, 2, 0, 1, 2, 64, 64}; std::memcpy(c.desc, d, sizeof(d)); } break;
        case 4: c.k = 256; c.n = 128; { const int32_t d[] = {2, 0, 2, 2, 0, 2, 0}; std::memcpy(c.desc, d, sizeof(d)); } break;
        case 5: c.k = 192; c.n = 192; { const int32_t d[] = {2, 0, 1, 2, 0, 1, 0}; std::memcpy(c.desc, d, sizeof(d)); } break;   // m = 12
        case 6: c.k = 320; c.n = 320; { const int32_t d[] = {2, 0, 1, 2, 0, 1, 0}; std::memcpy(c.desc, d, sizeof(d)); } break;   // m = 20
        case 7: c.k = 272; c.n = 272; { const int32_t d[] = {2, 0, 1, 2, 0, 1, 0}; std::memcpy(c.desc, d, sizeof(d)); } break;   // m = 68
        default: check(false, "bad mode");
    }
    return c;
}

static int desc_len(const case_desc & c) { return 7 + c.desc[6]; }

static const char * mode_name(int mode) {
    switch (mode) {
        case 0: return "identity";
        case 1: return "rht128";
        case 2: return "rht128+perm";
        case 3: return "hadfull+parts";
        case 4: return "hadfull+shards";
        case 5: return "hadfull+H12";
        case 6: return "hadfull+H20";
        case 7: return "hadfull+H68";
        default: return "?";
    }
}

// the op against the host reference product; returns the worst absolute error of this case
static double run(ggml_backend_t backend, int bits, int mode, int columns, const char * dump) {
    const case_desc c = make_case(mode);
    const int K = c.k, N = c.n;
    auto * ctx = ggml_init({64 * 1024 * 1024, nullptr, false});
    check(ctx != nullptr, "context allocation failed");
    const auto type = static_cast<ggml_type>(GGML_TYPE_TRL1 + bits - 1);
    auto * w = ggml_new_tensor_2d(ctx, type, K, N);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, columns);
    auto * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, K / 16);
    auto * pre = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    auto * post = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto * pin = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    auto * pout = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    // every bit pattern is a valid code: random codes are the coded weights, decoded exactly by the reference
    std::mt19937 rng(1234 + bits);
    for (size_t i = 0; i < ggml_nbytes(w); ++i) { static_cast<unsigned char *>(w->data)[i] = rng() & 255; }
    for (int i = 0; i < K * columns; ++i) { static_cast<float *>(x->data)[i] = float(int(rng() % 101) - 50) / 100; }
    for (int i = 0; i < K / 16; ++i) { static_cast<ggml_fp16_t *>(s->data)[i] = ggml_fp32_to_fp16(0.1f + (i % 7) * 0.01f); }
    for (int i = 0; i < K; ++i) {
        static_cast<float *>(pre->data)[i] = (i % 2 ? -1.f : 1.f) * (1.f + (i % 3) * 0.1f);
        static_cast<int32_t *>(pin->data)[i] = K - 1 - i;
    }
    for (int i = 0; i < N; ++i) {
        static_cast<float *>(post->data)[i] = 0.8f + (i % 5) * 0.1f;
        static_cast<int32_t *>(pout->data)[i] = N - 1 - i;
    }
    const bool permute = mode == 2;
    auto * ip = permute ? pin : nullptr;
    auto * op = permute ? pout : nullptr;
    const int nd = desc_len(c);
    auto * y = ggml_trellis_mul_mat(ctx, w, x, s, pre, post, ip, op, c.desc, nd);
    if (!ggml_backend_supports_op(backend, y)) {
        // The GPU coded product needs K a multiple of 128, which the H12/H20/H68 descriptors of K=192, 320 and 272 are
        // not; a real model always is. The case is skipped there and still checked on the CPU.
        ggml_free(ctx);
        if (backend == host_backend) { check(false, "the CPU backend rejected a valid coded product"); }
        ++skipped;
        return 0.0;
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "trellis graph failed");

    std::vector<float> dense(size_t(K) * N);
    check(ggml_trellis_dequantize(type, w->data, K, N, static_cast<ggml_fp16_t *>(s->data),
        static_cast<float *>(pre->data), static_cast<float *>(post->data),
        ip ? static_cast<int32_t *>(ip->data) : nullptr, op ? static_cast<int32_t *>(op->data) : nullptr,
        c.desc, nd, dense.data()), "dequantize failed");
    double worst = 0;
    for (int t = 0; t < columns; ++t) {
        for (int n = 0; n < N; ++n) {
            double expected = 0;
            for (int k = 0; k < K; ++k) { expected += double(dense[size_t(n) * K + k]) * static_cast<float *>(x->data)[t * K + k]; }
            const double got = static_cast<float *>(y->data)[t * N + n];
            check_close(got, expected, bits, "mul_mat");
            worst = std::max(worst, std::abs(got - expected));
        }
    }
    if (verbose) {
        std::printf("  TRL%d %-16s K=%3d N=%3d T=%2d  max|d|=%9.3g\n", bits, mode_name(mode), K, N, columns, worst);
    }

    // embeddings: rows unrotated, pre/post applied, plus a chained band with a vocab gather
    case_desc e = c;
    e.desc[0] = e.desc[1] = e.desc[2] = 0;
    check(ggml_trellis_dequantize(type, w->data, K, N, static_cast<ggml_fp16_t *>(s->data),
        static_cast<float *>(pre->data), static_cast<float *>(post->data), nullptr, nullptr,
        e.desc, desc_len(e), dense.data()), "embedding decode failed");
    const int rows[4] = {0, 17 % N, N - 1, 32 % N};
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    for (int i = 0; i < 4; ++i) { static_cast<int32_t *>(ids->data)[i] = rows[i]; }
    auto * emb = ggml_trellis_get_rows(ctx, w, ids, s, pre, post, e.desc, desc_len(e));
    graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, emb);
    ggml_backend_t eb = host_backend != nullptr ? host_backend : backend;
    check(ggml_backend_graph_compute(eb, graph) == GGML_STATUS_SUCCESS, "embedding graph failed");
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < K; ++k) {
            check_close(static_cast<float *>(emb->data)[i * K + k], dense[size_t(rows[i]) * K + k], bits, "get_rows");
        }
    }
    auto * gather = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2 * N);
    auto * post2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    for (int i = 0; i < N; ++i) { static_cast<float *>(post2->data)[i] = -static_cast<float *>(post->data)[i]; }
    for (int i = 0; i < 2 * N; ++i) { static_cast<int32_t *>(gather->data)[i] = 2 * N - 1 - i; }
    const int32_t tokens[4] = {0, 17 % N, N + 17 % N, 2 * N - 1};
    for (int i = 0; i < 4; ++i) { static_cast<int32_t *>(ids->data)[i] = tokens[i]; }
    auto * band = ggml_trellis_get_rows_band(ctx, nullptr, w, ids, gather, 0, s, pre, post, e.desc, desc_len(e));
    band = ggml_trellis_get_rows_band(ctx, band, w, ids, gather, N, s, pre, post2, e.desc, desc_len(e));
    graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, band);
    check(ggml_backend_graph_compute(eb, graph) == GGML_STATUS_SUCCESS, "banded embedding graph failed");
    for (int i = 0; i < 4; ++i) {
        const int row = 2 * N - 1 - tokens[i];
        for (int k = 0; k < K; ++k) {
            const double expected = (row >= N ? -1.0 : 1.0) * dense[size_t(row % N) * K + k];
            check_close(static_cast<float *>(band->data)[i * K + k], expected, bits, "band");
        }
    }

    if (dump != nullptr) {  // the same inputs, for tests/trellis-reference-check.py
        char path[1024];
        std::snprintf(path, sizeof(path), "%s/trl%d_mode%d.raw", dump, bits, mode);
        FILE * f = std::fopen(path, "wb");
        check(f != nullptr, "cannot write dump");
        const int32_t hdr[] = {bits, K, N, columns, nd};
        std::fwrite(hdr, sizeof(int32_t), 5, f);
        std::fwrite(c.desc, sizeof(int32_t), nd, f);
        std::fwrite(w->data, 1, ggml_nbytes(w), f);
        std::fwrite(s->data, sizeof(ggml_fp16_t), K / 16, f);
        std::fwrite(pre->data, sizeof(float), K, f);
        std::fwrite(post->data, sizeof(float), N, f);
        if (ip != nullptr) { std::fwrite(ip->data, sizeof(int32_t), K, f); }
        if (op != nullptr) { std::fwrite(op->data, sizeof(int32_t), N, f); }
        std::fwrite(x->data, sizeof(float), size_t(K) * columns, f);
        std::fwrite(y->data, sizeof(float), size_t(N) * columns, f);
        std::fclose(f);
    }
    ggml_free(ctx);
    return worst;
}

int main(int argc, char ** argv) {
    const char * dump = nullptr;
    std::string backend_name;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v") {
            verbose = true;
        } else if (std::string(argv[i]) == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (std::string(argv[i]) == "--tol" && i + 1 < argc) {
            tolerance = std::atof(argv[++i]);
        } else if (std::string(argv[i]) == "--dump" && i + 1 < argc) {
            dump = argv[++i];
        } else {
            check(false, "usage: test-trellis [-v] [--dump DIR] [--backend NAME]");
        }
    }
    ggml_backend_load_all();
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr, "CPU backend unavailable");
    host_backend = cpu;
    if (!backend_name.empty() && tolerance == 2e-4) {
        tolerance = 2e-2;  // f16 operands, see above
    }
    // the numeric checks run on --backend when one is named (e.g. CUDA0), on the CPU otherwise
    ggml_backend_t be = cpu;
    if (!backend_name.empty()) {
        be = ggml_backend_init_by_name(backend_name.c_str(), nullptr);
        check(be != nullptr, "the named backend is unavailable");
        std::fprintf(stderr, "checking the coded ops on %s\n", ggml_backend_name(be));
    }

    // descriptor validation rejects what the format does not allow
    const int32_t bad_block[] = {1, 3, 1, 0, 0, 1, 0};
    check(!ggml_trellis_desc_supported(256, 128, bad_block, 7), "invalid Hadamard block accepted");
    check(!ggml_trellis_desc_supported(256, 128, nullptr, 0), "missing descriptor accepted");
    check(!ggml_trellis_desc_supported(250, 128, bad_block, 7), "K not a multiple of 16 accepted");

    {
        // a coded weight has no per-row decode or dot product: only the trellis ops may read it
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TRL2, 256, 128);
        auto * e = ggml_new_tensor_3d(ctx, GGML_TYPE_TRL2, 256, 128, 2);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        auto * eids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 4);
        ggml_tensor * ops[] = {
            ggml_get_rows(ctx, w, ids),
            ggml_mul_mat(ctx, w, x),
            ggml_mul_mat_id(ctx, e, ggml_reshape_3d(ctx, x, 256, 1, 4), eids),
        };
        for (ggml_tensor * op : ops) {
            if (ggml_backend_supports_op(cpu, op)) {
                std::fprintf(stderr, "CPU backend accepts %s on a coded weight\n", ggml_op_name(op->op));
                std::exit(1);
            }
        }
        check(ggml_get_type_traits(GGML_TYPE_TRL1)->to_float == nullptr, "trellis type has a per-row decode");
        check(ggml_get_type_traits(GGML_TYPE_TRL6)->type_size == 12, "TRL6 block size wrong");
        check(ggml_is_trellis_type(GGML_TYPE_TRL1) && ggml_is_trellis_type(GGML_TYPE_TRL6), "range test broken");
        check(!ggml_is_trellis_type(GGML_TYPE_Q1_T_g128) && !ggml_is_trellis_type(GGML_TYPE_Q1_R_g128),
              "range test claims Q1_T/Q1_R are trellis");
        ggml_free(ctx);
    }

    for (int bits = 1; bits <= 6; ++bits) {
        for (int mode = 0; mode < 8; ++mode) {
            for (int columns : {1, 7, 96}) {  // 96 crosses the GEMV/GEMM split at 64 columns (prefill)
                run(be, bits, mode, columns, nullptr);
            }
        }
        if (dump != nullptr) {
            run(be, bits, 2, 3, dump);
            run(be, bits, 0, 3, dump);
        }
    }

    for (int bits = 1; bits <= 6; ++bits) {
        std::printf("TRL%d: max abs error %.3g, max relative error %.3g\n", bits, max_abs[bits - 1], max_rel[bits - 1]);
    }
    if (mismatches > 0) {
        std::fprintf(stderr, "%d coded values are off by more than %.3g (relative)\n", mismatches, tolerance);
        std::exit(1);
    }
    if (skipped > 0) {
        std::printf("%d cases skipped by %s (K not a multiple of 128)\n", skipped, ggml_backend_name(be));
    }
    std::puts("TRL1-6: coded products, embeddings and bands match the host reference");
    if (be != cpu) {
        ggml_backend_free(be);
    }
    ggml_backend_free(cpu);
    return 0;
}
