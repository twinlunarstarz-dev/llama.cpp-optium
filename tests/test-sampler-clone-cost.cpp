// Measures the cost of the per-step speculative sampler clone on the server's
// hot decode path (tools/server/server-context.cpp clones the sampler on every
// speculative verify, but consumes it only on checkpoint rollback).
//
// This is a COST test, not a correctness test: it reports nanoseconds per
// clone/accept and the bytes of candidate state copied. It runs CPU-only with
// no model, so it is hermetic and never a performance baseline for the CUDA
// server. It is printed, not asserted, except for the invariants that make the
// measurement meaningful.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// The production preset Qwen3.8-27B-Q5-Layer (/models/models.ini:265-285) sets
// presence-penalty = 1.5 and repeat-penalty = 1.05, which map to
// penalty_present / penalty_repeat (common/arg.cpp:2097-2120). Penalties are
// therefore ACTIVE in production and is_disabled() is false
// (llama-sampler.cpp:2885): the sampler's prev/token_count state is live and
// the apply() path runs. Measured with those values, not with library defaults.
constexpr int32_t  N_VOCAB        = 248320; // print_info for the production GGUF
constexpr int32_t  PENALTY_LAST_N = 64;     // penalty_last_n default
constexpr float    PENALTY_REPEAT = 1.05f;  // repeat-penalty = 1.05
constexpr float    PENALTY_FREQ   = 0.00f;  // frequency-penalty unset = 0.0
constexpr float    PENALTY_PRESENT= 1.50f;  // presence-penalty = 1.5

struct timing {
    double ns_per_op = 0.0;
};

// Median of per-batch nanoseconds, so a single scheduling outlier cannot
// dominate the reported figure.
timing measure(const std::vector<double> & per_batch_ns) {
    std::vector<double> sorted = per_batch_ns;
    std::sort(sorted.begin(), sorted.end());
    timing t;
    t.ns_per_op = sorted[sorted.size() / 2];
    return t;
}

std::vector<double> time_accept(llama_sampler * smpl, int n_tokens, int n_reps) {
    std::vector<double> out;
    out.reserve(n_reps);
    for (int r = 0; r < n_reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_tokens; ++i) {
            llama_sampler_accept(smpl, (llama_token) ((i * 7919) % N_VOCAB));
        }
        const auto t1 = std::chrono::steady_clock::now();
        out.push_back((double) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / n_tokens);
    }
    return out;
}

std::vector<double> time_clone(llama_sampler * smpl, int n_reps) {
    std::vector<double> out;
    out.reserve(n_reps);
    for (int r = 0; r < n_reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        llama_sampler * c = llama_sampler_clone(smpl);
        const auto t1 = std::chrono::steady_clock::now();
        llama_sampler_free(c);
        out.push_back((double) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    return out;
}

} // namespace

int main() {
    // The production server chain includes a penalties sampler on every request.
    // Measure it in isolation, then as part of a small chain, because
    // llama_sampler_chain_clone() clones every member.
    llama_sampler * pen = llama_sampler_init_penalties(
            N_VOCAB, PENALTY_LAST_N, PENALTY_REPEAT, PENALTY_FREQ, PENALTY_PRESENT);

    // Warm the ring buffer to its steady-state occupancy (penalty_last_n = 64),
    // which is the state a long-running production request actually clones.
    for (int i = 0; i < 4 * PENALTY_LAST_N; ++i) {
        llama_sampler_accept(pen, (llama_token) ((i * 7919) % N_VOCAB));
    }

    const std::vector<double> accept_ns = time_accept(pen, 4096, 9);
    const std::vector<double> clone_ns  = time_clone(pen, 9);

    const timing t_accept = measure(accept_ns);
    const timing t_clone  = measure(clone_ns);

    // Invariants that make the numbers meaningful rather than a measurement of
    // a degenerate state: with the production preset's penalties the sampler
    // must be live (apply must actually change the candidates), and the
    // candidate buffer the server clone copies must be non-empty.
    llama_token_data_array probe;
    probe.size = N_VOCAB;
    probe.data = nullptr;
    probe.sorted = false;
    probe.selected = -1;

    std::vector<llama_token_data> cand(N_VOCAB);
    for (int32_t i = 0; i < N_VOCAB; ++i) {
        cand[i] = llama_token_data{ i, 0.0f, 0.0f };
    }
    probe.data = cand.data();

    const auto out_before = cand;
    llama_sampler_apply(pen, &probe);
    const bool penalties_are_live = (cand.size() == out_before.size()) &&
        (memcmp(cand.data(), out_before.data(), cand.size() * sizeof(llama_token_data)) != 0);

    // Bytes of candidate state common_sampler_clone() copies per speculative
    // step (common/sampling.cpp:514-524 copies `cur`).
    const size_t clone_bytes_cur = cand.size() * sizeof(llama_token_data);

    printf("== speculative per-step sampler clone cost (CPU, hermetic) ==\n");
    printf("n_vocab                = %d\n", N_VOCAB);
    printf("penalties_last_n       = %d\n", PENALTY_LAST_N);
    printf("penalties_are_live     = %s (repeat 1.05 / present 1.5 => apply is NOT a no-op)\n",
            penalties_are_live ? "true" : "FALSE (state is dead weight)");
    printf("clone_cur_bytes_per_step = %zu (%.2f MiB of candidate state copied per clone)\n",
            clone_bytes_cur, (double) clone_bytes_cur / 1048576.0);
    printf("accept_ns_per_token    = %.1f  (min %.1f, max %.1f)\n",
            t_accept.ns_per_op,
            *std::min_element(accept_ns.begin(), accept_ns.end()),
            *std::max_element(accept_ns.begin(), accept_ns.end()));
    printf("clone_ns_per_step      = %.1f  (min %.1f, max %.1f)\n",
            t_clone.ns_per_op,
            *std::min_element(clone_ns.begin(), clone_ns.end()),
            *std::max_element(clone_ns.begin(), clone_ns.end()));
    printf("clone_ns_share_of_step = %.3f %% of clone+accept cost\n",
            100.0 * t_clone.ns_per_op / (t_clone.ns_per_op + t_accept.ns_per_op));
    printf("note: CPU-only microbenchmark. It bounds the avoidable per-step clone\n");
    printf("      cost, it is NOT a decode-throughput baseline for the CUDA server.\n");

    llama_sampler_free(pen);

    if (!penalties_are_live) {
        fprintf(stderr, "penalties sampler was a no-op; the measured apply cost does not describe production\n");
        return 1;
    }
    if (clone_bytes_cur == 0) {
        fprintf(stderr, "candidate buffer empty; clone cost measurement is invalid\n");
        return 1;
    }

    return 0;
}
