#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string messages;

static void log_message(ggml_log_level, const char * text, void *) {
    messages += text;
}

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n%s", message, messages.c_str());
        std::exit(1);
    }
}

// loads meta (with its tensors' data) on the CPU; the log of the load is in messages
static llama_model * load(gguf_context * meta, bool load_mtp = false) {
    auto * file = std::tmpfile();
    check(file != nullptr, "tmpfile failed");
    check(gguf_write_to_file_ptr(meta, file, false), "write GGUF failed");
    std::rewind(file);
    messages.clear();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.load_mtp = load_mtp;
    auto * model = llama_model_load_from_file_ptr(file, params);
    std::fclose(file);
    return model;
}

static void check_versions() {
    for (int version : {-1, 0, 1, 2, 3}) {
        auto * ctx = ggml_init({1024 * 1024, nullptr, false});
        auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TRL1, 256, 16);
        ggml_set_name(w, "token_embd.weight");
        auto * meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "llama");
        if (version == 3) {
            gguf_set_val_str(meta, "trellis.version", "1");
        } else if (version >= 0) {
            gguf_set_val_u32(meta, "trellis.version", version);
        }
        gguf_add_tensor(meta, w);
        auto * model = load(meta);
        check(model == nullptr, "incomplete fixture must not load as a model");
        const bool rejected_version = messages.find("trellis: coded tensors require trellis.version") != std::string::npos;
        check(rejected_version == (version != 1), "unexpected format version validation result");
        gguf_free(meta);
        ggml_free(ctx);
    }
}

// a GGUF with trellis.version 1 and its tensors' data
struct builder {
    ggml_context * ctx;
    gguf_context * meta;

    builder() {
        ctx = ggml_init({64 * 1024 * 1024, nullptr, false});
        meta = gguf_init_empty();
        gguf_set_val_u32(meta, "trellis.version", 1);
    }

    ~builder() {
        gguf_free(meta);
        ggml_free(ctx);
    }

    ggml_tensor * add(const std::string & name, ggml_type type, const std::vector<int64_t> & ne) {
        auto * t = ggml_new_tensor(ctx, type, (int) ne.size(), ne.data());
        ggml_set_name(t, name.c_str());
        std::memset(t->data, 0, ggml_nbytes(t));
        if (type == GGML_TYPE_F32 && ne.size() == 1) {
            for (int64_t i = 0; i < ne[0]; ++i) {
                ((float *) t->data)[i] = 1.0f;
            }
        }
        gguf_add_tensor(meta, t);
        return t;
    }

    // a coded weight base.weight [k, n] with its sides and no rotations; pin_type GGML_TYPE_COUNT: no input gather
    void coded(const std::string & base, int64_t k, int64_t n, ggml_type pin_type) {
        add(base + ".weight", GGML_TYPE_TRL2, {k, n});
        add(base + ".trellis_s16", GGML_TYPE_F16, {k / 16});
        add(base + ".trellis_pre", GGML_TYPE_F32, {k});
        add(base + ".trellis_post", GGML_TYPE_F32, {n});
        if (pin_type == GGML_TYPE_I32) {
            auto * pin = add(base + ".trellis_pin", GGML_TYPE_I32, {k});
            for (int64_t i = 0; i < k; ++i) {
                ((int32_t *) pin->data)[i] = (int32_t) (k - 1 - i);
            }
        } else if (pin_type != GGML_TYPE_COUNT) {
            add(base + ".trellis_pin", pin_type, {k});
        }
        const int32_t desc[] = {0, 0, 1, 0, 0, 1, 1, (int32_t) n};
        gguf_set_arr_data(meta, ("trellis.desc." + base).c_str(), GGUF_TYPE_INT32, desc, 8);
    }
};

// a one-layer llama (n_embd 256, n_vocab 32) with a coded output.weight, or a coded token_embd.weight that is also
// the output (tied); moe: coded expert weights
struct fixture : builder {
    static constexpr int64_t n_embd = 256, n_vocab = 32, n_expert = 2;

    fixture(bool tied, bool moe, ggml_type pin_type = GGML_TYPE_I32, int64_t vocab = n_vocab) {
        gguf_set_val_str(meta, "general.architecture", "llama");
        gguf_set_val_u32(meta, "llama.context_length", 64);
        gguf_set_val_u32(meta, "llama.embedding_length", n_embd);
        gguf_set_val_u32(meta, "llama.block_count", 1);
        gguf_set_val_u32(meta, "llama.feed_forward_length", n_embd);
        gguf_set_val_u32(meta, "llama.attention.head_count", 2);
        gguf_set_val_u32(meta, "llama.attention.head_count_kv", 2);
        gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_u32(meta, "llama.vocab_size", vocab);
        gguf_set_val_str(meta, "tokenizer.ggml.model", "none");

        if (tied) {
            coded("token_embd", n_embd, vocab, GGML_TYPE_COUNT);
        } else {
            add("token_embd.weight", GGML_TYPE_F32, {n_embd, vocab});
            coded("output", n_embd, vocab, pin_type);
        }
        add("output_norm.weight", GGML_TYPE_F32, {n_embd});
        add("blk.0.attn_norm.weight", GGML_TYPE_F32, {n_embd});
        for (const char * name : {"attn_q", "attn_k", "attn_v", "attn_output"}) {
            add(std::string("blk.0.") + name + ".weight", GGML_TYPE_F32, {n_embd, n_embd});
        }
        add("blk.0.ffn_norm.weight", GGML_TYPE_F32, {n_embd});
        if (moe) {
            gguf_set_val_u32(meta, "llama.expert_count", n_expert);
            gguf_set_val_u32(meta, "llama.expert_used_count", 1);
            add("blk.0.ffn_gate_inp.weight", GGML_TYPE_F32, {n_embd, n_expert});
            for (const char * name : {"ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"}) {
                add(std::string("blk.0.") + name + ".weight", GGML_TYPE_TRL2, {n_embd, n_embd, n_expert});
            }
        } else {
            for (const char * name : {"ffn_gate", "ffn_down", "ffn_up"}) {
                add(std::string("blk.0.") + name + ".weight", GGML_TYPE_F32, {n_embd, n_embd});
            }
        }
    }

    int32_t * pin() {
        return (int32_t *) ggml_get_tensor(ctx, "output.trellis_pin")->data;
    }
};

// a qwen35 with one recurrent layer and an MTP block (n_embd 256, n_vocab 32), its token_embd.weight coded;
// nextn_embd: the MTP block has its own plain embedding
struct mtp_fixture : builder {
    static constexpr int64_t n_embd = 256, n_vocab = 32, n_head = 2, n_head_dim = 128, n_ssm = 16;

    mtp_fixture(bool nextn_embd) {
        gguf_set_val_str(meta, "general.architecture", "qwen35");
        gguf_set_val_u32(meta, "qwen35.context_length", 64);
        gguf_set_val_u32(meta, "qwen35.embedding_length", n_embd);
        gguf_set_val_u32(meta, "qwen35.block_count", 2);
        gguf_set_val_u32(meta, "qwen35.nextn_predict_layers", 1);
        gguf_set_val_u32(meta, "qwen35.feed_forward_length", n_embd);
        gguf_set_val_u32(meta, "qwen35.attention.head_count", n_head);
        gguf_set_val_u32(meta, "qwen35.attention.head_count_kv", n_head);
        gguf_set_val_u32(meta, "qwen35.attention.key_length", n_head_dim);
        gguf_set_val_u32(meta, "qwen35.attention.value_length", n_head_dim);
        gguf_set_val_f32(meta, "qwen35.attention.layer_norm_rms_epsilon", 1e-5f);
        const int32_t sections[4] = {16, 16, 16, 0};
        gguf_set_arr_data(meta, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
        gguf_set_val_u32(meta, "qwen35.ssm.conv_kernel", 4);
        gguf_set_val_u32(meta, "qwen35.ssm.inner_size", n_ssm);
        gguf_set_val_u32(meta, "qwen35.ssm.state_size", n_ssm);
        gguf_set_val_u32(meta, "qwen35.ssm.time_step_rank", 1);
        gguf_set_val_u32(meta, "qwen35.ssm.group_count", 1);
        gguf_set_val_u32(meta, "qwen35.full_attention_interval", 4);
        gguf_set_val_u32(meta, "qwen35.vocab_size", n_vocab);
        gguf_set_val_str(meta, "tokenizer.ggml.model", "none");

        coded("token_embd", n_embd, n_vocab, GGML_TYPE_COUNT);
        add("output_norm.weight", GGML_TYPE_F32, {n_embd});
        add("output.weight", GGML_TYPE_F32, {n_embd, n_vocab});

        // blk.0: gated delta net (one key and one value head of n_ssm)
        for (const char * name : {"attn_norm", "post_attention_norm"}) {
            add(std::string("blk.0.") + name + ".weight", GGML_TYPE_F32, {n_embd});
        }
        add("blk.0.attn_qkv.weight", GGML_TYPE_F32, {n_embd, 3 * n_ssm});
        add("blk.0.attn_gate.weight", GGML_TYPE_F32, {n_embd, n_ssm});
        add("blk.0.ssm_conv1d.weight", GGML_TYPE_F32, {4, 3 * n_ssm});
        add("blk.0.ssm_dt.bias", GGML_TYPE_F32, {1});
        add("blk.0.ssm_a", GGML_TYPE_F32, {1});
        add("blk.0.ssm_beta.weight", GGML_TYPE_F32, {n_embd, 1});
        add("blk.0.ssm_alpha.weight", GGML_TYPE_F32, {n_embd, 1});
        add("blk.0.ssm_norm.weight", GGML_TYPE_F32, {n_ssm});
        add("blk.0.ssm_out.weight", GGML_TYPE_F32, {n_ssm, n_embd});

        // blk.1: the MTP block
        for (const char * name : {"attn_norm", "post_attention_norm", "nextn.enorm", "nextn.hnorm"}) {
            add(std::string("blk.1.") + name + ".weight", GGML_TYPE_F32, {n_embd});
        }
        add("blk.1.attn_q.weight", GGML_TYPE_F32, {n_embd, 2 * n_head * n_head_dim});
        add("blk.1.attn_k.weight", GGML_TYPE_F32, {n_embd, n_head * n_head_dim});
        add("blk.1.attn_v.weight", GGML_TYPE_F32, {n_embd, n_head * n_head_dim});
        add("blk.1.attn_output.weight", GGML_TYPE_F32, {n_head * n_head_dim, n_embd});
        add("blk.1.attn_q_norm.weight", GGML_TYPE_F32, {n_head_dim});
        add("blk.1.attn_k_norm.weight", GGML_TYPE_F32, {n_head_dim});
        add("blk.1.nextn.eh_proj.weight", GGML_TYPE_F32, {2 * n_embd, n_embd});
        if (nextn_embd) {
            add("blk.1.nextn.embed_tokens.weight", GGML_TYPE_F32, {n_embd, n_vocab});
        }

        for (int il : {0, 1}) {
            for (const char * name : {"ffn_gate", "ffn_down", "ffn_up"}) {
                add("blk." + std::to_string(il) + "." + name + ".weight", GGML_TYPE_F32, {n_embd, n_embd});
            }
        }
    }
};

// loads b; expected: a part of the rejection's message, or nullptr if the model must load
static void check_load(const char * what, builder & b, const char * expected, bool load_mtp = false) {
    auto * model = load(b.meta, load_mtp);
    if (expected == nullptr) {
        check(model != nullptr, what);
        llama_model_free(model);
        return;
    }
    check(model == nullptr && messages.find(expected) != std::string::npos, what);
}

int main() {
    ggml_backend_load_all();
    llama_log_set(log_message, nullptr);

    check_versions();

    {
        fixture f(false, false);
        check_load("a valid coded fixture must load", f, nullptr);
    }
    {
        fixture f(false, false);
        f.pin()[3] = (int32_t) fixture::n_embd;
        check_load("an out-of-range input gather must be rejected", f, "output.trellis_pin is not a permutation");
    }
    {
        fixture f(false, false);
        f.pin()[0] = -1;
        check_load("a negative input gather must be rejected", f, "output.trellis_pin is not a permutation");
    }
    {
        fixture f(false, false);
        f.pin()[1] = f.pin()[0];
        check_load("a repeated input gather must be rejected", f, "output.trellis_pin is not a permutation");
    }
    {
        fixture f(false, false, GGML_TYPE_F32);
        check_load("a gather of the wrong type must be rejected", f, "output.trellis_pin is f32, expected i32");
    }
    {
        fixture f(false, false, GGML_TYPE_I32, 40);
        check_load("a coded weight whose rows are not a multiple of 16 must be rejected", f,
                "trellis: unsupported shape for output.weight");
    }
    {
        fixture f(true, false);
        check_load("a coded tied embedding must be rejected", f, "token_embd.weight is coded and also used as another tensor");
    }
    {
        fixture f(false, true);
        check_load("coded expert weights must be rejected", f,
                "blk.0.ffn_gate_exps.weight is coded, but only linear weights and the token embedding can be");
    }

    {
        fixture f(false, false);
        gguf_set_val_u32(f.meta, "prism.hadamard.version", 1);
        gguf_set_val_u32(f.meta, "prism.hadamard.block_size", 256);
        gguf_set_val_str(f.meta, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
        gguf_set_val_str(f.meta, "prism.hadamard.axis", "input-last-dimension");
        gguf_set_val_str(f.meta, "prism.hadamard.sign_mode", "identity");
        const char * folded[] = {"output.weight"};
        gguf_set_arr_str(f.meta, "prism.hadamard.weight_names", folded, 1);
        check_load("a Hadamard-folded coded weight must be rejected", f, "output.weight is coded and Hadamard-folded");
    }
    {
        mtp_fixture f(false);
        check_load("a coded embedding must load without the MTP block", f, nullptr);
    }
    {
        mtp_fixture f(false);
        check_load("an MTP block without a plain embedding must be rejected", f,
                "the MTP block needs a plain token embedding", true);
    }
    {
        mtp_fixture f(true);
        check_load("an MTP block with its own embedding must load", f, nullptr, true);
    }

    std::puts("trellis loading: format versions, gathers, tied embeddings, expert weights, Hadamard folds and MTP checked");
}
