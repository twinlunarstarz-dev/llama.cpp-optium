#include "models.h"

void llama_model_eagle4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("EAGLE4 model requires 'eagle4.target_layers' in GGUF metadata");
    }
    if (target_layer_ids.size() != 3) {
        throw std::runtime_error("EAGLE4 requires exactly 3 entries in 'eagle4.target_layers'");
    }
    LLAMA_LOG_INFO("%s: EAGLE4 target_layers = [%d, %d, %d]\n", __func__,
            target_layer_ids[0],
            target_layer_ids[1],
            target_layer_ids[2]);

    uint32_t n_embd_tgt = 0;
    ml.get_key(LLM_KV_TARGET_HIDDEN_SIZE, n_embd_tgt);
    LLAMA_LOG_INFO("%s: EAGLE4 n_embd_tgt = %u (draft n_embd = %u)\n", __func__, n_embd_tgt, hparams.n_embd);

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * n_embd_tgt;

    // eagle4 fc_norm (optional, default false): per-segment RMSNorm before fc
    ml.get_key(LLM_KV_FC_NORM, hparams.norm_before_fc, false);
    if (hparams.norm_before_fc) {
        LLAMA_LOG_INFO("%s: EAGLE4 fc_norm = true\n", __func__);
    }

    // eagle4 norm_output (optional, default false)
    ml.get_key(LLM_KV_NORM_OUTPUT, hparams.norm_output, false);
    if (hparams.norm_output) {
        LLAMA_LOG_INFO("%s: EAGLE4 norm_output = true\n", __func__);
    }

    // sliding-window attention (optional): the drafter is trained with a limited
    // attention window (e.g. 4096) and must not attend over the whole context.
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        // SWA layers use the *_swa rope parameters; keep them identical to the
        // dense ones unless the file overrides them explicitly
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
        // per-layer pattern array if present, otherwise mark every layer as SWA
        if (!ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, false)) {
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                hparams.is_swa_impl[il] = 1;
            }
        }
        LLAMA_LOG_INFO("%s: EAGLE4 n_swa = %u, is_swa_any = %d\n", __func__, hparams.n_swa, (int) hparams.is_swa_any());
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_eagle4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // Feature fusion layer: projects 3 target layers to draft hidden size
    fc = create_tensor(tn(LLM_TENSOR_FC, "weight"), {n_embd_inp, n_embd}, 0);

    // fc_norm: one RMSNorm weight per target-layer segment (eagle4.fc_norm)
    if (hparams.norm_before_fc) {
        const int64_t n_embd_tgt = n_embd_inp / 3;
        fc_norm[0] = create_tensor(tn(LLM_TENSOR_FC_NORM_0, "weight"), {n_embd_tgt}, 0);
        fc_norm[1] = create_tensor(tn(LLM_TENSOR_FC_NORM_1, "weight"), {n_embd_tgt}, 0);
        fc_norm[2] = create_tensor(tn(LLM_TENSOR_FC_NORM_2, "weight"), {n_embd_tgt}, 0);
        LLAMA_LOG_INFO("%s: EAGLE4 fc_norm loaded (3 x %lld)\n", __func__, (long long) n_embd_tgt);
    }

    // Output layer (uses draft vocab size)
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // Token embeddings (optional - may have its own)
    const struct ggml_tensor * tok_embd_meta = ml->get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    if (tok_embd_meta) {
        const int64_t n_target_vocab = tok_embd_meta->ne[1];
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_target_vocab}, 0);
        LLAMA_LOG_INFO("%s: EAGLE4 using its own token_embd (vocab = %lld)\n", __func__, (long long)n_target_vocab);
    }

    // Single decoder layer
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // input_layernorm: applied to token embeddings
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // eagle4 specific: hidden_norm applied to fused target features
        layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);

        // Attention takes input_embeds_normed + fused_target_normed as input
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd * 2, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd * 2, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd * 2, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // rope_freqs for rope scaling (optional)
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED);
    }
}

template <>
ggml_tensor * llama_model_eagle4::graph<true>::build_inp_embd_enc() const {
    ggml_tensor * cur = nullptr;

    // Input: Target model features (3 layers concatenated: low, mid, high)
    // Data will be provided via ubatch->embd in encode_eagle4_features()
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// eagle4 Encoder: processes target model features through feature fusion layer
// Input: target_features e.g. [15360, n_tokens] from target model layers low, middle, high
// Output: g_embeddings e.g. [5120, n_tokens] stored in context
template <>
llama_model_eagle4::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = nullptr;

    cur = build_inp_embd_enc();

    // fc_norm: RMSNorm each target-layer segment independently before fusion.
    // The encoder input is [3 * n_embd_tgt, n_tokens] (segments concatenated on dim 0).
    if (hparams.norm_before_fc) {
        const int64_t n_seg = hparams.n_embd_inp_enc() / 3;
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(cur->ne[0] == 3 * n_seg);

        ggml_tensor * parts[3];
        for (int k = 0; k < 3; ++k) {
            ggml_tensor * seg = ggml_view_2d(ctx0, cur, n_seg, n_tokens,
                    cur->nb[1], (size_t) k * n_seg * ggml_element_size(cur));
            parts[k] = build_norm(seg, model.fc_norm[k], NULL, LLM_NORM_RMS, -1);
            cb(parts[k], "enc_seg_norm", -1);
        }
        cur = ggml_concat(ctx0, ggml_concat(ctx0, parts[0], parts[1], 0), parts[2], 0);
        cb(cur, "enc_input_norm", -1);
    }

    // Feature fusion layer
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    // Output: g_embeddings e.g. [5120, n_tokens]
    // store in t_h_nextn (same as MTP) so can be read via llama_get_embeddings_nextn(ctx_dft)
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// eagle4 Decoder: processes draft tokens using g_embeddings from encoder
// Input: draft tokens + g_embeddings from encoder
// Output: draft logits
template <>
llama_model_eagle4::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_layer == 1);  // eagle4 has only one decoder layer

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // eagle4 Decoder receives:
    // 1. Token embeddings (e.g. from eagle4's own tok_embd, or target model)
    // 2. g_embeddings from encoder
    auto * tok_embd = model.tok_embd;
    if (model.tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "EAGLE4 decoder requires token embeddings (own or from target model)");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * inp_embd = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inp_embd, "inp_embd", -1);

    ggml_tensor * inp_g = inp->embd;
    cb(inp_g, "inp_g_embeddings", -1);

    res->add_input(std::move(inp));

    inpL = inp_g;

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // SWA drafters need the iswa attention input (windowed KV cache)
    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (hparams.is_swa_any()) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // Single decoder layer (il = 0)
    const int il = 0;
    {
        // Apply input_layernorm to the token embeddings
        ggml_tensor * embd_norm = build_norm(inp_embd,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // Apply hidden_norm to inp_g
        ggml_tensor * g_norm = build_norm(inp_g,
                model.layers[il].attn_norm_2, NULL,
                LLM_NORM_RMS, -1);
        cb(g_norm, "g_norm", il);

        // Concatenate normalized inp_embd and normalized inp_g
        cur = ggml_concat(ctx0, embd_norm, g_norm, il);
        cb(cur, "concat_embd", il);

        // Self-attention with concatenated input
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // rope freq factors, returns nullptr if not available
        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        // RoPE
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        cur = inp_attn
            ? build_attn(inp_attn,
                model.layers[il].wo, NULL, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
            : build_attn(inp_attn_iswa,
                model.layers[il].wo, NULL, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        // Add residual and update it
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        // Apply FFN norm to the sum
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Output norm with residual
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "eagle4_prenorm", il);

        inpL = cur;
    }

    cur = inpL;

    // EAGLE-4 norm_output: when set, the per-step hidden state returned by the
    // drafter is post-norm, and both the next TTT step and lm_head consume it.
    // When unset (EAGLE-3 behaviour), the returned state is pre-norm and the
    // final norm is applied only on the lm_head path.
    if (hparams.norm_output) {
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
    }

    // Output state used as g_embeddings for the next draft step
    ggml_set_output(cur);
    res->t_h_nextn = cur;

    if (!hparams.norm_output) {
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
    }

    // lm_head - projects to draft vocabulary
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->output != nullptr && "EAGLE4 decoder requires an output projection (own or from target model)");
        output = model_other->output;
    }
    cur = build_lora_mm(output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::unique_ptr<llm_graph_context> llama_model_eagle4::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}
