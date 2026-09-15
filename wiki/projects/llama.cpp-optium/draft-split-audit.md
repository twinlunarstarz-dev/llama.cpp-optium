---
title: DSpark/DFlash Support and Row-Split Failure Analysis
type: note
tags: project-llama-cpp-optium,debugging,cuda,handoff
---

# Context
User reports row-split failures while tensor-split works for DSpark/DFlash speculative decoding in llama.cpp-optium. Need to audit target/draft setup, cache restrictions, graph execution, acceptance, and state handling to identify root causes and propose low-risk fixes.

# What Was Done
- Examined source code for DFlash/DSpark implementation in `src/models/dflash.cpp`
- Reviewed speculative decoding framework in `common/speculative.h` and `common/speculative.cpp`
- Analyzed model loader and tensor splitting logic in `src/llama-model-loader.h`, `src/llama-model.cpp`, and `src/llama-arch.cpp`
- Checked metadata handling and KV cache injection mechanics
- Verified model architecture specifics for Qwen3.8-27B variants present in `/models`

# Findings
## Target/Draft Setup
- DFlash model loads target layer IDs via `LLM_KV_TARGET_LAYERS` metadata, requiring explicit extraction layers
- DSpark adds Markov head (`markov_w1.weight`, `markov_w2.weight`) and confidence head (`conf_proj`)
- DFlash2 uses selector lattice via `dflash_selector_hidden` tensor for top-k candidate scoring
- Draft model hidden size (`n_embd_dec`) may differ from target model hidden size (`n_embd_tgt`)
- Target features extracted via `llama_get_embeddings_layer_inp` from `ctx_tgt`, concatenated across `target_layer_ids_n` layers

## Cache Restrictions
- KV cache injection in `common_speculative_impl_draft_dflash::process`:
  - Computes Kcur/Vcur from target features using target model weights (`layer.wk`, `layer.wv`)
  - Applies norm and RoPE before injecting into draft model KV cache via `ctx_dft->mctx->cpy_k/v`
  - Assumes weight matrix multiplication produces locally computable results without cross-device communication
- Row-split mode (`LLAMA_SPLIT_MODE_ROW`) splits weight matrices by output feature rows
- Input activations (target features) may be split inconsistently with weight matrix row partitioning
- Tensor-split (`LLAMA_SPLIT_MODE_TENSOR`) works because whole tensors reside on single devices

## Graph Execution
- Encoder graph: processes target features through FC layer and norm (`build_inp_embd_enc` → `fc_out` → `enc_norm_out`)
- Decoder graph has dual paths:
  - *Embd batch*: KV injection pass (target features → draft KV cache)
  - *Token batch*: noise block denoising → draft token sampling
- DSpark adds Markov head biasing and confidence thresholding during drafting
- DFlash2 uses selector lattice from `h_nextn` for token selection without consuming logits

## Acceptance & State Handling
- Acceptance updates per-implementation state (e.g., Eagle3's deferred boundary, MTP's pending hidden state)
- DFlash/DSpark implementations do not override `accept()` (no-op), relying on base class state tracking
- Draft parameters track `n_past`, `id_last`, and per-sequence drafting status
- State serialization handles per-implementation checkpoints (e.g., Eagle3's `pending_g_last`)

## Row-Split Failure Hypotheses
1. **Weight Activation Misalignment**: Row-split partitions weight matrices by rows but does not partition activations compatibly for the target feature → KV injection matrix multiplication. Tensor-split avoids this by keeping tensors intact.
2. **Device Mismatch**: Target model context (`ctx_tgt`) and draft model context (`ctx_dft`) may reside on different devices when using row-split, requiring explicit tensor transfers during KV injection that are not currently handled.
3. **Partial Tensor Operations**: Row-split may cause intermediate tensors (like Kcur/Vcur) to be fragmented across devices, breaking the assumption that `build_lora_mm` produces a contiguous tensor suitable for direct KV cache injection.
4. **Metadata Extraction**: Target layer extraction (`llama_get_embeddings_layer_inp`) may return split tensors that cannot be concatenated without cross-device communication when row-split is active.

# Fix / Solution
Low-risk mitigations (no code changes required):
1. **Avoid Row-Split with DFlash/DSpark**: Use tensor-split or no-split for speculative decoding workloads involving DFlash/DSpark draft models.
2. **Match Context Devices**: Ensure `ctx_tgt` and `ctx_dft` are placed on identical device sets when row-split is unavoidable (via identical `n_gpu_layers` and `tensor_split` parameters).
3. **Uniform Layer Extraction**: Verify all `target_layer_ids` map to layers with identical row-split partitioning to enable per-device feature concatenation.
4. **Fallback to Tensor-Split**: For troubleshooting, temporarily switch to tensor-split to isolate whether failures are specific to row-split mechanics.

Notes / Gotchas
- DFlash2 selector operations involve additional hidden-state projections that may exacerbate row-split alignment issues.
- DSpark Markov head introduces recurrence that may be sensitive to hidden-state fragmentation.
- User's specific setup (Qwen3.8-27B with DSpark/DFlash sidecars) likely uses tensor-split for base model but row-split for speculative contexts, causing device mismatches.
- No evidence of incorrectness in tensor-split path; failures appear isolated to row-split configuration.

Handoff
Further investigation would require:
- Device-aware tracing of tensor allocations during KV injection
- Verification of weight activation alignment under row-split via GGML tensor metadata
- Comparison of split statistics between working tensor-split and failing row-split runs
Links: [[index]] [[current-state]]