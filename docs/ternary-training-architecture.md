---
title: "llama.cpp-optium: Ternary Training, QLoRA, Tiered Memory, and Bonsai 2"
type: note
aliases:
  - "Optium ternary training architecture"
date: 2026-09-19
status: implementation-in-progress
repository: "https://github.com/twinlunarstarz-dev/llama.cpp-optium"
branch: testing
tags:
  - ai/llm
  - ai/ternary
  - ai/finetuning
  - ai/qlora
  - infrastructure/gguf
  - infrastructure/tiered-memory
  - projects/llama-cpp-optium
---

# llama.cpp-optium: Ternary training architecture

> Status: production training-graph integration and CPU/reference Bonsai codec support are implemented on `testing`. CUDA compilation and real Bonsai inference remain in validation; assistant-only loss masking, QLoRA/QAT/ternary optimizer state, and tiered backward execution are still not implemented.

## Current implementation status

- `src/llama-graph.h`, `src/models/llama.cpp`, and `src/llama-context.cpp` now select a differentiable no-cache training graph, retain logits, and force `LLM_GRAPH_TYPE_TRAINING` during optimizer iterations.
- `llama-finetune` disables FlashAttention for the current backward path and defaults to the explicit `finetuned-model.gguf` output name. The CI workflow builds production source and exercises train -> export -> bounded single-turn reload without runner-side source patching.
- PQ2_0 (GGML type 142, 34-byte group-128 blocks) and PTQ1_0 (GGML type 143, 28-byte group-128 trit blocks) reference codecs, CPU Q8_0 dot paths, GGUF file-type mapping, and initial CUDA MMQ/MMVQ/dequant dispatch are in progress; real GPU logits parity is not yet a passing gate.
- Prism Hadamard metadata is validated and persistent rotation/sign tensors are allocated; the shared dense-matmul and token-embedding paths apply the corresponding transforms. `gdn_v_grouped` is deliberately fail-closed until its permutation path is verified.
- Assistant-only masking must still carry role boundaries into the dataset, zero ignored-label gradients in both CE forward and backward, and cover multi-turn/tool transcripts. Quantized primitives remain reference utilities, not a ternary optimizer.

## Existing implementation, verified from source

- `examples/CMakeLists.txt` includes `add_subdirectory(training)` and `examples/training/CMakeLists.txt` builds the `llama-finetune` target. The previous version of this note incorrectly inferred its absence from `tools/CMakeLists.txt`.
- `examples/training/finetune.cpp` loads a model, tokenizes `params.prompt`, creates `common_opt_dataset_init`, calls `llama_opt_init` and `llama_opt_epoch`, and exports GGUF. It forces non-mmap loading for writable weights and F32 KV cache. Its README describes FP32 training on a limited set of hardware; this is NOT a general quantized or out-of-core trainer.
- `src/llama-train-quant.h` exposes bounded row dequantization, per-tensor deterministic stochastic rounding, FP32 error feedback, and type-support checks. Its interface expressly excludes optimizer, paging, and CLI.
- `README.md` documents sequential VRAM/RAM/disk inference and overlapped CUDA transfer/compute. This is a useful foundation but does not imply that backward graphs, activations, and optimizer state can already be paged safely.
- `examples/training/teacher_dataset.py` now launches a teacher `llama-server` in an isolated process, generates OpenAI-compatible chat records, writes an atomic JSONL output, and terminates and waits for the teacher. It requires supplied fixture responses for tool calls; it does not execute arbitrary tools or fabricate results.
- `examples/training/test_teacher_dataset.py` covers four cases, including simulated-server process teardown. Its local tests passed. Real GGUF generation and end-to-end training integration remain untested.

## Objective

Build on `llama-finetune`, not a competing executable. Load any inference-compatible GGUF as a teacher; generate a varied, validated, provenance-tracked training corpus; fully unload the teacher; then initialize the student and train using full parameters or adapters. Add strictly ternary deployed full weights and LoRA factors; conventional F16/BF16 and nonternary QAT/QLoRA; safe VRAM -> RAM -> disk placement for weights, activations, gradients, optimizer states, and checkpoints; and verified Bonsai 2 inference support.

## Numerical invariants

1. During ternary training, *every forward step* must use packed or equivalent ternary weights, `W_forward = scale * Q` with `Q in {-1,0,+1}`. Trainable master weights, surrogate gradients, error feedback, and optimizer moments may use higher precision as transient and tiered training state. Do not train a floating-point student and only ternarize once at export.
2. Preserve exact architecture-specific transforms: Bonsai requires corresponding weight and activation rotations. Type-ID registration without the forward transform is not sufficient.
3. Strictly ternary deployed adapters need ternary A and B factors (plus required scales). Conventional floating-point LoRA does not meet that requirement; merging a ternary adapter into a single ternary plane is not generally lossless.
4. A frozen quantized base plus a trainable adapter is QLoRA, whereas full QAT updates quantized forward weights. Keep their checkpoint formats and numerical tests distinct.
5. Higher-precision normalization, recurrent state, and scales may be essential even when all eligible trainable *weight matrices* are ternary. Define the strict-ternary format contract in metadata rather than silently removing required numerical state.
6. Support as a teacher any model currently supported for inference, subject to the usual working-set requirements. Student training additionally requires backward operators for its particular architecture; provide explicit diagnostics when they are unavailable.
7. No arbitrary model size can be promised on every machine. Admission checks must cover disk capacity, device/host working-set minima, backend operations, and checkpoint spill space; allow CPU fallback where genuinely supported.

## Teacher-first lifecycle and dataset format

`validate teacher -> load teacher in isolated server process -> generate and verify examples -> write and fsync complete dataset -> shut down and wait for teacher -> load student and train -> atomically checkpoint/export`.

- Preserve exact roles, content, OpenAI-compatible `assistant.tool_calls` and corresponding `tool_call_id` results. Record model hash, tokenizer/chat-template version, sampling seed/settings, tool schema, and record provenance in a future manifest.
- Use supplied tool fixtures or real authorized tool execution in a sandbox; never label fabricated search results as observed facts. Tool outputs are untrusted. The current generator only supports fixtures and at most one tool-call round.
- Generate and validate diverse categories: general tasks, coding and debugging, abstention, research with actual retrieved sources, tool selection, sequential and parallel tools, tool errors, and multi-turn recovery. The six built-in prompts are smoke tests, not a balanced dataset.
- Tool descriptions are an OpenAI JSON `tools` array. Map Hermes and zoo-code harness message formats to this intermediate schema only after verifying the exact deployed harness versions. MCP is a transport and tool discovery protocol, not a universal transcript encoding.
- Crucial missing integration: the `llama-finetune` dataset loader presently consumes tokenized text. Do **not** pass the generated JSONL as ordinary text and assume its roles or loss masks are respected. Add chat-template-aware tokenization, assistant-only targets, tool-call target handling, splitting, and packing first.

## Training implementation sequence

1. Add chat-aware dataset loader and masks to the existing `llama-finetune`. Keep the old `--file` text path working. Integrate the isolated teacher helper and resource-lifecycle tests.
2. Establish full F16/BF16 training and conventional LoRA with numerical gradient tests, loss-decrease tests, resume, and export on tiny supported models; preserve current FP32 behavior.
3. Extend the scheduler with bounded residency for weights, saved activations, backward gradients, train-only master weights, optimizer shards, and checkpoint writes. Make dependency and last-use accounting correct before introducing overlap.
4. Add out-of-core synchronous reference execution; then double-buffered prefetch/compute/writeback with bounded queue depths and backpressure. Compare gradients and updates against reference execution.
5. Add frozen quantized base plus LoRA, nonternary QAT, and then projected ternary full training and ternary LoRA. Each mode needs validated backward behavior and checkpoint metadata.
6. Independently reproduce Bonsai 2's custom GGML codec and Hadamard/architecture path against the Prism reference, and verify actual model loading, short deterministic logits, and offloaded inference before declaring the repair complete.

## Tiered residency and correctness

Model a tensor as resident, prefetched, in-use, dirty, writing, or evictable; retain durable identity/version and explicit read/write ownership. Use separate capacity budgets for forward weights, saved activations, gradients, optimizer state, and prefetch. Never evict an activation before its last backward consumer or a dirty tensor before durable writeback. Store optimizer and master shards in host/disk tiers when necessary, implement activation checkpointing/recomputation, and make RNG and update order independent of prefetch timing. Measure read/write amplification, queue depth, cache hit rate, stalls, VRAM/RAM peaks, and throughput. Reuse existing checkpoint primitives only after validating the new training state schema and recovery guarantees.

## Verification gates

- Build `llama-finetune` and the selected backends. Prove old training still works.
- Test teacher startup/termination and error paths using fake and real server runs, and ensure no teacher resources remain when student load begins.
- Verify tokenizer/chat template, tool-call transcripts, loss masks, dataset provenance, and safe parsing.
- Check numerical gradient and finite/decreasing loss in F16 full, F16 LoRA, QLoRA, QAT, ternary full, and ternary LoRA modes on small models.
- Verify checkpoint resume including tier placement, RNG, optimizer and quantization state.
- Validate strict ternary export by reloading it and checking expected codebooks and correct logits.
- Test CPU-only, constrained RAM, and GPU+RAM+disk paths; compare the offloaded implementation to in-memory numerical baselines.
- Benchmark Bonsai 2 against the reference Prism fork: exact custom type layouts, required transforms, full-file loading, logits, and inference with constrained memory. Do not claim support based on parsing alone.

## Current example: isolated teacher data generation

```sh
python3 examples/training/teacher_dataset.py \
  --server ./build/bin/llama-server \
  --model /path/to/teacher.gguf \
  --tasks /path/to/prompts.jsonl \
  --tools /path/to/openai-tools.json \
  --output /path/to/teacher-records.jsonl
python3 -m unittest discover -s examples/training -p 'test_teacher_dataset.py' -v
```

`prompts.jsonl` contains one JSON object per line with `prompt`, optional `category`, and optional `tool_results` mapping tool names to trusted fixtures. This command only generates data; the resulting JSONL is **not** yet a valid drop-in input for the existing `llama-finetune` text training path.

## References

- [Existing fine-tune implementation](https://github.com/twinlunarstarz-dev/llama.cpp-optium/blob/testing/examples/training/finetune.cpp)
- [Training CMake target](https://github.com/twinlunarstarz-dev/llama.cpp-optium/blob/testing/examples/training/CMakeLists.txt)
- [Training quantization primitives](https://github.com/twinlunarstarz-dev/llama.cpp-optium/blob/testing/src/llama-train-quant.h)
- [Sequential loading design](https://github.com/twinlunarstarz-dev/llama.cpp-optium/blob/testing/docs/architecture-sequential-loading-v2.md)
- [Prism Bonsai model](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- [Prism llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp/tree/prism)
