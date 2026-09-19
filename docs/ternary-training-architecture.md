---
title: "llama.cpp-optium: Ternary Training, QLoRA, Tiered Memory, and Bonsai 2"
aliases:
  - "Optium ternary training architecture"
date: 2026-09-19
status: design-proposal
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

> Status: design proposal only. This note is not an implementation report; no Bonsai 2 inference repair or end-to-end training build has been verified.

## Objective

Add `llama-finetune` with (1) teacher-generated supervised data from a separate base GGUF, (2) full-parameter training in a ternary forward representation, (3) LoRA and QLoRA for ordinary and ternary base models, (4) a strictly ternary *deployed* adapter option, and (5) bounded VRAM -> RAM -> disk storage for model weights, activations, gradients, optimizer state, and checkpoints. Preserve conventional F16/BF16 training and quantization-aware training (QAT) for supported quantized types. Fix the Bonsai 2 inference loader and its required rotation rather than merely accepting custom tensor IDs.

## Confirmed repository starting points

- `README.md` describes sequential storage-backed loading with mmap/direct I/O, bounded host/device windows, and overlapped CUDA transfer and compute; this is an inference foundation, not proven backward/optimizer support.
- `src/llama-train-quant.h` exposes bounded row dequantization, per-tensor deterministic stochastic rounding, FP32 error-feedback state, and type-support checks. Its own header places optimizer, paging, and CLI out of scope.
- `tools/CMakeLists.txt` on `testing` does not register a `llama-finetune` executable.
- `docs/architecture-sequential-loading-v2.md` records earlier scheduler allocation/OOM/NaN failure modes. Treat it as architecture history, not evidence that all configurations now work.

## Non-negotiable numerical contracts

1. A packed ternary forward weight is represented as `W_forward = scale * Q`, `Q in {-1,0,+1}`. If a required Hadamard/sign transform exists, preserve the exact transform and its placement in the graph.
2. A training algorithm needs gradients and an update state. For full ternary fine-tuning, use a bounded, shardable higher-precision *training-only* latent/master or error-feedback state and a straight-through or otherwise specified surrogate gradient. Re-ternarize each update in the forward path; do not train a floating student and perform a one-time final ternarization.
3. The deployed GGUF may contain only ternary trainable weight tensors plus indispensable scalar/group scales, metadata, and architecture-specific non-weight tensors. A strict interpretation of 'all fully ternary' cannot also prohibit floating-point group scales or normalization/state values without a new numerical design.
4. A conventional LoRA update `B @ A` stored as F16 is not a ternary-only adapter. A ternary LoRA mode must explicitly quantize/store BOTH factors (with their necessary scales), run their quantized forward path during training, and test merged and unmerged behavior. Merging a ternary LoRA into a one-plane ternary base is not generally lossless.
5. QLoRA keeps the base quantized/frozen and trains adapters; ordinary full QAT updates quantized forward weights. These are distinct training modes, with distinct checkpoint schemas.
6. 'Any model supported for inference' means teacher generation can use the existing inference loader. Training support requires an explicit architecture-specific differentiable graph and supported backward kernels; fail with a precise unsupported-op diagnostic otherwise.
7. No universal model-size guarantee: admission control must check disk capacity, minimum live working set, supported compute backend, temporary spill capacity, and predicted wall-clock/I/O costs.

## Teacher-first execution pipeline

`validate teacher GGUF -> load teacher through inference API -> generate diverse candidate prompts and responses -> parse/validate and deduplicate -> persist append-only dataset manifest + records -> flush and fsync -> destroy inference contexts and release model/backend buffers -> load student checkpoint and training state -> train -> atomically publish checkpoint/export`.

- Teacher and student may be different GGUF files. Record teacher model hash, tokenizer and chat-template hashes, generation parameters, tool schemas, sampling seeds, record provenance, and split assignment.
- Store transcripts in a structured format with role, content parts, tool-call ID/name/JSON arguments, tool responses, and target token masks. Never fabricate tool execution results: execute authorized sandboxed tools or mark responses as synthetic scenarios. Tool outputs are untrusted data.
- Default schema: OpenAI-compatible `tools`, `tool_choice`, `assistant.tool_calls`, matching `tool_call_id` responses. Supply configurable adapters for Hermes-style tool templates and the specific zoo-code harness schema after examining those versions. MCP is a tool discovery/transport protocol, not itself a single universal transcript format.
- Generate categories including coding, debugging, tool selection, multi-step and parallel tool calls, research, search with citations, general knowledge, abstention, clarification, and recovery from tool failures. Balance categories and hold out task families and prompts for evaluation. Avoid training on benchmark test sets or assuming teacher generations are ground truth.
- Resume teacher generation without duplicates after crashes; commit the completed dataset manifest before unloading the teacher.

## Storage and overlapped scheduling

Implement a shared tensor-store abstraction with durable tensor identity and versioning across VRAM, pinned/pageable RAM, and local disk. Track distinct states: resident, prefetched, computing, dirty, writing, evictable. Reserve bounded windows for **forward weights, backward activations/gradients, optimizer state, and prefetch**, rather than sharing one unbounded cache. Reuse sequential inference placement only after verifying its allocator and synchronization assumptions for backward graphs.

- Precompute a dependency-aware forward/backward/optimizer schedule. Use double-buffered asynchronous reads, host->device copies, compute, and writeback where the backend supports them; fall back to synchronous execution elsewhere.
- Do not evict dirty tensors until writes are durable, or activations until the last backward consumer; support recomputation and activation checkpointing under a memory budget.
- Keep master/error-feedback and optimizer shards in RAM or disk rather than requiring a full-device copy. Maintain deterministic update ordering and counters, not dependent on prefetch completion order.
- Bound outstanding I/O and queue depth; use locality-aware layer/block ordering and grouped writes. Measure disk throughput, read/write amplification, cache hit rate, GPU idle time, stalls, and write endurance. Backpressure rather than unbounded prefetch.
- Checkpoint atomically with the existing `llama-train-state` / `llama-train-checkpoint` primitives if their schemas meet new requirements. Include optimizer, RNG, schedule cursor, dataset cursor, latent states, ternary scales/codes, and hashes.

## Bonsai 2 inference repair: investigation checklist

1. Obtain the exact failing model variant, shard set, runtime command, platform, and error/log. Record the failing commit and compare against Prism's matching llama.cpp fork.
2. Confirm custom GGML type IDs, block sizes, validation, row dequantization, quantized matmul kernels, GGUF loader type tables, tensor allocation, and serialization; do not alias custom types to stock names with different layouts.
3. Confirm the model architecture, tensor naming and shapes, hybrid/recurrent attention operations, required Hadamard/sign metadata, and activation transform placement. Reject absent or incompatible rotation metadata rather than silently producing invalid logits.
4. Add focused small-fixture tests: encode/decode round-trip against reference, one-layer numerical parity, deterministic short-token logits against the reference runtime, reload/unload, and small-memory/offloaded inference. Check errors, NaNs and OOM handling.
5. Only label Bonsai 2 supported after full-file load and reference-logit/inference tests on an accessible real checkpoint. Custom GGML type parsing alone does not establish compatibility.

## Delivery increments and acceptance gates

1. **Compatibility audit:** document exact failing Bonsai configuration and reference expected outputs. No speculative fix before reproduction.
2. **Training CLI baseline:** build `llama-finetune` for a small supported architecture, F16 full fine-tune and F16 LoRA with finite-loss, decreasing-loss, resume and export tests.
3. **Teacher pipeline:** validate generation, dataset format, tool-call parsing, durable manifest, teacher unload and student load using explicit memory-resource tests.
4. **Tiered training:** test CPU-only, RAM-constrained and GPU+RAM+disk runs; verify deterministic resume and numerical agreement with an in-memory baseline on a tiny model.
5. **Quantized modes:** frozen quantized base + LoRA (QLoRA), then nonternary QAT; compare against float baselines and quantify quantization drift.
6. **Ternary modes:** train using a ternary forward graph, implement train-only master/error feedback; export strict ternary full model and strict ternary adapter; reload and compare training/inference logits. State explicitly where scales/non-weight state remain higher precision.
7. **Performance:** profile overlapping reads/transfers/computation and show throughput, peak VRAM/RAM, spill bytes, write amplification and correctness relative to synchronous mode.
8. **Model coverage:** maintain a feature matrix for architecture, quantization type and backend. Teacher coverage may exceed student-training coverage until backward operators exist.

## Suggested initial CLI contract (design, not implemented)

```sh
llama-finetune generate --teacher teacher.gguf --output dataset.jsonl --tools tools.json --format openai --categories coding,tools,research,general --seed 42
llama-finetune train --student student.gguf --dataset dataset.jsonl --mode ternary-full --vram-budget 8GiB --ram-budget 24GiB --disk-budget 300GiB --output checkpoint/
llama-finetune train --student student.gguf --dataset dataset.jsonl --mode ternary-lora --rank 16 --output adapter/
llama-finetune export --checkpoint checkpoint/ --format gguf --strict-ternary --output result.gguf
```

## Outstanding decisions to record during implementation

- Exact ternary codec and group-size support by architecture (PTQ1_0/PQ2_0 or a new backward-friendly internal representation).
- Straight-through gradient, alternate estimator, or projection method; optimizer and scale training policy.
- Whether a strict ternary adapter is permitted to retain FP16 group scales and non-weight tensors; define this unambiguously in format metadata.
- Whether optional tool execution is offline/sandboxed or remote, and how data licensing and provenance are enforced.
- Which GPUs, operating systems, disk types, and minimum working sets are in the supported tiered-training matrix.

## Source pointers

- [Fork README](https://github.com/twinlunarstarz-dev/llama.cpp-optium/blob/testing/README.md)
- [Existing quantized training primitives](https://github.com/twinlunarstarz-dev/llama.cpp-optium/blob/testing/src/llama-train-quant.h)
- [Sequential loading architecture](https://github.com/twinlunarstarz-dev/llama.cpp-optium/blob/testing/docs/architecture-sequential-loading-v2.md)
- [Prism Bonsai 2 model](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- [Prism llama.cpp implementation](https://github.com/PrismML-Eng/llama.cpp/tree/prism)
