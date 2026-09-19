---
title: "llama.cpp-optium — Ternary Training Implementation Status"
aliases:
  - "Optium ternary finetune status"
date: 2026-09-19
status: incomplete
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

# llama.cpp-optium — Ternary Training Implementation Status

## Required end state

A single teacher-first workflow should accept any inference-compatible GGUF; produce genuinely varied, provenance-checked OpenAI-compatible chat and tool-call training data; terminate and unload the teacher; then run validated full, conventional LoRA, QLoRA, quantization-aware or directly ternary student training with bounded VRAM → RAM → disk residency, overlapped I/O and compute, resumable checkpoints, and reloadable GGUF/adapters. Bonsai 2 loading and numerical inference must be independently verified against Prism. These requirements are **not yet met**.

## Implemented and verified

- `examples/training/teacher_dataset.py` creates OpenAI-style JSONL using a child `llama-server`, fixed tool-result fixtures, an atomic dataset write, and a terminating/waited teacher subprocess. It does not execute external research tools or guarantee training-data quality.
- `examples/training/teacher_student.py` validates the generated corpus and starts `llama-finetune` only after teacher generation has exited. Real subprocess lifecycle and failure-path tests passed in GitHub Actions run [35429523263](https://github.com/twinlunarstarz-dev/llama.cpp-optium/actions/runs/35429523263).
- `llama-finetune` builds in CPU CI; `--teacher-jsonl` renders OpenAI-compatible messages with the student chat template. It currently optimizes all transcript roles and has no assistant-only loss mask.
- Existing CPU quantized-training reference primitives now have a CMake test target `llama-train-quant-test`, and their tests passed in run [35430007334](https://github.com/twinlunarstarz-dev/llama.cpp-optium/actions/runs/35430007334). They are **not wired into the training optimizer**.

## Confirmed blocker: actual full fine-tuning

A real CPU test on pinned `ggml-org/tiny-llamas` `stories260K.gguf` builds and loads the model and reaches optimizer initialization, then aborts in `ggml_build_backward_expand` on a view-backed `SET_ROWS` write to `cache_k_l0` (source `cache_k_l0`). The diagnostic CI run is [35432182852](https://github.com/twinlunarstarz-dev/llama.cpp-optium/actions/runs/35432182852). The inference KV-cache write is in the gradient path, and the generic backward implementation does not implement `SET_ROWS` gradients. Removing the safety assertion or declaring the cache write gradient-free without proving the K/V gradient path would be incorrect. A dedicated differentiable training attention graph is required, with small-model gradient and loss-decrease tests.

## Additional reference numerical defect

The counter-based stochastic-rounding hash in `src/llama-train-quant.cpp` shifts a 64-bit integer right by 11 (53 output bits), but divides by `2^52` instead of `2^53`. This can produce numbers in `[0,2)` rather than the documented `[0,1)` and biases rounding. A guarded, test-before-commit correction is tracked in [one-off workflow run 35432592876](https://github.com/twinlunarstarz-dev/llama.cpp-optium/actions/runs/35432592876); verify its outcome before treating the fix as merged.

## Not implemented or demonstrated

- Full F16/BF16 training, LoRA, QLoRA, quantized QAT integrated with `llama_opt`, direct ternary full/adapter updates, strictly ternary persisted adapters, or all-architecture backward support.
- Tiered training-state residency across GPU, host RAM, and disk; overlapped prefetch/backprop/writeback; bounded memory admission; checkpoint/resume correctness.
- Real automatic diversified dataset generation for research and authenticated/sandboxed tools, assistant-only targets, or verified Hermes/zoo-code harness-specific round-trip transcripts.
- Bonsai 2 PQ2_0/PTQ1_0 decode/matmul, rotation placement, end-to-end model load, reference-logit parity, or functional inference repair in this fork.

## Verification before pulling for functional testing

Check the latest `testing` commit and GitHub Actions status. The teacher lifecycle and reference quant tests passing do **not** establish real training. Do not use this branch for irreversible model fine-tunes until a real GGUF finishes optimizer steps, produces a reloadable output, and shows a finite and decreasing loss. See `docs/ternary-training-architecture.md` for the detailed design and numerical invariants.
