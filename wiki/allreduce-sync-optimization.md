---
title: "AllReduce sync optimization"
type: note
tags:
  - cuda
  - allreduce
  - tensor-parallelism
  - performance
source:
  - codebase
date: "2026-08-08"
---

# AllReduce improvements for tensor-split mode

## Benchmark baseline (before any changes)

| Metric | Result |
|---|---|
| TG eval | 51.4 ms/tok (19.5 tok/s) |
| PP 1728 tok | 2.67 ms/tok (374 tok/s) |

## Changes implemented

### 1. Host sync → device-side wait + pool 2→4

`acquire_slot()` no longer blocks the CPU; it records `ev.ker` early and uses `cudaStreamWaitEvent`. Result: **TG flat** (~52ms both ways) — host stall was not the decode bottleneck. Decode uses the small-tensor kernel path.

### 2. BF16 threshold raised: 1 → 64 KB (default)

Decode's tiny F32 collectives now skip the two extra BF16 conversion kernels (F32→BF16 and back). Old default of `1` (always on) added launch overhead for small tensors. Override: `GGML_CUDA_AR_BF16_THRESHOLD`.

### 3. Tunable kernel block count, lower default: 8 → 4

Each kernel block spins one thread waiting on the peer GPU. Fewer blocks = less wasted SM capacity during decode. Prefill can raise via `GGML_CUDA_AR_KERNEL_BLOCKS`. The arrival ring still sizes for 8 (compile-time constant), so the runtime count is purely a launch choice.

### 4. NCCL BF16 crossover now tunable

New env var `GGML_CUDA_NCCL_BF16_THRESHOLD` (element count) overrides the hard-coded RTX-4090-PCIe-4 heuristic in [`ggml-cuda.cu`](ggml/src/ggml-cuda/ggml-cuda.cu:1018). Value `0` disables the FP32 fast-path.

## Why accuracy is preserved

All changes are scheduling/tuning: device-side ordering replaces host-blocking with identical guarantees, and threshold/env-var changes affect only *which* existing path runs — no new numerics, no layout changes.

## Tuning knobs (no recompile)

| Env var | Purpose | Default |
|---|---|---|
| `GGML_CUDA_AR_BF16_THRESHOLD` | bytes above which internal AR uses BF16 round-trip | 65536 |
| `GGML_CUDA_AR_COPY_THRESHOLD` | bytes above which copy-engine path kicks in | 1048576 |
| `GGML_CUDA_AR_KERNEL_BLOCKS` | chunked-kernel block count (spin-wait threads) | 4 |
| `GGML_CUDA_NCCL_BF16_THRESHOLD` | NCCL BF16 crossover element count (NEW) | old heuristic |
| `GGML_CUDA_AR_COPY_CHUNK_BYTES` | fixed copy-engine chunk size | heuristic |

## Deferred (architect-mode scope)

- Per-size custom-AR-vs-NCCL hybrid dispatcher.
- Graph-level collective-count reduction.

## Artifacts

- [`ggml/src/ggml-cuda/allreduce.cu`](ggml/src/ggml-cuda/allreduce.cu)
- [`ggml/src/ggml-cuda/ggml-cuda.cu`](ggml/src/ggml-cuda/ggml-cuda.cu)
