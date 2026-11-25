# Verified MoE tuning: Inkling-2-Mini on a 12 GB Turing card

Measured on RTX 2060 (12 GB, sm_75), i9-12900K, cgroup capped 60 GiB. Every number below comes from
`/completion` timings on a 1321-token prompt (`PP`) and a 90-line generation (`TG`), with the
`17*23 -> 391` correctness probe run at the same settings. Do not trust any figure here without the
correctness probe: a 24-token budget lands mid-echo and looks degenerate.

## Winning configuration

```
load-mode       = none          # required: -ot cannot move experts under mmap
gpu-layers      = 99
moe-mode        = strata        # or basalt; identical on this host
moe-cache-mib   = 0             # VRAM expert bank measured neutral (+2%)
threads         = 8             # E-cores hurt quantized dots
threads-batch   = 16
ubatch-size     = 512
cache-type-k/v  = q4_0
kv-offload      = false         # +8% decode, -2% prompt (paired A/B, 2 rounds)
override-tensor = blk\.(0|1|2|3|4|5|6|7|8|9|10)\.ffn_(gate_up|down)_exps\.weight=CUDA0
```

Result on a clean GPU: **PP ~45.2, TG ~8.0 at ctx 32768; PP 44.9, TG 7.9 at ctx 180000**,
`391` correct in both, VRAM ~7.2 GiB, host RAM available ~25.7 GiB at 180000.

## Measure on a clean GPU or the numbers are fiction

A leaked `llama-server` from an earlier run held **6824 MiB** of VRAM while subsequent tests ran.
Every "improvement" measured in that window was real throughput degraded by contention, and an
11-block arm read PP 43.5 / TG 7.6 instead of its true PP 45.2 / TG 8.0. Always verify with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before a sweep, and confirm the GPU
returns to ~780 MiB after each arm. A test harness must kill its own child by process group;
`start_new_session=True` + `os.killpg` works, port-based `pkill -f "port 609xx"` can match the
harness's own command line and kill the shell.

## Why 11 expert blocks and not more

Blocks 0-19 are all IQ4_NL, 547 MiB each; blocks 20-23 are Q8_0, 1033 MiB each (verified from GGUF).

| blocks on GPU | result |
|---|---|
### Only `-ot` puts expert weights in VRAM on this build

Measured with `nvidia-smi` at load time, same flags (`--load-mode none`, ctx 16384):

| placement | VRAM at load | TG | correct |
|---|---|---|---|
| `-ot` blk 0-10 | **7055 MiB** | 9.13 | yes |
| `--n-cpu-moe 4` | **1037 MiB** | 7.91 | yes |
| `--n-cpu-moe 0` | ~790 MiB | 7.9 | yes |

`--n-cpu-moe N` only keeps the first N layers' experts on the CPU; it does **not** move the remaining
experts into VRAM. With `-ngl 99 --load-mode none` and no `-ot`, experts stay in `CPU_REPACK`
regardless of `--n-cpu-moe`. An earlier revision of this document claimed `--n-cpu-moe` gave
"correct GPU expert placement to 20 blocks" — that was wrong: those arms never had GPU experts at
all (1037 MiB VRAM), which is exactly why their decode was flat. `-ot` is the only mechanism that
produces a CUDA expert buffer here.

### The 11-block ceiling is real, and 12+ is corrupt

| placement | VRAM at load | result |
|---|---|---|
| `-ot` blk 0-10 | 7055 MiB | correct |
| `-ot` blk 0-11 | 7608-7726 MiB | **garbage** |
| `-ot` blk 0-11, ctx 2048, KV on GPU | 7726 MiB | **garbage** |
| `-ot` blk 1-11 | - | **garbage** |
| `-ot` blk 0-9 + 11 | - | **garbage** |

Not VRAM pressure (7726 MiB of 11830 used), not KV placement, not context size, not CUDA graphs
(`GGML_CUDA_DISABLE_GRAPHS=1`), not op offload (`--no-op-offload`), not the CPU-lane accounting
(`--moe-cache-cpu-share 0`). It reproduces at every context size tested and is the current
open bug. 11 blocks is the shipping ceiling until it is fixed.

### Decode is CPU-expert bound, and the mode layer is not the bottleneck

Decode is flat at ~8.7-9.1 tok/s across every axis tested on a clean GPU:

- MoE mode: plain 8.83, `default` 8.96, `strata` 9.03, `ngram-cache` 8.73
- `--moe-profile auto` on vs off: 9.03 vs 8.59 (noise)
- context 4096 / 16384 / 32768: 8.88 / 9.03 / 8.71
- threads 8 / 12 / 16: 8.79 / 8.32 / 7.44 (E-cores hurt quantized dot products)
- KV on host vs GPU: 8.13 vs 7.87

With 13 of 24 expert blocks on the CPU, ~208 expert computes per token (~2.2 GFLOP) is the
bound. The lever that would move it is putting more expert blocks in VRAM — blocked by the
`-ot` corruption above 11 — or faster CPU expert kernels. No cache, profile, context, thread or
speculation setting moves it, so tuning this axis is finished.

## Negative results (measured, do not re-litigate)

- VRAM expert bank (`moe-cache-mib`): measured three separate times on a clean GPU — 0 / 3000 /
  4500 / 5500 MiB all land at PP 45.2-46.6, TG 7.9-8.1. Neutral. It does not add expert residency
  that the `-ot` prefix does not already provide.

- `mlock` of expert pages: impossible, `ulimit -Hl` = 8192 KiB.
- Faster NVMe (1.8 -> 5.7 GB/s): cold load 116 s both, decode unchanged. Not IO-bound.
- Page-cache warmer, host RAM arena, VRAM expert bank: neutral or negative here.
- `ngram-simple` speculation: 7.75 vs 7.89 tok/s.
- Fused `gate_up` IQ4_NL experts on CUDA are **correct**; the old "no kernel, silent corruption"
  claim was a VRAM-exhaustion artifact.

## `-ngl` is inert for this model; `-ot` is the only VRAM path

Measured memory breakdown at `-ngl 99 --load-mode none`, no `-ot` (`-lv 4`):

```
load_tensors: offloaded 25/25 layers to GPU
load_tensors:          CPU model buffer size =  5392.31 MiB
load_tensors:   CPU_REPACK model buffer size = 10935.00 MiB
load_tensors:        CUDA0 model buffer size =     3.28 MiB
load_tensors:    CUDA_Host model buffer size =   469.31 MiB
```

The GPU holds **3.28 MiB** while the log claims every layer is offloaded. `-ngl` does not place
weight bytes on the card for `inkling2_loop`; only a tensor override does (`-ot blk 0-10` gives
`CUDA0 model buffer size = 6017.53 MiB`, exactly 11 x 547 MiB). This is why experts, attention and
dense FFN all compute on the CPU, and why decode is insensitive to expert placement.

## The remaining ~23% is real but blocked by the CUDA backend

Forcing non-expert tensors to CUDA0 raises decode materially, and every variation is *incorrect*:

| override added to the 11 expert blocks | VRAM | TG | result |
|---|---|---|---|
| none (shipping) | 7056 MiB | 8.67 | correct |
| `blk\.\d+\.attn_.*` | 9353 MiB | **10.99** | garbage |
| `blk\.\d+\.attn_.*` + dense ffn | 9353 MiB | **11.08** | garbage |
| `blk\.\d+\.attn_q\.weight` | 7235 MiB | 9.17 | garbage |
| `blk\.\d+\.attn_(q|k|v|output)` | 7519 MiB | 9.87 | garbage |
| whole blocks 0-2 | 8125 MiB | 8.94 | garbage |
| whole blocks 0-5 | 9360 MiB | 9.09 | garbage |
| `attn_*` under `--load-mode mmap` | 9362 MiB | 10.93 | garbage |

Even a single `attn_q` tensor on CUDA corrupts output, under both load modes, so this is not a
repack-layout artefact. Decode is bound by CPU-side attention/mixer matmuls (the arch is recurrent:
decode is flat across ctx 4K/16K/32K), and those weights cannot legally move to the GPU on this
build. Recovering the 23% requires a correct CUDA path for the `inkling2_loop` mixer ops - the same
thing Strata ships as custom kernels. No flag combination substitutes for it.

## Root cause of inert -ngl, and the fix (commit efcfbb6a80889)

`weight_buft_supported()` probes a candidate buffer type by allocating a temporary
zero-sized buffer. `ggml_backend_buft_alloc_buffer(buft, 0)` returns a *dummy*
buffer with an empty iface, so the CUDA host-weight guard

```cpp
if (a->buffer != nullptr && !ggml_backend_buffer_is_cuda(a->buffer)) return false;
```

evaluated `false` for the probe (it tests `iface.free_buffer`, which is null on the
dummy) and rejected CUDA for every tensor. `select_weight_buft()` then fell through
to CPU, producing:

```
load_tensors: offloaded 25/25 layers to GPU
load_tensors:  CPU model buffer size =  5392.31 MiB
load_tensors:  CPU_REPACK buffer size = 10935.00 MiB
load_tensors: CUDA0 model buffer size =     3.28 MiB
```

Comparing the buffer *type* (`ggml_backend_buft_is_cuda(a->buffer->buft)`) fixes the
probe. Attention and dense weights now land in VRAM and execute there.

Ship configuration for Inkling (all three presets): `gpu-layers 99`, `n-cpu-moe 24`,
`ubatch-size 64`, `load-mode none`, `moe-cache-mib 0`, no `override-tensor`.

| measurement (ctx 180000, router) | before | after |
|---|---|---|
| warm decode tok/s | ~8.0 | **11.85** |
| long-prompt decode tok/s | 7.63 | **10.65** |
| long-prompt PP tok/s (1541 tok) | 45.03 | 42.89 |
| long-prompt PP tok/s (ctx 16k, ub 512) | 45.03 | **58.80** |
| VRAM | 7231 MiB | 4522 MiB |
| `17*23` | correct | correct |

`ubatch-size 64` is required at 180K: without flash attention for this architecture the
prefill buffer grows with ubatch (17.6 GiB at ub 512), so a smaller ubatch is what keeps
long context inside 12 GB.

## Experts must stay on the CPU

Every attempt to place expert blocks in VRAM now corrupts output, with attention either
on the GPU or on the CPU:

| arm | VRAM | TG | result |
|---|---|---|---|
| `n-cpu-moe 24` (all experts CPU) | 4756 | 12.80 | correct |
| `n-cpu-moe 20` | 4758 | 12.70 | correct (after Q8_0 guard) |
| `n-cpu-moe 16` | 6945 | 13.60 | garbage |
| `n-cpu-moe 12` | 9132 | 15.50 | garbage |
| `n-cpu-moe 16` + attn forced to CPU | — | 9.00 | garbage |

Routed Q8_0 experts are refused on CUDA explicitly (guard in `ggml-cuda.cu`), which is
what makes `n-cpu-moe 20` correct. IQ4_NL routed experts are still wrong once more than
the 20-layer CPU prefix is involved, so expert weights stay host-side.

## Same fix on Qwen3.8-Flash-Next

The placement-probe fix moved Qwen's attention and dense weights to the GPU as well.
Presets now pin experts to the CPU with `n-cpu-moe 48` (the router's coverage estimator
otherwise refuses the model: "CUDA0 needs 61763 MiB but 10849 MiB is free").

| measurement (router, ctx 32768, MTP draft) | before | after |
|---|---|---|
| warm decode tok/s | ~5.4-6.4 | **10.64** |
| long-prompt PP tok/s (1541 tok) | ~10-22 | **56.52** |
| cold load | ~110 s | 109 s |
| `17*23` | correct | correct |
| VRAM | 7231-8972 MiB | 8972 MiB |


## CRITICAL: quantized KV cache types corrupt generations (verified 2026-11-25)

Any **legacy/quantized KV type corrupts output in this fork**; `f16` and `q8_0` are correct.

Back-to-back, same binary, same model (`Inkling-2-Mini-IQ4_NL`), same flags
(`-c 4096/8192/32768/180000 -ngl 10 --load-mode mmap --flash-attn on --kv-unified`,
greedy, prompt `Q: What is 17*23? A:`):

| KV type | KV location | answer | behaviour |
|---|---|---|---|
| `f16` | host | `391` correct | normal |
| `q8_0` | host | `391` correct | normal |
| `q4_0` | host **and** GPU | wrong | degenerate loop `17*23= 17*23= ...` |
| `q5_1`/`q4_1` | GPU | wrong | same degenerate loop |

Not caused by: `--kv-unified`, `--cache-prompt`/`--cache-ram`, `--jinja`,
thread geometry (`8/8`, `10/16`), context size (4K–180K), or endpoint
(`/completion` and `/v1/completions` both reproduce).

`llama-cli` hides the bug: its default sampling (temp 0.8 / top-k 40) masks the
corrupted logits. The same `llama-cli` invocation with `--temp 0` is **correct**,
so the server-path difference is batch/KV layout, not the CLI.

Reference check: Unsloth's `llama.cpp` binary at identical settings and `-ctk q4_0 -ctv q4_0`
answers `391` correctly, so this is a fork-side regression, not an upstream property.

**Operational rule until the kernel bug is fixed:** use `cache-type-k = q8_0`,
`cache-type-v = q8_0` everywhere. Cost at 180K context on Qwen3.8-Flash-Next is
~220 MiB VRAM (measured with `llama-fit-params`: q4_0 284 MiB vs q8_0 504 MiB context),
i.e. free correctness.

### Upstream parity after the KV fix (same preset, same machine)

Inkling-2-Mini, `-c 180000 -b 2048 -ub 128 -t 8 -tb 8 -ngl 10 --load-mode mmap`,
KV `q8_0` on host, greedy:

| binary | PP tok/s | TG tok/s | correct |
|---|---|---|---|
| Unsloth | 22.94 / 17.72 | 8.84 / 9.00 | yes |
| this fork (router) | 21.92 / 20.70 | 9.71 / 8.82 | yes |

Parity within run-to-run noise. Qwen3.8-Flash-Next (q8_0 KV, `spec-type = draft-mtp`):
PP 6.70, TG 5.25 warm, correct.
