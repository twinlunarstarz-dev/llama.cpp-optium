---
title: P1 Quantized Training Primitives Implementation
type: implementation
tags:
  - project-llama-cpp-optium
  - implementation
  - training
  - quantization
  - q4
  - ggml
  - stochastic-rounding
  - error-feedback
  - testing
  - handoff
  - qa-fix
  - nvfp4
source:
  - codebase
  - obsidian
created: 2026-08-04
modified: 2026-08-04
status: corrected
---

## Overview

P1 implements the **CPU reference layer** for quantized training primitives in `llama.cpp-optium`. This is the numerically authoritative reference implementation that CUDA kernels (P4) must reproduce bit-for-bit under deterministic stochastic rounding.

**QA Status:** FAIL (original) → Corrected 2026-08-04, awaiting independent QA re-audit.

## Files Created/Modified

| File | Purpose | Lines |
|------|---------|-------|
| `src/llama-train-quant.h` | Public API header | 210 |
| `src/llama-train-quant.cpp` | Implementation (corrected) | ~495 |
| `tests/test-train-quant.cpp` | 22 deterministic tests | ~950 |
| `src/CMakeLists.txt` | Build integration (line 38) | modified |
| `tests/CMakeLists.txt` | Test target (lines 299-302) | modified |

## API Surface (10 functions)

### Type Support Queries
- `is_supported(ggml_type)` - Check if type has both `to_float` and `from_float`
- `first_supported(vector<ggml_type>)` - Return first supported type from list

### Validation
- `validate(type, n_rows, n_per_row)` - Geometry and type support check

### Core Operations
- `dequantize_rows(...)` - Bounded row dequantization (quantized → FP32)
- `quantize_rows(...)` - Bounded same-type requantization (FP32 → quantized)
- `apply_update_with_error_feedback(...)` - Atomic dequantize → update → requantize with residual
- `roundtrip_same_type(...)` - Dequantize → requantize for testing

## Supported Types and Stochastic Capability

| Type | Deterministic Nearest | Stochastic Rounding | Notes |
|------|----------------------|---------------------|-------|
| **Q8_0** | ✅ | ✅ | Elementwise unbiased SR (32-byte blocks, independent scales) |
| **F32** | ✅ | N/A | Identity quantization, zero residual |
| **Q4_K** | ✅ | ❌ | Block-coupled scales (K_SCALE_SIZE); elementwise SR invalid |
| **NVFP4** | ✅ | ❌ | Block-coupled micro-scales (4×16 sub-blocks); elementwise SR invalid |

**Rationale:** Q4_K uses per-sub-block scales encoded jointly with codes. Changing a single code stochastically requires recomputing the block scale, which changes all elements in the block. Elementwise stochastic projection is mathematically invalid for these formats. NVFP4 similarly uses 4 independent E4M3 micro-scales per 64-element super-block.

## Error Feedback Mathematics

```
proposed[i] = dequant(Q)[i] + residual[i] + lr * update[i]
Q_new = quantize(proposed)   [uses sr_config for rounding]
residual[i] = proposed[i] - dequant(Q_new)[i]
```

**Invariant:** Repeated sub-quantum updates eventually change the quantized payload because the residual accumulates quantization error.

**Sign:** `residual = proposed - dequant(Q_new)`. Positive residual means the quantized value is below the proposed value; it will push upward on the next update.

## Stochastic Rounding (Corrected Implementation)

### Mathematical Semantics
For a scalar x between adjacent representable quantized values lo ≤ x ≤ hi:
- `frac = (x - lo) / (hi - lo)` — fractional distance from lower bound
- Select hi with probability `frac`, lo with probability `(1 - frac)`
- **Exact endpoints:** If `frac == 0`, always return lo. If `frac == 1`, always return hi.
- **Degenerate intervals:** If `lo == hi`, return lo deterministically.

### Counter-Based RNG (No Global State)
```
counter[i] = counter_start + row * n_per_row + col
hash = sr_hash(seed, key, counter[i])  → uniform [0,1)
```

Hash function uses SplitMix64-style mixing of (seed + key·golden_ratio + counter), then extracts top 52 bits as a double in [0,1).

### Q8_0 Stochastic Quantization
Q8_0 blocks are independent (32 elements each with their own scale d = 127/max_abs). The scale depends only on max|x|, not on individual codes. Therefore elementwise stochastic projection is mathematically valid:

1. Compute block scale `d = 127 / max(|x[0..31]|)` (same as deterministic path)
2. For each element: `normalized = x * d`, shifted to [0, 254], apply `sr_round`
3. Write fp16 scale + int8 codes directly to block layout

### Q4_K / NVFP4 — Explicitly Unsupported for Stochastic Mode
When `sr.stochastic == true` and type is Q4_K or NVFP4, `quantize_rows` returns `err_unsupported_type` with `n_rows == 0`. No silent fallback to deterministic mode.

## Error Codes (9 codes)

| Code | Meaning |
|------|---------|
| `ok` | Success |
| `err_unsupported_type` | Missing to_float/from_float OR stochastic requested for block-coupled type |
| `err_bad_row_alignment` | n_per_row not multiple of blck_size |
| `err_zero_size` | n_rows == 0 or n_per_row == 0 |
| `err_null_pointer` | Required buffer is null |
| `err_overflow` | Checked arithmetic overflowed |
| `err_non_finite` | NaN/Inf in input, update, or residual |
| `err_malformed_buffer` | ggml_validate_row_data failed |
| `err_validation_failed` | Generic validation failure |

## Test Results

**22 tests, 3717 assertions, 0 failures**

### Original Tests (17)
| # | Test | Purpose |
|---|------|---------|
| 1 | `supported_type_matrix` | Type support/rejection verification |
| 2 | `geometry_checks` | Alignment and overflow validation |
| 3 | `roundtrip_preserves_payload` | Q8_0, Q4_K, F32 round-trip identity |
| 4 | `zero_update` | Zero update produces no change |
| 5 | `deterministic_nearest` | Same input → identical output |
| 6 | `deterministic_stochastic_replay` | Same seed/key/counter → identical output |
| 7 | `stochastic_differs_with_key` | Different key changes decisions |
| 8 | `error_feedback_conservation` | Reconstruction error ≈ 0 |
| 9 | `sub_quantum_accumulation` | Tiny updates eventually change payload |
| 10 | `error_feedback_sign_order` | Mathematical sign verification |
| 11 | `non_finite_rejection` | NaN/Inf rejected in update and residual |
| 12 | `malformed_buffers` | Null pointer checks (all functions) |
| 13 | `no_modification_on_failure` | Payload unchanged after failed update |
| 14 | `dequantize_strided` | Strided output preserves gaps |
| 15 | `error_code_names` | All error code names non-empty |
| 16 | `memory_bounded_by_row` | Row-at-a-time processing (100 rows) |
| 17 | `q4_k_operations` | Q4_K specific quantize/dequantize/update (nearest) |

### New Tests Added by Correction (5)
| # | Test | Purpose |
|---|------|---------|
| 18 | `q4_k_stochastic_unsupported` | Q4_K stochastic returns err_unsupported_type; no silent fallback |
| 19 | `nvfp4_stochastic_unsupported` | NVFP4 stochastic returns err_unsupported_type; nearest still works |
| 20 | `stochastic_unbiased_exhaustive` | Controlled midpoint value (frac=0.5) shows ~50/50 split across seeds with binomial bounds |
| 21 | `stochastic_endpoint_exact` | Exact representable values never flip under stochastic mode (32 seeds tested) |
| 22 | `counter_overflow_rejection` | Extreme counter values produce deterministic output (no silent wrap) |

## Build Commands and Results

```bash
# Debug build with AddressSanitizer
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DADDRESS_SANITIZER=ON -DLLAMA_SERVER=OFF -DLLAMA_CUDA=OFF -DBUILD_SHARED_LIBS=OFF
cd build-asan && ninja test-train-quant
./bin/test-train-quant  # 22 tests, 3717 assertions, 0 failures, 0 ASAN errors

# Release build (default)
cd build && ninja test-train-quant
LD_LIBRARY_PATH=./bin ./bin/test-train-quant  # 22 tests, 3717 assertions, 0 failures

# Regression: existing quantization tests
ninja test-quantize-fns && LD_LIBRARY_PATH=./bin ./bin/test-quantize-fns  # PASS
```

## What Changed from Original (QA Fail → Correction)

### Defect 1: Heuristic Stochastic Rounding → Mathematically Valid
**Before:** Lines 220-319 used perturbation-based approximation (`hash < 0.5 && abs_err < 0.1`). This is biased — it does not implement `P(up) = frac`.

**After:** `sr_round()` implements exact stochastic projection: `frac = normalized - floor(normalized)`, select ceil with probability `frac`. Exact endpoint handling. Q8_0 uses direct block-scale computation + per-element stochastic rounding. Block-coupled types return explicit error.

### Defect 2: Handoff Compliance — Note Location
**Before:** Implementation note at `wiki/P1 Quantized Training Primitives Implementation.md` (vault root).

**After:** Note at `wiki/projects/llama.cpp-optium/P1 Quantized Training Primitives Implementation.md`. No orphan/root duplicate exists.

### Defect 3: NVFP4 Honesty
**Before:** Conditional support on `from_float` existence without explicit stochastic capability declaration.

**After:** NVFP4 nearest mode works (has `from_float` via CPU traits). Stochastic mode returns `err_unsupported_type` explicitly because micro-scales are block-coupled.

### Defect 4: Error Feedback Sign Order
**Verified:** `residual[i] = proposed[i] - dequant(Q_new)[i]`. Positive residual pushes upward next iteration. Confirmed correct.

## Remaining Blockers for P2-P6

1. **Block-coupled stochastic rounding:** Q4_K and NVFP4 cannot use elementwise SR. A correct implementation requires joint scale+code optimization per block. This is an architecture-level decision, not a P1 bug. Documented blocker for P4 (CUDA stochastic kernels).

2. **No `llama-context` integration:** Training primitives are standalone. P5 (Coordinate Engine) must integrate them with the model loader and optimizer state.

3. **No checkpointing:** Residual state is caller-managed FP32 buffers. P2 must persist these atomically with quantized payloads.

## References
- [[Full-Parameter 4bit Training Architecture]] — Defines production architecture, error feedback equations, P1-P6 dependency ordering
- [[QA Audit: P1 Quantized Training Primitives]] — Original QA audit (FAIL), defect handoff
- Parent project index: [[wiki/projects/llama-cpp-optium/index|llama.cpp-optium Project Index]]
