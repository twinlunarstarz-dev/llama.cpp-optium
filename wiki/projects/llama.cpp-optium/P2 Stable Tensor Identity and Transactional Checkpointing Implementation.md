---
title: "P2 — Stable Tensor Identity & Transactional Checkpointing Implementation"
type: implementation
tags:
  - p2
  - tensor-identity
  - checkpoint
  - sha256
  - openssl
  - resumable-training
  - llama-cpp-optium
source:
  - codebase
  - obsidian
status: complete
tests_passed: "32 tests, 7115 assertions, 0 failures"
created: 2026-08-04
---

# P2 — Stable Tensor Identity & Transactional Checkpointing

## Overview

P2 implements stable tensor identity (SHA-256 based) and transactional resumable checkpointing for the llama.cpp-optium fine-tuning pipeline. P1 (quantized training primitives) independently passed QA before P2 implementation began.

**Scope:** Strictly P2 only. No P3 (GGUF export), P4 (CUDA kernels/backward), P5 (coordinate engine/context integration), or P6 (CLI/multi-GPU/e2e) work included.

## Files Created/Modified

| File | Lines | Purpose |
|------|-------|---------|
| [`src/llama-train-identity.h`](src/llama-train-identity.h) | 174 | Stable tensor identity API — `tensor_id`, `sha256_context`, `compute_tensor_id()`, `tied_group`, `validate_tied_groups()` |
| [`src/llama-train-identity.cpp`](src/llama-train-identity.cpp) | 315 | OpenSSL EVP SHA-256 implementation, canonical little-endian encoding, tied group validation |
| [`src/llama-train-state.h`](src/llama-train-state.h) | 273 | External optimizer state types — `tensor_state`, `dtype_matrix`, `session_cursors`, `train_state` class |
| [`src/llama-train-state.cpp`](src/llama-train-state.cpp) | 374 | State allocation with overflow checks, cursor validation, LR computation, fingerprint helpers |
| [`src/llama-train-checkpoint.h`](src/llama-train-checkpoint.h) | 242 | Transactional checkpoint interface — `ckpt_error`, `checkpoint_config`, `manifest_entry`, `checkpoint_store` |
| [`src/llama-train-checkpoint.cpp`](src/llama-train-checkpoint.cpp) | 1655 | Little-endian binary serialization, JSON manifest with cursors + shuffle_rng_state[8], atomic save/reload protocol, failure injection support |
| [`tests/test-train-state-checkpoint.cpp`](tests/test-train-state-checkpoint.cpp) | 1406 | 32 comprehensive tests with failure injection, resume equivalence, corruption detection |
| [`src/CMakeLists.txt`](src/CMakeLists.txt) | modified | Added P2 source files + OpenSSL Crypto linkage to llama library |
| [`tests/CMakeLists.txt`](tests/CMakeLists.txt) | modified | Added `test-train-state-checkpoint` target |

## Key Technical Decisions

### SHA-256 via OpenSSL EVP API
- **Decision:** Use OpenSSL EVP API rather than self-contained implementation.
- **Rationale:** Initial self-contained SHA-256 had fundamental algorithmic errors (producing incorrect hashes vs RFC 6234 test vectors). OpenSSL EVP is cryptographically verified and maintained.
- **Verification:** SHA-256("abc") = `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad` ✓
- **Build:** Links `OpenSSL::Crypto` via CMake when `LLAMA_OPENSSL=ON` (default).

### Canonical Encoding Format
- **String:** LE uint32 length + UTF-8 bytes
- **Type:** LE uint32 of `ggml_type` enum value
- **Shape:** LE uint32 rank + each dimension as LE int64_t
- **Tensor ID:** SHA-256(string_encoding || type_encoding || shape_encoding)

### is_valid() Bug Fix
- **Bug:** Original `is_valid(tensor_id)` used `std::all_of(b != 0)` which rejected any hash containing a zero byte. Valid SHA-256 hashes frequently contain zero bytes.
- **Fix:** Changed to `std::any_of(b != 0)` — only rejects the all-zeros sentinel (default-constructed ID).
- **Impact:** This was causing `p1_resume_equivalence` test segfault and `identity_dimension_type_diff` / `duplicate_tensor_id` failures.

### Transactional Checkpoint Protocol
1. Write shard data + manifest into `__staging_<rand>/` directory
2. fsync staging directory
3. Atomic rename to `<base>/<generation>/`
4. fsync parent directory (COMMIT)
5. On crash: staging directory left orphaned; safe to clean up on next save

### Binary Shard Format
- **File header:** Magic "CNRT" (0x54524E43 LE) + version (uint32 LE)
- **Records:** Per-tensor `shard_record_header` (magic + version + tensor_id_hex + offset + size) followed by raw binary data
- **Loading:** Uses `std::ifstream::read()` for direct binary reads into pre-allocated buffers
- **Manifest:** JSON with schema_version, generation, source_fingerprint, tensor_count, optimizer_step, cursors (all fields including shuffle_rng_state[8] array), shards (array of {path, size, sha256})

### Session Cursors (Complete Resumability)
- `global_optimizer_step` — optimizer step counter
- `micro_step` — position within gradient accumulation cycle
- `grad_accum_steps` — gradient accumulation steps
- `sr_counter` — stochastic rounding counter for deterministic replay
- `epoch_step`, `sample_step`, `token_step`, `shard_step` — data pipeline cursors
- `base_lr`, `min_lr`, `lr_warmup_steps`, `lr_schedule_kind` — LR schedule state
- `loss_scale` — dynamic loss scaling state
- `rng_state` — primary RNG state
- `committed_generation` — last committed checkpoint generation
- `shuffle_rng_state[8]` — SplitMix64 shuffle RNG state (8 x uint64_t)

## Test Results

### P2 Test Suite — [`tests/test-train-state-checkpoint.cpp`](tests/test-train-state-checkpoint.cpp)
```
tests      : 32
assertions : 7115
failures   : 0
```

**Test Categories:**
- **Identity stability** — SHA-256 produces stable IDs independent of graph pointers/allocation order
- **Canonical encoding** — string, type, shape encoding correctness
- **Tied-weight aliasing** — compatible ties accepted, incompatible rejected
- **Dtype matrix validation** — param/grad/residual/optimizer dtype checks
- **Overflow detection** — byte sizing overflow rejection
- **Session state roundtrip** — cursors serialize/deserialize correctly
- **Invalid cursor combinations** — micro_step > grad_accum_steps rejected
- **Checkpoint empty roundtrip** — zero-tensor checkpoint save/load
- **Multi-tensor roundtrip** — multiple tensors with different dtypes
- **Exact optimizer state** — FP32 optimizer moments preserved byte-exact
- **RNG/cursor preservation** — all cursor fields including shuffle_rng_state[8] preserved across save/load
- **Fingerprint mismatch detection** — source model fingerprint mismatch rejected
- **Schema version mismatch** — future schema versions rejected gracefully
- **Corrupt data detection** — appended garbage byte detected via size mismatch
- **Missing COMMIT detection** — incomplete checkpoint rejected
- **Path traversal defense** — `../` in shard paths rejected
- **Oversized declaration** — manifest size > actual file detected
- **No partial mutation** — failed save doesn't corrupt existing checkpoint
- **Interrupted save** — staging cleanup on interrupted operations
- **Staging cleanup safety** — orphaned staging directories handled safely
- **Symlink defense** — symlink targets rejected for shard paths
- **P1 resume equivalence** — checkpoint/resume produces identical results to uninterrupted run (full error-feedback quantization replay)
- **SHA-256 correctness** — RFC 6234 test vectors verified

### P1 Regression Check — [`tests/test-train-quant.cpp`](tests/test-train-quant.cpp)
```
tests      : 22
assertions : 3717
failures   : 0
```
P1 tests pass unchanged, confirming P2 introduces no regressions.

## Build Configuration

```bash
cmake .. -DLLAMA_BUILD_TESTS=ON -DLLAMA_OPENSSL=ON
cmake --build . --target test-train-state-checkpoint -- -j$(nproc)
./bin/test-train-state-checkpoint
```

OpenSSL linkage added to `src/CMakeLists.txt`:
```cmake
if (LLAMA_OPENSSL)
    find_package(OpenSSL QUIET)
    if (OPENSSL_FOUND)
        target_link_libraries(llama PRIVATE OpenSSL::Crypto)
    endif()
endif()
```

## Critical Bugs Fixed During Implementation

1. **`is_valid()` zero-byte rejection** — `all_of(b != 0)` → `any_of(b != 0)` in [`src/llama-train-identity.h:48`](src/llama-train-identity.h:48)
2. **Missing OpenSSL linkage** — Added `OpenSSL::Crypto` to llama library targets
3. **Cursor parsing missing in `read_manifest()`** — Added JSON parsing for all cursor fields including nested `cursors` object and `shuffle_rng_state[8]` array
4. **CHECK_STR_EQ pointer comparison** — Fixed macro to use `std::string(a) != std::string(b)` instead of pointer inequality
5. **Path concatenation in tests** — Fixed `/tmp/p2_test_` + pid string concatenation
6. **Private member access** — Fixed `checkpoint_config` construction to use proper constructor

## Integration Points for P3-P6

- **P3 (GGUF export):** Use `tensor_id` as stable key for weight mapping; `source_fingerprint` verifies model integrity
- **P4 (CUDA backward):** `dtype_matrix` defines independent parameter/gradient dtypes for quantized gradient paths
- **P5 (Context integration):** `train_state` + `checkpoint_store` provide save/resume API for training loops
- **P6 (Multi-GPU):** `tied_group` validation ensures parameter tying correctness across devices; `session_cursors` coordinate distributed state

## Linked Notes

- [[P1 Quantized Training Primitives Implementation]] — P1 prerequisite, independently QA-passed
- Parent project: Stage 2/3 implementation (see Obsidian MCP for architecture notes)
