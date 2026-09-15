---
title: KV Cache Residency and Concurrency Audit
type: note
tags: project-llama.cpp-optium,architecture,testing,handoff
---

# Context
This audit examines the bounded KV residency and concurrency mechanisms in llama.cpp-optium to determine gaps in supporting >=32 logical slots with parallel4 slots each capable of using 262144 tokens without full preallocation per slot. The analysis focuses on RAM/disk spill mechanisms, target/draft/hybrid state integrity, and backpressure handling.

# What Was Done
Reviewed the following source files:
- AGENTS.md (contribution guidelines)
- src/llama-kv-cache.h/.cpp (KV cache implementation)
- src/llama-kv-cells.h (KV cell structure)
- src/llama-context.h (context structure)
- src/llama-memory.h (memory interface)
- Searched for tiering, swap, offload, and resize mechanisms in the codebase.

# Findings
## Current Implementation
1. **KV Cache Structure**:
   - Uses fixed-size ring buffers (`llama_kv_cells`) per stream.
   - Streams count: `n_stream = unified ? 1 : n_seq_max` (line 85, llama-kv-cache.cpp).
   - Each stream's buffer is preallocated to fixed size `kv_size` at construction (line 145).
   - K/V tensors are allocated as contiguous 3D tensors: `[n_embd_k_gqa, kv_size, n_stream]` (lines 234-235).

2. **Logical Slots (Sequences)**:
   - Maximum sequences: `n_seq_max` (constructor arg, line 76), limited by `LLAMA_MAX_SEQ=256` (llama-cparams.h).
   - In non-unified mode (`unified=false`), each sequence gets a dedicated stream of size `kv_size`.
   - In unified mode (`unified=true`), all sequences share a single stream of size `kv_size`.

3. **Memory Allocation**:
   - Full preallocation of KV cache memory occurs at initialization:
     - Total K/V memory = `2 * kv_size * n_stream * (size_per_element)`.
     - No dynamic resizing or tiered storage (VRAM→RAM→disk) mechanisms exist.
   - The `offload` parameter only controls device placement (CPU/GPU), not tiered spill.

4. **Concurrency & Backpressure**:
   - `find_slot()` method searches for free cells in the ring buffer (lines 903-1100).
   - Returns empty `slot_info` on failure, propagating to `init_batch()` returning `LLAMA_MEMORY_STATUS_FAILED_PREPARE` (line 738).
   - Provides backpressure when KV cache is full but does not support graceful degradation via spill.

5. **Target/Draft/Hybrid State**:
   - Unified KV cache (`unified=true`) shares cells between two caches (target/draft) via `mem_other` parameter (line 81).
   - No versioning or hybrid state tracking for progressive cache updates.

6. **Gap Analysis for Requirements**:
   - **≥32 slots**: Supported via `n_seq_max` (max 256).
   - **Parallel4 slots @ 262144 tokens/slot without full preallocation**:
     - Non-unified mode with `n_seq_max=4`, `kv_size=262144` requires preallocating `4 * 262144` tokens worth of KV cache upfront.
     - Unified mode with `n_seq_max=4`, `kv_size=262144` shares 262144 tokens across all slots (insufficient for independent 262144-token contexts).
     - **Gap**: No mechanism to allocate KV cache per slot on-demand up to 262144 tokens without preallocating the full amount.
   - **RAM/Disk Spill**: Absent; KV cache is fixed-size and resident in a single backend buffer (CPU/GPU).
   - **State Integrity**: Target/draft sharing works but lacks versioning for hybrid states (e.g., merge conflicts).
   - **Backpressure**: Present via slot allocation failure but no spill-to-disk fallback.

## Code-Specific Observations
- KV cell metadata (`llama_kv_cells`) tracks per-cell sequence occupancy and positions but assumes fixed buffer size.
- No existing swap/paging layer; `llama-context.cpp` implements output swapping but not KV cache.
- `seq_rm`/`seq_add` operations enable sequence eviction but do not spill evicted data to disk.

# Fix / Solution
## Recommended Implementation Slices
To achieve the goals with minimal, safe changes:

### Slice 1: Per-Slot Dynamic KV Allocation (Non-Unified Mode)
**Files**: `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`
**Changes**:
1. Add optional `bool paged` constructor parameter (default `false`).
2. When `paged=true`:
   - Replace contiguous K/V tensor allocation with a page allocator (fixed-size pages, e.g., 512 tokens).
   - Maintain a free list of pages per stream.
   - On `find_slot()`, allocate pages on-demand for requested token count.
   - Track per-sequence page tables (map logical token → physical page).
3. Preserve existing API: `llama_kv_cache` interface unchanged; new behavior opt-in via constructor.
**Lines to modify**:
   - Constructor (lines 67-88): Add `paged` param, conditional allocation.
   - `get_k_storage()`/`get_k()`/`get_v()`: Indirect via page table.
   - `apply_ubatch()`: Allocate pages if needed before writing.
   - `find_slot()`: Check page availability instead of cell count.

### Slice 2: Tiered Storage Spill (VRAM→RAM→Disk)
**Files**: New file `src/llama-kv-cache-paged.cpp` (or extend Slice 1)
**Changes**:
1. Implement a three-tier page pool:
   - Tier 0: VRAM (GPU-backed GGML buffers)
   - Tier 1: RAM (CPU-backed GGML buffers)
   - Tier 2: Disk (memory-mapped files or swap space)
2. On allocation attempt:
   - Try Tier 0 → Tier 1 → Tier 2.
   - On eviction, move least-recently-used pages to lower tiers.
3. Add LRU tracking per page (timestamp or access counter).
**Integration**:
   - Extend Slice 1's page allocator to use tiered pools.
   - No API changes; transparent to existing code.

### Slice 3: Hybrid State Versioning (Target/Draft)
**Files**: `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`
**Changes**:
1. Add `uint64_t version` to `llama_kv_cache` (increment on mutation).
2. For unified caches, store version per sequence to detect stale reads.
3. On `state_read`/`state_write`, include version in checkpoint.
**Lines**:
   - Add version field near line 230.
   - Increment in `apply_ubatch()`, `seq_*()` methods.
   - Update `state_write()`/`state_read()` to handle version.

## Safety & Compatibility
- All changes are opt-in (via new constructor params or features).
- Existing unified/non-unified behavior preserved when new params are defaulted.
- No changes to `llama_memory_i` interface; maintains pluggability.
- Leverages existing GGML backend abstraction for tiered buffers.

# Notes / Gotchas
- **Page Size Trade-off**: Smaller pages reduce internal fragmentation but increase page table overhead. Recommend 512-token pages (tested in similar systems).
- **TLB Pressure**: Paged access may increase latency; consider hardware prefetching hints.
- **Disk Spill Performance**: Ensure disk backend is asynchronous to avoid blocking inference.
- **Unified Mode Interaction**: Paged allocation in unified mode requires careful sharing semantics (consider copy-on-write for target/draft).
- **Testing**: Unit tests for page allocation, spill, and recovery should be added to `tests/` (but per instructions, not run here).

# Handoff
This audit informs the parent task's implementation plan. The recommended slices are ordered by dependency:
1. Slice 1 enables dynamic per-slot allocation (foundation).
2. Slice 2 adds tiered spill (builds on Slice 1).
3. Slice 3 enhances hybrid state (orthogonal, can be done independently).

Refer to [[index]] for related architecture notes and [[current-state]] for baseline KV cache behavior.
Parent should verify that existing models.ini and concurrency experiments remain compatible with opt-in features.