# KV Concurrency Audit Summary

## What was done
- Read-only audit of bounded KV residency and concurrency in llama.cpp-optium
- Analyzed source code for KV cache structures, allocation, and concurrency limits
- Identified gaps to meet requirements: >=32 logical slots, parallel4 slots @ 262144 tokens each without full preallocation
- Traced RAM/disk spill mechanisms, target/draft/hybrid state integrity, and backpressure
- Recommended minimal implementation slices with exact file/line references

## Key findings
1. Current implementation uses **fixed-size preallocation** of KV cache memory
2. Logical slots limited by `n_seq_max` (max 256 via `LLAMA_MAX_SEQ`)
3. **No tiered storage** (VRAM→RAM→disk) or dynamic per-slot allocation exists
4. Backpressure present via slot allocation failure but no spill-to-disk fallback
5. Unified KV cache shares cells between target/draft but lacks versioning

## Recommended solution slices
1. **Per-slot dynamic KV allocation** (Slice 1):
   - Files: `src/llama-kv-cache.h/.cpp`
   - Add optional `paged` constructor parameter for on-demand page allocation
   - Lines: Constructor (~L67), get_k/get_v (~L234), apply_ubatch (~L1102)

2. **Tiered storage spill** (Slice 2):
   - New file or extension of Slice 1
   - Implement VRAM→RAM→disk page pools with LRU eviction

3. **Hybrid state versioning** (Slice 3):
   - Files: `src/llama-kv-cache.h/.cpp`
   - Add version tracking to llama_kv_cache for target/draft integrity

## Artifact created
- `/opt/llama.cpp-optium/wiki/projects/llama.cpp-optium/kv-concurrency-audit.md` (YAML frontmatter note with full details)

## Verification
- File written and verified via write_file tool
- Content matches required format: Context, What Was Done, Findings, Fix/Solution, Notes/Gotchas, Handoff
- No source code modifications made (read-only audit as requested)