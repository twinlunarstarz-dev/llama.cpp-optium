#!/usr/bin/env python3
from pathlib import Path
import re


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {n}: {old[:100]!r}")
    p.write_text(s.replace(old, new, 1))


def sub_once(path: str, pattern: str, repl: str) -> None:
    p = Path(path)
    s = p.read_text()
    out, n = re.subn(pattern, repl, s, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one regex match, found {n}: {pattern[:100]!r}")
    p.write_text(out)


# Public scheduler knob: split the total sequential VRAM window into a bounded
# persistent-resident tier and transient current/next pipeline space.
replace_once(
    "ggml/include/ggml-backend.h",
    """    GGML_API void                 ggml_backend_sched_set_weight_residency(\n            ggml_backend_sched_t sched, ggml_backend_t backend, bool enabled);\n\n    // Keep resident host-weight copies across scheduler graph resets. The next\n""",
    """    GGML_API void                 ggml_backend_sched_set_weight_residency(\n            ggml_backend_sched_t sched, ggml_backend_t backend, bool enabled);\n\n    // Cap persistent sequential weight residency on a backend. Transient current/next\n    // split buffers may still use the rest of the weight window, and oversized single\n    // tensors may evict residents and temporarily borrow the full window.\n    GGML_API void                 ggml_backend_sched_set_weight_residency_budget(\n            ggml_backend_sched_t sched, ggml_backend_t backend, size_t budget_bytes);\n\n    // Keep resident host-weight copies across scheduler graph resets. The next\n""",
)

replace_once(
    "ggml/src/ggml-backend.cpp",
    """    bool residency_enabled[GGML_SCHED_MAX_BACKENDS];\n    bool persistent_weight_residency;\n""",
    """    bool residency_enabled[GGML_SCHED_MAX_BACKENDS];\n    size_t residency_budget[GGML_SCHED_MAX_BACKENDS];\n    bool persistent_weight_residency;\n""",
)

replace_once(
    "ggml/src/ggml-backend.cpp",
    """    for (int b = 0; b < n_backends; ++b) {\n        sched->transient_metrics.backends[b].backend_index = b;\n        sched->transient_metrics.backends[b].backend = backends[b];\n    }\n""",
    """    for (int b = 0; b < n_backends; ++b) {\n        sched->transient_metrics.backends[b].backend_index = b;\n        sched->transient_metrics.backends[b].backend = backends[b];\n        // Preserve legacy behavior until a sequential context explicitly partitions\n        // the weight window into resident/current/next tiers.\n        sched->residency_budget[b] = SIZE_MAX;\n    }\n""",
)

# Replace the resident-space policy. Dense sequential weights are a cyclic scan:
# evicting an equal-frequency resident for every new layer degenerates into cache
# pollution. Dense admissions therefore stop at the configured resident budget;
# MoE/expert entries retain replacement behavior. Active transient execution has
# a separate escape hatch that may evict residents when required for correctness.
sub_once(
    "ggml/src/ggml-backend.cpp",
    r"static bool ggml_backend_sched_make_resident_space\(\n        ggml_backend_sched_t sched, int backend_id, size_t request\) \{.*?\n\}\n\nstatic bool ggml_backend_sched_grow_expert_slab",
    r'''static ggml_backend_sched_resident_map::iterator ggml_backend_sched_resident_victim(
        ggml_backend_sched_t sched, int backend_id) {
    auto victim = sched->residents->end();
    for (auto it = sched->residents->begin(); it != sched->residents->end(); ++it) {
        if (it->second.backend_id != backend_id || it->second.executing) {
            continue;
        }
        if (victim == sched->residents->end() || it->second.frequency < victim->second.frequency ||
                (it->second.frequency == victim->second.frequency &&
                 it->second.completed_use < victim->second.completed_use)) {
            victim = it;
        }
    }
    return victim;
}

static bool ggml_backend_sched_make_resident_space(
        ggml_backend_sched_t sched, int backend_id, size_t request, bool scan_resistant) {
    const size_t window = sched->weight_window_limit[backend_id];
    const size_t resident_budget = std::min(window, sched->residency_budget[backend_id]);
    if (!sched->weight_window_configured[backend_id] || !sched->weight_window_memory_valid[backend_id] ||
            request == 0 || request > resident_budget) {
        return false;
    }

    while (true) {
        const auto & row = sched->transient_metrics.backends[backend_id];
        const bool budget_ok = row.current_resident_bytes <= resident_budget - request;
        bool unknown = false;
        bool live_rejected = false;
        const bool window_ok = budget_ok && ggml_backend_sched_weight_window_admit(
            sched, backend_id, request, &unknown, &live_rejected);
        if (window_ok) {
            return true;
        }
        if (unknown || scan_resistant) {
            // Dense transformer weights are revisited in the same long cyclic scan.
            // Once the dedicated resident tier is full, retaining its current subset
            // produces real hits; replacing equal-frequency entries produces LRU scan
            // thrash. The active transient path below can still evict this subset when
            // a larger current split actually needs the bytes to make progress.
            return false;
        }

        auto victim = ggml_backend_sched_resident_victim(sched, backend_id);
        if (victim == sched->residents->end()) {
            return false;
        }
        ggml_backend_sched_evict_resident(sched, victim);
    }
}

static bool ggml_backend_sched_make_transient_space(
        ggml_backend_sched_t sched, int backend_id, size_t request,
        bool * unknown_memory, bool * live_guard_rejected) {
    *unknown_memory = false;
    *live_guard_rejected = false;
    while (true) {
        if (ggml_backend_sched_weight_window_admit(
                sched, backend_id, request, unknown_memory, live_guard_rejected)) {
            return true;
        }
        if (*unknown_memory) {
            return false;
        }

        // Correctness/oversized-tensor escape hatch: persistent residency is a cache,
        // never a requirement. If the current working tensor needs the space, demote
        // cached VRAM weights and continue from RAM/disk rather than failing inference.
        auto victim = ggml_backend_sched_resident_victim(sched, backend_id);
        if (victim == sched->residents->end()) {
            return false;
        }
        ggml_backend_sched_evict_resident(sched, victim);
        *live_guard_rejected = false;
    }
}

static bool ggml_backend_sched_grow_expert_slab''',
)

replace_once(
    "ggml/src/ggml-backend.cpp",
    """    if (!ggml_backend_sched_make_resident_space(sched, backend_id, new_alloc_size)) {\n""",
    """    if (!ggml_backend_sched_make_resident_space(sched, backend_id, new_alloc_size, false)) {\n""",
)

# Early N+1 prefetch must not consume the resident tier after its partition is
# full; it should remain an ephemeral transient and preserve double-buffer space.
replace_once(
    "ggml/src/ggml-backend.cpp",
    """    const bool cache_eligible = sched->residency_enabled[backend_id] && !full_moe_prefetch;\n    const std::vector<int32_t> empty_experts;\n    const auto resident_key = ggml_backend_sched_resident_key_make(input, backend_id, empty_experts);\n    if (cache_eligible && sched->residents->find(resident_key) != sched->residents->end()) {\n        return false;\n    }\n\n    const size_t alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], input_cpy);\n    if (alloc_size == 0) {\n        return false;\n    }\n""",
    """    const bool cache_enabled = sched->residency_enabled[backend_id] && !full_moe_prefetch;\n    const std::vector<int32_t> empty_experts;\n    const auto resident_key = ggml_backend_sched_resident_key_make(input, backend_id, empty_experts);\n    if (cache_enabled && sched->residents->find(resident_key) != sched->residents->end()) {\n        return false;\n    }\n\n    const size_t alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], input_cpy);\n    if (alloc_size == 0) {\n        return false;\n    }\n    const size_t resident_budget = sched->residency_budget[backend_id];\n    const size_t resident_bytes = sched->transient_metrics.backends[backend_id].current_resident_bytes;\n    const bool cache_eligible = cache_enabled && alloc_size <= resident_budget &&\n        resident_bytes <= resident_budget - alloc_size;\n""",
)

# A split cap is a pipeline target, not a hard model-compatibility limit. A tensor
# larger than the target is isolated in its own split and may borrow the whole
# runtime window; only additional tensors in that split are rejected by the cap.
replace_once(
    "ggml/src/ggml-backend.cpp",
    """            const bool limit_rejected = split_limit > 0 &&\n                alloc_size > split_limit - std::min(split_transient_bytes, split_limit);\n            bool unknown_memory = false;\n            bool live_guard_rejected = false;\n            const bool resident_admitted = cache_eligible && alloc_size > 0 &&\n                ggml_backend_sched_make_resident_space(sched, split_backend_id, alloc_size);\n            const bool window_rejected = alloc_size > 0 && !resident_admitted &&\n                !ggml_backend_sched_weight_window_admit(sched, split_backend_id, alloc_size, &unknown_memory, &live_guard_rejected);\n""",
    """            const bool limit_rejected = split_limit > 0 && split_transient_bytes > 0 &&\n                alloc_size > split_limit - std::min(split_transient_bytes, split_limit);\n            bool unknown_memory = false;\n            bool live_guard_rejected = false;\n            const bool resident_admitted = cache_eligible && alloc_size > 0 &&\n                ggml_backend_sched_make_resident_space(sched, split_backend_id, alloc_size, !expert_tier);\n            const bool transient_admitted = alloc_size > 0 && !resident_admitted &&\n                ggml_backend_sched_make_transient_space(\n                    sched, split_backend_id, alloc_size, &unknown_memory, &live_guard_rejected);\n            const bool window_rejected = alloc_size > 0 && !resident_admitted && !transient_admitted;\n""",
)

# Setter lives next to the existing residency switch.
replace_once(
    "ggml/src/ggml-backend.cpp",
    """void ggml_backend_sched_set_persistent_weight_residency(ggml_backend_sched_t sched, bool persistent) {\n""",
    """void ggml_backend_sched_set_weight_residency_budget(\n        ggml_backend_sched_t sched, ggml_backend_t backend, size_t budget_bytes) {\n    GGML_ASSERT(sched);\n    const int backend_id = ggml_backend_sched_backend_id(sched, backend);\n    GGML_ASSERT(backend_id >= 0);\n    sched->residency_budget[backend_id] = budget_bytes;\n}\n\nvoid ggml_backend_sched_set_persistent_weight_residency(ggml_backend_sched_t sched, bool persistent) {\n""",
)

# System-RAM safety should use MemAvailable on Linux rather than total physical
# memory. This preserves the disk fallback when the machine is already under RAM
# pressure instead of overcommitting staging/page-cache space.
replace_once(
    "src/llama-context.cpp",
    """#include <cstring>\n#include <limits>\n""",
    """#include <cstring>\n#include <fstream>\n#include <limits>\n""",
)

replace_once(
    "src/llama-context.cpp",
    """//\n// llama_context\n//\n\nstatic llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {\n""",
    """//\n// llama_context\n//\n\nstatic size_t llama_system_mem_available() {\n#if defined(__linux__)\n    std::ifstream f(\"/proc/meminfo\");\n    std::string key;\n    uint64_t value = 0;\n    std::string unit;\n    while (f >> key >> value >> unit) {\n        if (key == \"MemAvailable:\") {\n            return value > SIZE_MAX / 1024 ? SIZE_MAX : (size_t) value * 1024;\n        }\n    }\n#endif\n    return 0;\n}\n\nstatic llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {\n""",
)

replace_once(
    "src/llama-context.cpp",
    """            // on Linux the CPU backend reports total physical RAM as \"free\"\n            // (see ggml_backend_cpu_device_get_memory); use that as the ceiling\n            const size_t phys_ram = ram_free > 0 ? ram_free : ram_total;\n            if (phys_ram > 0) {\n                // use at most 1/4 of physical RAM per split:\n                //   1/4 current split, 1/4 prefetch, 1/2 headroom for OS + other processes\n                const size_t ram_budget = phys_ram / 4;\n                if (sequential_weight_budget == 0 || ram_budget < sequential_weight_budget) {\n                    sequential_weight_budget = ram_budget;\n                }\n                LLAMA_LOG_INFO(\"%s: sequential load: RAM budget  = %.1f GiB (physical = %.1f GiB)\\n\",\n                    __func__,\n                    ram_budget / (1024.0 * 1024.0 * 1024.0),\n                    phys_ram / (1024.0 * 1024.0 * 1024.0));\n            }\n""",
    """            // Prefer Linux MemAvailable: total physical RAM is not a safe bound when\n            // other processes and the OS already own a large fraction of memory.\n            const size_t phys_ram = ram_free > 0 ? ram_free : ram_total;\n            const size_t mem_available = llama_system_mem_available();\n            const size_t usable_ram = mem_available > 0 ?\n                (phys_ram > 0 ? std::min(mem_available, phys_ram) : mem_available) : phys_ram;\n            if (usable_ram > 0) {\n                // Two active host-side streaming windows consume at most half of the\n                // currently available RAM; the other half stays available to the OS,\n                // page cache, KV/checkpoints and unrelated processes.\n                const size_t ram_budget = usable_ram / 4;\n                if (sequential_weight_budget == 0 || ram_budget < sequential_weight_budget) {\n                    sequential_weight_budget = ram_budget;\n                }\n                LLAMA_LOG_INFO(\"%s: sequential load: RAM budget  = %.1f GiB (available = %.1f GiB, physical = %.1f GiB)\\n\",\n                    __func__,\n                    ram_budget / (1024.0 * 1024.0 * 1024.0),\n                    usable_ram / (1024.0 * 1024.0 * 1024.0),\n                    phys_ram / (1024.0 * 1024.0 * 1024.0));\n            }\n""",
)

# Runtime partition: default 1/3 persistent cache, 1/3 current split, 1/3 next
# split. The percentage is tunable, but current+next always share the remainder.
replace_once(
    "src/llama-context.cpp",
    """            ggml_backend_sched_set_weight_residency(sched.get(), backend, memory_valid && window_bytes > 0);\n            ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), backend, window_bytes);\n            LLAMA_LOG_INFO(\"%s: sequential weight window: total = %.2f MiB, post-reservation free = %.2f MiB, \"\n                           \"safety reserve = %.2f MiB, admitted = %.2f MiB, memory valid = %s\\n\",\n                __func__, total_bytes / 1024.0 / 1024.0, free_bytes / 1024.0 / 1024.0,\n                safety_reserve_bytes / 1024.0 / 1024.0, window_bytes / 1024.0 / 1024.0,\n                memory_valid ? \"yes\" : \"no\");\n""",
    """            ggml_backend_sched_set_weight_residency(sched.get(), backend, memory_valid && window_bytes > 0);\n\n            int resident_pct = 33;\n            if (const char * env = getenv(\"GGML_SEQUENTIAL_VRAM_CACHE_PERCENT\")) {\n                char * end = nullptr;\n                const long parsed = strtol(env, &end, 10);\n                if (end != env) {\n                    resident_pct = (int) std::max<long>(0, std::min<long>(80, parsed));\n                }\n            }\n            size_t resident_budget = memory_valid ? window_bytes * (size_t) resident_pct / 100 : 0;\n            size_t split_budget = memory_valid ? (window_bytes - resident_budget) / 2 : 0;\n            constexpr size_t min_pipeline_split = (size_t) 64 * 1024 * 1024;\n            if (window_bytes >= 2 * min_pipeline_split && split_budget < min_pipeline_split) {\n                // On very tight cards, execution capacity wins over persistent cache.\n                resident_budget = 0;\n                split_budget = window_bytes / 2;\n            }\n            if (memory_valid && split_budget == 0) {\n                split_budget = window_bytes;\n            }\n\n            ggml_backend_sched_set_weight_residency_budget(sched.get(), backend, resident_budget);\n            ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), backend, split_budget);\n            LLAMA_LOG_INFO(\"%s: sequential weight window: total = %.2f MiB, post-reservation free = %.2f MiB, \"\n                           \"safety reserve = %.2f MiB, admitted = %.2f MiB, resident = %.2f MiB, \"\n                           \"current/next target = %.2f/%.2f MiB, memory valid = %s\\n\",\n                __func__, total_bytes / 1024.0 / 1024.0, free_bytes / 1024.0 / 1024.0,\n                safety_reserve_bytes / 1024.0 / 1024.0, window_bytes / 1024.0 / 1024.0,\n                resident_budget / 1024.0 / 1024.0, split_budget / 1024.0 / 1024.0,\n                split_budget / 1024.0 / 1024.0, memory_valid ? \"yes\" : \"no\");\n""",
)

print("sequential VRAM/RAM window partition applied")
