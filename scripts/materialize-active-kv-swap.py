from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {n}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))


# ---- prompt-cache RAM/disk active state tier ----
replace_once(
    "tools/server/server-task.h",
    '''struct server_gpu_prompt_cache_state {\n    server_prompt prompt;\n    std::vector<uint8_t> spec;\n    llama_seq_id seq_id = -1;\n};\n\nstruct server_prompt_cache {''',
    '''struct server_gpu_prompt_cache_state {\n    server_prompt prompt;\n    std::vector<uint8_t> spec;\n    llama_seq_id seq_id = -1;\n};\n\n// Exact state for a live request that has been temporarily removed from the\n// unified KV pool. RAM-resident target/draft bytes share --cache-ram with the\n// ordinary prompt cache; when they cannot fit, target/draft state spills to\n// files and only the small speculative sidecar remains in host memory.\nstruct server_active_prompt_cache_state {\n    server_prompt_data data;\n    bool on_disk = false;\n    std::string file_tgt;\n    std::string file_dft;\n\n    size_t size() const { return data.size(); }\n};\n\nstruct server_prompt_cache {''')

replace_once(
    "tools/server/server-task.h",
    '''        if (!lmcache_endpoint.empty()) {\n            this->lmcache = std::make_unique<common_lmcache_client>(lmcache_endpoint);\n            this->lmcache_namespace = lmcache_namespace;\n        }\n    }\n\n    std::list<server_prompt_cache_state> states;\n    std::list<server_gpu_prompt_cache_state> gpu_states;''',
    '''        if (!lmcache_endpoint.empty()) {\n            this->lmcache = std::make_unique<common_lmcache_client>(lmcache_endpoint);\n            this->lmcache_namespace = lmcache_namespace;\n        }\n    }\n\n    ~server_prompt_cache();\n\n    std::list<server_prompt_cache_state> states;\n    std::list<server_gpu_prompt_cache_state> gpu_states;\n    std::map<int32_t, server_active_prompt_cache_state> active_states;\n    std::string active_disk_dir;''')

replace_once(
    "tools/server/server-task.h",
    '''    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);\n\n    // Configure hidden sequence ids''',
    '''    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);\n\n    // Active request residency. These methods preserve an exact sequence while\n    // the request object/sampler remains live, allowing another request to use\n    // the fixed-size unified KV pool without truncating either context.\n    bool active_store(int32_t id_slot, llama_context * ctx_tgt, llama_context * ctx_dft, const std::vector<uint8_t> & spec);\n    bool active_restore(int32_t id_slot, llama_context * ctx_tgt, llama_context * ctx_dft, std::vector<uint8_t> & spec);\n    bool active_has(int32_t id_slot) const;\n    void active_erase(int32_t id_slot);\n\n    // Configure hidden sequence ids''')

replace_once(
    "tools/server/server-task.cpp",
    '''#include <cstring>\n#include <limits>\n#include <sstream>''',
    '''#include <cstring>\n#include <filesystem>\n#include <limits>\n#include <sstream>''')

replace_once(
    "tools/server/server-task.cpp",
    '''size_t server_prompt_cache::size() const {\n    size_t res = 0;\n\n    for (const auto & state : states) {\n        res += state.size();\n    }\n\n    return res;\n}''',
    '''server_prompt_cache::~server_prompt_cache() {\n    if (!active_disk_dir.empty()) {\n        std::error_code ec;\n        std::filesystem::remove_all(active_disk_dir, ec);\n    }\n}\n\nsize_t server_prompt_cache::size() const {\n    size_t res = 0;\n\n    for (const auto & state : states) {\n        res += state.size();\n    }\n    for (const auto & entry : active_states) {\n        res += entry.second.size();\n    }\n\n    return res;\n}''')

replace_once(
    "tools/server/server-task.cpp",
    '''    if (limit_size > 0) {\n        // make room before allocating the new vectors to avoid breaching the limit\n        while (!states.empty() && size() + state_size_new > limit_size) {\n            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\\n",\n                    states.front().size() / (1024.0 * 1024.0));\n\n            states.pop_front();\n        }\n    }\n\n    std::vector<uint8_t> state_data_tgt;''',
    '''    if (limit_size > 0) {\n        // make room before allocating the new vectors to avoid breaching the limit.\n        // Active suspended states are never evicted; ordinary reusable prompt\n        // entries yield to them because dropping a live request is not allowed.\n        while (!states.empty() && size() + state_size_new > limit_size) {\n            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\\n",\n                    states.front().size() / (1024.0 * 1024.0));\n\n            states.pop_front();\n        }\n        if (size() + state_size_new > limit_size) {\n            SRV_TRC(" - prompt cache RAM is reserved by active suspended state; skipping reusable entry\\n");\n            return nullptr;\n        }\n    }\n\n    std::vector<uint8_t> state_data_tgt;''')

marker = '''void server_prompt_cache::store_remote(server_prompt_cache_state * state) {'''
p = Path("tools/server/server-task.cpp")
text = p.read_text()
if text.count(marker) != 1:
    raise SystemExit("server-task.cpp: store_remote marker mismatch")
active_impl = r'''bool server_prompt_cache::active_has(int32_t id_slot) const {
    return active_states.find(id_slot) != active_states.end();
}

void server_prompt_cache::active_erase(int32_t id_slot) {
    auto it = active_states.find(id_slot);
    if (it == active_states.end()) {
        return;
    }
    if (it->second.on_disk) {
        std::error_code ec;
        if (!it->second.file_tgt.empty()) {
            std::filesystem::remove(it->second.file_tgt, ec);
            ec.clear();
        }
        if (!it->second.file_dft.empty()) {
            std::filesystem::remove(it->second.file_dft, ec);
        }
    }
    active_states.erase(it);
}

bool server_prompt_cache::active_store(
        int32_t id_slot,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        const std::vector<uint8_t> & spec) {
    GGML_ASSERT(ctx_tgt);
    active_erase(id_slot);

    const size_t size_tgt = llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t size_dft = ctx_dft
        ? llama_state_seq_get_size_ext(ctx_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE)
        : 0;
    const size_t ram_need = size_tgt + size_dft + spec.size();

    bool use_ram = local_enabled;
    if (use_ram && limit_size > 0) {
        while (!states.empty() && size() + ram_need > limit_size) {
            SRV_TRC(" - evicting reusable prompt cache entry for active KV suspension (%.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));
            states.pop_front();
        }
        use_ram = ram_need <= limit_size && size() <= limit_size - ram_need;
    }

    server_active_prompt_cache_state state;
    state.data.spec = spec;

    if (use_ram) {
        try {
            state.data.main.resize(size_tgt);
            state.data.drft.resize(size_dft);
        } catch (const std::bad_alloc &) {
            state.data.main.clear();
            state.data.drft.clear();
            use_ram = false;
        }
    }

    if (use_ram) {
        const size_t n_tgt = llama_state_seq_get_data_ext(
                ctx_tgt, state.data.main.data(), size_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n_tgt != size_tgt) {
            SRV_ERR("failed to capture active target state for slot %d (%zu/%zu bytes)\n",
                    id_slot, n_tgt, size_tgt);
            return false;
        }
        if (ctx_dft && size_dft > 0) {
            const size_t n_dft = llama_state_seq_get_data_ext(
                    ctx_dft, state.data.drft.data(), size_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (n_dft != size_dft) {
                SRV_ERR("failed to capture active draft state for slot %d (%zu/%zu bytes)\n",
                        id_slot, n_dft, size_dft);
                return false;
            }
        }
        active_states.emplace(id_slot, std::move(state));
        SRV_INF("active KV slot %d suspended to RAM: %.3f MiB\n",
                id_slot, ram_need / (1024.0 * 1024.0));
        return true;
    }

    // Disk is the unbounded correctness tier. It is used when --cache-ram is
    // disabled/full or when one exact live state is larger than that budget.
    // The OS page cache remains free to keep hot pages in RAM, but it can reclaim
    // them under pressure, so this does not create another fixed host-RAM budget.
    try {
        if (active_disk_dir.empty()) {
            const char * env = getenv("LLAMA_SERVER_KV_SWAP_DIR");
            std::filesystem::path base = env && env[0]
                ? std::filesystem::path(env)
                : std::filesystem::temp_directory_path();
            std::ostringstream name;
            name << "llama-kv-swap-" << static_cast<const void *>(this);
            active_disk_dir = (base / name.str()).string();
            std::filesystem::create_directories(active_disk_dir);
            SRV_INF("active KV disk tier: %s\n", active_disk_dir.c_str());
        }

        const auto base = std::filesystem::path(active_disk_dir) /
            ("slot-" + std::to_string(id_slot));
        state.file_tgt = base.string() + ".target.bin";
        state.file_dft = size_dft > 0 ? base.string() + ".draft.bin" : std::string();

        const llama_token marker_token = 0;
        const size_t n_tgt = llama_state_seq_save_file(
                ctx_tgt, state.file_tgt.c_str(), id_slot, &marker_token, 1);
        if (n_tgt == 0) {
            SRV_ERR("failed to spill active target state for slot %d to disk\n", id_slot);
            std::error_code ec;
            std::filesystem::remove(state.file_tgt, ec);
            return false;
        }
        if (ctx_dft && size_dft > 0) {
            const size_t n_dft = llama_state_seq_save_file(
                    ctx_dft, state.file_dft.c_str(), id_slot, &marker_token, 1);
            if (n_dft == 0) {
                SRV_ERR("failed to spill active draft state for slot %d to disk\n", id_slot);
                std::error_code ec;
                std::filesystem::remove(state.file_tgt, ec);
                std::filesystem::remove(state.file_dft, ec);
                return false;
            }
        }
        state.on_disk = true;
        active_states.emplace(id_slot, std::move(state));
        SRV_INF("active KV slot %d suspended to disk (state %.3f MiB exceeds/avoids RAM tier)\n",
                id_slot, ram_need / (1024.0 * 1024.0));
        return true;
    } catch (const std::exception & e) {
        SRV_ERR("active KV disk spill failed for slot %d: %s\n", id_slot, e.what());
        return false;
    }
}

bool server_prompt_cache::active_restore(
        int32_t id_slot,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        std::vector<uint8_t> & spec) {
    auto it = active_states.find(id_slot);
    if (it == active_states.end()) {
        return false;
    }

    auto fail_clean = [&]() {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        }
        return false;
    };

    auto & state = it->second;
    if (!state.on_disk) {
        const size_t size_tgt = state.data.main.size();
        const size_t n_tgt = llama_state_seq_set_data_ext(
                ctx_tgt, state.data.main.data(), size_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n_tgt != size_tgt) {
            SRV_ERR("failed to restore active target RAM state for slot %d (%zu/%zu bytes)\n",
                    id_slot, n_tgt, size_tgt);
            return fail_clean();
        }
        if (!state.data.drft.empty()) {
            if (!ctx_dft) {
                return fail_clean();
            }
            const size_t size_dft = state.data.drft.size();
            const size_t n_dft = llama_state_seq_set_data_ext(
                    ctx_dft, state.data.drft.data(), size_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (n_dft != size_dft) {
                SRV_ERR("failed to restore active draft RAM state for slot %d (%zu/%zu bytes)\n",
                        id_slot, n_dft, size_dft);
                return fail_clean();
            }
        }
    } else {
        llama_token marker_token = 0;
        size_t n_token_count = 0;
        const size_t n_tgt = llama_state_seq_load_file(
                ctx_tgt, state.file_tgt.c_str(), id_slot, &marker_token, 1, &n_token_count);
        if (n_tgt == 0 || n_token_count != 1) {
            SRV_ERR("failed to restore active target disk state for slot %d\n", id_slot);
            return fail_clean();
        }
        if (!state.file_dft.empty()) {
            if (!ctx_dft) {
                return fail_clean();
            }
            n_token_count = 0;
            const size_t n_dft = llama_state_seq_load_file(
                    ctx_dft, state.file_dft.c_str(), id_slot, &marker_token, 1, &n_token_count);
            if (n_dft == 0 || n_token_count != 1) {
                SRV_ERR("failed to restore active draft disk state for slot %d\n", id_slot);
                return fail_clean();
            }
        }
    }

    spec = state.data.spec;
    const bool was_disk = state.on_disk;
    active_erase(id_slot);
    SRV_INF("active KV slot %d restored from %s tier\n", id_slot, was_disk ? "disk" : "RAM");
    return true;
}

'''
p.write_text(text.replace(marker, active_impl + marker, 1))


# ---- server slot state + scheduler ----
replace_once(
    "tools/server/server-context.cpp",
    '''    SLOT_STATE_DONE_PROMPT,\n    SLOT_STATE_GENERATING,\n};''',
    '''    SLOT_STATE_DONE_PROMPT,\n    SLOT_STATE_GENERATING,\n    SLOT_STATE_SUSPENDED, // live request; target/draft state is in RAM/disk instead of the unified KV pool\n};''')

replace_once(
    "tools/server/server-context.cpp",
    '''    // state\n    slot_state state = SLOT_STATE_IDLE;\n\n    server_prompt prompt;''',
    '''    // state\n    slot_state state = SLOT_STATE_IDLE;\n    slot_state resume_state = SLOT_STATE_IDLE;\n    int64_t t_suspended = -1;\n\n    server_prompt prompt;''')

replace_once(
    "tools/server/server-context.cpp",
    '''        spec_is_replay = false;\n\n        last_nl_pos''',
    '''        spec_is_replay = false;\n        resume_state   = SLOT_STATE_IDLE;\n        t_suspended    = -1;\n\n        last_nl_pos''')

replace_once(
    "tools/server/server-context.cpp",
    '''    bool is_processing() const {\n        return state != SLOT_STATE_IDLE;\n    }\n\n    bool can_speculate() const {''',
    '''    bool is_processing() const {\n        return state != SLOT_STATE_IDLE;\n    }\n\n    bool is_suspended() const {\n        return state == SLOT_STATE_SUSPENDED;\n    }\n\n    bool is_resident_processing() const {\n        return state != SLOT_STATE_IDLE && state != SLOT_STATE_SUSPENDED;\n    }\n\n    bool can_speculate() const {''')

replace_once(
    "tools/server/server-context.cpp",
    '''            t_last_used = ggml_time_us();\n\n            state = SLOT_STATE_IDLE;\n\n            // do not keep context of the child slots - the parent's context is enough\n            if (task->is_child()) {\n                prompt_clear();\n            }''',
    '''            t_last_used = ggml_time_us();\n\n            const bool was_suspended = state == SLOT_STATE_SUSPENDED;\n            state = SLOT_STATE_IDLE;\n\n            // A suspended slot has no live sequence state. Do not leave its prompt\n            // attached to an idle slot after cancellation/error, because that would\n            // make later prompt matching believe the missing KV is still resident.\n            if (task->is_child() || was_suspended) {\n                prompt_clear();\n            }''')

replace_once(
    "tools/server/server-context.cpp",
    '''        // Reserve additional sequence IDs for inactive agent contexts without\n        // creating additional active server slots.  With unified KV this does not\n        // divide n_ctx_seq: hidden sequences share the same physical KV pool.\n        int32_t gpu_agent_cache_seqs = 0;\n        if (!utility_model && params_base.kv_unified && params_base.lora_adapters.empty()) {\n            gpu_agent_cache_seqs = 8;\n            if (const char * env = getenv("LLAMA_SERVER_GPU_AGENT_CACHE_SEQS")) {\n                gpu_agent_cache_seqs = std::clamp<int32_t>(atoi(env), 0, 32);\n            }\n        }''',
    '''        // Hidden sequence ownership consumes cells from the same unified KV\n        // pool and therefore makes very large concurrent contexts collide sooner.\n        // Keep it as an explicit experimental opt-in; active RAM/disk suspension\n        // below removes inactive request cells from the live pool instead.\n        int32_t gpu_agent_cache_seqs = 0;\n        if (!utility_model && params_base.kv_unified && params_base.lora_adapters.empty()) {\n            if (const char * env = getenv("LLAMA_SERVER_GPU_AGENT_CACHE_SEQS")) {\n                gpu_agent_cache_seqs = std::clamp<int32_t>(atoi(env), 0, 32);\n                if (gpu_agent_cache_seqs > 0) {\n                    SRV_WRN("GPU hidden agent cache explicitly enabled with %d sequences; hidden states consume live unified KV capacity\\n",\n                            gpu_agent_cache_seqs);\n                }\n            }\n        }''')

replace_once(
    "tools/server/server-context.cpp",
    '''            slot.callback_on_release = [this](int id_slot) {\n                queue_tasks.pop_deferred_task(id_slot);\n            };''',
    '''            slot.callback_on_release = [this](int id_slot) {\n                if (prompt_cache) {\n                    prompt_cache->active_erase(id_slot);\n                }\n                queue_tasks.pop_deferred_task(id_slot);\n            };''')

replace_once(
    "tools/server/server-context.cpp",
    '''        if (params_base.cache_ram_mib != 0 || !params_base.lmcache_endpoint.empty() || gpu_agent_cache_seqs > 0) {''',
    '''        const bool active_kv_swap = !utility_model && params_base.kv_unified && params_base.n_parallel > 1;\n        if (params_base.cache_ram_mib != 0 || !params_base.lmcache_endpoint.empty() || gpu_agent_cache_seqs > 0 || active_kv_swap) {''')

replace_once(
    "tools/server/server-context.cpp",
    '''            prompt_cache = std::make_unique<server_prompt_cache>(\n                    params_base.cache_ram_mib, n_ctx, params_base.lmcache_endpoint, lmcache_namespace);\n            if (gpu_agent_cache_seqs > 0''',
    '''            prompt_cache = std::make_unique<server_prompt_cache>(\n                    params_base.cache_ram_mib, n_ctx, params_base.lmcache_endpoint, lmcache_namespace);\n            if (active_kv_swap) {\n                SRV_INF("active unified-KV suspension enabled: VRAM -> shared cache RAM -> disk (ctx=%d, parallel=%d)\\n",\n                        n_ctx, params_base.n_parallel);\n            }\n            if (gpu_agent_cache_seqs > 0''')

marker = '''    // returns false to decline the task, it is offered again after the decode is done\n    bool process_single_task(server_task && task, bool is_yielding) {'''
p = Path("tools/server/server-context.cpp")
text = p.read_text()
if text.count(marker) != 1:
    raise SystemExit("server-context.cpp: process_single_task marker mismatch")
helpers = r'''    size_t slot_projected_kv_tokens(const server_slot & slot) const {
        if (!slot.is_resident_processing() || !slot.task) {
            return 0;
        }

        size_t cur = slot.prompt.tokens.size();
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
        if (pos_max >= 0) {
            cur = std::max(cur, (size_t) pos_max + 1);
        }

        if (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT) {
            cur = std::max(cur, (size_t) slot.task->n_tokens());
        }

        if (slot.state == SLOT_STATE_GENERATING && cur < (size_t) slot.n_ctx) {
            cur++;
        }
        return std::min(cur, (size_t) slot.n_ctx);
    }

    size_t active_kv_headroom() const {
        if (!ctx_tgt) {
            return 0;
        }
        return std::max<size_t>(2048, 2ull * llama_n_ubatch(ctx_tgt));
    }

    bool slot_can_suspend(const server_slot & slot) const {
        if (!slot.is_resident_processing() || !slot.task || slot.task->is_parent() || slot.task->is_child()) {
            return false;
        }
        if (slot.state != SLOT_STATE_STARTED &&
                slot.state != SLOT_STATE_PROCESSING_PROMPT &&
                slot.state != SLOT_STATE_GENERATING) {
            return false;
        }
        return slot.spec_i_batch.empty() && slot.spec_draft.empty();
    }

    bool suspend_active_slot(server_slot & slot, const char * reason) {
        if (!prompt_cache || !slot_can_suspend(slot)) {
            return false;
        }

        llama_synchronize(ctx_tgt);
        if (ctx_dft) {
            llama_synchronize(ctx_dft);
        }

        std::vector<uint8_t> spec_state;
        if (slot.spec) {
            common_speculative_get_state(slot.spec, slot.id, spec_state);
        }

        if (!prompt_cache->active_store(slot.id, ctx_tgt, ctx_dft, spec_state)) {
            return false;
        }

        const slot_state saved_resume_state = slot.state;
        slot.mem.seq_rm(slot.id, -1, -1);
        slot.resume_state = saved_resume_state;
        slot.state = SLOT_STATE_SUSPENDED;
        slot.t_suspended = ggml_time_us();
        slot.i_batch = -1;

        SLT_INF(slot, "suspended live request (%s), preserved %zu prompt/context tokens outside VRAM KV\n",
                reason, slot.prompt.tokens.size());
        return true;
    }

    bool resume_active_slot(server_slot & slot) {
        if (!prompt_cache || !slot.is_suspended() || !prompt_cache->active_has(slot.id)) {
            return false;
        }

        llama_synchronize(ctx_tgt);
        if (ctx_dft) {
            llama_synchronize(ctx_dft);
        }
        slot.mem.seq_rm(slot.id, -1, -1);

        std::vector<uint8_t> spec_state;
        if (!prompt_cache->active_restore(slot.id, ctx_tgt, ctx_dft, spec_state)) {
            slot.mem.seq_rm(slot.id, -1, -1);
            return false;
        }
        if (slot.spec && !spec_state.empty()) {
            common_speculative_set_state(slot.spec, slot.id, spec_state);
        }

        slot.state = slot.resume_state;
        slot.resume_state = SLOT_STATE_IDLE;
        slot.t_suspended = -1;
        SLT_INF(slot, "resumed live request with %zu context tokens\n", slot.prompt.tokens.size());
        return true;
    }

    server_slot * largest_suspendable_resident() {
        server_slot * victim = nullptr;
        size_t best = 0;
        for (auto & slot : slots) {
            if (!slot_can_suspend(slot)) {
                continue;
            }
            const size_t cur = slot_projected_kv_tokens(slot);
            if (!victim || cur > best) {
                victim = &slot;
                best = cur;
            }
        }
        return victim;
    }

    bool make_room_for_incoming(const server_task & task) {
        if (!params_base.kv_unified || params_base.n_parallel <= 1 || !prompt_cache ||
                !task.need_sampling() || task.is_parent() || task.is_child()) {
            return true;
        }

        const size_t incoming = std::min<size_t>((size_t) task.n_tokens() + 1, (size_t) n_ctx);
        const size_t headroom = active_kv_headroom();
        const size_t shared_limit = (size_t) n_ctx > headroom ? (size_t) n_ctx - headroom : (size_t) n_ctx;

        while (true) {
            size_t projected = incoming;
            int n_resident = 0;
            for (const auto & slot : slots) {
                if (slot.is_resident_processing()) {
                    projected += slot_projected_kv_tokens(slot);
                    n_resident++;
                }
            }

            if (n_resident == 0 || projected <= shared_limit) {
                return true;
            }

            server_slot * victim = largest_suspendable_resident();
            if (!victim || !suspend_active_slot(*victim, "admitting another request would exceed unified KV capacity")) {
                return false;
            }
        }
    }

    void rebalance_active_kv_residency() {
        if (!params_base.kv_unified || params_base.n_parallel <= 1 || !prompt_cache) {
            return;
        }

        int n_resident = 0;
        for (const auto & slot : slots) {
            n_resident += slot.is_resident_processing() ? 1 : 0;
        }

        if (n_resident == 0) {
            server_slot * oldest = nullptr;
            for (auto & slot : slots) {
                if (!slot.is_suspended()) {
                    continue;
                }
                if (!oldest || slot.t_suspended < oldest->t_suspended) {
                    oldest = &slot;
                }
            }
            if (oldest && !resume_active_slot(*oldest)) {
                send_error(*oldest, "failed to restore suspended KV state", ERROR_TYPE_SERVER);
                oldest->release();
            }
            return;
        }

        const size_t headroom = active_kv_headroom();
        const size_t shared_limit = (size_t) n_ctx > headroom ? (size_t) n_ctx - headroom : (size_t) n_ctx;
        while (n_resident > 1) {
            size_t projected = 0;
            for (const auto & slot : slots) {
                if (slot.is_resident_processing()) {
                    projected += slot_projected_kv_tokens(slot);
                }
            }
            if (projected <= shared_limit) {
                break;
            }
            server_slot * victim = largest_suspendable_resident();
            if (!victim || !suspend_active_slot(*victim, "live contexts approached unified KV capacity")) {
                break;
            }
            n_resident--;
        }
    }

'''
p.write_text(text.replace(marker, helpers + marker, 1))

replace_once(
    "tools/server/server-context.cpp",
    '''                    if (slot->is_processing()) {\n                        // if requested slot is unavailable, we defer this task for processing later\n                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\\n", id_task);\n                        queue_tasks.defer(std::move(task));\n                        break;\n                    }\n\n                    if (task.is_parent()) {''',
    '''                    if (slot->is_processing()) {\n                        // if requested slot is unavailable, we defer this task for processing later\n                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\\n", id_task);\n                        queue_tasks.defer(std::move(task));\n                        break;\n                    }\n\n                    if (!make_room_for_incoming(task)) {\n                        SRV_DBG("unable to reach a safe active-KV suspension boundary; defer task, id_task = %d\\n", id_task);\n                        queue_tasks.defer(std::move(task));\n                        break;\n                    }\n\n                    if (task.is_parent()) {''')

replace_once(
    "tools/server/server-context.cpp",
    '''#endif\n\n        // check if all slots are idle\n        {''',
    '''#endif\n\n        rebalance_active_kv_residency();\n\n        // check if all slots are idle\n        {''')

replace_once(
    "tools/server/server-context.cpp",
    '''            {"id",            id},\n            {"n_ctx",         n_ctx},\n            {"speculative",   can_speculate()},\n            {"is_processing", is_processing()},''',
    '''            {"id",            id},\n            {"n_ctx",         n_ctx},\n            {"speculative",   can_speculate()},\n            {"is_processing", is_processing()},\n            {"is_suspended",  is_suspended()},''')

print("active KV residency source patches applied")
