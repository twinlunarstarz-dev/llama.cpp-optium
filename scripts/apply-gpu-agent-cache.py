#!/usr/bin/env python3
"""Add a GPU-resident inactive-agent tier in front of the RAM prompt cache.

For unified KV contexts, inactive conversations are retained under hidden sequence
IDs. Sequence-copy only changes ownership metadata, so A -> B -> A switching can
resume without serializing the KV to host. Under KV pressure the hidden LRU is
demoted to the existing exact RAM/LMCache snapshot format.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str):
    p = ROOT / path
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{path}: expected one match, found {n}: {old[:100]!r}")
    p.write_text(s.replace(old, new, 1))
    print(f"patched {path}")


def patch_header():
    path = "tools/server/server-task.h"
    old = '''struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

struct server_prompt_cache {
'''
    new = '''struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

// Level-1 agent cache: KV remains in the unified device cache and is retained
// under a hidden sequence id.  Only the small speculative-driver sidecar stays
// in host memory.  This avoids D2H/H2D state serialization on normal A->B->A
// agent switches.  Entries are demoted to server_prompt_cache_state on pressure.
struct server_gpu_prompt_cache_state {
    server_prompt prompt;
    std::vector<uint8_t> spec;
    llama_seq_id seq_id = -1;
};

struct server_prompt_cache {
'''
    replace_once(path, old, new)

    old = '''    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
'''
    new = '''    std::list<server_prompt_cache_state> states;
    std::list<server_gpu_prompt_cache_state> gpu_states;

    llama_context * gpu_ctx_tgt = nullptr;
    llama_context * gpu_ctx_dft = nullptr;
    llama_seq_id gpu_seq_first = 0;
    int32_t gpu_seq_count = 0;

    // in bytes, 0 = no limit
'''
    replace_once(path, old, new)

    old = '''    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();
};
'''
    new = '''    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    // Configure hidden sequence ids [seq_first, seq_first + seq_count) for a
    // unified-KV, GPU-resident inactive-agent tier.
    void configure_gpu(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq_first, int32_t seq_count);
    bool gpu_enabled() const { return gpu_ctx_tgt != nullptr && gpu_seq_count > 0; }
    bool gpu_save(const server_prompt & prompt, const std::vector<uint8_t> & spec, llama_seq_id active_seq);
    bool gpu_load(server_prompt & prompt, const server_tokens & tokens_new, llama_seq_id active_seq);
    bool gpu_demote_lru();

    void update();
};
'''
    replace_once(path, old, new)


def patch_cache_cpp():
    path = "tools/server/server-task.cpp"
    marker = '''void server_prompt_cache::update() {
'''
    impl = r'''void server_prompt_cache::configure_gpu(
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        llama_seq_id seq_first,
        int32_t seq_count) {
    gpu_ctx_tgt = ctx_tgt;
    gpu_ctx_dft = ctx_dft;
    gpu_seq_first = seq_first;
    gpu_seq_count = std::max<int32_t>(0, seq_count);

    if (gpu_enabled()) {
        SRV_INF("GPU agent cache enabled: %d hidden sequence ids [%d, %d)\n",
                gpu_seq_count, (int) gpu_seq_first, (int) (gpu_seq_first + gpu_seq_count));
    }
}

bool server_prompt_cache::gpu_demote_lru() {
    if (!gpu_enabled() || gpu_states.empty()) {
        return false;
    }

    auto it = gpu_states.begin();
    const llama_seq_id seq_id = it->seq_id;

    // Preserve the hidden entry in the existing level-2 cache when one exists.
    // If RAM/LMCache is disabled, dropping the oldest hidden entry is still the
    // correct response to device KV pressure and lets decode make progress.
    if (local_enabled || lmcache) {
        const size_t size_tgt = llama_state_seq_get_size_ext(gpu_ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t size_dft = gpu_ctx_dft
            ? llama_state_seq_get_size_ext(gpu_ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE)
            : 0;

        if (auto * dst = alloc(it->prompt, size_tgt, size_dft)) {
            const size_t n_tgt = llama_state_seq_get_data_ext(
                    gpu_ctx_tgt, dst->data.main.data(), size_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (n_tgt == size_tgt) {
                bool ok = true;
                if (gpu_ctx_dft && size_dft > 0) {
                    const size_t n_dft = llama_state_seq_get_data_ext(
                            gpu_ctx_dft, dst->data.drft.data(), size_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
                    ok = n_dft == size_dft;
                }
                if (ok) {
                    dst->data.spec = it->spec;
                    store_remote(dst);
                    SRV_TRC(" - demoted %d-token GPU agent state from seq %d to level-2 cache\n",
                            it->prompt.n_tokens(), (int) seq_id);
                } else {
                    SRV_WRN(" - failed to serialize hidden GPU agent seq %d; dropping it\n", (int) seq_id);
                }
            } else {
                SRV_WRN(" - failed to serialize hidden GPU agent target seq %d; dropping it\n", (int) seq_id);
            }
        } else {
            SRV_WRN(" - level-2 cache could not accept hidden GPU agent seq %d; dropping it\n", (int) seq_id);
        }
    }

    llama_memory_seq_rm(llama_get_memory(gpu_ctx_tgt), seq_id, -1, -1);
    if (gpu_ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(gpu_ctx_dft), seq_id, -1, -1);
    }
    gpu_states.erase(it);
    return true;
}

bool server_prompt_cache::gpu_save(
        const server_prompt & prompt,
        const std::vector<uint8_t> & spec,
        llama_seq_id active_seq) {
    if (!gpu_enabled() || prompt.tokens.empty()) {
        return false;
    }

    llama_seq_id seq_id = -1;
    auto it_exact = gpu_states.end();

    // Reuse the hidden id for an exact snapshot replacement.  Shorter prefixes
    // remain valid branch roots and are intentionally retained.
    for (auto it = gpu_states.begin(); it != gpu_states.end(); ++it) {
        const int lcp = it->prompt.tokens.get_common_prefix(prompt.tokens);
        if (lcp == (int) prompt.tokens.size() && it->prompt.tokens.size() == prompt.tokens.size()) {
            seq_id = it->seq_id;
            it_exact = it;
            break;
        }
    }

    if (seq_id < 0) {
        for (int32_t i = 0; i < gpu_seq_count; ++i) {
            const llama_seq_id candidate = gpu_seq_first + i;
            const bool used = std::any_of(gpu_states.begin(), gpu_states.end(),
                    [candidate](const server_gpu_prompt_cache_state & s) { return s.seq_id == candidate; });
            if (!used) {
                seq_id = candidate;
                break;
            }
        }
    }

    // Hidden-id metadata is deliberately bounded.  Demote only when every id is
    // occupied; normal alternation among a small agent set remains device-only.
    if (seq_id < 0) {
        if (!gpu_demote_lru()) {
            return false;
        }
        for (int32_t i = 0; i < gpu_seq_count; ++i) {
            const llama_seq_id candidate = gpu_seq_first + i;
            const bool used = std::any_of(gpu_states.begin(), gpu_states.end(),
                    [candidate](const server_gpu_prompt_cache_state & s) { return s.seq_id == candidate; });
            if (!used) {
                seq_id = candidate;
                break;
            }
        }
    }

    if (seq_id < 0) {
        return false;
    }

    // Remove an older exact version before re-tagging.  The active sequence still
    // owns the cells, so this only drops the stale hidden ownership bit.
    if (it_exact != gpu_states.end()) {
        llama_memory_seq_rm(llama_get_memory(gpu_ctx_tgt), seq_id, -1, -1);
        if (gpu_ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(gpu_ctx_dft), seq_id, -1, -1);
        }
        gpu_states.erase(it_exact);
    }

    llama_memory_seq_cp(llama_get_memory(gpu_ctx_tgt), active_seq, seq_id, -1, -1);
    if (gpu_ctx_dft) {
        llama_memory_seq_cp(llama_get_memory(gpu_ctx_dft), active_seq, seq_id, -1, -1);
    }

    gpu_states.push_back({ prompt.clone(), spec, seq_id });
    SRV_TRC(" - retained %d-token agent state on device as hidden seq %d\n",
            prompt.n_tokens(), (int) seq_id);
    return true;
}

bool server_prompt_cache::gpu_load(
        server_prompt & prompt,
        const server_tokens & tokens_new,
        llama_seq_id active_seq) {
    if (!gpu_enabled() || gpu_states.empty() || tokens_new.empty()) {
        return false;
    }

    const int lcp_base = prompt.tokens.get_common_prefix(tokens_new);
    int lcp_best = lcp_base;
    auto it_best = gpu_states.end();

    for (auto it = gpu_states.begin(); it != gpu_states.end(); ++it) {
        const int lcp = it->prompt.tokens.get_common_prefix(tokens_new);
        const float keep = it->prompt.tokens.size() > 0
            ? float(lcp) / it->prompt.tokens.size()
            : 0.0f;
        if (keep >= 0.25f && lcp > lcp_best) {
            lcp_best = lcp;
            it_best = it;
        }
    }

    if (it_best == gpu_states.end()) {
        return false;
    }

    // Remove the previous active ownership, then alias the selected hidden
    // sequence back into the active slot.  KV tensor bytes stay in-place.
    llama_memory_seq_rm(llama_get_memory(gpu_ctx_tgt), active_seq, -1, -1);
    llama_memory_seq_cp(llama_get_memory(gpu_ctx_tgt), it_best->seq_id, active_seq, -1, -1);
    if (gpu_ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(gpu_ctx_dft), active_seq, -1, -1);
        llama_memory_seq_cp(llama_get_memory(gpu_ctx_dft), it_best->seq_id, active_seq, -1, -1);
    }

    prompt = it_best->prompt.clone();
    restored_any_state = true;
    restored_spec_state = it_best->spec;
    restored_spec_state_valid = !restored_spec_state.empty();

    SRV_TRC(" - restored %d-token agent state from hidden GPU seq %d (lcp=%d)\n",
            prompt.n_tokens(), (int) it_best->seq_id, lcp_best);

    // Successful restores are MRU; retain the hidden tag so the same branch can
    // be selected again without another state copy.
    gpu_states.splice(gpu_states.end(), gpu_states, it_best);
    return true;
}

'''
    replace_once(path, marker, impl + marker)


def patch_server_context():
    path = "tools/server/server-context.cpp"

    old = '''        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
'''
    new = '''        // Reserve additional sequence IDs for inactive agent contexts without
        // creating additional active server slots.  With unified KV this does not
        // divide n_ctx_seq: hidden sequences share the same physical KV pool.
        int32_t gpu_agent_cache_seqs = 0;
        if (!utility_model && params_base.kv_unified && params_base.lora_adapters.empty()) {
            gpu_agent_cache_seqs = 8;
            if (const char * env = getenv("LLAMA_SERVER_GPU_AGENT_CACHE_SEQS")) {
                gpu_agent_cache_seqs = std::clamp<int32_t>(atoi(env), 0, 32);
            }
        }

        common_params params_ctx = params_base;
        params_ctx.n_parallel += gpu_agent_cache_seqs;
        llama_init = common_init_from_params(params_ctx);

        model_tgt = llama_init->model();
'''
    replace_once(path, old, new)

    old = '''                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
'''
    new = '''                common_params params_dft = common_base_params_to_speculative(params_base);
                // The draft/MTP context must understand the same hidden sequence
                // IDs as the target context.  Keep output limits from params_base;
                // only sequence-addressability is widened.
                params_dft.n_parallel = llama_n_seq_max(ctx_tgt);

                // progress callback
'''
    replace_once(path, old, new)

    old = '''            prompt_cache = std::make_unique<server_prompt_cache>(
                    params_base.cache_ram_mib, n_ctx, params_base.lmcache_endpoint, lmcache_namespace);
            if (!params_base.lmcache_endpoint.empty()) {
'''
    new = '''            prompt_cache = std::make_unique<server_prompt_cache>(
                    params_base.cache_ram_mib, n_ctx, params_base.lmcache_endpoint, lmcache_namespace);
            if (gpu_agent_cache_seqs > 0 && ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                const int32_t seq_total = (int32_t) llama_n_seq_max(ctx_tgt);
                const int32_t seq_first = params_base.n_parallel;
                const int32_t seq_count = std::max<int32_t>(0, seq_total - seq_first);
                prompt_cache->configure_gpu(ctx_tgt, ctx_dft, seq_first, seq_count);
            }
            if (!params_base.lmcache_endpoint.empty()) {
'''
    replace_once(path, old, new)

    old = '''        if (params_base.cache_ram_mib != 0 || !params_base.lmcache_endpoint.empty()) {
'''
    new = '''        if (params_base.cache_ram_mib != 0 || !params_base.lmcache_endpoint.empty() || gpu_agent_cache_seqs > 0) {
'''
    replace_once(path, old, new)

    old = '''        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }
'''
    new = '''        // Level-1: retain KV in the unified device pool under a hidden
        // sequence id.  This is metadata-only and avoids serializing the state.
        if (prompt_cache.gpu_save(prompt, spec_state, id)) {
            return true;
        }

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }
'''
    replace_once(path, old, new)

    old = '''        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
'''
    new = '''        // Prefer the device-resident tier.  If no hidden sequence has a better
        // exact prefix, fall back to the serialized RAM/LMCache state.
        bool res = prompt_cache.gpu_load(prompt, tokens, id);
        if (!res) {
            res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        }
        if (!res) {
'''
    replace_once(path, old, new)

    old = '''        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
'''
    new = '''        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        // If all active idle slots are already clear, reclaim the oldest hidden
        // GPU agent.  It is first demoted to RAM/LMCache when configured.
        if (!res && prompt_cache && prompt_cache->gpu_demote_lru()) {
            res = true;
        }

        return res;
'''
    replace_once(path, old, new)


def main():
    patch_header()
    patch_cache_cpp()
    patch_server_context()
    print("GPU-resident agent cache source changes applied")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
