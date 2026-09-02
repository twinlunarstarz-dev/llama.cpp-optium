#!/usr/bin/env python3
"""Apply token-stable agent/session resume optimizations to llama.cpp-optium.

This patcher deliberately edits only exact, audited source hunks.  It aborts if
any expected hunk is absent or appears more than once.  The target-model math,
KV precision, sampler order, and speculative verification remain authoritative.

What it applies:
  1. Persistent local prompt snapshots (read/MRU rather than consume-on-restore).
  2. Longest-exact-prefix cache selection for A -> B -> A agent switching.
  3. Preserve shorter branch roots instead of deleting them when a longer branch is saved.
  4. Save/restore state owned by speculative implementations alongside target/draft llama state.
  5. Versioned multiplexing of speculative implementation state, including MTP pending hidden state.
  6. Zero-copy speculative prompt view for text-only prompts; multimodal keeps the existing copy path.

It intentionally DOES NOT skip common_speculative_process() after target verification.
That deeper optimization changes MTP state-transition timing and is not enabled until
hardware-level token/state equivalence is demonstrated.  This is a stability guard,
not a missing performance flag.

Usage:
  python3 scripts/apply-agent-resume-optimizations.py --check
  python3 scripts/apply-agent-resume-optimizations.py --apply
  python3 scripts/apply-agent-resume-optimizations.py --apply --build

Recommended models.ini for long-context multi-agent use after applying:
  cache-prompt = true
  cache-ram = -1        # or a measured 8192/16384 MiB host-RAM budget

The source default for cache-idle-slots is already true.
"""

from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "task_h": ROOT / "tools/server/server-task.h",
    "task_cpp": ROOT / "tools/server/server-task.cpp",
    "context_cpp": ROOT / "tools/server/server-context.cpp",
    "spec_cpp": ROOT / "common/speculative.cpp",
}

class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise PatchError(f"{label}: expected exactly one source match, found {n}")
    return text.replace(old, new, 1)


def patch_task_h(s: str) -> str:
    s = replace_once(s, '''struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};''', '''struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    // State owned by speculative implementations (for example MTP's pending
    // target hidden row).  Target/draft llama state alone is not sufficient to
    // resume a speculative agent exactly after another agent used the slot.
    std::vector<uint8_t> spec;

    size_t size() const {
        return main.size() + drft.size() + spec.size();
    }
};''', "server_prompt_data.spec")

    s = replace_once(s, '''    std::unique_ptr<common_lmcache_client> lmcache;
    std::string lmcache_namespace;

    size_t size() const;''', '''    std::unique_ptr<common_lmcache_client> lmcache;
    std::string lmcache_namespace;

    // Result metadata for the most recent load().  This avoids changing the
    // public load signature while allowing server_slot to restore the matching
    // speculative-driver state after the llama target/draft state is restored.
    bool restored_any_state = false;
    bool restored_spec_state_valid = false;
    std::vector<uint8_t> restored_spec_state;

    size_t size() const;''', "server_prompt_cache.restore-metadata")
    return s


def patch_task_cpp(s: str) -> str:
    # A shorter cached branch is not obsolete.  It can be the branch root that
    # agent A needs after B/C were serviced.
    s = replace_once(s, '''    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\\n");
            return nullptr;
        }
    }
''', '''    // Replace only an exact snapshot.  A shorter prefix is a valid branch/session
    // root and must survive A -> B -> A agent switching.
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size() &&
                it->prompt.tokens.size() == prompt.tokens.size()) {
            SRV_TRC(" - replacing exact cached prompt with length %d\\n", cur_lcp_len);
            states.erase(it);
            break;
        }
    }
''', "cache.exact-dedupe")

    s = replace_once(s, '''    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

''', '', "cache.keep-branch-roots")

    # Reset load metadata at the start of each lookup.
    s = replace_once(s, '''bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (lmcache && !tokens_new.has_mtmd) {''', '''bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    restored_any_state = false;
    restored_spec_state_valid = false;
    restored_spec_state.clear();

    if (lmcache && !tokens_new.has_mtmd) {''', "cache.reset-load-metadata")

    # Remote cache: prefer absolute reusable token count.  Remote v1 records do
    # not carry speculative state, so server_slot will conservatively reject a
    # remote restore for a stateful speculator and fall back to recomputation.
    s = replace_once(s, '''                const int lcp_base = prompt.tokens.get_common_prefix(tokens_new);
                const int lcp_remote = remote.prompt.tokens.get_common_prefix(tokens_new);
                const float keep_base = prompt.tokens.size() > 0 ? float(lcp_base) / prompt.tokens.size() : -1.0f;
                const float sim_base = tokens_new.size() > 0 ? float(lcp_base) / tokens_new.size() : 0.0f;
                const float keep_remote = remote.prompt.tokens.size() > 0 ? float(lcp_remote) / remote.prompt.tokens.size() : 0.0f;
                const float sim_remote = tokens_new.size() > 0 ? float(lcp_remote) / tokens_new.size() : 0.0f;
                if (keep_remote >= 0.25f && keep_base < keep_remote && sim_base < sim_remote) {''', '''                const int lcp_base = prompt.tokens.get_common_prefix(tokens_new);
                const int lcp_remote = remote.prompt.tokens.get_common_prefix(tokens_new);
                const float keep_remote = remote.prompt.tokens.size() > 0 ? float(lcp_remote) / remote.prompt.tokens.size() : 0.0f;
                // Maximize actual avoided target work, not two relative ratios.
                if (keep_remote >= 0.25f && lcp_remote > lcp_base) {''', "cache.remote-absolute-lcp")

    s = replace_once(s, '''                    prompt = std::move(remote.prompt);
                    SRV_TRC(" - restored %zu-token prompt state from LMCache\\n", tokens.size());
                    return true;''', '''                    prompt = std::move(remote.prompt);
                    restored_any_state = true;
                    SRV_TRC(" - restored %zu-token prompt state from LMCache\\n", tokens.size());
                    return true;''', "cache.remote-restored-flag")

    s = replace_once(s, '''                SRV_TRC(" - LMCache prompt is not better, f_keep = %.3f, f_sim = %.3f\\n", keep_remote, sim_remote);''', '''                SRV_TRC(" - LMCache prompt is not better, lcp = %d (base = %d), f_keep = %.3f\\n",
                        lcp_remote, lcp_base, keep_remote);''', "cache.remote-log")

    s = replace_once(s, '''    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\\n", f_keep_best, f_sim_best);''', '''    int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = tokens_new.size() > 0 ? float(lcp_best) / tokens_new.size() : 0.0f;

    SRV_TRC(" - looking for better prompt, base lcp = %d, f_keep = %.3f, f_sim = %.3f\\n",
            lcp_best, f_keep_best, f_sim_best);''', "cache.local-base-lcp")

    s = replace_once(s, '''        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }''', '''        if (lcp_cur > lcp_best) {
            lcp_best    = lcp_cur;
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }''', "cache.local-absolute-lcp")

    # A local restore is a read, not a consume.  Retain the serialized bytes and
    # touch the list node as MRU.  Copy prompt metadata because it remains owned
    # by the cache node.
    s = replace_once(s, '''        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);''', '''        {
            const auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\\n", size);

                return false;
            }
        }

        {
            const auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\\n", size);

                    return false;
                }
            }
        }

        prompt = it_best->prompt.clone();
        restored_any_state = true;
        restored_spec_state = it_best->data.spec;
        restored_spec_state_valid = !restored_spec_state.empty();

        // True LRU behavior: a successfully restored agent snapshot becomes MRU.
        states.splice(states.end(), states, it_best);''', "cache.reusable-lru-restore")

    return s


def patch_context_cpp(s: str) -> str:
    # Save and restore speculative-driver state along with target/draft llama state.
    s = replace_once(s, '''        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;''', '''        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        std::vector<uint8_t> spec_state;
        const bool has_spec_state = spec && common_speculative_get_state(spec, id, spec_state);

        const size_t cur_size = cur_size_tgt + cur_size_dft + spec_state.size();''', "slot.save-spec-size")

    s = replace_once(s, '''        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        prompt_cache.store_remote(cur);''', '''        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        if (has_spec_state) {
            cur->data.spec = std::move(spec_state);
        }

        prompt_cache.store_remote(cur);''', "slot.save-spec-data")

    s = replace_once(s, '''    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\\n");
        }

        return res;
    }''', '''    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        // Determine whether this speculative stack owns persistent per-sequence
        // state.  Stateless n-gram helpers do not require a sidecar restore.
        std::vector<uint8_t> spec_probe;
        const bool spec_requires_state = spec && common_speculative_get_state(spec, id, spec_probe);

        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\\n");
            return false;
        }

        if (prompt_cache.restored_any_state && spec_requires_state) {
            if (!prompt_cache.restored_spec_state_valid) {
                // Legacy/remote cache records do not carry the sidecar.  Never
                // combine another agent's speculative state with restored KV.
                SLT_WRN(*this, "%s", "cached llama state has no matching speculative state; recomputing safely\\n");
                return false;
            }
            common_speculative_set_state(spec, id, prompt_cache.restored_spec_state);
        }

        return true;
    }''', "slot.restore-spec-data")

    # Zero-copy for ordinary text prompts.  The prompt vector is not mutated until
    # after common_speculative_draft() returns.  Multimodal keeps the filtered copy.
    s = replace_once(s, '''                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };''', '''                        const llama_tokens * spec_prompt_ptr = nullptr;
                        if (slot.prompt.tokens.has_mtmd) {
                            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();
                            spec_prompt_ptr = &slot.spec_prompt;
                        } else {
                            // Avoid copying the entire 50k/100k/1M-token context on
                            // every speculative generation round.
                            spec_prompt_ptr = &slot.prompt.tokens.get_tokens();
                        }

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ spec_prompt_ptr,
                            /* .result   = */ &slot.spec_draft,
                        };''', "spec.zero-copy-prompt")

    return s


def patch_spec_cpp(s: str) -> str:
    # MTP's pending_h is the cross-call boundary state.  Draft KV without the
    # matching pending target hidden row is not a true continuation.
    old = '''    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }
};'''
    new = '''    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }

        constexpr uint32_t magic = 0x3150544d; // "MTP1" little-endian
        constexpr uint32_t version = 1;
        const uint32_t width = (uint32_t) n_embd;
        const size_t header_size = 3*sizeof(uint32_t);
        const size_t row_bytes = (size_t) n_embd*sizeof(float);

        data.resize(header_size + row_bytes);
        size_t off = 0;
        std::memcpy(data.data() + off, &magic, sizeof(magic)); off += sizeof(magic);
        std::memcpy(data.data() + off, &version, sizeof(version)); off += sizeof(version);
        std::memcpy(data.data() + off, &width, sizeof(width)); off += sizeof(width);
        std::memcpy(data.data() + off, pending_h[seq_id].data(), row_bytes);
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        constexpr uint32_t magic_expected = 0x3150544d;
        constexpr uint32_t version_expected = 1;
        const size_t header_size = 3*sizeof(uint32_t);
        const size_t row_bytes = (size_t) n_embd*sizeof(float);
        if (data.size() != header_size + row_bytes) {
            return;
        }

        uint32_t magic = 0, version = 0, width = 0;
        size_t off = 0;
        std::memcpy(&magic, data.data() + off, sizeof(magic)); off += sizeof(magic);
        std::memcpy(&version, data.data() + off, sizeof(version)); off += sizeof(version);
        std::memcpy(&width, data.data() + off, sizeof(width)); off += sizeof(width);
        if (magic != magic_expected || version != version_expected || width != (uint32_t) n_embd) {
            return;
        }

        std::memcpy(pending_h[seq_id].data(), data.data() + off, row_bytes);
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        i_last[seq_id] = -1;
        if (chain_heads) {
            chain_h[seq_id].clear();
        }
    }
};'''
    s = replace_once(s, old, new, "mtp.persist-pending-h")

    # Multiplex all stateful speculative implementations.  This removes the old
    # TODO/ambiguity where the first state was returned but the same raw bytes
    # were broadcast to every implementation on restore.
    s = replace_once(s, '''// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}''', '''bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    constexpr uint32_t magic = 0x31504353; // "SCP1" little-endian
    constexpr uint32_t version = 1;

    struct item {
        uint32_t type;
        std::vector<uint8_t> data;
    };
    std::vector<item> items;
    for (auto & impl : spec->impls) {
        std::vector<uint8_t> state;
        if (impl->get_state(seq_id, state)) {
            items.push_back({ (uint32_t) impl->type, std::move(state) });
        }
    }
    if (items.empty()) {
        data.clear();
        return false;
    }

    size_t total = 3*sizeof(uint32_t);
    for (const auto & item : items) {
        total += sizeof(uint32_t) + sizeof(uint64_t) + item.data.size();
    }
    data.resize(total);

    size_t off = 0;
    auto put = [&](const void * src, size_t n) {
        std::memcpy(data.data() + off, src, n);
        off += n;
    };
    const uint32_t count = (uint32_t) items.size();
    put(&magic, sizeof(magic));
    put(&version, sizeof(version));
    put(&count, sizeof(count));
    for (const auto & item : items) {
        const uint64_t n = item.data.size();
        put(&item.type, sizeof(item.type));
        put(&n, sizeof(n));
        if (n > 0) put(item.data.data(), (size_t) n);
    }
    return true;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    constexpr uint32_t magic_expected = 0x31504353;
    constexpr uint32_t version_expected = 1;
    if (data.size() >= 3*sizeof(uint32_t)) {
        size_t off = 0;
        auto get = [&](void * dst, size_t n) -> bool {
            if (off > data.size() || data.size() - off < n) return false;
            std::memcpy(dst, data.data() + off, n);
            off += n;
            return true;
        };
        uint32_t magic = 0, version = 0, count = 0;
        if (get(&magic, sizeof(magic)) && get(&version, sizeof(version)) && get(&count, sizeof(count)) &&
                magic == magic_expected && version == version_expected) {
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t type = 0;
                uint64_t n = 0;
                if (!get(&type, sizeof(type)) || !get(&n, sizeof(n)) || n > data.size() - off) {
                    return;
                }
                std::vector<uint8_t> state((size_t) n);
                if (n > 0 && !get(state.data(), (size_t) n)) return;
                for (auto & impl : spec->impls) {
                    if ((uint32_t) impl->type == type) {
                        impl->set_state(seq_id, state);
                        break;
                    }
                }
            }
            return;
        }
    }

    // Backward-compatible fallback for old single-implementation checkpoint data.
    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}''', "spec.multiplex-state")

    return s


def produce() -> Dict[Path, Tuple[str, str]]:
    transforms = {
        FILES["task_h"]: patch_task_h,
        FILES["task_cpp"]: patch_task_cpp,
        FILES["context_cpp"]: patch_context_cpp,
        FILES["spec_cpp"]: patch_spec_cpp,
    }
    result: Dict[Path, Tuple[str, str]] = {}
    for path, fn in transforms.items():
        if not path.exists():
            raise PatchError(f"missing source file: {path.relative_to(ROOT)}")
        old = path.read_text()
        new = fn(old)
        if old == new:
            raise PatchError(f"no changes generated for {path.relative_to(ROOT)}")
        result[path] = (old, new)
    return result


def show_diff(changes: Dict[Path, Tuple[str, str]]) -> None:
    for path, (old, new) in changes.items():
        rel = str(path.relative_to(ROOT))
        diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
        sys.stdout.writelines(diff)


def apply_changes(changes: Dict[Path, Tuple[str, str]]) -> None:
    backup_root = ROOT / ".agent-resume-backup"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    for path, (old, new) in changes.items():
        rel = path.relative_to(ROOT)
        backup = backup_root / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(old)
        path.write_text(new)
    print(f"Backups written to {backup_root.relative_to(ROOT)}/")


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate hunks and print diff without writing")
    mode.add_argument("--apply", action="store_true", help="apply audited hunks and print git diff")
    ap.add_argument("--build", action="store_true", help="after --apply, run a CPU llama-server compile check")
    args = ap.parse_args()

    try:
        changes = produce()
    except PatchError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("No files were modified.", file=sys.stderr)
        return 2

    if args.check:
        show_diff(changes)
        print("\nAll guarded hunks matched exactly; no files modified.")
        return 0

    apply_changes(changes)
    run(["git", "diff", "--check"])
    run(["git", "diff", "--stat"])
    run(["git", "diff", "--", "tools/server/server-task.h", "tools/server/server-task.cpp",
         "tools/server/server-context.cpp", "common/speculative.cpp"])

    if args.build:
        build = ROOT / "build-agent-resume-check"
        run(["cmake", "-S", ".", "-B", str(build), "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF",
             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=ON",
             "-DCMAKE_BUILD_TYPE=Release"])
        run(["cmake", "--build", str(build), "--target", "llama-server", "-j", "2"])

    print("\nApplied token-stable agent resume optimizations.")
    print("For long-context multi-agent switching, increase models.ini cache-ram from 2048 MiB.")
    print("Use cache-ram=-1 for unlimited host-RAM cache, or measure and choose 8192/16384 MiB.")
    print("Do not commit until your normal CUDA build and A/B token-stability tests pass.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
