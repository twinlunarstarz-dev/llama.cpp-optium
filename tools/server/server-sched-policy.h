// Shared token budget for one scheduler step.
//
// pre_decode() used to fill the physical batch in slot iteration order. A slot
// with a long prompt therefore consumed the whole of n_batch before any
// generating slot was reached, so a single prefill could delay decode by an
// entire prompt. vLLM instead grants one budget per step, running requests
// first (scheduler.py::Scheduler.schedule, commit 378504a5).
//
// The policy here is the same idea reduced to what the server can express:
// decode is reserved before any prefill is offered, and the remaining budget is
// shared between waiting prompts. Fairness is bounded so a prompt can never be
// deferred forever, and a non-splittable slot is still granted whole.

#ifndef LLAMA_SERVER_SCHED_POLICY_H
#define LLAMA_SERVER_SCHED_POLICY_H

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace llama_server_sched {

// A slot waiting for its turn in the shared budget.
struct wait_entry {
    int32_t id_slot  = -1;
    int64_t t_queued = 0; // absolute us, from the slot's admission timestamp
    int32_t n_wait   = 0; // consecutive steps this slot was passed over
};

// A pending prompt competing for the prefill round.
struct prefill_candidate {
    int32_t id_slot        = -1;
    int32_t n_pending      = 0; // prompt tokens still to evaluate
    bool   can_split       = false;
};

// Result of the prefill round: tokens granted to each candidate, in the same
// order as the input.
struct prefill_grant {
    int32_t id_slot = -1;
    int32_t n_tokens = 0;
};

// Grants are computed per scheduler step:
//
//   begin_step() reserves the tokens every generating slot needs, so prefill
//   can never be offered budget that a decoder is waiting for;
//   run_prefill() then divides what is left between the waiting prompts.
class token_budget {
public:
    // Start a step with a physical capacity of n_batch, reserving n_running
    // tokens for the generating slots.
    //
    // Reserving up front is what makes the policy running-first: a later
    // prefill grant cannot reclaim it. If the reservation does not fit, the
    // decode is reported so the caller can shrink the step instead of
    // asserting - the same condition the old
    // GGML_ASSERT(handle_last_sampled_token) guarded.
    bool begin_step(int32_t n_batch, int32_t n_running) {
        n_budget = n_batch;

        if (n_running < 0 || n_running > n_budget) {
            return false;
        }

        n_budget        -= n_running;
        n_running_total = n_running;
        return true;
    }

    // Tokens already set aside for the generating slots this step.
    int32_t running_reserved() const {
        return n_running_total;
    }

    // Budget still available to prefill.
    int32_t remaining() const {
        return n_budget;
    }

    // max consecutive steps a slot may be passed over
    void set_max_wait(int32_t n_steps) {
        max_wait_steps = n_steps;
    }

    // Fairness: after max_wait_steps consecutive deferrals the slot is served
    // first, so no admitted request is postponed forever.
    void mark_wait(const wait_entry & entry) {
        auto & e = wait[entry.id_slot];

        e.id_slot  = entry.id_slot;
        e.t_queued = entry.t_queued;
        e.n_wait   = entry.n_wait;
    }

    bool is_due(int32_t id_slot) const {
        const auto it = wait.find(id_slot);
        if (it == wait.end()) {
            return false;
        }
        return it->second.n_wait >= max_wait_steps;
    }

    void note_progress(int32_t id_slot) {
        wait.erase(id_slot);
    }

    // Slots that received a grant in the prefill round of the current step.
    // The batch fill consults this, so a slot the policy passed over is not
    // filled anyway - otherwise the round would be a no-op.
    void mark_granted(int32_t id_slot, int32_t n_tokens) {
        granted[id_slot] = n_tokens;
    }

    bool has_slot(int32_t id_slot) const {
        return granted.find(id_slot) != granted.end();
    }

    int32_t grant_for(int32_t id_slot) const {
        const auto it = granted.find(id_slot);
        return it == granted.end() ? 0 : it->second;
    }

    // Drop the per-step grant table; call at the start of each step.
    void clear_grants() {
        granted.clear();
    }

    size_t n_tracked() const {
        return wait.size();
    }

    // The prefill round. Every candidate either receives a positive grant or
    // is recorded as having waited; the returned vector is index-aligned with
    // `cands`.
    //
    // Rules, in order:
    //   1. a due slot is served first, from the whole remainder;
    //   2. every other slot gets an equal share of what is left, rounded up
    //      so each makes at least one token of progress;
    //   3. a non-splittable slot is granted whole or not at all;
    //   4. the share is computed once, so leftover budget is not left idle -
    //      it is handed to the slots that can still take it.
    std::vector<prefill_grant> run_prefill(const std::vector<prefill_candidate> & cands) {
        std::vector<prefill_grant> grants(cands.size());

        for (size_t i = 0; i < cands.size(); ++i) {
            grants[i].id_slot = cands[i].id_slot;
        }

        if (n_budget <= 0 || cands.empty()) {
            return grants;
        }

        // 1. due slots first, so the fairness bound actually bounds the wait
        std::vector<size_t> order;
        order.reserve(cands.size());

        for (size_t i = 0; i < cands.size(); ++i) {
            if (is_due(cands[i].id_slot) && cands[i].n_pending > 0) {
                order.push_back(i);
            }
        }

        std::vector<bool> served(cands.size(), false);

        for (const size_t i : order) {
            int32_t take = cands[i].can_split
                ? std::min(cands[i].n_pending, n_budget)
                : (cands[i].n_pending <= n_budget ? cands[i].n_pending : 0);

            if (take > 0) {
                n_budget          -= take;
                grants[i].n_tokens = take;
                served[i]          = true;
            }
        }

        // 2. equal share of what is left, fixed once so the remainder is not
        //    left idle as the budget shrinks
        std::vector<size_t> rest;
        for (size_t i = 0; i < cands.size(); ++i) {
            if (!served[i] && cands[i].n_pending > 0) {
                rest.push_back(i);
            }
        }

        if (rest.empty()) {
            return grants;
        }

        // 3. a non-splittable slot is served whole when the budget can hold
        //    it, so it is never starved by a splittable slot that could have
        //    taken the room; otherwise it must wait for a roomier step
        for (const size_t i : rest) {
            if (cands[i].can_split || cands[i].n_pending > n_budget) {
                continue;
            }

            n_budget          -= cands[i].n_pending;
            grants[i].n_tokens = cands[i].n_pending;
            served[i]          = true;
        }

        std::vector<size_t> split;
        for (const size_t i : rest) {
            if (!served[i] && cands[i].can_split) {
                split.push_back(i);
            }
        }

        if (split.empty() || n_budget <= 0) {
            return grants;
        }

        // 4. one share, computed against the whole remainder, so budget left
        //    over as slots run out is not left idle
        const int32_t share = (n_budget + (int32_t) split.size() - 1) / (int32_t) split.size();

        for (size_t k = 0; k < split.size(); ++k) {
            const size_t i = split[k];

            int32_t take = std::min({ share, cands[i].n_pending, n_budget });

            // the last slot served takes the remainder, so no budget is idle
            if (k + 1 == split.size()) {
                take = std::min({ n_budget, cands[i].n_pending });
            }

            n_budget          -= take;
            grants[i].n_tokens = take;
            served[i]          = true;
        }

        return grants;
    }

private:
    int32_t n_budget        = 0;
    int32_t n_running_total = 0;
    int32_t max_wait_steps  = 8;

    std::map<int32_t, wait_entry> wait;

    // per-step prefill grants, keyed by slot
    std::map<int32_t, int32_t> granted;
};

} // namespace llama_server_sched

#endif // LLAMA_SERVER_SCHED_POLICY_H
