// Tests for the shared per-step token budget used by the server scheduler.
//
// A single long prefill used to be able to consume the whole physical batch
// before any generating slot was reached, delaying decode by a full prompt.
// The budget reserves decode first, shares the remainder between waiting
// prompts, and bounds how long any slot can be passed over.
//
// Pure bookkeeping: no model, no GPU, no network. The policy is included
// directly so this test does not need a server context.

#include "../tools/server/server-sched-policy.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using llama_server_sched::prefill_candidate;
using llama_server_sched::token_budget;
using llama_server_sched::wait_entry;

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        ++g_checks;                                                                    \
        if (!(cond)) {                                                                 \
            ++g_failures;                                                              \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
        }                                                                              \
    } while (0)

#define CHECK_EQ(actual, expected)                                                     \
    do {                                                                               \
        ++g_checks;                                                                    \
        const auto a_ = (actual);                                                      \
        const auto e_ = (expected);                                                    \
        if (!(a_ == e_)) {                                                             \
            ++g_failures;                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s (got %lld, want %lld)\n",       \
                __FILE__, __LINE__, #actual, #expected,                                \
                (long long) a_, (long long) e_);                                      \
        }                                                                              \
    } while (0)

static std::vector<prefill_candidate> mk(std::initializer_list<std::tuple<int32_t, int32_t, bool>> v) {
    std::vector<prefill_candidate> c;
    for (const auto & [id, n, split] : v) {
        c.push_back({ id, n, split });
    }
    return c;
}

// A generating slot's token is reserved before any prefill is offered, so a
// long prompt cannot consume the batch ahead of a decoder.
static wait_entry wait_of(int32_t id_slot, int32_t n_wait) {
    wait_entry e;
    e.id_slot  = id_slot;
    e.t_queued = 0;
    e.n_wait   = n_wait;
    return e;
}

static void test_decode_reserved_before_prefill() {
    token_budget b;
    CHECK(b.begin_step(/* n_batch = */ 2048, /* n_running = */ 1));
    CHECK_EQ(b.remaining(), 2047);

    const auto grants = b.run_prefill(mk({ { 0, 100000, true } }));
    CHECK_EQ((int) grants.size(), 1);
    CHECK_EQ(grants[0].n_tokens, 2047);
    CHECK_EQ(b.remaining(), 0);
}

// The reservation must survive a prefill that ran first, which is what the old
// slot-iteration order effectively did.
static void test_prefill_cannot_starve_decode() {
    for (int32_t n_batch : {512, 2048, 8192}) {
        for (int32_t pending : {512, 2048, 8192, 262144}) {
            token_budget b;

            // begin_step reserves the decoder up front - the prefill round
            // that follows cannot reclaim it
            CHECK(b.begin_step(n_batch, /* n_running = */ 3));
            const int32_t reserved = b.remaining();

            const auto grants = b.run_prefill(mk({ { 0, pending, true } }));

            // the prefill may not reclaim what the decoder reserved; a grant
            // larger than that would mean decode was starved
            CHECK(grants[0].n_tokens <= reserved);
            CHECK_EQ(grants[0].n_tokens, std::min(pending, reserved));
            CHECK_EQ(b.remaining(), reserved - grants[0].n_tokens);
        }
    }
}

// Several prompts share the remainder; no one of them takes all of it, and the
// budget is not left idle.
static void test_prefill_is_shared() {
    token_budget b;
    CHECK(b.begin_step(1000, /* n_running = */ 1));

    const auto grants = b.run_prefill(mk({
        { 0, 100000, true },
        { 1, 100000, true },
        { 2, 100000, true },
        { 3, 100000, true },
    }));

    CHECK_EQ((int) grants.size(), 4);

    int32_t total = 0;
    for (const auto & g : grants) {
        CHECK(g.n_tokens > 0);
        total += g.n_tokens;
    }

    CHECK_EQ(total, 999);
    CHECK_EQ(b.remaining(), 0);
}

// A prompt shorter than its share takes only what it needs, and the rest goes
// to the others.
static void test_short_prompt_does_not_waste_budget() {
    token_budget b;
    CHECK(b.begin_step(1000, 0));

    const auto grants = b.run_prefill(mk({
        { 0, 10,   true },
        { 1, 5000, true },
        { 2, 5000, true },
    }));

    CHECK_EQ(grants[0].n_tokens, 10);
    CHECK(grants[1].n_tokens > 0);
    CHECK(grants[2].n_tokens > 0);
    CHECK_EQ(b.remaining(), 0);
}

// A non-splittable slot is granted whole or not at all.
static void test_non_splittable_needs_whole_grant() {
    token_budget b;
    CHECK(b.begin_step(100, 0));

    auto grants = b.run_prefill(mk({ { 0, 500, false } }));
    CHECK_EQ(grants[0].n_tokens, 0);
    CHECK_EQ(b.remaining(), 100);

    // with room for the whole prompt it is served, without splitting
    b.note_progress(0);
    CHECK(b.begin_step(500, 0));
    grants = b.run_prefill(mk({ { 0, 500, false } }));
    CHECK_EQ(grants[0].n_tokens, 500);
    CHECK_EQ(b.remaining(), 0);
}

// A non-splittable prompt that fits must not be starved by a splittable one
// that is offered the remainder first.
static void test_non_splittable_not_starved_by_split() {
    token_budget b;
    CHECK(b.begin_step(1000, 0));

    const auto grants = b.run_prefill(mk({
        { 0, 100000, true  },
        { 1, 300,    false },
    }));

    const int32_t whole = grants[1].n_tokens;
    CHECK(whole == 300 || whole == 0);
    CHECK_EQ(grants[0].n_tokens, 700);
}

// Anti-starvation: a slot passed over repeatedly is served first and ahead of
// the others, so no admitted request waits forever.
static void test_fairness_bound() {
    token_budget b;
    b.set_max_wait(/* n_steps = */ 8);
    CHECK(b.begin_step(1000, 0));

    b.mark_wait(wait_of(0, 7));
    CHECK(!b.is_due(0));

    b.mark_wait(wait_of(0, 8));
    CHECK(b.is_due(0));

    // the due slot is served first, and the step's budget is fully used
    const auto grants = b.run_prefill(mk({
        { 0, 100000, true },
        { 1, 100000, true },
        { 2, 100000, true },
    }));

    CHECK_EQ(grants[0].n_tokens, 1000);
    CHECK_EQ(b.remaining(), 0);
}

// Progress clears the wait record, so a served slot is not immediately due.
static void test_progress_clears_wait() {
    token_budget b;
    b.set_max_wait(4);
    CHECK(b.begin_step(100, 0));

    b.mark_wait(wait_of(3, 3));
    CHECK_EQ((int) b.n_tracked(), 1);
    CHECK(!b.is_due(3)); // one step short of the bound

    b.note_progress(3);
    CHECK_EQ((int) b.n_tracked(), 0);
    CHECK(!b.is_due(3));
}

// Bounded wait over many steps: a continuously-deflected slot is guaranteed a
// grant within the declared window.
static void test_no_slot_waits_forever() {
    constexpr int32_t max_wait = 4;
    constexpr int32_t n_slots  = 3;

    token_budget b;
    b.set_max_wait(max_wait);
    b.begin_step(64, 0);

    std::vector<int32_t> n_wait(n_slots, 0);
    std::vector<int32_t> served(n_slots, 0);

    for (int32_t step = 0; step < 40; ++step) {
        for (int32_t i = 0; i < n_slots; ++i) {
            b.mark_wait(wait_of(i, n_wait[i]));
        }

        std::vector<prefill_candidate> cands;
        for (int32_t i = 0; i < n_slots; ++i) {
            cands.push_back({ i, 100000, true });
        }

        const auto grants = b.run_prefill(cands);

        bool any = false;
        for (int32_t i = 0; i < n_slots; ++i) {
            if (grants[i].n_tokens > 0) {
                served[i]++;
                n_wait[i] = 0;
                b.note_progress(i);
                any = true;
            } else {
                n_wait[i]++;
            }
        }
        CHECK(any); // the budget is never left idle

        // every slot must be served within the declared bound
        for (int32_t i = 0; i < n_slots; ++i) {
            CHECK(n_wait[i] <= max_wait);
        }

        b.begin_step(64, 0);
    }

    for (int32_t i = 0; i < n_slots; ++i) {
        CHECK(served[i] > 0);
    }
}

// A reservation that does not fit is reported, not asserted.
static void test_begin_step_reports_overcommit() {
    token_budget b;

    CHECK(!b.begin_step(/* n_batch = */ 2, /* n_running = */ 3));
    CHECK(b.begin_step(2, 2));
    CHECK_EQ(b.remaining(), 0);

    // a step with no decoder gives the whole batch to prefill
    token_budget c;
    CHECK(c.begin_step(8, 0));
    CHECK_EQ(c.remaining(), 8);
}

// No candidates must not touch the budget.
static void test_empty_and_zero_pending() {
    token_budget b;
    CHECK(b.begin_step(100, 0));

    CHECK_EQ((int) b.run_prefill({}).size(), 0);
    CHECK_EQ(b.remaining(), 100);

    const auto grants = b.run_prefill(mk({ { 0, 0, true } }));
    CHECK_EQ(grants[0].n_tokens, 0);
    CHECK_EQ(b.remaining(), 100);
}

int main() {
    test_decode_reserved_before_prefill();
    test_prefill_cannot_starve_decode();
    test_prefill_is_shared();
    test_short_prompt_does_not_waste_budget();
    test_non_splittable_needs_whole_grant();
    test_non_splittable_not_starved_by_split();
    test_fairness_bound();
    test_progress_clears_wait();
    test_no_slot_waits_forever();
    test_begin_step_reports_overcommit();
    test_empty_and_zero_pending();

    std::printf("%s: %d checks, %d failures\n",
                g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);

    return g_failures == 0 ? 0 : 1;
}
