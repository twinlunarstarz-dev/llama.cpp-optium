// Regression tests for the recurrent physical-row pool and slot-capacity honesty.
//
// The recurrent cache has more logical sequences (server slots, --parallel) than
// it has physical R/S rows, because a row costs n_embd_r + n_embd_s F32 values
// per recurrent layer and is sized from measured free VRAM rather than from the
// logical slot count.  That split is only safe if:
//
//   P1  row occupancy is tracked explicitly, so "how many states are actually
//       resident" is a number the server can enforce and report;
//   P2  rows are reclaimed when a sequence stops using them, so the pool does
//       not drift toward permanently-full and does not leak;
//   P3  a physical row is owned by at most one logical sequence, and a logical
//       sequence owns at most one row - this is the no-cross-slot-bleed
//       invariant, since a bleed means one agent reading another's R/S state;
//   P4  occupancy is reconciled from the authoritative cell table, so the
//       accounting cannot drift away from the state it describes;
//   P5  asking for more rows than the pool holds is refused up front, instead
//       of aborting later inside slot search.
//
// No model, no GPU, and no network: the pool is pure bookkeeping.
//
// (Upstream finding reference: `mamba`/`jamba`/`nemotron_h`/`plamo2` and the
// hybrid `qwen3next`/`qwen35` paths all share llama_memory_recurrent.)

#include "../src/llama-recurrent-rows.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

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

using llama_recurrent_rows::row_pool;

// A stand-in for the cell table: entry i is the logical sequence owning cell i,
// or -1 when the cell is empty.  llama_memory_recurrent derives occupancy from
// exactly this information.
static std::vector<int32_t> table(const std::vector<std::pair<int, int>> & live) {
    std::vector<int32_t> t;
    for (const auto & [cell, seq] : live) {
        if ((int) t.size() <= cell) {
            t.resize(cell + 1, -1);
        }
        t[cell] = seq;
    }
    return t;
}

static void test_p1_counts_and_bytes() {
    // capacity and per-row byte cost are the two numbers the server needs in
    // order to decide whether an incoming request can be admitted.
    row_pool pool(/* capacity = */ 4, /* plane_bytes = */ 2048);

    CHECK_EQ(pool.capacity(), 4u);
    CHECK_EQ(pool.n_resident(), 0u);
    CHECK_EQ(pool.peak(), 0u);
    CHECK_EQ(pool.bytes_resident(), (uint64_t) 0);

    CHECK(pool.acquire(/* row = */ 0, /* seq = */ 7));
    CHECK_EQ(pool.n_resident(), 1u);
    CHECK_EQ(pool.bytes_resident(), (uint64_t) 2048);

    CHECK(pool.acquire(3, 9));
    CHECK_EQ(pool.n_resident(), 2u);
    CHECK_EQ(pool.bytes_resident(), (uint64_t) 4096);
    CHECK_EQ(pool.peak(), 2u);

    // byte cost must track occupancy exactly, not a high-water mark
    CHECK(pool.release_row(0));
    CHECK_EQ(pool.n_resident(), 1u);
    CHECK_EQ(pool.bytes_resident(), (uint64_t) 2048);
    CHECK_EQ(pool.peak(), 2u);   // peak is remembered, bytes are not

    // a pool with no rows can never admit
    row_pool empty(/* capacity = */ 0, /* plane_bytes = */ 2048);
    CHECK_EQ(empty.capacity(), 0u);
    CHECK(!empty.acquire(0, 0));
    CHECK_EQ(empty.n_resident(), 0u);
}

static void test_p2_reclaim_and_no_leak() {
    row_pool pool(/* capacity = */ 3, /* plane_bytes = */ 16);

    // churn well past capacity; every release must be matched by an acquire
    for (int i = 0; i < 1000; ++i) {
        CHECK(pool.acquire((uint32_t) (i % 3), (int32_t) i));
        CHECK(pool.release_row((uint32_t) (i % 3)));
        CHECK_EQ(pool.n_resident(), 0u);
    }
    CHECK_EQ(pool.peak(), 1u);

    // releasing a row that is already free is reported, not silently accepted
    CHECK(!pool.release_row(0));
    CHECK_EQ(pool.n_resident(), 0u);

    // releasing a row out of range is refused
    CHECK(!pool.release_row(3));

    // per-sequence release frees the row that sequence owned
    CHECK(pool.acquire(1, 42));
    CHECK_EQ(pool.n_resident(), 1u);
    CHECK_EQ(pool.release_seq(42), 1);
    CHECK_EQ(pool.n_resident(), 0u);

    // releasing a sequence that owns nothing changes nothing
    CHECK_EQ(pool.release_seq(42), -1);
    CHECK_EQ(pool.n_resident(), 0u);

    // full reset drops every owner and every counter except peak
    CHECK(pool.acquire(0, 1));
    CHECK(pool.acquire(1, 2));
    CHECK(pool.acquire(2, 3));
    CHECK_EQ(pool.n_resident(), 3u);
    pool.reset();
    CHECK_EQ(pool.n_resident(), 0u);
    CHECK_EQ(pool.bytes_resident(), (uint64_t) 0);
    CHECK_EQ(pool.peak(), 3u);
    CHECK(pool.free_count() == pool.capacity());
}

static void test_p3_no_cross_slot_bleed() {
    row_pool pool(/* capacity = */ 4, /* plane_bytes = */ 8);

    CHECK(pool.acquire(0, 100));
    CHECK(pool.acquire(1, 101));

    // a row already owned by another sequence must never be re-acquired: that is
    // exactly the bleed where one slot would read another slot's R/S state
    CHECK(!pool.acquire(0, 999));
    CHECK_EQ(pool.owner_of(0), 100);

    // a sequence already holding a row must not silently take a second one
    CHECK(!pool.acquire(2, 100));
    CHECK_EQ(pool.n_resident(), 2u);
    CHECK_EQ(pool.owner_of(2), -1);

    // distinct sequences on distinct rows, and each sees only its own row
    CHECK_EQ(pool.row_of(100), 0);
    CHECK_EQ(pool.row_of(101), 1);
    CHECK_EQ(pool.row_of(999), -1);

    // freeing one sequence's row must not disturb the other's ownership
    CHECK_EQ(pool.release_seq(100), 0);
    CHECK_EQ(pool.owner_of(0), -1);
    CHECK_EQ(pool.owner_of(1), 101);
    CHECK_EQ(pool.n_resident(), 1u);
}

static void test_p4_reconcile_matches_cells() {
    // The pool must be reconcilable from the authoritative cell table, and the
    // reconciled state must match what incremental acquire/release produced.
    row_pool pool(/* capacity = */ 6, /* plane_bytes = */ 512);

    // 6 live cells; cell 2 and cell 5 share logical sequence 20 via seq_id sets
    // collapsed to one owner each in the table
    const std::vector<int32_t> t = table({ {0, 10}, {1, 11}, {2, 20}, {3, 11}, {4, 10}, {5, 20} });

    std::vector<int32_t> expected_owner(t.size(), -1);
    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i] >= 0) {
            expected_owner[i] = t[i];
        }
    }
    pool.reconcile(expected_owner);

    CHECK_EQ(pool.n_resident(), 6u);
    CHECK_EQ(pool.bytes_resident(), (uint64_t) 6 * 512);
    for (size_t i = 0; i < t.size(); ++i) {
        CHECK_EQ(pool.owner_of((uint32_t) i), t[i]);
    }

    // reconcile is idempotent
    pool.reconcile(expected_owner);
    CHECK_EQ(pool.n_resident(), 6u);

    // reconcile after the cells drain back to empty leaves nothing resident
    pool.reconcile(std::vector<int32_t>(t.size(), -1));
    CHECK_EQ(pool.n_resident(), 0u);
    CHECK_EQ(pool.bytes_resident(), (uint64_t) 0);

    // reconcile must refuse a table larger than the pool, and must leave the
    // pool untouched when it does
    CHECK(!pool.reconcile(std::vector<int32_t>(pool.capacity() + 1, -1)));
    CHECK_EQ(pool.n_resident(), 0u);
}

static void test_p5_admission_refused_not_aborted() {
    row_pool pool(/* capacity = */ 2, /* plane_bytes = */ 64);

    CHECK_EQ(pool.can_admit(0), true);
    CHECK_EQ(pool.can_admit(2), true);
    CHECK_EQ(pool.can_admit(3), false);

    CHECK(pool.acquire(0, 1));
    CHECK_EQ(pool.can_admit(0), true);
    CHECK_EQ(pool.can_admit(1), true);
    CHECK_EQ(pool.can_admit(2), false);

    CHECK(pool.acquire(1, 2));
    CHECK_EQ(pool.can_admit(1), false);

    // freeing one row re-opens exactly one admission
    CHECK(pool.release_row(0));
    CHECK_EQ(pool.can_admit(1), true);
    CHECK_EQ(pool.can_admit(2), false);

    // the ceiling the server enforces is the physical capacity, not the
    // advertised logical slot count
    CHECK_EQ(pool.capacity(), 2u);
}

static void test_ownership_invariant_under_random_churn() {
    // Property check: after any sequence of acquires and releases, the pool's
    // owner map agrees with an independent model of who owns what.
    row_pool pool(/* capacity = */ 8, /* plane_bytes = */ 32);

    // Independent mirror of the pool: forward map row->seq and reverse map
    // seq->row, so the model never consults the code under test.
    std::vector<int32_t> model(pool.capacity(), -1);
    std::vector<int32_t> model_row(40, -1);

    uint32_t seed = 12345;
    const auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 16) % 64u;
    };

    for (int step = 0; step < 20000; ++step) {
        const uint32_t row  = rnd() % pool.capacity();
        const int32_t  seq  = (int32_t) (rnd() % 40u);

        if (rnd() % 2u) {
            const bool want = model[row] == -1 && model_row[seq] == -1;
            const bool got  = pool.acquire(row, seq);
            CHECK_EQ(got, want);
            if (got) {
                model[row]    = seq;
                model_row[seq] = (int32_t) row;
            }
        } else {
            const bool got = pool.release_row(row);
            CHECK_EQ(got, model[row] != -1);
            if (got) {
                model_row[model[row]] = -1;
                model[row] = -1;
            }
        }

        if ((step % 512) == 0) {
            int32_t n_live = 0;
            for (auto v : model) {
                n_live += v >= 0 ? 1 : 0;
            }
            CHECK_EQ(pool.n_resident(), n_live);
            CHECK_EQ(pool.bytes_resident(), (uint64_t) n_live * 32);

            // no bleed: no row has two owners, no sequence has two rows
            std::set<int32_t> seen_seqs;
            for (uint32_t r = 0; r < pool.capacity(); ++r) {
                const int32_t o = pool.owner_of(r);
                CHECK(model[r] == o);
                if (o >= 0) {
                    CHECK(seen_seqs.insert(o).second);
                }
            }
        }
    }

    pool.reset();
    CHECK_EQ(pool.n_resident(), 0u);
}

int main() {
    test_p1_counts_and_bytes();
    test_p2_reclaim_and_no_leak();
    test_p3_no_cross_slot_bleed();
    test_p4_reconcile_matches_cells();
    test_p5_admission_refused_not_aborted();
    test_ownership_invariant_under_random_churn();

    printf("%s: %d checks, %d failures\n", __func__, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
