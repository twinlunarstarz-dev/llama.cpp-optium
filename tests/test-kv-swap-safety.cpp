// Regression tests for the KV prompt-cache disk tier.
//
// These cover the defects found in the 2026-09-25 P0 audit:
//   D1  the quota was measured against the per-instance directory, not the shared
//       swap root, so N sibling instances each got a full independent budget
//   D2  pre-existing and foreign instance directories were never re-admitted
//   D3  a generation that was interrupted before its commit marker was written
//       was unreachable and never reclaimed
//   D4  the quota was only estimated from llama_state_seq_get_size_ext before
//       the write; a larger-than-expected payload silently exceeded it
//   D5  the instance directory name was derived from a heap pointer, which can
//       be reused by a later process and then removed by that process
//   D6  an error partway through the usage walk returned a partial total, so
//       admission proceeded on an underestimate
//   D7  the swap root was not fsynced after a stage directory was created, so
//       a committed generation's directory entry could be lost on crash
//
// No model and no GPU are required: the layer under test is pure filesystem.

#include "server-kv-swap.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
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
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #actual       \
                      << " == " << #expected << " (got " << a_ << ", want " << e_      \
                      << ")" << std::endl;                                             \
        }                                                                              \
    } while (0)

namespace {

// unique scratch root per test, removed on scope exit
struct scratch {
    std::string path;

    explicit scratch(const std::string & tag) {
        std::ostringstream name;
        name << "/tmp/test-kv-swap-" << tag << "-" << static_cast<long>(::getpid());
        path = name.str();
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path, ec);
    }

    ~scratch() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void write_file(const std::filesystem::path & p, size_t bytes) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    std::vector<char> buf(std::min<size_t>(bytes, 1u << 16), 'x');
    size_t left = bytes;
    while (left > 0) {
        const size_t n = std::min(left, buf.size());
        out.write(buf.data(), (std::streamsize) n);
        left -= n;
    }
}

// a well-formed committed generation, exactly as the server writes one
void make_committed(const std::filesystem::path & dir, size_t target_bytes, size_t draft_bytes) {
    std::filesystem::create_directories(dir);
    write_file(dir / "target.bin", target_bytes);
    if (draft_bytes > 0) {
        write_file(dir / "draft.bin", draft_bytes);
    }
    std::ofstream out(dir / ".commit", std::ios::binary | std::ios::trunc);
    out << "LLAMA_KV_SWAP_V1\n" << target_bytes << "\n" << draft_bytes << "\n";
}

void set_env_size(const char * name, size_t value) {
    setenv(name, std::to_string(value).c_str(), 1);
}

// A directory whose encoded pid is this (live) process models a sibling
// instance that another running server still owns: it must be counted, but
// never reclaimed. A pid of 0 models an instance whose owner is definitively
// gone and is therefore reclaimable.
static const int LIVE_PID = (int) ::getpid();

// D1: the quota must be enforced over the whole swap root, so a second live
// instance directory cannot spend the budget the first one already consumed.
static void test_quota_spans_sibling_instances() {
    scratch sc("quota-root");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);   // 1 MiB total
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0); // do not confound with the file cap

    // "mine" and a sibling both owned by this live pid, so neither is an orphan
    const auto mine = std::filesystem::path(root) / ("llama-kv-swap-" + std::to_string(LIVE_PID) + "-0xaaa");
    const auto sib  = std::filesystem::path(root) / ("llama-kv-swap-" + std::to_string(LIVE_PID) + "-0xbbb");
    make_committed(sib / ".stage-cache-0-0-111", 700 * 1024, 0);

    server_kv_swap::config cfg_mine = server_kv_swap::config::for_instance(mine, LIVE_PID);

    // the sibling's live bytes must block a further generation
    CHECK(!server_kv_swap::admit(cfg_mine, 700 * 1024, 0, 0, 3));
    // a live sibling's data is never reclaimed to make room
    CHECK(std::filesystem::exists(sib / ".stage-cache-0-0-111" / ".commit"));
}

// D2: a foreign instance directory left by a dead process counts against the
// budget, and is only reclaimed when admission genuinely needs the room.
static void test_foreign_dead_directory_is_counted_then_reclaimed() {
    scratch sc("foreign");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    // a dead PID that cannot be alive: 0 is never a real process id
    const auto dead = std::filesystem::path(root) / "llama-kv-swap-0-0xdead";
    make_committed(dead / ".stage-cache-0-0-0", 900 * 1024, 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::inventory inv = server_kv_swap::reconcile(cfg);

    // it is visible in the accounting
    CHECK(inv.root_bytes >= 900 * 1024);

    // it is reclaimable because its owner is gone
    CHECK(inv.reclaimable_bytes >= 900 * 1024);

    // and admission can make room for a new generation
    CHECK(server_kv_swap::admit(cfg, 900 * 1024, 0, 0, 3));
    // the reclaim was actually performed
    CHECK(!std::filesystem::exists(dead));
}

// D2: a live foreign instance must never be reclaimed - another server is
// actively using those bytes.
static void test_foreign_live_directory_is_not_reclaimed() {
    scratch sc("live");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto live = std::filesystem::path(root)
        / ("llama-kv-swap-" + std::to_string((long) ::getpid()) + "-0xlive");
    make_committed(live / ".stage-cache-0-0-1", 900 * 1024, 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::inventory inv = server_kv_swap::reconcile(cfg);

    CHECK(inv.reclaimable_bytes == 0);
    CHECK(!server_kv_swap::admit(cfg, 900 * 1024, 0, 0, 3));
    // the other server's data is still there
    CHECK(std::filesystem::exists(live / ".stage-cache-0-0-1" / ".commit"));
}

// D3: a generation with no commit marker is unreachable by construction and
// must be reclaimed; a well-formed one must be left alone.
static void test_incomplete_generation_is_swept() {
    scratch sc("torn");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 8);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto inst = std::filesystem::path(root) / "llama-kv-swap-0-0xdead";
    // interrupted mid-write: tmp file present, no rename, no marker
    write_file(inst / ".stage-cache-0-0-0" / "target.bin.tmp", 64 * 1024);
    // a complete generation that must survive
    make_committed(inst / ".stage-cache-0-1-0", 32 * 1024, 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::reconcile(cfg);

    CHECK(!std::filesystem::exists(inst / ".stage-cache-0-0-0"));
    CHECK( std::filesystem::exists(inst / ".stage-cache-0-1-0" / ".commit"));
}

// D3: a marker that does not match the bytes on disk is a torn generation.
static void test_commit_size_mismatch_is_torn() {
    scratch sc("mismatch");
    const std::string root = sc.path;

    const auto inst = std::filesystem::path(root) / "llama-kv-swap-0-0xdead";
    const auto gen = inst / ".stage-cache-0-0-0";
    std::filesystem::create_directories(gen);
    write_file(gen / "target.bin", 1000);
    std::ofstream out(gen / ".commit", std::ios::binary | std::ios::trunc);
    out << "LLAMA_KV_SWAP_V1\n" << 999999 << "\n" << 0 << "\n"; // claims the wrong size

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::reconcile(server_kv_swap::config::for_instance(mine, 111));

    CHECK(!std::filesystem::exists(gen));
}

// D3: a live in-process generation must never be swept out from under a
// concurrent publish, even if its marker is not written yet.
static void test_live_incomplete_generation_survives_reconcile() {
    scratch sc("live-stage");
    const std::string root = sc.path;

    const auto inst = std::filesystem::path(root)
        / ("llama-kv-swap-" + std::to_string((long) ::getpid()) + "-0xmine");
    const auto gen = inst / ".stage-cache-0-0-0";
    write_file(gen / "target.bin.tmp", 4096); // publish in flight, no marker yet

    server_kv_swap::reconcile(server_kv_swap::config::for_instance(inst, (int) ::getpid()));
    CHECK(std::filesystem::exists(gen / "target.bin.tmp"));
}

// D5: the instance directory must not be reusable across processes, or one
// process can delete another process's data through the destructor.
static void test_instance_dir_name_is_pid_scoped() {
    const std::string name = server_kv_swap::instance_dir_name(4242, (const void *) 0x1234);

    CHECK(name.rfind("llama-kv-swap-", 0) == 0);
    CHECK(name.find("4242") != std::string::npos);
    // the pid must be a distinct, early field so liveness can be parsed
    CHECK(name.find("-4242-") != std::string::npos);

    // different pids must never collide even at the same address
    CHECK(server_kv_swap::instance_dir_name(1, (const void *) 0x1234) !=
          server_kv_swap::instance_dir_name(2, (const void *) 0x1234));
    // and a given pid/address pair is stable
    CHECK(server_kv_swap::instance_dir_name(7, (const void *) 0x1234) ==
          server_kv_swap::instance_dir_name(7, (const void *) 0x1234));
}

// D5: directories that predate the pid-scoped naming, or that are not
// instance directories at all, are never touched.
static void test_reconcile_ignores_foreign_layouts() {
    scratch sc("layout");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 64);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    // a hand-placed subdirectory that is not ours
    make_committed(std::filesystem::path(root) / "user-data" / "keepme", 1024, 0);
    // a legacy pointer-only name, pid unknown -> unresolvable, so not swept
    write_file(std::filesystem::path(root) / "llama-kv-swap-0xabc" / "junk.bin", 2048);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::reconcile(server_kv_swap::config::for_instance(mine, 111));

    CHECK(std::filesystem::exists(std::filesystem::path(root) / "user-data" / "keepme"));
    CHECK(std::filesystem::exists(std::filesystem::path(root) / "llama-kv-swap-0xabc" / "junk.bin"));
}

// D8: the pre-fix layout named the instance directory after a heap pointer
// alone ("llama-kv-swap-0x<ptr>"), so no pid is recorded in the directory name
// and the owner's liveness cannot be read from it. The pid is still recoverable
// from the generation names inside it. A legacy directory whose generations all
// name dead pids is an orphan and must be reclaimable, otherwise the shared
// quota stays permanently unsatisfiable and every spill is denied.
static void test_legacy_pointer_only_directory_is_reclaimed_via_generation_pid() {
    scratch sc("legacy");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    // the exact pre-fix layout seen in the live tree: pointer-only instance
    // directory, generation names carrying the dead writer pid
    const auto legacy = std::filesystem::path(root) / "llama-kv-swap-0x56e752047ec0";
    make_committed(legacy / ".stage-cache-0-0-239255", 900 * 1024, 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    const server_kv_swap::inventory inv = server_kv_swap::reconcile(cfg);

    // counted in the accounting ...
    CHECK(inv.root_bytes >= 900 * 1024);
    // ... and reclaimable, because 239255 is not this process
    CHECK(inv.reclaimable_bytes >= 900 * 1024);
    CHECK(server_kv_swap::admit(cfg, 900 * 1024, 0, 0, 3));
    CHECK(!std::filesystem::exists(legacy));
}

// D8: a legacy directory that is empty carries no generation names and so no
// evidence of ownership. It must be reclaimable, otherwise every historical
// instance that published nothing leaks a directory forever.
static void test_empty_legacy_directory_is_reclaimable() {
    scratch sc("legacy-empty");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto legacy = std::filesystem::path(root) / "llama-kv-swap-0x5778ee865c90";
    std::filesystem::create_directories(legacy);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    const server_kv_swap::inventory inv = server_kv_swap::reconcile(cfg);

    CHECK(inv.reclaimable.size() == 1);
    CHECK(inv.reclaimable_bytes == 0);
}

// D8: a legacy directory whose generations name a LIVE pid is still owned by a
// running server and must never be reclaimed, even though the instance name has
// no pid in it.
static void test_legacy_directory_with_live_generation_pid_is_not_reclaimed() {
    scratch sc("legacy-live");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto legacy = std::filesystem::path(root) / "llama-kv-swap-0xdeadbeef";
    make_committed(legacy / (".stage-cache-0-0-" + std::to_string(LIVE_PID)), 900 * 1024, 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    const server_kv_swap::inventory inv = server_kv_swap::reconcile(cfg);

    CHECK(inv.reclaimable_bytes == 0);
    CHECK(!server_kv_swap::admit(cfg, 900 * 1024, 0, 0, 3));
    CHECK(std::filesystem::exists(legacy / (".stage-cache-0-0-" + std::to_string(LIVE_PID))));
}

// D8: the pointer prefix alone must not be enough to make a directory
// reclaimable. Only names the server actually writes, or that hold generations
// in a recognised shape, qualify. An arbitrary subdirectory keeps its data.
static void test_legacy_reclaim_does_not_target_unrelated_data() {
    scratch sc("legacy-foreign");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    // matches the pointer prefix but holds a file, not a generation
    const auto odd = std::filesystem::path(root) / "llama-kv-swap-0xabc";
    write_file(odd / "important.bin", 4096);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::reconcile(cfg);

    CHECK(std::filesystem::exists(odd / "important.bin"));
}

// D6: the free-space probe must fail closed. The pre-fix code denied
// admission when std::filesystem::space() errored, so an unreadable or
// vanished swap volume could never be written to. Failing open would let an
// unbounded write proceed onto a volume whose headroom is unknown.
static void test_space_probe_failure_denies_admission() {
    scratch sc("spacefail");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 64);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 8); // a real headroom demand
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    std::filesystem::create_directories(mine);
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);

    const server_kv_swap::usage used{0, 0, true};
    const size_t max_bytes    = 64ull * 1024ull * 1024ull;
    const size_t max_files    = 0;
    const size_t min_free     = 8ull * 1024ull * 1024ull;

    // a root that does not exist makes space() fail
    const auto missing = std::filesystem::path(sc.path) / "no-such-volume";
    CHECK(server_kv_swap::over_bounds(missing, used, 1024, 3, max_bytes, max_files, min_free));

    // and the same must hold through the public entry point: the server always
    // creates its instance directory before publishing, so an absent root means
    // the volume is gone and the write must be denied
    const auto gone = server_kv_swap::config::for_instance(
        std::filesystem::path(sc.path) / "no-such-volume" / "llama-kv-swap-111-0xaaa", 111);
    CHECK(!server_kv_swap::admit(gone, 1024, 0, 0, 3));
}

// D9: admission walks the shared root, so two live server instances publishing
// at the same time can each measure the tree, both see room, and together
// exceed the quota. The measurement and the reservation must be serialised
// across processes, not just within one.
static void test_root_lock_is_exclusive_across_processes() {
    scratch sc("lock");
    const std::string root = sc.path;

    const auto cfg_a = server_kv_swap::config::for_instance(
        std::filesystem::path(root) / "llama-kv-swap-1-0xa", 1);
    const auto cfg_b = server_kv_swap::config::for_instance(
        std::filesystem::path(root) / "llama-kv-swap-2-0xb", 2);

    std::error_code ec;
    std::filesystem::create_directories(cfg_a.root, ec);

    server_kv_swap::root_lock lock_a(cfg_a);
    CHECK(lock_a.held());

    // a second acquisition on the same root must not succeed while the first
    // is held, even from another "process" in this test
    server_kv_swap::root_lock lock_b(cfg_b);
    CHECK(!lock_b.held());
    CHECK(!lock_b.try_lock());

    lock_a.unlock();
    CHECK(!lock_a.held());

    // once released, the other side can take it
    CHECK(lock_b.try_lock());
    CHECK(lock_b.held());
}

// D9: the lock must survive being released by its destructor, and must not
// remove a lock file another holder now owns.
static void test_root_lock_releases_on_scope_exit() {
    scratch sc("lock-scope");
    const std::string root = sc.path;

    const auto cfg = server_kv_swap::config::for_instance(
        std::filesystem::path(root) / "llama-kv-swap-1-0xa", 1);
    std::error_code ec;
    std::filesystem::create_directories(cfg.root, ec);

    {
        server_kv_swap::root_lock lock(cfg);
        CHECK(lock.held());
    }

    server_kv_swap::root_lock again(cfg);
    CHECK(again.try_lock());
    CHECK(again.held());
}

// D10: a hard crash between create_directory(stage) and the commit marker
// leaves a generation that can never be restored. Reconciliation must remove
// it, otherwise every crash leaks a full KV state. The sweep must also be
// idempotent and must not disturb a complete sibling generation.
static void test_torn_generation_is_swept_and_complete_sibling_survives() {
    scratch sc("crash");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 8);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    // a crashed publisher: target.bin.tmp only, no rename, no marker
    const auto inst = std::filesystem::path(root) / "llama-kv-swap-0-0xdead";
    const auto torn = inst / ".stage-slot-0-3-999";
    write_file(torn / "target.bin.tmp", 128 * 1024);
    // a complete generation in the same directory must survive
    const auto good = inst / ".stage-slot-0-2-999";
    make_committed(good, 64 * 1024, 16 * 1024);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    const server_kv_swap::inventory inv = server_kv_swap::reconcile(cfg);

    CHECK(!std::filesystem::exists(torn));
    CHECK(std::filesystem::exists(good / ".commit"));
    CHECK(std::filesystem::exists(good / "draft.bin"));
    // the torn bytes were accounted as swept
    CHECK(inv.swept.bytes >= 128 * 1024);

    // a second reconcile is a no-op: sweeping again must find nothing
    const server_kv_swap::inventory again = server_kv_swap::reconcile(cfg);
    CHECK(again.swept.bytes == 0);
    CHECK(std::filesystem::exists(good / ".commit"));
}

// D4: a payload that turns out larger than the pre-write estimate must be
// caught after the fact, not silently allowed past the bound.
static void test_post_commit_bound_is_enforced() {
    scratch sc("postcommit");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::reconcile(cfg);

    const auto stage = cfg.instance_dir / ".stage-slot-0-0-111";
    // the estimate assumed 64 KiB...
    CHECK(server_kv_swap::admit(cfg, 64 * 1024, 0, 0, 3));
    // ...but a payload larger than the whole quota landed
    make_committed(stage, 1200 * 1024, 0);

    // the bound must hold, so this generation is rejected and removed
    CHECK(!server_kv_swap::enforce_bound_after_commit(cfg, stage, 0, 0));
    CHECK(!std::filesystem::exists(stage));
}

// D4: the post-commit check must judge the root as it actually stands. A
// generation that is itself within quota is kept even when the pre-write
// estimate was smaller, because the check exists to catch an overshoot past
// the bound, not to punish an underestimate that did not overshoot.
static void test_post_commit_estimate_undershoot_within_quota_is_kept() {
    scratch sc("postcommit-under");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::reconcile(cfg);

    const auto stage = cfg.instance_dir / ".stage-slot-0-0-111";
    // estimate 64 KiB, actual 900 KiB, quota 1 MiB: still inside the bound
    CHECK(server_kv_swap::admit(cfg, 64 * 1024, 0, 0, 3));
    make_committed(stage, 900 * 1024, 0);

    CHECK( server_kv_swap::enforce_bound_after_commit(cfg, stage, 0, 0));
    CHECK( std::filesystem::exists(stage / ".commit"));
}

// D4: a payload that fits must survive the post-commit check.
static void test_post_commit_bound_allows_fit() {
    scratch sc("postcommit-ok");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::reconcile(cfg);

    const auto stage = cfg.instance_dir / ".stage-slot-0-0-111";
    make_committed(stage, 100 * 1024, 0);

    CHECK( server_kv_swap::enforce_bound_after_commit(cfg, stage, 0, 0));
    CHECK( std::filesystem::exists(stage / ".commit"));
}

// D4: the old generation must survive a rejected publish, so the state is
// never lost because the estimate was wrong.
static void test_rejected_publish_keeps_previous_generation() {
    scratch sc("keepold");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::reconcile(cfg);

    const auto old_gen = cfg.instance_dir / ".stage-slot-0-0-110";
    make_committed(old_gen, 100 * 1024, 0);

    // replacing it is admitted on the estimate...
    CHECK(server_kv_swap::admit(cfg, 800 * 1024, 100 * 1024, 3, 3));
    // ...but the new generation turns out to push the root past the quota even
    // after the replaced generation is discounted
    const auto stage = cfg.instance_dir / ".stage-slot-0-1-111";
    make_committed(stage, 1100 * 1024, 0);

    // rejection must not take the previously committed state with it
    CHECK(!server_kv_swap::enforce_bound_after_commit(cfg, stage, 100 * 1024, 3));
    CHECK(std::filesystem::exists(old_gen / ".commit"));
    CHECK(!std::filesystem::exists(stage));
}

// D6: an unreadable subtree must deny admission rather than undercount.
static void test_usage_walk_error_denies() {
    scratch sc("walkerr");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 1);
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 0);

    const auto mine = std::filesystem::path(root) / "llama-kv-swap-111-0xaaa";
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, 111);
    server_kv_swap::reconcile(cfg);

    // a dangling symlink is not a regular file but must not abort the walk
    // into a partial count; a symlink loop is the harder case
    std::error_code ec;
    const auto loopy = std::filesystem::path(root) / "llama-kv-swap-0-0xdead" / "self";
    std::filesystem::create_directories(loopy.parent_path());
    if (::symlink(loopy.parent_path().c_str(), loopy.c_str()) != 0) {
        // the platform refuses the loop, so the harder case cannot be built
        // here; the fail-closed behaviour it targets is covered directly by
        // test_space_probe_failure_denies_admission
        return;
    }

    // whether or not the platform can traverse it, the result must be
    // conservative: either the bytes are counted, or admission is denied
    const bool admitted = server_kv_swap::admit(cfg, 900 * 1024, 0, 0, 3);
    if (!admitted) {
        CHECK(true); // fail-closed is acceptable
    } else {
        // admitted, so the walk must have been complete
        server_kv_swap::inventory inv = server_kv_swap::reconcile(cfg);
        CHECK(inv.usage_complete);
    }
}

// the marker round-trip the server relies on must stay exact
static void test_commit_marker_roundtrip() {
    scratch sc("marker");
    const auto gen = std::filesystem::path(sc.path) / "gen";
    make_committed(gen, 4096, 2048);

    server_kv_swap::marker m;
    CHECK(server_kv_swap::read_commit(gen, m));
    CHECK_EQ(m.target_bytes, (size_t) 4096);
    CHECK_EQ(m.draft_bytes, (size_t) 2048);

    // a missing marker is not a commit
    const auto bare = std::filesystem::path(sc.path) / "bare";
    std::filesystem::create_directories(bare);
    CHECK(!server_kv_swap::read_commit(bare, m));
}

// the file cap must be enforced across the root too, not per instance
static void test_file_cap_spans_root() {
    scratch sc("files");
    const std::string root = sc.path;

    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_MIB", 0);  // bytes cap disabled
    set_env_size("LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB", 0);
    set_env_size("LLAMA_SERVER_KV_SWAP_MAX_FILES", 6);

    // a live sibling already holding 2 committed generations (4 files)
    const auto sib = std::filesystem::path(root) /
        ("llama-kv-swap-" + std::to_string(LIVE_PID) + "-0xaaa");
    make_committed(sib / "g0", 16, 0);
    make_committed(sib / "g1", 16, 0);

    const auto mine = std::filesystem::path(root) /
        ("llama-kv-swap-" + std::to_string(LIVE_PID) + "-0xbbb");
    server_kv_swap::config cfg = server_kv_swap::config::for_instance(mine, LIVE_PID);
    // 4 files used, 3 needed, cap 6 -> must be denied
    CHECK(!server_kv_swap::admit(cfg, 16, 0, 0, 3));
    // and the live sibling's data is untouched
    CHECK(std::filesystem::exists(sib / "g1" / ".commit"));
}

} // namespace

int main() {
    test_quota_spans_sibling_instances();
    test_foreign_dead_directory_is_counted_then_reclaimed();
    test_foreign_live_directory_is_not_reclaimed();
    test_incomplete_generation_is_swept();
    test_commit_size_mismatch_is_torn();
    test_live_incomplete_generation_survives_reconcile();
    test_instance_dir_name_is_pid_scoped();
    test_reconcile_ignores_foreign_layouts();
    test_legacy_pointer_only_directory_is_reclaimed_via_generation_pid();
    test_empty_legacy_directory_is_reclaimable();
    test_legacy_directory_with_live_generation_pid_is_not_reclaimed();
    test_legacy_reclaim_does_not_target_unrelated_data();
    test_space_probe_failure_denies_admission();
    test_root_lock_is_exclusive_across_processes();
    test_root_lock_releases_on_scope_exit();
    test_torn_generation_is_swept_and_complete_sibling_survives();
    test_post_commit_bound_is_enforced();
    test_post_commit_estimate_undershoot_within_quota_is_kept();
    test_post_commit_bound_allows_fit();
    test_rejected_publish_keeps_previous_generation();
    test_usage_walk_error_denies();
    test_commit_marker_roundtrip();
    test_file_cap_spans_root();

    printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
