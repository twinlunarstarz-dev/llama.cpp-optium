// Verifies the ported Basalt tier policy against its documented invariants. No GPU, no model:
// this is the deterministic check that the placement rules behave as Basalt specifies before any
// of them drive a real expert tier.
#include "../src/llama-moe-tier-policy.h"
#include <cstdio>
#include <cstdlib>

using namespace llama_moe_tier;

static int failures = 0;

#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

int main() {
    // plan_homes: with RAM roomy enough for every place plus the card's homes, the card keeps
    // none and the RAM tier takes what the peer does not.
    {
        Homes h = plan_homes(512, 15, 0, 300, 600);
        CHECK(h.peer == 0);
        CHECK(h.main == 0);
        CHECK(h.ram  == 300);
        CHECK(h.file == 212);
    }
    // plan_homes: RAM too small to leave no doubt -> the card keeps its homes.
    {
        Homes h = plan_homes(512, 15, 0, 300, 100);
        CHECK(h.main == 15);
        CHECK(h.ram  == 300);
        CHECK(h.file == 197);
    }
    // plan_homes: RAM larger than what is left after the card's homes -> file holds nothing.
    {
        Homes h = plan_homes(512, 15, 0, 1000, 100);
        CHECK(h.main == 15);
        CHECK(h.ram  == 497);
        CHECK(h.file == 0);
    }
    // plan_homes: the peer takes its places first and never exceeds n_expert.
    {
        Homes h = plan_homes(512, 15, 8, 300, 100);
        CHECK(h.peer == 8);
        CHECK(h.main == 15);
        CHECK(h.ram  == 300);
        CHECK(h.file == 189);
        Homes g = plan_homes(4, 15, 8, 300, 100);
        CHECK(g.peer == 4);
    }
    // plan_homes: ram_cap -1 means the tier was sized explicitly; ram_slots is all there is.
    {
        Homes h = plan_homes(512, 15, 0, 700, -1);
        CHECK(h.main == 0);
        CHECK(h.ram  == 512);
    }
    // place_of_rank: card homes first, then RAM, then peer, then file.
    {
        Homes h; h.main = 15; h.peer = 8; h.ram = 300; h.file = 189;
        CHECK(place_of_rank(h, 0, 0)   == Place::Main);
        CHECK(place_of_rank(h, 0, 14)  == Place::Main);
        CHECK(place_of_rank(h, 0, 15)  == Place::Peer);  // card homes, then the peer's
        CHECK(place_of_rank(h, 0, 22)  == Place::Peer);
        CHECK(place_of_rank(h, 0, 23)  == Place::Ram);   // then the RAM tier
        CHECK(place_of_rank(h, 0, 322) == Place::Ram);
        CHECK(place_of_rank(h, 0, 323) == Place::File);
    }
    // place_of_rank: with no card homes, the first `copies` RAM places lead the peer's homes.
    {
        Homes h; h.main = 0; h.peer = 8; h.ram = 300;
        CHECK(place_of_rank(h, 10, 0)  == Place::Ram);
        CHECK(place_of_rank(h, 10, 9)  == Place::Ram);
        CHECK(place_of_rank(h, 10, 10) == Place::Peer);
        CHECK(place_of_rank(h, 10, 17) == Place::Peer);  // peer's homes after the RAM lead
        CHECK(place_of_rank(h, 10, 18) == Place::Ram);
        CHECK(place_of_rank(h, 10, 307) == Place::Ram);
        CHECK(place_of_rank(h, 10, 308) == Place::File);
    }
    // count_use: a slot is earned on the threshold-th use and the count starts over.
    {
        uint16_t u = 0;
        for (int i = 0; i < kPromoteUses - 1; ++i) {
            CHECK(!count_use(u));
        }
        CHECK(count_use(u));
        CHECK(u == 0);
        CHECK(count_use(u, 1));  // threshold 1 admits every use
    }
    // repeat_miss: a second miss inside the window promotes, one outside it does not.
    {
        CHECK(!repeat_miss(0, 50, kPromoteWindow));       // never missed before
        CHECK( repeat_miss(10, 20, kPromoteWindow));      // 11 tokens later
        CHECK( repeat_miss(10, 73, kPromoteWindow));      // 64 tokens later: edge included
        CHECK(!repeat_miss(10, 74, kPromoteWindow));      // 65: outside
    }
    // yields_before: fewer lookups gives way first; among equals the colder stamp does.
    {
        CHECK( yields_before(5, 100, 7, 100));
        CHECK(!yields_before(7, 100, 5, 100));
        CHECK( yields_before(7, 50, 7, 100));
        CHECK(!yields_before(7, 100, 7, 50));
        CHECK(!yields_before(7, 100, 7, 100));
    }
    // count_look saturates at kLookCap.
    {
        uint16_t l = 0;
        for (int i = 0; i < 100; ++i) {
            count_look(l);
        }
        CHECK(l == kLookCap);
    }
    // crosses / use_half_life: a boundary is crossed exactly when the span straddles it.
    {
        CHECK(!crosses(10, 10, kUseHalfLife));
        CHECK( crosses(100, 40, kUseHalfLife));
        CHECK(!crosses(0, 0, kUseHalfLife));
        CHECK(!crosses(5, 10, 0));               // period 0 never crosses
        CHECK( use_half_life(127, 2));
        CHECK(!use_half_life(120, 2));
    }
    // kPieceBytes: promotion copies are bounded so a promotion cannot stall the hot path.
    CHECK(kPieceBytes == (1ull << 20));

    if (failures == 0) {
        std::printf("PASS: all Basalt tier-policy invariants hold\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", failures);
    return 1;
}
