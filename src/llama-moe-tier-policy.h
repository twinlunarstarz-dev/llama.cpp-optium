// Ported from Basalt's core/tier_policy.hpp (see /opt/basalt, same constant values and
// semantics). Basalt's own CUDA engine is Blackwell-only (sm_120), so its kernels cannot
// run here; its admission and placement *rules*, however, are deliberately free of CUDA
// ("so a CPU test can drive them with synthetic id streams"). This fork adopts those rules
// to decide which experts earn VRAM homes, which fall back to the RAM tier and which stay
// in the file, so placement is driven by routing rank and reuse rather than a global LRU.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace llama_moe_tier {

constexpr int kPromoteWindow = 64;           // tokens within which a repeat miss earns a VRAM slot
constexpr uint64_t kPieceBytes = 1ull << 20; // largest single promotion copy
constexpr int kPromoteUses   = 12;           // CPU uses that earn a VRAM slot when the RAM tier caches the file
constexpr int kUseHalfLife   = 128;          // tokens after which those counts halve
constexpr uint16_t kLookCap  = 15;           // lookups a use-count tier remembers before saturating

// One more CPU use of an expert: true, and the count starts over, when it reaches `threshold`.
inline bool count_use(uint16_t & uses, int threshold = kPromoteUses) {
    if (++uses < threshold) {
        return false;
    }
    uses = 0;
    return true;
}

// Whether the tokens from `token` to `token + n` cross a multiple of `period` (0: never).
inline bool crosses(uint64_t token, uint64_t n, uint64_t period) {
    return period > 0 && token / period != (token + n) / period;
}

inline bool use_half_life(uint64_t token, uint64_t n) { return crosses(token, n, kUseHalfLife); }

inline void count_look(uint16_t & looks) {
    if (looks < kLookCap) {
        ++looks;
    }
}

// Fewer lookups first; the one used longest ago among equals.
inline bool yields_before(uint16_t looks_a, uint64_t stamp_a, uint16_t looks_b, uint64_t stamp_b) {
    return looks_a != looks_b ? looks_a < looks_b : stamp_a < stamp_b;
}

// `last` is the token + 1 of the previous miss of an expert (0 = never); true when it falls
// within `window` tokens. This is the promotion trigger: a second miss soon after the first
// means the expert is being recycled by the routing, so a slot is worth spending on it.
inline bool repeat_miss(uint64_t last, uint64_t token, int window) {
    return last != 0 && token + 1 - last <= (uint64_t) window;
}

// How many experts of every layer each place holds.
struct Homes {
    int main = 0;  // on the main card and nowhere else; 0 when its slots only copy the RAM tier
    int peer = 0;  // on the second card and nowhere else
    int ram  = 0;  // pinned in RAM
    int file = 0;  // read from the pack
};

// `main_slots`: what the main card can hold through a prompt at the full context; `peer_slots`:
// room on the second card (0 without one); `ram_slots`: room in RAM above the floor right now;
// `ram_cap`: places per layer the machine has for the tier whatever else is running (its total
// RAM less the engine's own buffers and the floor), or -1 when the tier's size was given and
// `ram_slots` is all there is. The main card keeps its homes unless `ram_cap` takes what the
// second card does not and as many places again as the main card would have homed: copies-only
// decodes slower than homes, so it is picked only where RAM leaves no doubt.
inline Homes plan_homes(int n_expert, int main_slots, int peer_slots, int ram_slots, int ram_cap) {
    Homes h;
    h.peer = peer_slots < n_expert ? peer_slots : n_expert;
    const int left  = n_expert - h.peer;
    const int homed = main_slots < left ? main_slots : left;
    const bool roomy = ram_cap < 0 ? ram_slots >= left : ram_cap >= left + homed;
    if (!roomy) {
        h.main = homed;
    }
    const int rest = left - h.main;
    h.ram  = ram_slots < rest ? ram_slots : rest;
    h.file = rest - h.ram;
    return h;
}

enum class Place { Main, Peer, Ram, File };

// Where the expert at routing rank `r` lives: the main card's homes first, then the head of the
// RAM tier when the main card's slots copy it, the second card's homes, the rest of the RAM
// tier, the file.
inline Place place_of_rank(const Homes & h, int copies, int r) {
    if (r < h.main) {
        return Place::Main;
    }
    const int lead = h.main > 0 ? 0 : (copies < h.ram ? copies : h.ram);
    if (r - h.main < lead) {
        return Place::Ram;
    }
    if (r - h.main - lead < h.peer) {
        return Place::Peer;
    }
    return r < h.main + h.peer + h.ram ? Place::Ram : Place::File;
}

} // namespace llama_moe_tier
