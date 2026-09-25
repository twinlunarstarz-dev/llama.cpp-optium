#pragma once

// Physical-row bookkeeping for the recurrent (SSM/linear-attention) cache.
//
// llama_memory_recurrent sizes its R/S buffers from measured free device memory
// rather than from the logical sequence count, so the number of logical
// sequences a server exposes (--parallel, server slots) can exceed the number of
// physical rows that actually fit.  Callers need three things from that split:
// how many states are resident right now, how many bytes that costs, and a
// refusal when the pool is full.  A refused admission is a scheduling decision;
// hitting the limit later, inside slot search, is a crash.
//
// This header is pure bookkeeping: it owns no buffers and no cell metadata.  The
// authoritative state stays in llama_memory_recurrent's cell table, and
// reconcile() rebuilds the owner map from that table so the counters cannot
// drift away from the state they describe.  Row identity is decided by the
// caller, which keeps the emitted graph and the serialized state bytes exactly
// what they were.

#include <cstdint>
#include <vector>

namespace llama_recurrent_rows {

class row_pool {
public:
    row_pool() : row_pool(/* capacity = */ 0, /* plane_bytes = */ 0) {}

    row_pool(uint32_t capacity, uint64_t plane_bytes) :
        capacity_(capacity), plane_bytes_(plane_bytes), owner_(capacity, k_free) {}

    uint32_t capacity()      const { return capacity_; }
    uint32_t n_resident()    const { return n_resident_; }
    uint32_t peak()          const { return peak_; }
    uint32_t free_count()    const { return capacity_ - n_resident_; }
    uint64_t bytes_resident() const { return (uint64_t) n_resident_ * plane_bytes_; }
    uint64_t plane_bytes()   const { return plane_bytes_; }

    // True when a sequence that owns no row could still take one.  This is the
    // admission check: a sequence that already holds a row needs no admission.
    bool can_admit(uint32_t n_resident_extra) const {
        return (uint64_t) n_resident_ + n_resident_extra <= capacity_;
    }

    // Take `row` for `seq_id`.  Refused when the row is already owned, when
    // `seq_id` already owns a different row, or when the row is out of range.
    bool acquire(uint32_t row, int32_t seq_id) {
        if (row >= capacity_ || owner_[row] != k_free) {
            return false;
        }
        if (seq_id >= 0 && row_of(seq_id) >= 0) {
            return false;
        }
        owner_[row] = seq_id;
        n_resident_++;
        peak_ = n_resident_ > peak_ ? n_resident_ : peak_;
        return true;
    }

    // Free a row regardless of owner.  False when the row was already free or
    // out of range.
    bool release_row(uint32_t row) {
        if (row >= capacity_ || owner_[row] == k_free) {
            return false;
        }
        owner_[row] = k_free;
        n_resident_--;
        return true;
    }

    // Free the row owned by `seq_id`; returns the row, or -1 when it owns none.
    int32_t release_seq(int32_t seq_id) {
        const int32_t row = row_of(seq_id);
        if (row < 0) {
            return -1;
        }
        owner_[row] = k_free;
        n_resident_--;
        return row;
    }

    int32_t owner_of(uint32_t row) const {
        return row < capacity_ ? owner_[row] : k_free;
    }

    int32_t row_of(int32_t seq_id) const {
        if (seq_id < 0) {
            return -1;
        }
        for (uint32_t i = 0; i < capacity_; ++i) {
            if (owner_[i] == seq_id) {
                return (int32_t) i;
            }
        }
        return -1;
    }

    // Rebuild from the authoritative per-cell owner list (k_free for empty
    // cells).  False when the table does not match the pool, in which case the
    // pool is left untouched.
    bool reconcile(const std::vector<int32_t> & table) {
        if (table.size() != capacity_) {
            return false;
        }
        owner_ = table;
        n_resident_ = 0;
        for (int32_t o : owner_) {
            n_resident_ += o != k_free ? 1 : 0;
        }
        peak_ = n_resident_ > peak_ ? n_resident_ : peak_;
        return true;
    }

    void reset() {
        owner_.assign(capacity_, k_free);
        n_resident_ = 0;
    }

    static constexpr int32_t k_free = -1;

private:
    uint32_t capacity_;
    uint64_t plane_bytes_;
    uint32_t n_resident_ = 0;
    uint32_t peak_       = 0;

    std::vector<int32_t> owner_;
};

} // namespace llama_recurrent_rows
