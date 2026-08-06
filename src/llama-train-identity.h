#pragma once

// P2: Stable tensor identity and tied-weight aliasing model
//
// Provides a versioned stable tensor identity independent of graph indices,
// pointer addresses, allocation order, or backend placement.
//
// Identity = SHA-256(canonical_encoding(name) || canonical_encoding(type) ||
//                       canonical_encoding(shape))
//
// Canonical byte encoding:
//   - Strings: little-endian uint32 length + UTF-8 bytes (no null terminator)
//   - ggml_type: little-endian uint32 enum value
//   - Shape: little-end4_t rank, then rank little-endian int64_t dimensions
//
// Collision resistance: SHA-256 provides 256-bit preimage resistance.
// Two tensors with different (name, type, shape) will collide with probability
// ~2^-256. The identity is NOT a security boundary but an integrity contract
// for checkpoint persistence and optimizer state keying.

#include <ggml.h>
#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace llama_train {

// ---------------------------------------------------------------------------
// Stable tensor identity (256-bit SHA-256 digest)
// ---------------------------------------------------------------------------

/// 32-byte SHA-256 digest representing a canonical tensor identity.
using tensor_id = std::array<uint8_t, 32>;

/// Human-readable hex encoding of a tensor_id (64 hex chars).
std::string tensor_id_to_hex(const tensor_id & id);

/// Parse hex string into tensor_id. Returns nullopt on invalid input.
std::optional<tensor_id> tensor_id_from_hex(std::string_view hex);

/// Default-constructed ID is all zeros (invalid sentinel).
/// Only rejects the all-zeros sentinel; valid SHA-256 hashes may contain zero bytes.
inline bool is_valid(tensor_id id) {
    return std::any_of(id.begin(), id.end(), [](uint8_t b) { return b != 0; });
}

// ---------------------------------------------------------------------------
// Canonical encoding functions
// ---------------------------------------------------------------------------

/// Encode a string as little-endian uint32 length + UTF-8 bytes.
std::vector<uint8_t> canonical_encode_string(std::string_view s);

/// Encode ggml_type as little-endian uint32.
std::vector<uint8_t> canonical_encode_type(ggml_type type);

/// Encode shape as little-endian uint32 rank + rank int64_t dimensions.
std::vector<uint8_t> canonical_encode_shape(
    const std::vector<int64_t> & dims);

/// Compute stable tensor identity from name, type, and shape.
/// Returns all-zero ID on invalid input (empty name, invalid type, empty shape).
tensor_id compute_tensor_id(
    std::string_view name,
    ggml_type type,
    const std::vector<int64_t> & dims);

// ---------------------------------------------------------------------------
// SHA-256 via OpenSSL EVP API
// ---------------------------------------------------------------------------

/// Incremental SHA-256 state.
/// Throws std::runtime_error if any OpenSSL EVP call fails.
class sha256_context {
public:
    /// Constructs and initializes the context. Throws on failure.
    sha256_context();
    ~sha256_context();

    /// Add data to the digest. Throws if context was previously poisoned.
    void update(const uint8_t * data, size_t len);

    /// Finalize and return the 32-byte digest. Throws on failure.
    std::array<uint8_t, 32> finalize();

private:
    void * m_ctx = nullptr; // EVP_MD_CTX*
};

/// One-shot SHA-256 hash.
std::array<uint8_t, 32> sha256_hash(const uint8_t * data, size_t len);

// ---------------------------------------------------------------------------
// Tensor descriptor (externally supplied, strictly validated)
// ---------------------------------------------------------------------------

/// Canonical tensor metadata used for identity computation and validation.
struct tensor_descriptor {
    std::string name;
    ggml_type type;
    std::vector<int64_t> dims;  // rank-ordered dimensions

    /// Compute the stable identity for this descriptor.
    tensor_id id() const;

    /// Byte size of one row (dims[0] elements of given type).
    size_t row_size() const;

    /// Total byte count.
    size_t total_bytes() const;

    /// Number of elements.
    int64_t nelements() const;
};

// ---------------------------------------------------------------------------
// Tied-weight alias model
// ---------------------------------------------------------------------------

/// Represents a group of tensors that share the same canonical storage
/// (tied embeddings, tied output projections, etc.).
struct tied_group {
    tied_group() = default;

    /// Canonical tensor_descriptor for the unique base parameter.
    tensor_descriptor base;

    /// All names that alias this base (includes base.name).
    std::vector<std::string> aliases;

    /// Compute the stable ID for this tied group (same as base.id()).
    tensor_id id() const;
};

/// Validate and build tied groups from a list of descriptors.
///
/// Rules:
/// - Each descriptor must have a valid non-empty name, valid type, non-empty dims.
/// - Tensors with the same logical tie key (same name pattern or explicit
///   grouping) must have identical (type, dims).
/// - Returns an error string on validation failure; empty string on success.
///
/// The `tie_key_fn` maps a tensor name to a tie group identifier.
/// Tensors with the same tie_key are considered tied and must match in type/dims.
struct tied_validation_result {
    bool ok;
    std::string error;
    std::vector<tied_group> groups;
};

tied_validation_result validate_tied_groups(
    const std::vector<tensor_descriptor> & descriptors,
    std::function<std::string(std::string_view)> tie_key_fn);

/// Default tie-key function: tensors whose name ends with ".weight" and whose
/// stem (name without the final component after the last '.') matches a known
/// tied pattern are grouped. For a general-purpose default, return the full
/// name (no tying).
std::function<std::string(std::string_view)> default_tie_key_fn();

/// Build tied groups assuming all descriptors with identical (type, dims)
/// and names that match token_embd.weight / output.weight patterns are tied.
tied_validation_result validate_standard_tied(
    const std::vector<tensor_descriptor> & descriptors);

// ---------------------------------------------------------------------------
// Unique parameter enumeration (for audit: prove every base weight changed)
// ---------------------------------------------------------------------------

/// Return the set of unique tensor IDs from a list of tied groups.
std::vector<tensor_id> get_unique_ids(const std::vector<tied_group> & groups);

} // namespace llama_train
