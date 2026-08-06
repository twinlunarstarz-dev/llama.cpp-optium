#include "llama-train-identity.h"

#include <openssl/evp.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace llama_train {

// ===========================================================================
// SHA-256 via OpenSSL EVP API (verified against RFC 6234 test vectors)
//   SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
//   SHA-256("")    = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
// ===========================================================================

sha256_context::sha256_context() {
    m_ctx = static_cast<void*>(EVP_MD_CTX_new());
    if (!m_ctx) {
        // EVP_MD_CTX_new failed — out of memory or OpenSSL error
        throw std::runtime_error("sha256_context: EVP_MD_CTX_new failed");
    }
    if (EVP_DigestInit_ex(static_cast<EVP_MD_CTX*>(m_ctx), EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(static_cast<EVP_MD_CTX*>(m_ctx));
        m_ctx = nullptr;
        throw std::runtime_error("sha256_context: EVP_DigestInit_ex failed (SHA-256 unavailable)");
    }
}

sha256_context::~sha256_context() {
    if (m_ctx) EVP_MD_CTX_free(static_cast<EVP_MD_CTX*>(m_ctx));
}

void sha256_context::update(const uint8_t * data, size_t len) {
    if (!m_ctx) {
        throw std::runtime_error("sha256_context::update: context not initialized");
    }
    if (EVP_DigestUpdate(static_cast<EVP_MD_CTX*>(m_ctx), data, len) != 1) {
        // Mark context as poisoned so finalize also fails
        void * bad = m_ctx;
        m_ctx = nullptr;
        EVP_MD_CTX_free(static_cast<EVP_MD_CTX*>(bad));
        throw std::runtime_error("sha256_context::update: EVP_DigestUpdate failed");
    }
}

std::array<uint8_t, 32> sha256_context::finalize() {
    std::array<uint8_t, 32> digest{};
    if (!m_ctx) {
        throw std::runtime_error("sha256_context::finalize: context not initialized (previous failure)");
    }
    unsigned int len = 0;
    if (EVP_DigestFinal_ex(static_cast<EVP_MD_CTX*>(m_ctx), digest.data(), &len) != 1) {
        void * bad = m_ctx;
        m_ctx = nullptr;
        EVP_MD_CTX_free(static_cast<EVP_MD_CTX*>(bad));
        throw std::runtime_error("sha256_context::finalize: EVP_DigestFinal_ex failed");
    }
    // OpenSSL guarantees SHA-256 produces exactly 32 bytes, but check anyway
    if (len != 32) {
        void * bad = m_ctx;
        m_ctx = nullptr;
        EVP_MD_CTX_free(static_cast<EVP_MD_CTX*>(bad));
        throw std::runtime_error("sha256_context::finalize: unexpected digest length");
    }
    return digest;
}

std::array<uint8_t, 32> sha256_hash(const uint8_t * data, size_t len) {
    sha256_context ctx;
    ctx.update(data, len);
    return ctx.finalize();
}

// ===========================================================================
// Canonical encoding
// ===========================================================================

std::vector<uint8_t> canonical_encode_string(std::string_view s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    std::vector<uint8_t> out(4 + len);
    // Little-endian uint32 length
    out[0] = static_cast<uint8_t>(len & 0xFF);
    out[1] = static_cast<uint8_t>((len >> 8) & 0xFF);
    out[2] = static_cast<uint8_t>((len >> 16) & 0xFF);
    out[3] = static_cast<uint8_t>((len >> 24) & 0xFF);
    std::memcpy(out.data() + 4, s.data(), len);
    return out;
}

std::vector<uint8_t> canonical_encode_type(ggml_type type) {
    uint32_t val = static_cast<uint32_t>(type);
    std::vector<uint8_t> out(4);
    out[0] = static_cast<uint8_t>(val & 0xFF);
    out[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    out[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
    out[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    return out;
}

std::vector<uint8_t> canonical_encode_shape(
    const std::vector<int64_t> & dims) {
    uint32_t rank = static_cast<uint32_t>(dims.size());
    size_t total = 4 + rank * 8;
    std::vector<uint8_t> out(total);
    out[0] = static_cast<uint8_t>(rank & 0xFF);
    out[1] = static_cast<uint8_t>((rank >> 8) & 0xFF);
    out[2] = static_cast<uint8_t>((rank >> 16) & 0xFF);
    out[3] = static_cast<uint8_t>((rank >> 24) & 0xFF);
    for (size_t i = 0; i < dims.size(); i++) {
        int64_t d = dims[i];
        uint8_t * p = out.data() + 4 + i * 8;
        for (int j = 0; j < 8; j++) {
            p[j] = static_cast<uint8_t>(d & 0xFF);
            d >>= 8;
        }
    }
    return out;
}

tensor_id compute_tensor_id(
    std::string_view name,
    ggml_type type,
    const std::vector<int64_t> & dims) {
    if (name.empty() || dims.empty()) {
        return {}; // all zeros
    }

    auto name_enc = canonical_encode_string(name);
    auto type_enc = canonical_encode_type(type);
    auto shape_enc = canonical_encode_shape(dims);

    sha256_context ctx;
    ctx.update(name_enc.data(), name_enc.size());
    ctx.update(type_enc.data(), type_enc.size());
    ctx.update(shape_enc.data(), shape_enc.size());
    return ctx.finalize();
}

std::string tensor_id_to_hex(const tensor_id & id) {
    std::ostringstream oss;
    for (uint8_t b : id) {
        oss << std::hex << std::setfill('0') << std::setw(2)
            << static_cast<int>(b);
    }
    return oss.str();
}

std::optional<tensor_id> tensor_id_from_hex(std::string_view hex) {
    if (hex.size() != 64) return std::nullopt;
    tensor_id id{};
    for (size_t i = 0; i < 32; i++) {
        std::string byte_str = std::string(hex.substr(i * 2, 2));
        try {
            uint8_t val = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
            id[i] = val;
        } catch (...) {
            return std::nullopt;
        }
    }
    return id;
}

// ===========================================================================
// Tensor descriptor
// ===========================================================================

tensor_id tensor_descriptor::id() const {
    return llama_train::compute_tensor_id(name, type, dims);
}

size_t tensor_descriptor::row_size() const {
    if (dims.empty()) return 0;
    int64_t ne = dims[0];
    return ggml_row_size(type, ne);
}

size_t tensor_descriptor::total_bytes() const {
    if (dims.empty()) return 0;
    // Checked multiplication for total elements
    const int64_t max_i64 = INT64_MAX;
    int64_t total = dims[0];
    for (size_t i = 1; i < dims.size(); i++) {
        // Overflow check
        if (dims[i] > 0 && total > max_i64 / dims[i]) {
            return SIZE_MAX; // overflow sentinel
        }
        total *= dims[i];
    }
    if (total <= 0) return 0;

    size_t row_bytes = ggml_row_size(type, dims[0]);
    int64_t rows = total / dims[0];
    // Check multiplication overflow for bytes
    if (rows > 0 && static_cast<int64_t>(row_bytes) > max_i64 / rows) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(rows) * row_bytes;
}

int64_t tensor_descriptor::nelements() const {
    if (dims.empty()) return 0;
    int64_t total = dims[0];
    for (size_t i = 1; i < dims.size(); i++) {
        total *= dims[i];
    }
    return total;
}

// ===========================================================================
// Tied-weight validation
// ===========================================================================

std::function<std::string(std::string_view)> default_tie_key_fn() {
    return [](std::string_view name) -> std::string {
        // Default: each tensor is its own group (no tying).
        // Callers should provide a custom function to detect ties.
        return std::string(name);
    };
}

tied_validation_result validate_tied_groups(
    const std::vector<tensor_descriptor> & descriptors,
    std::function<std::string(std::string_view)> tie_key_fn) {

    tied_validation_result result;
    result.ok = false;

    // Group descriptors by tie key
    struct group_info {
        const tensor_descriptor * base = nullptr;
        std::vector<std::string> aliases;
    };

    std::map<std::string, group_info> groups;

    for (const auto & desc : descriptors) {
        if (desc.name.empty()) {
            result.error = "Tensor has empty name";
            return result;
        }
        if (desc.dims.empty()) {
            result.error = "Tensor '" + desc.name + "' has empty dimensions";
            return result;
        }
        if (desc.type >= GGML_TYPE_COUNT) {
            result.error = "Tensor '" + desc.name + "' has invalid type " +
                          std::to_string(static_cast<int>(desc.type));
            return result;
        }

        std::string key = tie_key_fn(desc.name);
        auto & g = groups[key];
        if (!g.base) {
            g.base = &desc;
            g.aliases.push_back(desc.name);
        } else {
            // Check compatibility with base
            if (desc.type != g.base->type || desc.dims != g.base->dims) {
                result.error = "Tied tensors '" + g.base->name + "' and '" + desc.name +
                              "' have incompatible type/shape. Base: type=" +
                              std::to_string(static_cast<int>(g.base->type)) +
                              ", dims=[";
                for (size_t i = 0; i < g.base->dims.size(); i++) {
                    if (i) result.error += ", ";
                    result.error += std::to_string(g.base->dims[i]);
                }
                result.error += "]. Got: type=" +
                              std::to_string(static_cast<int>(desc.type)) +
                              ", dims=[";
                for (size_t i = 0; i < desc.dims.size(); i++) {
                    if (i) result.error += ", ";
                    result.error += std::to_string(desc.dims[i]);
                }
                result.error += "]";
                return result;
            }
            // Check not already aliased
            for (const auto & alias : g.aliases) {
                if (alias == desc.name) {
                    result.error = "Duplicate tensor name '" + desc.name + "'";
                    return result;
                }
            }
            g.aliases.push_back(desc.name);
        }
    }

    // Build tied groups
    for (auto & [key, info] : groups) {
        tied_group tg;
        tg.base = *info.base;
        tg.aliases = std::move(info.aliases);
        result.groups.push_back(std::move(tg));
    }

    result.ok = true;
    return result;
}

tied_validation_result validate_standard_tied(
    const std::vector<tensor_descriptor> & descriptors) {

    auto tie_key = [](std::string_view name) -> std::string {
        // Standard tying: token_embd.weight and output.weight are tied.
        // Detect by checking if the name ends with ".weight" and the stem
        // is "token_embd" or "output".
        if (name.size() > 6 && name.substr(name.size() - 6) == ".weight") {
            std::string stem(name.begin(), name.end() - 7); // without .weight
            if (stem == "token_embd" || stem == "output") {
                return "__tied_embedding__";
            }
        }
        return std::string(name);
    };

    return validate_tied_groups(descriptors, tie_key);
}

// ===========================================================================
// Tied group implementation
// ===========================================================================

tensor_id tied_group::id() const {
    return base.id();
}

std::vector<tensor_id> get_unique_ids(const std::vector<tied_group> & groups) {
    std::vector<tensor_id> ids;
    ids.reserve(groups.size());
    for (const auto & g : groups) {
        ids.push_back(g.id());
    }
    // Sort and unique (should already be unique, but defensive)
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

} // namespace llama_train
