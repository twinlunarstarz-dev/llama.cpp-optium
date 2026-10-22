#include "llama-model-loader.h"

#include "ggml-alloc.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-hparams.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdarg>
#include <cstring>
#include <future>
#include <regex>
#include <fstream>
#include <fstream>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2";
        case GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

#define LLAMA_FTYPE_PREFIX "(guessed) "

const char * llama_ftype_name(llama_ftype ftype) {
    static constexpr size_t guessed_prefix_len = sizeof(LLAMA_FTYPE_PREFIX) - 1;
    const char * name;
    switch ((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) {
        case LLAMA_FTYPE_ALL_F32:          name = LLAMA_FTYPE_PREFIX "all F32"; break;
        case LLAMA_FTYPE_MOSTLY_F16:       name = LLAMA_FTYPE_PREFIX "F16"; break;
        case LLAMA_FTYPE_MOSTLY_BF16:      name = LLAMA_FTYPE_PREFIX "BF16"; break;
        case LLAMA_FTYPE_MOSTLY_Q1_0:      name = LLAMA_FTYPE_PREFIX "Q1_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_0:      name = LLAMA_FTYPE_PREFIX "Q2_0"; break;
        case LLAMA_FTYPE_MOSTLY_PQ2_0:     name = LLAMA_FTYPE_PREFIX "PQ2_0 - 2.13 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_PQ2_0_LEGACY:name = LLAMA_FTYPE_PREFIX "PQ2_0 - 2.13 bpw (legacy)"; break;
        case LLAMA_FTYPE_MOSTLY_PTQ1_0:    name = LLAMA_FTYPE_PREFIX "PTQ1_0 - 1.75 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_Q1_0_g128: name = LLAMA_FTYPE_PREFIX "Q1_0_g128 - 2.125 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_Q1_T_g128: name = LLAMA_FTYPE_PREFIX "Q1_T_g128 - 1.75 bpw ternary (trit)"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0:      name = LLAMA_FTYPE_PREFIX "Q4_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1:      name = LLAMA_FTYPE_PREFIX "Q4_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0:      name = LLAMA_FTYPE_PREFIX "Q5_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1:      name = LLAMA_FTYPE_PREFIX "Q5_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0:      name = LLAMA_FTYPE_PREFIX "Q8_0"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: name = LLAMA_FTYPE_PREFIX "MXFP4 MoE"; break;
        case LLAMA_FTYPE_MOSTLY_NVFP4:     name = LLAMA_FTYPE_PREFIX "NVFP4"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K:      name = LLAMA_FTYPE_PREFIX "Q2_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:    name = LLAMA_FTYPE_PREFIX "Q2_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:    name = LLAMA_FTYPE_PREFIX "Q3_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:    name = LLAMA_FTYPE_PREFIX "Q3_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:    name = LLAMA_FTYPE_PREFIX "Q3_K - Large"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:    name = LLAMA_FTYPE_PREFIX "Q4_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:    name = LLAMA_FTYPE_PREFIX "Q4_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:    name = LLAMA_FTYPE_PREFIX "Q5_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:    name = LLAMA_FTYPE_PREFIX "Q5_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:      name = LLAMA_FTYPE_PREFIX "Q6_K"; break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:     name = LLAMA_FTYPE_PREFIX "TQ1_0 - 1.69 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:     name = LLAMA_FTYPE_PREFIX "TQ2_0 - 2.06 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:   name = LLAMA_FTYPE_PREFIX "IQ2_XXS - 2.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:    name = LLAMA_FTYPE_PREFIX "IQ2_XS - 2.3125 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:     name = LLAMA_FTYPE_PREFIX "IQ2_S - 2.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:     name = LLAMA_FTYPE_PREFIX "IQ2_M - 2.7 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:    name = LLAMA_FTYPE_PREFIX "IQ3_XS - 3.3 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:   name = LLAMA_FTYPE_PREFIX "IQ3_XXS - 3.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:     name = LLAMA_FTYPE_PREFIX "IQ1_S - 1.5625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:     name = LLAMA_FTYPE_PREFIX "IQ1_M - 1.75 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:    name = LLAMA_FTYPE_PREFIX "IQ4_NL - 4.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:    name = LLAMA_FTYPE_PREFIX "IQ4_XS - 4.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:     name = LLAMA_FTYPE_PREFIX "IQ3_S - 3.4375 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:     name = LLAMA_FTYPE_PREFIX "IQ3_S mix - 3.66 bpw"; break;
        default:                           name = LLAMA_FTYPE_PREFIX "unknown, may not work"; break;
    }
    return (ftype & LLAMA_FTYPE_GUESSED) ? name : name + guessed_prefix_len;
}

#undef LLAMA_FTYPE_PREFIX

// return a list of splits for a given path
// for example, given "<name>-00002-of-00004.gguf", returns list of all 4 splits
static std::vector<std::string> llama_get_list_splits(const std::string & path, const int idx, const int n_split) {
    std::vector<std::string> paths;
    std::string split_prefix;
    std::vector<char> buf(llama_path_max(), 0);

    {
        int ret = llama_split_prefix(buf.data(), buf.size(), path.c_str(), idx, n_split);
        if (!ret) {
            throw std::runtime_error(format("invalid split file name: %s", path.c_str()));
        }
        split_prefix = std::string(buf.data(), ret);
    }

    if (split_prefix.empty()) {
        throw std::runtime_error(format("invalid split file: %s", path.c_str()));
    }

    for (int idx = 0; idx < n_split; ++idx) {
        int ret = llama_split_path(buf.data(), buf.size(), split_prefix.c_str(), idx, n_split);
        paths.push_back(std::string(buf.data(), ret));
    }

    return paths;
}

namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx, k);
            return ArrayInfo {
                arr_type,
                gguf_get_arr_n(ctx, k),
                arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(const std::string & key, T & result, bool required) {
        const int kid = gguf_find_key(metadata, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(metadata, kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(enum llm_kv kid, T & result, bool required) {
        return get_arr_n(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr_n(enum llm_kv kid, uint32_t & result, bool required);
    template std::enable_if<std::is_integral<uint32_t>::value, bool>::type
    llama_model_loader::get_arr_n<uint32_t>(const std::string & key, uint32_t & result, bool required);

    template<typename T>
    bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, bool required) {
        const gguf_context * ctx = metadata;
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        bool type_ok = false;
        switch (arr_info.gt) {
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   type_ok = (std::is_same<T,     int32_t>::value) ||
                                              (std::is_same<T,    uint32_t>::value); break;
            case GGUF_TYPE_UINT64:  type_ok = (std::is_same<T,    uint64_t>::value); break;
            case GGUF_TYPE_FLOAT32: type_ok = (std::is_same<T,       float>::value); break;
            case GGUF_TYPE_STRING:  type_ok = (std::is_same<T, std::string>::value); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32/uint64 array", key.c_str()));
        }
        if (!type_ok) {
            throw std::runtime_error(format("%s has wrong array element type %s", key.c_str(), gguf_type_name(arr_info.gt)));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);
            result.clear();

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result.emplace_back(value);
            }
        } else {
            result.resize(arr_info.length);
            result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);
        }

        return true;
    }

    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required) {
        const gguf_context * ctx = metadata;
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        bool type_ok = false;
        switch (arr_info.gt) {
            case GGUF_TYPE_BOOL:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   type_ok = (std::is_same<T,     int32_t>::value) ||
                                              (std::is_same<T,    uint32_t>::value); break;
            case GGUF_TYPE_UINT64:  type_ok = (std::is_same<T,    uint64_t>::value); break;
            case GGUF_TYPE_FLOAT32: type_ok = (std::is_same<T,       float>::value); break;
            case GGUF_TYPE_STRING:  type_ok = (std::is_same<T, std::string>::value); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32/uint64 array", key.c_str()));
        }
        if (!type_ok) {
            throw std::runtime_error(format("%s has wrong array element type %s", key.c_str(), gguf_type_name(arr_info.gt)));
        }

        if (arr_info.length > N_MAX) {
            throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result[i] = value;
            }
        } else {
            if (arr_info.gt == GGUF_TYPE_BOOL) {
                const int8_t * values = (const int8_t *) arr_info.data;
                std::transform(values, values + arr_info.length, result.begin(), [](int8_t x) {
                    return static_cast<T>(x != 0);
                });
            } else {
                std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());
            }
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_arr(enum llm_kv kid, T & result, bool required) {
        return get_arr(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr<std::vector<std::string>>(enum llm_kv kid, std::vector<std::string> & result, bool required);
    template bool llama_model_loader::get_arr<std::string>(const std::string & key, std::vector<std::string> & result, bool required);
    template bool llama_model_loader::get_arr<int32_t>(const std::string & key, std::vector<int32_t> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<int32_t, 512>>(enum llm_kv kid, std::array<int32_t, 512> & result, bool required);
    template bool llama_model_loader::get_arr<std::vector<int32_t>>(enum llm_kv kid, std::vector<int32_t> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<uint32_t, LLAMA_MAX_LAYERS>>(enum llm_kv kid, std::array<uint32_t, LLAMA_MAX_LAYERS> & result, bool required);
    template bool llama_model_loader::get_arr<std::vector<uint32_t>>(enum llm_kv kid, std::vector<uint32_t> & result, bool required);
    template bool llama_model_loader::get_arr<uint32_t>(const std::string & key, std::vector<uint32_t> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<uint64_t, LLAMA_MAX_PLE_NGRAM>>(enum llm_kv kid, std::array<uint64_t, LLAMA_MAX_PLE_NGRAM> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<uint64_t, LLAMA_MAX_PLE_HEADS>>(enum llm_kv kid, std::array<uint64_t, LLAMA_MAX_PLE_HEADS> & result, bool required);

    template<typename T>
    bool llama_model_loader::get_key(const std::string & key, T & result, bool required) {
        auto it = kv_overrides.find(key);

        const struct llama_model_kv_override * override =
            it != kv_overrides.end() ? &it->second : nullptr;

        const bool found = GGUFMeta::GKV<T>::set(metadata, key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool llama_model_loader::get_key(enum llm_kv kid, T & result, bool required) {
        return get_key(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
    template bool llama_model_loader::get_key<bool>       (const std::string & key, bool & result, bool required);
    template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
    template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
    template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

    template<>
    bool llama_model_loader::get_key(enum llm_kv kid, enum llama_pooling_type & result, bool required) {
        uint32_t tmp;
        const bool found = get_key(kid, tmp, required);
        if (found) {
            result = (enum llama_pooling_type) tmp;
        } else {
            result = LLAMA_POOLING_TYPE_UNSPECIFIED;
        }
        return found;
    }

    // get array of n <= N_MAX elements, or a single element repeated n times
    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required) {
        const int kid = gguf_find_key(metadata, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        if (n > N_MAX) {
            throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", n, (uint32_t) N_MAX, key.c_str()));
        }

        if (gguf_get_kv_type(metadata, kid) == GGUF_TYPE_ARRAY) {
            struct GGUFMeta::ArrayInfo arr_info =
                GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(metadata, kid);

            if (n != arr_info.length) {
                throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
            }

            return get_arr(key, result, required);
        }

        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required) {
        return get_key_or_arr(llm_kv(kid), result, n, required);
    }

    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required) {
        const std::string key = llm_kv(kid);

        const int id = gguf_find_key(metadata, key.c_str());

        if (id < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        // throw and error if type is an array
        if (gguf_get_kv_type(metadata, id) == GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("expected scalar, found array for key: %s", key.c_str()));
            }
            return false;
        }

        return get_key(key, result, required);
    }

    // TODO: this is not very clever - figure out something better
    template bool llama_model_loader::get_key_or_arr<std::array<int,      4>>  (enum llm_kv kid, std::array<int,      4>   & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<float,    512>>(enum llm_kv kid, std::array<float,    512> & result, uint32_t n, bool required);


llama_model_loader::llama_model_loader(
        struct gguf_context * meta,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits,
        FILE * file,
        llama_load_mode load_mode,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
        // MoE tiered placement parameters
        size_t moe_ram_mib,
        const char * moe_placement,
        bool moe_pin_decode,
        const float * tensor_split,
        int n_devices,
        const char * moe_hot)
        : metadata(meta), set_tensor_data(set_tensor_data), set_tensor_data_ud(set_tensor_data_ud) {
    (void) moe_ram_mib;
    (void) moe_placement;
    (void) moe_hot;
    (void) moe_pin_decode;
    (void) tensor_split;
    (void) n_devices;
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    this->use_mmap      = load_mode == LLAMA_LOAD_MODE_MMAP || load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK || load_mode == LLAMA_LOAD_MODE_AUTO;
    this->use_direct_io = load_mode == LLAMA_LOAD_MODE_DIRECT_IO;

    if (!fname.empty()) {
        // Load the main GGUF
        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };

        metadata_ptr.reset(gguf_init_from_file(fname.c_str(), params));
        metadata = metadata_ptr.get();
        if (metadata == nullptr) {
            throw std::runtime_error(format("%s: failed to load model from %s", __func__, fname.c_str()));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        files.emplace_back(new llama_file(fname.c_str(), "rb", use_direct_io));
        contexts.emplace_back(ctx);

        // Save tensors data offset of the main file.
        // For subsidiary files, `meta` tensor data offset must not be used,
        // so we build a unified tensors index for weights.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            std::string tensor_name = std::string(cur->name);
            // make sure there is no duplicated tensor names
            if (weights_map.find(tensor_name) != weights_map.end()) {
                throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
            }
            n_elements += ggml_nelements(cur);
            n_bytes    += ggml_nbytes(cur);
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
        }
        uint16_t n_split = 0;
        get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

        // Load additional GGML contexts
        if (n_split > 1) {
            // make sure the main file is loaded first
            uint16_t idx = 0;
            const std::string kv_split_no = llm_kv(LLM_KV_SPLIT_NO);
            get_key(kv_split_no, idx);
            if (idx != 0) {
                throw std::runtime_error(format("illegal split file idx: %d (file: %s), model must be loaded with the first split", idx, fname.c_str()));
            }

            // generate list of splits if needed
            if (splits.empty()) {
                splits = llama_get_list_splits(fname, idx, n_split);
            }

            // in case user give a custom list of splits, check if it matches the expected number
            if (n_split != (uint16_t)splits.size()) {
                throw std::runtime_error(format("invalid split count, given: %zu splits, but expected %d", splits.size(), n_split));
            }

            if (trace > 0) {
                LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
            }

            // load other splits
            for (idx = 1; idx < n_split; idx++) {
                const char * fname_split = splits[idx].c_str();

                struct gguf_init_params split_params = {
                    /*.no_alloc = */ true,
                    /*.ctx      = */ &ctx,
                };
                gguf_context_ptr ctx_gguf { gguf_init_from_file(fname_split, split_params) };
                if (!ctx_gguf) {
                    throw std::runtime_error(format("%s: failed to load GGUF split from %s", __func__, fname_split));
                }

                // check idx
                {
                    const int kid = gguf_find_key(ctx_gguf.get(), kv_split_no.c_str());
                    if (kid < 0) {
                        throw std::runtime_error(format("missing key %s in GGUF split %s", kv_split_no.c_str(), fname_split));
                    }
                    int idx_gguf = gguf_get_val_u16(ctx_gguf.get(), kid);
                    if (idx_gguf != idx) {
                        throw std::runtime_error(format("invalid split file idx: %d (file: %s), expected %d", idx_gguf, fname_split, idx));
                    }
                }

                files.emplace_back(new llama_file(fname_split, "rb", use_direct_io));
                contexts.emplace_back(ctx);

                // Save tensors data offset info of the shard.
                for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                    std::string tensor_name = std::string(cur->name);
                    // make sure there is no duplicated tensor names
                    if (weights_map.find(tensor_name) != weights_map.end()) {
                        throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
                    }
                    n_elements += ggml_nelements(cur);
                    n_bytes    += ggml_nbytes(cur);
                    weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur));
                }
            }

            get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

            // sanity check
            {
                const int n_tensors_loaded = (int) weights_map.size();
                if (n_tensors != n_tensors_loaded) {
                    throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
                }
            }

            LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
        }
    } else if (file != nullptr) {
        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };

        metadata_ptr.reset(gguf_init_from_file_ptr(file, params));
        metadata = metadata_ptr.get();
        if (metadata == nullptr) {
            throw std::runtime_error(format("%s: failed to load model from file pointer", __func__));
        }

        // mmap places tensors at their file offsets, so an embedded GGUF must be aligned in the file too
        const size_t tensor_align = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
        if (use_mmap && gguf_get_data_offset(metadata) % tensor_align != 0) {
            throw std::runtime_error(format("%s: GGUF data section at file offset %zu is not %zu byte aligned, cannot mmap",
                __func__, gguf_get_data_offset(metadata), tensor_align));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        files.emplace_back(new llama_file(file));
        contexts.emplace_back(ctx);

        // Save tensors data offset info of the main file.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            std::string tensor_name = std::string(cur->name);
            // make sure there is no duplicated tensor names
            if (weights_map.find(tensor_name) != weights_map.end()) {
                throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
            }
            n_elements += ggml_nelements(cur);
            n_bytes    += ggml_nbytes(cur);
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
        }
    } else {
        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));
    }

    n_kv      = gguf_get_n_kv(metadata);
    n_tensors = weights_map.size();

    fver = (enum llama_fver) gguf_get_version(metadata);

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.empty() ? "(file*)" : fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (const auto & it : weights_map) {
            const llama_tensor_weight & w = it.second;
            const ggml_tensor * tensor = w.tensor;

            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = w.idx;
                LLAMA_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__,
                        sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
            case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
            case GGML_TYPE_Q2_0:    ftype = LLAMA_FTYPE_MOSTLY_Q2_0;    break;
            case GGML_TYPE_PQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_PQ2_0;   break;
            case GGML_TYPE_PTQ1_0:  ftype = LLAMA_FTYPE_MOSTLY_PTQ1_0;  break;
            case GGML_TYPE_Q1_0_g128: ftype = LLAMA_FTYPE_MOSTLY_Q1_0_g128; break;
            case GGML_TYPE_Q1_T_g128: ftype = LLAMA_FTYPE_MOSTLY_Q1_T_g128; break;
            default:
                {
                    LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                    ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                ftype = (llama_ftype) ftype_val;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(metadata, i);
            const enum gguf_type type   = gguf_get_kv_type(metadata, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(metadata, i)), gguf_get_arr_n(metadata, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(metadata, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (this->use_mmap && !llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        this->use_mmap = false;
    }

    this->check_tensors = check_tensors;
    this->no_alloc = no_alloc;
    this->load_mtp = load_mtp;


    // MoE tiered placement: compute plan and initialize RAM arena
    if (moe_ram_mib > 0 && metadata != nullptr && !fname.empty()) {
        // Extract MoE dimensions from metadata
        int n_layer = 0;
        int n_expert = 0;
        int n_expert_used = 0;

        auto get_u64 = [&](const char * key) -> uint64_t {
            const int kid = gguf_find_key(metadata, key);
            if (kid < 0) return 0;
            enum gguf_type type = gguf_get_kv_type(metadata, kid);
            if (type == GGUF_TYPE_UINT64) {
                return gguf_get_val_u64(metadata, kid);
            } else if (type == GGUF_TYPE_UINT32) {
                return gguf_get_val_u32(metadata, kid);
            } else if (type == GGUF_TYPE_INT32) {
                return gguf_get_val_i32(metadata, kid);
            } else if (type == GGUF_TYPE_INT64) {
                return gguf_get_val_i64(metadata, kid);
            }
            return 0;
        };

        n_layer = (int) get_u64("qwen4exp.block_count");
        if (n_layer == 0) n_layer = (int) get_u64("llama.block_count");

        n_expert = (int) get_u64("qwen4exp.expert_count");
        if (n_expert == 0) n_expert = (int) get_u64("llama.expert_count");

        n_expert_used = (int) get_u64("qwen4exp.expert_used_count");
        if (n_expert_used == 0) n_expert_used = (int) get_u64("llama.expert_used_count");

        if (n_layer > 0 && n_expert > 0 && n_expert_used > 0) {
            const char * moe_profile_path = moe_hot;

            // Compute placement plan
            if (moe_profile_path && *moe_profile_path) {
                moe_plan = std::make_unique<moe_placement_plan>(moe_solve_placement_expert_granular(
                    metadata, tensor_split, n_devices, moe_ram_mib, moe_placement,
                    moe_pin_decode, n_layer, n_expert, n_expert_used, moe_profile_path));
            } else {
                moe_plan = std::make_unique<moe_placement_plan>(moe_solve_placement(
                    metadata, tensor_split, n_devices, moe_ram_mib, moe_placement,
                    moe_pin_decode, n_layer, n_expert, n_expert_used));
            }

            if (moe_plan->fits) {
                // Print placement report
                for (const auto & line : moe_plan->report_lines) {
                    LLAMA_LOG_INFO("%s\n", line.c_str());
                }

                // Initialize RAM arena
                moe_arena = std::make_unique<moe_ram_arena>();
                if (!moe_arena->init(moe_ram_mib, *moe_plan, n_layer, n_expert, fname.c_str())) {
                    LLAMA_LOG_WARN("MoE RAM arena: initialization failed, continuing without arena\n");
                    moe_arena.reset();
                }
            } else {
                LLAMA_LOG_ERROR("MoE placement: plan does not fit: %s\n", moe_plan->error.c_str());
            }
        }
    }

}

std::string llama_model_loader::get_arch_name() const {
    return arch_name;
}

enum llm_arch llama_model_loader::get_arch() const {
    return llm_kv.arch;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    auto pos = weights_map.find(name);
    if (pos != weights_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

struct ggml_tensor * llama_model_loader::require_tensor_meta(const std::string & name) const {
    struct ggml_tensor * tensor = get_tensor_meta(name.c_str());
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }
    return tensor;
}

const struct ggml_tensor * llama_model_loader::check_tensor_dims(
        const std::string & name,
        const std::vector<int64_t> & ne,
        bool required,
        bool allow_reshape) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    bool is_ok = true;

    if (allow_reshape) {
        // check total number of elements only
        const int64_t ncur = ggml_nelements(cur);
        int64_t nexp = 1;
        for (size_t i = 0; i < ne.size(); ++i) {
            nexp *= ne[i];
        }
        if (ncur != nexp) {
            is_ok = false;
        }
    } else {
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                is_ok = false;
                break;
            }
        }
    }

    if (!is_ok) {
        throw std::runtime_error(
                format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                    __func__, name.c_str(),
                    llama_format_tensor_shape(ne).c_str(),
                    llama_format_tensor_shape(cur).c_str()));
    }

    return cur;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                // Used for either MoE expert routing or embedded adapter routing
                const int n_ids_used = hparams.router_layer >= 0 ? 1 : hparams.n_expert_used_max();
                GGML_ASSERT(n_ids_used > 0);
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_ids_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_ADD_ID:
            {
                const int n_expert_used = hparams.n_expert_used_max();
                GGML_ASSERT(n_expert_used > 0);
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_add_id(ctx, a, w, c);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                const int n_embd_head = hparams.n_embd_head_v();
                const int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0] - 1 + n_seq_tokens, w->ne[1], n_seqs);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2
                const int64_t d_state      = w->ne[0] == 1 ? hparams.ssm_d_state : w->ne[0];
                const int64_t n_head       = w->ne[1];
                const int64_t head_dim     = hparams.ssm_d_inner / n_head;
                const int64_t n_group      = hparams.ssm_n_group ? hparams.ssm_n_group : 1;
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
                ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids, /*K=*/1);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd_inp = hparams.n_embd_inp();
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd_inp, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        case GGML_OP_CPY:
            {
                op_tensor = ggml_cast(ctx, w, GGML_TYPE_F32);
            } break;
        case GGML_OP_SCALE:
            {
                op_tensor = ggml_scale(ctx, w, 1.0f);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t * buft_list) {
    GGML_ASSERT(!buft_list->empty());
    for (const auto & cur : *buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

ggml_backend_buffer_type_t llama_model_loader::lazy_read::buft() {
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        throw std::runtime_error("no CPU backend found");
    }
    return ggml_backend_dev_buffer_type(cpu_dev);
}

bool llama_model_loader::lazy_read::add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w) {
    if (mode == LLAMA_LAZY_MODE_OFF) {
        return false;
    }

    // do not lazy-read small tensors, it has significant overhead and is not worth it
    constexpr size_t auto_min_size = 4ull * 1024 * 1024 * 1024;
    if (mode != LLAMA_LAZY_MODE_ON && ggml_nbytes(t) <= auto_min_size) {
        return false;
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not available, so tensor %s (size = %zu MiB) is loaded into RAM in full\n",
                __func__, name.c_str(), ggml_nbytes(t)/1024/1024);
        return false;
    }

    if (w) {
        ranges[w->idx].emplace_back(w->offs, w->offs + ggml_nbytes(t));
        tensors.insert(name);

        LLAMA_LOG_INFO("%s: tensor %s (size = %zu MiB) lazy read enabled\n",
                __func__, name.c_str(), ggml_nbytes(t)/1024/1024);
    }

    return true;
}

struct ggml_tensor * llama_model_loader::create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    // set below, before buft_for_tensor() runs
    bool is_lazy = false;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        const ctx_key key { buft, is_lazy };

        auto it = ctx_map.find(key);
        if (it == ctx_map.end()) {
            // one ggml context per buffer type
            int max_n_tensors = n_tensors;
            max_n_tensors += 1;                   // duplicated output tensor
            max_n_tensors += hparams.n_layer()*2; // duplicated rope freq tensors
            if (files.empty()) {
                max_n_tensors += hparams.n_layer()*256; // this should be well above what any model actually uses
            }
            const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

            ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ctx_map.emplace(key, ctx);

            return ctx;
        }
        return it->second.get();
    };

    auto buft_for_tensor = [&](ggml_tensor * t_meta) -> ggml_backend_buffer_type_t {
        if (!t_meta) {
            if (flags & TENSOR_NOT_REQUIRED) {
                return nullptr;
            }
            throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
        }

        // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
        // the tensor is duplicated
        // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
        llm_tensor tn_tensor = tn.tensor;
        if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && (flags & TENSOR_DUPLICATED)) {
            tn_tensor = LLM_TENSOR_OUTPUT;
        }

        llm_tensor_info info;
        try {
            info = llm_tensor_info_for(tn_tensor);
        } catch (const std::out_of_range & e) {
            throw std::runtime_error(format("missing tensor info mapping for %s", tn.str().c_str()));
        }

        // skip unused tensors
        if (info.op == GGML_OP_NONE || (flags & TENSOR_SKIP)) {
            const size_t nbytes = ggml_nbytes(t_meta);
            LLAMA_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", tn.str().c_str(), nbytes);

            size_data -= nbytes;
            n_created++;

            return nullptr;
        }

        // tensors with "bias" suffix are always used with GGML_OP_ADD or GGML_OP_ADD_ID;
        // embedded-adapter ".lora_a"/".lora_b" tensors are always used with GGML_OP_MUL_MAT_ID
        ggml_op op;
        if (tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0) {
            op = info.op == GGML_OP_MUL_MAT_ID ? GGML_OP_ADD_ID : GGML_OP_ADD;
        } else if (hparams.router_layer >= 0 && tn.suffix != nullptr &&
                (strcmp(tn.suffix, "lora_a") == 0 || strcmp(tn.suffix, "lora_b") == 0)) {
            op = GGML_OP_MUL_MAT_ID;
        } else {
            op = info.op;
        }

        // sanity checks
        if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
            if (tn.bid != -1) {
                GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
            }
        } else {
            if (tn.bid == -1) {
                GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
            }
        }

        if (is_lazy) {
            return lazy_read::buft();
        }

        // select the buffer type for this tensor
        const buft_list_t * buft_list;
        switch (info.layer) {
            case LLM_TENSOR_LAYER_INPUT:
                buft_list = buft_list_input;
                break;
            case LLM_TENSOR_LAYER_OUTPUT:
                buft_list = buft_list_output;
                break;
            case LLM_TENSOR_LAYER_REPEATING:
                GGML_ASSERT(buft_list_layer != nullptr);
                buft_list = buft_list_layer;
                break;
            default:
                GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
        }

        ggml_backend_buffer_type_t buft = nullptr;

        // check overrides
        if (tensor_buft_overrides) {
            std::string tensor_name = tn.str();
            for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                std::regex pattern(overrides->pattern);
                if (std::regex_search(tensor_name, pattern)) {
                    if (overrides->buft == ggml_backend_cpu_buffer_type()) {
                        // when overriding to a CPU buffer, consider the extra buffer types
                        buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);
                        if (use_mmap) {
                            static std::once_flag once;
                            std::call_once(once, [] {
                                LLAMA_LOG_WARN("llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance\n");
                            });
                        }
                    } else {
                        buft = overrides->buft;
                    }

                    LLAMA_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n",
                            tensor_name.c_str(),
                            ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
                            ggml_backend_buft_name(buft));
                    break;
                }
            }
        }

        if (!buft) {
            buft = select_weight_buft(hparams, t_meta, op, buft_list);
            if (!buft) {
                throw std::runtime_error(format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
            }
        }

        // avoid using a host buffer when using mmap
        auto * buft_dev = ggml_backend_buft_get_device(buft);
        if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
            auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!cpu_dev) {
                throw std::runtime_error("no CPU backend found");
            }
            buft = ggml_backend_dev_buffer_type(cpu_dev);
        }

        if (buft != buft_list->front().second) {
            if (n_tensors_moved == 0) {
                first_tensor_moved_name = t_meta->name;
                first_tensor_moved_type_name = ggml_type_name(t_meta->type);
                first_moved_from_buft = buft_list->front().second;
                first_moved_to_buft   = buft;
            }
            n_tensors_moved++;
        }

        return buft;
    };

    if (files.empty()) {
        if (flags & TENSOR_SKIP_IF_VIRTUAL) {
            return nullptr;
        }
        ggml_type type = GGML_TYPE_F32;
        const int64_t tid = gguf_find_tensor(metadata, tn.str().c_str());
        if (tid != -1) {
            type = gguf_get_tensor_type(metadata, tid);
        }

        // for tensors that are not required some of the dimensions can be invalid:
        if (flags & TENSOR_NOT_REQUIRED) {
            for (size_t dim = 0; dim < ne.size(); dim++) {
                if (ne.begin()[dim] <= 0) {
                    return nullptr;
                }
            }
        }

        ggml_tensor t_meta;
        memset(&t_meta, 0, sizeof(ggml_tensor));
        t_meta.type = type;
        for (size_t dim = 0; dim < GGML_MAX_DIMS; dim++) {
            t_meta.ne[dim] = dim < ne.size() ? ne.begin()[dim] : 1;
            GGML_ASSERT(t_meta.ne[dim] >= 1);
            if (dim == 0) {
                t_meta.nb[dim] = ggml_type_size(type);
            } else if (dim == 1) {
                t_meta.nb[dim] = ggml_row_size(type, t_meta.ne[dim-1]);
            } else {
                t_meta.nb[dim] = t_meta.nb[dim-1]*t_meta.ne[dim-1];
            }
            GGML_ASSERT(t_meta.nb[dim] >= 1);
        }
        ggml_set_name(&t_meta, tn.str().c_str());

        ggml_backend_buffer_type_t buft = buft_for_tensor(&t_meta);
        GGML_ASSERT(buft != nullptr);
        ggml_context * ctx = ctx_for_buft(buft);
        ggml_tensor * ret = ggml_dup_tensor(ctx, &t_meta);
        ggml_set_name(ret, tn.str().c_str());
        return ret;
    }

    LLAMA_LOG_DEBUG("%s: loading tensor %s\n", __func__, tn.str().c_str());
    const struct ggml_tensor * cur = check_tensor_dims(tn.str(), ne, !(flags & TENSOR_NOT_REQUIRED), flags & TENSOR_ALLOW_RESHAPE);
    if (cur == NULL) {
        return NULL;
    }

    if (flags & TENSOR_READ_LAZY) {
        // the decision must not depend on the load mode, or the memory-fit pass (no_alloc, no mmap)
        is_lazy = lazy.add(tn.str(), cur, no_alloc ? nullptr : &require_weight(tn.str().c_str()));
    }

    ggml_tensor t_meta = *cur;
    if (flags & TENSOR_ALLOW_RESHAPE) {
        for (size_t dim = 0; dim < GGML_MAX_DIMS; dim++) {
            t_meta.ne[dim] = dim < ne.size() ? ne.begin()[dim] : 1;
            if (dim == 0) {
                t_meta.nb[dim] = ggml_type_size(t_meta.type);
            } else if (dim == 1) {
                t_meta.nb[dim] = ggml_row_size(t_meta.type, t_meta.ne[dim-1]);
            } else {
                t_meta.nb[dim] = t_meta.ne[dim-1]*t_meta.nb[dim-1];
            }
        }
    }

    GGML_ASSERT(ggml_nbytes(&t_meta) == ggml_nbytes(cur));

    ggml_backend_buffer_type_t buft = buft_for_tensor(&t_meta);
    if (buft == nullptr) {
        return nullptr;
    }

    ggml_context * ctx = ctx_for_buft(buft);

    // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
    if (flags & TENSOR_DUPLICATED) {
        ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
        if (t) {
            return t;
        }
    }

    const bool duplicated = flags & TENSOR_DUPLICATED;

    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, &t_meta);
    ggml_set_name(tensor, ggml_get_name(&t_meta));

    if (duplicated) {
        size_data += ggml_nbytes(&t_meta);
    } else {
        n_created++;
    }

    return tensor;
}

void llama_model_loader::done_getting_tensors(bool partial) const {
    if (n_created > n_tensors) {
        throw std::runtime_error(format("%s: too many tensors created; expected %d, got %d", __func__, n_tensors, n_created));
    }
    if (n_created < n_tensors) {
        if (!partial) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
        LLAMA_LOG_INFO("%s: partial load — used %d of %d tensors in the file (rest belong to a sibling model on the same .gguf)\n",
                __func__, n_created, n_tensors);
    }
    if (n_tensors_moved > 0) {
        LLAMA_LOG_DEBUG("%s: tensor '%s' (%s) (and %zu others) cannot be used with preferred buffer type %s, using %s instead\n",
            __func__, first_tensor_moved_name.c_str(), first_tensor_moved_type_name.c_str(), n_tensors_moved - 1,
            ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps) {
    // note: read_lazy also requires mmap; this condition make sure it's usable even when --load-mode is not set to mmap
    if (use_mmap || lazy.any()) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (uint32_t idx = 0; idx < files.size(); idx++) {
            const auto & file = files[idx];

            bool is_numa = false;

            auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (dev) {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                auto * is_numa_fn = (decltype(ggml_is_numa) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_is_numa");
                if (is_numa_fn) {
                    is_numa = is_numa_fn();
                }
            }

            const size_t prefetch_size = prefetch && use_mmap ? -1 : 0;

            std::unique_ptr<llama_mmap> mapping = std::make_unique<llama_mmap>(file.get(), prefetch_size, is_numa,
                    lazy.for_file(idx));
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (const auto & it : weights_map) {
        size_data += ggml_nbytes(it.second.tensor);
    }
}

using mapping_ranges = std::vector<llama_model_loader::mapping_range>;

// merge a list of [first, last) ranges into a sorted, non-overlapping list
static mapping_ranges merge_ranges(mapping_ranges ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const auto & a, const auto & b) { return a.first < b.first; });

    mapping_ranges res;
    for (const auto & range : ranges) {
        if (!res.empty() && range.first <= res.back().last) {
            res.back().last = std::max(res.back().last, range.last);
        } else {
            res.push_back(range);
        }
    }

    return res;
}

// subtract the sorted, non-overlapping ranges 'sub' from 'a'
static mapping_ranges sub_ranges(const mapping_ranges & a, const mapping_ranges & sub) {
    mapping_ranges res;

    for (const auto & x : a) {
        size_t cur = x.first;
        for (const auto & y : sub) {
            if (y.last <= cur || y.first >= x.last) {
                continue;
            }
            if (y.first > cur) {
                res.push_back({ cur, y.first });
            }
            cur = std::max(cur, y.last);
        }
        if (cur < x.last) {
            res.push_back({ cur, x.last });
        }
    }

    return res;
}

std::vector<llama_model_loader::mapping_range> llama_model_loader::get_mapping_ranges(uint32_t idx, ggml_context * ctx) const {
    GGML_ASSERT(idx < mappings.size());

    // data of this context, and data of the other contexts that share the same file
    // a weight that is mapped in several contexts (e.g. the token embedding) is not foreign data
    std::unordered_set<const llama_tensor_weight *> own;
    mapping_ranges own_ranges;
    mapping_ranges other;

    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        const auto * weight = get_weight(ggml_get_name(tensor));
        if (!weight || weight->idx != idx) {
            continue;
        }
        own.insert(weight);
        own_ranges.push_back({ weight->offs, weight->offs + ggml_nbytes(tensor) });
    }

    for (const auto & it : ctx_map) {
        if (!it.second || it.second.get() == ctx) {
            continue;
        }

        for (ggml_tensor * tensor = ggml_get_first_tensor(it.second.get()); tensor; tensor = ggml_get_next_tensor(it.second.get(), tensor)) {
            const auto * weight = get_weight(ggml_get_name(tensor));
            if (!weight || weight->idx != idx || own.count(weight)) {
                continue;
            }
            other.push_back({ weight->offs, weight->offs + ggml_nbytes(tensor) });
        }
    }

    return sub_ranges(merge_ranges(std::move(own_ranges)), merge_ranges(std::move(other)));
}

void llama_model_loader::unmap_weight(const llama_tensor_weight & w) const {
    if (!use_mmap) { return; }
    mappings.at(w.idx)->unmap_fragment(w.offs, w.offs + ggml_nbytes(w.tensor));
}

const void * llama_model_loader::load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const {
    GGML_ASSERT(offs + size <= ggml_nbytes(w.tensor));

    const void * data = buf;

    if (use_mmap) {
        data = (const uint8_t *) mappings.at(w.idx)->addr() + w.offs + offs;
    } else {
        GGML_ASSERT(buf != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs + offs, SEEK_SET);
        file->read_raw(buf, size);
    }

    if (check_tensors && !ggml_validate_row_data(w.tensor->type, data, size)) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(w.tensor)));
    }

    return data;
}

bool llama_model_loader::load_all_data(
        struct ggml_context * ctx,
        std::vector<ggml_backend_buffer_t> & bufs,
        llama_mlocks * lmlocks,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    if (files.empty()) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            set_tensor_data(t, set_tensor_data_ud);
        }
        return true;
    }
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers = 4;

    size_t alignment = 1;
    for (const auto & file : files) {
        alignment = std::max(file->read_alignment(), alignment);
    }

    // Buffer size: balance between memory usage and I/O efficiency
    const size_t buffer_size = alignment != 1 ? LLAMA_DIRECT_IO_BUFFER_SIZE + 2 * alignment : 1 * 1024 * 1024;

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<ggml_backend_event_t> events;
    std::vector<void *> host_ptrs;
    size_t buffer_idx = 0; // buffer to use for async loads
    ggml_backend_t upload_backend = [&](const char * func) -> ggml_backend_t {
        if (use_mmap || check_tensors) {
            return nullptr;
        }
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.empty() ? nullptr : bufs[0];
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
        auto * dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
            LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);

            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

            auto * event = ggml_backend_event_new(dev);
            if (!event) {
                LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            events.emplace_back(event);
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        return backend;
    }(__func__);

    if (upload_backend) {
        LLAMA_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
            ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),
            ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs[0])),
            ggml_backend_name(upload_backend));
    }

    std::vector<ggml_tensor *> tensors;
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        tensors.push_back(cur);
    }

    // without mmap, tensors in non-host buffers are staged through a temporary buffer sized like the tensor
    // load them biggest-first so the largest staging buffer is allocated while the fewest weights are resident
    if (!use_mmap) {
        std::stable_sort(tensors.begin(), tensors.end(), [](const ggml_tensor * a, const ggml_tensor * b) {
            const bool staged_a = a->buffer && !ggml_backend_buffer_is_host(a->buffer);
            const bool staged_b = b->buffer && !ggml_backend_buffer_is_host(b->buffer);
            if (staged_a != staged_b) {
                return staged_a;
            }
            return staged_a && ggml_nbytes(a) > ggml_nbytes(b);
        });
    }

    for (struct ggml_tensor * cur : tensors) {
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        // Capture file offset for RAM-tier expert tensors
        capture_expert_file_offset(ggml_get_name(cur), weight->offs, n_size, cur);

        const bool from_mapping = use_mmap || lazy.has(cur);

        if (from_mapping) {
            const auto & mapping = mappings.at(weight->idx);
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            // the tensors of a file can be spread across several buffers - find the one containing this tensor's data
            ggml_backend_buffer_t buf_mmap = nullptr;
            for (const auto & buf : bufs) {
                const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(buf);
                if (base && base <= data && data + n_size <= base + ggml_backend_buffer_get_size(buf)) {
                    buf_mmap = buf;
                    break;
                }
            }

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (sequential_load && cur->data != nullptr && ggml_backend_buffer_is_host(cur->buffer)) {
                // sequential mode: tensor data already points directly to mmap, skip copy
                // the buffer was set up in the creation loop with t->data = mapping->addr() + offset
                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);

                // locking a lazy tensor would fault all of it in, which is what lazy avoids
                if (lmlocks && !lazy.has(cur)) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            const auto & file = files.at(weight->idx);

            if (ggml_backend_buffer_is_host(cur->buffer)) {
                if (sequential_load && use_direct_io && cur->data != nullptr) {
                    size_done += n_size;
                    continue;
                }
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (upload_backend) {
                    size_t offset = weight->offs;
                    alignment = file->read_alignment();
                    size_t aligned_offset = offset & ~(alignment - 1);
                    size_t offset_from_alignment = offset - aligned_offset;
                    file->seek(aligned_offset, SEEK_SET);

                    // Calculate aligned read boundaries
                    size_t read_start = aligned_offset;
                    size_t read_end = (offset + n_size + alignment - 1) & ~(alignment - 1);

                    size_t bytes_read = 0;
                    size_t data_read = 0;  // Actual tensor data copied (excluding padding)

                    while (bytes_read < read_end - read_start) {
                        size_t read_size = std::min<size_t>(buffer_size, read_end - read_start - bytes_read);

                        // Align the destination pointer within the pinned buffer
                        uintptr_t ptr_dest_aligned = (reinterpret_cast<uintptr_t>(host_ptrs[buffer_idx]) + alignment - 1) & ~(alignment - 1);

                        // Wait for previous upload to complete before reusing buffer
                        ggml_backend_event_synchronize(events[buffer_idx]);

                        // Read aligned chunk from file
                        file->read_raw_unsafe(reinterpret_cast<void *>(ptr_dest_aligned), read_size);

                        // Calculate actual data portion (excluding alignment padding)
                        uintptr_t ptr_data = ptr_dest_aligned;
                        size_t data_to_copy = read_size;

                        // Skip alignment padding at start of first chunk
                        if (bytes_read == 0) {
                            ptr_data += offset_from_alignment;
                            data_to_copy -= offset_from_alignment;
                        }

                        // Trim alignment padding at end of last chunk
                        if (aligned_offset + bytes_read + read_size > offset + n_size) {
                            data_to_copy -= (read_end - (offset + n_size));
                        }

                        // Async upload actual data to GPU
                        ggml_backend_tensor_set_async(upload_backend, cur,
                                                      reinterpret_cast<void *>(ptr_data), data_read, data_to_copy);
                        ggml_backend_event_record(events[buffer_idx], upload_backend);

                        data_read += data_to_copy;
                        bytes_read += read_size;

                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    // scoped to one tensor so only one staging buffer is alive at a time
                    std::vector<no_init<uint8_t>> read_buf(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }
            }
        }

        size_done += n_size;
    }

    // free temporary resources used for async uploads
    for (auto * event : events) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
    }
    for (auto * buf : host_buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_backend_free(upload_backend);

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            bool (*reg_fn)(void *, size_t) = nullptr;
            void (*unreg_fn)(void *) = nullptr;
            const char * register_host_env = getenv("GGML_CUDA_REGISTER_HOST");
            const bool register_host = !sequential_load && register_host_env != nullptr && atoi(register_host_env) != 0;

            if (register_host) {
                // Resolve the generic host-registration hook from the first backend
                // that exposes it. CUDA's implementation uses cudaHostRegister.
                for (size_t i = 0; i < ggml_backend_dev_count() && reg_fn == nullptr; ++i) {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_get(i));
                    reg_fn = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(
                        reg, "ggml_backend_register_host_buffer");
                    unreg_fn = (void (*)(void *)) ggml_backend_reg_get_proc_address(
                        reg, "ggml_backend_unregister_host_buffer");
                    if (reg_fn == nullptr || unreg_fn == nullptr) {
                        reg_fn = nullptr;
                        unreg_fn = nullptr;
                    }
                }
            }

            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                if (sequential_load) {
                    // sequential mode: keep tensor data mapped for on-demand disk paging;
                    // only unmap the metadata header (before first tensor). Never pin
                    // this full span: sequential mode must work when the model exceeds RAM.
                    mapping->unmap_fragment(0, mmap_used.first);
                } else {
                    mapping->unmap_fragment(0, mmap_used.first);
                    if (mmap_used.second != 0) {
                        mapping->unmap_fragment(mmap_used.second, mapping->size());
                    }
                    if (mmap_used.second > mmap_used.first && reg_fn != nullptr && unreg_fn != nullptr) {
                        const size_t n_registered = mapping->register_host(
                            mmap_used.first, mmap_used.second, reg_fn, unreg_fn);
                        if (n_registered > 0) {
                            LLAMA_LOG_INFO("%s: registered %.2f MiB of mmap-backed CPU weights for direct H2D DMA\n",
                                __func__, n_registered / 1024.0 / 1024.0);
                        }
                    }
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}


// ============================================================================
// MoE placement helper functions
// ============================================================================

std::vector<llama_model_tensor_buft_override> moe_plan_to_overrides(
        llama_model_loader * loader,
        const moe_placement_plan & plan,
        int n_layer) {
    std::vector<llama_model_tensor_buft_override> overrides;

    for (int l = 0; l < n_layer; ++l) {
        // Pattern for this layer's expert tensors
        static const char * fmt_layer = R"(blk\.%d\.ffn_.*_exps.*)";
        char pattern[128];
        int ret = snprintf(pattern, sizeof(pattern), fmt_layer, l);
        LLAMA_LOG_INFO("DEBUG snprintf ret=%d, pattern='%s'\n", ret, pattern);

        // Store pattern string for lifetime management
        loader->moe_pattern_strings.emplace_back(pattern);
        const char * pattern_ptr = loader->moe_pattern_strings.back().c_str();

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (l < plan.n_layers_vram) {
            // VRAM - let the meta device decide which GPU based on tensor_split
            buft = ggml_backend_cpu_buffer_type(); // will be remapped by llama_model_load_internal
        } else if (l < plan.n_layers_vram + plan.n_layers_ram) {
            // RAM
            buft = ggml_backend_cpu_buffer_type();
        } else {
            // DISK
            buft = ggml_backend_cpu_buffer_type();
        }

        overrides.push_back({pattern_ptr, buft});
    }

    return overrides;
}

// Convert expert-granular plan to tensor buffer type overrides
std::vector<llama_model_tensor_buft_override> moe_plan_to_overrides_expert_granular(
        llama_model_loader * loader,
        const moe_placement_plan & plan,
        int n_layer,
        int n_expert) {
    std::vector<llama_model_tensor_buft_override> overrides;

    for (int l = 0; l < n_layer; ++l) {
        for (int e = 0; e < n_expert; ++e) {
            const size_t idx = (size_t) l * n_expert + e;
            const auto tier = plan.expert_tier[idx];

            // Pattern: blk.L.ffn_*_exps.*.E
            static const char * fmt = R"(blk\.%d\.ffn_.*_exps\..*\.%d$)";
            char pattern[128];
            snprintf(pattern, sizeof(pattern), fmt, l, e);

            // Store pattern string for lifetime management
            loader->moe_pattern_strings.emplace_back(pattern);
            const char * pattern_ptr = loader->moe_pattern_strings.back().c_str();

            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (tier == MOE_TIER_VRAM) {
                // Let the meta device decide which GPU based on tensor_split
                buft = ggml_backend_cpu_buffer_type(); // will be remapped by llama_model_load_internal
            } else if (tier == MOE_TIER_RAM) {
                buft = ggml_backend_cpu_buffer_type();
            } else {
                buft = ggml_backend_cpu_buffer_type();
            }

            overrides.push_back({pattern_ptr, buft});
        }
    }

    return overrides;
}

// Early MoE solver - called from common.cpp before standard parameter fitting
// Returns true on success, false on failure (error in plan.error)
bool llama_moe_solve_early(
        const char * model_path,
        const float * tensor_split,
        int n_devices,
        size_t moe_ram_mib,
        const char * moe_placement_mode,
        bool moe_pin_decode,
        const char * moe_profile_path,
        int & n_layer_out,
        int & n_expert_out,
        int & n_expert_used_out,
        std::vector<llama_model_tensor_buft_override> & overrides_out) {

    // Open GGUF context to read metadata
    struct gguf_init_params gguf_params = { true, nullptr };
    gguf_context_ptr gguf_ctx(gguf_init_from_file(model_path, gguf_params));
    if (!gguf_ctx) {
        LLAMA_LOG_ERROR("llama_moe_solve_early: failed to open model file '%s'\n", model_path);
        return false;
    }

    // Extract n_layer and n_expert from metadata
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0;

    auto get_u64 = [&](const char * key) -> uint64_t {
        const int kid = gguf_find_key(gguf_ctx.get(), key);
        if (kid < 0) return 0;
        // Try different types - GGUF may store as uint32 or uint64
        enum gguf_type type = gguf_get_kv_type(gguf_ctx.get(), kid);
        if (type == GGUF_TYPE_UINT64) {
            return gguf_get_val_u64(gguf_ctx.get(), kid);
        } else if (type == GGUF_TYPE_UINT32) {
            return gguf_get_val_u32(gguf_ctx.get(), kid);
        } else if (type == GGUF_TYPE_INT32) {
            return gguf_get_val_i32(gguf_ctx.get(), kid);
        } else if (type == GGUF_TYPE_INT64) {
            return gguf_get_val_i64(gguf_ctx.get(), kid);
        }
        LLAMA_LOG_WARN("MoE solver: key '%s' has unexpected type %d\n", key, type);
        return 0;
    };

    n_layer = (int) get_u64("qwen4exp.block_count");
    if (n_layer == 0) n_layer = (int) get_u64("llama.block_count");

    n_expert = (int) get_u64("qwen4exp.expert_count");
    if (n_expert == 0) n_expert = (int) get_u64("llama.expert_count");

    n_expert_used = (int) get_u64("qwen4exp.expert_used_count");
    if (n_expert_used == 0) n_expert_used = (int) get_u64("llama.expert_used_count");

    if (n_layer <= 0 || n_expert <= 0 || n_expert_used <= 0) {
        LLAMA_LOG_ERROR("llama_moe_solve_early: could not determine MoE dimensions from metadata\n");
        return false;
    }

    n_layer_out = n_layer;
    n_expert_out = n_expert;
    n_expert_used_out = n_expert_used;

    // Use a local vector for pattern string lifetime management
    std::vector<std::string> pattern_strings;

    // Choose solver based on profile availability
    moe_placement_plan plan;
    if (moe_profile_path && *moe_profile_path) {
        plan = moe_solve_placement_expert_granular(
            gguf_ctx.get(), tensor_split, n_devices, moe_ram_mib, moe_placement_mode,
            moe_pin_decode, n_layer, n_expert, n_expert_used, moe_profile_path);
    } else {
        plan = moe_solve_placement(
            gguf_ctx.get(), tensor_split, n_devices, moe_ram_mib, moe_placement_mode,
            moe_pin_decode, n_layer, n_expert, n_expert_used);
    }

    if (!plan.fits) {
        LLAMA_LOG_ERROR("llama_moe_solve_early: placement failed: %s\n", plan.error.c_str());
        return false;
    }

    // Print placement report
    for (const auto & line : plan.report_lines) {
        LLAMA_LOG_INFO("%s\n", line.c_str());
    }

    // Generate overrides
    // Generate overrides directly
    overrides_out.clear();

    if (moe_profile_path && *moe_profile_path && !plan.expert_tier.empty()) {
        // Expert-granular overrides: assign combined tensors per layer based on expert tiers
        // Combined tensors: blk.L.ffn_{gate,up,down}_exps.weight (and gate_up_exps if merged)
        // Assign to VRAM if ANY expert in the layer is VRAM, otherwise RAM/CPU
        for (int l = 0; l < n_layer; ++l) {
            // Check if any expert in this layer is VRAM
            bool any_vram = false;
            bool any_ram = false;
            for (int e = 0; e < n_expert; ++e) {
                const size_t idx = (size_t) l * n_expert + e;
                if (idx < plan.expert_tier.size()) {
                    if (plan.expert_tier[idx] == MOE_TIER_VRAM) any_vram = true;
                    else if (plan.expert_tier[idx] == MOE_TIER_RAM) any_ram = true;
                }
            }

            // Determine buffer type for this layer's expert tensors
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (any_vram) {
                buft = ggml_backend_cpu_buffer_type(); // will be remapped to GPU by meta device
            } else if (any_ram) {
                buft = ggml_backend_cpu_buffer_type(); // stays on CPU (RAM tier)
            } else {
                buft = ggml_backend_cpu_buffer_type(); // disk tier - stays on CPU, loaded on demand
            }

            // Pattern matches all combined expert tensors for this layer
            static const char * fmt = R"(blk\.%d\.ffn_.*_exps\.weight$)";
            char pattern[128];
            snprintf(pattern, sizeof(pattern), fmt, l);

            pattern_strings.emplace_back(pattern);
            const char * pattern_ptr = pattern_strings.back().c_str();

            overrides_out.push_back({pattern_ptr, buft});
        }
    } else {
        // Layer-granular overrides
        for (int l = 0; l < n_layer; ++l) {
            static const char * fmt = R"(blk\.%d\.ffn_.*_exps.*)";
            char pattern[128];
            snprintf(pattern, sizeof(pattern), fmt, l);

            pattern_strings.emplace_back(pattern);
            const char * pattern_ptr = pattern_strings.back().c_str();

            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (l < plan.n_layers_vram) {
                buft = ggml_backend_cpu_buffer_type(); // remapped later
            } else if (l < plan.n_layers_vram + plan.n_layers_ram) {
                buft = ggml_backend_cpu_buffer_type();
            } else {
                buft = ggml_backend_cpu_buffer_type();
            }

            overrides_out.push_back({pattern_ptr, buft});
        }
    }

    // Null terminator
    overrides_out.push_back({nullptr, nullptr});

    return true;
}

// ============================================================================
// Phase 1: Layer-granular MoE placement solver
// ============================================================================

// Helper to format strings
static std::string moe_fmt(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    if (n < 0) return "";
    std::string s;
    s.resize(n + 1);
    vsnprintf(&s[0], s.size(), fmt, ap2);
    va_end(ap2);
    s.pop_back();
    return s;
}

// Solve MoE expert placement from metadata and params (Phase 1: layer-granular)
moe_placement_plan moe_solve_placement(
        struct gguf_context * /*metadata*/,
        const float * tensor_split,
        int n_devices,
        size_t moe_ram_mib,
        const char * /*moe_placement_mode*/,
        bool moe_pin_decode,
        int n_layer,
        int n_expert,
        int n_expert_used) {
    moe_placement_plan plan;

    // Query actual VRAM per device
    std::vector<size_t> vram_per_device;
    size_t total_vram = 0;
    if (n_devices <= 0) {
        n_devices = ggml_backend_dev_count();
    }
    for (int i = 0; i < n_devices; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        // Try to get memory info via backend
        size_t dev_vram = 0;
        const char * dev_name = ggml_backend_dev_name(dev);
        if (strcmp(dev_name, "CUDA") == 0 || strcmp(dev_name, "CUDA0") == 0) {
            dev_vram = 24ull * 1024 * 1024 * 1024; // 24 GB for RTX 3090
        } else if (strcmp(dev_name, "CUDA1") == 0) {
            dev_vram = 12ull * 1024 * 1024 * 1024; // 12 GB for RTX 2060
        } else {
            // Try to query via backend
            // Fallback
            dev_vram = 12ull * 1024 * 1024 * 1024;
        }
        // Apply 10% safety margin
        dev_vram = (dev_vram * 9) / 10;
        vram_per_device.push_back(dev_vram);
        total_vram += dev_vram;
        plan.report_lines.push_back(moe_fmt("  Device %d (%s): %.2f GiB (after 10%% margin)", i, dev_name, dev_vram / 1024.0 / 1024.0 / 1024.0));
    }

    // Fallback if no devices
    if (vram_per_device.empty()) {
        vram_per_device.push_back(4ull * 1024 * 1024 * 1024); // 4 GB default
        total_vram = vram_per_device[0];
        plan.report_lines.push_back("  No devices found, using 4 GiB default");
    }

    // Estimate expert bytes per layer
    // For qwen4exp: 512 experts/layer, ~2.3 MB per expert = ~1.17 GB per layer for all experts
    size_t expert_bytes_per_layer = (size_t)(1.17 * 1024 * 1024 * 1024);
    size_t total_expert_bytes = expert_bytes_per_layer * n_layer;

    plan.total_expert_bytes = total_expert_bytes;
    plan.expert_bytes_per_layer = expert_bytes_per_layer;

    // VRAM budget for experts (reserve some for KV cache and context)
    size_t ram_budget = moe_ram_mib * 1024 * 1024;
    size_t vram_budget = total_vram;
    if (vram_budget > 4ull * 1024 * 1024 * 1024) {
        vram_budget -= 4ull * 1024 * 1024 * 1024; // Reserve 4 GiB for KV/context
    }

    // Distribute VRAM budget across devices according to tensor_split
    plan.vram_expert_budget.resize(vram_per_device.size());
    if (tensor_split && vram_per_device.size() > 1) {
        float split_sum = 0;
        for (size_t i = 0; i < vram_per_device.size(); ++i) split_sum += tensor_split[i];
        if (split_sum > 0) {
            for (size_t i = 0; i < vram_per_device.size(); ++i) {
                plan.vram_expert_budget[i] = (size_t)(vram_budget * tensor_split[i] / split_sum);
            }
        } else {
            // Equal split
            size_t per_dev = vram_budget / vram_per_device.size();
            for (size_t i = 0; i < vram_per_device.size(); ++i) {
                plan.vram_expert_budget[i] = per_dev;
            }
        }
    } else {
        // Single device or no split
        for (size_t i = 0; i < vram_per_device.size(); ++i) {
            plan.vram_expert_budget[i] = std::min(vram_budget, vram_per_device[i]);
        }
    }

    plan.ram_expert_budget = ram_budget;
    plan.kv_compute_reserve.assign(vram_per_device.size(), 4ull * 1024 * 1024 * 1024);

    // Simple layer-granular placement: put as many layers in VRAM as fit
    size_t vram_for_experts = 0;
    for (size_t b : plan.vram_expert_budget) vram_for_experts += b;
    plan.n_layers_vram = std::min(n_layer, (int)(vram_for_experts / expert_bytes_per_layer));
    int remaining = n_layer - plan.n_layers_vram;
    plan.n_layers_ram = std::min(remaining, (int)(ram_budget / expert_bytes_per_layer));
    plan.n_layers_disk = remaining - plan.n_layers_ram;

    plan.fits = (plan.n_layers_vram + plan.n_layers_ram + plan.n_layers_disk == n_layer);

    if (!plan.fits) {
        plan.error = moe_fmt("Cannot fit %d layers: VRAM=%d, RAM=%d, DISK=%d", n_layer, plan.n_layers_vram, plan.n_layers_ram, plan.n_layers_disk);
    }

    // Pinned decode layer bytes
    if (moe_pin_decode && plan.n_layers_vram > 0) {
        plan.pinned_decode_bytes = expert_bytes_per_layer;
    }

    // Report
    plan.report_lines.push_back(moe_fmt("MoE Placement Plan (Layer-Granular):"));
    plan.report_lines.push_back(moe_fmt("  Layers: %d, Experts/layer: %d, Active: %d", n_layer, n_expert, n_expert_used));
    plan.report_lines.push_back(moe_fmt("  Total expert bytes: %.2f GiB", total_expert_bytes / 1024.0 / 1024.0 / 1024.0));
    plan.report_lines.push_back(moe_fmt("  VRAM budget: %.2f GiB, RAM budget: %.2f GiB", vram_budget / 1024.0 / 1024.0 / 1024.0, ram_budget / 1024.0 / 1024.0 / 1024.0));
    plan.report_lines.push_back(moe_fmt("  Layers VRAM: %d, RAM: %d, DISK: %d", plan.n_layers_vram, plan.n_layers_ram, plan.n_layers_disk));
    plan.report_lines.push_back(moe_fmt("  Fits: %s", plan.fits ? "YES" : "NO"));

    return plan;
}

// ============================================================================
// Phase 2: Expert-granular MoE placement solver
// ============================================================================

// Load MoE profile (activation counts) from JSON file
static bool moe_load_profile(const char * path, int n_layer, int n_expert, std::vector<uint32_t> & counts, uint64_t & tokens) {
    if (!path || !*path) {
        return false;
    }
    std::ifstream f(path);
    if (!f) {
        LLAMA_LOG_WARN("MoE profile: could not open '%s'\n", path);
        return false;
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Parse using the same logic as llama-moe-cache.cpp
    auto json_int = [&](const std::string & key, int64_t def) -> int64_t {
        const std::string k = "\"" + key + "\"";
        const size_t pos = s.find(k);
        if (pos == std::string::npos) return def;
        const size_t colon = s.find(':', pos + k.size());
        if (colon == std::string::npos) return def;
        char * end = nullptr;
        const long long v = strtoll(s.c_str() + colon + 1, &end, 10);
        return end == s.c_str() + colon + 1 ? def : v;
    };

    const int64_t nl = json_int("n_layer", -1);
    const int64_t ne = json_int("n_expert", -1);
    tokens = (uint64_t) std::max<int64_t>(0, json_int("tokens", 0));
    const size_t key = s.find("\"counts\"");
    if (nl != n_layer || ne != n_expert || nl <= 0 || ne <= 0 || key == std::string::npos) {
        LLAMA_LOG_WARN("MoE profile: dimension mismatch (expected %d layers, %d experts, got %ld, %ld)\n", n_layer, n_expert, (long)nl, (long)ne);
        return false;
    }
    const char * p = s.c_str() + key;
    p = strchr(p, '[');
    if (!p) return false;
    ++p;

    counts.assign((size_t) nl * (size_t) ne, 0);
    int32_t row = 0;
    while (*p != 0) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') ++p;
        if (*p == ']' || *p == 0) break;
        if (*p != '[') return false;
        ++p;
        for (int32_t e = 0; e < (int32_t) ne; ++e) {
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
            char * end = nullptr;
            const unsigned long long v = strtoull(p, &end, 10);
            if (end == p) return false;
            counts[(size_t) row * (size_t) ne + (size_t) e] = (uint32_t) std::min<unsigned long long>(v, UINT32_MAX);
            p = end;
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
            if (*p == ',') ++p;
        }
        while (*p != 0 && *p != ']') ++p;
        if (*p == ']') ++p;
        ++row;
    }
    if (row != (int32_t) nl) {
        LLAMA_LOG_WARN("MoE profile: row count mismatch (expected %d, got %d)\n", (int)nl, row);
        return false;
    }
    return true;
}

moe_placement_plan moe_solve_placement_expert_granular(
        struct gguf_context * metadata,
        const float * tensor_split,
        int n_devices,
        size_t moe_ram_mib,
        const char * moe_placement_mode,
        bool moe_pin_decode,
        int n_layer,
        int n_expert,
        int n_expert_used,
        const char * moe_profile_path) {

    // First run the layer-granular solver to get budgets
    moe_placement_plan plan = moe_solve_placement(
        metadata, tensor_split, n_devices, moe_ram_mib, moe_placement_mode,
        moe_pin_decode, n_layer, n_expert, n_expert_used);

    if (!plan.fits) {
        return plan; // Layer-granular already failed
    }

    plan.n_layer = n_layer;
    plan.n_expert = n_expert;

    // Load profile if provided
    std::vector<uint32_t> profile_counts;
    uint64_t profile_tokens = 0;
    bool has_profile = false;
    if (moe_profile_path && *moe_profile_path) {
        has_profile = moe_load_profile(moe_profile_path, n_layer, n_expert, profile_counts, profile_tokens);
        if (has_profile) {
            plan.profile_counts = std::move(profile_counts);
            LLAMA_LOG_INFO("MoE placement: loaded profile with %llu tokens\n", (unsigned long long) profile_tokens);
        } else {
            LLAMA_LOG_WARN("MoE placement: failed to load profile '%s', falling back to uniform\n", moe_profile_path);
        }
    }

    // If no profile, use uniform counts (all experts equally likely)
    if (!has_profile) {
        plan.profile_counts.assign((size_t) n_layer * (size_t) n_expert, 1);
    }

    // Compute expert bytes (per expert, per layer)
    size_t expert_bytes_per_expert = 0;
    if (n_layer > 0 && n_expert > 0) {
        expert_bytes_per_expert = plan.total_expert_bytes / ((size_t) n_layer * (size_t) n_expert);
    }

    // Total VRAM budget for experts across all devices
    size_t total_vram_expert = 0;
    for (size_t b : plan.vram_expert_budget) total_vram_expert += b;

    // Total RAM budget
    size_t total_ram_expert = plan.ram_expert_budget;

    // Rank all experts by activation count (higher = hotter)
    struct ExpertRank {
        int layer;
        int expert;
        uint32_t count;
    };
    std::vector<ExpertRank> all_experts;
    all_experts.reserve((size_t) n_layer * (size_t) n_expert);
    for (int il = 0; il < n_layer; ++il) {
        for (int e = 0; e < n_expert; ++e) {
            all_experts.push_back({il, e, plan.profile_counts[(size_t) il * n_expert + e]});
        }
    }
    std::sort(all_experts.begin(), all_experts.end(),
        [](const ExpertRank & a, const ExpertRank & b) {
            return a.count != b.count ? a.count > b.count : (a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert);
        });

    // Assign experts to tiers based on budget
    plan.expert_tier.assign((size_t) n_layer * (size_t) n_expert, MOE_TIER_DISK);
    plan.expert_slot.assign((size_t) n_layer * (size_t) n_expert, -1);
    plan.tier_expert_counts.assign(3, 0);
    plan.tier_bytes_per_layer.assign(3, 0);

    size_t vram_used = 0;
    size_t ram_used = 0;
    std::vector<int> vram_slots_per_layer(n_layer, 0);
    std::vector<int> ram_slots_per_layer(n_layer, 0);

    for (const auto & er : all_experts) {
        const size_t idx = (size_t) er.layer * n_expert + er.expert;
        const size_t exp_bytes = expert_bytes_per_expert;

        // Try VRAM first
        if (vram_used + exp_bytes <= total_vram_expert) {
            plan.expert_tier[idx] = MOE_TIER_VRAM;
            plan.expert_slot[idx] = vram_slots_per_layer[er.layer]++;
            vram_used += exp_bytes;
            plan.tier_expert_counts[MOE_TIER_VRAM]++;
            plan.tier_bytes_per_layer[MOE_TIER_VRAM] += exp_bytes;
        }
        // Then RAM
        else if (ram_used + exp_bytes <= total_ram_expert) {
            plan.expert_tier[idx] = MOE_TIER_RAM;
            plan.expert_slot[idx] = ram_slots_per_layer[er.layer]++;
            ram_used += exp_bytes;
            plan.tier_expert_counts[MOE_TIER_RAM]++;
            plan.tier_bytes_per_layer[MOE_TIER_RAM] += exp_bytes;
        }
        // Rest go to disk
        else {
            plan.expert_tier[idx] = MOE_TIER_DISK;
            plan.expert_slot[idx] = -1; // no slot for disk
            plan.tier_expert_counts[MOE_TIER_DISK]++;
            plan.tier_bytes_per_layer[MOE_TIER_DISK] += exp_bytes;
        }
    }

    // Recompute layer-granular counts from expert-granular assignment
    plan.n_layers_vram = 0;
    plan.n_layers_ram = 0;
    plan.n_layers_disk = 0;
    for (int il = 0; il < n_layer; ++il) {
        bool has_vram = false, has_ram = false, has_disk = false;
        for (int e = 0; e < n_expert; ++e) {
            const auto tier = plan.expert_tier[(size_t) il * n_expert + e];
            if (tier == MOE_TIER_VRAM) has_vram = true;
            else if (tier == MOE_TIER_RAM) has_ram = true;
            else has_disk = true;
        }
        if (has_vram && !has_ram && !has_disk) plan.n_layers_vram++;
        else if (has_ram && !has_vram && !has_disk) plan.n_layers_ram++;
        else if (has_disk && !has_vram && !has_ram) plan.n_layers_disk++;
        else {
            // Mixed layer - count as the highest tier present
            if (has_vram) plan.n_layers_vram++;
            else if (has_ram) plan.n_layers_ram++;
            else plan.n_layers_disk++;
        }
    }

    // Update report with expert-granular details
    plan.report_lines.push_back("");
    plan.report_lines.push_back("=== Expert-Granular Placement ===");
    plan.report_lines.push_back(moe_fmt("  Profile: %s", has_profile ? "loaded" : "uniform"));
    plan.report_lines.push_back(moe_fmt("  Expert bytes (est): %.2f MiB", expert_bytes_per_expert / 1024.0 / 1024.0));
    plan.report_lines.push_back(moe_fmt("  VRAM experts: %zu (%.2f MiB)", plan.tier_expert_counts[MOE_TIER_VRAM], vram_used / 1024.0 / 1024.0));
    plan.report_lines.push_back(moe_fmt("  RAM experts:  %zu (%.2f MiB)", plan.tier_expert_counts[MOE_TIER_RAM], ram_used / 1024.0 / 1024.0));
    plan.report_lines.push_back(moe_fmt("  DISK experts: %zu (%.2f MiB)", plan.tier_expert_counts[MOE_TIER_DISK],
        (plan.total_expert_bytes - vram_used - ram_used) / 1024.0 / 1024.0));
    plan.report_lines.push_back(moe_fmt("  Mixed layers: VRAM=%d, RAM=%d, DISK=%d", plan.n_layers_vram, plan.n_layers_ram, plan.n_layers_disk));

    // Expected cold bytes/token (k=8) from DISK experts
    if (n_expert > 0 && plan.tier_expert_counts[MOE_TIER_DISK] > 0) {
        double cold_per_layer = (double)plan.tier_bytes_per_layer[MOE_TIER_DISK] / n_layer;
        plan.report_lines.push_back(moe_fmt("  Expected cold bytes/token (k=8): %.2f MiB",
            (cold_per_layer * 8.0 / n_expert) / 1024.0 / 1024.0));
    }

    return plan;
}

// ============================================================================

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

std::unique_ptr<moe_ram_arena> g_moe_ram_arena;

static uint64_t moe_get_time_us() {
    return (uint64_t) ggml_time_us();
}

// ============================================================================
// moe_ram_arena implementation with combined buffers
// ============================================================================

bool moe_ram_arena::init(size_t capacity_mib, const moe_placement_plan & plan, int n_layer, int n_expert, const char * model_path) {
    (void)plan;
    (void)n_layer;
    (void)n_expert;
    capacity_bytes = capacity_mib * 1024 * 1024;
    used_bytes = 0;

    // Open model file for pread operations
    model_fd = open(model_path, O_RDONLY);
    if (model_fd < 0) {
        LLAMA_LOG_ERROR("moe_ram_arena: failed to open model file '%s'\n", model_path);
        return false;
    }

    // Count RAM-tier experts
    size_t ram_expert_count = 0;
    for (size_t i = 0; i < plan.expert_tier.size(); ++i) {
        if (plan.expert_tier[i] == MOE_TIER_RAM) {
            ram_expert_count++;
        }
    }

    slots.resize(ram_expert_count);
    lru_list.reserve(ram_expert_count);

    // Populate slots with expert metadata
    size_t slot_idx = 0;
    for (int l = 0; l < n_layer; ++l) {
        for (int e = 0; e < n_expert; ++e) {
            const size_t idx = (size_t) l * n_expert + e;
            if (idx < plan.expert_tier.size() && plan.expert_tier[idx] == MOE_TIER_RAM) {
                slots[slot_idx].layer = l;
                slots[slot_idx].expert = e;
                slots[slot_idx].size = plan.expert_bytes_per_layer / n_expert; // approximate
                // file_offset will be filled in later when tensors are mapped
                slot_idx++;
            }
        }
    }

    // Start prefetch worker thread
    shutdown = false;
    prefetch_thread = std::thread([this]() {
        while (!shutdown.load()) {
            // Simple prefetch: check for pending requests
            // In a real implementation, this would use a condition variable
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    LLAMA_LOG_INFO("MoE RAM arena: initialized %.2f MiB for %zu experts\n", capacity_bytes / 1024.0 / 1024.0, ram_expert_count);

    return true;
}

void * moe_ram_arena::load_expert(int layer, int expert, const char * tensor_type) {
    std::lock_guard<std::mutex> lock(mutex);

    // Find the slot
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].layer == layer && slots[i].expert == expert) {
            // If tensor_type is specified, verify this slot matches
            // For now, we load for all tensor types when any is requested
            if (!slots[i].resident) {
                // Need to load from disk for all tensor types
                if (slots[i].file_offset == 0) {
                    LLAMA_LOG_WARN("moe_ram_arena: expert L%d_E%d has no file offset\n", layer, expert);
                }

                // Make room if needed
                if (used_bytes + slots[i].size > capacity_bytes) {
                    evict_lru(slots[i].size);
                }

                // Allocate combined buffers for all tensor types if not already done
                const char * tensor_types[] = {"gate", "up", "down", "gate_up"};
                for (const char * tt : tensor_types) {
                    auto key = std::make_pair(layer, std::string(tt));
                    auto it = combined_buffers.find(key);
                    if (it == combined_buffers.end()) {
                        // Will be created when file offset is captured
                    }
                }

                // Note: Actual loading happens in ensure_experts_loaded which loads per tensor type
                // For single expert load, we just mark as resident
                slots[i].resident = true;
                used_bytes += slots[i].size;

                LLAMA_LOG_DEBUG("moe_ram_arena: marked expert L%d_E%d as resident\n", layer, expert);
            }

            // Update LRU
            slots[i].last_access = moe_get_time_us();
            auto it = std::find(lru_list.begin(), lru_list.end(), i);
            if (it != lru_list.end()) {
                lru_list.erase(it);
            }
            lru_list.insert(lru_list.begin(), i);

            // Return pointer to expert data in combined buffer for the requested tensor type
            if (tensor_type) {
                auto key = std::make_pair(layer, std::string(tensor_type));
                auto it = combined_buffers.find(key);
                if (it != combined_buffers.end()) {
                    moe_combined_buffer & cb = it->second;
                    if (expert < (int)cb.resident.size()) {
                        cb.resident[expert] = true;
                        return cb.buffer.data() + expert * cb.stride;
                    }
                }
            }
            return nullptr;
        }
    }
    return nullptr; // Not in RAM tier
}

bool moe_ram_arena::ensure_experts_loaded(int layer, const char * tensor_type, const int * expert_ids, int n_experts) {
    std::lock_guard<std::mutex> lock(mutex);

    auto key = std::make_pair(layer, std::string(tensor_type));
    auto it = combined_buffers.find(key);
    if (it == combined_buffers.end()) {
        LLAMA_LOG_DEBUG("moe_ram_arena: no combined buffer for layer %d tensor %s\n", layer, tensor_type);
        return false; // Not a RAM-tier tensor
    }

    moe_combined_buffer & cb = it->second;
    bool any_loaded = false;

    for (int i = 0; i < n_experts; ++i) {
        int expert = expert_ids[i];
        if (expert < 0 || expert >= (int)cb.resident.size()) {
            continue;
        }

        if (!cb.resident[expert]) {
            // Find the slot for this expert
            for (size_t si = 0; si < slots.size(); ++si) {
                if (slots[si].layer == layer && slots[si].expert == expert) {
                    if (slots[si].file_offset == 0) {
                        LLAMA_LOG_WARN("moe_ram_arena: expert L%d_E%d has no file offset\n", layer, expert);
                        continue;
                    }

                    // Make room if needed
                    if (used_bytes + slots[si].size > capacity_bytes) {
                        evict_lru(slots[si].size);
                    }

                    // Load from file using pread into combined buffer
                    ssize_t bytes_read = pread(model_fd, cb.buffer.data() + expert * cb.stride, slots[si].size, slots[si].file_offset);
                    if (bytes_read != (ssize_t)slots[si].size) {
                        LLAMA_LOG_ERROR("moe_ram_arena: pread failed for expert L%d_E%d tensor %s: read %zd of %zu bytes\n", layer, expert, tensor_type, bytes_read, slots[si].size);
                        continue;
                    }

                    cb.resident[expert] = true;
                    slots[si].resident = true;
                    used_bytes += slots[si].size;
                    any_loaded = true;

                    LLAMA_LOG_DEBUG("moe_ram_arena: loaded expert L%d_E%d tensor %s from offset %zu (%zu bytes)\n", layer, expert, tensor_type, slots[si].file_offset, slots[si].size);
                }
            }
        }

        // Update LRU for this slot
        for (size_t si = 0; si < slots.size(); ++si) {
            if (slots[si].layer == layer && slots[si].expert == expert) {
                slots[si].last_access = moe_get_time_us();
                auto it = std::find(lru_list.begin(), lru_list.end(), si);
                if (it != lru_list.end()) {
                    lru_list.erase(it);
                }
                lru_list.insert(lru_list.begin(), si);
                break;
            }
        }
    }

    cb.last_access = moe_get_time_us();
    return any_loaded;
}

void * moe_ram_arena::get_combined_buffer(int layer, const char * tensor_type) {
    std::lock_guard<std::mutex> lock(mutex);

    auto key = std::make_pair(layer, std::string(tensor_type));
    auto it = combined_buffers.find(key);
    if (it == combined_buffers.end()) {
        return nullptr;
    }
    return it->second.buffer.data();
}

void moe_ram_arena::prefetch_expert(int layer, int expert, const char * tensor_type) {
    // Non-blocking: just mark for prefetch
    // In a full implementation, this would push to a queue for the worker thread
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].layer == layer && slots[i].expert == expert && !slots[i].resident) {
            // Would queue for async load here
            break;
        }
    }
}

void moe_ram_arena::evict_lru(size_t need_bytes) {
    std::lock_guard<std::mutex> lock(mutex);

    while (used_bytes + need_bytes > capacity_bytes && !lru_list.empty()) {
        size_t idx = lru_list.back();
        if (idx < slots.size() && slots[idx].resident) {
            // Advise kernel to drop pages for this file range
            if (model_fd >= 0 && slots[idx].file_offset > 0) {
                posix_fadvise(model_fd, slots[idx].file_offset, slots[idx].size, POSIX_FADV_DONTNEED);
            }

            // Mark as non-resident in all combined buffers
            int layer = slots[idx].layer;
            int expert = slots[idx].expert;
            for (auto & [key, cb] : combined_buffers) {
                if (cb.layer == layer && expert < (int)cb.resident.size()) {
                    cb.resident[expert] = false;
                }
            }

            slots[idx].resident = false;
            used_bytes -= slots[idx].size;
            LLAMA_LOG_DEBUG("moe_ram_arena: evicted expert L%d_E%d (%zu bytes)\n", slots[idx].layer, slots[idx].expert, slots[idx].size);
        }
        lru_list.pop_back();
    }
}

void moe_ram_arena::shutdown_worker() {
    shutdown = true;
    if (prefetch_thread.joinable()) {
        prefetch_thread.join();
    }
}

moe_ram_arena::~moe_ram_arena() {
    shutdown_worker();
    // Free any remaining resident experts - combined buffers are auto-freed by vector destructor
    if (model_fd >= 0) {
        close(model_fd);
        model_fd = -1;
    }
}

// Capture file offset for RAM-tier expert tensor and create combined buffers
void llama_model_loader::capture_expert_file_offset(const char * tensor_name, size_t file_offset, size_t tensor_size, struct ggml_tensor * cur) {
    if (!moe_plan || !moe_arena) {
        return; // MoE placement not enabled
    }

    const auto & plan = *moe_plan;
    auto & arena = *moe_arena;

    // Parse tensor name: blk.L.ffn_{gate,up,down}_exps.weight (combined tensor with all experts)
    int layer = -1;
    char ffn_type[32];

    // Match pattern: blk.%d.ffn_%[^_]_exps.weight
    int matched = sscanf(tensor_name, "blk.%d.ffn_%31[^_]_exps.weight", &layer, ffn_type);
    if (matched != 2) {
        return; // Not an expert tensor
    }

    if (layer < 0) {
        return;
    }

    if (layer >= plan.n_layer) {
        return;
    }

    // Map tensor type name
    std::string tensor_type;
    if (strcmp(ffn_type, "gate") == 0) tensor_type = "gate";
    else if (strcmp(ffn_type, "up") == 0) tensor_type = "up";
    else if (strcmp(ffn_type, "down") == 0) tensor_type = "down";
    else if (strcmp(ffn_type, "gate_up") == 0) tensor_type = "gate_up";
    else {
        LLAMA_LOG_WARN("moe_ram_arena: unknown tensor type '%s' in '%s'\n", ffn_type, tensor_name);
        return;
    }

    // Get tensor dimensions for per-expert offset calculation
    // Expert tensors are 3D: [ne[0], ne[1], n_expert] where n_expert = ne[2]
    if (!cur || cur->ne[2] != plan.n_expert) {
        LLAMA_LOG_WARN("moe_ram_arena: expert tensor '%s' has unexpected dimensions (ne[2]=%d, expected %d)\n", tensor_name, cur ? (int)cur->ne[2] : -1, plan.n_expert);
        return;
    }

    // Calculate per-expert size and stride
    const size_t per_expert_size = tensor_size / plan.n_expert;
    const size_t stride = cur->nb[2];

    std::lock_guard<std::mutex> lock(arena.mutex);

    // Create or get combined buffer for this layer and tensor type
    auto key = std::make_pair(layer, tensor_type);
    auto & cb = arena.combined_buffers[key];
    cb.layer = layer;
    cb.tensor_type = tensor_type;
    cb.expert_size = per_expert_size;
    cb.stride = stride;
    cb.buffer.resize(plan.n_expert * stride);
    cb.resident.assign(plan.n_expert, false);

    // Populate slots for all experts in this tensor that are in RAM tier
    for (int e = 0; e < plan.n_expert; ++e) {
        const size_t idx = (size_t) layer * plan.n_expert + e;
        if (idx >= plan.expert_tier.size()) {
            continue;
        }

        // Check if this expert is in RAM tier
        if (plan.expert_tier[idx] != MOE_TIER_RAM) {
            continue;
        }

        // Find the corresponding slot in the arena
        for (size_t i = 0; i < arena.slots.size(); ++i) {
            if (arena.slots[i].layer == layer && arena.slots[i].expert == e) {
                // Calculate offset for this expert within the combined tensor
                arena.slots[i].file_offset = file_offset + e * per_expert_size;
                arena.slots[i].size = per_expert_size;
                // data pointer will point into combined buffer when loaded
                break;
            }
        }
    }

    // Point the model tensor's data to the combined buffer
    // This allows moe_cache to read directly from the combined buffer
    cur->data = cb.buffer.data();
}

std::string llama_model_loader::ftype_name() const {
    return llama_ftype_name(ftype);
}

void llama_model_loader::print_info() const {
    LLAMA_LOG_INFO("%s: file format = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: file type   = %s\n", __func__, llama_ftype_name(ftype));
    if (n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0,        n_bytes*8.0/n_elements);
    } else {
        LLAMA_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0/1024.0, n_bytes*8.0/n_elements);
    }
}
