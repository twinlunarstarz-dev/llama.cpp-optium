#pragma once
#include <cstdint>
#include <string>
#include <vector>

// MoE placement profile persistence (shared by the moe-cache recorder and the
// arena-path recorder in llama-context.cpp). Format version 1.
bool moe_profile_read(const std::string & path, int32_t & n_layer, int32_t & n_expert, std::vector<uint32_t> & counts, uint64_t & tokens);
bool moe_profile_write(const std::string & path, int32_t n_layer, int32_t n_expert, int32_t n_expert_used, const std::vector<uint32_t> & counts, uint64_t tokens);
