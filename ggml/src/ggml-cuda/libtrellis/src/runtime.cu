// SPDX-License-Identifier: Apache-2.0
// Copyright Nari Labs
// libtrellis: errors, device queries, matrix helpers.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "common.cuh"

namespace {
thread_local char g_error[512] = "";
constexpr int MAX_DEVICES = 64;
// g_sms[dev] != 0 (stored last, with release) means that g_cc[dev] is set too
std::atomic<int> g_sms[MAX_DEVICES] = {};
std::atomic<int> g_cc[MAX_DEVICES] = {};
std::mutex g_mutex;

int current_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev < 0 || dev >= MAX_DEVICES ? 0 : dev;
}

void query(int dev) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_sms[dev].load(std::memory_order_relaxed) != 0) return;
    int sms = 0, major = 0, minor = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    g_cc[dev].store(10 * major + minor, std::memory_order_relaxed);
    g_sms[dev].store(sms > 0 ? sms : 1, std::memory_order_release);
}
}  // namespace

namespace tq {
int fail(int code, const char* what) {
    snprintf(g_error, sizeof(g_error), "%s", what);
    return code;
}

int check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return 0;
    snprintf(g_error, sizeof(g_error), "%s: %s", what, cudaGetErrorString(e));
    return static_cast<int>(e);
}

int device_sms() {
    const int dev = current_device();
    if (g_sms[dev].load(std::memory_order_acquire) == 0) query(dev);
    return g_sms[dev].load(std::memory_order_acquire);
}

int device_cc() {
    const int dev = current_device();
    if (g_sms[dev].load(std::memory_order_acquire) == 0) query(dev);
    return g_cc[dev].load(std::memory_order_relaxed);
}
}  // namespace tq

extern "C" {
const char* tq_last_error(void) { return g_error; }
int tq_version(void) { return TQ_VERSION; }

tq_matrix tq_matrix_exl3(const void* codes, int bits, int K, int N, const void* block_scale) {
    tq_matrix m;
    m.codes = codes, m.bits = bits, m.K = K, m.N = N;
    m.tile_stride_k = N / 16, m.tile_stride_n = 1;
    m.block_scale = block_scale, m.mult = TQ_INT8_MULT;
    return m;
}

tq_matrix tq_matrix_rows(const void* codes, int bits, int K, int N, const void* block_scale) {
    tq_matrix m;
    m.codes = codes, m.bits = bits, m.K = K, m.N = N;
    m.tile_stride_k = 1, m.tile_stride_n = K / 16;
    m.block_scale = block_scale, m.mult = TQ_INT8_MULT;
    return m;
}
}
