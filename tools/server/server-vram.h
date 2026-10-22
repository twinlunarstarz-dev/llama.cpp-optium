#pragma once

// Driver-level GPU memory queries and offline footprint measurement for the
// model router. Nothing here creates a CUDA context in the router process:
// device memory comes from NVML (or nvidia-smi), and per-model footprints come
// from a short-lived `llama-fit-params` child.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct server_vram_dev {
    std::string name;
    size_t total = 0; // bytes
    size_t free  = 0; // bytes
};

class server_vram {
public:
    static server_vram & instance();

    bool   available() const { return !devs.empty(); }
    size_t n_devices() const { return devs.size(); }

    // refresh the snapshot when it is older than max_age_ms
    void refresh(int64_t max_age_ms = 200);

    const server_vram_dev & dev(size_t index) const;

    // "CUDA0" / "cuda0" / "0" -> 0 ; anything else -> -1
    static int dev_index_from_name(const std::string & name);

private:
    server_vram();

    void init_nvml();
    bool refresh_nvml();
    bool refresh_smi();

    std::vector<server_vram_dev> devs;
    int64_t last_refresh = 0;
    bool    nvml_broken  = false;

    void * lib = nullptr;
    int  (*fn_init)    (void) = nullptr;
    int  (*fn_shutdown)(void) = nullptr;
    int  (*fn_count)   (unsigned int *) = nullptr;
    int  (*fn_handle)  (unsigned int, void **) = nullptr;
    int  (*fn_memory)  (void *, void *) = nullptr;
    int  (*fn_name)    (void *, char *, unsigned int) = nullptr;
};

// Runs `exe args...`, parses the `CUDA<n> <model> <context> <compute>` lines
// printed by `--fit-print on`, and returns the per-device sum in bytes.
// Returns an empty vector on failure and fills `err`.
std::vector<size_t> server_vram_probe(const std::string & exe,
                                      const std::vector<std::string> & args,
                                      std::string & err);
