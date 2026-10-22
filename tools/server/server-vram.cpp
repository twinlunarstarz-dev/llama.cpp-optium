#include "server-vram.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <fcntl.h>
#  include <poll.h>
#  include <signal.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace {

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr size_t MIB = 1024ull * 1024ull;

// the subset of nvmlMemory_t we need
struct nvml_memory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

} // namespace

server_vram & server_vram::instance() {
    static server_vram inst;
    return inst;
}

server_vram::server_vram() {
    init_nvml();
    refresh(0);
}

const server_vram_dev & server_vram::dev(size_t index) const {
    static const server_vram_dev empty;
    return index < devs.size() ? devs[index] : empty;
}

int server_vram::dev_index_from_name(const std::string & name) {
    size_t i = 0;
    while (i < name.size() && !isdigit((unsigned char) name[i])) {
        i++;
    }
    if (i == name.size()) {
        return -1;
    }
    // only "CUDA3", "cuda3", "3" and similar prefixes are accepted
    for (size_t k = 0; k < i; k++) {
        const char c = (char) tolower((unsigned char) name[k]);
        if (c != 'c' && c != 'u' && c != 'd' && c != 'a') {
            return -1;
        }
    }
    return atoi(name.c_str() + i);
}

void server_vram::init_nvml() {
#if defined(_WIN32)
    return; // nvidia-smi fallback only
#else
    // the WSL driver installs NVML outside the normal loader path
    static const char * candidates[] = {
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "/usr/lib/wsl/lib/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
    };
    for (const char * path : candidates) {
        lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (lib != nullptr) {
            break;
        }
    }
    if (lib == nullptr) {
        nvml_broken = true;
        return;
    }

    auto sym = [this](const char * v2, const char * v1) -> void * {
        void * p = dlsym(lib, v2);
        return p != nullptr ? p : dlsym(lib, v1);
    };

    fn_init     = (decltype(fn_init))     sym("nvmlInit_v2", "nvmlInit");
    fn_shutdown = (decltype(fn_shutdown)) sym("nvmlShutdown", "nvmlShutdown");
    fn_count    = (decltype(fn_count))    sym("nvmlDeviceGetCount_v2", "nvmlDeviceGetCount");
    fn_handle   = (decltype(fn_handle))   sym("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
    fn_memory   = (decltype(fn_memory))   sym("nvmlDeviceGetMemoryInfo", "nvmlDeviceGetMemoryInfo");
    fn_name     = (decltype(fn_name))     sym("nvmlDeviceGetName", "nvmlDeviceGetName");

    if (fn_init == nullptr || fn_count == nullptr || fn_handle == nullptr || fn_memory == nullptr) {
        dlclose(lib);
        lib = nullptr;
        nvml_broken = true;
        return;
    }
    if (fn_init() != 0) {
        dlclose(lib);
        lib = nullptr;
        nvml_broken = true;
    }
#endif
}

bool server_vram::refresh_nvml() {
    if (lib == nullptr) {
        return false;
    }
    unsigned int n = 0;
    if (fn_count(&n) != 0 || n == 0) {
        return false;
    }

    std::vector<server_vram_dev> out(n);
    for (unsigned int i = 0; i < n; i++) {
        void * handle = nullptr;
        if (fn_handle(i, &handle) != 0 || handle == nullptr) {
            continue;
        }
        nvml_memory_t mem = {};
        if (fn_memory(handle, &mem) != 0) {
            continue;
        }
        char name[96] = {0};
        if (fn_name != nullptr) {
            fn_name(handle, name, sizeof(name));
        }
        out[i].name  = name;
        out[i].total = (size_t) mem.total;
        out[i].free  = (size_t) mem.free;
    }

    // every entry kept, so indices stay aligned with CUDA device order
    devs = std::move(out);
    return true;
}

bool server_vram::refresh_smi() {
    const char * cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null";
#if defined(_WIN32)
    FILE * f = _popen(cmd, "r");
#else
    FILE * f = popen(cmd, "r");
#endif
    if (f == nullptr) {
        return false;
    }

    std::vector<server_vram_dev> out;
    char line[512];
    while (fgets(line, sizeof(line), f) != nullptr) {
        // "<name>, <total MiB>, <free MiB>"
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ')) {
            line[--len] = '\0';
        }
        if (len == 0 || strncmp(line, "N/A", 3) == 0) {
            continue;
        }
        char * last = strrchr(line, ',');
        if (last == nullptr) {
            continue;
        }
        char * prev = last;
        while (prev > line && *(prev - 1) != ',') {
            prev--;
        }
        server_vram_dev d;
        d.free  = (size_t) strtoull(last + 1, nullptr, 10) * MIB;
        d.total = (size_t) strtoull(prev,     nullptr, 10) * MIB;
        if (prev > line) {
            size_t name_len = (size_t) (prev - line);
            while (name_len > 0 && line[name_len - 1] == ' ') {
                name_len--;
            }
            d.name = std::string(line, name_len);
        }
        out.push_back(std::move(d));
    }
#if defined(_WIN32)
    _pclose(f);
#else
    pclose(f);
#endif
    if (out.empty()) {
        return false;
    }
    devs = std::move(out);
    return true;
}

void server_vram::refresh(int64_t max_age_ms) {
    if (last_refresh != 0 && now_ms() - last_refresh < max_age_ms) {
        return;
    }
    if (!refresh_nvml()) {
        refresh_smi(); // no-op when NVML already refreshed
    }
    // a transient NVML failure must not drop the last known snapshot
    last_refresh = now_ms();
}

//
// probe
//

std::vector<size_t> server_vram_probe(const std::string & exe,
                                      const std::vector<std::string> & args,
                                      std::string & err) {
#if defined(_WIN32)
    (void) exe; (void) args;
    err = "probing is not supported on this build";
    return {};
#else
    int fds[2] = { -1, -1 };
    if (pipe(fds) != 0) {
        err = "pipe() failed";
        return {};
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        err = "fork() failed";
        return {};
    }

    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(exe.c_str()));
        for (const auto & a : args) {
            argv.push_back(const_cast<char *>(a.c_str()));
        }
        argv.push_back(nullptr);
        execv(exe.c_str(), argv.data());
        _exit(127);
    }

    close(fds[1]);

    std::string out;
    char buf[4096];
    const int64_t deadline = now_ms() + 180000;
    bool timed_out = false;
    while (true) {
        const int64_t left = deadline - now_ms();
        if (left <= 0) {
            timed_out = true;
            break;
        }
        struct pollfd pfd = { fds[0], POLLIN, 0 };
        const int pr = poll(&pfd, 1, (int) std::min<int64_t>(left, 1000));
        if (pr == 0) {
            continue;
        }
        if (pr < 0) {
            break;
        }
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n <= 0) {
            break;
        }
        out.append(buf, (size_t) n);
    }
    close(fds[0]);

    int status = 0;
    if (timed_out) {
        kill(pid, SIGKILL);
    }
    waitpid(pid, &status, 0);

    if (timed_out) {
        err = "probe timed out";
        return {};
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        err = "probe exited with status " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return {};
    }

    std::vector<size_t> per_dev;
    size_t pos = 0;
    while (pos <= out.size()) {
        const size_t eol = out.find('\n', pos);
        const std::string line = out.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        pos = eol == std::string::npos ? out.size() + 1 : eol + 1;

        if (line.compare(0, 4, "CUDA") != 0) {
            continue;
        }
        const size_t idx_end = line.find_first_of(" \t", 4);
        if (idx_end == std::string::npos) {
            continue;
        }
        const int idx = atoi(line.c_str() + 4);
        if (idx < 0 || idx > 64) {
            continue;
        }

        // model, context, compute in MiB
        long long v[3] = {0, 0, 0};
        int got = 0;
        const char * p = line.c_str() + idx_end;
        while (got < 3 && *p != '\0') {
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\0') break;
            char * end = nullptr;
            const long long n = strtoll(p, &end, 10);
            if (end == p) break;
            v[got++] = n;
            p = end;
        }
        if (got < 3) {
            continue;
        }
        const size_t bytes = (size_t) (v[0] + v[1] + v[2]) * MIB;
        // llama-fit-params prints its own device numbering, starting at CUDA0 for
        // whatever subset it was given, so the results are returned in emission
        // order and the caller maps them back to real device indices.
        per_dev.push_back(bytes);
    }

    if (per_dev.empty()) {
        err = "probe printed no CUDA memory lines";
    }
    return per_dev;
#endif
}
