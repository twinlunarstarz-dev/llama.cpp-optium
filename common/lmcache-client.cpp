#include "lmcache-client.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <winsock2.h>
#    include <ws2tcpip.h>
using lmcache_socket_t                                   = SOCKET;
static constexpr lmcache_socket_t LMCACHE_INVALID_SOCKET = INVALID_SOCKET;
#else
#    include <netdb.h>
#    include <sys/socket.h>
#    include <sys/types.h>
#    include <unistd.h>
using lmcache_socket_t                                   = int;
static constexpr lmcache_socket_t LMCACHE_INVALID_SOCKET = -1;
#endif

namespace {

constexpr size_t LMCACHE_MAX_KEY_LENGTH     = 150;
constexpr size_t LMCACHE_CLIENT_HEADER_SIZE = 9 * sizeof(int32_t) + LMCACHE_MAX_KEY_LENGTH;
constexpr size_t LMCACHE_SERVER_HEADER_SIZE = 9 * sizeof(int32_t);

constexpr int32_t LMCACHE_COMMAND_PUT          = 1;
constexpr int32_t LMCACHE_COMMAND_GET          = 2;
constexpr int32_t LMCACHE_RETURN_SUCCESS       = 200;
constexpr int32_t LMCACHE_RETURN_FAIL          = 400;
constexpr int32_t LMCACHE_FORMAT_BINARY_BUFFER = 5;
constexpr int32_t LMCACHE_DTYPE_UINT8          = 6;

struct socket_handle {
    lmcache_socket_t value = LMCACHE_INVALID_SOCKET;

    ~socket_handle() {
        if (value != LMCACHE_INVALID_SOCKET) {
#ifdef _WIN32
            closesocket(value);
#else
            close(value);
#endif
        }
    }
};

#ifdef _WIN32
struct winsock_scope {
    winsock_scope() {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
    }

    ~winsock_scope() { WSACleanup(); }
};
#endif

static void parse_endpoint(const std::string & endpoint, std::string & host, std::string & port) {
    if (endpoint.empty()) {
        throw std::invalid_argument("LMCache endpoint is empty");
    }

    if (endpoint.front() == '[') {
        const size_t close = endpoint.find(']');
        if (close == std::string::npos || close + 2 >= endpoint.size() || endpoint[close + 1] != ':') {
            throw std::invalid_argument("invalid LMCache endpoint: " + endpoint);
        }
        host = endpoint.substr(1, close - 1);
        port = endpoint.substr(close + 2);
    } else {
        const size_t colon = endpoint.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size() || endpoint.find(':') != colon) {
            throw std::invalid_argument("invalid LMCache endpoint: " + endpoint);
        }
        host = endpoint.substr(0, colon);
        port = endpoint.substr(colon + 1);
    }

    size_t    parsed      = 0;
    const int port_number = std::stoi(port, &parsed);
    if (parsed != port.size() || port_number < 1 || port_number > 65535) {
        throw std::invalid_argument("invalid LMCache endpoint port: " + port);
    }
}

static void store_i32(uint8_t * dst, int32_t value) {
    std::memcpy(dst, &value, sizeof(value));
}

static int32_t load_i32(const uint8_t * src) {
    int32_t value;
    std::memcpy(&value, src, sizeof(value));
    return value;
}

static bool send_all(lmcache_socket_t socket, const uint8_t * data, size_t size, std::string & error) {
    while (size > 0) {
        const int chunk = (int) std::min<size_t>(size, std::numeric_limits<int>::max());
#ifdef _WIN32
        const int n = send(socket, reinterpret_cast<const char *>(data), chunk, 0);
#else
        const int n = (int) send(socket, data, chunk, MSG_NOSIGNAL);
#endif
        if (n <= 0) {
            error = "failed to send LMCache request";
            return false;
        }
        data += n;
        size -= n;
    }
    return true;
}

static bool recv_all(lmcache_socket_t socket, uint8_t * data, size_t size, std::string & error) {
    while (size > 0) {
        const int chunk = (int) std::min<size_t>(size, std::numeric_limits<int>::max());
#ifdef _WIN32
        const int n = recv(socket, reinterpret_cast<char *>(data), chunk, 0);
#else
        const int n = (int) recv(socket, data, chunk, 0);
#endif
        if (n <= 0) {
            error = "failed to receive LMCache response";
            return false;
        }
        data += n;
        size -= n;
    }
    return true;
}

static bool connect_socket(const std::string & host,
                           const std::string & port,
                           socket_handle &     result,
                           std::string &       error) {
    addrinfo hints    = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo * addresses = nullptr;
    const int  rc        = getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses);
    if (rc != 0) {
        error = "failed to resolve LMCache endpoint";
        return false;
    }

    for (addrinfo * address = addresses; address != nullptr; address = address->ai_next) {
        socket_handle candidate;
        candidate.value = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (candidate.value == LMCACHE_INVALID_SOCKET) {
            continue;
        }
        if (connect(candidate.value, address->ai_addr, (int) address->ai_addrlen) == 0) {
            result.value    = candidate.value;
            candidate.value = LMCACHE_INVALID_SOCKET;
            freeaddrinfo(addresses);
            return true;
        }
    }

    freeaddrinfo(addresses);
    error = "failed to connect to LMCache endpoint";
    return false;
}

static bool make_header(int32_t                                           command,
                        const std::string &                               key,
                        size_t                                            data_size,
                        std::array<uint8_t, LMCACHE_CLIENT_HEADER_SIZE> & header,
                        std::string &                                     error) {
    if (key.size() > LMCACHE_MAX_KEY_LENGTH) {
        error = "LMCache key exceeds 150 bytes";
        return false;
    }
    if (data_size > (size_t) std::numeric_limits<int32_t>::max()) {
        error = "LMCache object exceeds protocol size limit";
        return false;
    }

    header.fill(0);
    store_i32(header.data() + 0 * sizeof(int32_t), command);
    store_i32(header.data() + 1 * sizeof(int32_t), (int32_t) data_size);
    store_i32(header.data() + 2 * sizeof(int32_t), LMCACHE_FORMAT_BINARY_BUFFER);
    store_i32(header.data() + 3 * sizeof(int32_t), LMCACHE_DTYPE_UINT8);
    store_i32(header.data() + 4 * sizeof(int32_t), 0);
    store_i32(header.data() + 5 * sizeof(int32_t), (int32_t) data_size);
    std::memcpy(header.data() + 9 * sizeof(int32_t), key.data(), key.size());
    return true;
}

}  // namespace

common_lmcache_client::common_lmcache_client(const std::string & endpoint) : endpoint_(endpoint) {
    parse_endpoint(endpoint, host_, port_);
}

bool common_lmcache_client::put(const std::string & key, const std::vector<uint8_t> & data, std::string & error) const {
    std::array<uint8_t, LMCACHE_CLIENT_HEADER_SIZE> header;
    if (!make_header(LMCACHE_COMMAND_PUT, key, data.size(), header, error)) {
        return false;
    }

#ifdef _WIN32
    winsock_scope winsock;
#endif
    socket_handle socket;
    if (!connect_socket(host_, port_, socket, error)) {
        return false;
    }
    return send_all(socket.value, header.data(), header.size(), error) &&
           send_all(socket.value, data.data(), data.size(), error);
}

bool common_lmcache_client::get(const std::string &    key,
                                std::vector<uint8_t> & data,
                                bool &                 found,
                                std::string &          error) const {
    found = false;
    std::array<uint8_t, LMCACHE_CLIENT_HEADER_SIZE> request;
    if (!make_header(LMCACHE_COMMAND_GET, key, 0, request, error)) {
        return false;
    }

#ifdef _WIN32
    winsock_scope winsock;
#endif
    socket_handle socket;
    if (!connect_socket(host_, port_, socket, error) ||
        !send_all(socket.value, request.data(), request.size(), error)) {
        return false;
    }

    std::array<uint8_t, LMCACHE_SERVER_HEADER_SIZE> response;
    if (!recv_all(socket.value, response.data(), response.size(), error)) {
        return false;
    }

    const int32_t code = load_i32(response.data());
    const int32_t size = load_i32(response.data() + sizeof(int32_t));
    if (code == LMCACHE_RETURN_FAIL) {
        return true;
    }
    if (code != LMCACHE_RETURN_SUCCESS || size < 0) {
        error = "invalid LMCache response";
        return false;
    }

    try {
        data.resize((size_t) size);
    } catch (const std::bad_alloc &) {
        error = "failed to allocate LMCache response buffer";
        return false;
    }
    if (!recv_all(socket.value, data.data(), data.size(), error)) {
        data.clear();
        return false;
    }
    found = true;
    return true;
}

const std::string & common_lmcache_client::endpoint() const {
    return endpoint_;
}
