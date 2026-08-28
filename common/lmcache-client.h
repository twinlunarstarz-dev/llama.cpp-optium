#pragma once

#include <cstdint>
#include <string>
#include <vector>

class common_lmcache_client {
  public:
    explicit common_lmcache_client(const std::string & endpoint);

    bool put(const std::string & key, const std::vector<uint8_t> & data, std::string & error) const;
    bool get(const std::string & key, std::vector<uint8_t> & data, bool & found, std::string & error) const;

    const std::string & endpoint() const;

  private:
    std::string host_;
    std::string port_;
    std::string endpoint_;
};
