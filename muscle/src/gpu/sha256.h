#pragma once

// Minimal self-contained SHA-256 (FIPS 180-4) used for:
//   * HMM snapshot digests (explicit serialization, never struct padding)
//   * reference/input identity records in results/reference
// Only 32-bit arithmetic and std::string/std::vector are used, so this compiles
// identically in host and device-independent code.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace muscle_gpu {

class Sha256 {
public:
    Sha256() { Reset(); }
    void Reset();
    void Update(const void *data, size_t bytes);
    void Update(const std::string &s) { Update(s.data(), s.size()); }
    void Final(uint8_t out[32]);

    static void Digest(const void *data, size_t bytes, uint8_t out[32]);
    static std::string Hex(const uint8_t digest[32]);
    static std::string OfBuffer(const void *data, size_t bytes);

private:
    void Transform(const uint8_t block[64]);

    uint32_t m_H[8];
    uint64_t m_Bytes;
    uint8_t m_Buf[64];
    size_t m_BufLen;
};

} // namespace muscle_gpu
