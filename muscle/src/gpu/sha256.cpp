#include "gpu/sha256.h"

#include <cstring>

namespace muscle_gpu {

namespace {

inline uint32_t Rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

const uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

} // namespace

void Sha256::Reset() {
    m_H[0] = 0x6a09e667u;
    m_H[1] = 0xbb67ae85u;
    m_H[2] = 0x3c6ef372u;
    m_H[3] = 0xa54ff53au;
    m_H[4] = 0x510e527fu;
    m_H[5] = 0x9b05688cu;
    m_H[6] = 0x1f83d9abu;
    m_H[7] = 0x5be0cd19u;
    m_Bytes = 0;
    m_BufLen = 0;
    memset(m_Buf, 0, sizeof(m_Buf));
}

void Sha256::Transform(const uint8_t block[64]) {
    uint32_t W[64];
    for (unsigned i = 0; i < 16; ++i) {
        W[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
               (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
    }
    for (unsigned i = 16; i < 64; ++i) {
        const uint32_t s0 = Rotr(W[i - 15], 7) ^ Rotr(W[i - 15], 18) ^ (W[i - 15] >> 3);
        const uint32_t s1 = Rotr(W[i - 2], 17) ^ Rotr(W[i - 2], 19) ^ (W[i - 2] >> 10);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }

    uint32_t a = m_H[0], b = m_H[1], c = m_H[2], d = m_H[3];
    uint32_t e = m_H[4], f = m_H[5], g = m_H[6], h = m_H[7];

    for (unsigned i = 0; i < 64; ++i) {
        const uint32_t S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + S1 + ch + K[i] + W[i];
        const uint32_t S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    m_H[0] += a;
    m_H[1] += b;
    m_H[2] += c;
    m_H[3] += d;
    m_H[4] += e;
    m_H[5] += f;
    m_H[6] += g;
    m_H[7] += h;
}

void Sha256::Update(const void *data, size_t bytes) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    m_Bytes += bytes;
    while (bytes > 0) {
        const size_t take = (bytes < 64 - m_BufLen) ? bytes : (64 - m_BufLen);
        memcpy(m_Buf + m_BufLen, p, take);
        m_BufLen += take;
        p += take;
        bytes -= take;
        if (m_BufLen == 64) {
            Transform(m_Buf);
            m_BufLen = 0;
        }
    }
}

void Sha256::Final(uint8_t out[32]) {
    const uint64_t bit_len = m_Bytes * 8;
    const uint8_t pad = 0x80;
    Update(&pad, 1);
    const uint8_t zero = 0;
    while (m_BufLen != 56)
        Update(&zero, 1);
    uint8_t lenbuf[8];
    for (unsigned i = 0; i < 8; ++i)
        lenbuf[i] = uint8_t((bit_len >> (56 - 8 * i)) & 0xff);
    // Bypass Update's byte counter for the length block.
    memcpy(m_Buf + 56, lenbuf, 8);
    Transform(m_Buf);
    m_BufLen = 0;
    for (unsigned i = 0; i < 8; ++i) {
        out[i * 4] = uint8_t(m_H[i] >> 24);
        out[i * 4 + 1] = uint8_t(m_H[i] >> 16);
        out[i * 4 + 2] = uint8_t(m_H[i] >> 8);
        out[i * 4 + 3] = uint8_t(m_H[i]);
    }
    Reset();
}

void Sha256::Digest(const void *data, size_t bytes, uint8_t out[32]) {
    Sha256 s;
    s.Update(data, bytes);
    s.Final(out);
}

std::string Sha256::Hex(const uint8_t digest[32]) {
    static const char *hexd = "0123456789abcdef";
    std::string s;
    s.resize(64);
    for (unsigned i = 0; i < 32; ++i) {
        s[2 * i] = hexd[digest[i] >> 4];
        s[2 * i + 1] = hexd[digest[i] & 0xf];
    }
    return s;
}

std::string Sha256::OfBuffer(const void *data, size_t bytes) {
    uint8_t d[32];
    Digest(data, bytes, d);
    return Hex(d);
}

} // namespace muscle_gpu
