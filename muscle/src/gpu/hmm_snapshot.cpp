#include "gpu/hmm_snapshot.h"

#include <cstring>

#include "muscle.h"

namespace muscle_gpu {

namespace {

inline void PutFloatLE(std::vector<uint8_t> &out, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
    memcpy(&bits, &value, sizeof(bits));
    out.push_back(uint8_t(bits & 0xff));
    out.push_back(uint8_t((bits >> 8) & 0xff));
    out.push_back(uint8_t((bits >> 16) & 0xff));
    out.push_back(uint8_t((bits >> 24) & 0xff));
}

inline void PutU32LE(std::vector<uint8_t> &out, uint32_t value) {
    out.push_back(uint8_t(value & 0xff));
    out.push_back(uint8_t((value >> 8) & 0xff));
    out.push_back(uint8_t((value >> 16) & 0xff));
    out.push_back(uint8_t((value >> 24) & 0xff));
}

inline bool SameBits(float a, float b) {
    uint32_t x = 0, y = 0;
    memcpy(&x, &a, sizeof(x));
    memcpy(&y, &b, sizeof(y));
    return x == y;
}

} // namespace

Alphabet CurrentAlphabet() {
    switch (g_Alpha) {
    case ALPHA_Amino: return Alphabet::Amino;
    case ALPHA_Nucleo: return Alphabet::Nucleo;
    default: return Alphabet::Unknown;
    }
}

void SerializeHmmSnapshot(const HmmSnapshot &snap, std::vector<uint8_t> &bytes) {
    bytes.clear();
    bytes.reserve(HmmSnapshotDigestBytes());
    for (size_t i = 0; i < snap.start.size(); ++i)
        PutFloatLE(bytes, snap.start[i]);
    for (size_t i = 0; i < snap.trans.size(); ++i)
        PutFloatLE(bytes, snap.trans[i]);
    for (size_t i = 0; i < snap.ins.size(); ++i)
        PutFloatLE(bytes, snap.ins[i]);
    for (size_t i = 0; i < snap.match.size(); ++i)
        PutFloatLE(bytes, snap.match[i]);
    PutFloatLE(bytes, snap.log_zero);
    PutFloatLE(bytes, snap.min_sparse_score);
    PutFloatLE(bytes, snap.min_sparse_prob);
    PutU32LE(bytes, snap.alphabet);
    PutU32LE(bytes, snap.semantics_version);
}

std::string HmmSnapshotDigestHex(const HmmSnapshot &snap) {
    std::vector<uint8_t> bytes;
    SerializeHmmSnapshot(snap, bytes);
    uint8_t digest[32];
    Sha256::Digest(bytes.data(), bytes.size(), digest);
    return Sha256::Hex(digest);
}

bool CaptureHmmSnapshot(HmmSnapshot &out, std::string &err) {
    err.clear();

    // Guard against capturing zero-initialized statics: the reference path always
    // runs HMMParams::ToPairHMM() before any DP call, and a snapshot of untouched
    // tables would silently produce garbage probabilities.
    bool any_nonzero = false;
    for (uint s = 0; s < HMMSTATE_COUNT && !any_nonzero; ++s)
        any_nonzero = (PairHMM::m_StartScore[s] != 0.0f);
    for (uint i = 0; i < HMMSTATE_COUNT && !any_nonzero; ++i)
        for (uint j = 0; j < HMMSTATE_COUNT && !any_nonzero; ++j)
            any_nonzero = (PairHMM::m_TransScore[i][j] != 0.0f);
    for (uint b = 0; b < 256 && !any_nonzero; ++b)
        any_nonzero = (PairHMM::m_InsScore[b] != 0.0f);
    for (uint x = 0; x < 256 && !any_nonzero; ++x)
        for (uint y = 0; y < 256 && !any_nonzero; ++y)
            any_nonzero = (PairHMM::m_MatchScore[x][y] != 0.0f);
    if (!any_nonzero) {
        err = "PairHMM tables are all zero: HMMParams::ToPairHMM() has not run";
        return false;
    }

    for (uint s = 0; s < HMMSTATE_COUNT; ++s)
        out.start[s] = PairHMM::m_StartScore[s];
    for (uint i = 0; i < HMMSTATE_COUNT; ++i)
        for (uint j = 0; j < HMMSTATE_COUNT; ++j)
            out.trans[i * HMMSTATE_COUNT + j] = PairHMM::m_TransScore[i][j];
    for (uint b = 0; b < 256; ++b)
        out.ins[b] = PairHMM::m_InsScore[b];
    for (uint x = 0; x < 256; ++x)
        for (uint y = 0; y < 256; ++y)
            out.match[x * 256 + y] = PairHMM::m_MatchScore[x][y];
    out.log_zero = LOG_ZERO;
    out.min_sparse_score = MIN_SPARSE_SCORE;
    out.min_sparse_prob = MIN_SPARSE_PROB;
    out.alphabet = uint32_t(CurrentAlphabet());
    out.semantics_version = PAIR_SEMANTICS_VERSION;

    std::vector<uint8_t> bytes;
    SerializeHmmSnapshot(out, bytes);
    Sha256::Digest(bytes.data(), bytes.size(), out.digest.data());
    return true;
}

bool HmmSnapshotMatchesCurrent(const HmmSnapshot &snap, std::string &err) {
    err.clear();
    if (snap.semantics_version != PAIR_SEMANTICS_VERSION) {
        err = "snapshot semantics_version != build PAIR_SEMANTICS_VERSION";
        return false;
    }
    if (snap.log_zero != LOG_ZERO || snap.min_sparse_prob != MIN_SPARSE_PROB ||
        !SameBits(snap.min_sparse_score, MIN_SPARSE_SCORE)) {
        err = "snapshot constants differ from compiled constants";
        return false;
    }
    if (snap.alphabet != uint32_t(CurrentAlphabet())) {
        err = "snapshot alphabet differs from current alphabet";
        return false;
    }
    for (uint s = 0; s < HMMSTATE_COUNT; ++s)
        if (!SameBits(snap.start[s], PairHMM::m_StartScore[s])) {
            err = "PairHMM start scores changed after snapshot";
            return false;
        }
    for (uint i = 0; i < HMMSTATE_COUNT; ++i)
        for (uint j = 0; j < HMMSTATE_COUNT; ++j)
            if (!SameBits(snap.trans[i * HMMSTATE_COUNT + j], PairHMM::m_TransScore[i][j])) {
                err = "PairHMM transition scores changed after snapshot";
                return false;
            }
    for (uint b = 0; b < 256; ++b)
        if (!SameBits(snap.ins[b], PairHMM::m_InsScore[b])) {
            err = "PairHMM insert scores changed after snapshot";
            return false;
        }
    for (uint x = 0; x < 256; ++x)
        for (uint y = 0; y < 256; ++y)
            if (!SameBits(snap.match[x * 256 + y], PairHMM::m_MatchScore[x][y])) {
                err = "PairHMM match scores changed after snapshot";
                return false;
            }
    const std::string now = HmmSnapshotDigestHex(snap);
    if (now != DigestHex(snap.digest)) {
        err = "snapshot digest does not match its serialized fields";
        return false;
    }
    return true;
}

} // namespace muscle_gpu
