#pragma once

// Snapshot capture for the numerically frozen Pair-HMM tables.
//
// The MUSCLE reference path re-reads the *global* PairHMM static tables at every
// call (hmmscores.h is included inside the DP functions), so a batch is only
// well defined when the whole batch runs against one frozen set of tables.  The
// GPU/CPU backends capture that set here, digest it explicitly, and (for the CPU
// backend) verify bitwise that the globals did not change before entering the
// parallel batch.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu/pair_backend.h"
#include "gpu/sha256.h"

namespace muscle_gpu {

// Alphabet currently installed in the MUSCLE globals, mapped to our own enum.
Alphabet CurrentAlphabet();

// Capture the installed tables.  Returns false (with `err`) when the tables look
// uninitialized, i.e. HMMParams::ToPairHMM() has not run yet.
bool CaptureHmmSnapshot(HmmSnapshot &out, std::string &err);

// Bitwise comparison of a snapshot against the currently installed tables,
// including the digest-covered scalar constants.  Used by the CPU backend before
// it enters its OpenMP region, and by tests that switch models.
bool HmmSnapshotMatchesCurrent(const HmmSnapshot &snap, std::string &err);

// Little-endian serialization of the digest-covered fields (no padding).
void SerializeHmmSnapshot(const HmmSnapshot &snap, std::vector<uint8_t> &bytes);

// Lower-case hex SHA-256 of the serialized fields (recomputed, not trusted).
std::string HmmSnapshotDigestHex(const HmmSnapshot &snap);

inline std::string DigestHex(const std::array<uint8_t, 32> &digest) {
    return Sha256::Hex(digest.data());
}

} // namespace muscle_gpu
