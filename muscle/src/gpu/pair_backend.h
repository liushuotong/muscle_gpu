#pragma once

// MUSCLE5 GPU pair backend interface, version 1.
//
// This header is the frozen contract between the CPU control flow (MPCFlat) and
// any pair-probability backend (CPU batch, CUDA).  Rules:
//   * C++17 standard library only -- no CUDA header may ever be included here, so
//     that a MUSCLE_CUDA=OFF build includes and links this file unchanged.
//   * Field names/meanings follow HANDOFF_AI.md section 4.  Implementations may
//     add private state, never change a documented meaning.
//   * Everything the backend needs is passed in; the backend never reads MUSCLE
//     globals (labels, sequence maps, PairHMM statics) on device.
//
// Fidelity rules implemented by every backend (see RESEARCH_PLAN.md section 2.2):
//   * five-state order [M, IX, IY, JX, JY] for storage/enum order, while the
//     Forward M recurrence sums [M, IX, JX, IY, JY] -- the two orders differ.
//   * LOG_ZERO = -2e20f is a finite value, not -inf; the piecewise polynomial
//     LOG_ADD with its 7.5 cutoff and its right-nested operand order is kept
//     verbatim; no exact logsumexp, no max, no reordering.
//   * q(i,j) = F_M(i+1,j+1) + B_M(i+1,j+1) - Z with the host-side two-stage
//     threshold: first q < min_sparse_score, then q >= 0 ? 1 : expf(q).
//   * Post is a real probability in [0,1], row-major len_x*len_y.

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace muscle_gpu {

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

enum class BackendKind { Cpu, Gpu, Auto };

enum class PairStatus {
    Ok,            // complete, faithful result in `post`
    CpuFallback,   // backend declined this pair; caller must run the CPU path
    InvalidInput,  // input violates the CPU reference contract (caller dies)
    ResourceError, // out of memory / budget; caller may retry smaller or fall back
    DeviceError    // fatal device/context error; caller must not reuse results
};

// Alphabet is defined independently of the MUSCLE global g_Alpha; the snapshot
// must stay self-describing and must not depend on a global's address.
enum class Alphabet : uint32_t { Unknown = 0, Amino = 1, Nucleo = 2 };

inline const char *ToString(BackendKind k) {
    switch (k) {
    case BackendKind::Cpu: return "cpu";
    case BackendKind::Gpu: return "gpu";
    case BackendKind::Auto: return "auto";
    }
    return "?";
}

inline const char *ToString(PairStatus s) {
    switch (s) {
    case PairStatus::Ok: return "Ok";
    case PairStatus::CpuFallback: return "CpuFallback";
    case PairStatus::InvalidInput: return "InvalidInput";
    case PairStatus::ResourceError: return "ResourceError";
    case PairStatus::DeviceError: return "DeviceError";
    }
    return "?";
}

// Bumped whenever the numerical semantics of a backend change (operation order,
// thresholds, constants).  A cached or serialized result is only reusable when
// this value and the HMM digest both match.
static const uint32_t PAIR_SEMANTICS_VERSION = 1;

// ---------------------------------------------------------------------------
// HMM snapshot
// ---------------------------------------------------------------------------

// A byte-for-byte copy of the *actual* global PairHMM tables in use for the
// batch (start scores, transition scores, per-byte insert and match scores).
// `digest` is SHA-256 over the explicit little-endian serialization of the
// fields above it -- never over struct padding.
struct HmmSnapshot {
    std::array<float, 5> start;        // HMMSTATE_M, IX, IY, JX, JY
    std::array<float, 25> trans;       // src*5 + dst
    std::array<float, 256> ins;        // per input byte
    std::array<float, 65536> match;    // byte_x*256 + byte_y
    float log_zero;                    // LOG_ZERO, exactly as compiled
    float min_sparse_score;            // logf(MIN_SPARSE_PROB), host bit pattern
    float min_sparse_prob;             // MIN_SPARSE_PROB
    uint32_t alphabet;                 // Alphabet value
    uint32_t semantics_version;        // PAIR_SEMANTICS_VERSION
    std::array<uint8_t, 32> digest;    // SHA-256 of the serialized fields

    HmmSnapshot() : log_zero(0), min_sparse_score(0), min_sparse_prob(0),
                    alphabet(0), semantics_version(PAIR_SEMANTICS_VERSION) {
        start.fill(0);
        trans.fill(0);
        ins.fill(0);
        match.fill(0);
        digest.fill(0);
    }
};

// Serialized byte size of the digest-covered region (little-endian, no padding).
inline size_t HmmSnapshotDigestBytes() {
    return 5 * sizeof(float) + 25 * sizeof(float) + 256 * sizeof(float) +
           65536 * sizeof(float) + 3 * sizeof(float) + 2 * sizeof(uint32_t);
}

// ---------------------------------------------------------------------------
// Sequence pool / jobs / results
// ---------------------------------------------------------------------------

// Parsed bytes exactly as the CPU path sees them: no implicit 20-letter
// recoding, no gap stripping here, original X/Y orientation preserved.
struct SequencePool {
    std::vector<uint8_t> bytes;      // concatenated sequence bytes
    std::vector<uint64_t> offsets;   // n+1 entries, offsets[n] == bytes.size()
    std::vector<uint32_t> lengths;   // n entries, must equal offsets[i+1]-offsets[i]

    size_t SeqCount() const { return lengths.size(); }
    const uint8_t *Seq(uint32_t i) const { return bytes.data() + offsets[i]; }
};

struct PairJob {
    uint64_t job_id;        // unique within the call
    uint32_t pair_index;    // index into the original pair list (commit order)
    uint32_t seq_x;         // index into SequencePool (X < Y by construction)
    uint32_t seq_y;
    uint32_t len_x;
    uint32_t len_y;
};

struct PairResult {
    uint64_t job_id;
    uint32_t pair_index;
    uint32_t len_x;
    uint32_t len_y;
    std::vector<float> post;   // len_x*len_y, row-major, real probability
    float ea;                  // CalcAlnScoreFlat(post)/min(len_x,len_y)
    // Optional diagnostic only: q = F_M(i+1,j+1)+B_M(i+1,j+1)-Z.  Backends fill
    // it when verification/debug is requested; it never carries the result.
    std::vector<float> q;
    BackendKind used;
    PairStatus status;
    std::string reason;        // human-readable, always set for non-Ok

    PairResult()
        : job_id(0), pair_index(0), len_x(0), len_y(0), ea(0),
          used(BackendKind::Cpu), status(PairStatus::CpuFallback) {}
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct BackendConfig {
    BackendKind kind;                     // requested backend
    uint64_t device_budget_bytes;         // total device budget for this process
    uint64_t pinned_budget_bytes;         // bounded pinned host arena
    int device;                           // CUDA device ordinal, validated by backend
    bool verify;                          // per-pair CPU oracle check

    // Implementation-private additions (do not change documented meanings).
    uint64_t batch_budget_bytes;          // per-batch device budget (<= device_budget)
    uint32_t max_jobs_per_batch;          // scheduling bound
    bool debug_dump_fb;                   // debug backends may export F/B copies

    BackendConfig()
        : kind(BackendKind::Cpu), device_budget_bytes(0), pinned_budget_bytes(0),
          device(0), verify(false), batch_budget_bytes(0),
          max_jobs_per_batch(1024), debug_dump_fb(false) {}
};

// ---------------------------------------------------------------------------
// Device byte estimate shared by the scheduler and the CUDA backend
// ---------------------------------------------------------------------------

// v1 prototype workspace per pair: full five-state Forward + full five-state
// Backward in cell-major layout, plus a row-major q buffer and small metadata.
inline uint64_t EstimatePairWorkspaceBytes(uint32_t lx, uint32_t ly) {
    const uint64_t cells = (uint64_t(lx) + 1) * (uint64_t(ly) + 1);
    const uint64_t fb = 2 * 5 * cells * sizeof(float);   // F + B
    const uint64_t q = uint64_t(lx) * uint64_t(ly) * sizeof(float);
    return fb + q + 64;                                  // + per-pair metadata slop
}

// ---------------------------------------------------------------------------
// Backend interface
// ---------------------------------------------------------------------------

class PairBackend {
public:
    virtual ~PairBackend() = default;

    // Synchronous v1: when RunBatch returns, all device->host copies for this
    // batch have completed; no dangling device view escapes.  Results are in
    // input job order.  The model/sequence inputs are immutable and owned by the
    // caller for the duration of the call; device copies and arenas belong to the
    // backend; `post` is owned by the returned vector (never freed with myfree).
    virtual std::vector<PairResult> RunBatch(
        const HmmSnapshot &hmm,
        const SequencePool &pool,
        const std::vector<PairJob> &jobs) = 0;

    virtual const char *BackendName() const = 0;
};

// Optional diagnostics; a backend that cannot dump simply does not implement it.
class PairBackendDebug {
public:
    virtual ~PairBackendDebug() = default;
    // Dump the most recent batch's raw device intermediates (F, B, Z, q) into
    // `dir`.  Used by pair_oracle only.
    virtual bool DumpLastBatch(const std::string &dir, std::string &err) = 0;
    virtual bool DescribeDevice(std::string &json_text, std::string &err) = 0;
};

// ---------------------------------------------------------------------------
// Small checked-size helpers (all multiplies are checked in 64-bit)
// ---------------------------------------------------------------------------

inline bool CheckedMul64(uint64_t a, uint64_t b, uint64_t &out) {
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_mul_overflow(a, b, &out))
        return false;
    return true;
#else
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
        return false;
    out = a * b;
    return true;
#endif
}

} // namespace muscle_gpu
