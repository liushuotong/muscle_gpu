#pragma once

// T03/T06 integration: run the pair-posterior stage of MPCFlat through a
// pluggable backend, and commit the results back with the exact R0 semantics.
//
// Integration rules (HANDOFF_AI.md sections 3 and 4, RESEARCH_PLAN.md section 5):
//   * `-backend cpu` (the default) keeps the original per-pair OpenMP loop; the
//     batch path is only entered when the user asks for gpu/auto or explicitly
//     forces the CPU batch regression switch.
//   * No GPU work ever happens inside the original per-pair OpenMP loop.
//   * A batch is validated and committed as a whole; a failed pair never leaves a
//     half-written sparse matrix or distance-matrix entry behind.
//   * `-backend gpu` with no usable CUDA device/compiler exits non-zero; it must
//     never look like a successful GPU run.
//   * Whole-command CPU fallback is recorded with a reason (Super5, Mega,
//     nucleotide alphabet, ...).  Illegal input is still rejected exactly as the
//     CPU reference rejects it (Die with the original message).

#include <cstdint>
#include <memory>
#include <string>

#include "gpu/pair_backend.h"

class MPCFlat;

namespace muscle_gpu {

struct BackendOptions {
    BackendKind kind;              // resolved from -backend (default Cpu)
    bool verify;                   // -gpu_verify
    bool force_cpu_batch;          // -cpu_batch (regression switch, test only)
    uint64_t device_budget_bytes;  // -gpu_mem_mb N, 0 = documented default
    int device;                    // -gpu_device N
    std::string trace_path;        // -stage_trace FILE ("" = tracing off)
    bool options_valid;

    BackendOptions()
        : kind(BackendKind::Cpu), verify(false), force_cpu_batch(false),
          device_budget_bytes(0), device(0), options_valid(false) {}
};

// Validate the new CLI options (values, ranges) and open the stage trace.
// Invalid values are a command-line error: the caller must fail, not guess.
bool ResolveBackendOptions(BackendOptions &out, std::string &err);

// Entry point used by MPCFlat::CalcPosteriors().
//   returns true  -> every pair has been committed (GPU and/or CPU fallback);
//                    the caller must not run the original loop.
//   returns false -> the caller must run the original CPU loop; `reason`
//                    explains why (already logged once at verbose level).
bool RunPairBackendPosteriors(MPCFlat &msa, std::string &reason);

// Commit one complete (Ok or CpuFallback) result into MPCFlat: validates the
// pair/length mapping, calls MySparseMx::FromPost, sets the borrowed m_X/m_Y
// pointers to the MPCFlat-owned sequence bytes, and fills both EA directions.
bool CommitPairResult(MPCFlat &msa, const PairResult &result, std::string &err);

// One-line summary of what the last RunPairBackendPosteriors did (for -log).
std::string BackendRunSummary();

// Reset the summary counters (tests).
void ResetBackendRunSummary();

} // namespace muscle_gpu
