#pragma once

#include <memory>
#include <string>

#include "gpu/pair_backend.h"

namespace muscle_gpu {

// CPU batch backend: runs the frozen R0 DP routines over a batch of jobs.
//
// It refuses (CpuFallback for the whole batch, with a reason) when the installed
// global PairHMM tables differ bitwise from the snapshot, because the R0 DP
// routines read those globals and a mixed batch would silently mix models.  It
// does not reset the model or the RNG, and it never mutates PairHMM.
std::unique_ptr<PairBackend> CreateCpuPairBackend(const BackendConfig &config,
                                                  std::string &err);

// Validate a pool/job set against the documented contract.  Shared with the
// CUDA backend and the tests so that both accept exactly the same inputs.
bool ValidatePoolAndJobs(const SequencePool &pool, const std::vector<PairJob> &jobs,
                         std::string &err);

} // namespace muscle_gpu
