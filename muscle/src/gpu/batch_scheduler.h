#pragma once

// Batch planning for the pair backends.
//
// Jobs are grouped by size class (ceil_log2 of each length, X and Y kept
// separate because the mathematics is not tie-break symmetric), preserving the
// original job order inside a class, and packed into batches that fit the device
// budget.  A job whose single-pair workspace already exceeds the budget is
// reported as oversize instead of being truncated: the caller must run it on the
// CPU and record the reason.

#include <cstdint>
#include <vector>

#include "gpu/pair_backend.h"

namespace muscle_gpu {

struct PairBatch {
    std::vector<uint32_t> job_indices;  // indices into the input job vector
    uint64_t device_bytes;              // sum of per-pair workspace estimates

    PairBatch() : device_bytes(0) {}
};

// ceil_log2 of a sequence length, used as the size class.  Length 0 -> 0.
uint32_t LengthBin(uint32_t len);

// Group jobs into batches.  `max_jobs_per_batch == 0` means "no explicit bound".
void PlanPairBatches(const std::vector<PairJob> &jobs, uint64_t budget_bytes,
                     uint32_t max_jobs_per_batch, std::vector<PairBatch> &batches,
                     std::vector<uint32_t> &oversize);

} // namespace muscle_gpu
