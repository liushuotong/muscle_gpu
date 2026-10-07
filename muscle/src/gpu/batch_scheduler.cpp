#include "gpu/batch_scheduler.h"

#include <algorithm>

namespace muscle_gpu {

uint32_t LengthBin(uint32_t len) {
    uint32_t bin = 0;
    while ((uint64_t(1) << bin) < uint64_t(len) && bin < 63)
        ++bin;
    return bin;
}

namespace {

struct SortKey {
    uint32_t bin_x;
    uint32_t bin_y;
    uint32_t pair_index;
    uint32_t job_index;

    bool operator<(const SortKey &rhs) const {
        if (bin_x != rhs.bin_x) return bin_x < rhs.bin_x;
        if (bin_y != rhs.bin_y) return bin_y < rhs.bin_y;
        if (pair_index != rhs.pair_index) return pair_index < rhs.pair_index;
        return job_index < rhs.job_index;
    }
};

} // namespace

void PlanPairBatches(const std::vector<PairJob> &jobs, uint64_t budget_bytes,
                     uint32_t max_jobs_per_batch, std::vector<PairBatch> &batches,
                     std::vector<uint32_t> &oversize) {
    batches.clear();
    oversize.clear();
    if (jobs.empty())
        return;

    std::vector<SortKey> keys;
    keys.reserve(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        SortKey k;
        k.bin_x = LengthBin(jobs[i].len_x);
        k.bin_y = LengthBin(jobs[i].len_y);
        k.pair_index = jobs[i].pair_index;
        k.job_index = uint32_t(i);
        keys.push_back(k);
    }
    std::stable_sort(keys.begin(), keys.end());

    PairBatch current;
    for (size_t i = 0; i < keys.size(); ++i) {
        const uint32_t job_index = keys[i].job_index;
        const PairJob &job = jobs[job_index];
        const uint64_t pair_bytes = EstimatePairWorkspaceBytes(job.len_x, job.len_y);
        if (pair_bytes > budget_bytes) {
            oversize.push_back(job_index);
            continue;
        }
        if (!current.job_indices.empty() &&
            (current.device_bytes + pair_bytes > budget_bytes ||
             (max_jobs_per_batch != 0 &&
              current.job_indices.size() >= max_jobs_per_batch))) {
            batches.push_back(current);
            current = PairBatch();
        }
        current.job_indices.push_back(job_index);
        current.device_bytes += pair_bytes;
    }
    if (!current.job_indices.empty())
        batches.push_back(current);
}

} // namespace muscle_gpu
