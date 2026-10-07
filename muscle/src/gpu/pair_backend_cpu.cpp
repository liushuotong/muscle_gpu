#include "gpu/pair_backend_cpu.h"

#include <set>

#include "muscle.h"
#include "gpu/hmm_snapshot.h"
#include "gpu/pair_reference.h"

namespace muscle_gpu {

bool ValidatePoolAndJobs(const SequencePool &pool, const std::vector<PairJob> &jobs,
                         std::string &err) {
    err.clear();
    const size_t n = pool.lengths.size();
    if (pool.offsets.size() != n + 1) {
        err = "SequencePool: offsets.size() != lengths.size()+1";
        return false;
    }
    if (pool.offsets.empty() || pool.offsets.back() != pool.bytes.size()) {
        err = "SequencePool: offsets.back() != bytes.size()";
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (pool.offsets[i] > pool.offsets[i + 1]) {
            err = "SequencePool: offsets are not monotone";
            return false;
        }
        const uint64_t len = pool.offsets[i + 1] - pool.offsets[i];
        if (len != pool.lengths[i]) {
            err = "SequencePool: length does not match offset difference";
            return false;
        }
        if (pool.lengths[i] == 0) {
            err = "SequencePool: zero-length sequence";
            return false;
        }
    }
    std::set<uint64_t> ids;
    for (size_t k = 0; k < jobs.size(); ++k) {
        const PairJob &j = jobs[k];
        if (!ids.insert(j.job_id).second) {
            err = "PairJob: duplicate job_id";
            return false;
        }
        if (j.seq_x >= n || j.seq_y >= n) {
            err = "PairJob: sequence index out of range";
            return false;
        }
        if (j.seq_x == j.seq_y) {
            err = "PairJob: seq_x == seq_y (pairs are X<Y and never self-pairs)";
            return false;
        }
        if (j.len_x != pool.lengths[j.seq_x] || j.len_y != pool.lengths[j.seq_y]) {
            err = "PairJob: length does not match pool length";
            return false;
        }
        if (!PairLengthsAllowed(j.len_x, j.len_y)) {
            err = "PairJob: lengths exceed the CPU reference overflow gate";
            return false;
        }
    }
    return true;
}

namespace {

class CpuPairBackend : public PairBackend {
public:
    explicit CpuPairBackend(const BackendConfig &config) : m_Config(config) {}

    const char *BackendName() const override { return "cpu-batch"; }

    std::vector<PairResult> RunBatch(const HmmSnapshot &hmm, const SequencePool &pool,
                                     const std::vector<PairJob> &jobs) override {
        std::vector<PairResult> results;
        results.reserve(jobs.size());

        std::string err;
        if (!ValidatePoolAndJobs(pool, jobs, err)) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::InvalidInput, err));
            return results;
        }

        // The R0 DP routines read the global PairHMM tables; a batch is only valid
        // while those tables are bitwise identical to the snapshot.  All checks
        // happen before the OpenMP region and nothing may mutate the model until
        // the batch has completed.
        if (!HmmSnapshotMatchesCurrent(hmm, err)) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::CpuFallback,
                                             "snapshot mismatch: " + err));
            return results;
        }

        results.resize(jobs.size());
        for (size_t k = 0; k < jobs.size(); ++k)
            results[k] = MakeResult(jobs[k], PairStatus::CpuFallback, "not computed");

        unsigned threads = GetRequestedThreadCount();
        if (threads == 0)
            threads = 1;
        if (threads > jobs.size())
            threads = unsigned(jobs.size());

        const bool keep_fb = m_Config.debug_dump_fb;

#pragma omp parallel for num_threads(threads)
        for (long long k = 0; k < (long long)jobs.size(); ++k) {
            const PairJob &job = jobs[size_t(k)];
            PairResult &r = results[size_t(k)];
            ReferencePairOutput out;
            std::string perr;
            const uint8_t *x = pool.Seq(job.seq_x);
            const uint8_t *y = pool.Seq(job.seq_y);
            if (!RunReferencePair(x, job.len_x, y, job.len_y, keep_fb, false, out, perr)) {
                r.status = PairStatus::InvalidInput;
                r.reason = perr;
                continue;
            }
            r.post = out.post;
            r.ea = out.ea;
            r.status = PairStatus::Ok;
            r.used = BackendKind::Cpu;
            r.reason.clear();
        }
        return results;
    }

private:
    static PairResult MakeResult(const PairJob &job, PairStatus status,
                                 const std::string &reason) {
        PairResult r;
        r.job_id = job.job_id;
        r.pair_index = job.pair_index;
        r.len_x = job.len_x;
        r.len_y = job.len_y;
        r.used = BackendKind::Cpu;
        r.status = status;
        r.reason = reason;
        return r;
    }

    BackendConfig m_Config;
};

} // namespace

std::unique_ptr<PairBackend> CreateCpuPairBackend(const BackendConfig &config,
                                                  std::string &err) {
    err.clear();
    return std::unique_ptr<PairBackend>(new CpuPairBackend(config));
}

} // namespace muscle_gpu
