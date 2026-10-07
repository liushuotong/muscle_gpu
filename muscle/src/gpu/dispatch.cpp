#include "gpu/dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "muscle.h"
#include "mega.h"

#include "gpu/backend_registry.h"
#include "gpu/batch_scheduler.h"
#include "gpu/hmm_snapshot.h"
#include "gpu/pair_backend.h"
#include "gpu/pair_backend_cpu.h"
#include "gpu/pair_reference.h"
#include "gpu/posterior_finalize.h"
#include "gpu/stage_trace.h"

// Defined in myutils.cpp (muscle_core); declared here instead of in a header so
// that the shared library does not depend on the main translation unit.
extern std::vector<std::string> g_Argv;

namespace muscle_gpu {

namespace {

struct RunCounters {
    uint64_t pairs_total;
    uint64_t pairs_ok;
    uint64_t pairs_cpu_fallback;
    uint64_t verify_checked;
    uint64_t verify_failed;
    uint64_t batches;
    uint64_t device_bytes;
    uint64_t oversized_pairs;

    RunCounters() { Reset(); }
    void Reset() { memset(this, 0, sizeof(*this)); }
};

RunCounters g_Counters;
std::string g_Summary;

const uint64_t MIB = 1024ull * 1024ull;
const uint64_t CPU_BATCH_DEFAULT_BUDGET = 256 * MIB;

std::string BaseNameOfPath(const std::string &path) {
    const size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

void RecordFallback(const char *what, const std::string &reason,
                    uint64_t pairs = 0) {
    // Visible by default: ProgressLog goes to stderr unless -quiet, and Log adds
    // the -log copy.  A whole-command fallback must never be silent.
    ProgressLog("Backend: %s -> CPU (%s)\n", what, reason.c_str());
    Log("Backend: %s -> CPU (%s)\n", what, reason.c_str());
    StageTrace &trace = StageTrace::Instance();
    if (trace.IsOpen()) {
        StageRecord rec;
        rec.stage = "pair_backend.selected";
        rec.backend = "cpu";
        rec.thread_id = StageThreadId();
        rec.cells = pairs;
        rec.fallback_reason = std::string(what) + ": " + reason;
        trace.Record(rec);
    }
}

} // namespace

bool ResolveBackendOptions(BackendOptions &out, std::string &err) {
    out = BackendOptions();
    err.clear();

    if (optset_backend) {
        const string &s = opt(backend);
        if (s == "cpu")
            out.kind = BackendKind::Cpu;
        else if (s == "gpu")
            out.kind = BackendKind::Gpu;
        else if (s == "auto")
            out.kind = BackendKind::Auto;
        else {
            err = "invalid -backend value >" + s + "< (expected cpu, gpu or auto)";
            return false;
        }
    }

    if (optset_gpu_device) {
        const unsigned d = opt(gpu_device);
        if (d > 64) {
            err = "-gpu_device out of range (0..64)";
            return false;
        }
        out.device = int(d);
    }

    if (optset_gpu_mem_mb) {
        const unsigned mb = opt(gpu_mem_mb);
        if (mb == 0) {
            err = "-gpu_mem_mb must be at least 1 MiB";
            return false;
        }
        out.device_budget_bytes = uint64_t(mb) * MIB;
    }

    out.verify = optd(gpu_verify, false);
    out.force_cpu_batch = optd(cpu_batch, false);
    if (optset_stage_trace)
        out.trace_path = opt(stage_trace);

    out.options_valid = true;
    return true;
}

bool CommitPairResult(MPCFlat &msa, const PairResult &result, std::string &err) {
    err.clear();
    if (result.status != PairStatus::Ok && result.status != PairStatus::CpuFallback) {
        err = "refusing to commit an incomplete pair result";
        return false;
    }
    if (result.pair_index >= msa.m_Pairs.size()) {
        err = "commit: pair_index out of range";
        return false;
    }
    const pair<uint, uint> &pr = msa.GetPair(result.pair_index);
    const uint lx = msa.GetSeqLength(pr.first);
    const uint ly = msa.GetSeqLength(pr.second);
    if (result.len_x != lx || result.len_y != ly) {
        err = "commit: result lengths do not match the MPCFlat pair";
        return false;
    }
    uint64_t cells = 0;
    if (!CheckedMul64(lx, ly, cells) || cells != result.post.size()) {
        err = "commit: post size does not match len_x*len_y";
        return false;
    }
    for (size_t k = 0; k < result.post.size(); ++k) {
        const float p = result.post[k];
        if (!(p >= 0.0f && p <= 1.0f)) {
            err = "commit: post value outside [0,1]";
            return false;
        }
    }
    if (!std::isfinite(result.ea)) {
        err = "commit: EA is not finite";
        return false;
    }

    // Same sequence of operations as MPCFlat::CalcPosterior once Post exists.
    MySparseMx &SparsePost = msa.GetSparsePost(result.pair_index);
    SparsePost.FromPost(result.post.data(), lx, ly);
    SparsePost.m_X = msa.GetBytePtr(pr.first);
    SparsePost.m_Y = msa.GetBytePtr(pr.second);

    msa.m_DistMx[pr.first][pr.second] = result.ea;
    msa.m_DistMx[pr.second][pr.first] = result.ea;
    return true;
}

std::string BackendRunSummary() { return g_Summary; }

void ResetBackendRunSummary() {
    g_Counters.Reset();
    g_Summary.clear();
}

namespace {

// Recompute the pair on the CPU and compare against the candidate result.
bool VerifyPairResult(const PairResult &candidate, const HmmSnapshot &snap,
                      const SequencePool &pool, const PairJob &job,
                      PairGateReport &report, std::string &err) {
    ReferencePairOutput ref;
    if (!RunReferencePair(pool.Seq(job.seq_x), job.len_x, pool.Seq(job.seq_y),
                          job.len_y, false, true, ref, err))
        return false;

    float min_sparse_prob = snap.min_sparse_prob;
    const float *q_cand = (candidate.q.size() == ref.q.size() && !candidate.q.empty())
                              ? candidate.q.data()
                              : nullptr;
    PairGateLimits limits;
    const bool ok = ComparePairOutputs(q_cand, ref.q.data(), candidate.post.data(),
                                       ref.post.data(), candidate.len_x,
                                       candidate.len_y, candidate.ea, ref.ea,
                                       min_sparse_prob, limits, report);
    return ok;
}

PairResult MakeFallbackResult(const PairJob &job, PairStatus status,
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

} // namespace

bool RunPairBackendPosteriors(MPCFlat &msa, std::string &reason) {
    reason.clear();

    BackendOptions opts;
    std::string err;
    if (!ResolveBackendOptions(opts, err))
        Die("%s", err.c_str());

    // The stage trace is opened before any early return so that -stage_trace also
    // works for the CPU reference path (T01); it never writes into FASTA/EFA.
    StageTrace &trace = StageTrace::Instance();
    if (!opts.trace_path.empty() && !trace.IsOpen()) {
        std::string terr;
        if (!trace.Open(opts.trace_path, terr))
            Die("%s", terr.c_str());
        // The family identity is the aligned input file.  `-align FILE` passes it
        // as the first positional argument (g_Argv[2]); commands that take
        // -input use that option instead.  g_Argv is defined in myutils.cpp, so
        // this does not create a dependency on the main translation unit.
        std::string family = optd(input, std::string());
        if (family.empty() && g_Argv.size() > 2 && !g_Argv[2].empty() &&
            g_Argv[2][0] != '-')
            family = g_Argv[2];
        if (family.empty())
            family = "input";
        family = BaseNameOfPath(family);
        trace.SetRunId(family);
        trace.SetFamilyId(family);
        trace.SetReplicateId(-1); // replicate identity is owned by cmd_align (T01 follow-up)
    }

    // Until the T07 break-even table exists, `auto` is defined to be exactly the
    // CPU reference path, so no benchmark can be mislabelled as GPU.  The batch
    // path is only entered for an explicit gpu request or for the regression
    // switch -cpu_batch.
    if (!opts.force_cpu_batch &&
        (opts.kind == BackendKind::Cpu || opts.kind == BackendKind::Auto)) {
        reason = (opts.kind == BackendKind::Cpu)
                     ? "backend cpu: original per-pair loop"
                     : "backend auto == cpu until T07: original per-pair loop";
        if (trace.IsOpen()) {
            StageRecord rec;
            rec.stage = "pair_backend.selected";
            rec.backend = "cpu";
            rec.thread_id = StageThreadId();
            rec.fallback_reason = reason;
            trace.Record(rec);
        }
        return false;
    }

    // An explicit -backend gpu is never silently downgraded.  Asking for both an
    // explicit GPU run and the CPU regression switch is a contradiction, and a
    // missing device must fail loudly rather than look like a successful run.
    if (opts.kind == BackendKind::Gpu) {
        if (opts.force_cpu_batch)
            Die("-backend gpu cannot be combined with -cpu_batch "
                "(the latter forces the CPU batch backend)");
        std::string why;
        if (!CudaAvailable(why))
            Die("-backend gpu was requested but %s", why.c_str());
    }
    const bool gpu_requested = (opts.kind == BackendKind::Gpu);
    const bool verbose = optd(verbose, false);

    // Super5/Super4 cluster-local MPCFlat uses the same posterior backend.
    if (Mega::m_Loaded) {
        reason = "Mega input is CPU-only (GPU backend handles ordinary amino-acid pairs)";
        RecordFallback("mega", reason);
        return false;
    }
    if (g_Alpha != ALPHA_Amino) {
        reason = "non-amino alphabet is CPU-only in this version";
        RecordFallback("alphabet", reason);
        return false;
    }

    const uint SeqCount = msa.GetSeqCount();
    if (SeqCount < 2 || msa.m_Pairs.empty()) {
        reason = "no pairs to compute";
        return false;
    }

    HmmSnapshot snap;
    if (!CaptureHmmSnapshot(snap, err)) {
        if (gpu_requested)
            Die("-backend gpu: cannot snapshot the Pair-HMM tables: %s", err.c_str());
        reason = "cannot snapshot the Pair-HMM tables: " + err;
        RecordFallback("snapshot", reason);
        return false;
    }

    // ---- immutable sequence pool (unique sequences, original bytes) ----
    SequencePool pool;
    pool.offsets.push_back(0);
    for (uint i = 0; i < SeqCount; ++i) {
        const uint len = msa.GetSeqLength(i);
        const byte *bytes = msa.GetBytePtr(i);
        pool.bytes.insert(pool.bytes.end(), bytes, bytes + len);
        pool.lengths.push_back(len);
        pool.offsets.push_back(uint64_t(pool.bytes.size()));
    }

    // ---- jobs in the original pair order (m_Pairs is X<Y lexicographic) ----
    std::vector<PairJob> jobs;
    jobs.reserve(msa.m_Pairs.size());
    for (size_t k = 0; k < msa.m_Pairs.size(); ++k) {
        const pair<uint, uint> &pr = msa.m_Pairs[k];
        PairJob job;
        job.job_id = uint64_t(k);
        job.pair_index = uint32_t(k);
        job.seq_x = pr.first;
        job.seq_y = pr.second;
        job.len_x = msa.GetSeqLength(pr.first);
        job.len_y = msa.GetSeqLength(pr.second);
        jobs.push_back(job);
    }

    // ---- configuration / backend ----
    BackendConfig bcfg;
    bcfg.kind = gpu_requested ? BackendKind::Gpu : BackendKind::Cpu;
    bcfg.verify = opts.verify;
    bcfg.device = opts.device;
    bcfg.debug_dump_fb = false;

    if (gpu_requested) {
        std::string berr;
        const uint64_t budget = ResolveDeviceBudget(bcfg, berr);
        if (budget == 0)
            Die("-backend gpu: cannot resolve the device memory budget: %s", berr.c_str());
        bcfg.device_budget_bytes = budget;
        bcfg.batch_budget_bytes = budget;
        bcfg.pinned_budget_bytes = std::min<uint64_t>(budget / 8, 1024 * MIB);
    } else {
        bcfg.device_budget_bytes = 0;
        bcfg.batch_budget_bytes =
            (opts.device_budget_bytes != 0) ? opts.device_budget_bytes : CPU_BATCH_DEFAULT_BUDGET;
        bcfg.pinned_budget_bytes = 0;
    }

    std::unique_ptr<PairBackend> backend = CreatePairBackend(bcfg, err);
    if (!backend) {
        if (gpu_requested)
            Die("-backend gpu: cannot create the CUDA backend: %s", err.c_str());
        reason = "cannot create backend: " + err;
        RecordFallback("backend", reason);
        return false;
    }

    {
        std::string verr;
        if (!ValidatePoolAndJobs(pool, jobs, verr))
            Die("internal backend input error: %s", verr.c_str());
    }

    g_Counters.Reset();
    g_Counters.pairs_total = jobs.size();

    std::vector<PairBatch> batches;
    std::vector<uint32_t> oversize;
    PlanPairBatches(jobs, bcfg.batch_budget_bytes, bcfg.max_jobs_per_batch, batches, oversize);
    g_Counters.batches = batches.size();
    g_Counters.oversized_pairs = oversize.size();

    std::vector<PairResult> committed;
    committed.reserve(jobs.size());
    std::set<uint32_t> fallback_pairs;

    const char *backend_name = backend->BackendName();

    for (size_t b = 0; b < batches.size(); ++b) {
        std::vector<PairJob> batch_jobs;
        batch_jobs.reserve(batches[b].job_indices.size());
        for (size_t k = 0; k < batches[b].job_indices.size(); ++k)
            batch_jobs.push_back(jobs[batches[b].job_indices[k]]);

        g_Counters.device_bytes += batches[b].device_bytes;

        uint64_t batch_cells = 0;
        for (size_t k = 0; k < batch_jobs.size(); ++k)
            batch_cells += uint64_t(batch_jobs[k].len_x) * batch_jobs[k].len_y;

        std::vector<PairResult> results;
        {
            StageTrace::Scope scope("pair_backend.run_batch", backend_name);
            scope.SetBytes(batches[b].device_bytes);
            scope.SetCells(batch_cells);
            results = backend->RunBatch(snap, pool, batch_jobs);
        }

        if (results.size() != batch_jobs.size()) {
            if (gpu_requested)
                Die("-backend gpu: backend returned %u results for %u jobs",
                    unsigned(results.size()), unsigned(batch_jobs.size()));
            reason = "backend returned a wrong result count";
            RecordFallback("result-count", reason);
            return false;
        }

        for (size_t k = 0; k < results.size(); ++k) {
            PairResult &r = results[k];
            const PairJob &job = batch_jobs[k];
            if (r.job_id != job.job_id || r.pair_index != job.pair_index) {
                Die("backend result identity mismatch (job %llu -> %llu)",
                    (unsigned long long)job.job_id, (unsigned long long)r.job_id);
            }

            switch (r.status) {
            case PairStatus::Ok:
                break;
            case PairStatus::InvalidInput: {
                // The CPU reference rejects these inputs; the backend must not
                // silently approximate or truncate them.  Emit the same messages
                // the reference emits (calcposteriorflat.cpp::CalcPosterior).
                const pair<uint, uint> &pr = msa.GetPair(job.pair_index);
                if (!PairLengthsAllowed(job.len_x, job.len_y)) {
                    ProgressLog("\nSequence length %u >%s\n", job.len_x,
                                msa.GetLabel(pr.first));
                    ProgressLog("Sequence length %u >%s\n", job.len_y,
                                msa.GetLabel(pr.second));
                    Die("HMM overflow, sequence lengths %u, %u (max ~21k)", job.len_x,
                        job.len_y);
                }
                Die("invalid backend input for pair %u: %s", job.pair_index,
                    r.reason.empty() ? "unspecified" : r.reason.c_str());
                break;
            }
            case PairStatus::CpuFallback:
            case PairStatus::ResourceError:
                ++g_Counters.pairs_cpu_fallback;
                fallback_pairs.insert(job.pair_index);
                if (g_Counters.pairs_cpu_fallback <= 5) {
                    ProgressLog("CPU fallback pair %u: %s\n", job.pair_index,
                                r.reason.c_str());
                } else if (g_Counters.pairs_cpu_fallback == 6) {
                    ProgressLog("CPU fallback: further reasons are only counted "
                                "(see -stage_trace / the run summary)\n");
                }
                r.status = PairStatus::CpuFallback;
                break;
            case PairStatus::DeviceError:
                Die("-backend gpu: fatal device error on pair %u: %s "
                    "(restart the task on the CPU; results are not committed)",
                    job.pair_index, r.reason.c_str());
                break;
            }

            if (r.status == PairStatus::Ok && opts.verify) {
                ++g_Counters.verify_checked;
                PairGateReport report;
                std::string verr;
                if (!VerifyPairResult(r, snap, pool, job, report, verr)) {
                    ++g_Counters.verify_failed;
                    if (!verr.empty()) {
                        report.reason = report.reason.empty()
                                            ? verr
                                            : (verr + "; " + report.reason);
                    }
                    if (report.reason.empty())
                        report.reason = "verification failed";
                    if (verbose)
                        Log("gpu_verify FAIL pair %u: %s\n", job.pair_index,
                            FormatGateReport(report).c_str());
                    r.status = PairStatus::CpuFallback;
                    r.reason = "gpu_verify: " + report.reason;
                    ++g_Counters.pairs_cpu_fallback;
                    fallback_pairs.insert(job.pair_index);
                }
            }
            if (r.status == PairStatus::Ok)
                ++g_Counters.pairs_ok;
            committed.push_back(r);
        }
    }

    for (size_t k = 0; k < oversize.size(); ++k) {
        const PairJob &job = jobs[oversize[k]];
        ++g_Counters.pairs_cpu_fallback;
        fallback_pairs.insert(job.pair_index);
        committed.push_back(MakeFallbackResult(
            job, PairStatus::CpuFallback,
            "pair workspace exceeds the configured device budget"));
    }

    // ---- commit in original pair order; CPU fallbacks re-run the R0 path ----
    std::stable_sort(committed.begin(), committed.end(),
                     [](const PairResult &a, const PairResult &b) {
                         return a.pair_index < b.pair_index;
                     });

    const uint PairCount = uint(msa.m_Pairs.size());
    uint PairCounter = 0;
    for (size_t k = 0; k < committed.size(); ++k) {
        const PairResult &r = committed[k];
        ProgressStep(PairCounter++, PairCount, "Calc posteriors");

        if (r.status == PairStatus::CpuFallback) {
            if (verbose)
                Log("CPU fallback pair %u: %s\n", r.pair_index, r.reason.c_str());
            StageTrace::Scope scope("pair_backend.commit", "cpu");
            scope.SetPairIndex(r.pair_index);
            scope.SetFallbackReason(r.reason);
            msa.CalcPosterior(r.pair_index);
            continue;
        }
        std::string cerr;
        if (!CommitPairResult(msa, r, cerr))
            Die("backend commit failed for pair %u: %s", r.pair_index, cerr.c_str());
        {
            StageTrace::Scope scope("pair_backend.commit", backend_name);
            scope.SetPairIndex(r.pair_index);
            scope.SetCells(uint64_t(r.len_x) * uint64_t(r.len_y));
            scope.SetNnz(msa.GetSparsePost(r.pair_index).m_Offsets[r.len_x]);
        }
    }

    if (committed.size() != msa.m_Pairs.size()) {
        Die("backend committed %u of %u pairs; refusing to continue with an "
            "incomplete posterior set",
            unsigned(committed.size()), unsigned(msa.m_Pairs.size()));
    }

    {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "backend=%s batches=%llu pairs=%llu ok=%llu cpu_fallback=%llu "
                 "oversize=%llu verify_checked=%llu verify_failed=%llu device_bytes=%llu",
                 backend_name, (unsigned long long)g_Counters.batches,
                 (unsigned long long)g_Counters.pairs_total,
                 (unsigned long long)g_Counters.pairs_ok,
                 (unsigned long long)g_Counters.pairs_cpu_fallback,
                 (unsigned long long)g_Counters.oversized_pairs,
                 (unsigned long long)g_Counters.verify_checked,
                 (unsigned long long)g_Counters.verify_failed,
                 (unsigned long long)g_Counters.device_bytes);
        g_Summary = buf;
        if (!fallback_pairs.empty())
            g_Summary += " fallback_pairs=" + std::to_string(fallback_pairs.size());
    }
    ProgressLog("%s\n", g_Summary.c_str());
    Log("Backend: %s\n", g_Summary.c_str());
    return true;
}

} // namespace muscle_gpu
