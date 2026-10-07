// Modified 2026-10-08: Super5 CUDA adapter; GPLv3, see NOTICE.md.
#include "muscle.h"
#include "mega.h"
#include "gpu/super5_adapter.h"
#include "gpu/backend_registry.h"
#include "gpu/dispatch.h"
#include "gpu/hmm_snapshot.h"
#include "gpu/pair_backend_cpu.h"
#include "gpu/pair_reference.h"
#include "gpu/stage_trace.h"
#include <algorithm>
#include <cstring>
#include <map>

namespace muscle_gpu {
void PrepareSuper5Backend() {
    BackendOptions opts; std::string err;
    if (!ResolveBackendOptions(opts,err)) Die("%s",err.c_str());
    StageTrace &trace=StageTrace::Instance();
    if (!opts.trace_path.empty() && !trace.IsOpen()) {
        if (!trace.Open(opts.trace_path,err)) Die("%s",err.c_str());
        trace.SetRunId("super5");
        trace.SetFamilyId(optd(super5,optd(super4,std::string("input"))));
    }
    if (opts.kind==BackendKind::Gpu) {
        if (opts.force_cpu_batch) Die("-backend gpu cannot be combined with -cpu_batch");
        if (!CudaAvailable(err)) Die("Super5 GPU requested: %s",err.c_str());
        if (g_Alpha!=ALPHA_Amino || Mega::m_Loaded)
            ProgressLog("Super5 CPU fallback: unsupported alphabet/profile\n");
        else ProgressLog("Super5 hybrid CUDA: UClust/PPP/consensus/PProg posterior stages; EACluster decisions remain CPU\n");
    }
}
void RecordSuper5Partition(const char *stage,const std::vector<unsigned> &membership,
    const std::vector<std::string> &paths) {
    StageTrace &trace=StageTrace::Instance();
    if (!trace.IsOpen()) return;
    std::string data;
    for (unsigned x:membership) data+=std::to_string(x)+",";
    for (const auto &p:paths) data+=std::to_string(p.size())+":"+p;
    StageRecord rec; rec.stage=std::string(stage)+":"+Sha256::OfBuffer(data.data(),data.size());
    rec.backend="cpu-decision"; rec.cells=membership.size();
    trace.Record(rec);
}
bool RunSuper5Pairs(const char *stage,
    const std::vector<std::pair<std::string, std::string>> &labels,
    const std::function<void(size_t, const float *, unsigned, unsigned)> &consume) {
    if (!(optset_super5 || optset_super4) || labels.empty()) return false;
    BackendOptions opts;
    std::string err;
    if (!ResolveBackendOptions(opts, err)) Die("%s", err.c_str());
    if (opts.kind != BackendKind::Gpu) return false;
    if (opts.force_cpu_batch) Die("-backend gpu cannot be combined with -cpu_batch");
    if (g_Alpha != ALPHA_Amino || Mega::m_Loaded) return false;
    if (!CudaAvailable(err)) Die("Super5 GPU requested: %s", err.c_str());
    StageTrace &trace = StageTrace::Instance();
    if (!opts.trace_path.empty() && !trace.IsOpen()) {
        if (!trace.Open(opts.trace_path, err)) Die("%s", err.c_str());
        trace.SetRunId("super5");
        trace.SetFamilyId(optd(super5, optd(super4, std::string("input"))));
    }
    BackendConfig config;
    config.kind = BackendKind::Gpu;
    config.device = opts.device;
    config.device_budget_bytes = opts.device_budget_bytes;
    config.device_budget_bytes = ResolveDeviceBudget(config, err);
    if (!config.device_budget_bytes) Die("Super5 GPU budget: %s", err.c_str());
    config.batch_budget_bytes = config.device_budget_bytes;
    config.verify = opts.verify;
    // Reuse one backend/context/workspace across sequential Super5 stages. Each
    // RunBatch uploads the ACTUAL current model; no HMM/posterior cache exists.
    static std::unique_ptr<PairBackend> backend;
    static int device = -1;
    static uint64_t budget = 0;
    static bool verify = false;
    if (!backend || device != config.device || budget != config.device_budget_bytes ||
        verify != config.verify) {
        backend = CreatePairBackend(config, err);
        if (!backend) Die("Super5 GPU backend: %s", err.c_str());
        device = config.device; budget = config.device_budget_bytes; verify = config.verify;
    }
    HmmSnapshot snap;
    if (!CaptureHmmSnapshot(snap, err)) Die("Super5 HMM snapshot: %s", err.c_str());
    uint64_t gpu_pairs = 0, fallback = 0;
    // Bounded 64-job / 64 MiB pair workspace chunks; sequence pool includes only
    // the current chunk. Avoid retaining quadratic host Post for the whole job.
    size_t first = 0;
    while (first < labels.size()) {
        SequencePool pool;
        pool.offsets.push_back(0);
        std::map<std::string, uint32_t> indexes;
        std::vector<PairJob> jobs;
        uint64_t bytes = 0;
        auto add = [&](const std::string &label) -> uint32_t {
            auto it = indexes.find(label);
            if (it != indexes.end()) return it->second;
            const uint32_t id = uint32_t(pool.lengths.size());
            const uint32_t n = GetSeqLengthByGlobalLabel(label);
            const byte *p = GetGlobalByteSeqByLabel(label);
            indexes[label] = id;
            pool.bytes.insert(pool.bytes.end(), p, p + n);
            pool.lengths.push_back(n); pool.offsets.push_back(pool.bytes.size());
            return id;
        };
        size_t last = first;
        for (; last < labels.size() && jobs.size() < 64; ++last) {
            const auto &pr = labels[last];
            uint32_t lx = GetSeqLengthByGlobalLabel(pr.first);
            uint32_t ly = GetSeqLengthByGlobalLabel(pr.second);
            if (!PairLengthsAllowed(lx, ly))
                Die("HMM overflow, sequence lengths %u, %u (max ~21k)", lx, ly);
            uint64_t need = EstimatePairWorkspaceBytes(lx, ly);
            if (!jobs.empty() && bytes + need > 64ull * 1024 * 1024) break;
            PairJob job;
            job.job_id = last; job.pair_index = uint32_t(last);
            job.seq_x = add(pr.first); job.seq_y = add(pr.second);
            job.len_x = lx; job.len_y = ly;
            jobs.push_back(job); bytes += need;
        }
        std::vector<PairResult> results;
        {
            StageTrace::Scope scope(stage, "cuda-pair-v1");
            scope.SetCells(jobs.size());
            results = backend->RunBatch(snap, pool, jobs);
        }
        if (results.size() != jobs.size()) Die("Super5 GPU result count mismatch");
        for (size_t k = 0; k < results.size(); ++k) {
            PairResult &r = results[k];
            const PairJob &job = jobs[k];
            if (r.job_id != job.job_id || r.pair_index != job.pair_index ||
                r.len_x != job.len_x || r.len_y != job.len_y)
                Die("Super5 GPU result identity mismatch");
            if (r.status == PairStatus::DeviceError || r.status == PairStatus::InvalidInput)
                Die("Super5 GPU: %s", r.reason.c_str());
            if (r.status == PairStatus::Ok &&
                r.post.size() != size_t(job.len_x) * job.len_y)
                Die("Super5 GPU Post size mismatch");
            if (r.status == PairStatus::Ok && opts.verify) {
                ReferencePairOutput ref;
                if (!RunReferencePair(pool.Seq(job.seq_x), job.len_x,
                        pool.Seq(job.seq_y), job.len_y, false, false, ref, err))
                    Die("Super5 GPU oracle: %s", err.c_str());
                if (r.post.size() != ref.post.size() ||
                    memcmp(r.post.data(), ref.post.data(), ref.post.size()*sizeof(float)))
                    Die("Super5 GPU verification failed at pair %u", job.pair_index);
            }
            StageTrace::Scope commit("super5.pair.commit",
                r.status == PairStatus::Ok ? "cuda-pair-v1" : "cpu");
            commit.SetPairIndex(job.pair_index);
            commit.SetCells(uint64_t(job.len_x)*job.len_y);
            if (r.status == PairStatus::Ok) {
                ++gpu_pairs;
                consume(job.pair_index, r.post.data(), job.len_x, job.len_y);
            } else {
                ++fallback; commit.SetFallbackReason(r.reason);
                ProgressLog("Super5 CPU fallback pair %u: %s\n",job.pair_index,r.reason.c_str());
                float *post = CalcPost(labels[job.pair_index].first, labels[job.pair_index].second);
                consume(job.pair_index, post, job.len_x, job.len_y);
                myfree(post);
            }
        }
        first = last;
    }
    ProgressLog("%s: cuda-pair-v1 gpu_pairs=%llu cpu_fallback=%llu verify=%s\n",
        stage, (unsigned long long)gpu_pairs, (unsigned long long)fallback,
        opts.verify ? "bitwise-post" : "off");
    return true;
}
}
