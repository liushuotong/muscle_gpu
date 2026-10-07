// CUDA pair backend v1 (T04/T05/T06).
//
// Scope: the ordinary amino-acid pair posterior used by MPCFlat::CalcPosteriors.
// Everything after q (the two thresholds, expf, the sparse support set, the EA
// score, the tree, consistency, progressive alignment and refinement) stays on
// the CPU, exactly as RESEARCH_PLAN.md section 5 requires.
//
// Ownership and lifetime rules:
//   * one context/stream/arena per backend instance; no per-pair context or
//     allocation, and no cross-thread submission (the coordinator thread calls
//     RunBatch from MPCFlat::CalcPosteriors);
//   * the HMM snapshot, sequence pool and job list are borrowed read-only for the
//     duration of the call; device copies are re-uploaded per batch so that a
//     model change can never be silently mixed into a batch;
//   * RunBatch is synchronous: it returns only after every device->host copy has
//     completed, so no device view escapes.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gpu/backend_registry.h"
#include "gpu/cuda_context.h"
#include "gpu/hmm_snapshot.h"
#include "gpu/pair_backend_cpu.h"
#include "gpu/pair_kernels.cuh"
#include "gpu/pair_reference.h"
#include "gpu/posterior_finalize.h"
#include "gpu/stage_trace.h"

namespace muscle_gpu {

namespace {

const int kBlockThreads = 128;
const uint64_t kModelBytes = uint64_t(65536 + 256) * sizeof(float);
const size_t kAlign = 256;

struct ArenaLayout {
    uint64_t fwd_off;   // element offsets (floats) into the float region
    uint64_t bwd_off;
    uint64_t q_off;
};

class CudaPairBackend : public PairBackend, public PairBackendDebug {
public:
    explicit CudaPairBackend(const BackendConfig &config)
        : m_Config(config), m_LastDumpValid(false) {}

    ~CudaPairBackend() override {}

    bool Init(std::string &err) {
        if (!m_Ctx.Init(m_Config.device, m_Config.device_budget_bytes, err))
            return false;
        DeviceMemoryInfo info;
        if (!QueryDeviceMemory(m_Config.device, info, err))
            return false;
        if (m_Config.device_budget_bytes > info.free_bytes) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "configured device budget %llu MiB exceeds free memory %llu MiB",
                     (unsigned long long)(m_Config.device_budget_bytes / (1024 * 1024)),
                     (unsigned long long)(info.free_bytes / (1024 * 1024)));
            err = buf;
            return false;
        }
        m_FreeBytes = info.free_bytes;
        m_TotalBytes = info.total_bytes;
        m_DeviceName = info.name;
        return true;
    }

    const char *BackendName() const override { return "cuda-pair-v1"; }

    std::vector<PairResult> RunBatch(const HmmSnapshot &hmm, const SequencePool &pool,
                                     const std::vector<PairJob> &jobs) override {
        std::vector<PairResult> results;

        std::string err;
        if (!ValidatePoolAndJobs(pool, jobs, err)) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::InvalidInput, err));
            return results;
        }

        // The digest is recomputed from the fields so a corrupted snapshot cannot
        // be used just because its stored digest matches itself.
        if (hmm.semantics_version != PAIR_SEMANTICS_VERSION) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::CpuFallback,
                                             "snapshot semantics_version mismatch"));
            return results;
        }
        const std::string recomputed = HmmSnapshotDigestHex(hmm);
        if (recomputed != DigestHex(hmm.digest)) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::CpuFallback,
                                             "HMM snapshot digest mismatch"));
            return results;
        }
        // The batch is defined by the model that is installed now: if the live
        // tables moved after the snapshot was taken, refuse the batch instead of
        // mixing two models inside one run.
        if (!HmmSnapshotMatchesCurrent(hmm, err)) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::CpuFallback,
                                             "snapshot mismatch: " + err));
            return results;
        }

        const uint64_t required = RequiredBytes(pool, jobs);
        if (required > m_Config.device_budget_bytes) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "batch needs %llu MiB but the device budget is %llu MiB",
                     (unsigned long long)(required / (1024 * 1024)),
                     (unsigned long long)(m_Config.device_budget_bytes / (1024 * 1024)));
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::ResourceError, buf));
            return results;
        }

        if (!m_Ctx.Arena().Reserve(size_t(required), err)) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::ResourceError, err));
            return results;
        }

        if (!UploadAndLaunch(hmm, pool, jobs, err)) {
            for (size_t k = 0; k < jobs.size(); ++k)
                results.push_back(MakeResult(jobs[k], PairStatus::DeviceError, err));
            return results;
        }

        // ---- host finalization: thresholds, expf, EA (CPU semantics) ----
        results.resize(jobs.size());
        for (size_t k = 0; k < jobs.size(); ++k) {
            PairResult &r = MakeInto(results[k], jobs[k]);
            const size_t cells = size_t(jobs[k].len_x) * size_t(jobs[k].len_y);
            std::vector<float> post;
            FinalizePostFromQ(m_HostQ.data() + m_HostQOffsets[k], jobs[k].len_x,
                              jobs[k].len_y, hmm.min_sparse_score, post);
            r.ea = ComputeEaFromPost(post.data(), jobs[k].len_x, jobs[k].len_y);
            r.post.swap(post);
            if (m_Config.verify || m_Config.debug_dump_fb)
                r.q.assign(m_HostQ.begin() + m_HostQOffsets[k],
                           m_HostQ.begin() + m_HostQOffsets[k] + cells);
            r.used = BackendKind::Gpu;
            r.status = PairStatus::Ok;
            r.reason.clear();
        }

        return results;
    }

    bool DumpLastBatch(const std::string &dir, std::string &err) override {
        err.clear();
        if (!m_LastDumpValid) {
            err = "no debug dump captured (enable config.debug_dump_fb)";
            return false;
        }
        const size_t n = m_DumpLenX.size();
        for (size_t k = 0; k < n; ++k) {
            char suffix[64];
            if (n == 1)
                suffix[0] = '\0';
            else
                snprintf(suffix, sizeof(suffix), "_pair%06u", m_DumpPairIndex[k]);
            if (!WriteFloatFile(dir, "fwd", suffix, m_DumpFwd[k], err))
                return false;
            if (!WriteFloatFile(dir, "bwd", suffix, m_DumpBwd[k], err))
                return false;
            if (!WriteFloatFile(dir, "q", suffix, m_DumpQ[k], err))
                return false;
            if (!WriteFloatFile(dir, "z", suffix, m_DumpZ[k], err))
                return false;
        }
        return true;
    }

    bool DescribeDevice(std::string &json_text, std::string &err) override {
        err.clear();
        char buf[768];
        snprintf(buf, sizeof(buf),
                 "{\"backend\":\"cuda-pair-v1\",\"device\":%d,\"name\":\"%s\","
                 "\"cc_major\":%d,\"cc_minor\":%d,\"sm_count\":%d,"
                 "\"budget_bytes\":%llu,\"device_free_bytes\":%llu,"
                 "\"device_total_bytes\":%llu,\"arena_capacity\":%zu,"
                 "\"arena_peak\":%zu,\"block_threads\":%d}",
                 m_Config.device, m_DeviceName.c_str(), m_Ctx.Props().major,
                 m_Ctx.Props().minor, m_Ctx.Props().multiProcessorCount,
                 (unsigned long long)m_Config.device_budget_bytes,
                 (unsigned long long)m_FreeBytes, (unsigned long long)m_TotalBytes,
                 m_Ctx.Arena().Capacity(), m_Ctx.Arena().PeakUsed(), kBlockThreads);
        json_text = buf;
        return true;
    }

private:
    static PairResult MakeResult(const PairJob &job, PairStatus status,
                                 const std::string &reason) {
        PairResult r;
        r.job_id = job.job_id;
        r.pair_index = job.pair_index;
        r.len_x = job.len_x;
        r.len_y = job.len_y;
        r.used = BackendKind::Gpu;
        r.status = status;
        r.reason = reason;
        return r;
    }

    static PairResult &MakeInto(PairResult &r, const PairJob &job) {
        r.job_id = job.job_id;
        r.pair_index = job.pair_index;
        r.len_x = job.len_x;
        r.len_y = job.len_y;
        return r;
    }

    uint64_t RequiredBytes(const SequencePool &pool,
                           const std::vector<PairJob> &jobs) const {
        uint64_t bytes = kModelBytes;
        bytes += pool.bytes.size();
        bytes += (pool.offsets.size()) * sizeof(uint64_t);
        bytes += jobs.size() * (4 * sizeof(uint32_t) + 3 * sizeof(uint64_t) +
                                sizeof(float));
        for (size_t k = 0; k < jobs.size(); ++k)
            bytes += EstimatePairWorkspaceBytes(jobs[k].len_x, jobs[k].len_y);
        bytes += 4096;  // alignment slop
        return bytes;
    }

    bool UploadAndLaunch(const HmmSnapshot &hmm, const SequencePool &pool,
                         const std::vector<PairJob> &jobs, std::string &err) {
        err.clear();
        DeviceArena &arena = m_Ctx.Arena();
        arena.Reset();

        float *d_match = static_cast<float *>(arena.Alloc(sizeof(float) * 65536, kAlign));
        float *d_ins = static_cast<float *>(arena.Alloc(sizeof(float) * 256, kAlign));
        uint8_t *d_pool = static_cast<uint8_t *>(arena.Alloc(pool.bytes.size(), kAlign));
        uint64_t *d_offsets = static_cast<uint64_t *>(
            arena.Alloc(sizeof(uint64_t) * pool.offsets.size(), kAlign));
        const size_t n = jobs.size();
        uint32_t *d_seq_x = static_cast<uint32_t *>(arena.Alloc(sizeof(uint32_t) * n, kAlign));
        uint32_t *d_seq_y = static_cast<uint32_t *>(arena.Alloc(sizeof(uint32_t) * n, kAlign));
        uint32_t *d_len_x = static_cast<uint32_t *>(arena.Alloc(sizeof(uint32_t) * n, kAlign));
        uint32_t *d_len_y = static_cast<uint32_t *>(arena.Alloc(sizeof(uint32_t) * n, kAlign));
        uint64_t *d_fwd_off = static_cast<uint64_t *>(arena.Alloc(sizeof(uint64_t) * n, kAlign));
        uint64_t *d_bwd_off = static_cast<uint64_t *>(arena.Alloc(sizeof(uint64_t) * n, kAlign));
        uint64_t *d_q_off = static_cast<uint64_t *>(arena.Alloc(sizeof(uint64_t) * n, kAlign));
        float *d_z = static_cast<float *>(arena.Alloc(sizeof(float) * n, kAlign));
        if (d_match == nullptr || d_ins == nullptr || d_pool == nullptr ||
            d_offsets == nullptr || d_seq_x == nullptr || d_seq_y == nullptr ||
            d_len_x == nullptr || d_len_y == nullptr || d_fwd_off == nullptr ||
            d_bwd_off == nullptr || d_q_off == nullptr || d_z == nullptr) {
            err = "device arena exhausted while allocating batch metadata";
            return false;
        }

        // Per-job F/B/q regions (float element offsets inside one float region).
        uint64_t float_cursor = 0;
        std::vector<ArenaLayout> layout(n);
        for (size_t k = 0; k < n; ++k) {
            const uint64_t cells = (uint64_t(jobs[k].len_x) + 1) * (uint64_t(jobs[k].len_y) + 1);
            layout[k].fwd_off = float_cursor;
            float_cursor += 5 * cells;
            layout[k].bwd_off = float_cursor;
            float_cursor += 5 * cells;
            layout[k].q_off = float_cursor;
            float_cursor += uint64_t(jobs[k].len_x) * uint64_t(jobs[k].len_y);
        }
        float *d_floats =
            static_cast<float *>(arena.Alloc(sizeof(float) * size_t(float_cursor), kAlign));
        if (d_floats == nullptr) {
            err = "device arena exhausted while allocating pair workspaces";
            return false;
        }
        // Remembered for the debug dump: fwd_off/bwd_off/q_off are element offsets
        // inside this float region, not inside the whole arena.
        m_LastFloatBase = d_floats;

        std::vector<uint32_t> h_seq_x(n), h_seq_y(n), h_len_x(n), h_len_y(n);
        std::vector<uint64_t> h_fwd_off(n), h_bwd_off(n), h_q_off(n);
        for (size_t k = 0; k < n; ++k) {
            h_seq_x[k] = jobs[k].seq_x;
            h_seq_y[k] = jobs[k].seq_y;
            h_len_x[k] = jobs[k].len_x;
            h_len_y[k] = jobs[k].len_y;
            h_fwd_off[k] = layout[k].fwd_off;
            h_bwd_off[k] = layout[k].bwd_off;
            h_q_off[k] = layout[k].q_off;
        }

        cudaStream_t stream = m_Ctx.Stream();
        {
            StageTrace::Scope scope("pair_backend.h2d", BackendName());
            scope.SetBytes(kModelBytes + pool.bytes.size());
            if (!CudaCheck(cudaMemcpyAsync(d_match, hmm.match.data(), sizeof(float) * 65536,
                                           cudaMemcpyHostToDevice, stream),
                           "copy match scores", err))
                return false;
            if (!CudaCheck(cudaMemcpyAsync(d_ins, hmm.ins.data(), sizeof(float) * 256,
                                           cudaMemcpyHostToDevice, stream),
                           "copy insert scores", err))
                return false;
            if (!CudaCheck(cudaMemcpyAsync(d_pool, pool.bytes.data(), pool.bytes.size(),
                                           cudaMemcpyHostToDevice, stream),
                           "copy sequence pool", err))
                return false;
            if (!CudaCheck(cudaMemcpyAsync(d_offsets, pool.offsets.data(),
                                           sizeof(uint64_t) * pool.offsets.size(),
                                           cudaMemcpyHostToDevice, stream),
                           "copy pool offsets", err))
                return false;
            if (!CudaCheck(cudaMemcpyAsync(d_seq_x, h_seq_x.data(), sizeof(uint32_t) * n,
                                           cudaMemcpyHostToDevice, stream),
                           "copy job seq_x", err) ||
                !CudaCheck(cudaMemcpyAsync(d_seq_y, h_seq_y.data(), sizeof(uint32_t) * n,
                                           cudaMemcpyHostToDevice, stream),
                           "copy job seq_y", err) ||
                !CudaCheck(cudaMemcpyAsync(d_len_x, h_len_x.data(), sizeof(uint32_t) * n,
                                           cudaMemcpyHostToDevice, stream),
                           "copy job len_x", err) ||
                !CudaCheck(cudaMemcpyAsync(d_len_y, h_len_y.data(), sizeof(uint32_t) * n,
                                           cudaMemcpyHostToDevice, stream),
                           "copy job len_y", err) ||
                !CudaCheck(cudaMemcpyAsync(d_fwd_off, h_fwd_off.data(),
                                           sizeof(uint64_t) * n, cudaMemcpyHostToDevice,
                                           stream),
                           "copy fwd offsets", err) ||
                !CudaCheck(cudaMemcpyAsync(d_bwd_off, h_bwd_off.data(),
                                           sizeof(uint64_t) * n, cudaMemcpyHostToDevice,
                                           stream),
                           "copy bwd offsets", err) ||
                !CudaCheck(cudaMemcpyAsync(d_q_off, h_q_off.data(), sizeof(uint64_t) * n,
                                           cudaMemcpyHostToDevice, stream),
                           "copy q offsets", err))
                return false;
        }

        DevicePairParams params;
        params.log_zero = hmm.log_zero;
        for (unsigned s = 0; s < 5; ++s)
            params.start[s] = hmm.start[s];
        for (unsigned t = 0; t < 25; ++t)
            params.trans[t] = hmm.trans[t];
        params.ins = d_ins;
        params.match = d_match;

        DeviceBatch batch;
        batch.pool_bytes = d_pool;
        batch.pool_offsets = d_offsets;
        batch.seq_x = d_seq_x;
        batch.seq_y = d_seq_y;
        batch.len_x = d_len_x;
        batch.len_y = d_len_y;
        batch.fwd_off = d_fwd_off;
        batch.bwd_off = d_bwd_off;
        batch.q_off = d_q_off;
        batch.arena = d_floats;
        batch.z_out = d_z;
        batch.n_jobs = int(n);

        cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
        if (!CudaCheck(cudaEventCreate(&ev_start), "cudaEventCreate", err) ||
            !CudaCheck(cudaEventCreate(&ev_stop), "cudaEventCreate", err)) {
            if (ev_start) cudaEventDestroy(ev_start);
            return false;
        }
        cudaEventRecord(ev_start, stream);

        bool ok = LaunchForward(params, batch, kBlockThreads, stream, err);
        if (ok)
            ok = LaunchBackward(params, batch, kBlockThreads, stream, err);
        if (ok)
            ok = LaunchPosteriorQ(params, batch, kBlockThreads, stream, err);

        cudaEventRecord(ev_stop, stream);
        if (ok && !CudaCheck(cudaStreamSynchronize(stream), "stream synchronize", err))
            ok = false;
        if (ok) {
            float device_ms = 0.0f;
            std::string timing_err;
            if (CudaCheck(cudaEventElapsedTime(&device_ms, ev_start, ev_stop),
                          "cudaEventElapsedTime", timing_err)) {
                StageTrace::Scope scope("pair_backend.kernels", BackendName());
                scope.SetDeviceMs(double(device_ms));
            } else {
                // Timing only: the result is already correct, but the failure is
                // recorded rather than silently dropped.
                StageTrace::Scope scope("pair_backend.kernels.timing_error", BackendName());
                scope.SetFallbackReason(timing_err);
            }
        }
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        if (!ok)
            return false;

        // ---- device -> host: q (and Z for dumps) ----
        m_HostQOffsets.assign(n, 0);
        size_t total_cells = 0;
        for (size_t k = 0; k < n; ++k) {
            m_HostQOffsets[k] = total_cells;
            total_cells += size_t(jobs[k].len_x) * size_t(jobs[k].len_y);
        }
        m_HostQ.resize(total_cells);
        m_HostZ.resize(n);
        {
            StageTrace::Scope scope("pair_backend.d2h", BackendName());
            scope.SetBytes(sizeof(float) * (total_cells + n));
            for (size_t k = 0; k < n; ++k) {
                const size_t cells = size_t(jobs[k].len_x) * size_t(jobs[k].len_y);
                if (!CudaCheck(cudaMemcpyAsync(m_HostQ.data() + m_HostQOffsets[k],
                                               d_floats + layout[k].q_off,
                                               sizeof(float) * cells,
                                               cudaMemcpyDeviceToHost, stream),
                               "copy q to host", err))
                    return false;
            }
            if (!CudaCheck(cudaMemcpyAsync(m_HostZ.data(), d_z, sizeof(float) * n,
                                           cudaMemcpyDeviceToHost, stream),
                           "copy Z to host", err))
                return false;
            if (!CudaCheck(cudaStreamSynchronize(stream), "stream synchronize", err))
                return false;
        }

        // Keep the raw device intermediates for debug dumps before the arena is
        // reused by the next batch.
        if (m_Config.debug_dump_fb && !CaptureDebugFromDevice(jobs, layout, err))
            return false;

        return true;
    }

    bool CaptureDebugFromDevice(const std::vector<PairJob> &jobs,
                                const std::vector<ArenaLayout> &layout,
                                std::string &err) {
        err.clear();
        const size_t n = jobs.size();
        m_DumpLenX.resize(n);
        m_DumpLenY.resize(n);
        m_DumpPairIndex.resize(n);
        m_DumpFwd.assign(n, std::vector<float>());
        m_DumpBwd.assign(n, std::vector<float>());
        m_DumpQ.assign(n, std::vector<float>());
        m_DumpZ.assign(n, std::vector<float>());

        cudaStream_t stream = m_Ctx.Stream();
        for (size_t k = 0; k < n; ++k) {
            const size_t cells = (size_t(jobs[k].len_x) + 1) * (size_t(jobs[k].len_y) + 1);
            m_DumpLenX[k] = jobs[k].len_x;
            m_DumpLenY[k] = jobs[k].len_y;
            m_DumpPairIndex[k] = jobs[k].pair_index;
            m_DumpFwd[k].resize(5 * cells);
            m_DumpBwd[k].resize(5 * cells);
            const size_t qcells = size_t(jobs[k].len_x) * size_t(jobs[k].len_y);
            m_DumpQ[k].assign(m_HostQ.begin() + m_HostQOffsets[k],
                              m_HostQ.begin() + m_HostQOffsets[k] + qcells);
            m_DumpZ[k].assign(1, m_HostZ[k]);
            if (m_LastFloatBase == nullptr) {
                err = "debug dump requested but the float region was not recorded";
                return false;
            }
            if (!CudaCheck(cudaMemcpyAsync(m_DumpFwd[k].data(),
                                           m_LastFloatBase + layout[k].fwd_off,
                                           sizeof(float) * 5 * cells,
                                           cudaMemcpyDeviceToHost, stream),
                           "copy F to host", err))
                return false;
            if (!CudaCheck(cudaMemcpyAsync(m_DumpBwd[k].data(),
                                           m_LastFloatBase + layout[k].bwd_off,
                                           sizeof(float) * 5 * cells,
                                           cudaMemcpyDeviceToHost, stream),
                           "copy B to host", err))
                return false;
        }
        if (!CudaCheck(cudaStreamSynchronize(stream), "debug dump synchronize", err))
            return false;
        m_LastDumpValid = true;
        return true;
    }

    static bool WriteFloatFile(const std::string &dir, const char *what,
                               const char *suffix, const std::vector<float> &values,
                               std::string &err) {
        std::string path = dir + "/" + what + suffix + ".bin";
        FILE *f = fopen(path.c_str(), "wb");
        if (f == nullptr) {
            err = "cannot create " + path;
            return false;
        }
        const size_t written = fwrite(values.data(), sizeof(float), values.size(), f);
        fclose(f);
        if (written != values.size()) {
            err = "short write to " + path;
            return false;
        }
        return true;
    }

    BackendConfig m_Config;
    CudaContext m_Ctx;
    uint64_t m_FreeBytes = 0;
    uint64_t m_TotalBytes = 0;
    std::string m_DeviceName;

    std::vector<float> m_HostQ;
    std::vector<size_t> m_HostQOffsets;
    std::vector<float> m_HostZ;
    float *m_LastFloatBase = nullptr;

    bool m_LastDumpValid;
    std::vector<uint32_t> m_DumpLenX, m_DumpLenY, m_DumpPairIndex;
    std::vector<std::vector<float> > m_DumpFwd, m_DumpBwd, m_DumpQ, m_DumpZ;
};

} // namespace

std::unique_ptr<PairBackend> CreateCudaPairBackend(const BackendConfig &config,
                                                   std::string &err) {
    err.clear();
    std::unique_ptr<CudaPairBackend> backend(new CudaPairBackend(config));
    if (!backend->Init(err))
        return std::unique_ptr<PairBackend>();
    return std::unique_ptr<PairBackend>(backend.release());
}

} // namespace muscle_gpu
