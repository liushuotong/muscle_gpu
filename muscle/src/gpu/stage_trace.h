#pragma once

// T01 stage trace: one JSON object per line, written to an explicitly requested
// file (-stage_trace FILE).  It never writes into FASTA/EFA output, and when no
// file is requested no per-cell timing or file I/O happens at all.
//
// Field contract (HANDOFF_AI.md section 5):
//   schema, run_id, family_id, replicate_id, pair_index (nullable), stage,
//   thread_id, wall_ns, cpu_work_ns (nullable), device_ms (nullable), cells,
//   nnz, bytes, backend, fallback_reason.
// One additional field, `seq`, is a monotonically increasing record index inside
// this file; it is additive metadata and does not change any documented field.
//
// Outer stage wall time must never be reported as the sum of child stages: each
// record is a single measured interval, and nesting is expressed by stage names.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace muscle_gpu {

struct StageRecord {
    std::string stage;
    int64_t pair_index;        // < 0 => null
    int thread_id;
    uint64_t wall_ns;
    int64_t cpu_work_ns;       // < 0 => null
    double device_ms;          // < 0 => null
    uint64_t cells;
    uint64_t nnz;
    uint64_t bytes;
    std::string backend;
    std::string fallback_reason;

    StageRecord()
        : pair_index(-1), thread_id(0), wall_ns(0), cpu_work_ns(-1), device_ms(-1),
          cells(0), nnz(0), bytes(0) {}
};

class StageTrace {
public:
    static StageTrace &Instance();

    // Empty path disables tracing.  Returns false with `err` when the file cannot
    // be created (the caller decides whether that is fatal: it is fatal when the
    // user explicitly asked for a trace).
    bool Open(const std::string &path, std::string &err);
    void Close();
    bool IsOpen() const { return m_File.load(std::memory_order_acquire) != nullptr; }

    void SetRunId(const std::string &s);
    void SetFamilyId(const std::string &s);
    void SetReplicateId(int64_t id);

    void Record(const StageRecord &rec);

    // RAII wall-clock timer for one stage.
    class Scope {
    public:
        Scope(const char *stage, const std::string &backend);
        ~Scope();
        void SetPairIndex(int64_t pair_index);
        void SetCells(uint64_t cells);
        void SetNnz(uint64_t nnz);
        void SetBytes(uint64_t bytes);
        void SetDeviceMs(double ms);
        void SetCpuWorkNs(int64_t ns);
        void SetFallbackReason(const std::string &reason);

    private:
        Scope(const Scope &);
        Scope &operator=(const Scope &);
        StageRecord m_Rec;
        uint64_t m_StartNs;
    };

private:
    StageTrace();
    ~StageTrace();
    StageTrace(const StageTrace &);
    StageTrace &operator=(const StageTrace &);

    mutable std::mutex m_Mutex;
    // FILE*, kept opaque so this header stays stdlib-only; atomic so that the
    // tracing-off fast path performs no locked read.
    std::atomic<void *> m_File;
    std::string m_RunId;
    std::string m_FamilyId;
    int64_t m_ReplicateId;
    uint64_t m_Seq;
};

// Monotonic nanoseconds.
uint64_t StageNowNs();
// Thread id as reported in traces (OpenMP thread when inside a region).
int StageThreadId();

} // namespace muscle_gpu
