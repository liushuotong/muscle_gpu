#include "gpu/stage_trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

namespace muscle_gpu {

namespace {

const uint32_t TRACE_SCHEMA = 1;

void JsonEscapeTo(std::string &out, const std::string &s) {
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += char(c);
            }
        }
    }
}

void AppendU64(std::string &out, uint64_t v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v);
    out += buf;
}

void AppendI64(std::string &out, int64_t v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%lld", (long long)v);
    out += buf;
}

} // namespace

uint64_t StageNowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
}

int StageThreadId() { return omp_get_thread_num(); }

StageTrace &StageTrace::Instance() {
    static StageTrace instance;
    return instance;
}

StageTrace::StageTrace()
    : m_File(nullptr), m_ReplicateId(-1), m_Seq(0) {}

StageTrace::~StageTrace() { Close(); }

bool StageTrace::Open(const std::string &path, std::string &err) {
    err.clear();
    std::lock_guard<std::mutex> lock(m_Mutex);
    void *existing = m_File.load(std::memory_order_acquire);
    if (existing != nullptr) {
        fclose((FILE *)existing);
        m_File.store(nullptr, std::memory_order_release);
    }
    if (path.empty())
        return true;
    FILE *f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        err = "cannot create stage trace file: " + path;
        return false;
    }
    m_File.store((void *)f, std::memory_order_release);
    return true;
}

void StageTrace::Close() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    void *existing = m_File.load(std::memory_order_acquire);
    if (existing != nullptr) {
        fflush((FILE *)existing);
        fclose((FILE *)existing);
        m_File.store(nullptr, std::memory_order_release);
    }
}

void StageTrace::SetRunId(const std::string &s) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_RunId = s;
}

void StageTrace::SetFamilyId(const std::string &s) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_FamilyId = s;
}

void StageTrace::SetReplicateId(int64_t id) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_ReplicateId = id;
}

void StageTrace::Record(const StageRecord &rec) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    FILE *f = (FILE *)m_File.load(std::memory_order_acquire);
    if (f == nullptr)
        return;

    std::string line;
    line.reserve(320);
    line += "{\"schema\":";
    AppendU64(line, TRACE_SCHEMA);
    line += ",\"seq\":";
    AppendU64(line, m_Seq++);
    line += ",\"run_id\":\"";
    JsonEscapeTo(line, m_RunId);
    line += "\",\"family_id\":\"";
    JsonEscapeTo(line, m_FamilyId);
    line += "\",\"replicate_id\":";
    if (m_ReplicateId < 0)
        line += "null";
    else
        AppendI64(line, m_ReplicateId);
    line += ",\"pair_index\":";
    if (rec.pair_index < 0)
        line += "null";
    else
        AppendI64(line, rec.pair_index);
    line += ",\"stage\":\"";
    JsonEscapeTo(line, rec.stage);
    line += "\",\"thread_id\":";
    AppendI64(line, rec.thread_id);
    line += ",\"wall_ns\":";
    AppendU64(line, rec.wall_ns);
    line += ",\"cpu_work_ns\":";
    if (rec.cpu_work_ns < 0)
        line += "null";
    else
        AppendI64(line, rec.cpu_work_ns);
    line += ",\"device_ms\":";
    if (rec.device_ms < 0) {
        line += "null";
    } else {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.6f", rec.device_ms);
        line += buf;
    }
    line += ",\"cells\":";
    AppendU64(line, rec.cells);
    line += ",\"nnz\":";
    AppendU64(line, rec.nnz);
    line += ",\"bytes\":";
    AppendU64(line, rec.bytes);
    line += ",\"backend\":\"";
    JsonEscapeTo(line, rec.backend);
    line += "\",\"fallback_reason\":";
    if (rec.fallback_reason.empty())
        line += "null";
    else {
        line += "\"";
        JsonEscapeTo(line, rec.fallback_reason);
        line += "\"";
    }
    line += "}\n";

    fwrite(line.data(), 1, line.size(), f);
    fflush(f);
}

StageTrace::Scope::Scope(const char *stage, const std::string &backend)
    : m_StartNs(StageNowNs()) {
    m_Rec.stage = (stage != nullptr) ? stage : "?";
    m_Rec.backend = backend;
    m_Rec.thread_id = StageThreadId();
}

StageTrace::Scope::~Scope() {
    StageTrace &trace = StageTrace::Instance();
    if (!trace.IsOpen())
        return;
    m_Rec.wall_ns = StageNowNs() - m_StartNs;
    trace.Record(m_Rec);
}

void StageTrace::Scope::SetPairIndex(int64_t pair_index) { m_Rec.pair_index = pair_index; }
void StageTrace::Scope::SetCells(uint64_t cells) { m_Rec.cells = cells; }
void StageTrace::Scope::SetNnz(uint64_t nnz) { m_Rec.nnz = nnz; }
void StageTrace::Scope::SetBytes(uint64_t bytes) { m_Rec.bytes = bytes; }
void StageTrace::Scope::SetDeviceMs(double ms) { m_Rec.device_ms = ms; }
void StageTrace::Scope::SetCpuWorkNs(int64_t ns) { m_Rec.cpu_work_ns = ns; }
void StageTrace::Scope::SetFallbackReason(const std::string &reason) {
    m_Rec.fallback_reason = reason;
}

} // namespace muscle_gpu
