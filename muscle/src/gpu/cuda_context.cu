#include "gpu/cuda_context.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu/backend_registry.h"

namespace muscle_gpu {

// ---------------------------------------------------------------------------
// DeviceArena
// ---------------------------------------------------------------------------

DeviceArena::DeviceArena()
    : m_Base(nullptr), m_Capacity(0), m_Used(0), m_Peak(0) {}

DeviceArena::~DeviceArena() { Release(); }

bool DeviceArena::Reserve(size_t bytes, std::string &err) {
    err.clear();
    if (bytes <= m_Capacity)
        return true;
    // Growth is only allowed when nothing is in flight; a bigger region is
    // allocated first and the old one released after a full synchronize.
    size_t want = m_Capacity ? m_Capacity : (4u << 20);
    while (want < bytes) {
        if (want > (size_t(1) << 40)) {
            want = bytes;
            break;
        }
        want *= 2;
    }
    void *base = nullptr;
    cudaError_t status = cudaMalloc(&base, want);
    if (status != cudaSuccess) {
        char buf[256];
        snprintf(buf, sizeof(buf), "cudaMalloc(%zu bytes) failed: %s", want,
                 cudaGetErrorString(status));
        err = buf;
        return false;
    }
    if (m_Base != nullptr) {
        cudaError_t sync = cudaDeviceSynchronize();
        if (sync != cudaSuccess) {
            cudaFree(base);
            err = std::string("cudaDeviceSynchronize before arena growth failed: ") +
                  cudaGetErrorString(sync);
            return false;
        }
        cudaFree(m_Base);
    }
    m_Base = base;
    m_Capacity = want;
    m_Used = 0;
    return true;
}

void DeviceArena::Release() {
    if (m_Base != nullptr) {
        cudaFree(m_Base);
        m_Base = nullptr;
    }
    m_Capacity = 0;
    m_Used = 0;
    m_Peak = 0;
}

void DeviceArena::Reset() { m_Used = 0; }

void *DeviceArena::Alloc(size_t bytes, size_t align) {
    if (m_Base == nullptr || align == 0)
        return nullptr;
    const size_t aligned = (m_Used + align - 1) / align * align;
    if (aligned > m_Capacity || bytes > m_Capacity - aligned)
        return nullptr;
    void *p = static_cast<char *>(m_Base) + aligned;
    m_Used = aligned + bytes;
    if (m_Used > m_Peak)
        m_Peak = m_Used;
    return p;
}

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

bool CudaCheck(cudaError_t status, const char *what, std::string &err) {
    if (status == cudaSuccess)
        return true;
    err = std::string(what) + ": " + cudaGetErrorString(status);
    return false;
}

// ---------------------------------------------------------------------------
// CudaContext
// ---------------------------------------------------------------------------

CudaContext::CudaContext()
    : m_Initialized(false), m_Device(0), m_BudgetBytes(0), m_Stream(nullptr) {
    memset(&m_Props, 0, sizeof(m_Props));
}

CudaContext::~CudaContext() {
    m_Arena.Release();
    if (m_Stream != nullptr)
        cudaStreamDestroy(m_Stream);
}

bool CudaContext::Init(int device, uint64_t budget_bytes, std::string &err) {
    err.clear();
    if (m_Initialized)
        return true;

    int count = 0;
    if (!CudaCheck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err))
        return false;
    if (count <= 0) {
        err = "no CUDA device found";
        return false;
    }
    if (device < 0 || device >= count) {
        char buf[128];
        snprintf(buf, sizeof(buf), "device ordinal %d out of range (0..%d)", device,
                 count - 1);
        err = buf;
        return false;
    }
    if (!CudaCheck(cudaSetDevice(device), "cudaSetDevice", err))
        return false;
    if (!CudaCheck(cudaGetDeviceProperties(&m_Props, device), "cudaGetDeviceProperties",
                   err))
        return false;
    if (!CudaCheck(cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking),
                   "cudaStreamCreate", err))
        return false;

    m_Device = device;
    m_BudgetBytes = budget_bytes;
    m_Initialized = true;
    return true;
}

bool CudaContext::Synchronize(std::string &err) {
    return CudaCheck(cudaStreamSynchronize(m_Stream), "cudaStreamSynchronize", err);
}

std::string CudaContext::Describe() const {
    char buf[512];
    const size_t free_b = 0;
    (void)free_b;
    snprintf(buf, sizeof(buf),
             "device=%d name=\"%s\" cc=%d.%d sm_count=%d budget_bytes=%llu "
             "arena_capacity=%zu arena_peak=%zu",
             m_Device, m_Props.name, m_Props.major, m_Props.minor,
             m_Props.multiProcessorCount, (unsigned long long)m_BudgetBytes,
             m_Arena.Capacity(), m_Arena.PeakUsed());
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Availability / properties (used by dispatch before any context exists)
// ---------------------------------------------------------------------------

bool CudaAvailable(std::string &reason) {
    reason.clear();
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        reason = std::string("no usable CUDA runtime: ") + cudaGetErrorString(status);
        return false;
    }
    if (count <= 0) {
        reason = "no CUDA device present";
        return false;
    }
    return true;
}

bool QueryDeviceMemory(int device, DeviceMemoryInfo &info, std::string &err) {
    err.clear();
    int count = 0;
    if (!CudaCheck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err))
        return false;
    if (device < 0 || device >= count) {
        char buf[128];
        snprintf(buf, sizeof(buf), "device ordinal %d out of range (0..%d)", device,
                 count - 1);
        err = buf;
        return false;
    }
    int saved = 0;
    cudaGetDevice(&saved);
    if (!CudaCheck(cudaSetDevice(device), "cudaSetDevice", err))
        return false;
    size_t free_b = 0, total_b = 0;
    const bool ok = CudaCheck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err);
    cudaDeviceProp props;
    memset(&props, 0, sizeof(props));
    if (ok)
        cudaGetDeviceProperties(&props, device);
    if (saved != device)
        cudaSetDevice(saved);
    if (!ok)
        return false;
    info.free_bytes = free_b;
    info.total_bytes = total_b;
    info.name = props.name;
    info.major = props.major;
    info.minor = props.minor;
    info.multi_processor_count = props.multiProcessorCount;
    return true;
}

} // namespace muscle_gpu
