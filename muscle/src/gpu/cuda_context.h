#pragma once

// Private CUDA context/arena header.  It includes CUDA headers and therefore must
// never be included from a public MUSCLE5 GPU header (pair_backend.h stays
// CUDA-free so MUSCLE_CUDA=OFF builds keep compiling).

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace muscle_gpu {

// Bump allocator over one cudaMalloc region.  Growth/Reset are only legal while
// no work is in flight (RunBatch is synchronous, so batch boundaries are safe).
class DeviceArena {
public:
    DeviceArena();
    ~DeviceArena();

    bool Reserve(size_t bytes, std::string &err);  // grow if needed; keeps data
    void Release();
    void Reset();  // rewind the bump pointer; no free
    void *Alloc(size_t bytes, size_t align = 256);

    template <typename T>
    T *Base() {
        return static_cast<T *>(m_Base);
    }

    size_t Capacity() const { return m_Capacity; }
    size_t Used() const { return m_Used; }
    size_t PeakUsed() const { return m_Peak; }
    bool Valid() const { return m_Base != nullptr; }

private:
    DeviceArena(const DeviceArena &);
    DeviceArena &operator=(const DeviceArena &);

    void *m_Base;
    size_t m_Capacity;
    size_t m_Used;
    size_t m_Peak;
};

// Convert a CUDA error into a readable message; never swallows the error code.
bool CudaCheck(cudaError_t status, const char *what, std::string &err);

// One CUDA device + stream used by the pair backend.  A worker owns exactly one
// context; no per-pair context/stream creation is allowed.
class CudaContext {
public:
    CudaContext();
    ~CudaContext();

    bool Init(int device, uint64_t budget_bytes, std::string &err);
    bool Synchronize(std::string &err);

    cudaStream_t Stream() const { return m_Stream; }
    DeviceArena &Arena() { return m_Arena; }
    int Device() const { return m_Device; }
    const cudaDeviceProp &Props() const { return m_Props; }
    bool Initialized() const { return m_Initialized; }
    std::string Describe() const;

private:
    CudaContext(const CudaContext &);
    CudaContext &operator=(const CudaContext &);

    bool m_Initialized;
    int m_Device;
    uint64_t m_BudgetBytes;
    cudaStream_t m_Stream;
    DeviceArena m_Arena;
    cudaDeviceProp m_Props;
};

} // namespace muscle_gpu
