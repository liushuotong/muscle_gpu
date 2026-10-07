#pragma once

// Backend construction and CUDA availability probing.
//
// A MUSCLE_CUDA=OFF build must still compile every caller: the CUDA factory is
// provided either by src/gpu/pair_backend_cuda.cu (MUSCLE_CUDA=ON) or by
// src/gpu/pair_backend_cuda_stub.cpp (MUSCLE_CUDA=OFF), never by both.  The stub
// reports a clear, actionable reason and never fakes success.

#include <cstdint>
#include <memory>
#include <string>

#include "gpu/pair_backend.h"

namespace muscle_gpu {

struct DeviceMemoryInfo {
    uint64_t free_bytes;
    uint64_t total_bytes;
    std::string name;
    int major;
    int minor;
    int multi_processor_count;

    DeviceMemoryInfo()
        : free_bytes(0), total_bytes(0), major(0), minor(0),
          multi_processor_count(0) {}
};

// True when a usable CUDA runtime + at least one device exist.  When false,
// `reason` explains exactly why (no CUDA support compiled in, no driver, no
// device, ...).  Never throws and never creates a long-lived context.
bool CudaAvailable(std::string &reason);

// Query free/total memory and properties of `device`.  Fails with `reason` when
// the ordinal is out of range or the runtime is unavailable.
bool QueryDeviceMemory(int device, DeviceMemoryInfo &info, std::string &err);

// Implemented by the CUDA translation unit or by the stub.
std::unique_ptr<PairBackend> CreateCudaPairBackend(const BackendConfig &config,
                                                   std::string &err);

// min(16 GiB, 0.75 * free) applied to the queried device, as required by the
// 24 GB budget policy.  `config.device_budget_bytes` overrides it when non-zero.
uint64_t ResolveDeviceBudget(const BackendConfig &config, std::string &err);

// Create the backend for config.kind.  BackendKind::Auto resolves to the CPU
// backend until the T07 break-even table exists (documented behaviour, never a
// silent GPU switch).
std::unique_ptr<PairBackend> CreatePairBackend(const BackendConfig &config,
                                               std::string &err);

} // namespace muscle_gpu
