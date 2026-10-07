// Stub CUDA backend factory for MUSCLE_CUDA=OFF builds.
//
// This translation unit is compiled *instead of* pair_backend_cuda.cu.  It exists
// so that the control layer (dispatch/registry/oracle) links without any CUDA
// toolchain, and it must never pretend to have a device: every entry point
// reports a precise reason so that `-backend gpu` can exit non-zero and no
// benchmark can be silently mislabelled.

#include "gpu/backend_registry.h"

namespace muscle_gpu {

namespace {
const char *kNoCuda = "CUDA support is not compiled in (MUSCLE_CUDA=OFF)";
}

bool CudaAvailable(std::string &reason) {
    reason = kNoCuda;
    return false;
}

bool QueryDeviceMemory(int device, DeviceMemoryInfo &info, std::string &err) {
    (void)device;
    (void)info;
    err = kNoCuda;
    return false;
}

std::unique_ptr<PairBackend> CreateCudaPairBackend(const BackendConfig &config,
                                                   std::string &err) {
    (void)config;
    err = kNoCuda;
    return std::unique_ptr<PairBackend>();
}

} // namespace muscle_gpu
