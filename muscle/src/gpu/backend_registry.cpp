#include "gpu/backend_registry.h"

#include "gpu/pair_backend_cpu.h"

namespace muscle_gpu {

namespace {

const uint64_t GIB = 1024ull * 1024ull * 1024ull;
const uint64_t DEFAULT_BUDGET_CAP = 16 * GIB;  // 24 GB card is never assumed free

} // namespace

uint64_t ResolveDeviceBudget(const BackendConfig &config, std::string &err) {
    err.clear();
    if (config.device_budget_bytes != 0)
        return config.device_budget_bytes;

    DeviceMemoryInfo info;
    if (!QueryDeviceMemory(config.device, info, err))
        return 0;

    const uint64_t by_free = uint64_t(double(info.free_bytes) * 0.75);
    return (by_free < DEFAULT_BUDGET_CAP) ? by_free : DEFAULT_BUDGET_CAP;
}

std::unique_ptr<PairBackend> CreatePairBackend(const BackendConfig &config,
                                               std::string &err) {
    err.clear();
    switch (config.kind) {
    case BackendKind::Cpu:
        return CreateCpuPairBackend(config, err);
    case BackendKind::Gpu:
        return CreateCudaPairBackend(config, err);
    case BackendKind::Auto: {
        // T07 owns the frozen break-even table; until then `auto` is defined to be
        // exactly the CPU path so that no benchmark can be mislabelled as GPU.
        return CreateCpuPairBackend(config, err);
    }
    }
    err = "unknown BackendKind";
    return std::unique_ptr<PairBackend>();
}

} // namespace muscle_gpu
