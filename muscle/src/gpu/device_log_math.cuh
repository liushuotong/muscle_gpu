#pragma once

// Device transcription of MUSCLE's logarithmic arithmetic (src/scoretype.h).
//
// Rules that must not change (RESEARCH_PLAN.md section 2.2, HANDOFF_AI.md section 3):
//   * LOG_ZERO is a finite -2e20f, not IEEE -inf; it is passed in as a parameter
//     so the device always uses the exact value the host snapshot recorded.
//   * LOGEXP1 keeps the four piecewise cubic polynomials, their coefficients and
//     the 1.0 / 2.5 / 4.5 / 7.5 break points, evaluated in the same order.
//   * LOG_ADD keeps the branch structure and the >= 7.5 cutoff, including the
//     "other operand is exactly LOG_ZERO" short-circuit.
//   * The nested multi-operand LOG_ADD is right-nested:
//       LOG_ADD(x1..x5) = LOG_ADD(x1, LOG_ADD(x2, LOG_ADD(x3, LOG_ADD(x4, x5))))
//     Reordering these operands changes float results (proved in
//     planning/evidence/log_math_probe.json) and is forbidden.
//
// No transcendentals are used on the device: expf()/logf() stay on the host, so
// the CUDA math library cannot influence the probabilities.

namespace muscle_gpu {

const float DEVICE_LOG_UNDERFLOW_THRESHOLD = 7.5f;
const float DEVICE_LOG_ONE = 0.0f;

__device__ __forceinline__ float DeviceLogExp1(float x) {
    // Computes log(exp(x) + 1) for 0 <= x <= 7.5, exactly as LOGEXP1 does.
    if (x <= 1.00f)
        return ((-0.009350833524763f * x + 0.130659527668286f) * x + 0.498799810682272f) * x +
               0.693203116424741f;
    if (x <= 2.50f)
        return ((-0.014532321752540f * x + 0.139942324101744f) * x + 0.495635523139337f) * x +
               0.692140569840976f;
    if (x <= 4.50f)
        return ((-0.004605031767994f * x + 0.063427417320019f) * x + 0.695956496475118f) * x +
               0.514272634594009f;
    return ((-0.000458661602210f * x + 0.009695946122598f) * x + 0.930734667215156f) * x +
           0.168037164329057f;
}

__device__ __forceinline__ float DeviceLogAdd(float x, float y, float log_zero) {
    if (x < y)
        return (x == log_zero || y - x >= DEVICE_LOG_UNDERFLOW_THRESHOLD)
                   ? y
                   : DeviceLogExp1(y - x) + x;
    return (y == log_zero || x - y >= DEVICE_LOG_UNDERFLOW_THRESHOLD)
               ? x
               : DeviceLogExp1(x - y) + y;
}

__device__ __forceinline__ float DeviceLogAdd2(float x1, float x2, float log_zero) {
    return DeviceLogAdd(x1, x2, log_zero);
}

__device__ __forceinline__ float DeviceLogAdd5(float x1, float x2, float x3, float x4,
                                               float x5, float log_zero) {
    return DeviceLogAdd(x1,
                        DeviceLogAdd(x2, DeviceLogAdd(x3, DeviceLogAdd(x4, x5, log_zero),
                                                      log_zero),
                                     log_zero),
                        log_zero);
}

__device__ __forceinline__ void DeviceLogPlusEquals(float &x, float y, float log_zero) {
    if (x < y)
        x = (x == log_zero || y - x >= DEVICE_LOG_UNDERFLOW_THRESHOLD)
                ? y
                : DeviceLogExp1(y - x) + x;
    else
        x = (y == log_zero || x - y >= DEVICE_LOG_UNDERFLOW_THRESHOLD)
                ? x
                : DeviceLogExp1(x - y) + y;
}

} // namespace muscle_gpu
