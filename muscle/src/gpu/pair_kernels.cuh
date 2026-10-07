#pragma once

// Shared device-side layout for the v1 pair kernels.
//
// One CTA per pair job; the CTA walks anti-diagonals with __syncthreads() between
// them.  Cell (i,j) of an (Lx+1)x(Ly+1) grid lives at 5*((i)*(Ly+1)+j) + state in
// the cell-major five-state layout, with state order [M, IX, IY, JX, JY].

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "gpu/cuda_context.h"  // CudaCheck() for the launch helpers

namespace muscle_gpu {

struct DevicePairParams {
    float log_zero;         // LOG_ZERO as compiled on the host
    float start[5];         // HMMSTATE order
    float trans[25];        // src*5 + dst
    const float *ins;       // 256 entries
    const float *match;     // 65536 entries, x*256 + y
};

struct DeviceBatch {
    const uint8_t *pool_bytes;
    const uint64_t *pool_offsets;   // n+1
    const uint32_t *seq_x;          // per job
    const uint32_t *seq_y;
    const uint32_t *len_x;
    const uint32_t *len_y;
    const uint64_t *fwd_off;        // element offsets into arena
    const uint64_t *bwd_off;
    const uint64_t *q_off;
    float *arena;
    float *z_out;                   // per job, total probability Z
    int n_jobs;
};

// Launch one CTA per job; `block_threads` must be a multiple of 32 (default 128).
bool LaunchForward(const DevicePairParams &params, const DeviceBatch &batch,
                   int block_threads, cudaStream_t stream, std::string &err);
bool LaunchBackward(const DevicePairParams &params, const DeviceBatch &batch,
                    int block_threads, cudaStream_t stream, std::string &err);
// q(i,j) = F_M(i+1,j+1) + B_M(i+1,j+1) - Z, row-major, plus Z per job (computed by
// one lane per CTA in HMMSTATE order, exactly like CalcTotalProbFlat).
bool LaunchPosteriorQ(const DevicePairParams &params, const DeviceBatch &batch,
                      int block_threads, cudaStream_t stream, std::string &err);

} // namespace muscle_gpu
