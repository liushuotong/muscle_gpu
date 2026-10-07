// Five-state Backward kernel: line-by-line transcription of
// src/bwdflat3.cpp::CalcBwdFlat executed along anti-diagonals in reverse.
//
// Boundary contract (do not "simplify"):
//   * the (Lx,Ly) corner is seeded with the start scores in enum order
//     [tSM, tSI, tSI, tSJ, tSJ], not with zeros;
//   * interior cells depend on diagonals d+1 and d+2 and use the *next* symbols
//     X[i], Y[j] for the emissions;
//   * the right edge (j == Ly, i < Lx) only allows the X insertion states and the
//     Y states stay LOG_ZERO; the bottom edge is the mirror image;
//   * the i>0 / j>0 masks zero M, IX/JX and IY/JY exactly as the source does.

#include "gpu/pair_kernels.cuh"
#include "gpu/device_log_math.cuh"

namespace muscle_gpu {

namespace {

const unsigned HMM_M = 0;
const unsigned HMM_IX = 1;
const unsigned HMM_IY = 2;
const unsigned HMM_JX = 3;
const unsigned HMM_JY = 4;

__global__ void KernelBackward(const DevicePairParams params, DeviceBatch batch) {
    const int job = blockIdx.x;
    if (job >= batch.n_jobs)
        return;

    const uint32_t lx = batch.len_x[job];
    const uint32_t ly = batch.len_y[job];
    const uint8_t *x_seq = batch.pool_bytes + batch.pool_offsets[batch.seq_x[job]];
    const uint8_t *y_seq = batch.pool_bytes + batch.pool_offsets[batch.seq_y[job]];
    float *flat = batch.arena + batch.bwd_off[job];

    const float log_zero = params.log_zero;
    const float tSM = params.start[HMM_M];
    const float tSI = params.start[HMM_IX];
    const float tSJ = params.start[HMM_JX];
    const float tMM = params.trans[HMM_M * 5 + HMM_M];
    const float tMI = params.trans[HMM_M * 5 + HMM_IX];
    const float tMJ = params.trans[HMM_M * 5 + HMM_JX];
    const float tII = params.trans[HMM_IX * 5 + HMM_IX];
    const float tJJ = params.trans[HMM_JX * 5 + HMM_JX];
    // As in the Forward kernel (and hmmscores.h): one tIM/tJM pair taken from the
    // IX/JX rows is reused for the IY/JY terms.
    const float tIM = params.trans[HMM_IX * 5 + HMM_M];
    const float tJM = params.trans[HMM_JX * 5 + HMM_M];

    const uint32_t ly1 = ly + 1;
    const int lx_i = int(lx);
    const int ly_i = int(ly);

    for (int d = lx_i + ly_i; d >= 0; --d) {
        int lo = d - ly_i;
        if (lo < 0)
            lo = 0;
        int hi = d;
        if (hi > lx_i)
            hi = lx_i;

        for (int i = lo + int(threadIdx.x); i <= hi; i += int(blockDim.x)) {
            const int j = d - i;
            float *cell = flat + 5 * (size_t(i) * ly1 + size_t(j));

            if (i == lx_i && j == ly_i) {
                // End-of-alignment: start scores in HMMSTATE enum order.
                cell[HMM_M] = tSM;
                cell[HMM_IX] = tSI;
                cell[HMM_IY] = tSI;
                cell[HMM_JX] = tSJ;
                cell[HMM_JY] = tSJ;
            } else if (i < lx_i && j < ly_i) {
                const float emit_x = params.ins[x_seq[i]];
                const float emit_y = params.ins[y_seq[j]];
                const float emit_xy = params.match[unsigned(x_seq[i]) * 256u + unsigned(y_seq[j])];

                const float *diag = flat + 5 * (size_t(i + 1) * ly1 + size_t(j + 1));
                const float *up = flat + 5 * (size_t(i + 1) * ly1 + size_t(j));
                const float *left = flat + 5 * (size_t(i) * ly1 + size_t(j + 1));

                const float next_m = diag[HMM_M] + emit_xy;
                const float next_ix = up[HMM_IX] + emit_x;
                const float next_jx = up[HMM_JX] + emit_x;
                const float next_iy = left[HMM_IY] + emit_y;
                const float next_jy = left[HMM_JY] + emit_y;

                if (i > 0 && j > 0) {
                    const float m_m = tMM + next_m;
                    const float m_ix = tMI + next_ix;
                    const float m_jx = tMJ + next_jx;
                    const float m_iy = tMI + next_iy;
                    const float m_jy = tMJ + next_jy;
                    cell[HMM_M] = DeviceLogAdd5(m_m, m_ix, m_jx, m_iy, m_jy, log_zero);
                } else {
                    cell[HMM_M] = log_zero;
                }

                if (i > 0) {
                    const float ix_ix = tII + next_ix;
                    const float ix_m = tIM + next_m;
                    cell[HMM_IX] = DeviceLogAdd2(ix_ix, ix_m, log_zero);
                    const float jx_jx = tJJ + next_jx;
                    const float jx_m = tJM + next_m;
                    cell[HMM_JX] = DeviceLogAdd2(jx_jx, jx_m, log_zero);
                } else {
                    cell[HMM_IX] = log_zero;
                    cell[HMM_JX] = log_zero;
                }

                if (j > 0) {
                    const float iy_iy = tII + next_iy;
                    const float iy_m = tIM + next_m;
                    cell[HMM_IY] = DeviceLogAdd2(iy_iy, iy_m, log_zero);
                    const float jy_jy = tJJ + next_jy;
                    const float jy_m = tJM + next_m;
                    cell[HMM_JY] = DeviceLogAdd2(jy_jy, jy_m, log_zero);
                } else {
                    cell[HMM_IY] = log_zero;
                    cell[HMM_JY] = log_zero;
                }
            } else if (i < lx_i) {
                // Right edge: j == Ly, only the X insertion chain continues.
                const float emit_x = params.ins[x_seq[i]];
                if (i > 0) {
                    const float *up = flat + 5 * (size_t(i + 1) * ly1 + size_t(j));
                    const float next_ix = up[HMM_IX] + emit_x;
                    const float next_jx = up[HMM_JX] + emit_x;
                    const float m_ix = tMI + next_ix;
                    const float m_jx = tMJ + next_jx;
                    cell[HMM_M] = DeviceLogAdd2(m_ix, m_jx, log_zero);
                    cell[HMM_IX] = tII + next_ix;
                    cell[HMM_JX] = tJJ + next_jx;
                } else {
                    cell[HMM_M] = log_zero;
                    cell[HMM_IX] = log_zero;
                    cell[HMM_JX] = log_zero;
                }
                // The reference pre-set these two to LOG_ZERO and never rewrites
                // them on this edge.
                cell[HMM_IY] = log_zero;
                cell[HMM_JY] = log_zero;
            } else {
                // Bottom edge: i == Lx, only the Y insertion chain continues.
                const float emit_y = params.ins[y_seq[j]];
                if (j > 0) {
                    const float *left = flat + 5 * (size_t(i) * ly1 + size_t(j + 1));
                    const float next_iy = left[HMM_IY] + emit_y;
                    const float next_jy = left[HMM_JY] + emit_y;
                    const float m_iy = tMI + next_iy;
                    const float m_jy = tMJ + next_jy;
                    cell[HMM_M] = DeviceLogAdd2(m_iy, m_jy, log_zero);
                    cell[HMM_IY] = tII + next_iy;
                    cell[HMM_JY] = tJJ + next_jy;
                } else {
                    cell[HMM_M] = log_zero;
                    cell[HMM_IY] = log_zero;
                    cell[HMM_JY] = log_zero;
                }
                cell[HMM_IX] = log_zero;
                cell[HMM_JX] = log_zero;
            }
        }
        __syncthreads();
    }
}

} // namespace

bool LaunchBackward(const DevicePairParams &params, const DeviceBatch &batch,
                    int block_threads, cudaStream_t stream, std::string &err) {
    err.clear();
    if (batch.n_jobs <= 0)
        return true;
    if (block_threads <= 0)
        block_threads = 128;
    KernelBackward<<<batch.n_jobs, unsigned(block_threads), 0, stream>>>(params, batch);
    return CudaCheck(cudaGetLastError(), "KernelBackward launch", err);
}

// ---------------------------------------------------------------------------
// Total probability Z and q
// ---------------------------------------------------------------------------

namespace {

// Z is accumulated by exactly one lane per CTA, in HMMSTATE enum order, using the
// same LOG_PLUS_EQUALS recurrence as src/totalprobflat.cpp.  q is then computed by
// all threads as (F_M + B_M) - Z, the same float expression and operand order as
// src/calcposteriorflat.cpp::CalcPostFlat.  expf() stays on the host.
__global__ void KernelPosteriorQ(const DevicePairParams params, DeviceBatch batch) {
    const int job = blockIdx.x;
    if (job >= batch.n_jobs)
        return;

    const uint32_t lx = batch.len_x[job];
    const uint32_t ly = batch.len_y[job];
    const float log_zero = params.log_zero;

    const float *fwd = batch.arena + batch.fwd_off[job];
    const float *bwd = batch.arena + batch.bwd_off[job];
    float *q = batch.arena + batch.q_off[job];

    __shared__ float shared_z;
    if (threadIdx.x == 0) {
        const size_t end_cell = size_t(lx) * (size_t(ly) + 1) + size_t(ly);
        float sum = log_zero;
        for (unsigned s = 0; s < 5; ++s) {
            const float fwd_score = fwd[5 * end_cell + s];
            const float bwd_score = bwd[5 * end_cell + s];
            DeviceLogPlusEquals(sum, fwd_score + bwd_score, log_zero);
        }
        shared_z = sum;
        if (batch.z_out != nullptr)
            batch.z_out[job] = sum;
    }
    __syncthreads();

    const float z = shared_z;
    const size_t cells = size_t(lx) * size_t(ly);
    const uint32_t ly1 = ly + 1;
    for (size_t k = size_t(threadIdx.x); k < cells; k += size_t(blockDim.x)) {
        const uint32_t i = uint32_t(k / ly);
        const uint32_t j = uint32_t(k % ly);
        const size_t idx = 5 * (size_t(i + 1) * ly1 + size_t(j + 1));
        q[k] = fwd[idx] + bwd[idx] - z;
    }
}

} // namespace

bool LaunchPosteriorQ(const DevicePairParams &params, const DeviceBatch &batch,
                      int block_threads, cudaStream_t stream, std::string &err) {
    err.clear();
    if (batch.n_jobs <= 0)
        return true;
    if (block_threads <= 0)
        block_threads = 128;
    KernelPosteriorQ<<<batch.n_jobs, unsigned(block_threads), 0, stream>>>(params, batch);
    return CudaCheck(cudaGetLastError(), "KernelPosteriorQ launch", err);
}

} // namespace muscle_gpu
