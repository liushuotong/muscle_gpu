// Five-state Forward kernel: line-by-line transcription of
// src/fwdflat3.cpp::CalcFwdFlat executed along anti-diagonals.
//
// Every thread of the CTA takes part in every diagonal barrier, including threads
// that own no cell on that diagonal; no lane returns early.  Cross-CTA
// communication does not exist: one CTA owns one pair.

#include "gpu/pair_kernels.cuh"
#include "gpu/device_log_math.cuh"

namespace muscle_gpu {

namespace {

const unsigned HMM_M = 0;
const unsigned HMM_IX = 1;
const unsigned HMM_IY = 2;
const unsigned HMM_JX = 3;
const unsigned HMM_JY = 4;

__global__ void KernelForward(const DevicePairParams params, DeviceBatch batch) {
    const int job = blockIdx.x;
    if (job >= batch.n_jobs)
        return;

    const uint32_t lx = batch.len_x[job];
    const uint32_t ly = batch.len_y[job];
    const uint8_t *x_seq = batch.pool_bytes + batch.pool_offsets[batch.seq_x[job]];
    const uint8_t *y_seq = batch.pool_bytes + batch.pool_offsets[batch.seq_y[job]];
    float *flat = batch.arena + batch.fwd_off[job];

    const float log_zero = params.log_zero;
    const float tSM = params.start[HMM_M];
    const float tSI = params.start[HMM_IX];
    const float tSJ = params.start[HMM_JX];
    const float tMM = params.trans[HMM_M * 5 + HMM_M];
    const float tMI = params.trans[HMM_M * 5 + HMM_IX];
    const float tMJ = params.trans[HMM_M * 5 + HMM_JX];
    const float tII = params.trans[HMM_IX * 5 + HMM_IX];
    const float tJJ = params.trans[HMM_JX * 5 + HMM_JX];
    // The M recurrence takes its five operands in the order M, IX, JX, IY, JY,
    // and the IY/JY contributions use their own transition rows.
    // hmmscores.h defines a single tIM = TransScore[IX][M] and tJM =
    // TransScore[JX][M]; the source reuses those two constants for the IY and JY
    // terms as well, so we do the same instead of reading the IY/JY rows.
    const float tIM = params.trans[HMM_IX * 5 + HMM_M];
    const float tJM = params.trans[HMM_JX * 5 + HMM_M];

    const float ins_x0 = params.ins[x_seq[0]];
    const float ins_y0 = params.ins[y_seq[0]];
    const float emit_x0_y0 = params.match[unsigned(x_seq[0]) * 256u + unsigned(y_seq[0])];

    const uint32_t ly1 = ly + 1;
    const int lx_i = int(lx);
    const int ly_i = int(ly);
    const int ndiag = lx_i + ly_i + 1;

    for (int d = 0; d < ndiag; ++d) {
        int lo = d - ly_i;
        if (lo < 0)
            lo = 0;
        int hi = d;
        if (hi > lx_i)
            hi = lx_i;

        for (int i = lo + int(threadIdx.x); i <= hi; i += int(blockDim.x)) {
            const int j = d - i;
            float *cell = flat + 5 * (size_t(i) * ly1 + size_t(j));
            if (i == 0 && j == 0) {
                cell[HMM_M] = log_zero;
                cell[HMM_IX] = log_zero;
                cell[HMM_IY] = log_zero;
                cell[HMM_JX] = log_zero;
                cell[HMM_JY] = log_zero;
            } else if (j == 0) {
                // First column: only the X insertion states are reachable.
                cell[HMM_M] = log_zero;
                cell[HMM_IY] = log_zero;
                cell[HMM_JY] = log_zero;
                const float emit_x = params.ins[x_seq[i - 1]];
                if (i == 1) {
                    cell[HMM_IX] = tSI + ins_x0;
                    cell[HMM_JX] = tSJ + ins_x0;
                } else {
                    const float *prev = flat + 5 * (size_t(i - 1) * ly1 + 0);
                    cell[HMM_IX] = prev[HMM_IX] + tII + emit_x;
                    cell[HMM_JX] = prev[HMM_JX] + tJJ + emit_x;
                }
            } else if (i == 0) {
                // First row: only the Y insertion states are reachable.
                cell[HMM_M] = log_zero;
                cell[HMM_IX] = log_zero;
                cell[HMM_JX] = log_zero;
                const float emit_y = params.ins[y_seq[j - 1]];
                if (j == 1) {
                    cell[HMM_IY] = tSI + ins_y0;
                    cell[HMM_JY] = tSJ + ins_y0;
                } else {
                    const float *prev = flat + 5 * (size_t(0) * ly1 + size_t(j - 1));
                    cell[HMM_IY] = prev[HMM_IY] + tII + emit_y;
                    cell[HMM_JY] = prev[HMM_JY] + tJJ + emit_y;
                }
            } else {
                const float emit_x = params.ins[x_seq[i - 1]];
                const float emit_y = params.ins[y_seq[j - 1]];

                if (i == 1 && j == 1) {
                    // Special entry: the source seeds M(1,1) directly.
                    cell[HMM_M] = tSM + emit_x0_y0;
                } else {
                    const float *prev = flat + 5 * (size_t(i - 1) * ly1 + size_t(j - 1));
                    const float m_m = prev[HMM_M] + tMM;
                    const float ix_m = prev[HMM_IX] + tIM;
                    const float jx_m = prev[HMM_JX] + tJM;
                    const float iy_m = prev[HMM_IY] + tIM;
                    const float jy_m = prev[HMM_JY] + tJM;
                    const float emit_pair =
                        params.match[unsigned(x_seq[i - 1]) * 256u + unsigned(y_seq[j - 1])];
                    const float sum_prev =
                        DeviceLogAdd5(m_m, ix_m, jx_m, iy_m, jy_m, log_zero);
                    cell[HMM_M] = sum_prev + emit_pair;
                }

                const float *up = flat + 5 * (size_t(i - 1) * ly1 + size_t(j));
                const float *left = flat + 5 * (size_t(i) * ly1 + size_t(j - 1));

                const float prev_m_up = up[HMM_M];
                const float prev_m_left = left[HMM_M];

                const float m_ix = prev_m_up + tMI;
                const float ix_ix = up[HMM_IX] + tII;
                cell[HMM_IX] = DeviceLogAdd2(ix_ix, m_ix, log_zero) + emit_x;

                const float m_jx = prev_m_up + tMJ;
                const float jx_jx = up[HMM_JX] + tJJ;
                cell[HMM_JX] = DeviceLogAdd2(jx_jx, m_jx, log_zero) + emit_x;

                const float m_iy = prev_m_left + tMI;
                const float iy_iy = left[HMM_IY] + tII;
                cell[HMM_IY] = DeviceLogAdd2(iy_iy, m_iy, log_zero) + emit_y;

                const float m_jy = prev_m_left + tMJ;
                const float jy_jy = left[HMM_JY] + tJJ;
                cell[HMM_JY] = DeviceLogAdd2(jy_jy, m_jy, log_zero) + emit_y;
            }
        }
        __syncthreads();
    }
}

} // namespace

bool LaunchForward(const DevicePairParams &params, const DeviceBatch &batch,
                   int block_threads, cudaStream_t stream, std::string &err) {
    err.clear();
    if (batch.n_jobs <= 0)
        return true;
    if (block_threads <= 0)
        block_threads = 128;
    KernelForward<<<batch.n_jobs, unsigned(block_threads), 0, stream>>>(params, batch);
    return CudaCheck(cudaGetLastError(), "KernelForward launch", err);
}

} // namespace muscle_gpu
