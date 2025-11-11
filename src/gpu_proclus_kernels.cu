#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <limits>
#include <type_traits>
#include "../include/error_check.h"

// -------------------------- device utils --------------------------
template<typename T>
__device__ __forceinline__ T dabs(T x) { return x < T(0) ? -x : x; }

// -------------------------- kernels -------------------------------

template<int TILE_K>
__global__ void k_assign_l1_excl(
    const float* __restrict__ X,   // [n,d]
    const float* __restrict__ M,   // [k,d]
    const int32_t* __restrict__ excl, // [k]
    int32_t n, int32_t d, int32_t k,
    int32_t* __restrict__ labels)  // [n]
{
    extern __shared__ float smem[];
    float* Mtile = smem; // TILE_K * d

    const int32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    const float* x = X + static_cast<int64_t>(p) * d;

    float best = FLT_MAX;
    int32_t best_id = -1;

    for (int32_t m0 = 0; m0 < k; m0 += TILE_K) {
        const int32_t chunk = min(TILE_K, k - m0);

        for (int32_t t = threadIdx.x; t < chunk * d; t += blockDim.x) {
            Mtile[t] = M[static_cast<int64_t>(m0) * d + t];
        }
        __syncthreads();

        for (int32_t t = 0; t < chunk; ++t) {
            const float* mptr = Mtile + static_cast<int64_t>(t) * d;

            float dist = 0.f;
            for (int32_t j = 0; j < d; ++j) {
                dist += dabs(x[j] - mptr[j]);
            }
            const int32_t jx = excl[m0 + t];
            dist -= dabs(x[jx] - mptr[jx]);

            if (dist < best) { best = dist; best_id = m0 + t; }
        }
        __syncthreads();
    }
    labels[p] = best_id;
}

template<int TILE_K>
__global__ void k_assign_l1_mask(
    const float* __restrict__ X,   // [n,d]
    const float* __restrict__ M,   // [k,d]
    const bool*  __restrict__ D,   // [k,d]
    int32_t n, int32_t d, int32_t k,
    int32_t* __restrict__ labels)  // [n]
{
    extern __shared__ float smem[];
    float* Mtile = smem; // TILE_K * d

    const int32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    const float* x = X + static_cast<int64_t>(p) * d;

    float best = FLT_MAX;
    int32_t best_id = -1;

    for (int32_t m0 = 0; m0 < k; m0 += TILE_K) {
        const int32_t chunk = min(TILE_K, k - m0);

        for (int32_t t = threadIdx.x; t < chunk * d; t += blockDim.x) {
            Mtile[t] = M[static_cast<int64_t>(m0) * d + t];
        }
        __syncthreads();

        for (int32_t t = 0; t < chunk; ++t) {
            const float* mptr = Mtile + static_cast<int64_t>(t) * d;
            const bool*  dptr = D     + static_cast<int64_t>(m0 + t) * d;

            float dist = 0.f;
            for (int32_t j = 0; j < d; ++j) {
                if (dptr[j]) dist += dabs(x[j] - mptr[j]);
            }
            if (dist < best) { best = dist; best_id = m0 + t; }
        }
        __syncthreads();
    }
    labels[p] = best_id;
}

__global__ void k_replace_medoids_pre_safe(
    int32_t *d_M_idx,
    const int32_t *d_M_idx_best,
    int32_t *d_M_current,
    const int32_t *d_M_random,
    const int32_t *d_M, int32_t Bk,
    const int32_t *d_M_best,
    const bool    *d_bad,
    int32_t k, int32_t /*n_ignored*/)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int32_t j = 0;
    for (int32_t i = 0; i < k; ++i) {
        if (!d_bad[i]) {
            if (j < k) {
                d_M_current[j] = d_M_best[i];
                d_M_idx[j]     = d_M_idx_best[i];
                ++j;
            }
        }
    }

    int32_t p = 0;
    while (j < k) {
        if (p >= Bk) break;
        const int32_t cand = d_M[d_M_random[p]];
        bool is_in = false;
        for (int32_t i = 0; i < j; ++i) {
            if (cand == d_M_current[i]) { is_in = true; break; }
        }
        if (!is_in) {
            d_M_current[j] = cand;
            d_M_idx[j]     = d_M_random[p];
            ++j;
        }
        ++p;
    }
}

// -------------------------- host wrappers (exact signatures) ------------------

static inline int grid_1d(int64_t n, int block) {
    return static_cast<int>((n + block - 1) / block);
}

at::Tensor assign_projected_l1_excl(const at::Tensor& X,
                                    const at::Tensor& M,
                                    const at::Tensor& excl)
{
    TORCH_CHECK(X.is_cuda() && M.is_cuda() && excl.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(X.dtype()==at::kFloat && M.dtype()==at::kFloat, "X,M must be float32");
    TORCH_CHECK(excl.dtype()==at::kInt, "excl must be int32");
    TORCH_CHECK(X.is_contiguous() && M.is_contiguous() && excl.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(X.dim()==2 && M.dim()==2 && excl.dim()==1, "X(n,d), M(k,d), excl(k,)");
    TORCH_CHECK(M.size(1)==X.size(1) && excl.size(0)==M.size(0), "shape mismatch");

    const at::cuda::OptionalCUDAGuard guard(at::device_of(X));
    const int64_t n = X.size(0);
    const int64_t d = X.size(1);
    const int64_t k = M.size(0);

    auto labels = at::empty({n}, X.options().dtype(at::kInt));

    constexpr int BLOCK  = 256;
    constexpr int TILE_K = 32;
    const dim3 grid(grid_1d(n, BLOCK));
    const size_t smem = TILE_K * d * sizeof(float);

    k_assign_l1_excl<TILE_K><<<grid, BLOCK, smem, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), M.data_ptr<float>(), excl.data_ptr<int32_t>(),
        static_cast<int32_t>(n), static_cast<int32_t>(d), static_cast<int32_t>(k),
        labels.data_ptr<int32_t>());

    CUDA_CHECK(cudaPeekAtLastError());
    return labels;
}

at::Tensor assign_projected_l1_mask(const at::Tensor& X,
                                    const at::Tensor& M,
                                    const at::Tensor& D)
{
    TORCH_CHECK(X.is_cuda() && M.is_cuda() && D.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(X.dtype()==at::kFloat && M.dtype()==at::kFloat, "X,M must be float32");
    TORCH_CHECK(D.dtype()==at::kBool, "D must be bool");
    TORCH_CHECK(X.is_contiguous() && M.is_contiguous() && D.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(X.dim()==2 && M.dim()==2 && D.dim()==2, "X(n,d), M(k,d), D(k,d)");
    TORCH_CHECK(M.size(1)==X.size(1) && D.size(0)==M.size(0) && D.size(1)==M.size(1), "shape mismatch");

    const at::cuda::OptionalCUDAGuard guard(at::device_of(X));
    const int64_t n = X.size(0);
    const int64_t d = X.size(1);
    const int64_t k = M.size(0);

    auto labels = at::empty({n}, X.options().dtype(at::kInt));

    constexpr int BLOCK  = 256;
    constexpr int TILE_K = 32;
    const dim3 grid(grid_1d(n, BLOCK));
    const size_t smem = TILE_K * d * sizeof(float);

    k_assign_l1_mask<TILE_K><<<grid, BLOCK, smem, at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), M.data_ptr<float>(), D.data_ptr<bool>(),
        static_cast<int32_t>(n), static_cast<int32_t>(d), static_cast<int32_t>(k),
        labels.data_ptr<int32_t>());

    CUDA_CHECK(cudaPeekAtLastError());
    return labels;
}

void replace_medoids_pre_safe(at::Tensor d_M_idx,
                              at::Tensor d_M_idx_best,
                              at::Tensor d_M_current,
                              at::Tensor d_M_random,
                              at::Tensor d_M,
                              int32_t Bk,
                              at::Tensor d_M_best,
                              at::Tensor d_bad,
                              int32_t k,
                              int32_t n)
{
    TORCH_CHECK(d_M_idx.is_cuda() && d_M_idx_best.is_cuda() &&
                d_M_current.is_cuda() && d_M_random.is_cuda() &&
                d_M.is_cuda() && d_M_best.is_cuda() && d_bad.is_cuda(),
                "all inputs must be CUDA");

    TORCH_CHECK(d_M_idx.dtype()==at::kInt && d_M_idx_best.dtype()==at::kInt &&
                d_M_current.dtype()==at::kInt && d_M_random.dtype()==at::kInt &&
                d_M.dtype()==at::kInt && d_M_best.dtype()==at::kInt,
                "indices must be int32");
    TORCH_CHECK(d_bad.dtype()==at::kBool, "d_bad must be bool");

    const at::cuda::OptionalCUDAGuard guard(at::device_of(d_M_idx));

    k_replace_medoids_pre_safe<<<1,1,0, at::cuda::getCurrentCUDAStream()>>>(
        d_M_idx.data_ptr<int32_t>(),
        d_M_idx_best.data_ptr<int32_t>(),
        d_M_current.data_ptr<int32_t>(),
        d_M_random.data_ptr<int32_t>(),
        d_M.data_ptr<int32_t>(),
        Bk,
        d_M_best.data_ptr<int32_t>(),
        d_bad.data_ptr<bool>(),
        k, n);

    CUDA_CHECK(cudaPeekAtLastError());
}