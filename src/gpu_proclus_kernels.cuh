#pragma once
#include <cstdint>
#include <torch/extension.h>

// ---- Canonical API (use at::Tensor everywhere; exact same signatures in .cu) ----

// Fast path: l ≈ d (e.g., d=85, l=84). One excluded dim per medoid.
// X: (n,d) float32 CUDA contiguous
// M: (k,d) float32 CUDA contiguous
// excl: (k,) int32 CUDA contiguous
// returns (n,) int32 CUDA
at::Tensor assign_projected_l1_excl(const at::Tensor& X,
                                    const at::Tensor& M,
                                    const at::Tensor& excl);

// General path: boolean mask D (k,d). Works for arbitrary l.
// D: (k,d) bool CUDA contiguous
at::Tensor assign_projected_l1_mask(const at::Tensor& X,
                                    const at::Tensor& M,
                                    const at::Tensor& D);

// Safe single-thread medoid replacement helper (device-side).
// All tensors must be CUDA tensors with the dtypes described below.
// d_M_idx, d_M_idx_best, d_M_current, d_M_random, d_M, d_M_best: int32
// d_bad: bool
void replace_medoids_pre_safe(at::Tensor d_M_idx,
                              at::Tensor d_M_idx_best,
                              at::Tensor d_M_current,
                              at::Tensor d_M_random,
                              at::Tensor d_M,
                              int32_t Bk,
                              at::Tensor d_M_best,
                              at::Tensor d_bad,
                              int32_t k,
                              int32_t n);