import torch
import gpu_proclus_backend as _backend

def assign_l1_excl(X: torch.Tensor, M: torch.Tensor, excl: torch.Tensor) -> torch.Tensor:
    """
    X: (n,d) float32 cuda
    M: (k,d) float32 cuda
    excl: (k,) int32 cuda  (excluded dim per medoid)
    return: labels (n,) int32 cuda
    """
    return _backend.assign_projected_l1_excl(X, M, excl)

def assign_l1_mask(X: torch.Tensor, M: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    X: (n,d) float32 cuda
    M: (k,d) float32 cuda
    D: (k,d) bool cuda  (selected dims per medoid)
    return: labels (n,) int32 cuda
    """
    return _backend.assign_projected_l1_mask(X, M, D)

def replace_medoids_pre_safe(M_idx: torch.Tensor,
                             M_idx_best: torch.Tensor,
                             M_current: torch.Tensor,
                             M_random: torch.Tensor,
                             M: torch.Tensor,
                             Bk: int,
                             M_best: torch.Tensor,
                             bad: torch.Tensor,
                             k: int,
                             n: int) -> None:
    """
    In-place device helper; mirrors the safe single-thread kernel.
    """
    _backend.replace_medoids_pre_safe(M_idx, M_idx_best, M_current,
                                      M_random, M, int(Bk),
                                      M_best, bad, int(k), int(n))