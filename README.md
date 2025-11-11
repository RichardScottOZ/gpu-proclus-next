# gpu-proclus-next
An experiment in bigger K


A clean, scalable CUDA core for projected k‑medoids (PROCLUS/GPU‑FAST‑PROCLUS) that:
- assigns 6.48M × 85 features with k=1000 on a 16 GB Tesla
- uses constant shared memory (tiled over k)
- exposes a simple PyTorch API

This repo focuses on the **hot kernels**:
- `assign_projected_l1_excl()`: fast assignment when `l ≈ d` (e.g., d=85, l=84).  
  It uses the complement mask: one excluded dimension per medoid.
- `assign_projected_l1_mask()`: general assignment with a boolean mask D (k×d).
- `replace_medoids_pre_safe()`: safe medoid replacement helper.

## Build (editable install)
```bash
cd python
pip install -e .