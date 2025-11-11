import torch
from gpu_proclus import assign_l1_excl, assign_l1_mask

torch.manual_seed(0)
device = "cuda"

# synthetic: n=100_000, d=85, k=1000
n, d, k = 100_000, 85, 1000
X = torch.randn(n, d, device=device, dtype=torch.float32)

# medoids as random rows
idx = torch.randint(0, n, (k,), device=device)
M = X.index_select(0, idx).contiguous()

# case A: l ≈ d => exactly one excluded dim per medoid
# build excl vector (random single exclusion)
excl = torch.randint(0, d, (k,), device=device, dtype=torch.int32)

labels_excl = assign_l1_excl(X, M, excl)
print("labels_excl:", labels_excl.shape, labels_excl.dtype, labels_excl.device)

# case B: general mask D (choose l dims per medoid)
l = 84
D = torch.zeros(k, d, device=device, dtype=torch.bool)
for i in range(k):
    keep = torch.randperm(d, device=device)[:l]
    D[i, keep] = True
labels_mask = assign_l1_mask(X, M, D)
print("labels_mask:", labels_mask.shape, labels_mask.dtype, labels_mask.device)