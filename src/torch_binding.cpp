#include <torch/extension.h>
#include "gpu_proclus_kernels.cuh"

// Use explicit module name to ensure PyInit_gpu_proclus_backend is emitted.
PYBIND11_MODULE(gpu_proclus_backend, m) {
  m.def("assign_projected_l1_excl", &assign_projected_l1_excl,
        "Assign labels (projected L1) using per-medoid excluded index");
  m.def("assign_projected_l1_mask", &assign_projected_l1_mask,
        "Assign labels (projected L1) using boolean mask D");
  m.def("replace_medoids_pre_safe", &replace_medoids_pre_safe,
        "Safe medoid replacement helper (device-side)");
}
