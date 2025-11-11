from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, shutil

# --- robust CUDA discovery (keeps your conda CUDA 12.4 in sync with torch 2.5.1) ---
from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
if TORCH_CUDA_HOME is None and "CUDA_HOME" not in os.environ:
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_bin = os.path.dirname(nvcc)
        cuda_home = os.path.dirname(cuda_bin)
        if os.path.isdir(cuda_home):
            os.environ["CUDA_HOME"] = cuda_home
# -------------------------------------------------------------------------------

root = Path(__file__).resolve().parent.parent

extra_compile_args = {
    "cxx": ["/O2", "/std:c++17"],  # MSVC-friendly host flags
    "nvcc": [
        "-O3", "-std=c++17",
        "-gencode=arch=compute_70,code=sm_70",
        "-lineinfo",
    ],
}

ext = CUDAExtension(
    name="gpu_proclus_backend",
    sources=[
        str(root / "src" / "torch_binding.cpp"),
        str(root / "src" / "gpu_proclus_kernels.cu"),
    ],
    extra_compile_args=extra_compile_args,
)

setup(
    name="gpu_proclus",
    version="0.1.0",
    description="Scalable GPU PROCLUS core (assignment + medoid replacement)",
    packages=find_packages(where="."),
    package_dir={"": "."},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)