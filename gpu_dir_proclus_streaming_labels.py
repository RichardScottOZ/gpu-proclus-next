#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directory-based raster clustering with GPU-FAST-PROCLUS (projected clustering),
using streaming assignment via custom CUDA kernels.

Pipeline:
  discover rasters (each file = one band, same H×W/transform/CRS)
  -> compute per-band mean/std on CPU (optional)
  -> FIT PROCLUS on either: full data | sampled pixels | pre-saved artifacts
  -> STREAM tiles: standardize, assign projected L1 with GPU kernels
  -> write labels GeoTIFF + JSON artifacts

Requires:
  - PyTorch (CUDA), rasterio, numpy
  - Your compiled extension 'gpu_proclus_backend' exposing:
      assign_projected_l1_excl(X, M, excl)  # X(n,d) float32 cuda, M(k,d), excl(k,)
      assign_projected_l1_mask(X, M, D)     # X(n,d) float32 cuda, M(k,d), D(k,d) bool
"""

import sys, os
repo_root = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(repo_root, "python")
if os.path.isdir(pkg_dir) and pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

# NEW: Import labels-only FIT helper from parent directory
parent_dir = os.path.dirname(repo_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


try:
    from proclus_labels import fit_proclus_labels_py
except ImportError:
    fit_proclus_labels_py = None  # Will error if --labels_runtime is used



try:
    import torch
    _TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_TORCH_LIB):
        try:
            os.add_dll_directory(_TORCH_LIB)  # Python 3.8+ on Windows
        except Exception:
            os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
    # Conda often puts core DLLs in Library\bin
    _CONDA_BIN = os.path.join(os.environ.get("CONDA_PREFIX", ""), "Library", "bin")
    if os.path.isdir(_CONDA_BIN):
        try:
            os.add_dll_directory(_CONDA_BIN)
        except Exception:
            os.environ["PATH"] = _CONDA_BIN + os.pathsep + os.environ.get("PATH", "")
except Exception:
    # If torch cannot import, the extension will fail anyway; let the import show a clear error
    pass


#import gpu_proclus_backend

import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rasterio as rio
from rasterio.windows import Window
import torch

# ---------- repo kernels (bindings) ----------
try:
    import gpu_proclus_backend as _backend
except Exception as e:
    raise RuntimeError(
        "Could not import gpu_proclus_backend. "
        "Build your extension first (see instructions below)."
    ) from e

# Make <repo_root>/python importable when running from repo root
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent / "python"))

def assign_l1_excl(X: torch.Tensor, M: torch.Tensor, excl: torch.Tensor) -> torch.Tensor:
    return _backend.assign_projected_l1_excl(X, M, excl)

def assign_l1_mask(X: torch.Tensor, M: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    return _backend.assign_projected_l1_mask(X, M, D)

# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser(
        description="GPU-FAST-PROCLUS projected clustering with streaming assignment."
    )
    p.add_argument("--root_dir", required=True, help="Directory tree of *.tif/*.tiff (one band per file).")
    p.add_argument("--include_dirs_regex", default="", help="Regex to include subdirs by name.")
    p.add_argument("--exclude_dirs_regex", default="", help="Regex to exclude subdirs by name.")
    p.add_argument("--output_tif", required=True, help="Output labels GeoTIFF (int32, nodata=-1).")
    p.add_argument("--artifacts_dir", default="", help="Optional directory to write artifacts JSON.")
    p.add_argument("--save_band_list", default="", help="Optional: write discovered band list (txt).")

    # PROCLUS selection & params
    p.add_argument("--proclus_repo", required=True, help="Path to local GPU_PROCLUS repo root (must contain python/).")
    p.add_argument("--variant",
                   choices=["GPU_FAST_PROCLUS", "GPU_PROCLUS", "FAST_PROCLUS", "PROCLUS", "FAST*_PROCLUS", "GPU_FAST*_PROCLUS"],
                   default="GPU_FAST_PROCLUS",
                   help="Algorithm variant to use for FIT only.")
    p.add_argument("--k", type=int, required=True, help="Clusters (medoids).")
    p.add_argument("--l", type=int, required=True, help="Projected dimensions per cluster.")
    p.add_argument("--a", type=int, default=0, help="Candidate pool size (0 => min(100, N//k)).")
    p.add_argument("--b", type=int, default=10, help="Neighborhood parameter.")
    p.add_argument("--min_deviation", type=float, default=0.7, help="Min deviation threshold.")
    p.add_argument("--termination_rounds", type=int, default=5, help="Termination rounds.")

    # FIT mode
    p.add_argument("--fit_mode", choices=["full", "sample", "artifacts"], default="sample",
                   help="How to obtain medoids+projection for assignment.")
    p.add_argument("--max_fit_pixels", type=int, default=2_000_000,
                   help="Max valid pixels to reservoir-sample for fitting when fit_mode=sample.")
    p.add_argument("--fit_artifacts_in", default="",
                   help="JSON file with saved medoids/projection to LOAD (fit_mode=artifacts).")
    p.add_argument("--fit_artifacts_out", default="",
                   help="JSON file to SAVE fit artifacts (used by full/sample modes).")

    # IO & performance
    # NEW: labels runtime flag
    p.add_argument("--labels_runtime", action="store_true",
                   help="Use Python labels-only FIT (no O(n*k) memory)")

    p.add_argument("--tile_rows", type=int, default=512, help="Rows per tile when reading.")
    p.add_argument("--standardize", action="store_true", help="Z-score per band.")
    p.add_argument("--compress", choices=["LZW", "ZSTD", "DEFLATE", "NONE"], default="LZW")
    p.add_argument("--seed", type=int, default=123, help="Random seed for sampling.")
    # Assignment kernel choice
    p.add_argument("--assign_mode", choices=["auto", "mask", "excl"], default="auto",
                   help="Use D-mask, excl index, or auto (excl only if l == d-1).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()

# ---------- discovery ----------
def discover_tifs(root: Path, inc_regex: str, exc_regex: str) -> List[Path]:
    inc = re.compile(inc_regex) if inc_regex else None
    exc = re.compile(exc_regex) if exc_regex else None
    tifs: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        dname = Path(dirpath).name
        if inc and not inc.search(dname): continue
        if exc and exc.search(dname): continue
        for f in filenames:
            if f.lower().endswith((".tif", ".tiff")):
                tifs.append(Path(dirpath) / f)
    tifs = sorted(tifs)
    if not tifs:
        raise FileNotFoundError(f"No GeoTIFFs found under {root}")
    return tifs

# ---------- geo/meta ----------
def read_geometry_meta(first_path: Path) -> Tuple[int, int, dict]:
    with rio.open(first_path) as ds0:
        meta = ds0.meta.copy()
        H, W = ds0.height, ds0.width
    return H, W, meta

# ---------- stats ----------
def compute_band_stats_cpu(paths: List[Path], H: int, W: int, tile_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (means, stds) as float32 arrays of length B, skipping NODATA/NaN."""
    B = len(paths)
    sums = np.zeros(B, dtype=np.float64)
    sumsqs = np.zeros(B, dtype=np.float64)
    counts = np.zeros(B, dtype=np.int64)

    dsets = [rio.open(p) for p in paths]
    try:
        nodatas = [ds.nodata for ds in dsets]
        for r0 in range(0, H, tile_rows):
            h = min(tile_rows, H - r0)
            cols, masks = [], []
            for j, ds in enumerate(dsets):
                band = ds.read(1, window=Window(0, r0, W, h))
                c = band.reshape(-1)
                cols.append(c)
                nd = nodatas[j]
                if nd is not None and not np.isnan(nd):
                    m = (c == nd)
                else:
                    m = np.isnan(c) if np.issubdtype(c.dtype, np.floating) else np.zeros_like(c, dtype=bool)
                masks.append(m)
            Xb = np.stack(cols, axis=1)  # (h*W, B)
            mask = masks[0].copy()
            for m in masks[1:]:
                mask |= m
            valid = ~mask
            if valid.any():
                Xv = Xb[valid]
                sums   += Xv.sum(axis=0, dtype=np.float64)
                sumsqs += (Xv.astype(np.float64) ** 2).sum(axis=0, dtype=np.float64)
                counts += Xv.shape[0]
    finally:
        for ds in dsets:
            ds.close()
    means = sums / np.maximum(counts, 1)
    var   = np.maximum(sumsqs / np.maximum(counts, 1) - means**2, 1e-12)
    stds  = np.sqrt(var).astype(np.float32)
    return means.astype(np.float32), stds

# ---------- PROCLUS loader ----------
def import_proclus_old(repo_root: Path):
    sys.path.insert(0, str(repo_root / "python"))
    from GPU_PROCLUS import (
        GPU_FAST_PROCLUS, GPU_PROCLUS,
        FAST_PROCLUS, PROCLUS, FAST_star_PROCLUS, GPU_FAST_star_PROCLUS
    )
    return {
        "GPU_FAST_PROCLUS": GPU_FAST_PROCLUS,
        "GPU_PROCLUS": GPU_PROCLUS,
        "FAST_PROCLUS": FAST_PROCLUS,
        "PROCLUS": PROCLUS,
        "FAST*_PROCLUS": FAST_star_PROCLUS,
        "GPU_FAST*_PROCLUS": GPU_FAST_star_PROCLUS,
    }


from pathlib import Path
import sys
import importlib

def import_proclus_old2(repo_root: Path, verbose: bool = False):
    """
    Import PROCLUS entry points from <repo_root>/python/gpu_proclus (preferred)
    or <repo_root>/python/GPU_PROCLUS (fallback).
    """
    proclus_py_dir = str(repo_root / "python")
    if proclus_py_dir not in sys.path:
        sys.path.insert(0, proclus_py_dir)
        if verbose:
            print(f"[Import] Added to sys.path: {proclus_py_dir}")

    # Try gpu_proclus first
    try:
        gp = importlib.import_module("gpu_proclus")
        if verbose:
            print("[Import] Using package 'gpu_proclus' from", proclus_py_dir)
        return {
            "GPU_FAST_PROCLUS": gp.GPU_FAST_PROCLUS,
            "GPU_PROCLUS":      gp.GPU_PROCLUS,
            "FAST_PROCLUS":     gp.FAST_PROCLUS,
            "PROCLUS":          gp.PROCLUS,
            "FAST*_PROCLUS":    gp.FAST_star_PROCLUS,
            "GPU_FAST*_PROCLUS":gp.GPU_FAST_star_PROCLUS,
        }
    except Exception as e_lower:
        if verbose:
            print(f"[Import] gpu_proclus import failed: {e_lower!r}")

    # Fallback: GPU_PROCLUS
    try:
        gp = importlib.import_module("GPU_PROCLUS")
        if verbose:
            print("[Import] Using package 'GPU_PROCLUS' from", proclus_py_dir)
        return {
            "GPU_FAST_PROCLUS": gp.GPU_FAST_PROCLUS,
            "GPU_PROCLUS":      gp.GPU_PROCLUS,
            "FAST_PROCLUS":     gp.FAST_PROCLUS,
            "PROCLUS":          gp.PROCLUS,
            "FAST*_PROCLUS":    gp.FAST_star_PROCLUS,
            "GPU_FAST*_PROCLUS":gp.GPU_FAST_star_PROCLUS,
        }
    except Exception as e_upper:
        raise RuntimeError(
            f"Could not import PROCLUS from {proclus_py_dir}; "
            "tried 'gpu_proclus' and 'GPU_PROCLUS'. "
            "Verify your repo path and that python/gpu_proclus/__init__.py exists."
        ) from e_upper    


# --- imports already near the top of your file ---
# import importlib        # and
# from pathlib import Path
# import sys, os
# -------------------------------------------------

def import_proclus(repo_root: Path, verbose: bool = False):
    """
    Import PROCLUS entry points from <repo_root>/python/gpu_proclus (preferred)
    or <repo_root>/python/GPU_PROCLUS (fallback).

    Notes:
    - GPU_PROCLUS.py uses 'from data.generator import *' → needs <repo_root> on sys.path
    - GPU_PROCLUS JIT compiles with relative paths like 'src\...' → requires CWD=<repo_root>
    """
    proclus_root   = str(repo_root)             # e.g. ...\GPU_PROCLUS
    proclus_py_dir = str(repo_root / "python")  # e.g. ...\GPU_PROCLUS\python

    # Ensure both the repo root (for 'data/...') and python/ are importable
    if proclus_root not in sys.path:
        sys.path.insert(0, proclus_root)
        if verbose:
            print(f"[Import] Added to sys.path (root):   {proclus_root}")

    if proclus_py_dir not in sys.path:
        sys.path.insert(0, proclus_py_dir)
        if verbose:
            print(f"[Import] Added to sys.path (python): {proclus_py_dir}")

    # (Optional) shorten PyTorch extension build path to avoid MAX_PATH issues on Windows
    # os.environ.setdefault("TORCH_EXTENSIONS_DIR", r"C:\_torch_ext")

    # Temporarily change CWD so relative JIT source paths like 'src\map\...' resolve
    old_cwd = os.getcwd()
    try:
        os.chdir(proclus_root)

        # Try modern package first
        try:
            gp = importlib.import_module("gpu_proclus")
            if verbose:
                print("[Import] Using package 'gpu_proclus'")
            return {
                "GPU_FAST_PROCLUS":  gp.GPU_FAST_PROCLUS,
                "GPU_PROCLUS":       gp.GPU_PROCLUS,
                "FAST_PROCLUS":      gp.FAST_PROCLUS,
                "PROCLUS":           gp.PROCLUS,
                "FAST*_PROCLUS":     gp.FAST_star_PROCLUS,
                "GPU_FAST*_PROCLUS": gp.GPU_FAST_star_PROCLUS,
            }
        except Exception as e_lower:
            if verbose:
                print(f"[Import] gpu_proclus import failed: {e_lower!r}")

        # Fallback: legacy module (relies on 'data.generator')
        gp = importlib.import_module("GPU_PROCLUS")
        if verbose:
            print("[Import] Using module 'GPU_PROCLUS'")
        return {
            "GPU_FAST_PROCLUS":  gp.GPU_FAST_PROCLUS,
            "GPU_PROCLUS":       gp.GPU_PROCLUS,
            "FAST_PROCLUS":      gp.FAST_PROCLUS,
            "PROCLUS":           gp.PROCLUS,
            "FAST*_PROCLUS":     gp.FAST_star_PROCLUS,
            "GPU_FAST*_PROCLUS": gp.GPU_FAST_star_PROCLUS,
        }

    finally:
        os.chdir(old_cwd)
        
# ---------- sampling for fit ----------
def reservoir_sample_per_tile(paths: List[Path], H: int, W: int, tile_rows: int,
                              means: np.ndarray, stds: np.ndarray, standardize: bool,
                              max_fit_pixels: int, seed: int) -> np.ndarray:
    """
    Reservoir-sample up to max_fit_pixels valid rows into a (S,B) float32 array.
    """
    rng = np.random.default_rng(seed)
    B = len(paths)
    S = max_fit_pixels
    buf = np.empty((S, B), dtype=np.float32)
    seen = 0

    dsets = [rio.open(p) for p in paths]
    try:
        nodatas = [ds.nodata for ds in dsets]
        for r0 in range(0, H, tile_rows):
            h = min(tile_rows, H - r0)
            cols, masks = [], []
            for j, ds in enumerate(dsets):
                band = ds.read(1, window=Window(0, r0, W, h))
                c = band.reshape(-1).astype(np.float32, copy=False)
                cols.append(c)
                nd = nodatas[j]
                if nd is not None and not np.isnan(nd):
                    m = (c == nd)
                else:
                    m = np.isnan(c) if np.issubdtype(c.dtype, np.floating) else np.zeros_like(c, dtype=bool)
                masks.append(m)

            Xb = np.stack(cols, axis=1)  # (h*W, B)
            mask = masks[0].copy()
            for m in masks[1:]:
                mask |= m
            valid = ~mask
            if not valid.any():
                continue

            Xv = Xb[valid]
            if standardize:
                if Xv.shape[0] > 0:
                    Xv = (Xv - means) / stds

            # reservoir sample rows of Xv
            for row in Xv:
                if seen < S:
                    buf[seen] = row
                else:
                    j = rng.integers(0, seen + 1)
                    if j < S:
                        buf[j] = row
                seen += 1
    finally:
        for ds in dsets:
            ds.close()

    if seen == 0:
        raise RuntimeError("No valid pixels found for sampling.")
    take = min(S, seen)
    return buf[:take].copy()

# ---------- helpers to parse PROCLUS output ----------
def _flatten(seq):
    if torch.is_tensor(seq) or isinstance(seq, (np.ndarray,)):
        return [seq]
    if isinstance(seq, (list, tuple)):
        out = []
        for x in seq:
            out.extend(_flatten(x))
        return out
    return [seq]

def _len1d(x):
    if torch.is_tensor(x):      return x.numel() if x.ndim == 1 else None
    if isinstance(x, np.ndarray): return x.size if x.ndim == 1 else None
    if isinstance(x, (list, tuple)) and all(isinstance(i, (int, np.integer)) for i in x):
        return len(x)
    return None

def _to_numpy1d_int(x):
    if torch.is_tensor(x):   return x.detach().cpu().numpy().astype(np.int32, copy=False)
    if isinstance(x, np.ndarray): return x.astype(np.int32, copy=False)
    if isinstance(x, (list, tuple)): return np.asarray(x, dtype=np.int32)
    raise TypeError(f"Cannot convert type {type(x)} to 1D int array")

def parse_proclus_return(out, N_expected: Optional[int], k: int):
    """
    Try to extract (labels, medoids_idx_optional, cluster_dims_optional) from any variant.
    """
    flat = _flatten(out)
    labels = None
    medoids_idx = None
    cluster_dims = None

    # labels: first 1D whose length matches N_expected (if given). If not, pick the longest.
    if N_expected is not None:
        for x in flat:
            if _len1d(x) == N_expected:
                labels = x; break
    if labels is None:
        # pick the first largest 1D vector
        one_d = [(x, _len1d(x)) for x in flat if _len1d(x) is not None]
        if one_d:
            one_d.sort(key=lambda t: t[1], reverse=True)
            labels = one_d[0][0]

    # medoids: 1D length k (avoid identical object as labels)
    for x in flat:
        if _len1d(x) == k and x is not labels:
            medoids_idx = x; break

    # cluster_dims: list/listlike length k of per-cluster dims
    for x in out if isinstance(out, (list, tuple)) else []:
        if isinstance(x, (list, tuple)) and len(x) == k and all(isinstance(y, (list, tuple, np.ndarray, torch.Tensor)) for y in x):
            cluster_dims = x; break

    return labels, medoids_idx, cluster_dims

# ---------- build projector (D or excl) ----------
def projector_from_cluster_dims(cluster_dims, d: int, l: int):
    """
    Build boolean mask D (k,d) from cluster_dims (iterables of dims).
    """
    k = len(cluster_dims)
    D = np.zeros((k, d), dtype=bool)
    for i, dims in enumerate(cluster_dims):
        dims_np = dims.detach().cpu().numpy() if torch.is_tensor(dims) else np.asarray(dims)
        if dims_np.size < l:
            # clip/extend if needed
            dims_np = dims_np[:l]
        D[i, np.clip(dims_np, 0, d-1)] = True
    return D

def excl_from_dims_if_l_eq_d_minus_1(cluster_dims, d: int, l: int):
    """
    If l == d-1, we can represent the projection per-cluster by the single excluded dim.
    """
    if d - l != 1:
        return None
    k = len(cluster_dims)
    excl = np.empty((k,), dtype=np.int32)
    full = set(range(d))
    for i, dims in enumerate(cluster_dims):
        dims_np = dims.detach().cpu().numpy() if torch.is_tensor(dims) else np.asarray(dims)
        dims_set = set(int(x) for x in dims_np.tolist())
        missing = list(full - dims_set)
        if len(missing) != 1:
            # Fallback: cannot find exactly one missing dim
            return None
        excl[i] = missing[0]
    return excl

# ---------- write labels window ----------
def write_labels_window(dst, win: Window, labels_1d: np.ndarray, nodata_value: int = -1):
    """
    labels_1d: length h*W for this window (already includes -1 for invalid).
    """
    h, w = win.height, win.width
    img = labels_1d.reshape(h, w)
    dst.write(img, 1, window=win)

# ---------- main ----------
def main():
    args = get_args()
    root = Path(args.root_dir)
    Path(args.output_tif).parent.mkdir(parents=True, exist_ok=True)
    if args.artifacts_dir:
        Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)

    # Discover rasters
    paths = discover_tifs(root, args.include_dirs_regex, args.exclude_dirs_regex)
    if args.save_band_list:
        Path(args.save_band_list).write_text("\n".join(map(str, paths)), encoding="utf-8")
    print(f"[Discovery] Found {len(paths)} GeoTIFF bands.")

    # Geometry/meta
    H, W, meta = read_geometry_meta(paths[0])
    N, B = H * W, len(paths)
    print(f"[Info] Grid H={H} W={W} N={N:,} Bands={B}")

    # Stats
    if args.standardize:
        means, stds = compute_band_stats_cpu(paths, H, W, args.tile_rows)
        print("[Stats] Per-band mean/std computed on CPU.")
    else:
        means = np.zeros(B, dtype=np.float32)
        stds  = np.ones(B, dtype=np.float32)

    # ---- FIT PHASE (patched) ----
    k, l = args.k, args.l
    M_np: Optional[np.ndarray] = None  # medoid feature vectors (k,d) float32
    D_np: Optional[np.ndarray] = None  # (k,d) bool
    excl_np: Optional[np.ndarray] = None  # (k,) int32 (when l == d-1)

    def _build_X_for_fit(paths, H, W, B, means, stds, args):
        """
        Build the FIT matrix X as torch.float32 CUDA tensor [n, d].
        Reuses your existing 'full' and 'sample' logic.
        """
        if args.fit_mode == "full":
            # Build full feature matrix on CPU (like non-streaming), then H2D
            X_cpu = np.empty((H * W, B), dtype=np.float32)
            valid_mask = np.ones(H * W, dtype=bool)
            dsets = [rio.open(p) for p in paths]
            try:
                nodatas = [ds.nodata for ds in dsets]
                base = 0
                for r0 in range(0, H, args.tile_rows):
                    h = min(args.tile_rows, H - r0)
                    cols, masks = [], []
                    for j, ds in enumerate(dsets):
                        band = ds.read(1, window=Window(0, r0, W, h))
                        c = band.reshape(-1).astype(np.float32, copy=False)
                        cols.append(c)
                        nd = nodatas[j]
                        if nd is not None and not np.isnan(nd):
                            m = (c == nd)
                        else:
                            m = np.isnan(c) if np.issubdtype(c.dtype, np.floating) else np.zeros_like(c, dtype=bool)
                        masks.append(m)
                    Xb = np.stack(cols, axis=1)
                    mask = masks[0].copy()
                    for m in masks[1:]:
                        mask |= m
                    if args.standardize:
                        Xb_valid = Xb[~mask]
                        if Xb_valid.shape[0] > 0:
                            Xb[~mask] = (Xb_valid - means) / stds
                    X_cpu[base:base + h * W, :] = Xb
                    valid_mask[base:base + h * W] = ~mask
                    base += h * W
            finally:
                for ds in dsets:
                    ds.close()
            X = torch.from_numpy(X_cpu).pin_memory().to(device="cuda", dtype=torch.float32, non_blocking=True)
            return X

        # Sampled fit matrix
        X_sample = reservoir_sample_per_tile(paths, H, W, args.tile_rows,
                                            means, stds, args.standardize,
                                            args.max_fit_pixels, args.seed)
        print(f"[Fit] Sampled {X_sample.shape[0]:,} valid pixels for fitting.")
        X = torch.from_numpy(X_sample).to(device="cuda", dtype=torch.float32, non_blocking=False)
        return X


    if args.fit_mode in ("full", "sample"):
        # 1) Build X for BOTH labels_runtime and CUDA paths
        X = _build_X_for_fit(paths, H, W, B, means, stds, args)

        # 2) Branch: labels-only Python FIT vs CUDA PROCLUS
        if args.labels_runtime:
            # Python labels-only FIT path (memory-safe for large k)
            if fit_proclus_labels_py is None:
                raise RuntimeError("Could not import proclus_labels. Ensure proclus_labels.py is in parent directory.")
            print(f"[FIT][labels_runtime] k={k} l={l} b={args.b} (Python labels-only, no O(n*k) memory)")

            M_indices, D_mask_t = fit_proclus_labels_py(
                X, k, l, args.a, args.b, args.min_deviation, args.termination_rounds,
                seed=args.seed, tile_n=131072
            )

            # Extract medoid feature vectors and D-mask
            M_np = X[M_indices].detach().cpu().numpy().astype("float32", copy=False)
            D_np = D_mask_t.detach().cpu().numpy().astype("bool", copy=False)

            # Try to build excl if l == d-1
            excl_np = excl_from_dims_if_l_eq_d_minus_1(
                [D_mask_t[i].nonzero(as_tuple=True)[0] for i in range(k)],
                d=M_np.shape[1], l=l
            )

        else:
            # Original CUDA/compiled PROCLUS path
            proclus_map = import_proclus(Path(args.proclus_repo), verbose=args.verbose)
            proclus_fn = proclus_map[args.variant]

            # Candidate pool size 'a' (fixes previously undefined variable)
            a = min(100, X.shape[0] // k) if args.a == 0 else args.a
            print(
                f"[PROCLUS][{args.variant}] k={k} l={l} a={a} b={args.b} "
                f"min_dev={args.min_deviation} term_rounds={args.termination_rounds}"
            )

            # Run PROCLUS
            out = proclus_fn(X, k, l, a, args.b, args.min_deviation, args.termination_rounds)
            labels_fit, medoids_idx, cluster_dims = parse_proclus_return(out, X.shape[0], k)

            # Derive medoid vectors
            if medoids_idx is None:
                candidates = [x for x in _flatten(out)
                            if (torch.is_tensor(x) or isinstance(x, np.ndarray))
                            and getattr(x, "ndim", 0) == 2
                            and x.shape[0] == k and x.shape[1] == X.shape[1]]
                if len(candidates) > 0:
                    M_np = (candidates[0].detach().cpu().numpy()
                            if torch.is_tensor(candidates[0]) else candidates[0]).astype(np.float32, copy=False)
                else:
                    raise RuntimeError("Could not recover medoid feature vectors (k,d) from PROCLUS output.")
            else:
                idx = _to_numpy1d_int(medoids_idx)
                X_cpu_for_m = X.detach().cpu().numpy().astype(np.float32, copy=False)
                M_np = X_cpu_for_m[idx]

            # Build projector (prefer D mask; excl only when exactly l == d-1)
            if cluster_dims is not None:
                D_np = projector_from_cluster_dims(cluster_dims, d=M_np.shape[1], l=l)
                excl_np = excl_from_dims_if_l_eq_d_minus_1(cluster_dims, d=M_np.shape[1], l=l)
            else:
                D_np = np.ones((k, M_np.shape[1]), dtype=bool)
                excl_np = None

            # Save artifacts if requested
            if args.fit_artifacts_out:
                payload = {
                    "variant": args.variant,
                    "params": {
                        "k": int(k), "l": int(l), "a": int(a), "b": int(args.b),
                        "min_deviation": float(args.min_deviation),
                        "termination_rounds": int(args.termination_rounds)
                    },
                    "bands": [str(p) for p in paths],
                    "grid": {"H": int(H), "W": int(W), "B": int(B)},
                    "medoids": M_np.tolist(),
                    "D_mask": D_np.astype(bool).tolist() if D_np is not None else [],
                    "excl": excl_np.astype(int).tolist() if excl_np is not None else []
                }
                Path(args.fit_artifacts_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.fit_artifacts_out).write_text(json.dumps(payload, indent=2))
                print(f"[Artifacts] Saved fit artifacts -> {args.fit_artifacts_out}")

    elif args.fit_mode == "artifacts":
        if not args.fit_artifacts_in:
            raise ValueError("--fit_artifacts_in is required when fit_mode=artifacts")
        payload = json.loads(Path(args.fit_artifacts_in).read_text())
        M_np = np.asarray(payload["medoids"], dtype=np.float32)
        D_list = payload.get("D_mask", [])
        excl_list = payload.get("excl", [])
        D_np = np.asarray(D_list, dtype=bool) if len(D_list) > 0 else None
        excl_np = np.asarray(excl_list, dtype=np.int32) if len(excl_list) > 0 else None
        print(
            f"[Artifacts] Loaded medoids (k={M_np.shape[0]}, d={M_np.shape[1]}). "
            f"D_mask={'yes' if D_np is not None else 'no'}, excl={'yes' if excl_np is not None else 'no'}"
        )
    else:
        raise ValueError(f"Unknown fit_mode: {args.fit_mode}")

    # Guards before ASSIGN
    if M_np is None:
        raise RuntimeError("Medoids array (M_np) is None after FIT; check FIT path.")

    # ---- ASSIGN (STREAM) PHASE ----
    # Decide assignment mode
    d = M_np.shape[1]
    use_excl = (args.assign_mode == "excl") or (args.assign_mode == "auto" and (d - l == 1) and (excl_np is not None))
    use_mask = not use_excl

    # Move M and projector to CUDA
    M = torch.from_numpy(M_np).contiguous().to(device="cuda", dtype=torch.float32)
    D = None
    excl = None
    if use_mask:
        if D_np is None:
            raise RuntimeError("assign_mode requires D_mask but none was available. "
                               "Either fit produced no dims or you forced mode=mask.")
        D = torch.from_numpy(D_np).contiguous().to(device="cuda", dtype=torch.bool)
        print("[Assign] Using assign_projected_l1_mask (general).")
    else:
        if excl_np is None:
            raise RuntimeError("assign_mode=excl but 'excl' indices are unavailable.")
        excl = torch.from_numpy(excl_np).contiguous().to(device="cuda", dtype=torch.int32)
        print("[Assign] Using assign_projected_l1_excl (fast path l=d-1).")

    # Prepare output GeoTIFF
    meta2 = meta.copy()
    meta2.update(count=1, dtype='int32', nodata=-1)
    if args.compress != "NONE":
        meta2.update(compress=args.compress, tiled=True)
    with rio.open(args.output_tif, 'w', **meta2) as dst:
        # Stream tiles
        dsets = [rio.open(p) for p in paths]
        try:
            nodatas = [ds.nodata for ds in dsets]
            for r0 in range(0, H, args.tile_rows):
                h = min(args.tile_rows, H - r0)
                cols, masks = [], []
                for j, ds in enumerate(dsets):
                    band = ds.read(1, window=Window(0, r0, W, h))
                    c = band.reshape(-1).astype(np.float32, copy=False)
                    cols.append(c)
                    nd = nodatas[j]
                    if nd is not None and not np.isnan(nd):
                        m = (c == nd)
                    else:
                        m = np.isnan(c) if np.issubdtype(c.dtype, np.floating) else np.zeros_like(c, dtype=bool)
                    masks.append(m)

                Xb = np.stack(cols, axis=1)  # (h*W, B)
                mask = masks[0].copy()
                for m in masks[1:]:
                    mask |= m

                if args.standardize:
                    Xb_valid = Xb[~mask]
                    if Xb_valid.shape[0] > 0:
                        Xb[~mask] = (Xb_valid - means) / stds

                # Move to CUDA and assign
                X_tile = torch.from_numpy(Xb).contiguous().to(device="cuda", dtype=torch.float32)
                if use_mask:
                    labels_gpu = assign_l1_mask(X_tile, M, D)
                else:
                    labels_gpu = assign_l1_excl(X_tile, M, excl)

                labels = labels_gpu.detach().cpu().numpy().astype(np.int32, copy=False)
                labels[mask] = -1  # set invalid pixels to nodata label

                write_labels_window(dst, Window(0, r0, W, h), labels, nodata_value=-1)

                # free CUDA tensors for this tile
                del X_tile, labels_gpu
                torch.cuda.empty_cache()
                print(f"[Assign] Rows {r0:>7}..{r0+h-1:<7} done")
        finally:
            for ds in dsets:
                ds.close()

    print(f"[Done] Wrote labels GeoTIFF: {args.output_tif}")

    # Artifacts directory summary (optional)
    if args.artifacts_dir:
        # (Optional) compute simple cluster histogram by re-reading the output window-by-window
        # to avoid holding the label raster in memory.
        counts = None
        with rio.open(args.output_tif) as lab:
            for r0 in range(0, H, args.tile_rows):
                h = min(args.tile_rows, H - r0)
                arr = lab.read(1, window=Window(0, r0, W, h)).reshape(-1)
                arr = arr[arr >= 0]
                if arr.size == 0:
                    continue
                mx = arr.max()
                if counts is None:
                    counts = np.bincount(arr, minlength=int(mx)+1)
                else:
                    if mx + 1 > counts.size:
                        counts = np.pad(counts, (0, int(mx)+1-counts.size), constant_values=0)
                    counts[:mx+1] += np.bincount(arr, minlength=int(mx)+1)

        outdir = Path(args.artifacts_dir)
        artifacts = {
            "variant": args.variant,
            "fit_mode": args.fit_mode,
            "params": {
                "k": int(k), "l": int(l), "a": int(args.a), "b": int(args.b),
                "min_deviation": float(args.min_deviation),
                "termination_rounds": int(args.termination_rounds)
            },
            "bands": [str(p) for p in paths],
            "grid": {"H": int(H), "W": int(W), "N": int(N), "B": int(B)},
            "assignment": "excl" if use_excl else "mask",
            "medoids_shape": list(M_np.shape),
            "has_D": bool(D_np is not None),
            "has_excl": bool(excl_np is not None),
        }
        if counts is not None and counts.size > 0:
            order = np.argsort(counts)[::-1]
            artifacts["label_summary"] = {
                "labeled_pixels": int(counts.sum()),
                "non_empty_clusters": int(np.count_nonzero(counts)),
                "top_clusters": [{"cluster_id": int(cid), "count": int(counts[cid])}
                                 for cid in order[: min(50, counts.size)]]
            }
        (outdir / "proclus_streaming_artifacts.json").write_text(json.dumps(artifacts, indent=2))
        print(f"[Artifacts] {outdir / 'proclus_streaming_artifacts.json'}")


if __name__ == "__main__":
    main()