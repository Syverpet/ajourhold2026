# Demo Conda Project with Pixi

Windows Pixi demo project for host 6737osl with a default GPU environment and a CPU fallback.

## What Pixi Is

[Pixi](https://pixi.prefix.dev/latest/python/pytorch/) is the project-level environment manager for this repository. It reads `pyproject.toml`, resolves the Conda and PyPI dependencies declared there, creates reproducible environments, and runs named project tasks.

The main commands used here are:

- `pixi install`: resolve and install the project environment from `pyproject.toml` and `pixi.lock`
- `pixi run <task>`: run a named task from `pyproject.toml`
- `pixi add <package>`: add a dependency and update `pyproject.toml`
- `pixi add --pypi <package>`: add a PyPI dependency instead of a Conda package
- `pixi install -e cpu` or `pixi run -e cpu ...`: target the CPU fallback environment instead of the default GPU one

## Quick Start

GPU environment:

```powershell
cd hosts\6737osl\python\demo-conda-gpu
pixi install
pixi run install-kernel-gpu
pixi run validate-env
```

CPU fallback:

```powershell
pixi install -e cpu
pixi run -e cpu install-kernel-cpu
pixi run -e cpu validate-env
```

Open `demo-windows-conda.ipynb` and select either `Python (demo-conda-gpu)` or `Python (demo-conda-cpu)`.

## Why `conda-forge` for most packages and PyPI for GPU PyTorch

- `conda-forge` provides the base Conda environment: Python, Jupyter support, geospatial packages, and the general scientific stack.
- `conda-forge` also provides `pytorch-cpu` for the CPU fallback environment.
- The GPU environment installs `torch`, `torchvision`, and `lightning` from PyPI.

Why: `conda-forge` can lag behind the newest CUDA-enabled PyTorch builds. This repo uses the official PyTorch `cu132` wheel index so the GPU environment gets the intended CUDA 13.2 build directly. The wheel choice is driven by the target CUDA runtime and NVIDIA driver support, not by the GPU model name itself.

## Current GPU Runtime

On this machine, the resolved GPU environment is:

- `torch.__version__ = 2.12.1+cu132`
- `torch.version.cuda = 13.2`
- `nvidia-smi` also reports CUDA `13.2`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`

If a notebook reports a different Torch or CUDA version from the same Pixi interpreter path, restart the notebook kernel. A long-running kernel can keep older imports in memory after the environment changes.

## Add Packages

Common Conda packages:

```powershell
pixi add numpy pandas scipy scikit-learn xgboost
pixi add geopandas rasterio shapely pyarrow
```

GPU-specific PyPI packages:

```powershell
pixi add --pypi lightning --feature gpu
pixi add --pypi cupy-cuda13x --feature gpu
```

After adding packages sync the envs:

```powershell
# sync default (GPU env)
pixi install
# sync fallback CPU env
pixi install -e cpu
```

## Choosing a CuPy Wheel

Use the CUDA runtime from the project environment, not only the GPU model:

```powershell
nvidia-smi
pixi run python -c "import torch; print(torch.version.cuda)"
```

Then choose the CuPy wheel family that matches that runtime as closely as CuPy supports.

References:

- CuPy install guide: [https://docs.cupy.dev/en/stable/install.html](https://docs.cupy.dev/en/stable/install.html)
- CuPy CUDA 12 wheel on PyPI: [https://pypi.org/project/cupy-cuda13x/](https://pypi.org/project/cupy-cuda12x/)
