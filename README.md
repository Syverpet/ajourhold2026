# ml-ajourhold

Short setup and maintenance guide for this project.

## 1) Quick setup

### Prerequisites

- Install Pixi.
- NVIDIA GPU users: install a driver compatible with CUDA 13.2.

### Create environment

```powershell
pixi lock
pixi install -e gpu
pixi install -e cpu
```

### Validate GPU environment

```powershell
pixi run -e gpu python -c "import sys, torch; print(sys.version.split()[0]); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### Recommended execution style

Always run scripts through Pixi so environment variables and paths are correct:

```powershell
pixi run -e gpu python -u scripts/tessera_no_s2_teacher_student_train.py
```

Avoid calling the environment python executable directly.

## 2) Adding new packages

Use this decision rule:

- Put geospatial/native stack in Conda dependencies when possible.
- Put PyTorch GPU wheels and tightly coupled wheel-only packages in PyPI dependencies.

### Add Conda package

1. Add it in tool.pixi.dependencies (or feature-specific Conda section if needed).
2. Re-lock and install:

```powershell
pixi lock
pixi install -e gpu
pixi install -e cpu
```

### Add PyPI package

1. Add it in tool.pixi.pypi-dependencies (or feature-specific pypi-dependencies).
2. Re-lock and install:

```powershell
pixi lock
pixi install -e gpu
pixi install -e cpu
```

### Verify after dependency changes

```powershell
pixi run -e gpu python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
pixi run -e gpu python -c "from torchvision import ops; print(hasattr(ops, 'nms'))"
```

If changing geospatial packages, also run a small rasterio and GDAL smoke test in the same Pixi environment.

## 3) Special rules for this pyproject.toml

- Mixed dependency model is intentional.
  - Conda-forge is the base solver and binary stack.
  - Some GPU packages come from PyPI on purpose.

- GPU and CPU environments are separate features.
  - cpu environment uses pytorch-cpu from Conda.
  - gpu environment uses PyPI torch and torchvision pinned to CUDA 13.2 wheels.

- GPU PyTorch wheels are pinned and index-specific.
  - Keep torch and torchvision versions compatible.
  - Keep the CUDA wheel index for both when updating.

- GPU environment prefers OpenBLAS on Conda side.
  - This is used to reduce OpenMP runtime conflicts.
  - Do not move this pin to global dependencies.

- GPU feature has a setuptools upper bound.
  - Keep the constraint unless you validate a newer version with the full workflow.

- Workspace channel settings are strict.
  - channels = conda-forge
  - channel-priority = strict

- Supported platforms are win-64 and linux-64.
  - Test lock/install on both when changing core dependencies.

## 4) Common commands

Install kernels:

```powershell
pixi run -e gpu python -m ipykernel install --user --name demo-conda-gpu --display-name "ml_gpu"
pixi run -e cpu python -m ipykernel install --user --name demo-conda-cpu --display-name "ml_cpu"
```

Run environment validator task:

```powershell
pixi run -e gpu validate-env
```

Clean Pixi environments in this workspace:

```powershell
# Remove one environment folder only (example: default)
pixi clean -e default

# Remove all local workspace environments under .pixi/envs
pixi clean

# Recreate only what you need afterward
pixi install -e gpu
pixi install -e cpu
```

## 5) Troubleshooting notes

- If GDAL_DATA warnings appear, run through Pixi instead of direct python.
- If a long-running stage appears stalled, tail the run log and check process activity:

```powershell
Get-Content F:\TESSERA_NO_S2\tessera_no_s2_teacher_student_128\logs\tessera_no_run.log -Tail 40 -Wait
Get-Process python | Select-Object Id,CPU,StartTime,Responding
```
