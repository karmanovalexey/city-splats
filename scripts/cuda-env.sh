#!/usr/bin/env bash
# Source this file from any directory before starting the app on Linux/CUDA.
SCENES_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$SCENES_REPO/.venv/bin:$SCENES_REPO/tools/node/node_modules/node/bin:/usr/local/cuda/bin:$PATH"
export NODE_PATH="${NODE_PATH:+$NODE_PATH:}/usr/share/nodejs"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export SCENES_MODEL_DIR="$SCENES_REPO/tools/models"
export SCENES_DEVICE=cuda
export SCENES_CUDA_PIPELINE=1
export LOD_BACKEND="${LOD_BACKEND:-cuda_voxel}"
# A local Ceres/cuDSS build is optional; otherwise use the caller's library search path.
if [[ -d "$SCENES_REPO/tools/ceres-install/lib" ]]; then
    export LD_LIBRARY_PATH="$SCENES_REPO/tools/ceres-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export AUTO_TRAIN="${AUTO_TRAIN:-full}"
# One reconstruction at a time avoids GPU contention between the long 4K flights.
export SFM_VIDEOS="${SFM_VIDEOS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Ubuntu glog/libunwind and SciPy otherwise conflict when pybind11 raises StopIteration.
if [[ -f /lib/x86_64-linux-gnu/libgcc_s.so.1 ]]; then
    export LD_PRELOAD="/lib/x86_64-linux-gnu/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
# COLMAP's ONNX provider also needs the cuDNN/cuBLAS shipped with PyTorch.
for scenes_lib in "$SCENES_REPO"/.venv/lib/python*/site-packages/nvidia/*/lib; do
    if [[ -d "$scenes_lib" ]]; then
        export LD_LIBRARY_PATH="$scenes_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
unset scenes_lib
