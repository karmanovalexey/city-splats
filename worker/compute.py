"""Compute backend detection: picks CUDA on GPU boxes (DGX Spark), Metal/CPU on the Mac."""

from __future__ import annotations

import logging
import os
import platform
from functools import cache

import pycolmap

from backend.app.config import BRUSH_BIN
from worker.pipeline.splat import Trainer
from worker.pipeline.splat.brush import BrushTrainer

log = logging.getLogger(__name__)


@cache
def device() -> str:
    forced = os.environ.get("SCENES_DEVICE", "auto")
    if forced != "auto":
        return forced
    if pycolmap.has_cuda:
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def trainer() -> Trainer:
    if device() == "cuda":
        from worker.pipeline.splat.cuda import CudaTrainer
        return CudaTrainer()
    return BrushTrainer(BRUSH_BIN)
