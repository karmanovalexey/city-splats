"""gsplat CUDA trainer; isolate torch from the worker's COLMAP/OpenMP runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycolmap

from . import Snapshot, TrainingStopped
from backend.app import config as app_config


class CudaTrainer:
    name = "gsplat"

    def train(self, image_root, reconstruction, work_dir, steps, max_resolution,
              init_ply, snapshot_every, on_snapshot, should_stop=lambda: False,
              sh_degree=3, freeze_scales=False, eval_names=None) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        dataset = work_dir / "dataset"
        # gsplat rasterizes pinhole cameras, so rectify lens distortion once.
        pycolmap.undistort_images(
            output_path=dataset, input_path=reconstruction, image_path=image_root,
            undistort_options={"max_image_size": max_resolution},
        )
        rec = pycolmap.Reconstruction(dataset / "sparse")
        images = sorted(rec.images.values(), key=lambda image: image.name)
        points = list(rec.points3D.values())
        np.savez(work_dir / "scene.npz",
                 xyz=np.array([p.xyz for p in points], dtype=np.float32),
                 rgb=np.array([p.color for p in points], dtype=np.float32) / 255,
                 views=np.array([im.cam_from_world().matrix() for im in images]),
                 intrinsics=np.array([rec.cameras[im.camera_id].calibration_matrix() for im in images]))
        sky_setting = getattr(app_config, "TRAIN_SKYBOX", os.environ.get("TRAIN_SKYBOX"))
        # A frozen comparison policy survives progress/manifest updates. Explicit
        # TRAIN_SKYBOX overrides it; extended training uses its own quality entry.
        policy = work_dir.parents[3] / "training-policy.json"
        if sky_setting is None and policy.exists():
            frozen = json.loads(policy.read_text()).get(work_dir.parent.name, {})
            if "train_skybox" in frozen:
                sky_setting = "1" if frozen["train_skybox"] else "0"
        config = dict(eval_names=eval_names, train_skybox=sky_setting != "0", names=[im.name for im in images], steps=steps,
                      snapshot_every=snapshot_every, sh_degree=sh_degree,
                      freeze_scales=freeze_scales,
                      init_ply=str(init_ply.resolve()) if init_ply else None)
        (work_dir / "config.json").write_text(json.dumps(config))
        seen = set()
        with (work_dir / "gsplat.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m",
                                     "worker.pipeline.splat.cuda_job", str(work_dir.resolve())],
                                    stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    if should_stop():
                        raise TrainingStopped()
                    # The job renames exports atomically, so previews are always complete.
                    for path in sorted((work_dir / "exports").glob("splat_*.ply")):
                        step = int(path.stem.split("_")[-1])
                        if step not in seen:
                            on_snapshot(Snapshot(step, steps, path))
                            seen.add(step)
                    try:
                        code = proc.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        if code:
            raise RuntimeError(f"gsplat exited with {code}; see {work_dir / 'gsplat.log'}")
        final = work_dir / "exports" / f"splat_{steps}.ply"
        if not final.is_file():
            raise RuntimeError(f"gsplat did not export {final}")
        if steps not in seen:
            on_snapshot(Snapshot(steps, steps, final))
        return final
