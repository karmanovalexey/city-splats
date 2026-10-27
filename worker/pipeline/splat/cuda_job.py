"""Single-GPU 3DGS optimization; cameras and points arrive as arrays from COLMAP."""

from __future__ import annotations

import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fused_ssim import fused_ssim
from gsplat import export_splats, rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image
from scipy.spatial import cKDTree


def ssim(a, b):
    return fused_ssim(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None], padding="valid")


def sky_basis(d):
    x, y, z = d.unbind(-1)
    return torch.stack([torch.ones_like(x), x, y, z, x*y, y*z,
                        3*z*z-1, x*z, x*x-y*y], dim=-1)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but torch cannot access a GPU")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    root = Path(sys.argv[1])
    cfg = json.loads((root / "config.json").read_text())
    scene = np.load(root / "scene.npz")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda")
    xyz, rgb = scene["xyz"], scene["rgb"]
    if len(xyz) < 4:
        raise ValueError("at least four SfM points are needed for splat initialization")
    views = np.tile(np.eye(4), (len(scene["views"]), 1, 1))
    views[:, :3] = scene["views"]
    centres = np.linalg.inv(views)[:, :3, 3]
    scene_scale = max(float(np.linalg.norm(centres - centres.mean(0), axis=1).max()), 1e-3)
    scales = np.log(np.maximum(cKDTree(xyz).query(xyz, k=4)[0][:, 1:].mean(1), 1e-6))
    count, degree = len(xyz), cfg["sh_degree"]
    values = dict(means=tensor(xyz), scales=tensor(np.repeat(scales[:, None], 3, axis=1)),
                  quats=torch.randn(count, 4, device="cuda"),
                  opacities=torch.full((count,), math.log(0.1 / 0.9), device="cuda"),
                  sh0=tensor((rgb[:, None] - 0.5) / 0.28209479177387814),
                  shN=torch.zeros(count, (degree + 1)**2 - 1, 3, device="cuda"))
    if cfg["init_ply"]:
        from plyfile import PlyData
        vertices = PlyData.read(cfg["init_ply"])["vertex"].data
        field = lambda names: tensor(np.column_stack([vertices[n] for n in names]))
        count = len(vertices)
        values.update(means=field(["x", "y", "z"]),
                      scales=field([f"scale_{i}" for i in range(3)]),
                      quats=field([f"rot_{i}" for i in range(4)]),
                      opacities=tensor(vertices["opacity"].copy()),
                      sh0=field([f"f_dc_{i}" for i in range(3)])[:, None],
                      shN=torch.zeros(count, (degree + 1)**2 - 1, 3, device="cuda"))
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in values.items()})
    rates = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3,
                 opacities=5e-2, sh0=2.5e-3, shN=1.25e-4)
    # Uniform mode keeps both the number of splats and their sizes fixed.
    frozen = {"scales", "quats"} if cfg["freeze_scales"] else set()
    for key in frozen:
        params[key].requires_grad_(False)
    optimizers = {k: torch.optim.Adam([v], lr=rates[k], eps=1e-15)
                  for k, v in params.items() if k not in frozen}
    # A low-frequency world-space environment learns the background through the
    # rasterizer's transmittance, without introducing distant floating geometry.
    sky = torch.nn.Parameter(torch.zeros(9, 3, device="cuda"))
    sky_optimizer = torch.optim.Adam([sky], lr=0.01)
    use_sky = cfg.get("train_skybox", True)
    steps = cfg["steps"]
    strategy = DefaultStrategy(refine_start_iter=min(500, steps // 6),
                               refine_stop_iter=int(steps * 0.7),
                               reset_every=max(steps + 1, 3000))
    state = strategy.initialize_state(scene_scale=scene_scale)
    if not frozen:
        strategy.check_sanity(params, optimizers)
    view_t, intrinsics = tensor(views), tensor(scene["intrinsics"])
    names = cfg["names"]

    @lru_cache(maxsize=32)
    def load(index):
        with Image.open(root / "dataset/images" / names[index]) as image:
            return torch.from_numpy(np.array(image.convert("RGB"))).float() / 255

    def render(index, target):
        height, width = target.shape[:2]
        rendered, alpha, info = rasterization(means=params["means"], quats=params["quats"],
                             scales=params["scales"].exp(), opacities=params["opacities"].sigmoid(),
                             colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                             viewmats=view_t[index:index + 1], Ks=intrinsics[index:index + 1],
                             width=width, height=height, sh_degree=min(degree, iteration // 1000),
                             packed=False, near_plane=0.01, far_plane=1e10)
        if use_sky:
            yy, xx = torch.meshgrid(torch.arange(height, device="cuda", dtype=torch.float32)+0.5,
                                    torch.arange(width, device="cuda", dtype=torch.float32)+0.5,
                                    indexing="ij")
            k = intrinsics[index]
            rays = torch.stack([(xx-k[0,2])/k[0,0], (yy-k[1,2])/k[1,1], torch.ones_like(xx)], -1)
            rays = torch.nn.functional.normalize(rays @ view_t[index,:3,:3], dim=-1)
            background = (sky_basis(rays) @ sky).sigmoid()
            rendered = rendered + (1-alpha) * background[None]
        return rendered, alpha, info

    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    held_out = set(cfg["eval_names"]) if cfg.get("eval_names") is not None else set(names[::8])
    evaluation = [i for i,name in enumerate(names) if name in held_out]
    training = [i for i,name in enumerate(names) if name not in held_out]
    if not training:
        raise ValueError("no training cameras remain after excluding evaluation frames")
    generator = np.random.default_rng(42)
    started = time.monotonic()
    print(f"CUDA {torch.cuda.get_device_name()}; {len(names)} cameras; {count} initial splats", flush=True)
    order = []
    for iteration in range(steps):
        if not order:
            order = list(generator.permutation(training))
        index = order.pop()
        target = load(index).to("cuda")
        rendered, _, info = render(index, target)
        if not frozen:
            strategy.step_pre_backward(params, optimizers, state, iteration, info)
        loss = 0.8 * (rendered[0] - target).abs().mean() + 0.2 * (1 - ssim(rendered[0], target))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {iteration + 1}")
        loss.backward()
        if use_sky:
            sky_optimizer.step()
            sky_optimizer.zero_grad(set_to_none=True)
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not frozen:
            strategy.step_post_backward(params, optimizers, state, iteration, info)
        optimizers["means"].param_groups[0]["lr"] = rates["means"] * 0.01**((iteration + 1) / steps)
        step = iteration + 1
        if step % 100 == 0 or step == 1:
            print(f"step {step}/{steps} loss={loss.item():.5f} splats={len(params['means'])} seconds={time.monotonic()-started:.1f}", flush=True)
        if step % cfg["snapshot_every"] == 0 or step == steps:
            temporary = exports / "pending.ply"
            with torch.no_grad():
                export_splats(means=params["means"], scales=params["scales"],
                              quats=params["quats"], opacities=params["opacities"],
                              sh0=params["sh0"], shN=params["shN"], format="ply", save_to=str(temporary))
            temporary.replace(exports / f"splat_{step}.ply")
            if use_sky:
                sky_path = root.parent / "skybox.json"
                pending = sky_path.with_suffix(".pending")
                pending.write_text(json.dumps(dict(basis="polynomial-degree-2", frame="COLMAP",
                                                   coefficients=sky.detach().cpu().tolist(), step=step)))
                pending.replace(sky_path)
    scores = []
    with torch.no_grad():
        for index in evaluation:
            target = load(index).to("cuda")
            rendered, _, _ = render(index, target)
            scores.append(float(-10 * torch.log10((rendered[0] - target).square().mean())))
    metrics = dict(trainer="gsplat", device=torch.cuda.get_device_name(), steps=steps,
                   trainable_skybox=use_sky, skybox_parameters=27 if use_sky else 0,
                   splats=len(params["means"]), seconds=time.monotonic() - started,
                   training_frames=len(training), holdout_psnr=float(np.mean(scores)) if scores else None, holdout_frames=len(scores))
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
