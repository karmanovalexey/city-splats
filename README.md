# city-splats

Upload videos of a place (drone, phone, action camera) and they are merged, one by one, into a
single 3D scene: a dense point cloud by default, Gaussian splats on request, streamed to the
browser with levels of detail. Each video becomes a new version of the scene, so a bad one can
be reverted.

![Achapnyak, Yerevan: four DJI Mini 4 flights merged and trained as splats](screenshot.png)

## What it does

- **Per-video reconstruction.** Frames are extracted and filtered (blur, exposure), keyframes
  picked, then COLMAP (pycolmap) builds the video's own model with ALIKED + LightGlue features.
  Long videos are mapped in overlapping chunks of ~200 frames that are chained back together.
- **GPS priors.** DJI subtitle telemetry (lat/lon, height above takeoff, distance from home)
  becomes camera position priors, keeping scale consistent along a flight and placing videos
  relative to each other.
- **Fitting into the scene.** A new video's model is matched against the nearest frames of
  the existing scene (retrieval, limited to frames within 200 m when both sides have GPS) and
  placed by a robust similarity transform from 3D-3D correspondences. Existing parts never move.
- **Dense cloud, in tiles.** Depth Anything V2 depth maps, scaled to the SfM points, are fused
  and streamed to disk by region, then merged into an octree of tiles on a global voxel grid.
  A new video re-merges and rebuilds only the tiles it touches (in parallel, memory bounded by
  a tile), and the viewer streams each tile as its own LoD (Spark `.rad`), all under one splat
  budget.
- **Splats.** [Brush](https://github.com/ArthurBrussee/brush) trains on request (quick / full /
  uniform), with live previews while training; output is cleaned and built into a LoD as well.
  GPS scenes train in ground blocks of ~500 frames, each with the full steps, so quality per
  area doesn't drop as videos are added; each block is cropped to its area and streamed as
  its own LoD.
- **Viewer.** three.js + [Spark](https://github.com/sparkjsdev/spark): dense cloud, splats,
  sparse points by video, camera positions, and, for GPS scenes, an OpenStreetMap or
  satellite map laid under the scene on the fitted ground plane.
- **Versions.** Every accepted video publishes a version; HEAD can be moved back to any one.
  Progress (with ETAs learned from past runs) streams to the UI over SSE.

## Layout

```
backend/    FastAPI: scenes, uploads, versions, training requests, SSE, static files
worker/     job runner; worker/pipeline/ holds the stages (ingest, sfm, pieces, dense, splat, ...)
frontend/   Vite + React + TypeScript viewer and control panel
infra/      COLMAP build script and patches for COLMAP and Brush
scripts/    dev.sh (run everything) and maintenance scripts
spikes/     benchmarks and experiments
data/       scenes, database, logs (not in git)
tools/      third-party sources and builds (not in git)
```

## Setup (macOS, Apple Silicon)

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 20+, Rust, ffmpeg,
Homebrew `cmake ninja libomp` and COLMAP's build dependencies.

```bash
uv venv --python 3.12 && uv sync

# COLMAP + pycolmap with ONNX (ALIKED/LightGlue); applies infra/patches/colmap-*.patch
infra/build_colmap.sh

# Brush (splat training), patched for fixed/frozen scales and a chosen eval split (needs Rust 1.95)
git clone https://github.com/ArthurBrussee/brush tools/brush-src
git -C tools/brush-src checkout 1388f74c6fe0236f68ee4915564bf00e9d2e3747
git -C tools/brush-src apply ../../infra/patches/brush-local.patch
(cd tools/brush-src && cargo +1.95 build --release -p brush-cli)

# Spark's build-lod (LoD files for the viewer)
git clone https://github.com/sparkjsdev/spark tools/spark
git -C tools/spark checkout ffe210dc0a45d0515a06c9235878108112827d61
(cd tools/spark/rust && cargo build --release -p build-lod)

(cd frontend && npm install)
```

Tool paths can be overridden with `BRUSH_BIN` and `BUILD_LOD_BIN`; other settings (training
steps, link radius, parallelism, dense stride) are environment variables documented in
`backend/app/config.py`. On Linux with CUDA (e.g. DGX Spark) the same build script enables
CUDA in COLMAP.

## Running

```bash
scripts/dev.sh
```

starts the backend (http://localhost:8000), the worker and the frontend
(http://localhost:5173); logs go to `data/logs/`. Create a scene, drop videos onto it, and
watch them process. Viewer keys: WASD/drag to move (Q/E down/up, Shift faster), `V` cloud ⇄
splats, `1`–`4` view modes, `C` cameras, `K` splat kind, `M` map (street → satellite → off),
`[` `]` splat size.

## Status

A working prototype, developed on an M3 MacBook (CPU SfM, Metal for depth and Brush).
Known gaps: rigid placement leaves ~5–15 px of misalignment between videos; scenes without
GPS still train splats in one run; QC gates are minimal.


### Optional CUDA voxel LoD

Install `uv sync --extra cuda-lod` on a CUDA 12 system, then set
`LOD_BACKEND=cuda_voxel`. The default `spark` backend keeps upstream's `--quick`
for dense tiles and `--quality` for trained splats. The CUDA backend retains the
original leaf geometry and spherical harmonics and builds coarser nodes by voxel
grouping. Its parent hierarchy differs from both Spark builders; compare coarse
views before adopting it. It writes native chunked RAD files and `cuda-lod.json`.
Tile LoD jobs serialize on the GPU to bound peak VRAM; CPU tile preparation stays
parallel. There is no silent fallback when CUDA is explicitly selected.

The measured DJI_0327 dense-stage result (1,899.3 s to 141.4 s) compared our old
whole-cloud, default-quality baseline with the CUDA pipeline. Current upstream
already uses tiled clouds and explicit `--quick`; that 13.4x figure is **not** a
speedup measured against this main revision. Both old runs decoded the video and
recomputed depth/SfM, with 1,013 frames and about 49 million output points.


### CUDA pipeline profile

This branch includes `voxel-lod`, CUDA depth alignment and neighbour-consistency
fusion, per-tile CUDA voxel reduction/PLY packing, Ceres GPU solver requests, and
the gsplat trainer. Upstream's streaming spill, incremental tiles, configurable
depth model, block planning and CPU/Brush paths are retained. Camera/track
metadata, image decoding, spill/grouping and file I/O still execute on the CPU.

On Linux with CUDA 12.8 and an NVIDIA driver:

1. Install CUDA-enabled pycolmap 4.2.1. For actual GPU sparse solves, build it
   against CUDA-enabled Ceres + cuDSS; a CUDA feature extractor alone is insufficient.
   The tested stack used Ceres `07a952622c2118433c783b1c3bd3fabe68420fc5`
   (2.3.0 development) and cuDSS 1.0.0.35. The old Ceres 2.0 binding patch remains
   under `infra/patches/` for the CPU-solver compatibility build only.
2. `uv sync --extra cuda` (gsplat/fused-ssim builds need the matching installed Torch).
   Pycolmap's native wheel must come from the compatible local COLMAP build; do
   not replace it with a CPU-only wheel when syncing the environment.
3. Download ALIKED models with `PYTHONPATH=. .venv/bin/python scripts/download_models.py`.
4. `source scripts/cuda-env.sh`, then start the app normally. The profile selects
   `SCENES_DEVICE=cuda`, `SCENES_CUDA_PIPELINE=1`, and `LOD_BACKEND=cuda_voxel`.
   Override `LOD_BACKEND=spark` to retain upstream's LoD builders.

The CUDA trainer accepts upstream's explicit block evaluation-frame lists and
excludes those cameras from optimization. It records holdout PSNR in `metrics.json`,
which the existing evaluation reporting reads. `TRAIN_SKYBOX=0` is the default:
the optional learned directional background requires a matching viewer extension.
Do not compare different training schedules/block plans as GPU-only speedups.

Use `SCENES_CUDA_PIPELINE=0`, `LOD_BACKEND=spark`, and the original device selection
for a CPU/Brush baseline. The older 13.4x measurement above is not a claim about
this newer tiled baseline. No production scene is launched by this branch.


Validation against upstream `af72843` (2026-10-28): on 16 existing DJI frames,
CPU and CUDA streaming fusion both emitted 1,897,448 points; sampled p99 nearest
position error was 1.35e-6 scene units, and voxel selection on the same input was
exact. Eight tiles passed Spark's native RAD decoder. Two concurrent LoD requests
also decoded as valid finite trees (42,629 original leaves each). CUDA trainer
smoke checks passed with explicit block holdouts (6 train / 2 evaluation cameras),
both adaptive and fixed-scale modes. These are compatibility checks, not a new
end-to-end performance measurement or a full-scene quality assessment.
