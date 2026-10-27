"""Paths and tool locations; override with environment variables on other machines."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SCENES_DATA_ROOT", REPO_ROOT / "data")).resolve()
SCENES_ROOT = DATA_ROOT / "scenes"
DATABASE_URL = os.environ.get("SCENES_DATABASE_URL", f"sqlite:///{DATA_ROOT / 'scenes.db'}")

# Prefer brush-cli built from source (2.4x faster steady-state than the v0.3.0 release on an
# M3: 7.15 vs 2.94 steps/s, see data/bench/results.jsonl); fall back to the release binary.
_BRUSH_CANDIDATES = [
    REPO_ROOT / "tools/brush-src/target/release/brush-cli",
    REPO_ROOT / "tools/brush-dl/brush-app-aarch64-apple-darwin/brush_app",
]
BRUSH_BIN = Path(os.environ.get("BRUSH_BIN", next((p for p in _BRUSH_CANDIDATES if p.exists()), _BRUSH_CANDIDATES[-1])))
BUILD_LOD_BIN = Path(os.environ.get("BUILD_LOD_BIN", REPO_ROOT / "tools/spark/rust/target/release/build-lod"))

# Splat training per quality: (steps, max image side). Every new version is viewable as a
# dense point cloud; splats train on request (UI), or automatically with AUTO_TRAIN
# ("quick" | "full"; default "none").
# On the 3-video room (M3): quick ~20.6 dB holdout PSNR in ~6 min, full 22.1 dB in ~25 min.
TRAIN_SETTINGS = {
    "quick": (int(os.environ.get("TRAIN_QUICK_STEPS", 3000)), int(os.environ.get("TRAIN_QUICK_RES", 1024))),
    "full": (int(os.environ.get("TRAIN_FULL_STEPS", 7000)), int(os.environ.get("TRAIN_FULL_RES", 1600))),
    # Uniform splats: round, one color each (SH degree 0), sizes fixed during training at
    # UNIFORM_PIXELS on screen from the nearest camera; started from UNIFORM_INIT_POINTS of
    # the dense cloud plus a sky shell (worker.pipeline.uniform). The viewer scales them live.
    # Waterfall holdout PSNR vs quick (19.7 dB, 149 s): 18.4 dB, 210 s. One world size for
    # all splats gave 13.8 dB (far scenery as dots, black sky).
    "uniform": (int(os.environ.get("TRAIN_UNIFORM_STEPS", 3000)), int(os.environ.get("TRAIN_UNIFORM_RES", 1024))),
}
UNIFORM_PIXELS = float(os.environ.get("UNIFORM_PIXELS", 2.0))
# GPS scenes train splats in ground blocks (worker.pipeline.splat.blocks) of about
# TRAIN_BLOCK_FRAMES frames (blocks under half of it are merged into neighbours), each with
# the full steps above: one global run spread the same steps over every video, and quality
# dropped as videos were added. 500 is about the first Achapnyak video alone (469 frames);
# Achapnyak v5 (2805 frames) becomes 6 blocks. 0 trains globally.
TRAIN_BLOCK_FRAMES = int(os.environ.get("TRAIN_BLOCK_FRAMES", 500))
UNIFORM_INIT_POINTS = int(os.environ.get("UNIFORM_INIT_POINTS", 500_000))
AUTO_TRAIN = os.environ.get("AUTO_TRAIN", "none")
TRAIN_PREVIEW_EVERY = int(os.environ.get("TRAIN_PREVIEW_EVERY", 1000))

# Dense clouds: depth-map pixels sampled per keyframe (1 = every pixel of the network's
# 518 x 924 output; 2 = every other pixel, ~4x fewer points and a 2x coarser voxel). Stride 1
# gave the 4-part office 61M points, and rebuilding the whole-scene LoD took 3 h (v3) and
# 14 h (v4) on the M3 (whole-scene LoD, before tiles).
DENSE_STRIDE = int(os.environ.get("DENSE_STRIDE", 2))
# Monocular depth for the dense cloud (Hugging Face id). Small: 0.67 s/frame on the M3 (MPS);
# Base (~4x the parameters, sharper and more accurate at edges): 1.26 s/frame.
DEPTH_MODEL = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
# Dense tiles (worker.pipeline.tiles) split above DENSE_TILE_POINTS points. Achapnyak v5
# (121M points): 104 tiles at 4M, 150 at 2M, built in 2.5-4.5 min (one 3.9M-point tile LoD:
# 20 s, 1 GB) instead of one whole-scene LoD that was still running after an hour, swapping.
# Fewer tiles matter in the viewer: each holds at least one 64K-splat page of Spark's 256.
# LOD_JOBS tile LoDs are built at once (build-lod --quick is mostly single-threaded).
DENSE_TILE_POINTS = int(os.environ.get("DENSE_TILE_POINTS", 4_000_000))
LOD_JOBS = int(os.environ.get("LOD_JOBS", os.cpu_count() or 4))

# Work run ahead on queued videos (worker.prefetch): frame extraction for up to
# INGEST_VIDEOS videos at once, decoding in up to DECODE_THREADS segments in total (one
# ffmpeg + analysis thread uses ~half a core; 4 concurrent 4K decodes on the M3 take as long
# as one), and depth prediction one video at a time on the GPU.
INGEST_VIDEOS = int(os.environ.get("INGEST_VIDEOS", 3))
DECODE_THREADS = int(os.environ.get("DECODE_THREADS", 4))
# Videos whose own SfM model is built at once, splitting the cores. On the M3 (4 performance +
# 4 efficiency cores) two at 4 threads each were far slower than one after the other at 8: a
# global mapping attempt went from ~3 min to ~21 min. So more than one only on bigger CPUs.
SFM_VIDEOS = int(os.environ.get("SFM_VIDEOS", 1 if (os.cpu_count() or 8) <= 12 else 2))

# Linking a video's own model into the scene (worker.pipeline.pieces): every Nth frame of the
# new video is matched (LightGlue) against its LINK_NEIGHBORS most similar frames among every
# Mth frame of the existing pieces.
LINK_QUERY_STRIDE = int(os.environ.get("LINK_QUERY_STRIDE", 4))
LINK_CANDIDATE_STRIDE = int(os.environ.get("LINK_CANDIDATE_STRIDE", 2))
LINK_NEIGHBORS = int(os.environ.get("LINK_NEIGHBORS", 5))
# With GPS on both sides, a query frame is only matched to frames taken within this many
# metres (horizontally) of it, and pieces with no frame that close are skipped: link cost
# then depends on what is nearby, not on the scene's size.
LINK_RADIUS_M = float(os.environ.get("LINK_RADIUS_M", 200))

# Minimal QC gate (full QC is milestone M3): reject a video if too few frames register.
MIN_REGISTRATION_RATIO = float(os.environ.get("MIN_REGISTRATION_RATIO", 0.6))


def files_url(path: Path) -> str:
    """Public URL for a file under DATA_ROOT (served by the backend at /files)."""
    return "/files/" + path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()


def scene_setting(scene_settings: dict | None, name: str, default):
    """A setting above (`default`), unless the scene overrides it under the same name
    (Scene.settings), so scenes can be processed differently side by side."""
    value = (scene_settings or {}).get(name)
    return default if value is None else type(default)(value)

# Opt-in GPU voxel hierarchy; spark preserves upstream quick/quality LoD behavior.
# The full CUDA profile opts into GPU depth fitting/fusion and Ceres GPU options.
CUDA_PIPELINE = os.environ.get("SCENES_CUDA_PIPELINE", "0") == "1"
LOD_BACKEND = os.environ.get("LOD_BACKEND", "cuda_voxel" if CUDA_PIPELINE else "spark")
# Trainable sky remains opt-in until a matching viewer implementation is installed.
TRAIN_SKYBOX = os.environ.get("TRAIN_SKYBOX", "0")

# A fixed seed makes paired reconstruction runs easier to compare; -1 preserves upstream.
SFM_RANDOM_SEED = int(os.environ.get("SCENES_RANDOM_SEED", -1))
