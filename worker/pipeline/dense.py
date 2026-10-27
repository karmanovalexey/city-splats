"""Dense point clouds from monocular depth, anchored to the SfM model.

COLMAP's PatchMatch stereo needs CUDA, so on the Mac (and as the default everywhere for now):
1. a depth network (worker.pipeline.depth_net, run as a subprocess) predicts relative inverse
   depth per keyframe;
2. per frame, an affine map (inverse depth = a * prediction + b) is fitted robustly to the
   depths of the SfM points observed in that frame, giving metric-consistent depth;
3. pixels are unprojected through the real camera model and kept only if neighbouring
   keyframes of the same video agree on the depth (rejects flying pixels at depth edges and
   frames where the network is wrong);
4. the points are streamed into tiles (worker.pipeline.tiles), voxel-downsampled per tile.

Incremental scenes: each contribution densifies only its own frames. Registration keeps
existing frames fixed, so earlier contributions' dense points stay valid and a version's
cloud is its parent's plus the new contribution's, changed only in the tiles it touches.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

from worker.pipeline.cleanup import SH_C0, write_ply


@dataclass
class DenseStats:
    frames: int
    frames_used: int
    points: int
    voxel: float
    fit_error_median: float  # median relative depth error of the per-frame fits (SfM points)

    def as_dict(self) -> dict:
        return {"frames": self.frames, "frames_used": self.frames_used, "points": self.points,
                "voxel": round(self.voxel, 5), "fit_error_median": round(self.fit_error_median, 4)}


def predict_depth(image_root: Path, names: list[str], out_dir: Path, model: str,
                  on_progress: Callable[[int, int], None] | None = None) -> None:
    """Run the depth network over the frames in a subprocess (see depth_net)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names_file = out_dir / "names.txt"
    names_file.write_text("\n".join(names))
    proc = subprocess.Popen(
        [sys.executable, "-m", "worker.pipeline.depth_net", str(image_root), str(names_file), str(out_dir), model],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        tail = (tail + [line])[-20:]
        if line.startswith("depth ") and on_progress:
            done, total = line.split()[1].split("/")
            on_progress(int(done), int(total))
    if proc.wait() != 0:
        raise RuntimeError("depth network failed:\n" + "".join(tail))


def _sample(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample of a 2D grid at (u, v) in grid pixels; NaN outside."""
    h, w = grid.shape
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    inside = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    out = np.full(u.shape, np.nan, dtype=np.float32)
    out[inside] = grid[iv[inside], iu[inside]]
    return out


@dataclass
class _Frame:
    image: pycolmap.Image
    camera: pycolmap.Camera
    depth: np.ndarray  # metric-consistent depth at network resolution (NaN = invalid)
    scale: tuple[float, float]  # image pixels per depth pixel (x, y)
    fit_error: float


def _align(image: pycolmap.Image, camera: pycolmap.Camera, rec: pycolmap.Reconstruction,
           disparity: np.ndarray, min_points: int = 50) -> _Frame | None:
    h, w = disparity.shape
    sx, sy = camera.width / w, camera.height / h
    xy, xyz = [], []
    for p2d in image.points2D:
        if p2d.has_point3D():
            xy.append(p2d.xy)
            xyz.append(rec.points3D[p2d.point3D_id].xyz)
    if len(xy) < min_points:
        return None
    xy, xyz = np.asarray(xy), np.asarray(xyz)
    cam_from_world = image.cam_from_world().matrix()
    z = (xyz @ cam_from_world[:, :3].T + cam_from_world[:, 3])[:, 2]
    d = _sample(disparity, xy[:, 0] / sx, xy[:, 1] / sy)
    ok = np.isfinite(d) & (z > 0)
    d, inv_z = d[ok], 1.0 / z[ok]
    if len(d) < min_points:
        return None
    # Robust affine fit in inverse depth: iteratively drop residuals beyond 3 MADs.
    keep = np.ones(len(d), dtype=bool)
    for _ in range(4):
        A = np.stack([d[keep], np.ones(keep.sum())], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, inv_z[keep], rcond=None)
        residual = np.abs(a * d + b - inv_z)
        mad = np.median(residual[keep]) + 1e-12
        keep = residual < 3 * 1.4826 * mad
    if a <= 0:
        return None
    pred_inv = a * disparity.astype(np.float32) + b
    depth = np.where(pred_inv > 1e-6, 1.0 / np.maximum(pred_inv, 1e-6), np.nan).astype(np.float32)
    fit_z = 1.0 / np.maximum(a * d[keep] + b, 1e-6)
    fit_error = float(np.median(np.abs(fit_z - z[ok][keep]) / z[ok][keep]))
    return _Frame(image, camera, depth, (sx, sy), fit_error)


def align_frames(rec: pycolmap.Reconstruction, names: list[str], depth_dir: Path, stride: int = 2,
                 max_fit_error: float = 0.2) -> tuple[dict[int, _Frame], float]:
    """Frames of `names` (one video, in capture order) with metric depth, keyed by position,
    and the median world size of one sampled pixel (the voxel size of a scene's first video)."""
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    frames: dict[int, _Frame] = {}
    for k, name in enumerate(names):
        image = by_name.get(name)
        path = depth_dir / f"{name}.npy"
        if image is None or not path.exists():
            continue
        frame = _align(image, rec.cameras[image.camera_id], rec, np.load(path))
        if frame is not None and frame.fit_error <= max_fit_error:
            frames[k] = frame
    if not frames:
        raise RuntimeError("no frame's depth could be fitted to the SfM points")
    footprint = float(np.median([np.nanmedian(f.depth) * stride * f.scale[0] / f.camera.mean_focal_length()
                                 for f in frames.values()]))
    return frames, footprint


def fuse_frames(frames: dict[int, _Frame], image_root: Path, sink: Callable[[np.ndarray, np.ndarray], None],
                stride: int = 2, neighbours: tuple[int, ...] = (-4, -2, -1, 1, 2, 4), min_agree: int = 2,
                tolerance: float = 0.06, workers: int = 6,
                on_progress: Callable[[int, int], None] | None = None) -> int:
    """Unproject each frame's depth and pass the points its neighbours agree on to `sink`
    (xyz float32, rgb uint8), in frame order; returns how many. Nothing accumulates here, so
    memory doesn't grow with the length of the video.

    Tolerances match the network's accuracy: per-frame fits to SfM depths are off by 4-7%
    (median) on the office, so a 3% agreement test with neighbours 3-6 frames away threw away
    most correct pixels. Office part 2 (78 frames): 0.8M points with (±3, ±6, 3%), 4.0M with
    (±1, ±2, ±4, 6%), visibly filling walls, ceiling and floor without adding noise."""

    def fuse(k: int, frame: _Frame) -> tuple[np.ndarray, np.ndarray] | None:
        depth = frame.depth
        h, w = depth.shape
        sx, sy = frame.scale
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        u, v = u.ravel(), v.ravel()
        z = depth[v, u]
        valid = np.isfinite(z)
        u, v, z = u[valid], v[valid], z[valid]
        rays = frame.camera.cam_from_img(np.stack([(u + 0.5) * sx, (v + 0.5) * sy], axis=1))
        cam_pts = np.concatenate([rays * z[:, None], z[:, None]], axis=1)
        world_from_cam = frame.image.cam_from_world().inverse().matrix()
        world = cam_pts @ world_from_cam[:, :3].T + world_from_cam[:, 3]

        agree = np.zeros(len(world), dtype=int)
        for offset in neighbours:
            other = frames.get(k + offset)
            if other is None:
                continue
            m = other.image.cam_from_world().matrix()
            in_other = world @ m[:, :3].T + m[:, 3]
            front = in_other[:, 2] > 1e-6
            pix = np.full((len(world), 2), np.nan)
            pix[front] = other.camera.img_from_cam(in_other[front])
            measured = _sample(other.depth, pix[:, 0] / other.scale[0], pix[:, 1] / other.scale[1])
            agree += (np.abs(in_other[:, 2] - measured) < tolerance * measured).astype(int)
        keep = agree >= min_agree
        if not keep.any():
            return None
        rgb = np.asarray(Image.open(image_root / frame.image.name).convert("RGB").resize((w, h)))
        return world[keep].astype(np.float32), rgb[v[keep], u[keep]]

    # Frames fuse independently and numpy releases the GIL in the heavy parts; results are
    # passed on in frame order so the voxel downsampling picks the same points.
    count = 0
    with ThreadPoolExecutor(workers) as pool:
        for done, fused in enumerate(pool.map(fuse, frames.keys(), frames.values()), start=1):
            if fused is not None:
                sink(*fused)
                count += len(fused[0])
            if on_progress:
                on_progress(done, len(frames))
    if not count:
        raise RuntimeError("densification produced no consistent points")
    return count


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The first point in each voxel. Voxels are packed into one int64 key: np.unique over
    rows (axis=0) sorts them as byte strings, 12 s for 13M points vs ~1 s."""
    cells = np.floor(xyz / voxel).astype(np.int64)
    cells -= cells.min(axis=0)
    size = cells.max(axis=0) + 1
    if float(size[0]) * float(size[1]) * float(size[2]) < 2**62:
        keys = (cells[:, 0] * size[1] + cells[:, 1]) * size[2] + cells[:, 2]
        _, first = np.unique(keys, return_index=True)
    else:
        _, first = np.unique(cells, axis=0, return_index=True)
    return xyz[first], rgb[first]


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz.T
    data["red"], data["green"], data["blue"] = rgb.T
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def read_points_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    count = int(data[:end].split(b"element vertex ")[1].split()[0])
    rows = np.frombuffer(data, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1")], count=count, offset=end)
    return np.stack([rows["x"], rows["y"], rows["z"]], 1), np.stack([rows["red"], rows["green"], rows["blue"]], 1)


def write_cloud_splats(path: Path, xyz: np.ndarray, rgb: np.ndarray, voxel: float, opacity: float = 0.98) -> None:
    """A cloud as small opaque round splats (one per point, sized to the voxel), so it streams
    through the same LoD pipeline and viewer as trained splats."""
    n = len(xyz)
    rows = np.concatenate([
        xyz.astype(np.float32),
        np.full((n, 3), np.log(0.6 * voxel), dtype=np.float32),
        np.full((n, 1), np.log(opacity / (1 - opacity)), dtype=np.float32),
        np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (n, 1)),
        (rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0,
    ], axis=1)
    props = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    write_ply(path, header, rows)


# Keep CPU references for paired numerical validation. Tiling/spilling remains upstream.
from backend.app.config import CUDA_PIPELINE
cpu_align_frames, cpu_fuse_frames = align_frames, fuse_frames
cpu_voxel_downsample, cpu_write_cloud_splats = voxel_downsample, write_cloud_splats
if CUDA_PIPELINE:
    from .cuda_dense import align_frames, fuse_frames, voxel_downsample, write_cloud_splats
