"""Video ingest: probe, decode (with HDR -> SDR), and motion/sharpness-based keyframe selection."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# HLG (ARIB STD-B67) inverse OETF constants.
_HLG_A = 0.17883277
_HLG_B = 1.0 - 4.0 * _HLG_A
_HLG_C = 0.5 - _HLG_A * np.log(4.0 * _HLG_A)

# Linear-light BT.2020 -> BT.709 primaries.
_BT2020_TO_BT709 = np.array(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ],
    dtype=np.float32,
)


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    num_frames: int
    color_transfer: str
    color_space: str
    rotation: int
    tags: dict

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in ("arib-std-b67", "smpte2084")


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    meta = json.loads(out)
    video = next(s for s in meta["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in video["r_frame_rate"].split("/"))
    rotation = 0
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(side_data["rotation"])
    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=num / den,
        duration=float(meta["format"]["duration"]),
        num_frames=int(video.get("nb_frames", 0)),
        color_transfer=video.get("color_transfer", ""),
        color_space=video.get("color_space", ""),
        rotation=rotation,
        tags=meta["format"].get("tags", {}),
    )


def _build_hlg_luts() -> tuple[np.ndarray, np.ndarray, float]:
    # 16-bit HLG code -> scene-linear BT.2020.
    e = np.arange(65536, dtype=np.float64) / 65535.0
    inv_oetf = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - _HLG_C) / _HLG_A) + _HLG_B) / 12.0)
    # Linear BT.709 (after the primaries matrix) -> 8-bit sRGB. HLG reference white (75%
    # signal, ~0.265 linear) maps to ~0.8 and highlights roll off instead of clipping.
    max_linear = 1.5
    linear = np.arange(65536, dtype=np.float64) / 65535.0 * max_linear * 3.0
    linear = linear / (1.0 + linear / 4.0)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    to_srgb = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return inv_oetf.astype(np.float32), to_srgb, max_linear


_HLG_INV_OETF, _LINEAR_TO_SRGB8, _MAX_LINEAR = _build_hlg_luts()


def _hlg_to_srgb(rgb16: np.ndarray) -> np.ndarray:
    """HLG BT.2020 16-bit RGB -> 8-bit sRGB BT.709 (LUT-based)."""
    linear = _HLG_INV_OETF[rgb16] @ _BT2020_TO_BT709.T
    idx = np.clip(linear * (65535.0 / _MAX_LINEAR), 0, 65535).astype(np.uint16)
    return _LINEAR_TO_SRGB8[idx]


def to_bgr8(info: VideoInfo, frame: np.ndarray) -> np.ndarray:
    """A decoded frame (16-bit RGB for HDR) as 8-bit BGR (HDR tone-mapped)."""
    if not info.is_hdr:
        return frame
    return cv2.cvtColor(_hlg_to_srgb(frame), cv2.COLOR_RGB2BGR)


def _select_expr(indices: list[int]) -> str:
    """ffmpeg select expression matching these frame numbers, nested as a balanced sum: the
    expression parser recurses per operator, and a flat 300-term sum overflows it."""
    if len(indices) == 1:
        return f"eq(n\\,{indices[0]})"
    mid = len(indices) // 2
    return f"({_select_expr(indices[:mid])}+{_select_expr(indices[mid:])})"


def _seek(sample_fps: float, start: int, count: int | None) -> tuple[list[str], list[str]]:
    """ffmpeg input/output options decoding `count` samples from sample `start` on. Samples are
    numbered from the seek point, so two decodes with the same `start` see the same frames."""
    before = ["-ss", f"{start / sample_fps:.6f}"] if start else []
    return before, (["-frames:v", str(count)] if count is not None else [])


def iter_frames(info: VideoInfo, sample_fps: float, max_side: int | None = None,
                select: list[int] | None = None, start: int = 0, count: int | None = None):
    """Yield (frame_index, timestamp, bgr uint8 image) sampled at `sample_fps`; with `select`,
    only those sample indices are converted and returned (the rest are skipped in ffmpeg).
    `start`/`count` decode a segment; indices stay relative to the whole video."""
    w, h = info.width, info.height
    if info.rotation in (90, -90, 270, -270):
        w, h = h, w
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        w, h = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)

    filters = [f"fps={sample_fps}"]
    if select is not None:
        filters.append(f"select='{_select_expr([i - start for i in select])}'")
    if info.is_hdr:
        # Decode to 16-bit non-linear RGB with the BT.2020 matrix; transfer handled in numpy.
        filters.append(f"scale={w}:{h}:in_color_matrix=bt2020:in_range=tv:flags=bicubic,format=rgb48le")
        bytes_per_frame, dtype = w * h * 3 * 2, np.uint16
    else:
        filters.append(f"scale={w}:{h}:flags=bicubic,format=bgr24")
        bytes_per_frame, dtype = w * h * 3, np.uint8

    # Hardware decode when available (VideoToolbox on macOS, NVDEC on CUDA boxes); falls back to
    # software. Output is bit-identical, but HEVC decode drops ~3.4x wall / ~12x CPU on an M3.
    before, after = _seek(sample_fps, start, count)
    cmd = [
        "ffmpeg", "-v", "error", "-hwaccel", "auto", *before, "-i", str(info.path),
        "-vf", ",".join(filters), *after, "-vsync", "0", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    indices = iter(select) if select is not None else None
    index = start
    try:
        while True:
            buf = proc.stdout.read(bytes_per_frame)
            if len(buf) < bytes_per_frame:
                break
            frame = np.frombuffer(buf, dtype=dtype).reshape(h, w, 3)
            sample = next(indices) if indices is not None else index
            yield sample, sample / sample_fps, to_bgr8(info, frame)
            index += 1
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed decoding {info.path}")


def sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_ANALYSIS_WIDTH = 960


def iter_gray(info: VideoInfo, sample_fps: float, width: int = _ANALYSIS_WIDTH, start: int = 0,
              count: int | None = None):
    """Yield (sample_index, timestamp, float32 luma) at reduced width, for frame analysis.
    Cheap: ffmpeg scales and extracts luma (for HDR the non-linear HLG signal, fine for
    ranking sharpness and measuring motion within one video)."""
    src_w, src_h = (info.height, info.width) if info.rotation in (90, -90, 270, -270) else (info.width, info.height)
    height = int(round(width * src_h / src_w / 2) * 2)
    before, after = _seek(sample_fps, start, count)
    cmd = [
        "ffmpeg", "-v", "error", "-hwaccel", "auto", *before, "-i", str(info.path),
        "-vf", f"fps={sample_fps},scale={width}:{height}:flags=area,format=gray",
        *after, "-vsync", "0", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    index = start
    try:
        while True:
            buf = proc.stdout.read(width * height)
            if len(buf) < width * height:
                break
            yield index, index / sample_fps, np.frombuffer(buf, dtype=np.uint8).reshape(height, width).astype(np.float32)
            index += 1
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed decoding {info.path}")


_THUMB_WIDTH = 160


def _motion(previous: np.ndarray, current: np.ndarray) -> float:
    """Median optical-flow magnitude between two thumbnails, as a fraction of image width."""
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.median(np.linalg.norm(flow, axis=2))) / _THUMB_WIDTH


@dataclass
class Frame:
    name: str
    timestamp: float
    sharpness: float
    holdout: bool
    motion: float = 0.0  # view change since the previous keyframe, fraction of image width


def _segments(num_samples: int, parts: int, min_samples: int) -> list[tuple[int, int]]:
    """Split sample indices [0, num_samples) into up to `parts` ranges of >= min_samples."""
    parts = max(1, min(parts, num_samples // max(min_samples, 1)))
    bounds = [round(k * num_samples / parts) for k in range(parts + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def _analyse(info: VideoInfo, candidate_fps: float, first: int, last: int | None,
             tick: Callable[[], None]) -> list[tuple[int, float, float, float]]:
    """(sample index, ts, sharpness, motion since the previous sample) for samples [first, last).
    Decoding starts one sample early so the first motion spans the segment boundary."""
    start = max(first - 1, 0)
    out, previous_thumb = [], None
    count = None if last is None else last - start
    for i, ts, gray in iter_gray(info, candidate_fps, start=start, count=count):
        thumb = cv2.resize(gray, (_THUMB_WIDTH, round(_THUMB_WIDTH * gray.shape[0] / gray.shape[1])),
                           interpolation=cv2.INTER_AREA)
        motion = _motion(previous_thumb, thumb) if previous_thumb is not None else 0.0
        previous_thumb = thumb
        if i >= first:
            out.append((i, ts, float(cv2.Laplacian(gray, cv2.CV_32F).var()), motion))
            if i % 10 == 0:
                tick()
    return out


def _run(pool: Executor | None, jobs: list[Callable[[], object]]) -> list:
    if pool is None or len(jobs) == 1:
        return [job() for job in jobs]
    return [f.result() for f in [pool.submit(job) for job in jobs]]


def extract_keyframes(
    video: Path,
    out_dir: Path,
    candidate_fps: float = 15.0,
    target_motion: float = 0.04,
    min_interval: float = 0.1,
    max_interval: float = 1.0,
    max_side: int | None = 1920,
    min_sharpness_ratio: float = 0.35,
    holdout_every: int = 8,
    prefix: str = "",
    on_progress: Callable[[float], None] | None = None,
    pool: Executor | None = None,
    segments: int = 4,
    min_segment_seconds: float = 20.0,
) -> list[Frame]:
    """Motion-spaced keyframes, each the sharpest candidate of its slot.

    Candidates are decoded at candidate_fps. A slot closes once the view has moved
    target_motion (summed per-candidate median optical flow, fraction of image width; hand
    shake adds to it, so 4% gives ~3 keyframes/s on walking videos) since the slot started, or
    after max_interval seconds (standing still), but not before min_interval; its sharpest
    candidate becomes the keyframe. So fast turns get dense keyframes chosen from several
    candidates (motion blur is worst there) and slow stretches sparse ones. Keyframes far
    blurrier than the video's median are dropped where their neighbours still overlap.

    Every `holdout_every`-th kept frame is flagged as holdout (never trained on, used by QC).

    With a `pool`, the video is decoded as up to `segments` time ranges at once (both passes):
    one ffmpeg + analysis thread uses ~half a core and the hardware decoder isn't the limit
    (4 concurrent 4K H.264 decodes on an M3 take as long as one).
    """
    info = probe(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_fps = min(candidate_fps, info.fps)

    # Pass 1: per-candidate sharpness and motion on small grayscale frames, by segment.
    expected = max(int(info.duration * candidate_fps), 1)
    ranges = _segments(expected, segments if pool else 1, int(min_segment_seconds * candidate_fps))
    ranges[-1] = (ranges[-1][0], None)  # the last segment runs to the end of the video
    analysed = [0]
    lock = threading.Lock()

    def tick() -> None:
        with lock:
            analysed[0] += 10
            if on_progress:
                on_progress(min(0.8 * analysed[0] / expected, 0.8))

    samples = [s for part in _run(pool, [lambda a=a, b=b: _analyse(info, candidate_fps, a, b, tick)
                                         for a, b in ranges]) for s in part]

    # Slots over the candidates in order.
    picked: list[tuple[int, float, float, float]] = []  # (sample index, ts, sharpness, motion)
    best: tuple[int, float, float] | None = None  # sharpest candidate of the open slot
    slot_start, slot_motion = 0.0, 0.0
    for i, ts, score, motion in samples:
        slot_motion += motion
        if best is None or score > best[2]:
            best = (i, ts, score)
        elapsed = ts - slot_start
        if elapsed >= min_interval and (slot_motion >= target_motion or elapsed >= max_interval):
            picked.append((*best, slot_motion))
            best, slot_start, slot_motion = None, ts, 0.0
    if best is not None:
        picked.append((*best, slot_motion))
    # Drop very blurry keyframes, but only where the neighbours still overlap enough: blur
    # peaks during fast motion, where a hole hurts registration more than a soft frame.
    median = float(np.median([p[2] for p in picked])) if picked else 0.0
    kept: list[tuple[int, float, float, float]] = []
    carried = 0.0  # motion of dropped slots, added to the next kept keyframe
    for k, (index, ts, score, motion) in enumerate(picked):
        following = picked[k + 1][3] if k + 1 < len(picked) else 0.0
        if score < min_sharpness_ratio * median and carried + motion + following <= 2 * target_motion:
            carried += motion
            continue
        kept.append((index, ts, score, carried + motion))
        carried = 0.0
    picked = kept

    # Pass 2: decode just the chosen frames at full quality, from the same segment starts as
    # pass 1 so sample indices name the same frames.
    names = {p[0]: f"{prefix}{k:05d}.jpg" for k, p in enumerate(picked)}
    written = [0]

    def decode(first: int, last: int | None) -> None:
        select = [p[0] for p in picked if p[0] >= first and (last is None or p[0] < last)]
        if not select:
            return
        start = max(first - 1, 0)
        count = None if last is None else last - start
        for index, _, frame in iter_frames(info, candidate_fps, max_side, select=select, start=start, count=count):
            cv2.imwrite(str(out_dir / names[index]), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with lock:
                written[0] += 1
                if on_progress and written[0] % 10 == 0:
                    on_progress(0.8 + 0.19 * written[0] / len(picked))

    _run(pool, [lambda a=a, b=b: decode(a, b) for a, b in ranges])
    frames = [Frame(names[index], ts, score, holdout=False, motion=round(motion, 4))
              for index, ts, score, motion in picked]
    missing = [f.name for f in frames if not (out_dir / f.name).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} keyframes were not decoded (first: {missing[0]})")
    for k in range(holdout_every - 1, len(frames), holdout_every):
        frames[k].holdout = True
    return frames
