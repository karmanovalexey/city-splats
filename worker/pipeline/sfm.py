"""Initial reconstruction for a scene's first contribution.

Retries: the global mapper's positioning step starts from a random initialisation and
occasionally fails to converge; bundle adjustment then drags the camera into nonsense. An
office walk-through once came out with 214/224 frames "registered", k1=-0.55, fx/fy=1.07 and
125 points; rerunning the same input gave a healthy model (f=950 px, 15.8k points). So a
degenerate result is retried (see the strategies in reconstruct_initial), keeping the best.
The same office video also failed differently: parts of the walk came out 40x smaller than
others (see scale_ratio) in 6 of 7 global runs, even after LightGlue repaired its weak link
(strengthen_weak_links). Incremental mapping seeded with the global run's calibration got
it right: 224/224 frames, 19.1k points, scale ratio 1.6.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from backend.app.config import CUDA_PIPELINE, SFM_RANDOM_SEED

from worker.pipeline.devices.dji import camera_track_defect, similarity
from worker.pipeline.features import FeatureConfig
from worker.reporting import StepReporter, step

log = logging.getLogger(__name__)


@dataclass
class InitialResult:
    reconstruction_path: Path
    database_path: Path
    num_images: int
    num_registered: int
    num_points: int
    mean_reproj_error: float
    timings: dict[str, float]
    camera_model: str
    defect: str | None
    gps: dict | None = None  # agreement with the flight track (devices.dji), when there is one
    chunking: dict | None = None  # how a long video's chunks were joined (_map_chunks)

    @property
    def registration_ratio(self) -> float:
        return self.num_registered / max(self.num_images, 1)


# Fewer SfM points per registered frame than this means the reconstruction is degenerate
# (healthy videos give 100+).
MIN_POINTS_PER_FRAME = 20
# Camera speed along a video changes smoothly; if the median step between consecutive frames
# differs this much between parts of the trajectory, parts were solved at different scales.
# Healthy: room 2.1, office 1.5; broken office runs: 38-59.
MAX_SCALE_RATIO = 8.0


def scale_ratio(rec: pycolmap.Reconstruction, window: int = 15) -> float:
    """Max/min of windowed median camera steps along the video (frames sorted by name)."""
    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    centres = np.array([im.projection_center() for im in images])
    if len(centres) < 2 * window:
        return 1.0
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    medians = [np.median(steps[k:k + window]) for k in range(0, len(steps) - window + 1, window // 3)]
    return float(max(medians) / max(min(medians), 1e-12))


SCALE_DEFECT = "parts of the video were solved at different scales"


def agree(a: pycolmap.Reconstruction, b: pycolmap.Reconstruction, window: int = 30,
          max_error: float = 0.03, max_scale_drift: float = 1.25) -> bool:
    """Whether two independent solutions of the same video share one trajectory (up to a
    similarity): camera centres within max_error of the extent, and every `window`-frame
    stretch at the same local scale.

    A drone that hovers or only tilts its gimbal has near-zero camera steps, so scale_ratio
    flags it (Achapnyak: 61-65x). There the global and incremental mappers matched within 1.2%,
    local scales 0.96-1.03; on a really broken run (waterfall, 11x) they differed by 28% and
    local scales 0.43-5.6."""
    ca = {im.name: im.projection_center() for im in a.images.values() if im.has_pose}
    cb = {im.name: im.projection_center() for im in b.images.values() if im.has_pose}
    names = sorted(set(ca) & set(cb))
    if len(names) < 2 * window:
        return False
    x, y = np.array([ca[n] for n in names]), np.array([cb[n] for n in names])
    # Umeyama similarity x -> y.
    mx, my = x.mean(0), y.mean(0)
    u, s, vt = np.linalg.svd((y - my).T @ (x - mx) / len(x))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = np.trace(np.diag(s) @ d) / (x - mx).var(0).sum()
    aligned = scale * (x - mx) @ rotation.T + my
    error = np.median(np.linalg.norm(aligned - y, axis=1)) / max(np.ptp(y, axis=0).max(), 1e-12)
    drift = [np.linalg.norm(np.diff(y[k:k + window], axis=0), axis=1).sum()
             / max(np.linalg.norm(np.diff(aligned[k:k + window], axis=0), axis=1).sum(), 1e-12)
             for k in range(0, len(names) - window + 1, window)]
    return error <= max_error and max(drift) <= max_scale_drift and min(drift) >= 1 / max_scale_drift


def _track_check(rec: pycolmap.Reconstruction, positions: dict[str, np.ndarray],
                 groups: list[list[str]] | None = None, min_frames: int = 50) -> tuple[str | None, dict]:
    """The reconstruction against the flight track; a merged one (groups: each source model's
    image names) model by model, its larger models deciding."""
    by_name = {im.name: im for im in rec.images.values() if im.has_pose and im.name in positions}
    groups = groups or [list(by_name)]
    problems, stats = [], []
    for names in groups:
        names = sorted(n for n in names if n in by_name)
        if len(names) < 10 or (len(groups) > 1 and len(names) < min_frames):
            continue
        centres = np.array([by_name[n].projection_center() for n in names])
        problem, group_stats = camera_track_defect(centres, np.array([positions[n] for n in names]))
        stats.append(group_stats | {"frames": len(names)})
        if problem:
            problems.append(f"{problem} ({len(names)} frames)" if len(groups) > 1 else problem)
    if not stats:
        return "too few frames with GPS", {}
    return ("; ".join(problems) or None), (stats[0] if len(stats) == 1 else {"models": stats})


def defect(rec: pycolmap.Reconstruction) -> str | None:
    """Why a reconstruction is unusable, or None if it looks healthy."""
    if rec.num_points3D() < MIN_POINTS_PER_FRAME * max(rec.num_reg_images(), 1):
        return f"only {rec.num_points3D()} points for {rec.num_reg_images()} frames"
    camera = next(iter(rec.cameras.values()))
    fx, fy = camera.params[0], camera.params[1]
    if camera.model.name.startswith("OPENCV") and abs(fx / fy - 1) > 0.05:
        return f"camera calibration diverged (fx/fy = {fx / fy:.2f})"
    ratio = scale_ratio(rec)
    if ratio > MAX_SCALE_RATIO:
        return f"{SCALE_DEFECT} (camera step ratio {ratio:.0f}x)"
    return None


def write_position_priors(database: Path, positions: dict[str, np.ndarray], sigma: tuple[float, float, float]) -> int:
    """GPS positions (local metres) as COLMAP pose priors of the database's images."""
    covariance = np.diag(np.square(sigma))
    db = pycolmap.Database.open(str(database))
    try:
        written = 0
        for image in db.read_all_images():
            if image.name in positions:
                db.write_pose_prior(pycolmap.PosePrior(
                    position=np.asarray(positions[image.name], dtype=float), position_covariance=covariance,
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                    corr_data_id=pycolmap.data_t(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, image.camera_id),
                                                 image.image_id),
                ))
                written += 1
        return written
    finally:
        db.close()


def _merge_by_track(reconstructions: dict[int, pycolmap.Reconstruction], sparse_dir: Path,
                    positions: dict[str, np.ndarray], out_dir: Path,
                    min_frames: int = 20) -> tuple[pycolmap.Reconstruction | None, list[list[str]]]:
    """Incremental mapping that lost track leaves several models; each is placed on the GPS
    track by a similarity and they are combined (the result is in track metres). Also returns
    each model's image names: consecutive frames from different models are offset by the GPS
    error, so the result is judged model by model."""
    from worker.pipeline.pieces import Piece, compose

    parts, groups = [], []
    for k, rec in reconstructions.items():
        images = [im for im in rec.images.values() if im.has_pose and im.name in positions]
        if len(images) < min_frames:
            continue
        centres = np.array([im.projection_center() for im in images])
        scale, rotation, translation = similarity(centres, np.array([positions[im.name] for im in images]))
        parts.append(Piece(sparse_dir / str(k), Path(), pycolmap.Sim3d(np.hstack([scale * rotation,
                                                                                  translation[:, None]]))))
        groups.append([im.name for im in images])
    if not parts:
        return None, []
    log.info("incremental mapping with GPS: %d model(s) placed on the flight track (%s frames)", len(parts),
             "+".join(str(len(g)) for g in groups))
    return compose(parts, out_dir), groups


def _map(database: Path, image_root: Path, sparse_dir: Path, names: list[str], incremental: bool = False,
         num_threads: int = -1, positions: dict[str, np.ndarray] | None = None,
         ) -> tuple[pycolmap.Reconstruction | None, list[list[str]] | None]:
    """The mapper's largest model, or with GPS and incremental mapping all its models merged
    (then also each model's image names)."""
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if incremental:
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = num_threads
        options.image_names = names
        options.random_seed = SFM_RANDOM_SEED
        if CUDA_PIPELINE:
            options.ba_use_gpu = True
            options.ba_gpu_index = "0"
        if positions:
            options.use_prior_position = True
            options.use_robust_loss_on_prior_position = True
            # The priors hold the model in place between global adjustments, so fewer suffice
            # (after every 25% growth instead of 10%).
            options.ba_global_frames_ratio = options.ba_global_points_ratio = 1.25
        reconstructions = pycolmap.incremental_mapping(database, image_root, sparse_dir, options=options)
        if positions and reconstructions:
            return _merge_by_track(reconstructions, sparse_dir, positions, sparse_dir / "merged")
    else:
        options = pycolmap.GlobalPipelineOptions()
        options.num_threads = num_threads
        options.image_names = names
        options.random_seed = SFM_RANDOM_SEED
        if CUDA_PIPELINE:
            options.mapper.global_positioning.use_gpu = True
            options.mapper.global_positioning.gpu_index = "0"
            options.mapper.global_positioning.min_num_images_gpu_solver = 0
            options.mapper.bundle_adjustment.ceres.use_gpu = True
            options.mapper.bundle_adjustment.ceres.gpu_index = "0"
            options.mapper.bundle_adjustment.ceres.min_num_images_gpu_solver = 0
            options.mapper.ba_gpu_index = "0"
        reconstructions = pycolmap.global_mapping(database, image_root, sparse_dir, options)
    if not reconstructions:
        return None, None
    return max(reconstructions.values(), key=lambda r: (r.num_reg_images(), r.num_points3D())), None


def _seed_camera(database: Path, camera: pycolmap.Camera) -> None:
    """Use a previous attempt's calibration as the starting point (incremental mapping
    otherwise starts from a 1.2 x width focal guess, far off for wide lenses)."""
    db = pycolmap.Database.open(str(database))
    try:
        for stored in db.read_all_cameras():
            stored.model, stored.params, stored.has_prior_focal_length = camera.model, camera.params, True
            db.update_camera(stored)
    finally:
        db.close()


# Weak links: a cut between consecutive frames crossed by fewer verified pairs than this
# (fast turns: brute-force ALIKED matching drops most pairs there).
MIN_CROSSING_PAIRS = 4
WEAK_LINK_WINDOW = 6


def strengthen_weak_links(database: Path, image_names: list[str], features: FeatureConfig,
                          report: StepReporter | None = None) -> int:
    """Re-match the pairs around weak cuts with the cross-video matcher (LightGlue).

    On the office video, a fast turn left 2 verified pairs across frame 73; LightGlue on
    the surrounding window brought it to 52. Returns the number of pairs re-matched."""
    db = pycolmap.Database.open(str(database))
    try:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        order = [ids[n] for n in sorted(image_names) if n in ids]
        position = {image_id: k for k, image_id in enumerate(order)}
        pair_ids, geometries = db.read_two_view_geometries()
        crossing = np.zeros(len(order) + 1, dtype=int)
        for pair_id, geometry in zip(pair_ids, geometries):
            pair = pycolmap.pair_id_to_image_pair(pair_id)
            # During registration the database also holds other videos' frames.
            if len(geometry.inlier_matches) < 15 or any(i not in position for i in pair):
                continue
            a, b = sorted(position[i] for i in pair)
            crossing[a] += 1  # crosses every cut k with a <= k < b
            crossing[b] -= 1
        crossing = np.cumsum(crossing)[: len(order) - 1]
        # Cuts near the ends are naturally crossed by fewer pairs.
        weak = [k for k in range(WEAK_LINK_WINDOW, len(order) - 1 - WEAK_LINK_WINDOW)
                if crossing[k] < MIN_CROSSING_PAIRS]
        pairs = {
            (order[i], order[j])
            for k in weak
            for i in range(k - WEAK_LINK_WINDOW + 1, k + 1)
            for j in range(k + 1, k + WEAK_LINK_WINDOW + 1)
        }
        for a, b in pairs:  # the matcher skips pairs that already have matches
            db.delete_matches(a, b)
            db.delete_two_view_geometry(a, b)
    finally:
        db.close()
    if not pairs:
        return 0
    log.info("weak links at frames %s: re-matching %d pairs with %s", weak, len(pairs), features.cross_matching.type)
    names = {image_id: name for name, image_id in ids.items()}
    pair_file = database.parent / "weak_link_pairs.txt"
    pair_file.write_text("".join(f"{names[a]} {names[b]}\n" for a, b in pairs))
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pair_file)
    pairing.block_size = 50  # COLMAP logs progress per block (default 1225 pairs)
    with step(report, "match_weak_links", len(pairs)):
        pycolmap.match_image_pairs(database, matching_options=features.cross_matching, pairing_options=pairing,
                                   device=features.device)
    return len(pairs)


def _use_fisheye(database: Path) -> None:
    """Switch the database's cameras to OPENCV_FISHEYE, focal prior for a ~120 degree lens."""
    db = pycolmap.Database.open(str(database))
    try:
        for camera in db.read_all_cameras():
            f = 0.48 * camera.width
            camera.model = pycolmap.CameraModelId.OPENCV_FISHEYE
            camera.params = [f, f, camera.width / 2, camera.height / 2, 0, 0, 0, 0]
            camera.has_prior_focal_length = True
            db.update_camera(camera)
    finally:
        db.close()


# Long videos are mapped in chunks of about this many frames, overlapping by CHUNK_OVERLAP:
# mapping time grows faster than the frame count (bundle adjustment; incremental mapping
# re-adjusts as it grows), so 695 frames took 77 min while 200-frame chunks take minutes each.
CHUNK_FRAMES = 200
CHUNK_OVERLAP = 40


def chunks(names: list[str], size: int = CHUNK_FRAMES, overlap: int = CHUNK_OVERLAP) -> list[list[str]]:
    """Consecutive runs of about `size` frames, neighbours sharing at least `overlap`."""
    names = sorted(names)
    if len(names) <= size * 1.25:
        return [names]
    count = max(2, round((len(names) - overlap) / (size - overlap)))
    size = -(-(len(names) + (count - 1) * overlap) // count)
    starts = np.round(np.linspace(0, len(names) - size, count)).astype(int)
    return [names[s:s + size] for s in starts]


def _sim3(matrix: np.ndarray) -> pycolmap.Sim3d:
    """A similarity from a 3x4 or 4x4 matrix (scale times rotation, translation)."""
    return pycolmap.Sim3d(np.asarray(matrix)[:3])


def _matrix(sim: pycolmap.Sim3d) -> np.ndarray:
    return np.vstack([sim.matrix(), [0, 0, 0, 1]])


def align_overlap(a: pycolmap.Reconstruction, b: pycolmap.Reconstruction, tolerance: float = 0.03,
                  min_inliers: int = 100, min_inlier_ratio: float = 0.5) -> tuple[pycolmap.Sim3d | None, dict]:
    """a_from_b for two chunks of one video mapped from the same database: in a frame both
    registered, keypoint i is the same keypoint, so 3D points seen through it in both are
    3D-3D correspondences. Robust Sim3, inliers within tolerance x median viewing distance."""
    images_a = {im.name: im for im in a.images.values() if im.has_pose}
    shared = [(images_a[im.name], im) for im in b.images.values() if im.has_pose and im.name in images_a]
    seen: set[tuple[int, int]] = set()
    src, tgt, depth = [], [], []
    for image_a, image_b in shared:
        center = image_a.projection_center()
        for p_a, p_b in zip(image_a.points2D, image_b.points2D):
            if p_a.has_point3D() and p_b.has_point3D() and (p_b.point3D_id, p_a.point3D_id) not in seen:
                seen.add((p_b.point3D_id, p_a.point3D_id))
                src.append(b.points3D[p_b.point3D_id].xyz)
                tgt.append(a.points3D[p_a.point3D_id].xyz)
                depth.append(np.linalg.norm(tgt[-1] - center))
    stats = {"shared_frames": len(shared), "correspondences": len(src)}
    if len(src) < min_inliers:
        return None, stats
    src, tgt = np.asarray(src), np.asarray(tgt)
    ransac = pycolmap.RANSACOptions()
    ransac.max_error = threshold = tolerance * float(np.median(depth))
    estimate = pycolmap.estimate_sim3d_robust(src, tgt, ransac)
    if not estimate:
        return None, stats
    sim = estimate["tgt_from_src"]
    residual = np.linalg.norm(src @ sim.matrix()[:, :3].T + sim.matrix()[:, 3] - tgt, axis=1)
    inliers = residual < threshold
    stats |= {"inliers": int(inliers.sum()), "inlier_ratio": round(float(inliers.mean()), 3)}
    ok = inliers.sum() >= min_inliers and inliers.mean() >= min_inlier_ratio
    return (sim if ok else None), stats


def _place_on_track(rec: pycolmap.Reconstruction, positions: dict[str, np.ndarray],
                    names: list[str] | None = None) -> pycolmap.Sim3d | None:
    """track_from_rec by the camera centres of `names` (default: all registered frames)."""
    images = [im for im in rec.images.values() if im.has_pose and im.name in positions
              and (names is None or im.name in names)]
    if len(images) < 10:
        return None
    scale, rotation, translation = similarity(np.array([im.projection_center() for im in images]),
                                              np.array([positions[im.name] for im in images]))
    return _sim3(np.hstack([scale * rotation, translation[:, None]]))


def _solve(database: Path, image_root: Path, names: list[str], sparse_dir: Path, num_threads: int,
           positions: dict[str, np.ndarray] | None, report: StepReporter | None, label: str = "",
           ) -> tuple[pycolmap.Reconstruction | None, str | None, str, dict | None]:
    """Map `names` (all or a chunk of the database's frames), retrying with other strategies
    until a result looks healthy: (reconstruction, problem or None, camera model, GPS stats)."""
    rec, problem, camera_model, gps = None, "no reconstruction", "OPENCV", None
    # Global mapping is fast but its positioning step is random and can solve parts of
    # a walk-forward video at different scales; incremental mapping (seeded with the
    # global attempt's calibration) is slower but steadier; fisheye last.
    strategies = ["global", "incremental", "global", "fisheye"]
    # Results flagged only for uneven camera steps; two that agree are accepted (see agree).
    uneven: list[pycolmap.Reconstruction] = []
    for attempt, strategy in enumerate(strategies):
        if strategy == "incremental":
            camera = next(iter(rec.cameras.values())) if rec is not None else None
            if camera is not None and abs(camera.params[0] / camera.params[1] - 1) <= 0.05:
                _seed_camera(database, camera)
            elif not positions:
                continue  # no usable calibration to start from (GPS priors steady it otherwise)
        if strategy == "fisheye":
            _use_fisheye(database)
        name = f"map_{strategy}" + ("_2" if strategy == "global" and attempt > 0 else "")
        with step(report, name, len(names)):
            candidate, groups = _map(database, image_root, sparse_dir / f"attempt{attempt}", names,
                                     strategy == "incremental", num_threads, positions)
        if candidate is None:
            continue
        candidate_gps = None
        if positions:
            # The flight track decides: camera-step heuristics can't tell hovering from a
            # broken solution.
            candidate_problem, candidate_gps = _track_check(candidate, positions, groups)
        else:
            candidate_problem = defect(candidate)
        if not positions and candidate_problem and candidate_problem.startswith(SCALE_DEFECT):
            if any(agree(other, candidate) for other in uneven):
                log.info("%smapping attempt %d (%s) matches an earlier one: uneven camera steps are real (%s)",
                         label, attempt + 1, strategy, candidate_problem)
                candidate_problem = None
            else:
                uneven.append(candidate)
        log.info("%smapping attempt %d (%s): %d frames, %d points, %s%s", label, attempt + 1, strategy,
                 candidate.num_reg_images(), candidate.num_points3D(), candidate_problem or "ok",
                 f" (GPS: {candidate_gps})" if candidate_gps else "")
        # Prefer healthy results, then more points.
        if rec is None or (problem is not None and (candidate_problem is None
                                                    or candidate.num_points3D() > rec.num_points3D())):
            rec, problem, gps = candidate, candidate_problem, candidate_gps
            camera_model = "OPENCV_FISHEYE" if strategy == "fisheye" else "OPENCV"
        if problem is None:
            break
    return rec, problem, camera_model, gps


def _restore_camera(database: Path, camera: pycolmap.Camera) -> None:
    db = pycolmap.Database.open(str(database))
    try:
        for stored in db.read_all_cameras():
            stored.model, stored.params = camera.model, camera.params
            stored.has_prior_focal_length = camera.has_prior_focal_length
            db.update_camera(stored)
    finally:
        db.close()


def _map_chunks(database: Path, image_root: Path, parts: list[list[str]], sparse_dir: Path, num_threads: int,
                positions: dict[str, np.ndarray] | None, report: StepReporter | None,
                ) -> tuple[pycolmap.Reconstruction | None, str | None, str, dict | None, dict]:
    """Each chunk mapped on its own, then chained through the frames neighbours share
    (align_overlap); a chain that breaks (a chunk failed, or too little overlap) is placed by
    GPS when there is a track, otherwise only the longest chain is kept."""
    db = pycolmap.Database.open(str(database))
    original = db.read_all_cameras()[0]
    db.close()
    solved: list[tuple[pycolmap.Reconstruction | None, str | None, str]] = []
    for k, names in enumerate(parts):
        if hasattr(report, "section"):
            report.section(k, len(parts))
        _restore_camera(database, original)
        good = [(r, m) for r, p, m in solved if r is not None and p is None and m == "OPENCV"]
        if good:  # one camera: start from the calibration a previous chunk found
            _seed_camera(database, next(iter(good[-1][0].cameras.values())))
        rec, problem, model, _ = _solve(database, image_root, names, sparse_dir / f"chunk{k}", num_threads,
                                        positions, report, label=f"chunk {k + 1}/{len(parts)}: ")
        if rec is not None:
            (sparse_dir / f"chunk{k}" / "best").mkdir(parents=True, exist_ok=True)
            rec.write(sparse_dir / f"chunk{k}" / "best")
        solved.append((rec, problem, model))
    if hasattr(report, "section"):
        report.section(None, None)
    return _join_chunks(solved, len(parts), sparse_dir, positions)


def _join_chunks(solved: list[tuple[pycolmap.Reconstruction | None, str | None, str]], count: int,
                 sparse_dir: Path, positions: dict[str, np.ndarray] | None,
                 ) -> tuple[pycolmap.Reconstruction | None, str | None, str, dict | None, dict]:
    """Chunks (each written to sparse_dir/chunk<k>/best) as one model, see _map_chunks."""
    from worker.pipeline.pieces import Piece, compose

    parts = range(count)
    # Chains of chunks, each chunk as chain_from_chunk.
    chains: list[list[tuple[int, np.ndarray]]] = []
    links: list[dict] = []
    for k, (rec, problem, _) in enumerate(solved):
        if rec is None or problem is not None:
            continue
        previous = chains[-1][-1] if chains else None
        if previous is not None and previous[0] == k - 1:
            sim, stats = align_overlap(solved[k - 1][0], rec)
            links.append({"chunks": [k, k + 1], **stats})
            if sim is not None:
                chains[-1].append((k, previous[1] @ _matrix(sim)))
                continue
        chains.append([(k, np.eye(4))])
    if not chains:
        first = next(((r, p, m) for r, p, m in solved if r is not None), (None, "no reconstruction", "OPENCV"))
        return first[0], first[1], first[2], None, {"links": links}
    log.info("%d chunks in %d chain(s): %s", len(parts), len(chains),
             ", ".join("+".join(str(k + 1) for k, _ in chain) for chain in chains))

    placed: list[tuple[int, np.ndarray]] = []  # (chunk, out_from_chunk)
    groups: list[list[str]] = []
    if positions:
        for chain in chains:
            composed = compose([Piece(sparse_dir / f"chunk{k}" / "best", Path(), _sim3(m)) for k, m in chain],
                               sparse_dir / "chain")
            track = _place_on_track(composed, positions)
            if track is None:
                continue
            composed.transform(track)
            problem, _ = _track_check(composed, positions)
            if problem:  # chained chunks drifted apart: each on the track by itself
                log.info("chain %s off the GPS track (%s): placing its chunks by GPS",
                         "+".join(str(k + 1) for k, _ in chain), problem)
                for k, _ in chain:
                    own = _place_on_track(solved[k][0], positions)
                    if own is not None:
                        placed.append((k, _matrix(own)))
                        groups.append(sorted(im.name for im in solved[k][0].images.values() if im.has_pose))
            else:
                placed += [(k, _matrix(track) @ m) for k, m in chain]
                groups.append(sorted(im.name for im in composed.images.values() if im.has_pose))
    else:
        longest = max(chains, key=lambda chain: sum(solved[k][0].num_reg_images() for k, _ in chain))
        placed = list(longest)
    if not placed:
        return solved[chains[0][0][0]][0], "chunks could not be placed on the GPS track", "OPENCV", None, {}
    rec = compose([Piece(sparse_dir / f"chunk{k}" / "best", Path(), _sim3(m)) for k, m in placed],
                  sparse_dir / "chunks")
    problem, gps = (_track_check(rec, positions, groups) if positions else (None, None))
    models = {solved[k][2] for k, _ in placed}
    return rec, problem, models.pop() if len(models) == 1 else "mixed", gps, {
        "chunks": len(parts), "chains": [[k + 1 for k, _ in chain] for chain in chains], "links": links}


def reconstruct_initial(
    image_root: Path,
    image_names: list[str],
    out_dir: Path,
    features: FeatureConfig,
    sequential_overlap: int = 10,
    report: StepReporter | None = None,
    num_threads: int = -1,
    positions: dict[str, np.ndarray] | None = None,
    position_sigma: tuple[float, float, float] = (3.0, 3.0, 0.3),
    chunk_frames: int = CHUNK_FRAMES,
) -> InitialResult:
    """num_threads caps every COLMAP step (videos reconstructed side by side split the cores).
    positions: GPS camera positions per image (devices.dji); then each attempt is judged by
    how well it follows the flight track, and incremental mapping uses them as priors.
    Features and matches are computed once; a long video is mapped in chunks (see chunks)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for options in (features.extraction, features.sequential_matching, features.cross_matching):
        options.num_threads = num_threads
    database = out_dir / "database.db"
    database.unlink(missing_ok=True)
    timings: dict[str, float] = {}

    t = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    with step(report, "extract", len(image_names)):
        pycolmap.extract_features(
            database, image_root, image_names=image_names, camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader, extraction_options=features.extraction, device=features.device,
        )
    timings["extract"] = time.time() - t
    if positions:
        log.info("%d GPS position priors", write_position_priors(database, positions, position_sigma))

    t = time.time()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = sequential_overlap
    pairing.quadratic_overlap = True
    with step(report, "match_sequential", len(image_names)):
        pycolmap.match_sequential(
            database, matching_options=features.sequential_matching, pairing_options=pairing,
            device=features.device,
        )
    timings["match_sequential"] = time.time() - t

    t = time.time()
    weak_pairs = strengthen_weak_links(database, image_names, features, report)
    if weak_pairs:
        timings["match_weak_links"] = time.time() - t

    t = time.time()
    sparse_dir = out_dir / "sparse"
    parts = chunks(image_names, chunk_frames)
    chunking = None
    if len(parts) == 1:
        rec, problem, camera_model, gps = _solve(database, image_root, parts[0], sparse_dir, num_threads,
                                                 positions, report)
    else:
        log.info("mapping %d frames in %d chunks of ~%d", len(image_names), len(parts), chunk_frames)
        rec, problem, camera_model, gps, chunking = _map_chunks(database, image_root, parts, sparse_dir,
                                                                num_threads, positions, report)
    timings["map"] = time.time() - t
    if rec is None:
        raise RuntimeError("SfM produced no reconstruction (not enough overlap/texture?)")

    rec_path = sparse_dir / "0"
    rec_path.mkdir(parents=True, exist_ok=True)
    rec.write(rec_path)
    errors = [p.error for p in rec.points3D.values()]
    return InitialResult(
        reconstruction_path=rec_path,
        database_path=database,
        num_images=len(image_names),
        num_registered=rec.num_reg_images(),
        num_points=rec.num_points3D(),
        mean_reproj_error=float(np.mean(errors)) if errors else float("nan"),
        timings=timings,
        camera_model=camera_model,
        defect=problem,
        gps=gps,
        chunking=chunking,
    )
