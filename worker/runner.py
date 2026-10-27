"""Jobs: process one contribution (ingest -> SfM/register -> layers -> dense cloud ->
publish) and train one version's splat (train -> remove floaters -> LoD)."""

from __future__ import annotations

import json
from collections.abc import Callable
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap
from sqlmodel import func, select

from backend.app.config import (
    AUTO_TRAIN,
    CUDA_PIPELINE,
    LOD_BACKEND,
    BUILD_LOD_BIN,
    DATA_ROOT,
    DEPTH_MODEL,
    DENSE_STRIDE,
    DENSE_TILE_POINTS,
    LINK_CANDIDATE_STRIDE,
    LINK_NEIGHBORS,
    LINK_QUERY_STRIDE,
    LINK_RADIUS_M,
    LOD_JOBS,
    MIN_REGISTRATION_RATIO,
    SCENES_ROOT,
    TRAIN_BLOCK_FRAMES,
    TRAIN_PREVIEW_EVERY,
    TRAIN_SETTINGS,
    UNIFORM_INIT_POINTS,
    UNIFORM_PIXELS,
    files_url,
    scene_setting,
)
from backend.app.db import session_scope
from backend.app.models import Contribution, Scene, Version, utcnow
from worker import compute
from worker.pipeline.cleanup import clean_sparse, clean_splats, points_as_splats
from worker.pipeline.dense import (
    DenseStats, align_frames, fuse_frames, predict_depth,
)
from worker.pipeline.export import export_layers
from worker.pipeline.georef import georeference
from worker.pipeline.features import DEFAULT_FEATURES, feature_config
from worker.pipeline.ingest import probe
from worker.pipeline.lod import build_lod
from worker.pipeline.pieces import Piece, apply, compose, link_piece
from worker.pipeline.splat import Snapshot, TrainingStopped
from worker.pipeline.splat.blocks import Block, block_model, crop_to_block, plan_blocks
from worker.pipeline.splat.brush import eval_results
from worker.pipeline.tiles import Spill, lod_files, sample_cloud, update_tiles
from worker.pipeline.uniform import uniform_init
from worker.prefetch import Prefetcher
from worker.progress import Progress, TimingModel, contribution_sink, rel, training_sink

log = logging.getLogger(__name__)

# Typical keyframes per second of video from ingest's motion-based spacing; only used to
# pre-estimate SfM time before the frames exist.
EXPECTED_KEYFRAME_RATE = 3.0


def _update(contribution_id: int, **fields) -> None:
    with session_scope() as session:
        c = session.get(Contribution, contribution_id)
        for key, value in fields.items():
            setattr(c, key, value)
        c.updated_at = utcnow()
        session.add(c)
        session.commit()


def publish(
    contribution_id: int,
    scene_id: int,
    head: Version | None,
    database: Path,
    reconstruction: Path,
    work: Path,
    dense_rad: Path,
    metrics: dict,
) -> int:
    """Create a Version with its dense cloud (no splat yet), move HEAD, mark the
    contribution done and queue automatic splat training if AUTO_TRAIN is set."""
    with session_scope() as session:
        scene = session.get(Scene, scene_id)
        number = (
            session.exec(select(func.max(Version.number)).where(Version.scene_id == scene_id)).one() or 0
        ) + 1
        version = Version(
            scene_id=scene_id,
            number=number,
            parent_id=head.id if head else None,
            contribution_id=contribution_id,
            contribution_ids=(head.contribution_ids if head else []) + [contribution_id],
            database_path=rel(database),
            reconstruction_path=rel(reconstruction),
            splat_path="",
            splat_url="",
            dense_url=files_url(dense_rad),
            layers_url=files_url(work / "layers"),
            metrics=metrics,
        )
        if AUTO_TRAIN in TRAIN_SETTINGS:
            version.train_status, version.train_quality, version.train_auto = "queued", AUTO_TRAIN, True
        session.add(version)
        session.flush()
        scene.head_version_id = version.id
        scene.updated_at = utcnow()
        contribution = session.get(Contribution, contribution_id)
        contribution.status = "done"
        contribution.version_id = version.id
        contribution.progress, contribution.stage_progress, contribution.eta_seconds = 1.0, 1.0, 0.0
        contribution.stage = "published"
        contribution.metrics = metrics
        contribution.preview_url = contribution.preview_step = None
        contribution.finished_at = contribution.updated_at = utcnow()
        session.add_all([scene, contribution])
        session.commit()
        return number


def process(contribution_id: int, prefetch: Prefetcher) -> None:
    with session_scope() as session:
        c = session.get(Contribution, contribution_id)
        scene = session.get(Scene, c.scene_id)
        head = session.get(Version, scene.head_version_id) if scene.head_version_id else None
        seq_by_dir = {
            f"c{other.id}": other.seq
            for other in session.exec(select(Contribution).where(Contribution.scene_id == scene.id)).all()
        }

    scene_dir = SCENES_ROOT / scene.slug
    image_root = scene_dir / "images"
    work = scene_dir / "contributions" / c.image_dir_name
    # Frames and depth maps may already be there (worker.prefetch); anything else is left from
    # an interrupted run.
    for stale in (work / name for name in ("sfm", "link", "scene", "layers", "dense", "preview")):
        shutil.rmtree(stale, ignore_errors=True)

    device = compute.device()
    timing = TimingModel(device)
    video = Path(c.video_path)
    info = probe(video)
    est_frames = info.duration * EXPECTED_KEYFRAME_RATE
    stages = [("ingest", timing.estimate("ingest", info.duration)), ("sfm", timing.estimate("sfm", est_frames))]
    if head is not None:
        stages.append(("register", timing.estimate("register", est_frames)))
    stages += [("layers", timing.estimate("layers", est_frames)), ("dense", timing.estimate("dense", est_frames))]
    progress = Progress(contribution_sink(c.id), stages, timing)
    metrics: dict = {"device": device, "video_seconds": round(info.duration, 1)}
    try:
        # 1. Ingest (usually run ahead by the prefetcher)
        frames, seconds = _await(prefetch.ingest(c, scene.slug, retry=True), progress, "ingest")
        timing.record("ingest", info.duration, seconds)
        names = [f"{c.image_dir_name}/{f['name']}" for f in frames]
        metrics |= {"keyframes": len(names), "holdouts": sum(f["holdout"] for f in frames),
                    "ingest_seconds": seconds}
        for stage in ("sfm", "register", "layers"):
            if any(name == stage for name, _ in stages):
                progress.set_estimate(stage, timing.estimate(stage, len(names)))
        _update(c.id, metrics=metrics)

        # 2. The video's own model, its piece of the scene (usually built ahead). A scene keeps
        # the feature type it was built with; scenes predating the "features" metric used SIFT.
        features = feature_config(head.metrics.get("features", "sift") if head else DEFAULT_FEATURES)
        metrics["features"] = features.name
        sfm, seconds = _await(prefetch.piece(c, scene.slug, names, retry=True), progress, "sfm")
        timing.record("sfm", len(names), seconds)
        ratio = sfm.registration_ratio
        metrics |= {
            "registered": sfm.num_registered, "camera_model": sfm.camera_model,
            "registration_ratio": round(ratio, 3), "reproj_error_px": round(sfm.mean_reproj_error, 3),
            "sfm_seconds": seconds, "sfm_timings": {k: round(v, 1) for k, v in sfm.timings.items()},
        }
        if sfm.gps:
            metrics["gps"] = sfm.gps
        if sfm.chunking:
            metrics["chunking"] = sfm.chunking
        rejection = None
        if sfm.defect:
            rejection = f"reconstruction failed after retries: {sfm.defect}"
        elif ratio < MIN_REGISTRATION_RATIO:
            rejection = (f"only {ratio:.0%} of frames reconstructed (< {MIN_REGISTRATION_RATIO:.0%}); "
                         "the video is likely too blurry or textureless")

        # 3. Place the piece in the scene: matched against a subset of the existing pieces'
        # frames, aligned by a similarity transform; nothing else moves (worker.pipeline.pieces).
        gps = sfm.database_path.parent / "gps.json"
        piece = Piece(sfm.reconstruction_path, sfm.database_path, contribution_id=c.id,
                      gps=gps if gps.exists() else None)
        pieces = scene_pieces(head) if head is not None else []
        if head is not None and rejection is None:
            pairs = -(-sfm.num_registered // LINK_QUERY_STRIDE) * LINK_NEIGHBORS
            progress.start("register", substeps=[("match_retrieval", pairs)], rate_suffix=features.name)
            with prefetch.paused():
                link = link_piece(piece, pieces, work / "link", features, report=progress,
                                  query_stride=LINK_QUERY_STRIDE, candidate_stride=LINK_CANDIDATE_STRIDE,
                                  neighbors=LINK_NEIGHBORS, radius=LINK_RADIUS_M)
            timing.record("register", len(names), progress.finish())
            metrics |= {"strategy": "pieces", "gate": link.reason, "link": link.stats,
                        "sfm_timings": metrics["sfm_timings"] | {k: round(v, 1) for k, v in link.timings.items()}}
            if link.ok:
                piece.scene_from_piece = link.scene_from_piece
            else:
                rejection = f"could not place the video in the scene: {link.reason}"
        if rejection:
            _update(c.id, status="rejected", metrics=metrics, error=rejection, eta_seconds=0, finished_at=utcnow())
            return
        pieces.append(piece)
        _update(c.id, metrics=metrics)

        # 4. The pieces as one model (training and viewer layers); viewer layers (sparse points
        # by source video + camera centers)
        progress.start("layers")
        rec = compose(pieces, work / "scene")
        metrics |= {"points": rec.num_points3D(), "points_before": head.metrics.get("points") if head else None,
                    "pieces": [p.to_json(DATA_ROOT) for p in pieces]}
        export_layers(rec, work / "layers", lambda name: seq_by_dir.get(name.split("/")[0], 1) - 1)
        if geo := georeference(rec, [p.gps for p in pieces if p.gps]):
            metrics["geo"] = geo
        timing.record("layers", len(names), progress.finish())

        # 5. Dense point cloud (the scene's main view): this video's frames densified and
        # merged into the parent version's cloud. If densification fails, the SfM points
        # (as round splats) stand in so the version is still viewable.
        progress.start("dense")
        try:
            depth_job, relay = prefetch.depth(c, scene.slug, names, retry=True)
            if not depth_job.done():
                relay.attach(lambda f, detail: progress.update(0.6 * f, detail=detail))
            depth_dir = depth_job.result()
            with prefetch.paused():
                dense_rad = _densify(head, pycolmap.Reconstruction(piece.reconstruction), image_root, names,
                                     work, metrics, progress, depth_dir, scene_from_piece=piece.scene_from_piece,
                                     settings=scene.settings)
            shutil.rmtree(depth_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001 - the version is still useful without a dense cloud
            log.exception("densification failed for contribution %s; using the sparse points", c.id)
            clean_sparse(rec)
            preview_dir = work / "preview"
            preview_dir.mkdir(exist_ok=True)
            points_as_splats(rec, preview_dir / "splat.ply")
            dense_rad = build_lod(BUILD_LOD_BIN, preview_dir / "splat.ply", preview_dir / "lod", quality=False)
            metrics["dense_failed"] = True
            # The next video still merges into the cloud the scene had before this one.
            if head and "dense_tiles" in head.metrics:
                metrics |= {k: head.metrics[k] for k in ("dense_tiles", "dense_voxel")}
        timing.record("dense", len(names), progress.finish())

        # 6. Publish a new version and move HEAD
        number = publish(c.id, scene.id, head, piece.database, work / "scene", work, dense_rad, metrics)
        log.info("contribution %s published as version %s", c.id, number)
    finally:
        progress.close()


def scene_pieces(version: Version) -> list[Piece]:
    """The version's pieces; a version from before pieces is one piece (its whole model)."""
    if "pieces" in version.metrics:
        return [Piece.from_json(p, DATA_ROOT) for p in version.metrics["pieces"]]
    database = DATA_ROOT / version.database_path
    gps = database.parent / "gps.json"
    return [Piece(DATA_ROOT / version.reconstruction_path, database, gps=gps if gps.exists() else None)]


def _await(job: tuple, progress: Progress, stage: str):
    """Result of a prefetch job: a stage done ahead, or one whose progress we report while
    waiting for it."""
    future, relay = job
    if future.done() and future.exception() is None:
        result, seconds = future.result()
        progress.done_ahead(stage, seconds)
        return result, seconds
    progress.start(stage)
    relay.attach(lambda f, detail: progress.update(f, detail=detail))
    result, seconds = future.result()
    progress.finish()
    return result, seconds


def _densify(head: Version | None, rec: pycolmap.Reconstruction, image_root: Path, names: list[str],
             work: Path, metrics: dict, progress: Progress, depth_dir: Path | None = None,
             scene_from_piece: pycolmap.Sim3d | None = None, settings: dict | None = None) -> Path:
    """Dense cloud for this contribution, merged into the parent's tiles; returns the tile
    list the viewer loads. The scene keeps the first video's voxel size. Depth maps are
    predicted unless `depth_dir` already holds them. `rec` is the video's piece, placed by
    scene_from_piece. `settings`: the scene's overrides (scene_setting)."""
    stride = scene_setting(settings, "DENSE_STRIDE", DENSE_STRIDE)
    dense_dir = work / "dense"
    dense_dir.mkdir(exist_ok=True)
    if depth_dir is None:
        depth_dir = dense_dir / "depth"
        predict_depth(image_root, names, depth_dir, scene_setting(settings, "DEPTH_MODEL", DEPTH_MODEL),
                      on_progress=lambda done, total: progress.update(
                          0.6 * done / total, detail=f"Predicting depth {done}/{total}"))
    substage_started = time.perf_counter()
    frames, footprint = align_frames(rec, names, depth_dir, stride=stride)
    metrics["dense_alignment_seconds"] = time.perf_counter() - substage_started
    parent = head.metrics.get("dense_tiles") if head else None
    voxel = parent["voxel"] if parent else footprint * (scene_from_piece.scale if scene_from_piece else 1.0)
    spill = Spill(dense_dir / "spill", voxel)

    def sink(xyz: np.ndarray, rgb: np.ndarray) -> None:
        spill.add(apply(scene_from_piece, xyz).astype(np.float32) if scene_from_piece else xyz, rgb)

    substage_started = time.perf_counter()
    fused = fuse_frames(frames, image_root, sink, stride=stride,
                        on_progress=lambda done, total: progress.update(
                            0.6 + 0.2 * done / total, detail=f"Fusing depth maps {done}/{total}"))
    metrics["dense_fusion_seconds"] = time.perf_counter() - substage_started
    shutil.rmtree(dense_dir / "depth", ignore_errors=True)
    substage_started = time.perf_counter()
    manifest = update_tiles(parent, spill, dense_dir / "tiles", DATA_ROOT, BUILD_LOD_BIN,
                            scene_setting(settings, "DENSE_TILE_POINTS", DENSE_TILE_POINTS),
                            scene_setting(settings, "LOD_JOBS", LOD_JOBS),
                            on_progress=lambda done, total: progress.update(
                                0.8 + 0.2 * done / total, detail=f"Building tile LoDs {done}/{total}"))
    metrics["dense_tiles_seconds"] = time.perf_counter() - substage_started
    metrics.update(cuda_pipeline=CUDA_PIPELINE, dense_lod_backend=LOD_BACKEND)
    shutil.rmtree(dense_dir / "spill", ignore_errors=True)
    before = sum(t["points"] for t in parent["tiles"].values()) if parent else 0
    total = sum(t["points"] for t in manifest["tiles"].values())
    stats = DenseStats(len(names), len(frames), total - before, voxel,
                       float(np.median([f.fit_error for f in frames.values()])))
    metrics |= {
        "dense": stats.as_dict() | {"scene_points": total, "fused_points": fused,
                                    "tiles": len(manifest["tiles"]), "tiles_built": _tiles_built(parent, manifest)},
        "dense_tiles": manifest,
        "dense_voxel": voxel,
    }
    return write_tileset(manifest, dense_dir / "tiles.json")


def _tiles_built(parent: dict | None, manifest: dict) -> int:
    old = {t["dir"] for t in parent["tiles"].values()} if parent else set()
    return sum(t["dir"] not in old for t in manifest["tiles"].values())


def write_tileset(manifest: dict, path: Path) -> Path:
    """The viewer's list of the dense tiles' LoDs."""
    return _write_lod_list([rad for rad, _ in lod_files(manifest, DATA_ROOT)], path)


def _write_lod_list(rads: list[Path], path: Path) -> Path:
    """A list of LoDs the viewer streams together, each as its own paged mesh."""
    path.write_text(json.dumps({"tiles": [{"url": files_url(rad)} for rad in rads]}))
    return path


def _training_should_stop(version_id: int) -> bool:
    """Cancelled from the UI, or (automatic jobs only) a new video is waiting: it will
    produce a newer HEAD, which supersedes this training anyway."""
    with session_scope() as session:
        v = session.get(Version, version_id)
        if v is None or v.train_status == "cancelling":
            return True
        if not v.train_auto:
            return False
        return session.exec(
            select(Contribution).where(Contribution.scene_id == v.scene_id, Contribution.status == "queued")
        ).first() is not None


def _update_version(version_id: int, **fields) -> None:
    with session_scope() as session:
        v = session.get(Version, version_id)
        for key, value in fields.items():
            setattr(v, key, value)
        v.updated_at = utcnow()
        session.add(v)
        session.commit()


def _train_global(v: Version, quality: str, steps: int, max_resolution: int, reconstruction: Path,
                  rec: pycolmap.Reconstruction, image_root: Path, work: Path, progress: Progress,
                  timing: TimingModel, show_preview: Callable[[list[Path], int], None]) -> tuple[Path, dict]:
    """One run over the whole scene; returns the LoD and details for the metrics."""
    # Uniform splats can't grow to cover gaps, so they start from the dense cloud (plus
    # a sky shell), each with a fixed on-screen size.
    init_ply, uniform = None, None
    if quality == "uniform":
        if not v.metrics.get("dense_tiles"):
            raise RuntimeError("uniform splats need the version's dense point cloud")
        init_ply = work / "init.ply"
        cloud = sample_cloud(v.metrics["dense_tiles"], DATA_ROOT, UNIFORM_INIT_POINTS)
        uniform = uniform_init(cloud, rec, image_root, init_ply, max_resolution,
                               pixels=UNIFORM_PIXELS, num_points=UNIFORM_INIT_POINTS)

    progress.start("train")
    snapshots: list[Path] = []

    def on_snapshot(s: Snapshot) -> None:
        progress.update(s.step / s.total_steps, detail=f"Step {s.step}/{s.total_steps}")
        if s.step >= s.total_steps:
            return
        preview_dir = work / f"preview_{s.step}"
        show_preview([build_lod(BUILD_LOD_BIN, s.ply_path, preview_dir, quality=False)], s.step)
        snapshots.append(preview_dir)
        while len(snapshots) > 2:
            shutil.rmtree(snapshots.pop(0), ignore_errors=True)

    raw_ply = compute.trainer().train(
        image_root, reconstruction, work / "train", steps, max_resolution, init_ply, TRAIN_PREVIEW_EVERY,
        on_snapshot, should_stop=lambda: _training_should_stop(v.id),
        sh_degree=0 if uniform else 3, freeze_scales=uniform is not None,
    )
    timing.record(f"train.{quality}", steps, progress.finish())

    progress.start("cleanup")
    splat_ply = work / "splat.ply"
    stats = clean_splats(raw_ply, rec, splat_ply)
    shutil.rmtree(work / "train" / "exports", ignore_errors=True)
    timing.record("cleanup", 1, progress.finish())

    progress.start("lod")
    rad = build_lod(BUILD_LOD_BIN, splat_ply, work / "lod", quality=True)
    timing.record("lod", 1, progress.finish())
    return rad, {**({"uniform": uniform} if uniform else {}), "cleanup": stats.as_dict(),
                 "eval": eval_results(work / "train")}


def _train_blocks(v: Version, quality: str, steps: int, max_resolution: int, reconstruction: Path,
                  rec: pycolmap.Reconstruction, image_root: Path, work: Path, progress: Progress,
                  timing: TimingModel, show_preview: Callable[[list[Path], int], None]) -> tuple[Path, dict]:
    """Each ground block trained on its own frames (worker.pipeline.splat.blocks), cropped to
    its cells, cleaned and built into its own LoD; returns the list of block LoDs. A block
    whose frames and steps match the same block of an earlier version is reused."""
    geo = v.metrics["geo"]
    blocks = plan_blocks(rec, geo, TRAIN_BLOCK_FRAMES)
    plan = [(block, steps) for block in blocks]
    total = sum(n for _, n in plan)
    earlier = _earlier_blocks(v, quality)
    # Held out: every 8th frame by name, as one global Brush run would hold out, so block and
    # global splats are measured on the same unseen frames.
    names = sorted(image.name for image in rec.images.values() if image.has_pose)
    held_out = set(names[::8])
    full = pycolmap.Reconstruction(reconstruction)
    progress.start("train")
    done, rads, results = 0, [], {}
    for i, (block, n) in enumerate(plan, start=1):
        old = earlier.get(block.key)
        if old and old["images_hash"] == block.images_hash and old["steps"] == n:
            rads.append(DATA_ROOT / old["dir"] / "lod" / "splat-lod.rad")
            results[block.key] = old | {"reused": True}
            done += n
            continue
        block_dir = work / block.key
        (block_dir / "model").mkdir(parents=True)
        block_model(full, geo, block).write(block_dir / "model")
        snapshots: list[Path] = []

        def on_snapshot(s: Snapshot, block: Block = block, block_dir: Path = block_dir, i: int = i) -> None:
            progress.update((done + s.step) / total, detail=f"Block {i}/{len(plan)} · step {s.step}/{s.total_steps}")
            if s.step >= s.total_steps:
                return
            preview_dir = block_dir / f"preview_{s.step}"
            preview_dir.mkdir()
            crop_to_block(s.ply_path, preview_dir / "splat.ply", geo, block, blocks)
            show_preview(rads + [build_lod(BUILD_LOD_BIN, preview_dir / "splat.ply", preview_dir / "lod",
                                           quality=False)], done + s.step)
            snapshots.append(preview_dir)
            while len(snapshots) > 2:
                shutil.rmtree(snapshots.pop(0), ignore_errors=True)

        raw = compute.trainer().train(
            image_root, block_dir / "model", block_dir / "train", n, max_resolution, None, TRAIN_PREVIEW_EVERY,
            on_snapshot, should_stop=lambda: _training_should_stop(v.id),
            eval_names=sorted(held_out & set(block.images)),
        )
        kept = crop_to_block(raw, block_dir / "cropped.ply", geo, block, blocks)
        shutil.rmtree(block_dir / "train" / "exports", ignore_errors=True)
        stats = clean_splats(block_dir / "cropped.ply", block_model(rec, geo, block), block_dir / "splat.ply")
        (block_dir / "cropped.ply").unlink()
        rads.append(build_lod(BUILD_LOD_BIN, block_dir / "splat.ply", block_dir / "lod", quality=True))
        done += n
        evals = eval_results(block_dir / "train")
        results[block.key] = {
            "dir": rel(block_dir), "cells": block.cells, "grow": block.grow, "frames": len(block.core),
            "train_frames": len(block.images), "eval_frames": len(held_out & set(block.images)), "steps": n,
            "images_hash": block.images_hash, "eval": evals[-1] if evals else None, "kept": kept,
            "cleanup": stats.as_dict(),
        }
        log.info("version %s block %s (%d/%d): %s", v.id, block.key, i, len(plan), results[block.key]["eval"])
    timing.record(f"train.{quality}", total, progress.finish())
    return _write_lod_list(rads, work / "tiles.json"), {
        "block_frames": TRAIN_BLOCK_FRAMES, "steps_total": total, "blocks": results}


def _earlier_blocks(v: Version, quality: str) -> dict:
    """Blocks of the nearest earlier version trained in blocks at this quality."""
    with session_scope() as session:
        parent = session.get(Version, v.parent_id) if v.parent_id else None
        while parent is not None:
            blocks = parent.metrics.get(f"splat_{quality}", {}).get("blocks")
            if blocks:
                return blocks
            parent = session.get(Version, parent.parent_id) if parent.parent_id else None
    return {}


def train_version(version_id: int) -> None:
    """Train the version's splat at its requested quality, remove floaters, build the LoD
    and swap it in. GPS scenes train in ground blocks (_train_blocks), others in one run."""
    with session_scope() as session:
        v = session.get(Version, version_id)
        scene = session.get(Scene, v.scene_id)
    quality = v.train_quality or "quick"
    steps, max_resolution = TRAIN_SETTINGS[quality]
    max_resolution = scene_setting(scene.settings, f"TRAIN_{quality.upper()}_RES", max_resolution)
    # The frames live with the scene that made the reconstruction (a showcase scene such as
    # office-tsdf borrows another scene's reconstruction and has no images of its own).
    image_root = SCENES_ROOT / Path(v.reconstruction_path).parts[1] / "images"
    blocks = quality != "uniform" and TRAIN_BLOCK_FRAMES > 0 and bool(v.metrics.get("geo"))
    work = SCENES_ROOT / scene.slug / "versions" / f"v{v.number}" / (f"{quality}_blocks" if blocks else quality)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    timing = TimingModel(compute.device())
    stages = [("train", timing.estimate(f"train.{quality}", steps))]
    if not blocks:
        stages += [("cleanup", timing.estimate("cleanup", 1)), ("lod", timing.estimate("lod", 1))]
    progress = Progress(training_sink(v.id), stages, timing)
    previews: list[Path] = []
    started = time.time()
    try:
        # Train from all SfM points (initialising from the cleaned ones scored 0.4 dB lower
        # on the room), but carve floaters against the cleaned ones: a single bad point
        # would carve a tunnel through real geometry.
        reconstruction = DATA_ROOT / v.reconstruction_path
        rec = pycolmap.Reconstruction(reconstruction)
        removed = clean_sparse(rec)

        def show_preview(rads: list[Path], step: int) -> None:
            preview = work / f"preview_{len(previews)}.json"
            _write_lod_list(rads, preview)
            _update_version(v.id, preview_url=files_url(preview), preview_step=step)
            previews.append(preview)
            # Keep the latest two so a client mid-stream on the previous one doesn't 404.
            while len(previews) > 2:
                previews.pop(0).unlink(missing_ok=True)

        if blocks:
            rad, details = _train_blocks(v, quality, steps, max_resolution, reconstruction, rec, image_root, work,
                                         progress, timing, show_preview)
        else:
            rad, details = _train_global(v, quality, steps, max_resolution, reconstruction, rec, image_root, work,
                                         progress, timing, show_preview)

        with session_scope() as session:
            v = session.get(Version, version_id)
            v.metrics = v.metrics | {
                f"splat_{quality}": {
                    "steps": steps,
                    "max_resolution": max_resolution,
                    "sparse_points_removed": removed,
                    **details,
                    "seconds": round(time.time() - started, 1),
                },
            }
            v.splat_path, v.splat_url, v.splat_kind = rel(rad), files_url(rad), quality
            v.splats = v.splats | {quality: v.splat_url}
            v.train_status = v.train_detail = v.preview_url = v.preview_step = None
            v.train_progress, v.train_eta, v.updated_at = 1.0, 0.0, utcnow()
            session.add(v)
            session.commit()
        log.info("version %s: %s splat published", version_id, quality)
    except TrainingStopped:
        with session_scope() as session:
            v = session.get(Version, version_id)
            cancelled = v.train_status == "cancelling"
        # Automatic jobs go back in the queue (dropped later if no longer HEAD).
        _update_version(
            version_id, train_status=None if cancelled else "queued", preview_url=None, preview_step=None,
            train_detail="cancelled" if cancelled else "paused for a new video", train_progress=0.0, train_eta=None,
        )
        log.info("version %s training %s", version_id, "cancelled" if cancelled else "paused")
    finally:
        progress.close()
        for preview in previews:
            preview.unlink(missing_ok=True)
        for preview_dir in work.glob("**/preview_*"):
            shutil.rmtree(preview_dir, ignore_errors=True)
