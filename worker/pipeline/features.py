"""Feature extraction/matching configuration per scene.

A scene keeps the feature type it was built with (descriptors of different types can't be
matched against each other), recorded in each Version's metrics as "features".

ALIKED (learned) is the default: on the room videos it produced ~100x more verified
cross-video inliers than SIFT (night lighting, portrait vs landscape, wide baselines).
Measured on an M3 CPU: brute-force ALIKED matching is ~200x cheaper than LightGlue and plenty
for consecutive frames, but keeps only ~4% of cross-video pairs, so LightGlue is reserved for
cross-video (retrieval) pairs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from worker import compute

SIFT_VOCAB_TREE_URI = (
    "https://github.com/colmap/colmap/releases/download/3.11.1/"
    "vocab_tree_faiss_flickr100K_words256K.bin;"
    "vocab_tree_faiss_flickr100K_words256K.bin;"
    "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"
)
ALIKED_VOCAB_TREE_URI = (
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
    "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e"
)

DEFAULT_FEATURES = os.environ.get("SCENES_FEATURES", "aliked")


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    extraction: pycolmap.FeatureExtractionOptions
    sequential_matching: pycolmap.FeatureMatchingOptions
    cross_matching: pycolmap.FeatureMatchingOptions
    vocab_tree_uri: str
    device: pycolmap.Device
    # Retrieval budget for cross-video pairs: query every Nth new frame, top-k candidates.
    retrieval_query_stride: int
    retrieval_neighbors: int


def feature_config(name: str | None = None) -> FeatureConfig:
    name = name or DEFAULT_FEATURES
    on_cuda = compute.device() == "cuda"
    extraction = pycolmap.FeatureExtractionOptions()
    sequential = pycolmap.FeatureMatchingOptions()
    cross = pycolmap.FeatureMatchingOptions()
    if name == "sift":
        return FeatureConfig(
            name, extraction, sequential, cross, SIFT_VOCAB_TREE_URI, pycolmap.Device.auto,
            retrieval_query_stride=1, retrieval_neighbors=15,
        )
    if name != "aliked":
        raise ValueError(f"unknown feature type {name!r}")
    extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
    # 1024 px halves CPU extraction time vs 1600 and keeps ~2/3 of cross-video inliers. At
    # 1024 px ALIKED is limited by its score threshold (1.3k keypoints at the default 0.2);
    # 0.1 gives +52% keypoints and +41% verified cross-video inliers for +30% extraction and
    # 2x LightGlue time (0.05 or 1280 px: 3k keypoints but 6.5x LightGlue time). M3, 12 frames.
    extraction.max_image_size = 1600 if on_cuda else 1024
    extraction.aliked.min_score = 0.1
    extraction.aliked.max_num_features = 4096
    sequential.type = pycolmap.FeatureMatcherType.ALIKED_BRUTEFORCE
    cross.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    vocab_tree = ALIKED_VOCAB_TREE_URI
    # Local models also support COLMAP builds without libcurl/download support.
    if model_dir := os.environ.get("SCENES_MODEL_DIR"):
        models = Path(model_dir)
        extraction.aliked.n16rot_model_path = str(models / "aliked-n16rot.onnx")
        sequential.aliked.brute_force.model_path = str(models / "bruteforce-matcher.onnx")
        cross.aliked.lightglue.model_path = str(models / "aliked-lightglue.onnx")
        vocab_tree = str(models / "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin")
    return FeatureConfig(
        name, extraction, sequential, cross, vocab_tree,
        # CoreML can't compile the ALIKED graph, so the Mac stays on CPU ONNX.
        pycolmap.Device.cuda if on_cuda else pycolmap.Device.cpu,
        retrieval_query_stride=1 if on_cuda else 2,
        retrieval_neighbors=20 if on_cuda else 10,
    )
