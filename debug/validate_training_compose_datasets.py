#!/usr/bin/env python
"""Audit both production compose roots and validate one temporal GT batch each.

The output is intentionally self-contained below ``debug/dataset``: it records
manifest/bundle integrity, resolved config, tensor layout, camera reprojection,
GT-as-prediction losses, mask statistics, and visual overlays.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAINING_DIR = REPO / "training"
for _path in (str(REPO), str(TRAINING_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(REPO)

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in (
    ("bool", bool), ("int", int), ("float", float), ("complex", complex),
    ("object", object), ("str", str),
):
    if _name not in np.__dict__:
        setattr(np, _name, _value)
if not hasattr(np, "unicode"):
    np.unicode = str

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from debug.temporal_alignment.check_temporal_batch import (  # noqa: E402
    make_gt_as_prediction,
    mask_metrics,
    model_loss_batch,
    move_to_device,
    process_batch_like_trainer,
    reprojection_metrics,
    save_overlays,
    scalar_metrics,
    tensor_shapes,
    validate_layout,
)
from training.smpl_body import set_smplx_model_root  # noqa: E402
from training.temporal import flatten_temporal_batch_for_framewise_model  # noqa: E402


DEFAULT_OUTPUT = HERE / "dataset"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def camera_arrays_valid(annotation: dict[str, Any]) -> bool:
    try:
        intrinsic = np.asarray(annotation["intrinsics"])
        extrinsic = np.asarray(annotation["extrinsics"])
        original_size = np.asarray(annotation["original_size"])
        track_offset = np.asarray(annotation["track_offset"])
    except Exception:
        return False
    return bool(
        intrinsic.shape == (3, 3)
        and extrinsic.shape == (3, 4)
        and original_size.shape == (2,)
        and track_offset.shape == (2,)
        and np.isfinite(intrinsic).all()
        and np.isfinite(extrinsic).all()
        and np.isfinite(track_offset).all()
    )


def audit_compose_root(root: Path, min_views: int, decode_samples: int) -> dict[str, Any]:
    started = time.perf_counter()
    manifests = sorted(root.rglob("manifest.pkl"))
    counters: Counter[str] = Counter()
    view_histogram: Counter[int] = Counter()
    sample_bundles: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []

    for manifest_path in manifests:
        try:
            with manifest_path.open("rb") as stream:
                manifest = pickle.load(stream)
        except Exception as exc:
            counters["unreadable_manifests"] += 1
            errors.append(f"{manifest_path}: {exc}")
            continue
        if manifest.get("format") != "mamma_compose_518" or int(
            manifest.get("version", -1)
        ) != 1:
            counters["wrong_format_manifests"] += 1
            continue
        if tuple(int(value) for value in manifest.get("target_shape", ())) != (518, 518):
            counters["wrong_shape_manifests"] += 1
            continue
        counters["valid_manifests"] += 1
        frames = manifest.get("frames", {})
        counters["manifest_frames"] += len(frames)
        for frame, annotations in frames.items():
            view_histogram[len(annotations)] += 1
            if len(annotations) < min_views:
                counters["frames_below_min_views"] += 1
            people_counts = {int(annotation.get("num_people", -1)) for annotation in annotations}
            if len(people_counts) != 1:
                counters["frames_with_inconsistent_people"] += 1
            for annotation in annotations:
                counters["manifest_views"] += 1
                image_path = manifest_path.parent / annotation["image_path"]
                camera_path = manifest_path.parent / annotation["camera_path"]
                mask_rel = annotation.get("mask_path")
                mask_path = manifest_path.parent / mask_rel if mask_rel else None
                if not image_path.is_file():
                    counters["missing_images"] += 1
                if not camera_path.is_file():
                    counters["missing_cameras"] += 1
                if mask_path is None or not mask_path.is_file():
                    counters["missing_masks"] += 1
                if not camera_arrays_valid(annotation):
                    counters["bad_manifest_camera_arrays"] += 1
                if len(sample_bundles) < decode_samples:
                    sample_bundles.append((manifest_path.parent, annotation))

    for sequence_root, annotation in sample_bundles:
        image_path = sequence_root / annotation["image_path"]
        mask_path = sequence_root / annotation["mask_path"]
        camera_path = sequence_root / annotation["camera_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[:2] != (518, 518):
            counters["bad_sample_images"] += 1
        if mask is None or mask.shape != (518, 518):
            counters["bad_sample_masks"] += 1
        try:
            with np.load(camera_path) as camera:
                for key in ("intrinsics", "extrinsics", "original_size", "track_offset"):
                    if not np.allclose(camera[key], annotation[key], rtol=0.0, atol=1e-6):
                        counters["sample_camera_manifest_mismatches"] += 1
                        break
        except Exception:
            counters["bad_sample_cameras"] += 1

    summary = read_json(root / "compose_all_summary.json")
    marker = read_json(root / "compose_complete.json")
    summary_matches = bool(
        summary
        and int(summary.get("sequences", -1)) == len(manifests)
        and int(summary.get("frames", -1)) == counters["manifest_frames"]
        and int(summary.get("views", -1)) == counters["manifest_views"]
    )
    marker_matches = bool(
        marker
        and int(marker.get("sequences", -1)) == len(manifests)
        and int(marker.get("frames", -1)) == counters["manifest_frames"]
        and int(marker.get("views", -1)) == counters["manifest_views"]
    )
    fatal_keys = (
        "unreadable_manifests", "wrong_format_manifests", "wrong_shape_manifests",
        "missing_images", "missing_cameras", "missing_masks",
        "bad_manifest_camera_arrays", "bad_sample_images", "bad_sample_masks",
        "bad_sample_cameras", "sample_camera_manifest_mismatches",
        "frames_with_inconsistent_people",
    )
    bundle_integrity_pass = all(counters[key] == 0 for key in fatal_keys)
    strict_complete = bool(
        bundle_integrity_pass
        and summary_matches
        and marker_matches
        and counters["frames_below_min_views"] == 0
        and int((summary or {}).get("failed_views", 0)) == 0
        and not (summary or {}).get("incomplete_sequences")
    )
    usable_frames = counters["manifest_frames"] - counters["frames_below_min_views"]
    return {
        "root": str(root),
        "strict_complete": strict_complete,
        "training_usable": bool(bundle_integrity_pass and usable_frames > 0),
        "bundle_integrity_pass": bundle_integrity_pass,
        "completion_marker_present": marker is not None,
        "summary_matches_manifests": summary_matches,
        "completion_marker_matches_manifests": marker_matches,
        "usable_frames_at_min_views": usable_frames,
        "view_count_histogram": dict(sorted(view_histogram.items())),
        "counts": dict(counters),
        "summary": summary,
        "completion_marker": marker,
        "errors_preview": errors[:20],
        "sampled_bundle_count": len(sample_bundles),
        "seconds": time.perf_counter() - started,
    }


def one_source_config(base: Any, dataset_index: int, frames: int, views: int) -> Any:
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    selected = cfg.data.train.dataset.dataset_configs[dataset_index]
    with open_dict(cfg):
        cfg.num_workers = 0
        cfg.max_img_per_gpu = frames * views
        cfg.data.train.num_workers = 0
        cfg.data.train.max_img_per_gpu = frames * views
        cfg.data.train.shuffle = False
        cfg.data.train.pin_memory = False
        cfg.data.train.persistent_workers = False
        cfg.data.train.dataset.dataset_configs = [selected]
        selected = cfg.data.train.dataset.dataset_configs[0]
        selected.split = "test"
        selected.val_sequence_fraction = 0.0
        selected.max_sequences = 1
        selected.max_frames_per_sequence = max(frames + 2, 12)
        common = cfg.data.train.common_config
        common.training = False
        common.fixed_view_sampling = True
        common.fix_img_num = views
        common.img_nums = [views, views]
        common.include_metadata = True
        common.use_temporal_training = True
        common.temporal_clip_length = frames
        common.temporal_clip_stride = 1
        common.augs.cojitter = False
        common.augs.color_jitter = None
    return cfg


def validate_one_batch(
    cfg: Any,
    source: str,
    output: Path,
    device: torch.device,
    seed: int,
    projection_tolerance: float,
    core_loss_tolerance: float,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()
    dynamic = instantiate(cfg.data.train, _recursive_=False)
    init_seconds = time.perf_counter() - started
    dynamic.seed = seed
    started = time.perf_counter()
    raw_batch = next(iter(dynamic.get_loader(epoch=0)))
    load_seconds = time.perf_counter() - started

    processed = process_batch_like_trainer(
        raw_batch, scale_by_extrinsics=bool(cfg.scale_by_extrinsics)
    )
    flat = flatten_temporal_batch_for_framewise_model(processed)
    layout = validate_layout(raw_batch, flat)
    reprojection, reprojection_by_frame, projected = reprojection_metrics(raw_batch)
    overlays = save_overlays(raw_batch, projected, output / source / "overlays")

    compact = move_to_device(model_loss_batch(processed), device)
    flat_device = flatten_temporal_batch_for_framewise_model(compact)
    loss_module = instantiate(cfg.loss, _recursive_=False).to(device).eval()
    with torch.no_grad():
        predictions = make_gt_as_prediction(flat_device, cfg)
        losses = scalar_metrics(loss_module(predictions, flat_device))
    zero_core_keys = (
        "loss_camera", "loss_T", "loss_R", "loss_FL", "loss_smpl_losses",
        "loss_mesh_translate", "loss_smpl_presence", "loss_smpl_joints3d",
        "loss_smpl_vertices", "loss_mask",
    )
    max_core_loss = max(abs(losses.get(key, 0.0)) for key in zero_core_keys)
    projection_pass = bool(
        reprojection["max_px"] is not None
        and reprojection["max_px"] <= projection_tolerance
    )
    losses_finite = bool(losses) and all(np.isfinite(value) for value in losses.values())
    result = {
        "passed": bool(
            layout["passed"]
            and projection_pass
            and losses_finite
            and max_core_loss <= core_loss_tolerance
        ),
        "source": source,
        "sequence_names": raw_batch.get("seq_name"),
        "dataset_init_seconds": init_seconds,
        "first_batch_seconds": load_seconds,
        "shapes": tensor_shapes(raw_batch),
        "layout": layout,
        "reprojection": reprojection,
        "reprojection_by_frame": reprojection_by_frame,
        "projection_tolerance_px": projection_tolerance,
        "projection_pass": projection_pass,
        "mask": mask_metrics(raw_batch),
        "gt_as_prediction_losses": losses,
        "zero_core_keys": list(zero_core_keys),
        "max_zero_core_loss": max_core_loss,
        "core_loss_tolerance": core_loss_tolerance,
        "all_losses_finite": losses_finite,
        "overlays": overlays,
    }
    source_dir = output / source
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    OmegaConf.save(cfg, source_dir / "config_resolved.yaml", resolve=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--projection-tolerance-px", type=float, default=2.0)
    parser.add_argument("--core-loss-tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(TRAINING_DIR / "config"), version_base=None):
        base = compose(config_name=args.config)
    OmegaConf.resolve(base)
    roots = {
        "mamma": Path(base.mamma_compose_root),
        "harmony4d": Path(base.harmony4d_compose_root),
    }
    missing = [str(path) for path in roots.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing compose roots: {missing}")
    smplx_root = Path(base.loss.smplx_model_dir)
    set_smplx_model_root(smplx_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    integrity = {
        name: audit_compose_root(root, args.views, args.decode_samples)
        for name, root in roots.items()
    }
    (output / "integrity.json").write_text(
        json.dumps(integrity, indent=2, default=str), encoding="utf-8"
    )

    batches = {}
    for dataset_index, source in enumerate(("mamma", "harmony4d")):
        cfg = one_source_config(base, dataset_index, args.frames, args.views)
        batches[source] = validate_one_batch(
            cfg, source, output, device, args.seed,
            args.projection_tolerance_px, args.core_loss_tolerance,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    passed = bool(
        all(item["training_usable"] for item in integrity.values())
        and all(item["passed"] for item in batches.values())
    )
    result = {
        "passed": passed,
        "strict_completeness": {
            name: item["strict_complete"] for name, item in integrity.items()
        },
        "training_usable": {
            name: item["training_usable"] for name, item in integrity.items()
        },
        "batch_validation": batches,
        "integrity_report": str(output / "integrity.json"),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"[result] {'PASS' if passed else 'FAIL'}")
    print(f"[output] {output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
