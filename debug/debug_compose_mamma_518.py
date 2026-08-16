#!/usr/bin/env python
"""Compare one production YAML batch from raw MAMMA and its 518 compose cache.

The check uses ``mamma_harmony4d_mask_dpt.yaml`` through Hydra's normal
``DynamicTorchDataset`` path. It validates tensor equality, camera-aware GT
reprojection, GT-as-prediction loss values, masks, temporal layout, and loader
timing. Visual overlays are written below the requested debug output folder.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

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


DEFAULT_SEQUENCE = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/mamma/mamma/"
    "moyo_4-6_C_200_00/be_sf-zEiRjQqSO_seq_000000"
)
DEFAULT_CACHE = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/mamma/mamma_compose"
)
DEFAULT_SMPLX = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/mamma/smplx_models"
)
DEFAULT_OUTPUT = HERE / "compose_mamma_518_validation"


def configure(cfg, args, *, use_cache: bool):
    selected = next(
        item
        for item in cfg.data.train.dataset.dataset_configs
        if str(item._target_).endswith("sys_smpl_multi.SysSMPLMultiDataset")
    )
    with open_dict(cfg):
        cfg.num_workers = 0
        cfg.max_img_per_gpu = args.frames * args.views
        cfg.data.train.num_workers = 0
        cfg.data.train.max_img_per_gpu = args.frames * args.views
        cfg.data.train.shuffle = False
        cfg.data.train.pin_memory = False
        cfg.data.train.persistent_workers = False
        cfg.data.train.dataset.dataset_configs = [selected]
        selected = cfg.data.train.dataset.dataset_configs[0]
        common = cfg.data.train.common_config
        common.training = False
        common.fixed_view_sampling = True
        common.fix_img_num = args.views
        common.fix_aspect_ratio = 1.0
        common.img_nums = [args.views, args.views]
        common.augs.aspects = [1.0, 1.0]
        common.augs.scales = None
        common.augs.cojitter = False
        common.augs.color_jitter = None
        common.include_metadata = True
        common.emit_dense_geometry = False
        common.profile_data_loading = True
        common.profile_data_loading_every = 1000
        common.use_temporal_training = True
        common.temporal_clip_length = args.frames
        common.temporal_clip_stride = 1

        selected.SysSMPL_DIR = str(args.sequence)
        selected.SysSMPL_ANNOTATION_DIR = str(args.sequence)
        selected.min_num_images = args.views
        selected.max_num_people = args.max_people
        selected.max_sequences = 1
        selected.max_frames_per_sequence = args.max_index_frames
        selected.emit_landmarks = False
        selected.emit_contact = False
        selected.emit_person_mask = True
        selected.person_mask_stride = 1
        selected.compose_cache_root = str(args.cache_root)
        selected.prefer_compose_cache = use_cache
        selected.require_complete_compose_cache = False
        cfg.loss.smplx_model_dir = str(args.smplx_root)
    return cfg


def load_one(cfg, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()
    dynamic = instantiate(cfg.data.train, _recursive_=False)
    init_seconds = time.perf_counter() - started
    dynamic.seed = seed
    started = time.perf_counter()
    batch = next(iter(dynamic.get_loader(epoch=0)))
    next_seconds = time.perf_counter() - started
    composed = dynamic.dataset
    profile = {
        key: float(value)
        for key, value in getattr(composed, "_profile_totals", {}).items()
    }
    return batch, {
        "dataset_init_seconds": init_seconds,
        "first_batch_seconds": next_seconds,
        "profile_one_sample_seconds": profile,
    }


def compare_batches(raw: dict[str, Any], cached: dict[str, Any], args):
    exact_keys = (
        "frame_ids", "view_ids", "ids", "smpl_gender", "has_smpl",
        "num_people", "smpl_joints2d_confidence", "person_mask",
    )
    float_keys = (
        "images", "extrinsics", "intrinsics", "original_sizes", "smpl_pose",
        "smpl_beta", "smpl_trans", "smpl_joints2d", "smpl_joints3d_world",
    )
    values = {}
    for key in exact_keys:
        same_shape = key in raw and key in cached and raw[key].shape == cached[key].shape
        values[key] = {
            "same_shape": bool(same_shape),
            "exact": bool(same_shape and torch.equal(raw[key], cached[key])),
        }
    for key in float_keys:
        same_shape = key in raw and key in cached and raw[key].shape == cached[key].shape
        maximum = None
        mean = None
        if same_shape:
            delta = (raw[key].float() - cached[key].float()).abs()
            maximum = float(delta.max().item()) if delta.numel() else 0.0
            mean = float(delta.mean().item()) if delta.numel() else 0.0
        values[key] = {
            "same_shape": bool(same_shape),
            "max_abs": maximum,
            "mean_abs": mean,
        }
    checks = {
        "all_shapes_match": all(item["same_shape"] for item in values.values()),
        "discrete_gt_exact": all(values[key]["exact"] for key in exact_keys),
        "rgb_exact": values["images"]["max_abs"] == 0.0,
        "camera_close": max(
            values["intrinsics"]["max_abs"], values["extrinsics"]["max_abs"]
        ) <= args.camera_tolerance,
        "smpl_close": max(
            values["smpl_pose"]["max_abs"],
            values["smpl_beta"]["max_abs"],
            values["smpl_trans"]["max_abs"],
        ) <= args.smpl_tolerance,
        "joints2d_close": values["smpl_joints2d"]["max_abs"] <= args.joints_tolerance_px,
    }
    return values, checks


def gt_losses(cfg, batch, device):
    processed = process_batch_like_trainer(
        batch, scale_by_extrinsics=bool(cfg.scale_by_extrinsics)
    )
    compact = move_to_device(model_loss_batch(processed), device)
    flat = flatten_temporal_batch_for_framewise_model(compact)
    module = instantiate(cfg.loss, _recursive_=False).to(device).eval()
    with torch.no_grad():
        prediction = make_gt_as_prediction(flat, cfg)
        values = scalar_metrics(module(prediction, flat))
    return processed, flat, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--max-people", type=int, default=6)
    parser.add_argument("--max-index-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--projection-tolerance-px", type=float, default=2.0)
    parser.add_argument("--camera-tolerance", type=float, default=1e-4)
    parser.add_argument("--smpl-tolerance", type=float, default=1e-6)
    parser.add_argument("--joints-tolerance-px", type=float, default=2e-3)
    parser.add_argument("--loss-tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    for name in ("sequence", "cache_root", "smplx_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    required = (args.sequence, args.cache_root, args.smplx_root)
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing required directories: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    set_smplx_model_root(args.smplx_root)

    with initialize_config_dir(config_dir=str(TRAINING_DIR / "config"), version_base=None):
        base = compose(config_name=args.config)
    raw_cfg = configure(OmegaConf.create(OmegaConf.to_container(base, resolve=False)), args, use_cache=False)
    cache_cfg = configure(OmegaConf.create(OmegaConf.to_container(base, resolve=False)), args, use_cache=True)
    OmegaConf.save(cache_cfg, args.output / "config_cache_resolved.yaml")

    raw_batch, raw_timing = load_one(raw_cfg, args.seed)
    cache_batch, cache_timing = load_one(cache_cfg, args.seed)
    # Repeat after both paths and the lazily-created SMPL body models have been
    # touched. This separates steady-state data-path gains from cold-start cost.
    raw_repeat_batch, raw_repeat_timing = load_one(raw_cfg, args.seed)
    del raw_repeat_batch
    gc.collect()
    cache_repeat_batch, cache_repeat_timing = load_one(cache_cfg, args.seed)
    del cache_repeat_batch
    gc.collect()
    comparison, comparison_checks = compare_batches(raw_batch, cache_batch, args)

    device = torch.device(args.device)
    raw_processed, raw_flat, raw_loss = gt_losses(raw_cfg, raw_batch, device)
    cache_processed, cache_flat, cache_loss = gt_losses(cache_cfg, cache_batch, device)
    common_loss_keys = sorted(set(raw_loss) & set(cache_loss))
    loss_delta = {
        key: abs(raw_loss[key] - cache_loss[key]) for key in common_loss_keys
    }
    max_loss_delta = max(loss_delta.values(), default=0.0)

    raw_reprojection, raw_by_frame, raw_projected = reprojection_metrics(raw_batch)
    cache_reprojection, cache_by_frame, cache_projected = reprojection_metrics(cache_batch)
    raw_layout = validate_layout(raw_batch, flatten_temporal_batch_for_framewise_model(raw_processed))
    cache_layout = validate_layout(cache_batch, flatten_temporal_batch_for_framewise_model(cache_processed))
    overlays = save_overlays(cache_batch, cache_projected, args.output / "overlays_cache")

    projection_pass = all(
        value["max_px"] is not None and value["max_px"] <= args.projection_tolerance_px
        for value in (raw_reprojection, cache_reprojection)
    )
    loss_pass = max_loss_delta <= args.loss_tolerance
    comparison_pass = all(comparison_checks.values())
    passed = bool(
        comparison_pass and projection_pass and loss_pass
        and raw_layout["passed"] and cache_layout["passed"]
    )
    speedup = (
        raw_timing["first_batch_seconds"] / cache_timing["first_batch_seconds"]
        if cache_timing["first_batch_seconds"] > 0 else None
    )
    repeat_speedup = (
        raw_repeat_timing["first_batch_seconds"]
        / cache_repeat_timing["first_batch_seconds"]
        if cache_repeat_timing["first_batch_seconds"] > 0 else None
    )
    result = {
        "passed": passed,
        "config": args.config,
        "num_workers": 0,
        "raw_timing": raw_timing,
        "cache_timing": cache_timing,
        "first_batch_speedup": speedup,
        "raw_warm_repeat_timing": raw_repeat_timing,
        "cache_warm_repeat_timing": cache_repeat_timing,
        "warm_repeat_speedup": repeat_speedup,
        "raw_shapes": tensor_shapes(raw_batch),
        "cache_shapes": tensor_shapes(cache_batch),
        "comparison": comparison,
        "comparison_checks": comparison_checks,
        "raw_layout": raw_layout,
        "cache_layout": cache_layout,
        "raw_reprojection": raw_reprojection,
        "cache_reprojection": cache_reprojection,
        "raw_reprojection_by_frame": raw_by_frame,
        "cache_reprojection_by_frame": cache_by_frame,
        "projection_pass": projection_pass,
        "raw_mask": mask_metrics(raw_batch),
        "cache_mask": mask_metrics(cache_batch),
        "raw_gt_as_prediction_loss": raw_loss,
        "cache_gt_as_prediction_loss": cache_loss,
        "loss_abs_delta": loss_delta,
        "max_loss_abs_delta": max_loss_delta,
        "loss_pass": loss_pass,
        "overlays": overlays,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"[result] {'PASS' if passed else 'FAIL'}")
    print(f"[output] {args.output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
