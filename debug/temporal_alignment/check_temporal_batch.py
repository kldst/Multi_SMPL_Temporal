#!/usr/bin/env python
"""Validate one real temporal batch, reprojection, flattening, and losses.

This script mirrors the production path in ``Trainer._step``:

1. load one time-major clip as [B, T*V, ...];
2. apply the same camera/world gauge normalization as Trainer._process_batch;
3. keep the clip intact while the temporal VGGT model runs;
4. flatten GT to [B*T, V, ...] only after model forward;
5. evaluate the configured camera, SMPL, mask, and temporal losses.

The default sample is a local, real MAMMA sequence. Harmony4D can also be used
when its SMPL-X annotation/mask roots contain consecutive annotated frames.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAINING_DIR = REPO / "training"
for _path in (str(REPO), str(TRAINING_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(REPO)

# Compatibility for the legacy chumpy/SMPL-X pickle dependencies.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for _name, _value in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if _name not in np.__dict__:
        setattr(np, _name, _value)
if not hasattr(np, "unicode"):
    np.unicode = str

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from training.smpl_body import (  # noqa: E402
    compute_gt_mesh_rot,
    compute_gt_mesh_translate,
    set_smplx_model_root,
)
from training.temporal import flatten_temporal_batch_for_framewise_model  # noqa: E402
from training.train_utils.normalization import (  # noqa: E402
    normalize_camera_extrinsics_points_and_3djoints_batch,
)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402


DEFAULT_MAMMA_SEQUENCE = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/mamma/mamma/"
    "harmony4d_train_1_NC_200_00_contact/"
    "be_HsuS3iLSSWWZ_seq_000000"
)
DEFAULT_HARMONY_ROOT = Path(
    "/train-data-2-hdd/yian/Multi_SMPL_Dataset_real/Harmony4D"
)
DEFAULT_HARMONY_ANNOTATIONS = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/smpl_transfer/output"
)
DEFAULT_HARMONY_MASKS = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/smpl_transfer/masks"
)
DEFAULT_SMPLX_ROOT = Path(
    "/train-data-3-hdd/yian/Multi_SMPL_0706/mamma/smplx_models"
)
DEFAULT_OUTPUT = HERE / "outputs" / "one_batch"


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_loss_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Drop dense depth/point targets unused by this config before H2D copy."""
    needed = {
        "seq_name", "ids", "images", "extrinsics", "intrinsics",
        "raw_extrinsics", "avg_scale", "views_per_frame",
        "temporal_num_frames", "frame_ids", "view_ids", "smpl_pose",
        "smpl_beta", "smpl_trans", "smpl_gender", "has_smpl",
        "num_people", "smpl_joints2d", "smpl_joints2d_confidence",
        "smpl_joints3d_world", "person_mask",
    }
    return {key: value for key, value in batch.items() if key in needed}


def tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def scalar_metrics(values: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu().item())
        for key, value in values.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def process_batch_like_trainer(batch: dict[str, Any], scale_by_extrinsics: bool):
    out = clone_batch(batch)
    raw_extrinsics = out["extrinsics"].clone()
    (
        norm_extrinsics,
        norm_cam_points,
        norm_world_points,
        norm_joints,
        norm_depths,
        avg_scale,
    ) = normalize_camera_extrinsics_points_and_3djoints_batch(
        extrinsics=out["extrinsics"],
        cam_points=out.get("cam_points"),
        world_points=out.get("world_points"),
        joints3d_world=out.get("smpl_joints3d_world"),
        depths=out.get("depths"),
        scale_by_extrinsics=scale_by_extrinsics,
        point_masks=out.get("point_masks"),
    )
    out["raw_extrinsics"] = raw_extrinsics
    out["extrinsics"] = norm_extrinsics
    out["avg_scale"] = avg_scale
    for key, value in (
        ("cam_points", norm_cam_points),
        ("world_points", norm_world_points),
        ("smpl_joints3d_world", norm_joints),
        ("depths", norm_depths),
    ):
        if value is not None:
            out[key] = value
    return out


def project_joints(batch: dict[str, Any]):
    points = batch["smpl_joints3d_world"].float()
    extrinsics = batch["extrinsics"].float()
    intrinsics = batch["intrinsics"].float()
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    camera = torch.einsum("bsij,bspnj->bspni", rotation, points)
    camera = camera + translation[:, :, None, None]
    homogeneous = torch.einsum("bsij,bspnj->bspni", intrinsics, camera)
    depth = camera[..., 2]
    safe_depth = depth.clamp(min=1e-8)
    pixels = homogeneous[..., :2] / safe_depth[..., None]
    return pixels, depth


def expanded_has_smpl(batch: dict[str, Any]) -> torch.Tensor:
    has_smpl = batch["has_smpl"] > 0.5
    if has_smpl.dim() == 2:
        return has_smpl[:, None].expand(-1, batch["images"].shape[1], -1)
    if has_smpl.dim() != 3:
        raise ValueError(f"Unexpected has_smpl shape {tuple(has_smpl.shape)}")
    B, T, P = has_smpl.shape
    V = int(batch["views_per_frame"].reshape(-1)[0].item())
    return has_smpl[:, :, None].expand(B, T, V, P).reshape(B, T * V, P)


def reprojection_metrics(batch: dict[str, Any]):
    projected, depth = project_joints(batch)
    target = batch["smpl_joints2d"].float()
    confidence = batch["smpl_joints2d_confidence"] > 0.5
    valid = confidence & expanded_has_smpl(batch)[..., None]
    valid = valid & (depth > 1e-6) & torch.isfinite(projected).all(dim=-1)
    error = torch.linalg.vector_norm(projected - target, dim=-1)
    values = error[valid]
    overall = {
        "valid_joint_count": int(values.numel()),
        "mean_px": float(values.mean().cpu()) if values.numel() else None,
        "median_px": float(values.median().cpu()) if values.numel() else None,
        "max_px": float(values.max().cpu()) if values.numel() else None,
    }

    B = batch["images"].shape[0]
    T = int(batch["temporal_num_frames"].reshape(-1)[0].item())
    V = int(batch["views_per_frame"].reshape(-1)[0].item())
    by_frame = []
    for batch_index in range(B):
        for time_index in range(T):
            view_slice = slice(time_index * V, (time_index + 1) * V)
            frame_values = error[batch_index, view_slice][valid[batch_index, view_slice]]
            by_frame.append(
                {
                    "batch": batch_index,
                    "time": time_index,
                    "count": int(frame_values.numel()),
                    "mean_px": (
                        float(frame_values.mean().cpu())
                        if frame_values.numel()
                        else None
                    ),
                    "max_px": (
                        float(frame_values.max().cpu())
                        if frame_values.numel()
                        else None
                    ),
                }
            )
    return overall, by_frame, projected


def pad_people(
    tensor: torch.Tensor,
    target_people: int,
    people_axis: int,
    fill: float = 0.0,
) -> torch.Tensor:
    current = tensor.shape[people_axis]
    if current >= target_people:
        return tensor
    shape = list(tensor.shape)
    shape[people_axis] = target_people - current
    padding = torch.full(
        shape, fill, device=tensor.device, dtype=tensor.dtype
    )
    return torch.cat([tensor, padding], dim=people_axis)


@torch.no_grad()
def make_gt_as_prediction(batch: dict[str, Any], cfg):
    model_people = int(cfg.model.smpl_num_people)
    mesh_translate = compute_gt_mesh_translate(
        batch, normalize_cam=True, use_mamma=True
    )
    mesh_rot = compute_gt_mesh_rot(batch)
    pose = batch["smpl_pose"].clone()
    pose[..., :3] = mesh_rot.to(dtype=pose.dtype)

    pose = pad_people(pose, model_people, 1)
    beta = pad_people(batch["smpl_beta"].clone(), model_people, 1)
    mesh_translate = pad_people(mesh_translate, model_people, 1)
    has_smpl = pad_people(batch["has_smpl"].clone(), model_people, 1)
    pose_encoding = extri_intri_to_pose_encoding(
        batch["extrinsics"],
        batch["intrinsics"],
        batch["images"].shape[-2:],
        pose_encoding_type="absT_quaR_FoV",
    )
    predictions = {
        "pose_enc_list": [pose_encoding],
        "pose_enc": pose_encoding,
        "smpl_pose": pose,
        "pred_pose_0": pose.clone(),
        "smpl_beta": beta,
        "mesh_translate": mesh_translate,
        "mesh_rot": pose[..., :3],
        "smpl_presence_logits": torch.where(
            has_smpl > 0.5,
            torch.full_like(has_smpl, 20.0),
            torch.full_like(has_smpl, -20.0),
        ),
    }
    if "person_mask" in batch:
        mask = pad_people(batch["person_mask"], model_people, 2)
        predictions["person_mask_logits"] = torch.logit(
            mask.float().clamp(1e-6, 1.0 - 1e-6)
        )
    return predictions


def validate_layout(raw_batch: dict[str, Any], flat_batch: dict[str, Any]):
    B = int(raw_batch["images"].shape[0])
    T = int(raw_batch["temporal_num_frames"].reshape(-1)[0].item())
    V = int(raw_batch["views_per_frame"].reshape(-1)[0].item())
    frame_ids = raw_batch["frame_ids"]
    ids = raw_batch["ids"].reshape(B, T, V)
    view_ids = raw_batch["view_ids"]
    checks = {
        "batch_size_is_one": B == 1,
        "images_are_B_TxV": tuple(raw_batch["images"].shape[:2]) == (B, T * V),
        "flat_images_are_BxT_V": tuple(flat_batch["images"].shape[:2]) == (B * T, V),
        "frame_ids_are_consecutive": bool(
            torch.all(frame_ids[:, 1:] - frame_ids[:, :-1] == 1).item()
        ),
        "same_views_each_timestep": bool(
            torch.all(ids == ids[:, :1]).item()
        ),
        "view_ids_match_ids": bool(
            torch.all(ids[:, 0] == view_ids).item()
        ),
        "person_params_are_B_T_P": (
            raw_batch["smpl_pose"].shape[:2] == (B, T)
        ),
        "flat_person_params_are_BxT_P": flat_batch["smpl_pose"].shape[0] == B * T,
        "view_person_gt_is_B_TxV_P": (
            raw_batch["smpl_joints2d"].shape[:2] == (B, T * V)
        ),
        "flat_view_person_gt_is_BxT_V_P": (
            flat_batch["smpl_joints2d"].shape[:2] == (B * T, V)
        ),
        "temporal_shape_marker_is_B_T": bool(
            torch.equal(
                flat_batch["temporal_shape"].cpu(),
                torch.tensor([B, T], dtype=torch.long),
            )
        ),
    }
    return {
        "B": B,
        "T": T,
        "V": V,
        "frame_ids": frame_ids.cpu().tolist(),
        "view_ids": view_ids.cpu().tolist(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def mask_metrics(batch: dict[str, Any]):
    mask = batch.get("person_mask")
    if mask is None:
        return {"available": False}
    mask = mask.float()
    B = mask.shape[0]
    T = int(batch["temporal_num_frames"].reshape(-1)[0].item())
    V = int(batch["views_per_frame"].reshape(-1)[0].item())
    P = mask.shape[2]
    temporal = mask.reshape(B, T, V, P, *mask.shape[-2:])
    area = temporal.mean(dim=(-1, -2))
    overlap = (temporal.sum(dim=3) > 1.0).float().mean()
    return {
        "available": True,
        "finite": bool(torch.isfinite(mask).all().item()),
        "min": float(mask.min().cpu()),
        "max": float(mask.max().cpu()),
        "mean_area_fraction_B_T_V_P": area.cpu().tolist(),
        "multi_person_overlap_fraction": float(overlap.cpu()),
    }


def save_overlays(
    batch: dict[str, Any],
    projected: torch.Tensor,
    output: Path,
):
    output.mkdir(parents=True, exist_ok=True)
    B = batch["images"].shape[0]
    if B != 1:
        raise ValueError("Overlay writer expects the requested one-sample batch")
    T = int(batch["temporal_num_frames"].reshape(-1)[0].item())
    V = int(batch["views_per_frame"].reshape(-1)[0].item())
    images = batch["images"][0].detach().float().cpu()
    if images.dtype == torch.uint8:
        images = images.float().div(255)
    images = images.permute(0, 2, 3, 1).clamp(0, 1).numpy()
    images = np.rint(images * 255.0).astype(np.uint8)
    target = batch["smpl_joints2d"][0].cpu().numpy()
    confidence = batch["smpl_joints2d_confidence"][0].cpu().numpy()
    projected_np = projected[0].detach().cpu().numpy()
    has_smpl = expanded_has_smpl(batch)[0].cpu().numpy()
    masks = batch.get("person_mask")
    masks_np = masks[0].cpu().numpy() if masks is not None else None
    palette = (
        (255, 100, 40),
        (80, 200, 255),
        (140, 255, 80),
        (230, 100, 255),
        (255, 220, 80),
        (80, 255, 220),
    )
    saved = []
    for time_index in range(T):
        canvases = []
        for view_index in range(V):
            sequence_index = time_index * V + view_index
            canvas = cv2.cvtColor(images[sequence_index], cv2.COLOR_RGB2BGR)
            if masks_np is not None:
                for person_index in range(masks_np.shape[1]):
                    if not has_smpl[sequence_index, person_index]:
                        continue
                    foreground = masks_np[sequence_index, person_index] >= 0.5
                    color = np.asarray(palette[person_index % len(palette)])
                    canvas[foreground] = (
                        0.78 * canvas[foreground] + 0.22 * color
                    ).astype(np.uint8)
            for person_index in range(target.shape[1]):
                if not has_smpl[sequence_index, person_index]:
                    continue
                for joint_index in range(target.shape[2]):
                    if confidence[sequence_index, person_index, joint_index] <= 0.5:
                        continue
                    gt_xy = tuple(
                        np.rint(
                            target[sequence_index, person_index, joint_index]
                        ).astype(int)
                    )
                    pred_xy = tuple(
                        np.rint(
                            projected_np[
                                sequence_index, person_index, joint_index
                            ]
                        ).astype(int)
                    )
                    cv2.circle(canvas, gt_xy, 3, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.drawMarker(
                        canvas,
                        pred_xy,
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        7,
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                canvas,
                f"t={time_index} v={view_index}: green=GT, red-x=reprojected",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            path = output / f"t{time_index:02d}_v{view_index:02d}.jpg"
            if not cv2.imwrite(str(path), canvas):
                raise RuntimeError(f"Could not write {path}")
            saved.append(str(path))
            canvases.append(canvas)
        contact_sheet = output / f"t{time_index:02d}_contact_sheet.jpg"
        cv2.imwrite(str(contact_sheet), np.hstack(canvases))
        saved.append(str(contact_sheet))
    return saved


def load_checkpoint(model: torch.nn.Module, checkpoint: Path):
    state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "path": str(checkpoint),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_preview": list(missing[:20]),
        "unexpected_preview": list(unexpected[:20]),
    }


@torch.no_grad()
def run_model_and_loss(cfg, clip_batch, flat_batch, loss_module, device, checkpoint):
    model = instantiate(cfg.model, _recursive_=False).to(device).eval()
    checkpoint_info = None
    if checkpoint is not None:
        checkpoint_info = load_checkpoint(model, checkpoint)
    smpl_inputs = {
        key: clip_batch[key]
        for key in (
            "views_per_frame",
            "temporal_num_frames",
            "frame_ids",
            "view_ids",
        )
        if key in clip_batch
    }
    images = clip_batch["images"]
    if images.dtype == torch.uint8:
        images = images.float().div(255)
    amp_enabled = device.type == "cuda" and bool(cfg.optim.amp.enabled)
    amp_dtype = (
        torch.bfloat16
        if str(cfg.optim.amp.amp_dtype).lower() == "bfloat16"
        else torch.float16
    )
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        predictions = model(images=images, smpl_inputs=smpl_inputs)
        loss_dict = loss_module(predictions, flat_batch)
    losses = scalar_metrics(loss_dict)
    return {
        "checkpoint": checkpoint_info,
        "random_weights": checkpoint is None,
        "prediction_shapes": tensor_shapes(predictions),
        "losses": losses,
        "all_predictions_finite": all(
            bool(torch.isfinite(value).all().item())
            for value in predictions.values()
            if torch.is_tensor(value) and value.is_floating_point()
        ),
        "all_losses_finite": bool(losses) and all(
            np.isfinite(value) for value in losses.values()
        ),
    }


def configure_one_batch(cfg, args):
    target_suffix = (
        "sys_smpl_multi.SysSMPLMultiDataset"
        if args.source == "mamma"
        else "harmony4d.Harmony4DDataset"
    )
    candidates = list(cfg.data.train.dataset.dataset_configs)
    selected = next(
        item
        for item in candidates
        if str(item._target_).endswith(target_suffix)
    )
    with open_dict(cfg):
        cfg.num_workers = 0
        cfg.max_img_per_gpu = args.temporal_frames * args.views
        cfg.data.train.num_workers = 0
        cfg.data.train.max_img_per_gpu = args.temporal_frames * args.views
        cfg.data.train.shuffle = False
        cfg.data.train.pin_memory = False
        cfg.data.train.persistent_workers = False
        cfg.data.train.dataset.dataset_configs = [selected]
        # OmegaConf copies the selected node when assigning a new ListConfig;
        # mutate the installed copy rather than the stale source-node reference.
        selected = cfg.data.train.dataset.dataset_configs[0]
        common = cfg.data.train.common_config
        common.training = False
        common.fixed_view_sampling = True
        common.fix_img_num = args.views
        common.img_nums = [args.views, args.views]
        common.include_metadata = True
        common.use_temporal_training = True
        common.temporal_clip_length = args.temporal_frames
        common.temporal_clip_stride = args.temporal_stride
        common.augs.cojitter = False
        common.augs.color_jitter = None

        selected.min_num_images = args.views
        selected.max_num_people = args.max_people
        selected.max_sequences = 1
        selected.max_frames_per_sequence = args.max_index_frames
        if args.source == "mamma":
            selected.SysSMPL_DIR = str(args.mamma_sequence)
            selected.SysSMPL_ANNOTATION_DIR = str(args.mamma_sequence)
        else:
            selected.split = "train"
            selected.Harmony4D_DIR = str(args.harmony_root)
            selected.smplx_annotation_root = str(args.harmony_annotations)
            selected.person_mask_root = str(args.harmony_masks)
            selected.val_sequence_fraction = 0.0
        cfg.loss.smplx_model_dir = str(args.smplx_root)
    return cfg


def check_paths(args):
    required = [args.smplx_root]
    if args.source == "mamma":
        required.append(args.mamma_sequence)
    else:
        required.extend(
            [args.harmony_root, args.harmony_annotations, args.harmony_masks]
        )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Required directories do not exist: {missing}")
    if not (args.smplx_root / "neutral" / "model.pkl").is_file():
        raise FileNotFoundError(
            f"Expected {args.smplx_root / 'neutral' / 'model.pkl'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mamma_harmony4d_mask_dpt")
    parser.add_argument("--source", choices=("mamma", "harmony4d"), default="mamma")
    parser.add_argument("--mamma-sequence", type=Path, default=DEFAULT_MAMMA_SEQUENCE)
    parser.add_argument("--harmony-root", type=Path, default=DEFAULT_HARMONY_ROOT)
    parser.add_argument(
        "--harmony-annotations", type=Path, default=DEFAULT_HARMONY_ANNOTATIONS
    )
    parser.add_argument("--harmony-masks", type=Path, default=DEFAULT_HARMONY_MASKS)
    parser.add_argument("--smplx-root", type=Path, default=DEFAULT_SMPLX_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temporal-frames", type=int, default=3)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--max-people", type=int, default=6)
    parser.add_argument("--max-index-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--skip-model-forward", action="store_true")
    parser.add_argument("--projection-tolerance-px", type=float, default=2.0)
    parser.add_argument("--gt-core-loss-tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    for name in (
        "mamma_sequence",
        "harmony_root",
        "harmony_annotations",
        "harmony_masks",
        "smplx_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    check_paths(args)
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with initialize_config_dir(
        config_dir=str(TRAINING_DIR / "config"), version_base=None
    ):
        cfg = compose(config_name=args.config)
    cfg = configure_one_batch(cfg, args)
    OmegaConf.save(cfg, args.output / "config_resolved.yaml")
    set_smplx_model_root(args.smplx_root)

    dynamic_dataset = instantiate(cfg.data.train, _recursive_=False)
    dynamic_dataset.seed = args.seed
    raw_batch = next(iter(dynamic_dataset.get_loader(epoch=0)))
    processed_clip = process_batch_like_trainer(
        raw_batch, scale_by_extrinsics=bool(cfg.scale_by_extrinsics)
    )
    flat_batch = flatten_temporal_batch_for_framewise_model(processed_clip)

    layout = validate_layout(raw_batch, flat_batch)
    raw_reprojection, raw_by_frame, raw_projected = reprojection_metrics(raw_batch)
    normalized_reprojection, normalized_by_frame, normalized_projected = (
        reprojection_metrics(processed_clip)
    )
    overlay_paths = save_overlays(
        processed_clip, normalized_projected, args.output / "overlays"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Move one compact copy. The flattened tensors are reshaped views of the
    # clip tensors wherever possible, matching Trainer without duplicating all
    # dense depth/world-point targets on the GPU.
    processed_clip_device = move_to_device(
        model_loss_batch(processed_clip), device
    )
    flat_batch_device = flatten_temporal_batch_for_framewise_model(
        processed_clip_device
    )
    loss_module = instantiate(cfg.loss, _recursive_=False).to(device).eval()
    gt_predictions = make_gt_as_prediction(flat_batch_device, cfg)
    with torch.no_grad():
        gt_loss_dict = loss_module(gt_predictions, flat_batch_device)
    gt_losses = scalar_metrics(gt_loss_dict)
    temporal_loss_keys = {
        "loss_smpl_temporal_pose",
        "loss_smpl_temporal_beta",
        "loss_smpl_temporal_mesh_translate",
    }
    core_keys = (
        "loss_camera",
        "loss_T",
        "loss_R",
        "loss_FL",
        "loss_smpl_losses",
        "loss_mesh_translate",
        "loss_smpl_joints3d",
        "loss_smpl_vertices",
    )
    max_gt_core_loss = max(abs(gt_losses.get(key, 0.0)) for key in core_keys)
    gt_core_loss_pass = max_gt_core_loss <= args.gt_core_loss_tolerance
    del gt_predictions, gt_loss_dict
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_result = {"skipped": True}
    if not args.skip_model_forward:
        model_result = run_model_and_loss(
            cfg,
            processed_clip_device,
            flat_batch_device,
            loss_module,
            device,
            args.checkpoint,
        )
        model_result["skipped"] = False

    projection_pass = (
        normalized_reprojection["max_px"] is not None
        and normalized_reprojection["max_px"] <= args.projection_tolerance_px
    )
    model_pass = model_result.get("skipped") or (
        model_result["all_predictions_finite"]
        and model_result["all_losses_finite"]
    )
    result = {
        "passed": bool(
            layout["passed"]
            and projection_pass
            and gt_core_loss_pass
            and model_pass
        ),
        "config": args.config,
        "source": args.source,
        "sequence_names": raw_batch.get("seq_name"),
        "layout": layout,
        "raw_shapes": tensor_shapes(raw_batch),
        "processed_clip_shapes": tensor_shapes(processed_clip),
        "flattened_loss_shapes": tensor_shapes(flat_batch),
        "avg_scale": processed_clip["avg_scale"].cpu().tolist(),
        "raw_reprojection": raw_reprojection,
        "raw_reprojection_by_frame": raw_by_frame,
        "normalized_reprojection": normalized_reprojection,
        "normalized_reprojection_by_frame": normalized_by_frame,
        "projection_tolerance_px": args.projection_tolerance_px,
        "projection_pass": bool(projection_pass),
        "mask": mask_metrics(raw_batch),
        "gt_as_prediction": {
            "losses": gt_losses,
            "core_zero_keys": list(core_keys),
            "temporal_regularizer_keys_not_expected_to_be_zero": sorted(
                temporal_loss_keys
            ),
            "max_core_loss": max_gt_core_loss,
            "tolerance": args.gt_core_loss_tolerance,
            "passed": bool(gt_core_loss_pass),
        },
        "model_forward": model_result,
        "overlays": overlay_paths,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"[output] {args.output}")
    print(f"[result] {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
