"""SMPL pose/shape/joints/vertices + dense-landmark losses (split out of loss.py).

Body model / decode / gauge utils live in smpl_body.py; Hungarian matching in
smpl_matching.py; per-person mask loss in loss_mask.py.
"""
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from training.train_utils.general import check_and_fix_inf_nan

from training.smpl_body import *  # noqa: F401,F403
from training.smpl_body import (
    _TorchSMPLX,
    _decode_smpl_batch,
    _decode_smplx_batch,
    _normalize_gender_string,
    _resolve_batch_genders,
    _project_points_opencv,
    axis_angle_to_rotmat,
    rotmat_to_axis_angle,
    scale_joints_to_batch_gauge,
    normalize_joints_world_to_batch_gauge,
    subtract_pelvis,
    compute_gt_mesh_translate,
    compute_gt_mesh_rot,
)
from training.smpl_matching import apply_hungarian_matching, _binary_cross_entropy_prob
from training.loss_mask import compute_mask_loss


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss for presence/objectness logits.

    Normalizes by the number of positive slots, matching dense detection losses
    more closely than averaging over all positive and negative slots.
    `alpha` can optionally be used as the positive-class weight.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_weight = (1.0 - p_t).clamp(min=0.0).pow(gamma)

    if alpha is not None and alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_weight = focal_weight * alpha_t

    loss = focal_weight * bce
    num_pos = (targets > 0.5).to(dtype=loss.dtype).sum()
    if num_pos > 0:
        return loss.sum() / num_pos.clamp(min=1.0)
    return loss.sum()


def rotation_geodesic_angle(
    pred_rotmat: torch.Tensor,
    gt_rotmat: torch.Tensor,
) -> torch.Tensor:
    """Return the SO(3) geodesic angle in radians.

    ``atan2(sin(theta), cos(theta))`` retains a useful slope much closer to pi
    than the chordal/Frobenius pose loss used below.  The calculation is kept in
    fp32 because root flips are precisely where bf16 rounding is most harmful.
    """
    pred_rotmat = pred_rotmat.float()
    gt_rotmat = gt_rotmat.float()
    relative = gt_rotmat.transpose(-1, -2) @ pred_rotmat

    cos_theta = 0.5 * (
        relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    )
    cos_theta = cos_theta.clamp(-1.0, 1.0)

    # The norm of this vector is 2 * |sin(theta)|.
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sin_theta, cos_theta)



def smpl_losses_plus_from_axis_angle(
    pred_pose_aa: torch.Tensor,
    pred_beta: torch.Tensor,
    gt_pose_aa: torch.Tensor,
    gt_beta: torch.Tensor,
    pred_pose_aa_0: torch.Tensor | None = None,
    pose_weight: float = 1.0,
    beta_weight: float = 1.0,
    global_orient_weight: float = 0.0,
    init_w: float = 1.0,
    loss_type: str = "l1",
    has_smpl: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """TRAM-style SMPL+ loss but with axis-angle inputs.

    - GT only needs pose (axis-angle) + beta.
    - If pred_pose_aa_0 is provided, adds an "init" pose loss weighted by init_w.

    Shapes supported:
      pose: (B,72) or (B,S,72)
      beta: (B,10) or (B,S,10)
      has_smpl: (B,) or (B,S) boolean/0-1
    """

    def _ensure_BS(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    pred_pose_aa = _ensure_BS(pred_pose_aa)
    pred_beta = _ensure_BS(pred_beta)
    gt_pose_aa = _ensure_BS(gt_pose_aa)
    gt_beta = _ensure_BS(gt_beta)

    if pred_pose_aa_0 is None:
        pred_pose_aa_0 = pred_pose_aa
    else:
        pred_pose_aa_0 = _ensure_BS(pred_pose_aa_0)

    if has_smpl is None:
        has_smpl = torch.ones(pred_pose_aa.shape[:2], device=pred_pose_aa.device, dtype=pred_pose_aa.dtype)
    else:
        if has_smpl.dim() == 1:
            has_smpl = has_smpl[:, None]
        has_smpl = has_smpl.to(device=pred_pose_aa.device, dtype=pred_pose_aa.dtype)

    # rotmat regression (Frobenius) on 24 joints
    R_pred = axis_angle_to_rotmat(pred_pose_aa)
    R_gt = axis_angle_to_rotmat(gt_pose_aa)
    R_pred0 = axis_angle_to_rotmat(pred_pose_aa_0)

    pose_loss_final = ((R_pred - R_gt) ** 2).sum(dim=(-1, -2)).mean(dim=-1)  # (B,S)
    pose_loss_init = ((R_pred0 - R_gt) ** 2).sum(dim=(-1, -2)).mean(dim=-1)  # (B,S)

    # Supervise the cam0-frame root separately instead of letting it contribute
    # only 1/24 of the aggregate pose loss. Recompute these three rotations in
    # fp32 so the near-pi geodesic remains numerically meaningful under AMP.
    R_root_pred = axis_angle_to_rotmat(pred_pose_aa[..., :3].float()).squeeze(-3)
    R_root_gt = axis_angle_to_rotmat(gt_pose_aa[..., :3].float()).squeeze(-3)
    R_root_pred0 = axis_angle_to_rotmat(pred_pose_aa_0[..., :3].float()).squeeze(-3)
    global_orient_final = rotation_geodesic_angle(R_root_pred, R_root_gt)  # (B,S)
    global_orient_init = rotation_geodesic_angle(R_root_pred0, R_root_gt)  # (B,S)

    if loss_type == "l1":
        beta_loss = (pred_beta - gt_beta).abs().mean(dim=-1)  # (B,S)
    elif loss_type == "l2":
        beta_loss = ((pred_beta - gt_beta) ** 2).mean(dim=-1)
    else:
        raise ValueError(f"Unknown SMPL loss_type: {loss_type}")

    # apply has_smpl mask
    denom = has_smpl.sum().clamp(min=1.0)
    loss_pose_final = (pose_loss_final * has_smpl).sum() / denom
    loss_pose_init = (pose_loss_init * has_smpl).sum() / denom
    loss_beta = (beta_loss * has_smpl).sum() / denom
    loss_global_orient = (global_orient_final * has_smpl).sum() / denom
    loss_global_orient_init = (global_orient_init * has_smpl).sum() / denom

    total = (
        pose_weight * loss_pose_final
        + pose_weight * init_w * loss_pose_init
        + beta_weight * loss_beta
        + global_orient_weight * loss_global_orient
        + global_orient_weight * init_w * loss_global_orient_init
    )
    return total, {
        "loss_smpl_plus": total,
        "loss_smpl_plus_pose": loss_pose_final,
        "loss_smpl_plus_pose_init": loss_pose_init,
        "loss_smpl_plus_beta": loss_beta,
        "loss_smpl_global_orient": loss_global_orient,
        "loss_smpl_global_orient_init": loss_global_orient_init,
    }


def compute_temporal_smpl_smoothness(
    predictions,
    batch,
    *,
    pose_order: int = 2,
    beta_order: int = 1,
    mesh_translate_order: int = 2,
    loss_type: str = "l1",
    exclude_root: bool = True,
    root_rotation_order: int = 1,
    root_rotation_loss_type: str = "l1",
):
    """Temporal regularizers on Hungarian-matched, identity-ordered people.

    Inputs remain framewise ``[B*T,P,...]`` for compatibility with the existing
    loss. ``batch['temporal_shape'] == [B,T]`` is the sole opt-in marker.
    Missing people are removed with products of ``has_smpl`` over every frame in
    the finite-difference stencil. Local pose/beta/translation terms are
    prediction smoothness priors; the root term instead supervises predicted
    relative SO(3) motion against GT so that intentional turns are not suppressed.
    """
    pred_pose = predictions["smpl_pose"]
    zero = pred_pose.sum() * 0.0
    result = {
        "loss_smpl_temporal_pose": zero,
        "loss_smpl_temporal_beta": zero,
        "loss_smpl_temporal_mesh_translate": zero,
        "loss_smpl_temporal_root_rotation": zero,
    }
    temporal_shape = batch.get("temporal_shape")
    if temporal_shape is None:
        return result
    B, T = [int(x) for x in temporal_shape.reshape(-1)[:2].tolist()]
    if B * T != pred_pose.shape[0]:
        raise ValueError(
            f"Temporal shape {(B, T)} does not match predictions batch {pred_pose.shape[0]}"
        )

    P = min(pred_pose.shape[1], batch["has_smpl"].shape[1])
    valid = batch["has_smpl"][:, :P].to(pred_pose.dtype).reshape(B, T, P)

    def _difference(x, order):
        if order == 1:
            if T < 2:
                return None, None
            return x[:, 1:] - x[:, :-1], valid[:, 1:] * valid[:, :-1]
        if order == 2:
            if T < 3:
                return None, None
            return (
                x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2],
                valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2],
            )
        raise ValueError(f"Unsupported temporal order: {order}")

    def _reduce(diff, mask, feature_dims):
        if diff is None:
            return zero
        if loss_type == "l1":
            value = diff.abs()
        elif loss_type == "l2":
            value = diff.square()
        else:
            raise ValueError(f"Unsupported temporal loss type: {loss_type}")
        value = value.mean(dim=feature_dims)
        mask = mask.to(value.dtype)
        return (value * mask).sum() / mask.sum().clamp(min=1.0)

    pose = pred_pose[:, :P, 3 if exclude_root else 0 :72].reshape(B, T, P, -1)
    pose_rot = axis_angle_to_rotmat(pose)
    pose_diff, pose_mask = _difference(pose_rot, int(pose_order))
    result["loss_smpl_temporal_pose"] = _reduce(
        pose_diff, pose_mask, feature_dims=(-1, -2, -3)
    )

    # Supervise root *motion* rather than forcing the absolute cam0-frame root
    # orientation to be constant.  Comparing relative SO(3) rotations against
    # GT preserves genuine turns while penalizing prediction-only angular jumps.
    gt_pose = batch.get("smpl_pose")
    if gt_pose is not None:
        pred_root = pred_pose[:, :P, :3].reshape(B, T, P, 3)
        gt_root = gt_pose[:, :P, :3].reshape(B, T, P, 3).to(
            device=pred_root.device, dtype=pred_root.dtype
        )
        pred_root_rot = axis_angle_to_rotmat(pred_root).squeeze(-3)
        gt_root_rot = axis_angle_to_rotmat(gt_root).squeeze(-3)
        pred_velocity = (
            pred_root_rot[:, :-1].transpose(-1, -2) @ pred_root_rot[:, 1:]
        )
        gt_velocity = (
            gt_root_rot[:, :-1].transpose(-1, -2) @ gt_root_rot[:, 1:]
        )
        root_mask = valid[:, :-1] * valid[:, 1:]

        if int(root_rotation_order) == 1:
            pred_motion = pred_velocity
            gt_motion = gt_velocity
        elif int(root_rotation_order) == 2:
            if T < 3:
                pred_motion = gt_motion = None
            else:
                pred_motion = (
                    pred_velocity[:, :-1].transpose(-1, -2)
                    @ pred_velocity[:, 1:]
                )
                gt_motion = (
                    gt_velocity[:, :-1].transpose(-1, -2)
                    @ gt_velocity[:, 1:]
                )
                root_mask = root_mask[:, :-1] * root_mask[:, 1:]
        else:
            raise ValueError(
                f"Unsupported temporal root rotation order: {root_rotation_order}"
            )

        if pred_motion is not None:
            root_angle = rotation_geodesic_angle(pred_motion, gt_motion)
            if root_rotation_loss_type == "l1":
                root_value = root_angle
            elif root_rotation_loss_type == "l2":
                root_value = root_angle.square()
            else:
                raise ValueError(
                    "Unsupported temporal root rotation loss type: "
                    f"{root_rotation_loss_type}"
                )
            root_mask = root_mask.to(root_value.dtype)
            result["loss_smpl_temporal_root_rotation"] = (
                (root_value * root_mask).sum()
                / root_mask.sum().clamp(min=1.0)
            )

    pred_beta = predictions.get("smpl_beta")
    if pred_beta is not None:
        beta = pred_beta[:, :P].reshape(B, T, P, -1)
        beta_diff, beta_mask = _difference(beta, int(beta_order))
        result["loss_smpl_temporal_beta"] = _reduce(
            beta_diff, beta_mask, feature_dims=-1
        )

    pred_translate = predictions.get("mesh_translate")
    if pred_translate is not None:
        translate = pred_translate[:, :P].reshape(B, T, P, 3)
        trans_diff, trans_mask = _difference(
            translate, int(mesh_translate_order)
        )
        result["loss_smpl_temporal_mesh_translate"] = _reduce(
            trans_diff, trans_mask, feature_dims=-1
        )
    return result



def compute_smpl_loss(
    predictions,
    batch,
    loss_type: str = "l1",
    loss_type_joints2d: str | None = None,
    loss_type_joints3d: str | None = None,
    weight_pose: float = 1.0,
    weight_beta: float = 0.1,
    weight_global_orient: float = 0.0,
    weight_trans: float = 0.0,
    weight_mesh_translate: float = 0.0,
    weight_presence: float = 1.0,
    weight_joints2d: float = 0.0,
    weight_joints3d: float = 0.0,
    weight_vertices: float = 0.0,
    use_temporal_training: bool = False,
    weight_temporal_pose: float = 0.0,
    weight_temporal_beta: float = 0.0,
    weight_temporal_mesh_translate: float = 0.0,
    weight_temporal_root_rotation: float = 0.0,
    temporal_pose_order: int = 2,
    temporal_beta_order: int = 1,
    temporal_mesh_translate_order: int = 2,
    temporal_loss_type: str = "l1",
    temporal_exclude_root: bool = True,
    temporal_root_rotation_order: int = 1,
    temporal_root_rotation_loss_type: str = "l1",
    use_gt: bool = False,
    # Standalone: use the GT (cam0-normalized) camera ONLY for the joints2d
    # reprojection loss, WITHOUT the full use_gt swap of pose/beta/trans/mesh_translate.
    joints2d_use_gt_camera: bool = False,
    normalize_cam: bool = True,
    use_hungarian: bool = False,
    hungarian_cost_pose_weight: float = 1.0,
    hungarian_cost_beta_weight: float = 0.1,
    hungarian_cost_trans_weight: float = 0.0,
    hungarian_cost_mesh_trans_weight: float = 0.0,
    hungarian_cost_presence_weight: float = 0.0,
    hungarian_cost_mask_weight: float = 0.0,
    hungarian_mask_cost_grid: int = 32,
    use_mamma: bool = False,
    **kwargs,
):
    """
    SMPL supervision:
    - predictions["smpl_pose"]: (B, 72), axis-angle
    - predictions["smpl_beta"]: (B, 10)
    - predictions["smpl_trans"]: (B, S, 1, 3)
    - predictions["smpl_presence_logits"]: optional multi-person slot logits (B, P)
    - batch["smpl_pose"]:      (B, 72)
    - batch["smpl_beta"]:      (B, 10)
    - batch["smpl_trans"]: (B, S, 1, 3)

    joint confidence masks (only joints with confidence==1 contribute):
    - batch["smpl_joints2d_confidence"]: (B, S, J) / (B, J) / (S, J)

    """
    # Multi-person samples use shape [B, P, ...] for SMPL params and
    # [B, S, P, ...] for per-view joints.  Flatten people into the batch axis so
    # the existing single-person loss path can supervise every person jointly.
    matching_cost_metrics = {
        "presence_cost": predictions["smpl_pose"].new_zeros(()),
        "mask_cost": predictions["smpl_pose"].new_zeros(()),
    }
    if "mesh_translate" in predictions and "mesh_translate" not in batch:
        batch = dict(batch)
        # fp32: this derives the GT root (mesh_translate) via SMPL decode + gauge
        # normalize; under bf16 autocast its value is rounded, which then shifts every
        # projected joint and inflates loss_smpl_joints2d. Keep it fp32.
        with torch.cuda.amp.autocast(enabled=False):
            batch["mesh_translate"] = compute_gt_mesh_translate(
                batch,
                normalize_cam=normalize_cam,
                use_mamma=use_mamma,
            )

    # mesh_rot mode: the head predicts smpl_pose[:3] as the cam0-frame mesh root
    # rotation. Replace the GT global_orient (world frame) with the GT mesh_rot so
    # that Hungarian matching AND the pose loss compare like-for-like (cam0 frame).
    # Must run BEFORE matching and BEFORE overwriting any pose value.
    use_mesh_rot = predictions.get("mesh_rot", None) is not None
    if use_mesh_rot:
        if "raw_extrinsics" not in batch:
            raise KeyError("mesh_rot mode requires batch['raw_extrinsics'] (set normalize_cam=True)")
        batch = dict(batch)
        with torch.cuda.amp.autocast(enabled=False):
            gt_mesh_rot = compute_gt_mesh_rot(batch)  # (B,P,3) axis-angle, cam0 frame
        gt_pose_new = batch["smpl_pose"].clone()
        gt_pose_new[..., :3] = gt_mesh_rot.to(gt_pose_new.dtype)
        batch["smpl_pose"] = gt_pose_new
        batch["mesh_rot"] = gt_mesh_rot

    has_people_axis = "num_people" in batch or any(
        key in batch and batch[key].dim() >= 5
        for key in ("smpl_joints2d", "smpl_joints3d_world", "smpl_joints2d_confidence")
    )
    temporal_losses = {
        "loss_smpl_temporal_pose": predictions["smpl_pose"].sum() * 0.0,
        "loss_smpl_temporal_beta": predictions["smpl_pose"].sum() * 0.0,
        "loss_smpl_temporal_mesh_translate": predictions["smpl_pose"].sum() * 0.0,
        "loss_smpl_temporal_root_rotation": predictions["smpl_pose"].sum() * 0.0,
    }
    if predictions["smpl_pose"].dim() == 3 and has_people_axis:
        if use_hungarian:
            predictions, batch, matching_cost_metrics = apply_hungarian_matching(
                predictions,
                batch,
                cost_pose_weight=hungarian_cost_pose_weight,
                cost_beta_weight=hungarian_cost_beta_weight,
                cost_trans_weight=hungarian_cost_trans_weight,
                cost_mesh_trans_weight=hungarian_cost_mesh_trans_weight,
                cost_presence_weight=hungarian_cost_presence_weight,
                cost_mask_weight=hungarian_cost_mask_weight,
                mask_cost_grid=hungarian_mask_cost_grid,
                return_cost_metrics=True,
                use_mamma=use_mamma,
            )

        if use_temporal_training:
            if not use_hungarian:
                raise ValueError(
                    "Temporal multi-person loss requires use_hungarian=True so each "
                    "trajectory follows a fixed GT identity."
                )
            temporal_losses = compute_temporal_smpl_smoothness(
                predictions,
                batch,
                pose_order=temporal_pose_order,
                beta_order=temporal_beta_order,
                mesh_translate_order=temporal_mesh_translate_order,
                loss_type=temporal_loss_type,
                exclude_root=temporal_exclude_root,
                root_rotation_order=temporal_root_rotation_order,
                root_rotation_loss_type=temporal_root_rotation_loss_type,
            )

        B_people, P_people = predictions["smpl_pose"].shape[:2]

        def _pad_people_param(tensor, fill_value: float = 0.0):
            if tensor is None:
                return None
            if tensor.dim() >= 3 and tensor.shape[0] == B_people and tensor.shape[1] < P_people:
                pad_shape = list(tensor.shape)
                pad_shape[1] = P_people - tensor.shape[1]
                pad = torch.full(
                    pad_shape,
                    fill_value,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                return torch.cat([tensor, pad], dim=1)
            return tensor

        def _pad_people_view_tensor(tensor, fill_value: float = 0.0):
            if tensor is None:
                return None
            if tensor.dim() >= 4 and tensor.shape[0] == B_people and tensor.shape[2] < P_people:
                pad_shape = list(tensor.shape)
                pad_shape[2] = P_people - tensor.shape[2]
                pad = torch.full(
                    pad_shape,
                    fill_value,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                return torch.cat([tensor, pad], dim=2)
            return tensor

        batch = dict(batch)
        for key in ("smpl_pose", "smpl_beta", "smpl_trans", "mesh_translate"):
            if key in batch:
                batch[key] = _pad_people_param(batch[key])
        for key in (
            "smpl_joints2d", "smpl_joints3d_world", "smpl_joints2d_confidence",
            "smpl_landmarks2d", "smpl_landmarks2d_visibility", "person_mask",
            "smpl_contact", "smpl_floor_contact",
        ):
            if key in batch:
                batch[key] = _pad_people_view_tensor(batch[key])
        if "smpl_gender" in batch and batch["smpl_gender"].dim() >= 2 and batch["smpl_gender"].shape[1] < P_people:
            gender = batch["smpl_gender"]
            pad = torch.full(
                (B_people, P_people - gender.shape[1]),
                2,
                device=gender.device,
                dtype=gender.dtype,
            )
            batch["smpl_gender"] = torch.cat([gender, pad], dim=1)
        if "has_smpl" in batch and batch["has_smpl"].dim() >= 2 and batch["has_smpl"].shape[1] < P_people:
            has_smpl = batch["has_smpl"]
            pad = torch.zeros(
                (B_people, P_people - has_smpl.shape[1]),
                device=has_smpl.device,
                dtype=has_smpl.dtype,
            )
            batch["has_smpl"] = torch.cat([has_smpl, pad], dim=1)

        def _flatten_people_param(tensor):
            if tensor is None:
                return None
            if tensor.dim() == 3 and tensor.shape[:2] == (B_people, P_people):
                return tensor.reshape(B_people * P_people, *tensor.shape[2:])
            return tensor

        def _flatten_people_view_tensor(tensor):
            if tensor is None:
                return None
            if tensor.dim() >= 4 and tensor.shape[0] == B_people and tensor.shape[2] == P_people:
                permute_order = [0, 2, 1] + list(range(3, tensor.dim()))
                tensor = tensor.permute(*permute_order).contiguous()
                return tensor.reshape(B_people * P_people, *tensor.shape[2:])
            return tensor

        def _repeat_per_person_view_tensor(tensor):
            if tensor is None:
                return None
            if tensor.shape[0] != B_people:
                return tensor
            expanded = tensor[:, None].expand(
                B_people,
                P_people,
                *tensor.shape[1:],
            )
            return expanded.reshape(B_people * P_people, *tensor.shape[1:])

        predictions = dict(predictions)
        for key in ("smpl_pose", "smpl_beta", "smpl_trans", "mesh_translate", "pred_pose_0", "smpl_pose_0", "smpl_pose_init"):
            if key in predictions:
                predictions[key] = _flatten_people_param(predictions[key])
        if "smpl_presence_logits" in predictions and predictions["smpl_presence_logits"] is not None:
            presence_logits = predictions["smpl_presence_logits"]
            if presence_logits.dim() == 2 and presence_logits.shape == (B_people, P_people):
                predictions["smpl_presence_logits"] = presence_logits.reshape(B_people * P_people)
        if "pose_enc_list" in predictions:
            predictions["pose_enc_list"] = [
                _repeat_per_person_view_tensor(pose_enc)
                for pose_enc in predictions["pose_enc_list"]
            ]
        # dense-landmark / mask head outputs -> (B*P, S, ...)
        for key in (
            "smpl_landmarks2d",
            "smpl_landmarks_logvar",
            "smpl_landmarks_visibility_logits",
            "smpl_contact_logits",
            "smpl_floor_contact_logits",
            "person_mask_logits",
        ):
            if key in predictions and predictions[key] is not None:
                predictions[key] = _flatten_people_view_tensor(predictions[key])

        batch = dict(batch)
        for key in ("smpl_pose", "smpl_beta", "smpl_trans", "mesh_translate"):
            if key in batch:
                batch[key] = _flatten_people_param(batch[key])
        for key in ("smpl_joints2d", "smpl_joints3d_world",
                    "smpl_landmarks2d", "smpl_landmarks2d_visibility", "person_mask",
                    "smpl_contact", "smpl_floor_contact"):
            if key in batch:
                batch[key] = _flatten_people_view_tensor(batch[key])
        if "smpl_joints2d_confidence" in batch:
            batch["smpl_joints2d_confidence"] = _flatten_people_view_tensor(
                batch["smpl_joints2d_confidence"]
            )
        for key in ("images", "extrinsics", "intrinsics", "raw_extrinsics", "point_masks"):
            if key in batch:
                batch[key] = _repeat_per_person_view_tensor(batch[key])
        if "avg_scale" in batch and batch["avg_scale"].shape[0] == B_people:
            batch["avg_scale"] = batch["avg_scale"][:, None].expand(B_people, P_people).reshape(-1)
        if "smpl_gender" in batch and batch["smpl_gender"].dim() >= 2:
            batch["smpl_gender"] = batch["smpl_gender"].reshape(B_people * P_people)
        if "has_smpl" in batch and batch["has_smpl"].dim() >= 2:
            batch["has_smpl"] = batch["has_smpl"].reshape(B_people * P_people)

    pred_pose = predictions["smpl_pose"]   # [B, 72]
    pred_beta = predictions["smpl_beta"]   # [B, 10]

    gt_pose = batch["smpl_pose"]           # [B, 72]
    gt_beta = batch["smpl_beta"]           # [B, 10]

    # SMPL translation losses
    pred_trans = predictions.get("smpl_trans", None)
    gt_trans = batch.get("smpl_trans", None)
    pred_mesh_translate = predictions.get("mesh_translate", None)
    gt_mesh_translate = batch.get("mesh_translate", None)

    if use_gt:
        pred_pose = gt_pose
        pred_beta = gt_beta
        pred_trans = gt_trans
        pred_mesh_translate = gt_mesh_translate

    # ---- 保證都有 time 維度 S ----
    if pred_pose.dim() == 2:   # [B,72] -> [B,1,72]
        pred_pose = pred_pose.unsqueeze(1)
    if pred_beta.dim() == 2:   # [B,10] -> [B,1,10]
        pred_beta = pred_beta.unsqueeze(1)

    if gt_pose.dim() == 2:     # [B,72] -> [B,1,72]
        gt_pose = gt_pose.unsqueeze(1)
    if gt_beta.dim() == 2:     # [B,10] -> [B,1,10]
        gt_beta = gt_beta.unsqueeze(1)

    def _align_trans_to_pose(trans, pose, name: str):
        if trans is None:
            raise ValueError(f"{name} is required")

        # Accept [B,3], [B,T,3], [B,T,1,3], and occasional loader/model variants.
        if trans.dim() == 2:
            trans = trans.unsqueeze(1)
        elif trans.dim() == 4:
            if trans.shape[-2] == 1:
                trans = trans.squeeze(-2)
            else:
                trans = trans[:, :, 0, :]
        elif trans.dim() != 3:
            raise ValueError(f"Unsupported {name} shape: {tuple(trans.shape)}")

        if trans.shape[-1] != 3:
            raise ValueError(f"{name} last dimension must be 3, got {tuple(trans.shape)}")

        target_t = pose.shape[1]
        if trans.shape[1] == target_t:
            return trans
        if trans.shape[1] == 1:
            return trans.expand(-1, target_t, -1)
        if target_t == 1:
            return trans[:, :1, :]

        raise ValueError(
            f"Cannot align {name} shape {tuple(trans.shape)} to pose shape {tuple(pose.shape)}"
        )

    if pred_trans is not None and gt_trans is not None:
        pred_trans = _align_trans_to_pose(pred_trans, pred_pose, "pred_trans")
        gt_trans = _align_trans_to_pose(gt_trans, gt_pose, "gt_trans")
    if pred_mesh_translate is not None and gt_mesh_translate is not None:
        pred_mesh_translate = _align_trans_to_pose(pred_mesh_translate, pred_pose, "pred_mesh_translate")
        gt_mesh_translate = _align_trans_to_pose(gt_mesh_translate, gt_pose, "gt_mesh_translate")

    smpl_decode_uses_mesh_translate = pred_mesh_translate is not None

    # ---- Keep ONLY: keypoint_loss (2D), keypoint_3d_loss (3D), smpl_losses (pose/beta), vertice_loss ----
    loss_joints2d = (pred_pose * 0.0).mean()
    loss_joints3d = (pred_pose * 0.0).mean()
    loss_vertices = (pred_pose * 0.0).mean()
    loss_smpl_losses = (pred_pose * 0.0).mean()
    loss_global_orient = (pred_pose * 0.0).mean()
    loss_global_orient_init = (pred_pose * 0.0).mean()
    loss_trans = (pred_pose * 0.0).mean()
    loss_mesh_translate = (pred_pose * 0.0).mean()
    loss_presence = (pred_pose * 0.0).mean()
    smpl_presence_positive_prob_mean = (pred_pose * 0.0).mean()
    smpl_presence_empty_prob_mean = (pred_pose * 0.0).mean()
    # 投影爆掉偵測:in-frame 的 normalized 2D 應在 [-1,1];若 mesh_translate 把人放到太靠近
    # 相機(深度 z→0),投影 Jacobian ~ f/z 爆大,座標 >> 1 → 這個值會飆高。
    smpl_joints2d_max_abs = (pred_pose * 0.0).mean()

    # NOTE: don't use Python `or` with tensors; it triggers boolean evaluation.
    pred_pose_0 = None
    if "pred_pose_0" in predictions and predictions["pred_pose_0"] is not None:
        pred_pose_0 = predictions["pred_pose_0"]
    elif "smpl_pose_0" in predictions and predictions["smpl_pose_0"] is not None:
        pred_pose_0 = predictions["smpl_pose_0"]
    elif "smpl_pose_init" in predictions and predictions["smpl_pose_init"] is not None:
        pred_pose_0 = predictions["smpl_pose_init"]
    init_w = float(kwargs.get("init_w", 1.0))
    has_smpl = batch.get("has_smpl", None)
    presence_loss_type = str(kwargs.get("presence_loss_type", "bce")).lower()
    presence_focal_alpha = kwargs.get("presence_focal_alpha", None)
    if presence_focal_alpha is not None:
        presence_focal_alpha = float(presence_focal_alpha)
    presence_focal_gamma = float(kwargs.get("presence_focal_gamma", 2.0))

    presence_logits = predictions.get("smpl_presence_logits", None)
    if weight_presence > 0.0 and presence_logits is not None and has_smpl is not None:
        presence_logits = presence_logits.to(dtype=pred_pose.dtype)
        presence_target = has_smpl.to(device=presence_logits.device, dtype=presence_logits.dtype)
        if presence_target.shape != presence_logits.shape:
            if presence_target.numel() == presence_logits.numel():
                presence_target = presence_target.reshape_as(presence_logits)
            elif presence_logits.dim() == 2 and presence_logits.shape[-1] == 1 and presence_target.shape == presence_logits.shape[:1]:
                presence_target = presence_target[:, None]
            else:
                raise ValueError(
                    f"Cannot align has_smpl shape {tuple(has_smpl.shape)} to smpl_presence_logits shape {tuple(presence_logits.shape)}"
                )
        presence_probs = torch.sigmoid(presence_logits)
        positive_mask = presence_target > 0.5
        empty_mask = ~positive_mask
        if positive_mask.any():
            smpl_presence_positive_prob_mean = presence_probs[positive_mask].mean()
        if empty_mask.any():
            smpl_presence_empty_prob_mean = presence_probs[empty_mask].mean()
        if presence_loss_type == "bce":
            loss_presence = F.binary_cross_entropy_with_logits(
                presence_logits,
                presence_target,
            )
        elif presence_loss_type == "focal":
            loss_presence = binary_focal_loss_with_logits(
                presence_logits,
                presence_target,
                alpha=presence_focal_alpha,
                gamma=presence_focal_gamma,
            )
        else:
            raise ValueError(f"Unknown presence_loss_type: {presence_loss_type}")

    #* SMPL losses (pose/beta) with optional init pose (pred_pose_0)
    if weight_pose > 0.0 or weight_beta > 0.0 or weight_global_orient > 0.0:
        pred_pose_for_smpl = pred_pose[..., :72]
        gt_pose_for_smpl = gt_pose[..., :72]
        pred_pose_0_for_smpl = pred_pose_0[..., :72] if pred_pose_0 is not None else None

        smpl_plus_total, _smpl_plus_dict = smpl_losses_plus_from_axis_angle(
            pred_pose_aa=pred_pose_for_smpl,
            pred_beta=pred_beta,
            gt_pose_aa=gt_pose_for_smpl,
            gt_beta=gt_beta,
            pred_pose_aa_0=pred_pose_0_for_smpl,
            pose_weight=weight_pose,
            beta_weight=weight_beta,
            global_orient_weight=weight_global_orient,
            init_w=init_w,
            loss_type=loss_type,
            has_smpl=has_smpl,
        )
        loss_smpl_losses = smpl_plus_total
        loss_global_orient = _smpl_plus_dict["loss_smpl_global_orient"]
        loss_global_orient_init = _smpl_plus_dict["loss_smpl_global_orient_init"]

    # Shared SMPL decode (only when needed)
    need_smpl_decode = (
        (weight_joints3d > 0.0 and "smpl_joints3d_world" in batch)
        or (weight_joints2d > 0.0 and "smpl_joints2d" in batch)
        or (weight_vertices > 0.0)
    )
    pred_joints_world = None
    pred_vertices_smpl = None
    if need_smpl_decode:
        B, T = pred_pose.shape[:2]
        pred_pose_flat = pred_pose.reshape(B * T, -1)
        pred_beta_flat = pred_beta.reshape(B * T, -1)
        if smpl_decode_uses_mesh_translate:
            pred_trans_flat = torch.zeros(B * T, 3, device=pred_pose.device, dtype=pred_pose.dtype)
        else:
            pred_trans_flat = pred_trans.reshape(B * T, -1)

        genders = _resolve_batch_genders(batch, B)
        if len(genders) == B:
            genders = [g for g in genders for _ in range(T)]

        # Decode SMPL in fp32: under bf16 autocast the LBS/joint regression matmuls
        # round to bf16, and that error (scaled by scene magnitude at projection time)
        # is what inflates loss_smpl_joints2d. Keep the whole geometry chain fp32.
        with torch.cuda.amp.autocast(enabled=False):
            decode_out = _decode_smpl_batch(
                pose_aa=pred_pose_flat.float(),
                betas=pred_beta_flat.float(),
                trans=pred_trans_flat.float(),
                genders=genders,
                use_mamma=use_mamma,
            )
        pred_joints_world_flat, pred_vertices_smpl_flat = decode_out
        if pred_joints_world_flat is not None:
            pred_joints_world = pred_joints_world_flat.reshape(B, T, pred_joints_world_flat.shape[1], 3)
        if pred_vertices_smpl_flat is not None:
            pred_vertices_smpl = pred_vertices_smpl_flat.reshape(
                B, T, pred_vertices_smpl_flat.shape[1], 3
            )
        if smpl_decode_uses_mesh_translate:
            if not normalize_cam:
                raise ValueError("mesh_translate predictions require normalize_cam=True")
            with torch.cuda.amp.autocast(enabled=False):
                pred_mesh_translate_aligned = pred_mesh_translate.to(device=pred_pose.device).float()
                raw_extr_f = batch["raw_extrinsics"].float()
                avg_scale_f = batch["avg_scale"].float()
                # mesh_rot mode: smpl_pose[:3] is the cam0-frame mesh_rot, so the
                # decoded body is already oriented in cam0 -> scale-only gauge (no R0
                # rotation). Otherwise rotate world->cam0 then scale.
                if pred_joints_world is not None:
                    if use_mesh_rot:
                        pred_joints_world = scale_joints_to_batch_gauge(pred_joints_world.float(), avg_scale_f)
                    else:
                        pred_joints_world = normalize_joints_world_to_batch_gauge(
                            pred_joints_world.float(),
                            raw_extr_f,
                            avg_scale_f,
                        )
                    # detach root anchor:切掉「所有關節的梯度疊加到 decode 出的 root」這條
                    # 放大路徑(dL/d(jw_root) 會含 -Σ_i dL/d(final_i)),它讓 Grad/smpl 隨訓練
                    # 爆增、灌爆共享 aggregator 而拖垮相機。forward 值不變(root 仍 anchor 到
                    # mesh_translate),只移除這條梯度放大。
                    joint_offset = pred_mesh_translate_aligned - pred_joints_world[..., 0, :].detach()
                    pred_joints_world = pred_joints_world + joint_offset.unsqueeze(-2)
                if pred_vertices_smpl is not None:
                    if use_mesh_rot:
                        pred_vertices_smpl = scale_joints_to_batch_gauge(pred_vertices_smpl.float(), avg_scale_f)
                    else:
                        pred_vertices_smpl = normalize_joints_world_to_batch_gauge(
                            pred_vertices_smpl.float(),
                            raw_extr_f,
                            avg_scale_f,
                        )
                    pred_vertices_smpl = pred_vertices_smpl + joint_offset.unsqueeze(-2)
    
    if weight_trans > 0.0 and pred_trans is not None and gt_trans is not None:
        if loss_type == "l1":
            diff = (pred_trans - gt_trans).abs().mean(dim=-1)
        elif loss_type == "l2":
            diff = ((pred_trans - gt_trans) ** 2).mean(dim=-1)
        else:
            raise ValueError(loss_type)

        if has_smpl is not None:
            trans_mask = has_smpl.to(device=diff.device, dtype=diff.dtype)
            if trans_mask.dim() == 1:
                trans_mask = trans_mask[:, None].expand_as(diff)
            loss_trans = (diff * trans_mask).sum() / trans_mask.sum().clamp(min=1.0)
        else:
            loss_trans = diff.mean()

    if weight_mesh_translate > 0.0 and pred_mesh_translate is not None and gt_mesh_translate is not None:
        if loss_type == "l1":
            diff = (pred_mesh_translate - gt_mesh_translate).abs().mean(dim=-1)
        elif loss_type == "l2":
            diff = ((pred_mesh_translate - gt_mesh_translate) ** 2).mean(dim=-1)
        else:
            raise ValueError(loss_type)

        if has_smpl is not None:
            mesh_translate_mask = has_smpl.to(device=diff.device, dtype=diff.dtype)
            if mesh_translate_mask.dim() == 1:
                mesh_translate_mask = mesh_translate_mask[:, None].expand_as(diff)
            loss_mesh_translate = (
                diff * mesh_translate_mask
            ).sum() / mesh_translate_mask.sum().clamp(min=1.0)
        else:
            loss_mesh_translate = diff.mean()
    
    #* pred 3d joints -> normalize by gt -> 3d joints loss (GT joints are already normalized in _process_batch)
    if weight_joints3d > 0.0 and "smpl_joints3d_world" in batch:
        gt_joints3d_world = batch["smpl_joints3d_world"]
        joints3d_loss_type = loss_type_joints3d or loss_type
        loss_joints3d = compute_smpl_3d_joint_loss(
            pred_joints_world[..., :24, :],
            gt_joints3d_world[..., :24, :],
            batch=batch,
            loss_type=joints3d_loss_type,
            pelvis_center=True,
            pelvis_ids=(1, 2),
            normalize_cam=False if smpl_decode_uses_mesh_translate else normalize_cam,
        )
    
    #* 2D keypoint reprojection loss
    # Requirement: use Unity-space AND gauge-normalized 3D joints, projected with camera-head predicted extrinsics/intrinsics.
    if weight_joints2d > 0.0 and "smpl_joints2d" in batch:
        if "pose_enc_list" not in predictions:
            raise KeyError("pose_enc_list not found in predictions; enable camera head to use joints2d loss")

        # NOTE: the reprojection below (gauge normalize -> FoV pose-encoding
        # round-trip -> projection) is run in fp32 even under bf16 autocast. In bf16
        # these matmuls accumulate rounding error that scales with the scene's
        # coordinate magnitude, which inflates loss_smpl_joints2d and makes it
        # unstable / not comparable across datasets. fp32 keeps the geometry exact.
        if normalize_cam and not smpl_decode_uses_mesh_translate:
            # Normalize pred joints to the same batch gauge as camera normalization.
            with torch.cuda.amp.autocast(enabled=False):
                pred_joints_norm = normalize_joints_world_to_batch_gauge(
                    pred_joints_world.float(),
                    batch["raw_extrinsics"].float(),
                    batch["avg_scale"].float(),
                )
        else:
            pred_joints_norm = pred_joints_world.float()

        # print(f"[CGV LOG] pred_joints_norm shape : {pred_joints_norm.shape}")

        pred_pose_enc = predictions["pose_enc_list"][-1]  # (B,S,9)
        # Optional: stop gradients from joints2d reprojection loss to camera head.
        # This keeps joints2d supervising SMPL (pose/beta) while not updating pose_enc.

        pred_pose_enc = pred_pose_enc.detach()
        image_hw = batch["images"].shape[-2:]
        pose_encoding_type = kwargs.get("pose_encoding_type", "absT_quaR_FoV")

        with torch.cuda.amp.autocast(enabled=False):
            # Project with the GT (cam0-normalized) camera when either the global
            # use_gt sanity mode is on, OR the standalone joints2d_use_gt_camera flag
            # is set (which ONLY swaps the camera here -- pose/beta/trans/mesh_translate
            # stay the model's predictions, so joints3d/vertices/mesh_translate still
            # supervise the model normally).
            if use_gt or joints2d_use_gt_camera:
                gt_extrinsics = batch['extrinsics'].float()
                gt_intrinsics = batch['intrinsics'].float()
                gt_pose_encoding = extri_intri_to_pose_encoding(
                    gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
                )
                pred_pose_enc = gt_pose_encoding

            pred_extr, pred_intr = pose_encoding_to_extri_intri(
                pred_pose_enc.float(),
                image_size_hw=image_hw,
                pose_encoding_type=pose_encoding_type,
                build_intrinsics=True,
            )

        B, S = pred_pose_enc.shape[:2]
        if pred_joints_norm.dim() == 4:
            temporal_frames = pred_joints_norm.shape[1]
            views_per_frame = int(batch.get("views_per_frame", torch.tensor([S], device=pred_pose_enc.device))[0].item())
            if temporal_frames > 1 and S == temporal_frames * views_per_frame:
                points_world = pred_joints_norm[:, :, None, :, :].expand(
                    B, temporal_frames, views_per_frame, pred_joints_norm.shape[2], 3
                )
                points_world = points_world.reshape(B, S, pred_joints_norm.shape[2], 3)
            else:
                points_world = pred_joints_norm[:, :1, :, :].expand(B, S, pred_joints_norm.shape[2], 3)
        else:
            points_world = pred_joints_norm[:, None, :, :].expand(B, S, pred_joints_norm.shape[1], 3)
        # Depth clamp for projection: in the batch gauge the scene is ~O(1) (divided by
        # avg_scale), so a valid in-frame person has depth z~O(1). When mesh_translate puts
        # someone at z→0, u=fx*X/z explodes (smpl_joints2d_max_abs → 1e7) and floods the
        # shared aggregator. Clamping z to a sane positive min bounds the projection.
        joints2d_depth_min = float(kwargs.get("joints2d_depth_min", 1e-6))
        with torch.cuda.amp.autocast(enabled=False):
            pred_joints2d = _project_points_opencv(
                points_world.float(), pred_extr.float(), pred_intr.float(), eps=joints2d_depth_min
            )

        gt_joints2d = batch["smpl_joints2d"].to(device=pred_joints2d.device, dtype=pred_joints2d.dtype)
        if gt_joints2d.dim() == 2:
            gt_joints2d = gt_joints2d.unsqueeze(0).unsqueeze(0)
        elif gt_joints2d.dim() == 3:
            gt_joints2d = gt_joints2d.unsqueeze(1)
        elif gt_joints2d.dim() != 4:
            raise ValueError(f"Unsupported smpl_joints2d shape: {tuple(gt_joints2d.shape)}")
        
        # logging.info(f"avg_scale: {batch['avg_scale'].detach().cpu().numpy()}")
        # logging.info(f"loss gt_extrinsics: {batch['extrinsics'].detach().cpu().numpy()}")
        # logging.info(f"loss gt_intrinsics: {batch['intrinsics'].detach().cpu().numpy()}")
        # logging.info(f"loss smpl_joints3d_world: {batch['smpl_joints3d_world'].detach().cpu().numpy()}")
        # logging.info(f"loss smpl_joints2d: {batch['smpl_joints2d']}")
        # logging.info(f"loss points_world: {points_world.detach().cpu().numpy()}")
        # logging.info(f"loss pred_joints2d: {pred_joints2d.detach().cpu().numpy()}")

        joints2d_conf = batch["smpl_joints2d_confidence"]
        joints2d_conf = joints2d_conf.to(device=pred_joints2d.device, dtype=pred_joints2d.dtype)
        if joints2d_conf.dim() == 1:
            joints2d_conf = joints2d_conf.unsqueeze(0).unsqueeze(0)
        elif joints2d_conf.dim() == 2:
            joints2d_conf = joints2d_conf.unsqueeze(1)
        elif joints2d_conf.dim() != 3:
            raise ValueError(f"Unsupported smpl_joints2d_confidence shape: {tuple(joints2d_conf.shape)}")
        if has_smpl is not None:
            has_smpl_2d = has_smpl.to(device=joints2d_conf.device, dtype=joints2d_conf.dtype)
            if has_smpl_2d.dim() == 1:
                has_smpl_2d = has_smpl_2d[:, None, None]
            elif has_smpl_2d.dim() == 2:
                has_smpl_2d = has_smpl_2d[:, :, None]
            else:
                raise ValueError(f"Unsupported has_smpl shape for 2D joints: {tuple(has_smpl.shape)}")
            if has_smpl_2d.shape[1] != joints2d_conf.shape[1]:
                if has_smpl_2d.shape[1] == 1:
                    has_smpl_2d = has_smpl_2d.expand(-1, joints2d_conf.shape[1], -1)
                elif joints2d_conf.shape[1] == 1:
                    has_smpl_2d = has_smpl_2d[:, :1]
                else:
                    raise ValueError(
                        f"Cannot align has_smpl shape {tuple(has_smpl.shape)} to joints2d_conf shape {tuple(joints2d_conf.shape)}"
                    )
            joints2d_conf = joints2d_conf * has_smpl_2d
        # print(f"pred_joints2d.shape: {pred_joints2d.shape}")
        # print(f"joints2d_conf.shape: {joints2d_conf.shape}")
        # print(f"joints2d_conf: {joints2d_conf}")

        B, S, J_pred = pred_joints2d.shape[:3]
        if gt_joints2d.shape[1] != S:
            gt_joints2d = gt_joints2d[:, :1].expand(-1, S, -1, -1)
        if joints2d_conf.shape[1] != S:
            joints2d_conf = joints2d_conf[:, :1].expand(-1, S, -1)

        # Normalize both pred/gt 2D joints to [-1, 1] in image coordinates (TRAM-style).
        # x_norm = (x - W/2) / (W/2), y_norm = (y - H/2) / (H/2)
        H, W = image_hw
        half_w = torch.tensor(W / 2.0, device=pred_joints2d.device, dtype=pred_joints2d.dtype)
        half_h = torch.tensor(H / 2.0, device=pred_joints2d.device, dtype=pred_joints2d.dtype)

        pred_joints2d_norm = pred_joints2d.clone()
        pred_joints2d_norm[..., 0] = (pred_joints2d_norm[..., 0] - half_w) / half_w.clamp(min=1.0)
        pred_joints2d_norm[..., 1] = (pred_joints2d_norm[..., 1] - half_h) / half_h.clamp(min=1.0)

        # 投影爆掉偵測(診斷用,no-grad):in-frame 應 ~[-1,1],飆高代表深度 z→0 投影爆掉
        smpl_joints2d_max_abs = pred_joints2d_norm.detach().abs().max()

        gt_joints2d_norm = gt_joints2d.clone()
        gt_joints2d_norm[..., 0] = (gt_joints2d_norm[..., 0] - half_w) / half_w.clamp(min=1.0)
        gt_joints2d_norm[..., 1] = (gt_joints2d_norm[..., 1] - half_h) / half_h.clamp(min=1.0)

        joints2d_loss_type = loss_type_joints2d or loss_type
        body_joint_count = min(24, J_pred, gt_joints2d.shape[2], joints2d_conf.shape[2])
        if weight_joints2d > 0.0 and body_joint_count > 0:
            pred_body_joints2d = pred_joints2d_norm[:, :, :body_joint_count]
            gt_body_joints2d = gt_joints2d_norm[:, :, :body_joint_count]
            body_joints2d_conf = joints2d_conf[:, :, :body_joint_count]
            if joints2d_loss_type == "l1":
                diff = (pred_body_joints2d - gt_body_joints2d).abs().sum(dim=-1)
            elif joints2d_loss_type == "l2":
                diff = ((pred_body_joints2d - gt_body_joints2d) ** 2).sum(dim=-1)
            else:
                raise ValueError(joints2d_loss_type)
            valid_f = (body_joints2d_conf > 0.5).to(dtype=diff.dtype)
            # Robust gate: drop joints whose normalized reprojection blew up (depth z→0 →
            # coords >> 1). In-frame is ~[-1,1]; anything beyond joints2d_max_norm is a bad
            # mesh_translate this step. Masking removes its huge gradient into the shared
            # backbone; the mesh_translate L1 loss still pulls these back toward valid depth.
            joints2d_max_norm = float(kwargs.get("joints2d_max_norm", 0.0))
            if joints2d_max_norm > 0.0:
                sane = (
                    pred_body_joints2d.detach().abs().amax(dim=-1) <= joints2d_max_norm
                ).to(dtype=diff.dtype)
                valid_f = valid_f * sane
            loss_joints2d = (diff * valid_f).sum()
            loss_joints2d = check_and_fix_inf_nan(loss_joints2d, "loss_joints2d")
            loss_joints2d = loss_joints2d / valid_f.sum().clamp(min=1.0)

    #* Vertices loss (SMPL space; mirrors TRAM behavior)
    if weight_vertices > 0.0:
        B, T = gt_pose.shape[:2]
        gt_pose_flat = gt_pose.reshape(B * T, -1)
        gt_beta_flat = gt_beta.reshape(B * T, -1)
        if smpl_decode_uses_mesh_translate:
            gt_trans_flat = torch.zeros(B * T, 3, device=gt_pose.device, dtype=gt_pose.dtype)
        else:
            gt_trans_flat = gt_trans.reshape(B * T, -1)

        genders = _resolve_batch_genders(batch, B)
        if len(genders) == B:
            genders = [g for g in genders for _ in range(T)]

        # Match the prediction decode path: keep SMPL LBS and gauge transforms in
        # fp32 so use_gt=True does not leave a small bf16/autocast vertex residual.
        with torch.cuda.amp.autocast(enabled=False):
            gt_joints_zero_flat, gt_vertices_smpl_flat = _decode_smpl_batch(
                pose_aa=gt_pose_flat.float(),
                betas=gt_beta_flat.float(),
                trans=gt_trans_flat.float(),
                genders=genders,
                use_mamma=use_mamma,
            )
        gt_vertices_smpl = gt_vertices_smpl_flat.reshape(B, T, gt_vertices_smpl_flat.shape[1], 3)
        if smpl_decode_uses_mesh_translate:
            gt_joints_zero = gt_joints_zero_flat.reshape(B, T, gt_joints_zero_flat.shape[1], 3)
            # mesh_rot mode: gt_pose[:3] is the cam0-frame mesh_rot (overwritten above),
            # so the decoded GT body is already in cam0 -> scale-only gauge.
            with torch.cuda.amp.autocast(enabled=False):
                avg_scale_f = batch["avg_scale"].float()
                if use_mesh_rot:
                    gt_vertices_smpl = scale_joints_to_batch_gauge(gt_vertices_smpl.float(), avg_scale_f)
                    gt_joints_zero = scale_joints_to_batch_gauge(gt_joints_zero.float(), avg_scale_f)
                else:
                    gt_vertices_smpl = normalize_joints_world_to_batch_gauge(
                        gt_vertices_smpl.float(),
                        batch["raw_extrinsics"].float(),
                        avg_scale_f,
                    )
                    gt_joints_zero = normalize_joints_world_to_batch_gauge(
                        gt_joints_zero.float(),
                        batch["raw_extrinsics"].float(),
                        avg_scale_f,
                    )
            gt_mesh_translate_aligned = gt_mesh_translate.to(device=gt_pose.device).float()
            gt_offset = gt_mesh_translate_aligned - gt_joints_zero[..., 0, :]
            gt_vertices_smpl = gt_vertices_smpl + gt_offset.unsqueeze(-2)

        # L1 in SMPL space
        vertex_diff = (pred_vertices_smpl - gt_vertices_smpl).abs().mean(dim=(-1, -2))
        if has_smpl is not None:
            vertex_mask = has_smpl.to(device=vertex_diff.device, dtype=vertex_diff.dtype)
            if vertex_mask.dim() == 1:
                vertex_mask = vertex_mask[:, None].expand_as(vertex_diff)
            loss_vertices = (vertex_diff * vertex_mask).sum() / vertex_mask.sum().clamp(min=1.0)
        else:
            loss_vertices = vertex_diff.mean()

    # ---- dense-landmark (direct 2D GNLL) + per-person mask losses ----
    # Both use the already-matched (Hungarian) + people-flattened predictions and
    # the shared has_smpl mask, so slots stay bound to the same identity.
    loss_landmark = (pred_pose * 0.0).mean()
    loss_landmark_l2 = (pred_pose * 0.0).mean().detach()
    loss_landmark_vis = (pred_pose * 0.0).mean()
    landmark_px = (pred_pose * 0.0).mean().detach()
    weight_landmark = float(kwargs.get("weight_landmark", 0.0))
    weight_landmark_vis = float(kwargs.get("weight_landmark_vis", 0.0))
    if (
        (weight_landmark > 0.0 or weight_landmark_vis > 0.0)
        and predictions.get("smpl_landmarks2d") is not None
        and batch.get("smpl_landmarks2d") is not None
    ):
        lmk_dict = compute_landmark_loss(
            pred_xy=predictions["smpl_landmarks2d"],
            pred_logvar=predictions.get("smpl_landmarks_logvar"),
            gt_xy=batch["smpl_landmarks2d"],
            visibility=batch.get("smpl_landmarks2d_visibility"),
            has_smpl=has_smpl,
            loss_type=str(kwargs.get("landmark_loss_type", "gnll")),
            weight_mode=str(kwargs.get("landmark_weight_mode", "visibility")),
            mamma_beta=float(kwargs.get("landmark_mamma_beta", 2.0)),
            mamma_hand_weight=float(kwargs.get("landmark_mamma_hand_weight", 1.0)),
        )
        loss_landmark = lmk_dict["loss_landmark"]
        loss_landmark_l2 = lmk_dict["loss_landmark_l2"]
        image_size = 1.0
        if batch.get("images") is not None:
            image_size = float(max(batch["images"].shape[-2:]))
        landmark_px = loss_landmark_l2 * image_size

    if (
        weight_landmark_vis > 0.0
        and predictions.get("smpl_landmarks_visibility_logits") is not None
        and batch.get("smpl_landmarks2d_visibility") is not None
    ):
        loss_landmark_vis = compute_landmark_visibility_loss(
            pred_logits=predictions["smpl_landmarks_visibility_logits"],
            visibility=batch["smpl_landmarks2d_visibility"],
            has_smpl=has_smpl,
        )

    # ---- MAMMA-style per-landmark contact losses (person-person + floor) ----
    loss_contact = (pred_pose * 0.0).mean()
    loss_floor_contact = (pred_pose * 0.0).mean()
    contact_positive_frac = (pred_pose * 0.0).mean().detach()
    weight_contact = float(kwargs.get("weight_contact", 0.0))
    weight_floor_contact = float(kwargs.get("weight_floor_contact", 0.0))
    contact_alpha = float(kwargs.get("contact_focal_alpha", 0.9))
    contact_gamma = float(kwargs.get("contact_focal_gamma", 2.0))
    if (
        weight_contact > 0.0
        and predictions.get("smpl_contact_logits") is not None
        and batch.get("smpl_contact") is not None
    ):
        loss_contact = compute_contact_loss(
            pred_logits=predictions["smpl_contact_logits"],
            target=batch["smpl_contact"],
            has_smpl=has_smpl,
            alpha=contact_alpha,
            gamma=contact_gamma,
        )
        _ct = batch["smpl_contact"].float()
        _valid = (_ct >= -0.5)
        contact_positive_frac = (
            (_ct.clamp(0, 1) * _valid).sum() / _valid.sum().clamp(min=1.0)
        ).detach()
    if (
        weight_floor_contact > 0.0
        and predictions.get("smpl_floor_contact_logits") is not None
        and batch.get("smpl_floor_contact") is not None
    ):
        loss_floor_contact = compute_contact_loss(
            pred_logits=predictions["smpl_floor_contact_logits"],
            target=batch["smpl_floor_contact"],
            has_smpl=has_smpl,
            alpha=contact_alpha,
            gamma=contact_gamma,
        )

    loss_mask = (pred_pose * 0.0).mean()
    mask_soft_iou = (pred_pose * 0.0).mean().detach()
    weight_mask = float(kwargs.get("weight_mask", 0.0))
    if (
        weight_mask > 0.0
        and predictions.get("person_mask_logits") is not None
        and batch.get("person_mask") is not None
    ):
        mask_dict = compute_mask_loss(
            pred_logits=predictions["person_mask_logits"],
            gt_mask=batch["person_mask"],
            has_smpl=has_smpl,
        )
        loss_mask = mask_dict["loss_mask"]
        mask_soft_iou = mask_dict["mask_soft_iou"]

    total = (
        loss_smpl_losses +
        weight_trans * loss_trans +
        weight_mesh_translate * loss_mesh_translate +
        weight_presence * loss_presence +
        weight_joints2d * loss_joints2d +
        weight_joints3d * loss_joints3d +
        weight_vertices * loss_vertices +
        weight_landmark * loss_landmark +
        weight_landmark_vis * loss_landmark_vis +
        weight_contact * loss_contact +
        weight_floor_contact * loss_floor_contact +
        weight_mask * loss_mask
        + weight_temporal_pose * temporal_losses["loss_smpl_temporal_pose"]
        + weight_temporal_beta * temporal_losses["loss_smpl_temporal_beta"]
        + weight_temporal_mesh_translate
        * temporal_losses["loss_smpl_temporal_mesh_translate"]
        + weight_temporal_root_rotation
        * temporal_losses["loss_smpl_temporal_root_rotation"]
    )

    return {
        "loss_smpl": total,
        "loss_contact": loss_contact,
        "loss_floor_contact": loss_floor_contact,
        "contact_positive_frac": contact_positive_frac,
        "loss_smpl_losses": loss_smpl_losses,
        "loss_smpl_global_orient": loss_global_orient,
        "loss_smpl_global_orient_init": loss_global_orient_init,
        "loss_smpl_trans": loss_trans,
        "loss_mesh_translate": loss_mesh_translate,
        "loss_smpl_presence": loss_presence,
        "loss_smpl_joints2d": loss_joints2d,
        "loss_smpl_joints3d": loss_joints3d,
        "loss_smpl_vertices": loss_vertices,
        "loss_landmark": loss_landmark,
        "loss_landmark_l2": loss_landmark_l2,
        "loss_landmark_vis": loss_landmark_vis,
        "landmark_px": landmark_px.detach(),
        "loss_mask": loss_mask,
        **temporal_losses,
        "mask_soft_iou": mask_soft_iou,
        # matched-pair Hungarian cost diagnostics (0 when the cost term is off).
        "hungarian_presence_cost": matching_cost_metrics["presence_cost"],
        "hungarian_mask_cost": matching_cost_metrics["mask_cost"],
        "smpl_presence_positive_prob_mean": smpl_presence_positive_prob_mean,
        "smpl_presence_empty_prob_mean": smpl_presence_empty_prob_mean,
        "smpl_joints2d_max_abs": smpl_joints2d_max_abs,
    }


def compute_smpl_3d_joint_loss(pred_joints_world, gt_joints_norm, batch, loss_type="l1",
                              pelvis_center=True, pelvis_ids=(1,2), normalize_cam=True):
    """
    pred_joints_world: predictions["smpl_joints3d_world"]  (raw world/unity)
    gt_joints_norm:    batch["smpl_joints3d_world"]       (你 _process_batch 後已 normalize)
    batch 需要提供 raw_extrinsics, avg_scale
    """
    raw_extr = batch["raw_extrinsics"]
    avg_scale = batch["avg_scale"]

    if normalize_cam:
        pred_norm = normalize_joints_world_to_batch_gauge(pred_joints_world, raw_extr, avg_scale)
    else:
        pred_norm = pred_joints_world

    if pred_norm.dim() == 4 and gt_joints_norm.dim() == 4:
        if pred_norm.shape[1] != gt_joints_norm.shape[1]:
            if gt_joints_norm.shape[1] % pred_norm.shape[1] == 0:
                views_per_frame = gt_joints_norm.shape[1] // pred_norm.shape[1]
                gt_joints_norm = gt_joints_norm[:, ::views_per_frame]
            elif pred_norm.shape[1] % gt_joints_norm.shape[1] == 0:
                pred_norm = pred_norm[:, :: pred_norm.shape[1] // gt_joints_norm.shape[1]]
            T = min(pred_norm.shape[1], gt_joints_norm.shape[1])
            pred_norm = pred_norm[:, :T]
            gt_joints_norm = gt_joints_norm[:, :T]
        B, T, J_pred, _ = pred_norm.shape
        _, _, J_gt, _ = gt_joints_norm.shape
        J = min(J_pred, J_gt)
        pred_norm = pred_norm[:, :, :J, :].reshape(B * T, J, 3)
        gt_joints_norm = gt_joints_norm[:, :, :J, :].reshape(B * T, J, 3)
    else:
        if pred_norm.dim() == 4:
            pred_norm = pred_norm[:, 0]
        if gt_joints_norm.dim() == 4:
            gt_joints_norm = gt_joints_norm[:, 0]

        if pred_norm.dim() != 3 or gt_joints_norm.dim() != 3:
            raise ValueError(
                f"Expected pred/gt joints to be 3D after alignment, got {tuple(pred_norm.shape)} and {tuple(gt_joints_norm.shape)}"
            )

    # If joint counts differ, fall back to matching the common prefix.
    if pred_norm.shape[1] != gt_joints_norm.shape[1]:
        J = min(pred_norm.shape[1], gt_joints_norm.shape[1])
        pred_norm = pred_norm[:, :J, :]
        gt_joints_norm = gt_joints_norm[:, :J, :]

    if pelvis_center:
        pred_norm, gt_joints_norm = subtract_pelvis(pred_norm, gt_joints_norm, pelvis_ids=pelvis_ids)

    if loss_type == "l1":
        loss_tensor = (pred_norm - gt_joints_norm).abs()
    elif loss_type == "l2":
        loss_tensor = (pred_norm - gt_joints_norm) ** 2
    else:
        raise ValueError(loss_type)

    has_smpl = batch.get("has_smpl", None)
    if has_smpl is not None:
        sample_loss = loss_tensor.mean(dim=(-1, -2))
        mask = has_smpl.to(device=sample_loss.device, dtype=sample_loss.dtype).reshape(-1)
        if sample_loss.numel() != mask.numel():
            if sample_loss.numel() % mask.numel() == 0:
                mask = mask.repeat_interleave(sample_loss.numel() // mask.numel())
            else:
                mask = mask[: sample_loss.numel()]
        return (sample_loss.reshape(-1) * mask).sum() / mask.sum().clamp(min=1.0)

    return loss_tensor.mean()


_BODY_PARTS_JSON = Path(__file__).resolve().parent / "data" / "assets" / "smplx_512_body_parts.json"


@lru_cache(maxsize=1)
def _load_landmark_body_parts() -> dict:
    """512-landmark -> body-part index lists (MAMMA's _smplx_512_body_parts.json).

    The repo's verts_512.pkl is the SAME MAMMA down-sampling file, so these indices
    align 1:1 with our smpl_landmarks2d ordering.
    """
    with open(_BODY_PARTS_JSON, "r") as f:
        return json.load(f)


def compute_mamma_landmark_weight(
    gt_xy: torch.Tensor,      # (..., L, 2) normalised [0,1] image coords
    beta: float = 2.0,
    hand_weight: float = 1.0,
) -> torch.Tensor:
    """MAMMA-style per-landmark loss weight (BEDLAM_WD.target_weight), NO occlusion gate.

    - in-frame (x,y in [0,1])           -> 1.0
    - out-of-frame                      -> exp(-beta*|dist_to_center - 1|)  (soft, >0)
    - hands / feet (if the part has any in-frame point) -> x2 ; hands additionally x hand_weight

    Occluded-but-in-frame landmarks keep weight 1 (unlike the repo's visibility gate),
    which is the whole point of the MAMMA weighting.
    """
    L = gt_xy.shape[-2]
    inframe = ((gt_xy >= 0.0) & (gt_xy <= 1.0)).all(dim=-1).to(gt_xy.dtype)  # (..., L)
    # out-of-frame soft decay: map [0,1] -> [-1,1], distance to centre, exp decay.
    coords_pm = 2.0 * (gt_xy - 0.5)
    dist = torch.linalg.vector_norm(coords_pm, dim=-1)                        # (..., L)
    w_out = torch.exp(-float(beta) * (dist - 1.0).abs())
    weight = torch.where(inframe > 0.5, torch.ones_like(w_out), w_out)        # (..., L)

    parts = _load_landmark_body_parts()
    dev = gt_xy.device
    for key in ("left_hand", "right_hand", "left_feet", "right_feet"):
        idx = parts.get(key)
        if not idx:
            continue
        idx = torch.as_tensor([i for i in idx if 0 <= i < L], device=dev, dtype=torch.long)
        if idx.numel() == 0:
            continue
        # per-sample gate: only boost this part where at least one of its pts is in-frame.
        part_has = inframe.index_select(-1, idx).amax(dim=-1, keepdim=True)   # (..., 1)
        factor = 1.0 + part_has                                              # x2 when present
        if key in ("left_hand", "right_hand") and hand_weight != 1.0:
            factor = factor * torch.where(part_has > 0.5, torch.full_like(part_has, float(hand_weight)), torch.ones_like(part_has))
        sub = weight.index_select(-1, idx) * factor
        weight = weight.index_copy(-1, idx, sub)
    return weight


def compute_landmark_loss(
    pred_xy: torch.Tensor,       # (B*P, S, L, 2) normalised [0,1]
    pred_logvar: torch.Tensor,   # (B*P, S, L)
    gt_xy: torch.Tensor,         # (B*P, S, L, 2)
    visibility: torch.Tensor | None,  # (B*P, S, L) in {0,1}
    has_smpl: torch.Tensor | None,    # (B*P,)
    loss_type: str = "gnll",
    weight_mode: str = "visibility",   # "visibility" (repo) | "mamma" (in-frame + soft-outside + part boost)
    mamma_beta: float = 2.0,
    mamma_hand_weight: float = 1.0,
) -> dict:
    """Dense 512-landmark loss (direct 2D).

    Gaussian NLL with a predicted per-landmark log-variance (mamma-style), so
    hard / occluded points down-weight themselves.  Only visible landmarks of
    matched (has_smpl) people contribute.
    """
    pred_xy = pred_xy.float()
    gt_xy = gt_xy.to(device=pred_xy.device, dtype=pred_xy.dtype)
    pred_logvar = pred_logvar.float()

    # per-landmark squared error summed over (x, y).
    sq = ((pred_xy - gt_xy) ** 2).sum(dim=-1)                     # (B*P, S, L)

    if loss_type == "gnll":
        # MAMMA clipped Gaussian-NLL: cap the precision*error term at 25 so a
        # confident (small-sigma) prediction can't earn unbounded reward -- this is
        # what prevents the x-variance collapse a plain 0.5*(exp(-logvar)*sq+logvar)
        # suffers under overfit. pred_logvar is used as log_sigma here.
        log_sigma = pred_logvar.clamp(min=math.log(1e-6))
        two_sigma_sq = 2.0 * torch.exp(log_sigma) ** 2
        per_lmk = torch.clip(sq / two_sigma_sq, max=25.0) + 2.0 * log_sigma
    elif loss_type == "l2":
        per_lmk = sq
    else:
        raise ValueError(f"Unknown landmark loss_type: {loss_type}")

    if weight_mode == "mamma":
        # MAMMA-style weight: in-frame=1 + soft out-of-frame decay + hand/feet boost.
        # Occlusion is NOT gated here (occluded-but-in-frame points keep weight 1);
        # occlusion is only supervised by the separate visibility BCE head.
        weight = compute_mamma_landmark_weight(
            gt_xy, beta=mamma_beta, hand_weight=mamma_hand_weight
        ).to(device=per_lmk.device, dtype=per_lmk.dtype)
    elif weight_mode == "visibility":
        # repo default: hard 0/1 gate by (occlusion x in-frame) visibility.
        weight = torch.ones_like(per_lmk)
        if visibility is not None:
            weight = weight * visibility.to(device=weight.device, dtype=weight.dtype)
    else:
        raise ValueError(f"Unknown landmark weight_mode: {weight_mode!r} (use 'visibility' or 'mamma')")
    if has_smpl is not None:
        w_person = has_smpl.to(device=weight.device, dtype=weight.dtype).reshape(-1, 1, 1)
        weight = weight * w_person

    denom = weight.sum().clamp(min=1.0)
    loss_landmark = (per_lmk * weight).sum() / denom
    # raw pixel-ish L2 (unweighted by sigma) for logging.
    loss_landmark_l2 = (sq.sqrt() * weight).sum() / denom

    return {
        "loss_landmark": loss_landmark,
        "loss_landmark_l2": loss_landmark_l2.detach(),
    }


def compute_landmark_visibility_loss(
    pred_logits: torch.Tensor,        # (B*P, S, L)
    visibility: torch.Tensor,         # (B*P, S, L) in {0,1}
    has_smpl: torch.Tensor | None,    # (B*P,)
) -> torch.Tensor:
    """BCE visibility loss for dense landmarks, ignored for empty person slots."""
    pred_logits = pred_logits.float()
    target = visibility.to(device=pred_logits.device, dtype=pred_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")

    weight = torch.ones_like(loss)
    if has_smpl is not None:
        w_person = has_smpl.to(device=weight.device, dtype=weight.dtype).reshape(-1, 1, 1)
        weight = weight * w_person

    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def compute_contact_loss(
    pred_logits: torch.Tensor,        # (B*P, S, L)
    target: torch.Tensor,            # (B*P, S, L) in {0,1}
    has_smpl: torch.Tensor | None,    # (B*P,)
    alpha: float = 0.9,
    gamma: float = 2.0,
) -> torch.Tensor:
    """MAMMA-style per-landmark contact loss: sigmoid focal loss (alpha, gamma).

    Faithful port of MAMMA's ``focal_loss`` (landmarks/lib/models/models_2d/loss.py),
    with an added ``has_smpl`` mask so empty person slots don't contribute.
    """
    pred_logits = pred_logits.float()
    target = target.to(device=pred_logits.device, dtype=pred_logits.dtype)
    # target < 0 is a sentinel for "no GT" (e.g. sdf_vertices absent for this sample);
    # those entries are masked out. Clamp to [0,1] so BCE stays well-defined.
    valid = (target >= -0.5).to(dtype=pred_logits.dtype)
    target = target.clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p = torch.sigmoid(pred_logits)
    pt = target * p + (1.0 - target) * (1.0 - p)
    loss = float(alpha) * (1.0 - pt) ** float(gamma) * bce      # (B*P, S, L)

    weight = valid
    if has_smpl is not None:
        w_person = has_smpl.to(device=weight.device, dtype=weight.dtype).reshape(-1, 1, 1)
        weight = weight * w_person
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)
