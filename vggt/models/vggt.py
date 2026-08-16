# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.smpl_head import SMPLHead
from vggt.heads.smpl_multi_query_head import SMPLMultiQueryHead
from vggt.heads.smpl_multi_query_trans_head import SMPLMultiQueryTransHead
from vggt.heads.smpl_multi_query_trans_rot_head import SMPLMultiQueryTransRotHead
from vggt.heads.smpl_multi_query_trans_rot_temporal_rel_head import (
    SMPLMultiQueryTemporalRelConfig,
    SMPLMultiQueryTransRotTemporalRelHead,
)
from vggt.heads.smpl_dense_landmark_head import DenseLandmarkHeadConfig, SMPLDenseLandmarkHead
from vggt.heads.person_mask_head import PersonMaskDPTHead, PersonMaskHead
from vggt.heads.smpl_direct_mask_head import SMPLDirectMaskCamHead, SMPLDirectMaskHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, patch_embed="dinov2_vitl14_reg", patch_embed_checkpoint=None,
                 out_channels=[256, 512, 1024, 1024], intermediate_layer_idx=[4, 11, 17, 23], frames_chunk_size=8,
                 enable_camera=True, enable_point=False, enable_depth=False, enable_track=False, enable_smpl=False,
                 enable_smpl_multi_query=False, enable_smpl_multi_query_trans=False,
                 enable_smpl_multi_query_trans_rot=False, smpl_num_people=1,
                 enable_temporal_path=False, enable_temporal_heads=False,
                 enable_temporal_rel_pe=False,
                 temporal_rel_num_buckets=32, temporal_rel_max_distance=31,
                 temporal_head_max_T=128,
                 enable_smpl_dense_landmark=False, enable_person_mask=False,
                 person_mask_head_type="dot", person_mask_down_ratio=2,
                 person_mask_embed_dim=None,
                 landmark_use_mask_embedding=False, landmark_mask_embed_dim=256,
                 landmark_detach_mask_context=True, landmark_predict_contact=False,
                 ):
        super().__init__()

        if enable_temporal_heads and not enable_smpl_multi_query_trans_rot:
            raise ValueError(
                "enable_temporal_heads currently requires "
                "enable_smpl_multi_query_trans_rot=True"
            )
        if enable_temporal_heads and not enable_temporal_path:
            raise ValueError(
                "Temporal SMPL heads require enable_temporal_path=True so the "
                "aggregator remains per-timestep."
            )
        if enable_temporal_heads and not enable_temporal_rel_pe:
            raise ValueError(
                "Only the relative-PE temporal multi-person head is supported; "
                "set enable_temporal_rel_pe=True."
            )

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, patch_embed=patch_embed, patch_embed_checkpoint=patch_embed_checkpoint)
        self.use_temporal_path_enabled = bool(enable_temporal_path)
        self.use_temporal_smpl_head = bool(enable_temporal_heads)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", out_channels=out_channels, intermediate_layer_idx=intermediate_layer_idx, frames_chunk_size=frames_chunk_size) if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", out_channels=out_channels, intermediate_layer_idx=intermediate_layer_idx, frames_chunk_size=frames_chunk_size) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.smpl_head  = SMPLHead(dim_in=2 * embed_dim) if enable_smpl else None
        self.smpl_multi_query_head = SMPLMultiQueryHead(dim_in=2 * embed_dim, num_people=smpl_num_people) if enable_smpl_multi_query else None
        self.smpl_multi_query_trans_head = SMPLMultiQueryTransHead(dim_in=2 * embed_dim, num_people=smpl_num_people) if enable_smpl_multi_query_trans else None
        if enable_smpl_multi_query_trans_rot and enable_temporal_heads:
            temporal_cfg = SMPLMultiQueryTemporalRelConfig(
                max_T=int(temporal_head_max_T),
                rel_num_buckets=int(temporal_rel_num_buckets),
                rel_max_distance=int(temporal_rel_max_distance),
            )
            self.smpl_multi_query_trans_rot_head = (
                SMPLMultiQueryTransRotTemporalRelHead(
                    dim_in=2 * embed_dim,
                    num_people=smpl_num_people,
                    smpl_cfg=temporal_cfg,
                )
            )
        else:
            self.smpl_multi_query_trans_rot_head = (
                SMPLMultiQueryTransRotHead(
                    dim_in=2 * embed_dim, num_people=smpl_num_people
                )
                if enable_smpl_multi_query_trans_rot
                else None
            )
        # Auxiliary heads that condition on the SMPL head's person tokens.
        landmark_cfg = DenseLandmarkHeadConfig(
            use_mask_embedding=bool(landmark_use_mask_embedding),
            mask_embed_dim=int(landmark_mask_embed_dim),
            detach_mask_context=bool(landmark_detach_mask_context),
            predict_contact=bool(landmark_predict_contact),
        )
        self.smpl_dense_landmark_head = (
            SMPLDenseLandmarkHead(context_dim=2 * embed_dim, query_dim=embed_dim, cfg=landmark_cfg)
            if enable_smpl_dense_landmark
            else None
        )
        # "dot": patch-grid dot-product head (37x37). "dpt": pixel-level DPT head
        # (H/person_mask_down_ratio, e.g. 259x259 for 518) -- sharper boundaries
        # for people in contact.
        self.person_mask_head_type = str(person_mask_head_type)
        if enable_person_mask:
            if self.person_mask_head_type == "dpt":
                self.person_mask_head = PersonMaskDPTHead(
                    dim_in=2 * embed_dim,
                    query_dim=embed_dim,
                    embed_dim=int(person_mask_embed_dim or 128),
                    intermediate_layer_idx=intermediate_layer_idx,
                    down_ratio=int(person_mask_down_ratio),
                    patch_size=patch_size,
                    frames_chunk_size=frames_chunk_size,
                )
            elif self.person_mask_head_type == "dot":
                self.person_mask_head = PersonMaskHead(
                    query_dim=embed_dim,
                    context_dim=2 * embed_dim,
                    embed_dim=int(person_mask_embed_dim or 256),
                )
            elif self.person_mask_head_type == "direct":
                # Ablation: decode the mask straight from the person token, with no
                # image features. View-agnostic by construction (see the head's
                # docstring); output is full-res (img_size) logits.
                self.person_mask_head = SMPLDirectMaskHead(
                    query_dim=embed_dim,
                    out_size=int(img_size),
                )
            elif self.person_mask_head_type == "direct_cam":
                # Ablation: token decoder + per-view camera-token conditioning, so
                # the mask can differ per view via geometry (still no image patch
                # features). Output is full-res (img_size) logits.
                self.person_mask_head = SMPLDirectMaskCamHead(
                    query_dim=embed_dim,
                    cam_dim=2 * embed_dim,
                    out_size=int(img_size),
                )
            else:
                raise ValueError(f"Unknown person_mask_head_type: {person_mask_head_type!r}")
        else:
            self.person_mask_head = None

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        smpl_inputs: Optional[Dict[str, object]] = None,
        return_encoder_feature_map: bool = False,
        return_aggregator_feature_map: bool = False,
        return_camera_tokens: bool = False,
        return_depth_feature_map: bool = False,
        return_point_feature_map: bool = False,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            smpl_inputs (dict, optional): Aux data for the SMPL head (e.g., gender tags, pelvis targets).
                Default: None
            return_encoder_feature_map (bool, optional): If True, also returns the ViT encoder
                patch tokens (reshaped to [B, S, H/patch, W/patch, C]) and their per-patch L2 norm map.
                Default: False.
            return_aggregator_feature_map (bool, optional): If True, also returns the final aggregator
                patch tokens reshaped to [B, S, H/patch, W/patch, 2C] and their per-patch L2 norm map.
                Default: False.
            return_camera_tokens (bool, optional): If True, include the raw camera tokens used by
                the camera head (shape [B, S, C]) in the predictions dict. Default: False.
            return_depth_feature_map (bool, optional): If True, include the fused DPT feature map
                (pre-conv tokens) for the depth head with shape [B, S, C, H, W]. Default: False.
            return_point_feature_map (bool, optional): If True, include the fused DPT feature map
                for the point head with shape [B, S, C, H, W]. Default: False.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                                - depth_feature_map (torch.Tensor): (Optional) Fused DPT tokens before the depth head
                                    convolutions with shape [B, S, C, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                                - point_feature_map (torch.Tensor): (Optional) Fused DPT tokens before the point head
                                    convolutions with shape [B, S, C, H, W]
                - smpl_pose (torch.Tensor): SMPL pose parameters with shape [B, 72] or [B, P, 72]
                - smpl_beta (torch.Tensor): SMPL shape parameters with shape [B, 10] or [B, P, 10]
                - smpl_confidence (torch.Tensor): Multi-person slot confidence probabilities with shape [B, P]
                - smpl_joints3d_world (torch.Tensor): Unity-space joints with shape [B, 24, 3]
                - smpl_joints3d_camera (torch.Tensor): Camera-frame joints with shape [B, S, 24, 3]
                - smpl_joints2d (torch.Tensor): Per-view 2D joints with shape [B, S, 24, 2]
                - ball_3d (torch.Tensor): Predicted 3D ball position with shape [B, 3]
                - images (torch.Tensor): Original input images, preserved for visualization
                - encoder_patch_tokens (torch.Tensor): (Optional) Encoder ViT tokens with shape [B, S, H/patch, W/patch, C]
                - encoder_feature_l2_map (torch.Tensor): (Optional) L2 norm map of encoder tokens with shape [B, S, H/patch, W/patch]
                - aggregator_feature_map (torch.Tensor): (Optional) Final aggregator patch tokens with shape [B, S, H/patch, W/patch, 2C]
                - aggregator_feature_l2_map (torch.Tensor): (Optional) L2 norm map of aggregator tokens with shape [B, S, H/patch, W/patch]
                - camera_tokens (torch.Tensor): (Optional) Raw camera tokens with shape [B, S, C] when requested

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        smpl_inputs = smpl_inputs or {}

        # Temporal samples arrive time-major as [B,T*V,3,H,W].  Match the
        # badminton temporal architecture by folding T into the batch axis before
        # the aggregator: global attention remains purely multi-view within one
        # timestep, while the temporal SMPL head performs all cross-time fusion.
        images_for_heads = images
        if self.use_temporal_path_enabled:
            temporal_num_frames = smpl_inputs.get("temporal_num_frames")
            views_per_frame = smpl_inputs.get("views_per_frame")
            if temporal_num_frames is not None and views_per_frame is not None:
                temporal_values = temporal_num_frames.reshape(-1)
                view_values = views_per_frame.reshape(-1)
                if not torch.all(temporal_values == temporal_values[0]):
                    raise ValueError("All clips in one batch must have the same T")
                if not torch.all(view_values == view_values[0]):
                    raise ValueError("All clips in one batch must have the same V")
                T = int(temporal_values[0].item())
                V = int(view_values[0].item())
                B, S = images.shape[:2]
                if S != T * V:
                    raise ValueError(
                        f"Temporal images have S={S}, expected T*V={T*V}"
                    )
                images_for_heads = images.reshape(
                    B, T, V, *images.shape[2:]
                ).reshape(B * T, V, *images.shape[2:])

        needs_encoder_patch_tokens = return_encoder_feature_map
        aggregated_tokens_list, patch_start_idx, encoder_patch_tokens = self.aggregator(
            images_for_heads, return_patch_tokens=needs_encoder_patch_tokens
        )

        aggregator_feature_map = None
        aggregator_feature_l2_map = None
        if return_aggregator_feature_map:
            final_tokens = aggregated_tokens_list[-1]
            patch_tokens = final_tokens[:, :, patch_start_idx:, :]
            patch_h = images_for_heads.shape[-2] // self.aggregator.patch_size
            patch_w = images_for_heads.shape[-1] // self.aggregator.patch_size
            if patch_h * patch_w != patch_tokens.shape[2]:
                raise ValueError(
                    "Mismatch between aggregator patch grid and token count: expected "
                    f"{patch_h * patch_w}, got {patch_tokens.shape[2]}. Check patch_size and image resolution."
                )
            aggregator_feature_map = patch_tokens.reshape(
                images_for_heads.shape[0],
                images_for_heads.shape[1],
                patch_h,
                patch_w,
                patch_tokens.shape[-1],
            ).contiguous()
            aggregator_feature_l2_map = torch.linalg.norm(aggregator_feature_map, dim=-1)

        camera_tokens = None
        if return_camera_tokens:
            # Camera tokens live at index 0 along the token dimension for each frame.
            camera_tokens = aggregated_tokens_list[-1][:, :, 0]

        predictions = {}

        cam_pose_enc = None

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_out = self.camera_head(aggregated_tokens_list)
                if isinstance(pose_out, list):
                    pose_enc_list = pose_out
                else:                           # simple head
                    pose_enc_list = [pose_out]
                cam_pose_enc = pose_enc_list[-1]
                predictions["pose_enc"] = cam_pose_enc  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth_out = self.depth_head(
                    aggregated_tokens_list,
                    images=images_for_heads,
                    patch_start_idx=patch_start_idx,
                    return_feature_map=return_depth_feature_map,
                )
                if return_depth_feature_map:
                    depth, depth_conf, depth_feat = depth_out
                    predictions["depth_feature_map"] = depth_feat
                else:
                    depth, depth_conf = depth_out
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                point_out = self.point_head(
                    aggregated_tokens_list,
                    images=images_for_heads,
                    patch_start_idx=patch_start_idx,
                    return_feature_map=return_point_feature_map,
                )
                if return_point_feature_map:
                    pts3d, pts3d_conf, point_feat = point_out
                    predictions["point_feature_map"] = point_feat
                else:
                    pts3d, pts3d_conf = point_out
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.smpl_head is not None:
                smpl_outputs = self.smpl_head(
                    aggregated_tokens_list,
                    patch_start_idx=patch_start_idx,
                )
                predictions.update(smpl_outputs)

            if self.smpl_multi_query_head is not None:
                smpl_outputs = self.smpl_multi_query_head(
                    aggregated_tokens_list,
                    patch_start_idx=patch_start_idx,
                )
                predictions.update(smpl_outputs)

            if self.smpl_multi_query_trans_head is not None:
                smpl_outputs = self.smpl_multi_query_trans_head(
                    aggregated_tokens_list,
                    patch_start_idx=patch_start_idx,
                )
                predictions.update(smpl_outputs)

            if self.smpl_multi_query_trans_rot_head is not None:
                head_kwargs = {}
                if self.use_temporal_smpl_head:
                    head_kwargs["smpl_inputs"] = smpl_inputs
                smpl_outputs = self.smpl_multi_query_trans_rot_head(
                    aggregated_tokens_list,
                    patch_start_idx=patch_start_idx,
                    **head_kwargs,
                )
                predictions.update(smpl_outputs)

            # Auxiliary dense-landmark / per-person-mask heads. They reuse the
            # SMPL head's person tokens (predictions["person_tokens"]) so every
            # head's slot p stays bound to the same identity.
            person_tokens = predictions.get("person_tokens", None)
            if person_tokens is not None and (
                self.smpl_dense_landmark_head is not None or self.person_mask_head is not None
            ):
                # (B, S, N_patch, 2C) per-view patch tokens from the last block.
                patch_tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]
                B, S, N_patch, C = patch_tokens.shape
                patch_h = images_for_heads.shape[-2] // self.aggregator.patch_size
                patch_w = images_for_heads.shape[-1] // self.aggregator.patch_size

                if self.person_mask_head is not None:
                    if self.person_mask_head_type == "dpt":
                        # pixel-level (B, S, P, H/dr, W/dr) from the DPT trunk.
                        predictions["person_mask_logits"] = self.person_mask_head(
                            person_tokens,
                            aggregated_tokens_list,
                            images=images_for_heads,
                            patch_start_idx=patch_start_idx,
                        )
                    elif self.person_mask_head_type == "direct":
                        # full-res (B, S, P, H, W) decoded straight from the token.
                        predictions["person_mask_logits"] = self.person_mask_head(
                            person_tokens, images=images_for_heads
                        )
                    elif self.person_mask_head_type == "direct_cam":
                        # token + per-view camera token (idx 0 of the last block,
                        # (B, S, C)) -> per-view (B, S, P, H, W). Same camera token
                        # the camera head / return_camera_tokens use.
                        cam_tokens = aggregated_tokens_list[-1][:, :, 0]
                        predictions["person_mask_logits"] = self.person_mask_head(
                            person_tokens, cam_tokens=cam_tokens
                        )
                    else:
                        predictions["person_mask_logits"] = self.person_mask_head(
                            person_tokens, patch_tokens, patch_hw=(patch_h, patch_w)
                        )
                if self.smpl_dense_landmark_head is not None:
                    # Direct per-view 2D: queries attend each view's own patches.
                    # When enabled, the per-person mask logits are embedded as a
                    # MAMMA-style spatial prompt over the context tokens.
                    predictions.update(
                        self.smpl_dense_landmark_head(
                            person_tokens,
                            patch_tokens,
                            person_mask_logits=predictions.get("person_mask_logits"),
                        )
                    )

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images_for_heads, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if return_encoder_feature_map and encoder_patch_tokens is not None:
            encoder_feature_l2_map = torch.linalg.norm(encoder_patch_tokens, dim=-1)
            predictions["encoder_patch_tokens"] = encoder_patch_tokens
            predictions["encoder_feature_l2_map"] = encoder_feature_l2_map

        if return_aggregator_feature_map and aggregator_feature_map is not None:
            predictions["aggregator_feature_map"] = aggregator_feature_map.detach()
            predictions["aggregator_feature_l2_map"] = aggregator_feature_l2_map.detach()

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        if return_camera_tokens and camera_tokens is not None:
            predictions["camera_tokens"] = camera_tokens.detach()

        return predictions
