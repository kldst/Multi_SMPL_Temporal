"""CPU shape/gradient smoke tests for the temporal multi-person SMPL head."""

import unittest

import torch

from vggt.heads.pose_transformer_rel import relative_position_bucket
from vggt.heads.smpl_multi_query_trans_rot_temporal_rel_head import (
    SMPLMultiQueryTemporalRelConfig,
    SMPLMultiQueryTransRotTemporalRelHead,
)


class TemporalRelHeadTest(unittest.TestCase):
    def test_relative_bucket_is_translation_invariant(self):
        times = torch.arange(5)
        first = relative_position_bucket(
            times[None] - times[:, None], num_buckets=16, max_distance=15
        )
        shifted = relative_position_bucket(
            (times + 17)[None] - (times + 17)[:, None],
            num_buckets=16,
            max_distance=15,
        )
        self.assertTrue(torch.equal(first, shifted))

    def test_output_shapes_and_gradient(self):
        config = SMPLMultiQueryTemporalRelConfig(
            transformer_depth=2,
            transformer_heads=4,
            transformer_mlp_dim=64,
            transformer_dim_head=8,
            transformer_dim=32,
            max_T=8,
            rel_num_buckets=16,
            rel_max_distance=15,
        )
        head = SMPLMultiQueryTransRotTemporalRelHead(
            dim_in=48, num_people=3, smpl_cfg=config
        )
        B, T, V, N, C = 2, 3, 2, 5, 48
        tokens = torch.randn(B * T, V, N + 1, C, requires_grad=True)
        outputs = head(
            [tokens],
            patch_start_idx=1,
            smpl_inputs={
                "temporal_num_frames": torch.full((B,), T),
                "views_per_frame": torch.full((B,), V),
            },
        )
        expected = {
            "smpl_pose": (B * T, 3, 72),
            "smpl_beta": (B * T, 3, 10),
            "mesh_translate": (B * T, 3, 3),
            "smpl_presence_logits": (B * T, 3),
            "person_tokens": (B * T, 3, 32),
        }
        for key, shape in expected.items():
            self.assertEqual(tuple(outputs[key].shape), shape)

        loss = sum(outputs[key].float().mean() for key in expected)
        loss.backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())


if __name__ == "__main__":
    unittest.main()
