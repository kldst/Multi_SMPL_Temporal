"""Unit tests for supervised temporal root-rotation motion loss."""

import unittest

import torch

from training.loss_smpl import compute_temporal_smpl_smoothness


class TemporalRootRotationLossTest(unittest.TestCase):
    @staticmethod
    def _inputs(pred_root, gt_root, valid=None):
        B, T, P, _ = pred_root.shape
        pred_pose = torch.zeros(B, T, P, 72, dtype=pred_root.dtype)
        gt_pose = torch.zeros_like(pred_pose)
        pred_pose[..., :3] = pred_root
        gt_pose[..., :3] = gt_root
        pred_pose = pred_pose.reshape(B * T, P, 72).requires_grad_()
        gt_pose = gt_pose.reshape(B * T, P, 72)
        if valid is None:
            valid = torch.ones(B, T, P, dtype=pred_root.dtype)
        predictions = {"smpl_pose": pred_pose}
        batch = {
            "smpl_pose": gt_pose,
            "has_smpl": valid.reshape(B * T, P),
            "temporal_shape": torch.tensor([B, T]),
        }
        return predictions, batch

    def test_matching_gt_motion_is_zero_even_during_turn(self):
        root = torch.zeros(1, 3, 2, 3)
        root[0, :, 0, 1] = torch.tensor([0.0, 0.3, 0.8])
        root[0, :, 1, 2] = torch.tensor([0.2, -0.1, -0.5])
        predictions, batch = self._inputs(root, root)

        losses = compute_temporal_smpl_smoothness(predictions, batch)

        self.assertLess(
            float(losses["loss_smpl_temporal_root_rotation"]), 1e-6
        )

    def test_prediction_only_root_jump_is_penalized_and_differentiable(self):
        gt_root = torch.zeros(1, 3, 1, 3)
        gt_root[0, :, 0, 1] = torch.tensor([0.0, 0.1, 0.2])
        pred_root = gt_root.clone()
        pred_root[0, 2, 0, 1] = 0.9
        predictions, batch = self._inputs(pred_root, gt_root)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch
        )["loss_smpl_temporal_root_rotation"]

        # The 0.7-rad final-step error is averaged with the exact first step.
        self.assertGreater(float(loss), 0.3)
        loss.backward()
        self.assertIsNotNone(predictions["smpl_pose"].grad)
        self.assertTrue(torch.isfinite(predictions["smpl_pose"].grad).all())

    def test_invalid_person_is_excluded(self):
        gt_root = torch.zeros(1, 3, 2, 3)
        pred_root = gt_root.clone()
        pred_root[0, 2, 1, 1] = 1.5
        valid = torch.ones(1, 3, 2)
        valid[..., 1] = 0.0
        predictions, batch = self._inputs(pred_root, gt_root, valid)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch
        )["loss_smpl_temporal_root_rotation"]

        self.assertLess(float(loss), 1e-6)

    def test_second_order_matching_motion_is_zero(self):
        root = torch.zeros(1, 4, 1, 3)
        root[0, :, 0, 1] = torch.tensor([0.0, 0.2, 0.5, 0.9])
        predictions, batch = self._inputs(root, root)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch, root_rotation_order=2
        )["loss_smpl_temporal_root_rotation"]

        self.assertLess(float(loss), 1e-6)


if __name__ == "__main__":
    unittest.main()
