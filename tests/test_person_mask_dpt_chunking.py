import unittest

import torch

from vggt.heads.person_mask_head import PersonMaskDPTHead


class PersonMaskDPTChunkingTest(unittest.TestCase):
    @staticmethod
    def make_head(chunk_size):
        return PersonMaskDPTHead(
            dim_in=64,
            query_dim=32,
            embed_dim=16,
            features=32,
            out_channels=[32, 32, 32, 32],
            intermediate_layer_idx=[0, 1, 2, 3],
            down_ratio=2,
            patch_size=14,
            frames_chunk_size=chunk_size,
        )

    @staticmethod
    def make_inputs(requires_grad=False):
        torch.manual_seed(7)
        person_tokens = torch.randn(1, 3, 32, requires_grad=requires_grad)
        # 28x28 with patch_size=14 -> four patch tokens plus two special tokens.
        token_layers = [
            torch.randn(1, 2, 6, 64, requires_grad=requires_grad)
            for _ in range(4)
        ]
        images = torch.rand(1, 2, 3, 28, 28)
        return person_tokens, token_layers, images

    def test_chunked_forward_matches_unchunked(self):
        full = self.make_head(chunk_size=8).eval()
        chunked = self.make_head(chunk_size=1).eval()
        chunked.load_state_dict(full.state_dict())
        person_tokens, token_layers, images = self.make_inputs()
        with torch.no_grad():
            expected = full(person_tokens, token_layers, images, patch_start_idx=2)
            actual = chunked(person_tokens, token_layers, images, patch_start_idx=2)
        self.assertEqual(tuple(actual.shape), (1, 2, 3, 14, 14))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_chunked_training_backward_is_finite(self):
        head = self.make_head(chunk_size=1).train()
        person_tokens, token_layers, images = self.make_inputs(requires_grad=True)
        logits = head(person_tokens, token_layers, images, patch_start_idx=2)
        logits.square().mean().backward()
        self.assertIsNotNone(head.pixel_proj.weight.grad)
        self.assertIsNotNone(head.trunk.projects[0].weight.grad)
        self.assertTrue(torch.isfinite(head.pixel_proj.weight.grad).all())
        self.assertTrue(torch.isfinite(head.trunk.projects[0].weight.grad).all())
        self.assertTrue(torch.isfinite(person_tokens.grad).all())


if __name__ == "__main__":
    unittest.main()
