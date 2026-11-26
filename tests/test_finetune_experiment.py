import unittest

import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer

from core.d65_claim import holm
from core.finetune_experiment import ARMS, CONTRASTS, TRAINABLE_BLOCKS, index_source, select_lr
from core.p1c_head_variants import group_pooled_sequences_by_video
from core.vit_split import CachedTailHead, Tail, frozen_digest, state_digest, trunk_tokens
from models.dinov2_backbone import pool_cls_plus_mean_patch


def tiny_vit(depth: int = 6) -> VisionTransformer:
    torch.manual_seed(0)
    return VisionTransformer(img_size=28, patch_size=14, embed_dim=32, depth=depth, num_heads=2, reg_tokens=4,
                             no_embed_class=True, num_classes=0, dynamic_img_size=True).eval()


class Split(unittest.TestCase):
    def test_trunk_plus_tail_equals_whole(self):
        net = tiny_vit()
        x = torch.rand(3, 3, 28, 28)
        with torch.no_grad():
            whole = pool_cls_plus_mean_patch(net.forward_features(x), net.num_prefix_tokens)
            for split in (0, 2, 4, 6):
                tail = Tail(net, split, 0, net.num_prefix_tokens).eval()
                torch.testing.assert_close(tail(trunk_tokens(net, x, split)), whole, rtol=0, atol=1e-6)

    def test_trainable_set_is_last_k_blocks_and_norm(self):
        net = tiny_vit()
        tail = Tail(net, 2, 2, net.num_prefix_tokens)
        names = tail.trainable_names()
        self.assertTrue(names)
        self.assertTrue(all(n.startswith(("blocks.2.", "blocks.3.", "norm.")) for n in names))
        self.assertTrue(any(n.startswith("norm.") for n in names))
        self.assertFalse(any(p.requires_grad for p in tail.blocks[0].parameters()))
        self.assertEqual({k.split(".")[0] + ("." + k.split(".")[1] if k.startswith("blocks") else "") for k in tail.trainable_state()},
                         {"blocks.2", "blocks.3", "norm"})
        frozen = Tail(net, 4, 0, net.num_prefix_tokens)
        self.assertEqual(frozen.trainable_names(), set())

    def test_bad_split_rejected(self):
        net = tiny_vit()
        with self.assertRaises(ValueError):
            Tail(net, 4, 3, net.num_prefix_tokens)

    def test_training_a_tail_never_changes_the_source_network(self):
        net = tiny_vit()
        before, trunk_before = state_digest(net.state_dict().items()), frozen_digest(net, 4)
        tail = Tail(net, 4, 2, net.num_prefix_tokens)
        opt = torch.optim.AdamW([p for p in tail.parameters() if p.requires_grad], lr=1e-2)
        h = trunk_tokens(net, torch.rand(2, 3, 28, 28), 4).detach()
        for _ in range(3):
            opt.zero_grad()
            tail.train()(h).pow(2).mean().backward()
            opt.step()
        self.assertEqual(state_digest(net.state_dict().items()), before)
        self.assertEqual(frozen_digest(net, 4), trunk_before)
        self.assertNotEqual(state_digest(tail.state_dict().items()), state_digest(Tail(net, 4, 2, net.num_prefix_tokens).state_dict().items()))

    def test_checkpointed_gradients_match_plain(self):
        net = tiny_vit()
        h = trunk_tokens(net, torch.rand(2, 3, 28, 28), 3).detach()
        grads = []
        for ckpt in (True, False):
            tail = Tail(net, 3, 3, net.num_prefix_tokens, grad_checkpoint=ckpt).train()
            tail(h).sum().backward()
            grads.append(torch.cat([p.grad.flatten() for p in tail.parameters() if p.requires_grad]))
        torch.testing.assert_close(grads[0], grads[1])


class CachedHead(unittest.TestCase):
    def test_index_lookup_equals_direct_features(self):
        net = tiny_vit()
        x = torch.rand(10, 3, 28, 28)
        with torch.no_grad():
            cache = trunk_tokens(net, x, 4)
            tail = Tail(net, 4, 2, net.num_prefix_tokens).eval()
            pooled = tail(cache)

        class SumHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Linear(64, 3)

            def forward(self, seq, key_padding_mask=None):
                keep = (~key_padding_mask).float().unsqueeze(-1)
                return self.w((seq * keep).sum(1))

        head = SumHead()
        model = CachedTailHead(tail, head, cache).eval()
        idx = torch.tensor([[[3.0], [7.0], [0.0]], [[1.0], [0.0], [0.0]]])
        mask = torch.tensor([[False, False, False], [False, True, True]])
        with torch.no_grad():
            got = model(idx, key_padding_mask=mask)
            want = head(pooled[idx[..., 0].long()], key_padding_mask=mask)
        torch.testing.assert_close(got, want)

    def test_chunked_tail_matches_unchunked_outputs_and_gradients(self):
        net = tiny_vit()
        cache = trunk_tokens(net, torch.rand(12, 3, 28, 28), 4).detach()

        class MeanHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Linear(64, 3)

            def forward(self, seq, key_padding_mask=None):
                return self.w(seq.mean(1))

        idx = torch.tensor([[[3.0], [7.0], [11.0]], [[1.0], [5.0], [0.0]]])
        outs, grads = [], []
        for chunk in (None, 2):
            torch.manual_seed(1)
            model = CachedTailHead(Tail(net, 4, 2, net.num_prefix_tokens).train(), MeanHead(), cache, chunk_frames=chunk)
            out = model(idx)
            out.sum().backward()
            outs.append(out.detach())
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad]))
        torch.testing.assert_close(outs[0], outs[1])
        torch.testing.assert_close(grads[0], grads[1])

    def test_index_source_preserves_the_controls_permutation(self):
        rows = [{"source_dataset": "ucf", "source_video_id": v, "source_frame_index": f, "target": t}
                for v, t in (("a", 0), ("b", 1)) for f in (5, 1, 9, 3)]
        feats = np.random.default_rng(0).random((len(rows), 6)).astype(np.float32)
        direct = group_pooled_sequences_by_video(feats, rows, permute_frame_order=True, permute_seed=11)
        via = group_pooled_sequences_by_video(index_source(rows)["train"][0], rows, permute_frame_order=True, permute_seed=11)
        for key in direct:
            np.testing.assert_array_equal(feats[via[key][0][:, 0].astype(int)], direct[key][0])


class Rules(unittest.TestCase):
    def test_select_lr_mean_then_lower(self):
        self.assertEqual(select_lr({1e-5: [0.4, 0.5], 3e-5: [0.6, 0.5], 1e-4: [0.2, 0.9]}), 3e-5)
        self.assertEqual(select_lr({1e-5: [0.5], 3e-5: [0.5]}), 1e-5)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.02]), [True, True])      # 0.01 <= 0.0125, then 0.02 <= 0.025
        self.assertEqual(holm([0.013, 0.001]), [True, True])
        self.assertEqual(holm([0.014, 0.02]), [False, False])   # smallest fails 0.0125, stop
        self.assertEqual(holm([0.001, 0.03]), [True, False])
        self.assertEqual(holm([0.02]), [True])

    def test_registered_design(self):
        self.assertEqual(ARMS, ("B_frozen", "B_ft2", "B_ft4"))
        self.assertEqual(CONTRASTS, (("B_ft2", "B_frozen"), ("B_ft4", "B_frozen")))
        self.assertEqual((TRAINABLE_BLOCKS["B_ft2"], TRAINABLE_BLOCKS["B_ft4"]), (2, 4))


if __name__ == "__main__":
    unittest.main()
