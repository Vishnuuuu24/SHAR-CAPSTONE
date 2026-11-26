import unittest

import torch

from models.vjepa2_backbone import convert_encoder_keys, encoder_config


class Vjepa2KeyConversion(unittest.TestCase):
    def test_renames_and_qkv_split(self):
        d = 4
        qkv_w = torch.arange(3 * d * d, dtype=torch.float32).reshape(3 * d, d)
        qkv_b = torch.arange(3 * d, dtype=torch.float32)
        original = {
            "module.backbone.blocks.0.attn.qkv.weight": qkv_w,
            "module.backbone.blocks.0.attn.qkv.bias": qkv_b,
            "module.backbone.blocks.0.attn.proj.weight": torch.zeros(d, d),
            "module.backbone.blocks.3.mlp.fc1.bias": torch.zeros(d),
            "module.backbone.patch_embed.proj.weight": torch.zeros(d, 3, 2, 16, 16),
            "module.backbone.norm.weight": torch.zeros(d),
        }
        out = convert_encoder_keys(original, emb_dim=d)
        self.assertEqual(
            set(out),
            {"layer.0.attention.query.weight", "layer.0.attention.key.weight", "layer.0.attention.value.weight",
             "layer.0.attention.query.bias", "layer.0.attention.key.bias", "layer.0.attention.value.bias",
             "layer.0.attention.proj.weight", "layer.3.mlp.fc1.bias", "embeddings.patch_embeddings.proj.weight",
             "layernorm.weight"})
        self.assertTrue(torch.equal(out["layer.0.attention.key.weight"], qkv_w[d:2 * d]))
        self.assertTrue(torch.equal(out["layer.0.attention.value.bias"], qkv_b[2 * d:]))

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            convert_encoder_keys({"module.backbone.cls_token": torch.zeros(1)}, emb_dim=4)

    def test_config_matches_input(self):
        c = encoder_config()
        self.assertEqual((c.crop_size, c.frames_per_clip, c.hidden_size, c.num_hidden_layers), (224, 16, 1024, 24))


if __name__ == "__main__":
    unittest.main()
