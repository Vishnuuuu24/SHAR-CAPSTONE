import csv
import io
import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from eval.image_quality import psnr, ssim
from eval.image_quality_torch import psnr_batch, ssim_batch
from models.denoising_vae import ConvDenoisingVAE, denoiser_loss

REPO = Path(__file__).resolve().parents[1]
SSIM_OPTS = dict(data_range=1.0, channel_axis=2, win_size=11, gaussian_weights=True, sigma=1.5,
                 use_sample_covariance=False, k1=0.01, k2=0.03)


def _images(seed, n=3, h=48, w=64):
    rng = np.random.default_rng(seed)
    base = np.cumsum(np.cumsum(rng.normal(size=(n, h, w, 3)), 1), 2)
    base = (base - base.min()) / (base.max() - base.min())
    noisy = np.clip(base + rng.normal(scale=0.08, size=base.shape), 0, 1)
    return base.astype(np.float32), noisy.astype(np.float32)


class TorchQualityMatchesReference(unittest.TestCase):
    def test_ssim_and_psnr_equal_skimage(self):
        clean, noisy = _images(0)
        ct, nt = (torch.from_numpy(a).permute(0, 3, 1, 2) for a in (clean, noisy))
        got_s, got_p = ssim_batch(ct, nt, **SSIM_OPTS), psnr_batch(ct, nt, data_range=1.0)
        for i in range(len(clean)):
            self.assertAlmostEqual(got_s[i], ssim(clean[i], noisy[i], **SSIM_OPTS), places=6)
            self.assertAlmostEqual(got_p[i], psnr(clean[i], noisy[i], data_range=1.0), places=6)

    def test_identical_images(self):
        clean, _ = _images(1, n=1)
        ct = torch.from_numpy(clean).permute(0, 3, 1, 2)
        self.assertEqual(psnr_batch(ct, ct.clone(), data_range=1.0), [math.inf])
        self.assertAlmostEqual(ssim_batch(ct, ct.clone(), **SSIM_OPTS)[0], 1.0, places=9)

    def test_unregistered_configuration_refused(self):
        clean, noisy = _images(2, n=1)
        ct, nt = (torch.from_numpy(a).permute(0, 3, 1, 2) for a in (clean, noisy))
        with self.assertRaises(ValueError):
            ssim_batch(ct, nt, **{**SSIM_OPTS, "use_sample_covariance": True})


def _tiny(variational):
    return ConvDenoisingVAE(base_channels=8, latent_channels=4, downsamplings=2, norm_groups=4, variational=variational)


class DenoisingVAE(unittest.TestCase):
    def test_shapes_range_and_deterministic_restore(self):
        torch.manual_seed(0)
        for variational in (True, False):
            m = _tiny(variational).eval()
            x = torch.rand(2, 3, 24, 32)
            out = m.restore(x)
            self.assertEqual(tuple(out.shape), (2, 3, 24, 32))
            self.assertTrue(bool((out >= 0).all() and (out <= 1).all()))
            self.assertTrue(torch.equal(out, m.restore(x)))

    def test_loss_finite_and_ablation_is_mse(self):
        torch.manual_seed(0)
        clean, noisy = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
        recon, mu, logvar = _tiny(True)(noisy)
        loss, parts = denoiser_loss(recon, clean, mu, logvar, log_sigma_min=-6.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(parts["kl"], 0.0)
        recon, mu, logvar = _tiny(False)(noisy)
        self.assertIsNone(logvar)
        loss, parts = denoiser_loss(recon, clean, mu, logvar)
        self.assertAlmostEqual(float(loss.detach()), float(((recon - clean) ** 2).mean().detach()), places=6)

    def test_small_subset_overfit(self):
        torch.manual_seed(0)
        clean = torch.rand(1, 3, 16, 16)
        noisy = (clean + 0.2 * torch.randn_like(clean)).clamp(0, 1)
        for variational in (True, False):
            m = _tiny(variational)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            first = None
            for _ in range(300):
                recon, mu, logvar = m(noisy)
                loss, parts = denoiser_loss(recon, clean, mu, logvar, log_sigma_min=-6.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                first = parts["mse"] if first is None else first
            self.assertLess(parts["mse"], 0.25 * first, msg=f"variational={variational}")


class NoisyCrops(unittest.TestCase):
    def _pool(self):
        from PIL import Image
        rng = np.random.default_rng(0)
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(buf, format="JPEG")
        return [{"video": "v", "frame": 3, "filepath": "Abuse/v.mp4", "file_digest": "a" * 64, "jpeg": buf.getvalue()}]

    def test_items_are_a_function_of_seed_and_index(self):
        from core.p2a_conventions import load_frozen_p2a_conventions
        from core.p2b_training import NoisyCropDataset
        specs = load_frozen_p2a_conventions(REPO / "configs/p2a_conventions.yaml").degradation_specs
        a = NoisyCropDataset(self._pool(), specs, crop=32, seed=0, noise_global_seed=1000, length=8)
        b = NoisyCropDataset(self._pool(), specs, crop=32, seed=0, noise_global_seed=1000, length=8)
        c = NoisyCropDataset(self._pool(), specs, crop=32, seed=1, noise_global_seed=1001, length=8)
        for i in range(8):
            n1, c1 = a[i]
            n2, c2 = b[i]
            self.assertTrue(torch.equal(n1, n2) and torch.equal(c1, c2))
            self.assertEqual(tuple(n1.shape), (3, 32, 32))
        self.assertFalse(all(torch.equal(a[i][0], c[i][0]) for i in range(8)))


class SharedTableReproducesClassicalRun(unittest.TestCase):
    def test_condition_table_and_summarise_match_committed_summary(self):
        from core.p2_downstream import condition_table
        from core.p2_pipeline import conditions, summarise
        from core.p2a_conventions import load_frozen_p2a_conventions
        d = REPO / "results/p2/p2-classical"
        if not (d / "summary.json").exists() or not (d / "quality.csv").exists():
            self.skipTest("classical results not present (quality.csv lists dataset frames and stays local)")
        conds = conditions(load_frozen_p2a_conventions(REPO / "configs/p2a_conventions.yaml"))
        quality = list(csv.DictReader((d / "quality.csv").open()))
        runs = json.loads((d / "classification_runs.json").read_text())
        got = json.loads(json.dumps(summarise(condition_table(conds, quality, runs, [1, 2, 3, 4]), conds[0].name)))
        self.assertEqual(got, json.loads((d / "summary.json").read_text()))


class PinnedComparator(unittest.TestCase):
    def _net(self):
        from models.nafnet_arch import NAFNet
        torch.manual_seed(0)
        return NAFNet(img_channel=3, width=8, middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])

    def test_upstream_layer_names_and_shape(self):
        net = self._net().eval()
        keys = set(net.state_dict())
        for k in ("intro.weight", "ending.weight", "encoders.0.0.norm1.weight", "encoders.0.0.beta",
                  "downs.1.weight", "middle_blks.0.sca.1.weight", "ups.0.0.weight", "decoders.1.0.conv5.bias"):
            self.assertIn(k, keys)
        x = torch.rand(2, 3, 20, 28)  # not a multiple of 4: padded then cropped back
        self.assertEqual(tuple(net(x).shape), (2, 3, 20, 28))

    def test_restorer_chunks_and_clamps(self):
        from models.restoration_comparator import PretrainedRestorer
        net = self._net()
        x = torch.rand(5, 3, 16, 16)
        a, b = PretrainedRestorer(net, chunk=2).restore(x), PretrainedRestorer(net, chunk=5).restore(x)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(bool((a >= 0).all() and (a <= 1).all()))

    def test_wrong_weight_digest_refused(self):
        import tempfile, yaml
        from models.restoration_comparator import load_comparator
        cfg = yaml.safe_load((REPO / "configs/p2b_pretrained_comparator.yaml").read_text())
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "w.pth").write_bytes(b"not the registered file")
            cfg["weights"]["path"] = "w.pth"
            (Path(d) / "c.yaml").write_text(yaml.safe_dump(cfg))
            with self.assertRaises(RuntimeError):
                load_comparator(Path(d) / "c.yaml", Path(d), "cpu")


def _adaptive_median_reference(img: np.ndarray, s_max: int) -> np.ndarray:
    """Per-pixel loop of the textbook two-level algorithm (Gonzalez & Woods sec. 5.3.3), replicate borders."""
    h, w = img.shape
    out = img.copy()
    for y in range(h):
        for x in range(w):
            for k in range(3, s_max + 1, 2):
                p = k // 2
                win = img[np.clip(np.arange(y - p, y + p + 1), 0, h - 1)][:, np.clip(np.arange(x - p, x + p + 1), 0, w - 1)]
                zmin, zmed, zmax = win.min(), np.median(win), win.max()
                if zmin < zmed < zmax:
                    out[y, x] = img[y, x] if zmin < img[y, x] < zmax else zmed
                    break
            else:
                out[y, x] = zmed
    return out


class AdaptiveMedian(unittest.TestCase):
    def test_matches_textbook_loop(self):
        from models.adaptive_median import adaptive_median
        rng = np.random.default_rng(0)
        img = rng.random((2, 3, 17, 23)).astype(np.float32)
        img = np.round(img * 8) / 8  # ties and flat regions exercise every branch
        mask = rng.random(img.shape)
        img[mask < 0.1], img[mask > 0.9] = 0.0, 1.0
        got = adaptive_median(torch.from_numpy(img), 7).numpy()
        for n in range(2):
            for c in range(3):
                np.testing.assert_array_equal(got[n, c], _adaptive_median_reference(img[n, c], 7))

    def test_leaves_non_impulse_pixels(self):
        from models.adaptive_median import adaptive_median
        x = torch.linspace(0.2, 0.8, 16 * 16).reshape(1, 1, 16, 16)
        y = x.clone()
        y[0, 0, 5, 5], y[0, 0, 9, 3] = 1.0, 0.0
        out = adaptive_median(y, 7)
        self.assertLess(float(out[0, 0, 5, 5]), 1.0)
        self.assertGreater(float(out[0, 0, 9, 3]), 0.0)
        untouched = torch.ones_like(y, dtype=torch.bool)
        untouched[0, 0, 4:7, 4:7] = False
        untouched[0, 0, 8:11, 2:5] = False
        untouched[0, 0, 0, :] = untouched[0, 0, -1, :] = untouched[0, 0, :, 0] = untouched[0, 0, :, -1] = False
        self.assertTrue(torch.equal(out[untouched], y[untouched]))


class ExtraRestorers(unittest.TestCase):
    def test_registered_weights_load_strictly_and_run(self):
        import yaml
        from models.restoration_comparator import load_restorers
        cfg = yaml.safe_load((REPO / "configs/p2b_extra_restorers.yaml").read_text())
        if not all((REPO / e["weights"]["path"]).exists() for e in cfg["restorers"] if "weights" in e):
            self.skipTest("restorer weight files not present")
        loaded = load_restorers(REPO / "configs/p2b_extra_restorers.yaml", REPO, "cpu")
        self.assertEqual([n for n, _, _ in loaded], [e["name"] for e in cfg["restorers"]])
        x = torch.rand(2, 3, 64, 72)
        for name, restorer, _ in loaded:
            out = restorer.restore(x)
            self.assertEqual(tuple(out.shape), (2, 3, 64, 72), msg=name)
            self.assertTrue(bool((out >= 0).all() and (out <= 1).all()), msg=name)


if __name__ == "__main__":
    unittest.main()
