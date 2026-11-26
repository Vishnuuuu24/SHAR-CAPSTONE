"""Diagnose NAFNet-SIDD behaviour on UCF frames: checkpoint keys, pre-clamp range, RGB vs BGR, noise sweep."""
import sys, json
from pathlib import Path
REPO = Path.home() / "SHAR"
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
from models.nafnet_arch import NAFNet
from core.p2_pipeline import decode_rgb_float
from core.subset7 import FAMILY
from core.validation_frames import select_validation_frames
from eval.image_quality_torch import psnr_batch

dev = torch.device("cuda:0")
cfg = yaml.safe_load((REPO / "configs/p2b_pretrained_comparator.yaml").read_text())
ck = torch.load(REPO / cfg["weights"]["path"], map_location="cpu", weights_only=True)
print(json.dumps({"checkpoint_keys": list(ck)[:10]}))
a = cfg["architecture"]
nets = {}
for key in ck:
    if not isinstance(ck[key], dict):
        continue
    n = NAFNet(img_channel=3, width=a["width"], middle_blk_num=a["middle_blk_num"], enc_blk_nums=a["enc_blk_nums"], dec_blk_nums=a["dec_blk_nums"])
    n.load_state_dict(ck[key], strict=True); nets[key] = n.to(dev).eval()
videos = select_validation_frames(REPO, [1], sorted(FAMILY), 32)
v = sorted(videos)[0]; meta = videos[v]
clean = np.stack([decode_rgb_float(meta["dataset"].raw_jpeg(i)) for i in meta["rows"][:8]])
x = torch.from_numpy(clean).permute(0, 3, 1, 2).to(dev)
g = torch.Generator(device=dev).manual_seed(0)
for key, net in nets.items():
    for order in ("rgb", "bgr"):
        for sigma in (0, 5, 10, 25, 50):
            noisy = (x + torch.randn(x.shape, generator=g, device=dev) * sigma / 255).clamp(0, 1)
            inp = noisy if order == "rgb" else noisy.flip(1)
            with torch.no_grad():
                out = net(inp)
            out = out if order == "rgb" else out.flip(1)
            print(json.dumps({"weights": key, "order": order, "sigma": sigma,
                              "in_psnr": round(float(np.mean(psnr_batch(x, noisy, data_range=1.0) if sigma else [99])), 2),
                              "out_psnr": round(float(np.mean(psnr_batch(x, out.clamp(0, 1), data_range=1.0))), 2),
                              "out_min": round(float(out.min()), 3), "out_max": round(float(out.max()), 3),
                              "out_minus_in_absmean": round(float((out - inp if order == 'rgb' else out - noisy).abs().mean()), 4)}), flush=True)
