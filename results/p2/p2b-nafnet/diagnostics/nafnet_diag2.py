"""Is NAFNet's sigma=50 blow-up numerical (TF32) or genuine? Same frames, three precisions."""
import sys, json
from pathlib import Path
REPO = Path.home() / "SHAR"; sys.path.insert(0, str(REPO))
import numpy as np, torch
from models.restoration_comparator import load_comparator
from core.p2_pipeline import decode_rgb_float
from core.subset7 import FAMILY
from core.validation_frames import select_validation_frames
from eval.image_quality_torch import psnr_batch
dev = torch.device("cuda:0")
_, r, _ = load_comparator(REPO / "configs/p2b_pretrained_comparator.yaml", REPO, dev)
net = r.net
videos = select_validation_frames(REPO, [1], sorted(FAMILY), 32); meta = videos[sorted(videos)[0]]
x = torch.from_numpy(np.stack([decode_rgb_float(meta["dataset"].raw_jpeg(i)) for i in meta["rows"][:2]])).permute(0, 3, 1, 2).to(dev)
for sigma in (25, 50):
    g = torch.Generator(device=dev).manual_seed(0)
    noisy = (x + torch.randn(x.shape, generator=g, device=dev) * sigma / 255).clamp(0, 1)
    for mode in ("tf32_on", "tf32_off", "float64"):
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32_on"; torch.backends.cudnn.allow_tf32 = mode == "tf32_on"
        m, inp = (net.double(), noisy.double()) if mode == "float64" else (net.float(), noisy)
        with torch.no_grad():
            out = m(inp).float()
        print(json.dumps({"sigma": sigma, "mode": mode, "out_psnr": round(float(np.mean(psnr_batch(x, out.clamp(0, 1), data_range=1.0))), 2),
                          "out_min": round(float(out.min()), 3), "out_max": round(float(out.max()), 3)}), flush=True)
    net.float()
