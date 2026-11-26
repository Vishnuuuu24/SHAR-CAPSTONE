"""Pinned pretrained restorers for RESULT 1: the NAFNet comparator (D-61) and the D-62 follow-ups (configs/p2b_extra_restorers.yaml)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from models.nafnet_arch import NAFNet


class PretrainedRestorer:
    """Published weights used as-is: RGB [0,1] in, output clamped to [0,1], fixed-size chunks per forward.

    `pad_multiple` > 1 pads H and W up to that multiple (as the model's own test script does) and crops back.
    """

    def __init__(self, net: torch.nn.Module, chunk: int, pad_multiple: int = 1, pad_mode: str = "reflect") -> None:
        self.net, self.chunk, self.pad_multiple, self.pad_mode = net.eval(), chunk, pad_multiple, pad_mode

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.pad_multiple, (-w) % self.pad_multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode=self.pad_mode)
        return self.net(x)[..., :h, :w].clamp_(0.0, 1.0)

    @torch.no_grad()
    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._forward(x[i:i + self.chunk]) for i in range(0, x.shape[0], self.chunk)])


def _verified_state(repo_root: Path, w: dict) -> tuple[dict, str]:
    path = Path(repo_root) / w["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != w["sha256"]:
        raise RuntimeError(f"{path} sha256 {digest} does not match the registered {w['sha256']}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return (state[w["state_dict_key"]] if w.get("state_dict_key") else state), digest


def _build(entry: dict) -> torch.nn.Module:
    a = entry["architecture"]
    if entry["type"] == "scunet":
        from models.scunet_arch import SCUNet
        return SCUNet(in_nc=int(a["in_nc"]), config=list(a["config"]), dim=int(a["dim"]))
    if entry["type"] == "restormer":
        from models.restormer_arch import Restormer
        return Restormer(**a)
    raise ValueError(f"unknown restorer type {entry['type']!r}")


def load_restorers(config_path: Path, repo_root: Path, device) -> list[tuple[str, object, dict]]:
    """Every entry of a multi-restorer config -> (name, restorer, provenance); weights sha256-checked, strict load."""

    from models.adaptive_median import AdaptiveMedianRestorer

    cfg = yaml.safe_load(Path(config_path).read_text())
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    out = []
    for e in cfg["restorers"]:
        if e["type"] == "adaptive_median":
            out.append((e["name"], AdaptiveMedianRestorer(int(e["s_max"]), int(e["frames_per_forward"])),
                        {"config_sha256": config_sha, "s_max": int(e["s_max"])}))
            continue
        net = _build(e)
        state, digest = _verified_state(repo_root, e["weights"])
        net.load_state_dict(state, strict=True)
        io = e["io"]
        out.append((e["name"], PretrainedRestorer(net.to(device), int(io["frames_per_forward"]), int(io.get("pad_multiple", 1)),
                                                  io.get("pad_mode", "reflect")),
                    {"config_sha256": config_sha, "weights_sha256": digest, "code_commit": e["code"]["commit"]}))
    return out


def load_comparator(config_path: Path, repo_root: Path, device) -> tuple[str, PretrainedRestorer, dict]:
    """(name, restorer, provenance). Refuses a weight file whose sha256 differs from the registered one."""

    cfg = yaml.safe_load(Path(config_path).read_text())
    w = cfg["weights"]
    path = Path(repo_root) / w["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != w["sha256"]:
        raise RuntimeError(f"{path} sha256 {digest} does not match the registered {w['sha256']}")
    a = cfg["architecture"]
    net = NAFNet(img_channel=3, width=int(a["width"]), middle_blk_num=int(a["middle_blk_num"]),
                 enc_blk_nums=list(a["enc_blk_nums"]), dec_blk_nums=list(a["dec_blk_nums"]))
    state = torch.load(path, map_location="cpu", weights_only=True)[w["state_dict_key"]]
    net.load_state_dict(state, strict=True)
    provenance = {"weights_sha256": digest, "code_commit": cfg["code"]["commit"],
                  "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()}
    return cfg["name"], PretrainedRestorer(net.to(device), int(cfg["io"]["frames_per_forward"])), provenance
