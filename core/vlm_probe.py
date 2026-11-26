"""Zero-shot video-language probe: multiple-choice prompt, letter-logit class probabilities, evaluation against DINOv2."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.specialist_eval import evaluate_system

LETTERS = "ABCDEFGHIJKLMN"
#: Option text shown to the model; registered before any run (2026-10-28). Names only, no hand-written definitions.
CLASS_TEXT = {
    "Abuse": "Abuse", "Arrest": "Arrest", "Arson": "Arson", "Assault": "Assault", "Burglary": "Burglary",
    "Explosion": "Explosion", "Fighting": "Fighting", "Normal_Videos_event": "Normal activity (no crime)",
    "RoadAccidents": "Road accident", "Robbery": "Robbery", "Shooting": "Shooting", "Shoplifting": "Shoplifting",
    "Stealing": "Stealing", "Vandalism": "Vandalism",
}
ORDERS = ("forward", "reverse")


def option_order(classes: Sequence[str], order: str) -> list[str]:
    if len(classes) != len(LETTERS):
        raise ValueError(f"need {len(LETTERS)} classes, got {len(classes)}")
    if order == "forward":
        return list(classes)
    if order == "reverse":
        return list(reversed(classes))
    raise ValueError(f"unknown order {order!r}")


def build_question(options: Sequence[str], frames: int) -> str:
    lines = [f"These are {frames} frames, in time order, sampled from one surveillance video.",
             "Which one category best describes what happens in the video?"]
    lines += [f"{LETTERS[i]}. {CLASS_TEXT[c]}" for i, c in enumerate(options)]
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def letter_probs_to_classes(letter_logits: Sequence[float], options: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Softmax over the 14 letter logits, returned in `classes` order."""

    z = np.asarray(letter_logits, dtype=np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    out = np.zeros(len(classes))
    for i, c in enumerate(options):
        out[list(classes).index(c)] = p[i]
    return out


def evaluate(classes: Sequence[str], runs: Mapping, vlm: Mapping[str, np.ndarray], weight: float) -> dict:
    """`runs`: core.confusion_groups.load_runs output (DINOv2). `vlm`: video id -> class probabilities (order-averaged)."""

    base, alone, fused = {}, {}, {}
    for key, r in runs.items():
        missing = [v for v in r["videos"] if v not in vlm]
        if missing:
            raise KeyError(f"{key}: no VLM probabilities for {missing[:3]}")
        v = np.stack([vlm[x] for x in r["videos"]])
        truth = np.asarray(r["truth"])
        base[key] = (truth, np.asarray(r["probs"]).argmax(1))
        alone[key] = (truth, v.argmax(1))
        fused[key] = (truth, ((1 - weight) * np.asarray(r["probs"]) + weight * v).argmax(1))
    return {"vlm_alone": evaluate_system(classes, base, alone),
            f"fusion_w{weight}": evaluate_system(classes, base, fused)}
