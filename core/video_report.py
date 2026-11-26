"""Turn `DemoModel.predict` results into something readable: label handling, a fair score, and a one-file HTML report.

Nothing here touches a model, so all of it can be tested without a GPU. Only videos some unseen head could judge count
towards the score: a UCF protocol video seen by every fold is reported but excluded (`fair` is False), and unlabelled
videos have nothing to score against. Numbers from a handful of chosen videos are anecdotes, and the report says so.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
LABEL_ALIASES = {"normal": "Normal_Videos_event", "normal_videos": "Normal_Videos_event", "roadaccident": "RoadAccidents"}
SMALL_SAMPLE = 30
RECORDED_SCORE = ("Recorded performance on held-out UCF-Crime test videos: macro-F1 0.414, accuracy 41.8% (D-57, 4 folds x 3 seeds), "
                  "scored against inherited video-level labels, not event-only (D-40). Fair validation macro-F1 0.3835 (D-70).")
LIMITS = ("The model always answers with one of its 14 classes; it has no &ldquo;none of these&rdquo;. It was trained on 34 videos per class "
          "of 320x240 CCTV, so phone or movie footage is outside what was evaluated and its confidence means less there. It is weakest on "
          "Shooting, Fighting and Assault. On unseen footage it averages all 12 heads, an ensemble whose score was never measured separately.")


def discover_videos(paths: Iterable[Path]) -> list[Path]:
    """Files as given, plus every video under any given folder (sorted); raises if a path is missing or nothing is found."""

    found: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or folder: {path}")
    if not found:
        raise FileNotFoundError("no video files found")
    return list(dict.fromkeys(found))


def canonical_label(label: str, classes: Sequence[str]) -> str:
    """A class name in any capitalisation, or a known alias such as 'Normal'; raises and lists the classes otherwise."""

    wanted = label.strip().lower().replace(" ", "")
    wanted = LABEL_ALIASES.get(wanted, wanted).lower()
    for name in classes:
        if name.lower() == wanted:
            return name
    raise ValueError(f"unknown label {label!r}; use one of: {', '.join(classes)}")


def read_labels_csv(path: Path, classes: Sequence[str]) -> dict[str, str]:
    """{file name: class} from a CSV with `video` and `label` columns."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"video", "label"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path} needs the columns video,label")
        return {Path(row["video"]).name: canonical_label(row["label"], classes) for row in reader}


def apply_label(result: dict, given: str | None) -> dict:
    """Attach `label`, `label_source` and `correct`. A protocol video keeps its dataset label over anything given."""

    expected = result.get("expected")
    if expected:
        result.update(label=expected["label"], label_source="dataset (inherited video-level label)")
        if given and given != expected["label"]:
            result["label_note"] = f"you gave {given}, but this file is {result['video_key']}, labelled {expected['label']}"
    else:
        result.update(label=given, label_source="you" if given else None)
    result["correct"] = None if result["label"] is None else result["label"] == result["prediction"]
    return result


def summarize(results: Sequence[Mapping]) -> dict:
    """Counts, plus accuracy and macro-F1 over the classes present among the scored videos."""

    ok = [r for r in results if not r.get("error")]
    labelled = [r for r in ok if r.get("label")]
    fair = [r for r in labelled if r.get("fair", True)]
    scored = list({r.get("video_sha256") or id(r): r for r in reversed(fair)}.values())  # a copy under another name counts once
    summary = {"videos": len(results), "errors": len(results) - len(ok), "unlabelled": len(ok) - len(labelled),
               "excluded_seen_by_all_heads": len(labelled) - len(fair), "duplicates": len(fair) - len(scored), "scored": len(scored), "accuracy": None,
               "macro_f1_over_present_classes": None, "per_class_f1": {}, "small_sample": len(scored) < SMALL_SAMPLE}
    if not scored:
        return summary
    summary["accuracy"] = sum(r["prediction"] == r["label"] for r in scored) / len(scored)
    for name in sorted({r["label"] for r in scored}):
        tp = sum(r["label"] == name and r["prediction"] == name for r in scored)
        fp = sum(r["label"] != name and r["prediction"] == name for r in scored)
        fn = sum(r["label"] == name and r["prediction"] != name for r in scored)
        summary["per_class_f1"][name] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    summary["macro_f1_over_present_classes"] = sum(summary["per_class_f1"].values()) / len(summary["per_class_f1"])
    return summary


def merge_windows(windows: Sequence[Mapping]) -> list[dict]:
    """Join consecutive windows with the same label into one stretch."""

    merged: list[dict] = []
    for w in windows:
        if merged and merged[-1]["label"] == w["label"]:
            merged[-1]["end_s"] = w["end_s"]
        else:
            merged.append({"start_s": w["start_s"], "end_s": w["end_s"], "label": w["label"]})
    return merged


def console_line(result: Mapping) -> str:
    if result.get("error"):
        return f"{result['filename']}: ERROR {result['error']}"
    top = ", ".join(f"{t['label']} {t['prob']:.0%}" for t in result["top5"][:3])
    line = f"{result['filename']}: {result['prediction']} ({result['confidence']:.0%}) | top 3: {top}"
    if result.get("label"):
        line += f" | label {result['label']}: {'correct' if result['correct'] else 'WRONG'}"
    if not result.get("fair", True):
        line += " | NOT A FAIR TEST: every fold saw this video"
    return line


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _colour(label: str, classes: Sequence[str]) -> str:
    return f"hsl({classes.index(label) * 360 // len(classes)} 60% 50%)" if label in classes else "#888"


def _card(r: Mapping, classes: Sequence[str]) -> str:
    e = html.escape
    if r.get("error"):
        return f'<section class="card"><h2>{e(str(r["filename"]))}</h2><p class="bad">Could not process: {e(str(r["error"]))}</p></section>'
    verdict = ""
    if r.get("label"):
        verdict = (f'<p class="{"good" if r["correct"] else "bad"}">{"Correct" if r["correct"] else "Wrong"}: label is <b>{e(r["label"])}</b> '
                   f'<span class="dim">({e(r["label_source"])})</span></p>')
    note = f'<p class="warn">{e(r["label_note"])}</p>' if r.get("label_note") else ""
    if not r.get("fair", True):
        note += '<p class="warn">Not a fair test: every fold trained or validated on this video, so it is left out of the score.</p>'
    if r.get("recognised_by") == "file contents" and r["video_key"] not in str(r["filename"]):
        note += f'<p class="warn">Recognised by its contents as UCF video {e(r["video_key"])}, so it is not treated as new footage.</p>'
    bars = "".join(f'<div class="bar"><span class="name">{e(t["label"])}</span><span class="track"><span class="fill" '
                   f'style="width:{t["prob"] * 100:.1f}%;background:{_colour(t["label"], classes)}"></span></span>'
                   f'<span class="pct">{t["prob"]:.1%}</span></div>' for t in r["top5"])
    stretches = merge_windows(r["windows"])
    total = sum(s["end_s"] - s["start_s"] for s in stretches) or 1.0
    strip = "".join(f'<span style="flex:{max(s["end_s"] - s["start_s"], total / 200):.3f};background:{_colour(s["label"], classes)}" '
                    f'title="{_clock(s["start_s"])}-{_clock(s["end_s"])} {e(s["label"])}"></span>' for s in stretches)
    longest = sorted(stretches, key=lambda s: s["start_s"] - s["end_s"])[:6]
    listed = "".join(f'<li>{_clock(s["start_s"])}&ndash;{_clock(s["end_s"])}: {e(s["label"])}</li>' for s in sorted(longest, key=lambda s: s["start_s"]))
    thumbs = "".join(f'<figure><img src="{t["image"]}" alt="frame at {_clock(t["t_s"])}"><figcaption>{_clock(t["t_s"])} {e(t["window_label"])}</figcaption></figure>'
                     for t in r.get("thumbnails", []))
    return f"""<section class="card"><h2>{e(str(r["filename"]))}</h2>
<p class="answer"><b>{e(r["prediction"])}</b> <span class="dim">{r["confidence"]:.1%} confidence</span></p>{verdict}{note}
{bars}
<h3>What each stretch resembles</h3><div class="strip">{strip}</div><ul class="stretches">{listed}</ul>
<p class="dim">Diagnostic only: the model never saw temporal labels, so this does not locate an event.</p>
<div class="thumbs">{thumbs}</div>
<p class="dim">{e(r["mode"])}. {r["frames_used"]} frames from {_clock(r["duration_s"])} of video, {r["timing_s"]["total"]} s on {e(r["device"])}.</p></section>"""


def render_html(results: Sequence[Mapping], summary: Mapping, meta: Mapping, classes: Sequence[str]) -> str:
    """One self-contained page. It embeds video frames, so keep it out of version control."""

    e = html.escape
    if summary["scored"]:
        score = f'<p class="big">{summary["accuracy"]:.1%} accuracy on {summary["scored"]} video{"s" if summary["scored"] != 1 else ""}</p>'
        if summary["scored"] > 1:
            score += (f'<p>Macro-F1 {summary["macro_f1_over_present_classes"]:.3f} over the {len(summary["per_class_f1"])} classes present. '
                      "Not comparable with the 14-class score unless every class is covered.</p>")
        if summary["small_sample"]:
            score += f'<p class="warn">Fewer than {SMALL_SAMPLE} videos: treat this as an example, not a measurement, especially if the videos were hand-picked.</p>'
    else:
        score = "<p>No score: give labels to compare the answers against.</p>"
    skipped = [f'{summary["unlabelled"]} unlabelled'] if summary["unlabelled"] else []
    if summary["excluded_seen_by_all_heads"]:
        skipped.append(f'{summary["excluded_seen_by_all_heads"]} seen by every head')
    if summary["duplicates"]:
        skipped.append(f'{summary["duplicates"]} duplicate file{"s" if summary["duplicates"] != 1 else ""}')
    if summary["errors"]:
        skipped.append(f'{summary["errors"]} failed')
    if skipped:
        score += f'<p class="dim">Not scored: {e(", ".join(skipped))}.</p>'
    provenance = " &middot; ".join(f"{e(str(k))}: {e(str(v))}" for k, v in meta.items())
    cards = "\n".join(_card(r, classes) for r in results)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SHAR Video Test</title><style>
:root{{--bg:#f6f6f4;--card:#fff;--ink:#1c1c1a;--dim:#666;--line:#ddd;--good:#17693a;--bad:#a32020;--warn:#7a5200}}
@media (prefers-color-scheme:dark){{:root{{--bg:#191917;--card:#242422;--ink:#eee;--dim:#a5a5a0;--line:#3a3a37;--good:#6fcf97;--bad:#f28b82;--warn:#e5c07b}}}}
body{{margin:0 auto;max-width:880px;padding:16px;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:16px 0;overflow-wrap:anywhere}}
h1{{font-size:22px;margin:8px 0}}h2{{font-size:16px;margin:0 0 8px}}h3{{font-size:14px;margin:16px 0 6px}}p{{margin:6px 0}}
.dim{{color:var(--dim);font-size:13px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.warn{{color:var(--warn)}}.big,.answer{{font-size:20px}}
.bar{{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:13px}}.name{{width:150px;flex:none}}.pct{{width:48px;flex:none;text-align:right}}
.track{{flex:1;height:10px;background:var(--line);border-radius:5px;overflow:hidden}}.fill{{display:block;height:100%}}
.strip{{display:flex;height:16px;border-radius:4px;overflow:hidden;gap:1px}}.stretches{{margin:6px 0;padding-left:18px;font-size:13px}}
.thumbs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:12px 0}}figure{{margin:0}}img{{width:100%;display:block;border-radius:4px}}figcaption{{font-size:12px;color:var(--dim)}}
</style></head><body><h1>SHAR video test</h1>
<section class="card">{score}<p class="dim">{RECORDED_SCORE}</p><p class="dim">{LIMITS}</p></section>
{cards}
<p class="dim">{provenance}</p></body></html>
"""


def write_outputs(out_dir: Path, results: Sequence[Mapping], summary: Mapping, meta: Mapping, classes: Sequence[str]) -> dict[str, Path]:
    """report.html (with frames), results.json and results.csv (without), so the numbers stay small and easy to read."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "thumbnails"} for r in results]
    paths = {"report": out_dir / "report.html", "json": out_dir / "results.json", "csv": out_dir / "results.csv"}
    paths["report"].write_text(render_html(results, summary, meta, classes), encoding="utf-8")
    paths["json"].write_text(json.dumps({"meta": dict(meta), "summary": dict(summary), "results": slim}, indent=1) + "\n", encoding="utf-8")
    columns = ["filename", "prediction", "confidence", "label", "label_source", "correct", "fair", "recognised_by", "video_sha256", "error"]
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(slim)
    return paths
