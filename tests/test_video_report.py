import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.demo_inference import resolve_video_key
from core.video_report import (
    apply_label,
    canonical_label,
    console_line,
    discover_videos,
    merge_windows,
    read_labels_csv,
    render_html,
    summarize,
    write_outputs,
)
from core.video_selfcheck import frame_cosines

CLASSES = ["Abuse", "Fighting", "Normal_Videos_event", "RoadAccidents"]


def result(name="a.mp4", prediction="Fighting", label=None, fair=True, expected=None):
    r = {"filename": name, "video_key": Path(name).stem, "prediction": prediction, "confidence": 0.6, "fair": fair, "expected": expected,
         "top5": [{"label": prediction, "prob": 0.6}, {"label": "Abuse", "prob": 0.2}], "mode": "all 12 heads", "frames_used": 16,
         "duration_s": 3.0, "timing_s": {"total": 1.0}, "device": "cpu", "recognised_by": None, "video_sha256": "d",
         "windows": [{"start_s": 0.0, "end_s": 1.0, "label": prediction, "prob": 0.6}], "thumbnails": []}
    return apply_label(r, label)


class Identity(unittest.TestCase):
    MEMBERSHIP = {"videos": {"Abuse014_x264": {"label": "Abuse", "seen_by_folds": [1, 3, 4]}}}

    def test_a_renamed_protocol_video_is_recognised_by_contents(self):
        self.assertEqual(resolve_video_key("holiday", "abc", self.MEMBERSHIP, {"abc": "Abuse014_x264"}), ("Abuse014_x264", "file contents"))

    def test_protocol_name_without_matching_contents_stays_protocol(self):
        self.assertEqual(resolve_video_key("Abuse014_x264", "zzz", self.MEMBERSHIP, {}), ("Abuse014_x264", "file name"))

    def test_unknown_file_is_unseen_footage(self):
        self.assertEqual(resolve_video_key("clip", "zzz", self.MEMBERSHIP, {}), ("clip", None))


class RegisteredFrames(unittest.TestCase):
    def test_frames_take_the_jpeg_round_trip_the_cache_took(self):
        import io

        from PIL import Image

        from core.demo_inference import registered_image

        frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(frame, mode="RGB").save(buffer, format="JPEG", quality=95)
        expected = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
        got = np.asarray(registered_image(frame))
        np.testing.assert_array_equal(got, expected)
        self.assertFalse(np.array_equal(got, frame))


class Labels(unittest.TestCase):
    def test_canonical_label_accepts_case_and_aliases(self):
        self.assertEqual(canonical_label("fighting", CLASSES), "Fighting")
        self.assertEqual(canonical_label("Normal", CLASSES), "Normal_Videos_event")
        self.assertEqual(canonical_label("road accident", CLASSES), "RoadAccidents")

    def test_unknown_label_lists_the_choices(self):
        with self.assertRaisesRegex(ValueError, "Abuse, Fighting"):
            canonical_label("Violence", CLASSES)

    def test_labels_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "labels.csv"
            path.write_text("video,label\nsub/a.mp4,fighting\n")
            self.assertEqual(read_labels_csv(path, CLASSES), {"a.mp4": "Fighting"})
            path.write_text("name,label\na.mp4,fighting\n")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_labels_csv(path, CLASSES)

    def test_dataset_label_beats_a_conflicting_given_label(self):
        r = result(expected={"label": "Abuse"}, label="Fighting")
        self.assertEqual(r["label"], "Abuse")
        self.assertFalse(r["correct"])
        self.assertIn("you gave Fighting", r["label_note"])


class Summary(unittest.TestCase):
    def test_only_fair_labelled_videos_are_scored(self):
        rows = [result("a.mp4", "Fighting", "Fighting"), result("b.mp4", "Abuse", "Fighting"), result("c.mp4", "Fighting", None),
                result("d.mp4", "Fighting", "Fighting", fair=False), {"filename": "e.mp4", "error": "bad"}]
        for number, row in enumerate(rows):
            row["video_sha256"] = str(number)
        s = summarize(rows)
        self.assertEqual((s["videos"], s["scored"], s["unlabelled"], s["excluded_seen_by_all_heads"], s["errors"]), (5, 2, 1, 1, 1))
        self.assertAlmostEqual(s["accuracy"], 0.5)
        self.assertAlmostEqual(s["per_class_f1"]["Fighting"], 2 / 3)
        self.assertTrue(s["small_sample"])

    def test_the_same_file_under_two_names_counts_once(self):
        a, b, c = result("a.mp4", "Fighting", "Fighting"), result("copy.mp4", "Fighting", "Fighting"), result("c.mp4", "Abuse", "Fighting")
        a["video_sha256"], b["video_sha256"], c["video_sha256"] = "same", "same", "other"
        s = summarize([a, b, c])
        self.assertEqual((s["scored"], s["duplicates"]), (2, 1))
        self.assertAlmostEqual(s["accuracy"], 0.5)

    def test_nothing_to_score(self):
        s = summarize([result()])
        self.assertIsNone(s["accuracy"])
        self.assertEqual(s["scored"], 0)

    def test_merge_windows(self):
        w = [{"start_s": 0, "end_s": 1, "label": "A"}, {"start_s": 1, "end_s": 2, "label": "A"}, {"start_s": 2, "end_s": 3, "label": "B"}]
        self.assertEqual(merge_windows(w), [{"start_s": 0, "end_s": 2, "label": "A"}, {"start_s": 2, "end_s": 3, "label": "B"}])


class Output(unittest.TestCase):
    def test_discover_videos(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "sub").mkdir()
            for name in ("b.MP4", "sub/a.avi", "notes.txt"):
                (root / name).write_bytes(b"x")
            self.assertEqual([p.name for p in discover_videos([root])], ["b.MP4", "a.avi"])
            with self.assertRaises(FileNotFoundError):
                discover_videos([root / "missing.mp4"])
            (root / "empty").mkdir()
            with self.assertRaises(FileNotFoundError):
                discover_videos([root / "empty"])

    def test_report_escapes_file_names(self):
        rows = [result("<script>alert(1)</script>.mp4", "Fighting", "Fighting")]
        page = render_html(rows, summarize(rows), {"git_revision": "abc"}, CLASSES)
        self.assertNotIn("<script>", page)
        self.assertIn("not a measurement", page.replace("an example, not a measurement", "not a measurement"))
        self.assertIn("inherited video-level labels", page)

    def test_failed_video_and_console_lines(self):
        rows = [{"filename": "bad.mp4", "error": "no frames"}, result("a.mp4", "Fighting", "Abuse", fair=False)]
        self.assertIn("Could not process: no frames", render_html(rows, summarize(rows), {}, CLASSES))
        self.assertEqual(console_line(rows[0]), "bad.mp4: ERROR no frames")
        self.assertIn("WRONG", console_line(rows[1]))
        self.assertIn("NOT A FAIR TEST", console_line(rows[1]))

    def test_write_outputs_keeps_frames_out_of_the_data_files(self):
        rows = [result("a.mp4", "Fighting", "Fighting")]
        rows[0]["thumbnails"] = [{"t_s": 0.0, "image": "data:image/jpeg;base64,AAAA", "window_label": "Fighting"}]
        with tempfile.TemporaryDirectory() as d:
            paths = write_outputs(Path(d), rows, summarize(rows), {"git_revision": "abc"}, CLASSES)
            self.assertIn("AAAA", paths["report"].read_text())
            self.assertNotIn("AAAA", paths["json"].read_text())
            self.assertEqual(json.loads(paths["json"].read_text())["summary"]["scored"], 1)
            self.assertIn("a.mp4,Fighting", paths["csv"].read_text())


class SelfCheckHelpers(unittest.TestCase):
    def test_frame_cosines(self):
        a = np.array([[1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(frame_cosines(a, np.array([[2.0, 0.0], [-1.0, -1.0]])), [1.0, -1.0])


class ProtocolDigests(unittest.TestCase):
    def test_every_protocol_video_has_a_digest(self):
        root = Path(__file__).resolve().parents[1] / "results/demo"
        if not (root / "protocol_digests.json").exists():
            self.skipTest("protocol digest list not present (lists dataset videos; kept out of the public copy)")
        digests = json.loads((root / "protocol_digests.json").read_text())["digests"]
        videos = json.loads((root / "membership.json").read_text())["videos"]
        self.assertEqual(set(digests.values()), set(videos))
        self.assertEqual(len(digests), 700)


if __name__ == "__main__":
    unittest.main()
