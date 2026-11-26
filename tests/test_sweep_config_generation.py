"""Tests for `scripts.sweep_p1c_resolution_batch` config generation.

The sweep itself needs a GPU and the real blob, so what is testable here is the
part that would silently corrupt every result: whether each generated config
actually carries the resolution and batch size its filename claims, and whether
the grid really is ordered cheapest-first (the property that keeps a
device-memory refusal at the end rather than aborting cheaper points).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep_p1c_resolution_batch import (  # noqa: E402
    BASE_CONFIG, NATIVE_HEIGHT, NATIVE_WIDTH, geometry_label, grid, parse_geometry,
    write_config,
)


class GridOrderTests(unittest.TestCase):
    def test_points_ascend_by_activation_cost(self) -> None:
        points = grid(("64", "112", "224"), (64, 256))
        costs = [h * w * b for h, w, b in points]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual(points[0], (64, 64, 64))
        self.assertEqual(points[-1], (224, 224, 256))

    def test_grid_covers_every_combination_exactly_once(self) -> None:
        points = grid(("64", "112"), (64, 256))
        self.assertEqual(len(points), 4)
        self.assertEqual(len(set(points)), 4)

    def test_native_sorts_above_224_because_it_has_more_pixels(self) -> None:
        # The corpus is 240x320 = 76,800 px against 224x224 = 50,176. "Native"
        # is the most expensive geometry available, not a middle ground.
        points = grid(("224", "native"), (64,))
        self.assertEqual(points, [(224, 224, 64), (240, 320, 64)])


class GeometryParsingTests(unittest.TestCase):
    def test_square_shorthand(self) -> None:
        self.assertEqual(parse_geometry("224"), (224, 224))
        self.assertEqual(parse_geometry(64), (64, 64))

    def test_explicit_height_x_width(self) -> None:
        self.assertEqual(parse_geometry("240x320"), (240, 320))
        self.assertEqual(parse_geometry("240X320"), (240, 320))

    def test_native_is_the_corpus_geometry(self) -> None:
        self.assertEqual(parse_geometry("native"), (NATIVE_HEIGHT, NATIVE_WIDTH))
        self.assertEqual(parse_geometry("native"), (240, 320))

    def test_labels_keep_square_ids_stable(self) -> None:
        # Existing artifacts embed "R224"; a rename would orphan them.
        self.assertEqual(geometry_label(224, 224), "224")
        self.assertEqual(geometry_label(240, 320), "240x320")


class ConfigGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.base = BASE_CONFIG.read_text(encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self, resolution, batch: int) -> dict:
        height, width = parse_geometry(resolution)
        path = write_config(height, width, batch, base_text=self.base, output_dir=self.out)
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_generated_config_carries_the_requested_geometry(self) -> None:
        for resolution, batch in ((64, 64), (112, 256), (224, 64)):
            with self.subTest(resolution=resolution, batch=batch):
                loaded = self._load(resolution, batch)
                pre = loaded["fixed_protocol"]["preprocessing"]
                self.assertEqual(pre["resize_height"], resolution)
                self.assertEqual(pre["resize_width"], resolution)
                self.assertEqual(loaded["fixed_protocol"]["training"]["batch_size"], batch)

    def test_native_config_is_non_square_in_the_right_order(self) -> None:
        # H and W must not be transposed: 240 tall by 320 wide, not the reverse.
        pre = self._load("native", 64)["fixed_protocol"]["preprocessing"]
        self.assertEqual(pre["resize_height"], 240)
        self.assertEqual(pre["resize_width"], 320)

    def test_precision_stays_bfloat16_across_the_grid(self) -> None:
        # The sweep varies geometry only; precision was decided separately.
        for resolution, batch in ((64, 64), (224, 256)):
            loaded = self._load(resolution, batch)
            self.assertEqual(loaded["fixed_protocol"]["training"]["precision"], "amp_bfloat16")

    def test_each_point_gets_a_distinct_experiment_id(self) -> None:
        ids = {self._load(f"{h}x{w}" if h != w else str(h), b)["experiment"]["id"]
               for h, w, b in grid(("64", "112"), (64, 256))}
        self.assertEqual(len(ids), 4)

    def test_batch_substitution_does_not_disturb_the_default_resolution(self) -> None:
        # Both fields start at 64 in the base config, so a careless string
        # replace could rewrite resolution when only batch was meant to change.
        loaded = self._load(224, 64)
        self.assertEqual(loaded["fixed_protocol"]["preprocessing"]["resize_height"], 224)
        self.assertEqual(loaded["fixed_protocol"]["training"]["batch_size"], 64)

    def test_manifest_digests_are_untouched(self) -> None:
        # The sweep must not change what data is read; only how it is shaped.
        base = yaml.safe_load(self.base)["fixed_protocol"]["data"]
        loaded = self._load(112, 256)["fixed_protocol"]["data"]
        self.assertEqual(loaded["train"]["sha256"], base["train"]["sha256"])
        self.assertEqual(loaded["validation"]["sha256"], base["validation"]["sha256"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
