import json
import tempfile
import unittest
from pathlib import Path

from core.code_graph import impact, scan, topic_of
from scripts.build_code_graph import main


def write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class CodeGraphTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        write(root, "core/__init__.py", "")
        write(root, "core/detect_model.py", '"""Frame scorer. Implements D-71."""\ndef score():\n    """Score frames."""\n')
        write(root, "core/detect_metrics.py", '"""Metrics."""\nfrom core import detect_model\nfrom .detect_model import score\n')
        write(root, "scripts/run_detector.py", "from core.detect_metrics import x\nOUT = 'results/detect/run-d71'\n")
        write(root, "tests/test_detect.py", "from core import detect_metrics\n")
        write(root, "core/broken.py", "def (:\n")
        write(root, "docs/DECISIONS.md", "| D-71 | **Detector protocol.** more | why |\n")

    def tearDown(self):
        self.tmp.cleanup()

    def edges(self, graph, kind):
        return {(e["source"], e["target"]) for e in graph["edges"] if e["kind"] == kind}

    def test_imports_resolve_absolute_relative_and_submodule_forms(self):
        graph = scan(self.root)
        imports = self.edges(graph, "imports")
        self.assertIn(("core/detect_metrics.py", "core/detect_model.py"), imports)
        self.assertIn(("scripts/run_detector.py", "core/detect_metrics.py"), imports)
        self.assertIn(("tests/test_detect.py", "core/detect_metrics.py"), imports)
        self.assertNotIn(("core/detect_metrics.py", "core/detect_metrics.py"), imports)

    def test_decisions_results_docstrings_and_syntax_errors(self):
        graph = scan(self.root)
        nodes = {n["id"]: n for n in graph["nodes"]}
        self.assertIn(("core/detect_model.py", "D-71"), self.edges(graph, "cites"))
        self.assertEqual(nodes["D-71"]["summary"], "Detector protocol.")
        self.assertIn(("scripts/run_detector.py", "results/detect/run-d71"), self.edges(graph, "names result"))
        self.assertFalse(nodes["results/detect/run-d71"]["exists"])
        self.assertEqual(nodes["core/detect_model.py"]["functions"][0]["doc"], "Score frames.")
        self.assertIn("syntax error", nodes["core/broken.py"]["problem"])
        self.assertEqual(nodes["tests/test_detect.py"]["topic"], "Live detector (temporal VAD)")

    def test_tour_lists_dependencies_first_and_excludes_tests(self):
        tour = scan(self.root)["tours"]["Live detector (temporal VAD)"]
        self.assertLess(tour.index("core/detect_model.py"), tour.index("core/detect_metrics.py"))
        self.assertLess(tour.index("core/detect_metrics.py"), tour.index("scripts/run_detector.py"))
        self.assertNotIn("tests/test_detect.py", tour)

    def test_impact_is_transitive(self):
        result = impact(scan(self.root), ["core/detect_model.py"])
        self.assertEqual(result["tests"], ["tests/test_detect.py"])
        self.assertEqual(result["code"], ["core/detect_metrics.py", "scripts/run_detector.py"])

    def test_scan_is_deterministic_and_topics_match_prefixes(self):
        self.assertEqual(json.dumps(scan(self.root), sort_keys=True), json.dumps(scan(self.root), sort_keys=True))
        self.assertEqual(topic_of("run_p2_classical"), "Denoising (RESULT 1)")
        self.assertEqual(topic_of("run_rwf2000"), "RWF-2000 (separate binary task)")
        self.assertEqual(topic_of("inner_cv_report"), "Evaluation protocol")

    def test_cli_writes_viewer_with_embedded_graph(self):
        out = self.root / "out"
        self.assertEqual(main(["--root", str(self.root), "--output", str(out)]), 0)
        html = (out / "index.html").read_text()
        self.assertNotIn("__GRAPH_JSON__", html)
        self.assertIn('"schema": "shar-code-graph/1"', html)
        self.assertEqual(json.loads((out / "graph.json").read_text())["schema"], "shar-code-graph/1")


if __name__ == "__main__":
    unittest.main()
