#!/usr/bin/env python3
"""Build the SHAR code map (graph JSON + offline HTML viewer), or list what a change affects.

    python3 scripts/build_code_graph.py                      # writes docs/code_graph/{graph.json,index.html}
    python3 scripts/build_code_graph.py --impact core/detect_model.py   # importers and tests of a file

Open docs/code_graph/index.html in any browser; it needs no server and no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.code_graph import impact, scan  # noqa: E402

TEMPLATE = ROOT / "ui/code_graph_template.html"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/code_graph")
    parser.add_argument("--impact", nargs="+", metavar="FILE", help="repo-relative files; print what imports them")
    args = parser.parse_args(argv)
    graph = scan(args.root)
    if args.impact:
        print(json.dumps(impact(graph, args.impact), indent=1))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(graph, indent=1, sort_keys=True)
    (args.output / "graph.json").write_text(text + "\n", encoding="utf-8")
    # `</` cannot appear inside the embedded script element.
    html = TEMPLATE.read_text(encoding="utf-8").replace("__GRAPH_JSON__", json.dumps(graph, sort_keys=True).replace("</", "<\\/"))
    (args.output / "index.html").write_text(html, encoding="utf-8")
    modules = [n for n in graph["nodes"] if n["kind"] == "module"]
    print(f"{len(modules)} files, {sum(n['lines'] for n in modules)} lines, {len(graph['edges'])} edges -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
