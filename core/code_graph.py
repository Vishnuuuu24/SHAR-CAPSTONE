"""Deterministic knowledge graph of the SHAR repository (modules, imports, decisions, results).

Idea taken from Egonex-AI/Understand-Anything (an AI-assistant plugin that maps a codebase into an
explorable graph). Its structural half is reproduced here with the standard-library `ast` parser
instead of tree-sitter; its LLM half (written summaries) is replaced by each module's own docstring,
so the same repository state always yields the same graph. SHAR-specific additions: edges to the
decisions (D-xx) and known problems (K-xx) a file cites and to the `results/` folders it names.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from pathlib import Path

PACKAGES = ("core", "models", "eval", "scripts", "tests", "ui", "paper")
LAYERS = {"core": "library", "models": "model", "eval": "metrics", "scripts": "entry point",
          "tests": "test", "ui": "ui", "paper": "paper"}
# First match wins; patterns are matched against the file stem.
TOPICS = (
    ("RWF-2000 (separate binary task)", r"rwf"),
    ("Live detector (temporal VAD)", r"detect|detector"),
    ("Captions and language", r"caption|uca|language|text"),
    ("Data preparation", r"p0|p1a|p1b|materiali[sz]e|inventory|mapping"),
    ("Evaluation protocol", r"inner_cv|final_test|fuse_inner|family|confusion|metrics|video_level|aggregation|topk|bootstrap|claim|near_dup|meanpool"),
    ("Denoising (RESULT 1)", r"(^|_)p2|result1|restor|denois|nafnet|scunet|adaptive_median|image_quality|dae|dvae|noise"),
    ("Backbones and feature caches", r"backbone|feature_cache|cache|extract|vjepa|siglip|dinov2|videomae|vit_split"),
    ("14-class recognition", r"p1c|head|probe|specialist|subset7|combined|finetune|failure|baseline|pose|shooting|frame|video|xd|classif|prediction"),
    ("Infrastructure", r"provenance|device|progress|reproduc|lifecycle|throughput|pagecache|ioaccount|precision|readiness|dgx|snapshot|manifest|pod|check_|calibrate|smoke|verify|code_graph|^__init__$"),
    ("Paper and UI", r"paper|docx|dashboard|demo|render|figure|server"),
)
DECISION = re.compile(r"\b([DK])-(\d{2,3})\b")
RESULT_PATH = re.compile(r"results/([A-Za-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)?)")


def topic_of(stem: str) -> str:
    for name, pattern in TOPICS:
        if re.search(pattern, stem):
            return name
    return "Other"


def first_line(text: str | None) -> str:
    return (text or "").strip().split("\n\n")[0].replace("\n", " ").strip()


def _module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(tree: ast.AST, module: str, is_package: bool) -> set[str]:
    """Dotted names this module imports, with relative imports resolved."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".")
                base = base[: len(base) - node.level + (1 if is_package else 0)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            # `from core import x` -> `core.x`; resolution falls back to `core` only if x is not a submodule.
            found.update(f"{prefix}.{alias.name}" for alias in node.names)
    return found


def _definitions(tree: ast.Module) -> tuple[list[dict], list[dict]]:
    functions, classes = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions.append({"name": node.name, "line": node.lineno, "doc": first_line(ast.get_docstring(node))})
        elif isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_")]
            classes.append({"name": node.name, "line": node.lineno, "doc": first_line(ast.get_docstring(node)), "methods": methods})
    return functions, classes


def decision_titles(decisions_md: Path) -> dict[str, str]:
    """`D-xx` -> the bold first sentence of its row in DECISIONS.md (trimmed)."""
    titles = {}
    for line in decisions_md.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(D-\d+)\s*\|\s*(.*?)\s*\|", line)
        if match:
            bold = re.search(r"\*\*(.+?)\*\*", match.group(2))
            text = (bold.group(1) if bold else match.group(2)).strip()
            titles[match.group(1)] = text if len(text) <= 240 else text[:237] + "..."
    return titles


def scan(root: Path) -> dict:
    """Build the graph for the repository at `root`. Pure function of the files on disk."""
    files = sorted(p.relative_to(root) for package in PACKAGES if (root / package).is_dir()
                   for p in (root / package).rglob("*.py") if "__pycache__" not in p.parts)
    by_name = {_module_name(p): str(p) for p in files}
    nodes, edges, parsed = [], [], {}
    for path in files:
        source = (root / path).read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            tree, problem = ast.Module(body=[], type_ignores=[]), f"syntax error line {error.lineno}"
        else:
            problem = None
        functions, classes = _definitions(tree)
        doc = first_line(ast.get_docstring(tree))
        nodes.append({"id": str(path), "kind": "module", "layer": LAYERS[path.parts[0]],
                      "topic": topic_of(path.stem) if path.parts[0] != "tests" else None,
                      "lines": source.count("\n") + 1, "summary": doc, "has_docstring": bool(doc),
                      "functions": functions, "classes": classes, "problem": problem,
                      "sha256": hashlib.sha256(source.encode()).hexdigest()[:12]})
        parsed[str(path)] = (tree, source, path.name == "__init__.py")

    for path_text, (tree, source, is_package) in parsed.items():
        module = _module_name(Path(path_text))
        targets = set()
        for name in _imports(tree, module, is_package):
            # Longest known prefix wins: `core.detect_model.load_store` -> core/detect_model.py
            parts = name.split(".")
            for cut in range(len(parts), 0, -1):
                target = by_name.get(".".join(parts[:cut]))
                if target:
                    if target != path_text:
                        targets.add(target)
                    break
        edges += [{"source": path_text, "target": t, "kind": "imports"} for t in sorted(targets)]
        cited = sorted({f"{k}-{n}" for k, n in DECISION.findall(source)})
        edges += [{"source": path_text, "target": c, "kind": "cites"} for c in cited]
        results = sorted(set(RESULT_PATH.findall(source)))
        edges += [{"source": path_text, "target": f"results/{r}", "kind": "names result"} for r in results]

    # Tests take the topic of the modules they exercise (most common), so the tour can place them.
    topic_by_id = {n["id"]: n["topic"] for n in nodes}
    for node in nodes:
        if node["topic"] is None:
            counts = defaultdict(int)
            for e in edges:
                if e["source"] == node["id"] and e["kind"] == "imports" and topic_by_id.get(e["target"]):
                    counts[topic_by_id[e["target"]]] += 1
            node["topic"] = max(sorted(counts), key=counts.get) if counts else topic_of(node["id"])
    titles = decision_titles(root / "docs/DECISIONS.md") if (root / "docs/DECISIONS.md").exists() else {}
    cited_ids = sorted({e["target"] for e in edges if e["kind"] == "cites"}, key=lambda d: (d[0], int(d[2:])))
    for ident in cited_ids:
        nodes.append({"id": ident, "kind": "decision" if ident.startswith("D") else "known problem",
                      "summary": titles.get(ident, ""), "topic": None, "layer": None})
    for result in sorted({e["target"] for e in edges if e["kind"] == "names result"}):
        nodes.append({"id": result, "kind": "result folder", "exists": (root / result).exists(), "topic": None, "layer": None})
    return {"schema": "shar-code-graph/1", "nodes": nodes, "edges": edges, "tours": tours(nodes, edges)}


def tours(nodes: list[dict], edges: list[dict]) -> dict[str, list[str]]:
    """Per topic, non-test modules ordered dependencies-first (the plugin's 'guided tour' idea)."""
    modules = {n["id"]: n for n in nodes if n["kind"] == "module" and n["layer"] != "test"}
    deps = defaultdict(set)
    for e in edges:
        if e["kind"] == "imports" and e["source"] in modules and e["target"] in modules:
            deps[e["source"]].add(e["target"])
    order, state = [], {}

    def visit(node_id: str) -> None:
        if state.get(node_id):
            return
        state[node_id] = 1                       # cycles are tolerated: a node is placed once
        for dep in sorted(deps[node_id]):
            visit(dep)
        order.append(node_id)

    for node_id in sorted(modules):
        visit(node_id)
    result = defaultdict(list)
    for node_id in order:
        result[modules[node_id]["topic"]].append(node_id)
    return dict(sorted(result.items()))


def impact(graph: dict, changed: list[str]) -> dict[str, list[str]]:
    """Everything that imports a changed file, directly or transitively, split into tests and code."""
    importers = defaultdict(set)
    for e in graph["edges"]:
        if e["kind"] == "imports":
            importers[e["target"]].add(e["source"])
    seen, stack = set(), list(changed)
    while stack:
        for parent in importers[stack.pop()]:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    seen -= set(changed)
    return {"tests": sorted(s for s in seen if s.startswith("tests/")),
            "code": sorted(s for s in seen if not s.startswith("tests/"))}
