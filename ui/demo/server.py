#!/usr/bin/env python3
"""SHAR demo server: serves ui/demo/index.html and classifies an uploaded video with the final E3 model (D-76).

Standard library only. POST /predict takes the raw video bytes (header X-Filename carries the original name) and
returns JSON from `core.e3_inference.E3DemoModel.predict` (or, with `--model d53`, the older
`core.demo_inference.DemoModel`). One prediction runs at a time. Run inside a DGX pod and
reach it with `kubectl port-forward` plus an SSH tunnel (see ui/demo/README.md).

With `--token` (or env SHAR_DEMO_TOKEN), /health and /predict require that token, sent as the X-Token header; the
page reads it from its own URL (`/?token=...`). Use it whenever the server is reachable by anyone but you.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAGE = Path(__file__).with_name("index.html")
MAX_BYTES = 2 * 1024**3


def make_handler(model, lock: threading.Lock, token: str | None = None, samples: dict | None = None):
    from core.demo_samples import byte_range, public_entry

    samples = samples or {}

    class Handler(BaseHTTPRequestHandler):
        def _authorised(self) -> bool:
            if not token:
                return True
            query = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            given = self.headers.get("X-Token") or query
            return hmac.compare_digest(given.encode(), token.encode())

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/health":
                if not self._authorised():
                    return self._json(401, {"error": "access token missing or wrong; open the link you were given"})
                self._json(200, {"ok": True, "classes": model.classes, "device": str(model.device),
                                 "busy": lock.locked(), **model.health()})
            elif path == "/samples":
                if not self._authorised():
                    return self._json(401, {"error": "access token missing or wrong; open the link you were given"})
                self._json(200, {"samples": [public_entry(e) for e in samples.values()]})
            elif path.startswith("/samples/"):
                if not self._authorised():
                    return self._json(401, {"error": "access token missing or wrong; open the link you were given"})
                self._send_sample(unquote(path[len("/samples/"):]))
            else:
                self._json(404, {"error": "not found"})

        def _send_sample(self, name: str) -> None:
            entry = samples.get(name)  # only listed names; never a path from the request
            if entry is None:
                return self._json(404, {"error": "no such sample"})
            size = entry["bytes"]
            try:
                rng = byte_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = rng if rng else (0, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "private, max-age=3600")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with entry["path"].open("rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left:
                        chunk = f.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the browser stopped reading (seek or close); normal for video

        def do_POST(self):  # noqa: N802
            if urlsplit(self.path).path != "/predict":
                return self._json(404, {"error": "not found"})
            if not self._authorised():
                return self._json(401, {"error": "access token missing or wrong; open the link you were given"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self._json(400, {"error": "empty upload"})
            if length > MAX_BYTES:
                return self._json(413, {"error": f"video is larger than {MAX_BYTES // 1024**3} GB"})
            name = unquote(self.headers.get("X-Filename") or "upload.mp4")
            suffix = Path(name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        return self._json(400, {"error": "upload ended early"})
                    tmp.write(chunk)
                    remaining -= len(chunk)
                tmp.flush()
                with lock:
                    try:
                        result = model.predict(Path(tmp.name), name)
                    except ValueError as e:
                        return self._json(422, {"error": str(e)})
                    except Exception as e:  # report, keep serving
                        traceback.print_exc()
                        return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            print(json.dumps({"file": name, "prediction": result["prediction"], "confidence": round(result["confidence"], 3),
                              "expected": (result["expected"] or {}).get("label"), "seconds": result["timing_s"]["total"]}), flush=True)
            self._json(200, result)

        def log_message(self, fmt, *args):  # quieter default log
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return Handler


def load_model(name: str, device: str):
    if name == "e3":
        from core.e3_inference import E3DemoModel

        return E3DemoModel(REPO_ROOT, device=device)
    from core.demo_inference import DemoModel

    return DemoModel(REPO_ROOT, device=device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", choices=["e3", "d53"], default="e3", help="e3: final model (default); d53: the older demo model")
    ap.add_argument("--token", default=os.environ.get("SHAR_DEMO_TOKEN"), help="require this access token (default: env SHAR_DEMO_TOKEN)")
    args = ap.parse_args(argv)
    from core.demo_samples import load_samples

    samples = load_samples(REPO_ROOT)
    model = load_model(args.model, args.device)
    print(json.dumps({"ready": True, "port": args.port, "device": str(model.device), "token_required": bool(args.token),
                      "samples": len(samples), **model.health()}), flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(model, threading.Lock(), args.token, samples)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
