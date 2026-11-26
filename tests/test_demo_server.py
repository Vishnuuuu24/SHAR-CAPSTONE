import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from pathlib import Path

from core.demo_samples import byte_range
from ui.demo.server import make_handler


class FakeModel:
    classes = ["A", "B"]
    device = "cpu"

    def health(self):
        return {"model": "fake", "members": 1}

    def predict(self, path, name):
        return {"prediction": "A", "confidence": 0.9, "expected": None, "timing_s": {"total": 0.0}}


class DemoServerTokenTests(unittest.TestCase):
    def serve(self, token, samples=None):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeModel(), threading.Lock(), token, samples))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def status(self, req):
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code

    def post(self, base, headers):
        return urllib.request.Request(base + "/predict", data=b"x", method="POST", headers={"X-Filename": "a.mp4", **headers})

    def test_token_required_on_health_and_predict_but_not_page(self):
        base = self.serve("s3cret")
        self.assertEqual(self.status(base + "/?token=s3cret"), 200)
        self.assertEqual(self.status(base + "/"), 200)
        self.assertEqual(self.status(base + "/health"), 401)
        self.assertEqual(self.status(urllib.request.Request(base + "/health", headers={"X-Token": "s3cret"})), 200)
        self.assertEqual(self.status(base + "/health?token=s3cret"), 200)
        self.assertEqual(self.status(self.post(base, {})), 401)
        self.assertEqual(self.status(self.post(base, {"X-Token": "wrong"})), 401)
        self.assertEqual(self.status(self.post(base, {"X-Token": "s3cret"})), 200)

    def test_no_token_configured_keeps_open_behaviour(self):
        base = self.serve(None)
        self.assertEqual(json.loads(urllib.request.urlopen(base + "/health").read())["model"], "fake")
        self.assertEqual(self.status(self.post(base, {})), 200)

    def test_samples_need_token_serve_ranges_and_only_listed_names(self):
        tmp = Path(tempfile.mkdtemp())
        clip = tmp / "a.mp4"
        clip.write_bytes(bytes(range(100)))
        samples = {"a.mp4": {"file": "a.mp4", "title": "A", "path": clip, "bytes": 100}}
        base = self.serve("s3cret", samples)
        self.assertEqual(self.status(base + "/samples"), 401)
        self.assertEqual(self.status(base + "/samples/a.mp4"), 401)
        listing = json.loads(urllib.request.urlopen(base + "/samples?token=s3cret").read())
        self.assertEqual(listing["samples"], [{"file": "a.mp4", "title": "A", "bytes": 100}])
        self.assertEqual(urllib.request.urlopen(base + "/samples/a.mp4?token=s3cret").read(), bytes(range(100)))
        r = urllib.request.urlopen(urllib.request.Request(base + "/samples/a.mp4?token=s3cret", headers={"Range": "bytes=10-19"}))
        self.assertEqual((r.status, r.read(), r.headers["Content-Range"]), (206, bytes(range(10, 20)), "bytes 10-19/100"))
        self.assertEqual(self.status(base + "/samples/b.mp4?token=s3cret"), 404)
        self.assertEqual(self.status(base + "/samples/..%2Fserver.py?token=s3cret"), 404)


class ByteRangeTests(unittest.TestCase):
    def test_forms(self):
        self.assertIsNone(byte_range(None, 100))
        self.assertEqual(byte_range("bytes=0-", 100), (0, 99))
        self.assertEqual(byte_range("bytes=90-200", 100), (90, 99))
        self.assertEqual(byte_range("bytes=-10", 100), (90, 99))
        with self.assertRaises(ValueError):
            byte_range("bytes=100-", 100)


if __name__ == "__main__":
    unittest.main()
