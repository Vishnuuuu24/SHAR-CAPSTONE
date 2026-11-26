import hashlib, json, os
e = json.load(open("results/vlm/qwen25vl7b-expected-sha256.json"))
D = ".cache/weights/Qwen2.5-VL-7B-Instruct"
out = {}
for name, want in e["sha256"].items():
    h = hashlib.sha256()
    with open(os.path.join(D, name), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    out[name] = {"expected": want, "actual": h.hexdigest(), "ok": h.hexdigest() == want}
    print(name, "OK" if out[name]["ok"] else "MISMATCH", flush=True)
json.dump({"revision": e["revision"], "all_ok": all(v["ok"] for v in out.values()), "files": out},
          open("results/vlm/qwen25vl7b-sha256-verify.json", "w"), indent=1)
print("ALL_OK" if all(v["ok"] for v in out.values()) else "FAILED", flush=True)
