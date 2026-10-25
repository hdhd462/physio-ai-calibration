#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download PhysioNet challenge-2021 georgia training set with cookie auth."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import os, re, sys, time, urllib.request, urllib.error

COOKIE = open(os.path.join(os.environ["TEMP"], "pn_ck.txt"), encoding="utf-8").read().strip()
BASE = "https://physionet.org/files/challenge-2021/1.0.3/training/georgia/"
DEST = os.path.join(DATA_DIR, "ext_georgia")
DIRS = [f"g{i}" for i in range(1, 12)]

def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"Cookie": COOKIE, "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def list_hrefs(html):
    return sorted(set(re.findall(r'href="([^"]+\.(?:hea|mat))"', html)))

def download_file(url, dest_path, retries=5):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return "skip"
    tmp = dest_path + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Cookie": COOKIE, "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest_path)
            return "ok"
        except Exception as e:
            if attempt == retries - 1:
                return f"fail:{e}"
            time.sleep(2 * (attempt + 1))
    return "fail"

def main():
    total_ok = total_skip = total_fail = 0
    failures = []
    for d in DIRS:
        ddir = os.path.join(DEST, d)
        os.makedirs(ddir, exist_ok=True)
        print(f"[{d}] listing...", flush=True)
        try:
            html = fetch(BASE + d + "/").decode("utf-8", "replace")
        except Exception as e:
            print(f"[{d}] LIST FAIL: {e}", flush=True)
            continue
        files = list_hrefs(html)
        print(f"[{d}] {len(files)} files", flush=True)
        for i, fname in enumerate(files, 1):
            url = f"{BASE}{d}/{fname}"
            dest = os.path.join(ddir, fname)
            res = download_file(url, dest)
            if res == "ok":
                total_ok += 1
            elif res == "skip":
                total_skip += 1
            else:
                total_fail += 1
                failures.append(f"{d}/{fname}: {res}")
            if i % 200 == 0:
                print(f"[{d}] {i}/{len(files)} ok={total_ok} skip={total_skip} fail={total_fail}", flush=True)
    print(f"\nDONE ok={total_ok} skip={total_skip} fail={total_fail}", flush=True)
    if failures:
        with open(os.path.join(DEST, "download_failures.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(failures))
        print(f"failures written: {len(failures)}", flush=True)

if __name__ == "__main__":
    main()
