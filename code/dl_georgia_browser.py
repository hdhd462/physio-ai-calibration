#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download georgia challenge dataset via real Edge browser (WAF fingerprint pass)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
from playwright.sync_api import sync_playwright
import os, re, time, base64, json, sys

COOKIE = open(os.path.join(os.environ["TEMP"], "pn_ck2.txt"), encoding="utf-8").read().strip()
BASE = "https://physionet.org/files/challenge-2021/1.0.3/training/georgia/"
DEST = os.path.join(DATA_DIR, "ext_georgia")
DIRS = [f"g{i}" for i in range(1, 12)]
BATCH = 30

def parse_cookies(s):
    out = []
    for part in s.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            out.append({"name": k.strip(), "value": v.strip(), "domain": ".physionet.org", "path": "/"})
    return out

JS_FETCH_BATCH = """
async (urls) => {
  const results = [];
  for (const u of urls) {
    try {
      const r = await fetch(u, {credentials: 'include'});
      if (!r.ok) { results.push({u, err: r.status}); continue; }
      const buf = await r.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let bin = '';
      const CH = 0x8000;
      for (let i = 0; i < bytes.length; i += CH) {
        bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
      }
      results.push({u, b64: btoa(bin)});
    } catch (e) { results.push({u, err: String(e)}); }
  }
  return results;
}
"""

def main():
    os.makedirs(DEST, exist_ok=True)
    logf = open(os.path.join(DEST, "dl_log.txt"), "a", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    stats = {"ok": 0, "skip": 0, "fail": 0}
    failures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context()
        ctx.add_cookies(parse_cookies(COOKIE))
        page = ctx.new_page()

        all_urls = []
        failed_dirs = []
        for d in DIRS:
            ok = False
            for attempt in range(6):
                try:
                    resp = page.goto(BASE + d + "/", wait_until="domcontentloaded", timeout=90000)
                    if resp and resp.status == 200:
                        ok = True
                        break
                    log(f"[{d}] attempt {attempt} status={resp.status if resp else None}")
                except Exception as e:
                    log(f"[{d}] list attempt {attempt} fail: {str(e)[:60]}")
                    time.sleep(10 + attempt * 5)
            if not ok:
                log(f"[{d}] LIST FAILED first pass, queue for retry")
                failed_dirs.append(d)
                continue
            time.sleep(2)
            body = page.content()
            files = sorted(set(re.findall(r'href="([^"/]+\.(?:hea|mat))"', body)))
            log(f"[{d}] {len(files)} files listed")
            for f in files:
                all_urls.append((d, f, BASE + d + "/" + f))
            time.sleep(1)

        # second pass for failed dirs
        if failed_dirs:
            log(f"retrying dirs: {failed_dirs}")
            time.sleep(30)
            still_failed = []
            for d in failed_dirs:
                ok = False
                for attempt in range(6):
                    try:
                        resp = page.goto(BASE + d + "/", wait_until="domcontentloaded", timeout=90000)
                        if resp and resp.status == 200:
                            ok = True
                            break
                    except Exception as e:
                        log(f"[{d}] retry attempt {attempt} fail: {str(e)[:60]}")
                        time.sleep(15 + attempt * 5)
                if not ok:
                    still_failed.append(d)
                    continue
                time.sleep(2)
                body = page.content()
                files = sorted(set(re.findall(r'href="([^"/]+\.(?:hea|mat))"', body)))
                log(f"[{d}] retry found {len(files)} files")
                for f in files:
                    all_urls.append((d, f, BASE + d + "/" + f))
                time.sleep(2)
            if still_failed:
                log(f"dirs still failed after retry: {still_failed}")
                failures.extend([f"{d}/DIR: list_failed" for d in still_failed])

        log(f"total urls: {len(all_urls)}")

        # filter already downloaded
        todo = []
        for d, f, url in all_urls:
            dest = os.path.join(DEST, d, f)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                stats["skip"] += 1
            else:
                todo.append((d, f, url))
        log(f"todo: {len(todo)} (skip {stats['skip']})")

        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            urls = [u for _, _, u in batch]
            try:
                results = page.evaluate(JS_FETCH_BATCH, urls)
            except Exception as e:
                log(f"batch@{i} eval error: {str(e)[:100]}")
                time.sleep(10)
                try:
                    results = page.evaluate(JS_FETCH_BATCH, urls)
                except Exception as e2:
                    log(f"batch@{i} retry failed: {str(e2)[:100]}")
                    for d, f, u in batch:
                        stats["fail"] += 1
                        failures.append(f"{d}/{f}: eval_fail")
                    continue
            got = {}
            for r in results:
                got[r["u"]] = r
            for d, f, u in batch:
                r = got.get(u)
                if r is None:
                    stats["fail"] += 1
                    failures.append(f"{d}/{f}: missing")
                elif "err" in r:
                    stats["fail"] += 1
                    failures.append(f"{d}/{f}: {r['err']}")
                else:
                    dest = os.path.join(DEST, d, f)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as fh:
                        fh.write(base64.b64decode(r["b64"]))
                    stats["ok"] += 1
            if (i // BATCH) % 5 == 0:
                log(f"progress {i + len(batch)}/{len(todo)} ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}")
            time.sleep(0.3)

        browser.close()

    log(f"DONE ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}")
    if failures:
        with open(os.path.join(DEST, "download_failures.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(failures))
        log(f"failures: {len(failures)}")

if __name__ == "__main__":
    main()
