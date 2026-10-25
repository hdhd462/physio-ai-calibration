#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify georgia access with real Edge browser."""
from playwright.sync_api import sync_playwright
import os, re, time

COOKIE = open(os.path.join(os.environ["TEMP"], "pn_ck2.txt"), encoding="utf-8").read().strip()
URL = "https://physionet.org/files/challenge-2021/1.0.3/training/georgia/"

def parse_cookies(s):
    out = []
    for part in s.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            out.append({"name": k.strip(), "value": v.strip(), "domain": ".physionet.org", "path": "/"})
    return out

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context()
    ctx.add_cookies(parse_cookies(COOKIE))
    page = ctx.new_page()
    resp = page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    print("status:", resp.status if resp else "none")
    time.sleep(1)
    body = page.content()
    print("len:", len(body))
    dirs = sorted(set(re.findall(r'href="[^"]*/(g\d+)/"', body)))
    print("subdirs:", dirs[:15])
    has403 = "dont have access" in body or "403" in page.title()
    print("is403:", has403)
    browser.close()
