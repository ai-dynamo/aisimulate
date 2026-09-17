#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the public FPM overview with offline branch fixtures in Chromium."""

from __future__ import annotations

import asyncio
import copy
import functools
import http.server
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

from fpm_accuracy.contract import artifact_key
from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


async def check():
    with tempfile.TemporaryDirectory() as directory:
        site = Path(directory)
        shutil.copytree(ROOT / "pages/fpm-accuracy", site / "fpm-accuracy")
        data = json.loads((ROOT / "tests/fpm_accuracy/fixtures/summary.json").read_text())
        entries = []
        for branch in ("main", "release/0.12.0"):
            summary = copy.deepcopy(data)
            summary["snapshot"]["branch"] = branch
            if branch != "main":
                summary["rows"][0]["model"] = "Release/Alpha"
            path = f"branches/{artifact_key(branch)}/summary.json"
            target = site / "fpm-accuracy" / path
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(summary))
            entries.append({"branch": branch, "status": "available", "summary_path": path, "head_sha": "e" * 40})
        entries.append({"branch": "release/0.13.0", "status": "unavailable", "summary_path": None})
        (site / "fpm-accuracy/branches.json").write_text(
            json.dumps({"schema_version": 1, "default_branch": "main", "branches": entries})
        )
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=site))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch()
                page = await browser.new_page(viewport={"width": 1600, "height": 1050})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                url = f"http://127.0.0.1:{server.server_port}/fpm-accuracy/"
                await page.goto(url + "?branch=main")
                await expect(page.locator(".overview-model-row")).to_have_count(2)
                await expect(page.locator("#freshness")).to_contain_text("Stale result")
                await expect(page.locator("thead th")).to_have_count(7)
                assert not await page.locator(
                    'a[href*="evaluation-detail"], a[href*="coverage.html"], a[href*="trends.html"]'
                ).count()
                assert "op-based" not in await page.locator("body").inner_text()
                await expect(page.locator(".overview-config-row").first).to_contain_text("80/100 predicted")
                await expect(page.locator(".overview-config-row").first).to_contain_text("80.0% coverage")
                await page.locator('[data-model="Example/Alpha"]').click()
                await expect(page.locator(".overview-config-row").first).to_be_hidden()
                await page.locator('[data-model="Example/Alpha"]').click()
                await expect(page.locator(".overview-config-row").first).to_be_visible()
                await page.locator('[data-sort="model"]').click()
                await expect(page.locator(".overview-model-row").first).to_contain_text("Example/Beta")
                await page.locator("#branch").select_option("release/0.12.0")
                await expect(page.locator(".overview-model-row")).to_contain_text(["Release/Alpha", "Example/Beta"])
                assert "branch=release%2F0.12.0" in page.url
                await page.reload()
                await expect(page.locator("#branch")).to_have_value("release/0.12.0")
                await expect(page.locator(".overview-model-row")).to_have_count(2)
                await page.locator("#branch").select_option("release/0.13.0")
                await expect(page.locator("#overview-body")).to_contain_text("No completed evaluation")
                await expect(page.locator(".overview-model-row")).to_have_count(0)
                await page.goto(url + "?branch=release/0.11.0")
                await expect(page.locator("#overview-body")).to_contain_text("No completed evaluation")
                await page.goto(url + "?branch=main")
                await expect(page.locator(".overview-model-row")).to_have_count(2)

                # Out-of-order fetch completion must not overwrite a newly selected branch.
                async def delayed(route):
                    await asyncio.sleep(0.5)
                    await route.continue_()

                await page.route("**/" + artifact_key("main") + "/summary.json", delayed)
                await page.locator("#branch").select_option("main")
                await page.locator("#branch").select_option("release/0.12.0")
                await expect(page.locator('[data-model="Release/Alpha"]')).to_be_visible()
                await page.wait_for_timeout(650)
                await expect(page.locator('[data-model="Release/Alpha"]')).to_be_visible()
                await page.unroute_all(behavior="wait")
                if screenshot := os.environ.get("FPM_SCREENSHOT"):
                    await page.screenshot(path=screenshot, full_page=True)
                await page.set_viewport_size({"width": 390, "height": 844})
                await expect(page.locator("#branch")).to_be_visible()
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                assert not errors, errors
                await browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print("FPM browser checks passed (offline fixtures).")


if __name__ == "__main__":
    asyncio.run(check())
