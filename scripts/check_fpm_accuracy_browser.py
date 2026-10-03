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
        shutil.copytree(ROOT / "pages/e2e-accuracy", site / "e2e-accuracy")
        data = json.loads((ROOT / "tests/fpm_accuracy/fixtures/summary.json").read_text())
        data["rows"][0]["skipped_count"] = 7
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
        shutil.copytree(ROOT / "tests/fpm_accuracy/fixtures/dashboard", site / "fpm-accuracy/data")
        history_path = site / "fpm-accuracy/data/history.json"
        history = json.loads(history_path.read_text())
        release_entry = copy.deepcopy(history["entries"][0])
        release_entry["snapshot"]["branch"] = "release/0.12.0"
        release_entry["trend"] = False
        history["entries"].append(release_entry)
        history_path.write_text(json.dumps(history))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=site))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch()
                page = await browser.new_page(viewport={"width": 1600, "height": 1050}, color_scheme="light")
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                url = f"http://127.0.0.1:{server.server_port}/fpm-accuracy/"
                await page.goto(url + "?branch=main")
                await expect(page.locator(".overview-model-row")).to_have_count(2)
                await expect(page.locator("html")).to_have_attribute("data-theme", "light")
                await page.get_by_role("button", name="Switch to dark theme").click()
                await expect(page.locator("html")).to_have_attribute("data-theme", "dark")
                await page.reload()
                await expect(page.locator("html")).to_have_attribute("data-theme", "dark")
                # Navigation between the accuracy pages preserves the shared preference.
                await page.goto(url.replace("fpm-accuracy/", "e2e-accuracy/"))
                await expect(page.locator("html")).to_have_attribute("data-theme", "dark")
                await page.get_by_role("button", name="Switch to light theme").click()
                await page.goto(url + "?branch=main")
                await expect(page.locator("html")).to_have_attribute("data-theme", "light")
                await expect(page.locator(".overview-model-row")).to_have_count(2)
                await expect(page.locator("#freshness")).to_contain_text("Stale result")
                await expect(page.locator("thead th")).to_have_count(7)
                assert not await page.locator('a[href*="coverage.html"]').count()
                await expect(page.locator(".fpm-tabs a")).to_have_count(3)
                assert "op-based" not in await page.locator("body").inner_text()
                await expect(page.locator(".overview-config-row").first).to_contain_text("7 excluded or unavailable")
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
                await page.get_by_role("button", name="Switch to dark theme").click()
                if screenshot:
                    await page.screenshot(
                        path=str(Path(screenshot).with_stem(Path(screenshot).stem + "-dark")), full_page=True
                    )
                await page.set_viewport_size({"width": 390, "height": 844})
                await expect(page.locator("#branch")).to_be_visible()
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.get_by_role("button", name="Switch to light theme").focus()
                await page.keyboard.press("Enter")
                await expect(page.locator("html")).to_have_attribute("data-theme", "light")
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                if screenshot:
                    await page.screenshot(
                        path=str(Path(screenshot).with_stem(Path(screenshot).stem + "-mobile")), full_page=True
                    )
                await page.set_viewport_size({"width": 1400, "height": 1000})
                await page.goto(url + "trends.html?branch=main")
                await expect(page.locator("#trend-table tbody tr")).to_have_count(1)
                await expect(page.locator("#trend-chart circle")).to_have_count(2)
                await page.locator("#phase-filter").select_option("decode")
                await page.locator("#trend-chart circle").first.focus()
                await expect(page.locator("#trend-chart circle").first).to_be_focused()
                await expect(page.locator("#trend-tooltip")).to_be_visible()
                await page.keyboard.press("Escape")
                await expect(page.locator("#trend-tooltip")).to_be_hidden()
                await page.locator("#branch").select_option("release/0.12.0")
                await expect(page.locator("#dashboard-status")).to_contain_text("main only")
                await page.goto(url + "evaluation-detail.html?branch=main")
                await expect(page.locator("#distribution table")).to_be_visible()
                await page.locator("#phase-filter").select_option("decode")
                await expect(page.locator("#error-heatmap table")).to_be_visible()
                await page.locator("#method-filter").select_option("nowarmup")
                await expect(page.locator("#error-heatmap")).to_contain_text("0/")
                await page.locator("#search-filter").fill("nonexistent")
                await expect(page.locator("#configuration-filter option")).to_have_count(0)
                await page.locator("#search-filter").fill("")
                await expect(page.locator("#distribution table")).to_be_visible()
                await page.set_viewport_size({"width": 390, "height": 844})
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.set_viewport_size({"width": 1400, "height": 1000})
                # Missing matching assets must leave the accuracy tables available.
                await page.goto(url + "evaluation-detail.html")
                await expect(page.locator("#distribution table")).to_be_visible()
                await expect(page.locator("#visualization-status")).to_contain_text("3D unavailable")
                await expect(page.locator("#gym-visualization")).to_be_hidden()
                # Match this browser fixture to the evaluation revision, then reload.
                catalog_path = site / "fpm-accuracy/data/visualization/catalog.json"
                catalog_data = json.loads(catalog_path.read_text())
                catalog_data["hf_revision"] = history["entries"][0]["snapshot"]["hf_revision"]
                catalog_path.write_text(json.dumps(catalog_data))
                await page.goto(url + "3d-visualization.html")
                await page.wait_for_url("**/evaluation-detail.html?view=3d")
                await page.locator("#phase-filter").select_option("all")
                await expect(page.locator("#gv-left-chart .plot-container")).to_be_visible(timeout=30000)
                await expect(page.locator("#phase-summary")).to_be_visible()
                assert await page.evaluate(
                    "document.querySelector('#gym-visualization').compareDocumentPosition(document.querySelector('#accuracy-panel')) & Node.DOCUMENT_POSITION_FOLLOWING"
                )
                held = asyncio.Event()
                release = asyncio.Event()

                async def hold_chunk(route):
                    held.set()
                    await release.wait()
                    await route.continue_()

                await page.route("**/data/visualization/*.json.gz", hold_chunk)
                await page.locator("#gv-density").select_option("all")
                await asyncio.wait_for(held.wait(), timeout=10)
                await page.locator("#gv-density").select_option("sample")
                try:
                    await page.wait_for_function(
                        "document.querySelector('#gym-visualization').dataset.ready === 'true'",
                        timeout=10000,
                    )
                    await expect(page.locator("#gv-left-chart .plot-container")).to_be_visible()
                    await expect(page.locator("#gv-left-chart")).to_have_attribute("data-density", "sample")
                finally:
                    release.set()
                axes = await page.locator("#gv-x-axis option").evaluate_all("nodes => nodes.map(n=>n.value)")
                assert len(axes) == 7
                for x in axes:
                    for y in axes:
                        await page.locator("#gv-x-axis").select_option(x)
                        await page.locator("#gv-y-axis").select_option(y)
                        await page.wait_for_function(
                            "document.querySelector('#gym-visualization').dataset.ready === 'true'"
                        )
                await page.evaluate(
                    "Plotly.relayout(document.querySelector('#gv-left-chart'), {'scene.camera': {eye:{x:2,y:1,z:1}}})"
                )
                await page.locator("#gv-density").select_option("all")
                await expect(page.locator("#gv-left-count")).to_contain_text("40")
                await page.locator("#phase-filter").select_option("decode")
                await page.locator("#gv-density").select_option("sample")
                await expect(page.locator("#gv-error")).to_be_hidden()
                await page.wait_for_function("document.querySelector('#gym-visualization').dataset.ready === 'true'")
                async with page.expect_download() as download_info:
                    await page.locator('#gv-left-chart [data-title*="Download"]').click()
                download = await download_info.value
                assert download.suggested_filename.endswith(".png")
                if screenshot:
                    await page.screenshot(path=str(Path(screenshot).with_stem("fpm-3d")), full_page=True)
                await page.set_viewport_size({"width": 390, "height": 844})
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.goto(url + "trends.html?branch=main")
                await page.route("**/data/history.json", lambda route: route.fulfill(status=404, body="{}"))
                await page.reload()
                await expect(page.locator("#dashboard-status")).to_contain_text("No completed dashboard evaluation")
                assert not errors, errors
                await browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print("FPM browser checks passed (offline fixtures).")


if __name__ == "__main__":
    asyncio.run(check())
