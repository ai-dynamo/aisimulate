#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the public FPM overview with offline branch fixtures in Chromium."""

from __future__ import annotations

import asyncio
import copy
import functools
import gzip
import hashlib
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

# Support direct execution as well as package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playwright.async_api import async_playwright, expect

from scripts.fpm_accuracy.contract import artifact_key

ROOT = Path(__file__).resolve().parents[2]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def prepare_visualization_fixtures(directory: Path):
    """Materialize readable fixtures as production-style hashed assets."""
    catalog = json.loads((directory / "catalog.json").read_text())
    files = {}

    def asset(name, compressed=False):
        source = directory / name
        content = json.dumps(json.loads(source.read_text()), separators=(",", ":"), sort_keys=True).encode()
        if compressed:
            content = gzip.compress(content, mtime=0)
        digest = hashlib.sha256(content).hexdigest()
        filename = digest + (".json.gz" if compressed else ".json")
        (directory / filename).write_bytes(content)
        files[filename] = digest
        source.unlink()
        return filename

    for group in catalog["groups"]:
        group["sample_file"] = asset(group["sample_file"])
        group["all_files"] = [asset(name, compressed=True) for name in group["all_files"]]
    content = json.dumps(catalog, separators=(",", ":"), sort_keys=True).encode()
    (directory / "catalog.json").write_bytes(content)
    files["catalog.json"] = hashlib.sha256(content).hexdigest()
    manifest = {key: catalog[key] for key in ("schema_version", "policy", "hf_revision", "repo_id")}
    manifest.update(files=files, observations=sum(group["n"] for group in catalog["groups"]))
    (directory / "manifest.json").write_text(json.dumps(manifest))


async def check_evaluation_banner(page, date="2026-09-17"):
    banner = page.locator("#evaluation-banner")
    await expect(banner).to_have_count(1)
    await expect(banner.locator("time")).to_have_text(date)
    await expect(banner.locator("a")).to_have_count(3)
    await expect(banner.locator("#evaluation-aisim")).to_contain_text("AISim ")
    await expect(banner.locator("#evaluation-hf")).to_contain_text("HF ")
    await expect(banner).not_to_contain_text("evaluated configurations")
    await expect(banner).not_to_contain_text("10:47")
    await expect(page.locator(".scope-bar, .snapshot-value, #nav-status-text")).to_have_count(0)
    assert await banner.evaluate("node => getComputedStyle(node).justifyContent === 'flex-start'")


async def check_predictor_views(page, url, data, screenshot):
    """Unequal counts distinguish configuration means from weighted/model means."""
    summary = copy.deepcopy(data)
    second = copy.deepcopy(summary["rows"][0])
    second.update(configuration_id="alpha-small", parallelism="tp8", measurement_count=10)
    for method, mape in (("regression", 20), ("warmup", 0), ("nowarmup", None)):
        second["results"][method]["metrics"]["all"].update(
            mape_pct=mape, predicted_count=10 if mape is not None else 0, measured_count=10
        )
    summary["rows"].append(second)
    for method in ("regression", "warmup", "nowarmup"):
        summary["rows"][1]["results"][method]["metrics"]["all"]["mape_pct"] = (
            "nonfinite" if method == "nowarmup" else 30
        )
    missing = copy.deepcopy(summary["rows"][0])
    missing.update(model="Example/Missing", configuration_id="missing")
    for result in missing["results"].values():
        result["metrics"]["all"].update(mape_pct=None, predicted_count=0)
    summary["rows"].append(missing)
    empty = copy.deepcopy(missing)
    empty.update(model="Example/Empty", configuration_id="empty", measurement_count=0)
    summary["rows"].append(empty)
    pattern = "**/" + artifact_key("main") + "/summary.json"
    await page.route(pattern, lambda route: route.fulfill(body=json.dumps(summary).replace('"nonfinite"', "1e400")))
    await page.set_viewport_size({"width": 1400, "height": 1000})
    await page.goto(url + "?branch=main")
    await expect(page.locator("#overall-value")).to_have_text("14.17%")
    await expect(page.locator("#overall-count")).to_have_text("3 / 4 configurations with MAPE")
    alpha = page.locator(".overview-model-row").filter(has_text="Example/Alpha")
    await expect(alpha).to_contain_text("6.25%")
    await expect(alpha).to_contain_text("Mixed predictors")
    await expect(page.locator("#overview-body")).not_to_contain_text("predicted")
    await expect(page.locator("#overview-body")).not_to_contain_text("coverage")
    await expect(page.locator("#overview-body")).not_to_contain_text("configurations with MAPE")
    await expect(page.locator(".overview-config-row").first).to_contain_text("Regression")
    await expect(page.locator(".overview-config-row").nth(1)).to_contain_text("0.00%")
    await expect(page.locator(".overview-config-row").nth(1)).to_contain_text("FPM (KV warmup on)")
    await expect(page.locator(".overview-model-row").last).to_contain_text("Unavailable")
    await page.locator('[data-model="Example/Alpha"]').click()
    await expect(page.locator("#overall-value")).to_have_text("14.17%")
    await page.locator('[data-sort="best"]').click()
    await expect(page.locator(".overview-model-row").first).to_contain_text("Example/Beta")
    await page.locator('[data-sort="best"]').click()
    await expect(page.locator(".overview-model-row").first).to_contain_text("Example/Alpha")
    await expect(page.locator(".overview-model-row").last).to_contain_text("Example/Missing")
    await page.get_by_role("link", name="Predictors", exact=True).click()
    await expect(page.locator('.fpm-tabs [aria-current="page"]')).to_have_text("Predictors")
    await check_evaluation_banner(page)
    await expect(page.locator(".summary-card")).to_have_count(3)
    await expect(page.locator("thead th")).to_have_count(7)
    await expect(page.locator("#overview-body")).not_to_contain_text("predicted")
    await expect(page.locator("#overview-body")).not_to_contain_text("coverage")
    await expect(page.locator("#regression-value")).to_have_text("20.83%")
    await expect(page.locator("#warmup-value")).to_have_text("14.17%")
    await expect(page.locator("#nowarmup-value")).to_have_text("12.50%")
    await expect(page.locator("#nowarmup-count")).to_have_text("1 / 4 configurations with MAPE")
    await expect(alpha).to_contain_text("13.33%")
    await expect(alpha).to_contain_text("11.11%")
    await page.locator(".predictor-reference summary").first.click()
    await expect(page.locator(".predictor-reference pre").first).to_be_visible()
    await expect(page.locator(".predictor-reference")).to_contain_text("tune_with_fpms")
    await page.locator(".predictor-reference summary").first.click()
    await page.locator('[data-sort="method:regression"]').click()
    await expect(page.locator(".overview-model-row").first).to_contain_text("Example/Beta")
    for theme in ("dark", "light"):
        await page.get_by_role("button", name=f"Switch to {theme} theme").click()
        await expect(page.locator("html")).to_have_attribute("data-theme", theme)
        if screenshot:
            await page.evaluate("scrollTo(0, 0)")
            await page.screenshot(path=str(Path(screenshot).with_stem(f"fpm-predictors-{theme}")), full_page=True)
    await page.set_viewport_size({"width": 390, "height": 844})
    assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    if screenshot:
        await page.screenshot(path=str(Path(screenshot).with_stem("fpm-predictors-mobile")), full_page=True)
    await page.locator("#branch").select_option("release/0.12.0")
    await expect(page.locator(".overview-model-row")).to_have_count(2)
    await expect(page.locator("#evaluation-run")).to_have_attribute(
        "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/456/attempts/2"
    )
    await expect(page.get_by_role("link", name="Overview", exact=True)).to_have_attribute(
        "href", "index.html?branch=release%2F0.12.0"
    )
    await page.locator("#branch").select_option("release/0.13.0")
    await expect(page.locator("#regression-value")).to_have_text("—")
    await expect(page.locator("#evaluation-run")).to_be_hidden()
    await page.unroute(pattern)


async def check_trend_tooltip(page, screenshot):
    await check_evaluation_banner(page)
    point = page.locator(".trend-point").first
    tooltip = page.locator("#trend-tooltip")
    await point.evaluate("node => node.blur()")
    await point.hover()
    await expect(tooltip).to_be_visible()
    await expect(tooltip.locator("dt")).to_contain_text(
        ["Predicted / measured", "Coverage", "Errors", "Evaluated", "AISim", "HF dataset", "Evaluator"]
    )
    await expect(point.locator("title")).to_have_count(0)
    await expect(page.locator("#evaluation-run")).to_have_attribute(
        "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/123/attempts/1"
    )
    await tooltip.hover()
    await page.wait_for_timeout(180)
    await expect(tooltip).to_be_visible()
    if screenshot:
        await page.screenshot(path=str(Path(screenshot).with_stem("fpm-trend-tooltip")))
    await page.mouse.move(0, 0)
    await expect(tooltip).to_be_hidden()
    await point.focus()
    await expect(tooltip).to_be_visible()
    await page.locator("#phase-filter").focus()
    await expect(tooltip).to_be_hidden()
    await point.dispatch_event("pointerdown", {"pointerType": "touch"})
    await expect(tooltip).to_be_visible()
    await page.locator("h2").first.click()
    await expect(tooltip).to_be_hidden()
    await point.focus()
    await page.set_viewport_size({"width": 390, "height": 844})
    await expect(tooltip).to_be_hidden()
    await point.scroll_into_view_if_needed()
    await point.dispatch_event("pointerdown", {"pointerType": "touch"})
    await expect(tooltip).to_be_visible()
    assert await tooltip.evaluate(
        "node => { const r = node.getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth "
        "&& r.top >= 0 && r.bottom <= innerHeight; }"
    )
    await page.locator("#phase-filter").select_option("prefill")
    await expect(tooltip).to_be_hidden()
    await page.set_viewport_size({"width": 1400, "height": 1000})


async def check_history_links(page, url):
    fixtures = ROOT / "tests/fpm_accuracy/fixtures/dashboard"
    history = json.loads((fixtures / "history.json").read_text())
    summary = json.loads((fixtures / "synthetic-summary.json").read_text())
    summary["snapshot"].update(run_id="789", run_attempt="3", completed_at="2026-09-18T12:00:00Z")
    summary["rows"][0]["model"] = "Org/New"
    entry = copy.deepcopy(history["entries"][0])
    entry.update(snapshot=summary["snapshot"], summary_path="new-summary.json", revision_order=1)
    history["entries"].append(entry)
    await page.route("**/data/history.json", lambda route: route.fulfill(json=history))
    await page.route("**/data/new-summary.json", lambda route: route.fulfill(json=summary))
    await page.goto(url + "trends.html?branch=main")
    await expect(page.locator("#evaluation-run")).to_have_attribute(
        "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/789/attempts/3"
    )
    await page.locator(".trend-point").first.hover()
    await expect(page.locator("#trend-tooltip")).to_be_visible()
    await check_evaluation_banner(page, "2026-09-18")
    await page.locator("#model-filter").select_option("Org/Model")
    await check_evaluation_banner(page)
    await expect(page.locator("#evaluation-run")).to_have_attribute(
        "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/123/attempts/1"
    )
    await page.goto(url + "evaluation-detail.html?branch=main")
    await expect(page.locator("#evaluation-run")).to_have_attribute(
        "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/789/attempts/3"
    )
    await check_evaluation_banner(page, "2026-09-18")
    await page.locator("#evaluation-filter").select_option("1")
    await check_evaluation_banner(page)
    await expect(page.locator("#evaluation-run")).to_have_attribute(
        "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/123/attempts/1"
    )
    await page.unroute("**/data/history.json")
    await page.unroute("**/data/new-summary.json")


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
                summary["snapshot"].update(run_id="456", run_attempt="2")
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
        prepare_visualization_fixtures(site / "fpm-accuracy/data/visualization")
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
                await expect(page.locator("#freshness")).to_have_count(0)
                await check_evaluation_banner(page)
                await expect(page.locator(".evaluation-banner")).to_contain_text("Daily evaluation")
                await expect(page.locator(".evaluation-banner #evaluation-run")).to_have_attribute(
                    "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/123/attempts/1"
                )
                await expect(page.locator("thead th")).to_have_count(5)
                assert not await page.locator('a[href*="coverage.html"]').count()
                await expect(page.locator(".fpm-tabs a")).to_have_count(5)
                assert "op-based" not in await page.locator("body").inner_text()
                await expect(page.locator(".overview-config-row").first).to_contain_text("7 excluded or unavailable")
                await expect(page.locator("#overview-body")).not_to_contain_text("predicted")
                await expect(page.locator("#overview-body")).not_to_contain_text("coverage")
                await expect(
                    page.locator(".overview-config-row").first.get_by_role("link", name="3D Viz →")
                ).to_have_attribute(
                    "href",
                    "3d-visualization.html?branch=main&configuration="
                    + quote(data["rows"][0]["configuration_id"], safe="")
                    + "&snapshot="
                    + data["rows"][0]["snapshot_id"],
                )
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
                await expect(page.locator("#evaluation-run")).to_be_hidden()
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
                await check_predictor_views(page, url, data, screenshot)
                await page.set_viewport_size({"width": 1400, "height": 1000})
                await page.goto(url + "trends.html?branch=main")
                await expect(page.locator("#trend-table")).to_have_count(0)
                await expect(page.locator("#trend-chart .trend-point")).to_have_count(2)
                await page.locator("#phase-filter").select_option("decode")
                await page.locator("#trend-chart .trend-point").first.focus()
                await expect(page.locator("#trend-chart .trend-point").first).to_be_focused()
                await expect(page.locator("#trend-tooltip")).to_be_visible()
                await page.keyboard.press("Escape")
                await expect(page.locator("#trend-tooltip")).to_be_hidden()
                await check_trend_tooltip(page, screenshot)
                await page.locator("#branch").select_option("release/0.12.0")
                await expect(page.locator("#evaluation-run")).to_be_hidden()
                await expect(page.locator("#dashboard-status")).to_contain_text("main only")
                await page.goto(url + "evaluation-detail.html?branch=main")
                await expect(page.locator("#distribution table")).to_be_visible()
                await expect(page.locator("#evaluation-run")).to_have_attribute(
                    "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/123/attempts/1"
                )
                await page.locator("#phase-filter").select_option("decode")
                await expect(page.locator("#error-heatmap table")).to_be_visible()
                await page.locator("#method-filter").select_option("nowarmup")
                await expect(page.locator("#error-heatmap")).to_contain_text("No FPM input")
                await expect(page.locator("#error-heatmap table")).to_have_count(0)
                await expect(page.locator("#distribution table")).to_be_visible()
                await expect(page.locator("#variant-filter")).to_have_count(0)
                await page.locator("#method-filter").select_option("warmup")
                await expect(page.locator("#variant-evidence a")).to_contain_text("fpm-1")
                await expect(page.locator("#error-heatmap table")).to_be_visible()
                await page.locator("#search-filter").fill("nonexistent")
                await expect(page.locator("#configuration-filter option")).to_have_count(0)
                await page.locator("#search-filter").fill("")
                await expect(page.locator("#distribution table")).to_be_visible()
                await page.set_viewport_size({"width": 390, "height": 844})
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.set_viewport_size({"width": 1400, "height": 1000})
                await check_history_links(page, url)
                await page.goto(url + "3d-visualization.html?branch=release/0.12.0")
                await expect(page.get_by_role("link", name="Evaluation run")).to_have_attribute(
                    "href", "https://github.com/ai-dynamo/aisimulate/actions/runs/456/attempts/2"
                )
                await expect(page.locator("#gv-left-chart .plot-container")).to_be_visible(timeout=30000)
                held = asyncio.Event()
                release = asyncio.Event()

                async def hold_chunk(route):
                    held.set()
                    await release.wait()
                    await route.continue_()

                await page.route("**/data/visualization/*.json.gz", hold_chunk)
                # Capture the superseded redraw's completion, including all chunk loads.
                await page.evaluate("""() => {
                    const original = Promise.allSettled;
                    Promise.allSettled = function(values) {
                        Promise.allSettled = original;
                        window.heldRedraw = original.call(Promise, values);
                        return window.heldRedraw;
                    };
                }""")
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
                await page.evaluate("async () => { await window.heldRedraw; }")
                await expect(page.locator("#gv-left-chart")).to_have_attribute("data-density", "sample")
                await expect(page.locator("#gv-right-chart")).to_have_attribute("data-density", "sample")
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
                await check_evaluation_banner(page)
                await expect(page.locator("#gv-left-count")).to_contain_text("40")
                await page.locator("#gv-phase").select_option("decode")
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
                mobile = await browser.new_page(viewport={"width": 390, "height": 844}, has_touch=True)
                await mobile.goto(url + "trends.html?branch=main")
                await mobile.locator(".trend-point").first.tap()
                await expect(mobile.locator("#trend-tooltip")).to_be_visible()
                await mobile.locator("h2").first.tap()
                await expect(mobile.locator("#trend-tooltip")).to_be_hidden()
                await mobile.close()
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
