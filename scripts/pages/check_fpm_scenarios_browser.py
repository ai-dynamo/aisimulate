#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the strict-C scenario page with synthetic browser-only fixtures."""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import http.server
import os
import re
import sys
import threading
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.fpm_accuracy.scenario_contract import (
    GBDT_FROZEN_RECIPE,
    GBDT_ONLINE_RECIPE,
    HYBRID_ONLINE_RECIPE,
    METHODS,
    SCENARIOS,
    SELECTOR_ONLINE_RECIPE,
    applicable,
)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def artifact_key(branch):
    return hashlib.sha256(branch.encode()).hexdigest()[:16]


def fixture(branch="main"):
    """Synthetic display evidence only; never written to deployable assets."""
    snapshot = {
        "branch": branch,
        "commit_sha": "1" * 40,
        "hf_revision": "2" * 40,
        "run_id": "1234",
        "run_attempt": "1",
        "completed_at": "2026-10-29T10:47:00Z",
    }

    def result(method, scenario, percent, mode):
        if not applicable(method, scenario):
            return None
        online = scenario in ("c", "ac", "bc")
        test_count = 100 - percent if scenario in ("b", "bc") else 100
        training_count = percent if scenario in ("b", "bc") else 0 if scenario == "c" else 50
        status = {"missing": "chronology_unavailable" if online else "no_fpm_input", "error": "predictor_error"}.get(
            mode, "evaluated"
        )
        predicted = 0 if mode in ("missing", "zero", "error") else test_count - (20 if scenario == "c" else 0)
        mape = None if not predicted else percent / 10 + METHODS.index(method) + 1
        metrics = {}
        for workload in ("all", "prefill", "decode", "mixed"):
            measured = test_count if workload in ("all", "decode") else 0
            count = predicted if measured else 0
            metrics[workload] = {
                "measured_count": measured,
                "predicted_count": count,
                "unavailable_count": measured - count,
                "error_count": 0,
                "tuning_error_count": 0,
                "mape_pct": mape if count else None,
                "coverage_pct": 100 * count / measured if measured else None,
            }
        calls = min(test_count, 24) if online and status == "evaluated" else 0
        value = {
            "status": status,
            "training_count": training_count,
            "test_count": test_count,
            "online_update_count": calls,
            "metrics": metrics,
            "cost": {
                "fit_ms": 0 if scenario == "c" else 2.5,
                "predict_us": 4.2,
                "update_us": 3.1 if calls else None,
                "update_count": calls,
            },
            "cold_start": {"unavailable_before_first_prediction": test_count - predicted if scenario == "c" else 0},
            "artifact": {"id": "synthetic-self-benchmark", "path": "synthetic/table.parquet", "sha256": "3" * 64}
            if scenario in ("a", "ac")
            else None,
            "training_population_sha256": "4" * 64 if scenario in ("b", "bc") else None,
            "scoring_population_sha256": "5" * 64,
        }
        if status != "evaluated":
            value["reason"] = "Synthetic browser fixture: " + status
        if online:
            value["online_schedule"] = {
                "interval_ms": 5000,
                "time_basis": "event_time_ms",
                "selection_policy": "first_observation_at_or_after_tick",
                "warmup_policy": "scheduled_only" if scenario == "c" else "none",
                "scheduled_update_count": calls,
                "warmup_update_count": 0,
                "elapsed_ms": 120000 if status == "evaluated" else None,
                "submitted_observation_count": calls,
                "skipped_observation_count": test_count - calls,
                "update_error_count": 0,
            }
        if method in ("warmup", "nowarmup") and status == "evaluated":
            value["interpolation"] = {
                "method": "direct",
                "timing_only": True,
                "text_only": False,
                "metadata_recovered": False,
            }
        if method in ("gbdt", "hybrid", "selector"):
            recipe = {
                "gbdt": GBDT_ONLINE_RECIPE if online else GBDT_FROZEN_RECIPE,
                "hybrid": HYBRID_ONLINE_RECIPE,
                "selector": SELECTOR_ONLINE_RECIPE,
            }[method]
            value["online_model" if online else "offline_model"] = {
                "recipe": copy.deepcopy(recipe),
                "diagnostics": {"synthetic_fixture": True, "native_state": {"ready": predicted > 0}}
                if status == "evaluated"
                else None,
            }
        return value

    rows = []
    for index, mode in enumerate(("ready", "missing", "zero", "error")):
        row = {
            "configuration_id": "synthetic-" + mode,
            "snapshot_id": "synthetic-snapshot",
            "model": "Synthetic model " + str(index + 1),
            "gpu": "Synthetic GPU",
            "framework": "Synthetic framework",
            "parallelism": "TP1",
            "ordering": "file_order_fallback" if mode == "missing" else "chronological",
        }
        row["results"] = {
            scenario: {method: result(method, scenario, 50, mode) for method in METHODS} for scenario in SCENARIOS
        }
        row["split_results"] = {
            str(percent): {
                scenario: {method: result(method, scenario, percent, mode) for method in METHODS}
                for scenario in ("b", "bc")
            }
            for percent in range(10, 100, 10)
        }
        rows.append(row)
    return {
        "schema_version": 13,
        "protocol_id": "fpm-scenarios-v13",
        "snapshot": snapshot,
        "methods": list(METHODS),
        "scenarios": list(SCENARIOS),
        "rows": rows,
    }


async def main():
    for filename in ("index.html", "predictors.html", "trends.html", "evaluation-detail.html", "3d-visualization.html"):
        html = (ROOT / "pages/fpm-accuracy" / filename).read_text()
        assert html.index('href="3d-visualization.html"') < html.index('href="scenario-comparison.html"')
    handler = functools.partial(QuietHandler, directory=ROOT / "pages")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/fpm-accuracy/scenario-comparison.html"
    current = fixture()
    summary = {"snapshot": copy.deepcopy(current["snapshot"])}
    delayed = asyncio.Event()
    release_requested = asyncio.Event()
    errors = []
    catalog = {
        "schema_version": 1,
        "default_branch": "main",
        "branches": [
            {"branch": branch, "status": "available", "summary_path": f"branches/{artifact_key(branch)}/summary.json"}
            for branch in ("main", "release/0.12.0", "empty")
        ],
    }
    try:
        async with async_playwright() as playwright:
            launch = {"headless": True}
            if executable := os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"):
                launch["executable_path"] = executable
            browser = await playwright.chromium.launch(**launch)
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.route("https://avatars.githubusercontent.com/**", lambda route: route.abort())
            await page.route("**/branches.json", lambda route: route.fulfill(json=catalog))

            async def serve_summary(route):
                value = summary
                if artifact_key("release/0.12.0") in route.request.url:
                    value = {"snapshot": fixture("release/0.12.0")["snapshot"]}
                await route.fulfill(json=value)

            async def serve_scenarios(route):
                if artifact_key("release/0.12.0") in route.request.url:
                    release_requested.set()
                    await delayed.wait()
                    await route.fulfill(json=fixture("release/0.12.0"))
                elif artifact_key("empty") in route.request.url:
                    await route.fulfill(status=404, body="Unavailable fixture")
                else:
                    await route.fulfill(json=current)

            await page.route("**/branches/*/summary.json", serve_summary)
            await page.route("**/branches/*/scenarios.json", serve_scenarios)

            async def ready(scenario="a"):
                await page.goto(url + "?branch=main&scenario=" + scenario)
                await expect(page.locator("#scenario-comparison")).to_have_attribute("data-ready", "true")

            async def field(label, value):
                element = page.locator("#sc-evidence-values > div").filter(
                    has=page.locator("dt", has_text=re.compile("^" + re.escape(label) + "$"))
                )
                await expect(element.locator("dd")).to_have_text(value)

            await ready()
            await expect(page.locator('.fpm-tabs a[aria-current="page"]')).to_have_text("Scenario Comparison")
            await expect(page.locator(".fpm-tabs a")).to_have_count(6)
            await expect(page.locator("#sc-table th[data-method]")).to_have_count(7)
            await expect(page.locator("#sc-body .sc-config")).to_have_count(3)
            await expect(page.locator("#sc-count")).to_contain_text("1 unavailable hidden")
            await expect(page.locator("#evaluation-hf")).to_have_attribute("href", re.compile("2{40}$"))
            await expect(page.locator("#sc-protocol-version-note")).to_contain_text("including warmup")
            await expect(page.locator("#sc-cadence-note")).to_contain_text("five seconds after the first")
            for scenario in SCENARIOS:
                await page.locator(f'input[name="scenario"][value="{scenario}"]').check()
                await expect(page.locator("#sc-detail-body tr")).to_have_count(5)
                assert "scenario=" + scenario in page.url
                expected_methods = [method for method in METHODS if applicable(method, scenario)]
                row = page.locator("#sc-body tr").first
                await expect(row.locator(".sc-score")).to_have_count(len(expected_methods))
            await page.locator('input[name="scenario"][value="c"]').check()
            await page.locator('#sc-body [data-method="selector"]').first.click()
            await field("Offline preparation inputs selected", "0")
            await field("Cold-start warmup policy", "Same five-second schedule from cold start")
            await field("Immediate warmup calls", "0")
            await field("Scheduled single-observation calls · including warmup", "24")
            await expect(page.locator("#sc-evidence-heading")).to_be_focused()
            await expect(page.locator("#sc-evidence-protocol")).to_contain_text(
                "Warmup follows this same five-second schedule"
            )
            await page.locator(".sc-native-evidence summary").click()
            await expect(page.locator(".sc-native-evidence pre")).to_contain_text('"synthetic_fixture": true')
            await expect(page.locator("#sc-fallback-note")).to_contain_text("B, C, and B+C receive no original-A prior")
            await page.locator("#sc-metric").select_option("coverage")
            await expect(page.locator('#sc-body [data-method="selector"]').first.locator("strong")).to_have_text("80%")
            zero_row = page.locator("#sc-body tr").filter(has_text="Synthetic model 3")
            await expect(zero_row.locator('[data-method="linear"] strong')).to_have_text("0%")
            await page.locator("#sc-metric").select_option("mape")
            await expect(zero_row.locator('[data-method="linear"]')).to_have_text("No predictions")
            await expect(
                page.locator("#sc-body tr").filter(has_text="Synthetic model 4").locator('[data-method="linear"]')
            ).to_have_text("Predictor error")
            await page.locator("#sc-workload").select_option("prefill")
            await expect(page.locator('#sc-body [data-method="linear"]').first).to_have_text("No predictions")
            await page.locator("#sc-metric").select_option("predict")
            await expect(page.locator('#sc-body [data-method="linear"]').first.locator("strong")).to_have_text("4.2 µs")
            await page.locator("#sc-reset").click()

            # A/C do not depend on the B split. B/B+C select the exact grid entry.
            original = await page.locator('#sc-body [data-method="linear"]').first.text_content()
            await page.locator("#sc-train-percent").fill("30")
            await expect(page.locator('#sc-body [data-method="linear"]').first).to_have_text(original)
            await page.locator('input[name="scenario"][value="b"]').check()
            await expect(page.locator('#sc-body [data-method="linear"] strong').first).to_have_text("6%")
            await page.locator("#sc-train-percent").focus()
            await page.keyboard.press("ArrowRight")
            await expect(page.locator("#sc-train-percent")).to_have_value("40")
            await expect(page.locator('#sc-body [data-method="linear"] strong').first).to_have_text("7%")
            assert "train_percent=40" in page.url
            await page.reload()
            await expect(page.locator("#sc-train-percent")).to_have_value("40")
            await page.locator('[data-sort="linear"]').click()
            await expect(page.locator('th[data-method="linear"]')).to_have_attribute("aria-sort", "ascending")
            await page.locator("#sc-model").select_option("Synthetic model 1")
            await expect(page.locator("#sc-body .sc-config")).to_have_count(1)
            await page.locator("#sc-reset").click()
            await expect(page.locator("#sc-train-percent")).to_have_value("50")

            # Missing chronology is inspectable and never presented as a score.
            await page.locator('input[name="scenario"][value="c"]').check()
            await page.locator("#sc-show-unavailable").check()
            await expect(page.locator("#sc-body .sc-config")).to_have_count(4)
            missing = page.locator("#sc-body tr").filter(has_text="Synthetic model 2")
            await missing.locator('[data-method="gbdt"]').click()
            await expect(page.locator("#sc-evidence-count-note")).to_contain_text(
                "No fitting, prediction, or update ran"
            )
            await expect(missing.locator("strong")).to_have_count(0)
            await page.locator("#sc-reset").click()
            await page.locator('#sc-body [data-method="hybrid"]').first.click()
            for width in (1440, 390):
                await page.set_viewport_size({"width": width, "height": 1000})
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.get_by_role("button", name=re.compile("Switch to .* theme")).click()
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                if screenshot := os.environ.get("SCREENSHOT_PATH"):
                    target = Path(screenshot).with_stem(Path(screenshot).stem + f"-{width}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    await page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
                    await page.wait_for_timeout(180)
                    await page.screenshot(path=str(target), full_page=True)
                    await page.screenshot(path=str(target.with_stem(target.stem + "-viewport")))
            await page.set_viewport_size({"width": 1440, "height": 1000})

            # Malformed split data is unavailable rather than replaced by the 50% score.
            current = fixture()
            del current["rows"][0]["split_results"]["30"]
            await ready("b")
            await expect(page.locator("#sc-split-note")).to_contain_text("incomplete or invalid split grid")
            await page.locator("#sc-show-unavailable").check()
            await expect(
                page.locator("#sc-body tr").filter(has_text="Synthetic model 1").locator("strong")
            ).to_have_count(0)

            # Render hostile data only as text, including recipe labels and native diagnostics.
            current = fixture()
            payload = '<img src=x onerror="window.injected=true">'
            current["rows"][0]["model"] = payload
            current["rows"][0]["results"]["c"]["gbdt"]["online_model"]["recipe"]["label"] = payload
            current["rows"][0]["results"]["c"]["gbdt"]["online_model"]["diagnostics"] = {"text": payload}
            await ready("c")
            await page.locator('#sc-body [data-method="gbdt"]').first.click()
            assert not await page.evaluate("Boolean(window.injected)")
            await expect(page.locator("#scenario-comparison img")).to_have_count(0)
            await expect(page.locator("#sc-body")).to_contain_text(payload)

            # Reject old fast-warmup data, offline C seeds, non-strict schedules, and stale provenance.
            invalid = []
            old = fixture()
            old.update(schema_version=12, protocol_id="fpm-scenarios-v12")
            invalid.append((old, "requires the strict five-second protocol"))
            for field_name, value in (
                ("warmup_policy", "until_first_prediction_per_worker_workload"),
                ("warmup_update_count", 1),
                ("interval_ms", 1000),
            ):
                broken = fixture()
                broken["rows"][0]["results"]["c"]["gbdt"]["online_schedule"][field_name] = value
                invalid.append((broken, "strict five-second admission schedule"))
            broken = fixture()
            broken["rows"][0]["results"]["c"]["gbdt"]["training_count"] = 20
            invalid.append((broken, "cannot contain offline training"))
            for name in ("hf_revision", "commit_sha", "run_id", "completed_at"):
                broken = fixture()
                broken["snapshot"][name] = {
                    "hf_revision": "a" * 40,
                    "commit_sha": "b" * 40,
                    "run_id": "9999",
                    "completed_at": "2026-10-28T10:47:00Z",
                }[name]
                invalid.append((broken, "does not match this evaluation snapshot"))
            for current, message in invalid:
                await page.goto(url + "?branch=main&scenario=c")
                await expect(page.locator("#scenario-comparison")).to_have_attribute("data-ready", "error")
                await expect(page.locator("#sc-status")).to_contain_text(message)
                await expect(page.locator("#evaluation-run")).to_have_count(0)
                await expect(page.locator("#sc-detail")).to_be_hidden()

            # A slower previous branch cannot replace the selected branch's empty state.
            current = fixture()
            await ready()
            await page.locator("#branch").select_option("release/0.12.0")
            await asyncio.wait_for(release_requested.wait(), timeout=5)
            await page.locator("#branch").select_option("empty")
            await expect(page.locator("#scenario-comparison")).to_have_attribute("data-ready", "empty")
            delayed.set()
            await page.wait_for_timeout(200)
            await expect(page.locator("#sc-status")).to_contain_text("has not been published")
            await expect(page.locator("#evaluation-run")).to_have_count(0)
            await page.locator("#branch").select_option("main")
            await expect(page.locator("#scenario-comparison")).to_have_attribute("data-ready", "true")
            assert not errors, errors
            await browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print("FPM scenario browser checks passed (synthetic fixtures only).")


if __name__ == "__main__":
    asyncio.run(main())
