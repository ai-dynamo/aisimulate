#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise AgentX navigation using synthetic numerical fixtures, never published."""

import asyncio
import functools
import http.server
import json
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from scripts.e2e_accuracy.agentx_comparison import build as build_comparison
from scripts.e2e_accuracy.agentx_contract import canonical
from scripts.pages.agentx_pages import write_assets
from scripts.pages.prepare_e2e_accuracy_pages import artifact_key
from test_agentx_accuracy import bundle, result


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def fixtures(site):
    destination = site / "e2e-accuracy"
    shutil.copytree(ROOT / "pages/e2e-accuracy", destination)
    source = destination / "agentx"
    write_assets(ROOT, source)
    runs = []
    for point, lanes in ((446063, 1), (446068, 4), (446069, 8), (446060, 10), (446061, 13), (446062, 14)):
        runs.append(result(id=f"agentx-{point}", lanes=lanes))
    for length in (4096, 32768):
        for lanes in (1, 4, 16):
            for cache in ("cold", "warm"):
                for repeat in range(3):
                    runs.append(
                        result(
                            id=f"self-p{length}-l{lanes}-{cache}-r{repeat}",
                            category="self_collected",
                            model="Qwen/Qwen3.5-397B-A17B",
                            backend="vllm",
                            lanes=lanes,
                            initial_input_tokens=length,
                            cache_state=cache,
                            repeat=repeat,
                            boundary="complete_finite_cohort",
                        )
                    )
    report = {
        "schema": "aisimulate.replay.dashboard.v1",
        "metadata": {"title": "Replay", "model": "Test model"},
        "prediction": {},
        "telemetry": {"completion_status": "completed", "profile_start_ms": 1000},
        "phases": {"preparation": {"start_ms": 0, "end_ms": 1000}, "profile": {"start_ms": 1000, "end_ms": 3000}},
        "requests": [
            {
                "uuid": "0123456789abcdef",
                "request_id": "0123456789abcdef",
                "phase": "profile",
                "arrival_time_ms": 1000,
                "terminal_time_ms": 2000,
                "ttft_ms": 120,
                "itl_ms": 15,
                "e2e_latency_ms": 1000,
                "input_length": 10,
                "output_length": 100,
                "terminal_status": "completed",
                "agentic": {},
                "routing_history": [],
                "admission_history": [],
            }
        ],
        "samples": [],
        "lifecycle": [],
    }
    report["requests"].append(
        {
            **report["requests"][0],
            "uuid": "fedcba9876543210",
            "request_id": "fedcba9876543210",
            "phase": "preparation",
            "arrival_time_ms": 0,
            "terminal_time_ms": 500,
        }
    )
    reference = [
        {
            "key": row["request_id"],
            "request_id": row["request_id"],
            "phase": row["phase"],
            "start_ns": str(row["arrival_time_ms"] * 1_000_000),
            "end_ns": str(row["terminal_time_ms"] * 1_000_000),
            "input_tokens": 10,
            "output_tokens": 100,
            "ttft_ms": 100,
            "tpot_ms": 10,
            "e2e_ms": 1000,
            "conversation_id": "fixture",
            "error": None,
        }
        for row in report["requests"]
    ]
    silicon = {
        "phases": {
            key: {
                "start_ns": str(value["start_ms"] * 1_000_000),
                "end_ns": str(value["end_ms"] * 1_000_000),
                "duration_s": (value["end_ms"] - value["start_ms"]) / 1000,
                "series": {},
            }
            for key, value in report["phases"].items()
        }
    }
    report["comparison"] = build_comparison(
        {"reference": reference, "silicon": silicon, "namespace_map": {}}, report, report, runs[0]
    )
    summary = json.loads(bundle()["summary.json"])
    summary["runs"] = []
    for run in runs:
        run["report"] = report
        entry = {k: v for k, v in run.items() if k not in {"pairs", "report"}}
        entry["detail_path"] = f"details/{run['id']}.json"
        summary["runs"].append(entry)
        target = source / "branches" / artifact_key("main") / entry["detail_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical({"schema": summary["schema"], "snapshot": summary["snapshot"], **run}))
    (source / "branches" / artifact_key("main") / "summary.json").write_bytes(canonical(summary))
    (source / "branches.json").write_bytes(
        canonical(
            {
                "schema_version": 1,
                "branches": [
                    {
                        "branch": "main",
                        "status": "evaluated",
                        "summary_path": f"branches/{artifact_key('main')}/summary.json",
                    },
                    {"branch": "release/empty", "status": "unavailable"},
                ],
            }
        )
    )
    # InfX may have no snapshot while AgentX is available. These views are independent.
    (destination / "branches.json").write_bytes(
        canonical(
            {
                "schema_version": 1,
                "default_branch": "main",
                "branches": [
                    {"branch": branch, "status": "unavailable", "published_from_commit": None, "summary_path": None}
                    for branch in ("main", "release/empty")
                ],
            }
        )
    )


async def check():
    with tempfile.TemporaryDirectory() as directory:
        site = Path(directory)
        fixtures(site)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=site))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            async with async_playwright() as browser_api:
                browser = await browser_api.chromium.launch()
                page = await browser.new_page(viewport={"width": 1500, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(f"http://127.0.0.1:{server.server_port}/e2e-accuracy/?tab=agentx&branch=main")
                await expect(page.locator("#agentx-view")).to_be_visible()
                await expect(page.locator("#branch-select option:checked")).to_have_text("main")
                await expect(page.locator(".agentx-table")).to_have_count(1)
                await expect(page.locator("#tab-agentx .agentx-new")).to_have_text("NEW")
                await expect(page.locator(".agentx-beta a")).to_have_attribute(
                    "href", "../feature-support-matrix/#agentx"
                )
                await expect(page.locator(".agentx-beta")).to_have_text(
                    "Beta · check the AgentX feature support matrix ↗."
                )
                table = page.locator(".agentx-table")
                await expect(table.locator("tr[data-agentx-parent]:visible")).to_have_count(9)
                await expect(table.locator("thead th")).to_have_count(8)
                await expect(table.locator(".source-tag")).to_have_text(["AgentX official", "Self-collected"])
                qwen = table.locator(".agentx-model").nth(1)
                await qwen.locator("button").focus()
                await page.keyboard.press("Enter")
                await expect(table.locator("tr[data-agentx-parent]:visible")).to_have_count(6)
                await page.keyboard.press("Enter")
                await expect(table.locator("tr[data-agentx-parent]:visible")).to_have_count(9)
                await expect(table).to_contain_text("AISim self-collected dataset: coming soon")
                assert "huggingface.co" not in await table.inner_html()
                await table.locator("tr[data-agentx-parent]").nth(6).locator("a").first.click()
                await expect(page.locator(".agentx-beta")).to_have_count(0)
                await expect(page.locator("#length")).to_have_value("4096")
                await expect(page.locator("#cache")).to_have_value("cold")
                await expect(page.locator("#repeat")).to_have_value("0")
                await expect(page.locator("#selected-run-label")).to_contain_text("4,096 tokens")
                frame = page.locator(".replay-dashboard")
                await expect(page.locator("iframe")).to_have_count(0)
                await expect(frame.locator(".main-tabs [role=tab]")).to_have_text(
                    ["Dashboard", "Request timeline", "Evidence"]
                )
                await expect(frame.locator("#profile-accuracy")).to_be_visible()
                await expect(frame.locator("#profile-accuracy")).to_contain_text("Selected run accuracy · Profile")
                await expect(frame).not_to_contain_text("Request accounting")
                await expect(frame).not_to_contain_text("All measured request comparisons")
                assert await frame.evaluate(
                    "root => root.querySelector('#profile-accuracy').nextElementSibling.id === 'chart-grid'"
                )
                await expect(frame.locator("#plot-reference_ttft .plotly")).to_have_count(1)
                await expect(frame.locator("#plot-reference_tpot .plotly")).to_have_count(1)
                await expect(frame.locator("#scores-reference_ttft")).to_contain_text("Request MAPE 20%")
                await frame.locator('[data-expand="reference_ttft"]').click()
                await expect(frame.locator("#expanded-scores")).to_contain_text("Request MAPE 20%")
                await frame.locator('[aria-label="Close expanded chart"]').click()
                await frame.locator('[data-view="timeline"]').click()
                await expect(page).to_have_url(re.compile("view=timeline"))
                await expect(frame.locator('[data-origin="silicon"]')).to_have_count(1)
                await expect(frame.locator('[data-origin="simulated"]')).to_have_count(1)
                await frame.locator('[data-origin="silicon"]').click()
                await expect(frame.locator("#request-details")).to_contain_text("20%")
                await frame.locator('[aria-label="Close request details"]').click()
                await frame.locator('[data-stage="preparation"]').click()
                await expect(page).to_have_url(re.compile("stage=preparation"))
                await frame.locator('[data-origin="silicon"]').click()
                assert "%" not in await frame.locator("#request-details").inner_text()
                await frame.locator('[aria-label="Close request details"]').click()
                await page.reload()
                await expect(frame.locator('[data-stage="preparation"]')).to_have_attribute("aria-selected", "true")
                await page.locator("#length").select_option("32768")
                await page.locator("#cache").select_option("warm")
                await page.locator("#repeat").select_option("2")
                await expect(page.locator("#selected-run-label")).to_contain_text("warm · repeat 2")
                assert "length=32768" in page.url and "repeat=2" in page.url
                await page.reload()
                await expect(page.locator("#repeat")).to_have_value("2")
                await page.locator("#theme-mode").select_option("system")
                await page.emulate_media(color_scheme="light")
                await expect(page.locator("html")).to_have_attribute("data-theme", "light")
                await page.emulate_media(color_scheme="dark")
                await expect(page.locator("html")).to_have_attribute("data-theme", "dark")
                await page.locator("#theme-mode").select_option("light")
                await page.emulate_media(color_scheme="dark")
                await expect(page.locator("html")).to_have_attribute("data-theme", "light")
                await frame.locator('[data-view="dashboard"]').click()
                await expect(frame.locator("#profile-accuracy")).to_be_hidden()
                await frame.locator('[data-stage="profile"]').click()
                await expect(frame.locator("#profile-accuracy")).to_be_visible()
                await expect(frame.locator("#profile-accuracy")).to_contain_text("Concurrency summary")
                await page.reload()
                await expect(frame.locator("#profile-accuracy")).to_be_visible()
                await page.goto(page.url.split("#")[0] + "#view=accuracy&stage=profile&variant=observed")
                await expect(frame.locator("#profile-accuracy")).to_be_visible()
                await expect(page).to_have_url(re.compile("view=point"))
                await frame.locator('[data-view="dashboard"]').focus()
                await page.keyboard.press("ArrowRight")
                await page.keyboard.press("ArrowRight")
                await expect(frame.locator("#view-evidence")).to_be_visible()
                await expect(frame.locator("#view-evidence")).to_contain_text("Silicon dataset revision")
                await page.set_viewport_size({"width": 390, "height": 844})
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                await page.locator("#back").click()
                await expect(page.locator("#agentx-view")).to_be_visible()
                await page.locator("#branch-select").select_option("release/empty")
                await expect(page.locator("#agentx-content")).to_contain_text("No AgentX evaluation")
                await page.reload()
                await expect(page.locator("#agentx-view")).to_be_visible()
                await expect(page.locator("#agentx-content")).to_contain_text("No AgentX evaluation")
                await page.locator("#branch-select").select_option("main")
                await expect(page.locator(".agentx-table")).to_have_count(1)
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                assert not errors, errors
                await browser.close()
        finally:
            server.shutdown()
            server.server_close()
    print("AgentX browser checks passed")


if __name__ == "__main__":
    asyncio.run(check())
