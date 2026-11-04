#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise AgentX navigation using synthetic numerical fixtures, never published."""

import asyncio
import functools
import hashlib
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
from test_agentx_accuracy import bundle, result
from test_benchmark_features import metadata as feature_fixture

from scripts.e2e_accuracy.agentx_comparison import build as build_comparison
from scripts.e2e_accuracy.agentx_contract import canonical
from scripts.pages.agentx_pages import write_assets
from scripts.pages.prepare_e2e_accuracy_pages import artifact_key


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def fixtures(site, weka=False):
    destination = site / "e2e-accuracy"
    shutil.copytree(ROOT / "pages/e2e-accuracy", destination, dirs_exist_ok=True)
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
    if weka:
        archive = source / "archive/finite-v1"
        shutil.copytree(source / "branches", archive / "branches")
        shutil.copyfile(source / "branches.json", archive / "branches.json")
        runs = [run for run in runs if run["category"] == "agentx"]
        for lanes in (1, 4, 8, 10, 13, 14):
            feature = feature_fixture()
            feature["benchmark_id"] = f"qwen-weka-l{lanes}"
            runs.append(
                result(
                    id=feature["benchmark_id"],
                    category="self_collected",
                    model="Qwen/Qwen3.5-397B-A17B",
                    backend="vllm",
                    lanes=lanes,
                    initial_input_tokens=None,
                    cache_state="recorded",
                    repeat=0,
                    boundary="successful_profile_span",
                    feature_metadata=feature,
                )
            )
    report = {
        "schema": "aisimulate.replay.dashboard.v1",
        "metadata": {"title": "Replay", "model": "Test model"},
        "prediction": {},
        "telemetry": {"completion_status": "completed", "profile_start_ms": 1000},
        "phases": {
            "preparation": {"start_ms": 0, "end_ms": 1000},
            "profile": {"start_ms": 1000, "end_ms": 3_601_000},
        },
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
    # Display fixture: zeros, one-sided and shared gaps, and a final five-second bucket.
    footprint = next(c for c in report["comparison"]["phases"]["profile"]["charts"] if c["id"] == "unique")
    footprint["score_window_end_s"] = 125.5
    for series in footprint["series"]:
        predicted = series["origin"] == "simulated"
        series["x"] = [t for t in range(1, 126) if t != 100 and not (predicted and t == 65)]
        levels = (1000, 5000, 9000) if predicted else (2000, 4000, 6000)
        series["y"] = [
            0 if t <= 30 else levels[0] if t <= 60 else levels[1] if t <= 120 else levels[2] for t in series["x"]
        ]
    footprint["scores"][0].update(
        count=93,
        denominator_count=125,
        missing_count=2,
        zero_reference_count=30,
        mape_percent=3200 / 93,
        mae=103000 / 123,
        coverage=93 / 125,
        window_seconds=[0, 125.5],
    )
    summary = json.loads(bundle()["summary.json"])
    summary["runs"] = []
    for run in runs:
        run["config_summary"] = [{"name": "Estimation", "value": "op_level"}]
        run["measured_backend_version"] = "0.31.0" if run["backend"] == "vllm" else "0.5.20"
        run["prediction_backend_version"] = "0.31.0" if run["backend"] == "vllm" else "0.5.21"
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


async def check_timeline_windows(page, site, port):
    path = site / "e2e-accuracy/agentx/branches" / artifact_key("main") / "details/agentx-446063.json"
    detail = json.loads(path.read_bytes())
    phase = detail["report"]["comparison"]["phases"]["profile"]
    template = phase["requests"][0]
    phase["requests"] = []
    identities = {}
    for name, actual, predicted in [
        ("early", (60, 90), (70, 100)),
        ("crossing", (590, 610), (595, 615)),
        ("boundary", (600, 620), (650, 670)),
        ("split", (100, 110), (1250, 1260)),
        ("failed", (700, 710), None),
        ("untimed", (None, None), None),
        ("instant", (600, 600), None),
        ("spanning", (500, 1500), None),
        ("late", (3050, 3060), None),
    ]:
        pair = []
        for origin, bounds in (("silicon", actual), ("simulated", predicted)):
            if bounds is None:
                continue
            identity = hashlib.sha256(f"{name}-{origin}".encode()).hexdigest()[:16]
            pair.append(
                {
                    **template,
                    "uuid": identity,
                    "request_id": identity,
                    "origin": origin,
                    "counterpart": None,
                    "terminal_status": "failed" if name == "failed" else "completed",
                    "arrival_time_ms": None if bounds[0] is None else 1000 + bounds[0] * 1000,
                    "terminal_time_ms": None if bounds[1] is None else 1000 + bounds[1] * 1000,
                }
            )
        identities[name] = pair[0]["uuid"]
        if len(pair) == 2:
            pair[0]["counterpart"], pair[1]["counterpart"] = pair[1]["uuid"], pair[0]["uuid"]
        phase["requests"].extend(pair)
    path.write_bytes(canonical(detail))
    await page.goto(
        f"http://127.0.0.1:{port}/e2e-accuracy/agentx/point.html?branch=main&category=agentx&model=nvidia%2FGLM-5.3-NVFP4&lanes=1#view=timeline&stage=profile"
    )
    await expect(page.locator("#timeline-page-label")).to_contain_text("1 / 6")
    scores = await page.locator("#timeline-summary .score-badges").text_content()
    for expected in [
        {"early", "crossing", "split", "spanning"},
        {"crossing", "boundary", "failed", "instant", "spanning"},
        {"split", "spanning"},
        set(),
        set(),
        {"late"},
    ]:
        visible = page.locator(".comparison-row:visible .timeline-label button")
        await expect(visible).to_have_count(len(expected))
        assert set(await visible.evaluate_all("nodes => nodes.map(n => n.dataset.request)")) == {
            identities[name] for name in expected
        }
        await expect(page.locator("#timeline-summary .score-badges")).to_have_text(scores)
        await expect(page.locator("#timeline-visible-summary")).to_have_text(f"{len(expected)} / 9 rows in this window")
        if expected != {"late"}:
            await page.locator("#timeline-next").click()
    # Scrolling between page boundaries applies the same overlap rule.
    await page.locator("#timeline-plot").evaluate("""node => {
        const label = node.querySelector('.timeline-label').getBoundingClientRect().width;
        node.scrollLeft = (node.clientWidth - label) * 1.5;
    }""")
    await expect(page.locator(".comparison-row:visible .timeline-label button")).to_have_count(2)
    await page.locator(f'.timeline-label [data-request="{identities["split"]}"]').click()
    await expect(page.locator("#request-details")).to_contain_text("Source paired")
    await page.locator('[aria-label="Close request details"]').click()
    await expect(page.locator("#timeline-summary")).to_contain_text("1 untimed requests not placed in time")


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
                await expect(table.locator("thead th")).to_have_count(9)
                hardware = table.locator(".agentx-model td:first-of-type")
                await expect(hardware.nth(0)).to_contain_text("B300 · sglang 0.5.20")
                await expect(hardware.nth(0)).not_to_contain_text("sglang 0.5.21")
                await expect(hardware.nth(1)).to_contain_text("B300 · vllm 0.31.0")
                await expect(table.locator("tr[data-agentx-parent] .agentx-perf-model")).to_have_text(
                    ["Op-based (sglang 0.5.21)"] * 6 + ["Op-based (vllm 0.31.0)"] * 3
                )
                await expect(table.locator("thead th").nth(2)).to_have_text("Performance model")
                await expect(table.locator(".agentx-model .agentx-perf-model")).to_have_text(
                    ["Op-based (sglang 0.5.21)", "Op-based (vllm 0.31.0)"]
                )
                await expect(table.locator("tr[data-agentx-parent] th .agentx-perf-model")).to_have_count(0)
                info = page.get_by_role("button", name="About point TTFT and TPOT error")
                tooltip = page.get_by_role("tooltip")
                await expect(table.locator("thead th").nth(3)).to_have_text("Point TPOT error")
                await info.hover()
                await expect(tooltip).to_be_visible()
                await expect(tooltip).to_contain_text(
                    "aggregate result across the benchmark\u2019s conversation requests"
                )
                await expect(tooltip).to_contain_text("compare requests individually, then average their errors")
                await expect(tooltip).to_contain_text("Tool-call waits are excluded from TTFT/TPOT calculations")
                await expect(tooltip).to_contain_text("over- and under-predictions cancel when averaged")
                await tooltip.hover()
                await expect(tooltip).to_be_visible()
                await page.mouse.move(5, 5)
                await expect(tooltip).to_be_hidden()
                await info.focus()
                await expect(tooltip).to_be_visible()
                await page.keyboard.press("Escape")
                await expect(tooltip).to_be_hidden()
                await info.click()
                await expect(tooltip).to_be_visible()
                await page.mouse.click(5, 5)
                await expect(tooltip).to_be_hidden()
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
                await expect(frame.locator(".agentx-summary-main h2")).to_have_text("Concurrency summary")
                await expect(frame.locator(".agentx-summary-main > p").first).to_contain_text(
                    "requests matched · Op-based perf model (vllm 0.31.0)"
                )
                await expect(frame.locator("#profile-accuracy .agentx-summary-table")).to_have_count(1)
                await expect(frame.locator("#profile-accuracy .agentx-feature-table")).to_have_count(1)
                await expect(frame.locator(".agentx-summary-table thead th")).to_have_text(
                    ["Metric", "Silicon", "AISim", "Error", "Samples"]
                )
                await expect(frame.locator(".agentx-summary-table tbody tr")).to_have_count(5)
                await expect(frame.locator("#profile-accuracy")).not_to_contain_text("Selected run accuracy")
                for metric, silicon, prediction in [
                    ("point-tpot_ms", "10", "15"),
                    ("point-ttft_ms", "100", "120"),
                    ("request-ttft_ms", "100", "120"),
                    ("request-tpot_ms", "10", "15"),
                ]:
                    row = frame.locator(f'tr[data-metric="{metric}"]')
                    await expect(row.locator("td").nth(0)).to_have_text(silicon)
                    await expect(row.locator("td").nth(1)).to_have_text(prediction)
                await expect(frame.locator('tr[data-metric="request-ttft_ms"] .agentx-samples')).to_have_text(
                    "12 requests"
                )
                await expect(frame.locator(".comparison-source-dot.silicon")).to_have_count(1)
                await expect(frame.locator(".comparison-source-dot.aisim")).to_have_count(1)
                concurrency_summary = await frame.locator(".agentx-summary-table").text_content()
                await expect(frame.locator(".agentx-summary-main ul")).to_have_count(0)
                await expect(frame.locator("#agentx-features-title")).to_have_text("Feature Parity")
                await expect(frame.locator('[data-source="silicon"] .agentx-feature-status.unknown')).to_have_count(8)
                await expect(frame.locator(".agentx-feature-table thead th")).to_have_text(
                    ["Feature", "Silicon", "AISim"]
                )
                await expect(frame.locator(".agentx-feature-icon")).to_have_count(0)
                await expect(frame.locator(".agentx-feature-note")).to_have_count(0)
                await expect(frame.locator(".agentx-feature-name")).to_have_text(
                    [
                        "Prefix caching",
                        "Sub-block prefix reuse",
                        "Host offload (G2)",
                        "Storage offload (G3)",
                        "Hybrid KV/state cache",
                        "Speculative decoding",
                        "Dynamo - KVCR",
                        "Dynamo - KV-aware routing",
                    ]
                )
                await expect(frame.locator(".agentx-more-features")).to_have_count(0)
                await expect(frame.locator(".agentx-combinations")).to_have_count(0)
                await expect(frame.locator(".agentx-features a")).to_have_attribute(
                    "href", "../../feature-support-matrix/#agentx"
                )
                panels = await frame.locator(".agentx-summary-layout > *").evaluate_all(
                    "nodes => nodes.map(n => ({tag:n.tagName, ...n.getBoundingClientRect().toJSON()}))"
                )
                assert [p["tag"] for p in panels] == ["ASIDE", "SECTION"]
                assert panels[0]["right"] <= panels[1]["left"] + 1
                assert abs(panels[0]["top"] - panels[1]["top"]) <= 1
                assert abs(panels[0]["height"] - panels[1]["height"]) <= 1
                await expect(frame).not_to_contain_text("Request accounting")
                await expect(frame).not_to_contain_text("All measured request comparisons")
                assert await frame.evaluate(
                    "root => root.querySelector('#profile-accuracy').nextElementSibling.id === 'chart-grid'"
                )
                await expect(frame.locator("#plot-reference_ttft .plotly")).to_have_count(1)
                await expect(frame.locator("#plot-reference_tpot .plotly")).to_have_count(1)
                assert await frame.locator("#plot-footprint").evaluate(
                    "plot => plot.data.some(trace => trace.name.includes('Silicon'))"
                )
                footprint = frame.locator("#plot-footprint")
                assert await footprint.evaluate("plot => plot.data.map(t => t.y)") == [
                    [1, None, 6],
                    [0.5, None, 9],
                    [-0.5, None, 3],
                ]
                footprint_score = await frame.locator("#scores-footprint").text_content()
                await expect(frame.locator("#scores-footprint")).to_contain_text("Original 1s · Curve MAPE 34.41%")
                await frame.locator('#controls-footprint [data-value="raw"]').click()
                assert await footprint.evaluate("plot => plot.data[0].y[29]") == 0
                assert await footprint.evaluate("plot => plot.data[1].y[64]") is None
                assert await footprint.evaluate("plot => plot.data[2].y[64]") is None
                assert await footprint.evaluate("plot => plot.data[0].y[99]") is None
                assert await footprint.evaluate("plot => plot.data[2].y[99]") is None
                await footprint.evaluate("plot => Plotly.relayout(plot, {'xaxis2.range': [0.5, 1]})")
                assert await footprint.evaluate("plot => plot._fullLayout.xaxis.range") == [0.5, 1]
                await expect(frame.locator("#scores-footprint")).to_have_text(footprint_score)
                await frame.locator('[data-expand="footprint"]').click()
                await expect(frame.locator('#expanded-controls [data-value="raw"]')).to_have_attribute(
                    "aria-pressed", "true"
                )
                await frame.locator('#expanded-controls [data-value="average"]').click()
                await expect(frame.locator('#controls-footprint [data-value="average"]')).to_have_attribute(
                    "aria-pressed", "true"
                )
                assert await frame.locator("#expanded-plot").evaluate("plot => plot.data[2].y") == [-0.5, None, 3]
                await expect(frame.locator("#expanded-scores")).to_have_text(footprint_score)
                await frame.locator('[aria-label="Close expanded chart"]').click()
                await expect(frame.locator("#note-cache")).to_contain_text("Silicon reference unavailable")
                await expect(frame.locator("#note-input")).to_contain_text("select In-flight")
                await expect(frame.locator("#scores-reference_ttft")).to_contain_text("Request MAPE 20%")
                await frame.locator('[data-expand="reference_ttft"]').click()
                await expect(frame.locator("#expanded-scores")).to_contain_text("Request MAPE 20%")
                await frame.locator('[aria-label="Close expanded chart"]').click()
                await frame.locator('[data-view="timeline"]').click()
                await expect(page).to_have_url(re.compile("view=timeline"))
                await expect(frame.locator(".timeline-panel > .hint")).to_contain_text("10 minutes per screen")
                await expect(frame.locator("#view-timeline .legend")).to_have_text("SiliconAISimFailed · red outline")
                await expect(frame.locator("#plot-reference_ttft .legend")).not_to_contain_text("Failed")
                await expect(frame.locator('[data-origin="silicon"]')).to_have_count(1)
                await expect(frame.locator('[data-origin="simulated"]')).to_have_count(1)
                pager = frame.locator("#timeline-page-label")
                previous = frame.locator("#timeline-previous")
                following = frame.locator("#timeline-next")
                timeline = frame.locator("#timeline-plot")
                await expect(pager).to_have_text("0\u201310 min · 1 / 6")
                await expect(previous).to_be_disabled()
                page_width = await timeline.evaluate(
                    "node => node.clientWidth - node.querySelector('.timeline-label').getBoundingClientRect().width"
                )
                assert await timeline.evaluate(
                    "(node, width) => Math.abs(node.scrollWidth - node.clientWidth - 5 * width) <= 2", page_width
                )
                label_x = (await frame.locator(".timeline-axis .timeline-label").bounding_box())["x"]
                score_badges = await frame.locator("#timeline-summary .score-badges").text_content()
                await following.focus()
                await page.keyboard.press("Enter")
                await expect(pager).to_have_text("10\u201320 min · 2 / 6")
                assert abs((await timeline.evaluate("node => node.scrollLeft")) - page_width) <= 1
                assert abs((await frame.locator(".timeline-axis .timeline-label").bounding_box())["x"] - label_x) <= 1
                await expect(frame.locator(".comparison-row:visible")).to_have_count(0)
                await expect(frame.locator("#timeline-window-empty")).to_be_visible()
                for _ in range(4):
                    await following.click()
                await expect(pager).to_have_text("50\u201360 min · 6 / 6")
                await expect(following).to_be_disabled()
                await expect(frame.locator(".comparison-row:visible")).to_have_count(0)
                await expect(frame.locator("#timeline-summary .score-badges")).to_have_text(score_badges)
                await timeline.evaluate("(node, width) => node.scrollLeft = width * 1.5", page_width)
                await expect(pager).to_have_text("15\u201325 min · 2 / 6")
                await following.click()
                await expect(pager).to_have_text("20\u201330 min · 3 / 6")
                await previous.click()
                await expect(pager).to_have_text("10\u201320 min · 2 / 6")
                viewport = page.viewport_size
                await page.set_viewport_size({"width": 390, "height": 844})
                await expect(pager).to_have_text("10\u201320 min · 2 / 6")
                await page.wait_for_function("document.documentElement.scrollWidth <= innerWidth + 1")
                await page.wait_for_function("""() => {
                    const node = document.querySelector('#timeline-plot');
                    return Math.abs(node.scrollLeft - (node.clientWidth - 140)) <= 1;
                }""")
                await page.set_viewport_size(viewport)
                await page.wait_for_function("""() => {
                    const node = document.querySelector('#timeline-plot');
                    return Math.abs(node.scrollLeft - (node.clientWidth - 230)) <= 1;
                }""")
                await previous.click()
                await expect(pager).to_have_text("0\u201310 min · 1 / 6")
                await expect(frame.locator(".comparison-row:visible")).to_have_count(1)
                await expect(frame.locator("#timeline-window-empty")).to_be_hidden()
                row = frame.locator(".comparison-row").first
                assert (await row.bounding_box())["height"] == 22
                await expect(row.locator(".timeline-label button")).to_have_text(
                    re.compile(r"^…[a-f0-9]{8} · [a-f0-9]{16}$")
                )
                await expect(row.locator(".timeline-label small")).to_have_count(0)
                await row.locator(".timeline-label button").focus()
                await page.keyboard.press("Enter")
                await expect(frame.locator("#request-details")).to_be_visible()
                await frame.locator('[aria-label="Close request details"]').click()
                await frame.locator('[data-origin="silicon"]').click()
                await expect(frame.locator("#request-details")).to_contain_text("20%")
                await frame.locator('[aria-label="Close request details"]').click()
                await frame.locator('[data-stage="preparation"]').click()
                await expect(page).to_have_url(re.compile("stage=preparation"))
                await expect(pager).to_have_text("0\u201310 min · 1 / 1")
                await expect(previous).to_be_disabled()
                await expect(following).to_be_disabled()
                await frame.locator('[data-origin="silicon"]').click()
                assert "%" not in await frame.locator("#request-details").inner_text()
                await frame.locator('[aria-label="Close request details"]').click()
                await page.reload()
                await expect(frame.locator('[data-stage="preparation"]')).to_have_attribute("aria-selected", "true")
                await page.locator("#length").select_option("32768")
                await page.locator("#cache").select_option("warm")
                await page.locator("#repeat").select_option("2")
                await expect(page.locator("#selected-run-label")).to_contain_text("warm · repeat 2")
                await expect(frame.locator(".agentx-summary-table")).to_have_text(concurrency_summary)
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
                await expect(frame.locator('#controls-footprint [data-value="average"]')).to_be_disabled()
                await expect(frame.locator('#controls-footprint [data-value="raw"]')).to_have_attribute(
                    "aria-pressed", "true"
                )
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
                await frame.locator('[data-view="dashboard"]').click()
                await expect(frame.locator(".agentx-summary-table .agentx-samples:visible")).to_have_count(0)
                await expect(frame.locator(".agentx-inline-count:visible")).to_have_count(5)
                assert await frame.locator("#profile-accuracy .table-scroll").evaluate(
                    "node => node.scrollWidth <= node.clientWidth + 1"
                )
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
                await check_timeline_windows(page, site, server.server_port)
                fixtures(site, weka=True)
                await page.goto(f"http://127.0.0.1:{server.server_port}/e2e-accuracy/?tab=agentx&branch=main")
                await expect(page.locator(".agentx-table tr[data-agentx-parent]")).to_have_count(12)
                qwen = page.locator(".agentx-model").filter(has_text="Qwen/")
                await expect(qwen.locator("button")).to_have_attribute("aria-expanded", "true")
                await page.goto(
                    f"http://127.0.0.1:{server.server_port}/e2e-accuracy/agentx/point.html?branch=main&category=self_collected&model=Qwen%2FQwen3.5-397B-A17B&lanes=14"
                )
                await expect(page.locator("#selected-run-label")).to_contain_text("Weka profile")
                await expect(page.locator("#variant-controls")).to_be_hidden()
                await expect(page.locator("#length")).to_have_count(0)
                await page.locator('[data-view="evidence"]').click()
                await expect(page.locator("#view-evidence")).to_contain_text("HF dataset record")
                await expect(page.locator("#view-evidence")).to_contain_text("Enabled does not imply exercised")
                await page.goto(
                    f"http://127.0.0.1:{server.server_port}/e2e-accuracy/agentx/point.html?branch=main&category=self_collected&model=Qwen%2FQwen3.5-397B-A17B&lanes=16&length=32768&cache=warm&repeat=2"
                )
                await expect(page).to_have_url(re.compile("campaign=finite-v1"))
                await expect(page.locator("#selected-run-label")).to_contain_text("warm · repeat 2")
                await expect(page.locator("#repeat")).to_have_value("2")
                await page.locator("#back").click()
                await expect(page.locator("#agentx-content")).to_contain_text("Archived finite campaign")
                archive = site / "e2e-accuracy/agentx/archive/finite-v1"
                key = artifact_key("main")
                summary_path = archive / "branches" / key / "summary.json"
                original_summary = summary_path.read_bytes()
                summary = json.loads(original_summary)
                records = []
                for run in summary["runs"]:
                    feature = feature_fixture()
                    feature["benchmark_id"] = run["id"]
                    if run["repeat"] == 1:
                        next(f for f in feature["features"] if f["id"] == "cache.g1_prefix")["state"] = "disabled"
                    records.append(
                        {
                            "benchmark_id": run["id"],
                            "feature_artifact_sha256": hashlib.sha256(canonical(feature)).hexdigest(),
                            "feature_metadata": feature,
                        }
                    )
                sidecar = {
                    "schema": "aisimulate.agentx.features.v1",
                    "measurement_dataset_revision": summary["snapshot"]["dataset_revision"],
                    "feature_dataset_revision": "f" * 40,
                    "measurement_catalog_sha256": "a" * 64,
                    "feature_catalog_sha256": "b" * 64,
                    "records": records,
                }
                catalog = json.loads((archive / "branches.json").read_bytes())
                entry = next(e for e in catalog["branches"] if e["branch"] == "main")
                content = canonical(sidecar)
                sidecar_path = archive / "branches" / key / "feature-evidence.json"
                sidecar_path.write_bytes(content)
                entry["feature_evidence"] = {
                    "path": f"branches/{key}/feature-evidence.json",
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                (archive / "branches.json").write_bytes(canonical(catalog))
                await page.goto(
                    f"http://127.0.0.1:{server.server_port}/e2e-accuracy/agentx/point.html?branch=main&campaign=finite-v1&category=self_collected&model=Qwen%2FQwen3.5-397B-A17B&lanes=1"
                )
                cache_feature = page.locator('[data-feature="cache.g1_prefix"]')
                await expect(cache_feature.locator('[data-source="silicon"] .agentx-feature-status')).to_have_text("On")
                await expect(cache_feature.locator('[data-source="aisim"] .agentx-feature-status')).to_have_text(
                    "?Legacy missing evidence"
                )
                assert (
                    await cache_feature.locator(".on .agentx-status-symbol").evaluate(
                        "node => getComputedStyle(node).backgroundColor"
                    )
                    == "rgb(22, 163, 74)"
                )
                await expect(page.locator(".agentx-feature-revision")).to_contain_text("f" * 12)
                await page.locator("#repeat").select_option("1")
                await expect(cache_feature.locator('[data-source="silicon"] .agentx-feature-status')).to_have_text(
                    "Off"
                )
                assert (
                    await cache_feature.locator(".off .agentx-status-symbol").evaluate(
                        "node => getComputedStyle(node).backgroundColor"
                    )
                    == "rgb(220, 38, 38)"
                )
                await page.locator('[data-view="evidence"]').click()
                await expect(page.locator("#view-evidence")).to_contain_text("Feature metadata dataset revision")
                await expect(page.locator("#view-evidence")).to_contain_text(summary["snapshot"]["dataset_revision"])
                assert summary_path.read_bytes() == original_summary
                sidecar_path.write_bytes(content + b" ")
                await page.reload()
                await expect(page.locator('[role="alert"]')).to_contain_text("checksum mismatch")
                sidecar["measurement_dataset_revision"] = "0" * 40
                content = canonical(sidecar)
                sidecar_path.write_bytes(content)
                entry["feature_evidence"]["sha256"] = hashlib.sha256(content).hexdigest()
                (archive / "branches.json").write_bytes(canonical(catalog))
                await page.reload()
                await expect(page.locator('[role="alert"]')).to_contain_text("dataset mismatch")
                assert not errors, errors
                await browser.close()
        finally:
            server.shutdown()
            server.server_close()
    print("AgentX browser checks passed")


if __name__ == "__main__":
    asyncio.run(check())
