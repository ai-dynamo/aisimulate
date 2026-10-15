#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check public E2E tabs and chart interactions with local fixture data."""

import asyncio
import functools
import http.server
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


async def check():
    with tempfile.TemporaryDirectory() as directory:
        site = Path(directory)
        shutil.copytree(ROOT / "pages/e2e-accuracy", site / "e2e-accuracy")
        path = site / "e2e-accuracy/summary.json"
        data = json.loads(path.read_text())
        # Synthetic chart values exercise the rich contract; never publish this fixture.
        for model in data["models"]:
            for workload in model["workloads"]:
                for gpu in workload["gpus"]:
                    for topology in gpu["topologies"]:
                        topology.update(total_gpus=8, is_multinode=False)
                        for point in topology["points"]:
                            for name in ("measured", "aic", "aisimulate"):
                                values = point[name]
                                for metric in ("ttft", "tpot"):
                                    relative = values[f"{metric}_relative"]
                                    values[f"{metric}_ms"] = relative * 20 if relative is not None else None
                                exists = values["ttft_ms"] is not None
                                values.update(
                                    e2e_ms=3000 if exists else None,
                                    output_per_gpu=100 if exists else None,
                                    total_per_gpu=200 if exists else None,
                                )
        path.write_text(json.dumps(data))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=site))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={"width": 1500, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(f"http://127.0.0.1:{server.server_port}/e2e-accuracy/")
                await expect(page.locator(".model-row").first).to_be_visible()
                await expect(page.locator("#exclude-multinode")).not_to_be_checked()
                await page.locator("#tab-details").click()
                await expect(page.locator("#details-view")).to_be_visible()
                await expect(page.locator("#matrix-layout")).to_be_hidden()
                await expect(page.locator("#details-view svg")).to_have_count(3)
                await expect(page.locator('#details-view svg[role="group"]')).to_have_count(3)
                await page.locator("#details-view .point.aic").first.focus()
                await page.keyboard.press("Enter")
                await expect(page.locator("#point-dialog")).to_be_visible()
                await expect(page.locator("#point-content")).to_contain_text("Measured silicon")
                await page.locator("#close-point").click()
                await page.locator("#exclude-outliers").check()
                await page.locator('#details-view [data-chart="throughput"]').select_option("total")
                await page.locator('#details-view [data-chart="view"]').select_option("e2e")
                await page.locator('#details-view [data-series="aic"]').focus()
                await page.keyboard.press("Shift+Enter")
                await expect(page.locator("#details-view .point.measured")).to_have_count(0)
                await expect(page.locator('#details-view [data-series="aic"]')).to_have_attribute(
                    "aria-pressed", "true"
                )
                await page.locator('#details-view [data-series="measured"]').click()
                await expect(page.locator("#details-view .point.measured").first).to_be_visible()
                topology = await page.locator('[data-filter="topology"]').input_value()
                await page.reload()
                await expect(page.locator('[data-filter="topology"]')).to_have_value(topology)
                await expect(page.locator("#exclude-outliers")).to_be_checked()
                await page.locator("#tab-overview").click()
                await expect(page.locator("#matrix-layout")).to_be_visible()
                await expect(page.locator("#drilldown")).to_be_visible()
                await page.locator("#tab-details").click()
                await expect(page.locator('[data-filter="topology"]')).to_have_value(topology)
                if output := os.environ.get("E2E_SCREENSHOT"):
                    await page.screenshot(path=output, full_page=True)
                await page.set_viewport_size({"width": 390, "height": 844})
                await expect(page.locator("#details-view")).to_be_visible()
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                assert not errors, errors
                await browser.close()
        finally:
            server.shutdown()
            server.server_close()
    print("E2E browser checks passed")


if __name__ == "__main__":
    asyncio.run(check())
