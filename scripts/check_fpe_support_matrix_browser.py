#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shipped FPE page against a local, branch-specific fixture site.

Executed by Pages CI with Playwright and Chromium installed.
Run directly with ``python scripts/check_fpe_support_matrix_browser.py``.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import functools
import http.server
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/fpe-support-matrix")
CATALOG = {
    "schema_version": 1,
    "default": "main",
    "branches": [
        {"name": "main", "path": ".", "status": "available"},
        {"name": "release/0.13.0", "path": "branches/release/0.13.0", "status": "available"},
        {"name": "release/0.14.0", "path": "branches/release/0.14.0", "status": "available"},
        {"name": "release/0.15.0", "status": "unavailable"},
    ],
}


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class FpeBrowserTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.site = Path(temporary.name)
        page = cls.site / "fpe-support-matrix/index.html"
        page.parent.mkdir()
        shutil.copyfile(ROOT / "pages/fpe-support-matrix/index.html", page)
        template_path = ROOT / "python/aisimulate/src/aisimulate_core/systems/fpe_support_matrix/b200_sxm.csv"
        with template_path.open() as handle:
            template = next(csv.DictReader(handle))
        cls.csvs = {}
        for branch, model in (
            ("main", "Fixture/Main"),
            ("release/0.13.0", "Fixture/ReleaseAlpha"),
            ("release/0.14.0", "Fixture/ReleaseBeta"),
            ("release/0.15.0", "Fixture/ReleaseGamma"),
        ):
            data = cls.site / DATA
            if branch != "main":
                data = data / "branches" / branch
            data.mkdir(parents=True)
            snapshot = {
                "branch": branch,
                "source_sha": "a" * 40,
                "generated_at": "2026-09-15T00:00:00Z",
                "run_url": "https://github.com/ai-dynamo/aisimulate/actions/runs/1",
            }
            (data / "index.json").write_text(json.dumps({"files": ["b200_sxm.csv"], "snapshot": snapshot}))
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=template.keys())
            writer.writeheader()
            writer.writerow({**template, "HuggingFaceID": model, "Status": "PASS", "SourceSHA": "a" * 40})
            cls.csvs[branch] = output.getvalue()
            (data / "b200_sxm.csv").write_text(output.getvalue())
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(cls.site))
        )
        cls.addClassCleanup(server.server_close)
        cls.addClassCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{server.server_port}/fpe-support-matrix/"

    async def asyncSetUp(self):
        playwright = await async_playwright().start()
        self.addAsyncCleanup(playwright.stop)
        browser = await playwright.chromium.launch(executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"))
        self.addAsyncCleanup(browser.close)
        self.page = await browser.new_page()
        self.page.set_default_timeout(30000)
        self.errors = []
        self.requests = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("request", lambda request: self.requests.append(request.url))
        self.catalog = copy.deepcopy(CATALOG)

        async def catalog(route):
            await route.fulfill(json=self.catalog)

        await self.page.route("**/data/fpe-support-matrix/branches.json", catalog)

    async def asyncTearDown(self):
        self.assertEqual(self.errors, [])

    async def test_main_and_release_deep_links_load_their_own_data(self):
        await self.page.goto(self.url)
        await expect(self.page.get_by_text("Fixture/Main", exact=True)).to_be_visible()
        await expect(self.page.locator("tbody tr")).to_have_count(1)
        self.requests.clear()
        await self.page.goto(self.url + "?branch=release%2F0.13.0&q=ReleaseAlpha")
        await expect(self.page.get_by_text("Fixture/ReleaseAlpha", exact=True)).to_be_visible()
        await expect(self.page.get_by_text("Qualified snapshot:", exact=False)).to_be_visible()
        await expect(self.page.get_by_label("Branch:")).to_have_value("release/0.13.0")
        await expect(self.page.get_by_placeholder("Search models...")).to_have_value("ReleaseAlpha")
        csv_requests = [urlsplit(url).path for url in self.requests if urlsplit(url).path.endswith(".csv")]
        self.assertEqual(csv_requests, ["/data/fpe-support-matrix/branches/release/0.13.0/b200_sxm.csv"])
        await expect(self.page.get_by_text("Fixture/Main", exact=True)).to_have_count(0)

    async def test_unavailable_retry_and_branch_switch_preserve_search(self):
        await self.page.goto(self.url + "?branch=release%2F0.15.0&q=Fixture")
        await expect(self.page.get_by_text("Results not available yet", exact=True)).to_be_visible()
        await expect(self.page.locator("tbody tr")).to_have_count(0)
        await expect(self.page.get_by_role("link", name="View coverage runs")).to_have_attribute(
            "href", "https://github.com/ai-dynamo/aisimulate/actions/workflows/release-nightly-ci.yml"
        )
        self.catalog["branches"][-1].update(status="available", path="branches/release/0.15.0")
        await self.page.get_by_role("button", name="Check again", exact=True).click()
        await expect(self.page.get_by_text("Fixture/ReleaseGamma", exact=True)).to_be_visible()
        await expect(self.page.get_by_placeholder("Search models...")).to_have_value("Fixture")
        await self.page.get_by_label("Branch:").select_option("main")
        await expect(self.page.get_by_text("Fixture/Main", exact=True)).to_be_visible()
        self.assertEqual(parse_qs(urlsplit(self.page.url).query), {"q": ["Fixture"]})

    async def test_delayed_previous_branch_response_cannot_replace_current_rows(self):
        pending = {}
        requested = asyncio.Event()

        async def delay(route):
            pending["route"] = route
            requested.set()

        delayed_csv = "**/branches/release/0.14.0/b200_sxm.csv"
        await self.page.route(delayed_csv, delay)
        await self.page.goto(self.url + "?branch=release%2F0.13.0")
        await expect(self.page.get_by_text("Fixture/ReleaseAlpha", exact=True)).to_be_visible()
        await self.page.evaluate("""() => {
            const originalParse = Papa.parse;
            Papa.parse = (text, ...args) => {
                const result = originalParse(text, ...args);
                if (text.includes('Fixture/ReleaseBeta')) window.delayedCsvParsed = true;
                return result;
            };
        }""")
        await self.page.get_by_label("Branch:").select_option("release/0.14.0")
        await asyncio.wait_for(requested.wait(), timeout=30)
        await expect(self.page.locator("tbody tr")).to_have_count(0)
        await self.page.get_by_label("Branch:").select_option("release/0.13.0")
        await expect(self.page.get_by_text("Fixture/ReleaseAlpha", exact=True)).to_be_visible()
        async with self.page.expect_response(delayed_csv) as response:
            await pending["route"].fulfill(body=self.csvs["release/0.14.0"], content_type="text/csv")
        self.assertEqual((await (await response.value).body()).decode(), self.csvs["release/0.14.0"])
        # The browser must consume and parse its body; Playwright reading it is insufficient.
        await self.page.wait_for_function("window.delayedCsvParsed === true")
        # Drain the page's promise continuations and allow React to render their result.
        await self.page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        await expect(self.page.get_by_label("Branch:")).to_have_value("release/0.13.0")
        await expect(self.page.get_by_text("Fixture/ReleaseAlpha", exact=True)).to_be_visible()
        await expect(self.page.get_by_text("Fixture/ReleaseBeta", exact=True)).to_have_count(0)

    async def test_mismatched_snapshot_fails_before_any_csv_request(self):
        index = json.loads((self.site / DATA / "index.json").read_text())

        async def wrong_owner(route):
            await route.fulfill(json=index)

        await self.page.route("**/branches/release/0.13.0/index.json", wrong_owner)
        await self.page.goto(self.url + "?branch=release%2F0.13.0")
        await expect(self.page.get_by_text("FPE snapshot is missing or belongs to a different branch")).to_be_visible()
        await expect(self.page.locator("tbody tr")).to_have_count(0)
        self.assertFalse(any(urlsplit(url).path.endswith(".csv") for url in self.requests))


if __name__ == "__main__":
    unittest.main(verbosity=2)
