#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and browser-test legacy matrix provenance; optionally capture the real page.

Install Chromium with ``uv run --python 3.12 --with playwright playwright install chromium``,
then run ``uv run --python 3.12 --with playwright python scripts/check_legacy_support_matrix_browser.py``.
Use --browser-executable to select an already installed Chrome/Chromium instead.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import tempfile
import threading
from pathlib import Path

from build_pages_site import ROOT, build_site
from playwright.sync_api import expect, sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass


def check_page(url: str, index: dict, browser_executable: str | None, screenshot: Path | None) -> None:
    with sync_playwright() as playwright:
        options = {"executable_path": browser_executable} if browser_executable else {}
        with playwright.chromium.launch(headless=True, **options) as browser:
            page = browser.new_page(viewport={"width": 1440, "height": 960}, locale="en-US", timezone_id="UTC")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url)
            expect(page.get_by_text("Historical snapshot", exact=True)).to_be_visible()
            expect(page.locator("tbody tr").first).to_be_visible()
            rows = page.locator("tbody").inner_text()
            snapshot = index["snapshot"]
            assert snapshot["qualification"] == "not_recorded"

            def check_snapshot(value: dict | None, valid: bool) -> None:
                expect(page.get_by_text("Historical snapshot", exact=True)).to_be_visible()
                expect(page.locator("tbody")).to_have_text(rows, use_inner_text=True)
                expect(page.get_by_text("Qualification not recorded", exact=False)).to_be_visible()
                if valid:
                    assert value is not None
                    expect(page.locator("time")).to_have_attribute("datetime", value["data_updated_at"])
                    commit_url = f"https://github.com/ai-dynamo/aisimulate/commit/{value['data_commit']}"
                    expect(page.locator(f'a[href="{commit_url}"]')).to_have_count(2)
                else:
                    expect(page.get_by_text("Latest data change: not recorded", exact=False)).to_be_visible()
                    expect(page.locator("time")).to_have_count(0)
                    expect(page.locator('a[href^="https://github.com/ai-dynamo/aisimulate/commit/"]')).to_have_count(0)

            # Capture the unmodified, packaged data before introducing test fixtures.
            check_snapshot(snapshot, "data_commit" in snapshot)
            if screenshot:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.evaluate("document.fonts.ready")
                page.locator("img").evaluate_all(
                    "images => Promise.all(images.map(img => img.decode().catch(() => {})))"
                )
                page.screenshot(path=str(screenshot))

            valid = {
                "kind": "historical",
                "qualification": "not_recorded",
                "data_commit": "a" * 40,
                "data_updated_at": "2026-09-04T18:42:50Z",
            }
            cases = [
                (valid, True),
                ({**valid, "data_updated_at": "2026-09-04T18:42:50.000Z"}, True),
                ({**valid, "data_updated_at": "2026-09-04T18:42:50.123Z"}, True),
                ({**valid, "data_updated_at": "2024-02-29T00:00:00Z"}, True),
                (None, False),
                ({}, False),
                ({**valid, "kind": "qualified"}, False),
                ({**valid, "qualification": "qualified"}, False),
                ({key: value for key, value in valid.items() if key != "qualification"}, False),
                ({**valid, "data_commit": "not-a-commit"}, False),
                ({**valid, "data_commit": ["a" * 40]}, False),
                ({**valid, "data_updated_at": "2026-02-31T00:00:00Z"}, False),
                ({**valid, "data_updated_at": "2026-02-31T00:00:00.000Z"}, False),
                ({**valid, "data_updated_at": "2026-02-29T00:00:00Z"}, False),
                ({**valid, "data_updated_at": "2026-09-04T24:00:00Z"}, False),
                ({**valid, "data_updated_at": "invalid"}, False),
                ({**valid, "data_updated_at": ["2026-09-04T18:42:50Z"]}, False),
            ]
            served_index = dict(index)
            page.route("**/data/support-matrix/index.json", lambda route: route.fulfill(json=served_index))
            for value, is_valid in cases:
                if value is None:
                    served_index.pop("snapshot", None)
                else:
                    served_index["snapshot"] = value
                page.reload()
                check_snapshot(value, is_valid)
            assert not errors, errors
            print(f"Passed real dataset and {len(cases)} browser cases; no JavaScript errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-executable", help="Path to an installed Chrome or Chromium executable")
    parser.add_argument(
        "--screenshot", type=Path, help="Capture the real packaged page before testing altered metadata"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="legacy-matrix-browser-") as temporary:
        site = Path(temporary) / "site"
        build_site(ROOT, site)
        index = json.loads((site / "data/support-matrix/index.json").read_text())
        handler = functools.partial(QuietHandler, directory=str(site))
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                check_page(
                    f"http://127.0.0.1:{server.server_port}/support-matrix/",
                    index,
                    args.browser_executable,
                    args.screenshot,
                )
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    main()
