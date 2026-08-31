# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_pages_site.py"
SPEC = importlib.util.spec_from_file_location("build_pages_site", SCRIPT_PATH)
assert SPEC and SPEC.loader
PAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAGES)


class PagesSiteTest(unittest.TestCase):
    def test_public_pages_artifact_is_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "site"
            files = PAGES.build_site(ROOT, output_dir)

            self.assertIn(Path("index.html"), files)
            self.assertIn(Path("support-matrix/index.html"), files)
            self.assertIn(Path("data/support-matrix/index.json"), files)
            self.assertTrue(
                any(path.match("data/support-matrix/*.csv") for path in files)
            )

            self.assertFalse(any(path.parts[0] == "universe" for path in files))
            self.assertFalse(any(path.suffix == ".md" for path in files))
            self.assertFalse(any("src" in path.parts for path in files))

            for path in files:
                if path.suffix not in {".html", ".js"}:
                    continue
                text = (output_dir / path).read_text()
                self.assertNotIn("raw.githubusercontent.com", text)
                self.assertNotIn("api.github.com/repos/ai-dynamo/aisimulate", text)

    def test_deployed_legacy_matrix_uses_packaged_public_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "site"
            PAGES.build_site(ROOT, output_dir)

            page = (output_dir / "support-matrix" / "index.html").read_text()
            self.assertIn("../data/support-matrix", page)
            self.assertNotIn("raw.githubusercontent.com", page)


if __name__ == "__main__":
    unittest.main()
