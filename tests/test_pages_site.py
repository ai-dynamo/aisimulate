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
            self.assertIn(Path("e2e-accuracy/index.html"), files)
            self.assertIn(Path("e2e-accuracy/app.js"), files)
            self.assertIn(Path("e2e-accuracy/styles.css"), files)
            self.assertIn(Path("e2e-accuracy/summary.json"), files)
            self.assertIn(Path("fpe-support-matrix/index.html"), files)
            self.assertIn(Path("fpe-support-matrix/fpe-support-matrix-preview.png"), files)
            self.assertIn(Path("support-matrix/index.html"), files)
            self.assertIn(Path("data/fpe-support-matrix/index.json"), files)
            self.assertIn(Path("data/support-matrix/index.json"), files)
            self.assertTrue(any(path.match("data/fpe-support-matrix/*.csv") for path in files))
            self.assertTrue(any(path.match("data/support-matrix/*.csv") for path in files))

            self.assertFalse(any(path.parts[0] == "universe" for path in files))
            self.assertFalse(any(path.suffix == ".md" for path in files))
            self.assertFalse(any("src" in path.parts for path in files))

            landing_page = (output_dir / "index.html").read_text()
            self.assertIn('href="./e2e-accuracy/"', landing_page)
            self.assertIn('href="./fpe-support-matrix/"', landing_page)
            self.assertIn('href="./support-matrix/"', landing_page)
            self.assertNotIn('href="./universe/"', landing_page)

            for path in files:
                if path.suffix not in {".html", ".js"}:
                    continue
                text = (output_dir / path).read_text()
                self.assertNotIn("raw.githubusercontent.com", text)
                self.assertNotIn("api.github.com/repos/ai-dynamo/aisimulate", text)

    def test_deployed_matrices_use_packaged_public_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "site"
            PAGES.build_site(ROOT, output_dir)

            legacy_page = (output_dir / "support-matrix" / "index.html").read_text()
            self.assertIn("../data/support-matrix", legacy_page)
            self.assertNotIn("raw.githubusercontent.com", legacy_page)

            fpe_page = (output_dir / "fpe-support-matrix" / "index.html").read_text()
            self.assertIn("../data/fpe-support-matrix", fpe_page)
            self.assertIn('href="../"', fpe_page)
            self.assertNotIn("raw.githubusercontent.com", fpe_page)
            self.assertNotIn("api.github.com/repos/ai-dynamo/aisimulate", fpe_page)

    def test_public_artifact_rejects_symlinked_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            private_file = temporary_root / "private.html"
            symlink = temporary_root / "public.html"
            destination = temporary_root / "site" / "public.html"
            private_file.write_text("private")
            symlink.symlink_to(private_file)

            with self.assertRaisesRegex(PAGES.PagesBuildError, "cannot be a symlink"):
                PAGES._copy_file(symlink, destination)


if __name__ == "__main__":
    unittest.main()
