# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
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


def test_accuracy_catalog_packages_main_and_release_data_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    relative = PAGES.DOCS_ROOT / "e2e-accuracy/summary.json"
    source = repo / relative
    source.parent.mkdir(parents=True)
    main_summary = json.loads((ROOT / relative).read_text())
    release_summary = json.loads(json.dumps(main_summary))
    release_summary["snapshot"]["evaluated_revision"] = {"branch": "release/0.12.0", "commit_sha": "a" * 40}
    source.write_text(json.dumps(release_summary))
    (source.parent / "app.js").write_text("untrusted release javascript")
    git("add", ".")
    git("commit", "-qm", "release snapshot")
    release_sha = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/release/0.12.0", release_sha)
    git("update-ref", "refs/remotes/origin/release/nested/rc1", release_sha)
    source.unlink()
    git("add", ".")
    git("commit", "-qm", "no snapshot yet")
    git("update-ref", "refs/remotes/origin/release/0.13.0", git("rev-parse", "HEAD"))
    source.write_text(json.dumps(main_summary))
    output = tmp_path / "site"
    (output / "e2e-accuracy").mkdir(parents=True)
    PAGES._build_accuracy_catalog(repo, output, True)
    catalog = json.loads((output / "e2e-accuracy/branches.json").read_text())
    entries = {entry["branch"]: entry for entry in catalog["branches"]}
    assert list(entries) == ["main", "release/0.12.0", "release/0.13.0", "release/nested/rc1"]
    assert entries["main"]["status"] == "historical"
    assert entries["release/0.12.0"]["status"] == "evaluated"
    assert entries["release/nested/rc1"]["status"] == "inherited"
    assert entries["release/0.13.0"]["summary_path"] is None
    assert entries["release/0.13.0"]["status"] == "unavailable"
    assert entries["release/0.12.0"]["published_from_commit"] == release_sha
    assert not list(output.rglob("*.js"))
    assert len({entry["summary_path"] for entry in entries.values() if entry["summary_path"]}) == 3
    published = json.loads((output / "e2e-accuracy" / entries["release/0.12.0"]["summary_path"]).read_text())
    assert published == release_summary
    # A deleted release disappears on the next catalog build.
    git("update-ref", "-d", "refs/remotes/origin/release/0.12.0")
    PAGES._build_accuracy_catalog(repo, output, True)
    assert "release/0.12.0" not in [
        entry["branch"] for entry in json.loads((output / "e2e-accuracy/branches.json").read_text())["branches"]
    ]


def test_malformed_accuracy_data_fails_publication() -> None:
    for value in (
        "[]",
        "{}",
        "not json",
        '{"schema_version": 1, "models": [], "snapshot": {"evaluated_revision": {"commit_sha": "oops"}}}',
    ):
        with unittest.TestCase().assertRaises(PAGES.PagesBuildError):
            PAGES._accuracy_summary(value)


def test_incomplete_branch_summary_cannot_replace_public_site() -> None:
    from copy import deepcopy

    valid = json.loads((ROOT / PAGES.DOCS_ROOT / "e2e-accuracy/summary.json").read_text())
    invalid_cases = [
        {"schema_version": 1, "models": [], "snapshot": {}},
        {**valid, "scope": {}},
        {**valid, "totals": {}},
    ]
    missing_metric = deepcopy(valid)
    del missing_metric["models"][0]["workloads"][0]["gpus"][0]["aic"]["ttft_mape_pct"]
    invalid_cases.append(missing_metric)
    broken_coverage = deepcopy(valid)
    broken_coverage["totals"]["aisimulate"]["status_counts"]["success"] += 1
    invalid_cases.append(broken_coverage)
    bad_topology = deepcopy(valid)
    bad_topology["models"][0]["workloads"][0]["gpus"][0]["topologies"] = [{"id": "missing-points"}]
    invalid_cases.append(bad_topology)
    for invalid in invalid_cases:
        with unittest.TestCase().assertRaises(PAGES.PagesBuildError):
            PAGES._accuracy_summary(json.dumps(invalid))


def test_generated_topology_summary_satisfies_publication_contract() -> None:
    spec = importlib.util.spec_from_file_location("accuracy_tests", ROOT / "tests/test_e2e_accuracy_overview.py")
    accuracy_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accuracy_tests)
    summary = accuracy_tests._summary()
    assert PAGES._accuracy_summary(json.dumps(summary)) == summary
    summary["models"][0]["workloads"][0]["gpus"][0]["topologies"][0]["points"][0]["measured"]["ttft_relative"] = None
    with unittest.TestCase().assertRaises(PAGES.PagesBuildError):
        PAGES._accuracy_summary(json.dumps(summary))
