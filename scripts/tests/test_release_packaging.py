# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-wheel release, immutable dependencies, and atomic nightly stamping."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import build_release_artifacts as release


class ReleasePackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source_tree(self, *, binding: bool = True, version: str = "0.13.0") -> None:
        files = ["Cargo.toml", "Cargo.lock", "crates/core/Cargo.toml", "python/aisimulate/pyproject.toml"]
        if binding:
            files.append("crates/python/Cargo.toml")
        for name in files:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            text = (ROOT / name).read_text().replace("0.13.0", version)
            if not binding:
                if name == "Cargo.lock":
                    text = "[[package]]".join(
                        record
                        for record in text.split("[[package]]")
                        if not record.startswith('\nname = "aisimulate-python"\n')
                    )
                text = text.replace('    "crates/python",\n', "")
                text = text.replace("../../crates/python/Cargo.toml", "../../crates/core/Cargo.toml")
            target.write_text(text)

    def manifests(self) -> tuple[str, str]:
        with patch.multiple(
            release,
            ROOT=self.root,
            EXPECTED_PYTHON_PROJECTS={self.root / "python/aisimulate/pyproject.toml": "aisimulate"},
            EXPECTED_CRATE=self.root / "crates/core/Cargo.toml",
        ):
            return release.check_manifests()

    def replace(self, relative: str, old: str, new: str) -> None:
        path = self.root / relative
        self.assertIn(old, path.read_text())
        path.write_text(path.read_text().replace(old, new, 1))

    def test_checked_in_single_wheel_and_pinned_router_graph_match(self) -> None:
        self.source_tree()
        self.assertEqual(self.manifests(), ("0.13.0", "0.13.0"))

    def test_versions_dependency_pins_and_resolved_sources_must_match(self) -> None:
        for relative, before, after in [
            ("Cargo.toml", 'version = "0.13.0"', 'version = "0.12.0"'),
            ("crates/python/Cargo.toml", 'version = "0.13.0"', 'version = "0.12.0"'),
            ("crates/python/Cargo.toml", 'version = "=0.13.0"', 'version = "=0.12.0"'),
            ("crates/python/Cargo.toml", release.DYNAMO_REVISION, "a" * 40),
            ("crates/python/Cargo.toml", 'rev = "', 'path = "../router", rev = "'),
            ("crates/python/Cargo.toml", "default-features = false", "default-features = true"),
            ("Cargo.lock", 'name = "dynamo-kv-hashing"', 'name = "dynamo-mocker"'),
            ("Cargo.lock", f"#{release.DYNAMO_REVISION}", "#" + "b" * 40),
            (
                "Cargo.lock",
                'name = "aisimulate-core"\nversion = "0.13.0"',
                'name = "aisimulate-core"\nversion = "0.12.0"',
            ),
            (
                "Cargo.lock",
                'name = "aisimulate-python"\nversion = "0.13.0"',
                'name = "aisimulate-python"\nversion = "0.12.0"',
            ),
        ]:
            with self.subTest(relative=relative, before=before):
                self.source_tree()
                self.replace(relative, before, after)
                with self.assertRaises(AssertionError):
                    self.manifests()

    def test_local_overrides_and_core_back_edges_are_rejected(self) -> None:
        self.source_tree()
        with (self.root / "Cargo.toml").open("a") as handle:
            handle.write('\n[patch."https://github.com/ai-dynamo/dynamo"]\ndynamo-kv-router = { path = "../router" }\n')
        with self.assertRaisesRegex(AssertionError, "overrides"):
            self.manifests()
        self.source_tree()
        self.replace(
            "Cargo.lock",
            'name = "aisimulate-core"\nversion = "0.13.0"\ndependencies = [',
            'name = "aisimulate-core"\nversion = "0.13.0"\ndependencies = [\n "aisimulate-python",',
        )
        with self.assertRaisesRegex(AssertionError, "independent"):
            self.manifests()

    def test_nightly_stamp_is_idempotent_and_preserves_dependencies(self) -> None:
        for binding, version in [(True, "0.13.0"), (False, "0.12.0")]:
            for suffix in (".dev20260922", ".dev202609220000001234"):
                with self.subTest(binding=binding, suffix=suffix), tempfile.TemporaryDirectory() as temp:
                    self.root = Path(temp)
                    self.source_tree(binding=binding, version=version)
                    for args in (
                        ["init", "-q"],
                        ["add", "."],
                        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "source"],
                    ):
                        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)
                    before = self.foreign_packages()
                    command = [sys.executable, str(ROOT / "scripts/apply_dev_version.py"), suffix, str(self.root)]
                    subprocess.run(command, check=True, capture_output=True)
                    stamped = self.contents()
                    subprocess.run(command, check=True, capture_output=True)
                    self.assertEqual(stamped, self.contents())
                    self.assertEqual(before, self.foreign_packages())
                    packages = tomllib.loads((self.root / "Cargo.lock").read_text())["package"]
                    for package in packages:
                        if package["name"] in {"aisimulate-core", "aisimulate-python"}:
                            self.assertEqual(package["version"], f"{version}-dev.{suffix[4:]}")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/build_release_artifacts.py"),
                            "--root",
                            str(self.root),
                            "--check-only",
                        ],
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_binding_pin_and_lock_records_fail_before_writes(self) -> None:
        cases = [("crates/python/Cargo.toml", 'version = "=0.13.0", ', "")]
        for name in ("aisimulate-core", "aisimulate-python"):
            record = f'[[package]]\nname = "{name}"\nversion = "0.13.0"\n'
            cases.extend(
                ("Cargo.lock", record, replacement)
                for replacement in (record.replace(name, "unrelated"), record + record)
            )
        for relative, before, after in cases:
            with self.subTest(relative=relative, before=before):
                self.source_tree()
                self.replace(relative, before, after)
                self.assert_stamp_rejected_unchanged()

    def test_missing_manifest_versions_fail_before_writes(self) -> None:
        for relative in (
            "Cargo.toml",
            "crates/core/Cargo.toml",
            "crates/python/Cargo.toml",
            "python/aisimulate/pyproject.toml",
        ):
            with self.subTest(relative=relative):
                self.source_tree()
                self.replace(relative, 'version = "0.13.0"', "# missing package version")
                self.assert_stamp_rejected_unchanged()

    def assert_stamp_rejected_unchanged(self) -> None:
        before = self.contents()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/apply_dev_version.py"), ".dev202609220000001234", str(self.root)],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, self.contents(), "failed stamp must preserve every manifest and lockfile")

    def contents(self) -> dict[Path, bytes]:
        return {path: path.read_bytes() for pattern in ("*.toml", "Cargo.lock") for path in self.root.rglob(pattern)}

    def foreign_packages(self) -> list[dict]:
        return [
            package
            for package in tomllib.loads((self.root / "Cargo.lock").read_text())["package"]
            if package["name"] not in {"aisimulate-core", "aisimulate-python"}
        ]


if __name__ == "__main__":
    unittest.main()
