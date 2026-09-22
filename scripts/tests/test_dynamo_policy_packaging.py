# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matching wheel contracts, including nightly and pre-adapter release trees."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import build_release_artifacts as release
from scripts.build_dynamo_policy import DYNAMO_REVISION, check_policy_dependency, source_identity, verify_wheels


class PolicyPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source_tree(self, *, plugin: bool = True, version: str = "0.13.0") -> None:
        files = ["Cargo.toml", "Cargo.lock", "crates/core/Cargo.toml", "python/aisimulate/pyproject.toml"]
        if plugin:
            files += [
                "crates/dynamo-policy/Cargo.toml",
                "crates/dynamo-policy/Cargo.lock",
                "python/aisimulate-dynamo-policy/pyproject.toml",
            ]
        for name in files:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((ROOT / name).read_text().replace("0.13.0", version))

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

    def test_checked_in_manifests_and_immutable_router_graph_match(self) -> None:
        self.source_tree()
        self.assertEqual(self.manifests(), ("0.13.0", "0.13.0"))
        check_policy_dependency(self.root)

    def test_mismatched_versions_pins_and_locks_fail(self) -> None:
        cases = [
            ("Cargo.toml", 'version = "0.13.0"'),
            ("python/aisimulate-dynamo-policy/pyproject.toml", 'version = "0.13.0"'),
            ("python/aisimulate-dynamo-policy/pyproject.toml", '"aisimulate==0.13.0"'),
            ("crates/dynamo-policy/Cargo.toml", 'version = "0.13.0"'),
            ("crates/dynamo-policy/Cargo.toml", 'version = "=0.13.0"'),
            ("Cargo.lock", 'name = "aisimulate-core"\nversion = "0.13.0"'),
            ("crates/dynamo-policy/Cargo.lock", 'name = "aisimulate-core"\nversion = "0.13.0"'),
            ("crates/dynamo-policy/Cargo.lock", 'name = "aisimulate-dynamo-policy"\nversion = "0.13.0"'),
        ]
        for relative, text in cases:
            with self.subTest(relative=relative, text=text):
                self.source_tree()
                self.replace(relative, text, text.replace("0.13.0", "0.12.0"))
                with self.assertRaises(AssertionError):
                    self.manifests()

    def test_local_router_overrides_and_heavy_dependencies_fail(self) -> None:
        for relative, before, after in [
            ("crates/dynamo-policy/Cargo.toml", DYNAMO_REVISION, "a" * 40),
            ("crates/dynamo-policy/Cargo.toml", 'rev = "', 'path = "../router", rev = "'),
            ("crates/dynamo-policy/Cargo.toml", "default-features = false", "default-features = true"),
            ("crates/dynamo-policy/Cargo.lock", 'name = "dynamo-kv-hashing"', 'name = "dynamo-mocker"'),
            ("crates/dynamo-policy/Cargo.lock", f"#{DYNAMO_REVISION}", "#" + "b" * 40),
        ]:
            with self.subTest(relative=relative, before=before):
                self.source_tree()
                self.replace(relative, before, after)
                with self.assertRaises(AssertionError):
                    check_policy_dependency(self.root)
        self.source_tree()
        with (self.root / "Cargo.toml").open("a") as handle:
            handle.write('\n[patch."https://github.com/ai-dynamo/dynamo"]\ndynamo-kv-router = { path = "../router" }\n')
        with self.assertRaisesRegex(AssertionError, "overrides"):
            check_policy_dependency(self.root)

    def test_archive_provenance_is_explicit_and_does_not_claim_cleanliness(self) -> None:
        with self.assertRaisesRegex(ValueError, "require --source-revision"):
            source_identity(self.root, None)
        with self.assertRaisesRegex(ValueError, "full 40-character"):
            source_identity(self.root, "main")
        self.assertEqual(source_identity(self.root, DYNAMO_REVISION), (DYNAMO_REVISION, None))
        self.source_tree()
        for args in (
            ["init", "-q"],
            ["add", "."],
            ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "source"],
        ):
            subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        self.assertEqual(source_identity(self.root, revision), (revision, False))
        with self.assertRaisesRegex(ValueError, "differs"):
            source_identity(self.root, DYNAMO_REVISION)
        (self.root / "untracked.txt").write_text("dirty")
        self.assertEqual(source_identity(self.root, revision), (revision, True))

    def test_wheel_metadata_and_exact_runtime_dependency_must_match(self) -> None:
        def wheel(name: str, version: str = "0.13.0", pin: str = "0.13.0") -> Path:
            path = self.root / f"{name}.whl"
            prefix = f"{name.replace('-', '_')}-{version}.dist-info"
            with zipfile.ZipFile(path, "w") as archive:
                metadata = f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
                if name == "aisimulate-dynamo-policy":
                    metadata += f"Requires-Dist: aisimulate=={pin}\n"
                archive.writestr(f"{prefix}/METADATA", metadata)
                for legal in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
                    archive.writestr(f"{prefix}/licenses/{legal}", (ROOT / legal).read_bytes())
            return path

        base = wheel("aisimulate")
        policy = wheel("aisimulate-dynamo-policy")
        verify_wheels([base, policy], "0.13.0")
        for wheels in ([base], [base, base], [base, wheel("wrong-package")]):
            with self.subTest(wheels=wheels), self.assertRaises(AssertionError):
                verify_wheels(wheels, "0.13.0")
        with self.assertRaisesRegex(AssertionError, "version differs"):
            verify_wheels([base, wheel("aisimulate-dynamo-policy", version="0.12.0")], "0.13.0")
        with self.assertRaisesRegex(AssertionError, "exact base pin"):
            verify_wheels([base, wheel("aisimulate-dynamo-policy", pin="0.12.0")], "0.13.0")

    def test_stamp_preserves_resolved_dependencies_and_supports_historical_roots(self) -> None:
        for plugin, base_version in [(True, "0.13.0"), (False, "0.12.0")]:
            for suffix in (".dev20260922", ".dev202609220000001234"):
                with (
                    self.subTest(plugin=plugin, base=base_version, suffix=suffix),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    self.root = Path(temp)
                    self.source_tree(plugin=plugin, version=base_version)
                    for args in (
                        ["init", "-q"],
                        ["add", "."],
                        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "source"],
                    ):
                        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)
                    locked_before = self.foreign_packages()
                    command = [sys.executable, str(ROOT / "scripts/apply_dev_version.py"), suffix, str(self.root)]
                    subprocess.run(command, check=True, capture_output=True)
                    stamped = {path: path.read_bytes() for path in self.root.rglob("*.toml")}
                    stamped.update({path: path.read_bytes() for path in self.root.rglob("Cargo.lock")})
                    subprocess.run(command, check=True, capture_output=True)
                    self.assertTrue(all(path.read_bytes() == content for path, content in stamped.items()))
                    self.assertEqual(self.foreign_packages(), locked_before)
                    for path in self.root.rglob("Cargo.lock"):
                        packages = tomllib.loads(path.read_text())["package"]
                        for package in packages:
                            if package["name"] in {"aisimulate-core", "aisimulate-dynamo-policy"}:
                                self.assertEqual(package["version"], f"{base_version}-dev.{suffix[4:]}")
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
                    if plugin:
                        check_policy_dependency(self.root)

    def foreign_packages(self) -> dict[str, list[dict]]:
        return {
            str(path.relative_to(self.root)): [
                package
                for package in tomllib.loads(path.read_text())["package"]
                if package["name"] not in {"aisimulate-core", "aisimulate-dynamo-policy"}
            ]
            for path in self.root.rglob("Cargo.lock")
        }


if __name__ == "__main__":
    unittest.main()
