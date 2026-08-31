# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_runner.py"
SPEC = importlib.util.spec_from_file_location("build_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_runner)


class CargoCommandTest(unittest.TestCase):
    def test_direct_cargo_command_has_no_rustup_shorthand(self) -> None:
        manifest = Path("/tmp/runner/Cargo.toml")
        command = build_runner._cargo_build_command(
            cargo="/opt/homebrew/bin/cargo",
            rustup="rustup",
            toolchain=None,
            manifest=manifest,
        )
        self.assertEqual(
            command,
            [
                "/opt/homebrew/bin/cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(manifest),
            ],
        )

    def test_toolchain_command_uses_explicit_rustup_run(self) -> None:
        manifest = Path("/tmp/runner/Cargo.toml")
        command = build_runner._cargo_build_command(
            cargo="cargo",
            rustup="/opt/homebrew/bin/rustup",
            toolchain="1.93.1",
            manifest=manifest,
        )
        self.assertEqual(
            command,
            [
                "/opt/homebrew/bin/rustup",
                "run",
                "1.93.1",
                "cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(manifest),
            ],
        )


if __name__ == "__main__":
    unittest.main()
