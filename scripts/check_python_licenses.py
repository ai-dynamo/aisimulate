#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply one Python dependency license policy in Full CI and nightlies."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "python/aisimulate/pyproject.toml"
PIP_LICENSES = "pip-licenses==5.5.5"
ALLOWED_LICENSES = (
    "MIT;MIT License;MIT-CMU;MIT AND PSF-2.0;MIT OR AFL-2.1;Apache-2.0;"
    "Apache Software License;Apache-2.0 OR BSD-2-Clause;Apache Software License; BSD License;"
    "BSD License;BSD-2-Clause;BSD-3-Clause;3-Clause BSD License;"
    "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0;MPL-2.0 AND MIT;"
    "PSF-2.0;Python Software Foundation License"
)


def check_licenses(python: str, inventory: Path | None = None, pyproject: Path | None = None) -> int:
    with (pyproject or PYPROJECT).open("rb") as manifest:
        dependencies = tomllib.load(manifest)["project"]["dependencies"]
    dependencies = [d for d in dependencies if not d.lower().startswith("aisimulate-core")]
    with tempfile.TemporaryDirectory(prefix="aisimulate-licenses-") as temporary:
        requirements = Path(temporary) / "runtime-deps.txt"
        requirements.write_text("\n".join(dependencies), encoding="utf-8")
        subprocess.run(
            [python, "-m", "pip", "install", "--quiet", "-r", str(requirements), PIP_LICENSES],
            check=True,
        )

    # pip-licenses otherwise excludes prettytable and wcwidth even though
    # both are also AISimulate runtime dependencies.
    # Audit the installed environment, including tooling; blanket exclusions
    # must not silently exempt a future runtime dependency with the same name.
    command = [python, "-m", "piplicenses", "--with-system"]
    # Preserve the release policy: detailed findings stay out of public CI logs.
    result = subprocess.run(
        [*command, "--allow-only", ALLOWED_LICENSES],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        print("::error::Python dependency license check failed; run pip-licenses locally for details")
        return 1
    if inventory is not None:
        inventory.parent.mkdir(parents=True, exist_ok=True)
        with inventory.open("w", encoding="utf-8", newline="") as output:
            subprocess.run([*command, "--format=csv"], stdout=output, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Interpreter whose environment is checked")
    parser.add_argument("--inventory", type=Path, help="Optional CSV output after the license check succeeds")
    parser.add_argument("--pyproject", type=Path, help="Package manifest from the source revision being built")
    args = parser.parse_args()
    return check_licenses(args.python, args.inventory, args.pyproject)


if __name__ == "__main__":
    sys.exit(main())
