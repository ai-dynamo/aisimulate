#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose the default systems root (packaged specs plus the pinned op-latency dataset) and print its path.

Rust tests and other consumers that never run the Python data library read operation tables from a directory named
by ``AICONFIGURATOR_SYSTEMS_PATH``. The first call downloads the pinned dataset into the data cache; later calls
reuse it. Only needs ``pydantic``, ``pyarrow`` and ``pyyaml`` (it does not import the native extension)::

    export AICONFIGURATOR_SYSTEMS_PATH="$(python scripts/ci/compose_systems_root.py)"
    # GitHub Actions: python scripts/ci/compose_systems_root.py --github-env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "aisim" / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--github-env", action="store_true", help="append AICONFIGURATOR_SYSTEMS_PATH to $GITHUB_ENV")
    args = parser.parse_args(argv)
    from aisim_core.data.default_systems import default_systems_path

    root = default_systems_path()
    if args.github_env:
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as env:
            env.write(f"AICONFIGURATOR_SYSTEMS_PATH={root}\n")
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
