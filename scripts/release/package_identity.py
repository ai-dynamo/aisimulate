# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve current and historical artifact identities without renaming imports."""

import argparse
import importlib.metadata
import tomllib
from pathlib import Path

PACKAGE_NAMES = ("aisim", "aisimulate")


def wheels(directory: Path) -> list[Path]:
    return sorted(path for name in PACKAGE_NAMES for path in directory.glob(f"{name}-*.whl"))


def single_wheel(directory: Path) -> Path:
    matches = wheels(directory)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one AISim wheel in {directory}, found {[p.name for p in matches]}")
    return matches[0]


def installed_distribution() -> importlib.metadata.Distribution:
    installed = []
    for name in PACKAGE_NAMES:
        try:
            installed.append(importlib.metadata.distribution(name))
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) > 1:
        raise RuntimeError(
            "aisim and aisimulate are both installed and own overlapping files. "
            "Use a clean environment, or uninstall aisimulate then force-reinstall aisim. "
            "Update dependencies that still require aisimulate before migrating."
        )
    if installed:
        return installed[0]
    raise importlib.metadata.PackageNotFoundError("aisim (or historical aisimulate)")


def core_package(core: Path) -> str:
    name = tomllib.loads((core / "Cargo.toml").read_text())["package"]["name"]
    if name not in ("aisim-core", "aisimulate-core"):
        raise ValueError(f"unsupported AISim core package: {name}")
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(single_wheel(parser.parse_args().directory))
