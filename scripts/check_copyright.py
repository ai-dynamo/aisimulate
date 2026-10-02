#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check NVIDIA SPDX headers on every tracked source-like file.

The check is intentionally full-tree rather than diff-only. In addition to
Python, Rust, and shell sources, it covers the template and support-file types
called out in the OSRB review so a future change cannot reintroduce the same
gap.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

HEADER_LINES = 15
SOURCE_SUFFIXES = {
    ".env",
    ".j2",
    ".js",
    ".ps1",
    ".py",
    ".pyi",
    ".rs",
    ".rule",
    ".sh",
}
APACHE_LICENSE_IDENTIFIER = r"SPDX-License-Identifier:\s*Apache-2\.0"
LICENSE_MARKER = re.compile(
    rf"^\s*(?:"
    rf"(?:\#|//)\s*{APACHE_LICENSE_IDENTIFIER}"
    rf"|/\*+\s*{APACHE_LICENSE_IDENTIFIER}\s*\*/"
    rf"|<!--\s*{APACHE_LICENSE_IDENTIFIER}\s*-->"
    rf"|\*\s*{APACHE_LICENSE_IDENTIFIER}(?:\s*\*/)?"
    rf"|{APACHE_LICENSE_IDENTIFIER}"
    rf")\s*$",
    re.MULTILINE,
)
COPYRIGHT_MARKER = re.compile(
    r"SPDX-FileCopyrightText: (?:Modifications )?Copyright \(c\) "
    r"\d{4}(?:-\d{4})? NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)


# Unmodified MIT distribution and license; provenance is in THIRD_PARTY_NOTICES.md.
# Pin bytes instead of falsely applying the repository's NVIDIA/Apache header.
VENDORED_FILES = {
    "pages/fpm-accuracy/assets/plotly.min.js": "28498fa2ea4ba45c8633218088eb223436ca0ca02fc57027fd6fa841ad1901f9",
    "pages/fpm-accuracy/assets/PLOTLY-LICENSE.txt": "764f5e789cad7b339fa1dce1b88f8e316bc39433d265a5754738f31888ad8b2b",
}


def invalid_vendored_files(root: Path) -> list[str]:
    return [
        name
        for name, digest in VENDORED_FILES.items()
        if not (root / name).is_file() or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    ]


def is_source(path: Path) -> bool:
    """Return whether the tracked path is subject to the header policy."""

    if path.suffix == ".patch":
        # Third-party patch files are byte-addressed provenance inputs. Their
        # attribution belongs in THIRD_PARTY_NOTICES.md; changing the patch
        # preamble invalidates recorded collection and overlay hashes.
        return False
    if path.suffix in SOURCE_SUFFIXES or path.name.startswith("Dockerfile"):
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return False


def has_required_license_identifier(head: str) -> bool:
    """Return whether the header contains one complete Apache-2.0 SPDX tag."""

    return LICENSE_MARKER.search(head) is not None


def main() -> int:
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
        .stdout.decode(errors="surrogateescape")
        .split("\0")
    )
    invalid = invalid_vendored_files(Path.cwd())
    if invalid:
        print("Missing or modified pinned third-party files: " + ", ".join(invalid))
        return 1
    sources = [Path(name) for name in tracked if name and name not in VENDORED_FILES and is_source(Path(name))]

    missing: list[str] = []
    for path in sources:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = "".join(handle.readline() for _ in range(HEADER_LINES))
        if not has_required_license_identifier(head) or COPYRIGHT_MARKER.search(head) is None:
            missing.append(str(path))

    if missing:
        print(f"{len(missing)} file(s) missing the required NVIDIA/Apache-2.0 SPDX header:")
        for name in missing:
            print(f"  {name}")
        return 1

    print(f"all {len(sources)} tracked source files carry the required NVIDIA/Apache-2.0 SPDX header")
    return 0


if __name__ == "__main__":
    sys.exit(main())
