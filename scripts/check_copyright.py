#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if any tracked source file is missing its SPDX license header.

Checks every non-empty git-tracked *.py, *.rs, and *.sh file for an
`SPDX-License-Identifier` line within its first 15 lines. The whole tree is
checked (not just a diff): the repository is fully compliant today, so any
regression is introduced by the change under review.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATTERNS = ["*.py", "*.rs", "*.sh"]
HEADER_LINES = 15
MARKER = "SPDX-License-Identifier"


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "--", *PATTERNS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    missing = []
    for name in tracked:
        path = Path(name)
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = "".join(handle.readline() for _ in range(HEADER_LINES))
        if MARKER not in head:
            missing.append(name)

    if missing:
        print(f"{len(missing)} file(s) missing an {MARKER} header:")
        for name in missing:
            print(f"  {name}")
        return 1
    print(f"all {len(tracked)} tracked source files carry an {MARKER} header")
    return 0


if __name__ == "__main__":
    sys.exit(main())
