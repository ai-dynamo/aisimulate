#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block release staging while source-controlled downstream migrations remain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATES = Path(__file__).resolve().parents[1] / ".github" / "release-gates.json"


def require_completed_migrations(path: Path = GATES) -> None:
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or set(document) != {"pending_migrations"}:
        raise ValueError("release gates must declare pending_migrations")
    pending = document["pending_migrations"]
    if not isinstance(pending, list):
        raise ValueError("pending_migrations must be a list")
    for item in pending:
        if (
            not isinstance(item, dict)
            or set(item) != {"pull_request", "requirement"}
            or any(not isinstance(value, str) or not value.strip() for value in item.values())
        ):
            raise ValueError("each release gate needs a pull_request and requirement")
    if pending:
        details = "\n".join(f"- {item['pull_request']}: {item['requirement']}" for item in pending)
        raise RuntimeError(f"Release publication is blocked by pending downstream migrations:\n{details}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-gates", type=Path, help="Also require the publication target's migrations to be complete"
    )
    args = parser.parse_args(argv)
    try:
        require_completed_migrations(GATES)
        if args.target_gates is not None:
            require_completed_migrations(args.target_gates)
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
