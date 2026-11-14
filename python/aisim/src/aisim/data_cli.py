# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim data fetch``: a thin command line over ``aisim_core.data.fetch``.

Selection, revision pinning, caching and verification all live in the library;
this module only maps flags to library calls and library results to JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aisim_core.data.fetch import (
    Cache,
    DataFetchError,
    HfTransport,
    RevisionError,
    Selection,
    SelectionError,
    Transport,
    fetch,
    plan,
)

from .data_parser import _PREDICTION_PURPOSE, PURPOSES, SUPPORTED_CATEGORIES, add_data_parser  # noqa: F401

EXIT_FAILURE = 1
EXIT_USAGE = 2


class UnsupportedSelectionError(SelectionError):
    code = "unsupported"


def make_transport() -> Transport:
    """Transport factory; tests patch this to inject a fake."""
    return HfTransport()


def build_selection(args: argparse.Namespace) -> Selection:
    if args.category not in SUPPORTED_CATEGORIES:
        raise UnsupportedSelectionError(
            f"category {args.category!r} is not supported by 'data fetch' yet; "
            f"supported categories: {', '.join(SUPPORTED_CATEGORIES)}"
        )
    if args.purpose in ("validation", "results"):
        raise UnsupportedSelectionError(
            f"purpose {args.purpose!r} is not supported by 'data fetch' yet; use 'prediction' or 'full-evidence'"
        )
    purpose = None
    if args.purpose == "prediction":
        purpose = _PREDICTION_PURPOSE.get(args.category)
        if purpose is None:
            raise UnsupportedSelectionError(
                f"purpose 'prediction' has no data for category {args.category!r}; use 'full-evidence'"
            )
    configurations = tuple(c.strip() for raw in args.configuration for c in raw.split(",") if c.strip())
    selection = Selection(
        category=args.category,
        configuration_ids=configurations,
        model=args.model,
        hardware=args.hardware,
        framework=args.framework,
        framework_version=args.framework_version,
        operation_family=args.operation_family,
        purpose=purpose,
    )
    selection.validate()
    return selection


def run_data_command(args: argparse.Namespace) -> int:
    try:
        selection = build_selection(args)
        cache = Cache(Path(args.cache_dir) if args.cache_dir else None)
        transport = None if args.offline else make_transport()
        common = {
            "repo": args.repo,
            "revision": args.revision,
            "transport": transport,
            "cache": cache,
            "offline": args.offline,
        }
        if args.dry_run:
            payload: dict[str, Any] = {"dry_run": True, **plan(selection, **common).to_dict()}
        else:
            payload = {"dry_run": False, **fetch(selection, **common).to_dict()}
    except SelectionError as exc:  # includes UnsupportedSelectionError
        return _fail(exc, EXIT_USAGE)
    except DataFetchError as exc:
        return _fail(exc, EXIT_FAILURE)
    except Exception as exc:  # transport errors (network, auth, unknown repo) are not library errors
        return _fail(RevisionError(f"{type(exc).__name__}: {exc}"), EXIT_FAILURE)
    sys.stdout.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return 0


def _fail(exc: DataFetchError, status: int) -> int:
    sys.stderr.write(f"aisim data fetch: {exc}\n")
    sys.stderr.write(json.dumps(exc.to_dict(), allow_nan=False) + "\n")
    return status
