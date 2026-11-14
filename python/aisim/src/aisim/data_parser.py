# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parser for ``aisim data``.

Kept free of dataset-library imports so building the CLI parser stays
runtime-free; :mod:`aisim.data_cli` holds the command implementation.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_CATEGORIES = ("op-latency", "fpm", "e2e", "eval")
PURPOSES = ("prediction", "validation", "results", "full-evidence")
# Library entry purposes that hold the data a predictor reads, per category.
# ``full-evidence`` applies no purpose filter. ``validation`` and ``results``
# select evidence tiers that the library cannot filter yet.
_PREDICTION_PURPOSE = {"op-latency": "operation-tables", "fpm": "execution-profile"}


def add_data_parser(subparsers: Any) -> None:
    data = subparsers.add_parser("data", help="Work with the AISim dataset.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch_parser = data_commands.add_parser(
        "fetch",
        help="Download selected dataset files into the local cache.",
        description=(
            "Resolve a dataset revision to one commit SHA, plan the files a selection needs, and download and "
            "verify them into the cache. A selection needs --configuration or --hardware (plus --model for "
            "fpm, e2e and eval)."
        ),
    )
    fetch_parser.add_argument("--category", required=True, help=f"one of: {', '.join(SUPPORTED_CATEGORIES)}")
    fetch_parser.add_argument("--model")
    fetch_parser.add_argument("--hardware")
    fetch_parser.add_argument("--framework")
    fetch_parser.add_argument("--framework-version")
    fetch_parser.add_argument(
        "--configuration",
        action="append",
        default=[],
        metavar="ID",
        help="configuration ID; repeat or comma-separate for several",
    )
    fetch_parser.add_argument("--operation-family")
    fetch_parser.add_argument("--purpose", choices=PURPOSES, default="prediction")
    fetch_parser.add_argument("--repo", required=True, help="Hugging Face dataset repository, owner/name")
    fetch_parser.add_argument("--revision", default="main", help="branch, tag or commit SHA (default: main)")
    fetch_parser.add_argument("--cache-dir", help="cache root (default: $AISIM_DATA_CACHE or the user cache)")
    fetch_parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; downloads only the small catalog and manifest files"
    )
    fetch_parser.add_argument("--offline", action="store_true", help="make no network requests")
