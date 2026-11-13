# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command identity shared by the canonical CLI and its compatibility alias."""

from __future__ import annotations

import os
import sys

DISTRIBUTION = "aisim"
COMMANDS = ("aisim", "aisimulate")
# Console scripts have a recognizable argv[0]; private worker modules do not.
PROGRAM_ENVIRONMENT = "_AISIM_PROG"


def program_name(argv0: str | None = None) -> str:
    """Preserve console/module invocation names across supervised workers."""
    raw = sys.argv[0] if argv0 is None else argv0
    path = (raw or "").replace("\\", "/").split("/")
    name = path[-1]
    for suffix in ("-script.py", ".exe"):
        name = name.removesuffix(suffix)
    if name in COMMANDS:
        return name
    # runpy resolves the legacy package to canonical source paths. Its module
    # spec retains the requested spelling even when argv[0] does not.
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    module_name = getattr(spec, "name", "") or ""
    prefix, _, suffix = module_name.partition(".")
    if prefix in COMMANDS and suffix in {"__main__", "main", "supervision"}:
        return prefix
    inherited = os.environ.get(PROGRAM_ENVIRONMENT)
    if inherited in COMMANDS:
        return inherited
    if name == "__main__.py" and len(path) > 1 and path[-2] in COMMANDS:
        return path[-2]
    return DISTRIBUTION
