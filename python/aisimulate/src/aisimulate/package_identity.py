# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report the installed AISim identity: command name, version, and legacy overlap.

AISim 0.14 renamed the ``aisimulate`` distribution to ``aisim`` without renaming
imports, so both distributions own the same ``aisimulate/`` files. Installing
``aisim`` does not remove ``aisimulate``, and uninstalling ``aisimulate``
afterwards deletes files that ``aisim`` needs.

``scripts/release/package_identity.py`` raises the same diagnosis for release
tooling. It stays standalone because it also inspects historical releases that
lack this module; keep the two lead sentences identical.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys

DISTRIBUTION = "aisim"
LEGACY_DISTRIBUTION = "aisimulate"
COMMANDS = ("aisim", "aisimulate")
# Carries the invoked command name into supervised worker processes, whose argv[0]
# is the private worker module rather than the console script.
PROGRAM_ENVIRONMENT = "_AISIMULATE_PROG"
_DEFAULT_PROGRAM = "aisimulate"
_SHARED_PACKAGES = ("aisimulate/", "aisimulate_core/")


def program_name(argv0: str | None = None) -> str:
    """Name the command the user invoked, ``aisim`` or ``aisimulate``.

    ``python -m aisimulate`` and other launchers fall back to ``aisimulate``.
    """

    raw = sys.argv[0] if argv0 is None else argv0
    name = os.path.basename(raw or "")
    for suffix in ("-script.py", ".exe"):
        name = name.removesuffix(suffix)
    if name in COMMANDS:
        return name
    inherited = os.environ.get(PROGRAM_ENVIRONMENT)
    return inherited if inherited in COMMANDS else _DEFAULT_PROGRAM


def _distribution(name: str) -> importlib.metadata.Distribution | None:
    try:
        return importlib.metadata.distribution(name)
    except Exception:
        # Missing or unreadable metadata (source trees, broken dist-info) is not an overlap.
        return None


def _owns_shared_packages(distribution: importlib.metadata.Distribution) -> bool:
    """Return whether the legacy distribution installs files under the shared imports.

    A future dependency-only ``aisimulate`` shim would own none, so it must not warn.
    Without a RECORD, assume the historical layout, which always owned them.
    """

    record = distribution.read_text("RECORD")
    if record is None:
        return True
    return any(line.startswith(_SHARED_PACKAGES) for line in record.splitlines())


def overlap_message() -> str | None:
    """Return repair guidance when ``aisim`` and the legacy ``aisimulate`` both own the imports.

    Returns ``None`` on any metadata error: the check must never block the CLI.
    """

    try:
        current = _distribution(DISTRIBUTION)
        if current is None:
            return None
        legacy = _distribution(LEGACY_DISTRIBUTION)
        if legacy is None or not _owns_shared_packages(legacy):
            return None
        version = current.version
        legacy_version = legacy.version
    except Exception:
        return None
    pin = f"aisim=={version}"
    return (
        "aisim and aisimulate are both installed and own overlapping files.\n"
        f"  aisimulate {legacy_version} is the pre-0.14 name of aisim {version}; installing aisim did not remove it.\n"
        "  Do not run `pip uninstall aisimulate` on its own: it deletes files aisim needs, and the\n"
        "  aisim command then fails with ModuleNotFoundError. Repair this environment with:\n"
        f"    pip uninstall -y aisimulate && pip install --force-reinstall --no-deps {pin}\n"
        "  or, with uv:\n"
        f"    uv pip uninstall aisimulate && uv pip install --reinstall --no-deps {pin}\n"
        f"  If aisim came from a local wheel or checkout, reinstall that path instead of {pin}.\n"
        "  Update dependencies that still require aisimulate before migrating.\n"
    )


def warn_on_overlap(stream=None) -> bool:
    """Print the overlap warning once to stderr; return whether it was printed."""

    message = overlap_message()
    if message is None:
        return False
    target = sys.stderr if stream is None else stream
    target.write(f"{program_name()}: warning: {message}")
    target.flush()
    return True
