# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted migration warnings for legacy AIConfigurator entry points."""

from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

_MIGRATION_GUIDE = "https://github.com/ai-dynamo/aisimulate/blob/main/docs/cli/migrate-from-aiconfigurator.md"
_warned_entry_points: set[str] = set()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def warn_legacy_cli(mode: str | None) -> None:
    """Warn once when the legacy compatibility CLI is run."""

    command = "aiconfigurator cli" + (f" {mode}" if mode else "")
    key = f"cli:{mode or '<root>'}"
    if key in _warned_entry_points:
        return
    _warned_entry_points.add(key)
    message = (
        f"`{command}` is a deprecated compatibility command shipped by AISimulate 0.13.0. "
        "Command removal is targeted for AISimulate 0.14.0 after every remaining "
        "workflow has a verified unified-CLI replacement. "
        f"Until then, keep running `{command} ...`; AISimulate preserves the "
        "established command name and arguments. "
        f"Migration guide: {_MIGRATION_GUIDE}"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=4)
    logger.warning("%s", message)


def deprecated_sweeper_entry_point(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Mark one legacy sweep function with a warn-once migration message."""

    entry_point = f"{function.__module__}.{function.__name__}"

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if entry_point not in _warned_entry_points:
            _warned_entry_points.add(entry_point)
            warnings.warn(
                f"`{entry_point}()` is deprecated. Migrate to "
                "`aisimulate.sweeper.Sweeper(...).run(config)`. "
                "AISimulate 0.13.0 temporarily retains the legacy "
                f"`aiconfigurator` import namespace. Migration guide: {_MIGRATION_GUIDE}",
                DeprecationWarning,
                stacklevel=2,
            )
        return function(*args, **kwargs)

    return wrapper
