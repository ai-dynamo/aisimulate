# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, scoped selection of additive model-owned history points."""

from contextlib import contextmanager
from contextvars import ContextVar

_LONG_HISTORY = ContextVar("collector_include_long_history", default=False)


@contextmanager
def long_history_coverage(enabled: bool):
    """Select coverage while planning; generated worker rows remain explicit."""
    token = _LONG_HISTORY.set(enabled)
    try:
        yield
    finally:
        _LONG_HISTORY.reset(token)


def includes_long_history() -> bool:
    return _LONG_HISTORY.get()


def history_points(profile: dict, phase: str, max_position: int | None) -> list[tuple[int, int, int]]:
    """Read exact (batch, query-or-past length, prefix) tuples, never a product of axes.

    Context uses query + prefix; generation uses past + one new token. Dense
    context declarations use the full lookup sequence and a zero prefix.
    """
    if not includes_long_history():
        return []
    declared = profile.get("long_history_cases", {})
    if not isinstance(declared, dict) or set(declared) - {"context", "generation"}:
        raise ValueError("long_history_cases must map context/generation to lists of [batch, length, prefix]")
    points = declared.get(phase, [])
    if not isinstance(points, list):
        raise ValueError(f"long_history_cases.{phase} must be a list")
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise ValueError(f"long_history_cases.{phase}: expected [batch, length, prefix], got {point!r}")
        if any(type(value) is not int for value in point):
            raise ValueError(f"long_history_cases.{phase}: point values must be integers: {point!r}")
        batch, length, prefix = point
        if batch <= 0 or length <= 0 or prefix < 0 or (phase == "generation" and prefix):
            raise ValueError(f"long_history_cases.{phase}: invalid point {point!r}")
        history = length + prefix if phase == "context" else length + 1
        if max_position is not None and history > max_position:
            raise ValueError(
                f"long_history_cases.{phase}: {profile.get('model_path')} point {point!r} "
                f"requires history {history}, exceeding model limit {max_position}"
            )
        result.append((batch, length, prefix))
    return list(dict.fromkeys(result))
