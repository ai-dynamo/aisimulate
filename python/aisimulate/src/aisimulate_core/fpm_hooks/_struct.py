# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helper: extend a producer's ``ScheduledRequestMetrics`` msgspec struct with the two lists."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

REQUEST_FIELDS = ("extend_lengths", "past_kv_lengths")


def has_request_fields(struct_type: type) -> bool:
    fields = getattr(struct_type, "__struct_fields__", ())
    return all(name in fields for name in REQUEST_FIELDS)


def extend_struct(base: type) -> type:
    """Return a frozen subclass of ``base`` with ``extend_lengths`` / ``past_kv_lengths`` list fields."""
    import msgspec

    namespace = {
        "__annotations__": {"extend_lengths": list[int], "past_kv_lengths": list[int]},
        "extend_lengths": msgspec.field(default_factory=list),
        "past_kv_lengths": msgspec.field(default_factory=list),
        "__module__": base.__module__,
        "__doc__": (base.__doc__ or "") + "\n\nExtended with per-request extend_lengths / past_kv_lengths.",
    }
    return type(base.__name__, (base,), namespace, frozen=True, gc=False)


def with_pairs(extended: type, base_instance: Any, pairs: Sequence[tuple[int, int]]) -> Any:
    """Copy ``base_instance`` into ``extended`` adding the (extend, past) pairs."""
    import msgspec

    values = msgspec.structs.asdict(base_instance)
    values.pop("extend_lengths", None)
    values.pop("past_kv_lengths", None)
    return extended(
        **values,
        extend_lengths=[int(e) for e, _ in pairs],
        past_kv_lengths=[int(p) for _, p in pairs],
    )
