# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helper: extend a producer's ``ScheduledRequestMetrics`` msgspec struct with hook fields.

Two groups of optional fields are added:

- per-request lists ``extend_lengths`` / ``past_kv_lengths`` (all hooks);
- step diagnostics (filled by the SGLang hook; other hooks leave the
  defaults): speculative configuration and per-request accepted tokens, device
  time per timer category, and host-clock timestamps. Defaults are empty or
  zero, so a record without the source data keeps its meaning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REQUEST_FIELDS = ("extend_lengths", "past_kv_lengths")

#: name -> (annotation, default value or default factory), in encoding order.
STEP_FIELDS: dict[str, tuple[Any, Any]] = {
    # Active speculative configuration of the worker (0 when speculation is off).
    "spec_num_steps": (int, 0),
    "spec_num_draft_tokens": (int, 0),
    # Tokens committed per request by a speculative decode step, bonus token
    # included; aligned with the request lists. Empty for other steps.
    "spec_accept_lengths": (list[int], list),
    # Device time in seconds per DeviceTimer category (for example ``decode``,
    # ``target_verify``, ``eagle_draft``, ``eagle_draft_extend``). Covers the
    # same intervals as ``wall_time`` when the producer times on the device.
    "device_time_s_by_category": (dict[str, float], dict),
    # Host monotonic clock (seconds, per worker process) when the scheduler
    # started building this batch, and when this record was emitted.
    "host_scheduled_at_s": (float, 0.0),
    "host_emitted_at_s": (float, 0.0),
}


def has_request_fields(struct_type: type) -> bool:
    fields = getattr(struct_type, "__struct_fields__", ())
    return all(name in fields for name in REQUEST_FIELDS)


def extend_struct(base: type) -> type:
    """Return a frozen subclass of ``base`` with the request lists and the step diagnostics."""
    import msgspec

    annotations: dict[str, Any] = {"extend_lengths": list[int], "past_kv_lengths": list[int]}
    namespace: dict[str, Any] = {
        "extend_lengths": msgspec.field(default_factory=list),
        "past_kv_lengths": msgspec.field(default_factory=list),
    }
    for name, (annotation, default) in STEP_FIELDS.items():
        annotations[name] = annotation
        namespace[name] = msgspec.field(default_factory=default) if callable(default) else default
    namespace["__annotations__"] = annotations
    namespace["__module__"] = base.__module__
    namespace["__doc__"] = (
        base.__doc__ or ""
    ) + "\n\nExtended with per-request extend_lengths / past_kv_lengths and step diagnostics."
    return type(base.__name__, (base,), namespace, frozen=True, gc=False)


def with_pairs(
    extended: type,
    base_instance: Any,
    pairs: Sequence[tuple[int, int]],
    step: Mapping[str, Any] | None = None,
) -> Any:
    """Copy ``base_instance`` into ``extended`` adding the (extend, past) pairs and step fields."""
    import msgspec

    values = msgspec.structs.asdict(base_instance)
    for name in (*REQUEST_FIELDS, *STEP_FIELDS):
        values.pop(name, None)
    extra = {name: value for name, value in (step or {}).items() if name in STEP_FIELDS}
    return extended(
        **values,
        **extra,
        extend_lengths=[int(e) for e, _ in pairs],
        past_kv_lengths=[int(p) for _, p in pairs],
    )
