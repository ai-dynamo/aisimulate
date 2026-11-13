# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang hook: per-request lists and step diagnostics on the FPM the scheduler emits.

Applied to ``sglang.srt.managers.scheduler_components.metrics_reporter`` after
import. Values come from the schedule-time ``ScheduleBatch.extend_lens`` /
``prefix_lens`` (aligned with ``batch.reqs``); the per-request attributes
(``req.extend_input_len``) are already reset when metrics are emitted, so they
are only a fallback. Decode batches contribute ``(1, seq_len)`` per request
from ``batch.seq_lens_cpu``, the same source as the aggregate. Under
speculative decoding a decode request's extend is the verify width
``spec_num_draft_tokens`` (``1 + k``, the convention of the learned-model
docs), and the tokens it committed are in ``spec_accept_lengths``.

Step diagnostics (see ``_struct.STEP_FIELDS``) come from three more wrappers:

- ``_init_fpm`` adds a DeviceTimer reporter that sums device time per
  category (``target_verify``, ``eagle_draft``, ``eagle_draft_extend``, ...);
  the record drains it, so the categories cover the intervals of ``wall_time``.
- ``_emit_forward_pass_metrics`` hands the batch result and the host emission
  time to the builder; ``accept_lens`` is read only when it is already on the
  CPU, so the hook never adds a device synchronization.
- The schedule-time host timestamp is ``batch.fpm_start_time``.

The stock emitter sums every forward interval that finished since the last
record, so under the overlap scheduler a record can include the next step
(see the learned-model docs). The host timestamps make such records visible.
"""

from __future__ import annotations

import importlib
import logging
import time
from types import ModuleType
from typing import Any

from ._struct import extend_struct, has_request_fields, with_pairs

logger = logging.getLogger("aisim.fpm_hooks")

_PATCHED = "_aisim_fpm_request_lists"
_STEP_CONTEXT = "_aisim_fpm_step_context"
_DEVICE_TIME = "_aisim_fpm_device_time_by_category"
_TIMER_HOOKED = "_aisim_fpm_category_reporter"


def request_pairs(batch) -> list[tuple[int, int]] | None:
    """(extend, past) per scheduled request, or ``None`` when the batch shape is not understood."""
    mode = getattr(batch, "forward_mode", None)
    reqs = list(getattr(batch, "reqs", None) or [])
    if mode is None or not reqs:
        return None
    if mode.is_decode():
        # Same source as the stock aggregate (sum_decode_kv_tokens): the
        # schedule-time sequence lengths. req.seqlen is already +1 when the
        # metrics are emitted after the step.
        seq_lens = getattr(batch, "seq_lens_cpu", None)
        if seq_lens is not None and len(seq_lens) == len(reqs):
            return [(1, int(sl)) for sl in seq_lens]
        return [(1, max(int(getattr(req, "seqlen", 1)) - 1, 0)) for req in reqs]
    if not (mode.is_extend() or mode.is_mixed()):
        return None
    ext = getattr(batch, "extend_lens", None)
    pre = getattr(batch, "prefix_lens", None)
    if ext is not None and pre is not None and len(ext) == len(reqs) == len(pre):
        return [(int(e), int(p)) for e, p in zip(ext, pre, strict=True)]
    # Fallback: per-request attributes (valid when read before the step resets them).
    decoding = {id(req) for req in (getattr(batch, "decoding_reqs", None) or [])}
    pairs: list[tuple[int, int]] = []
    for req in reqs:
        if id(req) in decoding:
            pairs.append((1, int(getattr(req, "seqlen", 0))))
            continue
        prefix = getattr(req, "prefix_indices", None)
        past = len(prefix) if prefix is not None else 0
        pairs.append((int(getattr(req, "extend_input_len", 0) or 0), int(past)))
    return pairs


def accept_lengths(result, num_requests: int) -> list[int] | None:
    """Tokens committed per request by a speculative step (bonus included), or ``None``."""
    if result is None:
        return None
    lens = getattr(result, "accept_lens", None)
    if lens is not None:
        if not getattr(lens, "is_cpu", True):
            return None  # reading it would synchronize with the device
        values = lens.tolist() if hasattr(lens, "tolist") else list(lens)
    else:
        correct = getattr(result, "num_correct_drafts_per_req_cpu", None)
        if correct is None:
            return None
        bonus = int(getattr(result, "num_non_draft_tokens_per_req", 1) or 1)
        values = [int(c) + bonus for c in correct]
    if len(values) != num_requests:
        return None
    return [int(v) for v in values]


def attach_device_time_reporter(reporter) -> bool:
    """Add a per-category DeviceTimer reporter to an SGLang metrics reporter (idempotent)."""
    timer = getattr(reporter, "forward_pass_device_timer", None)
    add = getattr(timer, "add_reporter", None)
    if add is None or getattr(timer, _TIMER_HOOKED, False):
        return False
    totals: dict[str, float] = {}

    def by_category(t, category="unknown", **_metadata):
        totals[category] = totals.get(category, 0.0) + float(t)

    add(by_category)
    setattr(timer, _TIMER_HOOKED, True)
    setattr(reporter, _DEVICE_TIME, totals)
    return True


def step_fields(reporter, batch, num_requests: int) -> dict[str, Any]:
    """Step diagnostics for the record being emitted; empty outside an emission."""
    context = getattr(reporter, _STEP_CONTEXT, None)
    if context is None:
        return {}
    result, emitted_at = context
    step: dict[str, Any] = {"host_emitted_at_s": float(emitted_at)}
    scheduled_at = getattr(batch, "fpm_start_time", None)
    if scheduled_at:
        step["host_scheduled_at_s"] = float(scheduled_at)
    totals = getattr(reporter, _DEVICE_TIME, None)
    if totals:
        step["device_time_s_by_category"] = dict(totals)
        totals.clear()
    snapshot = getattr(reporter, "_active_spec_config_snapshot", None)
    if snapshot is not None:
        spec = snapshot()
        step["spec_num_steps"] = int(spec.get("num_steps", 0) or 0)
        step["spec_num_draft_tokens"] = int(spec.get("num_draft_tokens", 0) or 0)
    accepted = accept_lengths(result, num_requests)
    if accepted is not None:
        step["spec_accept_lengths"] = accepted
    return step


def speculative_pairs(batch, pairs: list[tuple[int, int]], step: dict[str, Any]) -> list[tuple[int, int]]:
    """Decode pairs of a speculative step carry the verify width (``1 + k``) as the extend.

    A decode batch with speculation active runs one target verify over
    ``spec_num_draft_tokens`` tokens per request; prefill (extend) batches are unchanged.
    """
    width = int(step.get("spec_num_draft_tokens", 0) or 0)
    mode = getattr(batch, "forward_mode", None)
    if width <= 1 or mode is None or not mode.is_decode():
        return pairs
    return [(width, past) for _, past in pairs]


def _owner_class(module: ModuleType) -> type | None:
    # The owner class is SchedulerMetricsReporter in current SGLang; find it by
    # method name so a rename does not silently disable the hook.
    return next(
        (
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and "_build_scheduled_request_metrics" in vars(obj)
        ),
        None,
    )


def _wrap_init_fpm(owner: type) -> None:
    init_fpm = vars(owner).get("_init_fpm")
    if init_fpm is None:
        logger.info("fpm_hooks: SGLang metrics reporter has no _init_fpm; device time per category is lazy")
        return

    def wrapped_init_fpm(self, *args, **kwargs):
        out = init_fpm(self, *args, **kwargs)
        attach_device_time_reporter(self)
        return out

    wrapped_init_fpm.__wrapped__ = init_fpm  # type: ignore[attr-defined]
    owner._init_fpm = wrapped_init_fpm


def _wrap_emit(owner: type) -> None:
    emit = vars(owner).get("_emit_forward_pass_metrics")
    if emit is None:
        logger.warning("fpm_hooks: SGLang metrics reporter has no _emit_forward_pass_metrics; no step diagnostics")
        return

    def wrapped_emit(self, batch, *args, **kwargs):
        # Fallback when _init_fpm ran before the hook: the first record may
        # then miss the device time reported before this point.
        attach_device_time_reporter(self)
        result = args[0] if args else kwargs.get("result")
        setattr(self, _STEP_CONTEXT, (result, time.monotonic()))
        try:
            return emit(self, batch, *args, **kwargs)
        finally:
            setattr(self, _STEP_CONTEXT, None)

    wrapped_emit.__wrapped__ = emit  # type: ignore[attr-defined]
    owner._emit_forward_pass_metrics = wrapped_emit


def patch_sglang_metrics_reporter(module: ModuleType) -> bool:
    """Install the hook on an imported ``metrics_reporter`` module. Returns ``True`` when applied."""
    owner = _owner_class(module)
    build = getattr(owner, "_build_scheduled_request_metrics", None)
    if owner is None or build is None:
        logger.warning(
            "fpm_hooks: no class in SGLang metrics_reporter defines _build_scheduled_request_metrics; skipping"
        )
        return False
    if getattr(build, _PATCHED, False):
        return False
    fpm_mod = importlib.import_module("sglang.srt.observability.forward_pass_metrics")
    base = fpm_mod.ScheduledRequestMetrics
    if has_request_fields(base):
        logger.info("fpm_hooks: SGLang already emits per-request FPM fields; nothing to do")
        return False
    extended = extend_struct(base)
    # metrics_reporter imports the class name inside the builder at call time,
    # so rebinding the module attribute is enough for the encoder to see it.
    fpm_mod.ScheduledRequestMetrics = extended
    diagnostics_failed = False

    def wrapped(self, batch):
        nonlocal diagnostics_failed
        metrics = build(self, batch)
        pairs = request_pairs(batch)
        try:
            # Drain per-category device time even for records without lists.
            step = step_fields(self, batch, len(pairs) if pairs is not None else 0)
        except Exception:
            # Diagnostics must never break the scheduler loop.
            if not diagnostics_failed:
                logger.exception("fpm_hooks: SGLang step diagnostics failed; emitting lists only")
                diagnostics_failed = True
            step = {}
        if pairs is None:
            return metrics
        return with_pairs(extended, metrics, speculative_pairs(batch, pairs, step), step)

    setattr(wrapped, _PATCHED, True)
    wrapped.__wrapped__ = build  # type: ignore[attr-defined]
    owner._build_scheduled_request_metrics = wrapped
    _wrap_init_fpm(owner)
    _wrap_emit(owner)
    logger.info("fpm_hooks: SGLang FPM now carries per-request lists and step diagnostics")
    return True
