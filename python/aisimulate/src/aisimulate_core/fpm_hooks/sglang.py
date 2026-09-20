# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang hook: per-request (extend, past) lists on the FPM the scheduler emits.

Applied to ``sglang.srt.managers.scheduler_components.metrics_reporter`` after
import. Values come from the schedule-time ``ScheduleBatch.extend_lens`` /
``prefix_lens`` (aligned with ``batch.reqs``); the per-request attributes
(``req.extend_input_len``) are already reset when metrics are emitted, so they
are only a fallback. Decode batches contribute ``(1, seqlen)`` per request.

Collect with ``--disable-overlap-schedule``: the stock FPM emitter accumulates
every finished forward interval, which merges the next step into the current
record under the overlap scheduler (see the learned-model docs).
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

from ._struct import extend_struct, has_request_fields, with_pairs

logger = logging.getLogger("aisimulate.fpm_hooks")

_PATCHED = "_aisimulate_fpm_request_lists"


def request_pairs(batch) -> list[tuple[int, int]] | None:
    """(extend, past) per scheduled request, or ``None`` when the batch shape is not understood."""
    mode = getattr(batch, "forward_mode", None)
    reqs = list(getattr(batch, "reqs", None) or [])
    if mode is None or not reqs:
        return None
    if mode.is_decode():
        return [(1, int(getattr(req, "seqlen", 0))) for req in reqs]
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


def patch_sglang_metrics_reporter(module: ModuleType) -> bool:
    """Install the hook on an imported ``metrics_reporter`` module. Returns ``True`` when applied."""
    # The owner class is SchedulerMetricsReporter in current SGLang; find it by
    # method name so a rename does not silently disable the hook.
    mixin = next(
        (
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and "_build_scheduled_request_metrics" in vars(obj)
        ),
        None,
    )
    build = getattr(mixin, "_build_scheduled_request_metrics", None)
    if mixin is None or build is None:
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

    def wrapped(self, batch):
        metrics = build(self, batch)
        pairs = request_pairs(batch)
        if pairs is None:
            return metrics
        return with_pairs(extended, metrics, pairs)

    setattr(wrapped, _PATCHED, True)
    wrapped.__wrapped__ = build  # type: ignore[attr-defined]
    mixin._build_scheduled_request_metrics = wrapped
    logger.info("fpm_hooks: SGLang FPM now carries extend_lengths / past_kv_lengths")
    return True
