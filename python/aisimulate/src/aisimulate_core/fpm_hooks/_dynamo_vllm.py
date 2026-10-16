# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo vLLM hook: per-request (extend, past) lists on the FPM the InstrumentedScheduler emits.

Applied to ``dynamo.vllm.instrumented_scheduler`` after import. Values come
from the ``SchedulerOutput`` the scheduler just produced: ``num_scheduled_tokens``
per request is the extend, ``num_computed_tokens`` the past KV. Emitted sorted by
past descending, then extend descending; consumers do not depend on the order.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

from ._struct import extend_struct, has_request_fields, with_pairs

logger = logging.getLogger("aisimulate.fpm_hooks")

_PATCHED = "_aisimulate_fpm_request_lists"


def request_pairs(scheduler, output) -> list[tuple[int, int]] | None:
    new_reqs = getattr(output, "scheduled_new_reqs", None)
    cached = getattr(output, "scheduled_cached_reqs", None)
    num_scheduled = getattr(output, "num_scheduled_tokens", None)
    if new_reqs is None or cached is None or num_scheduled is None:
        return None
    counts_as_decode = getattr(scheduler, "_bench_new_request_counts_as_decode", None)
    pairs: list[tuple[int, int]] = []
    for req in new_reqs:
        past = int(getattr(req, "num_computed_tokens", 0))
        if counts_as_decode is not None and counts_as_decode(req.req_id):
            pairs.append((int(num_scheduled.get(req.req_id, 1)), past))
        else:
            pairs.append((int(num_scheduled.get(req.req_id, 0)), past))
    computed = getattr(cached, "num_computed_tokens", [])
    for i, req_id in enumerate(getattr(cached, "req_ids", [])):
        pairs.append((int(num_scheduled.get(req_id, 0)), int(computed[i])))
    pairs.sort(key=lambda t: (-t[1], -t[0]))
    return pairs


def patch_dynamo_vllm_instrumented_scheduler(module: ModuleType) -> bool:
    """Install the hook on an imported ``dynamo.vllm.instrumented_scheduler`` module."""
    scheduler_cls = getattr(module, "InstrumentedScheduler", None)
    extract = getattr(scheduler_cls, "_extract_scheduled", None)
    if scheduler_cls is None or extract is None:
        logger.warning("fpm_hooks: dynamo.vllm.instrumented_scheduler has no _extract_scheduled; skipping")
        return False
    if getattr(extract, _PATCHED, False):
        return False
    fpm_mod = importlib.import_module("dynamo.common.forward_pass_metrics")
    base = fpm_mod.ScheduledRequestMetrics
    if has_request_fields(base):
        logger.info("fpm_hooks: Dynamo already emits per-request FPM fields; nothing to do")
        return False
    extended = extend_struct(base)
    fpm_mod.ScheduledRequestMetrics = extended
    module.ScheduledRequestMetrics = extended  # imported at module level there

    def wrapped(self, output):
        metrics = extract(self, output)
        pairs = request_pairs(self, output)
        if pairs is None:
            return metrics
        return with_pairs(extended, metrics, pairs)

    setattr(wrapped, _PATCHED, True)
    wrapped.__wrapped__ = extract  # type: ignore[attr-defined]
    scheduler_cls._extract_scheduled = wrapped
    logger.info("fpm_hooks: Dynamo vLLM FPM now carries extend_lengths / past_kv_lengths")
    return True
