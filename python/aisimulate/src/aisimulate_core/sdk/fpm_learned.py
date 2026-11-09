# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Feature definitions adapted from (re-implemented; no upstream source copied):
# - SGLang simulator `MLTimePredictor.FEATURE_NAMES` (18 batch features),
#   revision 59799a368793b9f795baf59b067233a65ad8e38e,
#   path tools/sglang-simulator/src/sglang_simulator/time_predictor/ml.py,
#   https://github.com/sgl-project/sglang/blob/59799a368793b9f795baf59b067233a65ad8e38e/tools/sglang-simulator/src/sglang_simulator/time_predictor/ml.py,
#   copyright the SGLang project contributors, Apache-2.0.
# - HiSim (Alibaba tair-kvcache) XGBoost predictor (32 present/past/extend slots),
#   revision 85402f1a012fb4279be5bb4b57613bb0a067d361,
#   path hisim/src/hisim/time_predictor/aiconfigurator.py,
#   https://github.com/alibaba/tair-kvcache/blob/85402f1a012fb4279be5bb4b57613bb0a067d361/hisim/src/hisim/time_predictor/aiconfigurator.py,
#   copyright Alibaba Group, Apache-2.0.
# Modified: names, ordering and formulas adapted to AISimulate FPM fields.
# See THIRD_PARTY_NOTICES.md.

"""Train a learned forward-pass model from real-deployment FPM telemetry.

Workflow (Python trains, Rust predicts):

1. Run the real deployment with Dynamo's FPM trace enabled
   (``DYN_FPM_TRACE=1 DYN_FPM_MODE=full``) or capture the FPM event stream
   with a subscriber sink. Every scheduler iteration yields one
   ``ForwardPassMetrics`` record per attention-DP rank with the scheduled
   batch composition and the observed ``wall_time``.
2. ``python -m aisimulate_core.sdk.fpm_learned train --fpm <files> \\
   --worker-type decode --out model.json`` groups the records into
   iterations, derives the named iteration features (the same formulas as
   ``crates/core/src/perfmodel/fpm/learned.rs``), fits one gradient-boosted
   tree ensemble per regression workload store on ``log(wall_ms)``, and
   writes the ``aic_fpm_learned_forward_perf`` artifact.
3. ``RustForwardPassPerfModel.best_available(config)`` with explicit
   ``estimation_mode="fpm_learned"`` and
   ``estimator_config.fpm_learned.adaptation="frozen"`` loads the artifact.
   Training checkpoints retain exact rows and bins for native online updates
   through ``tune_with_fpms``; online adaptation is the default when omitted.

The feature ABI is the ordered ``features`` list in the artifact; every name
must be one of :data:`FEATURE_NAMES` and is computed identically here and in
Rust. Prediction and evaluation go through the compiled Rust model
(``RustForwardPassPerfModel.from_learned``), the single inference oracle; the
parity tests keep a pure-Python mirror of the tree walk under
``tests/unit/sdk/fpm_reference.py``.

scikit-learn is an optional dependency (``pip install aisimulate[learned]``);
it is imported lazily by :func:`train`.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import logging
import math
import os
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_NAME = "aic_fpm_learned_forward_perf"
SCHEMA_VERSION = 1
FPM_VERSION = 1
TRAINING_CHECKPOINT_SCHEMA = "aisimulate_fpm_training_checkpoint"
TRAINING_CHECKPOINT_VERSION = 1
TRAINING_EXPORTER_VERSION = "1"
# Export reads sklearn's fitted bin mapper and predictor arrays. A new version
# needs explicit round-trip coverage before its private fitted state is accepted.
ONLINE_SUPPORTED_SKLEARN_VERSIONS = frozenset({"1.9.1"})
_TRAINING_METADATA_FIELDS = frozenset(
    {
        "trainer",
        "model",
        "hyperparameters",
        "train_rows",
        "launch_censored",
        "launch_refused",
        "trees_fitted",
        "sklearn_version",
        "exporter_version",
    }
)

WORKER_TYPES = ("prefill", "decode", "aggregated")
WORKLOAD_KINDS = ("pure_decode", "contains_locally_mixed", "cross_rank_aggregated", "pure_prefill")
TARGETS = ("log_ms", "ms")

# Must stay byte-identical (names and order) with the tables in
# ``crates/core/src/perfmodel/fpm/learned.rs``.
AGGREGATE_FEATURE_NAMES: tuple[str, ...] = (
    "num_active_ranks",
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "var_prefill_length",
    "num_decode_requests",
    "sum_decode_kv_tokens",
    "var_decode_kv_tokens",
    "max_rank_prefill_tokens",
    "max_rank_decode_kv_tokens",
    "max_rank_num_decode_requests",
    "mean_prefill_chunk",
    "mean_prefill_kv",
    "prefill_attention_pairs",
    "mean_decode_kv",
    "sum_decode_kv_squared",
    "log1p_sum_prefill_tokens",
    "log1p_sum_prefill_kv_tokens",
    "log1p_sum_decode_kv_tokens",
    "log1p_prefill_attention_pairs",
    "log1p_sum_decode_kv_squared",
)
# The 18 features of the SGLang simulator ``MLTimePredictor`` over the
# per-request ``(extend, past)`` list, prefixed ``req_``; NaN when the
# producer did not emit ``extend_lengths`` / ``past_kv_lengths``.
REQUEST_FEATURE_NAMES: tuple[str, ...] = (
    "req_batch_size",
    "req_sum_extend",
    "req_max_extend",
    "req_min_extend",
    "req_sum_past",
    "req_max_past",
    "req_min_past",
    "req_sum_extend_x_past",
    "req_sum_extend_squared",
    "req_sum_past_squared",
    "req_sum_attn_flops",
    "req_sum_extend_x_max_past",
    "req_log1p_sum_past",
    "req_log1p_sum_attn_flops",
    "req_batch_size_x_sum_extend",
    "req_max_past_minus_min_past",
    "req_is_decode",
    "req_is_prefill",
)
# HiSim-style request slots: requests sorted by past KV descending, slot i ->
# (present, past, extend); mirrors ``_build_xgb_feature_maxbs_2`` padding.
SLOT_COUNT = 32
SLOT_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"slot{i}_{kind}" for i in range(SLOT_COUNT) for kind in ("present", "past", "extend")
)
FEATURE_NAMES: tuple[str, ...] = AGGREGATE_FEATURE_NAMES + REQUEST_FEATURE_NAMES + SLOT_FEATURE_NAMES

# Atomic per-request feature sets: the smallest subsets of the 18 from which all
# 18 can be computed back (docs/perf-model/fpm_ml/design.md §3.1). Prefill: the ten that are
# not functions of the others. Decode: every request extends by one token, so
# Σe = n, max e = min e = 1, Σe² = n and Σe·p = Σp, and five remain.
ATOMIC_PREFILL_FEATURE_NAMES: tuple[str, ...] = (
    "req_batch_size",
    "req_sum_extend",
    "req_max_extend",
    "req_min_extend",
    "req_sum_past",
    "req_max_past",
    "req_min_past",
    "req_sum_extend_x_past",
    "req_sum_extend_squared",
    "req_sum_past_squared",
)
ATOMIC_DECODE_FEATURE_NAMES: tuple[str, ...] = (
    "req_batch_size",
    "req_sum_past",
    "req_max_past",
    "req_min_past",
    "req_sum_past_squared",
)

# Aggregates computable from the five FPM v1 counts and sums alone
# (num_prefill_requests, sum_prefill_tokens, sum_prefill_kv_tokens,
# num_decode_requests, sum_decode_kv_tokens), e.g. FPM Gym data. Drops the two
# variances and the decode sum of squares recovered from var_decode_kv_tokens;
# without a variance those would assume equal request lengths.
FPM5_FEATURE_NAMES: tuple[str, ...] = tuple(
    name
    for name in AGGREGATE_FEATURE_NAMES
    if name
    not in ("var_prefill_length", "var_decode_kv_tokens", "sum_decode_kv_squared", "log1p_sum_decode_kv_squared")
)

# Feature presets: ``v1`` uses aggregates only (works on any FPM stream);
# ``fpm5`` (the default) uses the aggregates that need no variance (counts and sums only);
# ``atomic5`` / ``atomic10`` / ``sglang18`` / ``hisim`` need per-request lists;
# ``all`` is the union. ``sglang18`` is the full SGLang-simulator set the atomic
# sets were derived from; it is kept for comparison and for prefill models that
# must extrapolate to traffic unlike their training data (design.md §8.8).
FEATURE_PRESETS: dict[str, tuple[str, ...]] = {
    "v1": AGGREGATE_FEATURE_NAMES,
    "fpm5": FPM5_FEATURE_NAMES,
    "atomic5": ATOMIC_DECODE_FEATURE_NAMES,
    "atomic10": ATOMIC_PREFILL_FEATURE_NAMES,
    "sglang18": REQUEST_FEATURE_NAMES,
    "hisim": ("req_batch_size",) + SLOT_FEATURE_NAMES,
    "all": FEATURE_NAMES,
}
# Defaults: the 17 ``fpm5`` aggregates for every worker role. They need only the
# five FPM v1 counts and sums, so any FPM stream trains without per-request lists
# or variances. The atomic sets (``atomic5`` for decode, ``atomic10`` for prefill
# and aggregated workers) remain explicit opt-ins when the producer emits
# extend_lengths / past_kv_lengths.
DEFAULT_FEATURES: dict[str, tuple[str, ...]] = {
    "decode": FPM5_FEATURE_NAMES,
    "prefill": FPM5_FEATURE_NAMES,
    "aggregated": FPM5_FEATURE_NAMES,
}
DEFAULT_FEATURE_PRESET: dict[str, str] = {
    "decode": "fpm5",
    "prefill": "fpm5",
    "aggregated": "fpm5",
}

MIN_POSITIVE_PREDICTION_MS = 1e-6


# ---------------------------------------------------------------------------
# FPM record loading and iteration grouping
# ---------------------------------------------------------------------------


def _open_text(path: str | os.PathLike[str]):
    path = os.fspath(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _unwrap_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Accept a flat FPM record, a Dynamo trace envelope, or a sink record.

    The trace sink's ``observed_at_unix_ms`` (envelope level) is copied onto the
    FPM mapping so callers can window records by wall-clock time.
    """
    if "scheduled_requests" in raw:
        return raw
    fpm = None
    event = raw.get("event")
    if isinstance(event, dict):
        candidate = event.get("fpm")
        if isinstance(candidate, dict) and "scheduled_requests" in candidate:
            fpm = candidate
    if fpm is None:
        candidate = raw.get("fpm")
        if isinstance(candidate, dict) and "scheduled_requests" in candidate:
            fpm = candidate
    if fpm is None:
        return None
    # The trace sink keeps the wall-clock stamp on the envelope (``event``);
    # the ZMQ sink keeps it on the top level. Accept both.
    for holder in (raw, event if isinstance(event, dict) else {}):
        for key in ("observed_at_unix_ms", "recv_ms"):
            if key in holder and key not in fpm:
                fpm[key] = holder[key]
    return fpm


def iter_fpm_records(paths: Iterable[str | os.PathLike[str]]) -> Iterator[dict[str, Any]]:
    """Yield FPM v1 records from jsonl / jsonl.gz files, skipping heartbeats."""
    for path in paths:
        with _open_text(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                fpm = _unwrap_record(raw)
                if fpm is None or int(fpm.get("version", FPM_VERSION)) != FPM_VERSION:
                    continue
                fpm.setdefault("worker_id", "")
                fpm.setdefault("dp_rank", 0)
                fpm.setdefault("wall_time", 0.0)
                yield fpm


def group_iterations(records: Iterable[dict[str, Any]], join_ranks: str = "none") -> list[list[dict[str, Any]]]:
    """Group FPM records into iterations (one list of per-rank FPMs each).

    ``join_ranks="none"`` treats every record as a single-rank iteration.
    ``join_ranks="counter"`` keys records by ``(worker_id, counter_id)`` and
    rejects repeated keys. It refuses to join records of different
    ``dp_rank`` values: Dynamo's publisher assigns ``counter_id`` per rank and
    also advances it on idle heartbeats, so equal counters on two ranks are
    not the same scheduler step. A cross-rank join needs a step id that the
    scheduler synchronizes across ranks. Iterations without any positive
    ``wall_time`` (idle heartbeats) are dropped.
    """
    if join_ranks not in ("none", "counter"):
        raise ValueError(f"join_ranks must be 'none' or 'counter', got {join_ranks!r}")
    iterations: list[list[dict[str, Any]]] = []
    if join_ranks == "none":
        for fpm in records:
            if float(fpm.get("wall_time", 0.0)) > 0.0:
                iterations.append([fpm])
        return iterations
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for fpm in records:
        if "counter_id" not in fpm:
            raise ValueError(
                "join_ranks='counter' needs counter_id on every record; this producer does not emit it, "
                "use --join-ranks none"
            )
        worker_id = str(fpm.get("worker_id") or "")
        if not worker_id:
            raise ValueError(
                "join_ranks='counter' needs a non-empty worker_id on every record; without it records of "
                "different workers with the same counter_id would merge into one iteration"
            )
        key = (worker_id, int(fpm["counter_id"]))
        rank = int(fpm.get("dp_rank", 0))
        if rank in grouped[key]:
            raise ValueError(
                f"duplicate FPM record for worker_id={key[0]!r} counter_id={key[1]} dp_rank={rank}; "
                "counter_id repeats (worker restart or concatenated runs) make the rank join ambiguous"
            )
        grouped[key][rank] = fpm
    for key in sorted(grouped, key=lambda k: (k[0], k[1])):
        ranks = grouped[key]
        if len(ranks) > 1:
            raise ValueError(
                f"worker_id={key[0]!r} counter_id={key[1]} has records from dp_ranks {sorted(ranks)}; "
                "counter_id is rank-local (each rank's publisher advances it, heartbeats included), so it "
                "cannot align attention-DP ranks. Train per rank with --join-ranks none"
            )
        metrics_by_rank = [ranks[rank] for rank in sorted(ranks)]
        if any(float(m.get("wall_time", 0.0)) > 0.0 for m in metrics_by_rank):
            iterations.append(metrics_by_rank)
    return iterations


# ---------------------------------------------------------------------------
# Feature extraction and workload classification (mirror learned.rs)
# ---------------------------------------------------------------------------


def _validate_request_lists(sched: dict[str, Any], index: int) -> None:
    """Same rule as the Rust validator: both lists empty, or both aligned with one entry per request."""
    extend = sched.get("extend_lengths") or []
    past = sched.get("past_kv_lengths") or []
    if not extend and not past:
        return
    expected = int(sched.get("num_prefill_requests", 0)) + int(sched.get("num_decode_requests", 0))
    if len(extend) != expected or len(past) != expected:
        raise ValueError(
            f"iteration {index}: per-request lists must both be empty or both have {expected} entries "
            f"(num_prefill_requests + num_decode_requests); got extend_lengths={len(extend)}, "
            f"past_kv_lengths={len(past)}"
        )


def _request_pairs(sched: dict[str, Any]) -> list[tuple[int, int]] | None:
    """Aligned, non-empty ``(extend, past)`` pairs of one rank, or ``None`` when absent."""
    extend = sched.get("extend_lengths") or []
    past = sched.get("past_kv_lengths") or []
    if not extend or len(extend) != len(past):
        return None
    return [(int(e), int(p)) for e, p in zip(extend, past, strict=True)]


def _pairs_consistent(sched: dict[str, Any], pairs: Sequence[tuple[int, int]]) -> bool:
    """Same bound as ``IterationFeatureVector``: the past-KV total cannot exceed what the
    aggregates describe (``+ sum_prefill_tokens`` for producers that read the KV sum after the
    step, ``+ num_decode_requests`` for per-request decode lengths read one token late)."""
    bound = (
        int(sched.get("sum_prefill_kv_tokens", 0))
        + int(sched.get("sum_decode_kv_tokens", 0))
        + int(sched.get("sum_prefill_tokens", 0))
        + int(sched.get("num_decode_requests", 0))
    )
    return sum(p for _, p in pairs) <= bound


def _has_request_lists(sched: dict[str, Any]) -> bool:
    """Both per-request lists present, non-empty, aligned and consistent with the aggregates."""
    pairs = _request_pairs(sched)
    return pairs is not None and _pairs_consistent(sched, pairs)


def _rank_active(sched: dict[str, Any]) -> bool:
    return int(sched.get("sum_prefill_tokens", 0)) > 0 or int(sched.get("num_decode_requests", 0)) > 0


def _validate_metrics(fpm: dict[str, Any], index: int) -> None:
    """Mirror ``validate_forward_pass_metrics``: version, aggregate/count agreement, list alignment."""
    version = fpm.get("version", 1)
    if version != 1:
        raise ValueError(f"iteration {index}: unsupported FPM version {version!r}; expected 1")
    sched = _sched(fpm)
    np_ = int(sched.get("num_prefill_requests", 0))
    nd = int(sched.get("num_decode_requests", 0))
    if np_ == 0 and (int(sched.get("sum_prefill_tokens", 0)) > 0 or int(sched.get("sum_prefill_kv_tokens", 0)) > 0):
        raise ValueError(f"iteration {index}: prefill tokens reported with num_prefill_requests == 0")
    if nd == 0 and int(sched.get("sum_decode_kv_tokens", 0)) > 0:
        raise ValueError(f"iteration {index}: decode KV tokens reported with num_decode_requests == 0")
    _validate_request_lists(sched, index)


def iteration_has_request_lists(metrics_by_rank: Sequence[dict[str, Any]]) -> bool:
    """``True`` when every active rank carries usable per-request lists (what the Rust model requires
    before it will read ``req_*`` / ``slot*`` features)."""
    active = [_sched(fpm) for fpm in metrics_by_rank if _rank_active(_sched(fpm))]
    return bool(active) and all(_has_request_lists(sched) for sched in active)


def _sched(fpm: dict[str, Any]) -> dict[str, Any]:
    return fpm.get("scheduled_requests") or {}


def classify_workload(metrics_by_rank: Sequence[dict[str, Any]], worker_type: str) -> str | None:
    """Return the regression workload kind for one iteration, or ``None`` if idle.

    Mirrors ``RegressionIterationFeatures::from_metrics`` in Rust, including
    the role-compatibility errors for dedicated prefill/decode workers.
    """
    if worker_type not in WORKER_TYPES:
        raise ValueError(f"invalid worker_type {worker_type!r}")
    if not metrics_by_rank:
        raise ValueError("at least one attention-DP rank metric is required")
    has_prefill = has_decode = has_locally_mixed = False
    for fpm in metrics_by_rank:
        s = _sched(fpm)
        rank_prefill = int(s.get("sum_prefill_tokens", 0)) > 0
        rank_decode = int(s.get("num_decode_requests", 0)) > 0
        if worker_type == "prefill" and rank_decode:
            raise ValueError("prefill regression worker received scheduled decode work")
        if worker_type == "decode" and rank_prefill:
            raise ValueError("decode regression worker received scheduled prefill work")
        has_prefill |= rank_prefill
        has_decode |= rank_decode
        has_locally_mixed |= rank_prefill and rank_decode
    if not has_prefill and not has_decode:
        return None
    if has_prefill and has_decode:
        return "contains_locally_mixed" if has_locally_mixed else "cross_rank_aggregated"
    return "pure_prefill" if has_prefill else "pure_decode"


def compute_features(metrics_by_rank: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Named iteration features; formulas identical to ``IterationFeatureVector``."""
    num_active_ranks = 0.0
    num_prefill = sum_ptok = sum_pkv = var_plen = 0.0
    num_decode = sum_dkv = var_dkv = 0.0
    max_rank_ptok = max_rank_dkv = max_rank_nd = 0.0
    attention_pairs = sum_dkv_sq = 0.0
    pairs: list[tuple[float, float]] | None = []
    for fpm in metrics_by_rank:
        s = _sched(fpm)
        np_ = float(int(s.get("num_prefill_requests", 0)))
        ptok = float(int(s.get("sum_prefill_tokens", 0)))
        pkv = float(int(s.get("sum_prefill_kv_tokens", 0)))
        nd = float(int(s.get("num_decode_requests", 0)))
        dkv = float(int(s.get("sum_decode_kv_tokens", 0)))
        vp = max(float(s.get("var_prefill_length", 0.0)), 0.0)
        vd = max(float(s.get("var_decode_kv_tokens", 0.0)), 0.0)
        has_prefill = ptok > 0.0
        has_decode = nd > 0.0
        if not has_prefill and not has_decode:
            continue
        num_active_ranks += 1.0
        num_prefill += np_
        sum_ptok += ptok
        sum_pkv += pkv
        var_plen = max(var_plen, vp)
        num_decode += nd
        sum_dkv += dkv
        var_dkv = max(var_dkv, vd)
        max_rank_ptok = max(max_rank_ptok, ptok)
        max_rank_dkv = max(max_rank_dkv, dkv)
        max_rank_nd = max(max_rank_nd, nd)
        if has_prefill and np_ > 0.0:
            attention_pairs += pkv * ptok / np_ + ptok * ptok / (2.0 * np_) + ptok / 2.0
        if has_decode:
            mean = dkv / nd
            sum_dkv_sq += nd * vd + nd * mean * mean
        rank_pairs = _request_pairs(s)
        if pairs is not None and rank_pairs is not None and _pairs_consistent(s, rank_pairs):
            pairs.extend((float(e), float(p)) for e, p in rank_pairs)
        else:
            pairs = None
    mean_prefill_chunk = sum_ptok / num_prefill if num_prefill > 0.0 else 0.0
    mean_prefill_kv = sum_pkv / num_prefill if num_prefill > 0.0 else 0.0
    mean_decode_kv = sum_dkv / num_decode if num_decode > 0.0 else 0.0
    values = (
        num_active_ranks,
        num_prefill,
        sum_ptok,
        sum_pkv,
        var_plen,
        num_decode,
        sum_dkv,
        var_dkv,
        max_rank_ptok,
        max_rank_dkv,
        max_rank_nd,
        mean_prefill_chunk,
        mean_prefill_kv,
        attention_pairs,
        mean_decode_kv,
        sum_dkv_sq,
        math.log1p(sum_ptok),
        math.log1p(sum_pkv),
        math.log1p(sum_dkv),
        math.log1p(attention_pairs),
        math.log1p(sum_dkv_sq),
    )
    if pairs:
        pairs.sort(key=lambda t: (-t[1], -t[0]))
        n = float(len(pairs))
        sum_e = sum(e for e, _ in pairs)
        sum_p = sum(p for _, p in pairs)
        sum_attn = sum(e * (p + e / 2.0) for e, p in pairs)
        max_p, min_p = max(p for _, p in pairs), min(p for _, p in pairs)
        req = (
            n,
            sum_e,
            max(e for e, _ in pairs),
            min(e for e, _ in pairs),
            sum_p,
            max_p,
            min_p,
            sum(e * p for e, p in pairs),
            sum(e * e for e, _ in pairs),
            sum(p * p for _, p in pairs),
            sum_attn,
            sum_e * max_p,
            math.log1p(sum_p),
            math.log1p(sum_attn),
            n * sum_e,
            max_p - min_p,
            1.0 if all(e <= 1.0 for e, _ in pairs) else 0.0,
            1.0 if any(e > 1.0 for e, _ in pairs) else 0.0,
        )
        slots: list[float] = []
        for i in range(SLOT_COUNT):
            if i < len(pairs):
                e, p = pairs[i]
                slots.extend((1.0, p, e))
            else:
                slots.extend((0.0, 0.0, 0.0))
    else:
        req = (math.nan,) * len(REQUEST_FEATURE_NAMES)
        slots = [0.0] * (3 * SLOT_COUNT)
    return dict(zip(FEATURE_NAMES, values + req + tuple(slots), strict=True))


def iteration_wall_ms(metrics_by_rank: Sequence[dict[str, Any]]) -> float | None:
    """Maximum finite positive ``wall_time`` across ranks, in milliseconds."""
    best = 0.0
    for fpm in metrics_by_rank:
        wall = float(fpm.get("wall_time", 0.0))
        if math.isfinite(wall) and wall > best:
            best = wall
    return best * 1000.0 if best > 0.0 else None


def _launch_kind(fpm: dict[str, Any], max_cudagraph_capture_size: int) -> str:
    s = _sched(fpm)
    prefill = int(s.get("sum_prefill_tokens", 0))
    lists = s.get("extend_lengths")
    tokens = sum(lists) if lists else prefill + int(s.get("num_decode_requests", 0))
    if tokens > max_cudagraph_capture_size:
        return "eager"
    return "piecewise" if prefill > 0 else "decode"


_OWN_WORK_AXES = {
    "decode": ("num_decode_requests", "decode_kv_per_request"),
    "prefill": ("num_decode_requests", "sum_prefill_tokens"),
}
_OWN_WORK_CELLS = ("n", "ms", "sd_ms")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_own_work_buckets(buckets: Any, max_cudagraph_capture_size: Any, cells: tuple[str, ...] = ()) -> None:
    if (
        not isinstance(max_cudagraph_capture_size, int)
        or isinstance(max_cudagraph_capture_size, bool)
        or max_cudagraph_capture_size <= 0
    ):
        raise ValueError(f"max_cudagraph_capture_size must be a positive integer, got {max_cudagraph_capture_size!r}")
    if not isinstance(buckets, dict) or set(buckets) != set(_OWN_WORK_AXES):
        raise ValueError(f"own-work buckets need exactly the carriers {sorted(_OWN_WORK_AXES)}, got {buckets!r}")
    for carrier, axes in _OWN_WORK_AXES.items():
        table = buckets[carrier]
        if not isinstance(table, dict) or set(table) != {*axes, *cells}:
            raise ValueError(f"own-work buckets[{carrier!r}] need exactly {sorted({*axes, *cells})}, got {table!r}")
        for axis in axes:
            edges = table[axis]
            if (
                not isinstance(edges, list)
                or not edges
                or not all(_is_number(edge) for edge in edges)
                or edges[0] != 0
                or any(low >= high for low, high in pairwise(edges))
            ):
                raise ValueError(
                    f"own-work buckets[{carrier!r}][{axis!r}] must list increasing bucket lower edges from 0, "
                    f"got {edges!r}"
                )
        rows, cols = (len(table[axis]) for axis in axes)
        for cell in cells:
            grid = table[cell]
            if (
                not isinstance(grid, list)
                or len(grid) != rows
                or not all(isinstance(row, list) and len(row) == cols for row in grid)
            ):
                raise ValueError(f"own-work buckets[{carrier!r}][{cell!r}] must be {rows} rows of {cols}, got {grid!r}")
        if cells and not all(ms is None or (_is_number(ms) and ms > 0) for row in table["ms"] for ms in row):
            raise ValueError(f"own-work buckets[{carrier!r}]['ms'] must hold positive numbers or null")


def _carrier(fpm: dict[str, Any]) -> str:
    return "prefill" if int(_sched(fpm).get("sum_prefill_tokens", 0)) > 0 else "decode"


def _own_work_bucket(table: dict[str, Any], carrier: str, fpm: dict[str, Any]) -> tuple[int, int]:
    s = _sched(fpm)
    decodes = int(s.get("num_decode_requests", 0))
    if carrier == "prefill":
        second = float(s.get("sum_prefill_tokens", 0))
    else:
        second = int(s.get("sum_decode_kv_tokens", 0)) / decodes if decodes else 0.0
    first_axis, second_axis = _OWN_WORK_AXES[carrier]
    return (
        bisect.bisect_right(table[first_axis], decodes) - 1,
        bisect.bisect_right(table[second_axis], second) - 1,
    )


def _launch_chain(
    iterations: Sequence[Sequence[dict[str, Any]]], caller: str
) -> tuple[list[dict[str, Any]], list[tuple[str, int]], dict[tuple[str, int], int]]:
    """Return the records, each one's ``(worker_id, counter_id)`` and the index of each key."""
    if any(len(ranks) != 1 for ranks in iterations):
        raise ValueError(
            f"{caller} needs one rank per iteration; only a one-GPU vLLM worker launches in its engine loop"
        )
    fpms = [ranks[0] for ranks in iterations]
    position: dict[tuple[str, int], int] = {}
    for i, fpm in enumerate(fpms):
        worker = str(fpm.get("worker_id") or "")
        if not worker or "counter_id" not in fpm:
            raise ValueError(
                f"{caller} needs counter_id and a non-empty worker_id on every record to find the pass after it"
            )
        key = (worker, int(fpm["counter_id"]))
        if key in position:
            raise ValueError(
                f"duplicate FPM record for worker_id={key[0]!r} counter_id={key[1]}; counter_id repeats (worker "
                "restart or concatenated runs) make the pass after it ambiguous"
            )
        position[key] = i
    return fpms, list(position), position


def mean_own_work_ms(
    iterations: Sequence[Sequence[dict[str, Any]]],
    buckets: dict[str, dict[str, Any]],
    max_cudagraph_capture_size: int,
    *,
    min_rows: int = 20,
) -> dict[str, dict[str, Any]]:
    """Measure the own-work reference that :func:`charge_own_launch` keeps, per bucket.

    A record whose next pass is decode measures its own work, since a decode
    launch hides under it. Those records are grouped by carrier and bucket:
    the carrier is ``"prefill"`` for a record with prefill tokens, else
    ``"decode"``, and ``buckets[carrier]`` lists bucket lower edges from 0 on
    two axes: ``num_decode_requests``, then ``decode_kv_per_request``
    (``sum_decode_kv_tokens`` per decode request) for a decode carrier or
    ``sum_prefill_tokens`` for a prefill carrier.

    Returns ``buckets`` with ``n``, ``ms`` and ``sd_ms`` added per carrier:
    each bucket's record count and the mean and standard deviation of their
    ``wall_time`` in milliseconds. A bucket with fewer than ``min_rows``
    records gets no mean (``None``), and :func:`charge_own_launch` refuses a
    record that needs one. Raises ``ValueError`` on the inputs
    :func:`charge_own_launch` refuses, or when ``buckets`` or
    ``max_cudagraph_capture_size`` is malformed.
    """
    _check_own_work_buckets(buckets, max_cudagraph_capture_size)
    fpms, keys, position = _launch_chain(iterations, "mean_own_work_ms")
    samples: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for fpm, (worker, counter) in zip(fpms, keys, strict=True):
        nxt = position.get((worker, counter + 1))
        if nxt is not None and _launch_kind(fpms[nxt], max_cudagraph_capture_size) == "decode":
            carrier = _carrier(fpm)
            row, col = _own_work_bucket(buckets[carrier], carrier, fpm)
            samples[(carrier, row, col)].append(1000.0 * float(fpm["wall_time"]))
    table: dict[str, dict[str, Any]] = {}
    for carrier, (first, second) in _OWN_WORK_AXES.items():
        edges = buckets[carrier]
        cells = [[samples[(carrier, i, j)] for j in range(len(edges[second]))] for i in range(len(edges[first]))]
        kept = [[ms if ms and len(ms) >= min_rows else None for ms in row] for row in cells]
        table[carrier] = {
            first: list(edges[first]),
            second: list(edges[second]),
            "n": [[len(ms) for ms in row] for row in cells],
            "ms": [[None if ms is None else statistics.fmean(ms) for ms in row] for row in kept],
            "sd_ms": [[None if ms is None else statistics.pstdev(ms) for ms in row] for row in kept],
        }
    return table


def charge_own_launch(
    iterations: Sequence[Sequence[dict[str, Any]]],
    own_work_ms: dict[str, dict[str, Any]],
    max_cudagraph_capture_size: int,
    *,
    flag_thin_buckets: bool = False,
) -> list[list[dict[str, Any]]]:
    """Move the next pass's launch out of each vLLM record, for one-GPU workers.

    Under async scheduling, vLLM launches pass k+1 before it processes pass k's
    output, and the vLLM producer ends pass k's ``wall_time`` when that output
    arrives, so pass k's record lasts about the longer of its own work and that
    launch. When the next pass is ``"piecewise"`` (with prefill, at most
    ``max_cudagraph_capture_size`` scheduled tokens) or ``"eager"`` (more),
    pass k's record keeps at most its own-work reference ``w`` and moves the
    rest to the next record; a decode next pass takes nothing. A record that
    gives time is set to ``w`` and flagged ``launch_censored``, since its own
    work is then unknown.

    ``w`` is the mean of the record's bucket in ``own_work_ms``, the table
    :func:`mean_own_work_ms` returns. Two records are consecutive when they
    share ``worker_id`` and their ``counter_id`` values differ by one, so a
    pass after an idle heartbeat keeps its own launch. Each worker's total
    ``wall_time`` is unchanged, and a record that gives and takes nothing is
    returned as it was.

    A record before a piecewise or eager pass whose bucket has no mean stops
    the charge, unless ``flag_thin_buckets``: then it moves nothing, and it
    and the next record are flagged ``launch_refused``, which :func:`evaluate`
    counts as ``_refused`` and :func:`train` skips. Use it for evaluation
    records.

    Raises ``ValueError`` unless every iteration has one rank, every record
    has ``counter_id`` and a non-empty ``worker_id`` that no other record
    shares with the same ``counter_id``, ``own_work_ms`` has the shape
    :func:`mean_own_work_ms` returns, and, without ``flag_thin_buckets``,
    every record that gives time falls in a bucket with a mean.
    """
    _check_own_work_buckets(own_work_ms, max_cudagraph_capture_size, _OWN_WORK_CELLS)
    fpms, keys, position = _launch_chain(iterations, "charge_own_launch")
    wall_ms = [1000.0 * float(fpm["wall_time"]) for fpm in fpms]
    moved = []
    refused: set[int] = set()
    for i, (fpm, (worker, counter), raw_ms) in enumerate(zip(fpms, keys, wall_ms, strict=True)):
        nxt = position.get((worker, counter + 1))
        if nxt is None or _launch_kind(fpms[nxt], max_cudagraph_capture_size) == "decode":
            moved.append(0.0)
            continue
        carrier = _carrier(fpm)
        table = own_work_ms[carrier]
        row, col = _own_work_bucket(table, carrier, fpm)
        own_ms = table["ms"][row][col]
        if own_ms is None:
            if flag_thin_buckets:
                refused.update((i, nxt))
                moved.append(0.0)
                continue
            first, second = _OWN_WORK_AXES[carrier]
            raise ValueError(
                f"the {carrier} own-work bucket {first}>={table[first][row]}, {second}>={table[second][col]} has "
                f"{table['n'][row][col]} records whose next pass is decode, too few for a mean; use wider buckets"
            )
        moved.append(max(raw_ms - own_ms, 0.0))
    out = []
    for i, (fpm, (worker, counter), raw_ms, given) in enumerate(zip(fpms, keys, wall_ms, moved, strict=True)):
        prev = position.get((worker, counter - 1))
        taken = moved[prev] if prev is not None else 0.0
        if not given and not taken and i not in refused:
            out.append([fpm])
            continue
        record = {**fpm, "wall_time": (raw_ms - given + taken) / 1000.0} if given or taken else dict(fpm)
        if given:
            record["launch_censored"] = True
        if i in refused:
            record["launch_refused"] = True
        out.append([record])
    return out


# ---------------------------------------------------------------------------
# Dataset, training, export
# ---------------------------------------------------------------------------


def build_dataset(
    iterations: Sequence[Sequence[dict[str, Any]]],
    worker_type: str,
    features: Sequence[str],
) -> dict[str, tuple[list[list[float]], list[float]]]:
    """Per-store ``(X, y_ms)`` rows from iterations with positive wall time."""
    return _build_dataset(iterations, worker_type, features)


def _build_dataset(
    iterations: Sequence[Sequence[dict[str, Any]]],
    worker_type: str,
    features: Sequence[str],
    *,
    provenance: dict[str, list[tuple[int, str]]] | None = None,
    censored: dict[str, int] | None = None,
    refused: dict[str, int] | None = None,
) -> dict[str, tuple[list[list[float]], list[float]]]:
    unknown = [name for name in features if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown features {unknown}; valid names: {list(FEATURE_NAMES)}")
    dataset: dict[str, tuple[list[list[float]], list[float]]] = defaultdict(lambda: ([], []))
    needs_lists = any(name.startswith(("req_", "slot")) for name in features)
    skipped_role = skipped_lists = 0
    for index, metrics_by_rank in enumerate(iterations):
        for fpm in metrics_by_rank:
            _validate_metrics(fpm, index)
        try:
            kind = classify_workload(metrics_by_rank, worker_type)
        except ValueError:
            skipped_role += 1
            continue
        if kind is None:
            continue
        wall_ms = iteration_wall_ms(metrics_by_rank)
        if wall_ms is None:
            continue
        if needs_lists and not iteration_has_request_lists(metrics_by_rank):
            # the Rust model refuses these at inference; do not teach the trees a NaN route
            skipped_lists += 1
            continue
        if any(fpm.get("launch_refused") for fpm in metrics_by_rank):
            # charge_own_launch could not tell this record's launch from its own work
            if refused is not None:
                refused[kind] = refused.get(kind, 0) + 1
            continue
        if censored is not None and any(fpm.get("launch_censored") for fpm in metrics_by_rank):
            censored[kind] = censored.get(kind, 0) + 1
        named = compute_features(metrics_by_rank)
        rows, y = dataset[kind]
        rows.append([named[name] for name in features])
        y.append(wall_ms)
        if provenance is not None:
            source_fields = ("worker_id", "dp_rank", "counter_id", "observed_at_unix_ms", "recv_ms")
            source = [{key: fpm[key] for key in source_fields if key in fpm} for fpm in metrics_by_rank]
            provenance.setdefault(kind, []).append(
                (index, json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False))
            )
    if skipped_role:
        logger.warning("skipped %d iterations incompatible with worker_type=%s", skipped_role, worker_type)
    if skipped_lists:
        logger.warning(
            "skipped %d iterations without usable per-request lists (features need extend_lengths/past_kv_lengths)",
            skipped_lists,
        )
    return dict(dataset)


def _export_hgb(model: Any) -> dict[str, Any]:
    """Export a fitted ``HistGradientBoostingRegressor`` to the artifact tree format."""
    import numpy as np

    if getattr(model, "loss", "squared_error") not in ("squared_error", "absolute_error"):
        raise ValueError("only identity-link HGB losses (squared_error/absolute_error) are exportable")
    baseline = float(np.asarray(model._baseline_prediction).reshape(-1)[0])
    trees = []
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise ValueError("multi-output HGB models are not supported")
        nodes = predictors[0].nodes
        is_leaf = nodes["is_leaf"].astype(bool)
        left = np.where(is_leaf, -1, nodes["left"].astype(np.int64)).tolist()
        right = np.where(is_leaf, -1, nodes["right"].astype(np.int64)).tolist()
        feature = np.where(is_leaf, -1, nodes["feature_idx"].astype(np.int64)).tolist()
        thresholds = np.where(is_leaf, 0.0, nodes["num_threshold"].astype(np.float64))
        # sklearn uses +inf for a split separating every numeric value from
        # NaN. Our feature contract permits finite values and NaN, so the
        # largest finite float preserves that routing in strict JSON.
        thresholds[np.isposinf(thresholds)] = np.finfo(np.float64).max
        threshold = thresholds.tolist()
        value = nodes["value"].astype(np.float64).tolist()
        missing_left = nodes["missing_go_to_left"].astype(bool).tolist()
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported by the artifact format")
        trees.append(
            {
                "left": left,
                "right": right,
                "feature": feature,
                "threshold": threshold,
                "value": value,
                "missing_left": missing_left,
            }
        )
    return {"baseline": baseline, "trees": trees}


def _checkpoint_identity(identity: dict[str, Any] | None, worker_type: str) -> dict[str, Any]:
    """Use Rust's shared identity projection for exported artifacts and checkpoints."""
    if not isinstance(identity, dict):
        raise ValueError("online_checkpoint_dir requires an identity containing the canonical FPM configuration")
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    resolved = RustForwardPassPerfModel.learned_identity(identity)
    if resolved.get("worker_type") != worker_type:
        raise ValueError("checkpoint identity worker_type must match the training worker_type")
    return resolved


def write_original_a_manifest(
    path: str | os.PathLike[str],
    iterations: Sequence[Sequence[dict[str, Any]]],
    worker_type: str,
    *,
    identity: dict[str, Any],
    min_store_rows: int,
    seed_artifact_paths: Sequence[str | os.PathLike[str]],
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Pin the original self-benchmark groups and fitted seeds for mixed routing.

    Use only the immutable A population, before admitting online feedback.
    Eligibility follows the fitted seeds' shared ordered feature list, including
    groups in stores below ``min_store_rows``. Rust validates source identity,
    seed usability, complete rank groups, class counts and calibration geometry
    at load time.
    An existing manifest is never overwritten. The result is the native
    ``temporary_mixed_composition.original_a`` configuration value.
    """
    if isinstance(min_store_rows, bool) or not isinstance(min_store_rows, int) or min_store_rows < 1:
        raise ValueError("original A min_store_rows must be a positive integer")
    if not seed_artifact_paths:
        raise ValueError("original A requires at least one fitted seed artifact")
    canonical = _checkpoint_identity(identity, worker_type)
    seeds = []
    features: tuple[str, ...] | None = None
    for source in seed_artifact_paths:
        source = Path(source).resolve(strict=True)
        content = source.read_bytes()
        artifact = json.loads(content)
        seed_features = artifact.get("features") if isinstance(artifact, dict) else None
        if not isinstance(seed_features, list) or not seed_features:
            raise ValueError("original A seed features must be a non-empty list")
        unknown = [name for name in seed_features if name not in FEATURE_NAMES]
        if unknown:
            raise ValueError(f"unknown original A seed features {unknown}")
        _reject_duplicate_features(seed_features)
        if features is None:
            features = tuple(seed_features)
        elif tuple(seed_features) != features:
            raise ValueError("original A seeds must use the same ordered features")
        reference = {"path": str(source), "sha256": hashlib.sha256(content).hexdigest()}
        del artifact, content  # Retain only the feature ABI and authenticated references.
        checkpoint_manifest = source.parent / "manifest.json"
        if source.name == "model.json" and checkpoint_manifest.is_file():
            reference["checkpoint_manifest_sha256"] = hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest()
        seeds.append(reference)
    assert features is not None
    accepted: dict[str, list[tuple[int, str]]] = {}
    _build_dataset(iterations, worker_type, features, provenance=accepted)
    indices = sorted({index for rows in accepted.values() for index, _ in rows})
    del accepted  # Eligibility is resolved; release serialized per-group provenance.
    if not indices:
        raise ValueError("original A has no eligible measured groups")
    manifest = {
        "schema_version": 1,
        "source_kind": "self_benchmark",
        "identity": canonical,
        "min_store_rows": min_store_rows,
        "accepted_iterations": [iterations[index] for index in indices],
        "seed_artifacts": seeds,
        "provenance": provenance,
    }
    destination = Path(path).absolute()
    digest = hashlib.sha256()
    with destination.open("xb") as output:

        def emit(text: str) -> None:
            content = text.encode("utf-8")
            output.write(content)
            digest.update(content)

        def encode(value: Any) -> str:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

        try:
            emit("{")
            for field_index, key in enumerate(sorted(manifest)):
                if field_index:
                    emit(",")
                emit(encode(key) + ":")
                if key == "accepted_iterations":
                    emit("[")
                    for group_index, group in enumerate(manifest[key]):
                        if group_index:
                            emit(",")
                        emit(encode(group))
                    emit("]")
                else:
                    emit(encode(manifest[key]))
            emit("}")
        except BaseException:
            # Only remove the file this exclusive open created. A failed export
            # must not leave a partial manifest that prevents a corrected retry.
            destination.unlink()
            raise
    return {"manifest_path": str(destination), "source_sha256": digest.hexdigest()}


def _online_store_metadata(model: Any, feature_count: int) -> dict[str, Any]:
    """Describe fitted state supported by the scalar, unconstrained Rust updater."""
    import numpy as np
    import sklearn

    if sklearn.__version__ not in ONLINE_SUPPORTED_SKLEARN_VERSIONS:
        raise ValueError(
            f"online checkpoint export does not support sklearn {sklearn.__version__}; "
            f"supported versions: {sorted(ONLINE_SUPPORTED_SKLEARN_VERSIONS)}"
        )
    if model.loss != "squared_error":
        raise ValueError("online checkpoint export requires squared_error loss")
    if model.early_stopping is not False or getattr(model, "do_early_stopping_", True):
        raise ValueError("online checkpoint export requires early_stopping=False and all fit rows")
    # train() owns this model's fit call and never supplies sample weights.
    # sklearn does not retain sufficient metadata to infer that afterward.
    if model.monotonic_cst is not None or model.interaction_cst is not None:
        raise ValueError("online checkpoint export does not support monotonic or interaction constraints")
    if model.max_features != 1.0:
        raise ValueError("online checkpoint export does not support feature subsampling")
    mapper = model._bin_mapper
    if np.any(mapper.is_categorical_):
        raise ValueError("online checkpoint export does not support categorical features")
    thresholds = [np.asarray(values, dtype=np.float64).tolist() for values in mapper.bin_thresholds_]
    counts = np.asarray(mapper.n_bins_non_missing_, dtype=np.uint32).tolist()
    missing_bin = int(mapper.missing_values_bin_idx_)
    if len(thresholds) != feature_count or len(counts) != feature_count:
        raise ValueError("fitted bin mapper feature count does not match the checkpoint feature ABI")
    for values, count in zip(thresholds, counts, strict=True):
        if (
            count != len(values) + 1
            or not 1 <= count <= missing_bin
            or any(not math.isfinite(value) for value in values)
            or any(left >= right for left, right in pairwise(values))
        ):
            raise ValueError("fitted bin mapper has invalid numeric thresholds or bin counts")
    return {
        "bin_thresholds": thresholds,
        "missing_values_bin_idx": missing_bin,
        "n_bins_non_missing": counts,
        "loss": model.loss,
        "learning_rate": float(model.learning_rate),
        "l2_regularization": float(model.l2_regularization),
        "min_samples_leaf": int(model.min_samples_leaf),
        "max_depth": int(model.max_depth) if model.max_depth is not None else None,
        "max_leaf_nodes": int(model.max_leaf_nodes) if model.max_leaf_nodes is not None else None,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint_files(directory: Path, manifest: dict[str, Any]) -> None:
    """Verify written bytes before publishing the checkpoint's commit manifest."""
    entries = [(manifest["artifact_file"], manifest["artifact_sha256"])]
    entries.extend((store["rows_file"], store["rows_sha256"]) for store in manifest["stores"].values())
    for filename, expected in entries:
        if Path(filename).name != filename or _file_sha256(directory / filename) != expected:
            raise ValueError(f"checkpoint file hash mismatch or invalid filename: {filename}")


def _write_training_checkpoint(
    destination: str | os.PathLike[str],
    artifact: dict[str, Any],
    identity: dict[str, Any],
    dataset: dict[str, tuple[list[list[float]], list[float]]],
    provenance: dict[str, list[tuple[int, str]]],
    store_metadata: dict[str, dict[str, Any]],
) -> None:
    """Publish a new checkpoint directory atomically, with its manifest written last."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import sklearn

    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_artifact(artifact)
    artifact_bytes = json.dumps(artifact, separators=(",", ":"), allow_nan=False).encode("utf-8")
    manifest = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "schema_version": TRAINING_CHECKPOINT_VERSION,
        "artifact_file": "model.json",
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "identity": identity,
        "features": list(artifact["features"]),
        "target": artifact["target"],
        "sklearn_version": sklearn.__version__,
        "exporter_version": TRAINING_EXPORTER_VERSION,
        "shrinkage": "included_in_leaf_values",
        "stores": {},
    }
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.tmp-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        (staging / "model.json").write_bytes(artifact_bytes)
        for kind, metadata in sorted(store_metadata.items()):
            rows, wall_ms = dataset[kind]
            values = np.asarray(rows, dtype=np.float64)
            wall = np.asarray(wall_ms, dtype=np.float64)
            if values.shape != (len(wall), len(artifact["features"])) or np.isinf(values).any():
                raise ValueError(f"store {kind}: invalid training features")
            if not np.isfinite(wall).all() or (wall <= 0).any():
                raise ValueError(f"store {kind}: training wall_time_ms must be finite and positive")
            # Match train() exactly, including Python's scalar log rounding.
            target = [math.log(value) for value in wall_ms] if artifact["target"] == "log_ms" else wall_ms
            source = provenance[kind]
            if len(source) != len(rows) or len({row_id for row_id, _ in source}) != len(source):
                raise ValueError(f"store {kind}: training row provenance must match rows and have unique IDs")
            columns = {
                "row_id": pa.array([row_id for row_id, _ in source], type=pa.uint64()),
                "y": pa.array(target, type=pa.float64()),
                "wall_time_ms": pa.array(wall, type=pa.float64()),
                **{
                    f"x{index}": pa.array(values[:, index], type=pa.float64(), from_pandas=False)
                    for index in range(values.shape[1])
                },
                "source_json": pa.array([record for _, record in source], type=pa.string()),
            }
            rows_file = f"{kind}.rows.parquet"
            pq.write_table(pa.table(columns), staging / rows_file, compression="zstd")
            manifest["stores"][kind] = {
                "rows_file": rows_file,
                "rows_sha256": _file_sha256(staging / rows_file),
                "row_count": len(rows),
                **metadata,
            }
        _verify_checkpoint_files(staging, manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        # Reserve once so concurrent exports cannot replace an empty destination.
        destination.mkdir()
        try:
            os.rename(staging, destination)
        except OSError:
            try:
                destination.rmdir()  # Only our still-empty reservation may be removed.
            except OSError:
                pass
            raise


def train(
    iterations: Sequence[Sequence[dict[str, Any]]],
    worker_type: str,
    *,
    features: Sequence[str] | None = None,
    target: str = "log_ms",
    max_iter: int = 600,
    learning_rate: float = 0.05,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 5,
    min_store_rows: int = 20,
    random_state: int = 0,
    early_stopping: bool = False,
    metadata: dict[str, Any] | None = None,
    online_checkpoint_dir: str | os.PathLike[str] | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit each workload store and return the prediction artifact unchanged.

    ``online_checkpoint_dir`` additionally exports the exact fitted rows and
    bin mapper for Rust online fitting. It requires a canonical ``identity``
    configuration, supported sklearn, and ``early_stopping=False``. Existing
    checkpoint directories are never overwritten.

    The metadata's ``launch_censored`` counts, per store, the rows that
    :func:`charge_own_launch` set to their own-work reference; they are fitted
    like the others. ``launch_refused`` counts the rows it flagged
    ``launch_refused``, which are skipped.
    """
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor

    if worker_type not in WORKER_TYPES:
        raise ValueError(f"invalid worker_type {worker_type!r}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    checkpoint_identity = _checkpoint_identity(identity, worker_type) if identity is not None else None
    if (
        checkpoint_identity is not None
        and metadata
        and "identity" in metadata
        and metadata["identity"] != checkpoint_identity
    ):
        raise ValueError("metadata.identity conflicts with the canonical training identity")
    provenance: dict[str, list[tuple[int, str]]] = {}
    checkpoint_stores: dict[str, dict[str, Any]] = {}
    if online_checkpoint_dir is not None:
        if early_stopping is not False:
            raise ValueError("online checkpoint export requires early_stopping=False and all fit rows")
        if sklearn.__version__ not in ONLINE_SUPPORTED_SKLEARN_VERSIONS:
            raise ValueError(f"online checkpoint export does not support sklearn {sklearn.__version__}")
        if protected := _TRAINING_METADATA_FIELDS.intersection(metadata or {}):
            raise ValueError(f"checkpoint metadata cannot override training fields: {sorted(protected)}")
        if os.path.lexists(online_checkpoint_dir):
            raise FileExistsError(f"checkpoint destination already exists: {online_checkpoint_dir}")
        if checkpoint_identity is None:
            checkpoint_identity = _checkpoint_identity(identity, worker_type)
    features = tuple(features) if features else DEFAULT_FEATURES[worker_type]
    _reject_duplicate_features(features)
    needs_lists = any(name.startswith(("req_", "slot")) for name in features)
    if needs_lists and not any(
        _has_request_lists(_sched(fpm)) for metrics_by_rank in iterations for fpm in metrics_by_rank
    ):
        raise ValueError(
            "per-request features requested but no FPM record carries extend_lengths/past_kv_lengths; "
            "collect with the per-request FPM extension or use the default aggregate --features fpm5 (or v1)"
        )
    censored: dict[str, int] = {}
    refused: dict[str, int] = {}
    dataset = _build_dataset(
        iterations,
        worker_type,
        features,
        provenance=provenance if online_checkpoint_dir is not None else None,
        censored=censored,
        refused=refused,
    )
    if not dataset:
        raise ValueError("no trainable iterations (all idle, zero wall_time, or role-incompatible)")
    stores: dict[str, Any] = {}
    counts: dict[str, int] = {}
    trees: dict[str, int] = {}
    for kind, (rows, y) in sorted(dataset.items()):
        if len(y) < min_store_rows:
            logger.warning("store %s has only %d rows (< %d); skipped", kind, len(y), min_store_rows)
            continue
        y_fit = [math.log(v) for v in y] if target == "log_ms" else list(y)
        # sklearn's default early_stopping="auto" switches on above 10k rows and
        # holds out 10 % of them; keep the fit identical across dataset sizes
        # unless the caller asks for it, and record what was actually fitted.
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            early_stopping=early_stopping,
        )
        model.fit(rows, y_fit)
        stores[kind] = _export_hgb(model)
        if online_checkpoint_dir is not None:
            checkpoint_stores[kind] = _online_store_metadata(model, len(features))
        counts[kind] = len(y)
        trees[kind] = int(getattr(model, "n_iter_", len(stores[kind]["trees"])))
        logger.info("store %s: %d rows, %d trees", kind, len(y), trees[kind])
    if not stores:
        raise ValueError("every store fell below min_store_rows; nothing to export")
    artifact = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "worker_type": worker_type,
        "target": target,
        "features": list(features),
        "stores": stores,
        "metadata": {
            "trainer": "aisimulate_core.sdk.fpm_learned",
            "model": "sklearn.HistGradientBoostingRegressor",
            "hyperparameters": {
                "max_iter": max_iter,
                "learning_rate": learning_rate,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": min_samples_leaf,
                "early_stopping": early_stopping,
            },
            "train_rows": counts,
            **({"launch_censored": censored} if censored else {}),
            **({"launch_refused": refused} if refused else {}),
            "trees_fitted": trees,
            **(metadata or {}),
            **({"identity": checkpoint_identity} if checkpoint_identity is not None else {}),
        },
    }
    if online_checkpoint_dir is not None:
        assert checkpoint_identity is not None
        _write_training_checkpoint(
            online_checkpoint_dir, artifact, checkpoint_identity, dataset, provenance, checkpoint_stores
        )
    return artifact


# ---------------------------------------------------------------------------
# Pure-Python reference inference (mirrors learned.rs) and evaluation
# ---------------------------------------------------------------------------


_TREE_ARRAYS = ("left", "right", "feature", "threshold", "value", "missing_left")


def _reject_duplicate_features(features: Sequence[str]) -> None:
    """Same rule as the Rust loader: unique names, so at most ``len(FEATURE_NAMES)`` of them."""
    seen: set[str] = set()
    duplicates = sorted({name for name in features if name in seen or seen.add(name)})
    if duplicates:
        raise ValueError(f"duplicate features {duplicates}")


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Structural pre-check of an artifact (schema, version, target, features, tree shape).

    Raises ``ValueError`` with the offending field. The Rust loader
    (``LearnedForwardPassModel::from_artifact``) is authoritative and additionally checks
    store kinds against the worker type and the finiteness of every number.
    """
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a JSON object")
    if artifact.get("schema") != SCHEMA_NAME:
        raise ValueError(f"unsupported artifact schema {artifact.get('schema')!r}; expected {SCHEMA_NAME!r}")
    version = artifact.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artifact schema_version {artifact.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if artifact.get("worker_type") not in WORKER_TYPES:
        raise ValueError(f"invalid worker_type {artifact.get('worker_type')!r}")
    if artifact.get("target") not in TARGETS:
        raise ValueError(f"unsupported target {artifact.get('target')!r}; expected one of {TARGETS}")
    features = artifact.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("artifact features must be a non-empty list")
    unknown = [name for name in features if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown features {unknown}")
    _reject_duplicate_features(features)
    stores = artifact.get("stores")
    if not isinstance(stores, dict) or not stores:
        raise ValueError("artifact stores must be a non-empty mapping")
    for kind, store in stores.items():
        if kind not in WORKLOAD_KINDS:
            raise ValueError(f"unknown store {kind!r}")
        if not isinstance(store, dict) or "baseline" not in store or not isinstance(store.get("trees"), list):
            raise ValueError(f"store {kind!r} needs a baseline and a trees list")
        for index, tree in enumerate(store["trees"]):
            arrays = [tree.get(key) for key in _TREE_ARRAYS[:-1]]
            if any(not isinstance(a, list) for a in arrays):
                raise ValueError(f"store {kind!r} tree {index}: every tree array must be a list")
            n = len(arrays[0])
            if n == 0 or any(len(a) != n for a in arrays):
                raise ValueError(f"store {kind!r} tree {index}: tree arrays must be non-empty and equal length")
            missing_left = tree.get("missing_left") or []
            if missing_left and len(missing_left) != n:  # absent/empty = route NaN left, like the Rust loader
                raise ValueError(f"store {kind!r} tree {index}: missing_left length mismatch")
            for node in range(n):
                lc, rc = tree["left"][node], tree["right"][node]
                if (lc < 0) != (rc < 0):
                    raise ValueError(
                        f"store {kind!r} tree {index} node {node}: children must both be leaves or both inner"
                    )
                if lc >= 0 and not (node < lc < n and node < rc < n):
                    raise ValueError(f"store {kind!r} tree {index} node {node}: children must point to larger indices")
                if lc >= 0 and not 0 <= tree["feature"][node] < len(features):
                    raise ValueError(f"store {kind!r} tree {index} node {node}: feature index out of range")


def evaluate(
    artifact: dict[str, Any],
    iterations: Sequence[Sequence[dict[str, Any]]],
    predict=None,
) -> dict[str, dict[str, float]]:
    """Per-store and overall APE statistics against observed wall time.

    Predictions come from the compiled Rust model (the single inference
    oracle). ``predict`` exists for the parity tests, which score the
    pure-Python mirror against it; it is not exposed on the CLI.
    ``_launch_censored`` counts the rows :func:`charge_own_launch` set to
    their own-work reference; they are scored like the others.
    """
    validate_artifact(artifact)
    if predict is None:
        predict = rust_predictor(artifact)
    apes: dict[str, list[float]] = defaultdict(list)
    for metrics_by_rank in iterations:
        try:
            kind = classify_workload(metrics_by_rank, artifact["worker_type"])
        except ValueError:
            continue
        if kind is None:
            continue
        truth = iteration_wall_ms(metrics_by_rank)
        if truth is None:
            continue
        if any(fpm.get("launch_refused") for fpm in metrics_by_rank):
            # charge_own_launch could not tell this record's launch from its own work
            apes["_refused"].append(0.0)
            continue
        if any(fpm.get("launch_censored") for fpm in metrics_by_rank):
            apes["_launch_censored"].append(0.0)
        try:
            pred = predict(metrics_by_rank)
        except (ValueError, RuntimeError):
            # the Rust model refuses iterations without per-request lists for req_*/slot artifacts
            apes["_refused"].append(0.0)
            continue
        if pred is None:
            apes["_unpredicted"].append(0.0)
            continue
        ape = abs(pred - truth) / truth
        apes[kind].append(ape)
        apes["_all"].append(ape)
    result: dict[str, dict[str, float]] = {}
    for kind, values in apes.items():
        if kind in ("_unpredicted", "_refused", "_launch_censored"):
            result[kind] = {"n": float(len(values))}
            continue
        ordered = sorted(values)
        n = len(ordered)
        result[kind] = {
            "n": float(n),
            "mape_pct": 100.0 * sum(ordered) / n,
            "median_ape_pct": 100.0 * ordered[n // 2],
            "p95_ape_pct": 100.0 * ordered[min(n - 1, int(0.95 * (n - 1)))],
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def rust_predictor(artifact: dict[str, Any]):
    """Return ``RustForwardPassPerfModel.estimate_forward_pass_time_ms`` bound to ``artifact``.

    Fails loudly when the compiled extension is unavailable instead of falling
    back to the Python mirror.
    """
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    return RustForwardPassPerfModel.from_learned(artifact).estimate_forward_pass_time_ms


def _split_holdout(
    iterations: list[list[dict[str, Any]]], holdout_frac: float, seed: int
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    if holdout_frac <= 0.0:
        return iterations, []
    rng = random.Random(seed)
    order = list(range(len(iterations)))
    rng.shuffle(order)
    cut = round(len(order) * (1.0 - holdout_frac))
    train_idx, test_idx = sorted(order[:cut]), sorted(order[cut:])
    return [iterations[i] for i in train_idx], [iterations[i] for i in test_idx]


def _format_report(report: dict[str, dict[str, float]]) -> str:
    lines = [f"{'store':<26}{'n':>9}{'MAPE%':>9}{'median%':>9}{'p95%':>9}"]
    for kind in sorted(report, key=lambda k: (k.startswith("_"), k)):
        stats = report[kind]
        if kind in ("_unpredicted", "_refused", "_launch_censored"):
            lines.append(f"{kind:<26}{int(stats['n']):>9}{'-':>9}{'-':>9}{'-':>9}")
            continue
        lines.append(
            f"{kind:<26}{int(stats['n']):>9}{stats['mape_pct']:>9.2f}"
            f"{stats['median_ape_pct']:>9.2f}{stats['p95_ape_pct']:>9.2f}"
        )
    return "\n".join(lines)


def _read_launch_stall(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or set(spec) != {"max_cudagraph_capture_size", "own_work_buckets"}:
        raise ValueError(f"--launch-stall needs exactly max_cudagraph_capture_size and own_work_buckets, got {spec!r}")
    _check_own_work_buckets(spec["own_work_buckets"], spec["max_cudagraph_capture_size"])
    return spec


def _load_iterations(
    paths: Sequence[str], join_ranks: str, launch_stall: dict[str, Any] | None
) -> list[list[dict[str, Any]]]:
    """Load evaluation records, charged with a trained own-work table when given."""
    iterations = group_iterations(iter_fpm_records(paths), join_ranks)
    if launch_stall is not None:
        iterations = charge_own_launch(
            iterations,
            launch_stall["own_work_ms"],
            launch_stall["max_cudagraph_capture_size"],
            flag_thin_buckets=True,
        )
    return iterations


def _format_own_work(own_work_ms: dict[str, dict[str, Any]]) -> str:
    lines = []
    for carrier, (first, second) in _OWN_WORK_AXES.items():
        table = own_work_ms[carrier]
        for i, low in enumerate(table[first]):
            for j, low2 in enumerate(table[second]):
                n, ms, sd = (table[cell][i][j] for cell in _OWN_WORK_CELLS)
                stats = "too few for a mean" if ms is None else f"mean {ms:.2f} ms, sd {sd:.2f} ms"
                lines.append(f"{carrier} own work, {first}>={low}, {second}>={low2}: n={n}, {stats}")
    return "\n".join(lines)


def _cmd_train(args: argparse.Namespace) -> int:
    if args.online_checkpoint_dir and not args.identity:
        raise ValueError("--online-checkpoint-dir requires --identity with the canonical FPM configuration JSON")
    identity = json.loads(Path(args.identity).read_text(encoding="utf-8")) if args.identity else None
    spec = _read_launch_stall(args.launch_stall)
    iterations = group_iterations(iter_fpm_records(args.fpm), args.join_ranks)
    if not iterations:
        print("no FPM iterations with positive wall_time found", file=sys.stderr)
        return 2
    launch_stall = None
    if spec is not None:
        # measured on --fpm only; evaluation records are charged with this table, never their own
        size = spec["max_cudagraph_capture_size"]
        own_work = mean_own_work_ms(iterations, spec["own_work_buckets"], size, min_rows=args.min_store_rows)
        print(_format_own_work(own_work))
        launch_stall = {"max_cudagraph_capture_size": size, "own_work_ms": own_work}
        iterations = charge_own_launch(iterations, own_work, size)
    train_set, holdout = _split_holdout(iterations, args.holdout_frac, args.seed)
    if args.holdout_fpm:
        holdout = holdout + _load_iterations(args.holdout_fpm, args.join_ranks, launch_stall)
    features = None
    if args.features:
        features = list(FEATURE_PRESETS.get(args.features, ())) or [f.strip() for f in args.features.split(",")]
    artifact = train(
        train_set,
        args.worker_type,
        features=features,
        target=args.target,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        min_store_rows=args.min_store_rows,
        random_state=args.seed,
        early_stopping=args.early_stopping,
        online_checkpoint_dir=args.online_checkpoint_dir,
        identity=identity,
        metadata={
            "sources": [os.fspath(p) for p in args.fpm],
            "join_ranks": args.join_ranks,
            "train_iterations": len(train_set),
            "holdout_iterations": len(holdout),
            "features_preset": args.features or DEFAULT_FEATURE_PRESET[args.worker_type],
            **({"launch_stall": launch_stall} if launch_stall is not None else {}),
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB); stores={sorted(artifact['stores'])}")
    print("train-set fit:")
    print(_format_report(evaluate(artifact, train_set)))
    if holdout:
        print(f"random holdout ({len(holdout)} iterations; consecutive steps are correlated, treat as a smoke check):")
        print(_format_report(evaluate(artifact, holdout)))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    artifact = json.loads(Path(args.model).read_text(encoding="utf-8"))
    validate_artifact(artifact)
    # a model trained with --launch-stall charges these records with its own table
    launch_stall = artifact.get("metadata", {}).get("launch_stall")
    print(_format_report(evaluate(artifact, _load_iterations(args.fpm, args.join_ranks, launch_stall))))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aisimulate_core.sdk.fpm_learned",
        description="Train / evaluate a learned forward-pass model from FPM telemetry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fpm", nargs="+", required=True, help="FPM jsonl / jsonl.gz files")
    common.add_argument(
        "--join-ranks",
        choices=("none", "counter"),
        default="none",
        help="'counter' keys records by (worker_id, counter_id) and rejects repeats; cross-rank joins are refused",
    )

    tr = sub.add_parser("train", parents=[common], help="fit and export an artifact")
    tr.add_argument("--worker-type", choices=WORKER_TYPES, required=True)
    tr.add_argument("--out", required=True, help="output artifact JSON path")
    tr.add_argument(
        "--features",
        default=None,
        help="preset (v1 | fpm5 | atomic5 | atomic10 | sglang18 | hisim | all) or comma-separated feature names; "
        "default: fpm5 for every worker type",
    )
    tr.add_argument("--target", choices=TARGETS, default="log_ms")
    tr.add_argument("--max-iter", type=int, default=600)
    tr.add_argument("--learning-rate", type=float, default=0.05)
    tr.add_argument("--max-leaf-nodes", type=int, default=31)
    tr.add_argument("--min-samples-leaf", type=int, default=5)
    tr.add_argument(
        "--min-store-rows",
        type=int,
        default=20,
        help="fewest rows a store needs to be fitted, and an own-work bucket (--launch-stall) to get a mean",
    )
    tr.add_argument(
        "--launch-stall",
        default=None,
        help="JSON file {max_cudagraph_capture_size, own_work_buckets}: measure each bucket's own work on --fpm and "
        "move each vLLM launch to the pass it starts (see charge_own_launch); evaluate reuses the artifact's table",
    )
    tr.add_argument(
        "--online-checkpoint-dir",
        help="also export fitted rows and bins to a new directory for Rust online fitting (early stopping must be off)",
    )
    tr.add_argument("--identity", help="canonical FPM configuration JSON; required with --online-checkpoint-dir")
    tr.add_argument(
        "--early-stopping",
        action="store_true",
        help=(
            "let sklearn hold out 10%% of each store and stop early "
            "(off by default; the artifact records the trees fitted)"
        ),
    )
    tr.add_argument("--holdout-frac", type=float, default=0.2, help="random holdout fraction (0 disables)")
    tr.add_argument("--holdout-fpm", nargs="*", default=None, help="extra files used only for evaluation")
    tr.add_argument("--seed", type=int, default=0)
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("evaluate", parents=[common], help="score an artifact against FPM files")
    ev.add_argument("--model", required=True)
    ev.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
