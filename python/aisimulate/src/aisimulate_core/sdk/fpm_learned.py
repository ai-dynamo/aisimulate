# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
3. ``RustForwardPassPerfModel.from_learned("model.json")`` loads the artifact
   for pure-Rust inference inside AISimulate / the Dynamo mocker; the online
   correction grid keeps tuning on top of it through ``tune_with_fpms``.

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
import gzip
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_NAME = "aic_fpm_learned_forward_perf"
SCHEMA_VERSION = 1
FPM_VERSION = 1

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

# Feature presets: ``v1`` uses aggregates only (works on any FPM stream),
# ``sglang18`` / ``hisim`` need per-request lists, ``all`` is the union.
FEATURE_PRESETS: dict[str, tuple[str, ...]] = {
    "v1": AGGREGATE_FEATURE_NAMES,
    "sglang18": REQUEST_FEATURE_NAMES,
    "hisim": ("req_batch_size",) + SLOT_FEATURE_NAMES,
    "all": FEATURE_NAMES,
}
# Default = the 18 per-request features (SGLang simulator / HiSim lineage); the
# producer must emit extend_lengths / past_kv_lengths. Aggregate-only streams
# need an explicit ``--features v1``.
DEFAULT_FEATURES: dict[str, tuple[str, ...]] = {
    "decode": REQUEST_FEATURE_NAMES,
    "prefill": REQUEST_FEATURE_NAMES,
    "aggregated": REQUEST_FEATURE_NAMES,
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
    ``join_ranks="counter"`` joins records sharing ``(worker_id, counter_id)``
    across ``dp_rank`` values, which matches attention-DP lockstep engines
    whose ranks publish aligned counters. Iterations without any positive
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
        metrics_by_rank = [ranks[rank] for rank in sorted(ranks)]
        if any(float(m.get("wall_time", 0.0)) > 0.0 for m in metrics_by_rank):
            iterations.append(metrics_by_rank)
    return iterations


# ---------------------------------------------------------------------------
# Feature extraction and workload classification (mirror learned.rs)
# ---------------------------------------------------------------------------


def _has_request_lists(sched: dict[str, Any]) -> bool:
    """Both per-request lists present, non-empty and aligned."""
    extend = sched.get("extend_lengths") or []
    past = sched.get("past_kv_lengths") or []
    return bool(extend) and len(extend) == len(past)


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
        ext_list = s.get("extend_lengths") or []
        past_list = s.get("past_kv_lengths") or []
        if pairs is not None and ext_list and len(ext_list) == len(past_list):
            pairs.extend((float(e), float(p)) for e, p in zip(ext_list, past_list, strict=True))
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


# ---------------------------------------------------------------------------
# Dataset, training, export
# ---------------------------------------------------------------------------


def build_dataset(
    iterations: Sequence[Sequence[dict[str, Any]]],
    worker_type: str,
    features: Sequence[str],
) -> dict[str, tuple[list[list[float]], list[float]]]:
    """Per-store ``(X, y_ms)`` rows from iterations with positive wall time."""
    unknown = [name for name in features if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown features {unknown}; valid names: {list(FEATURE_NAMES)}")
    dataset: dict[str, tuple[list[list[float]], list[float]]] = defaultdict(lambda: ([], []))
    skipped_role = 0
    for metrics_by_rank in iterations:
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
        named = compute_features(metrics_by_rank)
        rows, y = dataset[kind]
        rows.append([named[name] for name in features])
        y.append(wall_ms)
    if skipped_role:
        logger.warning("skipped %d iterations incompatible with worker_type=%s", skipped_role, worker_type)
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
        threshold = np.where(is_leaf, 0.0, nodes["num_threshold"].astype(np.float64)).tolist()
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
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one HGB ensemble per workload store and return the artifact mapping."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    if worker_type not in WORKER_TYPES:
        raise ValueError(f"invalid worker_type {worker_type!r}")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    features = tuple(features) if features else DEFAULT_FEATURES[worker_type]
    needs_lists = any(name.startswith(("req_", "slot")) for name in features)
    if needs_lists and not any(
        _has_request_lists(_sched(fpm)) for metrics_by_rank in iterations for fpm in metrics_by_rank
    ):
        raise ValueError(
            "per-request features requested but no FPM record carries extend_lengths/past_kv_lengths; "
            "collect with the per-request FPM extension or pass --features v1"
        )
    dataset = build_dataset(iterations, worker_type, features)
    if not dataset:
        raise ValueError("no trainable iterations (all idle, zero wall_time, or role-incompatible)")
    stores: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for kind, (rows, y) in sorted(dataset.items()):
        if len(y) < min_store_rows:
            logger.warning("store %s has only %d rows (< %d); skipped", kind, len(y), min_store_rows)
            continue
        y_fit = [math.log(v) for v in y] if target == "log_ms" else list(y)
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model.fit(rows, y_fit)
        stores[kind] = _export_hgb(model)
        counts[kind] = len(y)
        logger.info("store %s: %d rows, %d trees", kind, len(y), len(stores[kind]["trees"]))
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
            },
            "train_rows": counts,
            **(metadata or {}),
        },
    }
    return artifact


# ---------------------------------------------------------------------------
# Pure-Python reference inference (mirrors learned.rs) and evaluation
# ---------------------------------------------------------------------------


_TREE_ARRAYS = ("left", "right", "feature", "threshold", "value", "missing_left")


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Reject artifacts the Rust loader would reject (schema, version, target, features, tree shape).

    Raises ``ValueError`` with the offending field; mirrors ``LearnedForwardPassModel::from_artifact``.
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
        pred = predict(metrics_by_rank)
        if pred is None:
            apes["_unpredicted"].append(0.0)
            continue
        ape = abs(pred - truth) / truth
        apes[kind].append(ape)
        apes["_all"].append(ape)
    result: dict[str, dict[str, float]] = {}
    for kind, values in apes.items():
        if kind == "_unpredicted":
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
        if kind == "_unpredicted":
            lines.append(f"{kind:<26}{int(stats['n']):>9}{'-':>9}{'-':>9}{'-':>9}")
            continue
        lines.append(
            f"{kind:<26}{int(stats['n']):>9}{stats['mape_pct']:>9.2f}"
            f"{stats['median_ape_pct']:>9.2f}{stats['p95_ape_pct']:>9.2f}"
        )
    return "\n".join(lines)


def _cmd_train(args: argparse.Namespace) -> int:
    iterations = group_iterations(iter_fpm_records(args.fpm), args.join_ranks)
    if not iterations:
        print("no FPM iterations with positive wall_time found", file=sys.stderr)
        return 2
    train_set, holdout = _split_holdout(iterations, args.holdout_frac, args.seed)
    if args.holdout_fpm:
        holdout = holdout + group_iterations(iter_fpm_records(args.holdout_fpm), args.join_ranks)
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
        metadata={
            "sources": [os.fspath(p) for p in args.fpm],
            "join_ranks": args.join_ranks,
            "train_iterations": len(train_set),
            "holdout_iterations": len(holdout),
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB); stores={sorted(artifact['stores'])}")
    print("train-set fit:")
    print(_format_report(evaluate(artifact, train_set)))
    if holdout:
        print(f"holdout ({len(holdout)} iterations):")
        print(_format_report(evaluate(artifact, holdout)))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    artifact = json.loads(Path(args.model).read_text(encoding="utf-8"))
    validate_artifact(artifact)
    iterations = group_iterations(iter_fpm_records(args.fpm), args.join_ranks)
    print(_format_report(evaluate(artifact, iterations)))
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
        help="group per-rank records into iterations by (worker_id, counter_id)",
    )

    tr = sub.add_parser("train", parents=[common], help="fit and export an artifact")
    tr.add_argument("--worker-type", choices=WORKER_TYPES, required=True)
    tr.add_argument("--out", required=True, help="output artifact JSON path")
    tr.add_argument(
        "--features",
        default=None,
        help="preset (v1 | sglang18 | hisim | all) or comma-separated feature names; default: sglang18",
    )
    tr.add_argument("--target", choices=TARGETS, default="log_ms")
    tr.add_argument("--max-iter", type=int, default=600)
    tr.add_argument("--learning-rate", type=float, default=0.05)
    tr.add_argument("--max-leaf-nodes", type=int, default=31)
    tr.add_argument("--min-samples-leaf", type=int, default=5)
    tr.add_argument("--min-store-rows", type=int, default=20)
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
