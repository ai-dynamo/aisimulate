# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare correction policies on held-out chronological FPM observations.

Run from the repository root::

    python -m scripts.fpm_accuracy.compare_calibration --model model.json \
        --input heldout.jsonl.gz --output comparison.json

Each JSONL line must be an object with ``time_ms`` (finite numeric timestamp),
``worker_id`` (nonempty string or integer), and ``metrics`` (a nonempty list of
rank FPM objects). A rank's optional worker_id must match the wrapper. Targets
are positive ``wall_time`` values in seconds; the maximum across ranks is used.
Invalid rank labels become zero in the feedback copy, preserving the native
maximum-finite-positive target rule without passing malformed JSON field types.
Example: {"time_ms": 5000, "worker_id": "worker-1", "metrics": [{
"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1024},
"wall_time": 0.01}]}.

The caller supplies an already trained artifact and disjoint evaluation input.
There is no fitting, splitting, bootstrap, or inference of missing timestamps.
One clock covers the input stream; model state is separate for every worker.
Every event is predicted before feedback. Only paired valid predictions with a
valid target are eligible for feedback. Raw predictions are never tuned.
Reported MAPE is iteration-weighted on common coverage, not the hierarchical
36-case research score. This diagnostic does not reproduce that campaign.

Use --every-events 100 for event-count feedback instead of --interval-ms.
Optional --training-input adds a fitted_fixed arm with each upper bound set to
ceil(1.25 * the training maximum). Supply the disjoint training observations in
the same wrapper format; their labels are not used. Fixed/hybrid cohort labels
always refer to the original default grid. Fixed eviction ties are nondeterministic.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

METHODS = ("raw", "fixed", "dynamic", "hybrid")
COHORTS = ("all", "inside_fixed_grid", "outside_fixed_grid", "invalid_coordinates")
# Pin the experiment's fixed grid in model options, cohort labels, and metadata.
FIXED_MAX_NUM_TOKENS = 8192
FIXED_MAX_BATCH_SIZE = 512
FIXED_MAX_KV_TOKENS = 2_000_000


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def workload_coordinates(metrics):
    """Select critical workload rank by coordinate sum, with the last tie winning."""
    selected, largest = None, -1.0
    for rank in metrics:
        scheduled = rank.get("scheduled_requests", {})
        values = [
            scheduled.get(key, 0) for key in ("sum_prefill_tokens", "num_decode_requests", "sum_decode_kv_tokens")
        ]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            return None
        prefill, decode, kv = values
        if prefill > 0:
            kind, axes = ("mixed", (prefill, kv)) if decode > 0 or kv > 0 else ("prefill", (prefill,))
        elif decode > 0 or kv > 0:
            kind, axes = "decode", (decode, kv)
        else:
            continue
        if sum(axes) >= largest:
            largest = sum(axes)
            selected = kind, axes
    return selected


def fixed_cohort(metrics):
    selected = workload_coordinates(metrics)
    if selected is None:
        return "invalid_coordinates"
    kind, axes = selected
    limits = {
        "prefill": (FIXED_MAX_NUM_TOKENS,),
        "decode": (FIXED_MAX_BATCH_SIZE, FIXED_MAX_KV_TOKENS),
        "mixed": (FIXED_MAX_NUM_TOKENS, FIXED_MAX_KV_TOKENS),
    }[kind]
    return "inside_fixed_grid" if all(x <= hi for x, hi in zip(axes, limits, strict=True)) else "outside_fixed_grid"


def fitted_options(events):
    maxima = dict(max_num_tokens=0, max_batch_size=0, max_kv_tokens=0)
    active = 0
    for event in events:
        _, _, metrics = validate_event(event)
        selected = workload_coordinates(metrics)
        if selected is None:
            continue
        kind, axes = selected
        key = "max_batch_size" if kind == "decode" else "max_num_tokens"
        maxima[key] = max(maxima[key], axes[0])
        if kind != "prefill":
            maxima["max_kv_tokens"] = max(maxima["max_kv_tokens"], axes[1])
        active += 1
    if not active:
        raise ValueError("training input has no valid active workload coordinates")
    bounds = {key: max(1, math.ceil(1.25 * value)) for key, value in maxima.items()}
    if any(value > 2**32 - 1 for value in bounds.values()):
        raise ValueError("fitted bounds exceed the native u32 option range")
    return bounds, dict(training_maxima=maxima, bounds=bounds, active_events=active, margin=1.25)


def validate_event(event):
    timestamp = event.get("time_ms")
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
        raise ValueError("each event requires a finite numeric time_ms; timestamps are never inferred")
    worker = event.get("worker_id")
    if isinstance(worker, bool) or not isinstance(worker, (str, int)) or not str(worker):
        raise ValueError("each event requires a nonempty string or integer worker_id")
    metrics = event.get("metrics")
    if not isinstance(metrics, list) or not metrics or any(not isinstance(rank, dict) for rank in metrics):
        raise ValueError("metrics must be a nonempty list of rank FPM objects")
    for rank in metrics:
        if "worker_id" in rank and str(rank["worker_id"]) != str(worker):
            raise ValueError("rank worker_id differs from event worker_id")
        if not isinstance(rank.get("scheduled_requests", {}), dict):
            raise ValueError("scheduled_requests must be an object")
    return timestamp, str(worker), metrics


def compare_events(events, factory, interval_ms=5000.0, *, every_events=None, methods=METHODS):
    """factory(method) returns an independent model exposing predict/tune SDK methods."""
    if not finite_positive(interval_ms):
        raise ValueError("interval_ms must be finite and positive")
    if every_events is not None and (
        isinstance(every_events, bool) or not isinstance(every_events, int) or every_events < 1
    ):
        raise ValueError("every_events must be a positive integer")
    state, anchor, previous, tick = {}, None, None, 0
    result = dict(
        events=0,
        workers=0,
        valid_targets=0,
        invalid_targets=0,
        scheduled_events=0,
        feedback_events=0,
        available=dict.fromkeys(methods, 0),
        paired_predictions=0,
        cohorts={},
    )
    totals = {
        name: dict(valid_targets=0, count=0, target_sum=0.0, errors={m: dict(ape=0.0, absolute=0.0) for m in methods})
        for name in COHORTS
    }
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("each event must be an object")
        timestamp, worker, metrics = validate_event(event)
        if previous is not None and timestamp < previous:
            raise ValueError("time_ms must be nondecreasing; input is not sorted implicitly")
        previous = timestamp
        if anchor is None:
            anchor = timestamp
        if worker not in state:
            state[worker] = {method: factory(method) for method in methods}
        query = [{key: value for key, value in rank.items() if key != "wall_time"} for rank in metrics]
        predictions = {method: model.estimate_forward_pass_time_ms(query) for method, model in state[worker].items()}
        valid = {method: finite_positive(value) for method, value in predictions.items()}
        targets = [rank.get("wall_time") for rank in metrics if finite_positive(rank.get("wall_time"))]
        target = max(targets) * 1000 if targets else None
        target_valid = finite_positive(target)
        result["events"] += 1
        result["valid_targets" if target_valid else "invalid_targets"] += 1
        for method in methods:
            result["available"][method] += int(valid[method])
        paired = target_valid and all(valid.values())
        result["paired_predictions"] += int(paired)
        for name in ("all", fixed_cohort(metrics)):
            total = totals[name]
            total["valid_targets"] += int(target_valid)
            if paired:
                total["count"] += 1
                total["target_sum"] += target
                for method, prediction in predictions.items():
                    absolute = abs(prediction - target)
                    total["errors"][method]["absolute"] += absolute
                    total["errors"][method]["ape"] += absolute / target
        reached = result["events"] // every_events if every_events else math.floor((timestamp - anchor) / interval_ms)
        if reached > tick:
            result["scheduled_events"] += 1
            if paired:
                # All corrected arms see the same label after all predictions.
                # Native target selection ignores nonfinite/nonpositive values,
                # but its f64 JSON field cannot deserialize null/string/bool.
                feedback = [
                    dict(rank, wall_time=rank["wall_time"] if finite_positive(rank.get("wall_time")) else 0.0)
                    for rank in metrics
                ]
                for method in methods:
                    if method != "raw":
                        state[worker][method].tune_with_fpms([feedback])
                result["feedback_events"] += 1
        tick = max(tick, reached)
    result["workers"] = len(state)
    for name, total in totals.items():
        count, eligible = total["count"], total["valid_targets"]
        result["cohorts"][name] = dict(
            valid_targets=eligible,
            paired_predictions=count,
            coverage=count / eligible if eligible else None,
            methods={
                m: dict(
                    mape_pct=100 * e["ape"] / count if count else None,
                    wape_pct=100 * e["absolute"] / total["target_sum"] if count else None,
                )
                for m, e in total["errors"].items()
            },
        )
    result["protocol"] = dict(
        interval_ms=interval_ms if every_events is None else None,
        every_events=every_events,
        clock=(
            "one input event count; admit each Nth event after its prediction"
            if every_events
            else "one input clock; first event per reached tick; no catch-up"
        ),
        feedback="valid target and all arms finite positive; predict before update; raw never tuned",
        state="one independent model per worker and method; shared frozen artifact; no bootstrap",
        metric="iteration-weighted MAPE/WAPE on common coverage; invalid targets excluded",
        fixed_grid=(
            f"inclusive prefill<={FIXED_MAX_NUM_TOKENS}; "
            f"decode batch<={FIXED_MAX_BATCH_SIZE} and KV<={FIXED_MAX_KV_TOKENS}; "
            f"mixed prefill<={FIXED_MAX_NUM_TOKENS} and KV<={FIXED_MAX_KV_TOKENS}"
        ),
    )
    return result


def read_events(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="frozen learned artifact")
    parser.add_argument("--input", type=Path, required=True, help="held-out chronological wrapper JSONL or JSONL.gz")
    parser.add_argument("--output", type=Path, help="write JSON here; otherwise print to stdout")
    cadence = parser.add_mutually_exclusive_group()
    cadence.add_argument("--interval-ms", type=float, default=5000)
    cadence.add_argument("--every-events", type=int, help="feedback on each Nth input event, after predicting it")
    parser.add_argument("--training-input", type=Path, help="disjoint training wrapper JSONL for fitted fixed bounds")
    args = parser.parse_args()
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    model_bytes = args.model.read_bytes()
    artifact = json.loads(model_bytes)
    input_hash = file_hash(args.input)
    fitted, fitted_metadata, training_hash = None, None, None
    if args.training_input:
        training_hash = file_hash(args.training_input)
        if training_hash == input_hash:
            raise ValueError("training input must be disjoint from evaluation input")
        fitted, fitted_metadata = fitted_options(read_events(args.training_input))
        if file_hash(args.training_input) != training_hash:
            raise ValueError("training input changed while fitting bounds")

    def factory(method):
        return RustForwardPassPerfModel.from_learned(
            artifact,
            {
                "correction_mode": {"dynamic": "dynamic_local", "hybrid": "hybrid"}.get(method, "fixed"),
                "max_num_tokens": FIXED_MAX_NUM_TOKENS,
                "max_batch_size": FIXED_MAX_BATCH_SIZE,
                "max_kv_tokens": FIXED_MAX_KV_TOKENS,
                **(fitted if method == "fitted_fixed" else {}),
            },
        )

    methods = (*METHODS, "fitted_fixed") if fitted else METHODS
    result = compare_events(
        read_events(args.input), factory, args.interval_ms, every_events=args.every_events, methods=methods
    )
    if fitted_metadata:
        result["protocol"]["fitted_fixed"] = fitted_metadata
    if file_hash(args.input) != input_hash:
        raise ValueError("input changed during comparison")
    result["sources"] = dict(
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        input_sha256=input_hash,
        harness_sha256=file_hash(Path(__file__)),
        training_input_sha256=training_hash,
    )
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
