#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently audit prepared native controls and both recorded-clock replays."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def interval(events):
    if not events:
        raise ValueError("Missing numerical stream")
    previous_time, previous_count = -1, -1
    for event in events:
        at, count = int(event["at_ns"]), event["output_tokens"]
        if at < previous_time or type(count) is not int or not 0 <= count <= 128 or count < previous_count:
            raise ValueError("Invalid clock or emitted-token conservation")
        previous_time, previous_count = at, count
    if previous_count != 128:
        raise ValueError("Missing delivered output cap")
    first = next(event for event in events if event["output_tokens"] > 0)
    last = next(event for event in events if event["output_tokens"] == 128)
    return int(first["at_ns"]), (int(last["at_ns"]) - int(first["at_ns"])) / 1e6 / 127


def errors(pairs):
    if not pairs or any(
        not math.isfinite(predicted) or not math.isfinite(reference) or reference <= 0 for predicted, reference in pairs
    ):
        raise ValueError("Empty cohort or undefined positive-reference error")
    ape = [100 * abs(predicted - reference) / reference for predicted, reference in pairs]
    return {
        "n": len(pairs),
        "mape": statistics.mean(ape),
        "median_ape": statistics.median(ape),
        "mean_signed_percent": statistics.mean(100 * (p - r) / r for p, r in pairs),
        "mean_absolute_ms": statistics.mean(abs(p - r) for p, r in pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--client-replay", type=Path, required=True)
    parser.add_argument("--submission-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases, pooled = [], {}
    for mode in ("unprofiled-mtp", "instrumented-mtp"):
        root = args.controls / mode
        raw = (root / "client/requests.jsonl").read_bytes()
        observations = list(map(json.loads, raw.decode().splitlines()))
        measured = {row["rid"]: row for row in observations}
        expected = {f"rsi-submit-r{repeat}-c131072-o128-l8-i{lane}" for repeat in range(3) for lane in range(8)}
        if len(observations) != 24 or set(measured) != expected:
            raise ValueError("Measured cohort loses or duplicates a planned identity")
        ready = {}
        for path in root.glob("scheduler-*.jsonl"):
            for event in map(json.loads, path.read_text().splitlines()):
                if event["rank"] != 0 or event["rid"] not in expected:
                    continue
                generated, emitted, discarded = (
                    event[key] for key in ("generated_output_tokens", "output_tokens", "discarded_tail_tokens")
                )
                finished = event["finished_len"]
                if (
                    event["schema"] != "aisimulate.glm-rsi.output-ready.v2"
                    or any(type(n) is not int or n < 0 for n in (generated, emitted, discarded))
                    or generated != emitted + discarded
                    or emitted != (generated if finished is None else min(generated, finished))
                ):
                    raise ValueError("Output-ready counters fail explicit conservation")
                ready.setdefault(event["rid"], []).append(event)
        if mode == "instrumented-mtp" and set(ready) != expected:
            raise ValueError("Instrumented arm loses a server-ready identity")
        for repeat in range(3):
            name = f"{mode}-r{repeat}-c131072-l8"
            ids = [f"rsi-submit-r{repeat}-c131072-o128-l8-i{lane}" for lane in range(8)]
            origin = min(int(measured[rid]["start_ns"]) for rid in ids)
            reference = {}
            for rid in ids:
                row = measured[rid]
                first, tpot = interval(row["events"])
                dispatch = row["dispatch"]
                if (
                    row["status"] != "completed"
                    or dispatch["rid"] != rid
                    or dispatch["input_tokens"] != 131072
                    or row["input_tokens"] != 131072
                    or row["events"][-1]["meta"].get("prompt_tokens") != 131072
                    or row["events"][-1]["meta"].get("cached_tokens") != 0
                    or not int(row["prepared_at_ns"])
                    <= origin
                    <= int(row["start_ns"])
                    <= int(dispatch["at_ns"])
                    <= first
                ):
                    raise ValueError("Measured preparation, clock, input or cold-cache mismatch")
                reference[rid] = {
                    "ttft": (first - int(row["start_ns"])) / 1e6,
                    "tpot": tpot,
                    "first": first,
                    "server_tpot": interval(ready[rid])[1] if rid in ready else None,
                }
            for clock, directory in (
                ("client_start", args.client_replay),
                ("server_submission", args.submission_replay),
            ):
                prediction_origin = min(
                    int(measured[rid]["start_ns"] if clock == "client_start" else measured[rid]["dispatch"]["at_ns"])
                    for rid in ids
                )
                path = directory / name / "native.json"
                native = read(path)
                predictions = native["telemetry"]["requests"]
                predicted = {row["session_id"]: row for row in predictions}
                if set(predicted) != set(ids) or len(predictions) != 8:
                    raise ValueError("Native cohort loses or duplicates a planned identity")
                values = {"ttft": [], "tpot": [], "server_tpot": []}
                for rid in ids:
                    p, r = predicted[rid], reference[rid]
                    row = measured[rid]
                    arrival = int(row["start_ns"] if clock == "client_start" else row["dispatch"]["at_ns"])
                    ttft = p["first_token_ms"] - (int(row["start_ns"]) - prediction_origin) / 1e6
                    if (
                        p["terminal_status"] != "completed"
                        or p["input_length"] != 131072
                        or p["output_length"] != 128
                        or p["first_admission_reused_input_tokens"] != 0
                        or abs(p["arrival_time_ms"] - (arrival - prediction_origin) / 1e6) > 1e-6
                    ):
                        raise ValueError("Native completion, input/output, cache or arrival mismatch")
                    values["ttft"].append((ttft, r["ttft"]))
                    pred_tpot = (p["last_token_ms"] - p["first_token_ms"]) / 127
                    if abs(pred_tpot - p["itl_ms"]) > 1e-6:
                        raise ValueError("Native TPOT arithmetic differs from recorded boundaries")
                    values["tpot"].append((pred_tpot, r["tpot"]))
                    if r["server_tpot"] is not None:
                        values["server_tpot"].append((pred_tpot, r["server_tpot"]))
                observed_order = sorted(ids, key=lambda rid: reference[rid]["first"])
                predicted_order = sorted(ids, key=lambda rid: predicted[rid]["first_token_ms"])
                metrics = {metric: errors(pairs) if pairs else None for metric, pairs in values.items()}
                for metric, pairs in values.items():
                    pooled.setdefault(clock, {}).setdefault(metric, []).extend(pairs)
                cases.append(
                    {
                        "case": name,
                        "arrival_clock": clock,
                        "metrics": metrics,
                        "first_output_position_mismatches": sum(
                            a != b for a, b in zip(observed_order, predicted_order, strict=True)
                        ),
                        "mean_absolute_rank_difference": statistics.mean(
                            abs(observed_order.index(rid) - predicted_order.index(rid)) for rid in ids
                        ),
                        "source_requests_sha256": hashlib.sha256(raw).hexdigest(),
                        "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    result = {
        "schema": "aisimulate.glm-rsi.submission-audit.v1",
        "scope": "Finite native controls, not AgentX headline scores",
        "planned_measured_requests": 48,
        "retained_requests_per_clock": 48,
        "cases": cases,
        "pooled": {
            clock: {metric: errors(pairs) for metric, pairs in metrics.items()} for clock, metrics in pooled.items()
        },
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
