#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit synthetic server-ready/client-receipt observations, never headline MAPE."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MODES = ("unprofiled-mtp", "instrumented-mtp", "instrumented-plain")


def planned_ids():
    matrix = {
        f"rsi-r{repeat}-c{context}-o{length}-l{lanes}-i{lane}"
        for repeat in range(2)
        for context in (8192, 131072)
        for length in (16, 128)
        for lanes in (1, 8, 14)
        for lane in range(lanes)
    }
    chat = {
        f"rsi-chat-r{repeat}-c{context}-o{length}-l{lanes}-i{lane}"
        for repeat in range(2)
        for context, length, lanes in ((8192, 16, 1), (8192, 16, 14), (131072, 128, 1))
        for lane in range(lanes)
    }
    overlap = {f"rsi-overlap-{value}-{role}" for value in (False, True) for role in ("warm", "victim")}
    return matrix | chat | overlap | {"rsi-overlap-interferer"}


def span(events, target):
    """First positive counter to first attainment of delivered length; no EOF inflation."""
    if type(target) is not int or target <= 0:
        return {"unavailable": "invalid delivered output length"}
    if not events:
        return {"unavailable": "no counter observations"}
    counts = [row["output_tokens"] for row in events]
    times = [int(row["at_ns"]) for row in events]
    if any(type(n) is not int or n < 0 for n in counts):
        return {"unavailable": "invalid output counter"}
    if counts != sorted(counts) or times != sorted(times):
        return {"unavailable": "counter or clock moved backward"}
    positive = [row for row in events if row["output_tokens"] > 0]
    complete = [row for row in events if row["output_tokens"] >= target]
    if not positive or not complete:
        return {"unavailable": "delivered output length not covered"}
    first, final = int(positive[0]["at_ns"]), int(complete[0]["at_ns"])
    return {
        "first_ns": first,
        "final_ns": final,
        "span_ms": (final - first) / 1e6,
        "mean_interval_ms": (final - first) / 1e6 / (target - 1) if target > 1 else None,
        "interval_unavailable": "one delivered token" if target <= 1 else None,
        "first_ready_tokens": positive[0]["output_tokens"],
        "last_retained_tokens": counts[-1],
        "retained_excess_over_delivered": max(0, counts[-1] - target),
    }


def request_metrics(row, server):
    result = {"rid": row["rid"], "role": row.get("role", "synthetic_native"), "status": row["status"]}
    target = row["requested_output_tokens"]
    events = row["events"]
    if row["status"] != "completed":
        return dict(result, unavailable="client request did not complete with requested output length")
    times = [int(event["at_ns"]) for event in events]
    if times and (times != sorted(times) or times[0] < int(row["start_ns"]) or times[-1] > int(row["end_ns"])):
        return dict(result, unavailable="client clock ordering or request boundary invalid")
    if row.get("role") == "diagnostic_chat_frontend":
        if not events or row.get("output_tokens") != target:
            return dict(result, unavailable="chat content or exact usage unavailable")
        first, final = int(events[0]["at_ns"]), int(events[-1]["at_ns"])
        result.update(
            visible_ttft_ms=(first - int(row["start_ns"])) / 1e6,
            visible_span_ms=(final - first) / 1e6,
            tpot_unavailable="chat frames do not contain per-frame token counts; content/reasoning boundary only",
        )
    else:
        client = span(events, target)
        if "unavailable" in client:
            return dict(result, unavailable=client["unavailable"])
        if client["last_retained_tokens"] != target:
            return dict(result, unavailable="client final output count differs from requested length")
        result.update(client=client, ttft_ms=(client["first_ns"] - int(row["start_ns"])) / 1e6)
    rank0 = [event for event in server if event.get("rank") == 0]
    ready = span(rank0, target)
    result["server_ready"] = ready
    if "unavailable" not in ready:
        first_receipt = int(events[0]["at_ns"]) if "client" not in result else result["client"]["first_ns"]
        result["first_receipt_after_ready_ms"] = (first_receipt - ready["first_ns"]) / 1e6
        if "client" in result:
            result["final_receipt_after_ready_ms"] = (result["client"]["final_ns"] - ready["final_ns"]) / 1e6
        if any(result.get(k, 0) < 0 for k in ("first_receipt_after_ready_ms", "final_receipt_after_ready_ms")):
            result["comparison_unavailable"] = (
                "receipt precedes recorded readiness; boundary/clock qualification failed"
            )
    return result


def analyze(root):
    result = {}
    expected = planned_ids()
    for mode in MODES:
        directory = root / mode
        source = directory / "client/requests.jsonl"
        rows = [json.loads(line) for line in source.open()] if source.exists() else []
        counts = Counter(row["rid"] for row in rows)
        if any(n > 1 for n in counts.values()) or set(counts) - expected:
            raise ValueError("Duplicate or unexpected client identity")
        server = defaultdict(list)
        ranks = Counter()
        for path in directory.glob("scheduler-*.jsonl"):
            for line in path.open():
                event = json.loads(line)
                ranks[str(event.get("rank"))] += 1
                server[event["rid"]].append(event)
        for events in server.values():
            events.sort(key=lambda row: int(row["at_ns"]))
        metrics = [request_metrics(row, server[row["rid"]]) for row in rows]
        result[mode] = {
            "planned": len(expected),
            "recorded": len(rows),
            "unrecorded_or_unattempted": sorted(expected - set(counts)),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "server_event_ranks": dict(ranks),
            "collector_completed": (directory / "completion.json").exists(),
            "metrics": metrics,
        }
    return {
        "schema": "aisimulate.glm-rsi.timing-audit.v1",
        "scope": "synthetic diagnostics; no AgentX headline scoring",
        "modes": result,
    }


def analyze_frontend(root):
    """Qualify paired endpoint controls; never fit costs to benchmark errors."""
    result = {}
    expected = {
        f"rsi-frontend-c{context}-r{repeat}-{endpoint}{suffix}"
        for context in (8192, 131072, 393216)
        for repeat in range(3)
        for endpoint in ("native", "chat")
        for suffix in ("", "-warm")
    }
    for mode in MODES[:2]:
        directory = root / mode
        source = directory / "client/requests.jsonl"
        rows = [json.loads(line) for line in source.open()] if source.exists() else []
        counts = Counter(row["rid"] for row in rows)
        if any(n > 1 for n in counts.values()) or set(counts) - expected:
            raise ValueError("Duplicate or unexpected frontend identity")
        index = {row["rid"]: row for row in rows}
        stages = [json.loads(line) for path in directory.glob("frontend-*.jsonl") for line in path.open()]
        pairs = []
        for context in (8192, 131072, 393216):
            for repeat in range(3):
                identity = f"rsi-frontend-c{context}-r{repeat}"
                native, chat = (index.get(identity + "-" + endpoint) for endpoint in ("native", "chat"))
                pair = {"context_label": context, "repeat": repeat, "qualified": False, "reasons": []}
                if not native or not chat:
                    pair["reasons"].append("missing endpoint measurement")
                else:
                    for row in (native, chat):
                        warm = index.get(row["rid"] + "-warm")
                        if row["status"] != "completed" or not warm or warm["status"] != "completed":
                            pair["reasons"].append("incomplete measured request or preparation")
                        elif not warm["events"] or warm["events"][-1].get("meta", {}).get("prompt_tokens") != row.get(
                            "prefix_tokens"
                        ):
                            pair["reasons"].append("preparation input receipt differs")
                        if row.get("input_tokens") != row.get("expected_input_tokens"):
                            pair["reasons"].append("template input length differs")
                        if row.get("cached_tokens") != row.get("prefix_tokens"):
                            pair["reasons"].append("cache receipt missing or differs from seeded prefix")
                        if row["endpoint"] == "native" and "unavailable" in span(
                            row["events"], row["requested_output_tokens"]
                        ):
                            pair["reasons"].append("native output counters invalid")
                        times = [int(e["at_ns"]) for e in row["events"]]
                        if (
                            not times
                            or times != sorted(times)
                            or not int(row["start_ns"]) <= times[0] <= times[-1] <= int(row["end_ns"])
                        ):
                            pair["reasons"].append("invalid client boundaries")
                        elif row.get("output_tokens") != row["requested_output_tokens"]:
                            pair["reasons"].append("output length differs")
                        else:
                            pair[row["endpoint"] + "_ttft_ms"] = (times[0] - int(row["start_ns"])) / 1e6
                    if native.get("input_sha256") != chat.get("input_sha256"):
                        pair["reasons"].append("paired token identities differ")
                    matched = [
                        s
                        for s in stages
                        if int(chat["start_ns"]) <= int(s["start_ns"]) <= int(s["end_ns"]) <= int(chat["end_ns"])
                    ]
                    if mode == "instrumented-mtp":
                        if len(matched) != 1:
                            pair["reasons"].append("missing or ambiguous synchronous stage")
                        else:
                            stage = matched[0]
                            if (
                                stage["input_sha256"] != chat["input_sha256"]
                                or stage["input_tokens"] != chat["expected_input_tokens"]
                            ):
                                pair["reasons"].append("server token identities differ")
                            if stage.get("template_cache_requested") is not False:
                                pair["reasons"].append("template-cache policy differs")
                            if (
                                stage["service_ms"] < 0
                                or abs(stage["service_ms"] - (int(stage["end_ns"]) - int(stage["start_ns"])) / 1e6)
                                > 1e-9
                            ):
                                pair["reasons"].append("invalid synchronous stage units")
                            pair["synchronous_stage"] = stage
                    pair["qualified"] = not pair["reasons"]
                    if pair["qualified"]:
                        pair["chat_minus_native_ttft_ms"] = pair["chat_ttft_ms"] - pair["native_ttft_ms"]
                pairs.append(pair)
        result[mode] = {
            "planned": len(expected),
            "recorded": len(rows),
            "unrecorded_or_unattempted": sorted(expected - set(counts)),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "qualified_pairs": sum(pair["qualified"] for pair in pairs),
            "pairs": pairs,
            "collector_completed": (directory / "completion.json").exists(),
        }
    return {
        "schema": "aisimulate.glm-rsi.frontend-audit.v1",
        "scope": "paired synthetic diagnostics; no headline MAPE",
        "modes": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("timing", "frontend"), default="timing")
    args = parser.parse_args()
    result = analyze_frontend(args.root) if args.suite == "frontend" else analyze(args.root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: {f: v[f] for f in ("planned", "recorded", "status_counts", "collector_completed")}
                for k, v in result["modes"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
