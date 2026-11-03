#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit synthetic server-ready/client-receipt observations, never headline MAPE."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import pairwise
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
            visible_mean_interval_ms=(final - first) / 1e6 / (target - 1) if target > 1 else None,
            interval_scope="captured content/reasoning frames only; legacy probe did not retain tool-only boundaries",
            per_token_alignment_unavailable="chat frames do not contain per-frame token counts",
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
        runtime_path = directory / "resolved-server.json"
        settings = json.loads(runtime_path.read_bytes()) if runtime_path.exists() else {}
        cache_boundary_verified = (
            settings.get("speculative_algorithm") == "EAGLE"
            and settings.get("page_size") == 64
            and settings.get("disable_radix_cache") is False
        )
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
                        # EAGLE counts bigrams (N-1), then page-aligns. A one-
                        # output-token seed commits its raw input, so an aligned
                        # seed loses one page in this verified runtime.
                        expected_reuse = ((row["prefix_tokens"] - 1) // 64) * 64
                        pair["expected_reused_tokens"] = expected_reuse
                        if not cache_boundary_verified:
                            pair["reasons"].append("missing or unsupported cache-boundary runtime receipt")
                        elif row.get("cached_tokens") != expected_reuse:
                            pair["reasons"].append("cache receipt missing or differs from EAGLE bigram/page boundary")
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
        "schema": "aisimulate.glm-rsi.frontend-audit.v2",
        "qualification_revision": "EAGLE bigram N-1 then page alignment; initial full-seed assumption superseded",
        "scope": "paired synthetic diagnostics; no headline MAPE",
        "modes": result,
    }


def blocking_comparison(victim, actor, server_events, stages):
    client = span(victim["events"], victim["output_tokens"])
    server = span(server_events, victim["output_tokens"])
    result = {"client": client, "server": server, "reasons": []}
    if "unavailable" not in server and max(row["output_tokens"] for row in server_events) > victim["output_tokens"]:
        result["reasons"].append("rank-zero counter exceeds delivered output length")
    if "unavailable" in client or "unavailable" in server:
        result["reasons"].append("incomplete client or rank-zero output counters")
    else:
        if server["first_ns"] > client["first_ns"] or server["final_ns"] > client["final_ns"]:
            result["reasons"].append("output receipt precedes output-ready clock")
        result["client_minus_ready_first_ms"] = (client["first_ns"] - server["first_ns"]) / 1e6
    if actor is not None:
        matches = [
            row
            for row in stages
            if row["input_sha256"] == (actor.get("input_sha256") or actor["expected_input_sha256"])
            and int(actor["start_ns"]) <= int(row["start_ns"])
            and int(row["end_ns"]) <= int(actor["body_received_ns"] if "body_received_ns" in actor else actor["end_ns"])
        ]
        if len(matches) != 1:
            result["reasons"].append("CPU stage absent or ambiguous")
        elif "unavailable" not in client and "unavailable" not in server:
            stage = matches[0]
            if stage["pid"] != victim["dispatch"]["pid"]:
                result["reasons"].append("CPU stage and scheduler submission use different API processes")
            result["cpu_stage"] = stage
            begin, end = int(stage["start_ns"]), int(stage["end_ns"])
            result["fully_buffered_during_cpu_stage"] = (
                begin <= server["first_ns"] <= server["final_ns"] <= end <= client["first_ns"]
            )
            result["cpu_finished_before_first_ready"] = end < server["first_ns"]
    result["qualified"] = not result["reasons"]
    return result


def analyze_blocking(root, coding=False):
    modes = {}
    expected = {
        f"rsi-block-c{size}-r{repeat}-{kind}"
        for size in ((8192,) if coding else (8192, 131072))
        for repeat in range(3)
        for kind in ("none", "short", "long")
    }
    expected |= {rid + "-actor" for rid in expected if not rid.endswith("-none")}
    expected |= {"rsi-block-preparation-short", "rsi-block-preparation-long"}
    for mode in ("unprofiled-mtp", "instrumented-mtp"):
        directory = root / mode
        path = directory / "client/requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        counts = Counter(row["rid"] for row in rows)
        index = {row["rid"]: row for row in rows}
        ready = defaultdict(list)
        for path in directory.glob("scheduler-*.jsonl"):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event["rank"] == 0:
                    ready[event["rid"]].append(event)
        stages = [
            {**json.loads(line), "pid": int(path.stem.rsplit("-", 1)[1])}
            for path in directory.glob("frontend-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        cases = []
        for victim in rows:
            if victim["role"] != "gpu_victim":
                continue
            rid = victim["rid"]
            actor = index.get(rid + "-actor")
            reasons = []
            if victim["status"] != "completed" or victim["input_tokens"] != victim["expected_input_tokens"]:
                reasons.append("native request incomplete or input length differs")
            if victim["cached_tokens"] != 0:
                reasons.append("cold cache receipt missing or nonzero")
            dispatch = victim.get("dispatch")
            client = span(victim["events"], victim["output_tokens"])
            if not dispatch or "unavailable" in client:
                reasons.append("submission or client counters unavailable")
            elif not (int(victim["start_ns"]) <= int(dispatch["at_ns"]) <= client["first_ns"]):
                reasons.append("submission clock outside native request")
            elif dispatch["input_tokens"] != victim["input_tokens"]:
                reasons.append("submission input length differs")
            if victim["actor"] != "none":
                if actor is None or actor["status"] != "completed":
                    reasons.append("CPU actor missing or failed")
                elif actor["input_tokens"] != actor["expected_input_tokens"] or (
                    not coding and actor["input_sha256"] != actor["expected_input_sha256"]
                ):
                    reasons.append("CPU tokenization identity differs")
                elif actor["pid"] == victim["pid"]:
                    reasons.append("CPU actor shares native client process")
                elif dispatch and int(actor["start_ns"]) < int(dispatch["at_ns"]):
                    reasons.append("CPU actor precedes scheduler submission")
            case = {
                "rid": rid,
                "context_label": victim["context_label"],
                "actor": victim["actor"],
                "repeat": victim["repeat"],
                "client": client,
                "reasons": reasons,
            }
            if mode == "instrumented-mtp" and not reasons:
                comparison = blocking_comparison(victim, actor, ready[rid], stages)
                case["comparison"] = comparison
                reasons.extend(comparison["reasons"])
            case["qualified"] = not reasons
            cases.append(case)
        modes[mode] = {
            "planned": len(expected),
            "recorded": len(rows),
            "unrecorded_or_unattempted": sorted(expected - set(counts)),
            "unexpected": sorted(set(counts) - expected),
            "duplicates": sorted(rid for rid, n in counts.items() if n != 1),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "qualified_victims": sum(case["qualified"] for case in cases),
            "cases": cases,
            "collector_completed": (directory / "completion.json").exists(),
        }
    return {
        "schema": "aisimulate.glm-rsi.coding-blocking-audit.v1" if coding else "aisimulate.glm-rsi.blocking-audit.v1",
        "modes": modes,
        "scope": (
            "Unmarked coding-chat actor also executes GPU work; qualify buffering only inside its synchronous CPU stage"
            if coding
            else "synthetic CPU-only interference; no historical MAPE or timing coefficient"
        ),
        "collector_completed": (root / "completion.json").exists(),
        "exposure_rule": "All ready outputs fall within CPU stage; first client receipt follows stage end.",
    }


def forward_request(row):
    """Audit numerical counters independently of the collection qualifier."""
    result = request_metrics(row, [])
    events = row["events"]
    context = row["context"]
    role = row["role"]
    expected_output = 128 if role == "profile" else (16 if context == 512 else 1)
    reasons = []
    if result.get("unavailable"):
        reasons.append(result["unavailable"])
    if (context not in (512, 8192, 393216) or role not in ("profile", "preparation")
            or row["input_tokens"] != context or row["requested_output_tokens"] != expected_output):
        reasons.append("declared context/output differs from finite matrix")
    meta = events[-1].get("meta", {}) if events else {}
    cached = meta.get("cached_tokens")
    if (meta.get("prompt_tokens") != context or type(cached) is not int
            or not 0 <= cached <= context):
        reasons.append("native prompt/cache receipt unavailable or invalid")
    elif context == 393216 and role == "profile":
        if cached % 64 or cached < context - 64:
            reasons.append("seeded long prefix not retained within one physical page")
    elif cached != 0:
        reasons.append("cold/preparation request unexpectedly reused cache")
    result.update(context=context, arm=row["arm"], cached_tokens=cached, reasons=reasons,
                  qualified=not reasons)
    return result


def analyze_forward(root, profiler_only=False):
    modes = {}
    matrix = (("unprofiled-mtp", ("before", "after")),
              ("instrumented-mtp", ("before", "profiled", "after")))
    for mode, arms in matrix[1:] if profiler_only else matrix:
        directory = root / mode
        rows_path = directory / "client/requests.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
        expected = {"rsi-profile-warmup": (512, "preparation", "warmup")}
        for context in (8192, 393216):
            for arm in arms:
                expected[f"rsi-profile-c{context}-{arm}"] = (context, "profile", arm)
                if context == 393216:
                    expected[f"rsi-profile-seed-{arm}"] = (context, "preparation", arm)
        counts = Counter(row["rid"] for row in rows)
        cases = [forward_request(row) for row in rows]
        reasons = []
        if any(expected.get(row["rid"]) != (row["context"], row["role"], row["arm"]) for row in rows):
            reasons.append("request identity/role/context does not match finite matrix")
        if any(int(a["end_ns"]) > int(b["start_ns"]) for a, b in pairwise(rows)):
            reasons.append("single-lane requests overlap or are recorded out of order")
        caches = {case["cached_tokens"] for case in cases
                  if case["context"] == 393216 and case["role"] == "profile"}
        if len(caches) != 1:
            reasons.append("long-prefix cache receipts differ or are missing")
        manifest_path = directory / "client/manifest.json"
        manifest = json.loads(manifest_path.read_bytes()) if manifest_path.exists() else {}
        if manifest.get("status") != "completed" or not (directory / "completion.json").exists():
            reasons.append("finite collector did not complete")
        if (manifest.get("planned_requests") != len(expected)
                or manifest.get("attempted_requests") != len(rows)
                or manifest.get("qualified_requests") != sum(case["qualified"] for case in cases)
                or manifest.get("unattempted_requests") != len(expected) - len(rows)):
            reasons.append("collector manifest does not reconcile with independent accounting")
        captures = manifest.get("capture_ranges", [])
        if mode == "instrumented-mtp":
            profiled = [row for row in rows if row["arm"] == "profiled" and row["role"] == "profile"]
            if len(captures) != 4 or len(profiled) != 2:
                reasons.append("two bounded capture ranges unavailable")
            else:
                for i, row in enumerate(profiled):
                    start, stop = captures[2 * i:2 * i + 2]
                    times = [int(start["begin_ns"]), int(start["end_ns"]), int(row["start_ns"]),
                             int(row["end_ns"]), int(stop["begin_ns"]), int(stop["end_ns"])]
                    if times != sorted(times) or start["start"] is not True or stop["start"] is not False:
                        reasons.append("capture API boundary does not enclose measured request")
        elif captures:
            reasons.append("unprofiled control unexpectedly enabled capture")
        if set(counts) != set(expected) or any(n != 1 for n in counts.values()):
            reasons.append("request identity coverage is incomplete or duplicated")
        modes[mode] = {"planned": len(expected), "recorded": len(rows),
                       "status_counts": dict(Counter(row["status"] for row in rows)),
                       "collector_completed": (directory / "completion.json").exists(),
                       "missing": sorted(set(expected) - set(counts)), "cases": cases, "reasons": reasons,
                       "qualified": not reasons and all(case["qualified"] for case in cases)}
    return {"schema": "aisimulate.glm-rsi.forward-controls-audit.v1", "modes": modes,
            "profiler_only": profiler_only,
            "scope": "Finite synthetic controls only; no historical AgentX accuracy score.",
            "gpu_trace_qualification": "Separate GPU-projected rank/kernel/shape audit required.",
            "collector_completed": (root / "completion.json").exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("timing", "frontend", "blocking", "coding-blocking", "forward-profile",
                                           "forward-torch"),
                        default="timing")
    args = parser.parse_args()
    result = {
        "frontend": analyze_frontend,
        "blocking": analyze_blocking,
        "coding-blocking": lambda root: analyze_blocking(root, coding=True),
        "timing": analyze,
        "forward-profile": analyze_forward,
        "forward-torch": lambda root: analyze_forward(root, profiler_only=True),
    }[args.suite](args.root)
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
