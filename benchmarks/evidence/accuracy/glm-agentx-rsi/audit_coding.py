#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently audit E037 coding-chat interference without the timing analyzer."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def validate_counts(event):
    if event.get("schema") != "aisimulate.glm-rsi.output-ready.v2":
        raise ValueError("Explicit generated/emitted counters required")
    generated, emitted, discarded = (
        event[key] for key in ("generated_output_tokens", "output_tokens", "discarded_tail_tokens")
    )
    finished = event["finished_len"]
    if (
        any(type(n) is not int or n < 0 for n in (generated, emitted, discarded))
        or (finished is not None and (type(finished) is not int or finished < 0))
        or generated != emitted + discarded
        or emitted != (generated if finished is None else min(generated, finished))
    ):
        raise ValueError("Generated/emitted/discarded conservation fails")


def bounds(events, cap=16):
    if not events:
        raise ValueError("Missing numerical stream")
    previous_at, previous_count = -1, -1
    for event in events:
        at, count = int(event["at_ns"]), event["output_tokens"]
        if type(count) is not int or not 0 <= count <= cap or at < previous_at or count < previous_count:
            raise ValueError("Invalid time or emitted count")
        previous_at, previous_count = at, count
    if previous_count != cap:
        raise ValueError("Incomplete emitted stream")
    first = next(int(event["at_ns"]) for event in events if event["output_tokens"] > 0)
    last = next(int(event["at_ns"]) for event in events if event["output_tokens"] == cap)
    return first, last, (last - first) / 1e6 / (cap - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = read(args.preflight)
    if preflight["status"] != "passed" or preflight["openai_cache_usage"]["status"] != "passed":
        raise ValueError("Actual cache serializer qualification missing")
    runtime = read(args.root / "runtime.json")
    if runtime["sglang"] != "0.5.20" or runtime["model_revision"] != "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf":
        raise ValueError("Controlled runtime/model identity changed")
    for name, digest in preflight["source_sha256"].items():
        if runtime["source_files"][name] != digest:
            raise ValueError("Collector differs from CPU qualification")
    victims = {f"rsi-block-c8192-r{repeat}-{kind}" for repeat in range(3) for kind in ("none", "short", "long")}
    expected = victims | {rid + "-actor" for rid in victims if not rid.endswith("-none")}
    expected |= {"rsi-block-preparation-short", "rsi-block-preparation-long"}
    modes = {}
    for mode in ("unprofiled-mtp", "instrumented-mtp"):
        root = args.root / mode
        config = read(root / "resolved-server.json")
        if (
            config["enable_cache_report"] is not True
            or config["tp_size"] != 8
            or config["max_running_requests"] != 256
            or config["chunked_prefill_size"] != 8192
            or config["context_length"] != 1048576
        ):
            raise ValueError("Resolved measured settings changed")
        path = root / "client/requests.jsonl"
        observations = list(map(json.loads, path.read_text().splitlines()))
        rows = {row["rid"]: row for row in observations}
        if set(rows) != expected or len(observations) != 17:
            raise ValueError("Measured identity lost, duplicated or unexpected")
        dispatch = {}
        for row in observations:
            if (
                row["status"] != "completed"
                or row["input_tokens"] != row["expected_input_tokens"]
                or row["output_tokens"] != 16
                or row["cached_tokens"] != 0
            ):
                raise ValueError("Input/output/cache or completion qualification fails")
            receipt = read(root / "dispatch" / (hashlib.sha256(row["rid"].encode()).hexdigest() + ".json"))
            if (
                receipt["rid"] != row["rid"]
                or receipt["input_tokens"] != row["input_tokens"]
                or not int(row["start_ns"]) <= int(receipt["at_ns"]) <= int(row["end_ns"])
            ):
                raise ValueError("Submission clock or input identity differs")
            if row["role"] == "coding_chat_actor":
                name = "turn-0.json" if row["actor"] == "short" else "turn-128.json"
                expected_input = preflight["coding_inputs"]["files"][name]
                if (
                    row["endpoint"] != "chat"
                    or row["input_tokens"] != expected_input["input_tokens"]
                    or row["expected_input_sha256"] != expected_input["token_sha256"]
                ):
                    raise ValueError("Coding input differs from the qualified private primitive")
            dispatch[row["rid"]] = receipt
        preparation_end = max(int(row["end_ns"]) for row in observations if row["phase"] == "preparation")
        profile_start = min(int(row["start_ns"]) for row in observations if row["phase"] == "profile")
        if preparation_end > profile_start:
            raise ValueError("Preparation overlaps the measured finite cohort")
        ready = defaultdict(list)
        for p in root.glob("scheduler-*.jsonl"):
            for event in map(json.loads, p.read_text().splitlines()):
                validate_counts(event)
                if event["rank"] == 0:
                    ready[event["rid"]].append(event)
        stages = [
            dict(json.loads(line), pid=int(p.stem.rsplit("-", 1)[1]))
            for p in root.glob("frontend-*.jsonl")
            for line in p.read_text().splitlines()
        ]
        cases = []
        for rid in sorted(victims):
            row = rows[rid]
            first, last, client_ms = bounds(row["events"])
            if not int(dispatch[rid]["at_ns"]) <= first <= last <= int(row["end_ns"]):
                raise ValueError("Native client boundaries differ")
            case = {"rid": rid, "actor": row["actor"], "repeat": row["repeat"], "client_tpot_ms": client_ms}
            actor = rows.get(rid + "-actor")
            if actor is not None:
                receipt = dispatch[actor["rid"]]
                if actor["pid"] == row["pid"] or int(actor["start_ns"]) < int(dispatch[rid]["at_ns"]):
                    raise ValueError("Actor shares reader process or precedes trigger")
                case["actor_client_to_submission_ms"] = (int(receipt["at_ns"]) - int(actor["start_ns"])) / 1e6
            if mode == "instrumented-mtp":
                server_first, server_last, server_ms = bounds(ready[rid])
                if server_first > first or server_last > last:
                    raise ValueError("Receipt precedes output-ready event")
                case["server_ready_tpot_ms"] = server_ms
                if actor is not None:
                    matches = [
                        s
                        for s in stages
                        if s["input_sha256"] == actor["expected_input_sha256"]
                        and int(actor["start_ns"])
                        <= int(s["start_ns"])
                        <= int(s["end_ns"])
                        <= int(dispatch[actor["rid"]]["at_ns"])
                    ]
                    if len(matches) != 1:
                        raise ValueError("Actual frontend stage absent or ambiguous")
                    stage = matches[0]
                    if (
                        stage["pid"] != dispatch[rid]["pid"]
                        or stage["pid"] != dispatch[actor["rid"]]["pid"]
                        or stage["input_tokens"] != actor["input_tokens"]
                        or stage["template_cache_requested"] is not False
                    ):
                        raise ValueError("API process, token identity or template cache differs")
                    begin, end = int(stage["start_ns"]), int(stage["end_ns"])
                    case.update(
                        cpu_ms=(end - begin) / 1e6,
                        thread_cpu_ms=stage["thread_cpu_ms"],
                        fully_buffered=begin <= server_first <= server_last <= end <= first,
                        cpu_finished_before_ready=end < server_first,
                        actor_GPU_submission_after_victim_ready=server_last <= int(dispatch[actor["rid"]]["at_ns"]),
                        pre_render_ms=(begin - int(actor["start_ns"])) / 1e6,
                        post_render_to_submission_ms=(int(dispatch[actor["rid"]]["at_ns"]) - end) / 1e6,
                    )
            cases.append(case)
        if mode == "instrumented-mtp" and not expected <= set(ready):
            raise ValueError("A measured server-ready identity is missing")
        modes[mode] = {
            "retained": len(rows),
            "profile_victims": len(victims),
            "preparation_requests": 2,
            "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "auxiliary_server_ready_requests": sorted(set(ready) - expected),
            "cases": cases,
        }
    result = {
        "schema": "aisimulate.glm-rsi.coding-independent-audit.v1",
        "planned": 34,
        "retained": 34,
        "scope": "Finite coding controls; no historical request MAPE or model improvement claim",
        "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preflight_sha256": hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
        "modes": modes,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                mode: {"retained": value["retained"], "profile_victims": value["profile_victims"]}
                for mode, value in modes.items()
            }
        )
    )


if __name__ == "__main__":
    main()
