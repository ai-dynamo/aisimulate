#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently qualify bounded native forward geometry; never score synchronized cadence."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_timing import analyze_forward


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify_rows(rows, device):
    contexts, context = defaultdict(list), None
    for ordinal, row in enumerate(rows):
        if (row["schema"] != "aisimulate.glm-rsi.forward-geometry.v1" or row["ordinal"] != ordinal
                or row["device"] != device or row["batch_size"] != 1
                or int(row["end_ns"]) < int(row["begin_ns"])):
            raise ValueError("Geometry identity, rank, order or measurement boundary is invalid")
        lengths = row["seq_lens"]
        if (not isinstance(lengths, list) or len(lengths) != 1
                or type(lengths[0]) is not int or not 0 < lengths[0] < 1_048_576):
            raise ValueError("Missing or invalid live device sequence length")
        count = row["input_token_count"]
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid input geometry")
        stage = row["stage"]
        if stage.startswith("step[EXTEND "):
            context = lengths[0]
            if context not in (8192, 393216) or context in contexts:
                raise ValueError("Unexpected or repeated profile context")
            expected = 8192 if context == 8192 else 64
            if count != expected:
                raise ValueError("Prefill geometry differs from native cache receipt")
            phase = "target_prefill"
        elif stage.startswith("step[VERIFY "):
            phase = "target_verify"
        elif stage.startswith("step[DRAFT "):
            phase = "draft_prefill"
        elif stage in ("draft", "draft_extend") and row["graph_entry"]:
            phase = stage
        else:
            raise ValueError("Unknown numerical forward phase")
        if context is None or lengths[0] < context or lengths[0] > context + 132:
            raise ValueError("Decode KV geometry exceeds the controlled output cohort")
        expected = {"target_verify": 4, "draft": 1, "draft_extend": 4}.get(phase)
        if expected is not None and (count != expected or row["num_tokens_per_req"] != expected):
            raise ValueError("Runtime token width differs from the pinned phase contract")
        positions = row["positions_bounded"]
        if positions is not None:
            if (not isinstance(positions, list) or len(positions) != count
                    or any(type(value) is not int or value < 0 for value in positions)
                    or positions != list(range(positions[0], positions[0] + count))):
                raise ValueError("Positions do not conserve consecutive request tokens")
        elif count <= 64:
            raise ValueError("Small forward is missing numerical position evidence")
        contexts[context].append({"ordinal": ordinal, "phase": phase, "tokens": count,
                                  "seq_lens": lengths, "positions": positions,
                                  "dsa_topk_seed_present": row["dsa_topk_seed_present"],
                                  "metadata_copy_ns": int(row["end_ns"]) - int(row["begin_ns"])})
    if set(contexts) != {8192, 393216}:
        raise ValueError("Incomplete numerical profile contexts")
    result = {}
    for context, values in contexts.items():
        counts = Counter(row["phase"] for row in values)
        if (counts["target_prefill"] != 1 or counts["draft_prefill"] != 1
                or any(counts[phase] < 10 for phase in ("target_verify", "draft", "draft_extend"))):
            raise ValueError("Incomplete phase geometry; do not select or discard executions")
        result[str(context)] = {"counts": dict(counts), "forwards": values}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Geometry audit output exists")
    requests = analyze_forward(args.controls, profiler_only=True)
    if not requests["collector_completed"] or not requests["modes"]["instrumented-mtp"]["qualified"]:
        raise ValueError("Finite native request campaign does not qualify")
    command_path = args.controls / "instrumented-mtp/command.json"
    command = json.loads(command_path.read_bytes())
    patch_path = args.controls / "forward-nvtx-instrumentation.json"
    patch = json.loads(patch_path.read_bytes())
    if command.get("geometry_diagnostic") is not True or patch.get("geometry_patch", {}).get(
        "original_sha256") != "8fc9285c458d746874c96da7b16bef2733509dea3d1170cbad9e787bcde67754":
        raise ValueError("Numerical instrumentation does not match the pinned source")
    root = args.controls / "instrumented-mtp/model-receipts"
    receipts = [json.loads(path.read_bytes()) for path in root.glob("model-*.json")]
    if len(receipts) != 16:
        raise ValueError("Missing loaded-model receipts")
    rank_pids, pids = {}, {}
    for rank in range(8):
        selected = [row for row in receipts if row["tp_rank"] == rank]
        if (len(selected) != 2 or {row["draft"] for row in selected} != {False, True}
                or any(row["tp_size"] != 8 for row in selected)
                or len({row["pid"] for row in selected}) != 1
                or len({row["cuda_device"] for row in selected}) != 1):
            raise ValueError("Loaded-model role/rank/device identities disagree")
        pid, device = selected[0]["pid"], selected[0]["cuda_device"]
        if pid in pids:
            raise ValueError("Duplicate rank PID")
        pids[pid] = device
        rank_pids[rank] = pid
    if set(pids.values()) != set(range(8)):
        raise ValueError("Missing eight-device numerical evidence")
    paths = sorted(root.glob("geometry-*.jsonl"))
    if {path.name for path in paths} != {f"geometry-{pid}.jsonl" for pid in pids}:
        raise ValueError("Missing or unexpected geometry artifact")
    result = {"schema": "aisimulate.glm-rsi.geometry-audit.v1", "ranks": {}, "source_sha256": {},
              "auditor_sha256": checksum(Path(__file__)), "request_audit": requests,
              "scope": "Synchronized geometry only. No unperturbed cadence, op-timing or E2E accuracy claim. "
                       "Executed rounds remain separate from per-request reported rounds."}
    for rank, pid in rank_pids.items():
        path = root / f"geometry-{pid}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if any(row["pid"] != pid for row in rows):
            raise ValueError("Geometry process identity changed")
        result["ranks"][str(rank)] = qualify_rows(rows, pids[pid])
        result["source_sha256"][str(path.relative_to(args.controls))] = checksum(path)
    for context in ("8192", "393216"):
        signature = lambda item: [(row["phase"], row["tokens"], row["seq_lens"], row["positions"])
                                  for row in item[context]["forwards"]]
        expected = signature(result["ranks"]["0"])
        if any(signature(value) != expected for value in result["ranks"].values()):
            raise ValueError("TP ranks disagree on live numerical geometry")
    paths = [command_path, patch_path, *root.glob("model-*.json")]
    paths += [args.controls / relative for relative in (
        "runtime.json", "completion.json", "instrumented-mtp/client/requests.jsonl",
        "instrumented-mtp/client/manifest.json", "instrumented-mtp/resolved-server.json")]
    result["source_sha256"].update({str(path.relative_to(args.controls)): checksum(path) for path in paths})
    result["qualified_geometry"] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
