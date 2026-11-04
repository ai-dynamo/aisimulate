#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export baseline/shared/model coverage inventories without editing collector YAML.

The shared scenario reconstructs #664's broad history policy on the same #676
planner/model population. It is an audit comparison, never a collection default.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG_ROOT))

SYSTEMS = {
    "a100_sxm": 80,
    "l40s": 89,
    "h100_sxm": 90,
    "h200_sxm": 90,
    "b200_sxm": 100,
    "gb200": 100,
    "b300_sxm": 103,
    "gb300": 103,
}
OPS = (
    "attention_context",
    "attention_generation",
    "mla_context",
    "mla_generation",
    "mla_context_module",
    "mla_generation_module",
    "dsa_context_module",
    "dsa_generation_module",
    "dsa_context_module_skip_indexer",
    "dsa_generation_module_skip_indexer",
    "msa_context_module",
    "msa_generation_module",
    "moe",
    "kda",
)


@contextmanager
def shared_policy():
    """An explicit in-memory policy fixture, with the original planner restored."""
    generator = importlib.import_module("collector.case_generator")
    plans = importlib.import_module("collector.op_plans")
    original_data = generator._load_base_cases_data()
    data = copy.deepcopy(original_data)
    context = data["all_frameworks_op_cases"]["attention_context"]["cases"][0]
    context["sequence_lengths"] = sorted(
        set(context["sequence_lengths"]) | {24576, 32768, 40960, 65536, 131072, 524288, 1048576}
    )
    context["max_tokens_self_attention"] = 1048576
    context["max_tokens_grouped_query_attention"] = 1048576
    generation = data["all_frameworks_op_cases"]["attention_generation"]["cases"][0]
    generation["sequence_lengths"] = sorted(set(generation["sequence_lengths"]) | {262144, 524288, 1048576})
    generation["max_mha_tokens_per_step"] = generation["max_xqa_tokens_per_step"] = 16777216
    module = data["common_case_values"]["mla_module_plan"]
    module["context"]["chunk_prefill_size"] = 32768
    module["context"]["long_prefix_sequence_lengths"] = sorted(
        set(module["context"]["long_prefix_sequence_lengths"]) | {32768}
    )
    module["generation_sequence_lengths"] = sorted(
        set(module["generation_sequence_lengths"]) | {262144, 524288, 786432, 917504, 1048575}
    )
    module["generation"]["max_tokens"] = 37486592
    original_loader = generator._load_base_cases_data
    original_points = plans._attention_points

    def points(sweep, phase, heads, kv_heads, dim):
        if phase == "context":
            return original_points(sweep, phase, heads, kv_heads, dim)
        return [
            (batch, length - 1)
            for batch in sweep["batch_sizes"]
            for length in sweep["sequence_lengths"]
            if batch * length <= 16777216 and batch * length * kv_heads * dim * 2 < 2147483647
        ]

    generator._load_base_cases_data = lambda: data
    plans._attention_points = points
    try:
        yield
    finally:
        generator._load_base_cases_data = original_loader
        plans._attention_points = original_points


def export(backend: str, sm: int, output: Path, scenario: str) -> dict:
    plans = importlib.import_module("collector.op_plans")
    registry = importlib.import_module(f"collector.{backend}.registry").REGISTRY
    registered = {entry.op for entry in registry}
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for op in OPS:
        if op not in plans.OP_PLANS or op not in registered:
            continue
        with_context = {"include_long_history": True} if scenario == "model" else {}
        planned = plans.plan(op, backend, sm, **with_context)
        spec = plans.OP_PLANS[op]
        path = output / f"{backend}_sm{sm}_{op}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow((*spec.case_type._fields, "dispatch"))
            dispatches = planned.dispatches or [("",)] * len(planned.cases)
            writer.writerows(
                (*case, runner) for case, runners in zip(planned.cases, dispatches, strict=True) for runner in runners
            )
        summaries[op] = {
            "cases": len(planned.cases),
            "runs": sum(map(len, dispatches)),
            "groups": len({tuple(getattr(case, field) for field in spec.group_fields) for case in planned.cases}),
            "table": path.name,
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vllm", "sglang", "trtllm"), required=True)
    parser.add_argument("--system", choices=tuple(SYSTEMS), required=True)
    parser.add_argument("--scenario", choices=("baseline", "shared", "model"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scenario == "shared":
        with shared_policy():
            summary = export(args.backend, SYSTEMS[args.system], args.output, args.scenario)
    else:
        summary = export(args.backend, SYSTEMS[args.system], args.output, args.scenario)
    sha = subprocess.check_output(["git", "-C", str(PKG_ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = {
        "collector_commit": sha,
        "system": args.system,
        "backend": args.backend,
        "scenario": args.scenario,
        "scope": "planned cases; runtime failures/memory refusals not predicted",
        "ops": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
