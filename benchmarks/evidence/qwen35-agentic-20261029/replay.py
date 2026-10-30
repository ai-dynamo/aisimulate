#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay collected Qwen token chains through the public AISim engine API.

--functional-only uses fixed forward-pass timing solely for the cache gate.
Measured timing predictions require --engine-config with the full estimator
identity. Warmup and measured requests share one native engine/cache lifetime.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from aisimulate import (
    CorePredictionConfig,
    EngineReplayRunnerFactory,
    ReplayOutputRequirements,
)
from aisimulate.compiler import prediction_to_replay_spec

from cache_probe import SCHEMA, digest, make_receipt


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["phase"] != "kernel_warmup"]
    if not rows or len({row["cell_id"] for row in rows}) != 1:
        raise ValueError("one replay must contain exactly one nonempty cell")
    ids = [row["request_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate collected request IDs")
    for row in rows:
        if row.get("status") != "completed":
            raise ValueError(f"request {row['request_id']} did not complete")
        if row["phase"] not in ("warmup", "measured"):
            raise ValueError("unexpected request phase")
        for name in ("input_token_ids", "output_token_ids"):
            values = row[name]
            if not values or any(
                type(token) is not int or not 0 <= token < 2**32 for token in values
            ):
                raise ValueError(f"{name} must contain exact unsigned token IDs")
        delay = row.get("think_time_ms", 0)
        if not isinstance(delay, (float, int)) or not math.isfinite(delay) or delay < 0:
            raise ValueError("think time must be finite and nonnegative")
    return rows


def materialize(
    rows: list[dict[str, Any]], model: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use unit-size trace blocks and a bijection to retain all token equality.

    Agentic Mooncake interns prompt hash IDs in sorted request-ID order. Apply
    that same bijection to generated tokens so a later full prompt exactly
    matches the cached prompt-plus-output sequence. No text retokenization.
    """
    ordered = sorted(rows, key=lambda row: row["request_id"])
    token_map: dict[int, int] = {}
    for row in ordered:
        for token in row["input_token_ids"]:
            token_map.setdefault(token, len(token_map))
    for row in ordered:
        for token in row["output_token_ids"]:
            token_map.setdefault(token, len(token_map))
    if len(token_map) >= 2**32:
        raise ValueError("too many distinct token IDs")
    ids = {row["request_id"] for row in rows}
    warmup_ids = {row["request_id"] for row in rows if row["phase"] == "warmup"}
    graph = []
    metadata = {}
    for row in ordered:
        parents = set(row.get("dependency_ids", []))
        if row.get("parent_request_id"):
            parents.add(row["parent_request_id"])
        # Every measured lane starts after the same warmup barrier.
        if row["phase"] == "measured" and row["turn_index"] == 0:
            parents.update(warmup_ids)
        if not parents <= ids or row["request_id"] in parents:
            raise ValueError("request has an unknown or self dependency")
        graph.append(
            {
                "request_id": row["request_id"],
                "play_id": row["cell_id"],
                "session_id": f"{row['cell_id']}/lane-{row['lane_id']}",
                "model": model,
                "input_length": len(row["input_token_ids"]),
                "output_length": len(row["output_token_ids"]),
                "hash_ids": row["input_token_ids"],
                "output_token_ids": [
                    token_map[token] for token in row["output_token_ids"]
                ],
                "not_before_ms": 0,
                "dependencies": [
                    {
                        "request_id": parent,
                        "trigger": "completion",
                        "delay_ms": row.get("think_time_ms", 0),
                        "relation": "sequence",
                    }
                    for parent in sorted(parents)
                ],
            }
        )
        metadata[row["request_id"]] = {
            key: row[key] for key in ("lane_id", "turn_index", "phase", "cell_id")
        }
    source_digest = digest(rows)
    header = {
        "schema": "dynamo.agentic_mooncake",
        "version": 2,
        "block_size": 1,
        "hash_id_scope": "local",
        "source": {"format": "qwen35-agentic-request/v1", "digest": source_digest},
    }
    launch_spreads = {}
    for phase in ("warmup", "measured"):
        launches = [
            row["timing"]["dispatch_ms"]
            for row in rows
            if row["phase"] == phase
            and (phase == "warmup" or row["turn_index"] == 0)
            and "timing" in row
        ]
        launch_spreads[phase] = max(launches) - min(launches) if launches else None
    evidence = {
        "input_digest": source_digest,
        "token_map_digest": digest(token_map),
        "token_mapping": "bijection; unit trace blocks; generated tokens use native sorted-node interning",
        "observed_phase_launch_spread_ms": launch_spreads,
        "launch_timing_boundary": "native roots and post-warmup lanes release together; HTTP dispatch spread is recorded, not fitted",
        "requests": metadata,
    }
    return [header, *graph], evidence


def build_config(
    receipt: dict[str, Any],
    trace: Path,
    engine_config: dict[str, Any] | None,
    functional_only: bool,
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported cache geometry receipt")
    validated = make_receipt(receipt["workers"])
    if (
        validated["geometry_id"] != receipt["geometry_id"]
        or validated["manual_kv_cache"] != receipt["manual_kv_cache"]
    ):
        raise ValueError(
            "cache geometry receipt does not match its worker observations"
        )
    worker_receipt = receipt["workers"][0]
    if not functional_only and engine_config is None:
        raise ValueError("timing replay requires an explicit engine configuration")
    config = (
        copy.deepcopy(engine_config)
        if engine_config
        else {
            "engine": {
                "mode": "aggregated",
                "backend": "vllm",
                "model": "Qwen/Qwen3.5-397B-A17B",
                "hardware": "b300_sxm",
                "workers": {"aggregated": {}},
            },
        }
    )
    engine = config["engine"]
    if (
        engine["mode"] != "aggregated"
        or engine["backend"] != "vllm"
        or engine.get("nextn", 0)
        or engine.get("speculation") is not None
    ):
        raise ValueError("replay supports only aggregated vLLM without speculation")
    if set(engine["workers"]) != {"aggregated"}:
        raise ValueError("expected exactly one aggregated worker role")
    worker = engine["workers"]["aggregated"]
    if worker.get("speculation") is not None:
        raise ValueError("speculation is outside the qualified baseline")
    tp = worker_receipt["parallelism"]["tensor_parallel_size"]
    ep = worker_receipt["parallelism"]["enable_expert_parallel"]
    expected_parallelism = {
        "replicas": 1,
        "tensor": tp,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1 if ep else tp,
        "moe_expert": tp if ep else 1,
    }
    existing_parallelism = worker.get("parallelism", {})
    if any(
        value != expected_parallelism.get(key, 1)
        for key, value in existing_parallelism.items()
    ):
        raise ValueError("engine topology differs from the observed vLLM topology")
    worker["parallelism"] = expected_parallelism
    worker["scheduler"] = {
        "max_batched_tokens": worker_receipt["scheduler"]["max_num_batched_tokens"],
        "max_sequences": worker_receipt["scheduler"]["max_num_seqs"],
        "async_scheduling": False,
    }
    worker["kv_cache"] = copy.deepcopy(receipt["manual_kv_cache"])
    engine["context_length"] = worker_receipt["model"]["max_model_len"]
    if functional_only:
        worker["timing"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    config["traffic"] = {
        "source": {
            "type": "trace",
            "paths": [str(trace.resolve())],
            "format": "agentic_mooncake",
            "block_size": 1,
        },
        "load": {"type": "trace_timestamps", "speedup": 1},
    }
    return config


def compare_cache(rows: list[dict[str, Any]], native: dict[str, Any]) -> dict[str, Any]:
    predicted = {row["request_id"]: row for row in native["per_request"]}
    comparisons = []
    for observed in rows:
        row = predicted.get(observed["request_id"])
        silicon = observed.get("cached_tokens")
        simulated = row.get("first_admission_reused_input_tokens") if row else None
        reasons = []
        if silicon is None:
            reasons.append("vLLM did not report per-request cached_tokens")
        if row is None or row.get("terminal_status") != "completed":
            reasons.append("AISim request did not complete")
        elif row["output_length"] != len(observed["output_token_ids"]):
            reasons.append("AISim output count differs from collected output")
        if row and row.get("admission_count", 1) != 1:
            reasons.append(
                "preemption requires a separate admission-history comparison"
            )
        if silicon != simulated:
            reasons.append("reused-token counts differ")
        comparisons.append(
            {
                "request_id": observed["request_id"],
                "phase": observed["phase"],
                "observed_cached_tokens": silicon,
                "predicted_cached_tokens": simulated,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
    return {
        "passed": len(predicted) == len(rows)
        and all(row["passed"] for row in comparisons),
        "scope": "per-request reuse, completion and output lengths; excludes timing accuracy and pressure/eviction parity",
        "comparisons": comparisons,
    }


def replay_case(
    requests_path: Path,
    receipt: dict[str, Any],
    engine_config: dict[str, Any] | None,
    output: Path,
    functional_only: bool,
    prepare_only: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    rows = load_requests(requests_path)
    model = (
        engine_config["engine"]["model"] if engine_config else "Qwen/Qwen3.5-397B-A17B"
    )
    trace, metadata = materialize(rows, model)
    trace_path = output / "agentic.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    config = build_config(receipt, trace_path, engine_config, functional_only)
    (output / "prediction-config.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata.update(geometry_id=receipt["geometry_id"], functional_only=functional_only)
    (output / "mapping.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if prepare_only:
        CorePredictionConfig.model_validate(config)
        return {
            "prepared": True,
            "passed": False,
            "reason": "native replay not executed",
        }
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    factory = EngineReplayRunnerFactory()
    factory.capabilities().require_compatible(spec)
    runner = factory.create(worker_id=0)
    try:
        report = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True, capture_per_request=True
            ),
        )
    finally:
        runner.close()
    native = report.metadata["native_report"]
    (output / "native-report.json").write_text(json.dumps(native, indent=2) + "\n")
    (output / "runner-metadata.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in report.metadata.items()
                if key != "native_report"
            },
            indent=2,
        )
        + "\n"
    )
    alignment = compare_cache(rows, native)
    alignment.update(
        geometry_id=receipt["geometry_id"],
        input_digest=metadata["input_digest"],
        functional_only=functional_only,
    )
    (output / "cache-gate.json").write_text(json.dumps(alignment, indent=2) + "\n")
    return alignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        type=Path,
        required=True,
        help="one requests.jsonl, or a directory containing cells",
    )
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument(
        "--engine-config",
        type=Path,
        help="JSON core configuration with the complete timing identity",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--functional-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.cache_receipt.read_text())
    engine_config = (
        json.loads(args.engine_config.read_text()) if args.engine_config else None
    )
    paths = (
        sorted(args.requests.rglob("requests.jsonl"))
        if args.requests.is_dir()
        else [args.requests]
    )
    if not paths:
        parser.error("no requests.jsonl files found")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for path in paths:
        name = (
            str(path.parent.relative_to(args.requests))
            if args.requests.is_dir()
            else path.parent.name
        )
        results[name] = replay_case(
            path,
            receipt,
            engine_config,
            args.output_dir / name,
            args.functional_only,
            args.prepare_only,
        )
    summary = {
        "passed": all(result["passed"] for result in results.values()),
        "cells": results,
        "geometry_id": receipt["geometry_id"],
        "functional_only": args.functional_only,
    }
    (args.output_dir / "cache-gate-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "cells": len(results),
                "output_dir": str(args.output_dir),
            }
        )
    )
    if not args.prepare_only and not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
