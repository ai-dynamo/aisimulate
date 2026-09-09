# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate both production consumers against independent routed-kernel timings.

Input contract and limits: docs/EXPERT_POPULARITY_CONSUMER.md. No synthetic
routes or probability-sampled timings are accepted as held-out observations.
This tool evaluates predictions; it does not fabricate missing GPU measurements.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.config import ModelConfig
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.moe_routing import select_profile
from aiconfigurator_core.sdk.operations import MoEAllToAll, MoEExpertCompute
from aiconfigurator_core.sdk.perf_database import get_database

REQUIRED_MODELS = ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "deepseek-ai/DeepSeek-R1", "zai-org/GLM-5.2")
COMPONENTS = ("dispatch", "combine", "compute", "total")


def validate_observation(record, root):
    """Check independence, immutable raw evidence, Top-K and timing completeness."""
    evidence = record["measurement"]
    if evidence["kind"] not in {"routing_replay", "native_decode"}:
        raise ValueError("Held-out evidence must be real routing replay or native decode, not synthetic")
    if record["phase"] == "decode" and evidence["kind"] != "native_decode":
        raise ValueError("Prefill routing replay is not decode accuracy evidence")
    for key in ("kernel_revision", "gpu_type", "driver", "timing_method", "placement"):
        if not evidence.get(key):
            raise ValueError(f"Missing timing condition: {key}")
    if evidence["placement"] != "contiguous_expert_id":
        raise ValueError("Only fixed contiguous placement is supported")
    training = set(record["bundle_workload_sha256s"])
    if not training or record["workload_sha256"] in training:
        raise ValueError("Workload must be disjoint from the bundle's training workloads")
    for digest in [*training, record["workload_sha256"], record["route_sha256"]]:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Evidence identities must be SHA-256 digests")
    raw = (root / record["route_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["route_sha256"]:
        raise ValueError("Held-out route checksum mismatch")
    routes = json.loads(raw)
    ranks = record["model_config"]["moe_ep_size"]
    if len(routes) != record["per_rank_tokens"] * ranks:
        raise ValueError("Held-out route token count mismatch")
    for route in routes:
        if len(route) != record["top_k"] or len(set(route)) != len(route):
            raise ValueError("Held-out routes must use distinct Top-K")
        if any(type(expert) is not int or not 0 <= expert < record["num_experts"] for expert in route):
            raise ValueError("Invalid held-out expert ID")
    timing = evidence["latency_ms"]
    for name in COMPONENTS[:3]:
        if not math.isfinite(timing[name]) or timing[name] <= 0:
            raise ValueError("All three component timings must be finite and positive")
    return {**timing, "total": sum(timing[name] for name in COMPONENTS[:3])}


def predict_pair(record):
    """Run the actual native LL/compute operators, not a fitted surrogate loss."""
    options = dict(record["model_config"])
    for key, enum in {
        "gemm_quant_mode": common.GEMMQuantMode,
        "moe_quant_mode": common.MoEQuantMode,
        "fmha_quant_mode": common.FMHAQuantMode,
        "kvcache_quant_mode": common.KVCacheQuantMode,
        "comm_quant_mode": common.CommQuantMode,
    }.items():
        if isinstance(options.get(key), str):
            options[key] = enum[options[key]]
    config = ModelConfig(
        **(
            options
            | {
                "moe_routing_mode": "power-law",
                "moe_power_law_alpha": None,
                "workload_distribution": None,
                "moe_comm_backend": {"context": "deepep_ll", "generation": "deepep_ll"},
            }
        )
    )
    identity = record["database"]
    model = get_model(record["model_id"], config, identity["backend"])
    phase = "context" if record["phase"] == "prefill" else "generation"
    selection = select_profile(
        model_id=record["model_id"],
        revision=record["revision"],
        num_layers=model._num_layers,
        num_experts=model._num_experts,
        top_k=model._topk,
        phase=record["phase"],
        backend="deepep_ll",
    )
    if not selection["layers"]:
        raise ValueError("Accuracy comparison requires a compatible packaged profile")
    if (record["num_experts"], record["top_k"]) != (model._num_experts, model._topk):
        raise ValueError("Held-out/model routing dimension mismatch")
    profile = next(p for p in selection["layers"] if p["layer_id"] == record["layer_id"])
    database = get_database(**identity)
    collected = {}

    def visit(spec):
        kind, fields = next(iter(spec.items()))
        if kind == "Overlap":
            for child in [*fields["group_a"], *fields["group_b"]]:
                visit(child)
        elif kind in {"MoeAllToAll", "MoeExpertCompute"}:
            component = fields["phase"] if kind == "MoeAllToAll" else "compute"
            if component in collected:
                raise ValueError("Accuracy harness requires one unambiguous operator per component")
            collected[component] = fields

    for op in getattr(model, f"{phase}_ops"):
        visit(json.loads(op._spec_json()))
    output = {"power-law": {}, "measured": {}}
    for component in COMPONENTS[:3]:
        fields = collected[component]
        for source in output:
            import aiconfigurator_core._aiconfigurator_core as native

            item = copy.deepcopy(fields)
            item["scale_factor"] = 1.0
            kind = "MoeExpertCompute" if component == "compute" else "MoeAllToAll"
            cls = MoEExpertCompute if component == "compute" else MoEAllToAll
            if source == "measured":
                item["measured_routing"] = profile
                if component == "compute":
                    item["routing_attention_tp_size"] = config.tp_size * config.cp_size if phase == "context" else 1
            native_op = native.op_from_spec_json(json.dumps({kind: item}))
            args, kwargs = native_op.__getnewargs_ex__()
            op = cls(*args, **kwargs)
            tokens = record["per_rank_tokens"] * (config.tp_size * config.cp_size if phase == "context" else 1)
            output[source][component] = float(op._engine_query(database, x=tokens, is_context=phase == "context"))
    for values in output.values():
        values["total"] = sum(values.values())
    return output, selection["provenance"]


def summarize(rows, failures, required_models=REQUIRED_MODELS):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model_id"]].append(row)
    results = {}
    for model in sorted(set(required_models) | set(grouped)):
        samples = grouped[model]
        if not samples:
            results[model] = {"status": "INSUFFICIENT_EVIDENCE", "samples": 0}
            continue
        mape = {
            source: {
                component: 100
                * sum(abs(row[source][component] / row["actual"][component] - 1) for row in samples)
                / len(samples)
                for component in COMPONENTS
            }
            for source in ("power-law", "measured")
        }
        results[model] = {
            "samples": len(samples),
            "mape_percent": mape,
            "status": "PASS" if mape["measured"]["total"] <= mape["power-law"]["total"] else "REGRESSION",
        }
    overall = (
        {
            source: sum(abs(row[source]["total"] / row["actual"]["total"] - 1) for row in rows) / len(rows)
            for source in ("power-law", "measured")
        }
        if rows
        else None
    )
    passed = bool(rows) and not failures and all(value["status"] == "PASS" for value in results.values())
    passed = passed and overall["measured"] <= overall["power-law"]
    return {
        "status": "PASS" if passed else "NOT_READY",
        "models": results,
        "failures": failures,
        "overall_total_mape_percent": {k: v * 100 for k, v in overall.items()} if overall else None,
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, help="JSON array of held-out observation records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, failures = [], []
    for index, record in enumerate(json.loads(args.observations.read_text())):
        try:
            actual = validate_observation(record, args.observations.parent)
            predictions, provenance = predict_pair(record)
            rows.append(
                {
                    "model_id": record["model_id"],
                    "layer_id": record["layer_id"],
                    "phase": record["phase"],
                    "route_sha256": record["route_sha256"],
                    "actual": actual,
                    **predictions,
                    "provenance": provenance,
                }
            )
        except Exception as error:
            # Continue reporting every model, but no failed sample is eligible
            # to turn the overall gate green. Keep raw messages private.
            failures.append({"sample": index, "model_id": record.get("model_id"), "error_type": type(error).__name__})
    report = summarize(rows, failures)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
