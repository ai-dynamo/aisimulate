# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-batch prediction through the existing native estimator."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from typing import Any

from .config.cli import CorePredictionConfig
from .config.traffic import SyntheticSource

STATIC_MODES = ("static", "static_ctx", "static_gen")
_NOT_MODELED = ["traffic.load", "traffic.stop", "engine.workers.aggregated.scheduler", "evaluation"]


def static_prediction_kwargs(config: CorePredictionConfig, mode: str, batch_size: int) -> dict[str, Any]:
    """Validate the supported YAML projection before any estimator or output work."""
    if mode not in STATIC_MODES:
        raise ValueError(f"unsupported static estimate mode {mode!r}")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("static prediction requires a positive --batch-size")
    engine = config.engine
    source = config.traffic.source
    if engine.mode != "aggregated" or engine.workers.encoder is not None:
        raise ValueError("static prediction requires engine.mode=aggregated without an encoder pool")
    if not isinstance(source, SyntheticSource) or source.images is not None:
        raise ValueError("static prediction requires fixed-length synthetic text, without traces, sessions, or images")
    worker = engine.workers.aggregated
    assert worker is not None
    if worker.parallelism.replicas != 1 or worker.startup_seconds != 0:
        raise ValueError("static prediction estimates one worker; replicas must be 1 and startup_seconds must be 0")
    if worker.timing.type != "default":
        raise ValueError("static prediction requires default estimator timing, not fixed or polynomial timing")
    if worker.kv_cache.model_dump(exclude_defaults=True):
        raise ValueError("static prediction does not support non-default worker kv_cache settings")
    if engine.context_length != "max":
        raise ValueError("static prediction derives context from input/output tokens; omit engine.context_length")
    parallel = worker.parallelism
    sharded_moe = parallel.moe_tensor * parallel.moe_expert > 1
    return {
        "model_path": engine.model,
        "system_name": engine.hardware,
        "backend_name": engine.backend,
        "backend_version": engine.backend_version,
        "mode": mode,
        "batch_size": batch_size,
        "isl": source.input_tokens,
        "osl": source.output_tokens,
        "tp_size": parallel.tensor,
        "pp_size": parallel.pipeline,
        "attention_dp_size": parallel.attention_data,
        "moe_tp_size": parallel.moe_tensor if sharded_moe else None,
        "moe_ep_size": parallel.moe_expert if sharded_moe else None,
        "forward_model": worker.timing.forward_model,
        "nextn": 0,
        "prefix": 0,
    }


def _numbers(values: dict) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def run_static_prediction(kwargs: dict[str, Any], sections: tuple[str, ...]) -> dict[str, Any]:
    from aiconfigurator.cli.api import cli_estimate

    # Model/config diagnostics must not corrupt --format json stdout.
    with redirect_stdout(sys.stderr):
        result = cli_estimate(**kwargs)
    mode = kwargs["mode"]
    metrics = {"memory_gib": float(result.memory)}
    if mode != "static_gen":
        metrics["ttft_ms"] = float(result.ttft)
    if mode != "static_ctx":
        metrics["tpot_ms"] = float(result.tpot)
        if result.raw.get("generation_latency") is not None:
            metrics["generation_latency_ms"] = float(result.raw["generation_latency"])
    if mode == "static":
        metrics["request_latency_ms"] = float(result.request_latency)
    inputs = {**kwargs, "backend_version": result.backend_version}
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "prediction_kind": "static_estimate",
        "estimate_mode": mode,
        "inputs": inputs,
        "summary": metrics,
        "assumptions": {
            "scope": "one_worker_fixed_batch",
            "cached_prefix_tokens": 0,
            "serving_controls_not_modeled": list(_NOT_MODELED),
        },
        "warnings": [result.kv_cache_warning] if result.kv_cache_warning else [],
    }
    detail: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name in sections:
        if name == "summary":
            detail[name] = dict(metrics)
        elif result.summary is None:
            skipped[name] = "estimator did not return a static breakdown"
        elif name == "memory":
            memory = result.summary.get_memory()
            if memory:
                detail[name] = {
                    "scope": "per_rank_estimate",
                    "unit": "GiB",
                    "components": _numbers(memory),
                    "capacity_bytes": result.summary.get_mem_capacity_bytes(),
                }
            else:
                skipped[name] = "estimator did not return memory components"
        elif name == "time":
            phases = {}
            if mode != "static_gen":
                phases["prefill"] = _numbers(result.summary.get_context_latency_dict())
            if mode != "static_ctx":
                phases["decode"] = _numbers(result.summary.get_generation_latency_dict())
            if any(phases.values()):
                detail[name] = {"unit": "ms", "scope": "phase_total", "per_operation": phases}
            else:
                skipped[name] = "estimator did not return per-operation timing"
        else:
            raise ValueError(f"unsupported static detail section {name!r}")
    if sections:
        report["details"] = {"sections": detail, "skipped": skipped}
    # Fail on non-finite estimator output instead of producing non-standard JSON.
    json.dumps(report, allow_nan=False)
    return report


def format_static_prediction(report: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report, allow_nan=False, sort_keys=True)
    inputs = report["inputs"]
    lines = [
        f"Static prediction ({report['estimate_mode']})",
        f"  Model: {inputs['model_path']}",
        f"  Hardware: {inputs['system_name']}; backend: {inputs['backend_name']} {inputs['backend_version']}",
        f"  Fixed batch: {inputs['batch_size']}; input/output tokens: {inputs['isl']}/{inputs['osl']}",
        "  Scope: one worker; no request arrivals, queueing, scheduling, or SLA evaluation.",
    ]
    for key, value in report["summary"].items():
        lines.append(f"  {key}: {value:.3f}")
    for warning in report["warnings"]:
        lines.append(f"Warning: {warning}")
    details = report.get("details", {})
    for name, section in details.get("sections", {}).items():
        lines.extend([f"Detail: {name}", json.dumps(section, indent=2, sort_keys=True)])
    for name, reason in details.get("skipped", {}).items():
        lines.append(f"Skipped {name}: {reason}")
    return "\n".join(lines)
