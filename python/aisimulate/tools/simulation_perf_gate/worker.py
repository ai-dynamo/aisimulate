# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One fresh process, one full replay, using this revision's public runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.simulation_perf_gate import PROTOCOL_VERSION, digest
from tools.simulation_perf_gate.contract import MODEL_FIELDS, check_finite, fields


def runner_factory(item: dict):
    if item["runner"] == "engine" and item["determinism"] == "canonical_v1":
        from aisimulate import EngineReplayRunnerFactory

        return EngineReplayRunnerFactory(determinism=item["determinism"])
    if item["runner"] == "dynamo" and item["determinism"] == "random":
        from dynamo.replay.simulation import DynamoReplayRunnerFactory

        return DynamoReplayRunnerFactory()
    raise ValueError("unsupported benchmark runner or determinism")


def portable_model_identity(model: dict, systems_root: Path) -> dict:
    paths = model.get("systems_paths", [])
    if not paths or any(Path(path).resolve() != systems_root.resolve() for path in paths):
        raise ValueError("benchmark requires this installation's packaged model data")
    return {**fields(model, MODEL_FIELDS, "/model_identity"), "systems_paths": ["package:aisimulate_core/systems"]}


def run(request: dict) -> dict:
    if type(request.get("protocol_version")) is not int or request["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("incompatible simulation-performance protocol")
    item = request["case"]
    config = deepcopy(item["config"])
    if request["phase"] != "measure":
        raise ValueError("unknown benchmark phase")
    import aisimulate_core
    from aisimulate import CorePredictionConfig, ReplayOutputRequirements
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec

    if item.get("fixture"):
        trace = Path(__file__).parent / item["fixture"]["path"]
        if hashlib.sha256(trace.read_bytes()).hexdigest() != item["fixture"]["sha256"]:
            raise ValueError("local trace does not match the controller's input hash")
        config["traffic"]["source"]["paths"] = [str(trace)]
    adapters = {}
    if "router" in item:
        adapters["dynamo.router"] = AdapterReplaySpec(
            runtime_hooks=(RuntimeHookSpec("dynamo.router", "placement_policy", 1, item["router"]),)
        )
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config), adapter_specs=adapters)
    identity = {}
    provenance = {}
    for role, field in (
        ("aggregated", "agg_engine_args"),
        ("prefill", "prefill_engine_args"),
        ("decode", "decode_engine_args"),
    ):
        args = getattr(spec.backend_deployment, field)
        if args is not None:
            timing = args["timing_model"]
            model = timing["config"]
            if (
                timing.get("provider") != "aic"
                or model.get("estimation_mode") != "op_level"
                or model.get("fallback_policy") != "deny"
                or model.get("database_mode") != "SILICON"
                or model.get("enable_shared_layer") is not True
            ):
                raise ValueError(f"{role} did not retain the pinned real-model policy")
            identity[role] = portable_model_identity(model, Path(aisimulate_core.__file__).parent / "systems")
            provenance[role] = {**model, "provider": timing["provider"]}
    if not identity:
        raise ValueError("replay has no real forward model")
    runner = runner_factory(item).create(0)
    try:
        result = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=False),
        )
    finally:
        runner.close()
    report = result.metadata["native_report"]
    if item["runner"] == "dynamo":
        report = report["summary"]
    return {
        "status": "OK",
        "wall_time_ms": report.get("wall_time_ms"),
        "model_identity": identity,
        "model_provenance": provenance,
        "report": report,
    }


def main() -> int:
    response = {"protocol_version": PROTOCOL_VERSION}
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("worker request must be a JSON object")
        check_finite(request)
        item = request.get("case", {})
        if not isinstance(item, dict):
            raise ValueError("worker case must be a JSON object")
        response.update(
            case_id=item.get("case_id"),
            case_hash=digest(item),
            revision=request.get("revision"),
            phase=request.get("phase"),
        )
        with contextlib.redirect_stdout(sys.stderr):
            response.update(run(request))
    except Exception as error:
        response.update(status="ERROR", error={"type": type(error).__name__, "message": str(error)})
    print(json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
