# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated exact-wheel replay process; all outputs remain private until scored."""

import argparse
import hashlib
import importlib
import importlib.resources
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.e2e_accuracy.simulation_features import build as build_features


def predict(config_path, output):
    started = time.monotonic()
    # Branch wheels have different capabilities. Import inside this isolated
    # boundary so an older wheel produces an explicit unsupported outcome.
    try:
        api = importlib.import_module("aisim")
        compiler = importlib.import_module("aisim.compiler")
        database = importlib.import_module("aisim.sdk.perf_database")
        dashboard = importlib.import_module("aisim.replay.dashboard")
        requirements = api.ReplayOutputRequirements(
            include_raw_report=True, capture_per_request=True, capture_telemetry=True
        )
    except (ImportError, AttributeError, TypeError):
        return {"outcome": "unsupported", "reason": "replay_telemetry_unavailable"}
    config = json.loads(config_path.read_text())
    engine = config["engine"]
    version = config.pop("prediction_database_version", None) or database.get_latest_database_version(
        system=engine["hardware"], backend=engine["backend"]
    )
    if not version:
        return {"outcome": "unsupported", "reason": "database_unavailable"}
    engine["backend_version"] = version
    try:
        data_root = importlib.resources.files("aisim_core") / "systems/data"
    except ImportError:
        return {"outcome": "unsupported", "reason": "database_unavailable", "prediction_backend_version": version}
    relative = (
        f"b300_sxm/sparse_attention/sglang/{version}/dsa_generation_module_perf.parquet"
        if engine["backend"] == "sglang"
        else f"b300_sxm/attention/vllm/{version}/context_attention_perf.parquet"
    )
    data_file = data_root / relative
    if not data_file.is_file():
        return {"outcome": "unsupported", "reason": "database_unavailable", "prediction_backend_version": version}
    parsed = api.CorePredictionConfig.model_validate(config)
    spec = compiler.prediction_to_replay_spec(parsed)
    factory = api.EngineReplayRunnerFactory()
    try:
        factory.capabilities().require_compatible(spec)
    except (ValueError, NotImplementedError):
        return {
            "outcome": "unsupported",
            "reason": "replay_capability_unavailable",
            "prediction_backend_version": version,
        }
    (output / "resolved-config.json").write_text(json.dumps(parsed.model_dump(mode="json"), allow_nan=False))
    (output / "compiled-replay.json").write_text(json.dumps(asdict(spec), allow_nan=False))
    context_path = output / "feature-context.json"
    context = json.loads(context_path.read_text()) if context_path.exists() else None
    architecture, dspark_architectures = None, None
    if context is not None:
        try:
            utils = importlib.import_module("aisim_core.sdk.utils")
            common = importlib.import_module("aisim_core.sdk.common")
            architecture = utils.get_model_config_from_model_path(parsed.engine.model).get("architecture")
        except (ImportError, AttributeError):
            return {
                "outcome": "unsupported",
                "reason": "replay_telemetry_unavailable",
                "prediction_backend_version": version,
            }
        dspark_architectures = getattr(common, "DSPARK_ARCHITECTURES", None)
        features = build_features(
            parsed, spec, {}, context, architecture=architecture, dspark_architectures=dspark_architectures
        )
        (output / "simulation-features.json").write_text(json.dumps(features, allow_nan=False))
    runner = factory.create(worker_id=0)
    try:
        result = runner.run(spec, output_requirements=requirements)
    finally:
        runner.close()
    native = result.metadata["native_report"]
    if context is not None:
        features = build_features(
            parsed, spec, native, context, architecture=architecture, dspark_architectures=dspark_architectures
        )
        (output / "simulation-features.json").write_text(json.dumps(features, allow_nan=False))
    (output / "native.json").write_text(json.dumps(native, allow_nan=False))
    (output / "config.json").write_text(json.dumps(config, allow_nan=False))
    payload = dashboard.build_replay_dashboard_data(native, native["telemetry"], config=config)
    (output / "dashboard.json").write_text(json.dumps(payload, allow_nan=False))
    op_data = (
        [{"file": relative, "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest()}]
        if data_file.is_file()
        else []
    )
    return {
        "outcome": "completed",
        "reason": None,
        "prediction_backend_version": version,
        "wall_seconds": time.monotonic() - started,
        "op_data": op_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        outcome = predict(args.config, args.output)
    except Exception as exc:
        # Detailed diagnostics remain encrypted, never copied into public labels.
        (args.output / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        outcome = {"outcome": "failed", "reason": "replay_failed"}
    (args.output / "outcome.json").write_text(json.dumps(outcome))
