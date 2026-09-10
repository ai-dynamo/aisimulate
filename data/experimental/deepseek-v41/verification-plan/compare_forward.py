# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare admitted native forward holdouts through the shared prediction API.

Run with the SILICON checkout's source and rebuilt native extension available.
This does not fit correction factors or turn missing predictions into zeroes.
The observed target includes native preparation and sampling, not HTTP E2E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path

import analyze_e2e
import normalize_fpm
from analyze_e2e import quantile
from normalize_fpm import prediction_input


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prediction_contract(config, observations):
    """Admit only the runtime and default precision used by this GB300 study."""
    required = {
        "model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "system_name": "gb300",
        "backend": "sglang",
        "backend_version": "0.0.0.dev0",
        "tp_size": 4,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "enable_shared_layer": False,
        "strict_provenance": True,
    }
    defaults = {"schema_version": 1, "pp_size": 1, "attention_dp_size": 1, "cp_size": 1, "nextn": 0}
    for key, expected in (required | defaults).items():
        actual = config.get(key, defaults.get(key))
        if key == "nextn" and actual is None:
            actual = 0
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"unqualified prediction contract field: {key}")
    profile = observations.get("execution_profile")
    replay = config.get("decoder_replay", False)
    if (
        profile not in {"full", "decoder_bounded"}
        or type(replay) is not bool
        or replay != (profile == "decoder_bounded")
    ):
        raise ValueError("prediction decoder profile differs from observations")
    if config.get("database_mode") not in {"SOL", "HYBRID", "SILICON"}:
        raise ValueError("unqualified prediction database_mode")
    if config.get("forward_model", "op_level") not in {"op_level", "fpm"}:
        raise ValueError("unqualified prediction forward_model")
    # The measured study resolves the checkpoint's native mixed precision. A
    # caller override, even one that appears equivalent, needs its own proof.
    precision = {"weight_dtype", "moe_dtype", "activation_dtype", "kv_cache_dtype"}
    optional_none = precision | {"kv_block_size", "transfer_policy"}
    if any(config.get(key) is not None for key in optional_none):
        raise ValueError("study requires checkpoint precision and default kernel/cache policies without overrides")
    allowed = (
        set(required)
        | set(defaults)
        | optional_none
        | {
            "decoder_replay",
            "database_mode",
            "forward_model",
            "systems_path",
        }
    )
    if set(config) - allowed:
        raise ValueError("prediction config contains unsupported contract fields")
    if not isinstance(config.get("systems_path"), str) or not config["systems_path"].strip():
        raise ValueError("study requires one explicit systems overlay")


def system_data_identity(systems_path):
    """Hash the selected overlay, including policy files, using relative labels.

    This is the complete input inventory, not a claim that every file was read
    by every operator. Shared-layer inheritance is disabled by admission.
    """
    import yaml

    root = Path(systems_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("systems overlay is not a directory")
    spec_path = root / "gb300.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    raw_data_dir = Path(spec["data_dir"])
    if raw_data_dir.is_absolute() or ".." in raw_data_dir.parts:
        raise ValueError("study system data_dir must remain inside its overlay")
    data_root = (root / raw_data_dir).resolve(strict=True)
    if not data_root.is_relative_to(root) or not data_root.is_dir():
        raise ValueError("study system data directory escapes its overlay")
    files = {}
    for path in sorted(root.rglob("*")):
        # Reject directory links too: Path.rglob does not traverse them.
        if path.is_symlink() or (path.is_file() and not path.resolve().is_relative_to(root)):
            raise ValueError("study overlay must not contain symlinked input files/directories")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_hash(path)
    if "gb300.yaml" not in files:
        raise ValueError("missing selected system specification")
    return {
        "inventory_scope": "complete explicit systems overlay; includes unused files",
        "system_spec_sha256": files["gb300.yaml"],
        "data_dir": raw_data_dir.as_posix(),
        "files_sha256": files,
        "inventory_sha256": hashlib.sha256(canonical(files).encode()).hexdigest(),
    }


def resolved_model_identity(config, observations):
    """Bind the actual SDK-resolved config to the observed pinned checkpoint."""
    from aiconfigurator_core.sdk import utils

    name = config["model_name"]
    checkpoint_path = utils._get_model_config_path() / f"{name.replace('/', '--')}_config.json"
    checkpoint = utils._load_pre_downloaded_hf_config(name)
    checkpoint_sha = hashlib.sha256(canonical(checkpoint).encode()).hexdigest()
    if checkpoint_sha != observations["input_provenance"]["config_sha256"]:
        raise ValueError("prediction checkpoint config differs from native observations")
    resolved = utils.get_model_config_from_model_path(name)["raw_config"]
    expected = utils._attach_inferred_quant_fields(deepcopy(checkpoint))
    if canonical(resolved) != canonical(expected):
        raise ValueError("resolved model config differs from the observed checkpoint and inferred precision")
    identity = {
        "checkpoint_config_file_sha256": file_hash(checkpoint_path),
        "checkpoint_config_canonical_sha256": checkpoint_sha,
        "resolved_config_canonical_sha256": hashlib.sha256(canonical(resolved).encode()).hexdigest(),
        "normalization": "SDK _attach_inferred_quant_fields on the pinned checkpoint; no precision overrides",
    }
    if config.get("activation_dtype") is not None:
        identity["normalization"] = (
            "SDK _attach_inferred_quant_fields on the pinned checkpoint; execution override recorded separately"
        )
        identity["execution_quantization_override"] = {"activation_dtype": config["activation_dtype"]}
    if config.get("fpm_fmha_dtype") is not None:
        identity["fpm_query_identity_override"] = {"fpm_fmha_dtype": config["fpm_fmha_dtype"]}
        identity["fpm_selector_semantics"] = "Table identity only; checkpoint arithmetic and memory remain unchanged"
    return identity


def case_metrics(case):
    batch = case["batch_size"]
    context = case["phase"] == "context"
    if case["phase"] not in {"context", "generation"}:
        raise ValueError("unknown native workload phase")
    return {
        "version": 1,
        "scheduled_requests": {
            "num_prefill_requests": batch if context else 0,
            "sum_prefill_tokens": batch * case["query"] if context else 0,
            "sum_prefill_kv_tokens": batch * case["prefix"] if context else 0,
            "num_decode_requests": 0 if context else batch,
            "sum_decode_kv_tokens": 0 if context else batch * case["native_inclusive_kv"],
            "var_prefill_length": 0.0,
            "var_decode_kv_tokens": 0.0,
        },
    }


def error_summary(rows):
    paired = [row for row in rows if row["status"] == "predicted"]
    if not paired:
        return {"planned_points": len(rows), "predicted_points": 0}
    signed = [row["signed_error_percent"] for row in paired]
    absolute = [abs(value) for value in signed]
    return {
        "planned_points": len(rows),
        "predicted_points": len(paired),
        "weighting": (
            "signed/APE statistics: equal logical configurations; "
            "WAPE: observed-latency weighted; observed median of rank maxima"
        ),
        "mean_signed_error_percent": statistics.mean(signed),
        "median_absolute_error_percent": statistics.median(absolute),
        "p90_absolute_error_percent_across_configurations": quantile(absolute, 0.9),
        "wape_percent": 100
        * sum(abs(r["predicted_ms"] - r["observed_ms"]) for r in paired)
        / sum(r["observed_ms"] for r in paired),
    }


def compare_cases(cases, predictor, *, forward_model):
    if forward_model not in {"op_level", "fpm"}:
        raise ValueError("unknown forward model")
    target = "whole_forward_past_kv" if forward_model == "fpm" else "op_level_inclusive_query"
    rows = []
    for case in cases:
        repeats = case["rank_max_ms"]
        if len(repeats) < 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in repeats):
            raise ValueError("invalid or incomplete measured repetitions")
        observed = statistics.median(repeats)
        if observed != case["median_ms"]:
            raise ValueError("reported observation differs from measured median")
        metrics, bridge = prediction_input(
            case_metrics(case), producer_semantics="sglang_inclusive_query", target_axis=target
        )
        row = dict(case, observed_ms=observed, prediction_input=metrics, axis_bridge=bridge)
        try:
            predicted = predictor(metrics)
            if type(predicted) not in (int, float) or not math.isfinite(predicted) or predicted <= 0:
                raise ValueError("native estimator returned no finite positive prediction")
        except Exception as error:
            row.update(status="prediction_unavailable", failure_type=type(error).__name__, failure=str(error))
        else:
            row.update(
                status="predicted", predicted_ms=predicted, signed_error_percent=100 * (predicted / observed - 1)
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--heldout-plan", type=Path, required=True)
    parser.add_argument("--prediction-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from collector.sglang.dsv41_forward_results import BOUNDARY
    from collector.sglang.dsv41_workloads import freeze_workloads

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk import engine, rust_engine_step, utils
    from aiconfigurator_core.sdk.models import deepseek_v41
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    observations = json.loads(args.observations.read_bytes())
    plan = freeze_workloads(json.loads(args.heldout_plan.read_bytes()))
    config = json.loads(args.prediction_config.read_bytes())
    if (
        observations.get("status") != "accepted"
        or observations["timing_boundary"] != BOUNDARY
        or observations["plan_sha256"] != plan["source_sha256"]
        or len(observations["cases"]) != len(plan["cases"])
    ):
        raise ValueError("only complete admitted observations for this frozen holdout plan are accepted")
    for observed, planned in zip(observations["cases"], plan["cases"], strict=True):
        if any(observed.get(key) != value for key, value in planned.items()):
            raise ValueError("observation geometry/order differs from frozen holdout plan")
    validate_prediction_contract(config, observations)
    model_identity = resolved_model_identity(config, observations)
    systems_identity = system_data_identity(config["systems_path"])
    model = RustForwardPassPerfModel.from_native(config)
    rows = compare_cases(
        observations["cases"],
        model.estimate_forward_pass_time_ms,
        forward_model=config.get("forward_model", "op_level"),
    )
    if systems_identity != system_data_identity(config["systems_path"]):
        raise ValueError("systems inputs changed while computing predictions")
    if model_identity != resolved_model_identity(config, observations):
        raise ValueError("resolved model inputs changed while computing predictions")
    report = {
        "schema": "dsv41.forward.comparison.v1",
        "observed_target": BOUNDARY,
        "prediction_input_origin": "frozen logical workload manifest, not observed FPM timing",
        "execution_profile": observations["execution_profile"],
        "database_mode": config["database_mode"],
        "forward_model": config.get("forward_model", "op_level"),
        "correction_fitting": False,
        "observation_sha256": file_hash(args.observations),
        "heldout_plan_sha256": file_hash(args.heldout_plan),
        "prediction_config_sha256": file_hash(args.prediction_config),
        "prediction_sources": {
            "model_python_sha256": file_hash(deepseek_v41.__file__),
            "engine_python_sha256": file_hash(engine.__file__),
            "native_extension_sha256": file_hash(native.__file__),
            "prediction_adapter_sha256": file_hash(rust_engine_step.__file__),
            "model_resolution_sha256": file_hash(utils.__file__),
            "comparison_analysis_sha256": file_hash(__file__),
            "statistics_analysis_sha256": file_hash(analyze_e2e.__file__),
            "axis_normalization_sha256": file_hash(normalize_fpm.__file__),
        },
        "resolved_model_identity": model_identity,
        "systems_identity": systems_identity,
        "summary": error_summary(rows),
        "by_phase": {
            phase: error_summary([r for r in rows if r["phase"] == phase]) for phase in ("context", "generation")
        },
        "cases": rows,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
