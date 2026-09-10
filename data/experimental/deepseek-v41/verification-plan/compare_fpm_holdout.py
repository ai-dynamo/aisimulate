# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare fresh, single-sample GB200 FPM holdouts with an immutable calibration.

Public normalized contract: dsv41.fpm.holdout.v1 contains role=independent_holdout,
status=accepted, runtime_identity, and two artifacts (prefill/decode). Each artifact
contains phase and the exact UTF-8 strings native_json, token_stream_jsonl,
collector_provenance_json, cell_json. Preserve the entire native artifact and
sidecar, including warmup records. Never use a measured point as calibration.

The separate dsv41.fpm.holdout.admission.v1 receipt binds normalized_sha256,
normalizer_source_sha256, validator_source_sha256, collector_plan_sha256,
collector_plan_file_sha256, collector_checkpoint_sha256, calibration_files_sha256,
calibration_manifest_sha256, holdout_manifest_sha256 and original_admission_json.
The latter is the independently rerun dsv41.parity.admission.v1 result;
its original UTF-8 text is stored as original_admission_json and its SHA is bound
by original_admission_sha256. The comparator reruns the existing complete native
Collector validator, including real token history, rather than trusting counts.
Runtime identity is the selected calibration collection-receipt fields listed
by RUNTIME_KEYS; source and point/grid identities remain attached to each phase.

This is descriptive accuracy on 38 fixed geometries, one sample per point.
No repeat statistics, population confidence intervals, or correction fitting.
The strict FPM query uses the qualified table's explicit FMHA FP8 identity.
The analytical SOL baseline keeps checkpoint-native precision without an
activation override; the table selector is not evidence for FP8 arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

from normalize_fpm import prediction_input

ROOT = Path(__file__).resolve().parent
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
VERSION = "0.1.dev20904+g179dd0fa9"
EXECUTION = {
    "model_config_sha256": "22c3912140adeb60ecbd3c7a9b54e62997adf65ad4c69440f8082cf29f5ef0d4",
    "execution_profile": "full",
    "engram_residency": "hbm_tp_sharded",
    "input_modality": "text",
}
DTYPES = {
    "comm_quant_mode": "half",
    "fmha_quant_mode": "fp8",
    "fmha_resolution": "checkpoint_native",
    "gemm_quant_mode": "fp8_block",
    "kvcache_quant_mode": "fp8",
    "moe_quant_mode": "w4a8_mxfp4_mxfp8",
}
TOPOLOGY = {"tp": 4, "pp": 1, "dp": 1, "moe_tp": 4, "moe_ep": 1, "cp": 1}
RUNTIME_KEYS = (
    "model_revision",
    "base_arm64_image_digest",
    "prepared_runtime_image_sha256",
    "dynamo_installed_packages",
    "recipe_commit",
    "actual_vllm_package",
    "actual_moe_kernel",
    "tp",
    "pp",
    "dp",
    "moe_ep",
    "pure_tp",
    "text_only",
    "dspark",
    "decoder_replay",
    "cuda_graphs",
    "kv_cache_dtype",
    "measurement_policy",
    "repetitions_per_point",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected):
    return canonical(actual) == canonical(expected)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid source digest")
    return value


def geometry(phase, point):
    values = (point["batch_size"], point.get("total_prefill_tokens", 0), point["total_kv_read_tokens"])
    require(all(type(v) is int and v >= 0 for v in values) and values[0] > 0, "invalid geometry")
    batch, query, prefix = values
    require(query % batch == prefix % batch == 0, "only homogeneous frozen geometries are qualified")
    rows = point.get("rows")
    expected_rows = [[query // batch, prefix // batch] for _ in range(batch)]
    require(rows is None or same(rows, expected_rows), "per-request geometry differs from aggregate totals")
    require(point.get("partition") is None, "partitioned geometry is outside the frozen domain")
    return phase, *values


def frozen_geometries(name):
    manifest = read(ROOT / name)
    require(manifest.get("schema_version") == 3, "unknown frozen workload schema")
    return {phase: [geometry(phase, p) for p in manifest[phase]] for phase in ("prefill", "decode")}


def load_calibration(directory):
    import pyarrow.parquet as pq

    directory = Path(directory).resolve(strict=True)
    files = list((directory / "systems").rglob("fpm_forward_perf.parquet"))
    require(len(files) == 1 and all(not p.is_symlink() for p in directory.rglob("*")), "ambiguous calibration overlay")
    table = files[0]
    metadata_path = table.with_suffix(".metadata.json")
    collection = read(directory / "collection-receipt.json")
    admission = read(directory / "admission-receipt.json")
    metadata = read(metadata_path)
    rows = pq.read_table(table).to_pylist()
    expected = frozen_geometries("calibration.json")
    keys = [geometry(r["workload_kind"], r) for r in rows]
    require(
        len(keys) == len(set(keys)) == 126 and set(keys) == {key for group in expected.values() for key in group},
        "calibration must be exact 126",
    )
    require(
        same(read(directory / "point-manifest.json"), read(ROOT / "calibration.json")), "wrong calibration manifest"
    )
    require(metadata["schema_version"] == 7 and metadata["row_count"] == 126, "wrong calibration table schema")
    require(metadata["parquet_sha256"] == sha_file(table), "calibration table changed")
    require(collection["role"] == "calibration_only" and admission["valid"] is True, "calibration is not admitted")
    for row in rows:
        expected_identity = {
            "model_path": MODEL,
            "system": "gb200",
            "backend": "vllm",
            "backend_version": VERSION,
            "parallel_strategy": "pure_tp",
            "weight_quantization": "fp8_block",
            "kv_cache_dtype": "fp8",
            "measurement_policy": "dynamo_native_single_sample_v1",
            "measurement_repeats": 1,
            "enable_wideep": False,
            "enable_eplb": False,
            **TOPOLOGY,
            **EXECUTION,
            **{key: value for key, value in DTYPES.items() if key != "kvcache_quant_mode"},
        }
        require(
            all(same(row.get(k), v) for k, v in expected_identity.items()), "calibration execution/precision mismatch"
        )
        require(
            type(row["latency_ms"]) in (int, float) and math.isfinite(row["latency_ms"]) and row["latency_ms"] > 0,
            "invalid calibration latency",
        )
        require(row["source_plan_sha256"] == admission["collector_plan_sha256"], "calibration plan mismatch")
    require(
        set(metadata["runtime_run_ids"]) == {r["runtime_run_id"] for r in rows}
        and set(metadata["collector_attempt_ids"]) == {r["collector_attempt_id"] for r in rows},
        "calibration run/attempt metadata differs from table rows",
    )
    require(
        {r["collector_attempt_id"] for r in rows} == {a["attempt_id"] for a in admission["artifacts"]},
        "calibration original admission differs from table attempts",
    )
    require(
        collection["actual_vllm_package"] == VERSION and collection["cuda_graphs"] is False,
        "unqualified calibration runtime",
    )
    chosen = [
        table,
        metadata_path,
        directory / "systems/gb200.yaml",
        directory / "collection-receipt.json",
        directory / "admission-receipt.json",
        directory / "point-manifest.json",
        directory / "prediction-config.json",
    ]
    return {
        "directory": directory,
        "rows": rows,
        "keys": set(keys),
        "collection": collection,
        "admission": admission,
        "table": table,
        "files_sha256": {p.relative_to(directory).as_posix(): sha_file(p) for p in chosen},
    }


def make_cell(value):
    from collector.fpm_forward.planner import BackendPolicy, FPMCell
    from collector.fpm_forward.types import ParallelTopology

    from aiconfigurator_core.sdk.fpm_identity import EXECUTION_COLUMNS

    require(
        same(value["topology"], TOPOLOGY) and same(value["execution_identity"], EXECUTION), "wrong topology/profile"
    )
    require(same(value["resolved_dtypes"], DTYPES), "wrong actual FMHA/model precision")
    policy = value["backend_policy"]
    require(
        policy["policy_id"] == "baseline_auto"
        and policy["generator_overrides"] == {}
        and policy["aic_fields"]
        == {"attention_backend": None, "moe_backend": None, "enable_wideep": False, "enable_eplb": False}
        and policy["expected_markers"] == {"config.engine_args.enforce_eager": "True"},
        "cell backend policy differs from the qualified eager baseline",
    )
    require(
        value["parallel_strategy"] == "pure_tp"
        and value["weight_quantization"] == "fp8_block"
        and value["kv_cache_dtype"] == "fp8",
        "wrong cell precision or parallel strategy",
    )
    return FPMCell(
        cell_id=value["cell_id"],
        workload_kind=value["workload_kind"],
        topology=ParallelTopology(**TOPOLOGY),
        weight_quantization=value["weight_quantization"],
        kv_cache_dtype=value["kv_cache_dtype"],
        backend_policy=BackendPolicy(**value["backend_policy"]),
        parallel_strategy=value["parallel_strategy"],
        execution_identity=tuple(EXECUTION[k] for k in EXECUTION_COLUMNS),
        input_text_sha256=value["input_text_sha256"],
        **{key: value for key, value in DTYPES.items() if key != "kvcache_quant_mode"},
    )


def qualify_holdout(payload, receipt, calibration, *, normalized_sha256, normalizer_sha256):
    from collector.fpm_forward.native_artifact import validate_native_collection

    require(
        payload.get("schema") == "dsv41.fpm.holdout.v1"
        and payload.get("role") == "independent_holdout"
        and payload.get("status") == "accepted",
        "not an admitted independent holdout",
    )
    require(
        receipt.get("schema") == "dsv41.fpm.holdout.admission.v1" and receipt.get("valid") is True,
        "holdout admission receipt is missing",
    )
    require(receipt["normalized_sha256"] == normalized_sha256, "normalized holdout checksum mismatch")
    require(receipt["normalizer_source_sha256"] == normalizer_sha256, "normalizer source mismatch")
    validator_sha = sha_file(inspect.getsourcefile(validate_native_collection))
    require(receipt["validator_source_sha256"] == validator_sha, "native validator source changed")
    require(same(receipt["calibration_files_sha256"], calibration["files_sha256"]), "calibration inputs changed")
    for name, filename in (("calibration", "calibration.json"), ("holdout", "heldout.json")):
        require(receipt[f"{name}_manifest_sha256"] == sha_file(ROOT / filename), "frozen point manifest changed")
    original_raw = receipt["original_admission_json"].encode()
    require(sha_bytes(original_raw) == receipt["original_admission_sha256"], "original Collector admission changed")
    original = json.loads(original_raw)
    require(
        original.get("schema") == "dsv41.parity.admission.v1" and original.get("valid") is True,
        "original Collector admission was not accepted",
    )
    require(original["validator_source_sha256"] == validator_sha, "original admission used another validator")
    digest(original["exporter_source_sha256"])
    for output_key, source_key in (
        ("collector_plan_sha256", "collector_plan_sha256"),
        ("collector_plan_file_sha256", "plan_file_sha256"),
        ("collector_checkpoint_sha256", "checkpoint_sha256"),
    ):
        require(digest(receipt[output_key]) == original[source_key], "Collector plan/checkpoint source mismatch")
    require(
        receipt["collector_plan_sha256"] != calibration["admission"]["collector_plan_sha256"],
        "calibration cannot be relabeled as an independent run",
    )
    reference = calibration["collection"]
    require(
        same(payload["runtime_identity"], {k: reference[k] for k in RUNTIME_KEYS}), "runtime differs from calibration"
    )
    holds = frozen_geometries("heldout.json")
    hold_keys = [key for group in holds.values() for key in group]
    require(
        len(hold_keys) == len(set(hold_keys)) == 38 and not (set(hold_keys) & calibration["keys"]),
        "calibration/holdout geometry overlap",
    )
    require(len(payload["artifacts"]) == len(original["artifacts"]) == 2, "missing or extra native phase artifact")
    admitted = {a["cell_id"]: a for a in original["artifacts"]}
    require(len(admitted) == 2, "duplicated original cell identity")
    old_runs = {row["runtime_run_id"] for row in calibration["rows"]}
    old_attempts = {row["collector_attempt_id"] for row in calibration["rows"]}
    old_raw = {row["artifact_sha256"] for row in calibration["admission"]["artifacts"]}
    output, phases, runs, attempts = [], set(), set(), set()
    for artifact in payload["artifacts"]:
        phase = artifact["phase"]
        require(phase in holds and phase not in phases, "duplicated or unknown phase")
        phases.add(phase)
        native_raw = artifact["native_json"].encode()
        tokens_raw = artifact["token_stream_jsonl"].encode()
        provenance_raw = artifact["collector_provenance_json"].encode()
        native, provenance, value = (
            json.loads(native_raw),
            json.loads(provenance_raw),
            json.loads(artifact["cell_json"]),
        )
        cell = make_cell(value)
        require(cell.workload_kind == phase, "cell phase mismatch")
        binding = admitted[cell.cell_id]
        require(
            sha_bytes(native_raw) == binding["artifact_sha256"]
            and sha_bytes(tokens_raw) == binding["token_stream_sha256"],
            "original native artifact or complete token sidecar changed",
        )
        require(
            binding["point_manifest_sha256"] == sha_bytes(canonical(read(ROOT / "heldout.json")).encode()),
            "native artifact does not belong to the frozen holdout plan",
        )
        require(
            provenance["attempt_id"] == binding["attempt_id"] and provenance["runtime"]["backend_version"] == VERSION,
            "native Collector attempt/backend mismatch",
        )
        run, attempt = native["run_id"], binding["attempt_id"]
        require(
            run not in old_runs | runs
            and attempt not in old_attempts | attempts
            and binding["artifact_sha256"] not in old_raw,
            "native run/attempt reused calibration or another holdout phase",
        )
        runs.add(run)
        attempts.add(attempt)
        expected_phase = reference["phases"][phase]
        require(same(native["producer"], expected_phase["producer"]), "producer source/runtime changed")
        input_base = {k: v for k, v in native["input_provenance"].items() if k != "token_stream_manifest"}
        require(same(input_base, expected_phase["input_provenance"]), "real text/tokenizer identity changed")
        require(
            same(native["execution_identity"], EXECUTION) and native["execution_mode"] == "eager",
            "wrong native execution",
        )
        require(
            all(native["cudagraph"][key] == "NONE" for key in ("mode", "prefill_mode", "decode_mode"))
            and native["cudagraph"]["max_capture_size"] == 0,
            "native CUDA graph metadata contradicts eager execution",
        )
        require(
            all(
                native["limits"][key] == expected_phase["native_capacity"]["common"][key]
                for key in ("max_model_len", "max_num_scheduled_tokens", "max_num_running_reqs")
            ),
            "native scheduler capacity configuration differs from calibration",
        )
        require(
            native["measurement_policy"] == {"decode": "steady_state_second_step", "prefill": "single_step"},
            "wrong native timing boundary",
        )
        with tempfile.TemporaryDirectory(prefix="fpm-holdout-admission-") as temp:
            raw = Path(temp)
            pod = raw / "rank0"
            pod.mkdir()
            (pod / "benchmark.json").write_bytes(native_raw)
            stream_name = native["input_provenance"]["token_stream_manifest"]["file"]
            require(
                Path(stream_name).name == stream_name and stream_name.endswith(".token-streams.jsonl"),
                "unsafe sidecar name",
            )
            (pod / stream_name).write_bytes(tokens_raw)
            (pod / "collector-provenance.json").write_bytes(provenance_raw)
            checked = validate_native_collection(
                cell, raw, expected_plan_sha256=receipt["collector_plan_sha256"], expected_attempt_id=attempt
            )
        actual = [geometry(phase, row.point) for row in checked.points]
        require(
            len(actual) == len(set(actual)) == binding["measured_points"] and set(actual) == set(holds[phase]),
            "native measured cells differ from the exact frozen holdouts",
        )
        by_key = {geometry(phase, row["point"]): row for row in native["results"]}
        for index, key in enumerate(holds[phase]):
            row = by_key[key]
            fpm = row["fpms"][0]
            scheduled = fpm["scheduled_requests"]
            require(
                scheduled.get("var_prefill_length") == scheduled.get("var_decode_kv_tokens") == 0,
                "heterogeneous native variance is outside this domain",
            )
            output.append(
                {
                    "case_id": f"{phase}-{index:04d}",
                    "phase": phase,
                    "point": row["point"],
                    "native_fpm": fpm,
                    "observed_ms": fpm["wall_time"] * 1000,
                    "observation_repeats": 1,
                    "native_run_id": run,
                    "collector_attempt_id": attempt,
                    "artifact_sha256": binding["artifact_sha256"],
                    "token_stream_sha256": row["real_kv_witness"]["token_stream_sha256"],
                }
            )
    return output


def validate_prediction_config(config, calibration):
    expected = read(calibration["directory"] / "prediction-config.json")
    require(
        (config.get("database_mode"), config.get("forward_model")) in (("SILICON", "fpm"), ("SOL", "op_level")),
        "only strict FPM or explicit analytical op-level baseline is qualified",
    )
    expected.update(database_mode=config["database_mode"], forward_model=config["forward_model"])
    if config["forward_model"] == "op_level":
        expected.pop("activation_dtype", None)
    require(set(config) == set(expected), "unknown prediction configuration fields")
    require(
        all(same(value, expected[key]) for key, value in config.items() if key != "systems_path"),
        "prediction model/backend/precision/policy differs from the qualified prediction contract",
    )
    require(
        Path(config["systems_path"]).resolve() == (calibration["directory"] / "systems").resolve(),
        "prediction must use the exact immutable calibration overlay",
    )


def prediction_precision_identity(config):
    if config["forward_model"] == "fpm":
        return {
            "precision_contract": "qualified_fpm_table_fmha_fp8_selector_not_analytical_operand_precision",
            "execution_quantization_override": {"activation_dtype": "fp8"},
        }
    return {"precision_contract": "checkpoint_native_analytical_precision_without_activation_override"}


def statistics_for(rows):
    usable = [r for r in rows if r["status"] == "predicted"]
    result = {
        "planned_points": len(rows),
        "predicted_points": len(usable),
        "missing_predictions": len(rows) - len(usable),
    }
    if not usable:
        return result
    signed = [100 * (r["predicted_ms"] / r["observed_ms"] - 1) for r in usable]
    absolute = sorted(abs(x) for x in signed)
    position = 0.9 * (len(absolute) - 1)
    lower = int(position)
    result.update(
        mean_signed_error_percent=statistics.mean(signed),
        median_absolute_error_percent=statistics.median(absolute),
        p90_absolute_error_percent=absolute[lower]
        + (position - lower) * (absolute[min(lower + 1, len(absolute) - 1)] - absolute[lower]),
        wape_percent=100
        * sum(abs(r["predicted_ms"] - r["observed_ms"]) for r in usable)
        / sum(r["observed_ms"] for r in usable),
    )
    return result


def compare_rows(rows, predictor, *, forward_model):
    require(forward_model in {"fpm", "op_level"}, "unknown forward modeling path")
    target = "whole_forward_past_kv" if forward_model == "fpm" else "op_level_inclusive_query"
    results = []
    for row in rows:
        metrics, bridge = prediction_input(row["native_fpm"], producer_semantics="vllm_past_kv", target_axis=target)
        result = dict(row, prediction_input=metrics, axis_bridge=bridge)
        try:
            value = predictor(metrics)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, "missing positive prediction")
        except Exception as error:
            # Native errors can contain private data-root paths. Retain type and
            # a content digest; the original exception remains in private logs.
            result.update(
                status="prediction_unavailable",
                failure_type=type(error).__name__,
                failure_sha256=sha_bytes(str(error).encode()),
            )
        else:
            result.update(
                status="predicted", predicted_ms=value, signed_error_percent=100 * (value / row["observed_ms"] - 1)
            )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("holdout", "admission", "calibration-dir", "prediction-config", "normalizer-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    input_paths = {
        "normalized_holdout": args.holdout,
        "holdout_admission": args.admission,
        "prediction_config": args.prediction_config,
        "normalizer_source": args.normalizer_source,
    }
    input_hashes = {key: sha_file(path) for key, path in input_paths.items()}
    calibration = load_calibration(args.calibration_dir)
    payload, receipt = read(args.holdout), read(args.admission)
    rows = qualify_holdout(
        payload,
        receipt,
        calibration,
        normalized_sha256=sha_file(args.holdout),
        normalizer_sha256=sha_file(args.normalizer_source),
    )
    config = read(args.prediction_config)
    validate_prediction_config(config, calibration)
    import normalize_fpm

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk import engine, rust_engine_step, utils
    from aiconfigurator_core.sdk.models import deepseek_v41
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    resolved = utils.get_model_config_from_model_path(MODEL)["raw_config"]
    require(
        sha_bytes(canonical(resolved).encode()) == EXECUTION["model_config_sha256"],
        "actual SDK model resolution differs from measured execution identity",
    )
    checkpoint_path = utils._get_model_config_path() / f"{MODEL.replace('/', '--')}_config.json"
    model_identity = {
        "checkpoint_config_file_sha256": sha_file(checkpoint_path),
        "checkpoint_config_canonical_sha256": sha_bytes(canonical(read(checkpoint_path)).encode()),
        "resolved_config_canonical_sha256": EXECUTION["model_config_sha256"],
        **prediction_precision_identity(config),
    }
    model = RustForwardPassPerfModel.from_native(config)
    results = compare_rows(rows, model.estimate_forward_pass_time_ms, forward_model=config["forward_model"])
    require(
        load_calibration(args.calibration_dir)["files_sha256"] == calibration["files_sha256"],
        "calibration mutated during prediction",
    )
    require(
        {key: sha_file(path) for key, path in input_paths.items()} == input_hashes,
        "normalized observations or prediction input changed during comparison",
    )
    output = {
        "schema": "dsv41.fpm.holdout.comparison.v1",
        "role": "independent_holdout",
        "observed_target": "dynamo_native_fpm_single_sample_seconds_to_ms",
        "observation_repeats_per_point": 1,
        "measurement_policy": "dynamo_native_single_sample_v1",
        "warmup_observations_excluded": {
            artifact["phase"]: len(json.loads(artifact["native_json"])["warmup_results"])
            for artifact in payload["artifacts"]
        },
        "calibration_points": 126,
        "calibration_inventory_scope": "immutable input overlay; FPM table is not used by the analytical op baseline",
        "confidence_intervals": None,
        "correction_fitting": False,
        "weighting": (
            "equal configurations for signed/APE statistics; observed-latency weighted WAPE; supported subset only"
        ),
        "prediction_contract": {k: v for k, v in config.items() if k != "systems_path"},
        "input_files_sha256": input_hashes,
        "resolved_model_identity": model_identity,
        "calibration_files_sha256": calibration["files_sha256"],
        "prediction_sources": {
            "comparison": sha_file(__file__),
            "normalization": sha_file(normalize_fpm.__file__),
            "native_extension": sha_file(native.__file__),
            "model": sha_file(deepseek_v41.__file__),
            "engine": sha_file(engine.__file__),
            "adapter": sha_file(rust_engine_step.__file__),
            "model_resolution": sha_file(utils.__file__),
        },
        "summary": statistics_for(results),
        "by_phase": {
            phase: statistics_for([r for r in results if r["phase"] == phase]) for phase in ("prefill", "decode")
        },
        "cases": results,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
