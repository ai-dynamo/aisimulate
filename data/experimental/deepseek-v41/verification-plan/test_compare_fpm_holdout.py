# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic fixtures only; no fixture latency is GPU evidence or calibration."""

import inspect
import json
from copy import deepcopy

import compare_fpm_holdout as subject
import pytest
from collector.fpm_forward.native_artifact import validate_native_collection


def encode(value):
    return subject.canonical(value)


def original_result(phase, index, geometry, *, warmup=False):
    _, batch, query, kv = geometry
    decode = phase == "decode"
    point = {
        "point_type": phase,
        "benchmark_id": index,
        "batch_size": batch,
        "total_prefill_tokens": query,
        "total_kv_read_tokens": kv,
        "rows": None,
        "partition": None,
        "expected_cudagraph_mode": "NONE",
        "sample_reasons": ["eager_warmup"] if warmup else ["explicit", "kvwarm_real_kv"],
    }
    prompt = (kv - batch if decode else kv + query) // batch
    stream = {
        "benchmark_id": index,
        "sampling_role": "warmup" if warmup else "measurement",
        "requests": [
            {
                "request_index": i,
                "prompt_token_ids": [i % 5] * prompt,
                "output_token_ids": [3],
                "computed_tokens": prompt + (2 if decode else 0),
            }
            for i in range(batch)
        ],
    }
    line = encode(stream)
    fpm = {
        "version": 1,
        "worker_id": f"fixture-{phase}",
        "dp_rank": 0,
        "counter_id": index,
        "wall_time": 0.2,
        "scheduled_requests": {
            "num_prefill_requests": 0 if decode else batch,
            "sum_prefill_tokens": 0 if decode else query,
            "sum_prefill_kv_tokens": 0 if decode else kv,
            "num_decode_requests": batch if decode else 0,
            "sum_decode_kv_tokens": kv if decode else 0,
            "var_prefill_length": 0.0,
            "var_decode_kv_tokens": 0.0,
        },
        "queued_requests": {},
    }
    row = {
        "point": point,
        "fpms": [fpm],
        "kv_seed_regime": "real_kv",
        "real_kv_witness": {
            "same_request": True,
            "allocated_fake_tokens": 0,
            "completed_seed_tokens": kv - (batch if decode else 0),
            "token_stream_sha256": subject.sha_bytes(line.encode()),
        },
    }
    group = {
        "benchmark_id": index,
        "point": point,
        "expected_dp_ranks": [0],
        "complete": True,
        "wall_time": 0.2,
        "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
    }
    return row, group, line


def reseal(payload, receipt):
    original = json.loads(receipt["original_admission_json"])
    original["artifacts"] = []
    for artifact in payload["artifacts"]:
        native = json.loads(artifact["native_json"])
        provenance = json.loads(artifact["collector_provenance_json"])
        original["artifacts"].append(
            {
                "artifact_sha256": subject.sha_bytes(artifact["native_json"].encode()),
                "token_stream_sha256": subject.sha_bytes(artifact["token_stream_jsonl"].encode()),
                "cell_id": provenance["cell_id"],
                "attempt_id": provenance["attempt_id"],
                "measured_points": len(native["results"]),
                "point_manifest_sha256": subject.sha_bytes(
                    encode(subject.read(subject.ROOT / "heldout.json")).encode()
                ),
            }
        )
    receipt["original_admission_json"] = encode(original)
    receipt["original_admission_sha256"] = subject.sha_bytes(receipt["original_admission_json"].encode())
    receipt["normalized_sha256"] = subject.sha_bytes(encode(payload).encode())


@pytest.fixture
def bundle():
    runtime = dict.fromkeys(subject.RUNTIME_KEYS, "fixture")
    phase_identity = {
        "producer": {"fixture_source": "immutable"},
        "native_capacity": {
            "common": {"max_model_len": 2050, "max_num_scheduled_tokens": 512, "max_num_running_reqs": 2}
        },
        "input_provenance": {
            "source": "tokenizer_text",
            "text_sha256": "1" * 64,
            "token_ids_sha256": "2" * 64,
            "tokenizer_revision": "fixture-revision",
            "token_count": 5,
            "unique_token_count": 5,
        },
    }
    cal = {
        "collection": runtime | {"phases": {p: deepcopy(phase_identity) for p in ("prefill", "decode")}},
        "files_sha256": {"table.parquet": "3" * 64},
        "keys": {key for rows in subject.frozen_geometries("calibration.json").values() for key in rows},
        "rows": [{"runtime_run_id": "old-run", "collector_attempt_id": "old-attempt"}],
        "admission": {"collector_plan_sha256": "a" * 64, "artifacts": [{"artifact_sha256": "b" * 64}]},
    }
    payload = {
        "schema": "dsv41.fpm.holdout.v1",
        "role": "independent_holdout",
        "status": "accepted",
        "runtime_identity": runtime,
        "artifacts": [],
    }
    for phase, geometries in subject.frozen_geometries("heldout.json").items():
        triples = [original_result(phase, i + 1, key) for i, key in enumerate(geometries)]
        warm = original_result(phase, len(triples) + 1, geometries[0], warmup=True)
        sidecar = "\n".join([triple[2] for triple in triples] + [warm[2]]) + "\n"
        native = {
            "schema_version": 2,
            "artifact_type": "rank",
            "status": "complete",
            "valid": True,
            "usable": True,
            "timing_valid": True,
            "stop_reason": None,
            "error": None,
            "skipped_points": [],
            "missing_phases": [],
            "run_id": "new-" + phase,
            "grid_digest": phase + "-grid",
            "dp": {"rank": 0, "size": 1},
            "config": {"mode": phase},
            "coverage": {"expected_points": len(triples), "completed_points": len(triples), "skipped_points": 0},
            "results": [triple[0] for triple in triples],
            "warmup_results": [warm[0]],
            "iteration_groups": [triple[1] for triple in triples],
            "timing": {"benchmark_elapsed_seconds": 100.0, "measured_iteration_seconds": 0.2 * len(triples)},
            "kvwarm": {"enabled": True, "warm_eligible": True, "skip_reason": None},
            "producer": deepcopy(phase_identity["producer"]),
            "execution_identity": subject.EXECUTION,
            "execution_mode": "eager",
            "cudagraph": {"mode": "NONE", "prefill_mode": "NONE", "decode_mode": "NONE", "max_capture_size": 0},
            "limits": deepcopy(phase_identity["native_capacity"]["common"]),
            "measurement_policy": {"decode": "steady_state_second_step", "prefill": "single_step"},
            "input_provenance": deepcopy(phase_identity["input_provenance"])
            | {
                "token_stream_manifest": {
                    "file": "benchmark.token-streams.jsonl",
                    "schema_version": 2,
                    "sha256": subject.sha_bytes(sidecar.encode()),
                    "records": len(triples) + 1,
                    "warmup_benchmark_ids": [len(triples) + 1],
                }
            },
        }
        cell = {
            "cell_id": "fixture-" + phase,
            "workload_kind": phase,
            "topology": subject.TOPOLOGY,
            "resolved_dtypes": subject.DTYPES,
            "execution_identity": subject.EXECUTION,
            "parallel_strategy": "pure_tp",
            "weight_quantization": "fp8_block",
            "kv_cache_dtype": "fp8",
            "input_text_sha256": "1" * 64,
            "backend_policy": {
                "policy_id": "baseline_auto",
                "aic_fields": {
                    "attention_backend": None,
                    "moe_backend": None,
                    "enable_eplb": False,
                    "enable_wideep": False,
                },
                "generator_overrides": {},
                "expected_markers": {"config.engine_args.enforce_eager": "True"},
                "admission_reason": "semantic fixture",
            },
        }
        provenance = {
            "schema_name": "aic_fpm_collector_provenance",
            "schema_version": 1,
            "cell_id": cell["cell_id"],
            "attempt_id": "new-attempt-" + phase,
            "plan_sha256": "c" * 64,
            "runtime": {"backend": "vllm", "backend_version": subject.VERSION},
        }
        payload["artifacts"].append(
            {
                "phase": phase,
                "native_json": encode(native),
                "token_stream_jsonl": sidecar,
                "collector_provenance_json": encode(provenance),
                "cell_json": encode(cell),
            }
        )
    validator = subject.sha_file(inspect.getsourcefile(validate_native_collection))
    original = {
        "schema": "dsv41.parity.admission.v1",
        "valid": True,
        "collector_plan_sha256": "c" * 64,
        "plan_file_sha256": "d" * 64,
        "checkpoint_sha256": "e" * 64,
        "validator_source_sha256": validator,
        "exporter_source_sha256": "f" * 64,
        "artifacts": [],
    }
    receipt = {
        "schema": "dsv41.fpm.holdout.admission.v1",
        "valid": True,
        "normalizer_source_sha256": "0" * 64,
        "validator_source_sha256": validator,
        "calibration_files_sha256": cal["files_sha256"],
        "collector_plan_sha256": "c" * 64,
        "collector_plan_file_sha256": "d" * 64,
        "collector_checkpoint_sha256": "e" * 64,
        "calibration_manifest_sha256": subject.sha_file(subject.ROOT / "calibration.json"),
        "holdout_manifest_sha256": subject.sha_file(subject.ROOT / "heldout.json"),
        "original_admission_json": encode(original),
    }
    reseal(payload, receipt)
    return payload, receipt, cal


def admit(bundle):
    payload, receipt, cal = bundle
    return subject.qualify_holdout(
        payload, receipt, cal, normalized_sha256=subject.sha_bytes(encode(payload).encode()), normalizer_sha256="0" * 64
    )


def test_complete_native_single_sample_fixture_and_warmup_retention(bundle):
    rows = admit(bundle)
    assert len(rows) == 38 and all(row["observation_repeats"] == 1 and row["observed_ms"] == 200 for row in rows)
    assert len([row for row in rows if row["phase"] == "decode"]) == 10
    assert all(json.loads(a["native_json"])["warmup_results"] for a in bundle[0]["artifacts"])


@pytest.mark.parametrize(
    "mutation",
    [
        "warmup_removed",
        "warmup_measured",
        "fake_kv",
        "missing_rank",
        "zero_latency",
        "duplicate_point",
        "wrong_geometry",
        "wrong_fmha",
        "old_run",
        "cuda_graph",
        "capacity",
        "source_changed",
    ],
)
def test_resealed_semantic_corruption_is_rejected(bundle, mutation):
    payload, receipt, _ = bundle
    artifact = payload["artifacts"][0]
    native = json.loads(artifact["native_json"])
    if mutation == "warmup_removed":
        artifact["token_stream_jsonl"] = "\n".join(artifact["token_stream_jsonl"].splitlines()[:-1]) + "\n"
    elif mutation == "warmup_measured":
        native["results"][0]["point"]["sample_reasons"].append("eager_warmup")
    elif mutation == "fake_kv":
        native["results"][0]["real_kv_witness"]["allocated_fake_tokens"] = 1
    elif mutation == "missing_rank":
        native["results"][0]["fpms"] = []
    elif mutation == "zero_latency":
        native["results"][0]["fpms"][0]["wall_time"] = 0.0
    elif mutation == "duplicate_point":
        native["results"][1] = deepcopy(native["results"][0])
    elif mutation == "wrong_geometry":
        native["results"][0]["point"]["total_prefill_tokens"] += 1
    elif mutation == "wrong_fmha":
        cell = json.loads(artifact["cell_json"])
        cell["resolved_dtypes"]["fmha_quant_mode"] = "bfloat16"
        artifact["cell_json"] = encode(cell)
    elif mutation == "old_run":
        native["run_id"] = "old-run"
    elif mutation == "cuda_graph":
        native["cudagraph"]["decode_mode"] = "FULL"
    elif mutation == "capacity":
        native["limits"]["max_num_scheduled_tokens"] = 4096
    else:
        native["producer"]["fixture_source"] = "changed"
    artifact["native_json"] = encode(native)
    reseal(payload, receipt)
    with pytest.raises((ValueError, TypeError)):
        admit(bundle)


@pytest.mark.parametrize("mutation", ["calibration_hash", "manifest_hash", "normalized_hash", "same_plan", "overlap"])
def test_receipt_and_calibration_separation(bundle, mutation):
    _, receipt, cal = bundle
    if mutation == "calibration_hash":
        receipt["calibration_files_sha256"] = {"table.parquet": "0" * 64}
    elif mutation == "manifest_hash":
        receipt["holdout_manifest_sha256"] = receipt["calibration_manifest_sha256"]
    elif mutation == "normalized_hash":
        receipt["normalized_sha256"] = "0" * 64
    elif mutation == "same_plan":
        cal["admission"]["collector_plan_sha256"] = receipt["collector_plan_sha256"]
    else:
        cal["keys"].add(subject.frozen_geometries("heldout.json")["decode"][0])
    with pytest.raises(ValueError):
        admit(bundle)


def test_past_kv_fpm_and_inclusive_sol_bridge_without_fitting(bundle):
    rows = [row for row in admit(bundle) if row["phase"] == "decode"][:1]
    calls = []

    def predict(metrics):
        calls.append(metrics["scheduled_requests"]["sum_decode_kv_tokens"])
        return 180.0

    fpm = subject.compare_rows(rows, predict, forward_model="fpm")
    sol = subject.compare_rows(rows, predict, forward_model="op_level")
    assert calls == [96, 97]
    assert fpm[0]["axis_bridge"]["delta"] == 0 and sol[0]["axis_bridge"]["delta"] == 1
    assert subject.statistics_for(fpm)["wape_percent"] == pytest.approx(10)


def test_missing_predictions_keep_denominator_and_private_paths_out(bundle):
    rows = admit(bundle)[:2]

    def missing(_):
        raise ValueError("No FPM cell at /private/calibration/location")

    result = subject.compare_rows(rows, missing, forward_model="fpm")
    assert subject.statistics_for(result) == {"planned_points": 2, "predicted_points": 0, "missing_predictions": 2}
    assert "/private" not in encode(result)
    partial = subject.compare_rows(rows[:1], lambda _: 180.0, forward_model="fpm") + result[1:]
    assert subject.statistics_for(partial)["planned_points"] == 2
    assert subject.statistics_for(partial)["predicted_points"] == 1


def test_opposing_errors_do_not_cancel_wape():
    rows = [{"status": "predicted", "observed_ms": 100, "predicted_ms": value} for value in (80, 120)]
    summary = subject.statistics_for(rows)
    assert summary["mean_signed_error_percent"] == pytest.approx(0)
    assert summary["wape_percent"] == pytest.approx(20)


def prediction_configuration(tmp_path):
    config = {
        "schema_version": 1,
        "model_name": subject.MODEL,
        "system_name": "gb200",
        "backend": "vllm",
        "backend_version": subject.VERSION,
        "systems_path": str(tmp_path / "systems"),
        "tp_size": 4,
        "pp_size": 1,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "attention_dp_size": 1,
        "decoder_replay": False,
        "activation_dtype": "fp8",
        "enable_shared_layer": False,
        "strict_provenance": True,
        "database_mode": "SILICON",
        "forward_model": "fpm",
    }
    (tmp_path / "prediction-config.json").write_text(encode(config))
    return config, {"directory": tmp_path}


@pytest.mark.parametrize("mode,forward", [("SILICON", "fpm"), ("SOL", "op_level")])
def test_qualified_configurations(tmp_path, mode, forward):
    config, calibration = prediction_configuration(tmp_path)
    config.update(database_mode=mode, forward_model=forward)
    if forward == "op_level":
        config.pop("activation_dtype")
    subject.validate_prediction_config(config, calibration)
    identity = subject.prediction_precision_identity(config)
    if forward == "op_level":
        assert identity == {"precision_contract": "checkpoint_native_analytical_precision_without_activation_override"}
    else:
        assert identity["execution_quantization_override"] == {"activation_dtype": "fp8"}
        assert "not_analytical_operand_precision" in identity["precision_contract"]


@pytest.mark.parametrize("override", ["fp8", "bfloat16", None])
def test_sol_does_not_inherit_fpm_activation_override(tmp_path, override):
    config, calibration = prediction_configuration(tmp_path)
    config.update(database_mode="SOL", forward_model="op_level", activation_dtype=override)
    with pytest.raises(ValueError, match="configuration fields"):
        subject.validate_prediction_config(config, calibration)


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation_dtype", "bfloat16"),
        ("decoder_replay", True),
        ("strict_provenance", False),
        ("enable_shared_layer", True),
        ("backend_version", "0.24.0"),
        ("attention_dp_size", True),
        ("forward_model", "op_level"),
        ("new_field", 1),
        ("systems_path", "/another/overlay"),
    ],
)
def test_configuration_cannot_change_precision_runtime_or_source(tmp_path, field, value):
    config, calibration = prediction_configuration(tmp_path)
    with pytest.raises(ValueError):
        subject.validate_prediction_config(config | {field: value}, calibration)
