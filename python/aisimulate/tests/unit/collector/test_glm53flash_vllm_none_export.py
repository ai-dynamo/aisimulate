# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY original-shaped NONE evidence; never native GPU qualification."""

import copy
import json

import pytest
from collector import glm53flash_vllm_none_activity as activity
from collector import glm53flash_vllm_serving_export as serving
from collector.glm53flash_contract import sha256_json
from collector.glm53flash_graph_nodes import trace_forward_identity
from collector.glm53flash_jsonl import file_sha256, iter_records
from collector.glm53flash_vllm_graph_policy import select_descriptor
from collector.glm53flash_vllm_none import SOURCE_PINS, validate_model_receipt

from .test_glm53flash_ops_evidence import put, put_lines
from .test_glm53flash_vllm_serving_export import complete_full_files

pytestmark = pytest.mark.unit


def model_receipt():
    glm5 = "vllm.models.glm5next.nvidia.model."
    glm4 = "vllm.model_executor.models.glm4_1v.Glm4vForConditionalGeneration."
    return validate_model_receipt(
        {
            "schema_name": "glm53flash_native_serving_none_model",
            "schema_version": 1,
            "source_pins": SOURCE_PINS,
            "classes": [
                glm5 + name for name in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM", "Glm5NextModel")
            ],
            "methods": [
                {"method": glm4 + "forward", "source": "model_executor/models/glm4_1v.py"},
                {"method": glm4 + "compute_logits", "source": "model_executor/models/glm4_1v.py"},
                {"method": glm5 + "Glm5NextForCausalLM.compute_logits", "source": "models/glm5next/nvidia/model.py"},
                {"method": glm5 + "Glm5NextModel.forward", "source": "models/glm5next/nvidia/model.py"},
            ],
            "compiled_model": False,
            "admission": "DIAGNOSTIC_ONLY_NATIVE_CALLS_STILL_REQUIRED",
        }
    )


def none_files(root, monkeypatch, role="calibration"):
    run, root = complete_full_files(root, monkeypatch, role)
    run["key"] = (*run["key"][:3], "prefill")
    run["points"] = [
        {
            "benchmark_id": 1,
            "point_type": "prefill",
            "batch_size": 1,
            "total_prefill_tokens": 8,
            "total_kv_read_tokens": 0,
        }
    ]
    manifest = json.loads((root / "manifest.json").read_bytes())
    entries = manifest["phases"]["context"]
    provenance = json.loads((root / "provenance.json").read_bytes())
    model = model_receipt()

    def scope(name, start, end):
        return {"name": name, "cat": "user_annotation", "ph": "X", "ts": start, "dur": end - start, "pid": 1, "tid": 2}

    for rank in range(2):
        put(root / f"serving-none-model-rank-{rank}.json", model)
        policy = json.loads((root / f"vllm-graph-policy-rank-{rank}.json").read_bytes())
        descriptor = select_descriptor(policy, batch=1, query=8, is_context=True)
        assert descriptor["cg_mode"] == "NONE"
        targets = list(iter_records(root / f"forward-rank-{rank}.jsonl"))
        all_ops = []
        for row in targets:
            row.update(
                phase="context",
                runtime_mode="NONE",
                used_cuda_graph=False,
                num_padded_tokens=8,
                query_lengths=[8],
                prefix_lengths=[0],
                total_new_tokens=8,
                total_past_kv_tokens=0,
                native_dispatch={
                    "descriptor": descriptor,
                    "policy_sha256": sha256_json(policy),
                    "physical_tokens": 8,
                    "physical_requests": 1,
                },
                native_graph_replay_completed=False,
                native_none_forward_completed=True,
                serving_none_model_sha256=sha256_json(model),
                measurement_contract=activity.NONE_MEASUREMENT_CONTRACT,
                native_runtime_boundary_gpu_ms={
                    "prepared_inputs_to_raw_model_entry": 0.002,
                    "raw_model_return_to_logits_entry": 0.003,
                },
                profiled=role == "calibration" and row["repetition"] == 4,
            )
            if role != "calibration":
                continue
            row["native_operation_calls"] = dict.fromkeys((entry["name"] for entry in entries), 1)
            events = [
                scope(activity.NONE_EXECUTION_RANGE, 0, 1000),
                scope(activity.NONE_SETUP_RANGES["prepared_inputs_to_raw_model_entry"], 1, 10),
                scope(activity.NONE_MODEL_RANGE, 11, 850),
                scope(activity.NONE_SETUP_RANGES["raw_model_return_to_logits_entry"], 851, 900),
            ]
            calls = []
            for index, entry in enumerate(entries):
                name = entry["name"]
                source = "TEST_ONLY.Native." + name
                geometry = json.loads(entry["geometry"])
                attention = entry["component"] == "attention"
                start = 950 if name == "logits" else 15 + index * 3
                events += [
                    scope(activity.OPERATION_RANGE_PREFIX + name, start, start + 2),
                    {
                        **scope("cudaLaunchKernel", start + 0.5, start + 1),
                        "cat": "cuda_runtime",
                        "args": {"correlation": index},
                    },
                    {
                        "name": "TEST_ONLY.kernel",
                        "cat": "kernel",
                        "ph": "X",
                        "ts": start + 1,
                        "dur": 0.5,
                        "args": {
                            "correlation": index,
                            "stream": 7,
                            "grid": [1, 1, 1],
                            "block": [32, 1, 1],
                            "shared memory": 0,
                        },
                    },
                ]
                calls.append(
                    {
                        "operation": name,
                        "source": source,
                        "completed": True,
                        "included_sources": [],
                        "excluded_collective_sources": [],
                        "parent_operation": None,
                    }
                )
                all_ops.append(
                    {
                        **provenance,
                        **entry,
                        "stage": "measure",
                        **{
                            key: row[key]
                            for key in (
                                "forward_id",
                                "tp_rank",
                                "phase",
                                "benchmark_id",
                                "repetition",
                                "sampling_role",
                                "dataset_role",
                                "request_set",
                                "corpus_sha256",
                                "request_ids",
                                "native_dispatch",
                                "serving_none_model_sha256",
                                "profiled",
                                "invocation",
                            )
                        },
                        "batch_size": 1,
                        "prefix": 0,
                        "x": 1 if geometry.get("token_selection") == "last_per_request" else 8,
                        "latency": 0.001,
                        "sample_count": 1,
                        "sample": row["repetition"],
                        "measurement_contract": activity.NONE_MEASUREMENT_CONTRACT,
                        "measurement_method": "native_module_cuda_events_v1",
                        "measurement_scope": "communication"
                        if geometry.get("role") == "allreduce"
                        else "compute_and_communication"
                        if geometry.get("role") == "logits"
                        else "local_compute",
                        "kv_seed_regime": "empty" if attention else "n/a",
                        "state_mode": "full_prefill" if attention else "token_only",
                        "dispatch_fingerprint": "",
                        "kernel_source": source,
                        "used_cuda_graph": False,
                        "excluded_collectives": [],
                    }
                )
            if row["profiled"]:
                trace = {
                    "traceEvents": events,
                    "aisim_native_forward": trace_forward_identity(row),
                    "aisim_native_none": {
                        "measurement_contract": activity.NONE_MEASUREMENT_CONTRACT,
                        "model_identity_sha256": sha256_json(model),
                        "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
                        "native_calls": calls,
                        "failed": False,
                    },
                }
                path = root / f"none-profile-rank-{rank}-forward-{row['invocation']}.json"
                put(path, trace)
                row["native_none_profile"] = {
                    "trace_file": path.name,
                    "trace_sha256": file_sha256(path),
                    "binding": activity.bind_none_execution(events, calls, [entry["name"] for entry in entries]),
                }
        put_lines(root / f"forward-rank-{rank}.jsonl", targets)
        if role == "calibration":
            put_lines(root / f"serving-none-measured-ops-rank-{rank}.jsonl", all_ops)
    return run, root


def test_none_reader_preserves_277_event_units_plus_two_direct_setup_intervals(tmp_path, monkeypatch):
    run, root = none_files(tmp_path, monkeypatch)
    proof = serving.read_serving_run(root, run)
    rows, _ = serving.aggregate_serving(proof, evidence_sha256="c" * 64)
    assert len(rows) == 278
    assert {row["runtime_mode"] for row in rows} == {"NONE"}
    assert {row["dispatch_fingerprint"] for row in rows} == {""}
    assert sum(row["latency"] for row in rows) == pytest.approx(0.282)
    setup = next(row for row in rows if row["component"] == "runtime")
    assert (setup["latency"], setup["contribution_count"], setup["measurement_method"]) == (
        0.005,
        2,
        "native_runtime_cuda_events_v1",
    )
    # Warmup activity lasts 0.5us per unit; measured event intervals remain 1us.
    assert all(row["latency"] == 0.001 for row in rows if row["component"] != "runtime")


@pytest.mark.parametrize(
    "defect",
    [
        "old_diagnostic",
        "old_writer",
        "profiled_measurement",
        "changed_model",
        "missing_trace",
        "foreign_trace",
        "missing_call",
        "repeated_unit",
        "source",
        "setup",
        "wrong_method",
    ],
)
def test_none_reader_rejects_legacy_or_incomplete_measured_contract(tmp_path, monkeypatch, defect):
    run, root = none_files(tmp_path, monkeypatch)
    path = root / "forward-rank-0.jsonl"
    rows = list(iter_records(path))
    if defect == "old_diagnostic":
        rows[0]["measurement_admission"] = "DIAGNOSTIC_ONLY_NO_TABLE_EXPORT"
    elif defect == "old_writer":
        rows[0].pop("measurement_contract")
    elif defect == "profiled_measurement":
        rows[5]["profiled"] = True
    elif defect == "changed_model":
        rows[0]["serving_none_model_sha256"] = "0" * 64
    elif defect == "missing_trace":
        (root / rows[4]["native_none_profile"]["trace_file"]).unlink()
    elif defect == "foreign_trace":
        trace_path = root / rows[4]["native_none_profile"]["trace_file"]
        trace = json.loads(trace_path.read_bytes())
        trace["aisim_native_forward"]["invocation"] += 1
        put(trace_path, trace)
        rows[4]["native_none_profile"]["trace_sha256"] = file_sha256(trace_path)
    elif defect == "missing_call":
        rows[0]["native_operation_calls"].pop("attention_0")
    elif defect == "setup":
        rows[0]["native_runtime_boundary_gpu_ms"] = {"unobserved_residual": 0.005}
    else:
        op_path = root / "serving-none-measured-ops-rank-0.jsonl"
        ops = list(iter_records(op_path))
        if defect == "repeated_unit":
            ops[-1] = copy.deepcopy(ops[-2])
        elif defect == "source":
            ops[0]["kernel_source"] = "TEST_ONLY.changed_source"
        else:
            ops[0]["measurement_method"] = serving.GRAPH_METHOD
        put_lines(op_path, ops)
    put_lines(path, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        serving.read_serving_run(root, run)
