# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY full366 source/activity/raw-event rederivation and SDK interoperability."""

import copy
import json
import shutil
from importlib.resources import files

import pytest

from collector import glm53flash_sglang_prefill_export as export
from collector.glm53flash_contract import sha256_json
from collector.glm53flash_graph_nodes import trace_forward_identity
from collector.glm53flash_jsonl import file_sha256
from collector.glm53flash_sglang_prefill_activity import (
    METHOD,
    PREFILL_RANGE,
    SETUP_RANGE,
    UNIT_PREFIX,
    bind_prefill_activity,
    dispatch_signatures,
)
from collector.glm53flash_validation import load_native

from .test_glm53flash_ops_evidence import put, put_lines
from .test_glm53flash_sglang_native_prefill import prefill_fixture

pytestmark = pytest.mark.unit


def fixture(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    run, root, records = prefill_fixture(tmp_path, "control")
    run["role"] = "calibration"
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest["phases"]["context"]

    def scope(name, ts, dur):
        return {"name": name, "cat": "user_annotation", "ph": "X", "ts": ts, "dur": dur, "pid": 1, "tid": 2}

    calls = [
        {
            "operation": entry["name"],
            "scope_name": f"{UNIT_PREFIX}{i}/{entry['name']}",
            "source": "TEST_ONLY.native." + entry["name"],
            "completed": True,
            "included_sources": [],
            "excluded_collective_sources": [],
            "parent_operation": None,
            "parent_scope_name": None,
        }
        for i, entry in enumerate(entries)
    ]
    events = [scope(PREFILL_RANGE, 0, 4000), scope(SETUP_RANGE, 1, 5)]
    for index, call in enumerate(calls):
        events.append(scope(call["scope_name"], 10 + index * 10, 8))
    for index in range(len(calls) + 1):
        ts = 2 if index == 0 else 11 + (index - 1) * 10
        events.append(
            {
                **scope("cudaMemsetAsync" if index == 0 else "cudaLaunchKernel", ts, 1),
                "cat": "cuda_runtime",
                "args": {"correlation": index + 1},
            }
        )
        events.append(
            {
                "name": "TEST_ONLY_Setup" if index == 0 else "TEST_ONLY_kernel_" + str(index),
                "cat": "gpu_memset" if index == 0 else "kernel",
                "ph": "X",
                "ts": ts + 1,
                "dur": 1,
                "args": {
                    "correlation": index + 1,
                    "stream": 7,
                    **({"bytes": 360} if index == 0 else {"grid": [1, 1, 1], "block": [32, 1, 1], "shared memory": 0}),
                },
            }
        )
    binding = bind_prefill_activity(events, calls, [entry["name"] for entry in entries])
    signatures = dispatch_signatures(binding)
    for rank, forwards in records.items():
        raw = []
        for row in forwards:
            row["ops_instrumented"] = True
            if row["stage"] != "measure":
                continue
            row["native_prefill_measurement_contract"] = METHOD
            row["native_prefill_calls"] = calls
            row["native_prefill_setup"] = {
                "buffer_size": 90,
                "dtype": "torch.float32",
                "device": f"cuda:{rank}",
                "source": "sglang.srt.utils.common.BumpAllocator.__init__",
                "completed": True,
                "latency": 0.002,
                "contribution_count": 1,
            }
            if row["repetition"] == 4:
                name = f"prefill-profile-rank-{rank}-forward-{row['invocation']}.json"
                put(
                    root / name,
                    {
                        "traceEvents": events,
                        "aisim_native_forward": trace_forward_identity(row),
                        "aisim_native_prefill": {
                            "measurement_contract": METHOD,
                            "model_identity_sha256": row["native_prefill_model_sha256"],
                            "native_calls": calls,
                            "setup": {
                                key: row["native_prefill_setup"][key]
                                for key in ("buffer_size", "dtype", "device", "source")
                            },
                            "failed": False,
                        },
                    },
                )
                row["native_prefill_profile"] = {
                    "trace_file": name,
                    "trace_sha256": file_sha256(root / name),
                    "binding": binding,
                }
            for entry in entries:
                shape = json.loads(entry["geometry"])
                attention = entry["component"] == "attention"
                unit = {
                    **{
                        key: row[key]
                        for key in (
                            "backend",
                            "backend_version",
                            "backend_revision",
                            "checkpoint_revision",
                            "source_sha256",
                            "config_sha256",
                            "runtime_digest",
                            "phase",
                            "invocation",
                            "tp_rank",
                            "stage",
                            "benchmark_id",
                            "repetition",
                            "sampling_role",
                            "request_ids",
                            "dataset_role",
                            "request_set",
                            "corpus_sha256",
                        )
                    },
                    **entry,
                    "batch_size": 1,
                    "prefix": 2 if attention else 0,
                    "x": 1,
                    "latency": 0.001,
                    "sample_count": 1,
                    "sample": row["repetition"],
                    "measurement_scope": "local_compute",
                    "kv_seed_regime": "real_kv" if attention else "n/a",
                    "used_cuda_graph": False,
                    "kernel_source": "TEST_ONLY.native." + entry["name"],
                    "state_mode": "chunked_prefill" if attention else "token_only",
                    "history_ids": [row["requests"][0]["previous_forward_id"]],
                    "excluded_collectives": [],
                    "dispatch_kernels": signatures[entry["name"]] if row["repetition"] >= 4 else [],
                    "dispatch_fingerprint": sha256_json(signatures[entry["name"]]) if row["repetition"] >= 4 else "",
                }
                if entry["component"] == "primitive":
                    unit["measurement_scope"] = (
                        "communication"
                        if shape["role"] == "allreduce"
                        else "compute_and_communication"
                        if shape["role"] == "logits"
                        else "local_compute"
                    )
                raw.append(unit)
        put_lines(root / f"forward-rank-{rank}.jsonl", forwards)
        put_lines(root / f"rank-{rank}.jsonl", raw)
    return run, root, records


def control_fixture(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    run, root, records = prefill_fixture(tmp_path, "control")
    # Separate request/run identities while preserving the exact native tokens.
    for path in root.iterdir():
        if path.suffix in (".json", ".jsonl"):
            path.write_text(
                path.read_text()
                .replace("request-", "control-request-")
                .replace('"authored-cpu-fixture"', '"independent-control"')
                .replace("authored-unit-run", "control-unit-run")
            )
    return run, root


def test_full_model_export_rederives_raw_trace_and_public_schema4_query(tmp_path):
    run, root, _ = fixture(tmp_path / "cal")
    control_run, control = control_fixture(tmp_path / "control")
    systems = tmp_path / "systems"
    data = systems / "data/gb300/sglang/0.5.20"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    path = data / export.BASENAME
    result = export.export_prefill(root, run, path, control_root=control, control_run=control_run)
    assert result["rows"] == 367
    native = load_native(run, root)
    binding = export.bind_calibration([path], run, native)
    assert binding["tables"][0]["sha256"] == file_sha256(path)
    from aisimulate_core.sdk.engine import EngineHandle

    engine = EngineHandle.compile(
        "zai-org/GLM-5.3-Flash",
        "gb300",
        "sglang",
        backend_version="0.5.20",
        tp_size=2,
        moe_tp_size=2,
        moe_ep_size=1,
        systems_path=str(systems),
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
    )
    assert engine.predict_prefill_latency(1, 3, 2) == pytest.approx(0.368)
    with pytest.raises(ValueError):
        engine.predict_prefill_latency(1, 4, 3)
    (tmp_path / "holdout").mkdir()
    holdout_run, holdout, _ = prefill_fixture(tmp_path / "holdout", "holdout")
    config = {
        "model": "zai-org/GLM-5.3-Flash",
        "system": "gb300",
        "backend": "sglang",
        "backend_version": "0.5.20",
        "worker_type": "aggregated",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp_size": 2,
        "moe_ep_size": 1,
        "kvcache_quant_mode": "fp8",
        "estimation_mode": "op_level",
        "database_mode": "SILICON",
        "systems_paths": [str(systems)],
        "fallback_policy": "deny",
        "strict_provenance": True,
        "enable_shared_layer": False,
    }
    result = export.predict_homogeneous(holdout_run, holdout, config, native, binding)
    assert result["rows"] == {1: {"prediction_ms": pytest.approx(0.368)}}
    assert result["calibration_binding"] == binding
    damaged = copy.deepcopy(binding)
    damaged["tables"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calibration table/policy"):
        export.predict_homogeneous(holdout_run, holdout, config, native, damaged)
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    rows[0]["latency"] *= 2
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="original native events/ownership"):
        export.bind_calibration([path], run, native)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_setup",
        "legacy_contract",
        "wrong_arguments",
        "zero_with_activity",
        "missing_unit",
        "changed_dispatch",
        "reused_trace",
        "wrong_trace_metadata",
        "missing_activity",
        "wrong_model_source",
        "changed_retained_call",
    ],
)
def test_original_forward_trace_event_closure_rejects_changes(tmp_path, defect):
    run, root, records = fixture(tmp_path)
    targets = [row for row in records[0] if row["stage"] == "measure"]
    row = targets[5]
    if defect == "missing_setup":
        row.pop("native_prefill_setup")
    elif defect == "legacy_contract":
        row.pop("prefill_measurement_contract")
    elif defect == "wrong_arguments":
        row["native_prefill_setup"]["buffer_size"] = 91
    elif defect == "zero_with_activity":
        row["native_prefill_setup"]["latency"] = 0
    elif defect == "changed_retained_call":
        row["native_prefill_calls"] = copy.deepcopy(row["native_prefill_calls"])
        row["native_prefill_calls"][0]["source"] = "TEST_ONLY_CHANGED"
    elif defect == "wrong_model_source":
        path = root / "prefill-model-rank-0.json"
        value = json.loads(path.read_text())
        value["source_pins"]["srt/utils/common.py"] = "0" * 64
        put(path, value)
    elif defect in ("missing_unit", "changed_dispatch"):
        path = root / "rank-0.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if defect == "missing_unit":
            rows.pop()
        else:
            rows[-1]["dispatch_fingerprint"] = "0" * 64
        put_lines(path, rows)
    elif defect == "reused_trace":
        row["native_prefill_profile"] = copy.deepcopy(targets[4]["native_prefill_profile"])
    else:
        profile = targets[4]["native_prefill_profile"]
        path = root / profile["trace_file"]
        value = json.loads(path.read_text())
        if defect == "wrong_trace_metadata":
            value["aisim_native_forward"]["repetition"] = 3
        else:
            value["traceEvents"] = [event for event in value["traceEvents"] if event["cat"] != "gpu_memset"]
        put(path, value)
        profile["trace_sha256"] = file_sha256(path)
    put_lines(root / "forward-rank-0.jsonl", records[0])
    with pytest.raises(ValueError):
        export.read_prefill_run(root, run)


def test_same_token_geometry_keeps_actual_prefix_and_operation_identity(tmp_path):
    run, root, _ = fixture(tmp_path)
    proof = export.read_prefill_run(root, run)
    rows, _ = export.aggregate_prefill(proof, evidence_sha256="a" * 64)
    assert len(rows) == 367
    assert {row["prefix"] for row in rows} == {2}
    assert len([row for row in rows if row["component"] == "mhc"]) == 182
    assert all(set(row) == set(export.COLUMNS) for row in rows)


@pytest.mark.parametrize("defect", ["policy", "reused_requests", "timing", "instrumented"])
def test_independent_control_cannot_change_policy_identity_or_hide_observer_cost(tmp_path, defect):
    run, root, _ = fixture(tmp_path / "cal")
    control_run, control = control_fixture(tmp_path / "control")
    if defect == "policy":
        path = control / "sglang-resolved-config.json"
        value = json.loads(path.read_text())
        value["mem_fraction_static"] = 0.82
        put(path, value)
    elif defect == "reused_requests":
        for path in control.iterdir():
            if path.suffix in (".json", ".jsonl"):
                path.write_text(path.read_text().replace("control-request-", "request-"))
    else:
        for rank in range(2):
            path = control / f"forward-rank-{rank}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if row["stage"] == "measure":
                    if defect == "timing":
                        row["whole_forward_gpu_ms"] *= 2
                    else:
                        row["native_prefill_calls"] = []
            put_lines(path, rows)
    with pytest.raises(ValueError):
        export.profile_control(root, export.read_prefill_run(root, run), control, control_run)


@pytest.mark.parametrize("defect", ["bool", "fractional", "duplicate", "empty"])
def test_frozen_points_require_unique_integer_homogeneous_geometry(tmp_path, defect):
    run, root, _ = fixture(tmp_path)
    if defect == "bool":
        run["points"][0]["batch_size"] = True
    elif defect == "fractional":
        run["points"][0]["batch_size"] = 2
    elif defect == "duplicate":
        run["points"].append(copy.deepcopy(run["points"][0]))
    else:
        run["points"] = []
    with pytest.raises(ValueError, match="frozen point"):
        export.read_prefill_run(root, run)


@pytest.mark.parametrize("changed", ["terminal_output", "actual_input"])
def test_control_compares_actual_model_inputs_and_preserves_independent_terminal_output(tmp_path, changed):
    run, root, _ = fixture(tmp_path / "cal")
    control_run, control = control_fixture(tmp_path / "control")
    for rank in range(2):
        path = control / f"forward-rank-{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            for request in row["requests"]:
                if changed == "actual_input":
                    # Keep a valid original same-request seed chain. Only the
                    # final corpus token differs between these independent
                    # runs; the common reader must still admit each run.
                    request["prompt_token_ids"][-1] = 8
                    if row["stage"] == "measure":
                        request["native_query_token_ids"] = [8]
                        request["input_tokens_sha256"] = sha256_json([4, 5, 8])
                elif row["stage"] == "measure":
                    request["sampled_token_id"] += 1
        put_lines(path, rows)
    native = load_native(control_run, control)
    assert native["values"] == {1: 5.0}
    proof = export.read_prefill_run(root, run)
    if changed == "actual_input":
        with pytest.raises(ValueError, match="changed native inputs/state"):
            export.profile_control(root, proof, control, control_run)
    else:
        result = export.profile_control(root, proof, control, control_run)
        assert result["results"][0]["ratio"] == 1


def test_lookup_analysis_metadata_preserves_original_native_policy_and_evidence(tmp_path):
    run, root, _ = fixture(tmp_path / "cal")
    control_run, control = control_fixture(tmp_path / "control")
    output = tmp_path / export.BASENAME
    export.export_prefill(
        root, run, output, control_root=control, control_run=control_run, lookup_contract=export.LOOKUP_CONTRACT
    )
    import pyarrow.parquet as pq

    rows = pq.read_table(output).to_pylist()
    assert {row["lookup_contract"] for row in rows} == {export.LOOKUP_CONTRACT}
    assert all(len(row["source_ownership_sha256"]) == 64 for row in rows)
    proof = export.read_prefill_run(root, run)
    assert all(json.loads(row["prefill_policy"]) == proof["policy"] for row in rows)
    assert "lookup_contract" not in proof["policy"]
    original = export.verify_evidence(root, proof)
    assert "lookup_contract" not in original
    native = load_native(run, root)
    binding = export.bind_calibration([output], run, native)
    assert binding["lookup_contract"] == export.LOOKUP_CONTRACT
    rows[0]["source_ownership_sha256"] = "f" * 64
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows), output)
    with pytest.raises(ValueError, match="original native events"):
        export.bind_calibration([output], run, native)


@pytest.mark.parametrize(
    "changes",
    [
        {"lookup_contract": "unknown", "source_ownership_sha256": "a" * 64},
        {"lookup_contract": export.LOOKUP_CONTRACT},
        {"source_ownership_sha256": "a" * 64},
        {"lookup_contract": None, "source_ownership_sha256": "a" * 64},
    ],
)
def test_lookup_metadata_cannot_silently_change_legacy_semantics(changes):
    with pytest.raises(ValueError):
        export.table_lookup_contract([changes])


def test_changed_native_calls_reject_even_when_table_geometry_is_equal(tmp_path):
    run, root, _ = fixture(tmp_path / "cal")
    proof = export.read_prefill_run(root, run)
    rows, _ = export.aggregate_prefill(proof, evidence_sha256="a" * 64)
    # Later selected sample calls differ from original fifth-warmup ownership.
    row = proof["forwards"][(1, 5)][0]
    row["whole_forward_gpu_ms"] = 1000
    row["native_prefill_calls"] = copy.deepcopy(row["native_prefill_calls"])
    row["native_prefill_calls"][0]["source"] = "TEST_ONLY.changed_source"
    with pytest.raises(ValueError, match="source ownership"):
        export.analysis_rows(proof, rows, export.LOOKUP_CONTRACT)


def test_existing_legacy_evidence_can_publish_optin_without_rewriting_any_raw(tmp_path):
    run, root, _ = fixture(tmp_path / "cal")
    control_run, control = control_fixture(tmp_path / "control")
    old = tmp_path / export.BASENAME
    export.export_prefill(root, run, old, control_root=control, control_run=control_run)
    before = {str(p): file_sha256(p) for base in (root, control) for p in base.iterdir() if p.is_file()}
    destination = tmp_path / "analysis" / export.BASENAME
    destination.parent.mkdir()
    result = export.publish_prefill(root, run, destination, lookup_contract=export.LOOKUP_CONTRACT)
    assert result["rows"] == 367
    assert {path: file_sha256(__import__("pathlib").Path(path)) for path in before} == before
    bound = export.bind_calibration([destination], run, load_native(run, root))
    assert bound["lookup_contract"] == export.LOOKUP_CONTRACT
