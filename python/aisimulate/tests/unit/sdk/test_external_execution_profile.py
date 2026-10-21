# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API contract with synthetic timings, no hardware-specific fixtures."""

import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aisimulate_core import _native
from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk import engine as engine_module
from aisimulate_core.sdk.errors import PerfDataNotAvailableError

pytestmark = pytest.mark.unit


@pytest.fixture
def profile(tmp_path, monkeypatch):
    root = tmp_path / "systems"
    root.mkdir()
    bundled = Path(engine_module.__file__).parents[1] / "systems"
    hardware = root / "h200_sxm.yaml"
    hardware.write_bytes((bundled / hardware.name).read_bytes())
    # Match the bundled data_dir but supply an otherwise empty local version.
    version = root / "data/h200_sxm/gemm/vllm/1.0"
    version.mkdir(parents=True)
    (version / "placeholder.txt").write_text("synthetic fixture")
    identity = {
        "schema_version": 1,
        "model_name": "test/model",
        "system_name": "h200_sxm",
        "systems_path": str(root),
        "backend": "vllm",
        "backend_version": "1.0",
        "forward_model": "op_level",
        "kv_block_size": 16,
        "tp_size": 1,
        "pp_size": 1,
        "weight_dtype": "bfloat16",
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "database_mode": "SILICON",
        "enable_shared_layer": False,
    }
    spec = {
        "schema_version": _native.engine_spec_schema_version(),
        "engine": identity,
        "context_ops": [],
        "generation_ops": [],
    }
    spec_bytes = bytes(_native.engine_spec_bincode_from_json(json.dumps(spec)))
    # The real canonical constructor, native admission, database loader and
    # execution remain active; only checkpoint-dependent graph compilation is stubbed.
    monkeypatch.setattr(engine_module, "compile_engine", lambda *args, **kwargs: spec_bytes)
    measured = {
        "name": "context_measured_scope",
        "system": "h200_sxm",
        "backend": "vllm",
        "version": "1.0",
        "is_context": True,
        "weight_bytes": 0.0,
        "scale_factor": 3.0,
        "cells": [
            {"shape": [1, 4, 0], "latency_ms": 2.0},
            {"shape": [1, 4, 4], "latency_ms": 5.0},
        ],
    }
    payload = {
        "schema_version": 1,
        "basis": json.loads(_native.engine_spec_json_from_bincode(spec_bytes)),
        "worker_type": "prefill",
        "ops": [{"MeasuredComposite": measured}],
        "retained_files": [
            {
                "path": str(p.relative_to(root)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ],
    }
    path = tmp_path / "profile.json"
    config = {
        "model": "test/model",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "1.0",
        "worker_type": "prefill",
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "database_mode": "SILICON",
        "enable_shared_layer": False,
        "systems_paths": [str(root)],
        "estimator_config": {"correction": {"enabled": False}, "op_level": {}},
    }

    def publish():
        path.write_text(json.dumps(payload))
        config["estimator_config"]["op_level"]["execution_profile"] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    publish()
    return config, payload, publish, root, path


def test_exact_cells_prefix_and_saved_config(profile):
    config, _, _, _, _ = profile
    model = RustForwardPassPerfModel.best_available(config)
    # Independent synthetic oracle: three repetitions of 2 ms or 5 ms.
    assert model.predict_prefill_latency(1, 4, 0) == 6.0
    assert model.predict_prefill_latency(1, 8, 4) == 15.0
    saved = model.diagnostics()["provenance"]["config"]
    assert saved["estimator_config"]["op_level"] == config["estimator_config"]["op_level"]
    assert RustForwardPassPerfModel.best_available(saved).predict_prefill_latency(1, 4, 0) == 6.0
    for shape in [(1, 5, 0), (2, 4, 0), (1, 8, 0)]:
        with pytest.raises(ValueError, match="no exact cell"):
            model.predict_prefill_latency(*shape)
    with pytest.raises(ValueError, match="phase is not qualified"):
        model.predict_decode_latency_total(1, 4)


def test_mutation_is_rejected_and_existing_model_owns_snapshot(profile):
    config, _, _, root, path = profile
    model = RustForwardPassPerfModel.best_available(config)
    (root / "h200_sxm.yaml").write_text("changed after construction")
    # First query after mutation still uses verified, engine-owned storage.
    assert model.predict_prefill_latency(1, 4, 0) == 6.0
    with pytest.raises(ValueError, match="retained input changed"):
        RustForwardPassPerfModel.best_available(config)
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        RustForwardPassPerfModel.best_available(config)
    assert model.predict_prefill_latency(1, 4, 0) == 6.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("estimation_mode", "auto"),
        ("fallback_policy", "allow"),
        ("database_mode", "HYBRID"),
        ("enable_shared_layer", True),
        ("worker_type", "aggregated"),
        ("nextn", 1),
        ("pp", 2),
        ("attention_dp", 2),
        ("cp_size", 2),
        ("dcp", 2),
        ("backend_version", "latest"),
    ],
)
def test_rejects_unsupported_modes_before_compilation(profile, field, value):
    config, _, _, _, _ = profile
    config[field] = value
    with pytest.raises(ValueError, match="external execution profile"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize(
    "change,pattern",
    [
        ("baseline", "compiled baseline"),
        ("weights", "resident weights"),
        ("runtime", "identity or phase"),
        ("phase", "identity or phase"),
        ("duplicate", "unique and sorted"),
        ("negative", "finite, positive"),
        ("empty", "invalid measured scope"),
        ("traversal", "bundle-relative"),
        ("duplicate_file", "duplicate retained"),
        ("extra_field", "unknown field"),
    ],
)
def test_profile_admission_fails_closed(profile, change, pattern):
    config, payload, publish, _, _ = profile
    op = payload["ops"][0]["MeasuredComposite"]
    if change == "baseline":
        payload["basis"]["engine"]["tp_size"] = 2
    elif change == "weights":
        op["weight_bytes"] = 1024
    elif change == "runtime":
        op["version"] = "another"
    elif change == "phase":
        op["is_context"] = False
        op["cells"] = op["cells"][:1]
    elif change == "duplicate":
        op["cells"].append(copy.deepcopy(op["cells"][-1]))
    elif change == "negative":
        op["cells"][0]["latency_ms"] = -2
    elif change == "empty":
        op["cells"] = []
    elif change == "traversal":
        payload["retained_files"][0]["path"] = "../outside"
    elif change == "duplicate_file":
        payload["retained_files"].append(copy.deepcopy(payload["retained_files"][0]))
    else:
        payload["unknown"] = True
    publish()
    with pytest.raises(ValueError, match=pattern):
        RustForwardPassPerfModel.best_available(config)


def test_decode_profile_and_separate_profile_identity(profile):
    config, payload, publish, _, _ = profile
    prefill = RustForwardPassPerfModel.best_available(config)
    payload["worker_type"] = config["worker_type"] = "decode"
    op = payload["ops"][0]["MeasuredComposite"]
    op["name"] = "generation_measured_scope"
    op["is_context"] = False
    op["cells"] = [{"shape": [2, 5, 0], "latency_ms": 7.0}]
    publish()
    decode = RustForwardPassPerfModel.best_available(config)
    assert decode.predict_decode_latency_total(2, 8) == 21.0
    assert prefill.predict_prefill_latency(1, 4, 0) == 6.0
    with pytest.raises(ValueError, match="homogeneous"):
        decode.predict_decode_latency_total(2, 9)
    with pytest.raises(ValueError, match="phase is not qualified"):
        decode.predict_prefill_latency(1, 4, 0)


def test_measured_scope_bincode_roundtrip(profile):
    _, payload, _, _, _ = profile
    spec = copy.deepcopy(payload["basis"])
    spec["context_ops"] = payload["ops"]
    encoded = bytes(_native.engine_spec_bincode_from_json(json.dumps(spec)))
    assert json.loads(_native.engine_spec_json_from_bincode(encoded)) == spec


def test_retained_symlink_is_rejected(profile):
    config, _, _, root, _ = profile
    hardware = root / "h200_sxm.yaml"
    target = root.parent / "outside.yaml"
    hardware.rename(target)
    hardware.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        RustForwardPassPerfModel.best_available(config)


def test_pinned_hardware_cannot_escape_snapshot(profile):
    config, payload, publish, root, _ = profile
    hardware = root / "h200_sxm.yaml"
    hardware.write_text(hardware.read_text().replace("data/h200_sxm", "/outside"))
    entry = next(e for e in payload["retained_files"] if e["path"] == hardware.name)
    entry.update(sha256=hashlib.sha256(hardware.read_bytes()).hexdigest(), size_bytes=hardware.stat().st_size)
    publish()
    with pytest.raises(ValueError, match="stay inside the bundle"):
        RustForwardPassPerfModel.best_available(config)


def test_profile_rejects_aggregate_telemetry(profile):
    config, _, _, _, _ = profile
    model = RustForwardPassPerfModel.best_available(config)
    with pytest.raises(ValueError, match="direct homogeneous phase queries"):
        model.estimate_forward_pass_time_ms(
            [{"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 4}}]
        )


def _communication_profile(profile, kind, version):
    config, payload, publish, root, _ = profile
    hardware = root / "h200_sxm.yaml"
    data = yaml.safe_load(hardware.read_bytes())
    data["misc"].pop("nccl_version", None)
    data["misc"][f"{kind}_version"] = version
    hardware.write_text(yaml.safe_dump(data))
    payload["ops"].append(
        {
            "Nccl": {
                "name": "context_collective",
                "scale_factor": 1.0,
                "hidden_size": 128.0,
                "num_gpus": 2,
                "dtype": "half",
                "operation": "all_reduce",
                "seq_split": 1,
            }
        }
    )
    payload["retained_files"] = [
        {
            "path": str(p.relative_to(root)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "size_bytes": p.stat().st_size,
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    publish()
    return config


def _communication_table(path, latency):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "op_name": ["all_reduce"],
                "nccl_dtype": ["half"],
                "num_gpus": [2],
                "message_size": [512],
                "latency": [latency],
            }
        ),
        path,
    )


@pytest.mark.parametrize("kind", ["nccl", "oneccl"])
@pytest.mark.parametrize("escape", ["absolute", "parent"])
def test_communication_versions_cannot_escape_snapshot(profile, kind, escape):
    _, _, _, root, _ = profile
    external = root.parent / "unretained"
    _communication_table(external / f"{kind}_perf.parquet", 1.0)
    if escape == "absolute":
        version = str(external)
    else:
        # Keep each prefix directory in the snapshot so traversal reaches a
        # real external table, rather than failing on an absent directory.
        anchor = root / f"data/h200_sxm/comm/{kind}/anchor"
        anchor.parent.mkdir(parents=True)
        anchor.write_text("retained directory marker")
        version = "../" * 32 + str(external).lstrip("/")
    config = _communication_profile(profile, kind, version)
    with pytest.raises(ValueError, match="stay inside the bundle"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("kind", ["nccl", "oneccl"])
@pytest.mark.parametrize("layout", ["comm/", ""])
def test_first_communication_query_uses_retained_snapshot(profile, kind, layout):
    _, _, _, root, _ = profile
    table = root / f"data/h200_sxm/{layout}{kind}/test/{kind}_perf.parquet"
    _communication_table(table, 1.0)
    config = _communication_profile(profile, kind, "test")
    model = RustForwardPassPerfModel.best_available(config)
    _communication_table(table, 99.0)
    # Independent oracle: 3 * 2 ms measured scope + the retained 1 ms collective.
    assert model.predict_prefill_latency(1, 4, 0) == 7.0
    with pytest.raises(ValueError, match="retained input changed"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("kind", ["nccl", "oneccl"])
def test_communication_table_must_be_retained(profile, kind):
    _, payload, publish, root, _ = profile
    table = root / f"data/h200_sxm/comm/{kind}/test/{kind}_perf.parquet"
    _communication_table(table, 1.0)
    config = _communication_profile(profile, kind, "test")
    payload["retained_files"] = [f for f in payload["retained_files"] if f["path"] != str(table.relative_to(root))]
    publish()
    with pytest.raises(PerfDataNotAvailableError, match="required op-level data unavailable"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("complete", [False, True])
def test_decode_profile_rejects_trajectories_even_with_all_exact_cells(profile, complete):
    config, payload, publish, _, _ = profile
    payload["worker_type"] = config["worker_type"] = "decode"
    op = payload["ops"][0]["MeasuredComposite"]
    op.update(name="generation_measured_scope", is_context=False)
    latencies = [2.0, 5.0, 7.0, 11.0] if complete else [2.0]
    op["cells"] = [{"shape": [2, 5 + i, 0], "latency_ms": t} for i, t in enumerate(latencies)]
    publish()
    model = RustForwardPassPerfModel.best_available(config)
    # Independent synthetic oracle: three repetitions of each measured cell.
    expected = [6.0, 15.0, 21.0, 33.0] if complete else [6.0]
    assert [model.predict_decode_latency_total(2, 8 + 2 * i) for i in range(len(latencies))] == expected
    assert model.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=2, prefill=False) == 6.0
    with pytest.raises(ValueError, match="one decode step"):
        model.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=5, prefill=False)


@pytest.mark.parametrize("metadata", [False, True])
def test_native_per_op_queries_enforce_profile_phase(profile, metadata):
    _, payload, _, root, _ = profile
    spec = copy.deepcopy(payload["basis"])
    spec["context_ops"] = payload["ops"]
    spec["engine"]["extra"]["execution_profile_phase"] = "prefill"
    handle = _native.AicEngine.from_spec(bytes(_native.engine_spec_bincode_from_json(json.dumps(spec))), str(root))
    decode = handle._decode_step_per_op_with_metadata if metadata else handle.decode_step_per_op
    for batch in [0, 1]:
        with pytest.raises(ValueError, match="phase is not qualified"):
            decode(batch, 4, 1)
    spec["context_ops"] = []
    spec["engine"]["extra"]["execution_profile_phase"] = "decode"
    handle = _native.AicEngine.from_spec(bytes(_native.engine_spec_bincode_from_json(json.dumps(spec))), str(root))
    run = handle._run_static_per_op_with_metadata if metadata else handle.run_static_per_op
    for mode in ["static_ctx", "static"]:
        with pytest.raises(ValueError, match="phase is not qualified"):
            run(1, 1, 4, 2, 0, 1.0, 1.0, mode=mode)


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("stride", [1, 32])
def test_native_per_op_decode_rejects_profile_trajectory(profile, metadata, stride):
    _, payload, _, root, _ = profile
    spec = copy.deepcopy(payload["basis"])
    spec["engine"]["extra"]["execution_profile_phase"] = "decode"
    op = copy.deepcopy(payload["ops"][0]["MeasuredComposite"])
    op.update(name="generation_measured_scope", is_context=False, cells=[{"shape": [2, 5, 0], "latency_ms": 2.0}])
    spec["generation_ops"] = [{"MeasuredComposite": op}]
    handle = _native.AicEngine.from_spec(bytes(_native.engine_spec_bincode_from_json(json.dumps(spec))), str(root))
    run = handle._run_static_per_op_with_metadata if metadata else handle.run_static_per_op
    assert run(2, 1, 4, 2, 0, 1.0, 1.0, mode="static_gen", stride=stride)[1][0][1] == 6.0
    assert run(2, 1, 4, 1, 0, 1.0, 1.0, mode="static_gen", stride=stride) == ([], [])
    with pytest.raises(ValueError, match="one decode step"):
        run(2, 1, 4, 5, 0, 1.0, 1.0, mode="static_gen", stride=stride)
