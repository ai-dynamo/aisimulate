# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API contract with synthetic timings, no hardware-specific fixtures."""

import copy
import hashlib
import json
import shutil
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
        "schema_version": 2,
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
    [evidence] = model.static_phase_diagnostics(batch_size=1, context_length=8, prefix=4, prefill=True)
    assert (evidence["latency_ms"], evidence["energy_wms"], evidence["source"]) == (15.0, 0.0, "silicon")
    saved = model.diagnostics()["provenance"]["config"]
    assert saved["estimator_config"]["op_level"] == config["estimator_config"]["op_level"]
    assert RustForwardPassPerfModel.best_available(saved).predict_prefill_latency(1, 4, 0) == 6.0
    for shape in [(1, 5, 0), (2, 4, 0), (1, 8, 0)]:
        with pytest.raises(ValueError, match="no exact cell"):
            model.predict_prefill_latency(*shape)
    with pytest.raises(ValueError, match="phase is not qualified"):
        model.predict_decode_latency_total(1, 4)


@pytest.mark.parametrize(
    "change,pattern",
    [
        ("missing", "external execution profile"),
        ("malformed", "expected ident"),
        ("pin", "SHA-256"),
        ("worker", "worker_type differs"),
    ],
)
def test_config_normalization_defers_profile_loading_to_construction(profile, change, pattern):
    config, payload, publish, _, path = profile
    selection = config["estimator_config"]["op_level"]["execution_profile"]
    if change == "missing":
        path.unlink()
    elif change == "malformed":
        path.write_text("not JSON")
        selection["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == "pin":
        path.write_text("{}")
    else:
        payload["worker_type"] = "decode"
        publish()
    normalized = RustForwardPassPerfModel.normalize_config(config)
    assert normalized["estimator_config"]["op_level"] == config["estimator_config"]["op_level"]
    saved = json.loads(json.dumps(normalized))
    assert RustForwardPassPerfModel.normalize_config(saved) == normalized
    with pytest.raises(ValueError, match=pattern):
        RustForwardPassPerfModel.best_available(saved)


@pytest.mark.parametrize("field,value", [("path", "relative/profile.json"), ("sha256", "invalid-pin")])
def test_config_normalization_still_validates_profile_selection(profile, field, value):
    config, _, _, _, _ = profile
    config["estimator_config"]["op_level"]["execution_profile"][field] = value
    with pytest.raises(ValueError, match="absolute path with a lowercase SHA-256 pin"):
        RustForwardPassPerfModel.normalize_config(config)


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
        ("nextn", 1),
        ("pp", 2),
        ("attention_dp", 2),
        ("cp_size", 2),
        ("dcp", 2),
        ("backend_version", "latest"),
        ("encoder_parallel", "tp"),
        ("encoder_parallel", "dp"),
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
    assert decode.predict_decode_latency_total(2, 10) == 21.0
    [evidence] = decode.static_phase_diagnostics(batch_size=2, context_length=4, prefill=False)
    assert (evidence["latency_ms"], evidence["energy_wms"], evidence["source"]) == (21.0, 0.0, "silicon")
    assert prefill.predict_prefill_latency(1, 4, 0) == 6.0
    # The common API retains ordinary integer mean handling for ragged totals.
    assert decode.predict_decode_latency_total(2, 11) == 21.0
    with pytest.raises(ValueError, match="no exact cell"):
        decode.predict_decode_latency_total(2, 9)
    with pytest.raises(ValueError, match="phase is not qualified"):
        decode.predict_prefill_latency(1, 4, 0)


def test_legacy_decode_profile_preserves_pinned_coordinate(profile):
    config, payload, publish, _, _ = profile
    payload["schema_version"] = 1
    payload["worker_type"] = config["worker_type"] = "decode"
    op = payload["ops"][0]["MeasuredComposite"]
    op.update(name="generation_measured_scope", is_context=False, scale_factor=1.0)
    op["cells"] = [
        {"shape": [2, 4, 0], "latency_ms": 10.0},
        {"shape": [2, 5, 0], "latency_ms": 20.0},
    ]
    for cells in [op["cells"], op["cells"][1:]]:
        payload["schema_version"] = 1
        op["cells"] = cells
        publish()
        model = RustForwardPassPerfModel.best_available(config)
        # v1 total decode has always queried past KV / batch + 1.
        assert model.predict_decode_latency_total(2, 8) == 20.0
        assert model.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=2, prefill=False) == 20.0
        with pytest.raises(ValueError, match="homogeneous"):
            model.predict_decode_latency_total(2, 9)
        with pytest.raises(ValueError, match="one decode step"):
            model.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=3, prefill=False)
        saved = model.diagnostics()["provenance"]["config"]
        assert RustForwardPassPerfModel.best_available(saved).predict_decode_latency_total(2, 8) == 20.0
        with pytest.raises(ValueError, match="schema_version 2"):
            model.estimate_forward_pass_time_ms(
                [{"scheduled_requests": {"num_decode_requests": 2, "sum_decode_kv_tokens": 8}}]
            )
        # Moving to the shared coordinate is explicit and changes the content pin.
        payload["schema_version"] = 2
        publish()
        assert (
            config["estimator_config"]["op_level"]["execution_profile"]["sha256"]
            != saved["estimator_config"]["op_level"]["execution_profile"]["sha256"]
        )
        migrated = RustForwardPassPerfModel.best_available(config)
        metrics = [{"scheduled_requests": {"num_decode_requests": 2, "sum_decode_kv_tokens": 8}}]
        if len(cells) == 2:
            assert migrated.predict_decode_latency_total(2, 8) == 10.0
            assert migrated.estimate_forward_pass_time_ms(metrics) == 10.0
        else:
            with pytest.raises(ValueError, match="no exact cell"):
                migrated.predict_decode_latency_total(2, 8)
        assert migrated.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=2, prefill=False) == 20.0
        assert model.predict_decode_latency_total(2, 8) == 20.0


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


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_profile_common_forward_pass_api(profile, phase):
    config, payload, publish, _, _ = profile
    config["worker_type"] = payload["worker_type"] = phase
    op = payload["ops"][0]["MeasuredComposite"]
    op["is_context"] = phase == "prefill"
    op["cells"] = [{"shape": [2, 5, 0], "latency_ms": 2.0}]
    publish()
    model = RustForwardPassPerfModel.best_available(config)
    scheduled = (
        {"num_prefill_requests": 2, "sum_prefill_tokens": 10}
        if phase == "prefill"
        else {"num_decode_requests": 2, "sum_decode_kv_tokens": 10}
    )
    metrics = [{"scheduled_requests": scheduled}]
    # Independent oracle: three repetitions of the measured 2 ms scope.
    assert model.estimate_forward_pass_time_ms(metrics) == 6.0
    detailed = model.estimate_forward_pass_detailed(metrics)
    assert detailed["latency_ms"] == detailed["native_latency_ms"] == 6.0
    saved = model.diagnostics()["provenance"]["config"]
    assert RustForwardPassPerfModel.best_available(saved).estimate_forward_pass_time_ms(metrics) == 6.0
    assert model.estimate_forward_pass_time_ms([{"scheduled_requests": {}}]) == 0.0
    assert model.estimate_forward_pass_time_ms(metrics * 2) == 6.0
    count = "num_prefill_requests" if phase == "prefill" else "num_decode_requests"
    tokens = "sum_prefill_tokens" if phase == "prefill" else "sum_decode_kv_tokens"
    variance = "var_prefill_length" if phase == "prefill" else "var_decode_kv_tokens"
    for extra, pattern in [
        ({tokens: scheduled[tokens] + 2}, "no exact cell"),
        ({count: 0}, "token sum|token sums"),
    ]:
        with pytest.raises(ValueError, match=pattern):
            model.estimate_forward_pass_time_ms([{"scheduled_requests": scheduled | extra}])
    # Preserve the existing integer-mean and variance handling.
    assert (
        model.estimate_forward_pass_time_ms(
            [{"scheduled_requests": scheduled | {tokens: scheduled[tokens] + 1, variance: 1.0}}]
        )
        == 6.0
    )
    with pytest.raises(ValueError, match="phase is not qualified"):
        model.estimate_forward_pass_time_ms(
            [
                {
                    "scheduled_requests": {
                        "num_prefill_requests": 2,
                        "sum_prefill_tokens": 10,
                        "num_decode_requests": 2,
                        "sum_decode_kv_tokens": 8,
                    }
                }
            ]
        )
    other_phase = (
        {"num_decode_requests": 2, "sum_decode_kv_tokens": 8}
        if phase == "prefill"
        else {"num_prefill_requests": 2, "sum_prefill_tokens": 10}
    )
    with pytest.raises(ValueError, match="phase is not qualified"):
        model.estimate_forward_pass_time_ms([{"scheduled_requests": other_phase}])


def test_profile_common_prefill_prefix_geometry(profile):
    config, _, _, _, _ = profile
    model = RustForwardPassPerfModel.best_available(config)
    # One request has four new tokens and four past KV tokens: 5 ms x 3.
    scheduled = {"num_prefill_requests": 1, "sum_prefill_tokens": 4, "sum_prefill_kv_tokens": 4}
    assert model.estimate_forward_pass_time_ms([{"scheduled_requests": scheduled}]) == 15.0
    # The ordinary path reaches the operator; this fixture has no batch-2 cell.
    with pytest.raises(ValueError, match="no exact cell"):
        model.estimate_forward_pass_time_ms(
            [
                {
                    "scheduled_requests": scheduled
                    | {"num_prefill_requests": 2, "sum_prefill_tokens": 8, "sum_prefill_kv_tokens": 8}
                }
            ]
        )


@pytest.fixture
def aggregated_profile(profile):
    config, payload, publish, _, _ = profile
    config["worker_type"] = payload["worker_type"] = "aggregated"
    template = payload["ops"][0]["MeasuredComposite"]

    def scope(name, context, cells):
        return {
            "MeasuredComposite": template
            | {
                "name": name,
                "is_context": context,
                "cells": [{"shape": shape, "latency_ms": ms} for shape, ms in sorted(cells)],
            }
        }

    payload["ops"] = [
        scope("shared_compute", True, [([2, 5, 0], 3.0), ([1, 12, 0], 5.0), ([1, 7, 4], 13.0)]),
        scope("context_attention", True, [([2, 5, 0], 2.0), ([1, 5, 4], 17.0)]),
    ]
    payload["generation_ops"] = [
        scope("decode_compute", False, [([2, 5, 0], 11.0)]),
        scope("generation_attention", False, [([2, 5, 0], 7.0)]),
    ]
    publish()
    return profile


def test_aggregated_profile_common_api_and_phase_queries(aggregated_profile):
    config, _, _, _, _ = aggregated_profile
    model = RustForwardPassPerfModel.best_available(config)
    prefill = {"num_prefill_requests": 2, "sum_prefill_tokens": 10}
    decode = {"num_decode_requests": 2, "sum_decode_kv_tokens": 10}
    metrics = [{"scheduled_requests": prefill | decode}]
    # Independent measured oracle: (5 shared + 2 prefill + 7 decode) ms x 3.
    # Common op-level queries use the supplied per-request KV length directly.
    assert model.estimate_forward_pass_time_ms(metrics) == 42.0
    detailed = model.estimate_forward_pass_detailed(metrics)
    assert detailed["latency_ms"] == detailed["native_latency_ms"] == 42.0
    assert model.estimate_forward_pass_time_ms([{"scheduled_requests": prefill}]) == 15.0
    assert model.predict_prefill_latency(2, 5, 0) == 15.0
    assert model.estimate_forward_pass_time_ms([{"scheduled_requests": decode}]) == 54.0
    assert model.predict_decode_latency_total(2, 10) == 54.0
    assert model.estimate_forward_pass_time_ms([{}]) == 0.0
    saved = model.diagnostics()["provenance"]["config"]
    assert RustForwardPassPerfModel.best_available(saved).estimate_forward_pass_time_ms(metrics) == 42.0
    prefix = {"num_prefill_requests": 1, "sum_prefill_tokens": 5, "sum_prefill_kv_tokens": 4}
    # One prefix-aware prefill plus decode: (13 + 17 + 7) ms x 3.
    assert model.estimate_forward_pass_time_ms([{"scheduled_requests": prefix | decode}]) == 111.0
    with pytest.raises(ValueError, match="no exact cell"):
        model.estimate_forward_pass_time_ms([{"scheduled_requests": prefill | decode | {"sum_decode_kv_tokens": 12}}])


@pytest.mark.parametrize("decode_latency", [70.0, 7000.0])
def test_opaque_mixed_scopes_fail_instead_of_omitting_decode(aggregated_profile, decode_latency):
    config, payload, publish, _, _ = aggregated_profile
    prefill = payload["ops"][0]["MeasuredComposite"]
    prefill.update(name="whole_prefill", scale_factor=1.0, cells=[{"shape": [1, 12, 0], "latency_ms": 100.0}])
    decode = payload["generation_ops"][0]["MeasuredComposite"]
    decode.update(name="whole_decode", scale_factor=1.0, cells=[{"shape": [2, 5, 0], "latency_ms": decode_latency}])
    payload["ops"] = [{"MeasuredComposite": prefill}]
    payload["generation_ops"] = [{"MeasuredComposite": decode}]
    publish()
    model = RustForwardPassPerfModel.best_available(config)
    assert model.predict_prefill_latency(1, 12, 0) == 100.0
    assert model.predict_decode_latency_total(2, 10) == decode_latency
    for kv in [10, 12]:
        metrics = [
            {
                "scheduled_requests": {
                    "num_prefill_requests": 2,
                    "sum_prefill_tokens": 10,
                    "num_decode_requests": 2,
                    "sum_decode_kv_tokens": kv,
                }
            }
        ]
        for query in [model.estimate_forward_pass_time_ms, model.estimate_forward_pass_detailed]:
            with pytest.raises(ValueError, match="top-level phase attention"):
                query(metrics)


@pytest.mark.parametrize(
    "change,pattern",
    [
        ("missing_decode", "require generation_ops"),
        ("legacy_schema", "schema_version 2"),
        ("wrong_phase", "identity or phase"),
        ("weights", "resident weights"),
        ("single_phase", "require an aggregated profile"),
    ],
)
def test_aggregated_profile_validates_phase_inputs(aggregated_profile, change, pattern):
    config, payload, publish, _, _ = aggregated_profile
    if change == "legacy_schema":
        payload["schema_version"] = 1
    elif change == "missing_decode":
        del payload["generation_ops"]
    elif change == "wrong_phase":
        payload["generation_ops"][1]["MeasuredComposite"]["is_context"] = True
    elif change == "weights":
        payload["generation_ops"][0]["MeasuredComposite"]["weight_bytes"] = 1e6
    else:
        config["worker_type"] = payload["worker_type"] = "prefill"
    publish()
    with pytest.raises(ValueError, match=pattern):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("role", ["prefill", "decode", "aggregated"])
def test_external_inputs_preserve_builtin_predictions(tmp_path, monkeypatch, role):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    bundled = Path(engine_module.__file__).parents[1] / "systems"
    root = tmp_path / "systems"
    # Copy the same published runtime inventory; no mocked compilation or predictions.
    files = [bundled / "b200_sxm.yaml", *sorted((bundled / "data/b200_sxm").glob("*/vllm/0.24.0/*"))]
    for source in files:
        destination = root / source.relative_to(bundled)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config = dict(
        model="Qwen/Qwen3-0.6B",
        system="b200_sxm",
        backend="vllm",
        backend_version="0.24.0",
        worker_type=role,
        estimation_mode="op_level",
        fallback_policy="deny",
        database_mode="SILICON",
        enable_shared_layer=False,
        estimator_config={"correction": {"enabled": False}},
    )
    builtin = RustForwardPassPerfModel.best_available(config)
    local_config = config | {"systems_paths": [str(root)]}
    local = RustForwardPassPerfModel.best_available(local_config)
    basis = json.loads(
        _native.engine_spec_json_from_bincode(
            engine_module.compile_engine(
                config["model"],
                config["system"],
                config["backend"],
                config["backend_version"],
                systems_path=str(root),
                database_mode="SILICON",
                shared_layer=False,
                forward_model="op_level",
            )
        )
    )
    payload = dict(
        schema_version=2,
        basis=basis,
        worker_type=role,
        ops=basis["generation_ops" if role == "decode" else "context_ops"],
        retained_files=[
            dict(
                path=str(p.relative_to(root)),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                size_bytes=p.stat().st_size,
            )
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ],
    )
    if role == "aggregated":
        payload["generation_ops"] = basis["generation_ops"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload))
    external_config = copy.deepcopy(local_config)
    external_config["estimator_config"]["op_level"] = {
        "execution_profile": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    }
    external = RustForwardPassPerfModel.best_available(external_config)
    restored = RustForwardPassPerfModel.best_available(external.diagnostics()["provenance"]["config"])
    prefill = {"num_prefill_requests": 2, "sum_prefill_tokens": 128}
    decode = {"num_decode_requests": 2, "sum_decode_kv_tokens": 256}
    cases = [prefill] if role == "prefill" else [decode]
    if role == "aggregated":
        cases = [prefill, decode, prefill | decode]
    # Existing op-level mean/variance handling must also be independent of input origin.
    cases += [
        cases[-1]
        | (
            {"sum_prefill_tokens": 129, "sum_prefill_kv_tokens": 64, "var_prefill_length": 7.0}
            if role != "decode"
            else {"sum_decode_kv_tokens": 257, "var_decode_kv_tokens": 9.0}
        ),
        {},
    ]
    for scheduled in cases:
        metrics = [{"scheduled_requests": scheduled}]
        expected = builtin.estimate_forward_pass_time_ms(metrics)
        assert expected == 0.0 if not scheduled else expected > 0.0
        for model in [local, external, restored]:
            assert model.estimate_forward_pass_time_ms(metrics) == expected
            assert model.estimate_forward_pass_detailed(metrics) == builtin.estimate_forward_pass_detailed(metrics)
    if role != "decode":
        assert external.predict_prefill_latency_total(2, 129, 64) == builtin.predict_prefill_latency_total(2, 129, 64)
    if role != "prefill":
        assert external.predict_decode_latency_total(2, 257) == builtin.predict_decode_latency_total(2, 257)


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
def test_decode_profile_reuses_static_trajectory_sampling(profile, complete):
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
    assert [model.predict_decode_latency_total(2, 10 + 2 * i) for i in range(len(latencies))] == expected
    assert model.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=2, prefill=False) == 6.0
    # The existing static API samples at stride 32 and repeats the first step.
    assert model.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=5, prefill=False) == 24.0
    with pytest.raises(ValueError, match="no exact cell"):
        model.static_phase_latency(batch_size=2, input_tokens=4, output_tokens=34, prefill=False)


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_native_per_op_queries_enforce_profile_phase(profile, metadata, phase):
    _, payload, _, root, _ = profile
    spec = copy.deepcopy(payload["basis"])
    spec["context_ops"] = payload["ops"] if phase == "prefill" else []
    spec["engine"]["extra"]["execution_profile_phase"] = phase
    handle = _native.AicEngine.from_spec(bytes(_native.engine_spec_bincode_from_json(json.dumps(spec))), str(root))
    if phase == "prefill":
        decode = handle._decode_step_per_op_with_metadata if metadata else handle.decode_step_per_op
        for batch in [0, 1]:
            with pytest.raises(ValueError, match="phase is not qualified"):
                decode(batch, 4, 1)
    else:
        run = handle._run_static_per_op_with_metadata if metadata else handle.run_static_per_op
        for mode in ["static_ctx", "static"]:
            with pytest.raises(ValueError, match="phase is not qualified"):
                run(1, 1, 4, 2, 0, 1.0, 1.0, mode=mode)


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("stride", [1, 32])
def test_native_per_op_decode_reports_missing_trajectory_cells(profile, metadata, stride):
    _, payload, _, root, _ = profile
    spec = copy.deepcopy(payload["basis"])
    spec["engine"]["extra"]["execution_profile_phase"] = "decode"
    spec["engine"]["extra"]["execution_profile_schema_version"] = "2"
    op = copy.deepcopy(payload["ops"][0]["MeasuredComposite"])
    op.update(name="generation_measured_scope", is_context=False, cells=[{"shape": [2, 5, 0], "latency_ms": 2.0}])
    spec["generation_ops"] = [{"MeasuredComposite": op}]
    handle = _native.AicEngine.from_spec(bytes(_native.engine_spec_bincode_from_json(json.dumps(spec))), str(root))
    run = handle._run_static_per_op_with_metadata if metadata else handle.run_static_per_op
    assert run(2, 1, 4, 2, 0, 1.0, 1.0, mode="static_gen", stride=stride)[1][0][1] == 6.0
    assert run(2, 1, 4, 1, 0, 1.0, 1.0, mode="static_gen", stride=stride) == ([], [])
    with pytest.raises(ValueError, match="no exact cell"):
        run(2, 1, 4, 34, 0, 1.0, 1.0, mode="static_gen", stride=stride)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_native_ad_hoc_queries_enforce_profile_phase(profile, phase):
    _, payload, _, root, _ = profile
    spec = copy.deepcopy(payload["basis"])
    spec["context_ops"] = payload["ops"] if phase == "prefill" else []
    spec["engine"]["extra"]["execution_profile_phase"] = phase
    handle = _native.AicEngine.from_spec(bytes(_native.engine_spec_bincode_from_json(json.dumps(spec))), str(root))
    qualified = phase == "prefill"
    opposite = [
        lambda: handle.evaluate_ops_json("[]", not qualified, 1, 4),
        lambda: handle.evaluate_ops_sol_json("[]", not qualified, 1, 4),
    ]
    if phase == "decode":
        opposite.append(lambda: handle.evaluate_context_attention_kernels_json("[]", 1, 4))
    for query in opposite:
        with pytest.raises(ValueError, match="phase is not qualified"):
            query()
    # The qualified phase still serves ad-hoc lists.
    assert handle.evaluate_ops_json("[]", qualified, 1, 4) == []
    assert handle.evaluate_ops_sol_json("[]", qualified, 1, 4) == []
    if qualified:
        assert handle.evaluate_context_attention_kernels_json("[]", 1, 4) == []


def test_basis_mismatch_names_the_differing_field(profile):
    config, payload, publish, _, _ = profile
    payload["basis"]["engine"]["tp_size"] = 2
    publish()
    with pytest.raises(ValueError, match=r"profile basis at engine\.tp_size \(compiled 1, profile 2\)"):
        RustForwardPassPerfModel.best_available(config)
