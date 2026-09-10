# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from compare_forward import compare_cases, error_summary


def sample():
    return {
        "case_id": "decode-0",
        "phase": "generation",
        "batch_size": 2,
        "query": 1,
        "prefix": 128,
        "canonical_past_kv": 128,
        "native_inclusive_kv": 129,
        "rank_max_ms": [9, 10, 11],
        "median_ms": 10,
    }


def test_distinct_forward_axes_and_signed_error():
    def predict(metrics):
        return metrics["scheduled_requests"]["sum_decode_kv_tokens"] / 20

    op = compare_cases([sample()], predict, forward_model="op_level")[0]
    fpm = compare_cases([sample()], predict, forward_model="fpm")[0]
    assert op["predicted_ms"] == 12.9
    assert fpm["predicted_ms"] == 12.8
    assert op["signed_error_percent"] == pytest.approx(29)


def test_missing_prediction_does_not_improve_accuracy_or_coverage():
    def absent(_):
        raise ValueError("missing exact prefix bucket")

    missing = compare_cases([sample()], absent, forward_model="op_level")
    assert error_summary(missing) == {"planned_points": 1, "predicted_points": 0}
    measured = compare_cases([sample()], lambda _: 8, forward_model="op_level")
    summary = error_summary(missing + measured)
    assert summary["planned_points"] == 2 and summary["predicted_points"] == 1
    assert summary["mean_signed_error_percent"] == pytest.approx(-20)
    assert summary["wape_percent"] == 20


def test_incomplete_or_changed_observations_are_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        compare_cases([sample() | {"rank_max_ms": [10]}], lambda _: 10, forward_model="op_level")
    with pytest.raises(ValueError, match="differs"):
        compare_cases([sample() | {"median_ms": 11}], lambda _: 10, forward_model="op_level")


def config():
    return {
        "model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "system_name": "gb300",
        "backend": "sglang",
        "backend_version": "0.0.0.dev0",
        "tp_size": 4,
        "moe_tp_size": 4,
        "moe_ep_size": 1,
        "enable_shared_layer": False,
        "strict_provenance": True,
        "database_mode": "SILICON",
        "systems_path": "study/full/systems",
    }


@pytest.mark.parametrize("mode", ["SOL", "HYBRID", "SILICON"])
@pytest.mark.parametrize("profile", ["full", "decoder_bounded"])
def test_qualified_current_modes_and_profile_defaults(mode, profile):
    from compare_forward import validate_prediction_contract

    value = config() | {"database_mode": mode, "decoder_replay": profile == "decoder_bounded"}
    validate_prediction_contract(value, {"execution_profile": profile})
    validate_prediction_contract(
        value | {"pp_size": 1, "attention_dp_size": 1, "cp_size": 1, "nextn": 0}, {"execution_profile": profile}
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend_version", "0.5.14"),
        ("pp_size", 2),
        ("attention_dp_size", 2),
        ("cp_size", 2),
        ("nextn", 3),
        ("tp_size", True),
        ("strict_provenance", False),
        ("strict_provenance", 1),
        ("enable_shared_layer", True),
        ("decoder_replay", "false"),
        ("database_mode", "EMPIRICAL"),
        ("weight_dtype", "bf16"),
        ("moe_dtype", "fp8"),
        ("activation_dtype", "fp16"),
        ("kv_cache_dtype", "fp8"),
        ("gemm_quant_mode", "bf16"),
        ("systems_path", None),
        ("extra", {"decoder_replay": "true"}),
        ("transfer_policy", ["xshape"]),
    ],
)
def test_contract_rejects_unqualified_or_ignored_configuration(field, value):
    from compare_forward import validate_prediction_contract

    with pytest.raises(ValueError):
        validate_prediction_contract(config() | {field: value}, {"execution_profile": "full"})


@pytest.mark.parametrize("field", ["enable_shared_layer", "strict_provenance", "backend_version"])
def test_contract_requires_explicit_policy(field):
    from compare_forward import validate_prediction_contract

    value = config()
    del value[field]
    with pytest.raises(ValueError):
        validate_prediction_contract(value, {"execution_profile": "full"})


def overlay(tmp_path):
    root = tmp_path / "private-location" / "systems"
    data = root / "data/gb300/dsv41/sglang/0.0.0.dev0"
    data.mkdir(parents=True)
    (root / "gb300.yaml").write_text("data_dir: data/gb300\ngpu: {sm_version: 103}\n")
    (data / "dsv41_module_perf.parquet").write_bytes(b"first table content")
    (data / "collection_meta.yaml").write_text("schema_version: 1\n")
    return root, data


def test_system_inventory_binds_actual_data_metadata_and_spec_without_private_paths(tmp_path):
    import json

    from compare_forward import system_data_identity

    root, data = overlay(tmp_path)
    before = system_data_identity(root)
    assert "private-location" not in json.dumps(before)
    assert str(tmp_path) not in json.dumps(before)
    assert len(before["files_sha256"]) == 3
    (data / "dsv41_module_perf.parquet").write_bytes(b"changed table content")
    changed = system_data_identity(root)
    assert changed["inventory_sha256"] != before["inventory_sha256"]
    (data / "collection_meta.yaml").write_text("schema_version: 2\n")
    assert system_data_identity(root)["inventory_sha256"] != changed["inventory_sha256"]


@pytest.mark.parametrize("kind", ["file_link", "directory_link", "absolute_data_dir", "parent_data_dir"])
def test_overlay_cannot_hide_external_inputs(tmp_path, kind):
    from compare_forward import system_data_identity

    root, data = overlay(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    file = outside / "table.parquet"
    file.write_bytes(b"external table")
    if kind == "file_link":
        (data / "external.parquet").symlink_to(file)
    elif kind == "directory_link":
        (root / "external").symlink_to(outside, target_is_directory=True)
    else:
        (root / "gb300.yaml").write_text(f"data_dir: {outside if kind == 'absolute_data_dir' else '../outside'}\n")
    with pytest.raises(ValueError):
        system_data_identity(root)


def test_model_identity_requires_same_checkpoint_and_actual_resolved_precision(tmp_path, monkeypatch):
    import hashlib
    from copy import deepcopy

    from compare_forward import canonical, resolved_model_identity

    from aiconfigurator_core.sdk import utils

    checkpoint = {"text_config": {"hidden_size": 5120}, "architectures": ["example"]}
    file = tmp_path / "deepseek-ai--DeepSeek-V4.1-Flash_config.json"
    file.write_text(canonical(checkpoint))
    resolved = checkpoint | {"quant_algo": "FP8"}
    monkeypatch.setattr(utils, "_get_model_config_path", lambda: tmp_path)
    monkeypatch.setattr(utils, "_load_pre_downloaded_hf_config", lambda _: deepcopy(checkpoint))
    monkeypatch.setattr(utils, "_attach_inferred_quant_fields", lambda raw: raw | {"quant_algo": "FP8"})
    monkeypatch.setattr(utils, "get_model_config_from_model_path", lambda _: {"raw_config": resolved})
    observation = {"input_provenance": {"config_sha256": hashlib.sha256(canonical(checkpoint).encode()).hexdigest()}}
    identity = resolved_model_identity(config(), observation)
    assert identity["checkpoint_config_canonical_sha256"] != identity["resolved_config_canonical_sha256"]
    resolved["quant_algo"] = "BF16"
    with pytest.raises(ValueError, match="resolved model"):
        resolved_model_identity(config(), observation)
    with pytest.raises(ValueError, match="checkpoint"):
        resolved_model_identity(config(), {"input_provenance": {"config_sha256": "wrong"}})
