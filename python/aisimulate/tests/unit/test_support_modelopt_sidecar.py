# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local ModelOpt metadata survives guided initialization and collector planning."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from collector.fpm_forward.cli import _parser
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.planner import build_collection_plan

import aisimulate.main as cli
from aisimulate.support.config_profile import derive_profile, load_model_config
from aisimulate.support.fpm import fpm_cli_args
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit


def _snapshot(tmp_path):
    source = Path(__file__).parents[2] / "src/aisimulate_core/model_configs"
    for filename in ("config.json", "hf_quant_config.json"):
        shutil.copyfile(source / f"nvidia--Llama-3.1-70B-Instruct-FP8_{filename}", tmp_path / filename)
    return tmp_path / "config.json"


@pytest.mark.parametrize(
    "algorithm,kv,gemm,moe", [("FP8", "FP8", "fp8_static", "fp8"), ("NVFP4", "none", "nvfp4", "nvfp4")]
)
@pytest.mark.parametrize(
    "metadata_layout",
    ["flat", "nested-modelopt", "null-fields", "dynamic-conflict", "inline-conflict", "nested-inline-conflict"],
)
@pytest.mark.parametrize("duplicate_sidecar", [False, True])
def test_public_init_with_modelopt_sidecar_plans_same_checkpoint(
    tmp_path, algorithm, kv, gemm, moe, metadata_layout, duplicate_sidecar
):
    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    if metadata_layout == "nested-modelopt":
        raw["quantization_config"] = {"quant_method": "modelopt"}
    elif metadata_layout == "null-fields":
        raw.update(quant_algo=None, quant_dynamic=None, kv_cache_quant_algo=None)
    elif metadata_layout == "dynamic-conflict":
        raw.update(
            quant_dynamic=False, quantization_config={"quant_method": "modelopt", "activation_scheme": "dynamic"}
        )
    elif metadata_layout in {"inline-conflict", "nested-inline-conflict"}:
        raw["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4" if algorithm == "FP8" else "FP8",
        }
    sidecar = tmp_path / "hf_quant_config.json"
    metadata = json.loads(sidecar.read_text())
    metadata["quantization"].update(quant_algo=algorithm, kv_cache_quant_algo=kv)
    if duplicate_sidecar:
        raw["hf_quant_config"] = metadata
    if metadata_layout.startswith("nested"):
        raw = {"architectures": ["ExampleMultimodalForConditionalGeneration"], "text_config": raw}
    path.write_text(json.dumps(raw))
    sidecar.write_text(json.dumps(metadata))
    kv_dtype = "fp8" if kv == "FP8" else "bfloat16"
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"fmha_quant_mode": kv_dtype, "kv_cache_dtype": kv_dtype}))
    output = tmp_path / "request.yaml"
    argv = [
        "onboard",
        "init",
        "--model-config",
        str(path),
        "--model",
        "nvidia/Llama-3.1-70B-Instruct-FP8",
        "--model-revision",
        "a" * 40,
        "--framework-version",
        "0.27.0",
        "--gpu",
        "h200_sxm",
        "--interconnect",
        "nvswitch",
        "--resource-overrides",
        str(overrides),
        "--context-length",
        "16384",
        "--tensor-parallel",
        "2",
        "--output",
        str(output),
    ]
    if metadata_layout.endswith("conflict"):
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 2
        assert not output.exists()
        from collector.fpm_forward.model_capability import load_model_config as collector_model_config

        conflict = "quant_dynamic" if metadata_layout == "dynamic-conflict" else "quant_algo"
        with pytest.raises(ValueError, match=f"conflicting {conflict}"):
            collector_model_config(str(path))
        return
    assert cli.main(argv) == 0
    request = SupportRequest.from_yaml(output)
    deployment = request.profile_deployment()
    assert (deployment.gemm_quant_mode, deployment.moe_quant_mode) == (gemm, moe)
    assert deployment.fmha_quant_mode == deployment.kv_cache_dtype == kv_dtype
    provenance = json.loads(request.fpm_profile.provenance)
    assert provenance["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["config_source_files"] == {
        "hf_quant_config.json": hashlib.sha256(sidecar.read_bytes()).hexdigest()
    }
    assert str(sidecar) in provenance["config_notes"]["quantization_sidecar"]
    assert "weights_bytes" not in provenance["planning_estimates"]
    assert "user override" in provenance["fields"]["kv_cache_dtype"]["source"]
    args = _parser().parse_args(fpm_cli_args(request, output_dir=tmp_path / "collection", plan_only=True)[3:])
    plan = build_collection_plan(
        backend="vllm",
        model_path=request.identity.model,
        model_architecture=args.model_architecture,
        model_config_path=str(path),
        system="h200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        options=FPMCollectionOptions.from_args(args),
        fpm_profile=request.fpm_profile,
    )
    assert {cell.weight_quantization for cell in plan.cells} == {gemm}


def test_sidecar_weights_do_not_supply_runtime_attention_or_unspecified_kv(tmp_path):
    path = _snapshot(tmp_path)
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "NVFP4"}}))
    draft = derive_profile(load_model_config(path))
    assert draft.resolved["gemm_quant_mode"] == draft.resolved["moe_quant_mode"] == "nvfp4"
    assert {"fmha_quant_mode", "kv_cache_dtype"} <= draft.missing.keys()
    assert "weights_bytes" not in draft.resolved


@pytest.mark.parametrize(
    "inline",
    [
        {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        {"quant_method": "fp8", "kv_cache_scheme": {"num_bits": 16, "type": "float"}},
        {"quant_method": "fp8", "kv_cache_scheme": {"num_bits": 8, "type": "int"}},
    ],
)
@pytest.mark.parametrize("duplicate_sidecar", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("consumer", ["onboarding", "collection"])
def test_conflicting_inline_and_sidecar_metadata_is_rejected(tmp_path, inline, duplicate_sidecar, nested, consumer):
    from collector.fpm_forward.model_capability import load_model_config as collector_model_config

    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    raw["quantization_config"] = inline
    if duplicate_sidecar:
        raw["hf_quant_config"] = json.loads((tmp_path / "hf_quant_config.json").read_text())
    if nested:
        raw = {"architectures": ["ExampleMultimodalForConditionalGeneration"], "text_config": raw}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="conflicting.*hf_quant_config"):
        if consumer == "onboarding":
            load_model_config(path)
        else:
            collector_model_config(str(path))


@pytest.mark.parametrize("top_level", [None, "fp8"])
def test_top_level_algorithm_does_not_hide_contradictory_inline_metadata(tmp_path, top_level):
    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    raw.update(quant_algo=top_level, quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"})
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="conflicting.*hf_quant_config"):
        load_model_config(path)


def test_explicit_precision_override_remains_visible_with_sidecar(tmp_path):
    config = load_model_config(_snapshot(tmp_path))
    draft = derive_profile(config, overrides={"gemm_quant_mode": "bfloat16", "kv_cache_dtype": "bfloat16"})
    assert draft.resolved["gemm_quant_mode"] == draft.resolved["kv_cache_dtype"] == "bfloat16"
    assert "user override" in draft.sources["gemm_quant_mode"]
    assert "user override" in draft.sources["kv_cache_dtype"]


@pytest.mark.parametrize(
    "inline,gemm,moe",
    [
        ({"quant_method": "modelopt"}, "fp8_static", "fp8"),
        ({"quant_method": "fp8", "activation_scheme": "dynamic"}, "fp8", "fp8"),
        ({"quant_method": "fp8", "weight_block_size": [128, 128]}, "fp8_block", "fp8_block"),
    ],
)
@pytest.mark.parametrize("duplicate_sidecar", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_matching_inline_quantization_retains_activation_and_block_metadata(
    tmp_path, inline, gemm, moe, duplicate_sidecar, nested
):
    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    raw["quantization_config"] = inline
    if duplicate_sidecar:
        raw["hf_quant_config"] = json.loads((tmp_path / "hf_quant_config.json").read_text())
    if nested:
        raw = {"architectures": ["ExampleMultimodalForConditionalGeneration"], "text_config": raw}
    path.write_text(json.dumps(raw))
    draft = derive_profile(load_model_config(path))
    assert (draft.resolved["gemm_quant_mode"], draft.resolved["moe_quant_mode"]) == (gemm, moe)
    assert draft.resolved["kv_cache_dtype"] == "fp8"
    assert "fmha_quant_mode" in draft.missing


def test_sidecar_conflict_with_string_kv_scheme_is_rejected(tmp_path):
    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    raw["quantization_config"] = {"kv_cache_scheme": "FP8"}
    path.write_text(json.dumps(raw))
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "none"}})
    )
    with pytest.raises(ValueError, match="conflicting.*hf_quant_config"):
        load_model_config(path)


@pytest.mark.parametrize("scheme,kv", [("FP8", "fp8"), ({"num_bits": 8, "type": "int"}, "int8")])
def test_weight_only_sidecar_preserves_explicit_inline_kv_scheme(tmp_path, scheme, kv):
    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    raw["quantization_config"] = {"kv_cache_scheme": scheme}
    path.write_text(json.dumps(raw))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "FP8"}}))
    draft = derive_profile(load_model_config(path))
    assert draft.resolved["kv_cache_dtype"] == kv
    assert draft.resolved["gemm_quant_mode"] == "fp8_static"


@pytest.mark.parametrize("sidecar", [{}, {"format": "unknown"}, {"quantization": {"quant_algo": "custom"}}])
def test_unknown_sidecar_does_not_reintroduce_unquantized_defaults(tmp_path, sidecar):
    path = _snapshot(tmp_path)
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(sidecar))
    draft = derive_profile(load_model_config(path))
    assert {"gemm_quant_mode", "moe_quant_mode"} <= draft.missing.keys()
    assert "weights_bytes" not in draft.resolved


def test_nested_decoder_and_hf_snapshot_symlinks_keep_adjacent_sidecar(tmp_path):
    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    path.write_text(json.dumps({"architectures": ["ExampleMultimodalForConditionalGeneration"], "text_config": raw}))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for filename in ("config.json", "hf_quant_config.json"):
        (snapshot / filename).symlink_to(tmp_path / filename)
    config = load_model_config(snapshot / "config.json")
    draft = derive_profile(config)
    assert draft.resolved["gemm_quant_mode"] == "fp8_static"
    assert draft.resolved["kv_cache_dtype"] == "fp8"
    assert str(snapshot / "hf_quant_config.json") in config.notes["quantization_sidecar"]
    assert config.raw["hidden_size"] == raw["hidden_size"]
    assert "text_config" not in config.raw


@pytest.mark.parametrize("layout", ["flat", "nested", "nested-inline"])
def test_already_supported_sidecar_keeps_frozen_collector_source_identity(tmp_path, layout):
    from collector.fpm_forward.model_capability import load_model_config as collector_model_config

    from aisimulate_core.sdk.utils import _attach_hf_quant_config

    path = _snapshot(tmp_path)
    raw = json.loads(path.read_text())
    if layout == "nested-inline":
        raw["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", "kv_cache_scheme": "FP8"}
    if layout != "flat":
        raw = {"architectures": ["ExampleMultimodalForConditionalGeneration"], "text_config": raw}
        path.write_text(json.dumps(raw))
    sidecar = json.loads((tmp_path / "hf_quant_config.json").read_text())
    # This is the unchanged general SDK attachment used before this FPM fix.
    baseline = _attach_hf_quant_config(raw, sidecar)
    resolved = collector_model_config(str(path))
    assert resolved.payload == baseline
    assert (
        resolved.sha256
        == hashlib.sha256(
            json.dumps(baseline, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    assert resolved.effective_payload["quant_algo"] == "fp8"


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"quantization": []}',
        '{"quantization": {"quant_algo": "FP8", "quant_algo": "NVFP4"}}',
        '{"quantization": {"quant_algo": "FP8", "quantization_algo": "NVFP4"}}',
    ],
)
def test_malformed_sidecar_fails_before_profile_creation(tmp_path, payload):
    path = _snapshot(tmp_path)
    (tmp_path / "hf_quant_config.json").write_text(payload)
    with pytest.raises(ValueError):
        load_model_config(path)
