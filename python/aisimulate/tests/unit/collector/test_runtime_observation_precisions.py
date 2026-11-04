# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.capabilities import resolve_model_capability
from collector.fpm_forward.runtime import fpm_memory_observer as observer
from collector.fpm_forward.runtime_observations import validate_observations

from .test_fpm_runtime_memory import _typed
from .test_runtime_observations import _mutate, _replace_launch, observation_fixture, stage_model_config

pytestmark = pytest.mark.unit


def _precision_fixture(tmp_path, *, cache_dtype="fp8", storage_dtype="torch.uint8", item_size=1, precision="fp8"):
    path, launches = observation_fixture(tmp_path, version="0.27.0")
    launches["tp2"]["precision"]["kvcache_quant_mode"] = precision
    _replace_launch(path, launches)
    if precision != "bfloat16":
        # The import takes the FMHA label of this KV cache from the configuration that the probe stages.
        stage_model_config(path)
    spec = _typed("vllm.v1.kv_cache_interface", "FullAttentionSpec")
    spec.block_size, spec.page_size_bytes, spec.dtype = 16, 64, storage_dtype
    observed_dtype = observer.cache_groups(
        SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer0"], kv_cache_spec=spec)])
    )[0]["dtype"]

    def change(record):
        record["resolved_config"]["cache_config"]["cache_dtype"] = cache_dtype
        for group in record["cache"]["groups"]:
            group["dtype"] = observed_dtype
        for tensor in record["cache"].get("layer_tensors", {}).values():
            tensor["shape"][-1] = 64 // item_size
            tensor["stride_bytes"][-1] = item_size
            tensor["element_size_bytes"] = item_size

    _mutate(path, change)
    return path, launches


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_fp8_uint8_storage_import_uses_resolved_precision_and_preserves_physical_evidence(tmp_path, cache_dtype):
    path, launches = _precision_fixture(tmp_path, cache_dtype=cache_dtype)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 83 * 128
    provenance = result["provenance"]
    assert provenance["runtime_settings"]["cache_config"]["cache_dtype"] == cache_dtype
    assert len(provenance["artifacts"]) == 12
    for artifact in provenance["artifacts"]:
        record = artifact["evidence"]
        assert {group["dtype"] for group in record["cache"]["groups"]} == {"torch.uint8"}
        for tensor in record["cache"].get("layer_tensors", {}).values():
            assert tensor["element_size_bytes"] == 1


@pytest.mark.parametrize("cache_dtype,storage_dtype,precision", [("bfloat16", "torch.bfloat16", "bfloat16")])
def test_unquantized_cache_storage_import_remains_supported(tmp_path, cache_dtype, storage_dtype, precision):
    path, launches = _precision_fixture(
        tmp_path, cache_dtype=cache_dtype, storage_dtype=storage_dtype, item_size=2, precision=precision
    )
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 83 * 128


def test_float16_cache_storage_import_is_refused_like_collection(tmp_path):
    # FPM collection resolves no FMHA label for a float16 KV cache, because AIC has no such KV-cache mode. The import
    # takes the label of a KV cache other than bfloat16 from collection's rule, so it refuses the cache with
    # collection's text.
    path, launches = _precision_fixture(
        tmp_path, cache_dtype="float16", storage_dtype="torch.float16", item_size=2, precision="half"
    )
    result = validate_observations(path, launches)["tp2"]
    assert result["diagnostics"] == [
        "tp2: unsupported FPM KV-cache dtype 'half'; AIC modes are ['bfloat16', 'fp8', 'int8']"
    ]


@pytest.mark.parametrize("fmha", ["half", "bfloat16"])
def test_float16_checkpoint_with_an_fp8_kv_cache_imports_the_fmha_label_of_collection(tmp_path, fmha):
    path, launches = _precision_fixture(tmp_path)
    launch = launches["tp2"]
    model = Path(launch["model_config"]["path"])
    model.write_text(json.dumps({"model_type": "example", "torch_dtype": "float16"}))
    launch["model_config"]["sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    launch["precision"].update(gemm_quant_mode="half", moe_quant_mode="half", fmha_quant_mode=fmha)
    _replace_launch(path, launches)

    def float16(record):
        record["model_config_sha256"] = launch["model_config"]["sha256"]
        record["resolved_config"]["model_config"].update(dtype="torch.float16", quantization=None)
        record["resolved_config"]["quantization_config"] = None

    _mutate(path, float16)
    stage_model_config(path)
    # The inference names no FMHA mode for this unquantized checkpoint, so collection labels the fp8 KV cache with
    # bfloat16, not with the compute dtype half that the import used to accept.
    capability = resolve_model_capability(
        backend="vllm",
        model_path=launch["identity"]["model"],
        model_architecture=None,
        selected_ops=set(),
        has_model_cases=False,
        system="gb300",
        requested_weight_quantizations=(),
        requested_kv_cache_dtypes=("fp8",),
        model_config_path=str(model),
        database_version="0.27.0",
        checkpoint_native_dtypes=True,
    )
    assert capability.dtype.fmha_by_kv_dtype == {"fp8": "bfloat16"}
    result = validate_observations(path, launches)["tp2"]
    if fmha == "bfloat16":
        assert result["status"] == "complete", result["diagnostics"]
    else:
        assert result["diagnostics"] == [
            "tp2: FMHA precision 'half' differs from the checkpoint-native 'bfloat16' for KV-cache dtype 'fp8'"
        ]


@pytest.mark.parametrize(
    "cache_dtype",
    [None, "auto", "bfloat16", "fp8_e5m2", "fp8_per_token_head", "int4_per_token_head", "nvfp4"],
)
def test_uint8_storage_does_not_establish_missing_or_other_semantic_precision(tmp_path, cache_dtype):
    path, launches = _precision_fixture(tmp_path, cache_dtype=cache_dtype)
    if cache_dtype is None:
        _mutate(path, lambda record: record["resolved_config"]["cache_config"].pop("cache_dtype"))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None
    assert "KV precision" in " ".join(result["diagnostics"])


def test_uint8_storage_is_not_accepted_for_validated_bf16_cache(tmp_path):
    path, launches = _precision_fixture(tmp_path, cache_dtype="bfloat16", precision="bfloat16")
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert "cache group dtype" in " ".join(result["diagnostics"])


@pytest.mark.parametrize("storage_dtype", ["torch.uint8", "fp8"])
def test_fp8_storage_requires_one_byte_tensor_elements(tmp_path, storage_dtype):
    path, launches = _precision_fixture(tmp_path, storage_dtype=storage_dtype, item_size=2)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert result["resources"] is None
    assert "tensor element size" in " ".join(result["diagnostics"])


def test_fp8_semantics_cannot_relabel_bf16_storage(tmp_path):
    path, launches = _precision_fixture(tmp_path, storage_dtype="torch.bfloat16", item_size=2)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert "cache group dtype" in " ".join(result["diagnostics"])
