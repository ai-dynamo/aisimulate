# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both predictor inputs must use the reviewed bytes, not a bundled model ID."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from e2e_accuracy_source.model_config_snapshot import (
    canonical_json_bytes,
    materialize_model_config,
    normalize_trt_snapshot,
)


def snapshot(config=None, companion=None):
    config = config or {"architectures": ["GptOssForCausalLM"], "torch_dtype": "bfloat16"}
    return {
        "config": config,
        "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "hf_quant_config": companion,
        "companion_sha256": hashlib.sha256(canonical_json_bytes(companion)).hexdigest()
        if companion is not None
        else None,
        "revision": "a" * 40,
    }


def test_snapshot_atomic_concurrent_materialization(tmp_path):
    data = snapshot(companion={"quantization": {"quant_algo": "NVFP4"}})
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: materialize_model_config(data, cache_root=tmp_path), range(20)))
    assert len(set(paths)) == 1
    target = Path(paths[0])
    assert json.loads((target / "config.json").read_text()) == data["config"]
    assert json.loads((target / "hf_quant_config.json").read_text()) == data["hf_quant_config"]
    assert {p.name for p in tmp_path.iterdir()} == {target.name}


def test_snapshot_rejects_wrong_hash_before_writing(tmp_path):
    data = snapshot()
    data["config"]["torch_dtype"] = "float16"
    with pytest.raises(ValueError, match="config_sha256 mismatch"):
        materialize_model_config(data, cache_root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_snapshot_detects_corrupted_cache(tmp_path):
    data = snapshot()
    path = Path(materialize_model_config(data, cache_root=tmp_path))
    (path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="cache content mismatch"):
        materialize_model_config(data, cache_root=tmp_path)


def test_original_json_text_can_be_verified_without_canonicalizing(tmp_path):
    data = snapshot()
    text = json.dumps(data["config"], indent=4) + "\n"
    data.update(config_json=text, config_sha256=hashlib.sha256(text.encode()).hexdigest())
    path = Path(materialize_model_config(data, cache_root=tmp_path))
    assert (path / "config.json").read_text() == text


def test_trt_normalization_is_auditable_and_does_not_modify_source(tmp_path):
    config = {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        "text_config": {"dtype": "bfloat16", "quantization_config": {"quant_method": "fp8"}},
    }
    data = snapshot(
        config, {"quantization": {"quant_algo": "NVFP4", "group_size": 16, "exclude_modules": ["self_attn*"]}}
    )
    normalized = normalize_trt_snapshot(data)
    assert data["config"] == config
    assert "quantization_config" in data["config"]
    assert "quantization_config" not in normalized["config"]
    assert "quantization_config" not in normalized["config"]["text_config"]
    assert normalized["source_config"] == config
    assert normalized["source_config_sha256"] == data["config_sha256"]
    assert normalized["config_sha256"] != data["config_sha256"]
    assert normalized["hf_quant_config"] == data["hf_quant_config"]
    assert normalize_trt_snapshot(normalized) == normalized
    path = Path(materialize_model_config(normalized, cache_root=tmp_path))
    assert json.loads((path / "config.json").read_text()) == normalized["config"]
    assert normalized["snapshot_transformation"]["removed_paths"] == [
        "quantization_config",
        "text_config.quantization_config",
    ]


def test_trt_without_companion_keeps_inline_metadata():
    data = snapshot({"quantization_config": {"quant_method": "fp8"}})
    assert normalize_trt_snapshot(data) == data


def test_unreviewed_trt_sidecar_stays_unchanged_for_conflict_reporting():
    data = snapshot({"quantization_config": {"quant_method": "fp8"}}, {"quantization": {"quant_algo": "FP8"}})
    assert normalize_trt_snapshot(data) == data
