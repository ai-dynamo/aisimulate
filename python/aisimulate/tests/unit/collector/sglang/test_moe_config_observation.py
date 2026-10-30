# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SGLang MoE collector records, never chooses, the Triton config source."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.sglang.moe_config_observation import (
    append_record,
    config_source,
    observe_triton_config_selection,
    sidecar_path,
)

pytestmark = pytest.mark.unit

DEFAULT = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32}


def _sglang_config_module(tuned):
    """Stand-in for fused_moe_triton_config: tuned JSON lookup, then default."""
    module = SimpleNamespace()
    module.get_moe_configs = lambda *_args, **_kwargs: tuned
    module.get_default_config = lambda *args, **kwargs: dict(DEFAULT)

    def try_get_optimal_moe_config(tokens):
        # Mirrors the v0.5.20 order: tuned map nearest to M, else default.
        configs = module.get_moe_configs(289, 512, "fp8_w8a8", 128, 128)
        if configs:
            return configs[min(configs, key=lambda x: abs(x - tokens))]
        return module.get_default_config(tokens, 289, 512, 4096, 9, "fp8_w8a8", False, [128, 128])

    module.try_get_optimal_moe_config = try_get_optimal_moe_config
    return module


def test_default_config_is_observed_without_changing_the_choice():
    module = _sglang_config_module(tuned=None)
    original = (module.get_moe_configs, module.get_default_config)
    with observe_triton_config_selection([module]) as observed:
        assert module.try_get_optimal_moe_config(512) == DEFAULT
    assert (module.get_moe_configs, module.get_default_config) == original
    assert observed == {"tuned_lookup_hits": 0, "tuned_lookup_misses": 1, "default_config_calls": 1}
    assert config_source(observed) == "default"


def test_tuned_config_is_observed_and_globals_namespaces_are_restored():
    tuned = {1: {"BLOCK_SIZE_M": 16}, 1024: {"BLOCK_SIZE_M": 128}}
    module = _sglang_config_module(tuned=tuned)
    direct = {"get_moe_configs": module.get_moe_configs, "get_default_config": module.get_default_config}
    with observe_triton_config_selection([module, direct]) as observed:
        assert module.try_get_optimal_moe_config(900) == {"BLOCK_SIZE_M": 128}
        assert direct["get_moe_configs"](289, 512, "fp8_w8a8", 128, 128) == tuned
    assert direct["get_moe_configs"] is module.get_moe_configs
    assert config_source(observed) == "tuned" and observed["tuned_lookup_hits"] == 2


def test_wrappers_are_removed_when_the_timed_call_fails():
    module = _sglang_config_module(tuned=None)
    original = module.get_default_config
    with pytest.raises(RuntimeError), observe_triton_config_selection([module]):
        raise RuntimeError("kernel failed")
    assert module.get_default_config is original


def test_record_lands_in_a_sidecar_outside_the_perf_staging_contract(tmp_path):
    perf = tmp_path / "moe_perf.txt"
    observed = {"tuned_lookup_hits": 0, "tuned_lookup_misses": 2, "default_config_calls": 2}
    row = {"moe_dtype": "fp8_block", "num_tokens": 64, "num_experts": 288, "topk": 8}
    record = append_record(str(perf), row, "sglang_fused_moe_triton", observed)
    path = sidecar_path(str(perf))
    assert path == tmp_path / "moe_perf.triton_config.jsonl"
    assert not path.name.endswith("_perf.txt")  # never finalized as a perf table
    assert not perf.exists()
    assert json.loads(path.read_text()) == record
    assert record["triton_config_source"] == "default" and record["kernel_source"] == "sglang_fused_moe_triton"
    assert config_source({"tuned_lookup_hits": 0, "tuned_lookup_misses": 0, "default_config_calls": 0}) == (
        "not_observed"
    )


def test_sglang_collector_wraps_the_timed_call_and_writes_triton_rows_only():
    source = (Path(__file__).resolve().parents[4] / "collector/sglang/collect_moe.py").read_text()
    assert "with observe_triton_config_selection(_triton_config_namespaces())" in source
    assert "return [sys.modules[get_config_dtype_str.__module__], globals()]" in source
    # one task times a num_tokens series: each point's row gets its own counters
    assert re.search(r'if moe_backend == "triton":\n\s+append_triton_config_record\(', source)
    assert "point_observed = dict(triton_config_observed)" in source
    assert "triton_config_observed[key] = 0" in source
