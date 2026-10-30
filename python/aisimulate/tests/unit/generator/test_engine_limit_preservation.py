# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import shlex
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from aisimulate.generator.api import generate_backend_artifacts
from aisimulate.generator.module_bridge import task_config_to_generator_config
from aisimulate.generator.rendering.engine import render_backend_templates
from aisimulate.generator.rendering.rule_engine import apply_rule_plugins
from aisimulate.generator.request import from_legacy_params, to_legacy_params


def _params(*, preserve: bool, rule: str | None = None) -> dict:
    params = {
        "preserve_engine_limits": preserve,
        "ServiceConfig": {},
        "DynConfig": {"mode": "disagg"},
        "WorkerConfig": {"prefill_workers": 1, "decode_workers": 1},
        "SlaConfig": {"isl": 128, "osl": 16},
        "ModelConfig": {"is_moe": False, "prefix": 0, "nextn": 0},
        "params": {
            "prefill": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "max_batch_size": 128,
                "max_num_tokens": 32768,
                "max_prefill_tokens": 32768,
                "max_seq_len": 8192,
                "tokens_per_block": 64,
                "cache_transceiver_max_tokens_in_buffer": 32768,
            },
            "decode": {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "max_batch_size": 256,
                "max_num_tokens": 8192,
                "max_prefill_tokens": 8192,
                "max_seq_len": 8192,
                "tokens_per_block": 64,
                "cache_transceiver_max_tokens_in_buffer": 8192,
            },
        },
    }
    if rule is not None:
        params["rule"] = rule
    return params


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("rule", [None, "benchmark"])
def test_externally_evaluated_engine_limits_are_preserved(backend: str, rule: str | None):
    params = _params(preserve=True, rule=rule)

    result = apply_rule_plugins(copy.deepcopy(params), backend)

    for role in ("prefill", "decode"):
        expected = params["params"][role]
        actual = result["params"][role]
        for key in (
            "max_batch_size",
            "max_num_tokens",
            "max_prefill_tokens",
            "max_seq_len",
            "cache_transceiver_max_tokens_in_buffer",
        ):
            assert actual[key] == expected[key]


def test_default_vllm_rules_still_resize_unpinned_requests():
    result = apply_rule_plugins(_params(preserve=False), "vllm")

    assert result["params"]["prefill"]["max_num_tokens"] == 1628
    assert result["params"]["decode"]["max_batch_size"] == 512
    assert result["params"]["decode"]["max_num_tokens"] == 512


@pytest.mark.parametrize("rule", [None, "benchmark"])
@pytest.mark.parametrize("block_size", [64, None])
def test_trtllm_preserved_limits_still_default_missing_transfer_buffer(rule, block_size):
    params = _params(preserve=True, rule=rule)
    params["params"]["prefill"].pop("cache_transceiver_max_tokens_in_buffer")
    if block_size is None:
        for role in ("prefill", "decode"):
            params["params"][role].pop("tokens_per_block")

    result = apply_rule_plugins(copy.deepcopy(params), "trtllm")

    if block_size is None:
        assert result["params"]["prefill"].get("cache_transceiver_max_tokens_in_buffer") is None
    else:
        assert result["params"]["prefill"]["cache_transceiver_max_tokens_in_buffer"] > 0
    assert result["params"]["decode"]["cache_transceiver_max_tokens_in_buffer"] == 8192
    for role in ("prefill", "decode"):
        for key in ("max_batch_size", "max_num_tokens", "max_seq_len"):
            assert result["params"][role][key] == params["params"][role][key]


@pytest.mark.parametrize("rule", [None, "benchmark"])
def test_trtllm_transfer_buffer_is_not_sized_without_request_lengths(rule):
    params = _params(preserve=True, rule=rule)
    params["SlaConfig"] = {}
    for role in ("prefill", "decode"):
        params["params"][role].pop("cache_transceiver_max_tokens_in_buffer")

    result = apply_rule_plugins(copy.deepcopy(params), "trtllm")

    for role in ("prefill", "decode"):
        assert result["params"][role].get("cache_transceiver_max_tokens_in_buffer") is None
        assert result["params"][role]["cache_transceiver_backend"] == "DEFAULT"


@pytest.mark.parametrize("rule", [None, "benchmark"])
def test_trtllm_explicit_transfer_backend_is_rendered(rule):
    params = _params(preserve=True, rule=rule)
    for role in ("prefill", "decode"):
        params["params"][role]["cache_transceiver_backend"] = "NIXL"

    artifacts = render_backend_templates(apply_rule_plugins(copy.deepcopy(params), "trtllm"), "trtllm")

    for role in ("prefill", "decode"):
        engine = yaml.safe_load(artifacts[f"extra_engine_args_{role}.yaml"])
        assert engine["cache_transceiver_config"]["backend"] == "NIXL"


@pytest.mark.parametrize("rule", [None, "benchmark"])
@pytest.mark.parametrize(
    ("block_size", "max_seq_len", "sla", "keep_buffer", "expected"),
    [
        (None, 8192, {}, False, "tokens_per_block is unset"),
        (64, 10000, {}, False, "max_seq_len 10000 is not a multiple of tokens_per_block 64"),
        (64, 8192, {}, False, None),
        (64, 10000, {"isl": 128, "osl": 16}, False, None),
        (None, 10000, {}, True, None),
    ],
    ids=["unset-block", "trace-unaligned", "trace-aligned", "rule-sized", "explicit-buffer"],
)
def test_public_trtllm_generation_warns_on_runtime_sized_transfer_buffer(
    caplog, rule, block_size, max_seq_len, sla, keep_buffer, expected
):
    # Direct generation bypasses the Sweeper bridge, so the renderer must warn.
    params = _params(preserve=True, rule=rule)
    params["SlaConfig"] = sla
    for role in ("prefill", "decode"):
        role_params = params["params"][role]
        role_params["max_seq_len"] = max_seq_len
        if not keep_buffer:
            role_params.pop("cache_transceiver_max_tokens_in_buffer")
        if block_size is None:
            role_params.pop("tokens_per_block")

    with caplog.at_level(logging.WARNING, logger="aisimulate.generator.rendering.engine"):
        generate_backend_artifacts(params, "trtllm")

    messages = [record.getMessage() for record in caplog.records if "tokens_per_block" in record.getMessage()]
    if expected is None:
        assert messages == []
    else:
        assert len(messages) == 2
        for message, role in zip(messages, ("prefill", "decode"), strict=True):
            assert message.startswith(f"TRT-LLM {role} {expected}")


@pytest.mark.parametrize("rule", [None, "benchmark"])
def test_trtllm_transfer_settings_stay_off_aggregated_workers(rule):
    params = _params(preserve=True, rule=rule)
    params["DynConfig"] = {"mode": "agg"}
    params["WorkerConfig"] = {"agg_workers": 1}
    params["params"] = {"agg": params["params"]["prefill"]}
    params["params"]["agg"].pop("cache_transceiver_max_tokens_in_buffer")

    result = apply_rule_plugins(copy.deepcopy(params), "trtllm")

    assert "cache_transceiver_backend" not in result["params"]["agg"]
    assert result["params"]["agg"].get("cache_transceiver_max_tokens_in_buffer") is None


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_sflow_benchmark_replays_sla_lengths_without_benchmark_lengths(backend):
    artifacts = render_backend_templates(apply_rule_plugins(_params(preserve=True), backend), backend)

    sflow = yaml.safe_load(artifacts["sflow.yaml"])
    assert "name: benchmark" in artifacts["sflow.yaml"]
    assert (sflow["variables"]["ISL"]["value"], sflow["variables"]["OSL"]["value"]) == (128, 16)


def test_preserve_engine_limits_survives_typed_request_round_trip():
    params = _params(preserve=True)

    result = to_legacy_params(from_legacy_params(params, backend="vllm"))

    assert result["preserve_engine_limits"] is True


@pytest.mark.parametrize("rule", [None, "benchmark"])
@pytest.mark.parametrize("preserve", [False, True])
def test_trtllm_rules_only_default_block_size_for_unpinned_requests(rule, preserve):
    params = _params(preserve=preserve, rule=rule)
    for role in ("prefill", "decode"):
        params["params"][role].pop("tokens_per_block")
        params["params"][role].pop("cache_transceiver_max_tokens_in_buffer")

    result = apply_rule_plugins(params, "trtllm")

    for role in ("prefill", "decode"):
        actual = result["params"][role]
        if preserve:
            assert "tokens_per_block" not in actual
            assert actual.get("cache_transceiver_max_tokens_in_buffer") is None
        else:
            assert actual["tokens_per_block"] == 32
            assert actual["cache_transceiver_max_tokens_in_buffer"] > 0
            assert actual["cache_transceiver_max_tokens_in_buffer"] % 32 == 0
        assert actual["cache_transceiver_backend"] == "DEFAULT"


@pytest.mark.parametrize("value", [True, False])
def test_module_bridge_propagates_preserve_engine_limits(value: bool):
    task = SimpleNamespace(
        primary_backend_name="vllm",
        primary_system_name="gb200",
        primary_backend_version="0.19.0",
        primary_model_path="meta-llama/Meta-Llama-3.1-8B",
        prefix=0,
        is_moe=False,
        nextn=0,
        nextn_accepted=None,
        serving_mode="agg",
        total_gpus=1,
        system_name="gb200",
        isl=128,
        osl=16,
        ttft=1000.0,
        tpot=100.0,
    )
    row = pd.Series({"workers": 1, "tp": 1, "pp": 1, "dp": 1, "bs": 16})

    result = task_config_to_generator_config(
        task,
        row,
        generator_overrides={"preserve_engine_limits": value},
        num_gpus_per_node=4,
    )

    assert result["preserve_engine_limits"] is value


@pytest.mark.unit
@pytest.mark.parametrize(
    "version",
    [
        None,
        "current",
        "0.10.2",
        "0.11.0",
        "0.12.0",
        "0.14.1",
        "0.16.0",
        "0.19.0",
        "0.20.1",
        "0.24.0",
        "0.25.1",
        "0.26.0",
        "0.27.0",
        "0.27.0+cu128",
        "0.28.0",
    ],
)
@pytest.mark.parametrize("rule", [None, "benchmark"])
@pytest.mark.parametrize("dtype", ["bfloat16", "fp8", "auto"])
def test_vllm_cache_dtype_preserves_explicit_selection(version, rule, dtype):
    params = _params(preserve=True, rule=rule)
    params["ServiceConfig"]["model_path"] = "example/model"
    for role in ("prefill", "decode"):
        params["params"][role]["kv_cache_dtype"] = dtype
    artifacts = render_backend_templates(params, "vllm", version=version)
    expected = "auto" if dtype == "bfloat16" and version == "0.10.2" else dtype
    for role in ("prefill", "decode"):
        tokens = shlex.split(artifacts[f"cli_args_{role}"])
        assert tokens[tokens.index("--kv-cache-dtype") + 1] == expected
