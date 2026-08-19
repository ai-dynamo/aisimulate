# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import aisimulate.sweeper.engine_request as engine_request_mod
from aiconfigurator_core.sdk.config_builders import build_model_config
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.engine_request import (
    EngineControlTemplate,
    materialize_engine_request,
    resolve_engine_controls,
)


def _config(**search_overrides) -> SmartSearchConfig:
    search_space = {
        "model_name": "example/moe",
        "hardware_sku": "gb200",
        "backend": ["sglang"],
        "deployment_mode": ["disagg"],
        "max_seq_len": 8192,
    }
    search_space.update(search_overrides)
    return SmartSearchConfig(
        search_space=search_space,
        workload={
            "isl": 4096,
            "osl": 1024,
            "concurrency": 2,
            "num_request_ratio": 1,
            "cached_prefix_tokens": 512,
        },
    )


def _stub_model(monkeypatch, *, is_moe=True, mla=True, family="DEEPSEEK") -> None:
    monkeypatch.setattr(engine_request_mod, "get_model_family", lambda model: family)
    monkeypatch.setattr(
        engine_request_mod,
        "resolve_model_hardware",
        lambda *args, **kwargs: SimpleNamespace(
            is_moe=is_moe,
            mla=mla,
            max_context=16384,
        ),
    )


def test_resolves_supported_engine_controls_before_replay(monkeypatch):
    _stub_model(monkeypatch)
    config = _config(
        enable_chunked_prefill=True,
        enable_wideep=True,
        enable_eplb=True,
        wideep_num_slots=64,
        moe_backend="deepep_moe",
        attention_backend="fa3",
        gemm_quant_mode="fp8",
        moe_quant_mode="fp8",
        kvcache_quant_mode="fp8",
        fmha_quant_mode="fp8",
        comm_quant_mode="fp8",
        aic_nextn=3,
        nextn_accepted=1.5,
        free_gpu_memory_fraction=0.82,
    )

    assert resolve_engine_controls(config) == {
        "sglang": EngineControlTemplate(
            backend="sglang",
            max_seq_len=8192,
            model_family="DEEPSEEK",
            is_moe=True,
            memory_fraction_kind="of_total",
        )
    }


def test_rejects_unsupported_model_backend_controls(monkeypatch):
    _stub_model(monkeypatch, is_moe=False, mla=False, family="GPT")
    with pytest.raises(ValueError, match="MoE controls require an MoE model"):
        resolve_engine_controls(_config(enable_eplb=True, enable_wideep=True))

    _stub_model(monkeypatch, is_moe=True, mla=True)
    with pytest.raises(ValueError, match="only with backend='sglang'"):
        resolve_engine_controls(_config(backend=["vllm"], attention_backend="fa3"))


def test_rejects_invalid_quant_and_sequence_capacity(monkeypatch):
    _stub_model(monkeypatch)
    with pytest.raises(ValueError, match="gemm_quant_mode has unsupported value"):
        resolve_engine_controls(_config(gemm_quant_mode="made_up"))
    with pytest.raises(ValueError, match="cannot hold the synthetic request"):
        resolve_engine_controls(_config(max_seq_len=5000))


def test_materializes_shared_controls_for_both_disaggregated_roles(monkeypatch):
    _stub_model(monkeypatch)
    config = _config(
        enable_chunked_prefill=True,
        free_gpu_memory_fraction=0.81,
        aic_nextn=2,
        nextn_accepted=1.25,
    )
    template = resolve_engine_controls(config)["sglang"]
    sample = {
        "deployment_mode": "disagg",
        "prefill_max_num_batched_tokens": 32768,
        "decode_max_num_batched_tokens": 8192,
        "prefill_gpu_memory_utilization": 0.9,
        "decode_gpu_memory_utilization": 0.88,
    }

    request = materialize_engine_request(template, config=config, sample=sample)

    assert request.cached_prefix_tokens == 512
    assert request.context_tokens == {"prefill": 32768, "decode": 8192}
    assert request.memory_fraction_by_role == {"prefill": 0.81, "decode": 0.81}
    assert request.max_seq_len == 8192
    assert request.nextn == 2
    assert request.nextn_accepted == 1.25


def test_aic_model_config_receives_backend_and_eplb_controls():
    model_config = build_model_config(
        tp_size=1,
        pp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        moe_backend="megamoe",
        attention_backend="fa3",
        enable_wideep=True,
        enable_eplb=True,
        wideep_num_slots=64,
    )

    assert model_config.moe_backend == "megamoe"
    assert model_config.attention_backend == "fa3"
    assert model_config.enable_wideep is True
    assert model_config.enable_eplb is True
    assert model_config.wideep_num_slots == 64
