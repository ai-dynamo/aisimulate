# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM MoE backend on Blackwell: WIDEEP only for releases whose MoeConfig accepts it
(through 1.3.0rc24); from rc29 an unset backend renders the framework default AUTO."""

from __future__ import annotations

import pytest
import yaml

from aisimulate.generator.api import generate_backend_artifacts
from aisimulate.generator.facts.apply import apply_moe_backend

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("version", "expected"),
    [("1.3.0rc14", "WIDEEP"), ("1.3.0rc24", "WIDEEP"), ("1.3.0rc25", None), ("1.3.0rc29", None), (None, "WIDEEP")],
)
def test_wideep_is_filled_only_for_releases_that_accept_it(version, expected):
    ctx = {"is_moe": True, "moe_config": {}}
    apply_moe_backend(ctx, {"moe_backend": {"trtllm": "WIDEEP"}}, backend="trtllm", backend_version=version)
    assert ctx["moe_config"].get("backend") == expected


def _params(system: str) -> dict:
    return {
        "ServiceConfig": {
            "model_path": "deepseek-ai/DeepSeek-V3",
            "served_model_path": "deepseek-ai/DeepSeek-V3",
            "served_model_name": "DeepSeek-V3",
            "include_frontend": True,
        },
        "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 8},
        "NodeConfig": {"num_gpus_per_node": 8, "system_name": system},
        "SlaConfig": {"isl": 1024, "osl": 256},
        "ModelConfig": {"is_moe": True, "prefix": 0},
        "BenchConfig": {},
        "params": {
            "agg": {
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "moe_tensor_parallel_size": 1,
                "moe_expert_parallel_size": 8,
                "max_batch_size": 32,
                "max_num_tokens": 4096,
                "max_seq_len": 4096,
            },
        },
    }


def _moe_backend(version: str, system: str) -> str:
    artifacts = generate_backend_artifacts(backend="trtllm", backend_version=version, params=_params(system))
    name = next(name for name in artifacts if "extra_engine_args" in name and "agg" in name)
    return yaml.safe_load(artifacts[name])["moe_config"]["backend"]


@pytest.mark.parametrize(("version", "expected"), [("1.3.0rc23", "WIDEEP"), ("1.3.0rc29", "AUTO")])
def test_blackwell_render_keeps_wideep_before_rc25_and_auto_from_rc29(version, expected):
    assert _moe_backend(version, "b200_sxm") == expected


def test_rc29_hopper_render_keeps_the_hardware_fact():
    assert _moe_backend("1.3.0rc29", "h200_sxm") == "CUTLASS"
