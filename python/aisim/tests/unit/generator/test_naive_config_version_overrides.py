# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for naive generator version override routing."""

import re
from unittest.mock import Mock

import pytest
import yaml

from aisim.generator import api, naive

pytestmark = pytest.mark.unit


def _minimal_generator_params(backend: str = "vllm") -> dict:
    return {
        "ServiceConfig": {"model_path": "Qwen/Qwen3-32B"},
        "K8sConfig": {"k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0"},
        "WorkerConfig": {"agg_workers": 8, "agg_gpus_per_worker": 1},
        "params": {
            "agg": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpus_per_worker": 1,
                "max_batch_size": 128,
            }
        },
        "backend": backend,
    }


def _patch_naive_dependencies(monkeypatch):
    calls = {"build": None, "render": None}

    def fake_build_naive_generator_params(**kwargs):
        calls["build"] = kwargs
        return _minimal_generator_params(kwargs["backend_name"])

    def fake_render_backend_templates(param_values, backend, templates_dir, backend_version, deployment_target):
        calls["render"] = {
            "param_values": param_values,
            "backend": backend,
            "templates_dir": templates_dir,
            "backend_version": backend_version,
            "deployment_target": deployment_target,
        }
        return {"cli_args_agg": "--model Qwen/Qwen3-32B"}

    monkeypatch.setattr(api, "build_naive_generator_params", fake_build_naive_generator_params)
    monkeypatch.setattr(api, "render_backend_templates", fake_render_backend_templates)
    return calls


def test_generate_naive_config_resolves_backend_version_from_dynamo(monkeypatch):
    calls = _patch_naive_dependencies(monkeypatch)

    result = api.generate_naive_config(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="vllm",
        generator_dynamo_version="1.2.0",
    )

    assert result["backend_version"] == "0.20.1"
    assert calls["build"]["generator_dynamo_version"] == "1.2.0"
    assert calls["render"]["backend_version"] == "0.20.1"


def test_generate_naive_config_generated_config_version_wins(monkeypatch):
    calls = _patch_naive_dependencies(monkeypatch)

    result = api.generate_naive_config(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="vllm",
        generated_config_version="9.9.9-does-not-exist",
        generator_dynamo_version="1.2.0",
    )

    assert result["backend_version"] == "9.9.9-does-not-exist"
    assert calls["build"]["generator_dynamo_version"] == "1.2.0"
    assert calls["render"]["backend_version"] == "9.9.9-does-not-exist"


def test_generate_naive_config_uses_perf_database_without_overrides(monkeypatch):
    calls = _patch_naive_dependencies(monkeypatch)

    def fake_latest_version(system, backend):
        assert system == "h200_sxm"
        assert backend == "vllm"
        return "0.19.0"

    monkeypatch.setattr("aisim.sdk.perf_database.get_latest_database_version", fake_latest_version)

    result = api.generate_naive_config(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="vllm",
    )

    assert result["backend_version"] == "0.19.0"
    assert calls["render"]["backend_version"] == "0.19.0"


def test_generate_naive_config_renders_valid_dgd_without_database_version(monkeypatch):
    latest_version = Mock(return_value=None)
    monkeypatch.setattr("aisim.sdk.perf_database.get_latest_database_version", latest_version)
    monkeypatch.setattr(
        naive,
        "_get_system_config",
        Mock(return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3}),
    )
    monkeypatch.setattr(naive, "_estimate_model_weight_bytes", Mock(return_value=30 * 1024**3))
    monkeypatch.setattr(
        naive,
        "get_model_config_from_model_path",
        Mock(return_value={"architecture": "Qwen3ForCausalLM", "num_experts": 0}),
    )

    result = api.generate_naive_config(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="vllm",
    )

    latest_version.assert_called_once_with(system="h200_sxm", backend="vllm")
    assert result["backend_version"] is None
    [dgd] = [
        doc
        for doc in yaml.safe_load_all(result["artifacts"]["k8s_deploy.yaml"])
        if doc["kind"] == "DynamoGraphDeployment"
    ]
    name = dgd["metadata"]["name"]
    for label in name.split("."):
        assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", label), name
    assert len(name) <= 253
    assert "None" not in name
    services = dgd["spec"]["services"]
    assert services["Frontend"]["replicas"] == 1
    args = services["VllmWorker"]["extraPodSpec"]["mainContainer"]["args"]
    assert args[args.index("--model") + 1] == "Qwen/Qwen3-32B"
