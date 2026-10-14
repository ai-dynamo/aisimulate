# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph option compatibility across SGLang's 0.5.20 CLI boundary."""

import json
import shlex
import subprocess

import pytest
import yaml

from aisimulate.generator.api import generate_from_request
from aisimulate.generator.request import ModelFacts, from_sweeper_candidate

from .test_sweeper_request import _agg_candidate, _disagg_candidate

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", ["0.5.11", "0.5.17", "0.5.19", "0.5.20", "0.5.21", "current"])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("target", ["dynamo-j2", "slurm"])
def test_sglang_graph_options_match_runtime(version, mode, target):
    candidate = (
        _agg_candidate(backend="sglang", backend_version=version, agg_max_num_seqs=4)
        if mode == "agg"
        else _disagg_candidate(backend="sglang", backend_version=version, adapters={}, decode_max_num_seqs=8)
    )
    request = from_sweeper_candidate(
        candidate,
        workload={"isl": 1024, "osl": 128},
        deployment_target=target,
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={
            "K8sConfig": {"k8s_image": "example/sglang:release"},
            "SlurmConfig": {"account": "test", "partition": "batch", "container_image": "/images/sglang.sqsh"},
        },
    )
    for sizing in request.topology.roles.values():
        sizing.extra["cuda_graph_max_batch_size"] = 16

    artifacts = generate_from_request(request)
    modern = version in {"0.5.20", "0.5.21", "current"}
    bs_flag = "--cuda-graph-bs-decode" if modern else "--cuda-graph-bs"
    max_flag = "--cuda-graph-max-bs-decode" if modern else "--cuda-graph-max-bs"
    incompatible_flags = (
        {"--cuda-graph-bs", "--cuda-graph-max-bs"}
        if modern
        else {"--cuda-graph-bs-decode", "--cuda-graph-max-bs-decode"}
    )
    for role, sizing in request.topology.roles.items():
        tokens = (
            next(
                worker["argv"]
                for worker in json.loads(artifacts["deployment.json"])["workers"]
                if worker["name"] == f"{role}-0"
            )
            if target == "slurm"
            else shlex.split(artifacts[f"cli_args_{role}"])
        )
        start = tokens.index(bs_flag) + 1
        assert tokens[start : start + sizing.max_batch_size] == [str(i) for i in range(1, sizing.max_batch_size + 1)]
        assert tokens[tokens.index(max_flag) + 1] == "16"
        # Request batch sizes must not become prefill token capture sizes.
        assert "--cuda-graph-bs-prefill" not in tokens
        assert "--cuda-graph-max-bs-prefill" not in tokens
        assert incompatible_flags.isdisjoint(tokens)

    if target == "slurm":
        for worker in json.loads(artifacts["deployment.json"])["workers"]:
            assert bs_flag in worker["argv"]
            assert max_flag in worker["argv"]
            assert incompatible_flags.isdisjoint(worker["argv"])
    else:
        services = yaml.safe_load(artifacts["k8s_deploy.yaml"])["spec"]["services"]
        for name, service in services.items():
            if "Worker" in name:
                script = service["extraPodSpec"]["mainContainer"]["args"][0]
                assert bs_flag + " " in script
                assert max_flag + " " in script
                assert incompatible_flags.isdisjoint(shlex.split(script))
        sflow = artifacts["sflow.yaml"]
        document = yaml.safe_load(sflow)
        tasks = {task["name"]: task for task in document["workflow"]["tasks"]}
        for role, sizing in request.topology.roles.items():
            prefix = {"agg": "AGG", "prefill": "CTX", "decode": "GEN"}[role]
            script = "\n".join(tasks[f"{role}_server"]["script"])
            tokens = shlex.split(script)
            # Every role retains its graph maximum, independent of request limits.
            assert tokens[tokens.index(max_flag) + 1] == "16"
            assert incompatible_flags.isdisjoint(tokens)
            assert bs_flag + " ${{ variables." + prefix + "_CUDA_GRAPH_BS }}" in script
            variables = document["variables"]
            graph = variables[f"{prefix}_CUDA_GRAPH_BS"]
            assert str(graph["value"]) == " ".join(str(i) for i in range(1, sizing.max_batch_size + 1))
    for name, content in artifacts.items():
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n"], input=content, text=True, capture_output=True, check=True)
