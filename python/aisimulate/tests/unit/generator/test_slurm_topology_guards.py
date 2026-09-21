# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys

import pytest

from aisimulate.generator.api import generate_backend_artifacts, generate_config_from_input_dict
from aisimulate.generator.builders.slurm_runtime import Supervisor


def _params(backend, extra, role="agg"):
    roles = ["agg"] if role == "agg" else ["prefill", "decode"]
    return generate_config_from_input_dict(
        {
            "ServiceConfig": {"model_path": "/models/test", "served_model_name": "test-model"},
            "DynConfig": {"mode": "agg" if role == "agg" else "disagg"},
            "Workers": {
                name: {
                    "tensor_parallel_size": 2,
                    "max_batch_size": 8,
                    "extra_cli_args": extra if name == role else [],
                }
                for name in roles
            },
            "NodeConfig": {"num_gpus_per_node": 8},
            "SlurmConfig": {
                "account": "test-account",
                "partition": "batch",
                "container_image": "/images/dynamo.sqsh",
            },
        },
        backend=backend,
    )


@pytest.mark.parametrize(
    "backend,extra",
    [
        ("vllm", ["--tensor-parallel-size", "16"]),
        ("vllm", ["--pipeline-parallel-size=16"]),
        ("vllm", ["--data_parallel_size", "16"]),
        ("vllm", ["-tp", "16"]),
        ("vllm", ["-pp=16"]),
        ("vllm", ["-dp", "16"]),
        ("vllm", ["-dpl", "16"]),
        ("vllm", ["-dpn", "1"]),
        ("vllm", ["-dpr", "1"]),
        ("vllm", ["-dcp", "16"]),
        ("vllm", ["-pcp", "16"]),
        ("vllm", ["-ep"]),
        ("vllm", ["--no-enable-expert-parallel"]),
        ("vllm", ["--tensor-parallel-s=16"]),
        ("vllm", ["--data-parallel-size-l", "16"]),
        ("vllm", ["-t", "16"]),
        ("vllm", ["-n2"]),
        ("vllm", ["-r1"]),
        ("vllm", ["--distributed-executor-backend", "external_launcher"]),
        ("vllm", ["--config", "/models/engine.yaml"]),
        ("sglang", ["--tensor-parallel-size", "16"]),
        ("sglang", ["--tp-size=16"]),
        ("sglang", ["--pp", "16"]),
        ("sglang", ["--dp", "16"]),
        ("sglang", ["--ep", "16"]),
        ("sglang", ["--attention-context-parallel-size", "4"]),
        ("sglang", ["--moe-dp-size", "4"]),
        ("sglang", ["--moe-dense-tp-size", "1"]),
        ("sglang", ["--enable-dp-attention"]),
        ("sglang", ["--base-gpu-id", "4"]),
        ("sglang", ["--gpu-id-step", "4"]),
        ("sglang", ["--nnodes", "2"]),
        ("sglang", ["--node-rank", "1"]),
        ("sglang", ["--config=/models/engine.yaml"]),
        ("trtllm", ["--tensor-parallel-size", "16"]),
        ("trtllm", ["--pipeline-parallel-s=16"]),
        ("trtllm", ["--expert-parallel-size", "4"]),
        ("trtllm", ["--gpus-per-node", "16"]),
        ("trtllm", ["--enable-attention-dp"]),
        ("trtllm", ["--no-enable-attention-dp"]),
        ("trtllm", ["--override-engine-args", '{"tensor_parallel_size": 16}']),
        ("trtllm", ['--override-engine-args={"pipeline_parallel_size": 16}']),
        ("trtllm", ["--override-engine", '{"moe_expert_parallel_size": 16}']),
        ("trtllm", ["--trtllm.tensor_parallel_size", "16"]),
        ("trtllm", ["--trtllm.moe_expert_parallel_size=16"]),
    ],
)
def test_slurm_rejects_topology_and_config_overrides_before_emission(backend, extra, tmp_path):
    with pytest.raises(ValueError, match=r"Workers\.agg\.extra_cli_args.*Slurm-owned"):
        generate_backend_artifacts(
            _params(backend, extra), backend, deployment_target="slurm", output_dir=str(tmp_path)
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_slurm_checks_topology_overrides_in_each_disaggregated_pool(role):
    with pytest.raises(ValueError, match=rf"Workers\.{role}\.extra_cli_args.*Slurm-owned"):
        generate_backend_artifacts(_params("vllm", ["-tp", "16"], role), "vllm", deployment_target="slurm")


@pytest.mark.parametrize(
    "backend,extra",
    [
        (
            "vllm",
            [
                "--max-parallel-loading-workers",
                "2",
                "--hf-overrides",
                '{"note": "--tensor-parallel-size", "description": "-tp 16"}',
                "--tokenizer-revision=--tensor-parallel-size",
                "--revision",
                "release--data-parallel-size",
            ],
        ),
        ("sglang", ["--mem-fraction-static", "0.8", "--revision", "release--tp-size"]),
        ("trtllm", ["--max-batch-size", "16", "--revision=--trtllm.tensor_parallel_size"]),
    ],
)
def test_slurm_preserves_tuning_options_and_values_without_changing_allocation(backend, extra):
    artifacts = generate_backend_artifacts(_params(backend, extra), backend, deployment_target="slurm")
    deployment = json.loads(artifacts["deployment.json"])
    worker = deployment["workers"][0]
    assert deployment["gpus"] == worker["gpu_count"] == 2
    assert "#SBATCH --gres=gpu:2" in artifacts["deploy.sbatch"]
    assert worker["argv"][-len(extra) :] == extra


def test_non_slurm_targets_keep_existing_extra_argument_behavior():
    artifacts = generate_backend_artifacts(
        _params("vllm", ["--tensor-parallel-size", "16"]), "vllm", deployment_target="dynamo-j2"
    )
    assert "run_0.sh" in artifacts
    assert "k8s_deploy.yaml" in artifacts


def test_slurm_rejects_trtllm_environment_engine_overlay_before_emission(tmp_path):
    params = _params("trtllm", [])
    params["SlurmConfig"]["env"] = {"DYN_TRTLLM_OVERRIDE_ENGINE_ARGS": '{"tensor_parallel_size": 16}'}
    with pytest.raises(ValueError, match=r"SlurmConfig\.env must not override"):
        generate_backend_artifacts(params, "trtllm", deployment_target="slurm", output_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_trtllm_worker_clears_inherited_engine_overlay(monkeypatch, tmp_path):
    monkeypatch.setenv("DYN_TRTLLM_OVERRIDE_ENGINE_ARGS", '{"tensor_parallel_size": 16}')
    artifacts = generate_backend_artifacts(_params("trtllm", []), "trtllm", deployment_target="slurm")
    deployment = json.loads(artifacts["deployment.json"])
    supervisor = Supervisor(deployment, tmp_path)
    try:
        process = supervisor.start(
            "env-probe",
            [sys.executable, "-c", "import os; print(repr(os.environ['DYN_TRTLLM_OVERRIDE_ENGINE_ARGS']))"],
            deployment["workers"][0]["env"],
        )
        assert process.wait(timeout=5) == 0
    finally:
        supervisor.stop()
    assert (tmp_path / "env-probe.log").read_text().strip() == "''"


@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_trtllm_pins_local_gpu_count_to_each_worker_allocation(mode):
    params = _params("trtllm", [], "agg" if mode == "agg" else "prefill")
    params["SlurmConfig"]["env"] = {"DYN_TRTLLM_GPUS_PER_NODE": "16"}
    if mode == "agg":
        params["WorkerConfig"]["agg_workers"] = 3
        expected = [(0, 2), (2, 2), (4, 2)]
    else:
        params["params"]["prefill"]["tensor_parallel_size"] = 4
        params["WorkerConfig"]["decode_workers"] = 2
        expected = [(0, 4), (4, 2), (6, 2)]
    artifacts = generate_backend_artifacts(params, "trtllm", deployment_target="slurm")
    deployment = json.loads(artifacts["deployment.json"])
    assert [(worker["gpu_offset"], worker["gpu_count"]) for worker in deployment["workers"]] == expected
    for worker in deployment["workers"]:
        argv = worker["argv"]
        assert argv[argv.index("--gpus-per-node") + 1] == str(worker["gpu_count"])
    assert deployment["gpus"] == sum(count for _, count in expected)
