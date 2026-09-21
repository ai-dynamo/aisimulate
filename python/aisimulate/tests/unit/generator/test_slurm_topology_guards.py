# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
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
        ("vllm", ["--device-ids", "GPU-outside-allocation"]),
        ("vllm", ["--device_i=GPU-outside-allocation"]),
        ("vllm", ["--data-parallel-backend", "ray"]),
        ("vllm", ["--data_parallel_b=ray"]),
        ("vllm", ["-dpb", "ray"]),
        ("vllm", ["--data-parallel-multi-port-external-lb"]),
        ("vllm", ["--data_parallel_multi"]),
        ("vllm", ["-dpm"]),
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
        ("sglang", ["--decode-context-parallel-size", "2"]),
        ("sglang", ["--dcp-size=2"]),
        ("sglang", ["--elastic-ep-backend", "mooncake"]),
        ("sglang", ["--elastic-ep-join-mode", "scale"]),
        ("sglang", ["--elastic-ep-join-rank-offset=8"]),
        ("sglang", ["--elastic-ep-initial-size", "8"]),
        ("sglang", ["--max-ep-size", "16"]),
        ("sglang", ["--elastic-ep-rejoin"]),
        ("sglang", ["--disagg-config", "/models/engine.yaml"]),
        ("sglang", ["--disagg-config-key=decode"]),
        ("sglang", ["--disagg-config-k=decode"]),
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
    "backend,option",
    [
        ("vllm", "--is-prefill-worker"),
        ("vllm", "--is-decode-worker"),
        ("vllm", "--no-is-prefill-worker"),
        ("vllm", "--no-is-decode-worker"),
        ("vllm", "--is_prefill_worker"),
        ("vllm", "--is_decode"),
        ("vllm", "--is-prefill"),
        ("vllm", "--multimodal-encode-worker"),
        ("vllm", "--multimodal-decode-worker"),
        ("vllm", "--no-multimodal-encode-worker"),
        ("vllm", "--no-multimodal-decode-worker"),
        ("sglang", "--multimodal-encode-worker"),
        ("sglang", "--multimodal-encode"),
        ("sglang", "--no-multimodal-encode-worker"),
    ],
)
@pytest.mark.parametrize("role", ["agg", "prefill", "decode"])
def test_slurm_rejects_legacy_role_overrides_before_emission(backend, option, role, tmp_path):
    with pytest.raises(ValueError, match=rf"Workers\.{role}\.extra_cli_args.*Slurm-owned"):
        generate_backend_artifacts(
            _params(backend, [option], role), backend, deployment_target="slurm", output_dir=str(tmp_path)
        )
    assert not list(tmp_path.iterdir())


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


_WORKER_ENV_DEFAULTS = {
    "vllm": {
        "VLLM_DP_SIZE": "1",
        "VLLM_DP_RANK": "0",
        "VLLM_DP_RANK_LOCAL": "0",
        "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": "0",
    },
    "sglang": {"DYN_SGL_DISAGG_CONFIG": "", "DYN_SGL_DISAGG_CONFIG_KEY": ""},
}


@pytest.mark.parametrize(
    "backend,key", [(backend, key) for backend, defaults in _WORKER_ENV_DEFAULTS.items() for key in defaults]
)
def test_slurm_rejects_environment_topology_overrides_before_emission(backend, key, tmp_path):
    params = _params(backend, [])
    params["SlurmConfig"]["env"] = {key: "16"}
    with pytest.raises(ValueError, match=r"SlurmConfig\.env must not override"):
        generate_backend_artifacts(params, backend, deployment_target="slurm", output_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("backend,data_parallel_size", [("vllm", 1), ("vllm", 2), ("sglang", 1)])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_worker_neutralizes_inherited_topology_environment(backend, mode, data_parallel_size, monkeypatch, tmp_path):
    expected = _WORKER_ENV_DEFAULTS[backend]
    for key in expected:
        monkeypatch.setenv(key, "16")
    monkeypatch.setenv("SLURM_TEST_TUNING", "preserved")
    params = _params(backend, [], "agg" if mode == "agg" else "prefill")
    if data_parallel_size > 1:
        params["ModelConfig"] = {"is_moe": True}
        for role in ["agg"] if mode == "agg" else ["prefill", "decode"]:
            params["params"][role].update(
                data_parallel_size=data_parallel_size,
                moe_tensor_parallel_size=2,
                moe_expert_parallel_size=2,
            )
    artifacts = generate_backend_artifacts(
        params,
        backend,
        backend_version={"vllm": "0.24.0", "sglang": "0.5.16"}[backend],
        deployment_target="slurm",
    )
    deployment = json.loads(artifacts["deployment.json"])
    supervisor = Supervisor(deployment, tmp_path)
    try:
        for worker in deployment["workers"]:
            argv = worker["argv"]
            actual_dp = int(argv[argv.index("--data-parallel-size") + 1]) if "--data-parallel-size" in argv else 1
            assert actual_dp == data_parallel_size
            process = supervisor.start(
                worker["name"],
                [
                    sys.executable,
                    "-c",
                    "import json, os, sys; print(json.dumps({key: os.environ[key] for key in sys.argv[1:]}))",
                    *expected,
                    "SLURM_TEST_TUNING",
                ],
                worker["env"],
            )
            assert process.wait(timeout=5) == 0
            assert json.loads((tmp_path / f"{worker['name']}.log").read_text()) == {
                **expected,
                "SLURM_TEST_TUNING": "preserved",
            }
    finally:
        supervisor.stop()


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


_ROLE_ENV = {
    "vllm": {
        "DYN_VLLM_DISAGGREGATION_MODE": "prefill",
        "DYN_VLLM_IS_PREFILL_WORKER": "true",
        "DYN_VLLM_IS_DECODE_WORKER": "true",
        "DYN_VLLM_MULTIMODAL_ENCODE_WORKER": "true",
        "DYN_VLLM_MULTIMODAL_DECODE_WORKER": "true",
    },
    "sglang": {"DYN_SGL_MULTIMODAL_ENCODE_WORKER": "true"},
    "trtllm": {"DYN_TRTLLM_DISAGGREGATION_MODE": "prefill"},
}


@pytest.mark.parametrize("backend,key", [(backend, key) for backend, values in _ROLE_ENV.items() for key in values])
def test_slurm_rejects_environment_role_overrides_before_emission(backend, key, tmp_path):
    params = _params(backend, [])
    params["SlurmConfig"]["env"] = {key: _ROLE_ENV[backend][key]}
    with pytest.raises(ValueError, match=r"SlurmConfig\.env must not override.*worker-role"):
        generate_backend_artifacts(params, backend, deployment_target="slurm", output_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "backend,version", [("vllm", "0.9.0"), ("vllm", "1.3.0"), ("sglang", "1.3.0"), ("trtllm", "1.3.0")]
)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_emitted_workers_remove_inherited_role_defaults(backend, version, mode, monkeypatch, tmp_path):
    inherited = _ROLE_ENV[backend]
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SLURM_TEST_TUNING", "preserved")
    original_env = dict(os.environ)
    params = _params(backend, [], "agg" if mode == "agg" else "prefill")
    params["generator_dynamo_version"] = version
    generate_backend_artifacts(params, backend, deployment_target="slurm", output_dir=str(tmp_path))
    deployment = json.loads((tmp_path / "deployment.json").read_text())
    module_spec = importlib.util.spec_from_file_location("emitted_slurm_runtime", tmp_path / "slurm_runtime.py")
    runtime = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runtime)
    supervisor = runtime.Supervisor(deployment, tmp_path / "logs")
    probe = [
        sys.executable,
        "-c",
        "import json, os, sys; print(json.dumps({key: os.environ.get(key) for key in sys.argv[1:]}))",
        *inherited,
        "SLURM_TEST_TUNING",
    ]
    try:
        for worker in deployment["workers"]:
            role = worker["name"].rsplit("-", 1)[0]
            argv = worker["argv"]
            if role != "agg":
                if backend == "vllm" and version == "0.9.0":
                    assert f"--is-{role}-worker" in argv
                    assert "--disaggregation-mode" not in argv
                else:
                    assert argv[argv.index("--disaggregation-mode") + 1] == role
            process = supervisor.start(worker["name"], probe, worker["env"])
            assert process.wait(timeout=5) == 0
            assert json.loads((tmp_path / "logs" / f"{worker['name']}.log").read_text()) == {
                **dict.fromkeys(inherited),
                "SLURM_TEST_TUNING": "preserved",
            }
        process = supervisor.start("service-probe", probe)
        assert process.wait(timeout=5) == 0
        assert json.loads((tmp_path / "logs/service-probe.log").read_text()) == {
            **inherited,
            "SLURM_TEST_TUNING": "preserved",
        }
    finally:
        supervisor.stop()
    assert dict(os.environ) == original_env


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("kvbm", [False, True])
def test_role_environment_does_not_enable_or_disable_kvbm(backend, mode, kvbm):
    params = _params(backend, [], "agg" if mode == "agg" else "prefill")
    if kvbm:
        params["DynConfig"]["kvbm_config"] = {"cpu_cache_gb": 1}
    artifacts = generate_backend_artifacts(params, backend, deployment_target="slurm")
    for worker in json.loads(artifacts["deployment.json"])["workers"]:
        role = worker["name"].rsplit("-", 1)[0]
        uses_kvbm = kvbm and role != "decode"
        assert ("DYN_KVBM_CPU_CACHE_GB" in worker["env"]) == uses_kvbm
        argv = worker["argv"]
        if backend == "trtllm":
            assert ("--connector" in argv) == uses_kvbm
            if uses_kvbm:
                assert argv[argv.index("--connector") + 1] == "kvbm"
        elif uses_kvbm or role != "agg":
            connector = json.loads(argv[argv.index("--kv-transfer-config") + 1])["kv_connector"]
            assert connector == (
                "DynamoConnector" if role == "agg" else "PdConnector" if uses_kvbm else "NixlConnector"
            )
        else:
            assert "--kv-transfer-config" not in argv
