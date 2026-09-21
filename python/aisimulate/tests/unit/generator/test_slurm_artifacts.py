# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from aisimulate.generator.api import (
    generate_backend_artifacts,
    generate_config_from_input_dict,
    generate_from_request,
)
from aisimulate.generator.builders.slurm_runtime import Supervisor
from aisimulate.generator.module_bridge import task_config_to_generator_config
from aisimulate.generator.naive import build_naive_generator_params
from aisimulate.generator.request import (
    ModelFacts,
    SweeperCandidateError,
    from_legacy_params,
    from_sweeper_candidate,
    to_legacy_params,
)

_BACKENDS = [("vllm", "0.20.1"), ("sglang", "0.5.11"), ("trtllm", "1.3.0rc14")]
_GOLDEN = Path(__file__).resolve().parents[2] / "golden/generator/slurm"


@pytest.fixture
def local_http_without_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def _params(backend="vllm", mode="agg"):
    return generate_config_from_input_dict(
        {
            "ServiceConfig": {"model_path": "/models/Qwen 3", "served_model_name": "test-model"},
            "DynConfig": {"mode": mode},
            "Workers": {
                role: {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                    "max_batch_size": 8,
                }
                for role in (["agg"] if mode == "agg" else ["prefill", "decode"])
            },
            "NodeConfig": {"num_gpus_per_node": 8},
            "SlaConfig": {"isl": 128, "osl": 64},
            "SlurmConfig": {
                "account": "test-account",
                "partition": "batch",
                "container_image": "/images/dynamo.sqsh",
                "container_mounts": ["/cache/models:/models:ro"],
                "env": {"HF_HUB_OFFLINE": "1"},
                "benchmark_concurrency": [1, 4],
            },
        },
        backend=backend,
    )


def _emitted_supervisor(tmp_path, params, backend="vllm", version=None):
    generate_backend_artifacts(
        params, backend, backend_version=version, deployment_target="slurm", output_dir=str(tmp_path)
    )
    module_spec = importlib.util.spec_from_file_location("emitted_slurm_runtime", tmp_path / "slurm_runtime.py")
    runtime = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runtime)
    return runtime, runtime.Supervisor(json.loads((tmp_path / "deployment.json").read_bytes()), tmp_path / "results")


@pytest.fixture
def free_service_ports():
    worker = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw"))
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    # A busy block must not make one xdist worker scan another worker's blocks.
    for base in range(20000 + worker * 40, 31960, workers * 40):
        try:
            with ExitStack() as sockets:
                for port in range(base, base + 40):
                    sock = sockets.enter_context(socket.socket())
                    sock.bind(("0.0.0.0", port))
        except OSError:
            continue
        return base
    pytest.fail("Could not find free control ports for the supervisor test")


def test_service_port_fixture_keeps_worker_blocks_disjoint_after_fallback(monkeypatch):
    def bind(address):
        if 20000 <= address[1] < 20160:
            raise OSError("First block occupied for each worker")

    # Simulate busy ports without occupying blocks belonging to other live
    # xdist workers. Emitted-worker tests below exercise real socket consumers.
    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.bind.side_effect = bind
    monkeypatch.setattr(socket, "socket", lambda: sock)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    selected = []
    for worker in range(4):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", f"gw{worker}")
        base = free_service_ports.__wrapped__()
        assert 20160 <= base < 31960
        selected.extend(range(base, base + 40))
    assert len(selected) == len(set(selected))
    monkeypatch.delenv("PYTEST_XDIST_WORKER")
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT")
    assert free_service_ports.__wrapped__() == 20160
    sock.bind.side_effect = OSError("Every block occupied")
    with pytest.raises(pytest.fail.Exception, match="Could not find free control ports"):
        free_service_ports.__wrapped__()


@pytest.mark.parametrize("backend,version", _BACKENDS)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_all_backends_render_standalone_slurm_bundle(backend, version, mode, tmp_path):
    params = _params(backend, mode)
    params["generator_dynamo_version"] = "1.2.0"
    artifacts = generate_backend_artifacts(
        params, backend, backend_version=version, deployment_target="slurm", output_dir=str(tmp_path)
    )
    # Every emitted byte has a static expected value. Common fixtures avoid
    # duplicating the unchanged supervisor and shared scripts for each backend.
    expected = {
        name: body
        for path in (_GOLDEN / "common.yaml", _GOLDEN / f"{mode}.yaml", _GOLDEN / backend / version / f"{mode}.yaml")
        for name, body in yaml.safe_load(path.read_text()).items()
    }
    assert artifacts.keys() == expected.keys()
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()} == expected.keys()
    for name, body in expected.items():
        emitted = (tmp_path / name).read_bytes()
        assert emitted == body.encode(), name
        if name.endswith(".json"):
            assert json.loads(artifacts[name]) == json.loads(emitted), name
        elif name.endswith(".yaml"):
            assert yaml.safe_load(artifacts[name]) == yaml.safe_load(emitted), name
        else:
            assert artifacts[name].encode() == emitted, name
    assert not {"k8s_deploy.yaml", "k8s_bench.yaml", "sflow.yaml", "run_0.sh"}.intersection(artifacts)
    for name in expected:
        if name.endswith((".sh", ".sbatch")):
            subprocess.run(["bash", "-n", str(tmp_path / name)], check=True, capture_output=True)
    compile((tmp_path / "slurm_runtime.py").read_bytes(), "slurm_runtime.py", "exec")
    spec = json.loads((tmp_path / "deployment.json").read_bytes())
    assert spec["gpus"] == (2 if mode == "agg" else 4)
    assert f"#SBATCH --gres=gpu:{spec['gpus']}" in artifacts["benchmark.sbatch"]
    for worker in spec["workers"]:
        assert worker["argv"][:3] == ["python3", "-m", f"dynamo.{backend}"]
        assert "/models/Qwen 3" in worker["argv"]
        if backend == "trtllm":
            config = worker["argv"][worker["argv"].index("--extra-engine-args") + 1]
            assert config in artifacts
        if mode == "disagg":
            assert "--disaggregation-mode" in worker["argv"]


@pytest.mark.parametrize("backend,version", _BACKENDS)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_moe_topology_matches_slurm_allocation_and_worker_commands(backend, version, mode):
    params = _params(backend, mode)
    params["ModelConfig"] = {"is_moe": True}
    # The attention TP starts at one; the public generator must derive the
    # four-GPU expert world. vLLM divides that world across attention DP ranks.
    for role in ["agg"] if mode == "agg" else ["prefill", "decode"]:
        params["params"][role].update(
            tensor_parallel_size=1,
            data_parallel_size=2 if backend == "vllm" else 1,
            moe_tensor_parallel_size=2,
            moe_expert_parallel_size=2,
        )
    artifacts = generate_backend_artifacts(params, backend, backend_version=version, deployment_target="slurm")
    spec = json.loads(artifacts["deployment.json"])
    expected_gpus = 4 if mode == "agg" else 8
    assert spec["gpus"] == expected_gpus
    assert [worker["gpu_count"] for worker in spec["workers"]] == [4] * (expected_gpus // 4)
    assert [worker["gpu_offset"] for worker in spec["workers"]] == list(range(0, expected_gpus, 4))
    for name in ("deploy.sbatch", "benchmark.sbatch"):
        assert f"#SBATCH --gres=gpu:{expected_gpus}\n" in artifacts[name]
    for worker in spec["workers"]:
        argv = worker["argv"]
        if backend == "trtllm":
            assert argv[argv.index("--gpus-per-node") + 1] == "4"
            assert worker["env"]["DYN_TRTLLM_OVERRIDE_ENGINE_ARGS"] == ""
            engine = yaml.safe_load(artifacts[argv[argv.index("--extra-engine-args") + 1]])
            assert engine["tensor_parallel_size"] == 4
            assert engine["moe_tensor_parallel_size"] == 2
            assert engine["moe_expert_parallel_size"] == 2
        else:
            assert argv[argv.index("--tensor-parallel-size") + 1] == ("2" if backend == "vllm" else "4")
            assert argv[argv.index("--data-parallel-size") + 1] == ("2" if backend == "vllm" else "1")
            if backend == "vllm":
                assert "--enable-expert-parallel" in argv
            else:
                assert argv[argv.index("--expert-parallel-size") + 1] == "2"


@pytest.mark.parametrize("backend,version", _BACKENDS)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_prefix_cache_enables_router_and_worker_events(backend, version, mode):
    params = _params(backend, mode)
    params["ModelConfig"] = {"prefix": 64}
    artifacts = generate_backend_artifacts(params, backend, backend_version=version, deployment_target="slurm")
    spec = json.loads(artifacts["deployment.json"])
    frontend = spec["frontend"]
    assert frontend[frontend.index("--router-mode") + 1] == "kv"
    for worker in spec["workers"]:
        argv = worker["argv"]
        if backend == "trtllm":
            assert "--publish-events-and-metrics" in argv
            engine = yaml.safe_load(artifacts[argv[argv.index("--extra-engine-args") + 1]])
            assert engine["kv_cache_config"]["enable_block_reuse"] is True
        else:
            assert "--no-enable-prefix-caching" not in argv
            assert "--disable-radix-cache" not in argv
            event = json.loads(argv[argv.index("--kv-events-config") + 1])
            assert event == {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": "tcp://*:@EVENT_PORT@",
                **({"enable_kv_cache_events": True} if backend == "vllm" else {}),
            }


@pytest.mark.parametrize("backend,version", _BACKENDS)
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("nextn", [0, 2])
def test_mtp_settings_reach_every_worker(backend, version, mode, nextn):
    params = _params(backend, mode)
    params["ModelConfig"] = {"nextn": nextn}
    artifacts = generate_backend_artifacts(params, backend, backend_version=version, deployment_target="slurm")
    spec = json.loads(artifacts["deployment.json"])
    for worker in spec["workers"]:
        argv = worker["argv"]
        if backend == "trtllm":
            engine = yaml.safe_load(artifacts[argv[argv.index("--extra-engine-args") + 1]])
            expected = {"decoding_type": "MTP", "num_nextn_predict_layers": nextn} if nextn else None
            assert engine.get("speculative_config") == expected
        elif not nextn:
            assert not any(token.startswith("--speculative-") for token in argv)
        elif backend == "vllm":
            assert json.loads(argv[argv.index("--speculative-config") + 1]) == {
                "method": "mtp",
                "num_speculative_tokens": nextn,
            }
        else:
            for flag, value in (
                ("--speculative-algorithm", "NEXTN"),
                ("--speculative-num-steps", "2"),
                ("--speculative-eagle-topk", "1"),
                ("--speculative-num-draft-tokens", "3"),
            ):
                assert argv[argv.index(flag) + 1] == value


def test_slurm_candidate_rejects_unsupported_prompt_lookup(tmp_path):
    candidate = {"config": {"speculation": {"kind": "ngram", "num_speculative_tokens": 2}}}
    with pytest.raises(SweeperCandidateError, match="ngram deployment generation is unsupported"):
        from_sweeper_candidate(candidate, deployment_target="slurm", output_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_typed_request_preserves_cluster_overrides(tmp_path):
    params = _params()
    req = from_legacy_params(params, "vllm")
    req = dataclasses.replace(req, emit=dataclasses.replace(req.emit, deployment_target="slurm"))
    assert to_legacy_params(req)["SlurmConfig"] == params["SlurmConfig"]
    assert "benchmark.sbatch" in generate_from_request(req, output_dir=str(tmp_path))


@pytest.mark.parametrize("source", ["sdk", "sweeper"])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_sdk_and_sweeper_inputs_reach_persisted_slurm_bundle(source, mode, tmp_path):
    overrides = {
        "generator_dynamo_version": "1.2.0",
        "rule": "benchmark",
        "preserve_engine_limits": True,
        "ServiceConfig": {"served_model_name": "selected-model"},
        "SlurmConfig": {
            "account": "selected-account",
            "partition": "selected-partition",
            "container_image": "/images/selected dynamo.sqsh",
            "container_mounts": ["/checkpoints:/models:ro"],
            "cpus_per_task": 24,
            "memory": "96G",
            "env": {"HF_HUB_OFFLINE": "1", "NCCL_DEBUG": "WARN"},
            "benchmark_concurrency": [2, 8],
            "benchmark_rounds": 3,
            "benchmark_timeout": 2700,
        },
    }
    original_overrides = copy.deepcopy(overrides)
    roles = {"agg": (2, 8)} if mode == "agg" else {"prefill": (1, 8), "decode": (2, 16)}
    if source == "sdk":
        task = SimpleNamespace(
            primary_backend_name="vllm",
            primary_system_name="h200_sxm",
            primary_backend_version="0.20.1",
            primary_model_path="/models/Qwen3",
            prefix=0,
            is_moe=False,
            nextn=0,
            nextn_accepted=None,
            serving_mode=mode,
            total_gpus=0,
            prefill_system_name="h200_sxm",
            decode_system_name="h200_sxm",
            isl=512,
            osl=128,
            ttft=2000.0,
            tpot=50.0,
        )
        row = {}
        for role, (count, batch_size) in roles.items():
            prefix = {"agg": "", "prefill": "(p)", "decode": "(d)"}[role]
            row.update(
                {f"{prefix}{key}": value for key, value in {"tp": 2, "workers": count, "bs": batch_size}.items()}
            )
        params = task_config_to_generator_config(task, pd.Series(row), overrides, num_gpus_per_node=8)
        generate_backend_artifacts(
            params, "vllm", backend_version="0.20.1", deployment_target="slurm", output_dir=str(tmp_path)
        )
    else:
        config = {
            "deployment_mode": mode,
            "model_name": "/models/Qwen3",
            "backend": "vllm",
            "backend_version": "0.20.1",
            "hardware_sku": "h200_sxm",
            "context_length": 8192,
            "used_gpus": 4 if mode == "agg" else 6,
        }
        for role, (count, batch_size) in roles.items():
            prefix = "" if role == "agg" else f"{role}_"
            config.update(
                {
                    f"{prefix}tp": 2,
                    f"{prefix}pp": 1,
                    f"{prefix}attention_dp": 1,
                    f"{prefix}moe_tp": 1,
                    f"{prefix}moe_ep": 1,
                    f"{prefix}replicas": count,
                    f"{role}_max_num_seqs": batch_size,
                    f"{role}_max_num_batched_tokens": 4096,
                    f"{role}_block_size": 16,
                    f"{role}_gpu_memory_utilization": 0.9,
                }
            )
        request = from_sweeper_candidate(
            {"config": config, "used_gpus": config["used_gpus"]},
            workload={"isl": 512, "osl": 128},
            deployment_target="slurm",
            output_dir=str(tmp_path),
            generator_overrides=overrides,
            model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        )
        generate_from_request(request)

    assert overrides == original_overrides
    spec = json.loads((tmp_path / "deployment.json").read_text())
    assert spec["mode"] == mode
    assert spec["gpus"] == (4 if mode == "agg" else 6)
    assert spec["model"] == "selected-model"
    assert spec["env"] == {"HF_HUB_OFFLINE": "1", "NCCL_DEBUG": "WARN"}
    assert spec["benchmark_concurrency"] == [2, 8]
    assert spec["benchmark_rounds"] == 3
    assert spec["benchmark_timeout"] == 2700
    assert spec["startup_timeout"] == 1800
    expected_workers = (
        [("agg-0", 8), ("agg-1", 8)] if mode == "agg" else [("prefill-0", 8), ("decode-0", 16), ("decode-1", 16)]
    )
    assert [worker["name"] for worker in spec["workers"]] == [name for name, _ in expected_workers]
    for index, (worker, (name, batch_size)) in enumerate(zip(spec["workers"], expected_workers, strict=True)):
        assert (worker["gpu_count"], worker["gpu_offset"]) == (2, index * 2)
        argv = worker["argv"]
        assert argv[:3] == ["python3", "-m", "dynamo.vllm"]
        for flag, value in (
            ("--model", "/models/Qwen3"),
            ("--served-model-name", "selected-model"),
            ("--tensor-parallel-size", "2"),
            ("--max-num-seqs", str(batch_size)),
        ):
            assert argv[argv.index(flag) + 1] == value
        if mode == "disagg":
            assert argv[argv.index("--disaggregation-mode") + 1] == name.split("-")[0]
        else:
            assert "--disaggregation-mode" not in argv
        if source == "sweeper":
            assert argv[argv.index("--max-num-batched-tokens") + 1] == "4096"
            assert argv[argv.index("--max-model-len") + 1] == "8192"

    for operation, filename in (("serve", "deploy.sbatch"), ("benchmark", "benchmark.sbatch")):
        job = (tmp_path / filename).read_text()
        for directive in (
            f"--job-name=aic-dynamo-{operation}",
            "--account=selected-account",
            "--partition=selected-partition",
            f"--gres=gpu:{spec['gpus']}",
            "--cpus-per-task=24",
            "--mem=96G",
            "--time=01:00:00",
        ):
            assert f"#SBATCH {directive}\n" in job
        assert f"python3 /work/slurm_runtime.py {operation} &" in job
    environment = (tmp_path / "environment.sh").read_text()
    assert "export AIC_SLURM_IMAGE='/images/selected dynamo.sqsh'" in environment
    assert "export AIC_SLURM_MOUNTS=/checkpoints:/models:ro" in environment
    benchmark = (tmp_path / "bench_run.sh").read_text()
    assert 'BENCH_MODEL="${AICONFIGURATOR_BENCH_MODEL:-selected-model}"' in benchmark
    assert 'BENCH_ISL="${AICONFIGURATOR_BENCH_ISL:-512}"' in benchmark
    assert 'BENCH_OSL="${AICONFIGURATOR_BENCH_OSL:-128}"' in benchmark


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("target", ["slurm", "dynamo-j2"])
@pytest.mark.parametrize("version,legacy", [("1.2.0", True), ("1.3.0", False)])
def test_benchmark_request_matches_dynamo_protocol(backend, target, version, legacy):
    params = _params(backend)
    params["generator_dynamo_version"] = version
    artifacts = generate_backend_artifacts(params, backend, deployment_target=target)
    for name in ["bench_run.sh", *(["k8s_bench.yaml"] if target == "dynamo-j2" else [])]:
        assert "--extra-inputs ignore_eos:true" in artifacts[name]
        assert ("nvext" in artifacts[name]) is legacy


def test_naive_cli_input_preserves_benchmark_sizing_rule(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.generator.naive._get_system_config",
        lambda _: {"gpus_per_node": 8, "vram_per_gpu": 192 * 1024**3},
    )
    monkeypatch.setattr("aisimulate.generator.naive._estimate_model_weight_bytes", lambda *a, **kw: 16 * 1024**3)
    monkeypatch.setattr(
        "aisimulate.generator.naive.get_model_config_from_model_path",
        lambda _: {"architecture": "Qwen3ForCausalLM", "num_experts": 0},
    )
    params = build_naive_generator_params(
        model_name="test/model",
        total_gpus=1,
        system_name="b200_sxm",
        backend_name="vllm",
        generator_overrides={
            "rule": "benchmark",
            "Workers": {"agg": {"max_batch_size": 8}},
            "SlurmConfig": _params()["SlurmConfig"],
        },
    )
    spec = json.loads(generate_backend_artifacts(params, "vllm", deployment_target="slurm")["deployment.json"])
    argv = spec["workers"][0]["argv"]
    assert argv[argv.index("--max-num-seqs") + 1] == "8"


@pytest.mark.parametrize(
    "key,value",
    [
        ("account", "bad\n#SBATCH --nodes=8"),
        ("partition", ""),
        ("container_image", ""),
        ("benchmark_concurrency", [0]),
        ("env", {"CUDA_VISIBLE_DEVICES": "0,1"}),
        ("container_mounts", ["/cache:/cache:ro,/other:/work"]),
    ],
)
def test_invalid_cluster_configuration_fails_before_emission(key, value, tmp_path):
    params = _params()
    params["SlurmConfig"][key] = value
    with pytest.raises(ValueError):
        generate_backend_artifacts(params, "vllm", deployment_target="slurm", output_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_over_capacity_and_encode_pool_fail_closed():
    params = _params(mode="disagg")
    params["WorkerConfig"]["prefill_workers"] = 4
    with pytest.raises(ValueError, match="one node"):
        generate_backend_artifacts(params, "vllm", deployment_target="slurm")
    params = _params()
    params["params"]["encode"] = {"tensor_parallel_size": 1}
    with pytest.raises(ValueError, match="encode"):
        generate_backend_artifacts(params, "vllm", deployment_target="slurm")


def test_submission_dry_run_and_receipt_prevent_duplicate_jobs(tmp_path):
    generate_backend_artifacts(_params(), "vllm", deployment_target="slurm", output_dir=str(tmp_path))
    binary = tmp_path / "bin"
    binary.mkdir()
    sbatch = binary / "sbatch"
    sbatch.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$PWD/calls"\nif [[ $1 == --parsable ]]; then echo "12345;cluster"; fi\n'
    )
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}")
    subprocess.run(["bash", str(tmp_path / "submit.sh"), "benchmark", "--test-only"], env=env, check=True)
    assert not (tmp_path / "job.id").exists()
    subprocess.run(["bash", str(tmp_path / "submit.sh"), "benchmark"], env=env, check=True)
    assert (tmp_path / "job.id").read_text().strip() == "12345;cluster"
    repeated = subprocess.run(["bash", str(tmp_path / "submit.sh"), "benchmark"], env=env, capture_output=True)
    assert repeated.returncode != 0
    assert sum(line.startswith("--parsable") for line in (tmp_path / "calls").read_text().splitlines()) == 1


def test_uncertain_submission_keeps_lock(tmp_path):
    generate_backend_artifacts(_params(), "vllm", deployment_target="slurm", output_dir=str(tmp_path))
    binary = tmp_path / "bin"
    binary.mkdir()
    sbatch = binary / "sbatch"
    sbatch.write_text("#!/bin/bash\nif [[ $1 == --parsable ]]; then exit 1; fi\n")
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}")
    result = subprocess.run(["bash", str(tmp_path / "submit.sh")], env=env, capture_output=True)
    assert result.returncode != 0
    assert (tmp_path / ".submission-lock").is_dir()
    assert not (tmp_path / "job.id").exists()


def test_supervisor_preserves_assigned_gpu_ids(monkeypatch, tmp_path):
    params = _params(mode="disagg")
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        params["ServiceConfig"]["port"] = sock.getsockname()[1]
    artifacts = generate_backend_artifacts(params, "vllm", deployment_target="slurm")
    spec = json.loads(artifacts["deployment.json"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    supervisor = Supervisor(spec, tmp_path)
    # Dynamo 1.2 accepts "etcd" here; "kv_store" is an internal runtime name.
    assert supervisor.env["DYN_DISCOVERY_BACKEND"] == "etcd"
    starts = []
    monkeypatch.setattr(supervisor, "start", lambda name, argv, env=None: starts.append((name, argv, env)))
    monkeypatch.setattr(supervisor, "wait_http", lambda *args: None)
    monkeypatch.setattr(supervisor, "wait_tcp", lambda *args: None)
    monkeypatch.setattr(supervisor, "smoke_test", lambda *args: None)
    monkeypatch.setattr("shutil.which", lambda name, **kwargs: f"/bin/{name}")
    supervisor.start_services()
    workers = [env for name, _, env in starts if name.startswith(("prefill", "decode"))]
    assert [env["CUDA_VISIBLE_DEVICES"] for env in workers] == ["GPU-a,GPU-b", "GPU-c,GPU-d"]
    assert len({env["DYN_SYSTEM_PORT"] for env in workers}) == 2
    assert all(0 < int(env["DYN_SYSTEM_PORT"]) <= 32767 for env in workers)


@pytest.mark.parametrize(
    "backend,version,nixl_offsets",
    [
        ("vllm", "0.11.0", (0, 1, 2, 3)),
        ("vllm", "0.24.0", (0, 1)),
        ("sglang", "0.5.11", (0,)),
        ("trtllm", "1.3.0rc14", (0,)),
    ],
)
def test_emitted_workers_have_noncolliding_service_ports(
    monkeypatch, tmp_path, free_service_ports, backend, version, nixl_offsets
):
    base = free_service_ports
    params = _params(backend, "disagg")
    params["ServiceConfig"]["port"] = base + 39
    if backend == "vllm":
        for role in ("prefill", "decode"):
            params["params"][role]["data_parallel_size"] = 2
    runtime, supervisor = _emitted_supervisor(tmp_path, params, backend, version)
    supervisor.env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(supervisor.spec["gpus"])))
    candidates = iter(base + offset for offset in (0, 1, 2, 3, 4, 5, 6, 8, 7, 9, *range(10, 39)))
    monkeypatch.setattr(runtime.secrets, "randbelow", lambda limit: next(candidates) - 20000)
    monkeypatch.setattr(runtime.shutil, "which", lambda name, **kwargs: f"/fake/{name}")
    commands = {}
    monkeypatch.setattr(supervisor, "start", lambda name, argv, env=None: commands.update({name: (argv, env)}))
    monkeypatch.setattr(supervisor, "wait_http", lambda *args: None)
    monkeypatch.setattr(supervisor, "wait_tcp", lambda *args: None)
    monkeypatch.setattr(supervisor, "smoke_test", lambda *args: None)
    supervisor.start_services()

    # TP=2/DP=2 uses four NIXL listeners in v0.11, two in v0.12+.
    # Bind after all launches are allocated to reproduce delayed rank startup.
    # These are consumer port sets, not a copy of the upstream implementation.
    ports = [base + i for i in (0, 1, 2, 3, 39)]  # etcd, NATS, frontend
    for worker in supervisor.spec["workers"]:
        argv, env = commands[worker["name"]]
        system, event, side = [
            int(env[key]) for key in ("DYN_SYSTEM_PORT", "DYN_VLLM_KV_EVENT_PORT", "VLLM_NIXL_SIDE_CHANNEL_PORT")
        ]
        ports.extend([system, event])
        if backend == "vllm":
            assert worker["gpu_count"] == 4
            assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
            assert argv[argv.index("--data-parallel-size") + 1] == "2"
        ports.extend(side + offset for offset in nixl_offsets)
        if backend == "sglang" and "--disaggregation-bootstrap-port" in argv:
            ports.append(int(argv[argv.index("--disaggregation-bootstrap-port") + 1]))
    with ExitStack() as sockets:
        for port in ports:
            sock = sockets.enter_context(socket.socket())
            sock.bind(("0.0.0.0", port))
    assert len(ports) == len(set(ports))
    # Other backends retain four single-port allocations per worker.
    expected_reservations = 19 if backend == "vllm" else 13
    assert len(supervisor.ports) == expected_reservations
    assert all(20000 <= port < 32000 for port in supervisor.ports)


@pytest.mark.parametrize("occupied", ["external", "reserved"])
def test_emitted_port_range_skips_busy_interior_without_partial_reservation(
    monkeypatch, tmp_path, free_service_ports, occupied
):
    runtime, supervisor = _emitted_supervisor(tmp_path, _params())
    base = free_service_ports
    candidates = iter([base, base + 10, base])
    monkeypatch.setattr(runtime.secrets, "randbelow", lambda limit: next(candidates) - 20000)
    with socket.socket() as busy:
        if occupied == "external":
            busy.bind(("0.0.0.0", base + 1))
        else:
            supervisor.ports.add(base + 1)
        assert supervisor.port(4) == base + 10
        expected = set(range(base + 10, base + 14))
        if occupied == "reserved":
            expected.add(base + 1)
        assert supervisor.ports == expected
        assert supervisor.port() == base


def test_emitted_port_ranges_are_bounded(monkeypatch, tmp_path):
    runtime, supervisor = _emitted_supervisor(tmp_path, _params())
    draws = MagicMock(side_effect=lambda limit: limit - 1)
    monkeypatch.setattr(runtime.secrets, "randbelow", draws)
    # A deterministic socket substitute makes the upper-bound check independent
    # of which ports happen to be occupied on the test host.
    monkeypatch.setattr(runtime.socket, "socket", MagicMock())
    assert supervisor.port(4) == 31996
    assert supervisor.ports == {31996, 31997, 31998, 31999}
    for count in (0, -1, 12001):
        with pytest.raises(ValueError, match="port range"):
            supervisor.port(count)
    draws.reset_mock()
    supervisor.ports = set(range(20000, 32000))
    with pytest.raises(RuntimeError, match="Dynamo-compatible service port"):
        supervisor.port(4)
    assert draws.call_count == 1000
    assert supervisor.ports == set(range(20000, 32000))


def test_supervisor_detects_worker_exit_and_cleans_up(tmp_path):
    supervisor = Supervisor({"env": {}, "port": 8000}, tmp_path)
    healthy = supervisor.start("healthy", [sys.executable, "-c", "import time; time.sleep(60)"])
    failing = supervisor.start("failing", [sys.executable, "-c", "raise SystemExit(7)"])
    failing.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="failing exited with code 7"):
            supervisor.wait_http("http://127.0.0.1:1/health", time.monotonic() + 5)
    finally:
        supervisor.stop()
    assert healthy.poll() is not None
    assert json.loads((tmp_path / "result.json").read_text())["services_stopped"]


def test_benchmark_finds_client_on_configured_path(monkeypatch, tmp_path):
    binary = tmp_path / "client-bin"
    binary.mkdir()
    client = binary / "aiperf"
    client.write_text("#!/bin/sh\nexit 0\n")
    client.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    supervisor = Supervisor(
        {
            "env": {"PATH": str(binary)},
            "port": 8000,
            "model": "test",
            "benchmark_concurrency": [1],
            "benchmark_rounds": 4,
            "benchmark_timeout": 5,
        },
        tmp_path,
    )
    process = subprocess.Popen([str(client)])
    process.wait(timeout=5)
    monkeypatch.setattr(supervisor, "start", lambda *args: process)
    monkeypatch.setattr(supervisor, "validate_benchmark", lambda: None)
    supervisor.benchmark()
    assert supervisor.result["status"] == "passed"


def test_cli_accepts_slurm_target(cli_args_factory):
    args = cli_args_factory(mode="generate", extra_args=["--deployment-target", "slurm"])
    assert args.deployment_target == "slurm"


@pytest.mark.usefixtures("local_http_without_proxy")
def test_health_accepts_plain_http_response(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ready")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        supervisor = Supervisor({"env": {}, "port": 8000}, tmp_path)
        assert supervisor.wait_http(f"http://127.0.0.1:{server.server_port}/health", time.monotonic() + 2) == b"ready"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_supervisor_loopback_requests_bypass_inherited_proxies(monkeypatch, tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ready")

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "Hello"}}]}).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.socket() as unavailable_proxy:
            unavailable_proxy.bind(("127.0.0.1", 0))
            proxy = f"http://127.0.0.1:{unavailable_proxy.getsockname()[1]}"
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                monkeypatch.setenv(key, proxy)
            for key in ("NO_PROXY", "no_proxy"):
                monkeypatch.setenv(key, "")
            # Other HTTP users retain their configured proxy behavior.
            environment = os.environ.copy()
            params = _params()
            params["ServiceConfig"]["port"] = server.server_port
            params["SlurmConfig"]["env"] = {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
            generate_backend_artifacts(params, "vllm", deployment_target="slurm", output_dir=str(tmp_path))
            module_spec = importlib.util.spec_from_file_location("slurm_probe_runtime", tmp_path / "slurm_runtime.py")
            runtime = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(runtime)
            supervisor = runtime.Supervisor(
                json.loads((tmp_path / "deployment.json").read_bytes()), tmp_path / "results"
            )
            assert supervisor.wait_http(f"{supervisor.base_url}/health", time.monotonic() + 2) == b"ready"
            supervisor.smoke_test(time.monotonic() + 2)
            assert supervisor.result["smoke_response"]["choices"][0]["message"]["content"] == "Hello"
            assert os.environ == environment
            assert supervisor.env["HTTP_PROXY"] == proxy
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "payload,outcome",
    [
        (b'{"data":[{"id":"test-model"}]}', "ready"),
        (b'{"data":[{"id":"other-model"}]}', "ready"),
        (b"", "ready"),
        (b"not-json", "ready"),
        (b"[]", "ready"),
        (b"null", "ready"),
        (b'"test-model"', "ready"),
        (b"42", "ready"),
        (b"{}", "ready"),
        (b'{"data":null}', "ready"),
        (b'{"data":{}}', "ready"),
        (b'{"data":"test-model"}', "ready"),
        (b'{"data":[null,"test-model",42,{}]}', "ready"),
        (b'{"data":[{"id":"test-model-other"}]}', "ready"),
        (b'{"data":null}', "timeout"),
        (b'{"data":null}', "child-exit"),
    ],
)
def test_emitted_model_readiness_retries_http_responses(monkeypatch, tmp_path, payload, outcome):
    params = _params()
    with socket.socket() as finder:
        finder.bind(("127.0.0.1", 0))
        params["ServiceConfig"]["port"] = finder.getsockname()[1]
    params["SlurmConfig"]["startup_timeout"] = 2 if outcome == "timeout" else 5
    runtime, supervisor = _emitted_supervisor(tmp_path, params)
    supervisor.env["CUDA_VISIBLE_DEVICES"] = "GPU-a,GPU-b"
    valid = b'{"data":[{"id":"test-model"}]}'
    requests = []
    server = None
    thread = None
    actual_start = supervisor.start

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            assert self.path == "/v1/models"
            if outcome == "child-exit":
                child = actual_start("failed-worker", [sys.executable, "-c", "raise SystemExit(7)"])
                child.wait(timeout=5)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload if len(requests) == 1 or outcome != "ready" else valid)

        def do_POST(self):
            requests.append(self.path)
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "test-model"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"Hello"}}]}')

    def start(name, argv, env=None):
        nonlocal server, thread
        if name == "frontend":
            server = HTTPServer(("127.0.0.1", supervisor.spec["port"]), Handler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()

    actual_wait_http = supervisor.wait_http

    def wait_http(url, deadline, predicate=None):
        if url.endswith("/v1/models"):
            return actual_wait_http(url, deadline, predicate)
        return b"healthy-synthetic-service"

    # Infrastructure and GPU launches are simulated. The persisted model
    # predicate, polling, HTTP requests, smoke test and child checks all run.
    monkeypatch.setattr(supervisor, "start", start)
    monkeypatch.setattr(supervisor, "wait_http", wait_http)
    monkeypatch.setattr(supervisor, "wait_tcp", lambda *args: None)
    monkeypatch.setattr(runtime.shutil, "which", lambda name, **kwargs: f"/fake/{name}")
    try:
        if outcome == "ready":
            supervisor.start_services()
            assert requests == ["/v1/models"] * (1 if payload == valid else 2) + ["/v1/chat/completions"]
            result = json.loads((tmp_path / "results/result.json").read_text())
            assert result["status"] == "ready"
            assert result["smoke_response"]["choices"][0]["message"]["content"] == "Hello"
        else:
            error, message = (
                (TimeoutError, "Readiness deadline exceeded.*v1/models")
                if outcome == "timeout"
                else (RuntimeError, "failed-worker exited with code 7")
            )
            with pytest.raises(error, match=message):
                supervisor.start_services()
            assert requests == ["/v1/models"] * (2 if outcome == "timeout" else 1)
            assert supervisor.result["status"] == "starting"
            assert "smoke_response" not in supervisor.result
    finally:
        supervisor.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join()


@pytest.mark.parametrize(
    "failure,outcome",
    [
        (404, "ready"),
        (503, "ready"),
        (400, "fatal"),
        ("disconnect", "ready"),
        ("disconnect", "timeout"),
        ("disconnect", "child-exit"),
    ],
)
def test_emitted_inference_readiness_retries_temporary_failures(tmp_path, failure, outcome):
    class Handler(BaseHTTPRequestHandler):
        attempts = 0

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            assert body["model"] == "test-model"
            Handler.attempts += 1
            if outcome == "child-exit":
                child = supervisor.start("failed-worker", [sys.executable, "-c", "raise SystemExit(7)"])
                child.wait(timeout=5)
            failing = Handler.attempts == 1 or outcome != "ready"
            if failure == "disconnect" and failing:
                self.close_connection = True
                return
            self.send_response(failure if failing else 200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "Hello"}}]}).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    params = _params()
    params["ServiceConfig"]["port"] = server.server_port
    _, supervisor = _emitted_supervisor(tmp_path, params)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if outcome == "fatal":
            with pytest.raises(urllib.error.HTTPError) as error:
                supervisor.smoke_test(time.monotonic() + 5)
            assert error.value.code == failure
            assert Handler.attempts == 1
            assert "readiness_retries" not in supervisor.result
        elif outcome == "ready":
            supervisor.smoke_test(time.monotonic() + 5)
            assert Handler.attempts == 2
            result = json.loads((tmp_path / "results/result.json").read_text())
            assert result["readiness_retries"] == 1
            assert result["last_readiness_error"]
            assert result["smoke_response"]["choices"][0]["message"]["content"] == "Hello"
        else:
            error, message = (
                (TimeoutError, "Readiness deadline exceeded waiting for frontend inference")
                if outcome == "timeout"
                else (RuntimeError, "failed-worker exited with code 7")
            )
            started = time.monotonic()
            with pytest.raises(error, match=message):
                supervisor.smoke_test(started + (1.5 if outcome == "timeout" else 5))
            assert time.monotonic() - started < 3
            assert Handler.attempts == (2 if outcome == "timeout" else 1)
            assert supervisor.result["readiness_retries"] == Handler.attempts
            assert "smoke_response" not in supervisor.result
    finally:
        supervisor.stop()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("failure", ["refused", "timeout"])
def test_emitted_inference_readiness_retries_socket_errors(monkeypatch, tmp_path, failure):
    class Handler(BaseHTTPRequestHandler):
        attempts = 0

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            Handler.attempts += 1
            if failure == "timeout" and Handler.attempts == 1:
                time.sleep(0.1)
                self.close_connection = True
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"Hello"}}]}')

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    if failure == "timeout":
        thread.start()
    else:
        server.server_close()
    params = _params()
    params["ServiceConfig"]["port"] = server.server_port
    _, supervisor = _emitted_supervisor(tmp_path, params)
    actual_request = supervisor.request
    errors = []

    def request(*args, **kwargs):
        nonlocal server, thread
        # Shorten the first transport timeout; retries retain the real deadline.
        if failure == "timeout" and not errors:
            kwargs["timeout"] = 0.05
        try:
            return actual_request(*args, **kwargs)
        except OSError as error:
            errors.append(error)
            if failure == "refused":
                server = HTTPServer(server.server_address, Handler)
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
            raise

    monkeypatch.setattr(supervisor, "request", request)
    try:
        supervisor.smoke_test(time.monotonic() + 5)
        assert len(errors) == 1
        if failure == "refused":
            assert isinstance(errors[0], urllib.error.URLError)
            assert isinstance(errors[0].reason, ConnectionRefusedError)
        else:
            assert isinstance(errors[0], TimeoutError)
        result = json.loads((tmp_path / "results/result.json").read_text())
        assert result["readiness_retries"] == 1
        assert result["last_readiness_error"] == str(errors[0])
        assert result["smoke_response"]["choices"][0]["message"]["content"] == "Hello"
    finally:
        if thread.ident is not None:
            server.shutdown()
            thread.join()
        server.server_close()


@pytest.mark.parametrize(
    "payload,error",
    [
        (b"not-json", json.JSONDecodeError),
        (b"{}", KeyError),
        (b'{"choices":[{"message":{"content":""}}]}', RuntimeError),
        (b'{"choices":[{"message":{"reasoning_content":"Hello"}}]}', None),
    ],
)
def test_emitted_inference_readiness_validates_generated_text(tmp_path, payload, error):
    class Handler(BaseHTTPRequestHandler):
        attempts = 0

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            Handler.attempts += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    params = _params()
    params["ServiceConfig"]["port"] = server.server_port
    _, supervisor = _emitted_supervisor(tmp_path, params)
    try:
        if error is None:
            supervisor.smoke_test(time.monotonic() + 5)
            assert supervisor.result["smoke_response"] == json.loads(payload)
        else:
            with pytest.raises(error):
                supervisor.smoke_test(time.monotonic() + 5)
        assert Handler.attempts == 1
        assert "readiness_retries" not in supervisor.result
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("count,errors,cancelled", [(0, 0, False), (4, 1, False), (4, 0, True)])
def test_benchmark_success_requires_complete_error_free_reports(tmp_path, count, errors, cancelled):
    supervisor = Supervisor({"env": {}, "port": 8000, "benchmark_concurrency": [1], "benchmark_rounds": 4}, tmp_path)
    report = tmp_path / "benchmark/concurrency_1/profile_export_aiperf.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {"request_count": {"avg": count}, "error_request_count": {"avg": errors}, "was_cancelled": cancelled}
        )
    )
    with pytest.raises(RuntimeError, match="Invalid benchmark"):
        supervisor.validate_benchmark()
    report.write_text(json.dumps({"request_count": {"avg": 4}, "error_request_count": {"avg": 0}}))
    supervisor.validate_benchmark()
    assert supervisor.result["benchmarks"][0]["requests"] == 4
