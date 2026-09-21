# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from aisimulate.generator.api import (
    generate_backend_artifacts,
    generate_config_from_input_dict,
    generate_from_request,
)
from aisimulate.generator.builders.slurm_runtime import Supervisor
from aisimulate.generator.naive import build_naive_generator_params
from aisimulate.generator.request import from_legacy_params, to_legacy_params


def _params(backend="vllm", mode="agg"):
    return generate_config_from_input_dict(
        {
            "ServiceConfig": {"model_path": "/models/Qwen 3", "served_model_name": "test-model"},
            "DynConfig": {"mode": mode},
            "Workers": {
                role: {"tensor_parallel_size": 2, "max_batch_size": 8}
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


@pytest.mark.parametrize("backend,version", [("vllm", "0.20.1"), ("sglang", "0.5.11"), ("trtllm", "1.3.0rc14")])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_all_backends_render_standalone_slurm_bundle(backend, version, mode, tmp_path):
    artifacts = generate_backend_artifacts(
        _params(backend, mode), backend, backend_version=version, deployment_target="slurm", output_dir=str(tmp_path)
    )
    assert set(artifacts) >= {
        "deploy.sbatch",
        "benchmark.sbatch",
        "submit.sh",
        "environment.sh",
        "deployment.json",
        "slurm_runtime.py",
        "bench_run.sh",
    }
    assert not {"k8s_deploy.yaml", "k8s_bench.yaml", "sflow.yaml", "run_0.sh"}.intersection(artifacts)
    for name, body in artifacts.items():
        assert (tmp_path / name).is_file()
        if name.endswith((".sh", ".sbatch")):
            subprocess.run(["bash", "-n"], input=body, text=True, check=True, capture_output=True)
    compile(artifacts["slurm_runtime.py"], "slurm_runtime.py", "exec")
    spec = json.loads(artifacts["deployment.json"])
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


def test_typed_request_preserves_cluster_overrides(tmp_path):
    params = _params()
    req = from_legacy_params(params, "vllm")
    req = dataclasses.replace(req, emit=dataclasses.replace(req.emit, deployment_target="slurm"))
    assert to_legacy_params(req)["SlurmConfig"] == params["SlurmConfig"]
    assert "benchmark.sbatch" in generate_from_request(req, output_dir=str(tmp_path))


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
    artifacts = generate_backend_artifacts(_params(mode="disagg"), "vllm", deployment_target="slurm")
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


@pytest.mark.parametrize("status", [404, 503, 400])
def test_inference_readiness_handles_late_frontend_registration(tmp_path, status):
    class Handler(BaseHTTPRequestHandler):
        attempts = 0

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            Handler.attempts += 1
            self.send_response(status if Handler.attempts == 1 else 200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "Hello"}}]}).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        supervisor = Supervisor({"env": {}, "port": server.server_port, "model": "test"}, tmp_path)
        if status == 400:
            with pytest.raises(urllib.error.HTTPError):
                supervisor.smoke_test(time.monotonic() + 5)
            assert Handler.attempts == 1
        else:
            supervisor.smoke_test(time.monotonic() + 5)
            assert supervisor.result["readiness_retries"] == 1
            assert supervisor.result["smoke_response"]["choices"][0]["message"]["content"] == "Hello"
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
