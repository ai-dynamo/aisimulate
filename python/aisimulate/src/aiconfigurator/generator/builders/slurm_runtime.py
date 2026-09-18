# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone supervisor copied into generated Slurm bundles (stdlib only)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class Supervisor:
    def __init__(self, spec, output):
        self.spec = spec
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.children = []
        self.logs = []
        self.ports = set()
        self.result = {
            "status": "starting",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "started_at": time.time(),
            "commands": {},
        }
        self.env = os.environ.copy()
        self.env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "DYN_REQUEST_PLANE": "tcp",
                "DYN_STORE_KV": "etcd",
                "DYN_DISCOVERY_BACKEND": "etcd",
                "DYN_NAMESPACE": f"aic-{os.getpid()}",
                "HF_HOME": str(self.output / "cache/huggingface"),
                "XDG_CACHE_HOME": str(self.output / "cache"),
                "TRITON_CACHE_DIR": str(self.output / "cache/triton"),
            }
        )
        self.env.update(spec["env"])
        self.base_url = f"http://127.0.0.1:{spec['port']}"

    def save(self):
        temporary = self.output / "result.json.tmp"
        temporary.write_text(json.dumps(self.result, indent=2) + "\n")
        temporary.replace(self.output / "result.json")

    def port(self):
        # Dynamo 1.2's SYSTEM_PORT parser uses a signed 16-bit integer.
        # OS-assigned ephemeral ports commonly exceed that range.
        for _ in range(1000):
            port = 20000 + secrets.randbelow(12000)
            if port in self.ports:
                continue
            with socket.socket() as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    continue
            self.ports.add(port)
            return port
        raise RuntimeError("Could not find a free Dynamo-compatible service port")

    def start(self, name, argv, env=None):
        log = (self.output / f"{name}.log").open("w")
        self.logs.append(log)
        child_env = self.env.copy()
        child_env.update(env or {})
        process = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append((name, process))
        self.result["commands"][name] = argv
        self.save()
        return process

    def check_children(self, ignore=None):
        for name, process in self.children:
            if process is not ignore and process.poll() is not None:
                raise RuntimeError(f"{name} exited with code {process.returncode}; see {name}.log")

    @staticmethod
    def request(url, body=None, timeout=5, decode_json=True):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
        return json.loads(payload) if payload and decode_json else payload

    def wait_http(self, url, deadline, predicate=None):
        while time.monotonic() < deadline:
            self.check_children()
            try:
                # Health only needs HTTP success; do not assume a JSON body.
                # Model discovery additionally checks the JSON model list.
                value = self.request(url, decode_json=predicate is not None)
                if predicate is None or predicate(value):
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError(f"Readiness deadline exceeded: {url}")

    def wait_tcp(self, port, deadline):
        while time.monotonic() < deadline:
            self.check_children()
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.2)
        raise TimeoutError(f"Readiness deadline exceeded for port {port}")

    def start_services(self):
        devices = [device.strip() for device in self.env.get("CUDA_VISIBLE_DEVICES", "").split(",") if device.strip()]
        if len(devices) < self.spec["gpus"] or any(device in {"-1", "NoDevFiles"} for device in devices):
            raise RuntimeError(f"Expected at least {self.spec['gpus']} Slurm-assigned GPUs, got {devices}")
        self.result["cuda_visible_devices"] = devices
        versions = {}
        for package in ("ai-dynamo", "ai-dynamo-runtime", self.spec["backend"], "aiperf", "torch"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        self.result["versions"] = versions
        # Reserve the user-facing port early so a busy port fails before model loading.
        with socket.socket() as sock:
            sock.bind(("0.0.0.0", self.spec["port"]))
        self.ports.add(self.spec["port"])
        etcd = shutil.which("etcd", path=self.env.get("PATH"))
        if etcd is None and Path("/usr/local/bin/etcd/etcd").is_file():
            etcd = "/usr/local/bin/etcd/etcd"
        nats = shutil.which("nats-server", path=self.env.get("PATH"))
        if not etcd or not nats:
            raise RuntimeError("The Dynamo image must include etcd and nats-server on PATH")
        client, peer, nats_port = self.port(), self.port(), self.port()
        self.env["ETCD_ENDPOINTS"] = f"http://127.0.0.1:{client}"
        self.env["NATS_SERVER"] = f"nats://127.0.0.1:{nats_port}"
        self.start(
            "etcd",
            [
                etcd,
                "--name",
                "aic",
                "--data-dir",
                str(self.output / "etcd"),
                "--listen-client-urls",
                self.env["ETCD_ENDPOINTS"],
                "--advertise-client-urls",
                self.env["ETCD_ENDPOINTS"],
                "--listen-peer-urls",
                f"http://127.0.0.1:{peer}",
                "--initial-advertise-peer-urls",
                f"http://127.0.0.1:{peer}",
                "--initial-cluster",
                f"aic=http://127.0.0.1:{peer}",
            ],
        )
        self.start(
            "nats", [nats, "-a", "127.0.0.1", "-p", str(nats_port), "-js", "--store_dir", str(self.output / "nats")]
        )
        deadline = time.monotonic() + self.spec["startup_timeout"]
        self.wait_http(f"http://127.0.0.1:{client}/health", deadline)
        self.wait_tcp(nats_port, deadline)
        self.start("frontend", self.spec["frontend"], {"DYN_SYSTEM_PORT": str(self.port())})
        health_ports = []
        for worker in self.spec["workers"]:
            system_port, event_port, side_port, bootstrap_port = [self.port() for _ in range(4)]
            health_ports.append(system_port)
            selected = devices[worker["gpu_offset"] : worker["gpu_offset"] + worker["gpu_count"]]
            worker_env = dict(worker["env"])
            worker_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": ",".join(selected),
                    "DYN_SYSTEM_PORT": str(system_port),
                    "DYN_VLLM_KV_EVENT_PORT": str(event_port),
                    "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_port),
                }
            )
            argv = [
                token.replace("@EVENT_PORT@", str(event_port)).replace("@BOOTSTRAP_PORT@", str(bootstrap_port))
                for token in worker["argv"]
            ]
            self.start(worker["name"], argv, worker_env)
        for port in health_ports:
            self.wait_http(f"http://127.0.0.1:{port}/health", deadline)
        self.wait_http(
            self.base_url + "/v1/models",
            deadline,
            lambda data: any(item.get("id") == self.spec["model"] for item in data.get("data", [])),
        )
        # P/D model discovery can precede frontend route registration. Require
        # real inference before either a persistent service or benchmark is ready.
        self.smoke_test(deadline)
        self.result.update(
            {
                "status": "ready",
                "endpoint": self.base_url,
                "ready_at": time.time(),
                "startup_seconds": time.time() - self.result["started_at"],
            }
        )
        self.save()

    def smoke_test(self, deadline):
        while time.monotonic() < deadline:
            self.check_children()
            try:
                response = self.request(
                    self.base_url + "/v1/chat/completions",
                    {
                        "model": self.spec["model"],
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "max_tokens": 32,
                        "temperature": 0,
                        "stream": False,
                    },
                    timeout=max(0.1, min(120, deadline - time.monotonic())),
                )
                break
            except urllib.error.HTTPError as error:
                if error.code not in {404, 503}:
                    raise
                self.result["last_readiness_error"] = str(error)
                self.result["readiness_retries"] = self.result.get("readiness_retries", 0) + 1
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        else:
            raise TimeoutError("Readiness deadline exceeded waiting for frontend inference")
        self.result["smoke_response"] = response
        message = response["choices"][0]["message"]
        if not (message.get("content") or message.get("reasoning_content")):
            raise RuntimeError("Smoke request returned no generated text")
        self.save()

    def benchmark(self):
        if shutil.which("aiperf", path=self.env.get("PATH")) is None:
            raise RuntimeError("Benchmark requires aiperf on PATH in the Dynamo container")
        self.result["status"] = "benchmarking"
        env = {
            "AICONFIGURATOR_BENCH_ENDPOINT_URL": self.base_url,
            "AICONFIGURATOR_BENCH_MODEL": self.spec["model"],
            "AICONFIGURATOR_BENCH_CONCURRENCY": " ".join(map(str, self.spec["benchmark_concurrency"])),
            "AICONFIGURATOR_BENCH_MULTI_ROUND": str(self.spec["benchmark_rounds"]),
            "BENCH_ARTIFACT_DIR": str(self.output / "benchmark"),
        }
        bench = self.start("benchmark", ["bash", "bench_run.sh"], env)
        deadline = time.monotonic() + self.spec["benchmark_timeout"]
        while bench.poll() is None:
            self.check_children(ignore=bench)
            if time.monotonic() >= deadline:
                raise TimeoutError("Benchmark deadline exceeded")
            time.sleep(1)
        if bench.returncode:
            raise RuntimeError(f"AIPerf exited with code {bench.returncode}; see benchmark.log")
        self.validate_benchmark()
        self.result["status"] = "passed"

    def validate_benchmark(self):
        summaries = []
        for concurrency in self.spec["benchmark_concurrency"]:
            report_path = self.output / "benchmark" / f"concurrency_{concurrency}" / "profile_export_aiperf.json"
            report = json.loads(report_path.read_text())
            count = (report.get("request_count") or {}).get("avg", 0)
            errors = (report.get("error_request_count") or {}).get("avg", 0)
            expected = concurrency * self.spec["benchmark_rounds"]
            if count != expected or errors or report.get("error_summary") or report.get("was_cancelled"):
                raise RuntimeError(
                    f"Invalid benchmark at concurrency {concurrency}: "
                    f"expected {expected} successful requests, got {count}, errors={errors}"
                )
            summaries.append(
                {
                    "concurrency": concurrency,
                    "requests": count,
                    "errors": errors,
                    "metrics": {
                        key: report.get(key)
                        for key in (
                            "time_to_first_token",
                            "inter_token_latency",
                            "output_token_throughput",
                            "request_throughput",
                            "request_latency",
                            "input_sequence_length",
                            "output_sequence_length",
                        )
                    },
                    "report": str(report_path),
                }
            )
        self.result["benchmarks"] = summaries

    def stop(self):
        for _, process in reversed(self.children):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 20
        for _, process in reversed(self.children):
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        # A launcher can exit before its GPU subprocesses; terminate every owned
        # process group even if the group leader has already exited.
        for _, process in reversed(self.children):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for log in self.logs:
            log.close()
        self.result["stopped_at"] = time.time()
        self.result["services_stopped"] = True
        self.save()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"serve", "benchmark"}:
        raise SystemExit("Usage: python3 slurm_runtime.py serve|benchmark")
    bundle = Path(__file__).resolve().parent
    os.chdir(bundle)
    spec = json.loads((bundle / "deployment.json").read_text())
    output = bundle / "results" / os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    supervisor = Supervisor(spec, output)

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        supervisor.start_services()
        if sys.argv[1] == "benchmark":
            supervisor.benchmark()
        else:
            while True:
                supervisor.check_children()
                time.sleep(1)
    except BaseException as error:
        supervisor.result.update({"status": "failed", "error": str(error)})
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        supervisor.stop()


if __name__ == "__main__":
    main()
