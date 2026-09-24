# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public GLM operation case population and real native serving launch."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from collector.case_generator import _framework_specific_model_case_values, get_base_common_case_values
from collector.glm53flash_contract import (
    CHECKPOINTS,
    WHOLE_FORWARD_RANK,
    aggregate_rank_records,
    build_model_manifest,
    runtime_source_pins,
    sha256_json,
    validate_native_workload,
)
from collector.glm53flash_jsonl import iter_records
from collector.glm53flash_protocol import MAX_MEASURED_CONTEXT, sglang_runtime_context_length, vllm_context_policy


def get_test_cases(backend: str) -> list[dict]:
    sweep = get_base_common_case_values("glm53flash_module")
    if not sweep:
        raise ValueError("missing native GLM workload declarations")
    cases = []
    prefill = [
        {
            "batch_size": point["batch_size"],
            "total_prefill_tokens": point["batch_size"] * point["query"],
            "total_kv_read_tokens": point["batch_size"] * point["prefix"],
        }
        for point in sweep["prefill_points"]
    ]
    decode = [
        {"batch_size": batch, "total_kv_read_tokens": batch * prefix}
        for batch in sweep["decode_batch_sizes"]
        for prefix in sweep["decode_prefix_lengths"]
    ]
    for model in _framework_specific_model_case_values("glm53flash_module", backend):
        fmt = model["checkpoint_format"]
        if CHECKPOINTS[fmt][0] != model["model_path"]:
            raise ValueError("native checkpoint aliases cannot share calibration identities")
        for tp in model["tensor_parallel_sizes"]:
            for phase, points in (("prefill", prefill), ("decode", decode)):
                cases.append(
                    {
                        "id": f"glm53flash_{backend}_{fmt}_tp{tp}_{phase}_eager",
                        "params": [backend, model["model_path"], fmt, tp, phase, points],
                    }
                )
    return cases


def verify_target_completeness(output: Path, tp_size: int) -> dict:
    mapping = json.loads((output / "requests.json").read_text())
    expected = {(entry["benchmark_id"], entry["repetition"]) for entry in mapping["requests"].values()}
    for rank in range(tp_size):
        observed = []
        for record in iter_records(output / f"forward-rank-{rank}.jsonl"):
            if record["stage"] == "measure":
                if record.get("gpu_completed") is not True or record.get("state_layout_admitted") is not True:
                    raise ValueError("target forward lacks native completion/state admission")
                observed.append((record["benchmark_id"], record["repetition"]))
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(f"rank{rank} did not complete every frozen point/repetition exactly once")
    return mapping


def native_command(
    backend,
    checkpoint,
    revision,
    tp,
    phase,
    output,
    corpus,
    *,
    ops_execution_mode="eager",
    sglang_mem_fraction_static=None,
):
    from collector.fpm_forward.config import validate_sglang_mem_fraction_static

    native_prefill = ops_execution_mode == "native_eager_prefill"
    if ops_execution_mode not in ("eager", "native_eager_prefill") or (
        native_prefill and (backend != "sglang" or phase != "prefill")
    ):
        raise ValueError("unsupported native operation command mode/backend/phase")
    validate_sglang_mem_fraction_static(sglang_mem_fraction_static)
    if backend != "sglang" and sglang_mem_fraction_static is not None:
        raise ValueError("SGLang memory policy requires the SGLang backend")
    common = ["--benchmark-mode", phase, "--benchmark-points-file", str(output / "points.json")]
    if backend == "sglang":
        return [
            sys.executable,
            "-m",
            "collector.fpm_forward.sglang_driver",
            "--model-path",
            checkpoint,
            "--revision",
            revision,
            "--tp-size",
            str(tp),
            "--context-length",
            str(sglang_runtime_context_length(MAX_MEASURED_CONTEXT)),
            "--benchmark-max-context-length",
            str(MAX_MEASURED_CONTEXT),
            "--max-running-requests",
            "32",
            "--chunked-prefill-size",
            "8192",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--disable-radix-cache",
            "--cuda-graph-backend-decode",
            "full" if native_prefill else "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--observation-purpose",
            "ops",
            "--tokenizer-revision",
            revision,
            "--input-text",
            str(corpus),
            "--benchmark-output",
            str(output / "benchmark.json"),
            *(["--ops-native-prefill"] if native_prefill else []),
            *(
                ["--mem-fraction-static", str(sglang_mem_fraction_static)]
                if sglang_mem_fraction_static is not None
                else []
            ),
            *common,
        ]
    return [
        sys.executable,
        "-m",
        "dynamo.vllm",
        "--model",
        checkpoint,
        "--revision",
        revision,
        "--tensor-parallel-size",
        str(tp),
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--max-model-len",
        str(vllm_context_policy(MAX_MEASURED_CONTEXT)["runtime_context_length"]),
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "8192",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--language-model-only",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--enforce-eager",
        "--cudagraph-metrics",
        "--distributed-executor-backend",
        "mp",
        "--benchmark-warmup-iterations",
        "0",
        "--benchmark-timeout",
        "10800",
        "--benchmark-output-path",
        str(output / "benchmark.json"),
        "--dump-config-to",
        str(output / "resolved-config.json"),
        *common,
    ]


def _execute(command, output, env, backend):
    with (output / "native.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if backend == "sglang":
                if process.wait(timeout=14400) != 0:
                    raise RuntimeError("native SGLang campaign failed; original native.log is retained")
            else:
                deadline = time.monotonic() + 14400
                while not (output / "benchmark.json").is_file():
                    if process.poll() is not None:
                        raise RuntimeError("native vLLM exited before its completed campaign receipt")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native vLLM campaign timed out; original evidence is retained")
                    time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)


def verify_sources(backend: str, pins: dict, output: Path) -> None:
    from collector.glm53flash_runtime_identity import observe_vllm_runtime_closure, validate_backend_version

    backend_version = validate_backend_version(backend, version(backend))
    audit = {"backend": backend, "backend_version": backend_version, "sources": {}}
    try:
        import torch

        audit["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if not audit["devices"] or any("GB300" not in name.upper() for name in audit["devices"]):
            raise ValueError("native GLM calibration requires actual GB300 devices")
        for name, expected in pins.items():
            module_path = name.removesuffix("/__init__.py") if name.endswith("/__init__.py") else name[:-3]
            module = module_path.replace("/", ".") if backend == "vllm" else "sglang." + module_path.replace("/", ".")
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"pinned native module is missing: {name}")
            actual = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
            audit["sources"][name] = actual
            if actual != expected:
                raise RuntimeError(f"pinned native module differs: {name}")
        if backend == "vllm":
            manifest = Path(__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"
            closure = observe_vllm_runtime_closure(backend_version, manifest)
            if closure is not None:
                audit["runtime_closure"] = closure
        audit["status"] = "passed"
    except BaseException as error:
        audit.update(status="failed", error=str(error))
        raise
    finally:
        (output / "runtime-preflight.json").write_text(json.dumps(audit, sort_keys=True, indent=2))


def run_native(backend, model_path, checkpoint_format, tp_size, phase, points, *, perf_filename, device="cuda:0"):
    from collector.helper import log_perf

    del device  # One native TP invocation owns its visible GPU group.
    from collector.glm53flash_runtime_identity import validate_backend_version

    backend_version = validate_backend_version(backend, version(backend))
    checkpoint = json.loads(Path(os.environ["AISIM_GLM53_MODEL_PATHS"]).read_text())[model_path]
    corpus = Path(os.environ["AISIM_GLM53_INPUT_TEXT"]).resolve()
    runtime_digest = os.environ["AISIM_GLM53_RUNTIME_DIGEST"]
    if backend == "vllm" and not os.environ.get("ETCD_ENDPOINTS"):
        raise RuntimeError("Dynamo native collection requires its allocation-local ETCD_ENDPOINTS")
    output = (
        Path(perf_filename).resolve().parent
        / f"glm53flash-{backend}-{checkpoint_format}-tp{tp_size}-{phase}"
        / uuid.uuid4().hex
    )
    output.mkdir(parents=True)
    manifest = build_model_manifest(backend, checkpoint_format, tp_size, backend_version)
    if sha256_json(json.loads((Path(checkpoint) / "config.json").read_bytes())) != manifest["config_sha256"]:
        raise RuntimeError("actual checkpoint config differs from pinned production model")
    runtime = (
        Path(__file__).parent / "fpm_forward/runtime" / ("glm53flash" if backend == "vllm" else "glm53flash_sglang")
    )
    pins = runtime_source_pins(backend, backend_version)
    provenance = {
        key: manifest[key]
        for key in ("backend", "backend_version", "backend_revision", "checkpoint_revision", "config_sha256")
    }
    provenance.update(source_sha256=sha256_json(pins), runtime_digest=runtime_digest)
    for name, value in (("manifest.json", manifest), ("points.json", {phase: points}), ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(value, sort_keys=True, indent=2))
    unsupported = []
    for benchmark_id, point in enumerate(points, start=1):
        batch = point["batch_size"]
        prefix = point["total_kv_read_tokens"] // batch
        query = point["total_prefill_tokens"] // batch if phase == "prefill" else 1
        try:
            validate_native_workload(backend, phase, prefix, query, backend_version)
        except ValueError as error:
            unsupported.append({"benchmark_id": benchmark_id, "point": point, "reason": str(error)})
    if unsupported:
        (output / "qualification-failures.json").write_text(
            json.dumps(
                {"status": "unqualified_native_geometry", "requested_points": len(points), "failures": unsupported},
                indent=2,
            )
        )
        raise ValueError(
            f"{len(unsupported)} frozen GLM points require unsupported native cached-prefill starts; evidence: {output}"
        )
    env = {
        **os.environ,
        "AISIM_GLM53_PURPOSE": "ops",
        "AISIM_GLM53_DISPATCH_PROFILING": "1",
        "AISIM_GLM53_TRACE_DIR": str(output),
        "AISIM_GLM53_OPS_MANIFEST": str(output / "manifest.json"),
        "AISIM_GLM53_PROVENANCE": str(output / "provenance.json"),
        "AISIM_GLM53_OPS_PROVENANCE": str(output / "provenance.json"),
        "AISIM_GLM53_REQUEST_MANIFEST": str(output / "requests.json"),
        "DYN_FPM_GLM53FLASH_REAL_KV": "1",
        "DYN_FPM_GLM53FLASH_MEASURED_CONTEXT": str(MAX_MEASURED_CONTEXT),
        "DYN_FPM_INPUT_TEXT": str(corpus),
        "DYN_FPM_TOKENIZER_REVISION": manifest["checkpoint_revision"],
        "DYN_FPM_DATASET_ROLE": "calibration",
    }
    if backend == "vllm":
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(runtime), env.get("PYTHONPATH")]))
    command = native_command(backend, checkpoint, manifest["checkpoint_revision"], tp_size, phase, output, corpus)
    (output / "command.json").write_text(json.dumps(command, indent=2))
    with (Path(perf_filename).parent / ".glm53flash-native-gpus.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify_sources(backend, pins, output)
        _execute(command, output, env, backend)
        if backend == "sglang":
            (output / "requests.json").write_bytes((output / "sglang-requests.json").read_bytes())
        verify_target_completeness(output, tp_size)
        from collector.glm53flash_validation import freeze_evidence

        evidence_sha256 = freeze_evidence(output, tp_size, manifest, aggregation_policy=WHOLE_FORWARD_RANK)
        rows = aggregate_rank_records(
            [output / f"rank-{rank}.jsonl" for rank in range(tp_size)],
            tp_size,
            manifest,
            evidence_sha256=evidence_sha256,
            aggregation_policy=WHOLE_FORWARD_RANK,
        )
        for row in rows:
            log_perf(
                [{key: value for key, value in row.items() if key != "kernel_source"}],
                backend,
                backend_version,
                "cuda",
                "glm53flash_module",
                row["kernel_source"],
                perf_filename,
            )
    (output / "COMPLETE").write_text("native measured campaign complete; accuracy acceptance not evaluated\n")
