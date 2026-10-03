# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify single-node server logs against same-job benchmark measurements."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

from e2e_accuracy_source.inferencex_recipe import (
    INFERENCEX_REPOSITORY_URL,
    InferenceXRecipeError,
    _normalize_yaml_server_args,
)
from e2e_accuracy_source.runtime_recipe import _logged_sglang_args, _logged_vllm_args, _verified_job_log
from e2e_accuracy_source.shell_recipe import command_args


class _ServerLog:
    """Adapt one verified server log to the shared typed runtime readers."""

    def __init__(self, text: str):
        self.members = {"server_aggregated_w0.out": None}
        self.text = text

    def read(self, name: str, *, errors: str = "strict") -> str:
        if name not in self.members:
            raise InferenceXRecipeError("unknown single-node server log member")
        return self.text


def _member(root: Path, record: dict, limit: int) -> bytes:
    path = root / record["path"]
    if path.resolve().parent != root.resolve():
        raise InferenceXRecipeError("single-node runtime archive path escapes cache")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record["sha256"]:
        raise InferenceXRecipeError("single-node runtime archive SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if item.filename == record["member"]]
        if len(members) != 1 or members[0].file_size > limit:
            raise InferenceXRecipeError("single-node runtime member is unavailable or too large")
        return archive.read(members[0])


def _benchmark_command(job: str, benchmark: dict, member: str) -> dict:
    commands = []
    for line in job.splitlines():
        if " + python3 " not in line or "benchmark_serving.py " not in line:
            continue
        command = line.split(" + python3 ", 1)[1]
        parsed = command_args(command, {})
        expected = {
            "model": benchmark["model"],
            "random_input_len": benchmark["isl"],
            "random_output_len": benchmark["osl"],
            "max_concurrency": benchmark["conc"],
            "dataset_name": "random",
            "result_filename": Path(member).name.removeprefix("agg_"),
        }
        if all(parsed.get(key) == value for key, value in expected.items()):
            commands.append(parsed)
    if not commands or any(value != commands[0] for value in commands[1:]):
        raise InferenceXRecipeError("single-node executed benchmark command is missing or ambiguous")
    parsed = commands[0]
    if type(parsed.get("num_prompts")) is not int or parsed["num_prompts"] < 1:
        raise InferenceXRecipeError("single-node benchmark request count is unresolved")
    if type(parsed.get("random_range_ratio")) not in (int, float):
        raise InferenceXRecipeError("single-node benchmark distribution is unresolved")
    if parsed["num_prompts"] % benchmark["conc"]:
        raise InferenceXRecipeError("single-node request count is not an integer concurrency multiple")
    return {
        "type": "benchmark_serving.py",
        "num_requests": parsed["num_prompts"],
        "num_prompts_mult": parsed["num_prompts"] // benchmark["conc"],
        "random_range_ratio": parsed["random_range_ratio"],
        "request_rate": parsed.get("request_rate"),
        "ignore_eos": parsed.get("ignore_eos", False),
        "use_chat_template": parsed.get("use_chat_template", False),
        "source": "same-job executed benchmark command",
    }


def runtime_deployment(parsed: dict) -> tuple:
    """Construct a deployment from measured logs when historical source expired."""
    args = _normalize_yaml_server_args(parsed["server_args"])
    backend = parsed["backend"]
    if backend == "sglang":
        args["effective_chunked_prefill_size"] = args["chunked_prefill_size"]
        args["enable_chunked_prefill"] = args["chunked_prefill_size"] > 0
        args["enable_prefix_caching"] = not args["disable_radix_cache"]
        args["cuda_graph_enabled"] = not args["disable_cuda_graph"]
    proof = parsed["evidence"]
    evidence = {
        "repository": INFERENCEX_REPOSITORY_URL,
        "git_sha": proof["source_git_sha"],
        "path": proof["server"]["member"],
        "content_sha256": proof["server_member_sha256"],
        "adapter": "single_node_runtime_artifact_v1",
        "server_args": args,
        "server_args_by_role": {},
        "runtime_image": parsed["image"],
        "runtime_defaults": {"aggregated": {"values": args}},
        "single_node_runtime": proof,
        "framework_identity_evidence": {
            "source": "matched server log",
            "value": parsed["runtime_framework_version"],
            "note": "Build environment variables are not installed package identity.",
        },
    }
    return (
        parsed["model"],
        {"aggregated": args},
        parsed["runtime_framework_version"],
        backend,
        parsed["benchmark"],
        evidence,
    )


def inspect_cached_single_node_runtime(row, cache_dir: Path) -> dict:
    """Return only observed controls, after source/job/metric identity checks.

    The cache index is an evidence manifest, not a list of trusted knob values.
    Every lookup rechecks its referenced bytes and derives values from logs.
    """
    result = {"parsed": None, "issues": []}
    if row.disagg:
        return result
    root = Path(cache_dir) / "runtime-evidence"
    index = root / "agg_index.json"
    if not index.exists():
        # Older evidence packages keep aggregate artifacts in a separate cache.
        root = Path(cache_dir) / "single-node-runtime"
        index = root / "index.json"
    if not index.exists():
        return result
    entries = json.loads(index.read_text()).get("artifacts", [])
    entries = [entry for entry in entries if str(entry.get("github_run_id")) == str(row.github_run_id)]
    matches = []
    for entry in entries:
        try:
            benchmark_bytes = _member(root, entry["benchmark"], 4 * 1024 * 1024)
            benchmark = json.loads(benchmark_bytes)
            expected = {
                "isl": row.isl,
                "osl": row.osl,
                "conc": row.conc,
                "framework": row.framework,
                "precision": row.precision,
                "disagg": False,
                "image": row.image,
                "tp": row.prefill_tp,
                "ep": row.prefill_ep,
            }
            hardware = {"b200-dgxc": "b200", "b200-dsv4": "b200"}.get(benchmark.get("hw"), benchmark.get("hw"))
            if hardware != row.hardware:
                continue
            if any(benchmark.get(key) != value for key, value in expected.items()):
                continue
            if str(benchmark.get("dp_attention")).lower() != str(row.prefill_dp_attention).lower():
                continue
            metrics = ("mean_ttft", "mean_tpot", "median_ttft", "median_tpot")
            if not all(
                type(benchmark.get(key)) in (int, float)
                and type(row.metrics.get(key)) in (int, float)
                and math.isclose(benchmark[key], row.metrics[key], rel_tol=1e-9, abs_tol=1e-9)
                for key in metrics
            ):
                continue
            proof = entry["run_attempt_evidence"]
            log_path = root / entry["logs_path"]
            if log_path.resolve().parent != root.resolve():
                raise InferenceXRecipeError("single-node job log path escapes cache")
            # Both uploads must be in exactly the same checked job log.
            job = _verified_job_log(
                log_path, proof, str(row.github_run_id), entry["run_attempt"], entry["server"]["id"]
            )
            _verified_job_log(log_path, proof, str(row.github_run_id), entry["run_attempt"], entry["benchmark"]["id"])
            source_sha = entry["source_git_sha"]
            if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not re.search(
                r"(?:ref: |Checking out the ref).*" + source_sha, job
            ):
                raise InferenceXRecipeError("single-node runtime workflow source revision is not proven")
            server_bytes = _member(root, entry["server"], 64 * 1024 * 1024)
            server_text = server_bytes.decode("utf-8", errors="replace")
            archive = _ServerLog(server_text)
            if row.framework == "sglang":
                args, _ = _logged_sglang_args(archive, "aggregated", 1)
                args["runtime_effective_scheduler"] = True
            elif row.framework == "vllm":
                observed, _ = _logged_vllm_args(archive, "aggregated", 1)
                declarations = []
                for line in server_text.splitlines():
                    if "non-default args: " not in line:
                        continue
                    value = ast.literal_eval(line.split("non-default args: ", 1)[1])
                    if not isinstance(value, dict):
                        raise InferenceXRecipeError("vLLM declared arguments are not a dictionary")
                    declarations.append(value)
                if not declarations or any(value != declarations[0] for value in declarations[1:]):
                    raise InferenceXRecipeError("vLLM declared arguments are missing or inconsistent")
                args = declarations[0] | observed
                dtypes = set(re.findall(r"with config: .*?\bdtype=torch\.(bfloat16|float16|float32)(?=,)", server_text))
                if len(dtypes) == 1:
                    args["dtype"] = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}[dtypes.pop()]
            else:
                continue
            if args.get("served_model_name") != benchmark.get("model"):
                raise InferenceXRecipeError("single-node runtime model differs from benchmark")
            model = benchmark["model"]
            model_evidence = {"served_model": model}
            if model.startswith("/"):
                canonical = set(re.findall(r"Z +MODEL: ([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\r?$", job, re.MULTILINE))
                exported = set(re.findall(r"Z \+ export MODEL=(/[^\s]+)\r?$", job, re.MULTILINE))
                if len(canonical) != 1 or exported != {model}:
                    raise InferenceXRecipeError("single-node local model path lacks exact job alias evidence")
                model = canonical.pop()
                model_evidence["canonical_model"] = model
                model_evidence["source"] = "same-job MODEL environment and executed local-path export"
            workload = _benchmark_command(job, benchmark, entry["benchmark"]["member"])
            matches.append(
                {
                    "model": model,
                    "backend": row.framework,
                    "image": row.image,
                    "benchmark": workload,
                    "server_args": args,
                    "runtime_framework_version": args.get("runtime_framework_version"),
                    "evidence": {
                        "kind": "runtime_observed",
                        "github_run_id": str(row.github_run_id),
                        "recorded_run_attempt": row.run_attempt,
                        "run_attempt": entry["run_attempt"],
                        "recorded_source_git_sha": row.head_sha,
                        "source_git_sha": source_sha,
                        "server": entry["server"],
                        "benchmark": entry["benchmark"],
                        "server_member_sha256": hashlib.sha256(server_bytes).hexdigest(),
                        "benchmark_member_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
                        "logs_path": entry["logs_path"],
                        "run_attempt_evidence": proof,
                        "matched_metrics": list(metrics),
                        "model_identity": model_evidence,
                    },
                }
            )
        except (
            KeyError,
            ValueError,
            TypeError,
            SyntaxError,
            OSError,
            zipfile.BadZipFile,
            InferenceXRecipeError,
        ) as error:
            result["issues"].append(str(error))
    if len(matches) == 1:
        result["parsed"] = matches[0]
    elif matches:
        result["issues"].append("single-node runtime evidence is ambiguous")
    return result
