# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match saved InferenceX server artifacts to a measured disaggregated point.

The reader never extracts files, executes logs, or exports environment dumps.
Artifact download and exact run-attempt attribution remain the caller's job.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import re
import tarfile
import zipfile
from pathlib import Path

from e2e_accuracy_source.recipes.inferencex_recipe import (
    InferenceXRecipeError,
    _load_yaml_mapping,
    _normalize_yaml_server_args,
)
from e2e_accuracy_source.recipes.shell_recipe import command_args
from e2e_accuracy_source.schema import SiliconRow


class RuntimeArtifact:
    def __init__(self, path: Path):
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive:
            matches = [item for item in archive.infolist() if item.filename == "multinode_server_logs.tar.gz"]
            if len(matches) != 1 or matches[0].file_size > 512 * 1024 * 1024:
                raise InferenceXRecipeError("runtime artifact lacks one bounded server log archive")
            # Owned by this reader and closed by read_runtime_recipe's finally.
            self.tar = tarfile.open(fileobj=io.BytesIO(archive.read(matches[0])), mode="r:gz")  # noqa: SIM115
        self.members = {}
        for member in self.tar.getmembers():
            name = member.name.removeprefix("./")
            if member.isfile():
                if name in self.members:
                    raise InferenceXRecipeError("runtime artifact has duplicate member names")
                self.members[name] = member
        self.used = {}

    def read(self, name: str, *, errors: str = "strict") -> str:
        return self._read_bytes(name, 64 * 1024 * 1024).decode("utf-8", errors=errors)

    def read_result(self, name: str) -> dict:
        # Per-token latency arrays can exceed the normal config/log size bound.
        if not re.fullmatch(r"sa-bench_isl_\d+_osl_\d+/results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+\.json", name):
            raise InferenceXRecipeError("large runtime result read requires an exact benchmark result path")
        return json.loads(self._read_bytes(name, 384 * 1024 * 1024))

    def _read_bytes(self, name: str, max_bytes: int) -> bytes:
        member = self.members.get(name)
        if member is None or member.size > max_bytes:
            raise InferenceXRecipeError(f"runtime artifact member unavailable or too large: {name}")
        content = self.tar.extractfile(member).read()
        self.used[name] = hashlib.sha256(content).hexdigest()
        return content


def _logged_worker_defaults(archive, role, workers):
    """Read two typed fields from complete printed runtime configurations."""
    collected, sources = [], []
    for worker in range(workers):
        names = [name for name in archive.members if re.fullmatch(rf"[^/]+_{role}_w{worker}\.out", name)]
        if not names:
            return {}, []
        if len(names) != 1:
            raise InferenceXRecipeError("runtime worker log is ambiguous")
        values = set()
        for line in archive.read(names[0], errors="replace").splitlines():
            # Do not evaluate Python representations or match arbitrary log text.
            if not line.startswith("model=") or " kv_cache_config=KvCacheConfig(" not in line:
                continue
            block = re.search(r"\btokens_per_block=(\d+)\b", line)
            chunk = re.search(r"\benable_chunked_prefill=(True|False)\b", line)
            if block and chunk:
                values.add((int(block[1]), chunk[1] == "True"))
        if not values:
            return {}, []
        if len(values) != 1:
            raise InferenceXRecipeError("runtime worker configurations disagree on defaults")
        collected.append(values.pop())
        sources.append(names[0])
    if len(set(collected)) != 1 or collected[0][0] < 1:
        raise InferenceXRecipeError("runtime workers disagree on block size/chunked prefill")
    return dict(zip(("block_size", "enable_chunked_prefill"), collected[0], strict=True)), sources


_SGLANG_RUNTIME_FIELDS = {
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
    "enable_dp_attention",
    "page_size",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "max_running_requests",
    "max_total_tokens",
    "context_length",
    "kv_cache_dtype",
    "dtype",
    "quantization",
    "mem_fraction_static",
    "disable_radix_cache",
    "disable_cuda_graph",
    "attention_backend",
    "moe_runner_backend",
    "served_model_name",
    "disaggregation_mode",
    "mamba_ssm_dtype",
    "swa_full_tokens_ratio",
    "disable_overlap_schedule",
    "enable_mixed_chunk",
    "schedule_policy",
    "enable_dynamic_chunking",
    "model_impl",
    "speculative_algorithm",
    "flashinfer_mxfp4_moe_precision",
}


def _logged_sglang_args(archive, role, workers):
    """Read allowlisted scalar fields without executing logged Python objects."""
    configurations, sources = [], []
    for worker in range(workers):
        names = [name for name in archive.members if re.fullmatch(rf"[^/]+_{role}_w{worker}\.out", name)]
        if len(names) != 1:
            raise InferenceXRecipeError("SGLang worker log is unavailable or ambiguous")
        seen, summaries, layouts = [], [], set()
        for line in archive.read(names[0], errors="replace").splitlines():
            layout = re.search(r"Auto-detected DSV4 routed-expert layout: is_fp4_experts=(True|False)\b", line)
            if layout:
                layouts.add(layout[1] == "True")
            summary = re.fullmatch(
                r"\[[^\]\n]+\] max_total_num_tokens=\d+, chunked_prefill_size=(\d+), "
                r"max_prefill_tokens=(\d+), max_running_requests=(\d+), context_len=(\d+), "
                r"available_gpu_mem=[\d.]+ GB",
                line,
            )
            if summary:
                summaries.append(tuple(int(value) for value in summary.groups()))
            if "server_args=ServerArgs(" not in line:
                continue
            try:
                expression = ast.parse(line.split("server_args=", 1)[1], mode="eval").body
            except SyntaxError as error:
                raise InferenceXRecipeError("SGLang runtime configuration is not parseable") from error
            if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
                raise InferenceXRecipeError("SGLang runtime configuration has unsupported structure")
            if expression.func.id != "ServerArgs" or expression.args:
                raise InferenceXRecipeError("SGLang runtime configuration has unsupported constructor")
            values = {}
            for keyword in expression.keywords:
                if keyword.arg not in _SGLANG_RUNTIME_FIELDS:
                    continue
                if keyword.arg in values or not isinstance(keyword.value, ast.Constant):
                    raise InferenceXRecipeError("SGLang runtime control is duplicated or nonliteral")
                value = ast.literal_eval(keyword.value)
                if value is not None and type(value) not in (int, float, str, bool):
                    raise InferenceXRecipeError("SGLang runtime control has unsupported type")
                values[keyword.arg] = value
            required = {"tp_size", "pp_size", "ep_size", "dp_size", "enable_dp_attention", "served_model_name"}
            if not required <= values.keys():
                raise InferenceXRecipeError("SGLang runtime topology/model is incomplete")
            seen.append(values)
        if not seen or any(value != seen[0] for value in seen[1:]):
            raise InferenceXRecipeError("SGLang worker configurations are missing or inconsistent")
        if not summaries or any(value != summaries[0] for value in summaries[1:]):
            raise InferenceXRecipeError("SGLang effective scheduler summary is missing or inconsistent")
        for key, value in zip(
            ("chunked_prefill_size", "max_prefill_tokens", "max_running_requests", "context_length"),
            summaries[0],
            strict=True,
        ):
            seen[0][key] = value
        if len(layouts) > 1:
            raise InferenceXRecipeError("SGLang runtime expert layouts disagree")
        if layouts:
            seen[0]["runtime_is_fp4_experts"] = layouts.pop()
        configurations.append(seen[0])
        sources.append(names[0])
    if any(value != configurations[0] for value in configurations[1:]):
        raise InferenceXRecipeError("SGLang workers disagree on effective configuration")
    return configurations[0], sources


def _logged_vllm_args(archive, role, workers):
    """Read the primitive prefix of V1 engine configuration log records."""
    configurations, sources = [], []
    types = {
        "tensor_parallel_size": int,
        "pipeline_parallel_size": int,
        "data_parallel_size": int,
        "decode_context_parallel_size": int,
        "max_seq_len": int,
        "enable_prefix_caching": bool,
        "enable_chunked_prefill": bool,
        "kv_cache_dtype": str,
        "served_model_name": str,
    }
    for worker in range(workers):
        names = [name for name in archive.members if re.fullmatch(rf"[^/]+_{role}_w{worker}\.out", name)]
        if len(names) != 1:
            raise InferenceXRecipeError("vLLM worker log is unavailable or ambiguous")
        seen = []
        for line in archive.read(names[0], errors="replace").splitlines():
            if "Initializing a V1 LLM engine (v" not in line or " with config: " not in line:
                continue
            if line.count("Initializing a V1 LLM engine (v") != 1:
                continue  # Interleaved rank output is not a complete configuration record.
            # Later compilation configs contain nested reprs. Never interpret them.
            prefix = line.split(" with config: ", 1)[1].split(", compilation_config=", 1)[0]
            version = re.search(r"Initializing a V1 LLM engine \(v([^()]+)\)", line)
            if version is None:
                raise InferenceXRecipeError("vLLM runtime version is unavailable")
            values = {"runtime_framework_version": version[1]}
            if any(len(re.findall(rf"(?:^|, ){key}=([^,\n]+)(?=, |$)", prefix)) != 1 for key in types):
                continue
            for key, kind in types.items():
                matches = re.findall(rf"(?:^|, ){key}=([^,\n]+)(?=, |$)", prefix)
                if len(matches) != 1:
                    raise InferenceXRecipeError("vLLM runtime control is unavailable or ambiguous: " + key)
                raw = matches[0]
                if kind is int and re.fullmatch(r"[1-9]\d*", raw):
                    values[key] = int(raw)
                elif kind is bool and raw in ("True", "False"):
                    values[key] = raw == "True"
                elif kind is str and re.fullmatch(r"[a-zA-Z0-9_./-]+", raw):
                    values[key] = raw
                else:
                    raise InferenceXRecipeError("vLLM runtime control has unsupported value: " + key)
            seen.append(values)
        if not seen or any(value != seen[0] for value in seen[1:]):
            raise InferenceXRecipeError("vLLM worker configurations are missing or inconsistent")
        configurations.append(seen[0])
        sources.append(names[0])
    if any(value != configurations[0] for value in configurations[1:]):
        raise InferenceXRecipeError("vLLM workers disagree on effective configuration")
    return configurations[0], sources


def _equal(value, expected, what):
    if value != expected:
        raise InferenceXRecipeError(f"runtime artifact {what} differs from measurement")


def _verified_job_log(path, proof, github_run_id, run_attempt, artifact_id):
    """Bind an artifact upload to checked bytes from one exact attempt job."""
    endpoint = f"repos/SemiAnalysisAI/InferenceX/actions/runs/{github_run_id}/attempts/{run_attempt}/logs"
    if not isinstance(proof, dict) or proof.get("logs_endpoint") != endpoint or path is None:
        raise InferenceXRecipeError("runtime attempt correction requires exact job log evidence")
    data = Path(path).read_bytes()
    if len(data) > 512 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != proof.get("logs_archive_sha256"):
        raise InferenceXRecipeError("runtime attempt log archive SHA256 mismatch")
    member = proof.get("job_log_member")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        matches = [item for item in archive.infolist() if item.filename == member]
        if len(matches) != 1 or matches[0].file_size > 64 * 1024 * 1024:
            raise InferenceXRecipeError("runtime attempt job log is unavailable or ambiguous")
        data = archive.read(matches[0])
    if hashlib.sha256(data).hexdigest() != proof.get("job_log_sha256"):
        raise InferenceXRecipeError("runtime attempt job log SHA256 mismatch")
    log = data.decode("utf-8")
    if not re.search(rf"successfully finalized\. Artifact ID {artifact_id}(?:\r?$)", log, re.MULTILINE):
        raise InferenceXRecipeError("runtime job log does not prove this artifact upload")
    return log


def _metric_matches(result: dict, row: SiliconRow) -> bool:
    for metric in ("mean_ttft", "mean_tpot", "median_ttft", "median_tpot"):
        recorded, measured = result.get(metric + "_ms"), row.metrics.get(metric)
        if not isinstance(recorded, (int, float)) or not isinstance(measured, (int, float)):
            return False
        if not math.isclose(recorded, measured * 1000, rel_tol=1e-9, abs_tol=1e-6):
            return False
    return True


def read_runtime_recipe(
    row: SiliconRow,
    artifact_path: Path,
    *,
    github_run_id: str,
    run_attempt: int,
    artifact_id: int,
    source_git_sha: str | None = None,
    source_model_path: str | None = None,
    source_model_evidence: dict | None = None,
    run_log_path: Path | None = None,
    run_attempt_evidence: dict | None = None,
    recorded_run_attempt: int | None = None,
    workload_only: bool = False,
) -> tuple:
    """Return the common parsed recipe tuple only after exact point validation.

    ``source_git_sha`` may be provided only from the actual checkout log; the
    workflow head remains separate and is never silently promoted to checkout.
    """
    _equal(str(github_run_id), str(row.github_run_id), "GitHub run")
    job_log = None
    if run_attempt_evidence is not None:
        job_log = _verified_job_log(run_log_path, run_attempt_evidence, github_run_id, run_attempt, artifact_id)
    if run_attempt != row.run_attempt and (job_log is None or recorded_run_attempt != row.run_attempt):
        raise InferenceXRecipeError("runtime artifact run attempt differs from measurement")
    if not row.disagg or row.framework not in ("dynamo-trt", "dynamo-sglang", "dynamo-vllm"):
        raise InferenceXRecipeError("runtime artifact adapter requires a supported disaggregated backend")
    if source_git_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", source_git_sha):
        raise InferenceXRecipeError("runtime checkout requires an immutable SHA")
    if (
        job_log is not None
        and source_git_sha is not None
        and not re.search(rf"git log -1 --format=%H\r?\n[^\n]*Z {source_git_sha}\r?$", job_log, re.MULTILINE)
    ):
        raise InferenceXRecipeError("runtime job log does not prove checkout SHA")
    archive = RuntimeArtifact(Path(artifact_path))
    try:
        parsed = _read_runtime_recipe(
            row,
            archive,
            github_run_id,
            run_attempt,
            artifact_id,
            source_git_sha,
            source_model_path,
            source_model_evidence,
            job_log,
            run_attempt_evidence,
            workload_only,
        )
        if run_attempt != row.run_attempt:
            parsed[-1]["artifact"]["recorded_run_attempt"] = row.run_attempt
            parsed[-1]["artifact"]["run_attempt_correction"] = run_attempt_evidence
        return parsed
    finally:
        archive.tar.close()


def _read_runtime_recipe(
    row,
    archive,
    github_run_id,
    run_attempt,
    artifact_id,
    source_git_sha,
    source_model_path,
    source_model_evidence,
    job_log,
    run_attempt_evidence,
    workload_only,
):
    recipe = _load_yaml_mapping(archive.read("config.yaml"), "runtime merged recipe")
    backend = {"dynamo-trt": "trtllm", "dynamo-sglang": "sglang", "dynamo-vllm": "vllm"}[row.framework]
    _equal(recipe.get("backend", {}).get("type"), backend, "backend")
    _equal(recipe.get("model", {}).get("precision"), row.precision, "precision")
    benchmark = recipe.get("benchmark", {})
    _equal(benchmark.get("type"), "sa-bench", "benchmark type")
    for name in ("isl", "osl"):
        _equal(benchmark.get(name), getattr(row, name), name)
    concurrencies = benchmark.get("concurrencies", [])
    if isinstance(concurrencies, str) and re.fullmatch(r"[1-9]\d*(?:x[1-9]\d*)*", concurrencies):
        concurrencies = [int(value) for value in concurrencies.split("x")]
    if not isinstance(concurrencies, list) or row.conc not in concurrencies:
        raise InferenceXRecipeError("runtime concurrency is absent from merged recipe")
    resources = recipe.get("resources", {})
    _equal(resources.get("gpu_type"), row.hardware, "GPU type")
    model = recipe.get("identity", {}).get("model", {}).get("repo")
    if model is None and source_model_path is not None:
        proof = source_model_evidence or {}
        if proof.get("source") == "github_job_log":
            line = proof.get("line")
            if (
                job_log is None
                or not isinstance(line, str)
                or line not in job_log.splitlines()
                or not (
                    line.endswith("+ export SRT_SLURM_MODEL_PREFIX=" + source_model_path)
                    or (
                        backend in ("sglang", "vllm")
                        and re.fullmatch(r"\S+Z\s+MODEL: " + re.escape(source_model_path), line)
                    )
                )
                or proof.get("path") != run_attempt_evidence["job_log_member"]
                or proof.get("content_sha256") != run_attempt_evidence["job_log_sha256"]
            ):
                raise InferenceXRecipeError("runtime source model lacks verified job export")
        else:
            raise InferenceXRecipeError("runtime source model requires verified job log evidence")
        model = source_model_path
    elif source_model_path is not None:
        _equal(model, source_model_path, "source model identity")
    if not isinstance(model, str) or "/" not in model or model.startswith("/"):
        raise InferenceXRecipeError("runtime model identity is unavailable")
    pattern = re.compile(
        rf"sa-bench_isl_{row.isl}_osl_{row.osl}/results_concurrency_{row.conc}_gpus_"
        rf"{row.num_prefill_gpu + row.num_decode_gpu}_ctx_{row.num_prefill_gpu}_gen_{row.num_decode_gpu}\.json"
    )
    matches = []
    for name in archive.members:
        if pattern.fullmatch(name):
            result = archive.read_result(name)
            if _metric_matches(result, row):
                matches.append((name, result))
    if len(matches) != 1:
        raise InferenceXRecipeError("runtime result does not uniquely match measured TTFT and TPOT")
    result_path, result = matches[0]
    _equal(result.get("max_concurrency"), row.conc, "result concurrency")
    if result.get("model_id") not in (model, model.rsplit("/", 1)[-1]):
        raise InferenceXRecipeError("runtime result model differs from measurement")
    commands = []
    for line in archive.read("benchmark.out").splitlines():
        if line.startswith("+ ") and "benchmark_serving.py " in line:
            args = command_args(line.split("benchmark_serving.py ", 1)[1], {})
            if args.get("result_filename") == result_path.rsplit("/", 1)[-1]:
                commands.append(args)
    if len(commands) != 1:
        raise InferenceXRecipeError("runtime benchmark command is missing or ambiguous")
    command = commands[0]
    for knob, expected in (
        ("random_input_len", row.isl),
        ("random_output_len", row.osl),
        ("max_concurrency", row.conc),
        ("num_prompts", result.get("num_prompts")),
        ("model", result.get("model_id")),
        ("dataset_name", "random"),
    ):
        _equal(command.get(knob), expected, "benchmark " + knob)
    _equal(str(command.get("request_rate")), str(result.get("request_rate")), "request rate")
    if benchmark.get("req_rate") is not None:
        _equal(str(command.get("request_rate")), str(benchmark["req_rate"]), "configured request rate")
    ratio = command.get("random_range_ratio")
    if not isinstance(ratio, (float, int)) or not 0 < ratio <= 1:
        raise InferenceXRecipeError("runtime benchmark length ratio is unresolved")
    count = command["num_prompts"]
    if not isinstance(count, int) or count < 1 or count % row.conc:
        raise InferenceXRecipeError("runtime request count is not a positive concurrency multiple")
    if "num_prompts_mult" in benchmark:
        _equal(benchmark["num_prompts_mult"], count // row.conc, "request multiplier")
    elif backend == "trtllm":
        raise InferenceXRecipeError("runtime request multiplier is unavailable")
    if "completed" in result:
        _equal(result["completed"], count, "completed requests")
    elif job_log is not None or backend != "trtllm" or workload_only:
        raise InferenceXRecipeError("runtime attempt recovery requires successful request count")
    roles, versions, fingerprints, logged_defaults = {}, {}, {}, {}
    workload = {
        "type": "sa-bench",
        "random_range_ratio": ratio,
        "num_prompts_mult": count // row.conc,
        "num_requests": count,
        "request_rate": command.get("request_rate"),
        "use_chat_template": command.get("use_chat_template"),
        "ignore_eos": command.get("ignore_eos"),
        "source": "runtime artifact benchmark.out",
    }
    evidence = {
        "adapter": "runtime_server_artifact_v1",
        "repository": "https://github.com/SemiAnalysisAI/InferenceX",
        "git_sha": source_git_sha,
        "workflow_head_sha": row.head_sha,
        "path": "config.yaml",
        "content_sha256": archive.used["config.yaml"],
        "runtime_image": recipe.get("identity", {}).get("container", {}).get("image"),
        "dynamo_installation": dict(recipe.get("dynamo") or {}),
        "server_args": {},
        "server_args_by_role": roles,
        "runtime_defaults": logged_defaults,
        "artifact": {
            "id": artifact_id,
            "sha256": archive.sha256,
            "github_run_id": str(github_run_id),
            "run_attempt": run_attempt,
            "members": archive.used,
        },
        "runtime_fingerprints": fingerprints,
        "unsupported_controls": {
            "workload": {
                "use_chat_template": {
                    "value": command["use_chat_template"],
                    "status": "not_represented",
                    "source": "benchmark.out saved-result command",
                    "reason": "Token-length replay does not reconstruct chat-template tokenization.",
                }
            }
            if command.get("use_chat_template")
            else {}
        },
        "measurement_match": {
            "result_path": result_path,
            "metrics": ["mean_ttft", "mean_tpot", "median_ttft", "median_tpot"],
            "source_units": "milliseconds",
            "db_units": "seconds",
        },
    }
    for key in ("custom_tokenizer",):
        if command.get(key):
            workload[key] = command[key]
            evidence["unsupported_controls"]["workload"][key] = {
                "value": command[key],
                "status": "not_represented",
                "source": "benchmark.out saved-result command",
                "reason": "Token-length replay does not execute benchmark tokenizer plugins.",
            }
    if source_model_path is not None and source_model_evidence is not None:
        evidence["source_model_identity"] = {"model_path": source_model_path, **source_model_evidence}
    if workload_only:
        evidence["adapter"] = "runtime_workload_artifact_v1"
        return model, roles, versions, backend, workload, evidence
    for role in ("prefill", "decode"):
        workers = getattr(row, role + "_num_workers")
        _equal(resources.get(role + "_workers"), workers, role + " workers")
        if not isinstance(workers, int) or workers < 1:
            raise InferenceXRecipeError("runtime worker count must be positive")
        gpus_per_worker = resources.get("gpus_per_" + role)
        if backend in ("sglang", "vllm"):
            effective = recipe.get("backend", {}).get(backend + "_config", {}).get(role)
            if not isinstance(effective, dict):
                raise InferenceXRecipeError(backend + " runtime role recipe is unavailable")
            reader = _logged_sglang_args if backend == "sglang" else _logged_vllm_args
            runtime_values, log_members = reader(archive, role, workers)
            # The API name can omit the organization; checkpoint identity still
            # comes from the independently verified job/config evidence above.
            _equal(runtime_values["served_model_name"], result["model_id"], role + " served model")
            if backend == "sglang":
                _equal(runtime_values.get("disaggregation_mode"), role, role + " disaggregation mode")
            effective = {**_normalize_yaml_server_args(effective), **runtime_values}
            if backend == "sglang":
                effective["effective_chunked_prefill_size"] = runtime_values.get("chunked_prefill_size")
            effective["recipe_environment"] = dict(recipe["backend"].get(role + "_environment", {}))
            if backend == "sglang":
                runtime_gpus = runtime_values["tp_size"] * runtime_values["pp_size"]
            else:
                width = runtime_values["tensor_parallel_size"] * runtime_values["data_parallel_size"]
                runtime_gpus = width * runtime_values["pipeline_parallel_size"]
                # InferenceX has reported either attention TP or TP*DP in this
                # column. Actual geometry is established by engine fields and
                # independently checked against measured total GPU counts.
                reported_tp = getattr(row, role + "_tp")
                if reported_tp not in (runtime_values["tensor_parallel_size"], width):
                    raise InferenceXRecipeError("vLLM reported TP differs from both runtime topology definitions")
                effective["inferencex_reported_tp"] = reported_tp
                ep = effective.get("enable_expert_parallel")
                if type(ep) is not bool:
                    raise InferenceXRecipeError("vLLM runtime recipe does not establish expert parallelism")
                _equal(width if ep else 1, getattr(row, role + "_ep"), role + " MoE EP")
                _equal(
                    runtime_values["data_parallel_size"] > 1,
                    getattr(row, role + "_dp_attention"),
                    role + " attention DP",
                )
            if gpus_per_worker is not None:
                _equal(gpus_per_worker, runtime_gpus, role + " runtime GPUs")
            gpus_per_worker = runtime_gpus
            logged_defaults[role] = {
                "values": runtime_values,
                "log_members": log_members,
                "workers_checked": workers,
            }
        else:
            effective = _load_yaml_mapping(archive.read(f"trtllm_config_{role}.yaml"), "runtime role configuration")
        if not isinstance(gpus_per_worker, int) or gpus_per_worker < 1:
            raise InferenceXRecipeError("runtime GPUs per worker are unresolved")
        _equal(gpus_per_worker * workers, getattr(row, "num_" + role + "_gpu"), role + " GPUs")
        # Prefer the actual role configuration written for the launched worker.
        args = _normalize_yaml_server_args(effective)
        for key, expected in (
            ("tensor_parallel_size", getattr(row, role + "_tp")),
            ("moe_expert_parallel_size", getattr(row, role + "_ep")),
            (
                "enable_dp_attention" if backend == "sglang" else "enable_attention_dp",
                getattr(row, role + "_dp_attention"),
            ),
        ):
            if backend != "vllm":
                _equal(args.get(key), expected, role + " " + key)
        pp = row.metrics.get(role + "_pp")
        if pp is not None:
            _equal(args.get("pipeline_parallel_size"), pp, role + " PP")
        identities = []
        for worker in range(workers):
            name = f"fingerprint_{role}_w{worker}.json"
            fingerprint = json.loads(archive.read(name))
            version = fingerprint.get("frameworks", {}).get("tensorrt_llm" if backend == "trtllm" else backend)
            dynamo_version = fingerprint.get("frameworks", {}).get("dynamo")
            revision = (fingerprint.get("model") or {}).get("hf_revision")
            if (
                not isinstance(version, str)
                or not isinstance(revision, str)
                or not re.fullmatch(r"[0-9a-f]{40}", revision)
            ):
                raise InferenceXRecipeError("runtime framework/checkpoint fingerprint is unresolved")
            framework_sha = fingerprint.get("env", {}).get("TRT_LLM_GIT_COMMIT") if backend == "trtllm" else None
            if framework_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", str(framework_sha)):
                raise InferenceXRecipeError("runtime framework source fingerprint is unresolved")
            if dynamo_version is not None and (not isinstance(dynamo_version, str) or not dynamo_version.strip()):
                raise InferenceXRecipeError("runtime Dynamo fingerprint is unresolved")
            identities.append((version, revision, framework_sha, dynamo_version))
        if len(set(identities)) != 1:
            raise InferenceXRecipeError("runtime workers disagree on framework/checkpoint identity")
        version, revision, framework_sha, dynamo_version = identities[0]
        if backend == "vllm":
            _equal(args["runtime_framework_version"], version, role + " framework version")
        versions[role] = version
        args["revision"] = revision
        roles[role] = args
        fingerprints[role] = {"framework_version": version, "hf_revision": revision, "workers_checked": workers}
        if dynamo_version is not None:
            fingerprints[role]["dynamo_version"] = dynamo_version
        if framework_sha is not None:
            fingerprints[role]["framework_git_sha"] = framework_sha
        defaults, log_members = _logged_worker_defaults(archive, role, workers) if backend == "trtllm" else ({}, [])
        for key, value in defaults.items():
            if key in args:
                _equal(args[key], value, role + " logged " + key)
            args[key] = value
        if defaults:
            logged_defaults[role] = {"values": defaults, "log_members": log_members, "workers_checked": workers}
    return model, roles, versions, backend, workload, evidence


def inspect_cached_runtime_recipe(row: SiliconRow, cache_dir: Path, *, workload_only: bool = False) -> dict:
    """Inspect optional cached artifacts, retaining rejection reasons for audit.

    Content/hash or index integrity errors are fatal. An otherwise valid artifact
    for another point is a normal mismatch and allows historical source fallback.
    """
    if not row.disagg:
        return {"parsed": None, "candidates": []}
    directory = Path(cache_dir) / "runtime-evidence"
    index_path = directory / "disagg_index.json"
    if not index_path.exists():
        # Older evidence packages keep disaggregated artifacts in a separate cache.
        directory = Path(cache_dir) / "runtime-artifacts"
        index_path = directory / "index.json"
    if not index_path.is_file():
        return {"parsed": None, "candidates": []}
    index_bytes = index_path.read_bytes()
    try:
        index = json.loads(index_bytes)
    except ValueError as error:
        raise InferenceXRecipeError("runtime artifact cache index is invalid JSON") from error
    if not isinstance(index, dict) or index.get("schema_version") != "runtime-artifact-cache/1":
        raise InferenceXRecipeError("runtime artifact cache index schema is unsupported")
    entries = index.get("artifacts")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise InferenceXRecipeError("runtime artifact cache index requires artifact entries")
    matches, diagnostics = [], []
    for entry in entries:
        if str(entry.get("github_run_id")) != str(row.github_run_id) or row.run_attempt not in (
            entry.get("run_attempt"),
            entry.get("recorded_run_attempt"),
        ):
            continue
        relative = entry.get("path")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(entry.get("id"), int)
        ):
            raise InferenceXRecipeError("runtime artifact cache entry has invalid path/hash/identity")
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise InferenceXRecipeError("runtime artifact cache member is unavailable or outside its cache")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise InferenceXRecipeError("runtime artifact cache content SHA256 mismatch")
        log_path = None
        if entry.get("run_log_path") is not None:
            relative_log = Path(entry["run_log_path"])
            log_path = directory / relative_log
            if (
                relative_log.is_absolute()
                or ".." in relative_log.parts
                or not log_path.resolve().is_relative_to(directory.resolve())
            ):
                raise InferenceXRecipeError("runtime job log is outside its cache")
            if not log_path.is_file():
                raise InferenceXRecipeError("runtime job log cache member is unavailable")
        try:
            parsed = read_runtime_recipe(
                row,
                path,
                github_run_id=str(entry["github_run_id"]),
                run_attempt=entry["run_attempt"],
                artifact_id=entry["id"],
                source_git_sha=entry.get("source_git_sha"),
                source_model_path=entry.get("source_model_path"),
                source_model_evidence=entry.get("source_model_evidence"),
                run_log_path=log_path,
                run_attempt_evidence=entry.get("run_attempt_evidence") if log_path is not None else None,
                recorded_run_attempt=entry.get("recorded_run_attempt"),
                workload_only=workload_only,
            )
        except (InferenceXRecipeError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as error:
            diagnostics.append({"artifact_id": entry["id"], "status": "unmatched", "error": str(error)})
            continue
        parsed[-1]["artifact"]["cache_index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
        parsed[-1]["artifact"]["cache_relative_path"] = relative
        matches.append(parsed)
        diagnostics.append({"artifact_id": entry["id"], "status": "matched"})
    if len(matches) > 1:
        raise InferenceXRecipeError("multiple runtime artifacts match the same measured point")
    return {"parsed": matches[0] if matches else None, "candidates": diagnostics}


def inspect_cached_runtime_workload(row: SiliconRow, cache_dir: Path) -> dict:
    """Match benchmark evidence independently of unavailable server fingerprints.

    The common parsed tuple contains empty roles/versions; only its model,
    benchmark, and evidence are valid. Callers retain their own server settings
    and must reject conflicting explicit recipe workload values.
    """
    return inspect_cached_runtime_recipe(row, cache_dir, workload_only=True)
