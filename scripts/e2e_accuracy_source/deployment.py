# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-qualified deployment shared by estimates and native replay.

No predictor result is used as source evidence. Unknown source layouts and
required settings fail closed before either predictor is called.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import replace
from functools import cache
from typing import Any

import requests

from e2e_accuracy_source.checkpoint_quantization import resolve_checkpoint_quantization
from e2e_accuracy_source.filter import FRAMEWORK_TO_AIC_BACKEND
from e2e_accuracy_source.framework_defaults import (
    apply_framework_defaults,
    has_reviewed_defaults,
    verified_dynamo_usage_context,
)
from e2e_accuracy_source.inferencex_recipe import (
    INFERENCEX_REPOSITORY_URL,
    InferenceXRecipeError,
    RecipeSource,
    _load_yaml_mapping,
    _normalize_yaml_server_args,
    _select_disaggregated_search_point,
    _single_config_file,
)
from e2e_accuracy_source.legacy_recipe import (
    load_source_config,
    verified_launcher_recipe_copy,
    verified_launcher_workload_identity,
)
from e2e_accuracy_source.mapping import HARDWARE_TO_SYSTEM, MOE_MODELS
from e2e_accuracy_source.model_config_snapshot import normalize_trt_snapshot
from e2e_accuracy_source.runtime_recipe import inspect_cached_runtime_recipe, inspect_cached_runtime_workload
from e2e_accuracy_source.schema import CliEstimateKwargs, SiliconRow
from e2e_accuracy_source.sglang_additional_defaults import apply_additional_sglang_defaults
from e2e_accuracy_source.shell_recipe import read_shell_recipe
from e2e_accuracy_source.single_node_runtime import inspect_cached_single_node_runtime, runtime_deployment
from e2e_accuracy_source.trt_additional_defaults import apply_additional_trt_defaults
from e2e_accuracy_source.workload_defaults import resolve_workload_defaults, unmodeled_workload_controls


def runtime_version(image: str | None, backend: str) -> str | None:
    """Do not mistake a Dynamo container version for a framework version."""
    if not image:
        return None
    release = r"(\d+\.\d+\.\d+(?:rc\d+)?(?:\.post\d+)?)(?=$|[-@+])"
    patterns = {
        "vllm": r"(?:^|/)vllm-openai:v?" + release,
        "sglang": r"(?:^|/)sglang:v?" + release,
        "trtllm": r"(?:^|/)tensorrt-llm(?:/release)?:" + release,
    }
    match = re.search(patterns.get(backend, r"(?!)"), image)
    return match.group(1) if match else None


def recipe_runtime_version(recipe: dict, backend: str, runtime_image: str | None) -> tuple[Any, dict]:
    """Keep exact framework identities; Dynamo tags never imply engine versions."""
    identity = recipe.get("identity", {})
    frameworks = identity.get("frameworks", {})
    keys = ("trtllm", "tensorrt_llm", "tensorrt-llm") if backend == "trtllm" else (backend,)
    for key in keys:
        if frameworks.get(key):
            return frameworks[key], {"source": f"identity.frameworks.{key}", "value": frameworks[key]}
    recorded = runtime_version(runtime_image, backend)
    if recorded:
        return recorded, {"source": "measurement.image", "image": runtime_image}
    for field, image in (
        ("identity.container.image", identity.get("container", {}).get("image")),
        ("model.container", recipe.get("model", {}).get("container")),
    ):
        if isinstance(image, str) and (version := runtime_version(image, backend)):
            return version, {"source": field, "image": image}
    return None, {"source": "unknown"}


@cache
def _framework_source(version: str, path: str) -> str:
    # This installed development identity is pinned in framework_default_sources.json.
    development_refs = {"0.23.1rc1.dev231+g8b00f4123": "8b00f4123776a47a6d8e315242ee5f0dd0b817cf"}
    ref = development_refs.get(version) or (version if re.fullmatch(r"[0-9a-f]{40}", version) else f"v{version}")
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{ref}/{path}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


@cache
def _checkpoint_config(model: str) -> tuple[dict, str, str | None]:
    response = requests.get(f"https://huggingface.co/{model}/resolve/main/config.json", timeout=30)
    response.raise_for_status()
    return response.json(), hashlib.sha256(response.content).hexdigest(), response.headers.get("x-repo-commit")


@cache
def _pinned_checkpoint_config(model: str, revision: str) -> tuple[dict, str, str]:
    response = requests.get(f"https://huggingface.co/{model}/resolve/{revision}/config.json", timeout=30)
    response.raise_for_status()
    return response.json(), hashlib.sha256(response.content).hexdigest(), revision


@cache
def _quantization_companion(model: str, revision: str) -> tuple[dict | None, str | None]:
    response = requests.get(f"https://huggingface.co/{model}/resolve/{revision}/hf_quant_config.json", timeout=30)
    if response.status_code == 404:
        return None, None
    response.raise_for_status()
    return response.json(), hashlib.sha256(response.content).hexdigest()


@cache
def _checkpoint_weight_dtype(model: str, revision: str) -> tuple[str | None, dict]:
    """Read revision-bound weight metadata when config.json omits dtype."""
    url = f"https://huggingface.co/api/models/{model}/revision/{revision}"
    response = requests.get(url, params={"expand[]": ["safetensors", "sha"]}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("sha") != revision:
        raise InferenceXRecipeError("checkpoint weight metadata revision does not match config revision")
    parameters = (data.get("safetensors") or {}).get("parameters", {})
    types = {name for name, count in parameters.items() if isinstance(count, int) and count > 0}
    dtype = None
    # common_broadcastable_dtype over floating compute weights and packed
    # integer quantized weights. Other weight formats need a separate trace.
    if types and types <= {"BF16", "F16", "F32", "U8", "F8_E4M3", "F8_E5M2"}:
        if "F32" in types:
            dtype = "float32"
        elif "BF16" in types and "F16" not in types:
            dtype = "bfloat16"
        elif "F16" in types and "BF16" not in types:
            dtype = "float16"
    return dtype, dict(
        source=url,
        revision=revision,
        sha256=hashlib.sha256(response.content).hexdigest(),
        parameter_counts=parameters,
        weight_dtype=dtype,
    )


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def resolve_quantization(
    args: dict, model: str, *, backend: str | None = None, hardware: str = "", framework_version: str | None = None
) -> dict:
    """Resolve all checkpoint categories and retain exact predictor config bytes."""
    method = args.get("quantization")
    supplied = args.get("quantization_config")
    if supplied is not None and not isinstance(supplied, dict):
        raise InferenceXRecipeError("quantization_config must be a mapping")
    if method == "fbgemm_fp8" and supplied is None:
        return {
            "gemm": "fp8",
            "moe": "fp8",
            "evidence": {"source": "recipe", "method": method, "gemm_profile_is_explicit": True},
        }
    evidence = {"source": "recipe", "method": method}
    snapshot = None
    if supplied is not None:
        checkpoint = {"quantization_config": supplied}
    else:
        revision = args.get("revision")
        if revision is None:
            checkpoint, content_hash, resolved_revision = _checkpoint_config(model)
        elif isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
            checkpoint, content_hash, resolved_revision = _pinned_checkpoint_config(model, revision)
        else:
            raise InferenceXRecipeError(f"unsupported checkpoint revision: {revision!r}")
        checkpoint = dict(checkpoint)
        companion, companion_hash = None, None
        text = checkpoint.get("text_config") or {}
        if backend == "trtllm" or (not checkpoint.get("quantization_config") and not text.get("quantization_config")):
            if not resolved_revision:
                raise InferenceXRecipeError("cannot pin companion quantization metadata without checkpoint revision")
            companion, companion_hash = _quantization_companion(model, resolved_revision)
        snapshot = {
            "config": checkpoint,
            "hf_quant_config": companion,
            "revision": resolved_revision,
            "config_sha256": _canonical_hash(checkpoint),
            "companion_sha256": _canonical_hash(companion) if companion is not None else None,
        }
        if companion is not None:
            checkpoint = checkpoint | {"hf_quant_config": companion}
        evidence.update(
            source="checkpoint",
            model=model,
            requested_revision=revision or "main",
            revision=resolved_revision,
            content_sha256=content_hash,
            companion_content_sha256=companion_hash,
            historical_revision_verified=revision is not None,
        )
    result = resolve_checkpoint_quantization(
        checkpoint,
        recipe_method=method,
        backend=backend,
        hardware=hardware,
        framework_version=framework_version,
        runtime_args=args,
    )
    result["evidence"] = result["evidence"] | evidence
    if checkpoint.get("expert_dtype"):
        result["evidence"]["expert_dtype"] = checkpoint["expert_dtype"]
    if supplied is not None:
        result["evidence"]["gemm_profile_is_explicit"] = True
    if snapshot is not None:
        if backend == "trtllm" and result["evidence"].get("predictor_snapshot_conflict"):
            snapshot = normalize_trt_snapshot(snapshot)
            if snapshot.get("snapshot_transformation"):
                result["evidence"]["predictor_snapshot_conflict"] = False
                result["evidence"]["snapshot_transformation"] = snapshot["snapshot_transformation"]
        result["_checkpoint_config"] = snapshot
    return result


def verified_defaults(
    args: dict, backend: str, version: str, *, aggregated: bool, hardware: str = "", model: str | None = None
) -> tuple[dict, list]:
    """Only static defaults and the reviewed OpenAI-server sequence-cap rule."""
    values = {}
    evidence = []
    defaults_needed = {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
        "kv_cache_dtype",
        "gpu_memory_utilization",
    }
    if aggregated:
        defaults_needed.add("max_num_seqs")
    if model:
        defaults_needed.add("enable_chunked_prefill")
    if backend != "vllm" or defaults_needed.issubset(args):
        return values, evidence
    try:
        if model and "enable_chunked_prefill" not in args:
            revision = args.get("revision")
            config, config_hash, checkpoint_revision = (
                _pinned_checkpoint_config(model, revision) if revision else _checkpoint_config(model)
            )
            architectures = config.get("architectures", [])
            model_path = "vllm/config/__init__.py" if version == "0.10.2" else "vllm/config/model.py"
            model_source = _framework_source(version, model_path)
            engine_source = _framework_source(version, "vllm/engine/arg_utils.py")
            decoder = architectures and all(str(a).endswith("ForCausalLM") for a in architectures)
            wrapper_evidence = None
            # Conditional generation can wrap a causal decoder. Review the
            # implementation at the exact ref; the class-name suffix is not
            # enough to infer encoder-decoder attention or chunking support.
            wrappers = {
                "e2fa28594f7baad142a426b0b6a2cfe2c79201c7": (
                    "KimiK25ForConditionalGeneration",
                    "vllm/model_executor/models/kimi_k25.py",
                ),
                "0.25.1": ("KimiK25ForConditionalGeneration", "vllm/model_executor/models/kimi_k25.py"),
                "5e35a6f4f9bbc217c599692157ca985c894373f7": (
                    "MiniMaxM3SparseForConditionalGeneration",
                    "vllm/models/minimax_m3/nvidia/model.py",
                ),
                "93d8f834dd8acf33eb0e2a75b2711b628cb6e226": (
                    "MiniMaxM3SparseForConditionalGeneration",
                    "vllm/models/minimax_m3/nvidia/model.py",
                ),
            }
            wrapper = wrappers.get(version)
            if wrapper and architectures == [wrapper[0]]:
                implementation = _framework_source(version, wrapper[1])
                if "self.language_model.compute_logits(hidden_states)" in implementation:
                    decoder = True
                    wrapper_evidence = dict(path=wrapper[1], sha256=hashlib.sha256(implementation.encode()).hexdigest())
            generation_rule = 'logger.debug("Generative models support chunked prefill.")\n            return True'
            if (
                decoder
                and not config.get("is_encoder_decoder", False)
                and args.get("runner") in {None, "auto", "generate"}
                and args.get("task") in {None, "auto", "generate"}
                and generation_rule in model_source
                and "default_chunked_prefill = model_config.is_chunked_prefill_supported" in engine_source
                and "self.enable_chunked_prefill = default_chunked_prefill" in engine_source
            ):
                values["enable_chunked_prefill"] = True
                evidence.append(
                    dict(
                        knob="enable_chunked_prefill",
                        version=version,
                        path=model_path,
                        sha256=hashlib.sha256(model_source.encode()).hexdigest(),
                        engine_sha256=hashlib.sha256(engine_source.encode()).hexdigest(),
                        model=model,
                        architectures=architectures,
                        checkpoint_config_sha256=config_hash,
                        checkpoint_revision=checkpoint_revision,
                        context="causal generative checkpoint",
                        decoder_wrapper=wrapper_evidence,
                    )
                )
        parallel_source = _framework_source(version, "vllm/config/parallel.py")
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "enable_expert_parallel"):
            match = re.search(r"\b" + name + r":\s*(?:int|bool)\s*=\s*(1|False)\b", parallel_source)
            if name not in args and match:
                values[name] = False if match.group(1) == "False" else 1
                evidence.append(
                    dict(
                        knob=name,
                        version=version,
                        path="vllm/config/parallel.py",
                        sha256=hashlib.sha256(parallel_source.encode()).hexdigest(),
                    )
                )
        cache_source = _framework_source(version, "vllm/config/cache.py")
        if "kv_cache_dtype" not in args and re.search(r'cache_dtype:\s*CacheDType\s*=\s*"auto"', cache_source):
            values["kv_cache_dtype"] = "auto"
            evidence.append(
                dict(
                    knob="kv_cache_dtype",
                    version=version,
                    path="vllm/config/cache.py",
                    sha256=hashlib.sha256(cache_source.encode()).hexdigest(),
                )
            )
        if "gpu_memory_utilization" not in args:
            match = re.search(r"gpu_memory_utilization:\s*float\s*=\s*(?:Field\(default=)?([0-9.]+)", cache_source)
            if match:
                values["gpu_memory_utilization"] = float(match.group(1))
                evidence.append(
                    dict(
                        knob="gpu_memory_utilization",
                        version=version,
                        path="vllm/config/cache.py",
                        sha256=hashlib.sha256(cache_source.encode()).hexdigest(),
                    )
                )
        # The caller establishes the entry-point usage context independently.
        if (
            aggregated
            and hardware in {"h100", "h200", "b200", "b300", "gb200", "gb300"}
            and "max_num_seqs" not in args
            and args.get("performance_mode") in {None, "balanced"}
            and isinstance(args.get("max_num_batched_tokens"), int)
        ):
            source = _framework_source(version, "vllm/engine/arg_utils.py")
            pattern = (
                r'if device_memory >= 70 \* GiB_bytes and "a100" not in device_name:'
                r".*?default_max_num_seqs\s*=\s*\{[^}]*UsageContext.OPENAI_API_SERVER:\s*(\d+)"
            )
            match = re.search(pattern, source, re.S)
            if match and "self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)" in source:
                values["max_num_seqs"] = min(int(match.group(1)), args["max_num_batched_tokens"])
                evidence.append(
                    dict(
                        knob="max_num_seqs",
                        version=version,
                        path="vllm/engine/arg_utils.py",
                        sha256=hashlib.sha256(source.encode()).hexdigest(),
                        context="OpenAI serve, NVIDIA >=70 GiB, explicit token cap",
                    )
                )
    except requests.RequestException as error:
        raise InferenceXRecipeError(f"cannot verify framework defaults: {error}") from error
    return values, evidence


def _single_node_result_gpu_count(processor: str, tp: int, metrics: dict) -> int | None:
    """Read the two historical single-node throughput denominators, without execution."""
    try:
        tree = ast.parse(processor)
    except SyntaxError:
        return None

    def assignment(name: str):
        values = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        return values[0] if len(values) == 1 else None

    def matches(node, expression: str) -> bool:
        return node is not None and ast.dump(node) == ast.dump(ast.parse(expression, mode="eval").body)

    if not matches(assignment("tp_size"), "int(single_node_env['TP'])"):
        return None
    data = assignment("single_node_data")
    if not isinstance(data, ast.Dict):
        return None
    fields = {
        key.value: value for key, value in zip(data.keys, data.values, strict=True) if isinstance(key, ast.Constant)
    }
    value = fields.get("tput_per_gpu")
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Div):
        return None
    if matches(value.right, "tp_size"):
        return tp
    if not matches(value.right, "num_gpus") or not matches(assignment("num_gpus"), "tp_size * pp * pcp_size"):
        return None
    # The modern result writer exports these dimensions alongside throughput.
    # Missing measurements cannot be replaced by the launcher's default values.
    factors = [metrics.get(key) for key in ("pp", "pcp_size")]
    if not all(type(factor) is int and factor > 0 for factor in factors):
        return None
    if not all(matches(fields.get(key), key) for key in ("pp", "pcp_size")):
        return None
    return tp * factors[0] * factors[1]


def role_topology(args: dict[str, Any], backend: str, workers: int, gpus: int) -> dict[str, int]:
    """Normalize process-group width versus attention TP for each backend."""
    tp = int(args.get("tensor_parallel_size", 1))
    pp = int(args.get("pipeline_parallel_size", 1))
    if backend == "vllm":
        dp = int(args.get("data_parallel_size", 1))
        width = tp * dp
        ep = width if args.get("enable_expert_parallel", False) else 1
    elif backend == "sglang":
        width = tp
        dp = int(args.get("data_parallel_size", 1)) if args.get("enable_dp_attention") else 1
        ep = int(args.get("expert_parallel_size", args.get("moe_expert_parallel_size", 1)))
        if width % dp:
            raise InferenceXRecipeError("attention DP does not divide process-group width")
        tp = width // dp
    else:
        width = tp
        dp = width if args.get("enable_attention_dp") else 1
        tp = width // dp
        ep = int(args.get("moe_expert_parallel_size", 1))
    if min(tp, pp, dp, ep, workers) < 1 or width % ep:
        raise InferenceXRecipeError("invalid recipe topology")
    if workers * tp * pp * dp != gpus:
        raise InferenceXRecipeError(f"recipe GPU count {workers * tp * pp * dp} differs from measurement {gpus}")
    return dict(tp=tp, pp=pp, attention_dp=dp, moe_tp=width // ep, moe_ep=ep, workers=workers)


def _merge_runtime_workload(model: str, backend: str, benchmark: dict, runtime: tuple) -> dict:
    """Combine independently proven workload and server sources without hiding conflicts."""
    if model != runtime[0] or backend != runtime[3]:
        raise InferenceXRecipeError("runtime workload model/backend differs from server recipe")
    observed = runtime[4]
    for key in (
        "random_range_ratio",
        "num_prompts_mult",
        "num_requests",
        "use_chat_template",
        "custom_tokenizer",
        "ignore_eos",
    ):
        if key in benchmark and key in observed and benchmark[key] != observed[key]:
            raise InferenceXRecipeError("runtime workload conflicts with server recipe: " + key)
    for key in ("request_rate", "req_rate"):
        if benchmark.get(key) is not None and str(benchmark[key]) != str(observed.get("request_rate")):
            raise InferenceXRecipeError("runtime workload conflicts with server recipe: " + key)
    return benchmark | observed


def read_deployment_recipe(row: SiliconRow, source: RecipeSource) -> tuple:
    """Parse historical source independently of defaults and predictor support."""
    original_row = row
    runtime_workload = None
    single_node_runtime = None
    cache_dir = getattr(source, "_cache_dir", None)
    if cache_dir is not None:
        cached = inspect_cached_runtime_recipe(row, cache_dir)
        if cached["parsed"] is not None:
            return cached["parsed"]
        if not row.disagg:
            single_node_runtime = inspect_cached_single_node_runtime(row, cache_dir)["parsed"]
            if single_node_runtime is not None:
                return runtime_deployment(single_node_runtime)
        if row.disagg:
            runtime_workload = inspect_cached_runtime_workload(row, cache_dir)["parsed"]
            if runtime_workload is not None:
                runtime_evidence = runtime_workload[-1]
                # Resolve server source at the actual measured attempt as well.
                row = replace(
                    row, head_sha=runtime_evidence["git_sha"], run_attempt=runtime_evidence["artifact"]["run_attempt"]
                )
    if not row.head_sha or not re.fullmatch(r"[0-9a-fA-F]{40}", row.head_sha):
        raise InferenceXRecipeError("missing immutable workflow source SHA")
    backend = FRAMEWORK_TO_AIC_BACKEND[row.framework]
    config_path, content, config = load_source_config(row, source)
    families = [
        (key, value)
        for key, value in config.items()
        if isinstance(value, dict)
        and value.get("model-prefix") in {row.silicon_model, {"gptoss120b": "gptoss"}.get(row.silicon_model)}
        and value.get("precision") == row.precision
        and value.get("framework") == row.framework
        and str(value.get("runner", "")).removeprefix("cluster:").split("-")[0] == row.hardware
        and bool(value.get("disagg", False)) == row.disagg
    ]
    benchmark = {}
    if row.disagg:
        candidates = []
        for key, family in families:
            try:
                point = _select_disaggregated_search_point(row, family)
                if point.get("spec-decoding", "none") == row.spec_method:
                    candidates.append((key, family, point))
            except InferenceXRecipeError:
                continue
        if len(candidates) != 1:
            raise InferenceXRecipeError(f"expected one immutable recipe point; found {len(candidates)}")
        key, family, point = candidates[0]
        relative = _single_config_file(point)
        if not relative.startswith("recipes/") or ".." in relative.split("/"):
            raise InferenceXRecipeError("unverified external recipe path")
        launcher_copy = verified_launcher_recipe_copy(row, relative, source)
        path = (
            launcher_copy["path"]
            if launcher_copy
            else "benchmarks/multi_node/srt-slurm-recipes/" + relative.removeprefix("recipes/")
        )
        text = source.read_text(row.head_sha, path)
        recipe = _load_yaml_mapping(text, "server recipe")
        raw_roles = recipe.get("backend", {}).get(backend + "_config", {})
        roles = {
            role: _normalize_yaml_server_args(raw_roles[role])
            for role in ("prefill", "decode")
            if isinstance(raw_roles.get(role), dict)
        }
        if len(roles) != 2:
            raise InferenceXRecipeError("recipe lacks both role configurations")
        for role in roles:
            environment = recipe.get("backend", {}).get(role + "_environment", {})
            if isinstance(environment, dict):
                roles[role]["recipe_environment"] = dict(environment)
            if isinstance(raw_roles[role].get("kv_cache_config"), dict):
                roles[role]["kv_cache_config"] = raw_roles[role]["kv_cache_config"]
        resources = recipe.get("resources", {})
        for role in roles:
            if resources.get(role + "_workers") != getattr(row, role + "_num_workers"):
                raise InferenceXRecipeError("recipe worker count differs from measurement")
        benchmark = recipe.get("benchmark", {})
        evidence = dict(
            repository=INFERENCEX_REPOSITORY_URL,
            git_sha=row.head_sha,
            path=path,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            adapter="resolved_deployment_v1",
            server_args={},
            server_args_by_role=roles,
            source_config_path=config_path,
            source_config_sha256=hashlib.sha256(content.encode()).hexdigest(),
            source_config_key=key,
            runtime_image=row.image,
            source_config_image=family.get("image"),
            runtime_defaults={},
            dynamo_installation=recipe.get("dynamo", {}),
        )
        if launcher_copy:
            evidence["launcher_recipe_copy"] = launcher_copy
        workload_identity = verified_launcher_workload_identity(original_row, relative, server_source_sha=row.head_sha)
        if workload_identity:
            evidence["benchmark_source_identity"] = workload_identity
        identity = recipe.get("identity", {})
        version, version_evidence = recipe_runtime_version(recipe, backend, row.image)
        evidence["framework_identity_evidence"] = version_evidence
        model = identity.get("model", {}).get("repo") or recipe.get("model", {}).get("path") or family.get("model")
        if isinstance(model, str) and "/" not in model and "/" in str(family.get("model", "")):
            evidence["recipe_model_alias"] = model
            model = family["model"]
    else:
        candidates = []
        for key, family in families:
            sequences = family.get("scenarios", {}).get("fixed-seq-len", family.get("seq-len-configs", []))
            for sequence in sequences:
                if sequence.get("isl") != row.isl or sequence.get("osl") != row.osl:
                    continue
                for point in sequence.get("search-space", []):
                    if (
                        point.get("tp") == row.decode_tp
                        and point.get("ep", 1) == row.decode_ep
                        and point.get("dp-attn", False) == row.decode_dp_attention
                        and point.get("spec-decoding", "none") == row.spec_method
                        and (
                            row.conc in point.get("conc-list", [])
                            or point.get("conc-start", float("inf")) <= row.conc <= point.get("conc-end", -1)
                        )
                    ):
                        candidates.append((key, family, point))
        if len(candidates) != 1:
            raise InferenceXRecipeError(f"expected one aggregate source point; found {len(candidates)}")
        key, family, point = candidates[0]
        if point.get("additional-settings"):
            raise InferenceXRecipeError("aggregate source point has unreviewed additional settings")
        model = family.get("model")
        path, text, args, benchmark = read_shell_recipe(
            replace(row, silicon_model=family["model-prefix"]), source, model, legacy_context=family
        )
        args = _normalize_yaml_server_args(args)
        roles = {"aggregated": args}
        evidence = dict(
            repository=INFERENCEX_REPOSITORY_URL,
            git_sha=row.head_sha,
            path=path,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            adapter="resolved_deployment_v1",
            server_args=args,
            server_args_by_role={},
            source_config_path=config_path,
            source_config_sha256=hashlib.sha256(content.encode()).hexdigest(),
            source_config_key=key,
            runtime_image=row.image,
            source_config_image=family.get("image"),
            runtime_defaults={},
        )
        version = runtime_version(row.image or family.get("image"), backend)
        if family.get("_legacy_sources"):
            evidence["legacy_sources"] = family["_legacy_sources"]
    if runtime_workload is not None:
        benchmark = _merge_runtime_workload(model, backend, benchmark, runtime_workload)
        evidence["runtime_workload"] = runtime_workload[-1]
        evidence["workflow_head_sha"] = original_row.head_sha
        evidence["recorded_run_attempt"] = original_row.run_attempt
    return model, roles, version, backend, benchmark, evidence


def inspect_deployment(
    row: SiliconRow, source: RecipeSource, *, parsed: tuple | None = None
) -> tuple[dict | None, dict, list[dict]]:
    """Collect independent blockers without substituting values to advance validation."""
    issues = []

    def issue(stage, message, role=None, **details):
        issues.append(dict(stage=stage, role=role, message=message, **details))

    model, roles, version, backend, benchmark, evidence = parsed or read_deployment_recipe(row, source)
    if not isinstance(model, str) or "/" not in model or model.startswith("/"):
        issue("model_identity", "source model path is an unresolved alias")
        return None, evidence, issues
    resolved_roles = {}
    checkpoint_snapshot = None
    for role, args in roles.items():
        args = dict(args)
        if backend == "trtllm":
            kv = args.get("kv_cache_config", {})
            args.update(
                {
                    k: v
                    for k, v in {
                        "kv_cache_dtype": kv.get("dtype"),
                        "block_size": kv.get("tokens_per_block"),
                        "free_gpu_memory_fraction": kv.get("free_gpu_memory_fraction"),
                        "enable_prefix_caching": kv.get("enable_block_reuse"),
                    }.items()
                    if v is not None
                }
            )
            for original, target in [("max_batch_size", "max_num_seqs"), ("max_num_tokens", "max_num_batched_tokens")]:
                if original in args:
                    args[target] = args[original]
        if args.get("no_enable_chunked_prefill"):
            args["enable_chunked_prefill"] = False
        # Explicit controls and independently reviewed runtime defaults only.
        role_version = version.get(role) if isinstance(version, dict) else version
        role_version = re.sub(r"^v(?=\d)", "", role_version) if isinstance(role_version, str) else None
        nightly = re.search(r"(?:^|/)vllm-openai:nightly-([0-9a-f]{40})(?:$|[-@])", row.image or "")
        framework_source_ref = role_version or (nightly.group(1) if nightly and backend == "vllm" else None)
        installed_commit = re.search(r"\+g([0-9a-f]{9,40})$", role_version or "")
        if (
            backend == "vllm"
            and nightly
            and installed_commit
            and nightly.group(1).startswith(installed_commit.group(1))
        ):
            framework_source_ref = nightly.group(1)
            evidence.setdefault("framework_source_identity", {})[role] = {
                "framework_version": role_version,
                "git_sha": framework_source_ref,
                "source": "runtime fingerprint Git suffix agrees with full nightly image revision",
            }
        verified, sources = {}, []
        usage_context = None
        try:
            if backend == "vllm" and row.disagg:
                usage_context, worker_sources = verified_dynamo_usage_context(
                    evidence.get("dynamo_installation", {}),
                    runtime_version=evidence.get("runtime_fingerprints", {}).get(role, {}).get("dynamo_version"),
                )
                if usage_context:
                    evidence["worker_usage_context"] = dict(value=usage_context, sources=worker_sources)
                    evidence.setdefault("worker_usage_context_by_role", {})[role] = evidence["worker_usage_context"]
            if framework_source_ref:
                verified, sources = verified_defaults(
                    args,
                    backend,
                    framework_source_ref,
                    aggregated=not row.disagg or usage_context == "openai_api_server",
                    hardware=row.hardware,
                    model=model,
                )
        except (InferenceXRecipeError, requests.RequestException, ValueError) as error:
            issue("framework_defaults", str(error), role)
        evidence.setdefault("verified_defaults", {})[role] = {"values": verified, "sources": sources}
        args = verified | args
        try:
            checkpoint = None
            weight_dtype = None
            if framework_source_ref and (
                backend in {"sglang", "trtllm"} or has_reviewed_defaults(backend, framework_source_ref)
            ):
                revision = args.get("revision")
                record = _pinned_checkpoint_config(model, revision) if revision else _checkpoint_config(model)
                checkpoint = record[0]
                evidence["verified_defaults"][role]["checkpoint"] = {
                    "model": model,
                    "source_config_sha256": record[1],
                    "revision": record[2],
                    "historical_revision_verified": bool(revision),
                }
                if (
                    backend == "vllm"
                    and args.get("kv_cache_dtype", "auto") == "auto"
                    and args.get("dtype", "auto") == "auto"
                    and checkpoint.get("architectures") in (["GptOssForCausalLM"], ["MiniMaxM2ForCausalLM"])
                    and not (checkpoint.get("dtype") or checkpoint.get("torch_dtype"))
                    and record[2]
                ):
                    weight_dtype, weight_evidence = _checkpoint_weight_dtype(model, record[2])
                    evidence["verified_defaults"][role]["checkpoint_weight_dtype"] = weight_evidence
                if backend == "trtllm" and record[2]:
                    companion, _ = _quantization_companion(model, record[2])
                    checkpoint = checkpoint | {"hf_quant_config": companion}
            args, effective_sources = apply_framework_defaults(
                args,
                backend,
                framework_source_ref,
                aggregated=not row.disagg,
                hardware=row.hardware,
                checkpoint=checkpoint,
                checkpoint_weight_dtype=weight_dtype,
                usage_context=usage_context,
            )
            if backend == "sglang":
                args, additional_sources = apply_additional_sglang_defaults(
                    args, framework_source_ref, hardware=row.hardware, checkpoint=checkpoint
                )
                effective_sources.extend(additional_sources)
            elif backend == "trtllm":
                args, additional_sources = apply_additional_trt_defaults(
                    args,
                    framework_source_ref,
                    aggregated=not row.disagg,
                    hardware=row.hardware,
                    checkpoint=checkpoint,
                )
                effective_sources.extend(additional_sources)
            evidence["verified_defaults"][role]["effective_rules"] = effective_sources
        except (InferenceXRecipeError, requests.RequestException, ValueError) as error:
            issue("framework_defaults", str(error), role)
        workers = getattr(row, role + "_num_workers") if row.disagg else 1
        gpus = getattr(row, "num_" + role + "_gpu") if row.disagg else row.num_decode_gpu
        if not row.disagg and not row.is_multinode and row.decode_ep > 1 and gpus == row.decode_tp * row.decode_ep:
            try:
                path = "utils/process_result.py"
                processor = source.read_text(row.head_sha, path)
                effective_gpus = _single_node_result_gpu_count(processor, row.decode_tp, row.metrics)
                if effective_gpus is not None:
                    evidence["gpu_count_validation"] = {
                        "reported": gpus,
                        "effective": effective_gpus,
                        "path": path,
                        "content_sha256": hashlib.sha256(processor.encode()).hexdigest(),
                        "dimensions": {key: row.metrics[key] for key in ("pp", "pcp_size") if key in row.metrics},
                        "reason": "historical single-node throughput denominator; EP reuses the TP world",
                    }
                    gpus = effective_gpus
            except InferenceXRecipeError:
                pass
        topology, quantization = None, None
        try:
            topology = role_topology(args, backend, workers, gpus)
        except (InferenceXRecipeError, ValueError, TypeError, ZeroDivisionError) as error:
            issue("topology", str(error), role)
        try:
            quantization = resolve_quantization(
                args, model, backend=backend, hardware=row.hardware, framework_version=role_version
            )
            if quantization.get("evidence", {}).get("predictor_snapshot_conflict"):
                issue(
                    "predictor_mapping",
                    "predictor checkpoint loader cannot preserve TRT sidecar precedence over inline quantization",
                    role,
                )
            snapshot = quantization.pop("_checkpoint_config", None)
            if checkpoint_snapshot is not None and snapshot != checkpoint_snapshot:
                issue("model_identity", "role checkpoint configurations differ", role)
            elif snapshot is not None:
                checkpoint_snapshot = snapshot
        except (InferenceXRecipeError, requests.RequestException, ValueError, KeyError) as error:
            kind = getattr(error, "kind", "unresolved_quantization")
            issue(
                "predictor_mapping"
                if kind in {"unsupported_predictor_representation", "unverified_kernel_mapping"}
                else "quantization",
                str(error),
                role,
                kind=kind,
                details=getattr(error, "details", {}),
            )
        estimated_knobs = {}
        required = [
            "max_num_seqs",
            "max_num_batched_tokens",
            "block_size",
            "enable_prefix_caching",
            "kv_cache_dtype",
            "enable_chunked_prefill",
        ]
        memory_key = {
            "vllm": "gpu_memory_utilization",
            "sglang": "mem_fraction_static",
            "trtllm": "free_gpu_memory_fraction",
        }[backend]
        missing = [k for k in [*required, memory_key] if args.get(k) is None]
        if missing:
            issue("required_knobs", f"{role}: unresolved required knobs: {', '.join(missing)}", role, knobs=missing)
        if args.get("kv_cache_dtype") == "auto":
            issue(
                "required_knobs", f"{role}: effective KV dtype for auto is unresolved", role, knobs=["kv_cache_dtype"]
            )
        resolved_roles[role] = dict(
            framework_version=role_version,
            topology=topology,
            args=args,
            quantization=quantization,
            estimated_knobs=estimated_knobs,
        )
    if row.disagg:
        evidence["server_args_by_role"] = {role: spec["args"] for role, spec in resolved_roles.items()}
    else:
        evidence["server_args"] = resolved_roles["aggregated"]["args"]
    workload = {"isl": row.isl, "osl": row.osl, "concurrency": row.conc}
    try:
        benchmark, workload_sources = resolve_workload_defaults(benchmark, evidence.get("benchmark_source_identity"))
        evidence["verified_workload_defaults"] = workload_sources
    except InferenceXRecipeError as error:
        issue("workload", str(error))
    workload["benchmark_controls"] = benchmark
    workload["unsupported_controls"] = unmodeled_workload_controls(benchmark)
    # A reviewed benchmark adapter must establish distribution, not just mean lengths.
    if "random_range_ratio" not in benchmark or "num_prompts_mult" not in benchmark:
        issue("workload", "benchmark token-length distribution or request count is not established by recipe")
    else:
        try:
            workload["random_range_ratio"] = float(benchmark["random_range_ratio"])
            workload["request_count"] = int(benchmark["num_prompts_mult"]) * row.conc
            if not 0 < workload["random_range_ratio"] <= 1 or workload["request_count"] <= 0:
                raise ValueError("benchmark range ratio and request count must be positive, with ratio <= 1")
        except (ValueError, TypeError) as error:
            issue("workload", str(error))
    evidence["resolution_issues"] = issues
    if issues:
        return None, evidence, issues
    return (
        dict(
            schema_version="resolved-deployment/1",
            configuration_quality="estimated" if evidence.get("estimated_knobs") else "verified",
            backend=backend,
            system=HARDWARE_TO_SYSTEM[row.hardware],
            model_path=model,
            checkpoint_config=checkpoint_snapshot,
            roles=resolved_roles,
            workload=workload,
        ),
        evidence,
        issues,
    )


def resolve_deployment(row: SiliconRow, source: RecipeSource, *, parsed: tuple | None = None) -> tuple[dict, dict]:
    deployment, evidence, issues = inspect_deployment(row, source, parsed=parsed)
    if issues:
        raise InferenceXRecipeError("; ".join(item["message"] for item in issues))
    assert deployment is not None
    return deployment, evidence


def estimate_kwargs(row: SiliconRow, deployment: dict) -> CliEstimateKwargs:
    roles = deployment["roles"]
    kvs = {r["args"]["kv_cache_dtype"] for r in roles.values()}
    if len(kvs) != 1:
        raise InferenceXRecipeError("AIC estimate API cannot represent different role KV dtypes")
    kv = next(iter(kvs))
    kv = {"fp8_e4m3": "fp8", "bf16": "bfloat16"}.get(kv, kv)
    common = dict(
        model_path=deployment["model_path"],
        system_name=deployment["system"],
        backend_name=deployment["backend"],
        mode="disagg" if row.disagg else "agg",
        isl=row.isl,
        osl=row.osl,
        kvcache_quant_mode=kv,
    )
    for name in ("gemm", "moe"):
        profiles = {spec.get("quantization", {}).get(name) for spec in roles.values()}
        if None in profiles:
            raise InferenceXRecipeError(f"unresolved {name} quantization profile")
        if len(profiles) != 1:
            raise InferenceXRecipeError(f"AIC estimate API cannot represent different role {name} profiles")
        inferred_gemm = name == "gemm" and all(
            spec["quantization"].get("evidence", {}).get("gemm_profile_is_explicit") is False for spec in roles.values()
        )
        if not inferred_gemm and (name != "moe" or row.silicon_model in MOE_MODELS):
            common[name + "_quant_mode"] = next(iter(profiles))
    kw = CliEstimateKwargs(**common)
    kw.model_config_snapshot = deployment.get("checkpoint_config")
    for role, spec in roles.items():
        t = spec["topology"]
        prefix = "" if role == "aggregated" else role + "_"
        for source, target in [
            ("tp", "tp_size"),
            ("pp", "pp_size"),
            ("attention_dp", "attention_dp_size"),
            ("moe_tp", "moe_tp_size"),
            ("moe_ep", "moe_ep_size"),
        ]:
            if source.startswith("moe") and row.silicon_model not in MOE_MODELS:
                continue
            setattr(kw, prefix + target, t[source])
        if role != "aggregated":
            setattr(kw, prefix + "num_workers", t["workers"])
        if "max_model_len" in spec["args"]:
            setattr(kw, prefix + "max_seq_len", spec["args"]["max_model_len"])
        memory_field = {
            "vllm": "gpu_memory_utilization",
            "sglang": "mem_fraction_static",
            "trtllm": "free_gpu_memory_fraction",
        }[deployment["backend"]]
        if memory_field in spec["args"]:
            setattr(kw, prefix + "free_gpu_memory_fraction", spec["args"][memory_field])
        denom = t["workers"] * t["attention_dp"]
        if role != "prefill" and row.conc % denom:
            raise InferenceXRecipeError("AIC fixed batch cannot represent concurrency / workers / DP")
        setattr(kw, prefix + "batch_size", 1 if role == "prefill" else row.conc // denom)
    return kw
