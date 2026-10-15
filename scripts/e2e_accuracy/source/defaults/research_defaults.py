# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in experiment assumptions; these do not establish historical runtime settings."""

from __future__ import annotations

import hashlib
import json

from scripts.e2e_accuracy.source.filter import FRAMEWORK_TO_AIC_BACKEND
from scripts.e2e_accuracy.source.mapping import MOE_MODELS, _resolve_worker_shape
from scripts.e2e_accuracy.source.recipes.inferencex_recipe import InferenceXRecipeError
from scripts.e2e_accuracy.source.recipes.legacy_recipe import load_source_config

MODEL_PATH_LOOKUP: dict[tuple[str, str], str] = {
    # MiniMax-M2.5
    ("minimaxm2.5", "bf16"): "MiniMaxAI/MiniMax-M2.5",
    ("minimaxm2.5", "fp4"): "MiniMaxAI/MiniMax-M2.5",
    ("minimaxm2.5", "fp8"): "MiniMaxAI/MiniMax-M2.5",
    # MiniMax-M2.7 uses its native ModelOpt artifact for InferenceX FP4.
    ("minimaxm2.7", "bf16"): "MiniMaxAI/MiniMax-M2.7",
    ("minimaxm2.7", "fp4"): "nvidia/MiniMax-M2.7-NVFP4",
    # DeepSeek-R1 (silicon ships fp4/fp8; AISim's DeepSeek-V3 config is the
    # closest match — same DeepseekV3ForCausalLM architecture).
    ("dsr1", "bf16"): "deepseek-ai/DeepSeek-V3",
    ("dsr1", "fp4"): "deepseek-ai/DeepSeek-V3",
    ("dsr1", "fp8"): "deepseek-ai/DeepSeek-V3",
    # Kimi-K2.5 (silicon ships fp4 + int4 + fp8; AISim has the bf16 source).
    ("kimik2.5", "bf16"): "moonshotai/Kimi-K2.5",
    ("kimik2.5", "fp4"): "moonshotai/Kimi-K2.5",
    ("kimik2.5", "fp8"): "moonshotai/Kimi-K2.5",
    ("kimik2.5", "int4"): "moonshotai/Kimi-K2.5",
    # Kimi-K2.6 and Kimi-K3 are registered AISim family members. Their FP4
    # checkpoints carry mixed-quant metadata that AISim must infer natively.
    ("kimik2.6", "fp4"): "nvidia/Kimi-K2.6-NVFP4",
    ("kimik3", "fp4"): "moonshotai/Kimi-K3",
    # Qwen3.5 (silicon ships bf16/fp4/fp8 with sglang).
    ("qwen3.5", "bf16"): "Qwen/Qwen3.5-397B-A17B",
    ("qwen3.5", "fp4"): "Qwen/Qwen3.5-397B-A17B",
    ("qwen3.5", "fp8"): "Qwen/Qwen3.5-397B-A17B",
    # Llama-3.1-70B-Instruct (dense).
    ("llama70b", "bf16"): "meta-llama/Meta-Llama-3.1-70B",
    ("llama70b", "fp4"): "meta-llama/Meta-Llama-3.1-70B",
    ("llama70b", "fp8"): "meta-llama/Meta-Llama-3.1-70B",
    # gpt-oss carries native MXFP4 metadata; AISim infers its operator modes.
    ("gptoss120b", "fp4"): "openai/gpt-oss-120b",
    # DeepSeek-V4-Pro has distinct native-FP4 and Hopper-compatible FP8 IDs.
    ("dsv4", "fp4"): "deepseek-ai/DeepSeek-V4-Pro",
    ("dsv4", "fp8"): "sgl-project/DeepSeek-V4-Pro-FP8",
    # GLM's precision-specific artifacts are registered independently in AISim.
    ("glm5", "bf16"): "zai-org/GLM-5",
    ("glm5", "fp4"): "nvidia/GLM-5-NVFP4",
    ("glm5", "fp8"): "zai-org/GLM-5-FP8",
    ("glm5.1", "bf16"): "zai-org/GLM-5.1",
    ("glm5.1", "fp4"): "nvidia/GLM-5.1-NVFP4",
    ("glm5.1", "fp8"): "zai-org/GLM-5.1-FP8",
    ("glm5.2", "bf16"): "zai-org/GLM-5.2",
    ("glm5.2", "fp4"): "nvidia/GLM-5.2-NVFP4",
    ("glm5.2", "fp8"): "zai-org/GLM-5.2-FP8",
    # Registered AISim family member. Individual SILICON points remain subject to
    # cli_estimate validation even when the model is absent from the active
    # support-matrix roster.
    ("minimaxm3", "bf16"): "MiniMaxAI/MiniMax-M3",
    ("minimaxm3", "fp4"): "MiniMaxAI/MiniMax-M3",
    ("minimaxm3", "fp8"): "MiniMaxAI/MiniMax-M3",
}

PROFILE = "coverage-experiment/1"
# Deliberately explicit experimental starting values, not universal framework defaults.
# Exact source-derived defaults are applied first by deployment.py.
SERVER_DEFAULTS = {
    "vllm": dict(
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        block_size=16,
        kv_cache_dtype="bfloat16",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    ),
    "sglang": dict(
        max_num_seqs=4096,
        max_num_batched_tokens=16384,
        block_size=1,
        kv_cache_dtype="bfloat16",
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        mem_fraction_static=0.9,
    ),
    "trtllm": dict(
        max_num_seqs=2048,
        max_num_batched_tokens=8192,
        block_size=64,
        kv_cache_dtype="bfloat16",
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        free_gpu_memory_fraction=0.9,
    ),
}
WORKLOAD_DEFAULTS = dict(random_range_ratio=0.8, num_prompts_mult=10)


def fill_missing(values: dict, defaults: dict, *, role: str) -> tuple[dict, list[dict]]:
    """Never overwrite a known value, including false/zero or an explicit 'auto'."""
    result = dict(values)
    assumptions = []
    for knob, value in defaults.items():
        if result.get(knob) is None:
            result[knob] = value
            assumptions.append(
                dict(
                    role=role,
                    knob=knob,
                    value=value,
                    profile=PROFILE,
                    kind="assumed_default",
                    historical_value_verified=False,
                )
            )
    return result, assumptions


def can_assume_recipe(error: str) -> bool:
    """Missing sources/layouts may be estimated; conflicting or corrupt evidence may not."""
    return "recipe does not exist at" in error or error in {
        "no reviewed legacy workflow family",
        "no reviewed legacy shell family",
        "recipe lacks both role configurations",
        "expected one immutable recipe point; found 0",
        "expected one aggregate source point; found 0",
    }


def resolve_auto_kv(args: dict, snapshot: dict | None, *, role: str) -> tuple[dict, list[dict]]:
    """An experimental interpretation of auto; never replace an explicit KV dtype."""
    if args.get("kv_cache_dtype") != "auto":
        return args, []
    config = (snapshot or {}).get("config", {})
    config = config.get("text_config") or config
    dtype = config.get("dtype") or config.get("torch_dtype") or "bfloat16"
    dtype = {"bf16": "bfloat16", "float16": "float16", "half": "float16"}.get(dtype, dtype)
    if dtype not in {"bfloat16", "float16"}:
        return args, []
    return args | {"kv_cache_dtype": dtype}, [
        dict(
            role=role,
            knob="kv_cache_dtype",
            value=dtype,
            requested_value="auto",
            profile=PROFILE,
            kind="assumed_default",
            historical_value_verified=False,
            reason="interpret auto as checkpoint weight dtype, or bfloat16 when absent",
        )
    ]


def assumed_recipe(row, error: str, source=None) -> tuple:
    """Use observed DB topology with an explicitly assumed model/configuration mapping."""
    if not can_assume_recipe(error):
        raise InferenceXRecipeError(error)
    backend = FRAMEWORK_TO_AIC_BACKEND[row.framework]
    # Reuse the existing DB topology conversion and its divisibility/ambiguity checks.
    model = MODEL_PATH_LOOKUP[(row.silicon_model, row.precision)]
    model_source = None
    if source is not None:
        try:
            path, content, config = load_source_config(row, source)
        except InferenceXRecipeError as source_error:
            if not can_assume_recipe(str(source_error)):
                raise
        else:
            models = {
                family.get("model")
                for family in config.values()
                if isinstance(family, dict)
                and family.get("model-prefix") in {row.silicon_model, {"gptoss120b": "gptoss"}.get(row.silicon_model)}
                and family.get("precision") == row.precision
                and family.get("framework") == row.framework
                and str(family.get("runner", "")).removeprefix("cluster:").split("-")[0] == row.hardware
                and bool(family.get("disagg", False)) == row.disagg
                and isinstance(family.get("model"), str)
            }
            if len(models) == 1:
                model = models.pop()
                model_source = dict(
                    path=path,
                    git_sha=row.head_sha,
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                )
    roles = {}
    for role in ("prefill", "decode") if row.disagg else ("aggregated",):
        observed_role = role if row.disagg else "decode"
        shape = _resolve_worker_shape(
            num_gpu=getattr(row, f"num_{observed_role}_gpu"),
            num_workers=getattr(row, f"{observed_role}_num_workers") if row.disagg else 1,
            tp=getattr(row, f"{observed_role}_tp"),
            ep=getattr(row, f"{observed_role}_ep"),
            dp_attention=getattr(row, f"{observed_role}_dp_attention"),
            is_moe=row.silicon_model in MOE_MODELS,
            backend=backend,
            single_node_vllm=not row.disagg and not row.is_multinode and backend == "vllm",
        )
        tp, dp, ep = shape.tp_size, shape.attention_dp_size, shape.moe_ep_size or 1
        args = dict(
            model_path=model,
            tensor_parallel_size=tp,
            pipeline_parallel_size=shape.pp_size,
            data_parallel_size=dp,
            enable_expert_parallel=ep > 1,
        )
        if backend != "vllm":
            args.update(
                tensor_parallel_size=tp * dp,
                expert_parallel_size=ep,
                moe_expert_parallel_size=ep,
                enable_dp_attention=dp > 1,
                enable_attention_dp=dp > 1,
            )
        roles[role] = args
    evidence = dict(
        adapter="resolved_deployment_v1",
        source="assumed_database_recipe",
        git_sha=None,
        content_sha256=hashlib.sha256(
            json.dumps(dict(profile=PROFILE, model=model, roles=roles), sort_keys=True).encode()
        ).hexdigest(),
        path=None,
        original_recipe_error=error,
        model_source=model_source,
        server_args_by_role=roles,
        estimated_defaults=[
            dict(
                kind="assumed_recipe",
                profile=PROFILE,
                model_path=model,
                roles=roles,
                historical_value_verified=False,
            )
        ],
    )
    # An image tag for Dynamo is not a backend version. Preserve unknown here.
    return model, roles, None, backend, {"type": "sa-bench"}, evidence
