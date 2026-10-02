# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower resolved deployments for historical wheels without the public adapter."""

from .inferencex_recipe import InferenceXRecipeError
from .mapping import MOE_MODELS
from .schema import CliEstimateKwargs, SiliconRow


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
