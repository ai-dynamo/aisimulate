# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower an evidence-qualified deployment using gym's modeled replay controls."""

from .model_config_snapshot import materialize_model_config

ENGINE_FIELDS = (
    "block_size",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "mem_fraction_static",
    "free_gpu_memory_fraction",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "max_model_len",
    "prefill_schedule_interval",
    "prefill_decode_interval",
)


def engine_args(deployment, role, database_version):
    spec = deployment["roles"][role]
    topology = spec["topology"]
    quantization = spec["quantization"]
    args = {
        "engine_type": deployment["backend"],
        "aic_backend": deployment["backend"],
        "aic_backend_version": database_version,
        "aic_system": deployment["system"],
        "aic_model_path": deployment["model_path"],
        "aic_gemm_dtype": quantization["gemm"],
        "aic_moe_dtype": quantization["moe"],
        "aic_tp_size": topology["tp"],
        "aic_pp_size": topology["pp"],
        "aic_attention_dp_size": topology["attention_dp"],
        "aic_moe_tp_size": topology["moe_tp"],
        "aic_moe_ep_size": topology["moe_ep"],
    }
    if snapshot := deployment.get("checkpoint_config"):
        args["aic_model_path"] = materialize_model_config(snapshot)
        evidence = quantization.get("evidence", {})
        if evidence.get("gemm_profile_is_explicit") is False or (
            "gemm_profile_is_explicit" not in evidence and evidence.get("requires_checkpoint_split")
        ):
            args.pop("aic_gemm_dtype")
    args.update({name: spec["args"][name] for name in ENGINE_FIELDS if name in spec["args"]})
    kv = spec["args"]["kv_cache_dtype"]
    args["aic_kv_cache_dtype"] = {"fp8_e4m3": "fp8", "bf16": "bfloat16"}.get(kv, kv)
    return args


def replay_spec(deployment, database_version, deployment_type, spec_type):
    roles = deployment["roles"]
    disagg = "prefill" in roles
    engines = {role: engine_args(deployment, role, database_version) for role in roles}
    workload = deployment["workload"]
    concurrency = workload["concurrency"]
    kwargs = (
        {
            "prefill_engine_args": engines["prefill"],
            "decode_engine_args": engines["decode"],
            "num_prefill_workers": roles["prefill"]["topology"]["workers"],
            "num_decode_workers": roles["decode"]["topology"]["workers"],
        }
        if disagg
        else {
            "agg_engine_args": engines["aggregated"],
            "num_workers": roles["aggregated"]["topology"]["workers"],
        }
    )
    return spec_type(
        backend_deployment=deployment_type(
            deployment_mode="disagg" if disagg else "agg",
            backend=deployment["backend"],
            backend_version=database_version,
            **kwargs,
        ),
        workload={
            "isl": workload["isl"],
            "osl": workload["osl"],
            "request_count": workload["request_count"],
            "concurrency": concurrency,
            "replay_concurrency": concurrency,
            "random_range_ratio": workload["random_range_ratio"],
            "random_seed": workload.get("random_seed", workload.get("benchmark_controls", {}).get("seed", 0)),
            "length_sampler": "numpy_random_state",
            "ignore_eos": True,
        },
        goal={},
        concurrency=concurrency,
    )
