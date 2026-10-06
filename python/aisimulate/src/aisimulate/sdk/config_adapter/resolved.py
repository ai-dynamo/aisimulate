# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt already resolved source evidence; no fetching or source execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .inferencex import MOE_MODELS
from .schema import AdaptationDiagnostic, AdaptationOutcome, AdaptationReport, AdapterOverrides, EstimateRequestV1


@dataclass(frozen=True)
class ResolvedInferenceXSource:
    deployment: Mapping[str, Any]
    config: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    source_reference: str


def adapt_resolved_inferencex(source: ResolvedInferenceXSource, overrides: AdapterOverrides) -> AdaptationReport:
    point_id = str(source.benchmark.get("id", "resolved"))
    try:
        supported_overrides = {
            "model_path",
            "backend_version",
            "database_mode",
            "system_name",
            "decode_system_name",
            "gemm_quant_mode",
            "moe_quant_mode",
            "kvcache_quant_mode",
            "free_gpu_memory_fraction",
            "prefill_free_gpu_memory_fraction",
            "decode_free_gpu_memory_fraction",
            "max_seq_len",
            "prefill_max_seq_len",
            "decode_max_seq_len",
        }
        unsupported = {
            name for name in overrides.model_fields_set - supported_overrides if getattr(overrides, name) is not None
        }
        if unsupported:
            raise ValueError(f"resolved source does not support these overrides: {', '.join(sorted(unsupported))}")
        deployment = source.deployment
        if deployment.get("schema_version") != "resolved-deployment/1":
            raise ValueError("unsupported resolved deployment version")
        roles = deployment["roles"]
        disagg = set(roles) == {"prefill", "decode"}
        if not disagg and set(roles) != {"aggregated"}:
            raise ValueError("resolved deployment must contain aggregated or prefill/decode roles")
        workload = deployment["workload"]
        if (
            any(
                workload[name] != source.benchmark[key]
                for name, key in (("isl", "isl"), ("osl", "osl"), ("concurrency", "conc"))
            )
            or disagg != source.config["disagg"]
        ):
            raise ValueError("resolved deployment does not match the benchmark")
        backend = deployment["backend"]
        is_moe = source.config.get("silicon_model", source.config.get("model")) in MOE_MODELS
        memory_field = {
            "vllm": "gpu_memory_utilization",
            "sglang": "mem_fraction_static",
            "trtllm": "free_gpu_memory_fraction",
        }[backend]
        topology, runtime = {}, {}
        for role, spec in roles.items():
            shape = spec["topology"]
            if any(
                type(shape[key]) is not int or shape[key] <= 0
                for key in ("tp", "pp", "attention_dp", "moe_tp", "moe_ep", "workers")
            ):
                raise ValueError("resolved worker dimensions must be positive integers")
            divisor = shape["workers"] * shape["attention_dp"]
            if role != "prefill" and workload["concurrency"] % divisor:
                raise ValueError("fixed batch cannot represent concurrency / workers / attention DP")
            topology["worker" if role == "aggregated" else role] = {
                "replicas": shape["workers"],
                "gpus_per_replica": shape["tp"] * shape["pp"] * shape["attention_dp"],
                "tp_size": shape["tp"],
                "pp_size": shape["pp"],
                "attention_dp_size": shape["attention_dp"],
                "moe_tp_size": shape["moe_tp"] if is_moe else None,
                "moe_ep_size": shape["moe_ep"] if is_moe else None,
                "batch_size": 1 if role == "prefill" else workload["concurrency"] // divisor,
            }
            prefix = "" if role == "aggregated" else role + "_"
            for key, field in ((memory_field, "free_gpu_memory_fraction"), ("max_model_len", "max_seq_len")):
                value = getattr(overrides, prefix + field)
                if value is None:
                    value = getattr(overrides, field)
                if value is None:
                    value = spec["args"].get(key)
                if value is not None:
                    runtime[prefix + field] = value
        quantization = {}
        for name, field in (("gemm", "gemm_quant_mode"), ("moe", "moe_quant_mode"), ("kvcache", "kvcache_quant_mode")):
            values = {
                spec["args"]["kv_cache_dtype"] if name == "kvcache" else spec["quantization"][name]
                for spec in roles.values()
            }
            if None in values or len(values) != 1:
                raise ValueError(f"estimate request cannot represent unresolved or different per-role {name}")
            value = next(iter(values))
            if name == "kvcache":
                value = {"fp8_e4m3": "fp8", "bf16": "bfloat16"}.get(value, value)
            if name == "gemm" and all(
                spec["quantization"].get("evidence", {}).get("gemm_profile_is_explicit") is False
                for spec in roles.values()
            ):
                value = None
            if name == "moe" and not is_moe:
                value = None
            quantization[name] = getattr(overrides, field) if getattr(overrides, field) is not None else value
        request = EstimateRequestV1.model_validate(
            {
                "model": {"path": overrides.model_path or deployment["model_path"]},
                "backend": {
                    "name": backend,
                    "version": overrides.backend_version,
                    "database_mode": overrides.database_mode or "SILICON",
                },
                "systems": {
                    "prefill": overrides.system_name or deployment["system"],
                    "decode": overrides.decode_system_name if disagg else None,
                },
                "workload": {name: workload[name] for name in ("isl", "osl", "concurrency")},
                "quantization": quantization,
                "topology": {"kind": "disagg" if disagg else "agg", **topology},
                "runtime": runtime,
                "provenance": {
                    "source_type": "inferencex",
                    "source_reference": source.source_reference,
                    "source_ids": {"config_id": source.config["id"], "benchmark_id": source.benchmark["id"]},
                },
            }
        )
        diagnostics = (
            ()
            if overrides.backend_version is not None
            else (
                AdaptationDiagnostic(
                    severity="warning",
                    code="backend_version_unpinned",
                    message="Backend version is not pinned; AIC will select its latest compatible database version.",
                    path="backend.version",
                ),
            )
        )
        outcome = AdaptationOutcome(point_id=point_id, status="adapted", request=request, diagnostics=diagnostics)
    except (ValueError, TypeError, KeyError) as error:
        outcome = AdaptationOutcome(
            point_id=point_id,
            status="rejected",
            diagnostics=(
                AdaptationDiagnostic(severity="error", code="resolved_inferencex_mapping_failed", message=str(error)),
            ),
        )
    return AdaptationReport(outcomes=(outcome,))
