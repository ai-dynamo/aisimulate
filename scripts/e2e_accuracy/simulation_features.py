# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI-owned effective simulation settings, never part of HF silicon intake."""

import hashlib
import json
import math
import re

from scripts.e2e_accuracy.benchmark_features import SILICON_FEATURES

SCHEMA = "aisimulate.agentx.simulation-features.v1"
STATES = {"enabled", "disabled", "not_applicable", "unsupported", "failed_to_record"}
PARAMETERS = {
    "match_granularity_tokens",
    "block_size",
    "scope",
    "cache_types",
    "bytes_per_request",
    "method",
    "draft_length",
    "verify_width",
    "accepted_length",
    "max_draft_length",
    "runtime_path",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def feature(identity, state, **parameters):
    return {"id": identity, "state": state, "parameters": parameters, "observed": {"state": "unknown", "count": None}}


def field_maximum(model, name):
    """Read this installed revision's typed bound; never assume a universal limit."""
    field = type(model).model_fields.get(name)
    bounds = [m.le for m in field.metadata if getattr(m, "le", None) is not None] if field else []
    return min(bounds) if bounds else None


def speculation_settings(engine, architecture, dspark_architectures):
    scheme = engine.speculation
    if scheme is not None:
        return {
            "method": scheme.kind,
            "draft_length": scheme.num_speculative_tokens,
            "verify_width": scheme.num_speculative_tokens + 1,
            "accepted_length": getattr(scheme, "expected_accepted_tokens", None),
            "max_draft_length": field_maximum(scheme, "num_speculative_tokens"),
        }
    depth = engine.nextn
    if not depth:
        return {
            "method": "off",
            "draft_length": None,
            "verify_width": 1,
            "accepted_length": None,
            "max_draft_length": None,
        }
    # The evaluated SDK owns this architecture set (Kimi's legacy graph is DSPARK).
    if architecture is None or dspark_architectures is None:
        return None
    return {
        "method": "dspark" if architecture in dspark_architectures else "mtp",
        "draft_length": depth,
        "verify_width": depth + 1,
        "accepted_length": engine.nextn_accepted,
        "max_draft_length": field_maximum(engine, "nextn"),
    }


def routing_feature(spec):
    adapter = spec.adapters.get("dynamo")
    if adapter is not None:
        policy = adapter.config.get("router", {}).get("policy")
        return feature(
            "routing.kv_aware",
            "enabled" if policy == "kv_router" else "disabled" if policy == "round_robin" else "failed_to_record",
            runtime_path="dynamo_adapter",
        )
    return feature(
        "routing.kv_aware",
        "not_applicable" if spec.execution_mode == "offline" and not spec.adapters else "failed_to_record",
        runtime_path="native_replay" if spec.execution_mode == "offline" else "adapter",
    )


def build(parsed, spec, native, context, *, architecture, dspark_architectures):
    """Project defaults from the evaluated typed config and overrides from compilation.

    The full resolved config/spec are retained privately by the caller. Public
    records contain only the feature vocabulary and immutable provenance.
    """
    resolved = parsed.model_dump(mode="json")
    deployment = spec.backend_deployment
    record = {
        "schema": SCHEMA,
        **context,
        "execution_mode": spec.execution_mode,
        "configuration_sha256": digest(resolved),
        "roles": [],
    }
    observed_requests = native.get("telemetry", {}).get("requests", [])
    for role, attr in (
        ("aggregated", "agg_engine_args"),
        ("prefill", "prefill_engine_args"),
        ("decode", "decode_engine_args"),
    ):
        compiled = getattr(deployment, attr)
        if compiled is None:
            continue
        worker = resolved["engine"]["workers"][role]
        cache = worker["kv_cache"]
        prefix = compiled.get("enable_prefix_caching")
        block = compiled.get("block_size")
        backend = compiled.get("engine_type", deployment.backend)
        unit = compiled.get("prefix_match_unit", block) if backend in {"vllm", "sglang"} else None
        rows = [
            feature(
                "cache.g1_prefix",
                "enabled" if prefix is True else "disabled" if prefix is False else "failed_to_record",
            )
        ]
        rows.append(
            feature(
                "cache.sub_block_prefix_reuse",
                "not_applicable"
                if prefix is False
                else "failed_to_record"
                if unit is None or block is None
                else "enabled"
                if unit < block
                else "disabled",
                match_granularity_tokens=unit,
                block_size=block,
            )
        )
        for identity, field, compiled_field in (
            ("cache.g2", "host_offload", "native_host_offload"),
            ("cache.g3", "g3_offload", "g3_offload"),
        ):
            value = compiled.get(compiled_field)
            state = (
                "enabled"
                if value is not None
                else "disabled"
                if field in cache and cache[field] is None
                else "failed_to_record"
            )
            rows.append(feature(identity, state, scope=value.get("scope") if value else None))
        state = compiled.get("state_cache")
        rows.append(
            feature(
                "cache.hybrid_state",
                "enabled"
                if state is not None
                else "disabled"
                if "state_cache" in cache and cache["state_cache"] is None
                else "failed_to_record",
                cache_types=["recurrent_state"] if state else [],
                bytes_per_request=state.get("bytes_per_request") if state else None,
            )
        )
        speculation = speculation_settings(parsed.engine, architecture, dspark_architectures)
        rows.append(
            feature(
                "speculation",
                "failed_to_record"
                if speculation is None
                else "disabled"
                if speculation["method"] == "off"
                else "enabled",
                **(speculation or {}),
            )
        )
        rows.append(
            feature(
                "cache.kvcr",
                "enabled"
                if compiled.get("kvcr") is not None
                else "disabled"
                if "kvcr" in cache and cache["kvcr"] is None
                else "failed_to_record",
                runtime_path="native_replay",
            )
        )
        rows.append(routing_feature(spec))
        if role == "aggregated" and observed_requests:
            reuse = [r.get("first_admission_reused_input_tokens") for r in observed_requests]
            if prefix is True and all(type(n) is int for n in reuse):
                count = sum(n > 0 for n in reuse)
                rows[0]["observed"] = {"state": "exercised" if count else "not_observed", "count": count}
        record["roles"].append(
            {
                "role": role,
                "backend": backend,
                "backend_version": deployment.backend_version,
                "model": resolved["engine"]["model"],
                "compiled_sha256": digest(compiled),
                "features": rows,
            }
        )
    return validate(record)


def unavailable(context, run, state):
    return validate(
        {
            "schema": SCHEMA,
            **context,
            "execution_mode": "unavailable",
            "configuration_sha256": None,
            "roles": [
                {
                    "role": "aggregated",
                    "backend": run["backend"],
                    "backend_version": None,
                    "model": run["model"],
                    "compiled_sha256": None,
                    "features": [feature(key, state) for key in sorted(SILICON_FEATURES)],
                }
            ],
        }
    )


def compare_speculation(record, silicon, run, recipe=None):
    """Keep actual source values; normalize only the evidenced NEXTN chain convention."""
    measured = next(row for row in silicon["features"] if row["id"] == "speculation")
    params = {p["name"]: p["value"] for p in measured["parameters"]}
    actual, method = params.get("draft_length"), params.get("method")
    proposals, width, convention = actual, actual + 1 if type(actual) is int else None, "draft_proposals"
    if run["backend"] == "sglang" and method == "mtp" and params.get("algorithm") == "NEXTN" and recipe:
        args = recipe["base"]["roles"]["agg"]["args"] | recipe.get(f"override_c{run['lanes']}", {}).get(
            "roles", {}
        ).get("agg", {}).get("args", {})
        if (
            type(actual) is int
            and actual > 1
            and args.get("speculative-eagle-topk") == 1
            and args.get("speculative-num-draft-tokens") == actual
            and args.get("speculative-num-steps") == actual - 1
        ):
            proposals, width, convention = actual - 1, actual, "verification_width_including_seed"
    comparisons = []
    for role in record["roles"]:
        configured = next(row for row in role["features"] if row["id"] == "speculation")
        values = configured["parameters"]
        maximum = values.get("max_draft_length")
        status = "unknown"
        if method == "off" and configured["state"] == "disabled":
            status = "matched"
        elif method == "off" and configured["state"] == "enabled":
            status = "mismatch"
        elif (
            method not in {None, "unknown", "off"}
            and type(proposals) is int
            and configured["state"] in {"enabled", "disabled"}
        ):
            status = (
                "unsupported"
                if method == values.get("method") and maximum is not None and proposals > maximum
                else (
                    "matched"
                    if method == values.get("method") and proposals == values.get("draft_length")
                    else "mismatch"
                )
            )
        comparisons.append(
            {
                "role": role["role"],
                "status": status,
                "actual_method": method,
                "actual_draft_length": actual,
                "actual_proposals": proposals,
                "actual_verify_width": width,
                "actual_convention": convention,
                "aisim_method": values.get("method"),
                "aisim_draft_length": values.get("draft_length"),
                "aisim_max": maximum,
            }
        )
    record["speculation_comparison"] = comparisons
    return validate(record)


def validate(value, benchmark_id=None):
    fields = {
        "schema",
        "benchmark_id",
        "evaluated_revision",
        "dataset_revision",
        "execution_mode",
        "configuration_sha256",
        "roles",
    }
    if not isinstance(value, dict) or set(value) - {"speculation_comparison"} != fields or value["schema"] != SCHEMA:
        raise ValueError("Invalid simulation feature record")
    if not re.fullmatch(r"[a-z0-9-]+", value["benchmark_id"]) or (
        benchmark_id is not None and value["benchmark_id"] != benchmark_id
    ):
        raise ValueError("Simulation features belong to another benchmark")
    for key in ("evaluated_revision", "dataset_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", value[key]):
            raise ValueError("Simulation features require pinned revisions")
    if value["execution_mode"] not in {"offline", "online", "unavailable"}:
        raise ValueError("Invalid execution mode")
    if not value["roles"]:
        raise ValueError("Simulation features require role context")
    seen = set()
    for role in value["roles"]:
        if (
            set(role) != {"role", "backend", "backend_version", "model", "compiled_sha256", "features"}
            or role["role"] not in {"aggregated", "prefill", "decode"}
            or role["role"] in seen
        ):
            raise ValueError("Invalid simulation feature role")
        seen.add(role["role"])
        if role["backend"] not in {"vllm", "sglang", "trtllm"} or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", role["model"]
        ):
            raise ValueError("Invalid simulation model/backend identity")
        if role["backend_version"] is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", role["backend_version"]):
            raise ValueError("Invalid simulation backend version")
        for checksum in (value["configuration_sha256"], role["compiled_sha256"]):
            if checksum is not None and not re.fullmatch(r"[a-f0-9]{64}", checksum):
                raise ValueError("Invalid configuration checksum")
        if len(role["features"]) != 8 or {row["id"] for row in role["features"]} != SILICON_FEATURES:
            raise ValueError("Incomplete simulation feature inventory")
        for row in role["features"]:
            if (
                set(row) != {"id", "state", "parameters", "observed"}
                or row["state"] not in STATES
                or set(row["parameters"]) - PARAMETERS
            ):
                raise ValueError("Invalid simulation feature settings")
            for name, scalar in row["parameters"].items():
                if name == "cache_types" and scalar in ([], ["recurrent_state"]):
                    continue
                if scalar is not None and type(scalar) not in {str, int, float, bool}:
                    raise ValueError("Invalid simulation feature parameter")
                if name == "method" and scalar not in {
                    None,
                    "off",
                    "mtp",
                    "dspark",
                    "eagle",
                    "eagle3",
                    "dflash",
                    "dflash2",
                    "ngram",
                    "draft_model",
                    "unknown",
                }:
                    raise ValueError("Unapproved simulation method")
                if name == "scope" and scalar not in {None, "dp_rank_local", "cluster_shared", "worker_local"}:
                    raise ValueError("Unapproved simulation offload scope")
                if name == "runtime_path" and scalar not in {"native_replay", "adapter", "dynamo_adapter"}:
                    raise ValueError("Unapproved simulation runtime path")
                if isinstance(scalar, str) and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", scalar):
                    raise ValueError("Unsafe simulation feature parameter")
                if type(scalar) in {int, float} and (not math.isfinite(scalar) or scalar < 0):
                    raise ValueError("Invalid simulation feature number")
            observed = row["observed"]
            if (
                set(observed) != {"state", "count"}
                or observed["state"] not in {"exercised", "not_observed", "unknown"}
                or (observed["count"] is not None and (type(observed["count"]) is not int or observed["count"] < 0))
            ):
                raise ValueError("Invalid simulation feature usage")
            if (
                (observed["state"] == "unknown") != (observed["count"] is None)
                or (observed["state"] == "exercised" and not observed["count"])
                or (observed["state"] == "not_observed" and observed["count"] != 0)
            ):
                raise ValueError("Simulation usage count disagrees with its state")
    if "speculation_comparison" in value and (
        len(value["speculation_comparison"]) != len(seen)
        or {c["role"] for c in value["speculation_comparison"]} != seen
    ):
        raise ValueError("Incomplete speculative role comparisons")
    for comparison in value.get("speculation_comparison", []):
        if set(comparison) != {
            "role",
            "status",
            "actual_method",
            "actual_draft_length",
            "actual_proposals",
            "actual_verify_width",
            "actual_convention",
            "aisim_method",
            "aisim_draft_length",
            "aisim_max",
        }:
            raise ValueError("Invalid speculative comparison")
        if (
            comparison["role"] not in seen
            or comparison["status"] not in {"unknown", "matched", "mismatch", "unsupported"}
            or comparison["actual_convention"] not in {"draft_proposals", "verification_width_including_seed"}
        ):
            raise ValueError("Invalid speculative parity state")
        for key in ("actual_method", "aisim_method"):
            if comparison[key] is not None and not re.fullmatch(r"[a-z0-9_-]{1,80}", comparison[key]):
                raise ValueError("Unsafe speculative method")
        for key in (
            "actual_draft_length",
            "actual_proposals",
            "actual_verify_width",
            "aisim_draft_length",
            "aisim_max",
        ):
            if comparison[key] is not None and (type(comparison[key]) is not int or comparison[key] < 0):
                raise ValueError("Invalid speculative token count")
    return value
