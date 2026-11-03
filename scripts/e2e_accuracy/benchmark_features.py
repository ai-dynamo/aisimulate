# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned silicon feature metadata; independent of prediction branch settings."""

import math
import re

SCHEMA = "aisimulate.benchmark.features.v1"
STATES = {"enabled", "disabled", "not_applicable", "unknown"}
USAGE = {"exercised", "not_observed", "unknown"}
PARAMETERS = {
    "size",
    "max_batched_tokens",
    "max_sequences",
    "block_size",
    "mamba_cache_mode",
    "state_dtype",
    "nodes",
    "prefill_size",
    "decode_size",
    "offload_size",
    "context_limit",
    "seed",
    "mode",
    "duration_seconds",
    "grace_seconds",
    "global_idle_cap_seconds",
    "requests_per_lane",
    "count_definition",
    "observation_seconds",
    "profile_seconds",
    "observed_requests",
    "unknown_requests",
    "phase_scope",
}
STRING_VALUES = {
    "phase_scope": {"profile", "preparation", "preparation_and_profile", "exported_harness_counters"},
    "mamba_cache_mode": {"align", "all", "none"},
    "state_dtype": {"float32", "bfloat16", "float16", "auto"},
    "mode": {"none", "first_turn_prefix", "all_turns_prefix"},
    "count_definition": {
        "distinct_observed_subagent_conversations",
        "distinct_observed_subagent_sessions",
        "requests_with_recorded_cached_tokens",
        "preemptions_in_retained_intervals",
        "dag_child_sessions_dispatched",
        "parents_resumed_after_branch_joins",
    },
}
FEATURES = {
    "agentx.weka": "Weka dependency graph",
    "agentx.seeded_snapshots": "Seeded snapshots",
    "agentx.live_warmup": "Live-trajectory warmup",
    "agentx.continuous_profile": "Continuous profile",
    "agentx.sequential_recycling": "Sequential lane recycling",
    "agentx.cache_bust": "Cache bust",
    "agentx.shuffle": "Random or shuffled recycling",
    "scheduling.continuous_batching": "Continuous batching",
    "scheduling.chunked_prefill": "Chunked prefill",
    "scheduling.async": "Asynchronous scheduling",
    "cache.g1_prefix": "GPU prefix caching",
    "cache.hybrid_state": "Hybrid KV and recurrent state",
    "cache.g2": "Host offload",
    "cache.g3": "Storage offload",
    "cache.kvcr": "Peer KV fetch",
    "topology.aggregated": "Aggregated serving",
    "topology.tensor_parallel": "Tensor parallelism",
    "topology.expert_parallel": "Expert parallelism",
    "topology.attention_dp": "Attention data parallelism",
    "topology.pipeline_parallel": "Pipeline parallelism",
    "topology.context_parallel": "Context parallelism",
    "topology.multi_node": "Multi-node serving",
    "topology.pd": "Prefill/decode disaggregation",
    "speculation": "Speculative decoding",
    "routing.planner": "KV-aware routing and Planner",
    "precision.quantized_weights": "Quantized weights",
    "precision.quantized_kv": "Quantized KV",
    "workload.multimodal": "Multimodal input",
    "events.subagent_spawn": "Subagent spawning",
    "events.join": "Dependency joins",
    "events.tool_wait": "Tool waits",
    "events.cache_reuse": "Cache reuse",
    "events.eviction": "Cache eviction",
    "events.preemption": "Request preemption",
    "events.recompute": "Recomputation",
    "events.state_restore": "Recurrent-state restoration",
}
QWEN_ENABLED = {
    "agentx.weka",
    "agentx.seeded_snapshots",
    "agentx.live_warmup",
    "agentx.continuous_profile",
    "agentx.sequential_recycling",
    "cache.g1_prefix",
    "cache.hybrid_state",
    "scheduling.continuous_batching",
    "scheduling.chunked_prefill",
    "topology.aggregated",
    "topology.tensor_parallel",
}
QWEN_DISABLED = set(FEATURES) - QWEN_ENABLED - {identity for identity in FEATURES if identity.startswith("events.")}


def validate_qwen_baseline(value):
    validate(value)
    indexed = {row["id"]: row for row in value["features"]}
    for identities, expected in ((QWEN_ENABLED, "enabled"), (QWEN_DISABLED, "disabled")):
        for identity in identities:
            if indexed[identity]["state"] != expected:
                raise ValueError(f"Qwen required feature changed or unresolved: {identity}")
    required = {
        "topology.tensor_parallel": {"size": 8},
        "scheduling.continuous_batching": {"max_sequences": 64},
        "scheduling.chunked_prefill": {"max_batched_tokens": 8192},
        "agentx.weka": {"context_limit": 262144, "seed": 42},
    }
    for identity, settings in required.items():
        parameters = {p["name"]: p["value"] for p in indexed[identity]["parameters"]}
        if any(parameters.get(name) != actual for name, actual in settings.items()):
            raise ValueError(f"Qwen required parameters changed: {identity}")
    return value


def validate(value, benchmark_id=None):
    if not isinstance(value, dict) or set(value) != {"schema", "benchmark_id", "taxonomy", "features"}:
        raise ValueError("Invalid feature metadata fields")
    if value["schema"] != SCHEMA or not re.fullmatch(r"[a-z0-9-]+", value["benchmark_id"]):
        raise ValueError("Invalid feature metadata identity")
    if benchmark_id is not None and value["benchmark_id"] != benchmark_id:
        raise ValueError("Feature metadata belongs to another benchmark")
    taxonomy = value["taxonomy"]
    if not isinstance(taxonomy, dict) or set(taxonomy) != {"revision", "sha256"}:
        raise ValueError("Invalid feature taxonomy")
    if not re.fullmatch(r"[a-f0-9]{40}", taxonomy["revision"]) or not re.fullmatch(r"[a-f0-9]{64}", taxonomy["sha256"]):
        raise ValueError("Feature taxonomy requires pinned revisions")
    seen = set()
    if not isinstance(value["features"], list):
        raise ValueError("Invalid feature inventory")
    for row in value["features"]:
        if not isinstance(row, dict) or set(row) != {"id", "state", "parameters", "observed", "evidence"}:
            raise ValueError("Invalid feature fields")
        identity = row["id"]
        if identity not in FEATURES or identity in seen or row["state"] not in STATES:
            raise ValueError("Unknown or duplicate feature")
        seen.add(identity)
        if not isinstance(row["parameters"], list):
            raise ValueError("Invalid feature parameters")
        names = set()
        for parameter in row["parameters"]:
            if not isinstance(parameter, dict) or set(parameter) != {"name", "value"}:
                raise ValueError("Invalid feature parameter fields")
            name, scalar = parameter["name"], parameter["value"]
            if not isinstance(name, str) or name not in PARAMETERS or name in names:
                raise ValueError("Invalid feature parameter name")
            names.add(name)
            if scalar is not None and type(scalar) not in {str, bool, int, float}:
                raise ValueError("Feature parameters must be scalar")
            if isinstance(scalar, str) and (len(scalar) > 200 or re.search(r"/|https?:|Bearer|hf_", scalar)):
                raise ValueError("Private or unsafe feature parameter")
            if isinstance(scalar, str) and scalar not in STRING_VALUES.get(name, set()):
                raise ValueError("Unapproved feature parameter value")
            if type(scalar) in {int, float} and (not math.isfinite(scalar) or scalar < 0):
                raise ValueError("Nonfinite feature parameter")
        observed = row["observed"]
        if not isinstance(observed, dict) or set(observed) != {"state", "count"} or observed["state"] not in USAGE:
            raise ValueError("Invalid observed feature usage")
        count = observed["count"]
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Invalid observed feature count")
        if (observed["state"] == "unknown") != (count is None):
            raise ValueError("Unknown usage must retain an unknown count")
        if observed["state"] == "exercised" and count == 0 or observed["state"] == "not_observed" and count != 0:
            raise ValueError("Observed feature count disagrees with state")
        if not isinstance(row["evidence"], list):
            raise ValueError("Invalid feature evidence")
        for evidence in row["evidence"]:
            if not isinstance(evidence, dict) or set(evidence) != {"sha256", "pointer"}:
                raise ValueError("Invalid feature evidence fields")
            if not re.fullmatch(r"[a-f0-9]{64}", evidence["sha256"]) or not re.fullmatch(
                r"/[A-Za-z0-9_~/.-]+", evidence["pointer"]
            ):
                raise ValueError("Invalid feature evidence reference")
            if re.search(r"/(?:Users|home|lustre|tmp|mnt)/", evidence["pointer"]):
                raise ValueError("Local paths are not JSON evidence pointers")
        if row["state"] in {"enabled", "disabled"} and not row["evidence"]:
            raise ValueError("Resolved feature states require evidence")
    if seen != set(FEATURES):
        raise ValueError("Incomplete feature inventory; retain unknowns explicitly")
    return value


def from_vllm_receipt(benchmark_id, receipt, receipt_sha256, taxonomy):
    """Resolve only observed server settings; absent settings remain unknown."""
    worker = receipt["workers"][0]
    result = {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "taxonomy": taxonomy,
        "features": [
            {
                "id": identity,
                "state": "unknown",
                "parameters": [],
                "observed": {"state": "unknown", "count": None},
                "evidence": [],
            }
            for identity in FEATURES
        ],
    }
    indexed = {row["id"]: row for row in result["features"]}

    def setting(identity, value, pointer, parameters=None):
        if type(value) is not bool:
            return
        indexed[identity].update(
            state="enabled" if value else "disabled",
            parameters=[{"name": key, "value": val} for key, val in (parameters or {}).items()],
            evidence=[{"sha256": receipt_sha256, "pointer": "/workers/0/" + pointer}],
        )

    parallel, scheduler, cache = (worker.get(k, {}) for k in ("parallelism", "scheduler", "cache_config"))
    for identity, field in [
        ("topology.tensor_parallel", "tensor_parallel_size"),
        ("topology.pipeline_parallel", "pipeline_parallel_size"),
        ("topology.attention_dp", "data_parallel_size"),
        ("topology.context_parallel", "decode_context_parallel_size"),
    ]:
        value = parallel.get(field)
        setting(identity, value > 1 if type(value) is int else None, "parallelism/" + field, {"size": value})
    setting("topology.expert_parallel", parallel.get("enable_expert_parallel"), "parallelism/enable_expert_parallel")
    setting(
        "scheduling.chunked_prefill",
        scheduler.get("enable_chunked_prefill"),
        "scheduler/enable_chunked_prefill",
        {"max_batched_tokens": scheduler.get("max_num_batched_tokens")},
    )
    setting("scheduling.async", scheduler.get("async_scheduling"), "scheduler/async_scheduling")
    sequences = scheduler.get("max_num_seqs")
    setting(
        "scheduling.continuous_batching",
        sequences > 1 if type(sequences) is int else None,
        "scheduler/max_num_seqs",
        {"max_sequences": sequences},
    )
    setting(
        "cache.g1_prefix",
        cache.get("enable_prefix_caching"),
        "cache_config/enable_prefix_caching",
        {"block_size": cache.get("block_size")},
    )
    groups = worker.get("groups")
    if isinstance(groups, list):
        setting(
            "cache.hybrid_state",
            any(g["spec_type"] == "MambaSpec" for g in groups),
            "groups",
            {"mamba_cache_mode": cache.get("mamba_cache_mode"), "state_dtype": cache.get("mamba_ssm_cache_dtype")},
        )
    if "speculative_config" in worker:
        setting("speculation", worker["speculative_config"] is not None, "speculative_config")
    if "quantization" in worker.get("model", {}):
        setting("precision.quantized_weights", worker["model"]["quantization"] is not None, "model/quantization")
    attention = [g for g in groups or [] if g["spec_type"] == "FullAttentionSpec"]
    if attention and all("kv_quant_mode" in g["spec"] for g in attention):
        setting("precision.quantized_kv", any(g["spec"]["kv_quant_mode"] != 0 for g in attention), "groups")
    resolved = worker.get("resolved_vllm_config", {})
    topology = resolved.get("parallel_config", {})
    if type(topology.get("nnodes")) is int:
        setting(
            "topology.multi_node",
            topology["nnodes"] > 1,
            "resolved_vllm_config/parallel_config/nnodes",
            {"nodes": topology["nnodes"]},
        )
    cp = [topology.get(name) for name in ("prefill_context_parallel_size", "decode_context_parallel_size")]
    if all(type(size) is int for size in cp):
        setting(
            "topology.context_parallel",
            max(cp) > 1,
            "resolved_vllm_config/parallel_config",
            {"prefill_size": cp[0], "decode_size": cp[1]},
        )
    if "kv_transfer_config" in resolved:
        transfer = resolved["kv_transfer_config"]
        # No connector means no disaggregated serving, peer retrieval, or external cache tiers.
        if transfer is None:
            for identity in ("topology.pd", "cache.kvcr", "cache.g3"):
                setting(identity, False, "resolved_vllm_config/kv_transfer_config")
            setting("topology.aggregated", True, "resolved_vllm_config/kv_transfer_config")
            size = resolved.get("cache_config", {}).get("kv_offloading_size", "unknown")
            if size is None or type(size) in (int, float):
                setting(
                    "cache.g2",
                    size is not None and size > 0,
                    "resolved_vllm_config/cache_config/kv_offloading_size",
                    {"offload_size": size},
                )
    multimodal = resolved.get("model_config", {}).get("multimodal_config")
    if isinstance(multimodal, dict) and type(multimodal.get("language_model_only")) is bool:
        setting(
            "workload.multimodal",
            not multimodal["language_model_only"],
            "resolved_vllm_config/model_config/multimodal_config/language_model_only",
        )
    return validate(result, benchmark_id)


def add_agentx_receipt(value, configuration, configuration_sha256, cuts, cuts_sha256, requests, requests_sha256):
    """Add actual harness settings and narrowly defined observed event counts."""
    rows = {row["id"]: row for row in value["features"]}

    def setting(identity, enabled, pointer, parameters=None, checksum=configuration_sha256):
        if type(enabled) is not bool:
            return
        rows[identity].update(
            state="enabled" if enabled else "disabled",
            parameters=[{"name": name, "value": scalar} for name, scalar in (parameters or {}).items()],
            evidence=[{"sha256": checksum, "pointer": pointer}],
        )

    datasets = configuration.get("datasets", [])
    phases = configuration.get("phases", [])
    if len(datasets) != 1 or len(phases) != 1:
        raise ValueError("Expected one resolved Weka dataset and one profiling phase")
    dataset, phase = datasets[0], phases[0]
    setting(
        "agentx.weka",
        dataset.get("dataset") == "semianalysis_cc_traces_weka_062126",
        "/datasets/0/dataset",
        {"context_limit": dataset.get("max_context_length"), "seed": dataset.get("random_seed")},
    )
    if dataset.get("sampling") in {"sequential", "random", "shuffle"}:
        sampling = dataset["sampling"]
        setting("agentx.sequential_recycling", sampling == "sequential", "/datasets/0/sampling")
        setting("agentx.shuffle", sampling != "sequential", "/datasets/0/sampling")
    target = dataset.get("cache_bust", {}).get("target")
    if isinstance(target, str):
        setting("agentx.cache_bust", target != "none", "/datasets/0/cache_bust/target", {"mode": target})
    if phase.get("timing_mode") == "agentic_replay":
        duration = phase.get("duration")
        setting(
            "agentx.continuous_profile",
            type(duration) in (int, float) and duration > 0,
            "/phases/0",
            {
                "duration_seconds": duration,
                "grace_seconds": phase.get("grace_period"),
                "global_idle_cap_seconds": phase.get("system_idle_gap_cap_seconds"),
            },
        )
        budget = phase.get("warmup_requests_per_lane")
        if type(budget) is int:
            setting(
                "agentx.live_warmup", budget > 0, "/phases/0/warmup_requests_per_lane", {"requests_per_lane": budget}
            )
    if cuts and all(isinstance(cut.get("snapshot"), dict) for cut in cuts):
        setting("agentx.seeded_snapshots", True, "/0/snapshot", {"seed": dataset.get("random_seed")}, cuts_sha256)
    endpoint = configuration.get("endpoint", {})
    # This client connects directly to the local serving endpoint; no routing service is involved.
    urls = endpoint.get("urls", [])
    url = urls[0] if len(urls) == 1 else None
    if isinstance(url, str) and re.fullmatch(r"http://(?:127\.0\.0\.1|localhost):8800/?", url):
        setting("routing.planner", False, "/endpoint/urls/0")

    def observed(identity, count, definition):
        rows[identity]["observed"] = {"state": "exercised" if count else "not_observed", "count": count}
        rows[identity]["parameters"].append({"name": "count_definition", "value": definition})
        rows[identity]["parameters"].append({"name": "phase_scope", "value": "preparation_and_profile"})
        rows[identity]["evidence"].append({"sha256": requests_sha256, "pointer": "/requests"})
        if count and rows[identity]["state"] == "unknown":
            rows[identity]["state"] = "enabled"

    known = [row for row in requests if type(row.get("agent_depth")) is int and row.get("session_id")]
    if known:
        observed(
            "events.subagent_spawn",
            len({row["session_id"] for row in known if row["agent_depth"] > 0}),
            "distinct_observed_subagent_sessions",
        )
        rows["events.subagent_spawn"]["parameters"].extend(
            [
                {"name": "observed_requests", "value": len(known)},
                {"name": "unknown_requests", "value": len(requests) - len(known)},
            ]
        )
    known = [row for row in requests if row.get("cached_tokens") is not None]
    if known:
        observed(
            "events.cache_reuse",
            sum(row["cached_tokens"] > 0 for row in known),
            "requests_with_recorded_cached_tokens",
        )
        rows["events.cache_reuse"]["parameters"].extend(
            [
                {"name": "observed_requests", "value": len(known)},
                {"name": "unknown_requests", "value": len(requests) - len(known)},
            ]
        )
    return validate(value, value["benchmark_id"])


def add_preemption_receipt(value, receipt, checksum):
    """Count observed counter increments; retain the exact covered interval scope."""
    intervals = receipt["intervals"]
    if not intervals:
        return value
    count = sum(row["preemptions"] for row in intervals)
    if type(count) is not int or count < 0:
        raise ValueError("Invalid recorded preemption count")
    row = next(row for row in value["features"] if row["id"] == "events.preemption")
    row.update(
        state="enabled" if count else "unknown",
        observed={"state": "exercised" if count else "not_observed", "count": count},
        parameters=[
            {"name": "phase_scope", "value": "profile"},
            {"name": "count_definition", "value": "preemptions_in_retained_intervals"},
            {
                "name": "observation_seconds",
                "value": sum((int(r["end_ns"]) - int(r["start_ns"])) / 1e9 for r in intervals),
            },
            {
                "name": "profile_seconds",
                "value": (int(receipt["profile_end_ns"]) - int(receipt["profile_start_ns"])) / 1e9,
            },
        ],
        evidence=[{"sha256": checksum, "pointer": "/intervals"}],
    )
    return validate(value)


def add_branch_receipt(value, summary, checksum):
    """Use the harness's actual dispatch/join counters, preserving their exported scope."""
    counters = summary.get("branch_stats")
    if not isinstance(counters, dict):
        return value
    rows = {row["id"]: row for row in value["features"]}
    for identity, key, definition in (
        ("events.subagent_spawn", "children_spawned", "dag_child_sessions_dispatched"),
        ("events.join", "parents_resumed", "parents_resumed_after_branch_joins"),
    ):
        count = counters.get(key)
        if key not in counters:
            continue
        if type(count) is not int or count < 0:
            raise ValueError("Invalid harness branch counter")
        rows[identity].update(
            state="enabled" if count else "unknown",
            observed={"state": "exercised" if count else "not_observed", "count": count},
            parameters=[
                {"name": "count_definition", "value": definition},
                {"name": "phase_scope", "value": "exported_harness_counters"},
            ],
            evidence=[{"sha256": checksum, "pointer": "/branch_stats/" + key}],
        )
    return validate(value)
