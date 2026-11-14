# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read portable silicon bundles and build private, reproducible CI inputs."""

import hashlib
import json
import math
import re

from scripts.e2e_accuracy.benchmark_features import SILICON_FEATURES, validate

CORPORA = "data/e2e/agentx/corpus.json"


def safe_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError("Unsafe AgentX artifact path")
    path = root / name
    current = root
    for part in name.split("/"):
        current = current / part
        if current.is_symlink():
            raise ValueError("Unsafe AgentX artifact path")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unsafe AgentX artifact path")
    return path


def checked(root, entry):
    if set(entry) != {"path", "sha256", "bytes"} or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError("Invalid AgentX artifact checksum")
    path = safe_path(root, entry["path"])
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if type(entry["bytes"]) is not int or path.stat().st_size != entry["bytes"] or checksum != entry["sha256"]:
        raise ValueError("AgentX input checksum mismatch")
    return path


def document(root, run, role):
    return json.loads(checked(root, run["artifacts"][role]).read_bytes())


def load_runs(root, catalog):
    runs, identities = [], set()
    for entry in catalog["runs"]:
        path = checked(root, entry["manifest"])
        run = json.loads(path.read_bytes())
        if run.get("schema") != "aisimulate.agentx.benchmark.v1" or any(
            run.get(k) != entry[k] for k in ("id", "category", "configuration_id")
        ):
            raise ValueError("AgentX manifest identity mismatch")
        run["artifacts"] = {
            role: {**value, "path": safe_path(path.parent, value["path"]).relative_to(root).as_posix()}
            for role, value in run["artifacts"].items()
        }
        features = document(root, run, "feature_metadata")
        validate(features, run["id"])
        if {r["id"] for r in features["features"]} != SILICON_FEATURES:
            raise ValueError("HF requires the curated eight-feature inventory")
        configuration = document(root, run, "configuration")
        if configuration.get("schema") != "aisimulate.agentx.silicon-config.v1" or configuration["identity"] != {
            k: run.get(k)
            for k in ("model", "hardware", "backend", "measured_backend_version", "precision", "tensor_parallel")
        }:
            raise ValueError("Silicon configuration identity mismatch")
        workload = document(root, run, "workload")
        if (
            workload.get("schema") != "aisimulate.agentx.workload.v1"
            or workload["benchmark_id"] != run["id"]
            or workload["corpus_id"] != run["corpus_id"]
        ):
            raise ValueError("Silicon workload identity mismatch")
        if run.get("role") == "profile" and run.get("profile_seconds", 0) > 0:
            run.setdefault("boundary", "successful_profile_span")
            run.setdefault("cache_state", "recorded")
            run.setdefault("initial_input_tokens", None)
            run.setdefault("repeat", 0)
            run.setdefault("benchmark_url", None)
            run.setdefault("source_run_url", None)
            effective = configuration.get("effective", [])
            dtype = effective[0].get("model", {}).get("dtype") if effective else None
            run.setdefault("precision", {"torch.bfloat16": "BF16", "bfloat16": "BF16"}.get(dtype))
        if not run.get("precision"):
            raise ValueError("Measured precision is unavailable")
        identity = tuple(
            run[k]
            for k in (
                "category",
                "model",
                "hardware",
                "backend",
                "measured_backend_version",
                "lanes",
                "initial_input_tokens",
                "cache_state",
                "repeat",
            )
        )
        if (
            identity in identities
            or run["boundary"] not in {"successful_profile_span", "complete_finite_cohort"}
            or type(run["lanes"]) is not int
            or run["lanes"] < 1
        ):
            raise ValueError("Duplicate or invalid AgentX workload identity")
        identities.add(identity)
        if {"trace", "config", "predictions", "simulation_features"} & run["artifacts"].keys():
            raise ValueError("Simulation inputs do not belong in silicon bundles")
        observations = document(root, run, "observations")
        if (
            type(run["measured_requests"]) is not int
            or sum(r.get("phase") == "profile" for r in observations) != run["measured_requests"]
        ):
            raise ValueError("Measured cohort count mismatch")
        references = [ref for row in features["features"] for ref in row["evidence"]]
        references += configuration.get("requested", []) + configuration.get("effective_evidence", [])
        by_hash = {a["sha256"]: a for a in run["artifacts"].values()}
        documents = {}
        for ref in references:
            if ref["sha256"] not in by_hash:
                raise ValueError("Unresolved silicon evidence reference")
            if ref["sha256"] not in documents:
                documents[ref["sha256"]] = json.loads(checked(root, by_hash[ref["sha256"]]).read_bytes())
            value = documents[ref["sha256"]]
            try:
                for part in ref["pointer"].split("/")[1:]:
                    part = part.replace("~1", "/").replace("~0", "~")
                    value = value[int(part)] if isinstance(value, list) else value[part]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise ValueError("Unresolved silicon evidence pointer") from exc
        runs.append(run)
    return runs


def peak_context(value):
    if isinstance(value, dict):
        own = value.get("in", 0) + value.get("out", 0) if "in" in value and "out" in value else 0
        return max([own, *(peak_context(v) for v in value.values())])
    if isinstance(value, list):
        return max((peak_context(v) for v in value), default=0)
    return 0


def reconstruct(descriptor, source, target):
    if set(descriptor) != {
        "repository",
        "revision",
        "filename",
        "sha256",
        "derived_sha256",
        "transformations",
    } or not re.fullmatch(r"[0-9a-f]{40}", descriptor["revision"]):
        raise ValueError("Unpinned AgentX corpus")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != descriptor["sha256"]:
            raise ValueError("Upstream corpus checksum mismatch")
    steps = descriptor["transformations"]
    if len(steps) > 1:
        raise ValueError("Unsupported corpus transformation")
    step = steps[0] if steps else None
    if step and (step["operation"] != "retain_complete_traces" or step["serialization"] != "original_jsonl_lines"):
        raise ValueError("Unsupported corpus transformation")
    included, excluded = [], []
    with source.open("rb") as incoming, target.open("wb") as outgoing:
        for line in incoming:
            row = json.loads(line)
            if step and peak_context(row) > step["maximum_tokens"]:
                excluded.append(row["id"])
            else:
                included.append(row["id"])
                outgoing.write(line)
    with target.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != descriptor["derived_sha256"] or (
        step and (included != step["eligible"] or excluded != step["excluded"])
    ):
        target.unlink()
        raise ValueError("Derived corpus differs from the reviewed workload")


def finite_trace(root, run, target):
    # Kernel warmup is deliberately outside the finite replay measurement.
    with checked(root, run["artifacts"]["raw"]).open() as stream:
        rows = [row for line in stream if (row := json.loads(line))["phase"] in {"warmup", "measured"}]
    rows.sort(key=lambda r: r["request_id"])
    tokens = {}
    for field in ("input_token_ids", "output_token_ids"):
        for row in rows:
            for token in row[field]:
                tokens.setdefault(token, len(tokens))
    ids = {r["request_id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("Duplicate finite request")
    warm = {r["request_id"] for r in rows if r["phase"] == "warmup"}
    with target.open("w") as stream:
        stream.write(
            json.dumps({"schema": "dynamo.agentic_mooncake", "version": 2, "block_size": 1, "hash_id_scope": "local"})
            + "\n"
        )
        for row in rows:
            parents = set(row.get("dependency_ids", []))
            if row.get("parent_request_id"):
                parents.add(row["parent_request_id"])
            if row["phase"] == "measured" and row["turn_index"] == 0:
                parents.update(warm)
            if not parents <= ids or row["request_id"] in parents:
                raise ValueError("Invalid finite dependency")
            value = {
                "request_id": row["request_id"],
                "play_id": row["cell_id"],
                "session_id": f"{row['cell_id']}/lane-{row['lane_id']}",
                "model": run["model"],
                "input_length": len(row["input_token_ids"]),
                "output_length": len(row["output_token_ids"]),
                "hash_ids": row["input_token_ids"],
                "output_token_ids": [tokens[t] for t in row["output_token_ids"]],
                "not_before_ms": 0,
                "dependencies": [
                    {
                        "request_id": p,
                        "trigger": "completion",
                        "delay_ms": row.get("think_time_ms", 0),
                        "relation": "sequence",
                    }
                    for p in sorted(parents)
                ],
            }
            stream.write(json.dumps(value) + "\n")


def derive_manual_geometry(worker):
    """Accept only the inspected, equal-page, 15:45 baseline; fail on drift."""
    if worker["vllm_version"].split("+")[0] != "0.31.0":
        raise ValueError("cache mapping requires the inspected vLLM 0.31.0 runtime")
    if worker["speculative_config"] is not None or worker["kv_transfer_config"] is not None:
        raise ValueError("baseline requires speculation and KV transfer/offload disabled")
    parallel = worker["parallelism"]
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if parallel[name] != 1:
            raise ValueError(f"baseline requires {name}=1")
    scheduler = worker["scheduler"]
    if scheduler["async_scheduling"] is not False or scheduler["enable_chunked_prefill"] is not True:
        raise ValueError("baseline requires synchronous scheduling with chunked prefill")
    if scheduler["disable_hybrid_kv_cache_manager"]:
        raise ValueError("hybrid cache manager must be enabled")
    cache = worker["cache_config"]
    block_size = cache["block_size"]
    if type(block_size) is not int or block_size < 2:
        raise ValueError("resolved physical block size is missing")
    if (
        cache["mamba_cache_mode"] != "align"
        or not cache["enable_prefix_caching"]
        or cache["prefix_match_unit"] not in (None, block_size)
        or cache["use_replayssm"]
    ):
        raise ValueError("baseline requires ordinary align state caching and full-block prefix matching")
    if worker["prefix_cache_retention_interval"] != 0:
        raise ValueError("baseline requires vLLM semantic-checkpoint retention=0")
    groups = worker["groups"]
    if len(groups) != 4 or sorted(group["spec_type"] for group in groups) != [
        "FullAttentionSpec",
        "MambaSpec",
        "MambaSpec",
        "MambaSpec",
    ]:
        raise ValueError("expected one full-attention group and three GDN state groups")
    if any(len(group["layer_names"]) != 15 for group in groups):
        raise ValueError("expected four groups of 15 layers without grouping padding")
    names = [name for group in groups for name in group["layer_names"]]
    if len(set(names)) != 60:
        raise ValueError("cache groups must partition the 60 target layers")
    pages = {group["page_size_bytes"] for group in groups}
    if len(pages) != 1 or next(iter(pages)) <= 0:
        raise ValueError("all cache groups must share one positive physical page size")
    for group in groups:
        spec = group["spec"]
        if spec["block_size"] != block_size:
            raise ValueError("token and state groups must use the same block size")
        if group["spec_type"] == "MambaSpec" and (
            spec["mamba_cache_mode"] != "align"
            or spec.get("num_speculative_blocks", 0) != 0
            or spec.get("num_prefill_checkpoint_blocks", 0) != 0
        ):
            raise ValueError("unexpected state scratch/checkpoint allocation")
    state_specs = [group["spec"] for group in groups if group["spec_type"] == "MambaSpec"]
    if any(spec != state_specs[0] for spec in state_specs[1:]):
        raise ValueError("all three state groups must have identical specs")
    pool_block_bytes = next(iter(pages)) * 15
    blocks = worker["num_blocks"]
    if type(blocks) is not int or blocks < 6 or pool_block_bytes % block_size:
        raise ValueError("invalid shared block pool geometry")
    # v0.31 tensor descriptors alias one backing allocation across groups.
    # Each descriptor's size is the complete backing size, not an additive size.
    tensors = worker["tensors"]
    if len(tensors) != 4:
        raise ValueError("expected one aliased tensor descriptor per cache group")
    for group, tensor in zip(groups, tensors, strict=True):
        if (
            tensor["size"] != blocks * pool_block_bytes
            or tensor["offset"] != 0
            or tensor.get("host_resident", False)
            or set(tensor["layers"]) != set(group["layer_names"])
        ):
            raise ValueError("tensor allocation disagrees with the shared-pool mapping")
        end = 14 * tensor["layer_stride"] + (blocks - 1) * tensor["block_stride"] + group["page_size_bytes"]
        if end != tensor["size"]:
            raise ValueError("unexpected cache tensor padding or strides")
    return {
        "block_size": block_size,
        "prefix_match_unit": block_size,
        "bytes_per_token": pool_block_bytes // block_size,
        "prefix_caching": True,
        "capacity": {"type": "fixed", "blocks": blocks - 1},
        "state_cache": {"bytes_per_request": 3 * pool_block_bytes},
    }


def manual_geometry(workers):
    if not workers:
        raise ValueError("probe returned no workers")
    mapping = derive_manual_geometry(workers[0])
    tensor_parallel = workers[0]["parallelism"]["tensor_parallel_size"]
    if (
        type(tensor_parallel) is not int
        or tensor_parallel < 1
        or any(type(worker["rank"]) is not int for worker in workers)
        or len(workers) != tensor_parallel
        or {worker["rank"] for worker in workers} != set(range(tensor_parallel))
    ):
        raise ValueError("probe must contain every tensor-parallel rank exactly once")
    if any(derive_manual_geometry(worker) != mapping for worker in workers[1:]):
        raise ValueError("rank cache geometry differs; a uniform AISim worker cannot represent it")
    for key in ("parallelism", "scheduler", "cache_config", "model"):
        if any(worker[key] != workers[0][key] for worker in workers[1:]):
            raise ValueError(f"rank configuration differs: {key}")
    return mapping


def measured_acceptance(point, depth):
    """Read a point's retained acceptance, bound to its verification depth."""
    if type(depth) is not int or depth < 0:
        raise ValueError("Invalid recorded verification depth")
    if depth == 0:
        return 0
    receipt = point.get("metrics", {}).get("speculative_acceptance")
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"verification_depth", "accepted_draft_tokens"}
        or type(receipt["verification_depth"]) is not int
        or receipt["verification_depth"] != depth
    ):
        raise ValueError("Missing depth-matched measured speculative acceptance")
    accepted = receipt["accepted_draft_tokens"]
    if type(accepted) not in {int, float} or not math.isfinite(accepted) or not 0 <= accepted <= depth:
        raise ValueError("Invalid measured speculative acceptance")
    return accepted


def replay_config(root, run):
    """Simulation policy lives in CI; measured settings come from verified receipts."""
    if (run["backend"], run["model"], run["hardware"], run["tensor_parallel"], run["precision"]) not in {
        ("sglang", "nvidia/GLM-5.3-NVFP4", "B300", 8, "NVFP4"),
        ("vllm", "Qwen/Qwen3.5-397B-A17B", "B300", 8, "BF16"),
    }:
        raise ValueError("No reviewed replay policy for this measured configuration")
    config = {
        "engine": {
            "mode": "aggregated",
            "backend": run["backend"],
            "backend_version": run["measured_backend_version"],
            "model": run["model"],
            "hardware": "b300_sxm",
            "systems_paths": ["default"],
            "database_mode": "SILICON",
            "estimation_mode": "op_level",
            "fallback_policy": "deny",
            "estimator_config": {"correction": {"enabled": False}},
        },
        "traffic": {"source": {"type": "trace", "paths": ["traces.jsonl"]}, "load": {"type": "trace_timestamps"}},
    }
    engine, traffic = config["engine"], config["traffic"]
    parallel = {
        "tensor": run["tensor_parallel"],
        "moe_tensor": run["tensor_parallel"],
        "moe_expert": 1,
        "attention_data": 1,
        "replicas": 1,
    }
    worker = {"parallelism": parallel, "timing": {"type": "default"}}
    engine["workers"] = {"aggregated": worker}
    if run["backend"] == "vllm":
        receipt = document(root, run, "cache_receipt" if "cache_receipt" in run["artifacts"] else "runtime_receipt")
        worker["kv_cache"] = manual_geometry(receipt["workers"])
        observed = receipt["workers"][0]
        engine.update(
            context_length=observed["model"]["max_model_len"],
            nextn=0,
            attention_backend="flashinfer",
            comm_quant_mode="half",
            enable_chunked_prefill=True,
            enable_eplb=False,
            enable_shared_layer=True,
            moe_kernel_source="vllm_unquantizedfusedmoe_flashinfer_trtllm_trtllmbf16expertsmonolithic",
        )
        engine.update(
            dict.fromkeys(("fmha_quant_mode", "gemm_quant_mode", "kvcache_quant_mode", "moe_quant_mode"), "bfloat16")
        )
        scheduler = observed["scheduler"]
        worker["scheduler"] = {
            "max_sequences": scheduler["max_num_seqs"],
            "max_batched_tokens": scheduler["max_num_batched_tokens"],
            "async_scheduling": scheduler["async_scheduling"],
        }
        parallel["pipeline"] = observed["parallelism"]["pipeline_parallel_size"]
        traffic["source"].update(format="agentic_mooncake", block_size=1)
        traffic["load"]["speedup"] = 1
    elif run["backend"] == "sglang":
        recipe = document(root, run, "recipe")
        args = recipe["base"]["roles"]["agg"]["args"] | recipe.get(f"override_c{run['lanes']}", {}).get(
            "roles", {}
        ).get("agg", {}).get("args", {})
        point = document(root, run, "feature_point_receipt")
        block = 64
        worker["kv_cache"] = {
            "block_size": block,
            "prefix_caching": True,
            "capacity": {
                "type": "fixed",
                "blocks": point["metrics"]["kv_cache_pool_tokens"] // block // run["tensor_parallel"],
            },
        }
        worker["scheduler"] = {
            "max_batched_tokens": args["chunked-prefill-size"],
            "max_sequences": args["max-running-requests"],
        }
        # Recorded verification depth and measured acceptance remain distinct from configured draft length.
        depth = args["speculative-num-steps"]
        engine.update(
            context_length=args["context-length"],
            nextn=depth,
            nextn_accepted=measured_acceptance(point, depth),
        )
        source = document(root, run, "source_manifest")["initial_selection"]
        snapshot = {
            "seed": source["seed"],
            "corpus_cursor": source["corpus_cursor"],
            "initial_selections": [
                {k: s[k] for k in ("source_play_id", "t_star_ms")} for s in source.get("initial_selections", [source])
            ],
        }
        namespaces = document(root, run, "workload")["silicon_reference"]["namespace_map"]
        for selection in snapshot["initial_selections"]:
            identity = selection["source_play_id"]
            if identity in namespaces:
                selection["source_play_id"] = f"{namespaces[identity]}:play:{identity}"
        if len(snapshot["initial_selections"]) != run["lanes"]:
            raise ValueError("Missing recorded lane selections")
        phase = document(root, run, "feature_harness_receipt")["input_config"]["phases"][0]
        traffic["source"].update(format="weka", block_size=64, nested_timestamp_basis="absolute")
        traffic["load"].update(
            agentic_lanes=run["lanes"],
            agentic_warmup=True,
            agentic_warmup_policy="live_trajectory_v1",
            agentic_snapshot=snapshot,
            agentic_profile={
                "duration_seconds": phase["duration"],
                "global_idle_cap_seconds": phase["system_idle_gap_cap_seconds"],
                "tree_idle_cap_seconds": 300,
                "response_grace_seconds": phase["grace_period"],
                "cancel_drain_seconds": 10,
            },
        )
    else:
        raise ValueError("No reviewed replay policy for this backend")
    if run["backend"] == "vllm" and run["boundary"] == "successful_profile_span":
        workload = document(root, run, "workload")
        namespaces = workload["silicon_reference"]["namespace_map"]
        registry = json.loads((root / CORPORA).read_bytes())
        descriptor = registry["sources"][run["corpus_id"]]
        steps = descriptor["transformations"]
        order = steps[0]["eligible"] if steps else list(namespaces)
        selections, positions = {}, {}
        for cut in document(root, run, "snapshot_receipt"):
            lane = cut["lane_index"]
            if lane is not None and lane not in selections:
                identity = cut["root_id"]
                positions[lane] = order.index(identity)
                selections[lane] = {
                    "source_play_id": f"{namespaces[identity]}:play:{identity}",
                    "t_star_ms": cut["snapshot"]["t_star_ms"],
                }
        if set(selections) != set(range(run["lanes"])) or list(positions.values()) != sorted(set(positions.values())):
            raise ValueError("Missing or unordered initial lane selections")
        harness = document(root, run, "harness_receipt")
        phase = harness["phases"][0]
        traffic["source"].update(format="weka", block_size=64, nested_timestamp_basis="absolute")
        traffic["load"] = {
            "type": "trace_timestamps",
            "agentic_lanes": run["lanes"],
            "agentic_warmup": True,
            "agentic_warmup_policy": "live_trajectory_v1",
            "agentic_snapshot": {
                "seed": harness["datasets"][0]["random_seed"],
                "corpus_cursor": max(positions.values()) + 1,
                "initial_selections": [selections[i] for i in range(run["lanes"])],
            },
            "agentic_profile": {
                "duration_seconds": phase["duration"],
                "global_idle_cap_seconds": phase["system_idle_gap_cap_seconds"],
                "tree_idle_cap_seconds": 300,
                "response_grace_seconds": phase["grace_period"],
                "cancel_drain_seconds": 10,
            },
        }
    config["prediction_database_version"] = {"vllm": "0.31.0", "sglang": "0.5.21"}[run["backend"]]
    return config
