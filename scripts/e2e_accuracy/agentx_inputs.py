# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read portable silicon bundles and build private, reproducible CI inputs."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

from scripts.e2e_accuracy.benchmark_features import SILICON_FEATURES, validate

CORPORA = "data/e2e/agentx/corpus.json"


def safe_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError("Unsafe AgentX artifact path")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [path, *path.parents]):
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
        if {"trace", "config", "predictions"} & run["artifacts"].keys():
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
        # Reuse the existing physical-geometry qualification, including every rank.
        probe_path = Path(__file__).resolve().parents[2] / "benchmarks/evidence/accuracy/qwen-weka/cache_probe.py"
        spec = importlib.util.spec_from_file_location("agentx_cache_probe", probe_path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        receipt = document(root, run, "cache_receipt" if "cache_receipt" in run["artifacts"] else "runtime_receipt")
        worker["kv_cache"] = probe.make_receipt(receipt["workers"])["manual_kv_cache"]
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
        engine.update(context_length=args["context-length"], nextn=args["speculative-num-steps"], nextn_accepted=1.99)
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
