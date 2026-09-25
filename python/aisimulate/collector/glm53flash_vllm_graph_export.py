# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recheck V2 FULL hidden-state captures and separately executed native logits.

Original readers of frozen native observations. Source/config dispatch precedes
requests; no holdout geometry supplies the prediction policy or missing costs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from collector.glm53flash_contract import build_model_manifest, canonical_json, sha256_json
from collector.glm53flash_graph_callbacks import QUALIFIED_CUPTI_SHA256, resolve_registry
from collector.glm53flash_graph_nodes import bind_replay_kernels, bind_vllm_execution_activity, trace_forward_identity
from collector.glm53flash_jsonl import file_sha256, iter_records
from collector.glm53flash_vllm_graph_ops import LOGITS_SOURCE_PIN
from collector.glm53flash_vllm_graph_ops import SOURCE_PINS as CAPTURE_PINS
from collector.glm53flash_vllm_graph_policy import build_full_policy, select_descriptor, validate_snapshot

BOUNDARY = "native_metadata_to_logits_gpu_v1"


def execution_policy(root):
    """Retain the native public EngineArgs separately from actual dispatch.

    Dynamo dumps parsed native EngineArgs before construction. The initialized
    per-worker graph snapshots and native state inventory independently verify
    the executed paths. Only the native random seed is excluded from equality.
    """
    raw = json.loads((root / "resolved-config-node0.json").read_bytes())
    args = raw.get("config", {}).get("engine_args")
    if not isinstance(args, dict) or not args or args.get("enforce_eager") is not False:
        raise ValueError("native FULL evidence lacks original public EngineArgs")
    normalized = {key: value for key, value in args.items() if key != "seed"}
    return {
        "execution_policy": {
            "normalization": "native_vllm_engine_args_except_seed_v1",
            "sha256": sha256_json(normalized),
        },
        "_execution_policy": normalized,
    }


def same_execution_policy(left, right):
    expected = "native_vllm_engine_args_except_seed_v1"
    for value in (left, right):
        policy = value.get("execution_policy", {})
        original = value.get("_execution_policy")
        if (
            policy.get("normalization") != expected
            or not isinstance(original, dict)
            or "seed" in original
            or policy.get("sha256") != sha256_json(original)
        ):
            raise ValueError("native V2 graph configuration comparison lacks its original EngineArgs")
    if left["_execution_policy"] != right["_execution_policy"]:
        raise ValueError("native V2 calibration/control/holdout changed public EngineArgs")


def check_dispatch(row, snapshot):
    """Verify every completed seed/target, including real NONE and PW seeds."""
    batch = row["batch_size"]
    queries, prefixes = row["query_lengths"], row["prefix_lengths"]
    if (
        type(batch) is not int
        or batch < 1
        or len(queries) != batch
        or len(prefixes) != batch
        or len(set(queries)) != 1
        or len(set(prefixes)) != 1
        or row["phase"] not in ("context", "generation")
    ):
        raise ValueError("native V2 graph forward lacks homogeneous actual coordinates")
    selected = select_descriptor(snapshot, batch=batch, query=queries[0], is_context=row["phase"] == "context")
    dispatch = {
        "descriptor": selected,
        "policy_sha256": sha256_json(snapshot),
        "physical_tokens": selected["num_tokens"],
        "physical_requests": selected["num_reqs"] or batch,
    }
    if (
        row.get("native_dispatch") != dispatch
        or row.get("runtime_mode") != selected["cg_mode"]
        or row.get("used_cuda_graph") is not (selected["cg_mode"] != "NONE")
        or type(row.get("num_padded_tokens")) is not int
        or row["num_padded_tokens"] != selected["num_tokens"]
        or (selected["cg_mode"] == "FULL" and row.get("native_graph_replay_completed") is not True)
    ):
        raise ValueError("native V2 actual dispatch/padding/completion differs from initialized source policy")
    if row.get("stage") == "measure" and (row["phase"], selected["cg_mode"]) != ("generation", "FULL"):
        raise ValueError("native FULL calibration cannot admit PW/eager or context targets")
    return selected


def _captures(root, rank, snapshot, manifest, provenance, files):
    from collector.glm53flash_graph_export import _completed_boundaries, _local, _receipt

    paths = sorted(root.glob(f"vllm-capture-rank-{rank}-*.json"))
    captures = {}
    source_ids, executable_ids, executable_handles = set(), set(), set()
    operations = [row for row in manifest["phases"]["generation"] if row["name"] != "logits"]
    if len(operations) != len(manifest["phases"]["generation"]) - 1:
        raise ValueError("native V2 manifest lacks exactly one external logits operation")
    for path in paths:
        if not re.fullmatch(rf"vllm-capture-rank-{rank}-[0-9]+\.json", path.name):
            raise ValueError("native FULL capture has an unknown descriptor artifact")
        recorded = json.loads(_local(root, path.name).read_bytes())
        files.add(path.name)
        receipt = recorded["instantiation_receipt"]
        if not re.fullmatch(rf"vllm-graph-clones-rank-{rank}-capture-[0-9]+-[0-9]+\.json", receipt["file"]):
            raise ValueError("native FULL instantiation belongs to another worker/capture")
        source_path = _local(root, receipt["file"][:-5] + "-source.json")
        files.add(source_path.name)
        source = json.loads(source_path.read_bytes())
        callbacks = _receipt(root, receipt, files)
        if callbacks.get("graph_mutations") is not False:
            raise ValueError("native FULL instantiation is not a read-only observation")
        key = canonical_json(source["native_shape_key"])
        if (
            key in captures
            or source["native_shape_key"] not in snapshot["full_graphs"]
            or source.get("tp_rank") != rank
            or source.get("provenance") != provenance
            or source.get("capture_scope") != "vllm_hidden_states"
            or source.get("uncaptured_operations") != ["logits"]
            or source.get("graph_mutations") is not False
            or source.get("operations") != operations
            or source.get("physical_padded_tokens") != source["native_shape_key"]["num_tokens"]
        ):
            raise ValueError("native V2 capture ownership differs from its actual descriptor/model/source")
        libraries = source.get("native_api_libraries", {})
        runtime = libraries.get("cudart", {})
        if (
            libraries.get("cupti", {}).get("sha256") != QUALIFIED_CUPTI_SHA256
            or type(runtime.get("runtime_version")) is not int
            or runtime["runtime_version"] // 1000 != 13
            or runtime.get("abi") != "CUDA13_capture7_edges5"
            or not re.fullmatch(r"[0-9a-f]{64}", runtime.get("sha256", ""))
            or any(not Path(libraries[name].get("path", "")).is_absolute() for name in ("cudart", "cupti"))
        ):
            raise ValueError("native V2 capture lacks actual qualified CUDA/CUPTI provider identity")
        _completed_boundaries(source, operations)
        type_receipt = recorded.get("node_type_receipt")
        type_proof = _receipt(root, type_receipt, files) if type_receipt is not None else None
        derived = resolve_registry(source, callbacks, type_proof, allow_pending_memcpy=True, allow_memset_query=True)
        derived["instantiation_receipt"] = receipt
        if type_receipt is not None:
            derived["node_type_receipt"] = type_receipt
        if derived != recorded:
            raise ValueError("native V2 executable differs from original capture/clone/query evidence")
        # All native FULL graph objects stay alive in this manager. Their
        # executable IDs/handles and CUPTI source graph IDs cannot be reused by
        # another descriptor. Source graph *handles* may be recycled after
        # instantiation, so do not infer identity from those pointer values.
        create = next(
            row
            for row in callbacks["callbacks"]
            if row["kind"] == "graph_exec_created" and row["graph_exec_id"] == callbacks["actual_graph_exec_id"]
        )
        identities = (source["graph_id"], callbacks["actual_graph_exec_id"], create["raw_fields"]["graphExec"])
        for identity, seen in zip(identities, (source_ids, executable_ids, executable_handles), strict=True):
            if type(identity) is not int or identity <= 0 or identity in seen:
                raise ValueError("native V2 FULL descriptors reuse a source or live executable identity")
            seen.add(identity)
        captures[key] = (derived, {"file": path.name, "sha256": file_sha256(path)})
    if set(captures) != {canonical_json(value) for value in snapshot["full_graphs"]}:
        raise ValueError("native V2 capture omits an actual initialized FULL descriptor")
    inventory = _local(root, f"vllm-graph-inventory-rank-{rank}.json")
    files.add(inventory.name)
    if json.loads(inventory.read_bytes()).get("source_pins") != CAPTURE_PINS:
        raise ValueError("native V2 capture manager/offloader source differs")
    return captures


def _binding(root, row, registry, files):
    from collector.glm53flash_graph_export import _receipt

    recorded = row["replay_nodes"]
    name = f"graph-profile-rank-{row['tp_rank']}-forward-{row['invocation']}.json"
    if recorded["trace_file"] != name or name in files:
        raise ValueError("native V2 trace must uniquely belong to this rank/forward")
    trace = _receipt(root, {"file": name, "sha256": recorded["trace_sha256"]}, files)
    if trace.get("aisim_native_forward") != trace_forward_identity(row) or trace.get("aisim_native_execution") != {
        "backend": "vllm",
        "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
        "logits_source_sha256": LOGITS_SOURCE_PIN,
        "failed": False,
    }:
        raise ValueError("native V2 trace belongs to another forward or source/logits boundary")
    launches = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "cuda_runtime" and event.get("name") == "cudaGraphLaunch"
    ]
    if len(launches) != 1:
        raise ValueError("native V2 FULL trace lacks one complete graph launch")
    nodes = bind_replay_kernels(registry, trace["traceEvents"], correlation=launches[0]["args"]["correlation"])
    binding = bind_vllm_execution_activity(nodes, trace["traceEvents"])
    binding.update(trace_file=name, trace_sha256=recorded["trace_sha256"])
    if binding != recorded:
        raise ValueError("native V2 recorded costs differ from original device activity")
    return binding


def read_vllm_run(root: Path, run: dict) -> dict:
    """Join actual V2 dispatch, unique traces and completed native request history."""
    from collector.glm53flash_graph_export import _compact_units, _local
    from collector.glm53flash_validation import _required_files, expected_runtime_version

    backend, fmt, tp, phase = run["key"]
    if (backend, phase) != ("vllm", "decode") or run["role"] not in ("calibration", "holdout", "control"):
        raise ValueError("V2 FULL evidence requires a declared decode calibration/control/holdout")
    calibrated = run["role"] == "calibration"
    version = expected_runtime_version(run)
    manifest = json.loads(_local(root, "manifest.json").read_bytes())
    if (
        manifest != build_model_manifest(backend, fmt, tp, version)
        or len(manifest["runtime_operations"]["generation"]) != 1
    ):
        raise ValueError("native V2 manifest must bind its complete production graph and single setup marker")
    provenance = json.loads(_local(root, "provenance.json").read_bytes())
    entries = manifest["phases"]["generation"] + manifest["runtime_operations"]["generation"]
    files = _required_files(tp, backend) - {f"rank-{rank}.jsonl" for rank in range(tp)}
    files.add("resolved-config-node0.json")
    snapshots, registries, state_hashes, forwards = {}, {}, {}, {}
    for rank in range(tp):
        path = _local(root, f"vllm-graph-policy-rank-{rank}.json")
        files.add(path.name)
        snapshot = validate_snapshot(json.loads(path.read_bytes()))
        if snapshot["tp_rank"] != rank or snapshot["tp_size"] != tp or snapshot["backend_version"] != version:
            raise ValueError("native V2 graph snapshot differs from frozen worker/runtime")
        snapshots[rank] = snapshot
        layout = _local(root, f"state-layout-rank-{rank}.json")
        files.add(layout.name)
        state_hashes[rank] = file_sha256(layout)
        captures = _captures(root, rank, snapshot, manifest, provenance, files) if calibrated else {}
        registries[rank] = (
            [captures[canonical_json(key)][1]["sha256"] for key in snapshot["full_graphs"]] if calibrated else []
        )
        forward_path = _local(root, f"forward-rank-{rank}.jsonl")
        files.add(forward_path.name)
        targets = {}
        for row in iter_records(forward_path):
            check_dispatch(row, snapshot)
            if row["stage"] == "measure":
                if row["invocation"] in targets:
                    raise ValueError("native V2 target invocation was reused")
                targets[row["invocation"]] = row
        if calibrated:
            graph_path = _local(root, f"graph-forward-rank-{rank}.jsonl")
            files.add(graph_path.name)
            records = iter_records(graph_path)
        else:
            records = iter(targets.values())
        observed = set()
        for row in records:
            forward = targets.get(row["invocation"])
            if (
                forward is None
                or row["invocation"] in observed
                or any(row.get(key) != value for key, value in forward.items())
            ):
                raise ValueError("native V2 graph activity differs from its completed target forward")
            observed.add(row["invocation"])
            shape = check_dispatch(row, snapshot)
            elapsed = row.get("whole_forward_gpu_ms")
            if (
                row.get("ops_instrumented") is not calibrated
                or row.get("whole_forward_boundary") != BOUNDARY
                or row.get("gpu_completed") is not True
                or any(row.get(key) != value for key, value in provenance.items())
                or row.get("native_graph_replay_completed") is not True
                or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed)
                or elapsed <= 0
            ):
                raise ValueError("native V2 FULL observation lacks its complete source/timing/dispatch identity")
            binding = None
            if calibrated:
                registry, artifact = captures[canonical_json(shape)]
                if (
                    row.get("capture_registry_file") != artifact["file"]
                    or row.get("capture_registry_sha256") != artifact["sha256"]
                    or row.get("logits_source_sha256") != LOGITS_SOURCE_PIN
                    or row.get("profiled") is not True
                    or row.get("measurement_method") != "native_cupti_graph_nodes_and_external_logits"
                ):
                    raise ValueError("native V2 measured replay lacks its actual capture and external logits")
                binding = _compact_units(_binding(root, row, registry, files), entries)
            elif (
                any(
                    row.get(key) is not None
                    for key in ("replay_nodes", "capture_registry_sha256", "capture_registry_file")
                )
                or row.get("profiled", False) is not False
            ):
                raise ValueError("independent native V2 truth contains operation profiling")
            key = row["benchmark_id"], row["repetition"]
            by_rank = forwards.setdefault(key, {})
            if rank in by_rank:
                raise ValueError("native V2 graph point repetition was observed twice")
            by_rank[rank] = {
                key: row[key]
                for key in (
                    "forward_id",
                    "invocation",
                    "request_ids",
                    "benchmark_id",
                    "repetition",
                    "sampling_role",
                    "dataset_role",
                    "request_set",
                    "corpus_sha256",
                    "batch_size",
                    "prefix_lengths",
                    "query_lengths",
                    "num_padded_tokens",
                    "whole_forward_gpu_ms",
                )
            }
            by_rank[rank]["binding"] = binding
            by_rank[rank]["token_witness"] = [
                {
                    "input_sha256": request["input_tokens_sha256"],
                    "prompt_sha256": sha256_json(request["prompt_token_ids"]),
                    "sampled_token_id": request["sampled_token_id"],
                }
                for request in row["requests"]
            ]
        if observed != targets.keys():
            raise ValueError("native V2 graph activity omits completed target forwards")
    comparable = [
        {key: value for key, value in snapshot.items() if key != "tp_rank"} for snapshot in snapshots.values()
    ]
    if any(value != comparable[0] for value in comparable[1:]):
        raise ValueError("native V2 TP workers used different graph dispatch policies")
    for group in forwards.values():
        if set(group) != set(range(tp)):
            raise ValueError("native V2 graph forward omits TP workers")
        first = group[0]
        fields = set(first) - {"binding", "whole_forward_gpu_ms", "forward_id", "invocation"}
        if any(any(row[key] != first[key] for key in fields) for row in group.values()):
            raise ValueError("native V2 graph workers do not join one actual forward")
    policy = (
        build_full_policy(
            snapshots,
            checkpoint_format=fmt,
            tp_size=tp,
            provenance=provenance,
            resolved_config_sha256=file_sha256(root / "resolved-config-node0.json"),
            state_layout_sha256=state_hashes,
            capture_registry_sha256=registries,
        )
        if calibrated
        else {"native_snapshot": comparable[0], "provenance": provenance}
    )
    return {
        **execution_policy(root),
        "policy": policy,
        "native_snapshot": comparable[0],
        "provenance": provenance,
        "forwards": forwards,
        "files": files,
        "manifest": manifest,
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
    }
