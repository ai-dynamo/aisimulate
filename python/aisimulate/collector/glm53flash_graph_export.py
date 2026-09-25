# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce native graph measurements from original capture and CUPTI evidence.

No whole-forward residual enters a unit cost. Native activity unions remain an
additive approximation whose accuracy requires independent whole-forward data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from collector.glm53flash_contract import (
    WHOLE_FORWARD_RANK,
    build_model_manifest,
    canonical_json,
    sha256_json,
)
from collector.glm53flash_graph_callbacks import QUALIFIED_CUPTI_SHA256, resolve_registry
from collector.glm53flash_graph_nodes import bind_execution_activity, bind_replay_kernels, trace_forward_identity
from collector.glm53flash_graph_policy import SOURCE_PINS, build_policy, padded_batch, validate_snapshot
from collector.glm53flash_jsonl import file_sha256, iter_records
from collector.glm53flash_native_hooks import SGLANG_PROJECTION_SOURCE_PINS

BASENAME = "glm53flash_graph_perf.parquet"
BOUNDARY = "native_full_graph_metadata_to_logits_gpu_v1"
KEYS = ("component", "geometry", "batch_size", "prefix", "padded_batch_size")
SCOPE = "disjoint_native_node_activity_union_v1"
NAMED_CONTRACT = "graph_named_operations_v1"


def _semantic_sha(value):
    # Capture records use the original writer's spaced sorted JSON convention.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _local(root, name):
    if not isinstance(name, str):
        raise ValueError("graph evidence path must be a string")
    path = root / name
    if path.name != name or path.resolve().parent != root.resolve():
        raise ValueError("graph evidence must reference a local, nonsymlinked file")
    if path.is_symlink() or not path.is_file():
        raise ValueError("graph evidence file is absent or aliased")
    return path


def _receipt(root, value, files):
    path = _local(root, value["file"])
    if file_sha256(path) != value.get("sha256"):
        raise ValueError("native graph evidence hash changed")
    files.add(path.name)
    return json.loads(path.read_bytes())


def _completed_boundaries(source, operations):
    expected = {row["name"] for row in operations}
    calls = source.get("calls", [])
    if len(calls) != len(expected) or {row.get("name") for row in calls} != expected:
        raise ValueError("native capture omits or duplicates physical operation calls")
    owners = defaultdict(list)
    for node in source["nodes"]:
        if node["name"] not in expected | {"native_graph_setup"}:
            raise ValueError("native graph node has an unknown physical owner")
        owners[node["name"]].append(node["node_id"])
    for call in calls:
        if (
            call.get("completed") is not True
            or not call.get("source")
            or call.get("owned_node_ids") != sorted(owners[call["name"]])
        ):
            raise ValueError("zero/positive graph cost requires a completed call with its exact node difference")


def _captures(root, rank, snapshot, manifest, provenance, files):
    source_path = _local(root, f"capture-source-nodes-rank-{rank}.jsonl")
    target_path = _local(root, f"capture-nodes-rank-{rank}.jsonl")
    files.update((source_path.name, target_path.name))
    # Worker installation receives the complete native driver's run identity,
    # then the hook records its independently verified lazy-projection sources.
    # Keep that exact execution binding separate from the stable input identity
    # used to compare independently launched calibration/control processes.
    native_provenance = json.loads(_local(root, "sglang-provenance.json").read_bytes())
    if any(native_provenance.get(key) != value for key, value in provenance.items()):
        raise ValueError("native graph capture runtime differs from its frozen input provenance")
    capture_provenance = {
        **native_provenance,
        "native_projection_source_sha256": SGLANG_PROJECTION_SOURCE_PINS,
    }
    sources = {}
    for source in iter_records(source_path):
        key = canonical_json(source["native_shape_key"])
        if key in sources:
            raise ValueError("native capture replaced a previously recorded descriptor")
        if (
            source.get("tp_rank") != rank
            or source.get("provenance") != capture_provenance
            or source.get("capture_scope") != "model_with_logits"
            or source.get("uncaptured_operations") != []
            or source.get("graph_mutations") is not False
            or source.get("operations") != manifest["phases"]["generation"]
            or source.get("physical_padded_tokens") != source["native_shape_key"]["size"]
        ):
            raise ValueError("native capture ownership differs from the actual model/runtime identity")
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
            raise ValueError("native graph capture lacks its qualified CUDA/CUPTI library identity")
        _completed_boundaries(source, manifest["phases"]["generation"])
        sources[key] = source
    captures = {}
    for recorded in iter_records(target_path):
        key = canonical_json(recorded["native_shape_key"])
        if key in captures or key not in sources:
            raise ValueError("executable graph does not uniquely identify an original capture")
        receipt = recorded["instantiation_receipt"]
        callbacks = _receipt(root, receipt, files)
        if callbacks.get("graph_mutations") is not False:
            raise ValueError("native graph clone receipt does not establish read-only observation")
        node_type_receipt = recorded.get("node_type_receipt")
        node_type_proof = _receipt(root, node_type_receipt, files) if node_type_receipt is not None else None
        derived = resolve_registry(sources[key], callbacks, node_type_proof, allow_pending_memcpy=True)
        derived["instantiation_receipt"] = receipt
        if node_type_receipt is not None:
            derived["node_type_receipt"] = node_type_receipt
        if derived != recorded:
            raise ValueError("executable graph differs from original native clone callbacks")
        captures[key] = derived
    if set(captures) != set(sources) or set(captures) != {canonical_json(key) for key in snapshot["captured_keys"]}:
        raise ValueError("graph registry omits an actual initialized capture bucket")
    return captures


def _binding(root, row, registry, files):
    recorded = row["replay_nodes"]
    name = f"graph-profile-rank-{row['tp_rank']}-forward-{row['invocation']}.json"
    if recorded["trace_file"] != name or name in files:
        raise ValueError("native graph trace must uniquely belong to its actual rank/invocation")
    trace = _receipt(root, {"file": recorded["trace_file"], "sha256": recorded["trace_sha256"]}, files)
    if trace.get("aisim_native_forward") != trace_forward_identity(row):
        raise ValueError("native graph trace belongs to another run/rank/forward or sampling role")
    launches = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "cuda_runtime" and event.get("name") == "cudaGraphLaunch"
    ]
    if len(launches) != 1:
        raise ValueError("graph profile has ambiguous actual launch identity")
    nodes = bind_replay_kernels(registry, trace["traceEvents"], correlation=launches[0]["args"]["correlation"])
    binding = bind_execution_activity(nodes, trace["traceEvents"])
    binding.update(trace_sha256=recorded["trace_sha256"], trace_file=recorded["trace_file"])
    if binding != recorded:
        raise ValueError("recorded graph costs differ from actual CUPTI activity")
    return binding


def _compact_units(binding, entries):
    """Keep timing/fingerprint scalars; full activity intervals remain on disk."""
    units = {row["operation"]: row for row in binding["operation_activity_unions"]}
    if not units.keys() <= {entry["name"] for entry in entries}:
        raise ValueError("CUPTI activity contains undeclared units")
    compact = {}
    for entry in entries:
        activities = [row for row in binding["activities"] if row["operation"] == entry["name"]]
        fingerprints = sorted(
            ({"activity": row["activity"], "fingerprint": row["fingerprint"]} for row in activities),
            key=canonical_json,
        )
        compact[entry["name"]] = {
            "latency": units.get(entry["name"], {}).get("active_union_us", 0.0) / 1000,
            "dispatch": sha256_json(fingerprints),
            "activity_count": len(activities),
        }
    return compact


def read_graph_run(root: Path, run: dict) -> dict:
    """Validate graph-specific evidence; the common loader verifies native state."""
    backend, fmt, tp, phase = run["key"]
    if backend == "vllm":
        from collector.glm53flash_vllm_graph_export import read_vllm_run

        return read_vllm_run(root, run)
    if (backend, phase) != ("sglang", "decode") or run["role"] not in ("calibration", "holdout", "control"):
        raise ValueError("graph evidence supports ordinary SGLang FULL decode only")
    calibrated = run["role"] == "calibration"
    from collector.glm53flash_validation import (
        _required_files,
        check_sglang_forward_allocator,
        sglang_allocator_evidence,
    )

    files = _required_files(tp, backend) - {f"rank-{rank}.jsonl" for rank in range(tp)}
    allocator = sglang_allocator_evidence(root, run, files)
    manifest = json.loads(_local(root, "manifest.json").read_bytes())
    production = build_model_manifest(backend, fmt, tp)
    if manifest != production or len(manifest["runtime_operations"]["generation"]) != 1:
        raise ValueError("native graph manifest must contain the complete production graph and one setup marker")
    provenance = json.loads(_local(root, "provenance.json").read_bytes())
    entries = manifest["phases"]["generation"] + manifest["runtime_operations"]["generation"]
    snapshots, registries, state_hashes, captures_by_rank, forwards = {}, {}, {}, {}, {}
    for rank in range(tp):
        path = _local(root, f"graph-policy-rank-{rank}.json")
        files.add(path.name)
        snapshot = validate_snapshot(json.loads(path.read_bytes()))
        if snapshot["tp_rank"] != rank:
            raise ValueError("native graph policy belongs to a different rank")
        snapshots[rank] = snapshot
        layout_path = _local(root, f"state-layout-rank-{rank}.json")
        files.add(layout_path.name)
        state_hashes[rank] = file_sha256(layout_path)
        captures = _captures(root, rank, snapshot, manifest, provenance, files) if calibrated else {}
        captures_by_rank[rank] = captures
        registries[rank] = (
            [_semantic_sha(captures[canonical_json(key)]) for key in snapshot["captured_keys"]] if calibrated else []
        )
        targets = {}
        forward_path = _local(root, f"forward-rank-{rank}.jsonl")
        graph_path = _local(root, f"graph-forward-rank-{rank}.jsonl")
        files.update((forward_path.name, graph_path.name))
        for row in iter_records(forward_path):
            check_sglang_forward_allocator(row, rank, allocator)
            if row["stage"] == "measure":
                if row["invocation"] in targets:
                    raise ValueError("native target invocation was reused")
                targets[row["invocation"]] = row
        observed = set()
        for row in iter_records(graph_path):
            forward = targets.get(row["invocation"])
            if (
                forward is None
                or row["invocation"] in observed
                or any(
                    row.get(key) != value
                    for key, value in forward.items()
                    if key not in ("graph_ops_profiled", "graph_ops_formal_admission")
                )
            ):
                raise ValueError("graph activity belongs to another completed native forward")
            observed.add(row["invocation"])
            shape = row["native_shape_key"]
            elapsed = row.get("whole_forward_gpu_ms")
            if (
                row.get("profiled") is not calibrated
                or row.get("ops_instrumented") is not calibrated
                or row.get("whole_forward_boundary") != BOUNDARY
                or row.get("native_execute_source_sha256")
                != SOURCE_PINS["srt/model_executor/runner/decode_cuda_graph_runner.py"]
                or row.get("gpu_completed") is not True
                or any(row.get(key) != value for key, value in provenance.items())
                or row.get("used_cuda_graph") is not True
                or row.get("runtime_mode") != "FULL"
                or row.get("phase") != "generation"
                or row.get("query_lengths") != [1] * row["batch_size"]
                or len(set(row["prefix_lengths"])) != 1
                or shape not in snapshot["captured_keys"]
                or shape["size"] != padded_batch(snapshot, row["batch_size"])
                or row.get("num_padded_tokens") != shape["size"]
                or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed)
                or elapsed <= 0
            ):
                raise ValueError("graph observation differs from actual source-bound dispatch/whole-forward policy")
            policy_receipt = row["native_dispatch_policy_receipt"]
            if policy_receipt["file"] != path.name or _receipt(root, policy_receipt, files) != snapshot:
                raise ValueError("graph observation did not use its actual initialized native policy")
            binding = None
            if calibrated:
                registry = captures[canonical_json(shape)]
                if (
                    row.get("capture_registry_sha256") != _semantic_sha(registry)
                    or row.get("capture_registry_file") != f"capture-nodes-rank-{rank}.jsonl"
                ):
                    raise ValueError("native replay points to another captured graph")
                binding = _compact_units(_binding(root, row, registry, files), entries)
            elif row.get("replay_nodes") is not None or row.get("capture_registry_sha256") is not None:
                raise ValueError("independent native graph truth contains module profiling")
            key = row["benchmark_id"], row["repetition"]
            by_rank = forwards.setdefault(key, {})
            if rank in by_rank:
                raise ValueError("native graph point repetition was observed twice")
            # Token/state payloads are checked by the common native loader and
            # remain on disk. Keep only compact join data and measured units.
            by_rank[rank] = {
                k: row[k]
                for k in (
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
            raise ValueError("native graph records omit completed target forwards")
    comparable = [{k: v for k, v in value.items() if k != "tp_rank"} for value in snapshots.values()]
    if any(value != comparable[0] for value in comparable[1:]):
        raise ValueError("TP ranks used different graph dispatch policies")
    for group in forwards.values():
        if set(group) != set(range(tp)):
            raise ValueError("native graph forward is missing TP ranks")
        first = group[0]
        fields = set(first) - {"binding", "whole_forward_gpu_ms", "forward_id", "invocation"}
        if any(any(row[key] != first[key] for key in fields) for row in group.values()):
            raise ValueError("graph ranks cannot be joined into one actual native forward")
    policy = (
        build_policy(
            snapshots,
            checkpoint_format=fmt,
            tp_size=tp,
            provenance=provenance,
            resolved_config_sha256=file_sha256(root / "sglang-resolved-config.json"),
            state_layout_sha256=state_hashes,
            capture_registry_sha256=registries,
        )
        if calibrated
        else {"native_snapshot": comparable[0], "provenance": provenance}
    )
    from collector.fpm_forward.glm53flash_validation import _sglang_execution_policy

    result = {
        **_sglang_execution_policy(
            json.loads(_local(root, "sglang-resolved-config.json").read_bytes()), allocator["normalized"]
        ),
        "policy": policy,
        "native_snapshot": comparable[0],
        "provenance": provenance,
        "forwards": forwards,
        "files": files,
        "manifest": manifest,
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
    }
    from collector.glm53flash_sglang_control import read_submission

    result["request_submission"] = read_submission(root, run, result, files)
    return result


def _named_contract(value):
    if value not in (None, NAMED_CONTRACT):
        raise ValueError("unknown native graph analysis lookup contract")
    return value == NAMED_CONTRACT


def _rank_selection(proof):
    """Reproduce the original whole-forward rank receipt independently of table keys."""
    selections = []
    for key, ranks in sorted(proof["forwards"].items()):
        selected = min(ranks, key=lambda rank: (-ranks[rank]["whole_forward_gpu_ms"], rank))
        selections.append(
            {
                "benchmark_id": key[0],
                "repetition": key[1],
                "selected_rank": selected,
                "ranks": [
                    {
                        "rank": rank,
                        "forward_id": row["forward_id"],
                        "invocation": row["invocation"],
                        "whole_forward_gpu_ms": row["whole_forward_gpu_ms"],
                    }
                    for rank, row in sorted(ranks.items())
                ],
            }
        )
    return {
        "schema": "glm53flash_graph_rank_selection_v1",
        "aggregation_policy": WHOLE_FORWARD_RANK,
        "forwards": selections,
    }


def aggregate_graph(proof, *, evidence_sha256, lookup_contract=None):
    """Select actual whole-forward slowest rank, then median physical samples."""
    named = _named_contract(lookup_contract)
    grouped = defaultdict(list)
    entries = proof["manifest"]["phases"]["generation"] + proof["manifest"]["runtime_operations"]["generation"]
    if named:
        policy = proof["policy"]
        expected = build_model_manifest(
            policy["backend"], policy["checkpoint_format"], policy["tp_size"], policy["backend_version"]
        )
        if proof["manifest"] != expected:
            raise ValueError("named graph analysis requires the complete original model manifest")
        repetitions = defaultdict(set)
        for (point, repetition), ranks in proof["forwards"].items():
            repetitions[point].add(repetition)
            for record in ranks.values():
                if set(record["binding"]) != {entry["name"] for entry in entries}:
                    raise ValueError("named graph forward lacks its complete physical unit inventory")
                if record["sampling_role"] != ("warmup" if repetition < 5 else "measurement"):
                    raise ValueError("named graph repetition differs from original 5+10 sampling roles")
        if not repetitions or any(ids != set(range(15)) for ids in repetitions.values()):
            raise ValueError("named graph analysis requires all original five warmups and ten measurements")
    for key, ranks in sorted(proof["forwards"].items()):
        selected = min(ranks, key=lambda rank: (-ranks[rank]["whole_forward_gpu_ms"], rank))
        record = ranks[selected]
        if record["sampling_role"] != "measurement":
            continue
        for entry in entries:
            unit = record["binding"][entry["name"]]
            identity = (
                entry["component"],
                entry["geometry"],
                record["batch_size"],
                record["prefix_lengths"][0],
                record["num_padded_tokens"],
            )
            if named:
                identity = (*identity, entry["name"])
            grouped[identity].append((key, unit["latency"], unit["dispatch"], unit["activity_count"]))
    selection = _rank_selection(proof)
    rows = []
    for identity, samples in sorted(grouped.items()):
        signatures = {(row[2], row[3]) for row in samples}
        repetitions = {row[0] for row in samples}
        if named and (len(samples) != 10 or len(repetitions) != 10):
            raise ValueError("named graph unit repeats or omits an original measurement")
        if len(signatures) != 1 or len(repetitions) < 10:
            raise ValueError("native graph physical key mixes dispatch identities or lacks ten repetitions")
        dispatch, count = signatures.pop()
        rows.append(
            {
                **dict(zip((*KEYS, "operation_name") if named else KEYS, identity, strict=True)),
                **({"graph_lookup_contract": NAMED_CONTRACT} if named else {}),
                "latency": statistics.median(row[1] for row in samples),
                "activity_count": count,
                "sample_count": len(repetitions),
                "dispatch_fingerprint": dispatch,
                "graph_policy": canonical_json(proof["policy"]),
                "graph_policy_sha256": sha256_json(proof["policy"]),
                "dataset_role": "calibration",
                "aggregation_policy": WHOLE_FORWARD_RANK,
                "rank_selection_sha256": sha256_json(selection),
                "evidence_sha256": evidence_sha256,
                "measurement_scope": SCOPE,
            }
        )
    return rows, selection


def verify_evidence(root, proof):
    receipt = json.loads(_local(root, "graph-calibration-evidence.json").read_bytes())
    if receipt.get("schema") != "glm53flash_graph_calibration_v1" or receipt.get("graph_policy_sha256") != sha256_json(
        proof["policy"]
    ):
        raise ValueError("graph calibration evidence has a different native policy")
    request_sets = {row["request_set"] for group in proof["forwards"].values() for row in group.values()}
    if request_sets != {receipt.get("request_set")} or any(
        receipt.get(key) != proof[key] for key in ("source_plan_sha256", "corpus_sha256")
    ):
        raise ValueError("graph calibration evidence differs from its frozen native run")
    files = receipt.get("files", {})
    if not proof["files"] <= files.keys() or any(
        file_sha256(_local(root, name)) != digest for name, digest in files.items()
    ):
        raise ValueError("graph calibration original evidence is incomplete or changed")
    selection = _rank_selection(proof)
    if json.loads(_local(root, "graph-rank-selection.json").read_bytes()) != selection:
        raise ValueError("graph rank selection differs from actual whole-forward intervals")
    control = json.loads(_local(root, "graph-profile-control.json").read_bytes())
    if "graph-profile-control.json" not in files or control.get("schema") != "glm53flash_graph_profile_control_v1":
        raise ValueError("graph calibration lacks its original unprofiled native control")
    control_root = Path(control["evidence_root"])
    current = profile_control(root, proof, control_root, control["frozen_run"])
    if current != control:
        raise ValueError("native graph profiler control differs from retained original evidence")
    return receipt


def _same_execution_policy(left, right, label):
    if left.get("execution_policy", {}).get("normalization") == "native_vllm_engine_args_except_seed_v1":
        from collector.glm53flash_vllm_graph_export import same_execution_policy

        same_execution_policy(left, right)
    else:
        from collector.fpm_forward.glm53flash_validation import _same_sglang_policy

        _same_sglang_policy(left, right, label)


def profile_control(root, proof, control_root, control_run):
    """Retain profiled/unprofiled elapsed differences, without fitting unit costs."""
    from collector.glm53flash_validation import load_native

    if control_run["role"] != "control" or control_run["spec"].get("ops_execution_mode") != "native_full_graph":
        raise ValueError("graph export requires an independent unprofiled same-calibration control")
    native = load_native(control_run, control_root)
    if Path(native["evidence_root"]) != control_root.resolve() or control_root.resolve() == root.resolve():
        raise ValueError("graph profiling control must retain a separate original native run")
    control = read_graph_run(control_root, control_run)
    _same_execution_policy(proof, control, "graph calibration/profile control")
    if control["native_snapshot"] != proof["native_snapshot"] or control["provenance"] != proof["provenance"]:
        raise ValueError("graph profiling control changes actual native policy/runtime identity")
    if control["forwards"].keys() != proof["forwards"].keys():
        raise ValueError("graph profiling control omits point repetitions")
    from collector.glm53flash_sglang_control import verify_pair

    verify_pair(root, proof, control)
    samples = defaultdict(list)
    calibration_ids, control_ids = set(), set()
    fields = (
        "benchmark_id",
        "repetition",
        "sampling_role",
        "batch_size",
        "prefix_lengths",
        "query_lengths",
        "num_padded_tokens",
        "corpus_sha256",
    )
    for key, ranks in proof["forwards"].items():
        other = control["forwards"][key]
        if ranks.keys() != other.keys() or any(
            any(row[field] != other[rank][field] for field in fields) for rank, row in ranks.items()
        ):
            raise ValueError("graph profiling control differs in actual cohort/tokens/dispatch")
        for rank, row in ranks.items():
            # Terminal sampling is after native metadata-to-logits execute.
            # Both original outputs remain in raw TP/state-chain evidence.
            actual_inputs = [{k: token[k] for k in ("input_sha256", "prompt_sha256")} for token in row["token_witness"]]
            control_inputs = [
                {k: token[k] for k in ("input_sha256", "prompt_sha256")} for token in other[rank]["token_witness"]
            ]
            if actual_inputs != control_inputs:
                raise ValueError("graph profiling control differs in actual model inputs")
        calibration_ids.update(ranks[0]["request_ids"])
        control_ids.update(other[0]["request_ids"])
        if ranks[0]["request_set"] == other[0]["request_set"]:
            raise ValueError("graph profiling control reuses its calibration native run")
        if ranks[0]["sampling_role"] == "measurement":
            samples[key[0]].append(
                {
                    "repetition": key[1],
                    "profiled_rank_ms": {str(rank): row["whole_forward_gpu_ms"] for rank, row in ranks.items()},
                    "control_rank_ms": {str(rank): row["whole_forward_gpu_ms"] for rank, row in other.items()},
                }
            )
    if calibration_ids & control_ids:
        raise ValueError("graph profiling control reuses calibration requests")
    results = []
    for bid, rows in sorted(samples.items()):
        if len(rows) < 10:
            raise ValueError("graph profiling control lacks ten measured repetitions")
        profiled = statistics.median(max(row["profiled_rank_ms"].values()) for row in rows)
        unprofiled = statistics.median(max(row["control_rank_ms"].values()) for row in rows)
        results.append(
            {
                "benchmark_id": bid,
                "profiled_median_ms": profiled,
                "control_median_ms": unprofiled,
                "profiled_to_control_ratio": profiled / unprofiled,
                "samples": rows,
            }
        )
    return {
        "schema": "glm53flash_graph_profile_control_v1",
        "execution_policy": control["execution_policy"],
        "evidence_root": str(control_root.resolve()),
        "frozen_run": control_run,
        "receipts": native["receipts"],
        "results": results,
        "timing_equivalence": "REPORTED_NOT_ASSUMED",
        "accuracy_acceptance": "NOT_EVALUATED",
    }


def export_graph(
    root: Path, run: dict, output: Path, *, control_root: Path, control_run: dict, lookup_contract=None
) -> dict:
    """Export only complete real calibration; this does not certify accuracy."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector.glm53flash_validation import _load_native

    _named_contract(lookup_contract)
    if run["role"] != "calibration" or run["spec"].get("ops_execution_mode") != "native_full_graph":
        raise ValueError("graph table export requires explicit native graph calibration")
    if output.name != BASENAME or output.exists():
        raise ValueError("graph export requires a new canonical table path")
    native = _load_native(run, root, calibration_evidence=False)
    if Path(native["evidence_root"]) != root.resolve():
        raise ValueError("graph export root differs from its frozen native run")
    proof = read_graph_run(root, run)
    control = profile_control(root, proof, control_root, control_run)
    control_path = root / "graph-profile-control.json"
    with control_path.open("x") as stream:
        stream.write(canonical_json(control))
    _, selection = aggregate_graph(proof, evidence_sha256="0" * 64, lookup_contract=lookup_contract)
    selection_path = root / "graph-rank-selection.json"
    with selection_path.open("x") as stream:
        stream.write(canonical_json(selection))
    files = proof["files"] | {row["path"] for row in native["receipts"]} | {selection_path.name, control_path.name}
    receipt = {
        "schema": "glm53flash_graph_calibration_v1",
        "graph_policy_sha256": sha256_json(proof["policy"]),
        "request_set": native["runtime_run_id"],
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
        "files": {name: file_sha256(_local(root, name)) for name in sorted(files)},
    }
    evidence = root / "graph-calibration-evidence.json"
    with evidence.open("x") as stream:
        stream.write(canonical_json(receipt))
    rows, _ = aggregate_graph(proof, evidence_sha256=file_sha256(evidence), lookup_contract=lookup_contract)
    pq.write_table(pa.Table.from_pylist(rows), output)
    return {"rows": len(rows), "table_sha256": file_sha256(output), "accuracy_acceptance": "NOT_EVALUATED"}


def republish_named_graph(root: Path, run: dict, output: Path) -> dict:
    """Derive a new named table from complete original traces, leaving raw files intact."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector.glm53flash_validation import _load_native

    if run["role"] != "calibration" or run["spec"].get("ops_execution_mode") != "native_full_graph":
        raise ValueError("named graph re-export requires original native FULL calibration")
    if output.name != BASENAME or output.exists():
        raise ValueError("named graph re-export requires a new canonical table path")
    native = _load_native(run, root, calibration_evidence=False)
    proof = read_graph_run(root, run)
    receipt = verify_evidence(root, proof)
    if Path(native["evidence_root"]) != root.resolve() or receipt["request_set"] != native["runtime_run_id"]:
        raise ValueError("named graph re-export differs from original native run")
    rows, _ = aggregate_graph(
        proof, evidence_sha256=file_sha256(root / "graph-calibration-evidence.json"), lookup_contract=NAMED_CONTRACT
    )
    pq.write_table(pa.Table.from_pylist(rows), output)
    return {"rows": len(rows), "table_sha256": file_sha256(output), "accuracy_acceptance": "NOT_EVALUATED"}


def bind_calibration(paths, run, native):
    import pyarrow.parquet as pq

    root = Path(native["evidence_root"])
    proof = read_graph_run(root, run)
    receipt = verify_evidence(root, proof)
    if receipt["request_set"] != native["runtime_run_id"] or receipt["source_plan_sha256"] != run["plan"]["sha256"]:
        raise ValueError("graph calibration evidence belongs to another frozen native run")
    selected = []
    tables = []
    identity = run["key"][:3]
    for path in paths:
        if path.name == BASENAME:
            tables.append({"path": str(path), "sha256": file_sha256(path)})
            for row in pq.read_table(path).to_pylist():
                policy = json.loads(row["graph_policy"])
                if (policy["backend"], policy["checkpoint_format"], policy["tp_size"]) == tuple(identity):
                    selected.append(row)
    contracts = {row.get("graph_lookup_contract") for row in selected}
    if len(contracts) != 1:
        raise ValueError("graph table mixes analysis lookup contracts or has no measurements")
    contract = contracts.pop()
    named = _named_contract(contract)
    if any((row.get("operation_name") is not None) != named for row in selected):
        raise ValueError("graph operation names require their explicit analysis contract")
    expected, _ = aggregate_graph(
        proof, evidence_sha256=file_sha256(root / "graph-calibration-evidence.json"), lookup_contract=contract
    )
    # A mixed-deployment Parquet may carry null named-analysis columns for a
    # different, legacy identity. Null optional metadata is equivalent to absence.
    comparable = [
        {
            key: value
            for key, value in row.items()
            if not (key in ("operation_name", "graph_lookup_contract") and value is None)
        }
        for row in selected
    ]
    if sorted(comparable, key=canonical_json) != sorted(expected, key=canonical_json):
        raise ValueError("consumer graph table differs from original native activity measurements")
    return {
        **({"lookup_contract": NAMED_CONTRACT} if named else {}),
        "rows": len(selected),
        "tables": tables,
        "graph_policy_sha256": sha256_json(proof["policy"]),
        "evidence_sha256": file_sha256(root / "graph-calibration-evidence.json"),
        "source_plan_sha256": run["plan"]["sha256"],
        "native_runtime_run_id": native["runtime_run_id"],
    }


def predict_homogeneous(run, base, config, calibration_native, *, calibration_binding=None):
    """Use public static geometry only after native per-request proof is checked.

    The common acceptance caller first binds every selected table to original
    calibration evidence and returns that binding plus exact table receipts.
    ``last_provenance() is None`` below checks only the absence of fallback; it
    is not a substitute for that positive measured-data binding.
    Policy selection comes from calibration. Holdout dispatch is only checked
    for policy mismatch; it never supplies prediction padding or unit latency.
    """
    from aisimulate_core.sdk.engine import EngineHandle
    from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig
    from collector.glm53flash_validation import load_native

    if run["spec"].get("ops_execution_mode") != "native_full_graph" or run["role"] != "holdout":
        raise ValueError("graph prediction requires explicit independent graph holdout")
    if len(config["systems_paths"]) != 1:
        raise ValueError("initial graph prediction requires one fully receipted calibration root")
    holdout = load_native(run, base)
    _same_execution_policy(calibration_native, holdout, "graph calibration/holdout")
    policy = calibration_native["graph_policy"]
    actual = holdout["graph_policy"]
    snapshot = actual["native_snapshot"]
    if policy["backend"] == "vllm":
        from collector.glm53flash_vllm_graph_policy import full_policy_fields

        snapshot = full_policy_fields({**snapshot, "tp_rank": 0})
    policy_fields = (
        "backend",
        "backend_version",
        "backend_revision",
        "capture_sizes",
        "disable_padding",
        "captured_req_width",
        "native_flags",
        "source_pins",
    )
    identity_fields = ("source_sha256", "config_sha256", "runtime_digest", "checkpoint_revision")
    if any(policy[key] != snapshot[key] for key in policy_fields) or any(
        policy[key] != actual["provenance"][key] for key in identity_fields
    ):
        raise ValueError("independent native holdout changed its frozen calibration graph policy")
    cfg = ForwardPassPerfModelConfig(**config)
    if (
        (cfg.backend, cfg.backend_version, cfg.tp, cfg.database_mode, cfg.estimation_mode, cfg.fallback_policy)
        != (policy["backend"], policy["backend_version"], policy["tp_size"], "SILICON", "op_level", "deny")
        or not cfg.strict_provenance
        or cfg.nextn
        or cfg.speculation
        or cfg.estimator_config
        or cfg.decoder_replay
    ):
        raise ValueError("graph prediction configuration differs from the strict measured contract")
    engine = EngineHandle.compile(
        cfg.model,
        cfg.system,
        cfg.backend,
        backend_version=cfg.backend_version,
        tp_size=cfg.tp,
        pp_size=cfg.pp,
        attention_dp_size=cfg.attention_dp,
        moe_tp_size=cfg.moe_tp_size,
        moe_ep_size=cfg.moe_ep_size,
        gemm_quant_mode=cfg.gemm_quant_mode,
        moe_quant_mode=cfg.moe_quant_mode,
        fmha_quant_mode=cfg.fmha_quant_mode,
        fpm_fmha_quant_mode=cfg.fpm_fmha_quant_mode,
        kvcache_quant_mode=cfg.kvcache_quant_mode,
        comm_quant_mode=cfg.comm_quant_mode,
        attention_backend=cfg.attention_backend,
        moe_backend=cfg.moe_backend,
        enable_eplb=cfg.enable_eplb,
        wideep_num_slots=cfg.wideep_num_slots,
        kv_block_size=cfg.kv_block_size,
        systems_path=cfg.systems_paths[0],
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
        transfer_policy=cfg.transfer_policy,
    )
    # This public graph path admits exactly one canonical, positively bound table.
    # Read its analysis metadata so an omitted binding cannot silently omit a named audit.
    import pyarrow.parquet as pq

    table = Path(cfg.systems_paths[0]) / "data" / cfg.system / cfg.backend / cfg.backend_version / BASENAME
    contract = None
    if table.is_file():
        selected = [
            row for row in pq.read_table(table).to_pylist() if row["graph_policy_sha256"] == sha256_json(policy)
        ]
        contracts = {row.get("graph_lookup_contract") for row in selected}
        if len(contracts) != 1:
            raise ValueError("graph prediction table has missing or mixed analysis contracts")
        contract = contracts.pop()
    named = _named_contract(contract)
    if named:
        evidence = Path(calibration_native["evidence_root"]) / "graph-calibration-evidence.json"
        original = json.loads(evidence.read_bytes())
        if not calibration_binding or (
            calibration_binding.get("lookup_contract") != contract
            or calibration_binding.get("graph_policy_sha256") != sha256_json(policy)
            or calibration_binding.get("native_runtime_run_id") != calibration_native["runtime_run_id"]
            or calibration_binding.get("evidence_sha256") != file_sha256(evidence)
            or calibration_binding.get("source_plan_sha256") != original["source_plan_sha256"]
            or {"path": str(table), "sha256": file_sha256(table)} not in calibration_binding.get("tables", [])
        ):
            raise ValueError("named graph prediction requires its exact original calibration binding")
    elif calibration_binding and calibration_binding.get("lookup_contract") is not None:
        raise ValueError("graph prediction binding differs from table analysis contract")
    rows, prediction_evidence = {}, {}
    for point in run["points"]:
        try:
            batch, total = point["batch_size"], point["total_kv_read_tokens"]
            if point["point_type"] != "decode" or point["total_prefill_tokens"] != 0 or total % batch:
                raise ValueError("graph static prediction requires homogeneous native past lengths")
            value = engine.predict_decode_latency(batch, total // batch, 2)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("public graph consumer returned no positive finite latency")
            if engine.last_provenance() is not None:
                raise ValueError("graph consumer fired a non-silicon fallback")
            if named:
                audit = engine.glm53flash_lookup_audit("generation", batch, 1, total // batch)
                if (
                    audit.get("lookup_contract") != NAMED_CONTRACT
                    or audit.get("graph_policy_sha256") != sha256_json(policy)
                    or len(audit.get("operations", [])) != (367 if policy["backend"] == "sglang" else 278)
                    or not math.isclose(sum(op["latency_ms"] for op in audit["operations"]), value, rel_tol=1e-12)
                ):
                    raise ValueError("named graph endpoint audit differs from public prediction")
                prediction_evidence[point["benchmark_id"]] = audit
            rows[point["benchmark_id"]] = {"prediction_ms": value}
        except Exception as error:
            rows[point["benchmark_id"]] = {"error": f"{type(error).__name__}: {error}"}
    return {
        "rows": rows,
        **({"prediction_evidence": prediction_evidence} if named else {}),
        "diagnostics": {
            "consumer": "public_EngineHandle_predict_decode_latency",
            "graph_policy_sha256": sha256_json(policy),
            "composition": "disjoint_native_unit_unions_additive_approximation",
        },
    }
