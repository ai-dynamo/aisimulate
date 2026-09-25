# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema3 identities and named-unit reduction for native V2 serving evidence.

The original schema1/2 reducers remain separate. These functions consume already
rederived native activity; they do not admit diagnostic NONE observations, grant
runtime qualification, or infer query dispatch from holdout measurements.
Native identity follows vllm-project/vllm ced6857afa0ea7b2e3f0846a62e1394e90f15607,
vllm/v1/worker/gpu/{model_runner,cudagraph_utils}.py and
vllm/compilation/breakable_cudagraph.py (Apache-2.0). This original reader copies
no compute implementation; see THIRD_PARTY_NOTICES.md.
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
    CHECKPOINTS,
    WHOLE_FORWARD_RANK,
    _runtime_contract,
    build_model_manifest,
    canonical_json,
    sha256_json,
)
from collector.glm53flash_graph_export import BASENAME, _local
from collector.glm53flash_jsonl import file_sha256
from collector.glm53flash_vllm_graph_export import BOUNDARY, same_execution_policy
from collector.glm53flash_vllm_graph_policy import SOURCE_PINS, select_descriptor, validate_snapshot

SCOPE = "native_vllm_serving_units_v1"
GRAPH_METHOD = "native_cupti_unit_union_v1"
LOOKUP_CONTRACT = "vllm_serving_bounded_p_q_v1"
KEYS = (
    "component",
    "operation_name",
    "geometry",
    "phase",
    "runtime_mode",
    "batch_size",
    "query_length",
    "prefix",
    "physical_num_tokens",
    "physical_num_requests",
)
COLUMNS = (
    *KEYS,
    "latency",
    "contribution_count",
    "sample_count",
    "dispatch_fingerprint",
    "measurement_method",
    "graph_policy",
    "graph_policy_sha256",
    "dataset_role",
    "aggregation_policy",
    "rank_selection_sha256",
    "evidence_sha256",
    "policy_evidence_sha256",
    "measurement_scope",
)
NATIVE_KEYS = {
    "backend",
    "backend_version",
    "backend_revision",
    "source_pins",
    "native_flags",
    "capture_sizes",
    "max_num_reqs",
    "max_capture_tokens",
    "decode_query_len",
    "graphs_captured",
    "lora_capture_cases",
    "dp_size",
    "tp_size",
    "resolved_mode",
    "use_breakable_cg",
    "capture_descriptors",
    "full_graphs",
    "candidates",
    "piecewise_entries",
}


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _elapsed(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)


def table_lookup_contract(rows):
    contracts = set()
    for row in rows:
        present = "lookup_contract" in row
        if present != ("source_ownership_sha256" in row):
            raise ValueError("incomplete serving lookup metadata")
        contract = row.get("lookup_contract")
        if present and (contract != LOOKUP_CONTRACT or not _hash(row["source_ownership_sha256"])):
            raise ValueError("unknown serving lookup contract or source ownership")
        contracts.add(contract)
    if len(contracts) != 1:
        raise ValueError("serving table mixes analysis lookup contracts")
    return contracts.pop()


def analysis_rows(proof, rows, lookup_contract):
    """Opt in using rederived source ownership, without changing native policy."""
    if lookup_contract is None:
        return rows
    if lookup_contract != LOOKUP_CONTRACT:
        raise ValueError("unknown serving bounded lookup contract")
    ownership = defaultdict(set)
    for (_, repetition), ranks in proof["forwards"].items():
        if repetition < 5:
            continue
        selected = min(ranks, key=lambda rank: (-ranks[rank]["whole_forward_gpu_ms"], rank))
        forward = ranks[selected]
        key = (forward["phase"], forward["batch_size"], forward["query_lengths"][0], forward["prefix_lengths"][0])
        for name, unit in forward["binding"].items():
            value = unit.get("source_ownership_sha256")
            if not _hash(value):
                raise ValueError("serving lookup lacks rederived native call ownership")
            ownership[(*key, name)].add(value)
    result = []
    for row in rows:
        key = tuple(row[name] for name in ("phase", "batch_size", "query_length", "prefix", "operation_name"))
        values = ownership.get(key, set())
        if len(values) != 1:
            raise ValueError("serving point changes source ownership across retained samples")
        result.append({**row, "lookup_contract": lookup_contract, "source_ownership_sha256": next(iter(values))})
    return result


def _capture_ownership(registry, entries, mode):
    """Keep physical call/fusion ownership, not graph handles or kernel grids."""
    from collector.glm53flash_vllm_graph_export import LOGITS_SOURCE_PIN

    result = {}
    for entry in entries:
        name = entry["name"]
        calls = [
            {
                key: call.get(key)
                for key in ("source", "included_sources", "excluded_collective_sources", "parent_operation")
            }
            for call in registry["calls"]
            if call["name"] == name
        ]
        if name == "logits":
            calls = [{"source_boundary": "GPUModelRunner.compute_logits", "source_sha256": LOGITS_SOURCE_PIN}]
        elif name == "native_graph_setup":
            calls = [
                {"source_boundary": "native_metadata_to_logits_outside_physical_operations", "source_pins": SOURCE_PINS}
            ]
        if not calls:
            raise ValueError("serving bounded lookup omits original captured call ownership")
        segments = []
        if mode == "PIECEWISE":
            for segment in registry["segments"]:
                owns = (
                    any(node["name"] == name for node in segment["nodes"])
                    if segment["kind"] == "graph"
                    else segment.get("name") == name
                )
                if owns:
                    segments.append(
                        {key: segment.get(key) for key in ("kind", "position", "qualname", "source_sha256")}
                    )
        result[name] = sha256_json({"operation": entry, "mode": mode, "calls": calls, "segments": segments})
    return result


def build_serving_policy(snapshots, manifest, provenance, execution):
    """Separate stable deployment identity from each attempt's raw file hashes."""
    tp = manifest["tp_size"]
    if type(tp) is not int or tp not in (2, 4) or set(snapshots) != set(range(tp)):
        raise ValueError("serving policy requires every actual TP2/TP4 worker")
    common = None
    for rank, snapshot in snapshots.items():
        if set(snapshot) != NATIVE_KEYS | {"tp_rank"}:
            raise ValueError("serving policy differs from the original native snapshot fields")
        validate_snapshot(snapshot)
        if snapshot["tp_rank"] != rank or snapshot["tp_size"] != tp:
            raise ValueError("serving snapshot belongs to another actual worker")
        value = {key: item for key, item in snapshot.items() if key != "tp_rank"}
        if common is not None and common != value:
            raise ValueError("serving workers changed their initialized native policy")
        common = value
    version = common["backend_version"]
    revision, source = _runtime_contract("vllm", version)
    formats = {json.loads(row["geometry"])["checkpoint_format"] for row in manifest["phases"]["context"]}
    if len(formats) != 1:
        raise ValueError("serving manifest mixes checkpoint formats")
    fmt = formats.pop()
    if manifest != build_model_manifest("vllm", fmt, tp, version):
        raise ValueError("serving manifest differs from the complete production model")
    expected = {
        "backend": "vllm",
        "backend_version": version,
        "backend_revision": revision,
        "checkpoint_revision": manifest["checkpoint_revision"],
        "config_sha256": manifest["config_sha256"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    if any(provenance.get(key) != value for key, value in expected.items()) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", provenance.get("runtime_digest", "")
    ):
        raise ValueError("serving provenance differs from exact native source/checkpoint/config")
    same_execution_policy(execution, execution)
    if execution["_execution_policy"].get("enforce_eager") is not False:
        raise ValueError("serving policy cannot reuse an enforce_eager deployment")
    return {
        "schema_version": 3,
        **expected,
        "checkpoint_format": fmt,
        "tp_size": tp,
        "runtime_digest": provenance["runtime_digest"],
        "source_pins": dict(SOURCE_PINS),
        "timing_boundary": BOUNDARY,
        "execution_policy_sha256": execution["execution_policy"]["sha256"],
        "native_policy": common,
        "native_policy_sha256": sha256_json(common),
    }


def check_serving_dispatch(row, snapshot):
    """Check real NONE/PW/FULL coordinates without changing their meaning."""
    batch, phase = row.get("batch_size"), row.get("phase")
    queries, prefixes = row.get("query_lengths"), row.get("prefix_lengths")
    if (
        type(batch) is not int
        or batch < 1
        or phase not in ("context", "generation")
        or not isinstance(queries, list)
        or not isinstance(prefixes, list)
        or len(queries) != batch
        or len(prefixes) != batch
        or any(type(value) is not int for value in (*queries, *prefixes))
        or len(set(queries)) != 1
        or len(set(prefixes)) != 1
        or prefixes[0] < (1 if phase == "generation" else 0)
        or queries[0] < 1
        or prefixes[0] + queries[0] > 131072
    ):
        raise ValueError("serving forward lacks bounded homogeneous actual coordinates")
    descriptor = select_descriptor(snapshot, batch=batch, query=queries[0], is_context=phase == "context")
    mode = descriptor["cg_mode"]
    expected = {
        "descriptor": descriptor,
        "policy_sha256": sha256_json(snapshot),
        "physical_tokens": descriptor["num_tokens"],
        "physical_requests": descriptor["num_reqs"] or batch,
    }
    if (
        row.get("native_dispatch") != expected
        or row.get("runtime_mode") != mode
        or row.get("used_cuda_graph") is not (mode != "NONE")
        or type(row.get("num_padded_tokens")) is not int
        or row["num_padded_tokens"] != descriptor["num_tokens"]
    ):
        raise ValueError("serving forward changed native dispatch or physical padding")
    if row.get("stage") == "measure":
        if row.get("measurement_admission") == "DIAGNOSTIC_ONLY_NO_TABLE_EXPORT":
            raise ValueError("diagnostic NONE evidence cannot become a measured serving table")
        if mode == "NONE":
            from collector.glm53flash_vllm_none_activity import NONE_MEASUREMENT_CONTRACT, NONE_SETUP_RANGES

            boundaries = row.get("native_runtime_boundary_gpu_ms", {})
            if (
                row.get("measurement_contract") != NONE_MEASUREMENT_CONTRACT
                or "measurement_admission" in row
                or phase != "context"
                or row.get("native_none_forward_completed") is not True
                or row.get("native_graph_replay_completed") is not False
                or not _hash(row.get("serving_none_model_sha256"))
                or set(boundaries) != set(NONE_SETUP_RANGES)
                or any(not _elapsed(value) for value in boundaries.values())
            ):
                raise ValueError("serving NONE measurement admission requires original native event/source proof")
            return descriptor
        if row.get("native_graph_replay_completed") is not True:
            raise ValueError("serving graph lacks its actual completed native replay")
        if mode == "PIECEWISE":
            entry = next(
                item for item in snapshot["piecewise_entries"] if item["num_tokens"] == descriptor["num_tokens"]
            )
            expected_replay = {
                "source_sha256": SOURCE_PINS["compilation/breakable_cudagraph.py"],
                "entry_descriptor": {
                    key: entry[key] for key in ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras")
                },
                "segment_count": entry["num_graphs"] + entry["num_eager_breaks"],
            }
            if (
                row.get("native_piecewise_replay_completed") is not True
                or row.get("native_piecewise_replay") != expected_replay
            ):
                raise ValueError("serving PIECEWISE replay changed its actual native entry/segments")
    return descriptor


def _entries(manifest, phase):
    entries = manifest["phases"][phase]
    runtime = manifest["runtime_operations"][phase]
    if (
        len(entries) != 277
        or len({row["name"] for row in entries}) != 277
        or any(row["component"] == "runtime" or row["name"] == "native_graph_setup" for row in entries)
        or len(runtime) != 1
        or runtime[0]["name"] != "native_graph_setup"
        or runtime[0]["component"] != "runtime"
    ):
        raise ValueError("serving reduction requires 277 named physical units and exactly one setup")
    return entries + runtime


def aggregate_serving(proof, *, evidence_sha256):
    """Select one actual forward's rank; never collapse layers sharing geometry.

    This reducer has no filesystem/export entry point. Its caller must rederive
    native source/trace ownership and the complete same-request history first.
    Diagnostic NONE rows are rejected, including exact geometry matches.
    """
    policy = build_serving_policy(proof["snapshots"], proof["manifest"], proof["provenance"], proof)
    if policy != proof["policy"] or not _hash(evidence_sha256) or not _hash(proof["policy_evidence_sha256"]):
        raise ValueError("serving reduction lacks exact stable policy and attempt evidence")
    tp = policy["tp_size"]
    entries = {phase: _entries(proof["manifest"], phase) for phase in ("context", "generation")}
    grouped, repetitions, selections = defaultdict(list), defaultdict(set), []
    observed = {rank: set() for rank in range(tp)}
    if not proof["forwards"]:
        raise ValueError("serving reduction has no actual measured forwards")
    for key, ranks in sorted(proof["forwards"].items()):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or type(key[0]) is not int
            or key[0] < 1
            or type(key[1]) is not int
            or not 0 <= key[1] < 15
            or set(ranks) != set(range(tp))
        ):
            raise ValueError("serving reduction requires exact all-rank 5+10 forward joins")
        repetitions[key[0]].add(key[1])
        first = ranks[0]
        join_fields = (
            "phase",
            "runtime_mode",
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
            "token_witness",
        )
        for rank, record in ranks.items():
            check_serving_dispatch(record, proof["snapshots"][rank])
            if (
                any(record.get(field) != first.get(field) for field in join_fields)
                or (record.get("benchmark_id"), record.get("repetition")) != key
                or record.get("tp_rank") != rank
                or record.get("sampling_role") != ("warmup" if key[1] < 5 else "measurement")
                or record.get("dataset_role") != "calibration"
                or record.get("stage") != "measure"
                or record.get("gpu_completed") is not True
                or record.get("whole_forward_boundary") != BOUNDARY
                or not _elapsed(record.get("whole_forward_gpu_ms"), positive=True)
                or type(record.get("invocation")) is not int
                or record["invocation"] < 1
                or record.get("forward_id") != f"rank-{rank}/forward-{record['invocation']}"
                or record["invocation"] in observed[rank]
            ):
                raise ValueError("serving rank selection changed a completed same-forward identity")
            observed[rank].add(record["invocation"])
            units = record.get("binding", {})
            if set(units) != {entry["name"] for entry in entries[record["phase"]]}:
                raise ValueError("serving forward has missing, duplicated or unknown named units")
            for name, unit in units.items():
                if record["runtime_mode"] == "NONE":
                    setup = name == "native_graph_setup"
                    if (
                        type(unit.get("activity_count")) is not int
                        or unit["activity_count"] != (2 if setup else 1)
                        or unit.get("dispatch") != ""
                        or unit.get("method")
                        != ("native_runtime_cuda_events_v1" if setup else "native_module_cuda_events_v1")
                        or not _elapsed(unit.get("latency"), positive=not setup)
                    ):
                        raise ValueError("serving NONE unit lacks its original native event interval")
                    continue
                if (
                    type(unit.get("activity_count")) is not int
                    or unit["activity_count"] < 0
                    or not _hash(unit.get("dispatch"))
                    or not _elapsed(unit.get("latency"), positive=unit["activity_count"] > 0)
                    or (unit["activity_count"] == 0 and unit["latency"] != 0)
                ):
                    raise ValueError("serving graph unit lacks actual activity/zero-boundary evidence")
        selected = min(ranks, key=lambda rank: (-ranks[rank]["whole_forward_gpu_ms"], rank))
        record = ranks[selected]
        selections.append(
            {
                "benchmark_id": key[0],
                "repetition": key[1],
                "selected_rank": selected,
                "ranks": [
                    {
                        "rank": rank,
                        **{field: row[field] for field in ("forward_id", "invocation", "whole_forward_gpu_ms")},
                    }
                    for rank, row in sorted(ranks.items())
                ],
            }
        )
        if record["sampling_role"] != "measurement":
            continue
        for entry in entries[record["phase"]]:
            unit = record["binding"][entry["name"]]
            identity = (
                entry["component"],
                entry["name"],
                entry["geometry"],
                record["phase"],
                record["runtime_mode"],
                record["batch_size"],
                record["query_lengths"][0],
                record["prefix_lengths"][0],
                record["native_dispatch"]["physical_tokens"],
                record["native_dispatch"]["physical_requests"],
            )
            grouped[identity].append((key, unit))
    if any(values != set(range(15)) for values in repetitions.values()):
        raise ValueError("serving reduction omitted original warmup or measured repetitions")
    selection = {
        "schema": "glm53flash_serving_rank_selection_v1",
        "aggregation_policy": WHOLE_FORWARD_RANK,
        "forwards": selections,
    }
    rows = []
    for identity, samples in sorted(grouped.items()):
        signatures = {(row["dispatch"], row["activity_count"], row.get("method", GRAPH_METHOD)) for _, row in samples}
        ids = {key for key, _ in samples}
        if len(signatures) != 1 or len(ids) != len(samples) or len(samples) < 10:
            raise ValueError("serving named unit mixes actual dispatch or repeats a physical measurement")
        dispatch, count, method = signatures.pop()
        rows.append(
            {
                **dict(zip(KEYS, identity, strict=True)),
                "latency": statistics.median(row["latency"] for _, row in samples),
                "contribution_count": count,
                "sample_count": len(samples),
                "dispatch_fingerprint": dispatch,
                "measurement_method": method,
                "graph_policy": canonical_json(policy),
                "graph_policy_sha256": sha256_json(policy),
                "dataset_role": "calibration",
                "aggregation_policy": WHOLE_FORWARD_RANK,
                "rank_selection_sha256": sha256_json(selection),
                "evidence_sha256": evidence_sha256,
                "policy_evidence_sha256": proof["policy_evidence_sha256"],
                "measurement_scope": SCOPE,
            }
        )
    return rows, selection


def _piecewise_captures(root, rank, snapshot, manifest, provenance, files, full_captures):
    """Rebuild every PW segment from original source and shared callbacks."""
    from collector.glm53flash_graph_callbacks import (
        QUALIFIED_CUPTI_SHA256,
        resolve_registry,
        vllm_memset_contract_options,
    )
    from collector.glm53flash_graph_export import _local, _receipt
    from collector.glm53flash_jsonl import file_sha256
    from collector.glm53flash_receipt_cache import ReceiptCache
    from collector.glm53flash_vllm_piecewise import BREAKABLE_SOURCE_PIN, EAGER_RANGE_PREFIX

    identities = [set(), set(), set()]

    def claim(source_id, receipt):
        creates = [
            row
            for row in receipt["callbacks"]
            if row["kind"] == "graph_exec_created" and row["graph_exec_id"] == receipt["actual_graph_exec_id"]
        ]
        if len(creates) != 1:
            raise ValueError("serving capture lacks its unique native executable creation")
        create = creates[0]
        values = source_id, receipt["actual_graph_exec_id"], create["raw_fields"]["graphExec"]
        for value, seen in zip(values, identities, strict=True):
            if type(value) is not int or value <= 0 or value in seen:
                raise ValueError("serving captures reuse a source or live executable identity")
            seen.add(value)
        return create

    for registry, _ in full_captures.values():
        receipt = _receipt(root, registry["instantiation_receipt"], files)
        claim(registry["capture_graph_id"], receipt)
    captures, shared_actual, shared_expected = {}, defaultdict(list), {}
    shared_receipts = ReceiptCache(root, files)
    expected_entries = {entry["num_tokens"]: entry for entry in snapshot["piecewise_entries"]}
    entries = manifest["phases"]["context"]
    expected_names = {row["name"] for row in entries} - {"logits"}
    paths = sorted(root.glob(f"vllm-graph-clones-rank-{rank}-capture-*-piecewise-*-bound.json"))
    for path in paths:
        match = re.fullmatch(
            rf"vllm-graph-clones-rank-{rank}-capture-([0-9]+)-piecewise-([0-9]+)-bound\.json", path.name
        )
        if match is None:
            raise ValueError("serving PW capture has an unknown artifact name")
        serial, index = map(int, match.groups())
        stem = f"vllm-graph-clones-rank-{rank}-capture-{serial}"
        recorded = json.loads(_local(root, path.name).read_bytes())
        files.add(path.name)
        source_ref = recorded["source_receipt"]
        if source_ref["file"] != f"vllm-piecewise-source-rank-{rank}-capture-{serial}-{index}.json":
            raise ValueError("serving PW source belongs to another worker/capture")
        source = _receipt(root, source_ref, files)
        shape = source.get("native_shape_key", {})
        tokens = shape.get("num_tokens")
        entry = expected_entries.get(tokens)
        if (
            entry is None
            or type(tokens) is not int
            or tokens in captures
            or shape
            != {key: entry[key] for key in ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras")}
            or source.get("tp_rank") != rank
            or source.get("provenance") != provenance
            or source.get("physical_padded_tokens") != tokens
            or source.get("operations") != entries
            or source.get("capture_scope") != "vllm_piecewise_hidden_states"
            or source.get("uncaptured_operations") != ["logits"]
            or source.get("graph_mutations") is not False
            or source.get("measurement_method") != "native_piecewise_capture_ownership"
        ):
            raise ValueError("serving PW capture differs from initialized native shape/model/source")
        libraries = source.get("native_api_libraries", {})
        runtime = libraries.get("cudart", {})
        from pathlib import Path

        if (
            libraries.get("cupti", {}).get("sha256") != QUALIFIED_CUPTI_SHA256
            or type(runtime.get("runtime_version")) is not int
            or runtime["runtime_version"] // 1000 != 13
            or runtime.get("abi") != "CUDA13_capture7_edges5"
            or not _hash(runtime.get("sha256"))
            or any(not Path(libraries[name].get("path", "")).is_absolute() for name in ("cudart", "cupti"))
        ):
            raise ValueError("serving PW capture lacks its actual qualified native libraries")
        calls = source["calls"]
        if (
            len(calls) != len(expected_names)
            or {row.get("name") for row in calls} != expected_names
            or any(
                row.get("index") != i or row.get("completed") is not True or not row.get("source")
                for i, row in enumerate(calls)
            )
        ):
            raise ValueError("serving PW capture omits a completed physical call, including zero activity")
        segments = source["segments"]
        if (
            len(segments) != entry["num_graphs"] + entry["num_eager_breaks"]
            or sum(row.get("kind") == "graph" for row in segments) != entry["num_graphs"]
            or sum(row.get("kind") == "eager" for row in segments) != entry["num_eager_breaks"]
            or any(row.get("position") != i for i, row in enumerate(segments))
            or len(recorded["segments"]) != len(segments)
        ):
            raise ValueError("serving PW capture omits or changes initialized native segments")

        def owner(value):
            idx, name = value.get("call_index"), value.get("name")
            if idx is None and name == "native_graph_setup":
                return
            if type(idx) is not int or not 0 <= idx < len(calls) or calls[idx]["name"] != name:
                raise ValueError("serving PW graph/eager contribution has an unknown physical owner")

        derived_segments, eager_ids = [], set()
        for segment, bound in zip(segments, recorded["segments"], strict=True):
            position = segment["position"]
            if segment["kind"] == "eager":
                owner(segment)
                eager_id = segment.get("eager_id")
                if (
                    type(eager_id) is not int
                    or eager_id < 0
                    or eager_id in eager_ids
                    or segment.get("source_sha256") != BREAKABLE_SOURCE_PIN
                    or segment.get("range") != EAGER_RANGE_PREFIX + str(eager_id)
                    or not segment.get("qualname")
                    or not Path(segment.get("source_file", "")).is_absolute()
                ):
                    raise ValueError("serving PW eager callable lacks its exact native identity")
                eager_ids.add(eager_id)
                derived_segments.append(segment)
                continue
            for node in segment["nodes"]:
                owner(node)
            original = {key: value for key, value in segment.items() if key not in ("kind", "position")}
            original["native_api_libraries"] = libraries
            shared_ref = bound["shared_callback_receipt"]
            if shared_ref["file"] != f"{stem}-piecewise-callbacks.json":
                raise ValueError("serving PW segment refers to another callback subscription")
            shared = shared_receipts.read(shared_ref)
            if (
                shared.get("schema") != "glm53flash_piecewise_callbacks_v1"
                or shared.get("callback_subscription_closed") is not True
                or shared.get("graph_mutations") is not False
                or shared.get("callback_errors") != []
            ):
                raise ValueError("serving PW callback stream is incomplete or changed")
            observed = bound["observed_executable"]
            if observed.get("capture_index") != index or observed.get("position") != position:
                raise ValueError("serving PW executable belongs to another native segment")
            receipt = {
                key: shared[key]
                for key in ("callbacks", "callback_errors", "callback_subscription_closed", "graph_mutations")
            }
            receipt["actual_graph_exec_id"] = observed["actual_graph_exec_id"]
            type_ref = bound.get("node_type_receipt")
            if type_ref is not None and type_ref["file"] != f"{stem}-piecewise-{index}-{position}-event-types.json":
                raise ValueError("serving PW node query belongs to another executable")
            type_proof = _receipt(root, type_ref, files) if type_ref is not None else None
            derived = resolve_registry(
                original, receipt, type_proof, allow_pending_memcpy=True, **vllm_memset_contract_options(bound)
            )
            create = claim(original["graph_id"], receipt)
            if observed.get("actual_graph_exec_handle") != create["raw_fields"]["graphExec"]:
                raise ValueError("serving PW executable handle differs from its actual native callback")
            derived.update(shared_callback_receipt=shared_ref, observed_executable=observed)
            if type_ref is not None:
                derived["node_type_receipt"] = type_ref
            derived_segments.append({"kind": "graph", "position": position, **derived})
            shared_actual[shared_ref["file"]].append(observed)
            if (
                shared_ref["file"] in shared_expected
                and shared_expected[shared_ref["file"]] != shared["observed_executables"]
            ):
                raise ValueError("serving PW shared callback evidence changed between segments")
            shared_expected[shared_ref["file"]] = shared["observed_executables"]
        if eager_ids != set(range(entry["num_eager_breaks"])):
            raise ValueError("serving PW omitted an original native eager callable")
        derived = {**source, "segments": derived_segments, "source_receipt": source_ref}
        if derived != recorded:
            raise ValueError("serving PW executable ownership differs from original native evidence")
        captures[tokens] = derived, {"file": path.name, "sha256": file_sha256(path)}
    if set(captures) != set(expected_entries):
        raise ValueError("serving PW capture omits an initialized native entry")
    for name, values in shared_actual.items():
        if sorted(map(canonical_json, values)) != sorted(map(canonical_json, shared_expected[name])):
            raise ValueError("serving PW shared callback receipt omits or repeats observed executables")
    shared_receipts.verify()
    return captures


def _replay_binding(root, row, registry, files):
    from collector.glm53flash_graph_export import _receipt
    from collector.glm53flash_graph_nodes import trace_forward_identity
    from collector.glm53flash_vllm_graph_export import LOGITS_SOURCE_PIN, _binding
    from collector.glm53flash_vllm_piecewise_activity import bind_piecewise_execution

    if row["runtime_mode"] == "FULL":
        return _binding(root, row, registry, files)
    recorded = row["replay_nodes"]
    name = f"graph-profile-rank-{row['tp_rank']}-forward-{row['invocation']}.json"
    if recorded["trace_file"] != name or name in files:
        raise ValueError("serving PW trace must uniquely belong to one actual rank/forward")
    trace = _receipt(root, {"file": name, "sha256": recorded["trace_sha256"]}, files)
    if trace.get("aisim_native_forward") != trace_forward_identity(row) or trace.get("aisim_native_execution") != {
        "backend": "vllm",
        "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
        "logits_source_sha256": LOGITS_SOURCE_PIN,
        "failed": False,
        "runtime_mode": "PIECEWISE",
    }:
        raise ValueError("serving PW trace belongs to another native forward or logits boundary")
    derived = bind_piecewise_execution(registry, trace["traceEvents"])
    derived.update(trace_file=name, trace_sha256=recorded["trace_sha256"])
    if derived != recorded:
        raise ValueError("serving PW costs differ from original graph/eager device activity")
    return derived


def _none_observations(root, rank, targets, entries, provenance, files, *, calibrated):
    """Join separate NONE event rows to excluded warmup source/activity proof."""
    from collector.glm53flash_contract import validate_row
    from collector.glm53flash_graph_export import _receipt
    from collector.glm53flash_graph_nodes import trace_forward_identity
    from collector.glm53flash_jsonl import iter_records
    from collector.glm53flash_vllm_none import validate_model_receipt
    from collector.glm53flash_vllm_none_activity import NONE_MEASUREMENT_CONTRACT, bind_none_execution

    model_file = f"serving-none-model-rank-{rank}.json"
    model = validate_model_receipt(json.loads(_local(root, model_file).read_bytes()))
    files.add(model_file)
    physical = {entry["name"]: entry for entry in entries if entry["component"] != "runtime"}
    profiles = {}
    for record in targets.values():
        if record.get("serving_none_model_sha256") != sha256_json(model):
            raise ValueError("native NONE forward changed its original model identity")
        profiled = calibrated and record["repetition"] == 4
        if record.get("profiled") is not profiled or ("native_none_profile" in record) is not profiled:
            raise ValueError("native NONE must profile only its fifth excluded calibration warmup")
        if calibrated and record.get("native_operation_calls") != dict.fromkeys(physical, 1):
            raise ValueError("native NONE forward omitted or repeated a physical native operation")
        if not profiled:
            continue
        recorded = record["native_none_profile"]
        name = f"none-profile-rank-{rank}-forward-{record['invocation']}.json"
        if recorded.get("trace_file") != name or name in files:
            raise ValueError("native NONE warmup trace was reused across actual forwards")
        trace = _receipt(root, {"file": name, "sha256": recorded["trace_sha256"]}, files)
        metadata = trace.get("aisim_native_none", {})
        calls = metadata.get("native_calls", [])
        if trace.get("aisim_native_forward") != trace_forward_identity(record) or metadata != {
            "measurement_contract": NONE_MEASUREMENT_CONTRACT,
            "model_identity_sha256": sha256_json(model),
            "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
            "native_calls": calls,
            "failed": False,
        }:
            raise ValueError("native NONE warmup source trace belongs to another forward")
        derived = bind_none_execution(trace["traceEvents"], calls, physical)
        if recorded.get("binding") != derived or record["benchmark_id"] in profiles:
            raise ValueError("native NONE warmup ownership differs from its original source/API activity")
        profiles[record["benchmark_id"]] = {row["operation"]: row for row in calls}
    if not calibrated:
        return {}
    if set(profiles) != {row["benchmark_id"] for row in targets.values()}:
        raise ValueError("native NONE point lacks its independently observed fifth warmup")
    path = f"serving-none-measured-ops-rank-{rank}.jsonl"
    files.add(path)
    units = {invocation: {} for invocation in targets}
    for row in iter_records(_local(root, path)):
        record = targets.get(row.get("invocation"))
        entry = physical.get(row.get("name"))
        if record is None or entry is None or row["name"] in units[record["invocation"]]:
            raise ValueError("native NONE event row has an unknown/repeated forward or physical unit")
        validate_row(row)
        call = profiles[record["benchmark_id"]][row["name"]]
        source = "+".join(sorted({call["source"], *call["included_sources"]}))
        geometry = json.loads(entry["geometry"])
        attention = entry["component"] == "attention"
        query, prefix, batch = record["query_lengths"][0], record["prefix_lengths"][0], record["batch_size"]
        coordinates = {
            "batch_size": batch if attention else 1,
            "prefix": prefix if attention else 0,
            "x": query
            if attention
            else batch
            if geometry.get("token_selection") == "last_per_request"
            else batch * query,
        }
        same_forward = (
            "stage",
            "forward_id",
            "tp_rank",
            "phase",
            "benchmark_id",
            "repetition",
            "sampling_role",
            "dataset_role",
            "request_set",
            "corpus_sha256",
            "request_ids",
            "native_dispatch",
            "serving_none_model_sha256",
            "profiled",
        )
        excluded = row.get("excluded_collectives", [])
        if (
            any(
                row.get(key) != value
                for source_row in (entry, provenance, coordinates)
                for key, value in source_row.items()
            )
            or any(row.get(key) != record.get(key) for key in same_forward)
            or row.get("sample") != record["repetition"]
            or row.get("measurement_contract") != NONE_MEASUREMENT_CONTRACT
            or "measurement_admission" in row
            or row.get("measurement_method") != "native_module_cuda_events_v1"
            or row.get("used_cuda_graph") is not False
            or row.get("sample_count") != 1
            or row.get("dispatch_fingerprint") != ""
            or row.get("kernel_source") != source
            or not _elapsed(row.get("latency"), positive=True)
            or not isinstance(excluded, list)
            or sorted(item.get("source", "") for item in excluded) != sorted(call["excluded_collective_sources"])
            or any(not _elapsed(item.get("latency")) for item in excluded)
        ):
            raise ValueError("native NONE event row differs from its original source/shape/forward identity")
        units[record["invocation"]][row["name"]] = {
            "latency": row["latency"],
            "dispatch": "",
            "activity_count": 1,
            "method": "native_module_cuda_events_v1",
            "source_ownership_sha256": sha256_json(
                {
                    "operation": entry,
                    "model_identity_sha256": sha256_json(model),
                    "calls": [
                        {
                            key: call.get(key)
                            for key in ("source", "included_sources", "excluded_collective_sources", "parent_operation")
                        }
                    ],
                }
            ),
        }
    for invocation, values in units.items():
        if set(values) != set(physical):
            raise ValueError("native NONE event evidence omits physical units")
        values["native_graph_setup"] = {
            "latency": sum(targets[invocation]["native_runtime_boundary_gpu_ms"].values()),
            "dispatch": "",
            "activity_count": 2,
            "method": "native_runtime_cuda_events_v1",
            "source_ownership_sha256": sha256_json(
                {
                    "operation_name": "native_graph_setup",
                    "model_identity_sha256": sha256_json(model),
                    "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
                    "regions": sorted(targets[invocation]["native_runtime_boundary_gpu_ms"]),
                    "source_pins": SOURCE_PINS,
                }
            ),
        }
    return units


def read_serving_run(root, run):
    """Recompute serving graph evidence; common loader still checks native state.

    Both independent truth and calibration retain the full initialization
    policy. Only calibration opens capture/trace evidence, never holdout data
    to supply ownership or dispatch. NONE diagnostic targets remain closed.
    """
    from pathlib import Path

    from collector.glm53flash_graph_export import _compact_units, _local
    from collector.glm53flash_jsonl import file_sha256, iter_records
    from collector.glm53flash_validation import _required_files, expected_runtime_version
    from collector.glm53flash_vllm_graph_export import LOGITS_SOURCE_PIN, _captures, execution_policy

    root = Path(root)
    backend, fmt, tp, phase = run["key"]
    if (
        backend != "vllm"
        or phase not in ("prefill", "decode")
        or run["spec"].get("ops_execution_mode") != "native_serving"
        or run["role"] not in ("calibration", "control", "holdout")
    ):
        raise ValueError("serving evidence requires an explicit vLLM serving phase/role")
    calibrated = run["role"] == "calibration"
    target_phase = "context" if phase == "prefill" else "generation"
    version = expected_runtime_version(run)
    manifest = json.loads(_local(root, "manifest.json").read_bytes())
    if manifest != build_model_manifest(backend, fmt, tp, version):
        raise ValueError("serving manifest differs from its frozen native production model")
    entries = _entries(manifest, target_phase)
    provenance = json.loads(_local(root, "provenance.json").read_bytes())
    files = _required_files(tp, backend) - {f"rank-{rank}.jsonl" for rank in range(tp)}
    files.add("resolved-config-node0.json")
    snapshots, forwards = {}, {}
    for rank in range(tp):
        name = f"vllm-graph-policy-rank-{rank}.json"
        snapshot = validate_snapshot(json.loads(_local(root, name).read_bytes()))
        files.add(name)
        snapshots[rank] = snapshot
        if (snapshot["tp_rank"], snapshot["tp_size"], snapshot["backend_version"]) != (rank, tp, version):
            raise ValueError("serving policy differs from the frozen native worker/runtime")
        files.add(f"state-layout-rank-{rank}.json")
        forward_name = f"forward-rank-{rank}.jsonl"
        files.add(forward_name)
        targets = {}
        for row in iter_records(_local(root, forward_name)):
            check_serving_dispatch(row, snapshot)
            if row["stage"] == "measure":
                if row["invocation"] in targets or row["phase"] != target_phase:
                    raise ValueError("serving target is duplicated or belongs to another frozen phase")
                targets[row["invocation"]] = row
        none_mode = any(row["runtime_mode"] == "NONE" for row in targets.values())
        if none_mode and any(row["runtime_mode"] != "NONE" for row in targets.values()):
            raise ValueError("native NONE measured producer cannot mix graph calibration targets")
        none_units = (
            _none_observations(root, rank, targets, entries, provenance, files, calibrated=calibrated)
            if none_mode
            else {}
        )
        full = _captures(root, rank, snapshot, manifest, provenance, files) if calibrated and not none_mode else {}
        piecewise = (
            _piecewise_captures(root, rank, snapshot, manifest, provenance, files, full)
            if calibrated and not none_mode
            else {}
        )
        if calibrated and not none_mode:
            graph_name = f"graph-forward-rank-{rank}.jsonl"
            files.add(graph_name)
            records = iter_records(_local(root, graph_name))
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
                raise ValueError("serving activity differs from its complete native target forward")
            observed.add(row["invocation"])
            shape = check_serving_dispatch(row, snapshot)
            if (
                row.get("tp_rank") != rank
                or row.get("ops_instrumented") is not calibrated
                or row.get("whole_forward_boundary") != BOUNDARY
                or row.get("gpu_completed") is not True
                or not _elapsed(row.get("whole_forward_gpu_ms"), positive=True)
                or any(row.get(key) != value for key, value in provenance.items())
            ):
                raise ValueError("serving observation lacks actual source/completion/timing identity")
            binding = None
            if calibrated and none_mode:
                binding = none_units[row["invocation"]]
            elif calibrated:
                registry, artifact = (
                    full[canonical_json(shape)] if shape["cg_mode"] == "FULL" else piecewise[shape["num_tokens"]]
                )
                method = (
                    "native_cupti_graph_nodes_and_external_logits"
                    if shape["cg_mode"] == "FULL"
                    else "native_cupti_piecewise_graphs_eager_and_external_logits"
                )
                if (
                    row.get("capture_registry_file") != artifact["file"]
                    or row.get("capture_registry_sha256") != artifact["sha256"]
                    or row.get("logits_source_sha256") != LOGITS_SOURCE_PIN
                    or row.get("profiled") is not True
                    or row.get("measurement_method") != method
                ):
                    raise ValueError("serving replay lacks its exact original capture/trace method")
                binding = _compact_units(_replay_binding(root, row, registry, files), entries)
                ownership = _capture_ownership(registry, entries, shape["cg_mode"])
                for name, unit in binding.items():
                    unit["source_ownership_sha256"] = ownership[name]
            elif (
                any(
                    row.get(key) is not None
                    for key in ("replay_nodes", "capture_registry_file", "capture_registry_sha256")
                )
                or row.get("profiled", False) is not False
            ):
                raise ValueError("independent serving truth contains operation profiling")
            key = row["benchmark_id"], row["repetition"]
            ranks = forwards.setdefault(key, {})
            if rank in ranks:
                raise ValueError("serving point repetition was observed twice")
            compact = {key: value for key, value in row.items() if key not in ("requests", "replay_nodes")}
            compact["binding"] = binding
            compact["token_witness"] = [
                {
                    "input_sha256": request["input_tokens_sha256"],
                    "prompt_sha256": sha256_json(request["prompt_token_ids"]),
                    "sampled_token_id": request["sampled_token_id"],
                }
                for request in row["requests"]
            ]
            ranks[rank] = compact
        if observed != targets.keys():
            raise ValueError("serving activity omits an actual completed target forward")
    execution = execution_policy(root)
    policy = build_serving_policy(snapshots, manifest, provenance, execution)
    for ranks in forwards.values():
        if set(ranks) != set(range(tp)):
            raise ValueError("serving forward omits actual TP workers")
        first = ranks[0]
        fields = (
            "phase",
            "runtime_mode",
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
            "token_witness",
        )
        if any(any(row.get(key) != first.get(key) for key in fields) for row in ranks.values()):
            raise ValueError("serving workers cannot join one actual native request forward")
    attempt = {
        "schema": "glm53flash_serving_attempt_v1",
        "graph_policy_sha256": sha256_json(policy),
        "files": {name: file_sha256(_local(root, name)) for name in sorted(files)},
    }
    proof = {
        **execution,
        "policy": policy,
        "snapshots": snapshots,
        "native_snapshot": policy["native_policy"],
        "provenance": provenance,
        "manifest": manifest,
        "forwards": forwards,
        "files": files,
        "policy_evidence": attempt,
        "policy_evidence_sha256": sha256_json(attempt),
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
    }
    from collector.glm53flash_observation_partition import check_native_proof

    check_native_proof(run, proof)
    if "observation_leaf" in run:
        proof["observation_leaf"] = run["observation_leaf"]
    return proof


def profile_control(root, proof, control_root, control_run):
    """Retain profiled/unprofiled elapsed differences, without fitting unit costs."""
    from collector.glm53flash_validation import load_native

    if control_run["role"] != "control" or control_run["spec"].get("ops_execution_mode") != "native_serving":
        raise ValueError("graph export requires an independent unprofiled same-calibration control")
    from collector.glm53flash_observation_partition import check_control_pair, freeze_leaf_run

    check_control_pair(proof, control_run)
    if "observation_leaf" in control_run:
        control_run = freeze_leaf_run(control_run)
    native = load_native(control_run, control_root)
    if Path(native["evidence_root"]) != control_root.resolve() or control_root.resolve() == root.resolve():
        raise ValueError("graph profiling control must retain a separate original native run")
    control = read_serving_run(control_root, control_run)
    same_execution_policy(proof, control)
    if control["policy"] != proof["policy"]:
        raise ValueError("graph profiling control changes actual native policy/runtime identity")
    if control["forwards"].keys() != proof["forwards"].keys():
        raise ValueError("graph profiling control omits point repetitions")
    samples = defaultdict(list)
    calibration_ids, control_ids = set(), set()
    fields = (
        "phase",
        "runtime_mode",
        "benchmark_id",
        "repetition",
        "sampling_role",
        "batch_size",
        "prefix_lengths",
        "query_lengths",
        "num_padded_tokens",
        "token_witness",
        "corpus_sha256",
    )
    for key, ranks in proof["forwards"].items():
        other = control["forwards"][key]
        if ranks.keys() != other.keys() or any(
            any(row[field] != other[rank][field] for field in fields) for rank, row in ranks.items()
        ):
            raise ValueError("graph profiling control differs in actual cohort/tokens/dispatch")
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
        "schema": "glm53flash_serving_profile_control_v1",
        "calibration_timing": (
            "unprofiled_native_module_cuda_events"
            if {row["runtime_mode"] for group in proof["forwards"].values() for row in group.values()} == {"NONE"}
            else "profiled_native_graph_activity"
        ),
        "execution_policy": control["execution_policy"],
        "evidence_root": str(control_root.resolve()),
        "frozen_run": control_run,
        "receipts": native["receipts"],
        "results": results,
        "timing_equivalence": "REPORTED_NOT_ASSUMED",
        "accuracy_acceptance": "NOT_EVALUATED",
    }


def require_none_timing_control(proof, control):
    """Require the original symmetric 5% control for measured NONE prefill.

    Keep the v1 all-point report unchanged. Call only after preserving it, or
    after independently rederiving an existing receipt from original evidence.
    Graph ratios remain descriptive and have no threshold here.
    """
    none_points = {
        bid
        for (bid, _), ranks in proof["forwards"].items()
        for row in ranks.values()
        if row["phase"] == "context" and row["runtime_mode"] == "NONE"
    }
    if not none_points:
        return
    results = control.get("results")
    expected = {bid for bid, _ in proof["forwards"]}
    if (
        not isinstance(results, list)
        or any(not isinstance(row, dict) or type(row.get("benchmark_id")) is not int for row in results)
        or len(results) != len(expected)
        or {row["benchmark_id"] for row in results} != expected
    ):
        raise ValueError("native NONE timing control lacks complete original point results")
    failed = []
    for row in results:
        if row["benchmark_id"] not in none_points:
            continue
        observed, original = row.get("profiled_median_ms"), row.get("control_median_ms")
        if (
            not _elapsed(observed, positive=True)
            or not _elapsed(original, positive=True)
            or not isinstance(row.get("samples"), list)
            or len(row["samples"]) != 10
            or abs(observed - original) > 0.05 * original
        ):
            failed.append(row["benchmark_id"])
    if failed:
        raise ValueError(
            "native NONE prefill observation failed the original five-percent independent timing control "
            f"at benchmark IDs {sorted(failed)}"
        )


def export_serving(
    root: Path, run: dict, output: Path, *, control_root: Path, control_run: dict, lookup_contract=None
) -> dict:
    """Export only complete real calibration; this does not certify accuracy."""
    if lookup_contract not in (None, LOOKUP_CONTRACT):
        raise ValueError("unknown serving bounded lookup contract")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector.glm53flash_validation import _load_native

    if run["role"] != "calibration" or run["spec"].get("ops_execution_mode") != "native_serving":
        raise ValueError("graph table export requires explicit native graph calibration")
    if output.name != BASENAME or output.exists():
        raise ValueError("graph export requires a new canonical table path")
    native = _load_native(run, root, calibration_evidence=False)
    if Path(native["evidence_root"]) != root.resolve():
        raise ValueError("graph export root differs from its frozen native run")
    proof = read_serving_run(root, run)
    control = profile_control(root, proof, control_root, control_run)
    control_path = root / "serving-profile-control.json"
    with control_path.open("x") as stream:
        stream.write(canonical_json(control))
    require_none_timing_control(proof, control)
    _, selection = aggregate_serving(proof, evidence_sha256="0" * 64)
    selection_path = root / "serving-rank-selection.json"
    with selection_path.open("x") as stream:
        stream.write(canonical_json(selection))
    files = proof["files"] | {row["path"] for row in native["receipts"]} | {selection_path.name, control_path.name}
    receipt = {
        "schema": "glm53flash_serving_calibration_v1",
        "graph_policy_sha256": sha256_json(proof["policy"]),
        "policy_evidence_sha256": proof["policy_evidence_sha256"],
        "request_set": native["runtime_run_id"],
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
        "files": {name: file_sha256(_local(root, name)) for name in sorted(files)},
    }
    evidence = root / "serving-calibration-evidence.json"
    with evidence.open("x") as stream:
        stream.write(canonical_json(receipt))
    rows, _ = aggregate_serving(proof, evidence_sha256=file_sha256(evidence))
    rows = analysis_rows(proof, rows, lookup_contract)
    pq.write_table(pa.Table.from_pylist(rows), output)
    return {"rows": len(rows), "table_sha256": file_sha256(output), "accuracy_acceptance": "NOT_EVALUATED"}


def bind_calibration(paths, run, native):
    import pyarrow.parquet as pq

    root = Path(native["evidence_root"])
    proof = read_serving_run(root, run)
    receipt = verify_evidence(root, proof)
    if receipt["request_set"] != native["runtime_run_id"] or receipt["source_plan_sha256"] != run["plan"]["sha256"]:
        raise ValueError("graph calibration evidence belongs to another frozen native run")
    expected, _ = aggregate_serving(proof, evidence_sha256=file_sha256(root / "serving-calibration-evidence.json"))
    selected = []
    tables = []
    identity = run["key"][:3]
    for path in paths:
        if path.name == BASENAME:
            tables.append({"path": str(path), "sha256": file_sha256(path)})
            for row in pq.read_table(path).to_pylist():
                policy = json.loads(row["graph_policy"])
                if (policy["backend"], policy["checkpoint_format"], policy["tp_size"]) == tuple(identity):
                    if policy.get("schema_version") != 3:
                        raise ValueError("serving calibration cannot mix legacy graph profiles")
                    if row["phase"] == ("context" if run["key"][3] == "prefill" else "generation"):
                        selected.append(row)
    lookup_contract = table_lookup_contract(selected)
    expected = analysis_rows(proof, expected, lookup_contract)
    if sorted(selected, key=canonical_json) != sorted(expected, key=canonical_json):
        raise ValueError("consumer graph table differs from original native activity measurements")
    return {
        "rows": len(selected),
        "tables": tables,
        "graph_policy_sha256": sha256_json(proof["policy"]),
        "evidence_sha256": file_sha256(root / "serving-calibration-evidence.json"),
        "source_plan_sha256": run["plan"]["sha256"],
        "native_runtime_run_id": native["runtime_run_id"],
        **({"lookup_contract": lookup_contract} if lookup_contract else {}),
    }


def verify_evidence(root, proof):
    """Recompute rank selection, control and every original raw file hash."""
    receipt = json.loads(_local(root, "serving-calibration-evidence.json").read_bytes())
    if (
        receipt.get("schema") != "glm53flash_serving_calibration_v1"
        or receipt.get("graph_policy_sha256") != sha256_json(proof["policy"])
        or receipt.get("policy_evidence_sha256") != proof["policy_evidence_sha256"]
        or any(receipt.get(key) != proof[key] for key in ("source_plan_sha256", "corpus_sha256"))
    ):
        raise ValueError("serving calibration changed its policy/original attempt identity")
    request_sets = {row["request_set"] for group in proof["forwards"].values() for row in group.values()}
    if request_sets != {receipt.get("request_set")}:
        raise ValueError("serving calibration belongs to another native request run")
    files = receipt.get("files", {})
    if not proof["files"] <= files.keys() or any(
        file_sha256(_local(root, name)) != digest for name, digest in files.items()
    ):
        raise ValueError("serving calibration original evidence is incomplete or changed")
    _, selection = aggregate_serving(proof, evidence_sha256=file_sha256(root / "serving-calibration-evidence.json"))
    if json.loads(_local(root, "serving-rank-selection.json").read_bytes()) != selection:
        raise ValueError("serving rank selection differs from actual whole-forward intervals")
    control = json.loads(_local(root, "serving-profile-control.json").read_bytes())
    if "serving-profile-control.json" not in files or control.get("schema") != "glm53flash_serving_profile_control_v1":
        raise ValueError("serving calibration lacks its independent unprofiled native control")
    if profile_control(root, proof, Path(control["evidence_root"]), control["frozen_run"]) != control:
        raise ValueError("serving control differs from original independent execution")
    require_none_timing_control(proof, control)
    return receipt


def predict_homogeneous(run, base, config, calibration_native, calibration_binding):
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

    if run["spec"].get("ops_execution_mode") != "native_serving" or run["role"] != "holdout":
        raise ValueError("graph prediction requires explicit independent graph holdout")
    if len(config["systems_paths"]) != 1:
        raise ValueError("initial graph prediction requires one fully receipted calibration root")
    holdout = load_native(run, base)
    same_execution_policy(calibration_native, holdout)
    policy = calibration_native["graph_policy"]
    if policy.get("schema_version") != 3 or holdout["graph_policy"] != policy:
        raise ValueError("independent serving holdout changed its initialized calibration policy")
    from collector.glm53flash_serving_shards import validate_prediction_binding

    validate_prediction_binding(calibration_native, calibration_binding)
    if (
        calibration_binding.get("graph_policy_sha256") != sha256_json(policy)
        or not calibration_binding.get("tables")
        or any(file_sha256(Path(item["path"])) != item["sha256"] for item in calibration_binding["tables"])
    ):
        raise ValueError("serving prediction lacks the bound actual calibration tables")
    cfg = ForwardPassPerfModelConfig(**config)
    if (
        (cfg.backend, cfg.backend_version, cfg.tp, cfg.database_mode, cfg.estimation_mode, cfg.fallback_policy)
        != (policy["backend"], policy["backend_version"], policy["tp_size"], "SILICON", "op_level", "deny")
        or not cfg.strict_provenance
        or cfg.nextn
        or cfg.speculation
        or cfg.estimator_config
        or cfg.decoder_replay
        or cfg.pp != 1
        or cfg.attention_dp != 1
        or cfg.moe_tp_size != policy["tp_size"]
        or cfg.moe_ep_size != 1
        or cfg.model != CHECKPOINTS[policy["checkpoint_format"]][0]
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
    lookup_contract = calibration_binding.get("lookup_contract")
    if lookup_contract not in (None, LOOKUP_CONTRACT):
        raise ValueError("unknown serving prediction lookup contract")
    rows, prediction_evidence = {}, {}
    for point in run["points"]:
        try:
            batch, total = point["batch_size"], point["total_kv_read_tokens"]
            context = run["key"][3] == "prefill"
            if (
                type(batch) is not int
                or batch < 1
                or type(total) is not int
                or total < 0
                or total % batch
                or point["point_type"] != ("prefill" if context else "decode")
                or type(point["total_prefill_tokens"]) is not int
                or (context and (point["total_prefill_tokens"] < batch or point["total_prefill_tokens"] % batch))
                or (not context and point["total_prefill_tokens"] != 0)
            ):
                raise ValueError("serving prediction requires complete homogeneous native B/Q/P")
            query, prefix = point["total_prefill_tokens"] // batch if context else 1, total // batch
            if point.get("partition") is not None or point.get("rows") not in (None, [[query, prefix]] * batch):
                raise ValueError("serving prediction cannot replace heterogeneous native requests")
            value = (
                engine.predict_prefill_latency(batch, prefix + query, prefix)
                if context
                else engine.predict_decode_latency(batch, prefix, 2)
            )
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("public graph consumer returned no positive finite latency")
            if engine.last_provenance() is not None:
                raise ValueError("graph consumer fired a non-silicon fallback")
            if lookup_contract:
                audit = engine.glm53flash_lookup_audit("context" if context else "generation", batch, query, prefix)
                if (
                    audit.get("lookup_contract") != lookup_contract
                    or audit.get("graph_policy_sha256") != sha256_json(policy)
                    or audit.get("native_policy_sha256") != policy["native_policy_sha256"]
                    or len(audit.get("operations", [])) != 278
                    or not math.isclose(sum(op["latency_ms"] for op in audit["operations"]), value, rel_tol=1e-12)
                ):
                    raise ValueError("serving endpoint audit differs from the actual public prediction")
                prediction_evidence[point["benchmark_id"]] = audit
            rows[point["benchmark_id"]] = {"prediction_ms": value}
        except Exception as error:
            rows[point["benchmark_id"]] = {"error": f"{type(error).__name__}: {error}"}
    return {
        "rows": rows,
        "calibration_binding": calibration_binding,
        **({"prediction_evidence": prediction_evidence} if lookup_contract else {}),
        "diagnostics": {
            "consumer": "public_EngineHandle_homogeneous_prefill_or_decode",
            "graph_policy_sha256": sha256_json(policy),
            "composition": "disjoint_native_unit_unions_additive_approximation",
            "interpolation": lookup_contract or "LEGACY_SAME_DISPATCH_PREFIX_ONLY",
        },
    }
