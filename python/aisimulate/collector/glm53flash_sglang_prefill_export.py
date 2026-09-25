# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce SG context schema4 from original native event and trace evidence.

Original reader for sgl-project/sglang
94602c9c2b7cbdb8efd5c52802dac6a1c180089e (Apache-2.0), source paths in
glm53flash_sglang_prefill_activity.SOURCE_PINS. No native compute copied;
see THIRD_PARTY_NOTICES.md. Legacy eager rows cannot enter this contract.
"""

from __future__ import annotations

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
    validate_row,
)
from collector.glm53flash_graph_export import _local
from collector.glm53flash_graph_nodes import trace_forward_identity
from collector.glm53flash_graph_policy import NATIVE_SOURCE_SHA256
from collector.glm53flash_jsonl import file_sha256, iter_records
from collector.glm53flash_sglang_prefill_activity import (
    METHOD,
    MODEL_CONTRACT,
    SOURCE_PINS,
    _calls,
    bind_prefill_activity,
    dispatch_signatures,
)

BASENAME = "glm53flash_sglang_prefill_perf.parquet"
BOUNDARY = "embedding_to_logits_gpu_v1"
SCOPE = "native_sglang_prefill_units_v1"
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
    "activity_count",
    "sample_count",
    "dispatch_fingerprint",
    "measurement_method",
    "prefill_policy",
    "prefill_policy_sha256",
    "dataset_role",
    "aggregation_policy",
    "rank_selection_sha256",
    "evidence_sha256",
    "policy_evidence_sha256",
    "measurement_scope",
)


def _elapsed(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json(root, name, files):
    path = _local(root, name)
    files.add(name)
    return json.loads(path.read_bytes())


def build_policy(manifest, provenance, execution, allocator=None):
    from collector.fpm_forward.glm53flash_validation import _sglang_execution_policy

    formats = {json.loads(row["geometry"])["checkpoint_format"] for row in manifest["phases"]["context"]}
    if len(formats) != 1:
        raise ValueError("native prefill manifest mixes checkpoint formats")
    fmt = formats.pop()
    tp = manifest["tp_size"]
    if (
        type(tp) is not int
        or tp not in (2, 4)
        or canonical_json(manifest) != canonical_json(build_model_manifest("sglang", fmt, tp))
    ):
        raise ValueError("native prefill manifest is not the complete production physical model")
    expected = {
        key: manifest[key]
        for key in ("backend", "backend_version", "backend_revision", "checkpoint_revision", "config_sha256")
    }
    if (
        any(provenance.get(key) != value for key, value in expected.items())
        or provenance.get("source_sha256") != NATIVE_SOURCE_SHA256
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", provenance.get("runtime_digest", ""))
    ):
        raise ValueError("native prefill runtime/checkpoint/source provenance differs")
    if (
        execution.get("cuda_graph_config", {}).get("prefill", {}).get("backend") != "disabled"
        or execution.get("cuda_graph_config", {}).get("decode", {}).get("backend") != "full"
    ):
        raise ValueError("native prefill requires actual disabled context and FULL decode policy")
    policy = _sglang_execution_policy(execution, allocator)
    return {
        "schema_version": 4,
        **expected,
        "checkpoint_format": fmt,
        "tp_size": tp,
        "runtime_digest": provenance["runtime_digest"],
        "source_sha256": NATIVE_SOURCE_SHA256,
        "source_pins": dict(SOURCE_PINS),
        "timing_boundary": BOUNDARY,
        "execution_policy_sha256": policy["execution_policy"]["sha256"],
        "prefill_backend": "disabled",
        "decode_backend": "full",
        "native_model_contract": dict(MODEL_CONTRACT),
    }


def _geometry(row):
    batch, queries, prefixes = row.get("batch_size"), row.get("query_lengths"), row.get("prefix_lengths")
    if (
        type(batch) is not int
        or batch < 1
        or not isinstance(queries, list)
        or not isinstance(prefixes, list)
        or len(queries) != batch
        or len(prefixes) != batch
        or any(type(value) is not int for value in queries + prefixes)
        or len(set(queries)) != 1
        or len(set(prefixes)) != 1
    ):
        raise ValueError("native prefill requires actual homogeneous B/Q/P coordinates")
    query, prefix = queries[0], prefixes[0]
    if (
        query <= 0
        or prefix < 0
        or prefix + query > 131072
        or row.get("phase") != "context"
        or row.get("runtime_mode") != "NONE"
        or row.get("used_cuda_graph") is not False
        or type(row.get("num_padded_tokens")) is not int
        or row["num_padded_tokens"] != batch * query
    ):
        raise ValueError("native prefill cannot relabel graph/padded/out-of-range state")
    return batch, query, prefix


def read_prefill_run(root, run):
    """Rebuild target proof; public callers also require the common seed reader."""
    from collector.fpm_forward.glm53flash_validation import _sglang_execution_policy

    root = Path(root)
    if (
        run["key"][0] != "sglang"
        or run["key"][3] != "prefill"
        or run["spec"].get("ops_execution_mode") != "native_eager_prefill"
        or run["role"] not in ("calibration", "control", "holdout")
    ):
        raise ValueError("native prefill reader requires explicit SG context purpose")
    from collector.glm53flash_validation import check_sglang_forward_allocator, sglang_allocator_evidence

    files = set()
    allocator = sglang_allocator_evidence(root, run, files)
    manifest = _json(root, "manifest.json", files)
    provenance = _json(root, "provenance.json", files)
    execution = _json(root, "sglang-resolved-config.json", files)
    policy = build_policy(manifest, provenance, execution, allocator["normalized"])
    if (policy["checkpoint_format"], policy["tp_size"]) != tuple(run["key"][1:3]):
        raise ValueError("native prefill policy belongs to another frozen cell")
    entries = manifest["phases"]["context"]
    runtime = manifest["runtime_operations"]["context"]
    if (
        len(entries) != 366
        or len({row["name"] for row in entries}) != 366
        or len(runtime) != 1
        or runtime[0]["name"] != "native_graph_setup"
    ):
        raise ValueError("native prefill requires 366 physical units and exactly one runtime marker")
    expected_entries = {row["name"]: row for row in entries}
    expected_points = {}
    for point in run["points"]:
        bid, batch, query, prefix = (
            point.get(key) for key in ("benchmark_id", "batch_size", "total_prefill_tokens", "total_kv_read_tokens")
        )
        if (
            any(type(value) is not int for value in (bid, batch, query, prefix))
            or bid in expected_points
            or batch < 1
            or query < batch
            or prefix < 0
            or query % batch
            or prefix % batch
            or point.get("point_type") != "prefill"
        ):
            raise ValueError("native prefill frozen points lack unique homogeneous integer coordinates")
        expected_points[bid] = (batch, query // batch, prefix // batch)
    if not expected_points:
        raise ValueError("native prefill cannot admit an empty frozen point set")
    forwards, model_files, consumed = defaultdict(dict), [], set()
    calibration = run["role"] == "calibration"
    for rank in range(policy["tp_size"]):
        model_file = f"prefill-model-rank-{rank}.json"
        identity = _json(root, model_file, files)
        if canonical_json(identity) != canonical_json(
            {"source_pins": SOURCE_PINS, "native_model_contract": MODEL_CONTRACT}
        ):
            raise ValueError("native prefill actual model/source identity differs")
        model_files.append({"rank": rank, "file": model_file, "sha256": file_sha256(_local(root, model_file))})
        name = f"forward-rank-{rank}.jsonl"
        files.add(name)
        observed, warmups = {}, {}
        for original in iter_records(_local(root, name)):
            check_sglang_forward_allocator(original, rank, allocator)
            if original.get("stage") != "measure":
                continue
            row = dict(original)
            key = row.get("benchmark_id"), row.get("repetition")
            if (
                key in observed
                or type(key[0]) is not int
                or type(key[1]) is not int
                or not 0 <= key[1] < 15
                or expected_points.get(key[0]) != _geometry(row)
                or type(row.get("tp_rank")) is not int
                or row["tp_rank"] != rank
                or row.get("gpu_completed") is not True
                or row.get("native_prefill_model_sha256") != sha256_json(identity)
                or row.get("sampling_role") != ("warmup" if key[1] < 5 else "measurement")
                or row.get("dataset_role") != ("holdout" if run["role"] == "holdout" else "calibration")
                or row.get("corpus_sha256") != run["corpus"]
                or row.get("ops_instrumented") is not calibration
                or row.get("whole_forward_boundary") != BOUNDARY
                or not _elapsed(row.get("whole_forward_gpu_ms"), positive=True)
            ):
                raise ValueError("native prefill target lacks its exact original completed run/rank/role")
            if (
                row.get("ops_execution_mode") != "native_eager_prefill"
                or row.get("prefill_measurement_contract") != METHOD
            ):
                raise ValueError("old eager observations cannot become source-bound prefill evidence")
            row["token_witness"] = [
                {
                    key: item[key]
                    for key in (
                        "prompt_token_ids",
                        "native_query_token_ids",
                        "computed_tokens_before",
                        "computed_tokens_after",
                        "input_tokens_sha256",
                        "sampled_token_id",
                    )
                }
                for item in row["requests"]
            ]
            if calibration:
                setup = row.get("native_prefill_setup", {})
                if (
                    row.get("native_prefill_measurement_contract") != METHOD
                    or setup.get("completed") is not True
                    or type(setup.get("contribution_count")) is not int
                    or setup["contribution_count"] != 1
                    or type(setup.get("buffer_size")) is not int
                    or setup["buffer_size"] != 90
                    or setup.get("dtype") != "torch.float32"
                    or setup.get("source") != "sglang.srt.utils.common.BumpAllocator.__init__"
                    or not isinstance(setup.get("device"), str)
                    or setup["device"].split(":")[0] != "cuda"
                    or not _elapsed(setup.get("latency"))
                ):
                    raise ValueError("native prefill runtime marker lacks original allocator events/arguments")
                _calls(row.get("native_prefill_calls", []), list(expected_entries))
                profile = row.get("native_prefill_profile")
                if (profile is not None) != (key[1] == 4):
                    raise ValueError("native prefill trace must be only the fifth excluded warmup")
                if profile is not None:
                    expected_name = f"prefill-profile-rank-{rank}-forward-{row['invocation']}.json"
                    if profile.get("trace_file") != expected_name or expected_name in consumed:
                        raise ValueError("native prefill trace was reused across rank/forward identity")
                    consumed.add(expected_name)
                    trace = _json(root, expected_name, files)
                    metadata = trace.get("aisim_native_prefill", {})
                    if (
                        file_sha256(_local(root, expected_name)) != profile.get("trace_sha256")
                        or trace.get("aisim_native_forward") != trace_forward_identity(row)
                        or metadata
                        != {
                            "measurement_contract": METHOD,
                            "model_identity_sha256": sha256_json(identity),
                            "native_calls": row["native_prefill_calls"],
                            "setup": {key: setup[key] for key in ("buffer_size", "dtype", "device", "source")},
                            "failed": False,
                        }
                    ):
                        raise ValueError("native prefill trace changed its actual source/model/forward metadata")
                    binding = bind_prefill_activity(
                        trace["traceEvents"], row["native_prefill_calls"], list(expected_entries)
                    )
                    if binding != profile.get("binding"):
                        raise ValueError("native prefill activity differs from original trace rederivation")
                    warmups[key[0]] = binding
            elif any(key in row for key in ("native_prefill_setup", "native_prefill_calls", "native_prefill_profile")):
                raise ValueError("native prefill control/holdout cannot contain operation instrumentation")
            observed[key] = row
        if set(observed) != {(bid, sample) for bid in expected_points for sample in range(15)}:
            raise ValueError("native prefill omits original frozen 5+10 target forwards")
        if calibration:
            raw = defaultdict(dict)
            name = f"rank-{rank}.jsonl"
            files.add(name)
            for unit in iter_records(_local(root, name)):
                key = unit.get("benchmark_id"), unit.get("repetition")
                row = observed.get(key)
                entry = expected_entries.get(unit.get("name"))
                if (
                    row is None
                    or entry is None
                    or unit["name"] in raw[key]
                    or any(unit.get(field) != value for field, value in entry.items())
                    or any(
                        unit.get(field) != row.get(field)
                        for field in (
                            "invocation",
                            "tp_rank",
                            "request_ids",
                            "phase",
                            "benchmark_id",
                            "repetition",
                            "sampling_role",
                            "corpus_sha256",
                            "request_set",
                            "dataset_role",
                        )
                    )
                ):
                    raise ValueError("native prefill event row differs from its physical unit/forward")
                # Only this reader subsequently proves completed zero-work
                # calls by original API/activity ownership; legacy stays >0.
                validate_row(unit, allow_zero_latency=True)
                batch, query, prefix = _geometry(row)
                geometry = json.loads(entry["geometry"])
                attention = entry["component"] == "attention"
                expected_x = (
                    query
                    if attention
                    else batch
                    if geometry.get("token_selection") == "last_per_request"
                    else batch * query
                )
                if (unit["batch_size"], unit["prefix"], unit["x"], unit["sample_count"], unit.get("sample")) != (
                    batch if attention else 1,
                    prefix if attention else 0,
                    expected_x,
                    1,
                    key[1],
                ) or any(
                    unit.get(field) != row.get(field)
                    for field in (
                        "backend",
                        "backend_version",
                        "backend_revision",
                        "checkpoint_revision",
                        "source_sha256",
                        "config_sha256",
                        "runtime_digest",
                    )
                ):
                    raise ValueError("native prefill event geometry/runtime differs from actual forward")
                parts = [call for call in row["native_prefill_calls"] if call["operation"] == unit["name"]]
                sources = sorted({source for call in parts for source in (call["source"], *call["included_sources"])})
                excluded = [source for call in parts for source in call["excluded_collective_sources"]]
                actual_excluded = unit.get("excluded_collectives", [])
                if (
                    unit.get("kernel_source") != "+".join(sources)
                    or sorted(item.get("source", "") for item in actual_excluded) != sorted(excluded)
                    or any(not _elapsed(item.get("latency"), positive=True) for item in actual_excluded)
                ):
                    raise ValueError("native prefill measured exclusive intervals changed source/collective ownership")
                raw[key][unit["name"]] = unit
            for key, row in observed.items():
                if (
                    set(raw[key]) != set(expected_entries)
                    or row["native_prefill_calls"] != warmups[key[0]]["native_calls"]
                ):
                    raise ValueError("native prefill event/call inventory changed across original repetitions")
                proof = warmups[key[0]]
                signatures = dispatch_signatures(proof)
                binding = {}
                for name in [*expected_entries, "native_graph_setup"]:
                    count = len(proof["operation_activity_indices"][name])
                    dispatch = sha256_json(signatures[name]) if count else ""
                    if name == "native_graph_setup":
                        latency = row["native_prefill_setup"]["latency"]
                    else:
                        unit = raw[key][name]
                        latency = unit["latency"]
                        if key[1] >= 4 and (
                            unit.get("dispatch_kernels") != signatures[name]
                            or unit.get("dispatch_fingerprint") != dispatch
                        ):
                            raise ValueError(
                                "native prefill event dispatch differs from authoritative warmup ownership"
                            )
                    if not _elapsed(latency, positive=count > 0):
                        raise ValueError("native prefill zero event lacks a completed no-activity source call")
                    binding[name] = {
                        "latency": latency,
                        "dispatch": dispatch,
                        "activity_count": count,
                        "contribution_count": proof["contribution_counts"][name],
                    }
                row["binding"] = binding
        for key, row in observed.items():
            forwards[key][rank] = row
    policy_evidence = {
        "model_files": model_files,
        "resolved_config_sha256": file_sha256(_local(root, "sglang-resolved-config.json")),
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
    }
    return {
        **_sglang_execution_policy(execution, allocator["normalized"]),
        "policy": policy,
        "manifest": manifest,
        "forwards": dict(forwards),
        "files": files,
        "policy_evidence_sha256": sha256_json(policy_evidence),
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
    }


# Analysis metadata only: never added to the actual native execution policy.
LOOKUP_CONTRACT = "sglang_prefill_bounded_p_q_v1"


def table_lookup_contract(rows):
    contracts = set()
    for row in rows:
        present = "lookup_contract" in row
        if present != ("source_ownership_sha256" in row):
            raise ValueError("incomplete prefill lookup metadata")
        contract = row.get("lookup_contract")
        if present and (contract != LOOKUP_CONTRACT or not _hash(row["source_ownership_sha256"])):
            raise ValueError("unknown prefill lookup contract or source ownership")
        contracts.add(contract)
    if len(contracts) != 1:
        raise ValueError("prefill publication mixes analysis lookup contracts")
    return contracts.pop()


def analysis_rows(proof, rows, lookup_contract):
    """Bind optional lookup to original call ownership; never rewrite raw policy."""
    if lookup_contract is None:
        return rows
    if lookup_contract != LOOKUP_CONTRACT:
        raise ValueError("unknown native prefill lookup contract")
    ownership = defaultdict(set)
    for (_, repetition), ranks in sorted(proof["forwards"].items()):
        if repetition < 5:
            continue
        selected = min(ranks, key=lambda rank: (-ranks[rank]["whole_forward_gpu_ms"], rank))
        forward = ranks[selected]
        coordinates = _geometry(forward)
        for name, unit in forward["binding"].items():
            if name == "native_graph_setup":
                calls = [{key: forward["native_prefill_setup"][key] for key in ("source", "buffer_size", "dtype")}]
            else:
                calls = [
                    {
                        key: call[key]
                        for key in ("source", "included_sources", "excluded_collective_sources", "parent_operation")
                    }
                    for call in forward["native_prefill_calls"]
                    if call["operation"] == name
                ]
            if not calls:
                raise ValueError("prefill lookup lacks original source call ownership")
            ownership[(*coordinates, name)].add(
                sha256_json(
                    {
                        "operation_name": name,
                        "calls": calls,
                        "contribution_count": unit["contribution_count"],
                    }
                )
            )
    result = []
    for row in rows:
        key = tuple(row[name] for name in ("batch_size", "query_length", "prefix", "operation_name"))
        values = ownership.get(key, set())
        if len(values) != 1:
            raise ValueError("prefill point changes native source ownership across retained samples")
        result.append({**row, "lookup_contract": lookup_contract, "source_ownership_sha256": next(iter(values))})
    return result


def aggregate_prefill(proof, *, evidence_sha256):
    if not _hash(evidence_sha256) or not _hash(proof["policy_evidence_sha256"]):
        raise ValueError("native prefill aggregate lacks bound original evidence")
    entries = proof["manifest"]["phases"]["context"] + proof["manifest"]["runtime_operations"]["context"]
    groups, selections = defaultdict(list), []
    for key, ranks in sorted(proof["forwards"].items()):
        if set(ranks) != set(range(proof["policy"]["tp_size"])):
            raise ValueError("native prefill requires every actual TP rank")
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
        for rank, row in ranks.items():
            if (
                any(row[field] != first[field] for field in fields)
                or row["tp_rank"] != rank
                or row["dataset_role"] != "calibration"
                or set(row.get("binding", {})) != {item["name"] for item in entries}
            ):
                raise ValueError("native prefill rank cohorts or physical bindings differ")
        selected = min(ranks, key=lambda rank: (-ranks[rank]["whole_forward_gpu_ms"], rank))
        row = ranks[selected]
        selections.append(
            {
                "benchmark_id": key[0],
                "repetition": key[1],
                "selected_rank": selected,
                "ranks": [
                    {
                        "rank": rank,
                        "forward_id": record["forward_id"],
                        "invocation": record["invocation"],
                        "whole_forward_gpu_ms": record["whole_forward_gpu_ms"],
                    }
                    for rank, record in sorted(ranks.items())
                ],
            }
        )
        if key[1] < 5:
            continue
        batch, query, prefix = _geometry(row)
        for entry in entries:
            identity = (
                entry["component"],
                entry["name"],
                entry["geometry"],
                "context",
                "NONE",
                batch,
                query,
                prefix,
                batch * query,
                batch,
            )
            groups[identity].append((key, row["binding"][entry["name"]]))
    selection = {
        "schema": "glm53flash_sglang_prefill_rank_selection_v1",
        "aggregation_policy": WHOLE_FORWARD_RANK,
        "forwards": selections,
    }
    rows = []
    for identity, samples in sorted(groups.items()):
        signatures = {(unit["dispatch"], unit["activity_count"], unit["contribution_count"]) for _, unit in samples}
        if len(samples) != 10 or len({key for key, _ in samples}) != 10 or len(signatures) != 1:
            raise ValueError("native prefill exact named geometry mixes actual dispatch or duplicate samples")
        dispatch, count, contributions = signatures.pop()
        rows.append(
            {
                **dict(zip(KEYS, identity, strict=True)),
                "latency": statistics.median(unit["latency"] for _, unit in samples),
                "contribution_count": contributions,
                "activity_count": count,
                "sample_count": 10,
                "dispatch_fingerprint": dispatch,
                "measurement_method": METHOD,
                "prefill_policy": canonical_json(proof["policy"]),
                "prefill_policy_sha256": sha256_json(proof["policy"]),
                "dataset_role": "calibration",
                "aggregation_policy": WHOLE_FORWARD_RANK,
                "rank_selection_sha256": sha256_json(selection),
                "evidence_sha256": evidence_sha256,
                "policy_evidence_sha256": proof["policy_evidence_sha256"],
                "measurement_scope": SCOPE,
            }
        )
    if not rows:
        raise ValueError("native prefill cannot export an empty phase")
    return rows, selection


def profile_control(root, proof, control_root, control_run):
    from collector.fpm_forward.glm53flash_validation import _same_sglang_policy
    from collector.glm53flash_validation import _load_native

    if Path(root).resolve() == Path(control_root).resolve() or control_run["role"] != "control":
        raise ValueError("native prefill requires an independently launched control")
    native = _load_native(control_run, control_root, calibration_evidence=False)
    control = read_prefill_run(control_root, control_run)
    _same_sglang_policy(proof, control, "prefill calibration/control")
    if proof["policy"] != control["policy"] or proof["forwards"].keys() != control["forwards"].keys():
        raise ValueError("native prefill control changed policy or original point repetitions")
    identities, control_ids, samples = set(), set(), defaultdict(list)
    for key, ranks in proof["forwards"].items():
        other = control["forwards"][key]
        if ranks.keys() != other.keys():
            raise ValueError("native prefill control omitted actual workers")
        for rank, row in ranks.items():
            compare = other[rank]
            fields = (
                "batch_size",
                "query_lengths",
                "prefix_lengths",
                "num_padded_tokens",
                "corpus_sha256",
            )
            # Terminal sampling happens after the embedding-to-logits boundary.
            # Preserve its raw evidence and common-reader state/TP checks, but
            # compare the actual measured inputs rather than requiring equal
            # independently produced terminal outputs.
            input_fields = (
                "prompt_token_ids",
                "native_query_token_ids",
                "computed_tokens_before",
                "computed_tokens_after",
                "input_tokens_sha256",
            )
            inputs = [{field: token[field] for field in input_fields} for token in row["token_witness"]]
            control_inputs = [{field: token[field] for field in input_fields} for token in compare["token_witness"]]
            if (
                any(row[field] != compare[field] for field in fields)
                or inputs != control_inputs
                or row["request_set"] == compare["request_set"]
            ):
                raise ValueError("native prefill control changed native inputs/state or reused its run")
        identities.update(ranks[0]["request_ids"])
        control_ids.update(other[0]["request_ids"])
        if key[1] >= 5:
            samples[key[0]].append(
                {
                    "repetition": key[1],
                    "observed_ms": max(row["whole_forward_gpu_ms"] for row in ranks.values()),
                    "control_ms": max(row["whole_forward_gpu_ms"] for row in other.values()),
                }
            )
    if identities & control_ids:
        raise ValueError("native prefill independent control reused request identities")
    results = []
    for bid, rows in sorted(samples.items()):
        observed = statistics.median(row["observed_ms"] for row in rows)
        original = statistics.median(row["control_ms"] for row in rows)
        if len(rows) != 10 or abs(observed / original - 1) > 0.05:
            raise ValueError("native prefill observation failed the original five-percent independent timing control")
        results.append(
            {
                "benchmark_id": bid,
                "observed_median_ms": observed,
                "control_median_ms": original,
                "ratio": observed / original,
                "samples": rows,
            }
        )
    return {
        "schema": "glm53flash_sglang_prefill_control_v1",
        "evidence_root": str(Path(control_root).resolve()),
        "frozen_run": control_run,
        "receipts": native["receipts"],
        "execution_policy": control["execution_policy"],
        "results": results,
        "accuracy_acceptance": "NOT_EVALUATED",
    }


def export_prefill(root, run, output, *, control_root, control_run, lookup_contract=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector.glm53flash_validation import _load_native

    root, output = Path(root), Path(output)
    if lookup_contract not in (None, LOOKUP_CONTRACT):
        raise ValueError("unknown native prefill lookup contract")
    if run["role"] != "calibration" or output.name != BASENAME or output.exists():
        raise ValueError("native prefill export needs fresh canonical calibration output")
    native = _load_native(run, root, calibration_evidence=False)
    if Path(native["evidence_root"]) != root.resolve():
        raise ValueError("native prefill frozen root differs from export root")
    proof = read_prefill_run(root, run)
    control = profile_control(root, proof, Path(control_root), control_run)
    _, selection = aggregate_prefill(proof, evidence_sha256="0" * 64)
    for name, value in (("prefill-profile-control.json", control), ("prefill-rank-selection.json", selection)):
        with (root / name).open("x") as stream:
            stream.write(canonical_json(value))
    files = (
        proof["files"]
        | {row["path"] for row in native["receipts"]}
        | {"prefill-profile-control.json", "prefill-rank-selection.json"}
    )
    receipt = {
        "schema": "glm53flash_sglang_prefill_calibration_v1",
        "prefill_policy_sha256": sha256_json(proof["policy"]),
        "policy_evidence_sha256": proof["policy_evidence_sha256"],
        "request_set": native["runtime_run_id"],
        "source_plan_sha256": run["plan"]["sha256"],
        "corpus_sha256": run["corpus"],
        "files": {name: file_sha256(_local(root, name)) for name in sorted(files)},
    }
    path = root / "prefill-calibration-evidence.json"
    with path.open("x") as stream:
        stream.write(canonical_json(receipt))
    rows, _ = aggregate_prefill(proof, evidence_sha256=file_sha256(path))
    rows = analysis_rows(proof, rows, lookup_contract)
    pq.write_table(pa.Table.from_pylist(rows), output)
    return {"rows": len(rows), "table_sha256": file_sha256(output), "accuracy_acceptance": "NOT_EVALUATED"}


def verify_evidence(root, proof):
    receipt = json.loads(_local(root, "prefill-calibration-evidence.json").read_bytes())
    if (
        receipt.get("schema") != "glm53flash_sglang_prefill_calibration_v1"
        or receipt.get("prefill_policy_sha256") != sha256_json(proof["policy"])
        or receipt.get("policy_evidence_sha256") != proof["policy_evidence_sha256"]
        or any(receipt.get(key) != proof[key] for key in ("source_plan_sha256", "corpus_sha256"))
    ):
        raise ValueError("native prefill calibration policy/attempt identity changed")
    if {row["request_set"] for ranks in proof["forwards"].values() for row in ranks.values()} != {
        receipt.get("request_set")
    }:
        raise ValueError("native prefill evidence reused another runtime run")
    files = receipt.get("files", {})
    if not (proof["files"] | {"prefill-profile-control.json", "prefill-rank-selection.json"}) <= files.keys() or any(
        file_sha256(_local(root, name)) != digest for name, digest in files.items()
    ):
        raise ValueError("native prefill evidence omitted or changed original files")
    _, selection = aggregate_prefill(proof, evidence_sha256=file_sha256(root / "prefill-calibration-evidence.json"))
    if json.loads(_local(root, "prefill-rank-selection.json").read_bytes()) != selection:
        raise ValueError("native prefill rank selection changed")
    control = json.loads(_local(root, "prefill-profile-control.json").read_bytes())
    if profile_control(root, proof, Path(control["evidence_root"]), control["frozen_run"]) != control:
        raise ValueError("native prefill control differs from independent original evidence")
    return receipt


def publish_prefill(root, run, output, *, lookup_contract=None):
    """Publish analysis from existing original evidence without changing raw files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from collector.glm53flash_validation import load_native

    root, output = Path(root), Path(output)
    if run["role"] != "calibration" or output.name != BASENAME or output.exists():
        raise ValueError("native prefill publication requires fresh canonical calibration output")
    if lookup_contract not in (None, LOOKUP_CONTRACT):
        raise ValueError("unknown native prefill lookup contract")
    load_native(run, root)
    proof = read_prefill_run(root, run)
    verify_evidence(root, proof)
    rows, _ = aggregate_prefill(proof, evidence_sha256=file_sha256(root / "prefill-calibration-evidence.json"))
    rows = analysis_rows(proof, rows, lookup_contract)
    pq.write_table(pa.Table.from_pylist(rows), output)
    return {
        "rows": len(rows),
        "table_sha256": file_sha256(output),
        "accuracy_acceptance": "NOT_EVALUATED",
        **({"lookup_contract": lookup_contract} if lookup_contract else {}),
    }


def bind_calibration(paths, run, native):
    import pyarrow.parquet as pq

    root = Path(native["evidence_root"])
    proof = read_prefill_run(root, run)
    receipt = verify_evidence(root, proof)
    if receipt["request_set"] != native["runtime_run_id"] or receipt["source_plan_sha256"] != run["plan"]["sha256"]:
        raise ValueError("native prefill calibration differs from frozen run")
    expected, _ = aggregate_prefill(proof, evidence_sha256=file_sha256(root / "prefill-calibration-evidence.json"))
    selected, tables = [], []
    for path in paths:
        if path.name != BASENAME:
            continue
        tables.append({"path": str(path), "sha256": file_sha256(path)})
        for row in pq.read_table(path).to_pylist():
            policy = json.loads(row["prefill_policy"])
            if (policy["backend"], policy["checkpoint_format"], policy["tp_size"]) == tuple(run["key"][:3]):
                selected.append(row)
    lookup_contract = table_lookup_contract(selected)
    expected = analysis_rows(proof, expected, lookup_contract)
    if sorted(selected, key=canonical_json) != sorted(expected, key=canonical_json):
        raise ValueError("consumer prefill table differs from original native events/ownership")
    return {
        "rows": len(selected),
        "tables": tables,
        "prefill_policy_sha256": sha256_json(proof["policy"]),
        "evidence_sha256": file_sha256(root / "prefill-calibration-evidence.json"),
        "source_plan_sha256": run["plan"]["sha256"],
        "native_runtime_run_id": native["runtime_run_id"],
        **({"lookup_contract": lookup_contract} if lookup_contract else {}),
    }


def predict_homogeneous(run, base, config, calibration_native, binding):
    from aisimulate_core.sdk.engine import EngineHandle
    from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig
    from collector.fpm_forward.glm53flash_validation import _same_sglang_policy
    from collector.glm53flash_contract import CHECKPOINTS
    from collector.glm53flash_sglang_prefill_shards import validate_prediction_binding
    from collector.glm53flash_validation import load_native

    if (
        run["role"] != "holdout"
        or run["key"][0] != "sglang"
        or run["key"][3] != "prefill"
        or run["spec"].get("ops_execution_mode") != "native_eager_prefill"
    ):
        raise ValueError("native prefill prediction requires independent homogeneous SG context")
    holdout = load_native(run, base)
    _same_sglang_policy(calibration_native, holdout, "prefill calibration/holdout")
    policy = calibration_native["prefill_policy"]
    if (
        policy != holdout["prefill_policy"]
        or binding.get("prefill_policy_sha256") != sha256_json(policy)
        or not binding.get("tables")
        or any(file_sha256(Path(item["path"])) != item["sha256"] for item in binding["tables"])
    ):
        raise ValueError("native prefill prediction lacks original calibration table/policy binding")
    validate_prediction_binding(calibration_native, binding)
    cfg = ForwardPassPerfModelConfig(**config)
    if (
        (cfg.backend, cfg.backend_version, cfg.tp, cfg.database_mode, cfg.estimation_mode, cfg.fallback_policy)
        != ("sglang", "0.5.20", policy["tp_size"], "SILICON", "op_level", "deny")
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
        or len(cfg.systems_paths) != 1
    ):
        raise ValueError("native prefill predictor configuration differs from measured exact identity")
    options = {
        key: getattr(cfg, key)
        for key in (
            "backend_version",
            "moe_tp_size",
            "moe_ep_size",
            "gemm_quant_mode",
            "moe_quant_mode",
            "fmha_quant_mode",
            "fpm_fmha_quant_mode",
            "kvcache_quant_mode",
            "comm_quant_mode",
            "attention_backend",
            "moe_backend",
            "enable_eplb",
            "wideep_num_slots",
            "kv_block_size",
            "transfer_policy",
        )
    }
    engine = EngineHandle.compile(
        cfg.model,
        cfg.system,
        cfg.backend,
        **options,
        tp_size=cfg.tp,
        pp_size=1,
        attention_dp_size=1,
        systems_path=cfg.systems_paths[0],
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
    )
    rows, prediction_evidence = {}, {}
    lookup_contract = binding.get("lookup_contract")
    if lookup_contract not in (None, LOOKUP_CONTRACT):
        raise ValueError("unknown native prefill prediction lookup contract")
    for point in run["points"]:
        try:
            batch, total_query, total_prefix = (
                point["batch_size"],
                point["total_prefill_tokens"],
                point["total_kv_read_tokens"],
            )
            if (
                type(batch) is not int
                or batch < 1
                or type(total_query) is not int
                or total_query < batch
                or total_query % batch
                or type(total_prefix) is not int
                or total_prefix < 0
                or total_prefix % batch
                or point["point_type"] != "prefill"
            ):
                raise ValueError("native prefill prediction needs exact homogeneous B/Q/P")
            query, prefix = total_query // batch, total_prefix // batch
            if point.get("partition") is not None or point.get("rows") not in (None, [[query, prefix]] * batch):
                raise ValueError("native prefill cannot replace heterogeneous request geometry")
            value = engine.predict_prefill_latency(batch, prefix + query, prefix)
            if not _elapsed(value, positive=True) or engine.last_provenance() is not None:
                raise ValueError("native prefill public query did not use complete measured silicon evidence")
            if lookup_contract:
                audit = engine.glm53flash_lookup_audit("context", batch, query, prefix)
                if (
                    audit.get("lookup_contract") != lookup_contract
                    or audit.get("native_policy_sha256") != binding["prefill_policy_sha256"]
                    or len(audit.get("operations", [])) != 367
                    or not math.isclose(sum(item["latency_ms"] for item in audit["operations"]), value, rel_tol=1e-12)
                ):
                    raise ValueError("native prefill selected endpoint audit differs from actual public prediction")
                prediction_evidence[point["benchmark_id"]] = audit
            rows[point["benchmark_id"]] = {"prediction_ms": value}
        except Exception as error:
            rows[point["benchmark_id"]] = {"error": f"{type(error).__name__}: {error}"}
    return {
        "rows": rows,
        "calibration_binding": binding,
        **({"prediction_evidence": prediction_evidence} if lookup_contract else {}),
        "diagnostics": {
            "consumer": "public_EngineHandle_homogeneous_prefill",
            "prefill_policy_sha256": sha256_json(policy),
            "interpolation": lookup_contract or "EXACT_ONLY",
            "accuracy_acceptance": "NOT_EVALUATED",
        },
    }
