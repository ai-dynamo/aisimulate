# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate SGLang's actual forward histories before publishing an FPM curve.

The schema-2 envelope is an interchange format, not a claim that SGLang uses
Dynamo's scheduler. Its latency is rank zero's native DeviceTimer interval.
All TP ranks must witness the same frozen requests, real prefixes and dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from collector.glm53flash_protocol import PROTOCOL, TIMING_BOUNDARIES

WARMUPS = 5
MEASUREMENTS = 10
TELEMETRY_POLICY = "native_device_timer_with_outside_interval_token_readback_v1"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_receipt(path: Path) -> dict:
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_receipt(parent: Path, receipt: dict) -> bytes:
    name = receipt.get("file")
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError("SGLang evidence must be an adjacent file")
    raw = (parent / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
        raise ValueError("SGLang native evidence digest mismatch")
    return raw


def read_observations(manifest: dict, traces: dict[int, bytes], points: list[dict]) -> dict:
    """Reconstruct completed prefix chains and select exact frozen occurrences."""
    mappings = manifest.get("requests")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("SGLang requires a frozen nonempty request manifest")
    if manifest.get("dataset_role") not in ("calibration", "holdout") or not manifest.get("request_set"):
        raise ValueError("SGLang request-set identity is missing")
    by_id = {point["benchmark_id"]: point for point in points}
    if len(by_id) != len(points):
        raise ValueError("SGLang duplicate benchmark point")
    expected = {}
    for rid, entry in mappings.items():
        key = (entry["benchmark_id"], entry["repetition"])
        if entry["benchmark_id"] not in by_id or not 0 <= entry["repetition"] < WARMUPS + MEASUREMENTS:
            raise ValueError("SGLang unexpected frozen request identity")
        point = by_id[entry["benchmark_id"]]
        batch = point["batch_size"]
        phase = "context" if point["point_type"] == "prefill" else "generation"
        query = point["total_prefill_tokens"] // batch if phase == "context" else 1
        prefix = point["total_kv_read_tokens"] // batch
        target = (phase, query, prefix, batch, "warmup" if entry["repetition"] < WARMUPS else "measurement")
        if (
            tuple(
                entry[k]
                for k in ("target_phase", "target_query", "target_prefix", "target_batch_size", "sampling_role")
            )
            != target
        ):
            raise ValueError("SGLang frozen request differs from benchmark geometry")
        expected.setdefault(key, set()).add(rid)
    for point in points:
        for repetition in range(WARMUPS + MEASUREMENTS):
            if len(expected.get((point["benchmark_id"], repetition), ())) != point["batch_size"]:
                raise ValueError("SGLang frozen request coverage is incomplete")
    observations = {}
    reference = None
    for rank, raw in sorted(traces.items()):
        previous = {}
        seen_forwards = set()
        selected = {}
        for line in raw.splitlines():
            record = json.loads(line)
            if record.get("tp_rank") != rank or record.get("state_protocol") != PROTOCOL:
                raise ValueError("SGLang native rank/state protocol mismatch")
            if record.get("allocated_fake_tokens") != 0 or record.get("gpu_completed") is not True:
                raise ValueError("SGLang requires a completed real forward")
            if record.get("ops_instrumented") is not False:
                raise ValueError("SGLang FPM cannot admit instrumented Ops latency")
            if record.get("timing_boundary") != TIMING_BOUNDARIES["sglang"]:
                raise ValueError("SGLang native timing boundary mismatch")
            forward_id = record.get("forward_id")
            if not isinstance(forward_id, str) or forward_id in seen_forwards:
                raise ValueError("SGLang native forward identity reused")
            seen_forwards.add(forward_id)
            requests = record.get("requests", [])
            ids = record.get("request_ids")
            if ids != [request.get("request_id") for request in requests] or len(set(ids)) != len(ids):
                raise ValueError("SGLang actual request order is invalid")
            batch = record["batch_size"]
            queries, prefixes = record["query_lengths"], record["prefix_lengths"]
            if len(ids) != batch or len(queries) != batch or len(prefixes) != batch:
                raise ValueError("SGLang native batch metadata is incomplete")
            if sum(queries) != record["total_new_tokens"] or sum(prefixes) != record["total_past_kv_tokens"]:
                raise ValueError("SGLang native token totals disagree")
            for request, query, prefix in zip(requests, queries, prefixes, strict=True):
                rid = request["request_id"]
                if rid not in mappings or type(query) is not int or query < 1 or type(prefix) is not int or prefix < 0:
                    raise ValueError("SGLang observed an unplanned request or invalid geometry")
                tokens = request.get("native_query_token_ids")
                if (
                    not isinstance(tokens, list)
                    or len(tokens) != query
                    or any(type(t) is not int or t < 0 for t in tokens)
                ):
                    raise ValueError("SGLang requires actual input token IDs")
                prior = previous.get(rid)
                if prefix:
                    if prior is None or len(prior[1]) != prefix or request.get("previous_forward_id") != prior[0]:
                        raise ValueError("SGLang real hybrid prefix chain is missing or discontinuous")
                    history = prior[1] + tokens
                else:
                    if prior is not None or request.get("previous_forward_id") is not None:
                        raise ValueError("SGLang request identity restarted")
                    history = tokens
                if request.get("same_request_real_prefix") is not True:
                    raise ValueError("SGLang native prefix was not computed by the same request")
                if (
                    request.get("computed_tokens_before") != prefix
                    or request.get("computed_tokens_after") != prefix + query
                ):
                    raise ValueError("SGLang computed history disagrees with actual dispatch")
                prompt = request.get("prompt_token_ids")
                if not isinstance(prompt, list) or not prompt or history[: len(prompt)] != prompt[: len(history)]:
                    raise ValueError("SGLang actual input differs from its prompt")
                digest = hashlib.sha256(json.dumps(history, separators=(",", ":")).encode()).hexdigest()
                if prior is not None and prompt != prior[2]:
                    raise ValueError("SGLang request prompt changed during its native history")
                if prefix >= len(prompt) and (
                    prior is None or tokens != [prior[3]] or record.get("phase") != "generation"
                ):
                    raise ValueError("SGLang native decode input differs from its preceding sampled token")
                if request.get("input_tokens_sha256") != digest:
                    raise ValueError("SGLang actual input history digest mismatch")
                if type(request.get("sampled_token_id")) is not int or request["sampled_token_id"] < 0:
                    raise ValueError("SGLang lacks an actual completed sampled token")
                previous[rid] = (forward_id, history, prompt, request["sampled_token_id"])
            if record.get("stage") != "measure":
                continue
            key = (record.get("benchmark_id"), record.get("repetition"))
            if key not in expected or set(ids) != expected[key] or key in selected:
                raise ValueError("SGLang exact target is missing, duplicated or misidentified")
            entry = mappings[ids[0]]
            if (
                record["phase"] != entry["target_phase"]
                or queries != [entry["target_query"]] * batch
                or prefixes != [entry["target_prefix"]] * batch
                or record.get("sampling_role") != entry["sampling_role"]
            ):
                raise ValueError("SGLang observed target does not match its frozen geometry")
            for field in ("request_set", "dataset_role", "corpus_sha256"):
                if record.get(field) != manifest.get(field):
                    raise ValueError("SGLang native request provenance mismatch")
            latency = record.get("native_forward_ms")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (float, int))
                or not math.isfinite(latency)
                or latency <= 0
            ):
                raise ValueError("SGLang native latency must be finite and positive")
            count, padded = sum(queries), record.get("num_padded_tokens")
            if (
                record.get("runtime_mode") not in ("NONE", "FULL", "PIECEWISE")
                or type(padded) is not int
                or padded < count
            ):
                raise ValueError("SGLang requires actual native graph mode and padding")
            if record.get("used_cuda_graph") != (record["runtime_mode"] != "NONE"):
                raise ValueError("SGLang actual graph witnesses disagree")
            selected[key] = record
        if set(selected) != set(expected):
            raise ValueError(
                f"SGLang rank {rank} missing exact frozen forwards: {sorted(set(expected) - set(selected))}"
            )
        geometry = {
            key: (
                row["request_ids"],
                row["query_lengths"],
                row["prefix_lengths"],
                row["runtime_mode"],
                row["num_padded_tokens"],
                [
                    (
                        request["request_id"],
                        request["prompt_token_ids"],
                        request["native_query_token_ids"],
                        request["input_tokens_sha256"],
                        request["sampled_token_id"],
                    )
                    for request in row["requests"]
                ],
            )
            for key, row in selected.items()
        }
        if reference is not None and geometry != reference:
            raise ValueError("SGLang TP ranks disagree on actual request/dispatch identity")
        reference = geometry
        observations[rank] = selected
    if 0 not in observations:
        raise ValueError("SGLang native timing requires rank zero")
    return observations


def validate_sglang_repetitions(cell, payload: dict, path: Path) -> None:
    if cell.state_protocol != PROTOCOL or payload.get("kvwarm", {}).get("state_protocol") != PROTOCOL:
        raise ValueError("SGLang hybrid state protocol mismatch")
    if payload.get("timing_boundary") != TIMING_BOUNDARIES["sglang"]:
        raise ValueError("SGLang timing boundary mismatch")
    producer = payload.get("producer", {})
    if (producer.get("warmup_repeats"), producer.get("measurement_repeats"), producer.get("telemetry_policy")) != (
        WARMUPS,
        MEASUREMENTS,
        TELEMETRY_POLICY,
    ):
        raise ValueError("SGLang repetition/telemetry policy mismatch")
    evidence = payload["input_provenance"]["native_forward_manifest"]
    manifest = json.loads(read_receipt(path.parent, evidence["requests"]))
    if manifest.get("corpus_sha256") != payload["input_provenance"]["text_sha256"]:
        raise ValueError("SGLang request manifest corpus mismatch")
    traces = {entry["tp_rank"]: read_receipt(path.parent, entry) for entry in evidence["traces"]}
    if len(traces) != len(evidence["traces"]) or set(traces) != set(range(cell.topology.tp)):
        raise ValueError("SGLang actual TP rank coverage is incomplete")
    layouts = {entry["tp_rank"]: json.loads(read_receipt(path.parent, entry)) for entry in evidence["state_layouts"]}
    if len(layouts) != len(evidence["state_layouts"]) or set(layouts) != set(traces):
        raise ValueError("SGLang hybrid state layout rank coverage is incomplete")
    expected_dtypes = {
        "kda_conv": "torch.bfloat16",
        "kda_temporal": "torch.float32",
        "pooled_index_packed": "torch.uint8",
        "index_tail_key": "torch.bfloat16",
        "index_tail_score": "torch.bfloat16",
    }
    for rank, layout in layouts.items():
        if layout.get("admitted") is not True or layout.get("logical_kv_dtype") != "torch.float8_e4m3fn":
            raise ValueError("SGLang actual hybrid state layout is not admitted")
        groups = layout.get("groups", {})
        if not groups.get("mla_latent"):
            raise ValueError("SGLang MLA allocation evidence is missing")
        for group, dtype in expected_dtypes.items():
            if not groups.get(group) or any(t.get("dtype") != dtype for t in groups[group]):
                raise ValueError(f"SGLang allocated {group} dtype differs from the frozen contract")
        digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
        for line in traces[rank].splitlines():
            record = json.loads(line)
            if record.get("stage") == "measure" and (
                record.get("state_layout_sha256") != digest or record.get("state_layout_admitted") is not True
            ):
                raise ValueError("SGLang forward is not bound to its allocated hybrid state")
    observations = read_observations(manifest, traces, [result["point"] for result in payload["results"]])
    for result in payload["results"]:
        bid = result["point"]["benchmark_id"]
        values = [
            observations[0][bid, rep]["native_forward_ms"] / 1000 for rep in range(WARMUPS, WARMUPS + MEASUREMENTS)
        ]
        if not math.isclose(result["fpms"][0]["wall_time"], statistics.median(values), rel_tol=1e-12):
            raise ValueError("SGLang published median differs from native observations")
