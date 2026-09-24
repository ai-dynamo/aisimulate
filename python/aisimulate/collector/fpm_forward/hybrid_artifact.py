# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate retained repeated native observations and real hybrid histories."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

PROTOCOL = "glm53flash_same_request_real_hybrid_v1"
TIMING_BOUNDARIES = {
    "vllm": "vllm_native_scheduler_output_interval",
    "sglang": "sglang_native_forward_device_timer",
}


def validate_real_hybrid_repetitions(cell, payload: dict, path: Path) -> None:
    from .native_artifact import _expected_scheduled

    if cell.state_protocol != PROTOCOL or payload.get("kvwarm", {}).get("state_protocol") != PROTOCOL:
        raise ValueError("GLM hybrid state protocol mismatch")
    if payload.get("ops_instrumented", False) or payload.get("observation_purpose", "fpm") != "fpm":
        raise ValueError("GLM FPM cannot admit operation-instrumented or eager Ops validation data")
    if payload.get("timing_boundary") != TIMING_BOUNDARIES[cell.backend]:
        raise ValueError("GLM native timing boundary mismatch")
    producer = payload.get("producer", {})
    warmups, measurements = producer.get("warmup_repeats"), producer.get("measurement_repeats")
    if type(warmups) is not int or warmups < 5 or type(measurements) is not int or measurements < 10:
        raise ValueError("GLM formal data requires at least 5 warmups and 10 measurements per point")
    manifest = payload["input_provenance"].get("token_stream_manifest", {})
    if manifest.get("schema_version") != 3:
        raise ValueError("GLM real hybrid token histories require schema 3")
    name = manifest.get("file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".token-streams.jsonl"):
        raise ValueError("GLM token history must be an adjacent JSONL file")
    stream_path = path.with_name(name)
    # Keep only offsets and digests across repetitions. Full long-context token
    # arrays are checked one record at a time and remain in the original JSONL.
    histories = {}
    stream_digest = hashlib.sha256()
    offset = 0
    with stream_path.open("rb") as source:
        for raw_line in source:
            stream_digest.update(raw_line)
            line = raw_line.rstrip(b"\r\n")
            record = json.loads(line)
            key = (record.get("benchmark_id"), record.get("repetition"))
            if any(type(value) is not int for value in key) or key in histories:
                raise ValueError("duplicate or malformed GLM history identity")
            histories[key] = (offset, len(raw_line), hashlib.sha256(line).hexdigest())
            offset += len(raw_line)
    if stream_digest.hexdigest() != manifest.get("sha256"):
        raise ValueError("GLM token history digest mismatch")
    # Do not retain the last indexed record while reading the next full record.
    record = None
    expected_count = len(payload["results"]) * (warmups + measurements)
    if manifest.get("records") != len(histories) or len(histories) != expected_count:
        raise ValueError("GLM real history coverage mismatch")
    consumed = set()
    request_ids = set()
    for result in payload["results"]:
        point = result["point"]
        repetitions = result.get("real_hybrid_repetitions")
        if not isinstance(repetitions, list) or len(repetitions) != warmups + measurements:
            raise ValueError("GLM repetition coverage mismatch")
        expected = _expected_scheduled(point)
        decode = point["point_type"] == "decode"
        batch = point["batch_size"]
        seed = point["total_kv_read_tokens"] - (batch if decode else 0)
        if cell.backend == "vllm" and not decode:
            if point.get("rows") is not None:
                rows = point["rows"]
            elif point.get("partition") is not None:
                raise ValueError("GLM stock vLLM partitioned cached-prefill requires native start qualification")
            else:
                prefixes = divmod(point["total_kv_read_tokens"], batch)
                queries = divmod(point["total_prefill_tokens"], batch)
                rows = [(queries[0] + (i < queries[1]), prefixes[0] + (i < prefixes[1])) for i in range(batch)]
            if any(prefix % 4 and query >= 2 for query, prefix in rows):
                raise ValueError("GLM stock vLLM IndexPool cached-prefill start is unqualified")
        values = []
        for index, repetition in enumerate(repetitions):
            role = "warmup" if index < warmups else "measurement"
            key = (point["benchmark_id"], index)
            if key not in histories:
                raise ValueError("GLM missing exact repetition history")
            consumed.add(key)
            offset, length, digest = histories[key]
            with stream_path.open("rb") as source:
                source.seek(offset)
                line = source.read(length).rstrip(b"\r\n")
            if hashlib.sha256(line).hexdigest() != digest:
                raise ValueError("GLM token history changed during validation")
            history = json.loads(line)
            if (
                repetition.get("repetition") != index
                or repetition.get("role") != role
                or history.get("sampling_role") != role
                or repetition.get("completed_seed_tokens") != seed
                or repetition.get("same_request") is not True
                or repetition.get("allocated_fake_tokens") != 0
                or repetition.get("token_stream_sha256") != digest
            ):
                raise ValueError("GLM invalid completed real-forward witness")
            requests = history.get("requests")
            if not isinstance(requests, list) or len(requests) != batch:
                raise ValueError("GLM request count mismatch")
            prompt_total = 0
            for request_index, request in enumerate(requests):
                rid = request.get("request_id")
                if request.get("request_index") != request_index or not isinstance(rid, str) or rid in request_ids:
                    raise ValueError("GLM request identity reused or reordered")
                request_ids.add(rid)
                for field in ("prompt_token_ids", "output_token_ids"):
                    tokens = request.get(field)
                    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
                        raise ValueError("GLM requires actual prompt and sampled output token histories")
                length = len(request["prompt_token_ids"])
                prompt_total += length
                if request.get("computed_tokens") != length + (2 if decode else 0):
                    raise ValueError("GLM computed history differs from the completed forward")
            if prompt_total != seed + (0 if decode else point["total_prefill_tokens"]):
                raise ValueError("GLM request histories disagree with scheduled geometry")
            fpms = repetition.get("fpms")
            if not isinstance(fpms, list) or len(fpms) != (2 if decode else 1):
                raise ValueError("GLM native FPM count mismatch")
            fpm = fpms[-1]
            if any(fpm.get("scheduled_requests", {}).get(k) != v for k, v in expected.items()):
                raise ValueError("GLM native measured coordinate mismatch")
            latency = fpm.get("wall_time")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency <= 0
            ):
                raise ValueError("GLM native latency must be finite and positive")
            dispatches = repetition.get("dispatches")
            if not isinstance(dispatches, list) or len(dispatches) != len(fpms):
                raise ValueError("GLM missing actual native dispatch evidence")
            dispatch = dispatches[-1]
            count = batch if decode else point["total_prefill_tokens"]
            if dispatch.get("stage") != "measure" or dispatch.get("num_unpadded_tokens") != count:
                raise ValueError("GLM graph dispatch geometry mismatch")
            mode = dispatch.get("runtime_mode")
            if not isinstance(mode, str) or mode not in {"NONE", "FULL", "PIECEWISE"}:
                raise ValueError("GLM unsupported or missing native graph mode")
            padded = dispatch.get("num_padded_tokens")
            if type(padded) is not int or padded < count or dispatch.get("num_paddings") != padded - count:
                raise ValueError("GLM graph padding witness mismatch")
            if index >= warmups:
                values.append(latency)
        if not math.isclose(result["fpms"][-1]["wall_time"], statistics.median(values), rel_tol=1e-12):
            raise ValueError("GLM published latency differs from retained observation median")
    if consumed != set(histories):
        raise ValueError("GLM contains unrelated token histories")
