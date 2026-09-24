# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded token-history reads preserve the native repetition acceptance gates."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.fpm_forward.hybrid_artifact import PROTOCOL, validate_real_hybrid_repetitions
from collector.fpm_forward.native_artifact import _expected_scheduled

pytestmark = pytest.mark.unit


def fixture(tmp_path, phase):
    batch, prefix, query = 2, 8, 3
    point = {
        "benchmark_id": 1,
        "point_type": phase,
        "batch_size": batch,
        "total_prefill_tokens": query * batch if phase == "prefill" else 0,
        "total_kv_read_tokens": prefix * batch,
    }
    decode = phase == "decode"
    length = prefix - 1 if decode else prefix + query
    histories, repetitions = [], []
    for index in range(15):
        role = "warmup" if index < 5 else "measurement"
        history = {
            "benchmark_id": 1,
            "repetition": index,
            "sampling_role": role,
            "requests": [
                {
                    "request_index": rank,
                    "request_id": f"request-{index}-{rank}",
                    "prompt_token_ids": [7] * length,
                    "output_token_ids": [9],
                    "computed_tokens": length + 2 * decode,
                }
                for rank in range(batch)
            ],
        }
        raw = json.dumps(history).encode()
        histories.append(raw)
        fpm = {"wall_time": 0.01, "scheduled_requests": _expected_scheduled(point)}
        dispatch = {
            "stage": "measure",
            "num_unpadded_tokens": batch if decode else batch * query,
            "num_padded_tokens": batch if decode else batch * query,
            "num_paddings": 0,
            "runtime_mode": "NONE",
        }
        repetitions.append(
            {
                "repetition": index,
                "role": role,
                "completed_seed_tokens": batch * (prefix - decode),
                "same_request": True,
                "allocated_fake_tokens": 0,
                "token_stream_sha256": hashlib.sha256(raw).hexdigest(),
                "fpms": [fpm] * (1 + decode),
                "dispatches": [dispatch] * (1 + decode),
            }
        )
    path = tmp_path / "benchmark.json"
    stream = path.with_suffix(".token-streams.jsonl")
    raw = b"\n".join(histories) + b"\n"
    stream.write_bytes(raw)
    payload = {
        "kvwarm": {"state_protocol": PROTOCOL},
        "timing_boundary": "vllm_native_scheduler_output_interval",
        "producer": {
            "warmup_repeats": 5,
            "measurement_repeats": 10,
            "context_policy_version": 1,
            "vllm_package_version": "0.30.0",
        },
        "limits": {"max_model_len": 131079},
        "context_policy": {
            "measured_context_limit": 131072,
            "runtime_context_length": 131079,
            "native_admission_headroom": 7,
        },
        "input_provenance": {
            "token_stream_manifest": {
                "schema_version": 3,
                "file": stream.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "records": 15,
            }
        },
        "results": [{"point": point, "fpms": [fpm], "real_hybrid_repetitions": repetitions}],
    }
    payload["input_provenance"]["context_policy"] = dict(payload["context_policy"])
    payload["kvwarm"]["max_context"] = 131072
    return SimpleNamespace(state_protocol=PROTOCOL, backend="vllm"), payload, path


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_history_validation_never_reads_whole_jsonl(tmp_path, monkeypatch, phase):
    cell, payload, path = fixture(tmp_path, phase)

    def forbidden(*args, **kwargs):
        raise AssertionError("cannot load whole token stream")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    validate_real_hybrid_repetitions(cell, payload, path)


@pytest.mark.parametrize("corruption", ["digest", "duplicate", "request", "geometry", "median", "coverage"])
def test_streaming_preserves_rejection_gates(tmp_path, corruption):
    cell, payload, path = fixture(tmp_path, "prefill")
    manifest = payload["input_provenance"]["token_stream_manifest"]
    stream = path.with_name(manifest["file"])
    lines = stream.read_bytes().splitlines()
    if corruption == "digest":
        manifest["sha256"] = "0" * 64
    elif corruption == "duplicate":
        lines[1] = lines[0]
    elif corruption == "request":
        row = json.loads(lines[1])
        row["requests"][0]["request_id"] = "request-0-0"
        lines[1] = json.dumps(row).encode()
        payload["results"][0]["real_hybrid_repetitions"][1]["token_stream_sha256"] = hashlib.sha256(
            lines[1]
        ).hexdigest()
    elif corruption == "geometry":
        payload["results"][0]["real_hybrid_repetitions"][0]["completed_seed_tokens"] += 1
    elif corruption == "median":
        payload["results"][0]["fpms"] = [{"wall_time": 0.1}]
    else:
        lines.pop()
    if corruption != "digest":
        raw = b"\n".join(lines) + b"\n"
        stream.write_bytes(raw)
        manifest["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="GLM"):
        validate_real_hybrid_repetitions(cell, payload, path)


@pytest.mark.parametrize("rows", [None, [[3, 9], [3, 9]]])
def test_reader_rejects_stock_vllm_unqualified_pool_start(tmp_path, rows):
    cell, payload, path = fixture(tmp_path, "prefill")
    point = payload["results"][0]["point"]
    point["total_kv_read_tokens"] = 18
    point["rows"] = rows
    with pytest.raises(ValueError, match="cached-prefill start is unqualified"):
        validate_real_hybrid_repetitions(cell, payload, path)


@pytest.mark.parametrize("field", ["missing", "version", "native", "policy", "provenance", "state"])
def test_context_headroom_cannot_be_stripped_or_substituted(tmp_path, field):
    cell, payload, path = fixture(tmp_path, "decode")
    if field == "missing":
        payload.pop("context_policy")
        payload["producer"].pop("context_policy_version")
    elif field == "version":
        payload["producer"]["context_policy_version"] = True
    elif field == "native":
        payload["limits"]["max_model_len"] = 131072
    elif field == "policy":
        payload["context_policy"]["native_admission_headroom"] = 8
    elif field == "provenance":
        payload["input_provenance"].pop("context_policy")
    else:
        payload["kvwarm"]["max_context"] = 131079
    with pytest.raises(ValueError, match="context|policy"):
        validate_real_hybrid_repetitions(cell, payload, path)


def test_only_receipted_legacy_overlay_preserves_historical_short_canary(tmp_path):
    cell, payload, path = fixture(tmp_path, "prefill")
    payload.pop("context_policy")
    payload["producer"].pop("context_policy_version")
    payload["input_provenance"].pop("context_policy")
    payload["limits"]["max_model_len"] = 131072
    payload["producer"]["overlay_sha256"] = "e391db177f53430c4280807fcc0eafdace5310cda6f6549ef7f2fb54e6cad984"
    validate_real_hybrid_repetitions(cell, payload, path)
    payload["producer"]["overlay_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="versioned context policy"):
        validate_real_hybrid_repetitions(cell, payload, path)


def test_internal_headroom_cannot_enlarge_measured_workload(tmp_path):
    from collector.glm53flash_protocol import vllm_context_policy

    cell, payload, path = fixture(tmp_path, "prefill")
    payload["context_policy"] = vllm_context_policy(10)
    payload["input_provenance"]["context_policy"] = payload["context_policy"]
    payload["limits"]["max_model_len"] = 17
    payload["kvwarm"]["max_context"] = 10
    with pytest.raises(ValueError, match="measured context"):
        validate_real_hybrid_repetitions(cell, payload, path)
