# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU evidence-integrity tests; the observations here are synthetic fixtures."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from collector.fpm_forward.sglang_artifact import read_observations, validate_sglang_repetitions
from collector.fpm_forward.sglang_driver import freeze_requests, result_payload
from collector.glm53flash_protocol import PROTOCOL, TIMING_BOUNDARIES

pytestmark = pytest.mark.unit


def fixture():
    point = {
        "benchmark_id": 1,
        "point_type": "decode",
        "batch_size": 1,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 2,
    }
    manifest = freeze_requests([point], request_set="independent", dataset_role="calibration", corpus_sha256="a" * 64)
    records = {0: [], 1: []}
    for rank in records:
        for rid, entry in manifest["requests"].items():
            rep = entry["repetition"]
            for step, (query, prefix, tokens, sampled) in enumerate(((2, 0, [4, 5], 7), (1, 2, [7], 9))):
                fid = f"rank-{rank}/forward-{rep * 2 + step}"
                history = [4, 5] if step == 0 else [4, 5, 7]
                record = {
                    "tp_rank": rank,
                    "state_protocol": PROTOCOL,
                    "allocated_fake_tokens": 0,
                    "gpu_completed": True,
                    "ops_instrumented": False,
                    "timing_boundary": TIMING_BOUNDARIES["sglang"],
                    "forward_id": fid,
                    "request_ids": [rid],
                    "batch_size": 1,
                    "query_lengths": [query],
                    "prefix_lengths": [prefix],
                    "total_new_tokens": query,
                    "total_past_kv_tokens": prefix,
                    "phase": "context" if not step else "generation",
                    "stage": "seed" if not step else "measure",
                    "benchmark_id": 1,
                    "repetition": rep,
                    "sampling_role": entry["sampling_role"],
                    "request_set": manifest["request_set"],
                    "dataset_role": "calibration",
                    "corpus_sha256": "a" * 64,
                    "native_forward_ms": 999 if rep < 5 else rep + 1,
                    "runtime_mode": "NONE" if not step else "FULL",
                    "used_cuda_graph": bool(step),
                    "num_padded_tokens": query,
                    "requests": [
                        {
                            "request_id": rid,
                            "native_query_token_ids": tokens,
                            "prompt_token_ids": [4, 5],
                            "sampled_token_id": sampled,
                            "previous_forward_id": f"rank-{rank}/forward-{rep * 2}" if step else None,
                            "same_request_real_prefix": True,
                            "computed_tokens_before": prefix,
                            "computed_tokens_after": prefix + query,
                            "input_tokens_sha256": hashlib.sha256(
                                json.dumps(history, separators=(",", ":")).encode()
                            ).hexdigest(),
                        }
                    ],
                }
                records[rank].append(record)
    return point, manifest, records


def raw(records):
    return {rank: ("\n".join(json.dumps(row) for row in rows) + "\n").encode() for rank, rows in records.items()}


def test_native_sglang_median_excludes_warmup_and_roundtrips(tmp_path):
    point, manifest, records = fixture()
    layout = {
        "admitted": True,
        "logical_kv_dtype": "torch.float8_e4m3fn",
        "groups": {
            name: [{"dtype": dtype, "shape": [16, 4], "stride": [4, 1]}]
            for name, dtype in {
                "mla_latent": "torch.float8_e4m3fn",
                "kda_conv": "torch.bfloat16",
                "kda_temporal": "torch.float32",
                "pooled_index_packed": "torch.uint8",
                "index_tail_key": "torch.bfloat16",
                "index_tail_score": "torch.bfloat16",
            }.items()
        },
    }
    for rank, rows in records.items():
        (tmp_path / f"state-layout-rank-{rank}.json").write_text(json.dumps(layout))
        for row in rows:
            row.update(
                state_layout_admitted=True,
                state_layout_sha256=hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
            )
    traces = raw(records)
    observations = read_observations(manifest, traces, [point])
    manifest_path = tmp_path / "sglang-requests.json"
    manifest_path.write_text(json.dumps(manifest))
    paths = [tmp_path / f"forward-rank-{rank}.jsonl" for rank in traces]
    for rank, path in enumerate(paths):
        path.write_bytes(traces[rank])
    payload = result_payload(
        [point],
        observations,
        output=tmp_path / "benchmark.json",
        manifest_path=manifest_path,
        trace_paths=paths,
        provenance={"run_id": "run", "execution_identity": {}},
        input_provenance={"text_sha256": "a" * 64},
        elapsed=100,
    )
    assert payload["results"][0]["fpms"][0]["wall_time"] == pytest.approx(0.0105)
    cell = SimpleNamespace(state_protocol=PROTOCOL, topology=SimpleNamespace(tp=2))
    validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")
    payload["results"][0]["fpms"][0]["wall_time"] = 0.999
    with pytest.raises(ValueError, match="median"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize(
    "corruption", ["seed", "fake", "tokens", "ops", "padding", "rank", "missing", "reused", "phase", "role"]
)
def test_native_sglang_rejects_invalid_observations(corruption):
    point, manifest, records = fixture()
    selected = records[1][11]
    if corruption == "seed":
        records[1].pop(10)
    elif corruption == "fake":
        selected["allocated_fake_tokens"] = 1
    elif corruption == "tokens":
        selected["requests"][0]["native_query_token_ids"] = [88]
    elif corruption == "ops":
        selected["ops_instrumented"] = True
    elif corruption == "padding":
        selected["num_padded_tokens"] = None
    elif corruption == "rank":
        selected["tp_rank"] = 0
    elif corruption == "missing":
        selected["stage"] = "seed"
    elif corruption == "reused":
        records[1].append(selected)
    elif corruption == "phase":
        selected["phase"] = "context"
    else:
        selected["sampling_role"] = "warmup"
    with pytest.raises(ValueError):
        read_observations(manifest, raw(records), [point])


def test_ops_provenance_is_bound_to_loaded_config_and_native_source(tmp_path):
    from collector.fpm_forward.sglang_driver import read_ops_provenance

    from aisimulate_core.sdk.glm53flash import BACKEND_REVISIONS

    def sha256_json(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    config = {"actual": "loaded"}
    audit = {"status": "passed", "sources": {"native.py": "a" * 64}}
    provenance = {
        "backend": "sglang",
        "backend_version": "0.5.20",
        "backend_revision": BACKEND_REVISIONS["sglang"],
        "checkpoint_revision": "pinned-checkpoint",
        "config_sha256": sha256_json(config),
        "source_sha256": sha256_json(audit["sources"]),
        "runtime_digest": "sha256:" + "b" * 64,
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(provenance))
    assert (
        read_ops_provenance(path, raw_config=config, checkpoint_revision="pinned-checkpoint", runtime_audit=audit)
        == provenance
    )
    with pytest.raises(ValueError, match="differs"):
        read_ops_provenance(
            path, raw_config={"other": "config"}, checkpoint_revision="pinned-checkpoint", runtime_audit=audit
        )
    path.write_text(json.dumps({**provenance, "execution_identity": {}}))
    with pytest.raises(ValueError, match="cannot replace"):
        read_ops_provenance(path, raw_config=config, checkpoint_revision="pinned-checkpoint", runtime_audit=audit)


@pytest.mark.parametrize("all_ranks", [False, True])
def test_native_sglang_rejects_rehashed_wrong_sample_chain(all_ranks):
    point, manifest, records = fixture()
    for rank in records if all_ranks else (1,):
        request = records[rank][11]["requests"][0]
        request["native_query_token_ids"] = [88]
        request["input_tokens_sha256"] = hashlib.sha256(b"[4,5,88]").hexdigest()
    with pytest.raises(ValueError, match="preceding sampled token"):
        read_observations(manifest, raw(records), [point])


def test_native_sglang_rejects_internally_valid_but_different_tp_token_chains():
    point, manifest, records = fixture()
    # Rank one is internally continuous, but its preceding sample and decode
    # input differ from rank zero. Geometry and dispatch are unchanged.
    records[1][10]["requests"][0]["sampled_token_id"] = 88
    request = records[1][11]["requests"][0]
    request["native_query_token_ids"] = [88]
    request["input_tokens_sha256"] = hashlib.sha256(b"[4,5,88]").hexdigest()
    with pytest.raises(ValueError, match="TP ranks disagree"):
        read_observations(manifest, raw(records), [point])


def test_native_sglang_rejects_different_tp_final_samples():
    point, manifest, records = fixture()
    records[1][11]["requests"][0]["sampled_token_id"] = 88
    with pytest.raises(ValueError, match="TP ranks disagree"):
        read_observations(manifest, raw(records), [point])
