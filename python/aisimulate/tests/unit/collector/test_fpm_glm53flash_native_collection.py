# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native collection accepts the rank artifacts of the GLM-5.3-Flash vLLM producer.

The artifacts are built in the shape of ``Glm53FlashPrefixSeedScheduler._glm_enrich``: a native rank artifact plus the
producer's execution, input, serving and evidence blocks and a repetition-evidence sidecar next to it.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from collector import glm53flash_attention_tokens as seeded
from collector.fpm_forward.native_artifact import (
    _expected_scheduled,
    select_native_measurements,
    validate_native_collection,
)
from collector.fpm_forward.planner import FPMCell

from aisimulate.fpm_contract import fpm_benchmark_result_name
from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS
from aisimulate_core.sdk.utils import get_model_config_from_model_path

from .test_fpm_runner import _cell, _native_payload, _write_provenance

pytestmark = pytest.mark.unit

GLM_MODEL = "zai-org/GLM-5.3-Flash"
PLAN_SHA256 = "plan-sha"
ATTEMPT_ID = "a1"
EVIDENCE_FILE = "benchmark.repetitions.jsonl"

PREFILL_COLD = {
    "batch_size": 1,
    "total_prefill_tokens": 64,
    "total_kv_read_tokens": 0,
    "sample_reasons": ["explicit"],
}
PREFILL_CACHED = {
    "batch_size": 1,
    "total_prefill_tokens": 64,
    "total_kv_read_tokens": 2176,
    "sample_reasons": ["explicit", "prefill_real_seed"],
}
DECODE_POINTS = [
    {
        "batch_size": 1,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 4096,
        "sample_reasons": ["explicit", "kvwarm_real_kv"],
    },
    {
        "batch_size": 2,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 8192,
        "sample_reasons": ["explicit", "kvwarm_real_kv"],
    },
]


@functools.cache
def glm_identity() -> tuple[str, ...]:
    """The execution identity that the producer computes from the loaded checkpoint configuration."""
    raw_config = get_model_config_from_model_path(GLM_MODEL)["raw_config"]
    return execution_identity(raw_config, backend="vllm", input_modality="text")


def glm_cell(phase: str, *, dp: int = 1) -> FPMCell:
    """A pure-TP cell that carries the real GLM-5.3-Flash execution identity."""
    return replace(_cell(phase=phase, dp=dp, strategy="pure_tp"), execution_identity=glm_identity())


def glm_native_payload(phase: str, *, points: list[dict], rank: int = 0, dp: int = 1) -> dict:
    """The rank artifact that the producer writes for ``points`` of one phase.

    A point gives ``batch_size``, ``total_prefill_tokens``, ``total_kv_read_tokens`` and ``sample_reasons``. Each
    row holds one FPM that schedules exactly what its point asks for. The evidence manifest and the per-row digests
    stay empty until ``write_glm_rank`` writes the sidecar that they describe.
    """
    payload = _native_payload(phase=phase, rank=rank, dp=dp)
    rows, groups = [], []
    for benchmark_id, spec in enumerate(points, start=1):
        point = {"point_type": phase, "benchmark_id": benchmark_id, **copy.deepcopy(spec)}
        fpms = [
            {
                "counter_id": benchmark_id,
                "dp_rank": dp_rank,
                "wall_time": 0.01 * benchmark_id + dp_rank / 1000,
                "scheduled_requests": _expected_scheduled(point),
            }
            for dp_rank in range(dp)
        ]
        if phase == "decode":
            regime = "real_kv"
        else:
            regime = "real_prefix" if point["total_kv_read_tokens"] > 0 else "not_applicable"
        rows.append(
            {"point": point, "fpms": [fpms[rank]], "kv_seed_regime": regime, "repetition_evidence_sha256": None}
        )
        groups.append(
            {
                "benchmark_id": benchmark_id,
                "point": point,
                "expected_dp_ranks": list(range(dp)),
                "complete": True,
                "wall_time": max(fpm["wall_time"] for fpm in fpms),
                "rank_results": [{"dp_rank": fpm["dp_rank"], "fpms": [fpm]} for fpm in fpms],
                # The producer stamps every group with the decode regime, whatever its phase.
                "kv_seed_regime": "real_kv",
            }
        )
    measured = sum(group["wall_time"] for group in groups)
    context_policy = {
        "measured_context_limit": 131072,
        "runtime_context_length": 131079,
        "native_admission_headroom": 7,
    }
    payload.update(
        coverage={"expected_points": len(rows), "completed_points": len(rows), "skipped_points": 0},
        results=rows,
        iteration_groups=groups,
        timing={"benchmark_elapsed_seconds": measured + 1.0, "measured_iteration_seconds": measured},
        context_policy=context_policy,
        execution_identity=dict(zip(EXECUTION_COLUMNS, glm_identity(), strict=True)),
        execution_mode="native_graph_policy",
        observation_purpose="fpm",
        ops_instrumented=False,
        timing_boundary="dynamo_vllm_instrumented_scheduler_fpm_wall_time",
        input_provenance={
            "source": "seeded_tokens",
            "generator": seeded.GENERATOR,
            "generator_version": seeded.GENERATOR_VERSION,
            "seed": seeded.SEED,
            "count": 131072 + 32 * seeded.REQUEST_STRIDE,
            "vocab_low": seeded.VOCAB_LOW,
            "vocab_high_exclusive": seeded.VOCAB_HIGH,
            "tokenizer_json_sha256": seeded.TOKENIZER_JSON_SHA256,
            "tokenizer_revision": MODEL_REVISIONS[GLM_MODEL],
            "token_ids_sha256": hashlib.sha256(b"seeded token ids").hexdigest(),
            "unique_token_count": 100000,
            "sampling": "prompt = concatenated [offset, length] rotations of the seeded stream (see prompt specs)",
            "chunk_start_alignment": 4,
            "decode_contexts_sha256": None,
            "context_policy": context_policy,
            "repetition_evidence_manifest": {
                "schema_version": 1,
                "file": Path(fpm_benchmark_result_name("benchmark.json", rank)).with_suffix(".repetitions.jsonl").name,
                "sha256": None,
                "records": 0,
            },
        },
        serving_config={
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "prefix_match_unit": None,
            "hash_block_size": 2176,
            "block_size": 2176,
            "need_mamba_block_aligned_split": True,
            "cache_dtype": "fp8_e4m3",
            "async_scheduling": False,
        },
        kvwarm={
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "native_prefix_cache_real_seed",
            "state_protocol": "glm53flash_prefix_cache_real_seed_v1",
            "max_batch": 32,
            "max_context": 131072,
        },
        producer={
            "instrumentation_revision": "54960177085413259859c88bd34ed0734d4c2ea9",
            "real_seed_port_source": "ai-dynamo/dynamo@99dae1f53e5a0534c274223ff9daa97f0b1fc2ea "
            "instrumented_scheduler._bench_realseed_*",
            "queued_metrics_port_source": "ai-dynamo/dynamo@395f02405c1dd1835ab73a39f7a5984ae5fd4552 "
            "instrumented_scheduler._compute_queued",
            "vllm_package_version": "0.31.0",
            "reviewed_scheduler_api_revision": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
            "runtime_source_manifest_sha256": hashlib.sha256(b"runtime source manifest").hexdigest(),
            "overlay_sha256": hashlib.sha256(b"producer overlay").hexdigest(),
            "context_policy_version": 1,
            "warmup_repeats": 5,
            "measurement_repeats": 10,
        },
    )
    return payload


def write_glm_rank(directory: Path, payload: dict) -> Path:
    """Write ``payload`` as ``benchmark.json`` and its sidecar of repetition evidence next to it.

    The sidecar holds one line per result row. As in the producer, a row's digest is the digest of its line without
    the line break, and the manifest holds the digest of the whole file. ``payload`` is updated in place with these
    values and the path of the artifact is returned. Rank 0 writes ``benchmark.json``. Another rank writes
    ``benchmark_dp<rank>.json`` with a sidecar of the same stem, as the engine names them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"benchmark_id": row["point"]["benchmark_id"], "point_type": row["point"]["point_type"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        for row in payload["results"]
    ]
    for row, line in zip(payload["results"], lines, strict=True):
        row["repetition_evidence_sha256"] = hashlib.sha256(line).hexdigest()
    sidecar = b"".join(line + b"\n" for line in lines)
    manifest = payload["input_provenance"]["repetition_evidence_manifest"]
    manifest.update(sha256=hashlib.sha256(sidecar).hexdigest(), records=len(lines))
    (directory / manifest["file"]).write_bytes(sidecar)
    artifact = directory / fpm_benchmark_result_name("benchmark.json", payload["dp"]["rank"])
    artifact.write_text(json.dumps(payload))
    return artifact


def _pod(tmp_path: Path, cell: FPMCell) -> Path:
    pod = tmp_path / "raw" / "node0000"
    pod.mkdir(parents=True)
    _write_provenance(
        pod / "collector-provenance.json", cell_id=cell.cell_id, plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID
    )
    return pod


def _validate(tmp_path: Path, cell: FPMCell):
    return validate_native_collection(
        cell, tmp_path / "raw", expected_plan_sha256=PLAN_SHA256, expected_attempt_id=ATTEMPT_ID
    )


def test_glm_producer_artifact_validates_and_selects_its_points(tmp_path):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    collection = _validate(tmp_path, cell)
    assert [m.kv_seed_regime for m in collection.points] == ["not_applicable", "real_prefix"]
    assert collection.input_provenance["source"] == "seeded_tokens"
    selected = select_native_measurements(collection, cell_id=cell.cell_id)
    assert [m.point["total_kv_read_tokens"] for m in selected] == [0, 2176]


def test_glm_producer_decode_artifact_validates(tmp_path):
    cell = glm_cell("decode")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("decode", points=DECODE_POINTS))
    collection = _validate(tmp_path, cell)
    assert [m.point["batch_size"] for m in collection.points] == [1, 2]
    assert {m.kv_seed_regime for m in collection.points} == {"real_kv"}
    assert collection.kvwarm_meta["method"] == "native_prefix_cache_real_seed"


def test_glm_producer_artifact_without_repetition_evidence_is_refused(tmp_path):
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    write_glm_rank(pod, glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    (pod / EVIDENCE_FILE).unlink()
    with pytest.raises(ValueError, match="native GLM-5.3-Flash result lacks valid repetition_evidence_manifest"):
        _validate(tmp_path, cell)


def test_glm_producer_row_regime_must_follow_its_point(tmp_path):
    cell = glm_cell("decode")
    payload = glm_native_payload("decode", points=DECODE_POINTS)
    payload["results"][0]["kv_seed_regime"] = "real_prefix"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError, match="native GLM-5.3-Flash result lacks valid kv_seed_regime"):
        _validate(tmp_path, cell)


def test_unknown_execution_mode_is_refused(tmp_path):
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD])
    payload["execution_mode"] = "graph"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError, match="native execution identity cell has an unaudited execution mode: 'graph'"):
        _validate(tmp_path, cell)


def test_v41_artifacts_keep_their_contract(tmp_path):
    identity = ("c" * 64, "full", "hbm_tp_sharded", "text")
    cell = replace(_cell(phase="prefill", strategy="pure_tp"), execution_identity=identity, input_text_sha256="a" * 64)
    payload = _native_payload(phase="prefill", rank=0, dp=1)
    payload.update(
        execution_identity=dict(zip(EXECUTION_COLUMNS, identity, strict=True)),
        execution_mode="eager",
        input_provenance={
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        },
    )
    (_pod(tmp_path, cell) / "benchmark.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="V4.1 native result lacks token-stream manifest"):
        _validate(tmp_path, cell)


def _drop_a_line(payload: dict, sidecar: Path) -> None:
    lines = sidecar.read_bytes().splitlines(keepends=True)[:-1]
    sidecar.write_bytes(b"".join(lines))
    payload["input_provenance"]["repetition_evidence_manifest"]["sha256"] = hashlib.sha256(b"".join(lines)).hexdigest()


def _add_a_line(payload: dict, sidecar: Path) -> None:
    raw = sidecar.read_bytes() + b'{"benchmark_id":99}\n'
    sidecar.write_bytes(raw)
    manifest = payload["input_provenance"]["repetition_evidence_manifest"]
    manifest.update(sha256=hashlib.sha256(raw).hexdigest(), records=len(raw.splitlines()))


def _evidence(payload: dict) -> dict:
    return payload["input_provenance"]


REFUSALS = [
    pytest.param(
        "execution_identity",
        lambda p, _: p["execution_identity"].update(execution_profile="decoder_bounded"),
        id="identity",
    ),
    pytest.param("input_provenance", lambda p, _: p.update(input_provenance=[]), id="provenance-not-a-mapping"),
    pytest.param(
        "input_provenance.source", lambda p, _: _evidence(p).update(source="tokenizer_text"), id="input-source"
    ),
    pytest.param("input_provenance.generator", lambda p, _: _evidence(p).update(generator=""), id="generator"),
    pytest.param(
        "input_provenance.generator_version",
        lambda p, _: _evidence(p).update(generator_version="1"),
        id="generator-version",
    ),
    pytest.param("input_provenance.sampling", lambda p, _: _evidence(p).update(sampling=None), id="sampling"),
    pytest.param("input_provenance.seed", lambda p, _: _evidence(p).update(seed=True), id="seed"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count="262240"), id="count"),
    pytest.param(
        "input_provenance.tokenizer_json_sha256",
        lambda p, _: _evidence(p).update(tokenizer_json_sha256="abc"),
        id="tokenizer-hash",
    ),
    pytest.param(
        "input_provenance.token_ids_sha256",
        lambda p, _: _evidence(p).update(token_ids_sha256="A" * 64),
        id="token-ids-hash-uppercase",
    ),
    pytest.param(
        "input_provenance.tokenizer_revision",
        lambda p, _: _evidence(p).update(tokenizer_revision="main"),
        id="tokenizer-revision",
    ),
    pytest.param(
        "input_provenance.chunk_start_alignment",
        lambda p, _: _evidence(p).update(chunk_start_alignment=1),
        id="chunk-alignment",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p).pop("repetition_evidence_manifest"),
        id="manifest-absent",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(schema_version=2),
        id="manifest-schema",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(schema_version=True),
        id="manifest-schema-is-a-boolean",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(file="other.repetitions.jsonl"),
        id="manifest-file-name",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(file=f"../{EVIDENCE_FILE}"),
        id="manifest-file-outside-the-directory",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(sha256="0" * 64),
        id="manifest-digest",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(records=3),
        id="manifest-records",
    ),
    pytest.param("repetition_evidence_manifest", _drop_a_line, id="sidecar-has-fewer-lines-than-records"),
    pytest.param("repetition_evidence_manifest", _add_a_line, id="sidecar-has-more-lines-than-rows"),
    pytest.param("kvwarm", lambda p, _: p.pop("kvwarm"), id="kvwarm-absent"),
    pytest.param("kvwarm", lambda p, _: p["kvwarm"].update(enabled=False), id="kvwarm-disabled"),
    pytest.param("kvwarm", lambda p, _: p["kvwarm"].update(enabled=1), id="kvwarm-enabled-is-not-a-boolean"),
    pytest.param("kvwarm", lambda p, _: p["kvwarm"].update(warm_eligible=False), id="kvwarm-not-eligible"),
    pytest.param("kvwarm", lambda p, _: p["kvwarm"].update(skip_reason="skipped"), id="kvwarm-skip-reason"),
    pytest.param(
        "kvwarm",
        lambda p, _: p["kvwarm"].update(method="native_radix_cache_real_seed"),
        id="kvwarm-method",
    ),
    pytest.param(
        "kvwarm", lambda p, _: p["kvwarm"].update(state_protocol="other_protocol"), id="kvwarm-state-protocol"
    ),
    pytest.param("timing_boundary", lambda p, _: p.update(timing_boundary="wall_clock"), id="timing-boundary"),
    pytest.param(
        "serving_config.enable_prefix_caching",
        lambda p, _: p["serving_config"].update(enable_prefix_caching=False),
        id="prefix-caching-off",
    ),
    pytest.param(
        "serving_config.async_scheduling",
        lambda p, _: p["serving_config"].update(async_scheduling=True),
        id="async-scheduling-on",
    ),
    pytest.param(
        "serving_config.enable_prefix_caching", lambda p, _: p.pop("serving_config"), id="serving-config-absent"
    ),
    pytest.param(
        "producer.runtime_source_manifest_sha256",
        lambda p, _: p["producer"].update(runtime_source_manifest_sha256=None),
        id="runtime-source-manifest",
    ),
    pytest.param("producer.runtime_source_manifest_sha256", lambda p, _: p.pop("producer"), id="producer-absent"),
    pytest.param(
        "repetition_evidence_sha256",
        lambda p, _: p["results"][1].pop("repetition_evidence_sha256"),
        id="row-digest-absent",
    ),
    pytest.param(
        "kv_seed_regime", lambda p, _: p["results"][0].update(kv_seed_regime="real_prefix"), id="regime-of-cold-row"
    ),
    pytest.param(
        "kv_seed_regime",
        lambda p, _: p["results"][1].update(kv_seed_regime="not_applicable"),
        id="regime-of-cached-row",
    ),
]


@pytest.mark.parametrize(("field", "tamper"), REFUSALS)
def test_glm_producer_artifact_refuses_each_unaudited_item(tmp_path, field, tamper):
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    artifact = write_glm_rank(pod, payload)
    tamper(payload, pod / EVIDENCE_FILE)
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=rf"native GLM-5\.3-Flash result lacks valid {re.escape(field)}: "):
        _validate(tmp_path, cell)


def _change_the_sidecar(payload: dict, sidecar: Path) -> None:
    raw = b'{"benchmark_id":1,"point_type":"prefill","note":"another rank"}\n'
    sidecar.write_bytes(raw)
    payload["input_provenance"]["repetition_evidence_manifest"]["sha256"] = hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(lambda p, _: _evidence(p).update(seed=_evidence(p)["seed"] + 1), id="input"),
        pytest.param(_change_the_sidecar, id="sidecar-digest"),
    ],
)
def test_glm_producer_dp_ranks_must_agree_on_input_provenance(tmp_path, tamper):
    cell = glm_cell("prefill", dp=2)
    pod = _pod(tmp_path, cell)
    payloads = [glm_native_payload("prefill", points=[PREFILL_COLD], rank=rank, dp=2) for rank in range(2)]
    artifacts = [write_glm_rank(pod, payload) for payload in payloads]
    # Each rank names its own sidecar, and the names alone are no disagreement.
    assert [artifact.name for artifact in artifacts] == ["benchmark.json", "benchmark_dp1.json"]
    assert len(_validate(tmp_path, cell).points) == 1
    tamper(payloads[1], artifacts[1].with_suffix(".repetitions.jsonl"))
    artifacts[1].write_text(json.dumps(payloads[1]))
    with pytest.raises(ValueError, match="native DP ranks disagree on input provenance"):
        _validate(tmp_path, cell)
