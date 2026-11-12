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
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector import glm53flash_attention_tokens as seeded
from collector.fpm_forward import database, native_artifact
from collector.fpm_forward.capabilities import GENERIC_RUNTIME_TEMPLATE_ID
from collector.fpm_forward.native_artifact import (
    NativeCollection,
    _expected_scheduled,
    _validate_execution_provenance,
    select_native_measurements,
    validate_native_collection,
)
from collector.fpm_forward.planner import FPMCell

from aisimulate.fpm_contract import fpm_benchmark_result_name
from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS
from aisimulate_core.sdk.utils import get_model_config_from_model_path

from .test_fpm_forward import _write_v41_token_streams
from .test_fpm_glm53flash_prefix_seed import load_producer
from .test_fpm_runner import _cell, _native_payload, _write_provenance

pytestmark = pytest.mark.unit

GLM_MODEL = "zai-org/GLM-5.3-Flash"
PLAN_SHA256 = "plan-sha"
ATTEMPT_ID = "a1"
EVIDENCE_FILE = "benchmark.repetitions.jsonl"
# The producer files that the collector ships and the launch mounts. The producer records their digests.
PRODUCER_DIR = Path(native_artifact.__file__).parent / "runtime" / "glm53flash"

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
            "pinned_model": GLM_MODEL,
            "pinned_revision": MODEL_REVISIONS[GLM_MODEL],
            "checkpoint_dir": f"/hf-hub/models--zai-org--GLM-5.3-Flash/snapshots/{MODEL_REVISIONS[GLM_MODEL]}",
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
            "mamba_prefill_checkpoint": {"blocks": True, "alignment": 16},
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
            "runtime_scheduler_revision": "9abd2ef0d908c4bb323e1b8956ea76a5772a1d41",
            "runtime_distribution": "ai-dynamo 1.7.0.dev20261109",
            "real_seed_port_source": "ai-dynamo/dynamo@99dae1f53e5a0534c274223ff9daa97f0b1fc2ea "
            "instrumented_scheduler._bench_realseed_*",
            "queued_metrics_port_source": "ai-dynamo/dynamo@9abd2ef0d908c4bb323e1b8956ea76a5772a1d41 "
            "instrumented_scheduler._compute_queued",
            "vllm_package_version": "0.33.0",
            "reviewed_scheduler_api_revision": "4ba787ec9959e566d3b587a796dff07673324da5",
            "runtime_source_manifest_sha256": hashlib.sha256(
                (PRODUCER_DIR / "runtime-source-sha256.json").read_bytes()
            ).hexdigest(),
            "overlay_sha256": hashlib.sha256(
                (PRODUCER_DIR / "glm53flash_prefix_scheduler.py").read_bytes()
            ).hexdigest(),
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


def _glm_producer_shape(tmp_path, breakage=None):
    """Rewrite a GLM-5.3-Flash rank artifact into the shape that the producer writes on Dynamo 1.7.

    The fake of the native scheduler runs the producer once for each phase. Its measurement protocol and the
    ``benchmark_sample`` of its first measured repetition are the templates. Each point then gets the record of the
    native save as the producer rewrites it: ten raw samples of the point's own FPM, each numbered by its position and
    with its own timing interval, and the producer's upper median of samples 1 to 9 as the retained wall time. With no
    global warm-up, Dynamo records an empty warm-up list. ``breakage`` moves the timing interval of one raw sample
    (``interval``) or moves the retained wall time to the slowest repetition (``median``), in the first point of each
    artifact.
    """
    from collector.fpm_forward.measurement_evidence import _content_key

    from .test_fpm_glm53flash_prefix_seed import Engine, Point, _run_prefill_point, make_scheduler
    from .test_fpm_glm53flash_prefix_seed import run as run_producer

    templates, estimates = {}, {}
    with pytest.MonkeyPatch.context() as patch:
        module = load_producer(patch)
        for phase, point in (
            ("prefill", Point("prefill", 1, 1, 32, 120, rows=[[32, 120]])),
            ("decode", Point("decode", 1, 3, 0, 3 * 201)),
        ):
            (tmp_path / f"producer-{phase}").mkdir(parents=True)
            scheduler = make_scheduler(module, tmp_path / f"producer-{phase}", [point])
            if phase == "prefill":
                _run_prefill_point(scheduler, measured_walls=[0.4] * 10)
            else:
                run_producer(scheduler, Engine(scheduler), "decode")
            ((produced,),) = [result.fpms for result in scheduler._bench_results]
            measurement = produced["benchmark_measurement"]
            assert (measurement["expected_internal_samples"], measurement["estimate"]["method"]) == (
                10,
                "adjacent_upper_median",
            )
            templates[phase] = measurement["raw_fpms"][0]["benchmark_sample"]
            estimates[phase] = measurement["estimate"]
        protocol = scheduler._bench_measurement_protocol()

    def shape(phase, payload):
        payload["measurement_protocol"] = copy.deepcopy(protocol)
        payload["warmup_evidence"] = {"status": "recorded", "records": []}
        forward = 0
        for position, (row, group) in enumerate(zip(payload["results"], payload["iteration_groups"], strict=True)):
            fpm = group["rank_results"][0]["fpms"][0]
            assert row["fpms"][0] is fpm
            walls = [fpm["wall_time"] * (1 + 0.01 * (index - 4)) for index in range(10)]
            tokens = row["point"]["total_prefill_tokens"] or row["point"]["batch_size"]
            samples = []
            for index, wall in enumerate(walls):
                sample = copy.deepcopy(templates[phase])
                start = 100.0 * position + index
                # The producer numbers each raw sample by its position, as Dynamo's merge requires.
                sample["sample_index"] = index
                sample["forward_index"] = forward
                sample["timing"].update(start_monotonic=start, end_monotonic=start + wall)
                sample["cudagraph"].update(num_unpadded_tokens=tokens, num_padded_tokens=tokens, num_paddings=0)
                # A decode repetition records two FPMs, and the second is its steady sample.
                forward += 2 if phase == "decode" else 1
                raw = {**copy.deepcopy(fpm), "wall_time": wall, "benchmark_sample": sample}
                if phase == "decode":
                    raw["kvwarm_steady_sample"] = True
                samples.append(raw)
            if phase == "decode":
                fpm["kvwarm_steady_sample"] = True
            # The producer's reduction: the upper median of the samples that its estimate names.
            chosen = sorted(walls[index] for index in estimates[phase]["raw_sample_indices"])
            fpm["wall_time"] = group["wall_time"] = chosen[len(chosen) // 2]
            fpm["benchmark_measurement"] = {
                "schema_version": 1,
                "point_key": _content_key(row["point"]),
                "dp_rank": 0,
                # The producer injects no prompt hashes.
                "prompts": {"status": "unavailable", "sha256": None, "requests": []},
                "preparation": {
                    "grid_digest": payload["grid_digest"],
                    "completed_points_before": position,
                    "kv_seed_regime": row["kv_seed_regime"],
                    "warmup_records_before": 0,
                },
                "expected_internal_samples": 10,
                "raw_fpms": samples,
                "estimate": copy.deepcopy(estimates[phase]),
            }
        first = payload["iteration_groups"][0]["rank_results"][0]["fpms"][0]
        if breakage == "interval":
            first["benchmark_measurement"]["raw_fpms"][3]["benchmark_sample"]["timing"]["start_monotonic"] -= 0.1
        elif breakage == "median":
            # The slowest repetition instead of the median.
            first["wall_time"] = max(sample["wall_time"] for sample in first["benchmark_measurement"]["raw_fpms"])
            payload["iteration_groups"][0]["wall_time"] = first["wall_time"]
        groups = payload["iteration_groups"]
        payload["timing"]["measured_iteration_seconds"] = sum(group["wall_time"] for group in groups)

    return shape


def _pod(tmp_path: Path, cell: FPMCell, *, backend_version: str = "0.33.0") -> Path:
    """The pod folder of a launch whose Collector provenance names the vLLM runtime ``backend_version``."""
    pod = tmp_path / "raw" / "node0000"
    pod.mkdir(parents=True)
    provenance = pod / "collector-provenance.json"
    _write_provenance(provenance, cell_id=cell.cell_id, plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID)
    record = json.loads(provenance.read_text())
    record["runtime"]["backend_version"] = backend_version
    provenance.write_text(json.dumps(record))
    return pod


def _validate(tmp_path: Path, cell: FPMCell):
    return validate_native_collection(
        cell, tmp_path / "raw", expected_plan_sha256=PLAN_SHA256, expected_attempt_id=ATTEMPT_ID
    )


def _glm_plan(template_id: str = GENERIC_RUNTIME_TEMPLATE_ID) -> SimpleNamespace:
    """The parts of a collection plan that publication reads, for the GLM-5.3-Flash checkpoint."""
    return SimpleNamespace(
        sha256=PLAN_SHA256,
        aic_revision="test-revision",
        model_path=GLM_MODEL,
        system="gb300",
        backend="vllm",
        options=SimpleNamespace(warmup_iterations=0),
        capability=SimpleNamespace(
            support_level="native_runtime", template_id=template_id, template_version=1, aic_database_version="0.33.0"
        ),
    )


def _restamp(measurement, stamps: list[str], regime: str):
    """The measurement of a point that carries ``stamps`` and whose row reports ``regime``."""
    return replace(
        measurement, point={**measurement.point, "sample_reasons": ["explicit", *stamps]}, kv_seed_regime=regime
    )


def _publish(monkeypatch, tmp_path: Path, cell: FPMCell, collection: NativeCollection, plan=None) -> list[dict]:
    """Publish ``collection`` as if the native validator had returned it.

    The validator refuses artifacts that the producer never writes, such as a GLM decode artifact with a warm-up
    skip. A test starts from a collection that the validator accepted and changes it, to reach the publication rules.
    The measurement record reader checks the artifact against the validated collection, so it would refuse every
    changed collection before these rules. It is replaced too, and its own rules are tested on unchanged artifacts.
    """
    monkeypatch.setattr(database, "validate_native_collection", lambda *_args, **_kwargs: collection)
    monkeypatch.setattr(database, "extract_measurement_evidence", lambda *_args, **_kwargs: {})
    return database.aggregate_cell(plan or _glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)


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


_V41_IDENTITY = ("c" * 64, "full", "hbm_tp_sharded", "text")


def test_v41_cell_refuses_an_artifact_that_declares_the_glm_contract(tmp_path):
    # The frozen cell, not the artifact, selects the producer contract. A V4.1 artifact that declares the GLM mode
    # with its own execution identity and the GLM provenance would otherwise skip the V4.1 eager and text checks.
    cell = replace(
        _cell(phase="prefill", strategy="pure_tp"), execution_identity=_V41_IDENTITY, input_text_sha256="a" * 64
    )
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["execution_identity"] = dict(zip(EXECUTION_COLUMNS, _V41_IDENTITY, strict=True))
    artifact = write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError) as error:
        _validate(tmp_path, cell)
    assert str(error.value) == (
        f"native execution identity cell expects execution mode 'eager', found 'native_graph_policy': {artifact}"
    )


def test_glm_cell_refuses_an_artifact_that_declares_the_v41_contract(tmp_path):
    # The reverse case: the GLM cell accepts only the GLM producer's mode, whatever V4.1 evidence the artifact holds.
    cell = glm_cell("prefill")
    payload = _native_payload(phase="prefill", rank=0, dp=1)
    payload.update(
        execution_identity=dict(zip(EXECUTION_COLUMNS, glm_identity(), strict=True)),
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
    artifact = _pod(tmp_path, cell) / "benchmark.json"
    _write_v41_token_streams(payload, artifact)
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as error:
        _validate(tmp_path, cell)
    assert str(error.value) == (
        f"native execution identity cell expects execution mode 'native_graph_policy', found 'eager': {artifact}"
    )


def test_cell_without_an_audited_contract_refuses_an_audited_mode(tmp_path):
    # An execution identity whose Engram residency no producer contract has stays refused, whatever mode it declares.
    identity = ("c" * 64, "full", "cpu_offloaded", "text")
    cell = SimpleNamespace(execution_identity=identity, input_text_sha256="a" * 64)
    payload = {"execution_identity": dict(zip(EXECUTION_COLUMNS, identity, strict=True)), "execution_mode": "eager"}
    path = tmp_path / "benchmark.json"
    with pytest.raises(ValueError) as error:
        _validate_execution_provenance(cell, payload, path)
    assert str(error.value) == (
        f"native execution identity cell has no audited producer contract: 'cpu_offloaded': {path}"
    )


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
    pytest.param("input_provenance.seed", lambda p, _: _evidence(p).update(seed=-1), id="seed-negative"),
    pytest.param("input_provenance.seed", lambda p, _: _evidence(p).update(seed=2**64), id="seed-above-64-bits"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count="262240"), id="count"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count=0), id="count-zero"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count=-1), id="count-negative"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count=True), id="count-is-a-boolean"),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=_evidence(p)["count"] + 1),
        id="unique-count-above-count",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=0),
        id="unique-count-zero",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=True),
        id="unique-count-is-a-boolean",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).pop("unique_token_count"),
        id="unique-count-absent",
    ),
    # The pinned vocabulary has 154,820 ids, so a stream of 262,240 tokens cannot hold more distinct tokens than that.
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=_evidence(p)["count"]),
        id="unique-count-equals-count-above-the-vocabulary",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=seeded.VOCAB_HIGH - seeded.VOCAB_LOW + 1),
        id="unique-count-one-above-the-vocabulary",
    ),
    # The vocabulary is the half-open range of its low and high ids, so ids 50 to 149 are 100 ids and not 150.
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(vocab_low=50, vocab_high_exclusive=150, unique_token_count=101),
        id="unique-count-above-a-vocabulary-that-starts-above-zero",
    ),
    pytest.param("input_provenance.vocab_low", lambda p, _: _evidence(p).update(vocab_low=-1), id="vocab-low-negative"),
    pytest.param(
        "input_provenance.vocab_low", lambda p, _: _evidence(p).update(vocab_low=False), id="vocab-low-is-a-boolean"
    ),
    pytest.param(
        "input_provenance.vocab_low", lambda p, _: _evidence(p).update(vocab_low=0.0), id="vocab-low-is-a-float"
    ),
    pytest.param("input_provenance.vocab_low", lambda p, _: _evidence(p).pop("vocab_low"), id="vocab-low-absent"),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=True),
        id="vocab-high-is-a-boolean",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=float(seeded.VOCAB_HIGH)),
        id="vocab-high-is-a-float",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).pop("vocab_high_exclusive"),
        id="vocab-high-absent",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=_evidence(p)["vocab_low"]),
        id="vocab-range-empty",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_low=10, vocab_high_exclusive=5),
        id="vocab-range-inverted",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=-1),
        id="vocab-high-negative",
    ),
    pytest.param(
        "input_provenance.vocab_low",
        lambda p, _: _evidence(p).update(vocab_low=-10, vocab_high_exclusive=-5),
        id="vocab-range-negative",
    ),
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
    pytest.param("producer", lambda p, _: p.pop("producer"), id="producer-absent"),
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


def test_glm_producer_token_evidence_passes_for_real_and_boundary_values(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    # The producer draws its largest measured context of 131,072 tokens plus a window of 4,099 tokens for each of up
    # to 32 requests, from seed 53 over the pinned vocabulary of 154,820 ids. Its distinct-token count is that of the
    # real stream, which does not draw every id.
    count = producer.MAX_CONTEXT + producer.MAX_BATCH * seeded.REQUEST_STRIDE
    tokens = seeded.generate(seeded.SEED, count, seeded.VOCAB_LOW, seeded.VOCAB_HIGH)
    assert (seeded.SEED, count, len(set(tokens))) == (53, 262240, 126418)
    cases = {
        "producer": {"seed": seeded.SEED, "count": count, "unique_token_count": len(set(tokens))},
        # The smallest seed and a stream of one token.
        "smallest": {"seed": 0, "count": 1, "unique_token_count": 1},
        # The largest seed and a stream that draws every id of the vocabulary, the most distinct tokens that it allows.
        "largest": {"seed": 2**64 - 1, "count": count, "unique_token_count": seeded.VOCAB_HIGH - seeded.VOCAB_LOW},
        # The ids 50 to 149 are 100 ids, because the upper bound is exclusive. They cap the distinct tokens of a longer
        # stream at 100.
        "narrow-vocabulary": {"count": count, "unique_token_count": 100, "vocab_low": 50, "vocab_high_exclusive": 150},
    }
    for name, fields in cases.items():
        cell = glm_cell("prefill")
        payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
        payload["input_provenance"].update(fields)
        write_glm_rank(_pod(tmp_path / name, cell), payload)
        assert len(_validate(tmp_path / name, cell).points) == 2


@pytest.mark.parametrize(
    ("high", "accepted"),
    [(2**63 + 1, True), (2**63 + 2, False)],
    ids=["range-at-the-generator-cap", "range-above-the-generator-cap"],
)
def test_glm_producer_vocabulary_range_follows_the_generator_cap(tmp_path, high, accepted):
    # The generator takes at most 2**63 ids in its range, so the validator takes the same. A lower bound of 1 puts
    # exactly 2**63 ids in the first range and one more in the second, and both upper bounds exceed 2**63.
    low = 1
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["input_provenance"].update(vocab_low=low, vocab_high_exclusive=high)
    write_glm_rank(_pod(tmp_path, cell), payload)
    if accepted:
        assert len(seeded.generate(seeded.SEED, 1, low, high)) == 1
        assert len(_validate(tmp_path, cell).points) == 2
    else:
        with pytest.raises(ValueError, match="invalid token generator request"):
            seeded.generate(seeded.SEED, 1, low, high)
        with pytest.raises(
            ValueError, match=r"native GLM-5\.3-Flash result lacks valid input_provenance\.vocab_high_exclusive: "
        ):
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


_NO_KEY = "no key"
_WITH_CHECKPOINT = {"blocks": True, "alignment": 16}
_WITHOUT_CHECKPOINT = {"blocks": False, "alignment": None}


@pytest.mark.parametrize(
    ("first", "second", "refused"),
    [
        pytest.param(_WITH_CHECKPOINT, _WITHOUT_CHECKPOINT, True, id="different-values"),
        pytest.param(_NO_KEY, _WITH_CHECKPOINT, True, id="first-rank-lacks-the-key"),
        pytest.param(_WITH_CHECKPOINT, _NO_KEY, True, id="second-rank-lacks-the-key"),
        pytest.param(_NO_KEY, _NO_KEY, False, id="no-rank-has-the-key"),
        pytest.param(_WITH_CHECKPOINT, dict(_WITH_CHECKPOINT), False, id="equal-values"),
    ],
)
def test_collection_refuses_ranks_that_disagree_on_the_checkpoint_evidence(tmp_path, first, second, refused):
    cell = glm_cell("prefill", dp=2)
    pod = _pod(tmp_path, cell)
    payloads = [glm_native_payload("prefill", points=[PREFILL_COLD], rank=rank, dp=2) for rank in range(2)]
    for payload, value in zip(payloads, (first, second), strict=True):
        if value == _NO_KEY:
            del payload["serving_config"]["mamba_prefill_checkpoint"]
        else:
            payload["serving_config"]["mamba_prefill_checkpoint"] = value
    for payload in payloads:
        write_glm_rank(pod, payload)
    if refused:
        with pytest.raises(
            ValueError,
            match=r"^native ranks disagree on serving_config\.mamba_prefill_checkpoint: .+benchmark_dp1\.json$",
        ):
            _validate(tmp_path, cell)
    else:
        assert len(_validate(tmp_path, cell).points) == 1


REAL_KV_REFUSAL = "execution-identity cells require real KV measurements for cached prefill and decode, got {}"
# Dynamo skips the decode warm-up of a pure tensor-parallel MoE model and stamps every point as a fake fallback.
WARMUP_SKIP = {"enabled": True, "warm_eligible": False, "skip_reason": "moe_tp_balanced_by_construction"}


def test_glm_producer_rows_publish_real_regimes(tmp_path):
    rows = {}
    shape = _glm_producer_shape(tmp_path)
    for phase, points in (("prefill", [PREFILL_COLD, PREFILL_CACHED]), ("decode", DECODE_POINTS)):
        cell = glm_cell(phase)
        payload = glm_native_payload(phase, points=points)
        shape(phase, payload)
        write_glm_rank(_pod(tmp_path / phase, cell), payload)
        rows[phase] = database.aggregate_cell(_glm_plan(), cell, tmp_path / phase, expected_attempt_id=ATTEMPT_ID)
    assert [(row["total_kv_read_tokens"], row["kv_seed_regime"]) for row in rows["prefill"]] == [
        (0, "n/a"),
        (2176, "real_kv"),
    ]
    assert [(row["total_kv_read_tokens"], row["kv_seed_regime"]) for row in rows["decode"]] == [
        (4096, "real_kv"),
        (8192, "real_kv"),
    ]
    for row in [*rows["prefill"], *rows["decode"]]:
        assert row["model_template_id"] == GENERIC_RUNTIME_TEMPLATE_ID
        assert row["execution_profile"] == "full"
        assert tuple(row[name] for name in EXECUTION_COLUMNS) == glm_identity()


def test_glm_publication_keeps_the_retained_values_of_a_producer_record(tmp_path):
    shape = _glm_producer_shape(tmp_path)
    for phase, points in (("prefill", [PREFILL_COLD, PREFILL_CACHED]), ("decode", DECODE_POINTS)):
        cell = glm_cell(phase)
        payload = glm_native_payload(phase, points=points)
        original = [row["fpms"][0]["wall_time"] for row in payload["results"]]
        shape(phase, payload)
        retained = [row["fpms"][0]["wall_time"] for row in payload["results"]]
        # The builder spreads ten repetitions around the original wall time, 1% apart from 0.96 to 1.05 times it, and
        # retains the upper median of repetitions 1 to 9: repetition 5, at 1.01 times the original.
        assert retained == [wall * (1 + 0.01 * (5 - 4)) for wall in original]
        write_glm_rank(_pod(tmp_path / phase, cell), payload)
        rows = database.aggregate_cell(_glm_plan(), cell, tmp_path / phase, expected_attempt_id=ATTEMPT_ID)
        # One rank, so the latency of a row is the retained wall time of its point.
        assert [row["latency_ms"] for row in rows] == [wall * 1000.0 for wall in retained]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_glm_publication_refuses_a_producer_record_whose_retained_median_moved(tmp_path, phase):
    cell = glm_cell(phase)
    points = [PREFILL_COLD, PREFILL_CACHED] if phase == "prefill" else DECODE_POINTS
    payload = glm_native_payload(phase, points=points)
    _glm_producer_shape(tmp_path, "median")(phase, payload)
    write_glm_rank(_pod(tmp_path, cell), payload)
    # The retained wall time, the group wall time and the timing totals moved together to the slowest repetition, so
    # the native validator accepts the artifact. Only the reduction rule of its measurement record refuses it.
    _validate(tmp_path, cell)
    with pytest.raises(ValueError) as error:
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)
    assert "benchmark_measurement reduction disagrees with retained timing" in str(error.value)


def test_glm_publication_refuses_an_artifact_without_the_producer_protocol(tmp_path):
    # The producer writes its measurement protocol into every rank artifact, so an artifact without it is not the
    # producer's, although the native validator accepts it.
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    _validate(tmp_path, cell)
    with pytest.raises(ValueError) as error:
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)
    assert "measurement_protocol of a GLM-5.3-Flash producer artifact must be present" in str(error.value)


@pytest.mark.parametrize(
    ("stamps", "regime", "template_id", "published"),
    [
        # The native row reports real_prefix, and publication labels the same prefix real_kv.
        pytest.param(["prefill_real_seed"], "real_prefix", GENERIC_RUNTIME_TEMPLATE_ID, "real_kv", id="real-seed"),
        pytest.param(["prefill_fake_prefix"], "fake_prefix", GENERIC_RUNTIME_TEMPLATE_ID, None, id="fake-prefix"),
        # The DeepSeek V4.1 adapter stamps real KV on every point, cached prefill included. Its plans have their own
        # template, and a point with this stamp keeps the label that it had before the prefix stamps counted.
        pytest.param(["kvwarm_real_kv"], "real_kv", "deepseek_v41_text_native", "real_kv", id="real-kv"),
        pytest.param(
            ["prefill_real_seed", "kvwarm_real_kv"], "real_kv", "deepseek_v41_text_native", "real_kv", id="both"
        ),
    ],
)
def test_execution_identity_cached_prefill_accepts_real_prefix_stamp(
    tmp_path, monkeypatch, stamps, regime, template_id, published
):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    valid = _validate(tmp_path, cell)
    cold, cached = valid.points
    collection = replace(valid, points=(cold, _restamp(cached, stamps, regime)))
    plan = _glm_plan(template_id)
    if published is None:
        with pytest.raises(ValueError) as error:
            _publish(monkeypatch, tmp_path, cell, collection, plan)
        assert str(error.value) == REAL_KV_REFUSAL.format(regime)
    else:
        rows = _publish(monkeypatch, tmp_path, cell, collection, plan)
        assert [row["kv_seed_regime"] for row in rows] == ["n/a", published]


@pytest.mark.parametrize("kvwarm", [None, WARMUP_SKIP], ids=["no-warmup-block", "warmup-skip"])
def test_cached_prefill_regime_follows_its_stamp_not_the_decode_warmup_block(tmp_path, monkeypatch, kvwarm):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    collection = replace(_validate(tmp_path, cell), kvwarm_meta=kvwarm)
    rows = _publish(monkeypatch, tmp_path, cell, collection)
    assert [row["kv_seed_regime"] for row in rows] == ["n/a", "real_kv"]


def test_cached_prefill_with_the_real_kv_stamp_follows_the_warmup_block(tmp_path, monkeypatch):
    """No adapter writes both stamps. A point with the real-KV stamp is classified by the warm-up block, as before."""
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    valid = _validate(tmp_path, cell)
    cold, cached = valid.points
    both = _restamp(cached, ["prefill_real_seed", "kvwarm_real_kv"], "real_prefix")
    collection = replace(valid, points=(cold, both), kvwarm_meta=WARMUP_SKIP)
    with pytest.raises(ValueError) as error:
        _publish(monkeypatch, tmp_path, cell, collection)
    assert str(error.value) == REAL_KV_REFUSAL.format("skip:moe_tp_balanced_by_construction")


def test_glm_identity_decode_without_real_kv_is_refused(tmp_path, monkeypatch):
    cell = glm_cell("decode")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("decode", points=DECODE_POINTS))
    valid = _validate(tmp_path, cell)
    fallback = tuple(_restamp(measurement, ["kvwarm_fake_fallback"], "fake_fallback") for measurement in valid.points)
    skipped = replace(valid, kvwarm_meta=WARMUP_SKIP, points=fallback)
    with pytest.raises(ValueError) as error:
        _publish(monkeypatch, tmp_path, cell, skipped)
    assert str(error.value) == REAL_KV_REFUSAL.format("skip:moe_tp_balanced_by_construction")


def test_glm_decode_artifact_with_a_warmup_skip_is_refused_at_validation(tmp_path):
    cell = glm_cell("decode")
    fallback_points = [{**point, "sample_reasons": ["explicit", "kvwarm_fake_fallback"]} for point in DECODE_POINTS]
    payload = glm_native_payload("decode", points=fallback_points)
    payload["kvwarm"] = dict(WARMUP_SKIP)
    for row in payload["results"]:
        row["kv_seed_regime"] = "fake_fallback"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError, match="native GLM-5.3-Flash result lacks valid kvwarm"):
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)


def test_published_glm_pair_loads_through_the_rust_fpm_loader(tmp_path):
    """The loader refuses a cached prefill or decode row of a cell with an execution identity unless it is real_kv."""
    import aisimulate_core
    from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    plan = _glm_plan()
    plan.backend_version = "glm-published-pair"
    rows = []
    shape = _glm_producer_shape(tmp_path)
    for phase, points in (("prefill", [PREFILL_COLD, PREFILL_CACHED]), ("decode", DECODE_POINTS)):
        # The loader selects cells by the quantization that the SDK resolves for the checkpoint.
        cell = replace(
            glm_cell(phase),
            weight_quantization="fp8_block",
            gemm_quant_mode="fp8_block",
            moe_quant_mode="fp8_block",
            fmha_quant_mode="bfloat16",
        )
        payload = glm_native_payload(phase, points=points)
        shape(phase, payload)
        write_glm_rank(_pod(tmp_path / phase, cell), payload)
        rows += database.aggregate_cell(plan, cell, tmp_path / phase, expected_attempt_id=ATTEMPT_ID)
    systems = tmp_path / "systems"
    database.write_formal_database(plan, rows, systems_root=systems / "data")
    system_file = f"{plan.system}.yaml"
    shutil.copyfile(Path(aisimulate_core.__file__).parent / "systems" / system_file, systems / system_file)
    engine = dict(
        schema_version=1,
        model_name=GLM_MODEL,
        system_name=plan.system,
        backend=plan.backend,
        backend_version=plan.backend_version,
        systems_path=str(systems),
        enable_shared_layer=False,
        strict_provenance=True,
        tp_size=1,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        database_mode="SILICON",
        forward_model="fpm",
    )
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig.from_legacy_engine_config(engine, "aggregated")
    )
    # Each published point is a measured coordinate, so the model returns its latency.
    for row in rows:
        decode = row["workload_kind"] == "decode"
        scheduled = dict(
            num_prefill_requests=0 if decode else row["batch_size"],
            num_decode_requests=row["batch_size"] if decode else 0,
            sum_prefill_tokens=row["total_prefill_tokens"],
            sum_prefill_kv_tokens=0 if decode else row["total_kv_read_tokens"],
            sum_decode_kv_tokens=row["total_kv_read_tokens"] if decode else 0,
            var_prefill_length=0.0,
            var_decode_kv_tokens=0.0,
        )
        fpm = dict(version=1, wall_time=1.0, scheduled_requests=scheduled)
        assert model.estimate_forward_pass_time_ms(fpm) == pytest.approx(row["latency_ms"])


def test_glm_producer_record_must_name_the_supported_vllm(tmp_path):
    # vLLM 0.33.1 is another runtime, although its version shares the release line of the supported one.
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["producer"]["vllm_package_version"] = "0.33.1"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(
        ValueError,
        match=r"^native GLM-5\.3-Flash result lacks valid producer\.vllm_package_version: expected '0\.33\.0', "
        r"found '0\.33\.1': .+benchmark\.json$",
    ):
        _validate(tmp_path, cell)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instrumentation_revision", "0" * 40),
        ("runtime_scheduler_revision", "0" * 40),
        ("runtime_distribution", "ai-dynamo 1.6.0.dev20261010"),
        ("real_seed_port_source", "ai-dynamo/dynamo@main instrumented_scheduler._bench_realseed_*"),
        ("queued_metrics_port_source", "ai-dynamo/dynamo@main instrumented_scheduler._compute_queued"),
        ("vllm_package_version", "0.33.0+local"),
        ("reviewed_scheduler_api_revision", "0" * 40),
        ("runtime_source_manifest_sha256", "0" * 64),
        ("overlay_sha256", "0" * 64),
        # The same number in another JSON type is another record.
        ("context_policy_version", 1.0),
        ("warmup_repeats", 4),
        ("measurement_repeats", 9),
        ("measurement_repeats", None),
    ],
)
def test_glm_producer_record_must_be_the_supported_producers(tmp_path, field, value):
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    expected = payload["producer"][field]
    if value is None:
        del payload["producer"][field]
    else:
        payload["producer"][field] = value
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError) as refusal:
        _validate(tmp_path, cell)
    assert re.fullmatch(
        rf"native GLM-5\.3-Flash result lacks valid producer\.{field}: expected {re.escape(repr(expected))}, "
        rf"found {re.escape(repr(value))}: .+benchmark\.json",
        str(refusal.value),
    ), str(refusal.value)


def test_glm_producer_record_must_not_add_a_field(tmp_path):
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["producer"]["patched"] = True
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(
        ValueError,
        match=r"^native GLM-5\.3-Flash result lacks valid producer\.patched: expected no such field, found True: "
        r".+benchmark\.json$",
    ):
        _validate(tmp_path, cell)


def test_glm_producer_record_digests_are_those_of_the_shipped_producer_files():
    record = native_artifact._glm_producer_record()
    for field, name in (
        ("runtime_source_manifest_sha256", "runtime-source-sha256.json"),
        ("overlay_sha256", "glm53flash_prefix_scheduler.py"),
    ):
        assert record[field] == hashlib.sha256((PRODUCER_DIR / name).read_bytes()).hexdigest()
    assert record["vllm_package_version"] == "0.33.0"


def test_glm_producer_vllm_must_be_the_collection_runtime(tmp_path):
    # The producer and the record are those of vLLM 0.33.0, and the Collector provenance binds the launch to 0.33.1.
    cell = glm_cell("prefill")
    write_glm_rank(
        _pod(tmp_path, cell, backend_version="0.33.1"),
        glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]),
    )
    with pytest.raises(
        ValueError,
        match=r"^native GLM-5\.3-Flash producer\.vllm_package_version differs from the Collector provenance runtime: "
        r"expected '0\.33\.1', found '0\.33\.0': .+benchmark\.json$",
    ):
        _validate(tmp_path, cell)


def test_glm_collection_carries_the_contract_of_its_cell(tmp_path):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    assert _validate(tmp_path, cell).execution_mode == "native_graph_policy"
