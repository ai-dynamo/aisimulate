# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict reader for Dynamo PR11509 native self-benchmark rank artifacts."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisim.fpm_contract import (
    FPM_BENCHMARK_RESULT_GLOB,
    FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
)
from aisim_core.sdk.fpm_identity import EXECUTION_COLUMNS
from collector.glm53flash_runtime_identity import BASELINE_VERSIONS

from .planner import FPMCell
from .types import KVWARM_STRATEGIES

logger = logging.getLogger(__name__)

COLLECTOR_PROVENANCE_FILENAME = "collector-provenance.json"


@dataclass(frozen=True, slots=True)
class NativePointMeasurement:
    point: dict[str, Any]
    rank_wall_times: tuple[tuple[int, float], ...]
    # Keep native row provenance for diagnostics, without reinterpreting the
    # formal database's historical prefill kv_seed_regime contract.
    kv_seed_regime: str | None = None


# Distinguishes "no artifact seen yet" from a legitimately absent (legacy)
# kvwarm block during cross-rank consistency checking.
_KVWARM_UNSEEN = object()
# The value of a rank whose serving configuration holds no checkpoint evidence, so that it differs from any record.
_NO_CHECKPOINT_EVIDENCE = object()


def _validate_seed_regime(row: dict[str, Any], path: Path) -> str | None:
    regime = row.get("kv_seed_regime")
    if regime is None:
        return None  # Historical artifacts did not report a row-level regime.
    if not isinstance(regime, str) or not regime:
        raise ValueError(f"native kv_seed_regime must be a non-empty string: {path}")
    # Dynamo's _kvwarm_seed_regime reports these point-injection stamps:
    # https://github.com/ai-dynamo/dynamo/blob/b83b1d9304ebfc624709ac46db32b1b6f1ff1615/components/src/dynamo/vllm/instrumented_scheduler.py#L4721
    stamps = {
        "real_prefix": "prefill_real_seed",
        "fake_prefix": "prefill_fake_prefix",
        "real_kv": "kvwarm_real_kv",
        "fake_fallback": "kvwarm_fake_fallback",
    }
    reasons = set(row["point"].get("sample_reasons") or [])
    observed_stamps = reasons.intersection(stamps.values())
    if regime in stamps and observed_stamps != {stamps[regime]}:
        raise ValueError(f"native kv_seed_regime disagrees with point injection stamps: {path}")
    if regime in {"real_prefix", "fake_prefix"} and row["point"]["point_type"] != "prefill":
        raise ValueError(f"native prefix seed regime requires prefill: {path}")
    # Real-seed staging stamps the path before checking whether any KV needs
    # seeding; an uncached prefill can therefore retain real_prefix provenance.
    # https://github.com/ai-dynamo/dynamo/blob/b83b1d9304ebfc624709ac46db32b1b6f1ff1615/components/src/dynamo/vllm/instrumented_scheduler.py#L4298
    if regime == "fake_prefix" and row["point"]["total_kv_read_tokens"] <= 0:
        raise ValueError(f"native fake prefix seed regime requires cached prefill: {path}")
    if regime not in stamps and observed_stamps:
        raise ValueError(f"native kv_seed_regime disagrees with point injection stamps: {path}")
    return regime


def _validate_kvwarm_contract(cell: FPMCell, kvwarm: object, path: Path) -> dict[str, Any] | None:
    """Validate that engine protocol evidence matches the rendered strategy."""

    if cell.workload_kind != "decode":
        if kvwarm is not None and not isinstance(kvwarm, dict):
            raise TypeError(f"native kvwarm block is not a mapping: {path}")
        return kvwarm
    requires_warm = cell.parallel_strategy in KVWARM_STRATEGIES
    if kvwarm is None:
        if requires_warm:
            raise ValueError(
                f"native decode result lacks required KV warm-up metadata for strategy "
                f"{cell.parallel_strategy!r}: {path}"
            )
        return None
    if not isinstance(kvwarm, dict):
        raise TypeError(f"native kvwarm block is not a mapping: {path}")
    enabled = kvwarm.get("enabled")
    warm_eligible = kvwarm.get("warm_eligible")
    skip_reason = kvwarm.get("skip_reason")
    if not isinstance(enabled, bool):
        raise TypeError(f"native kvwarm.enabled must be a boolean: {path}")
    if not isinstance(warm_eligible, bool):
        raise TypeError(f"native kvwarm.warm_eligible must be a boolean: {path}")
    if warm_eligible and skip_reason is not None:
        raise ValueError(f"native warm-eligible result must not declare kvwarm.skip_reason: {path}")
    if not warm_eligible and (not isinstance(skip_reason, str) or not skip_reason):
        raise ValueError(f"native warm-ineligible result must declare a non-empty kvwarm.skip_reason: {path}")
    if requires_warm and (not enabled or not warm_eligible):
        raise ValueError(
            f"native KV warm-up protocol mismatch for strategy {cell.parallel_strategy!r}: "
            f"enabled={enabled!r}, skip_reason={skip_reason!r}: {path}"
        )
    return kvwarm


@dataclass(frozen=True, slots=True)
class NativeCollection:
    points: tuple[NativePointMeasurement, ...]
    rank_timings: tuple[tuple[int, float, float], ...]
    backend_version: str
    collector_attempt_id: str
    runtime_run_id: str
    runtime_grid_digest: str
    # Engine-reported KV warm-up envelope (warm_eligible/skip_reason/...);
    # None only for artifacts predating the kvwarm-enabled runtime.
    kvwarm_meta: dict[str, Any] | None = None
    input_provenance: dict[str, Any] | None = None
    # The execution mode of the producer contract that the frozen cell selected: "eager" for DeepSeek V4.1,
    # "native_graph_policy" for GLM-5.3-Flash, None for a cell without an execution identity. Readers of the
    # measurement records apply the contract's rules by this value, never by the artifact's own fields.
    execution_mode: str | None = None


def _zero_kv_prefill_sample(measurements: list[NativePointMeasurement]) -> NativePointMeasurement | None:
    """Prefer the unique ordinary sample over equivalent real-prefix path samples."""
    ordinary = [m for m in measurements if m.kv_seed_regime == "not_applicable"]
    if len(ordinary) != 1:
        return None
    sample = ordinary[0]
    if sample.point["point_type"] != "prefill" or sample.point["total_kv_read_tokens"] != 0:
        return None
    expected = {key: value for key, value in sample.point.items() if key != "benchmark_id"}
    expected["sample_reasons"] = list(expected.get("sample_reasons") or [])
    for measurement in measurements:
        if measurement is sample:
            continue
        if measurement.kv_seed_regime != "real_prefix":
            return None
        point = {key: value for key, value in measurement.point.items() if key != "benchmark_id"}
        point["sample_reasons"] = [
            reason for reason in (point.get("sample_reasons") or []) if reason != "prefill_real_seed"
        ]
        if point != expected:
            return None
    return sample


def select_native_measurements(collection: NativeCollection, *, cell_id: str) -> list[NativePointMeasurement]:
    """Select the formal per-coordinate samples from one validated attempt.

    Used by publication and by repeatability source selection, so the original
    reference timing is identical. Never combine independent repeat attempts.
    """
    # The steady-state decode policy clamps every per-sequence context below
    # the measurable minimum up to it, so several requested plan points can
    # collapse onto one achieved physical coordinate (e.g. batch=3 with
    # requested total-kv 3/4/5/6 all measure total-kv 6). Repeated samples of
    # one coordinate would violate the database's unique-key contract, so keep
    # exactly one per coordinate: the native (unclamped) sample when present,
    # otherwise the clamped sample with the lowest benchmark_id.
    # Real-prefix staging can also emit zero-KV prefill samples beside the
    # ordinary sample of the same work. Only this provenance difference is
    # consolidatable: keep the unique ordinary sample with identical remaining
    # point metadata. Other native collisions remain a hard error. Raw samples
    # stay intact; neither reduction averages timings nor selects by latency.
    grouped: dict[tuple[str, int, int, int], list[NativePointMeasurement]] = {}
    for measurement in collection.points:
        point = measurement.point
        key = (
            str(point["point_type"]),
            int(point["batch_size"]),
            int(point["total_prefill_tokens"]),
            int(point["total_kv_read_tokens"]),
        )
        grouped.setdefault(key, []).append(measurement)
    selected: list[NativePointMeasurement] = []
    dropped_clamped = 0
    dropped_zero_kv = 0
    for key, measurements in grouped.items():
        natives = [m for m in measurements if "context_clamped" not in (m.point.get("sample_reasons") or ())]
        if len(natives) > 1:
            sample = _zero_kv_prefill_sample(measurements)
            if sample is not None:
                selected.append(sample)
                dropped_zero_kv += len(measurements) - 1
                continue
            raise ValueError(
                f"conflicting FPM measurements for physical coordinate {key} in "
                f"{cell_id}: {len(natives)} unclamped samples share one key"
            )
        if natives:
            selected.append(natives[0])
        else:
            selected.append(min(measurements, key=lambda m: int(m.point["benchmark_id"])))
        dropped_clamped += len(measurements) - 1
    if dropped_clamped:
        logger.info(
            "FPM %s: consolidated %d context-clamped duplicate sample(s) onto their achieved physical coordinates",
            cell_id,
            dropped_clamped,
        )
    if dropped_zero_kv:
        logger.info(
            "FPM %s: consolidated %d zero-KV prefill provenance duplicate sample(s) onto their ordinary samples",
            cell_id,
            dropped_zero_kv,
        )

    return selected


def _validate_execution_provenance(cell: FPMCell, payload: dict[str, Any], path: Path) -> dict[str, Any] | None:
    """Config-bound curves require engine evidence for execution and real input.

    The frozen cell selects the producer contract, never the artifact. The planner computes the cell's execution
    identity from the model configuration, and the cell ID and the plan hash cover it. A DeepSeek V4.1 cell accepts
    only the ``eager`` mode of its producer and a GLM-5.3-Flash cell only ``native_graph_policy``. A mode that no
    contract audits is refused, never read as one of them.
    """
    if not cell.execution_identity[0]:
        return None
    # The execution mode of each contract, by the Engram residency of the cell's execution identity.
    # aisim_core.sdk.fpm_identity.execution_identity gives DeepSeek V4.1 "hbm_tp_sharded" and GLM-5.3-Flash
    # "none", and every other model the legacy identity, which has no configuration hash and returned above.
    contract_modes = {"hbm_tp_sharded": "eager", "none": "native_graph_policy"}
    mode = payload.get("execution_mode")
    if mode not in contract_modes.values():
        raise ValueError(f"native execution identity cell has an unaudited execution mode: {mode!r}: {path}")
    residency = dict(zip(EXECUTION_COLUMNS, cell.execution_identity, strict=True))["engram_residency"]
    expected_mode = contract_modes.get(residency)
    if expected_mode is None:
        raise ValueError(f"native execution identity cell has no audited producer contract: {residency!r}: {path}")
    if mode != expected_mode:
        raise ValueError(
            f"native execution identity cell expects execution mode {expected_mode!r}, found {mode!r}: {path}"
        )
    if mode == "native_graph_policy":
        return _validate_glm_provenance(cell, payload, path)
    expected = dict(zip(EXECUTION_COLUMNS, cell.execution_identity, strict=True))
    if payload.get("execution_identity") != expected:
        raise ValueError(f"native execution identity differs from the frozen V4.1 cell: {path}")
    evidence = payload.get("input_provenance")
    if not isinstance(evidence, dict) or evidence.get("source") != "tokenizer_text":
        raise ValueError(f"V4.1 native result requires tokenizer-generated text provenance: {path}")
    for field in ("text_sha256", "token_ids_sha256"):
        value = evidence.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"V4.1 native result has invalid {field}: {path}")
    if cell.input_text_sha256 and evidence["text_sha256"] != cell.input_text_sha256:
        raise ValueError(f"V4.1 native input text differs from the frozen corpus: {path}")
    if not isinstance(evidence.get("tokenizer_revision"), str) or not evidence["tokenizer_revision"]:
        raise ValueError(f"V4.1 native result has no tokenizer revision: {path}")
    counts = [evidence.get(field) for field in ("token_count", "unique_token_count")]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 2 for value in counts):
        raise ValueError(f"V4.1 native input corpus must contain multiple tokenizer-generated tokens: {path}")
    if counts[1] > counts[0]:
        raise ValueError(f"V4.1 native input corpus token counts are inconsistent: {path}")
    return evidence


def _same(value: object, expected: object) -> bool:
    """Equality that does not let 1, 1.0 and True stand in for one another."""
    return type(value) is type(expected) and value == expected


def _is_hex(value: object, digits: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{digits}}}", value) is not None


_GLM_RUNTIME = Path(__file__).parent / "runtime" / "glm53flash"


@functools.cache
def _glm_producer_record() -> dict[str, Any]:
    """The ``producer`` record that the supported GLM-5.3-Flash vLLM producer writes.

    The producer (``_glm_enrich`` of runtime/glm53flash/glm53flash_prefix_scheduler.py) imports Dynamo and vLLM, so the
    reader cannot import its constants. The reader keeps them here, and a test keeps them equal to the record that the
    producer writes. The vLLM version is the exact supported runtime. The two digests are those of the pin manifest and
    of the producer file that this collector ships and that a launch of this collector mounts; the producer computes
    both the same way. A campaign of another producer revision is therefore refused.
    """
    return {
        "instrumentation_revision": "54960177085413259859c88bd34ed0734d4c2ea9",
        "runtime_scheduler_revision": "9abd2ef0d908c4bb323e1b8956ea76a5772a1d41",
        "runtime_distribution": "ai-dynamo 1.7.0.dev20261109",
        "real_seed_port_source": (
            "ai-dynamo/dynamo@99dae1f53e5a0534c274223ff9daa97f0b1fc2ea instrumented_scheduler._bench_realseed_*"
        ),
        "queued_metrics_port_source": (
            "ai-dynamo/dynamo@9abd2ef0d908c4bb323e1b8956ea76a5772a1d41 instrumented_scheduler._compute_queued"
        ),
        "vllm_package_version": BASELINE_VERSIONS["vllm"],
        "reviewed_scheduler_api_revision": "4ba787ec9959e566d3b587a796dff07673324da5",
        "runtime_source_manifest_sha256": hashlib.sha256(
            (_GLM_RUNTIME / "runtime-source-sha256.json").read_bytes()
        ).hexdigest(),
        "overlay_sha256": hashlib.sha256((_GLM_RUNTIME / "glm53flash_prefix_scheduler.py").read_bytes()).hexdigest(),
        "context_policy_version": 1,
        "warmup_repeats": 5,
        "measurement_repeats": 10,
    }


def _validate_glm_provenance(cell: FPMCell, payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Require the evidence that the GLM-5.3-Flash vLLM producer writes beside its graph-timed measurements.

    The producer builds every prefix and decode state from seeded tokens through vLLM's own scheduler, repeats each
    point and keeps the evidence of the repetitions in a sidecar file next to the artifact. Its ``producer`` record
    must equal the record of the supported producer (``_glm_producer_record``). A refusal names the first item that is
    absent or malformed. The caller has already checked that the rows are mappings with mapping points.
    """

    def lacks(field: str) -> ValueError:
        return ValueError(f"native GLM-5.3-Flash result lacks valid {field}: {path}")

    if payload.get("execution_identity") != dict(zip(EXECUTION_COLUMNS, cell.execution_identity, strict=True)):
        raise lacks("execution_identity")
    evidence = payload.get("input_provenance")
    if not isinstance(evidence, dict):
        raise lacks("input_provenance")
    if not _same(evidence.get("source"), "seeded_tokens"):
        raise lacks("input_provenance.source")
    for key in ("generator", "sampling"):
        if not isinstance(evidence.get(key), str) or not evidence[key]:
            raise lacks(f"input_provenance.{key}")
    # The generator version is an integer in the producer's record, like the seed and the token count.
    for key in ("generator_version", "seed", "count"):
        if type(evidence.get(key)) is not int:
            raise lacks(f"input_provenance.{key}")
    # The producer draws a nonempty token stream, so a count below 1 contradicts the input that it records.
    if evidence["count"] < 1:
        raise lacks("input_provenance.count")
    # The generator encodes the seed with int(seed).to_bytes(8, "big") (collector/glm53flash_attention_tokens.py:71), so
    # an unsigned 64-bit integer is the only seed that can have drawn the stream.
    if not 0 <= evidence["seed"] < 2**64:
        raise lacks("input_provenance.seed")
    # The generator draws ids from the half-open range [vocab_low, vocab_high_exclusive). It requires
    # 0 <= vocab_low < vocab_high_exclusive and at most 2**63 ids in the range
    # (collector/glm53flash_attention_tokens.py:67, 79), so the range holds vocab_high_exclusive - vocab_low ids. The
    # producer records both bounds in every artifact.
    low, high = evidence.get("vocab_low"), evidence.get("vocab_high_exclusive")
    if type(low) is not int or low < 0:
        raise lacks("input_provenance.vocab_low")
    if type(high) is not int or not 0 < high - low <= 2**63:
        raise lacks("input_provenance.vocab_high_exclusive")
    # A stream holds at most as many distinct tokens as tokens and as ids to draw from, and a nonempty one holds at
    # least one. The producer records this number in every artifact, so a record without it is refused like any other
    # absent item.
    unique = evidence.get("unique_token_count")
    if type(unique) is not int or not 1 <= unique <= min(evidence["count"], high - low):
        raise lacks("input_provenance.unique_token_count")
    for key in ("tokenizer_json_sha256", "token_ids_sha256"):
        if not _is_hex(evidence.get(key), 64):
            raise lacks(f"input_provenance.{key}")
    if not _is_hex(evidence.get("tokenizer_revision"), 40):
        raise lacks("input_provenance.tokenizer_revision")
    if not _same(evidence.get("chunk_start_alignment"), 4):
        raise lacks("input_provenance.chunk_start_alignment")

    # The sidecar is bound by its name, its digest and its record count, one record per measured point.
    sidecar = path.with_name(f"{path.stem}.repetitions.jsonl")
    manifest = evidence.get("repetition_evidence_manifest")
    raw = sidecar.read_bytes() if sidecar.is_file() else None
    if (
        not isinstance(manifest, dict)
        or raw is None
        or not _same(manifest.get("schema_version"), 1)
        or manifest.get("file") != sidecar.name
        or manifest.get("sha256") != hashlib.sha256(raw).hexdigest()
        or manifest.get("records") != len(raw.splitlines())
        or manifest.get("records") != len(payload["results"])
    ):
        raise lacks("repetition_evidence_manifest")

    kvwarm = payload.get("kvwarm")
    kvwarm = kvwarm if isinstance(kvwarm, dict) else {}
    expected_kvwarm = (
        ("enabled", True),
        ("warm_eligible", True),
        ("skip_reason", None),
        ("method", "native_prefix_cache_real_seed"),
        ("state_protocol", "glm53flash_prefix_cache_real_seed_v1"),
    )
    if any(not _same(kvwarm.get(key), value) for key, value in expected_kvwarm):
        raise lacks("kvwarm")
    if not _same(payload.get("timing_boundary"), "dynamo_vllm_instrumented_scheduler_fpm_wall_time"):
        raise lacks("timing_boundary")
    serving = payload.get("serving_config")
    serving = serving if isinstance(serving, dict) else {}
    for key, value in (("enable_prefix_caching", True), ("async_scheduling", False)):
        if not _same(serving.get(key), value):
            raise lacks(f"serving_config.{key}")
    # The producer record must be the one that the supported producer writes, field by field and with the same JSON
    # types. A field that the record lacks or adds is refused too.
    producer = payload.get("producer")
    if not isinstance(producer, dict):
        raise ValueError(
            f"native GLM-5.3-Flash result lacks valid producer: expected a mapping, found {producer!r}: {path}"
        )
    expected_producer = _glm_producer_record()
    for key, value in expected_producer.items():
        if not _same(producer.get(key), value):
            raise ValueError(
                f"native GLM-5.3-Flash result lacks valid producer.{key}: expected {value!r}, "
                f"found {producer.get(key)!r}: {path}"
            )
    added = sorted(set(producer) - set(expected_producer))
    if added:
        raise ValueError(
            f"native GLM-5.3-Flash result lacks valid producer.{added[0]}: expected no such field, "
            f"found {producer[added[0]]!r}: {path}"
        )

    for row in payload["results"]:
        point = row["point"]
        if not _is_hex(row.get("repetition_evidence_sha256"), 64):
            raise lacks("repetition_evidence_sha256")
        if point.get("point_type") == "decode":
            regime = "real_kv"
        elif _require_int(point, "total_kv_read_tokens") > 0:
            regime = "real_prefix"
        else:
            regime = "not_applicable"
        if row.get("kv_seed_regime") != regime:
            raise lacks("kv_seed_regime")
    return evidence


def _validate_token_streams(payload: dict[str, Any], path: Path) -> None:
    """Verify archived real request histories and completed-forward witnesses."""
    manifest = payload["input_provenance"].get("token_stream_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"V4.1 native result lacks token-stream manifest: {path}")
    name = manifest.get("file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".token-streams.jsonl"):
        raise ValueError(f"V4.1 token-stream path must be an adjacent JSONL file: {path}")
    raw = path.with_name(name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
        raise ValueError(f"V4.1 token-stream manifest SHA mismatch: {path}")
    lines = raw.splitlines()
    schema = manifest.get("schema_version", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError(f"V4.1 token-stream schema is unsupported: {path}")
    warmups = payload.get("warmup_results", [])
    if not isinstance(warmups, list) or (schema == 1 and warmups):
        raise ValueError(f"V4.1 warmup histories require token-stream schema 2: {path}")
    measured = payload["results"]
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("point"), dict)
        or type(row["point"].get("benchmark_id")) is not int
        or row["point"].get("point_type") not in ("prefill", "decode")
        or not isinstance(row["point"].get("sample_reasons", []), list)
        for row in measured + warmups
    ):
        raise ValueError(f"V4.1 native result has malformed token-stream points: {path}")
    warmup_ids = set()
    for row in warmups:
        point = row.get("point", {})
        benchmark_id = point.get("benchmark_id")
        if (
            type(benchmark_id) is not int
            or benchmark_id <= len(measured)
            or benchmark_id in warmup_ids
            or "eager_warmup" not in point.get("sample_reasons", [])
        ):
            raise ValueError(f"V4.1 warmup point is not a distinct native eager replica: {path}")
        warmup_ids.add(benchmark_id)
    if schema == 2 and manifest.get("warmup_benchmark_ids") != sorted(warmup_ids):
        raise ValueError(f"V4.1 native eager warmup coverage mismatch: {path}")
    if any("eager_warmup" in row["point"].get("sample_reasons", []) for row in measured):
        raise ValueError(f"V4.1 warmup timing cannot enter measured results: {path}")
    all_results = measured + warmups
    if manifest.get("records") != len(lines) or len(lines) != len(all_results):
        raise ValueError(f"V4.1 token-stream coverage mismatch: {path}")
    streams = {}
    for line in lines:
        stream = json.loads(line)
        if not isinstance(stream, dict):
            raise ValueError(f"V4.1 token-stream record must be an object: {path}")
        benchmark_id = stream.get("benchmark_id")
        if type(benchmark_id) is not int or benchmark_id in streams:
            raise ValueError(f"V4.1 token-stream benchmark ID is invalid or duplicated: {path}")
        expected_role = "warmup" if benchmark_id in warmup_ids else "measurement"
        if schema == 2 and stream.get("sampling_role") != expected_role:
            raise ValueError(f"V4.1 token-stream sampling role mismatch: {path}")
        streams[benchmark_id] = (stream, hashlib.sha256(line).hexdigest())
    if set(streams) != {row["point"]["benchmark_id"] for row in all_results}:
        raise ValueError(f"V4.1 token-stream point identity coverage mismatch: {path}")
    for row in all_results:
        point = row["point"]
        witness = row.get("real_kv_witness")
        stream, digest = streams.get(point["benchmark_id"], ({}, None))
        batch = _require_int(point, "batch_size")
        decode = point["point_type"] == "decode"
        expected_seed = _require_int(point, "total_kv_read_tokens") - (batch if decode else 0)
        if (
            not isinstance(witness, dict)
            or witness.get("same_request") is not True
            or witness.get("allocated_fake_tokens") != 0
            or witness.get("completed_seed_tokens") != expected_seed
            or witness.get("token_stream_sha256") != digest
        ):
            raise ValueError(f"V4.1 native result has invalid completed real-KV witness: {path}")
        requests = stream.get("requests")
        if not isinstance(requests, list) or len(requests) != batch:
            raise ValueError(f"V4.1 token-stream request count mismatch: {path}")
        prompt_total = 0
        for index, request in enumerate(requests):
            if not isinstance(request, dict) or request.get("request_index") != index:
                raise ValueError(f"V4.1 token-stream request order mismatch: {path}")
            for field in ("prompt_token_ids", "output_token_ids"):
                ids = request.get(field)
                if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
                    raise ValueError(f"V4.1 token-stream has invalid real token IDs: {path}")
            prompt_length = len(request["prompt_token_ids"])
            prompt_total += prompt_length
            if request.get("computed_tokens") != prompt_length + (2 if decode else 0):
                raise ValueError(f"V4.1 token-stream computed-token witness mismatch: {path}")
        expected_prompt = expected_seed + (0 if decode else _require_int(point, "total_prefill_tokens"))
        if prompt_total != expected_prompt:
            raise ValueError(f"V4.1 token-stream prompt history differs from the measured point: {path}")


def _without_sidecar_names(evidence: dict[str, Any]) -> dict[str, Any]:
    """Input evidence without the file names of its sidecar manifests, which differ per rank.

    The rest of each manifest, such as its digest and its record count, still has to agree across ranks.
    """
    sidecar_manifests = ("token_stream_manifest", "repetition_evidence_manifest")
    return {
        key: {name: item for name, item in value.items() if name != "file"} if key in sidecar_manifests else value
        for key, value in evidence.items()
    }


def _validate_collector_provenance(
    cell: FPMCell,
    raw_root: Path,
    rank_payloads: list[tuple[Path, dict[str, Any]]],
    *,
    expected_plan_sha256: str | None,
    expected_attempt_id: str | None,
) -> tuple[str, str]:
    pod_names = set()
    for path, _payload in rank_payloads:
        relative = path.relative_to(raw_root)
        if len(relative.parts) < 2:
            raise ValueError(f"native rank artifact is not scoped to a collected pod: {path}")
        pod_names.add(relative.parts[0])

    canonical: dict[str, Any] | None = None
    for pod_name in sorted(pod_names):
        path = raw_root / pod_name / COLLECTOR_PROVENANCE_FILENAME
        if not path.is_file():
            raise ValueError(f"native result is missing Collector provenance: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"Collector provenance must be a mapping: {path}")
        if payload.get("schema_name") != "aic_fpm_collector_provenance" or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported Collector provenance schema: {path}")
        if payload.get("cell_id") != cell.cell_id:
            raise ValueError(
                f"Collector provenance cell mismatch: actual={payload.get('cell_id')!r} expected={cell.cell_id!r}"
            )
        plan_sha256 = payload.get("plan_sha256")
        attempt_id = payload.get("attempt_id")
        if not isinstance(plan_sha256, str) or not plan_sha256:
            raise ValueError(f"Collector provenance has no plan identity: {path}")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError(f"Collector provenance has no attempt identity: {path}")
        if expected_plan_sha256 is not None and plan_sha256 != expected_plan_sha256:
            raise ValueError(
                f"Collector provenance plan mismatch: actual={plan_sha256!r} expected={expected_plan_sha256!r}"
            )
        if expected_attempt_id is not None and attempt_id != expected_attempt_id:
            raise ValueError(
                f"Collector provenance attempt mismatch: actual={attempt_id!r} expected={expected_attempt_id!r}"
            )
        runtime = payload.get("runtime")
        if (
            not isinstance(runtime, dict)
            or runtime.get("backend") != "vllm"
            or not isinstance(runtime.get("backend_version"), str)
            or not runtime["backend_version"]
        ):
            raise ValueError(f"Collector provenance has invalid runtime identity: {path}")
        # Dynamo package/release strings are provenance only. Backend identity
        # and every attempt binding still have to agree across all ranks.
        comparable = {
            **{key: value for key, value in payload.items() if key != "dynamo_version"},
            "runtime": {key: value for key, value in runtime.items() if key != "dynamo_version"},
        }
        if canonical is None:
            canonical = comparable
        elif comparable != canonical:
            raise ValueError(f"Collector provenance differs across pods: {path}")

    assert canonical is not None
    return str(canonical["runtime"]["backend_version"]), str(canonical["attempt_id"])


def _require_int(point: dict[str, Any], key: str) -> int:
    """Native grid coordinates are integers by contract (PR11509 declares int
    fields); a float or bool here means a noncompliant engine build or a
    tampered artifact, and silent truncation would publish a wrong-coordinate
    database row."""

    value = point.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        # ValueError, matching the module's other point-contract violations.
        raise ValueError(f"native point field {key!r} must be an integer, got {value!r}")
    return value


def _expected_scheduled(point: dict[str, Any]) -> dict[str, int]:
    phase = str(point["point_type"])
    batch = _require_int(point, "batch_size")
    total_prefill = _require_int(point, "total_prefill_tokens")
    total_kv = _require_int(point, "total_kv_read_tokens")
    if batch < 1 or min(total_prefill, total_kv) < 0:
        raise ValueError(f"native point has invalid totals: {point}")
    if phase == "prefill":
        # Every scheduled prefill request contributes at least one token, so a
        # total below the request count is a physically impossible coordinate.
        if total_prefill < batch:
            raise ValueError(f"native prefill point schedules fewer tokens than requests: {point}")
        return {
            "num_prefill_requests": batch,
            "sum_prefill_tokens": total_prefill,
            "sum_prefill_kv_tokens": total_kv,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        }
    if phase == "decode":
        if total_prefill != 0:
            raise ValueError(f"native decode point carries prefill tokens: {point}")
        # Every decode request reads at least one KV token of context.
        if total_kv < batch:
            raise ValueError(f"native decode point reads fewer KV tokens than requests: {point}")
        return {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": batch,
            "sum_decode_kv_tokens": total_kv,
        }
    raise ValueError(f"unknown native point type: {phase!r}")


def _validate_fpm(point: dict[str, Any], fpm: object, *, rank: int) -> float:
    if not isinstance(fpm, dict):
        raise TypeError("native FPM sample must be a mapping")
    benchmark_id = _require_int(point, "benchmark_id")
    if int(fpm.get("counter_id", -1)) != benchmark_id:
        raise ValueError(
            f"native FPM counter mismatch: counter_id={fpm.get('counter_id')!r}, benchmark_id={benchmark_id}"
        )
    if int(fpm.get("dp_rank", -1)) != rank:
        raise ValueError(f"native FPM rank mismatch: fpm={fpm.get('dp_rank')!r}, artifact={rank}")
    wall_time = float(fpm.get("wall_time", 0.0))
    if not math.isfinite(wall_time) or wall_time <= 0:
        raise ValueError(f"native FPM has invalid wall_time={wall_time}")
    scheduled = fpm.get("scheduled_requests")
    if not isinstance(scheduled, dict):
        raise TypeError("native FPM scheduled_requests must be a mapping")
    expected = _expected_scheduled(point)
    mismatches = {key: (scheduled.get(key), value) for key, value in expected.items() if scheduled.get(key) != value}
    if mismatches:
        raise ValueError(f"native FPM workload mismatch: {mismatches}")
    return wall_time


def _rank_artifacts(raw_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    artifacts = []
    for path in sorted(raw_root.glob(f"**/{FPM_BENCHMARK_RESULT_GLOB}")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"native benchmark artifact is not a mapping: {path}")
        if path.stem.endswith("_merged_worker_probe"):
            # Dynamo retains post-run RPC evidence separately from timing rows.
            if payload.get("schema") != "dynamo.fpm.benchmark_worker_probe" or payload.get("schema_version") != 1:
                raise ValueError(f"unsupported native worker probe sidecar: {path}")
            continue
        if payload.get("artifact_type") == "merged" or path.stem.endswith("_merged"):
            continue
        artifacts.append((path, payload))
    return artifacts


def validate_native_collection(
    cell: FPMCell,
    raw_root: Path,
    *,
    expected_plan_sha256: str | None = None,
    expected_attempt_id: str | None = None,
) -> NativeCollection:
    """Validate a complete native rank set and return synchronized measurements."""

    rank_payloads = _rank_artifacts(raw_root)
    if not rank_payloads:
        raise ValueError(f"integrated runtime emitted no benchmark results for {cell.cell_id}")
    backend_version, collector_attempt_id = _validate_collector_provenance(
        cell,
        raw_root,
        rank_payloads,
        expected_plan_sha256=expected_plan_sha256,
        expected_attempt_id=expected_attempt_id,
    )
    expected_ranks = list(range(cell.topology.dp))
    seen_ranks: set[int] = set()
    canonical_points: list[dict[str, Any]] | None = None
    canonical_groups: list[dict[str, Any]] | None = None
    run_identity: tuple[str, str] | None = None
    kvwarm_meta: dict[str, Any] | None = None
    kvwarm_seen: object = _KVWARM_UNSEEN
    checkpoint_evidence: list[object] = []
    input_provenance: dict[str, Any] | None = None
    execution_mode: str | None = None
    local_fpms: dict[tuple[int, int], dict[str, Any]] = {}
    seed_regimes: dict[int, str | None] = {}
    rank_timings: list[tuple[int, float, float]] = []

    for path, payload in rank_payloads:
        if (
            payload.get("schema_version") != FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION
            or payload.get("artifact_type") != "rank"
        ):
            raise ValueError(f"result is not a PR11509 native rank artifact: {path}")
        if (
            payload.get("status") != "complete"
            or payload.get("valid") is not True
            or payload.get("usable") is not True
            or payload.get("timing_valid") is not True
            or payload.get("stop_reason") is not None
            or payload.get("error") is not None
        ):
            raise ValueError(f"invalid native terminal result envelope: {path}")
        if payload.get("skipped_points") != [] or payload.get("missing_phases") != []:
            raise ValueError(f"native result skipped work: {path}")

        config = payload.get("config")
        if not isinstance(config, dict) or config.get("mode") != cell.workload_kind:
            raise ValueError(f"native benchmark mode mismatch: {path}")
        coverage = payload.get("coverage")
        rows = payload.get("results")
        groups = payload.get("iteration_groups")
        if not isinstance(coverage, dict) or not isinstance(rows, list) or not isinstance(groups, list):
            raise ValueError(f"native result is missing coverage/results/iteration_groups: {path}")
        if any(not isinstance(row, dict) or not isinstance(row.get("point"), dict) for row in rows):
            raise ValueError(f"native result has malformed result points: {path}")
        evidence = _validate_execution_provenance(cell, payload, path)
        # The validation above has proven that the artifact's mode is the mode of the cell's contract.
        execution_mode = payload.get("execution_mode") if evidence is not None else None
        if execution_mode == "eager":
            _validate_token_streams(payload, path)
        elif execution_mode == "native_graph_policy" and payload["producer"]["vllm_package_version"] != backend_version:
            # The producer reports the vLLM that it ran on. The Collector provenance binds the launch to its runtime.
            raise ValueError(
                "native GLM-5.3-Flash producer.vllm_package_version differs from the Collector provenance runtime: "
                f"expected {backend_version!r}, found {payload['producer']['vllm_package_version']!r}: {path}"
            )
        if input_provenance is None:
            input_provenance = evidence
        elif _without_sidecar_names(evidence) != _without_sidecar_names(input_provenance):
            raise ValueError(f"native DP ranks disagree on input provenance: {path}")
        # One launch shares one scheduler configuration, so the ranks must agree on the checkpoint evidence of it.
        serving = payload.get("serving_config")
        checkpoint = (
            serving.get("mamba_prefill_checkpoint", _NO_CHECKPOINT_EVIDENCE)
            if isinstance(serving, dict)
            else _NO_CHECKPOINT_EVIDENCE
        )
        checkpoint_evidence.append(checkpoint)
        if checkpoint != checkpoint_evidence[0]:
            raise ValueError(f"native ranks disagree on serving_config.mamba_prefill_checkpoint: {path}")
        expected = coverage.get("expected_points")
        if (
            not isinstance(expected, int)
            or expected < 1
            or coverage.get("completed_points") != expected
            or coverage.get("skipped_points") != 0
            or len(rows) != expected
            or len(groups) != expected
        ):
            raise ValueError(f"native benchmark coverage is incomplete: {path}")

        dp = payload.get("dp")
        if not isinstance(dp, dict) or dp.get("size") != cell.topology.dp or not isinstance(dp.get("rank"), int):
            raise ValueError(f"native DP metadata mismatch: {path}")
        rank = int(dp["rank"])
        if rank in seen_ranks:
            raise ValueError(f"duplicate native dp_rank={rank}: {path}")
        seen_ranks.add(rank)

        identity = (payload.get("run_id"), payload.get("grid_digest"))
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError(f"native run identity is missing: {path}")
        typed_identity = (str(identity[0]), str(identity[1]))
        if run_identity is None:
            run_identity = typed_identity
        elif typed_identity != run_identity:
            raise ValueError(f"native DP ranks have different run identities: {path}")

        # Engine-reported KV warm-up envelope. Absent on artifacts predating
        # the kvwarm-enabled runtime (legacy); when present, the regime facts
        # (warm_eligible/skip_reason) must agree across DP ranks -- they
        # describe one shared measurement protocol, not per-rank state.
        kvwarm = _validate_kvwarm_contract(cell, payload.get("kvwarm"), path)
        kvwarm_regime = (
            None if kvwarm is None else (kvwarm.get("enabled"), kvwarm.get("warm_eligible"), kvwarm.get("skip_reason"))
        )
        if kvwarm_seen is _KVWARM_UNSEEN:
            kvwarm_seen = kvwarm_regime
            kvwarm_meta = kvwarm
        elif kvwarm_regime != kvwarm_seen:
            raise ValueError(f"native DP ranks disagree on the KV warm-up regime (warm_eligible/skip_reason): {path}")

        timing = payload.get("timing")
        if not isinstance(timing, dict):
            raise TypeError(f"native result has no timing object: {path}")
        elapsed = float(timing.get("benchmark_elapsed_seconds", -1))
        measured = float(timing.get("measured_iteration_seconds", -1))
        if (
            not math.isfinite(elapsed)
            or not math.isfinite(measured)
            or min(elapsed, measured) < 0
            or measured > elapsed + 1e-12
        ):
            raise ValueError(f"native result has invalid timing: {path}")
        rank_timings.append((rank, elapsed, measured))

        points = []
        seen_ids = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("point"), dict):
                raise TypeError(f"native result entry is malformed: {path}")
            point = row["point"]
            required = {
                "point_type",
                "benchmark_id",
                "total_prefill_tokens",
                "total_kv_read_tokens",
                "batch_size",
            }
            if not required.issubset(point) or point["point_type"] != cell.workload_kind:
                raise ValueError(f"native point contract mismatch: {point}")
            benchmark_id = int(point["benchmark_id"])
            if benchmark_id in seen_ids:
                raise ValueError(f"native benchmark IDs are duplicated: {path}")
            seen_ids.add(benchmark_id)
            _expected_scheduled(point)
            fpms = row.get("fpms")
            if not isinstance(fpms, list) or len(fpms) != 1:
                raise ValueError(f"native point must contain exactly one local FPM: {path}")
            _validate_fpm(point, fpms[0], rank=rank)
            seed_regime = _validate_seed_regime(row, path)
            if (
                seed_regime == "real_kv"
                and cell.workload_kind == "decode"
                and kvwarm is not None
                and (not kvwarm["enabled"] or not kvwarm["warm_eligible"])
            ):
                raise ValueError(f"native real KV seed regime disagrees with warm-up eligibility: {path}")
            if benchmark_id in seed_regimes and seed_regimes[benchmark_id] != seed_regime:
                raise ValueError(f"native DP ranks disagree on per-point KV seed regime: {path}")
            seed_regimes[benchmark_id] = seed_regime
            local_fpms[(benchmark_id, rank)] = fpms[0]
            points.append(point)
        # The engine writes results in execution order, and KV warm-up's
        # point reordering (batch/kv-depth descending for warm-chain reuse,
        # fake points last) deliberately decouples that order from
        # benchmark_id -- so validate the ID set, not file positions, then
        # canonicalize by ID.
        if seen_ids != set(range(1, expected + 1)):
            raise ValueError(f"native benchmark IDs are not contiguous: {path}")
        points.sort(key=lambda item: int(item["benchmark_id"]))
        if canonical_points is None:
            canonical_points = points
        elif points != canonical_points:
            raise ValueError(f"native DP ranks generated different grids: {path}")
        if canonical_groups is None:
            canonical_groups = groups
        elif groups != canonical_groups:
            raise ValueError(f"native DP ranks have different synchronized iteration groups: {path}")

    if seen_ranks != set(expected_ranks):
        raise ValueError(f"native DP rank set mismatch: actual={sorted(seen_ranks)} expected={expected_ranks}")
    assert canonical_points is not None and canonical_groups is not None and run_identity is not None

    measurements = []
    measured_iteration_seconds = 0.0
    # iteration_groups are written in execution order too; canonicalize by
    # benchmark_id so the positional pairing with the ID-sorted points holds.
    for group in canonical_groups:
        if not isinstance(group, dict):
            raise TypeError("native iteration group is not a mapping")
    try:
        canonical_groups = sorted(canonical_groups, key=lambda group: int(group["benchmark_id"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"native iteration group benchmark_id is invalid: {error}") from error
    for benchmark_id, (point, group) in enumerate(zip(canonical_points, canonical_groups, strict=True), start=1):
        if (
            group.get("benchmark_id") != benchmark_id
            or group.get("point") != point
            or group.get("expected_dp_ranks") != expected_ranks
            or group.get("complete") is not True
        ):
            raise ValueError(f"native iteration group contract mismatch for benchmark_id={benchmark_id}")
        rank_results = group.get("rank_results")
        if (
            not isinstance(rank_results, list)
            or not all(isinstance(item, dict) for item in rank_results)
            or [item.get("dp_rank") for item in rank_results] != expected_ranks
        ):
            raise ValueError(f"native iteration group rank mismatch for benchmark_id={benchmark_id}")
        wall_times = []
        for rank_result in rank_results:
            rank = int(rank_result["dp_rank"])
            fpms = rank_result.get("fpms")
            if not isinstance(fpms, list) or len(fpms) != 1:
                raise ValueError(f"native iteration group has invalid FPM count for rank={rank}")
            wall_time = _validate_fpm(point, fpms[0], rank=rank)
            if fpms[0] != local_fpms[(benchmark_id, rank)]:
                raise ValueError(f"native local result differs from synchronized group for rank={rank}")
            wall_times.append((rank, wall_time))
        group_wall_time = float(group.get("wall_time", -1))
        expected_wall_time = max(value for _, value in wall_times)
        if not math.isfinite(group_wall_time) or not math.isclose(
            group_wall_time, expected_wall_time, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"native iteration wall_time mismatch for benchmark_id={benchmark_id}")
        measured_iteration_seconds += group_wall_time
        measurements.append(
            NativePointMeasurement(
                point=dict(point), rank_wall_times=tuple(wall_times), kv_seed_regime=seed_regimes[benchmark_id]
            )
        )

    for rank, _elapsed, measured in rank_timings:
        if not math.isclose(measured, measured_iteration_seconds, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"native measured timing mismatch for rank={rank}")
    return NativeCollection(
        points=tuple(measurements),
        rank_timings=tuple(sorted(rank_timings)),
        backend_version=backend_version,
        collector_attempt_id=collector_attempt_id,
        runtime_run_id=run_identity[0],
        runtime_grid_digest=run_identity[1],
        kvwarm_meta=kvwarm_meta,
        input_provenance=input_provenance,
        execution_mode=execution_mode,
    )
