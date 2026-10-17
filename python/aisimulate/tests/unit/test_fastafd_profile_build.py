# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from aisimulate.fastafd_profile import FastAFDMoEStageKey, FastAFDMoEStageProfile, build_profile, main

pytestmark = pytest.mark.unit

_REPOSITORY = "https://github.com/example/FastAFD"
_MODEL = {
    "key": "qwen3_235b_fp4",
    "model_path": "Qwen/Qwen3-235B-A22B-FP8",
    "routed_top_k": 8,
    "moe_quant_contract": "w4a8_mxfp4_mxfp8",
}
_STACK = [{"torch": "2.10.0+cu130", "cuda_runtime": "13.0"}]


def _colocated(tokens, *, routing="balanced", stable=True, p50=12.0, correctness=None):
    return {
        "schema": "fastafd.megamoe-colocated-model-benchmark.v2",
        "system_label": "gb200",
        "measurement_boundary": "colocated MoE stage",
        "primary_backend": "mega",
        "model_profile": _MODEL,
        "topology": {"ep_size": 8},
        "workload": {"logical_tokens_per_rank": tokens, "mtp_nextn": 0, "layers": 94, "routing": routing},
        "stage_cuda": {"samples": 40, "p50_ms": p50, "cv_percent": 0.5},
        "stable": stable,
        "correctness_passed": correctness,
        "source": {"commit": "0" * 40, "source_tree_sha256": "c" * 64},
        "provenance_by_rank": _STACK,
    }


def _split(sequences):
    return {
        "schema": "fastafd.moe-m2n-model-benchmark.v2",
        "system_label": "gb200",
        "measurement_boundary": "split AFD MoE stage",
        "moe_backend": "megamoe",
        "model_profile": _MODEL,
        "topology": {"ag_size": 16, "eg_size": 8},
        "workload": {
            "sequences_per_ag_rank": sequences,
            "mtp_nextn": 1,
            "microbatches": 2,
            "layers": 94,
            "routing": "balanced",
        },
        "stage_cuda": {"samples": 40, "p50_ms": 11.4, "cv_percent": 0.6},
        "stable": True,
        "source": {"commit": "0" * 40, "source_tree_sha256": "d" * 64},
        "provenance_by_rank": _STACK,
    }


def _write(directory, name, payload):
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builds_colocated_profile_and_skips_unqualified_results(tmp_path):
    results = tmp_path / "colocated"
    raw = _write(results, "t48.json", _colocated(48))
    _write(results, "t96.json", _colocated(96, p50=14.0))
    _write(results, "t48_hotset.json", _colocated(48, routing="hotset"))
    _write(results, "t8.json", _colocated(8, stable=False))
    _write(results, "t16.json", _colocated(16, correctness=False))

    profile, summary = build_profile(results, repository=_REPOSITORY)

    assert summary == {"accepted": 2, "skipped_routing": 1, "skipped_unstable": 1, "skipped_correctness": 1}
    assert profile["source"] == {
        "repository": _REPOSITORY,
        "commit": "0" * 40,
        "benchmark": "fastafd.megamoe-colocated-model-benchmark.v2",
        "boundary": "colocated MoE stage",
    }
    entry = profile["entries"][0]
    assert entry["key"] == {
        "model_path": "Qwen/Qwen3-235B-A22B-FP8",
        "system": "gb200",
        "stage": "agg",
        "topology": "ep8",
        "logical_batch_per_source_rank": 48,
        "mtp_nextn": 0,
        "microbatches": 1,
        "moe_layers": 94,
        "routed_topk": 8,
        "moe_precision": "w4a8_mxfp4_mxfp8",
        "moe_backend": "megamoe",
    }
    assert entry["measurement"]["raw_artifact"] == "t48.json"
    assert entry["measurement"]["raw_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert entry["measurement"]["method_version"] == "torch 2.10.0+cu130; cuda 13.0"
    assert entry["validation"] == {"stable": True, "correctness": None}


def test_builds_split_profile_through_the_command_line(tmp_path):
    results = tmp_path / "split"
    _write(results, "s64.json", _split(64))
    output = tmp_path / "profile.json"

    assert main([str(results), "--repository", _REPOSITORY, "--output", str(output)]) == 0

    profile = FastAFDMoEStageProfile.load(output)
    key = FastAFDMoEStageKey(
        model_path="Qwen/Qwen3-235B-A22B-FP8",
        system="gb200",
        stage="afd",
        topology="16A8F",
        logical_batch_per_source_rank=64,
        mtp_nextn=1,
        microbatches=2,
        moe_layers=94,
        routed_topk=8,
        moe_precision="w4a8_mxfp4_mxfp8",
        moe_backend="megamoe",
    )
    assert profile.require(key).latency_ms == pytest.approx(11.4)
    assert profile.source.benchmark == "fastafd.moe-m2n-model-benchmark.v2"


def test_rejects_mixed_directories_and_repeated_measurements(tmp_path):
    mixed = tmp_path / "mixed"
    _write(mixed, "a.json", _colocated(48))
    _write(mixed, "b.json", _split(64))
    with pytest.raises(ValueError, match="mixed benchmark"):
        build_profile(mixed, repository=_REPOSITORY)

    repeated = tmp_path / "repeated"
    _write(repeated, "a.json", _colocated(48))
    _write(repeated, "b.json", _colocated(48, p50=12.5))
    with pytest.raises(ValueError, match="repeats the measurement"):
        build_profile(repeated, repository=_REPOSITORY)

    with pytest.raises(ValueError, match="no qualified hotset"):
        build_profile(repeated, repository=_REPOSITORY, routing="hotset")
