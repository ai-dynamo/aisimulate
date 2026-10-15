# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aisimulate_core.sdk.fastafd_profile import (
    FASTAFD_PROFILE_SCHEMA,
    FastAFDMoEStageKey,
    FastAFDMoEStageProfile,
)

pytestmark = pytest.mark.unit

_REPOSITORY = "https://github.com/example/FastAFD"
_COMMIT = "0" * 40


def _entry():
    return {
        "key": {
            "model_path": "deepseek-ai/DeepSeek-V4-Flash",
            "system": "gb200",
            "stage": "afd",
            "topology": "16A8F",
            "logical_batch_per_source_rank": 48,
            "mtp_nextn": 0,
            "microbatches": 2,
            "moe_layers": 43,
            "routed_topk": 6,
            "moe_precision": "w4a8_mxfp4_mxfp8",
            "moe_backend": "megamoe",
        },
        "model_profile": "deepseek_v4_flash_fp4",
        "latency_ms": 16.7,
        "measurement": {
            "scope": "complete_moe_stage",
            "method": "cuda_event",
            "method_version": "torch 2.10.0+cu130; cuda 13.0",
            "statistic": "p50",
            "sample_count": 40,
            "cv_percent": 0.4,
            "routing": "balanced",
            "source_tree_sha256": "c" * 64,
            "raw_artifact": "deepseek_v4_flash_fp4_mega_16a8f_s48_n0_mb2_balanced.json",
            "raw_sha256": "b" * 64,
        },
        "validation": {"stable": True, "correctness": None},
    }


def _payload(entries=None):
    return {
        "schema": FASTAFD_PROFILE_SCHEMA,
        "lookup_policy": "exact-only",
        "source": {
            "repository": _REPOSITORY,
            "commit": _COMMIT,
            "benchmark": "fastafd.moe-m2n-model-benchmark.v2",
            "boundary": "split AFD MoE stage",
        },
        "entries": [_entry()] if entries is None else entries,
    }


def _write(tmp_path, payload):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_lookup_preserves_source_and_method(tmp_path):
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    key = FastAFDMoEStageKey(**_entry()["key"])
    measurement = profile.require(key)

    assert measurement.latency_ms == pytest.approx(16.7)
    assert profile.find(FastAFDMoEStageKey(**{**_entry()["key"], "topology": "32A8F"})) is None
    assert len(profile.profile_sha256) == 64
    provenance = measurement.provenance()
    assert provenance["repository"] == _REPOSITORY
    assert provenance["source_commit"] == _COMMIT
    assert provenance["benchmark"] == "fastafd.moe-m2n-model-benchmark.v2"
    assert provenance["method"] == "cuda_event"
    assert provenance["source_tree_sha256"] == "c" * 64
    assert provenance["correctness"] is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("repository", "", "non-empty string"),
        ("commit", "main", "invalid digest"),
        ("boundary", "", "non-empty string"),
    ],
)
def test_rejects_incomplete_source(tmp_path, field, value, match):
    payload = _payload()
    payload["source"][field] = value
    with pytest.raises(ValueError, match=match):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scope", "per_rank_gpu_span", "complete_moe_stage"),
        ("method", "", "non-empty string"),
        ("statistic", "mean", "must be p50"),
        ("sample_count", 0, "integer >= 1"),
        ("cv_percent", -1.0, ">= 0.0"),
        ("source_tree_sha256", "bad", "invalid digest"),
        ("raw_sha256", "bad", "invalid digest"),
    ],
)
def test_rejects_unqualified_measurement(tmp_path, field, value, match):
    payload = _payload()
    payload["entries"][0]["measurement"][field] = value
    with pytest.raises(ValueError, match=match):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))


def test_rejects_failed_validation_and_duplicate_keys(tmp_path):
    payload = _payload()
    payload["entries"][0]["validation"]["stable"] = False
    with pytest.raises(ValueError, match="not a qualified measurement"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    payload["entries"][0]["validation"] = {"stable": True, "correctness": False}
    with pytest.raises(ValueError, match="not a qualified measurement"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    with pytest.raises(ValueError, match="duplicate FastAFD MoE stage key"):
        FastAFDMoEStageProfile.load(_write(tmp_path, _payload([_entry(), _entry()])))


def test_rejects_other_schemas_and_duplicate_json_fields(tmp_path):
    payload = _payload()
    payload["schema"] = "unsupported.schema"
    with pytest.raises(ValueError, match="unsupported FastAFD profile schema"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    path = _write(tmp_path, _payload())
    path.write_text(path.read_text().replace('"stable": true', '"stable": true, "stable": false'))
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        FastAFDMoEStageProfile.load(path)
