# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aisimulate_core.sdk.fastafd_profile import (
    FASTAFD_OFFICIAL_REPOSITORY,
    FASTAFD_PROFILE_SCHEMA,
    FastAFDMoEStageKey,
    FastAFDMoEStageProfile,
)

pytestmark = pytest.mark.unit

_OFFICIAL_COMMIT = "3c7161949310b6d59d6b4cf9bf997a4935c8113b"


def _entry():
    return {
        "key": {
            "model_path": "deepseek-ai/DeepSeek-V4-Flash",
            "system": "b200_sxm",
            "stage": "afd",
            "topology": "4A4F",
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
            "method": "nsight-systems-cupti",
            "method_version": "2025.5.2",
            "statistic": "p50",
            "sample_count": 10,
            "procedure": "docs/profile-procedure.md",
            "procedure_sha256": "a" * 64,
            "raw_artifact": "raw/trace.nsys-rep",
            "raw_sha256": "b" * 64,
        },
        "validation": {"stable": True, "correctness": True, "evidence": "raw/validation.json"},
    }


def _payload(entries=None):
    return {
        "schema": FASTAFD_PROFILE_SCHEMA,
        "lookup_policy": "exact-only",
        "source": {"repository": FASTAFD_OFFICIAL_REPOSITORY, "commit": _OFFICIAL_COMMIT},
        "entries": [_entry()] if entries is None else entries,
    }


def _write(tmp_path, payload):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_lookup_preserves_official_source_and_method(tmp_path):
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    key = FastAFDMoEStageKey(**_entry()["key"])
    measurement = profile.require(key)

    assert measurement.latency_ms == pytest.approx(16.7)
    assert profile.find(FastAFDMoEStageKey(**{**_entry()["key"], "topology": "2A6F"})) is None
    assert len(profile.profile_sha256) == 64
    assert measurement.provenance()["repository"] == FASTAFD_OFFICIAL_REPOSITORY
    assert measurement.provenance()["source_commit"] == _OFFICIAL_COMMIT
    assert measurement.provenance()["method"] == "nsight-systems-cupti"
    assert measurement.provenance()["raw_sha256"] == "b" * 64


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("repository", "https://github.com/example/FastAFD", "official repository"),
        ("commit", "main", "invalid digest"),
    ],
)
def test_rejects_nonofficial_or_mutable_source(tmp_path, field, value, match):
    payload = _payload()
    payload["source"][field] = value
    with pytest.raises(ValueError, match=match):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scope", "per_rank_gpu_span", "complete_moe_stage"),
        ("method", "", "non-empty string"),
        ("method_version", "", "non-empty string"),
        ("statistic", "mean", "must be p50"),
        ("sample_count", 0, "integer >= 1"),
        ("procedure_sha256", "bad", "invalid digest"),
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

    payload["entries"][0]["validation"]["stable"] = True
    payload["entries"][0]["validation"]["correctness"] = None
    with pytest.raises(ValueError, match="not a qualified measurement"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    with pytest.raises(ValueError, match="duplicate FastAFD MoE stage key"):
        FastAFDMoEStageProfile.load(_write(tmp_path, _payload([_entry(), _entry()])))


def test_rejects_old_format_and_duplicate_json_fields(tmp_path):
    payload = _payload()
    payload["schema"] = "unsupported.schema"
    with pytest.raises(ValueError, match="unsupported FastAFD profile schema"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    path = _write(tmp_path, _payload())
    path.write_text(path.read_text().replace('"stable": true', '"stable": true, "stable": false'))
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        FastAFDMoEStageProfile.load(path)
