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


def _entry(**overrides):
    entry = {
        "model_path": "deepseek-ai/DeepSeek-V4-Flash",
        "model_profile": "deepseek_v4_flash_fp4",
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
        "latency_ms": 16.7,
        "validation": {"stable": True, "correctness": None, "evidence": "stable-split"},
        "source": {
            "commit": "e507eacf858d2046bdc2cca02ed86c0e58bd6c60",
            "source_tree_sha256": "fce9cfe9888adc31974e31434f16d3f1b87585a692a9adbbfe3fb481c4a564da",
            "result": "raw/split/point.json",
        },
    }
    entry.update(overrides)
    return entry


def _write(tmp_path, entries):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": FASTAFD_PROFILE_SCHEMA, "lookup_policy": "exact-only", "entries": entries}))
    return path


def _key(**overrides):
    fields = {
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
    }
    fields.update(overrides)
    return FastAFDMoEStageKey(**fields)


def test_exact_lookup_preserves_provenance(tmp_path):
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, [_entry()]))

    measurement = profile.require(_key())

    assert measurement.latency_ms == pytest.approx(16.7)
    assert len(profile.profile_sha256) == 64
    assert measurement.provenance()["source_commit"] == "e507eacf858d2046bdc2cca02ed86c0e58bd6c60"
    assert profile.find(_key(topology="2A6F")) is None


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"stage": "afd", "topology": "ep8"}, "invalid afd topology"),
        ({"validation": {"stable": False, "correctness": True, "evidence": "x"}}, "stable=true"),
        ({"validation": {"stable": True, "correctness": False, "evidence": "x"}}, "failed correctness"),
        ({"source": {"commit": "bad", "source_tree_sha256": "0" * 64, "result": "x"}}, "SHA-1"),
        ({"latency_ms": 0}, "finite positive"),
    ],
)
def test_rejects_unqualified_measurements(tmp_path, change, match):
    with pytest.raises((TypeError, ValueError), match=match):
        FastAFDMoEStageProfile.load(_write(tmp_path, [_entry(**change)]))


def test_rejects_duplicate_exact_keys(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        FastAFDMoEStageProfile.load(_write(tmp_path, [_entry(), _entry()]))


def test_rejects_duplicate_json_object_keys(tmp_path):
    path = _write(tmp_path, [_entry()])
    path.write_text(path.read_text().replace('"stable": true', '"stable": true, "stable": false'))

    with pytest.raises(ValueError, match="duplicate JSON object key: 'stable'"):
        FastAFDMoEStageProfile.load(path)
