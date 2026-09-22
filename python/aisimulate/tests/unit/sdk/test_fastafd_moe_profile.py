# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from aiconfigurator.sdk.fastafd_moe_profile import (
    PROFILE_SCHEMA,
    FastAFDMoEStageKey,
    FastAFDMoEStageProfile,
)


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
    path.write_text(json.dumps({"schema": PROFILE_SCHEMA, "lookup_policy": "exact-only", "entries": entries}))
    return path


def test_loads_exact_measured_stage_and_preserves_provenance(tmp_path):
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, [_entry()]))
    key = FastAFDMoEStageKey(
        model_path="deepseek-ai/DeepSeek-V4-Flash",
        system="b200_sxm",
        stage="afd",
        topology="4A4F",
        logical_batch_per_source_rank=48,
        mtp_nextn=0,
        microbatches=2,
        moe_layers=43,
        routed_topk=6,
        moe_precision="w4a8_mxfp4_mxfp8",
        moe_backend="megamoe",
    )

    measurement = profile.require(key)

    assert measurement.latency_ms == pytest.approx(16.7)
    assert measurement.source_commit == "e507eacf858d2046bdc2cca02ed86c0e58bd6c60"
    assert profile.find(FastAFDMoEStageKey(**{**key.__dict__, "topology": "2A6F"})) is None


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"stage": "afd", "topology": "ep8"}, "invalid afd topology"),
        ({"validation": {"stable": False, "correctness": True, "evidence": "x"}}, "stable=true"),
        ({"source": {"commit": "bad", "source_tree_sha256": "0" * 64, "result": "x"}}, "SHA-1"),
        ({"latency_ms": 0}, "finite positive"),
    ],
)
def test_rejects_unqualified_or_ambiguous_measurements(tmp_path, change, match):
    with pytest.raises((TypeError, ValueError), match=match):
        FastAFDMoEStageProfile.load(_write(tmp_path, [_entry(**change)]))


def test_rejects_duplicate_exact_keys(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        FastAFDMoEStageProfile.load(_write(tmp_path, [_entry(), _entry()]))
