# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded SD identity is exact, native-normalized, and cannot alias AR data."""

import copy
import json

import pytest
from pydantic import ValidationError

import aisimulate_core
from aisimulate_core.fpm_profile import FpmModelProfile, FpmSpeculativeExecutionProfile
from aisimulate_core.sdk import RustForwardPassPerfModel, fpm_profile
from aisimulate_core.sdk.fpm_identity import LEGACY_EXECUTION_IDENTITY, speculative_execution_identity

pytestmark = pytest.mark.unit


def execution():
    return {
        "kind": "speculative_round",
        "method": "eagle3",
        "num_draft_tokens": 3,
        "draft_model": "test/draft",
        "draft_revision": "2" * 40,
        "timing_scope": "model_step_cuda",
    }


def profile(*, speculative=True):
    deployment = {
        "system": "b200_sxm",
        "backend": "vllm",
        "backend_version": "0.28.0",
        "tp": 4,
        "dp": 1,
        "moe_tp": 4,
        "moe_ep": 1,
        "gemm_quant_mode": "nvfp4",
        "moe_quant_mode": "nvfp4",
        "fmha_quant_mode": "fp8",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "fp8",
        "resources": {
            "cache_layout": "linear",
            "kv_bytes_per_token": 128,
            "max_num_tokens": 16384,
            "max_batch_size": 1024,
            "provenance": "Synthetic pending-resource metadata, not silicon measurements.",
        },
    }
    if speculative:
        deployment["execution_profile"] = execution()
    return {
        "schema_version": 1,
        "model": "test/target",
        "model_revision": "1" * 40,
        "architecture": "TestDecoderForCausalLM",
        "context_length": 1048576,
        "num_experts": 128,
        "provenance": "Synthetic identity fixture.",
        "deployments": [deployment],
    }


def test_native_selector_is_self_describing_and_normalizes_input_key_order():
    deployment = FpmModelProfile.model_validate(profile()).deployments[0]
    suffix = deployment.execution_identity("1" * 40)
    assert suffix[0] == "" and suffix[2:] == ("none", "text")
    assert suffix[1].startswith("speculative_round_v1:")
    assert json.loads(suffix[1].split(":", 1)[1]) == {
        "model_revision": "1" * 40,
        "execution_profile": execution(),
    }
    reversed_keys = dict(reversed(list(execution().items())))
    assert speculative_execution_identity("1" * 40, reversed_keys) == suffix
    assert len(deployment.match_identity()) == 15


def test_legacy_profile_json_and_identity_do_not_gain_default_execution_fields():
    value = FpmModelProfile.model_validate(profile(speculative=False))
    before = value.model_dump_json()
    assert "execution_profile" not in before
    assert value.deployments[0].execution_identity("legacy-tag-remains-valid") == LEGACY_EXECUTION_IDENTITY
    explicit_none = copy.deepcopy(profile(speculative=False))
    explicit_none["deployments"][0]["execution_profile"] = None
    assert FpmModelProfile.model_validate(explicit_none).model_dump_json() == before
    assert len(value.deployments[0].match_identity()) == 15


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_draft_tokens", 0),
        ("num_draft_tokens", 6),
        ("num_draft_tokens", True),
        ("num_draft_tokens", 3.0),
        ("method", "ngram"),
        ("kind", "target_verify"),
        ("timing_scope", "scheduler_wall_time"),
        ("draft_revision", "main"),
        ("extra_field", "ignored"),
    ],
)
def test_python_and_rust_reject_invalid_descriptor(field, value):
    data = execution() | {field: value}
    with pytest.raises((ValueError, ValidationError)):
        FpmSpeculativeExecutionProfile.model_validate(data)
    with pytest.raises(ValueError):
        speculative_execution_identity("1" * 40, data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_draft_tokens", 4),
        ("draft_model", "test/other"),
        ("draft_revision", "3" * 40),
    ],
)
def test_draft_identity_changes_never_alias(field, value):
    original = speculative_execution_identity("1" * 40, execution())
    assert speculative_execution_identity("1" * 40, execution() | {field: value}) != original
    assert speculative_execution_identity("4" * 40, execution()) != original


@pytest.mark.parametrize("model_revision", ["main", "fixture-v1", "F" * 40, "1" * 39])
def test_recorded_target_revision_is_immutable(model_revision):
    with pytest.raises(ValueError, match="model_revision"):
        speculative_execution_identity(model_revision, execution())


def test_profile_rejects_attention_dp_or_expert_parallel_for_initial_contract():
    data = profile()
    data["deployments"][0].update(tp=1, dp=4, moe_tp=1, moe_ep=4)
    with pytest.raises(ValidationError, match="pure TP"):
        FpmModelProfile.model_validate(data)


def test_dense_tp_profile_accepts_unused_moe_tp_one():
    data = profile()
    data["num_experts"] = 0
    data["deployments"][0]["moe_tp"] = 1
    deployment = FpmModelProfile.model_validate(data).deployments[0]
    assert deployment.match_identity()[5:11] == ["4", "1", "1", "1", "1", "1"]
    assert deployment.execution_identity(data["model_revision"])[1].startswith("speculative_round_v1:")


def canonical_request(data, *, nextn=0):
    selected = data["deployments"][0]
    return {
        "model": data["model"],
        "system": selected["system"],
        "backend": selected["backend"],
        "backend_version": selected["backend_version"],
        "worker_type": "aggregated",
        "tp": selected["tp"],
        "moe_tp_size": selected["moe_tp"],
        "moe_ep_size": selected["moe_ep"],
        "nextn": nextn,
        "fpm_profile": data,
        "estimation_mode": "fpm_interpolation",
        "fallback_policy": "deny",
        "estimator_config": {"fpm_interpolation": {"method": "direct"}},
    }


def test_unrelated_speculative_deployment_does_not_block_selected_ar(monkeypatch):
    monkeypatch.setattr(
        fpm_profile, "_load_pinned_checkpoint_config", lambda *_: {"architectures": ["TestDecoderForCausalLM"]}
    )
    ar = profile(speculative=False)
    before = RustForwardPassPerfModel.normalize_config(canonical_request(ar))
    both = copy.deepcopy(ar)
    other = profile()["deployments"][0]
    other.update(tp=2, moe_tp=2)
    both["deployments"].append(other)
    after = RustForwardPassPerfModel.normalize_config(canonical_request(both))
    assert after["nextn"] == before["nextn"] == 0
    assert after["fpm_profile"]["deployments"][0] == before["fpm_profile"]["deployments"][0]
    assert after["fpm_profile"]["deployments"][1]["execution_profile"] == execution()


@pytest.mark.parametrize(
    "nextn,error",
    [(0, "nextn must match"), (2, "nextn must match"), (3, "observed runtime memory")],
)
@pytest.mark.parametrize("entry", ["normalize_config", "best_available"])
def test_speculative_execution_rejects_depth_mismatch_and_pending_memory(nextn, error, entry, monkeypatch):
    monkeypatch.setattr(
        fpm_profile, "_load_pinned_checkpoint_config", lambda *_: {"architectures": ["TestDecoderForCausalLM"]}
    )
    config = canonical_request(profile(), nextn=nextn)
    with pytest.raises(ValueError, match=error):
        getattr(RustForwardPassPerfModel, entry)(config)


def test_identity_helper_delegates_without_python_defaults(monkeypatch):
    calls = []

    class Native:
        @staticmethod
        def _normalize_fpm_execution_profile(revision, payload):
            calls.append((revision, json.loads(payload)))
            return "native-selector"

    monkeypatch.setattr(aisimulate_core, "RustForwardPassPerfModel", Native)
    assert speculative_execution_identity("pinned", {"supplied": True}) == ("", "native-selector", "none", "text")
    assert calls == [("pinned", {"supplied": True})]
