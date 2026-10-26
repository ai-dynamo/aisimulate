# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical cost identities are validated before estimator selection."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from aisimulate_core.sdk import models, utils
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.errors import InvalidEngineConfigurationError
from aisimulate_core.sdk.models.kimi_k3 import KimiK3Model
from aisimulate_core.sdk.speculation.base import SpeculationConfig, resolve_speculation_config

pytestmark = pytest.mark.unit

# Synthetic geometry: these tests exercise identity and validation, not timings.
DRAFT_GEOMETRY = {
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 512,
}


@pytest.mark.parametrize(
    "config",
    [
        SpeculationConfig(),
        SpeculationConfig("mtp", {"depth": 8}),
        SpeculationConfig("mtp", {"depth": 3.0}),
        SpeculationConfig("ngram", {"num_speculative_tokens": 3}),
        SpeculationConfig("eagle3", {"tree_shape": [1, 4, 4], "verify_token_budget": 12}, draft_config=DRAFT_GEOMETRY),
        SpeculationConfig(
            "dflash",
            draft_config={**DRAFT_GEOMETRY, "block_size": 16, "dflash_config": {"target_layer_ids": [1, 3]}},
        ),
        SpeculationConfig("dspark", draft_config={**DRAFT_GEOMETRY, "block_size": 8, "target_layer_ids": [1, 3]}),
    ],
)
def test_cost_identity_roundtrip_does_not_impose_replay_limits(config):
    resolved = resolve_speculation_config(config)
    assert resolved == config
    saved = json.loads(json.dumps(asdict(resolved)))
    assert resolve_speculation_config(SpeculationConfig(**saved)) == resolved


@pytest.mark.parametrize(
    "config",
    [
        SpeculationConfig(kind=None),
        SpeculationConfig(kind=" "),
        SpeculationConfig(kind="unknown"),
        SpeculationConfig(params=[]),
        SpeculationConfig(params={1: 2}),
        SpeculationConfig("mtp", {"depth": "3"}),
        SpeculationConfig("mtp", {"depth": True}),
        SpeculationConfig("mtp", {"depth": 1.5}),
        SpeculationConfig("mtp", {"depth": float("nan")}),
        SpeculationConfig("mtp", {"depth": 3, "acceptance_rate": 0.5}),
        SpeculationConfig("none", {"depth": 0}),
        SpeculationConfig("ngram", {"num_speculative_tokens": 0}),
        SpeculationConfig("eagle3", {"tree_shape": [1, False]}, draft_config=DRAFT_GEOMETRY),
        SpeculationConfig("eagle3", draft_model_path=""),
        SpeculationConfig("eagle3", draft_model_path=123),
        SpeculationConfig("eagle3", draft_config=[]),
        SpeculationConfig("eagle3", draft_config={"hidden_size": 128}),
        SpeculationConfig("draft_model", {"num_speculative_tokens": 3}, draft_config=DRAFT_GEOMETRY),
    ],
)
def test_invalid_cost_identity_is_rejected(config):
    with pytest.raises(ValueError):
        resolve_speculation_config(config)


@pytest.mark.parametrize("kind,params", [("none", {}), ("mtp", {"depth": 3}), ("ngram", {"num_speculative_tokens": 2})])
@pytest.mark.parametrize("draft", [{"draft_model_path": "unused/draft"}, {"draft_config": {}}])
def test_model_free_or_target_draft_schemes_reject_unused_checkpoint(kind, params, draft):
    with pytest.raises(ValueError, match="does not accept a draft checkpoint"):
        resolve_speculation_config(SpeculationConfig(kind, params, **draft))


@pytest.mark.parametrize("kind,params", [("unknown", {}), ("eagle3", {"unknown": 1})])
def test_invalid_registry_identity_fails_before_checkpoint_loading(monkeypatch, kind, params):
    def unexpected_load(*args):
        pytest.fail("invalid scheme identity must not load a checkpoint")

    monkeypatch.setattr(utils, "_load_model_config_from_model_path", unexpected_load)
    monkeypatch.setattr(utils, "get_model_config_from_model_path", unexpected_load)
    with pytest.raises(ValueError, match="Unknown"):
        resolve_speculation_config(SpeculationConfig(kind, params, draft_model_path="unused/draft"))


def test_learned_draft_retains_source_and_resolved_content_without_reloading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text(json.dumps(DRAFT_GEOMETRY))
    config = SpeculationConfig("eagle3", {"tree_shape": [1, 2]}, draft_model_path=str(tmp_path))
    resolved = resolve_speculation_config(config)
    assert config.draft_config is None
    assert resolved.draft_model_path == str(tmp_path)
    assert resolved.draft_config == DRAFT_GEOMETRY

    checkpoint.unlink()
    monkeypatch.setattr(
        utils,
        "_load_model_config_from_model_path",
        lambda path: pytest.fail("resolved learned-draft identity must not reload its checkpoint"),
    )
    saved = SpeculationConfig(**json.loads(json.dumps(asdict(resolved))))
    assert resolve_speculation_config(saved) == resolved
    changed = replace(saved, draft_config={**saved.draft_config, "hidden_size": 256})
    assert resolve_speculation_config(changed).identity_hash() != resolved.identity_hash()
    assert replace(resolved, draft_model_path="another/draft").identity_hash() != resolved.identity_hash()


def test_resolved_identity_owns_nested_snapshots():
    config = SpeculationConfig("eagle3", {"tree_shape": [1, 2]}, draft_config=deepcopy(DRAFT_GEOMETRY))
    resolved = resolve_speculation_config(config)
    identity = resolved.identity_hash()
    config.params["tree_shape"].append(4)
    config.draft_config["hidden_size"] = 256
    assert resolved.params == {"tree_shape": [1, 2]}
    assert resolved.draft_config == DRAFT_GEOMETRY
    assert resolved.identity_hash() == identity


def test_standalone_draft_identity_uses_the_path_parser_and_rejects_mismatch(monkeypatch):
    raw = deepcopy(DRAFT_GEOMETRY)
    paths = []

    def parse(path):
        paths.append(path)
        return {"raw_config": raw}

    monkeypatch.setattr(utils, "get_model_config_from_model_path", parse)
    config = SpeculationConfig("draft_model", {"num_speculative_tokens": 3}, draft_model_path="standalone/draft")
    resolved = resolve_speculation_config(config)
    assert resolved.draft_config == raw
    assert resolved.draft_model_path == "standalone/draft"
    assert resolve_speculation_config(resolved) == resolved
    assert paths == ["standalone/draft", "standalone/draft"]

    raw["hidden_size"] = 256
    assert resolved.draft_config["hidden_size"] == 128
    with pytest.raises(ValueError, match="draft_config must match"):
        resolve_speculation_config(resolved)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_explicit_mtp_cannot_construct_a_dspark_graph(monkeypatch, backend):
    def unexpected_create(*args, **kwargs):
        pytest.fail("explicit MTP must be rejected before constructing a DSPARK graph")

    monkeypatch.setattr(KimiK3Model, "create", unexpected_create)
    config = ModelConfig(
        tp_size=8,
        moe_tp_size=8,
        moe_ep_size=1,
        speculation=SpeculationConfig("mtp", {"depth": 3}),
    )
    with pytest.raises(InvalidEngineConfigurationError, match="Explicit MTP.*DSPARK"):
        models.get_model("moonshotai/Kimi-K3", config, backend)
    assert config.nextn == 0


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_legacy_nextn_still_constructs_the_dspark_graph(backend):
    config = ModelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1, nextn=3)
    model = models.get_model("moonshotai/Kimi-K3", config, backend)
    assert model.verify_width == 4
    assert "draft_kv_proj_gemm" in {op._name for op in model.generation_ops}
