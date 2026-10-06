# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay assumptions preserve the SDK scheme, geometry and cost identity."""

import json
from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError

from aisimulate.config.engine import SpeculationConfig
from aisimulate_core.sdk import models
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.memory import KVCacheEstimator, estimate_kv_cache
from aisimulate_core.sdk.speculation import SpeculationConfig as CostConfig

from .test_dense_draft_schemes import DFLASH_CONFIG, DSPARK_8B_CONFIG, EAGLE3_CONFIG

pytestmark = pytest.mark.unit
PUBLIC = TypeAdapter(SpeculationConfig)
TARGET = "Qwen/Qwen3-8B"
SCHEMES = [
    ({"kind": "mtp", "params": {"depth": 3}}, 3, 4),
    ({"kind": "ngram", "params": {"num_speculative_tokens": 7}}, 7, 8),
    ({"kind": "eagle3", "params": {"num_speculative_tokens": 3}, "draft_config": EAGLE3_CONFIG}, 3, 4),
    (
        {
            "kind": "eagle3",
            "params": {"tree_shape": [1, 4, 4], "verify_token_budget": 10},
            "draft_config": EAGLE3_CONFIG,
        },
        3,
        10,
    ),
    ({"kind": "dflash", "params": {}, "draft_config": DFLASH_CONFIG}, 15, 16),
    ({"kind": "dspark", "params": {}, "draft_config": DSPARK_8B_CONFIG}, 7, 8),
    ({"kind": "draft_model", "params": {"num_speculative_tokens": 3}, "draft_model_path": TARGET}, 3, 4),
]


@pytest.mark.parametrize("cost,depth,width", SCHEMES)
def test_registered_scheme_roundtrip_and_graph(cost, depth, width):
    public = PUBLIC.validate_python({**deepcopy(cost), "expected_accepted_tokens": 1.5, "seed": 73})
    assert public.max_accepted_draft_tokens == depth
    assert public.verify_width == width
    assert public.acceptance_rates == [1.0, 0.5] + [0.0] * (depth - 2)
    assert PUBLIC.validate_json(public.model_dump_json()) == public
    changed = PUBLIC.validate_python({**public.model_dump(), "expected_accepted_tokens": 0, "seed": 74})
    assert changed.cost_config() == public.cost_config()

    graph = models.get_model(TARGET, ModelConfig(speculation=CostConfig(**public.cost_config())), "vllm")
    assert graph.model_path == TARGET
    assert graph._nextn + 1 == width
    assert graph.spec_scheme.max_accepted_draft_tokens() == depth
    if public.kind not in {"mtp", "ngram"}:
        assert any(op._name.startswith("draft_") for op in graph.generation_ops)


def test_local_draft_resolution_retains_geometry_and_identity(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(EAGLE3_CONFIG))
    public = PUBLIC.validate_python(
        {
            "kind": "eagle3",
            "params": {"num_speculative_tokens": 3},
            "draft_model_path": str(tmp_path),
            "acceptance_rates": [1, 0.5, 0],
        }
    )
    assert public.draft_config == EAGLE3_CONFIG
    assert public.cost_config()["draft_model_path"] == str(tmp_path)
    saved = public.model_dump()
    (tmp_path / "config.json").unlink()
    assert PUBLIC.validate_python(saved) == public
    changed = deepcopy(saved)
    changed["draft_config"]["draft_vocab_size"] //= 2
    other = PUBLIC.validate_python(changed)
    assert CostConfig(**public.cost_config()).identity_hash() != CostConfig(**other.cost_config()).identity_hash()
    graphs = [
        models.get_model(TARGET, ModelConfig(speculation=CostConfig(**p.cost_config())), "vllm")
        for p in (public, other)
    ]
    assert [op._spec_json() for op in graphs[0].generation_ops] != [op._spec_json() for op in graphs[1].generation_ops]


@pytest.mark.parametrize(
    "updates",
    [
        {"expected_accepted_tokens": None},
        {"acceptance_rates": [1, 1, 1]},
        {"expected_accepted_tokens": 3.5},
        {"expected_accepted_tokens": float("nan")},
        {"params": {"tree_shape": [1, 4, 4], "unknown": 1}},
    ],
)
def test_tree_rejects_missing_conflicting_or_unreachable_acceptance(updates):
    cost = SCHEMES[3][0]
    with pytest.raises((ValidationError, ValueError)):
        PUBLIC.validate_python({**cost, "expected_accepted_tokens": 1.5, **updates})


def test_conditional_probabilities_follow_tree_depth_not_verify_budget():
    cost = SCHEMES[3][0]
    with pytest.raises(ValidationError, match="accepted draft depth"):
        PUBLIC.validate_python({**cost, "acceptance_rates": [1] * 9})
    public = PUBLIC.validate_python({**cost, "acceptance_rates": [1, 0.5, 0]})
    assert public.verify_width == 10 and public.acceptance_rates == [1, 0.5, 0]


def test_standalone_draft_cannot_silently_ignore_injected_geometry():
    public = PUBLIC.validate_python({**SCHEMES[-1][0], "expected_accepted_tokens": 1.5})
    changed = deepcopy(public.model_dump())
    changed["draft_config"]["num_hidden_layers"] += 1
    with pytest.raises(ValidationError, match="draft_config must match"):
        PUBLIC.validate_python(changed)


def test_profile_resources_cannot_silently_discard_a_draft():
    with pytest.raises(ValueError, match="speculation must be omitted"):
        estimate_kv_cache(
            TARGET,
            "h200_sxm",
            "vllm",
            "0.24.0",
            max_num_tokens=512,
            max_batch_size=8,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            fpm_profile={},
            speculation=SCHEMES[2][0],
        )


@pytest.mark.parametrize(
    "legacy",
    [
        {"kind": "mtp", "num_speculative_tokens": 3, "expected_accepted_tokens": 1.5},
        {"kind": "ngram", "num_speculative_tokens": 3, "acceptance_rates": [1, 0.5, 0]},
    ],
)
def test_flat_configuration_matches_generic_assumptions(legacy):
    flat = PUBLIC.validate_python(legacy)
    canonical = PUBLIC.validate_python({**flat.cost_config(), "acceptance_rates": flat.acceptance_rates})
    assert canonical.max_accepted_draft_tokens == flat.max_accepted_draft_tokens
    assert canonical.verify_width == flat.verify_width
    assert canonical.acceptance_rates == flat.acceptance_rates
    assert PUBLIC.validate_json(flat.model_dump_json()) == flat


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_automatic_capacity_includes_existing_draft_memory_hooks(backend):
    common = dict(
        model_path=TARGET,
        system="h200_sxm",
        backend=backend,
        backend_version="0.24.0" if backend == "vllm" else "0.5.14",
        max_num_tokens=8192,
        max_batch_size=8,
    )
    plain = KVCacheEstimator.from_request(**common).breakdown
    cost = SCHEMES[2][0]
    draft = KVCacheEstimator.from_request(**common, speculation=cost).breakdown
    graph = models.get_model(TARGET, ModelConfig(speculation=CostConfig(**cost)), backend)
    assert draft["weights_bytes"] - plain["weights_bytes"] == pytest.approx(
        graph.spec_scheme.draft_weights_bytes(graph)
    )
    budget = 8 * (1 << 30)
    assert 0 < draft["tokens_from_kv_bytes"](budget) < plain["tokens_from_kv_bytes"](budget)
