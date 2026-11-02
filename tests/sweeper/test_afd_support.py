# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD support boundaries, source metadata, and cross-language count contracts."""

import json

import pytest
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.config.engine import AFDSearchRecommendationConfig, AFDTopologyPredictionConfig
from aisimulate.sweeper.afd_parallel import (
    AFDInfeasible,
    AFDReasonCategory,
    AFDSearchConfig,
    AFDTopology,
    enumerate_afd_topologies,
    validate_afd_model,
)
from aisimulate_core._native import resolve_afd_parallelism, validate_afd_model_shape


def _topology(**overrides):
    values = dict(
        n_a_nodes=2,
        n_f_nodes=3,
        gpus_per_node=8,
        a_tp_size=1,
        batch_size_per_a_replica=12,
        is_moe=True,
        num_experts=384,
    )
    values.update(overrides)
    return AFDTopology(**values)


def test_rust_python_json_use_identical_counts_and_sizes():
    topology = _topology()
    native = json.loads(resolve_afd_parallelism(1, 2, 3, 8, None, 384))
    identity = topology.provenance()["topology"]
    assert native == {key: getattr(topology, key) for key in native}
    assert native == {key: identity[key] for key in native}
    assert native["a_replica_count"] == 16
    assert native["f_rank_count"] == 24
    assert topology.total_batch_size == 192
    assert topology.provenance()["schema_version"] == 2
    for old in ("attention_workers", "ffn_workers", "ffn_tp", "tp_a", "a_batch_size"):
        assert not hasattr(topology, old)
        assert old not in identity


def test_topology_mapping_round_trips_and_checks_derived_counts():
    source = _topology().provenance()["topology"]
    original = dict(source)
    restored = AFDTopology.from_mapping(json.loads(json.dumps(source)))
    assert restored == _topology()
    assert source == original
    with pytest.raises(ValueError, match="disagrees"):
        AFDTopology.from_mapping({**source, "f_rank_count": 7})


@pytest.mark.parametrize("value", [True, False])
def test_removed_boundary_option_is_rejected_at_public_inputs(value):
    prediction = dict(n_a_nodes=1, n_f_nodes=1, tp_a=1, a_batch_size=12)
    for config_type, values in (
        (AFDTopologyPredictionConfig, prediction),
        (AFDSearchRecommendationConfig, dict(a_batch_size=12)),
    ):
        assert "boundary_on_attn" not in config_type(**values).model_dump()
        with pytest.raises(ValidationError, match="boundary_on_attn"):
            config_type(**values, boundary_on_attn=value)

    topology = _topology().provenance()["topology"]
    assert "boundary_on_attn" not in topology
    with pytest.raises(TypeError, match="boundary_on_attn"):
        AFDTopology.from_mapping({**topology, "boundary_on_attn": value})
    with pytest.raises(TypeError, match="boundary_on_attn"):
        AFDSearchConfig(total_gpus=16, gpus_per_node=8, is_moe=True, boundary_on_attn=value)
    with pytest.raises(TypeError, match="boundary_on_attn"):
        resolve_afd_parallelism(1, 1, 1, 8, boundary_on_attn=value)


def test_omitted_ep_resolves_but_saved_ep1_is_explicit():
    assert _topology().f_moe_ep_size == 24
    with pytest.raises(NotImplementedError, match="f_moe_tp_size=24"):
        _topology(f_moe_ep_size=1)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"f_moe_ep_size": 12}, "f_moe_tp_size=2"),
        ({"is_moe": False, "num_experts": 0}, "Dense AFD"),
    ],
)
def test_unsupported_geometry_raises_not_implemented(changes, match):
    with pytest.raises(NotImplementedError, match=match):
        _topology(**changes)


@pytest.mark.parametrize(
    "shape,match",
    [
        (dict(num_shared_experts=1, num_moe_layers=48), "shared experts"),
        (dict(num_shared_experts=0, num_moe_layers=47), "Dense FFN"),
    ],
)
def test_model_gate_uses_source_features(shape, match, monkeypatch):
    from aisimulate_core.sdk.models import helpers

    monkeypatch.setattr(
        helpers,
        "_get_model_info",
        lambda model: dict(
            layers=48,
            num_experts=128,
            topk=8,
            architecture="TestMoeForCausalLM",
            extra_params=None,
            raw_config={"first_k_dense_replace": 48 - shape["num_moe_layers"]},
            **shape,
        ),
    )
    assert validate_afd_model("test/model")["num_shared_experts"] == shape["num_shared_experts"]


def test_missing_source_facts_are_not_defaulted_to_supported(monkeypatch):
    from aisimulate_core.sdk.models import helpers

    monkeypatch.setattr(helpers, "_get_model_info", lambda model: dict(layers=48, num_experts=128, topk=8))
    with pytest.raises(ValueError, match="num_shared_experts"):
        validate_afd_model("test/model")
    validate_afd_model_shape(48, 128, 8, 0, 48)


@pytest.mark.parametrize(
    "domain,feasible",
    [
        (dict(f_moe_ep_size_candidates=(4, "f_gpu_count")), True),
        (dict(f_moe_ep_size_candidates=("n_f_nodes",)), False),
    ],
)
def test_generated_ep_domains_keep_only_supported_shapes(domain, feasible):
    config = AFDSearchConfig(total_gpus=16, gpus_per_node=8, is_moe=True, num_experts=128, **domain)
    if feasible:
        result = enumerate_afd_topologies(config)
        assert result.candidates
        assert all(t.f_moe_ep_size == t.f_gpu_count == 8 for t in result.candidates)
    else:
        with pytest.raises(AFDInfeasible) as error:
            enumerate_afd_topologies(config)
        assert error.value.category is AFDReasonCategory.NO_FEASIBLE_TOPOLOGY


def test_default_domain_contains_only_dp_ep():
    domain = enumerate_afd_topologies(AFDSearchConfig(total_gpus=32, gpus_per_node=8, is_moe=True, num_experts=128))
    assert domain.candidates
    assert all(t.a_tp_size == t.f_moe_tp_size == 1 and t.f_moe_ep_size == t.f_gpu_count for t in domain.candidates)


def test_predict_tp_reaches_standard_data_preflight(monkeypatch):
    config = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 16, "output_tokens": 4},
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
            "engine": {
                "mode": "aggregated",
                "model": "missing/model",
                "hardware": "missing_system",
                "backend": "sglang",
                "workers": {
                    "aggregated": {
                        "type": "afd",
                        "afd": {
                            "n_a_nodes": 1,
                            "n_f_nodes": 1,
                            "tp_a": 2,
                            "a_batch_size": 12,
                        },
                    }
                },
            },
        }
    )
    monkeypatch.setattr("aisimulate.compiler.validate_afd_model", lambda *args: {"num_experts": 128})
    with pytest.raises(RuntimeError, match="no perf database"):
        prediction_to_replay_spec(config)
