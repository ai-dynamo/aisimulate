# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re

import pytest

from aisim import _runtime
from aisim.runner import canonical_performance_config, materialize_engine_launch_config
from aisim_core.sdk import RustForwardPassPerfModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_runner_exports_canonical_engine_launch(backend):
    config = materialize_engine_launch_config(
        backend,
        "",
        {},
        {
            "block_size": 16,
            "num_gpu_blocks": 64,
            "max_model_len": 128,
            "dp_size": 2,
            "tensor_parallel_size": 2,
            "timing_model": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 1.0},
        },
        "decode",
    )
    assert config["engine"]["backend"] == backend
    assert config["engine"]["worker_type"] == "decode"
    assert config["engine"]["max_model_len"] == 128
    assert config["dp_size"] == 2
    assert config["tensor_parallel_size"] == 2
    assert config["num_gpu_blocks_is_explicit"] is True
    assert "engine_type" not in config


def test_canonical_timing_metadata_preserves_controls_and_role():
    raw = {
        "model_path": "model",
        "system": "system",
        "backend": "vllm",
        "tp_size": 2,
        "attention_dp_size": 2,
        "pp_size": 1,
        "worker_type": "prefill",
        "estimator_config": {"correction": {"enabled": False}},
    }
    config = canonical_performance_config(raw, worker_type="prefill")
    assert config["model"] == "model"
    assert config["tp"] == 2
    assert config["attention_dp"] == 2
    assert config["estimator_config"]["correction"]["enabled"] is False
    with pytest.raises(ValueError, match="worker_type"):
        canonical_performance_config(raw, worker_type="decode")


@pytest.fixture
def local_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": 4,
                "hidden_size": 2048,
                "intermediate_size": 5632,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
            }
        )
    )
    return str(target)


def _speculative_cost(kind):
    cost = {
        "kind": kind,
        "params": {"depth": 3} if kind == "mtp" else {"num_speculative_tokens": 3},
    }
    if kind == "eagle3":
        cost["draft_config"] = {
            "num_hidden_layers": 1,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 32000,
        }
    return cost


def _timing_config(model, backend="vllm", role="decode"):
    return {
        "model": model,
        "backend_version": "0.24.0" if backend == "vllm" else "0.5.14",
        "system": "h200_sxm",
        "backend": backend,
        "worker_type": role,
        "estimation_mode": "op_level",
    }


def _native_launch(engine, *, explicit_capacity=True):
    return json.loads(
        _runtime.engine_launch_from_replay_role_json(
            json.dumps({"rank": engine, "num_gpu_blocks_is_explicit": explicit_capacity})
        )
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize("kind", ["mtp", "eagle3"])
@pytest.mark.parametrize("canonical_only", [False, True], ids=["public-selection", "nested-cost-only"])
def test_generic_speculation_survives_canonical_launch(local_target, backend, role, kind, canonical_only):
    timing = _timing_config(local_target, backend, role)
    cost = _speculative_cost(kind)
    rank = {
        "block_size": 16,
        "num_gpu_blocks": 64,
        "timing_model": {"type": "external", "provider": "aic", "config": timing},
    }
    if canonical_only:
        timing["speculation"] = cost
        rank.update(aic_nextn_accept_rates="1,1,0.4", aic_mtp_seed=73)
        # A canonical cost identity must derive scheduler depth even when no
        # public selection or legacy aic_nextn field accompanies it.
        raw = {"rank": rank}
    else:
        rank["speculation"] = {**cost, "expected_accepted_tokens": 2.4, "seed": 73}
        raw = rank

    config = materialize_engine_launch_config(backend, timing["backend_version"], {}, raw, role)
    engine = config["engine"]
    assert engine["aic_nextn"] == 3
    assert engine["aic_mtp_seed"] == 73
    assert [float(rate) for rate in engine["aic_nextn_accept_rates"].split(",")] == pytest.approx([1, 1, 0.4])
    saved_cost = engine["timing_model"]["config"]
    assert saved_cost["speculation"] == cost
    assert saved_cost["nextn"] == 0
    assert saved_cost["worker_type"] == role
    assert config["num_gpu_blocks_is_explicit"] is True
    assert RustForwardPassPerfModel.config_engine_launch_mismatches(config, config=saved_cost) == []
    # A serialized native descriptor retains cost identity and workload inputs.
    restored = _native_launch(engine)
    assert restored["engine"] == engine
    assert restored["num_gpu_blocks_is_explicit"] is True


@pytest.mark.parametrize("kind", ["mtp", "eagle3"])
@pytest.mark.parametrize(
    "change,message",
    [
        ("capacity-inferred", "generic speculation requires explicitly configured positive num_gpu_blocks"),
        ("capacity-zero", "num_gpu_blocks must be positive"),
        ("acceptance-missing", "generic speculative replay requires explicit aic_nextn_accept_rates"),
        ("depth-conflict", "canonical speculative depth conflicts with scheduler aic_nextn"),
        ("backend-conflict", "AIS backend must match engine backend"),
        ("unsupported-backend", "generic speculation requires backend=vllm or backend=sglang"),
    ],
)
def test_speculative_launch_and_mismatch_query_enforce_same_controls(local_target, kind, change, message):
    timing = {**_timing_config(local_target), "speculation": _speculative_cost(kind)}
    engine = {
        "backend": "vllm",
        "worker_type": "decode",
        "block_size": 16,
        "num_gpu_blocks": 64,
        "aic_nextn": 3,
        "aic_nextn_accept_rates": "1,1,0.4",
        "timing_model": {"type": "external", "provider": "ais", "config": timing},
    }
    explicit_capacity = change != "capacity-inferred"
    if change == "capacity-zero":
        engine["num_gpu_blocks"] = 0
    elif change == "acceptance-missing":
        engine.pop("aic_nextn_accept_rates")
    elif change == "depth-conflict":
        engine.update(aic_nextn=2, aic_nextn_accept_rates="1,1")
    elif change == "unsupported-backend":
        engine["backend"] = timing["backend"] = "trtllm"
        timing.pop("backend_version")
    elif change == "backend-conflict":
        engine["backend"] = "trtllm"
    arguments = {"engine": engine, "num_gpu_blocks_is_explicit": explicit_capacity}
    if change == "capacity-zero":
        # Invalid scheduler inputs are errors, not timing-identity mismatches.
        with pytest.raises(ValueError, match=re.escape(message)):
            RustForwardPassPerfModel.config_engine_launch_mismatches(arguments, config=timing)
    else:
        assert RustForwardPassPerfModel.config_engine_launch_mismatches(arguments, config=timing) == [message]
    with pytest.raises(ValueError, match=re.escape(message)):
        _native_launch(engine, explicit_capacity=explicit_capacity)


@pytest.mark.parametrize(
    "kind,params",
    [
        ("mtp", {"depth": 6}),
        ("eagle3", {"tree_shape": [1, 2], "verify_token_budget": 4}),
        ("eagle3", {"tree_shape": [1, 1], "verify_token_budget": 4}),
    ],
)
def test_speculative_launch_and_mismatch_query_reject_unsupported_chain(local_target, kind, params):
    timing = {
        **_timing_config(local_target),
        "speculation": {**_speculative_cost(kind), "params": params},
    }
    engine = {
        "backend": "vllm",
        "worker_type": "decode",
        "num_gpu_blocks": 64,
        "aic_nextn_accept_rates": "1,1,0.4",
        "timing_model": {"type": "external", "provider": "ais", "config": timing},
    }
    with pytest.raises(ValueError, match="speculative replay requires"):
        RustForwardPassPerfModel.config_engine_launch_mismatches({"engine": engine}, config=timing)
    with pytest.raises(ValueError, match="speculative replay requires"):
        _native_launch(engine)


@pytest.mark.parametrize("kind", ["ngram", "mtp", "eagle3"])
@pytest.mark.parametrize("depth_input", ["omitted", "null", "explicit"])
def test_canonical_depth_defaults_match_replay_role(local_target, kind, depth_input):
    timing = {**_timing_config(local_target), "speculation": _speculative_cost(kind)}
    engine = {
        "backend": "vllm",
        "worker_type": "decode",
        "num_gpu_blocks": 64,
        "aic_nextn_accept_rates": "1,1,0.4",
        "timing_model": {"type": "external", "provider": "ais", "config": timing},
    }
    if depth_input != "omitted":
        engine["aic_nextn"] = None if depth_input == "null" else 3
    arguments = {"engine": engine, "num_gpu_blocks_is_explicit": True}
    assert RustForwardPassPerfModel.config_engine_launch_mismatches(arguments, config=timing) == []
    restored = _native_launch(engine)
    assert restored["engine"]["aic_nextn"] == 3
    assert restored["engine"]["timing_model"]["config"]["nextn"] == 0
    assert _native_launch(restored["engine"]) == restored


@pytest.mark.parametrize("rates", [None, "", "  ", "1,0.5,0.25"])
def test_ngram_descriptor_preserves_native_acceptance_defaults(local_target, rates):
    timing = {**_timing_config(local_target), "speculation": _speculative_cost("ngram")}
    engine = {
        "backend": "vllm",
        "worker_type": "decode",
        "num_gpu_blocks": 64,
        "aic_nextn": None,
        "timing_model": {"type": "external", "provider": "ais", "config": timing},
    }
    if rates is not None:
        engine["aic_nextn_accept_rates"] = rates
    arguments = {"engine": engine, "num_gpu_blocks_is_explicit": False}
    assert RustForwardPassPerfModel.config_engine_launch_mismatches(arguments, config=timing) == []
    restored = _native_launch(engine, explicit_capacity=False)
    assert restored["engine"]["aic_nextn"] == 3
    assert restored["engine"]["async_scheduling"] is False
    expected = [1, 0.5, 0.25] if rates and rates.strip() else [0.85, 0.3, 0]
    assert [float(rate) for rate in restored["engine"]["aic_nextn_accept_rates"].split(",")] == expected
    assert restored["num_gpu_blocks_is_explicit"] is False


@pytest.mark.parametrize("kind", ["ngram", "mtp", "eagle3"])
def test_explicit_zero_depth_is_not_treated_as_omitted(local_target, kind):
    timing = {**_timing_config(local_target), "speculation": _speculative_cost(kind)}
    engine = {
        "backend": "vllm",
        "worker_type": "decode",
        "num_gpu_blocks": 64,
        "aic_nextn": 0,
        "aic_nextn_accept_rates": "1,1,0.4",
        "timing_model": {"type": "external", "provider": "ais", "config": timing},
    }
    with pytest.raises(ValueError, match="aic_nextn"):
        _native_launch(engine)
    with pytest.raises(ValueError, match="aic_nextn"):
        RustForwardPassPerfModel.config_engine_launch_mismatches(
            {"engine": engine, "num_gpu_blocks_is_explicit": True}, config=timing
        )


def test_ngram_launch_and_mismatch_query_reject_explicit_async_scheduling(local_target):
    timing = {**_timing_config(local_target), "speculation": _speculative_cost("ngram")}
    engine = {
        "backend": "vllm",
        "worker_type": "decode",
        "num_gpu_blocks": 64,
        "async_scheduling": True,
        "timing_model": {"type": "external", "provider": "ais", "config": timing},
    }
    message = "vLLM does not support async scheduling with ngram speculative decoding"
    with pytest.raises(ValueError, match=re.escape(message)):
        _native_launch(engine)
    assert RustForwardPassPerfModel.config_engine_launch_mismatches({"engine": engine}, config=timing) == [message]
