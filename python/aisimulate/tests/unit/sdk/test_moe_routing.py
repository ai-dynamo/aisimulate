# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import shutil
from types import SimpleNamespace

import pytest
import yaml

from aiconfigurator_core.sdk import expert_popularity as db
from aiconfigurator_core.sdk.config import ModelConfig
from aiconfigurator_core.sdk.moe_routing import apply_routing, select_profile
from aiconfigurator_core.sdk.operations import MoEAllToAll, MoEExpertCompute

pytestmark = pytest.mark.unit
MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"


def select(**kwargs):
    defaults = dict(
        model_id=MODEL, revision=None, num_layers=27, num_experts=64, top_k=6, phase="prefill", backend="deepep_ll"
    )
    return select_profile(**(defaults | kwargs))


def test_measured_hit_expert_id_order_and_proxy():
    result = select()
    assert result["provenance"]["selected_mode"] == "measured"
    assert result["provenance"]["revision_selection"] == "bundle_pinned"
    assert [item["layer_id"] for item in result["layers"]] == list(range(1, 27))
    assert all(sum(item["probabilities"]) == pytest.approx(1) for item in result["layers"])
    assert select(phase="decode")["provenance"]["selected_mode"] == "measured_prefill_proxy"
    with pytest.raises(ValueError, match="phase_mismatch"):
        select(phase="decode", mode="random")


@pytest.mark.parametrize("override", ["uniform", "power-law"])
def test_explicit_override_does_not_read_corrupt_bundle(tmp_path, override):
    (tmp_path / db.model_id_to_bundle_name(MODEL)).mkdir()
    assert select(data_root=tmp_path, mode=override)["provenance"]["selected_mode"] == override


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"model_id": "missing/Model"}, "missing_bundle"),
        ({"phase": "other"}, "phase_mismatch"),
        ({"backend": "deepep_ht"}, "unsupported_consumer_backend"),
    ],
)
def test_fallback_and_strict_random(changes, reason):
    assert select(**changes)["provenance"]["fallback_reason"] == reason
    with pytest.raises(ValueError, match=reason):
        select(mode="random", **changes)


@pytest.mark.parametrize(
    "changes", [{"revision": "bad"}, {"num_layers": 26}, {"num_experts": 32}, {"top_k": 4}, {"enable_eplb": True}]
)
def test_present_mismatch_is_error(changes):
    with pytest.raises(ValueError):
        select(**changes)


@pytest.mark.parametrize("fault", ["sha", "missing_file", "identity"])
def test_corrupt_candidate_is_not_fallback(tmp_path, fault):
    source = db._resolve_bundle(MODEL, None)
    bundle = tmp_path / source.name
    shutil.copytree(source, bundle)
    meta_path = bundle / db.METADATA_FILENAME
    metadata = yaml.safe_load(meta_path.read_text())
    if fault == "sha":
        metadata["files"]["parquet"]["sha256"] = "0" * 64
    elif fault == "identity":
        metadata["model"]["id"] = "wrong/Model"
    else:
        (bundle / db.PARQUET_FILENAME).unlink()
    meta_path.write_text(yaml.safe_dump(metadata))
    with pytest.raises((ValueError, FileNotFoundError)):
        select(data_root=tmp_path)


@pytest.mark.parametrize("alpha", [0, -1, float("inf"), float("nan")])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        ModelConfig(moe_routing_mode="power-law", moe_power_law_alpha=alpha)


def test_alpha_requires_power_law_and_legacy_conflict():
    with pytest.raises(ValueError):
        ModelConfig(moe_power_law_alpha=1.2)
    with pytest.raises(ValueError):
        ModelConfig(moe_routing_mode="random", workload_distribution="uniform")
    assert select(legacy="uniform")["provenance"]["selected_mode"] == "legacy"
    with pytest.raises(ValueError):
        ModelConfig(moe_routing_mode="uniform", workload_distribution="power_law")


def test_cli_sdk_yaml_config_plumbing():
    from aiconfigurator.cli.main import _build_common_cli_experiments_parser
    from aiconfigurator.sdk.config_builders import build_model_config
    from aiconfigurator.sdk.task_v2 import Task

    config = build_model_config(1, 1, 4, 1, 4, moe_routing_mode="power-law", moe_power_law_alpha=1.3)
    assert config.moe_power_law_alpha == 1.3
    assert "moe_routing_mode" in Task.__dataclass_fields__
    assert "moe_power_law_alpha" in Task.__dataclass_fields__
    # Check the real parser, not a parallel test-only schema.
    parser = _build_common_cli_experiments_parser()
    args = parser.parse_args(["--moe-routing-mode", "power-law", "--moe-power-law-alpha", "1.3"])
    assert args.moe_routing_mode == config.moe_routing_mode
    assert args.moe_power_law_alpha == config.moe_power_law_alpha


def test_layer_expansion_native_wire_pickle_and_no_dense_layer():
    cfg = ModelConfig(
        tp_size=2,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=4,
        moe_comm_backend={"context": "deepep_ll", "generation": "deepep_ll"},
    )
    comm = MoEAllToAll(
        "context_dispatch",
        27,
        phase="dispatch",
        comm_backend="deepep_ll",
        hidden_size=2048,
        topk=6,
        num_experts=64,
        moe_ep_size=4,
        node_num=1,
        attention_tp_size=2,
    )
    compute = MoEExpertCompute(
        "context_compute",
        27,
        hidden_size=2048,
        inter_size=1408,
        topk=6,
        num_experts=64,
        moe_ep_size=4,
        quant_mode="bfloat16",
        workload_distribution="power_law_1.2",
        attention_dp_size=2,
        inference_phase="context",
    )
    model = SimpleNamespace(
        config=cfg,
        model_path=MODEL,
        _num_layers=27,
        _num_experts=64,
        _topk=6,
        context_ops=[comm, compute],
        generation_ops=[comm, compute],
    )
    apply_routing(model, {"layers": 27})
    assert len(model.context_ops) == 52
    spec = json.loads(model.context_ops[0]._spec_json())["MoeAllToAll"]
    assert spec["scale_factor"] == 1
    assert spec["measured_routing"]["layer_id"] == 1
    assert spec["measured_routing"]["probabilities"] == select()["layers"][0]["probabilities"]
    compute_spec = json.loads(model.context_ops[26]._spec_json())["MoeExpertCompute"]
    assert compute_spec["routing_attention_tp_size"] == 2
    for op in model.context_ops:
        assert pickle.loads(pickle.dumps(op))._spec_json() == op._spec_json()


def test_native_compile_entry_preserves_selection_and_provenance(monkeypatch):
    import aiconfigurator_core
    from aiconfigurator_core.sdk import engine

    captured = {}

    def build_model(path, config, backend):
        captured["config"] = config
        return SimpleNamespace(config=config)

    monkeypatch.setattr(engine, "get_model", build_model)
    monkeypatch.setattr(engine, "_literal_backend_version", lambda *args: "fixed")
    monkeypatch.setattr(
        engine, "_maybe_load_database", lambda *args: SimpleNamespace(system_spec={"node": {"num_gpus_per_node": 4}})
    )

    def build_spec(model, **kwargs):
        captured["version"] = kwargs["backend_version"]
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", build_spec)
    monkeypatch.setattr(aiconfigurator_core, "engine_spec_bincode_from_json", lambda value: b"test")
    assert (
        engine.compile_engine(
            MODEL,
            "gb200",
            "sglang",
            moe_routing_mode="power-law",
            moe_power_law_alpha=1.2,
            moe_model_revision="pinned",
            moe_comm_backend={"context": "deepep_ll"},
        )
        == b"test"
    )
    config = captured["config"]
    assert config.moe_routing_mode == "power-law"
    assert config.moe_power_law_alpha == 1.2
    assert config.moe_model_revision == "pinned"
    assert config.moe_comm_backend == {"context": "deepep_ll"}
    assert config.num_gpus_per_node == 4
    assert captured["version"] == "fixed"


def test_invalid_profile_is_hard_native_configuration_error(tmp_path):
    from aiconfigurator_core.sdk.moe_routing import MoeRoutingError

    (tmp_path / db.model_id_to_bundle_name(MODEL)).mkdir()
    with pytest.raises(MoeRoutingError):
        select(data_root=tmp_path)


@pytest.mark.parametrize("pin_target_revision", [False, True])
def test_speculative_moe_draft_keeps_own_measured_profile_and_token_width(pin_target_revision):
    from aiconfigurator_core.sdk.engine import _ops_json
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.speculation import SpeculationConfig

    draft_path = "Qwen/Qwen3-30B-A3B"
    target_metadata = db.validate_expert_popularity_bundle(db._resolve_bundle(MODEL, None))
    config = ModelConfig(
        tp_size=2,
        attention_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=4,
        moe_comm_backend={"context": "deepep_ll", "generation": "deepep_ll"},
        num_gpus_per_node=4,
        moe_model_revision=target_metadata["model"]["revision"] if pin_target_revision else None,
        speculation=SpeculationConfig(
            kind="draft_model",
            params={"num_speculative_tokens": 3, "draft_tp_size": 1},
            draft_model_path=draft_path,
        ),
    )
    model = get_model(MODEL, config, "vllm")
    draft = model.spec_scheme._draft_model
    assert model.verify_width == 4
    assert model.moe_routing_provenance["decode"]["model_id"] == MODEL
    assert draft.moe_routing_provenance["decode"]["model_id"] == draft_path
    assert model.moe_routing_provenance["decode"]["revision_selection"] == (
        "explicit" if pin_target_revision else "bundle_pinned"
    )
    assert draft.moe_routing_provenance["decode"]["revision_selection"] == "bundle_pinned"
    target_digest = model.moe_routing_provenance["decode"]["profile_digest"]
    draft_digest = draft.moe_routing_provenance["decode"]["profile_digest"]
    assert target_digest != draft_digest

    target_compute = []
    draft_compute = []
    for spec in json.loads(_ops_json(model.generation_ops)):
        if "MoeExpertCompute" in spec:
            target_compute.append(spec["MoeExpertCompute"])
        elif "TokenScale" in spec and "MoeExpertCompute" in spec["TokenScale"]["op"]:
            wrapper = spec["TokenScale"]
            assert (wrapper["numerator"], wrapper["denominator"]) == (1, 4)
            draft_compute.append(wrapper["op"]["MoeExpertCompute"])
    # Coder Lite has 26 routed layers; the Qwen draft has 48, repeated for
    # three independent draft forwards. Target verification is unwrapped.
    assert len(target_compute) == 26
    assert len(draft_compute) == 3 * 48
    assert {op["measured_routing"]["profile_digest"] for op in target_compute} == {target_digest}
    assert {op["measured_routing"]["profile_digest"] for op in draft_compute} == {draft_digest}
    assert all(len(op["measured_routing"]["probabilities"]) == 64 for op in target_compute)
    assert all(len(op["measured_routing"]["probabilities"]) == 128 for op in draft_compute)
