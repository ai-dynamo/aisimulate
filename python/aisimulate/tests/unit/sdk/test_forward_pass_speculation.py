# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public speculative-cost contract, preserving existing compiled-engine answers.

The compatibility oracle is the existing compiled SDK engine, not Python timing
math. Draft fixtures below are synthetic geometries, with no checkpoint download.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
from aisimulate_core.sdk.engine import EngineHandle

pytestmark = pytest.mark.unit


def _config(**kwargs):
    defaults = {
        "model": "Qwen/Qwen3-8B",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "worker_type": "aggregated",
        "estimation_mode": "op_level",
        "gemm_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "kvcache_quant_mode": "bfloat16",
    }
    return ForwardPassPerfModelConfig(**(defaults | kwargs))


def _eagle_spec(**params):
    return {
        "kind": "eagle3",
        "params": {"num_speculative_tokens": 3} | params,
        "draft_config": {
            "num_hidden_layers": 1,
            "hidden_size": 4096,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 32768,
            "draft_vocab_size": 16384,
        },
    }


def _metrics(prefill, verify_width=1):
    # Existing OP ForwardPassMetrics uses already-packed token counters.
    # B requests at width v carry N=B*v and K=B*v*s, retaining kv_per_req=s.
    fields = (
        {"num_prefill_requests": 2, "sum_prefill_tokens": 1024}
        if prefill
        else {"num_decode_requests": 2 * verify_width, "sum_decode_kv_tokens": 2048 * verify_width}
    )
    return [{"version": 1, "dp_rank": 0, "scheduled_requests": fields}]


def _static(model, prefill=False):
    return model.static_phase_latency(batch_size=2, input_tokens=512, output_tokens=4, prefill=prefill)


@pytest.mark.parametrize("database_mode", ["SILICON", "SOL"])
@pytest.mark.parametrize("depth", [2, 5])
def test_mtp_public_cost_matches_legacy_nextn(database_mode, depth):
    config = _config(database_mode=database_mode)
    legacy = RustForwardPassPerfModel.best_available(replace(config, nextn=depth))
    explicit = RustForwardPassPerfModel.best_available(
        replace(config, speculation={"kind": "mtp", "params": {"depth": depth}})
    )
    try:
        for prefill in (True, False):
            assert _static(explicit, prefill) == pytest.approx(_static(legacy, prefill), rel=1e-12)
            assert explicit.estimate_forward_pass_time_ms(_metrics(prefill, depth + 1)) == pytest.approx(
                legacy.estimate_forward_pass_time_ms(_metrics(prefill, depth + 1)), rel=1e-12
            )
        assert explicit.diagnostics()["provenance"]["config"]["speculation"]["params"] == {"depth": depth}
    finally:
        legacy.close()
        explicit.close()


@pytest.mark.parametrize("database_mode", ["SILICON", "SOL"])
def test_eagle_public_cost_matches_existing_sdk_engine(database_mode):
    # A tree and width greater than five remain valid cost-model inputs.
    speculation = _eagle_spec(tree_shape=[2, 3], verify_token_budget=8)
    config = _config(database_mode=database_mode, speculation=speculation)
    model = RustForwardPassPerfModel.best_available(config)
    reference = EngineHandle.compile(
        config.model,
        config.system,
        config.backend,
        backend_version=config.backend_version,
        gemm_quant_mode=config.gemm_quant_mode,
        fmha_quant_mode=config.fmha_quant_mode,
        kvcache_quant_mode=config.kvcache_quant_mode,
        database_mode=database_mode,
        speculation=speculation,
    )
    try:
        assert _static(model, True) == pytest.approx(reference.predict_prefill_latency(2, 512), rel=1e-12)
        assert _static(model) == pytest.approx(reference.predict_decode_latency(2, 512, 4), rel=1e-12)
        # Existing static decode evaluates s=isl+1; osl=2 contains one decode
        # step. Thus isl=1023 matches the packed telemetry's s=1024 exactly.
        assert model.estimate_forward_pass_time_ms(_metrics(False, 8)) == pytest.approx(
            reference.predict_decode_latency(2, 1023, 2), rel=1e-12
        )
        with pytest.raises(ValueError, match="scale_num_tokens.*batch_size"):
            model.estimate_forward_pass_time_ms(_metrics(False))
        assert model.diagnostics()["provenance"]["selected_estimation_mode"] == "op_level"
    finally:
        model.close()


def test_mtp_static_cost_is_not_limited_by_replay_depth():
    config = _config(database_mode="SOL", speculation={"kind": "mtp", "params": {"depth": 7}})
    model = RustForwardPassPerfModel.best_available(config)
    reference = EngineHandle.compile(
        config.model,
        config.system,
        config.backend,
        backend_version=config.backend_version,
        database_mode="SOL",
        nextn=7,
    )
    try:
        assert _static(model) == pytest.approx(reference.predict_decode_latency(2, 512, 4), rel=1e-12)
    finally:
        model.close()


@pytest.mark.parametrize("speculation", [None, {"kind": "ngram", "params": {"num_speculative_tokens": 3}}])
def test_existing_no_sd_and_ngram_costs_are_preserved(speculation):
    config = _config(database_mode="SOL", speculation=speculation)
    model = RustForwardPassPerfModel.best_available(config)
    reference = EngineHandle.compile(
        config.model,
        config.system,
        config.backend,
        backend_version=config.backend_version,
        database_mode="SOL",
        speculation=speculation,
    )
    try:
        assert _static(model) == pytest.approx(reference.predict_decode_latency(2, 512, 4), rel=1e-12)
        resolved = model.diagnostics()["provenance"]["config"]
        if speculation is not None:
            assert resolved["speculation"]["kind"] == "ngram"
            assert resolved["speculation"]["params"] == speculation["params"]
        restored = RustForwardPassPerfModel.best_available(json.loads(json.dumps(resolved)))
        try:
            assert _static(restored) == pytest.approx(_static(model), rel=1e-12)
        finally:
            restored.close()
    finally:
        model.close()


def test_standalone_local_draft_preserves_resolved_geometry_and_cost(tmp_path):
    draft = tmp_path / "standalone-draft"
    draft.mkdir()
    geometry = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 2,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 32768,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
    }
    (draft / "config.json").write_text(json.dumps(geometry))
    speculation = {
        "kind": "draft_model",
        "params": {"num_speculative_tokens": 2},
        "draft_model_path": str(draft),
    }
    config = _config(database_mode="SOL", speculation=speculation)
    model = RustForwardPassPerfModel.best_available(config)
    reference = EngineHandle.compile(
        config.model,
        config.system,
        config.backend,
        backend_version=config.backend_version,
        database_mode="SOL",
        speculation=speculation,
    )
    try:
        assert _static(model) == pytest.approx(reference.predict_decode_latency(2, 512, 4), rel=1e-12)
        resolved = model.diagnostics()["provenance"]["config"]
        assert resolved["speculation"]["draft_config"] == geometry
        restored = RustForwardPassPerfModel.best_available(json.loads(json.dumps(resolved)))
        try:
            assert _static(restored) == pytest.approx(_static(model), rel=1e-12)
        finally:
            restored.close()
    finally:
        model.close()


def test_resolved_draft_content_survives_save_without_source_file(tmp_path):
    source = tmp_path / "synthetic-eagle"
    source.mkdir()
    content = _eagle_spec()["draft_config"]
    (source / "config.json").write_text(json.dumps(content))
    config = _config(
        database_mode="SOL",
        speculation={"kind": "eagle3", "params": {"num_speculative_tokens": 3}, "draft_model_path": str(source)},
    )
    original = RustForwardPassPerfModel.best_available(config)
    try:
        resolved = original.diagnostics()["provenance"]["config"]
        assert resolved["speculation"]["draft_model_path"] == str(source)
        assert resolved["speculation"]["draft_config"] == content
        saved = tmp_path / "saved-cost-config.json"
        saved.write_text(json.dumps(resolved))
        (source / "config.json").unlink()
        source.rmdir()
        reloaded = RustForwardPassPerfModel.best_available(json.loads(saved.read_text()))
        try:
            assert reloaded.diagnostics()["provenance"]["config"] == resolved
            assert _static(reloaded) == pytest.approx(_static(original), rel=1e-12)
            assert reloaded.estimate_forward_pass_time_ms(_metrics(False, 4)) == pytest.approx(
                original.estimate_forward_pass_time_ms(_metrics(False, 4)), rel=1e-12
            )
        finally:
            reloaded.close()
    finally:
        original.close()


def test_different_draft_geometry_keeps_distinct_identity_and_cost():
    first = _eagle_spec()
    second = copy.deepcopy(first)
    second["draft_config"]["intermediate_size"] *= 2
    models = [
        RustForwardPassPerfModel.best_available(_config(database_mode="SOL", speculation=spec))
        for spec in (first, second)
    ]
    try:
        identities = [model.diagnostics()["provenance"]["config"]["speculation"] for model in models]
        assert identities[0]["draft_config"] == first["draft_config"]
        assert identities[1]["draft_config"] == second["draft_config"]
        smaller = _static(models[0])
        assert _static(models[1]) > smaller
        # Querying the second geometry must not replace a cached first graph.
        assert _static(models[0]) == smaller
    finally:
        for model in models:
            model.close()


@pytest.mark.parametrize("entrypoint", ["normalize_config", "best_available"])
@pytest.mark.parametrize(
    ("speculation", "error"),
    [
        ({"kind": "unknown_scheme", "params": {}}, "[Uu]nknown"),
        ({"kind": "mtp", "params": {"depth": 3, "typo": 1}}, "[Uu]nknown.*typo"),
        ({"kind": "mtp", "params": {"depth": 0}}, "depth"),
        ({"kind": "mtp", "params": {"depth": 1.5}}, "integer"),
        ({"kind": "mtp", "params": {"depth": True}}, "integer"),
        ({"kind": "eagle3", "params": {}}, "draft"),
        ({"kind": "mtp", "params": {"depth": 3}, "seed": 42}, "unknown field"),
    ],
)
def test_invalid_speculation_never_uses_fallback(entrypoint, speculation, error):
    # Regression would otherwise construct successfully without a native graph.
    config = _config(estimation_mode="fpm_regression", fallback_policy="allow", speculation=speculation)
    with pytest.raises(ValueError, match=error):
        getattr(RustForwardPassPerfModel, entrypoint)(config)


@pytest.mark.parametrize("entrypoint", ["normalize_config", "best_available"])
@pytest.mark.parametrize("failure", ["download", "validation", "import", "runtime", "same_name"])
def test_checkpoint_resolution_classifies_only_recognized_availability_errors(monkeypatch, entrypoint, failure):
    from aisimulate_core.sdk import engine, utils

    error_type = {
        "download": utils.HuggingFaceDownloadError,
        "validation": ValueError,
        "import": ImportError,
        "runtime": RuntimeError,
        # A similarly named arbitrary exception is not the SDK's loader error.
        "same_name": type("HuggingFaceDownloadError", (Exception,), {}),
    }[failure]
    attempted_paths = []
    message = "checkpoint failure from the test loader"

    def fail_loading(path):
        attempted_paths.append(path)
        raise error_type(message)

    def unexpected_compile(*args, **kwargs):
        pytest.fail("unresolved draft identity must fail before estimator candidates are built")

    monkeypatch.setattr(utils, "_load_model_config_from_model_path", fail_loading)
    monkeypatch.setattr(engine, "compile_engine", unexpected_compile)
    path = "test-fixture/unavailable-draft"
    config = _config(
        estimation_mode="auto",
        fallback_policy="allow",
        speculation={"kind": "eagle3", "params": {"num_speculative_tokens": 3}, "draft_model_path": path},
    )
    expected = (
        "unsupported model for Rust core estimator: speculation checkpoint unavailable:"
        if failure == "download"
        else "invalid engine config: speculation:"
    )
    with pytest.raises(ValueError, match=expected) as caught:
        getattr(RustForwardPassPerfModel, entrypoint)(config)
    assert message in str(caught.value)
    assert attempted_paths == [path]


@pytest.mark.parametrize("mode", ["fpm_interpolation", "fpm_regression"])
def test_explicit_fpm_rejects_unimplemented_speculative_cost(mode):
    with pytest.raises(ValueError, match="speculat.*op_level|op_level.*speculat"):
        RustForwardPassPerfModel.best_available(
            _config(estimation_mode=mode, speculation={"kind": "mtp", "params": {"depth": 3}})
        )


@pytest.mark.parametrize(
    ("mode", "fallback"), [("auto", "deny"), ("fpm_interpolation", "allow"), ("fpm_regression", "allow")]
)
@pytest.mark.parametrize(
    "spec",
    [{"kind": "mtp", "params": {"depth": 3}}, {"kind": "ngram", "params": {"num_speculative_tokens": 3}}],
    ids=["mtp", "ngram"],
)
def test_selection_preserves_speculation(mode, fallback, spec):
    config = _config(database_mode="SOL", estimation_mode=mode, fallback_policy=fallback, speculation=spec)
    model = RustForwardPassPerfModel.best_available(config)
    reference = RustForwardPassPerfModel.best_available(replace(config, estimation_mode="op_level"))
    try:
        provenance = model.diagnostics()["provenance"]
        assert provenance["requested_estimation_mode"] == mode
        assert provenance["selected_estimation_mode"] == "op_level"
        assert provenance["config"]["speculation"]["kind"] == spec["kind"]
        assert provenance["config"]["speculation"]["params"] == spec["params"]
        assert _static(model) == pytest.approx(_static(reference), rel=1e-12)
    finally:
        model.close()
        reference.close()


def test_explicit_none_normalizes_to_disabled_speculation():
    resolved = RustForwardPassPerfModel.normalize_config(_config(speculation={"kind": "none"}))
    assert resolved["speculation"] is None
    assert RustForwardPassPerfModel.normalize_config(json.loads(json.dumps(resolved))) == resolved


def test_fpm_only_selector_cannot_be_discarded_by_speculative_fallback():
    # OP cannot honor this FPM identity field. Returning OP while retaining the
    # field would produce a saved configuration rejected by its own validator.
    with pytest.raises(ValueError, match="fpm_fmha_quant_mode"):
        RustForwardPassPerfModel.best_available(
            _config(
                database_mode="SOL",
                estimation_mode="fpm_interpolation",
                fallback_policy="allow",
                fpm_fmha_quant_mode="bfloat16",
                speculation={"kind": "mtp", "params": {"depth": 2}},
            )
        )


def test_explicit_and_legacy_speculation_cannot_be_combined():
    with pytest.raises(ValueError, match="nextn"):
        RustForwardPassPerfModel.normalize_config(_config(nextn=3, speculation={"kind": "mtp", "params": {"depth": 3}}))


def test_replay_requires_acceptance_in_addition_to_cost_identity():
    from pydantic import ValidationError

    from aisimulate.config.engine import EnginePredictionConfig

    with pytest.raises(ValidationError, match="expected_accepted_tokens"):
        EnginePredictionConfig.model_validate(
            {
                "model": "Qwen/Qwen3-8B",
                "hardware": "h200_sxm",
                "speculation": {"kind": "mtp", "params": {"depth": 3}},
            }
        )
