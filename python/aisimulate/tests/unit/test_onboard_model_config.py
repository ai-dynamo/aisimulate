# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public local-config onboarding, including saved-request replay."""

from __future__ import annotations

import builtins
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import aisimulate.main as cli
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.support.fpm import fpm_cli_args
from aisimulate.support.plan import request_id
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit

# Synthetic geometry and explicit illustrative bounds, not checkpoint measurements.
_CONFIG = {
    "_name_or_path": "example/local-decoder",
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "max_position_embeddings": 32768,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 1024,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
}
_OVERRIDES = {
    "weights_bytes": 2 * 1024**3,
    "activations_bytes": 32 * 1024**2,
    "runtime_overhead_bytes": 16 * 1024**2,
    "comm_overhead_bytes": 16 * 1024**2,
    "kv_bytes_per_token": 4096,
    "cache_layout": "linear",
    "max_num_tokens": 4096,
    "max_batch_size": 8,
    "gemm_quant_mode": "bfloat16",
    "moe_quant_mode": "bfloat16",
    "fmha_quant_mode": "bfloat16",
    "comm_quant_mode": "half",
    "kv_cache_dtype": "bfloat16",
    "provenance": "Synthetic CLI test bounds; not measured.",
}
_IDENTITY = {
    "model_revision": "checkpoint-revision-123",
    "framework_version": "0.25.1",
    "gpu": "h200_sxm",
    "interconnect": "nvswitch",
}
_PILOT = {
    "tensor_parallel": 1,
    "input_tokens": 1024,
    "output_tokens": 128,
    "concurrency": 1,
    "ttft_ms": 1000,
    "tpot_ms": 100,
}


def test_validation_changes_preserve_profile_and_collection_command(tmp_path, capsys):
    source, resources = _files(tmp_path)
    requests = []
    for name, validation in (
        ("small", {}),
        (
            "large",
            {"input_tokens": 4096, "output_tokens": 256, "concurrency": 16, "request_count": 32, "ttft_ms": 5000},
        ),
    ):
        output = tmp_path / f"{name}.yaml"
        assert cli.main(_args(output, source, resources, **validation)) == 0
        requests.append(SupportRequest.from_yaml(output))
    first, second = requests
    assert first.workload != second.workload
    assert first.fpm_profile == second.fpm_profile
    assert request_id(first) == request_id(second)
    assert fpm_cli_args(first, output_dir=tmp_path / "plan", plan_only=True) == fpm_cli_args(
        second, output_dir=tmp_path / "plan", plan_only=True
    )
    assert first.scheduler_limits() == {"max_batched_tokens": 4096, "max_sequences": 8}


@pytest.mark.parametrize("model_context,expected", [(32768, 32768), (262144, 256000)])
def test_model_context_and_reference_cap_define_initial_runtime_context(tmp_path, model_context, expected):
    source, resources = _files(tmp_path, config={**_CONFIG, "max_position_embeddings": model_context})
    output = tmp_path / "request.yaml"
    assert cli.main(_args(output, source, resources)) == 0
    request = SupportRequest.from_yaml(output)
    assert request.search.context_length == expected
    assert request.fpm_profile.context_length == model_context
    command = fpm_cli_args(request, output_dir=tmp_path / "plan", plan_only=True)
    assert command[command.index("--fpm-max-model-len") + 1] == str(expected)
    assert command[command.index("--fpm-max-prefill-isl") + 1] == "4096"


def test_guided_collection_edit_reestimates_resources_before_acceptance(tmp_path, monkeypatch):
    source, resources = _files(
        tmp_path,
        config={**_CONFIG, "hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 8192},
        overrides={"fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", "kv_cache_dtype": "bfloat16"},
    )
    output = tmp_path / "request.yaml"
    prompts = _terminal(
        monkeypatch,
        [
            "edit",
            "max_num_tokens",
            "4096",
            "edit",
            "runtime_context_length",
            "8192",
            "edit",
            "max_prefill_cudagraph_size",
            "512",
            "accept",
        ],
    )
    assert cli.main(_args(output, source, resources, max_num_tokens=16384) + ["--interactive"]) == 0
    request = SupportRequest.from_yaml(output)
    profile = request.profile_deployment()
    assert request.search.context_length == 8192
    assert request.collection.max_num_tokens == profile.resources.max_num_tokens == 4096
    assert request.collection.max_prefill_cudagraph_size == 512
    assert profile.resources.activations_bytes == 2 * 4096 * 4096 * 11
    assert request.scheduler_limits() == {"max_batched_tokens": 4096, "max_sequences": 256}
    assert not any("Input tokens" in prompt or "Concurrent requests" in prompt for prompt in prompts)
    command = fpm_cli_args(request, output_dir=tmp_path / "plan", plan_only=True)
    for option, expected in (
        ("--fpm-max-model-len", "8192"),
        ("--fpm-max-num-batched-tokens", "4096"),
        ("--fpm-max-num-seqs", "256"),
        ("--fpm-max-prefill-cudagraph-size", "512"),
    ):
        assert command[command.index(option) + 1] == expected


def test_conflicting_resource_and_collection_bounds_fail_before_saving(tmp_path):
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"
    with pytest.raises(SystemExit):
        cli.main(_args(output, source, resources, max_num_tokens=8192))
    assert not output.exists()


@pytest.mark.parametrize(
    "options,policy,limit",
    [
        ({}, "runtime", None),
        ({"prefill_cudagraph_policy": "runtime"}, "runtime", None),
        ({"max_prefill_cudagraph_size": 512}, "explicit", 512),
        ({"prefill_cudagraph_policy": "explicit"}, "explicit", 2048),
        ({"prefill_cudagraph_policy": "explicit", "max_prefill_cudagraph_size": 1024}, "explicit", 1024),
    ],
)
def test_new_runtime_policy_is_saved_and_preserved_in_collection_preview(tmp_path, options, policy, limit):
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"
    assert cli.main(_args(output, source, resources, **options)) == 0
    request = SupportRequest.from_yaml(output)
    assert request.collection.prefill_cudagraph_policy == policy
    assert request.collection.gpu_memory_utilization == 0.9
    assert request.collection_settings()["max_prefill_cudagraph_size"] == limit
    root = tmp_path / "plan"
    assert cli.main(["onboard", "plan", "--config", str(output), "--output-dir", str(root)]) == 0
    command = json.loads((root / "commands.json").read_text())["fpm_plan_local"]
    assert command[command.index("--fpm-prefill-cudagraph-policy") + 1] == policy
    assert command[command.index("--fpm-gpu-memory-utilization") + 1] == "0.9"
    if policy == "runtime":
        assert "--fpm-max-prefill-cudagraph-size" not in command
        assert "deferred" in request.collection_settings()["sources"]["max_prefill_cudagraph_size"]


@pytest.mark.parametrize(
    "options",
    [
        {"prefill_cudagraph_policy": "runtime", "max_prefill_cudagraph_size": 512},
        {"gpu_memory_utilization": "nan"},
        {"gpu_memory_utilization": "inf"},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": -0.1},
        {"gpu_memory_utilization": 1.1},
    ],
)
def test_invalid_runtime_settings_leave_request_unchanged(tmp_path, options):
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"
    output.write_text("original request\n")
    with pytest.raises(SystemExit):
        cli.main(_args(output, source, resources, **options) + ["--overwrite"])
    assert output.read_text() == "original request\n"


def test_review_reset_to_runtime_clears_capture_override_and_saves_edited_memory(tmp_path, monkeypatch):
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"
    _terminal(
        monkeypatch,
        [
            "edit",
            "prefill_cudagraph_policy",
            "invalid",
            "runtime",
            "edit",
            "gpu_memory_utilization",
            "nan",
            "1.1",
            "0.75",
            "accept",
        ],
    )
    assert cli.main(_args(output, source, resources, max_prefill_cudagraph_size=1024) + ["--interactive"]) == 0
    request = SupportRequest.from_yaml(output)
    assert request.collection.prefill_cudagraph_policy == "runtime"
    assert request.collection.max_prefill_cudagraph_size is None
    assert request.collection.gpu_memory_utilization == 0.75
    command = fpm_cli_args(request, output_dir=tmp_path / "collection", plan_only=True)
    assert "--fpm-max-prefill-cudagraph-size" not in command
    assert command[command.index("--fpm-gpu-memory-utilization") + 1] == "0.75"


@pytest.mark.parametrize("grouped", [False, True])
def test_memory_fraction_survives_plan_compilation_and_replay_validation(tmp_path, grouped):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.support.validation import _prediction_config

    overrides = dict(_OVERRIDES)
    if grouped:
        overrides.pop("kv_bytes_per_token")
        overrides.pop("cache_layout")
        overrides["cache_block_sizes"] = {"sliding_attention": 16}
    source, resources = _files(
        tmp_path, config={**_CONFIG, **({"sliding_window": 128} if grouped else {})}, overrides=overrides
    )
    output = tmp_path / "request.yaml"
    fraction = 0.73
    assert cli.main(_args(output, source, resources, gpu_memory_utilization=fraction)) == 0
    request = SupportRequest.from_yaml(output)
    root = tmp_path / "plan"
    assert cli.main(["onboard", "plan", "--config", str(output), "--output-dir", str(root)]) == 0
    plan = json.loads((root / "support-plan.json").read_text())
    budget = plan["resources"]
    assert budget["total_kv_size_bytes"] == int(budget["total_gpu_capacity_bytes"] * fraction) - sum(
        budget["memory_breakdown"].values()
    )
    prediction = CorePredictionConfig.from_yaml(root / "predict/pilot.yaml")
    recommendation = CoreRecommendationConfig.from_yaml(root / "recommend/pilot.yaml")
    replay = CorePredictionConfig.model_validate(_prediction_config(request, root, tmp_path / "trace.jsonl"))
    for config in (prediction, recommendation, replay):
        assert config.engine.workers.aggregated.kv_cache.capacity.memory_fraction == fraction
        if grouped:
            assert not config.engine.workers.aggregated.kv_cache.prefix_caching
        # Host RAM controls are independent of the simulated GPU budget.
        assert config.execution.resources.available_memory_fraction == 0.9
    for config in (prediction, replay):
        spec = prediction_to_replay_spec(config)
        args = spec.backend_deployment.agg_engine_args
        assert args["gpu_memory_utilization"] == fraction


def test_communication_override_is_retained_with_collection_compatibility_guidance(tmp_path, monkeypatch, capsys):
    source, resources = _files(tmp_path, overrides={**_OVERRIDES, "comm_quant_mode": "int8"})
    output = tmp_path / "request.yaml"
    _terminal(monkeypatch, ["accept"])
    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0
    deployment = SupportRequest.from_yaml(output).profile_deployment()
    assert deployment.comm_quant_mode == "int8"
    source = json.loads(deployment.resources.provenance)["fields"]["comm_quant_mode"]["source"]
    assert "user override" in source
    assert "collector supports half" in source
    assert "new collection will reject a mismatched identity" in capsys.readouterr().out


@pytest.mark.parametrize(
    "route,limits",
    [
        (route, limits)
        for route in ("identifier", "profile", "config")
        for limits in ({"max_num_tokens": 128, "max_batch_size": 256}, {"max_num_tokens": 128})
    ]
    + [("profile", {}), ("config", {})],
)
def test_invalid_resolved_scheduler_cannot_be_saved_or_planned(tmp_path, capsys, route, limits):
    source, resources = _files(tmp_path, overrides={**_OVERRIDES, "max_num_tokens": 8192, "max_batch_size": 256})
    baseline = tmp_path / "baseline.yaml"
    assert cli.main(_args(baseline, source, resources)) == 0
    payload = yaml.safe_load(baseline.read_text())
    profile = payload["fpm_profile"]
    output = tmp_path / "invalid.yaml"
    if route == "config":
        resources.write_text(yaml.safe_dump({**_OVERRIDES, "max_num_tokens": 128, "max_batch_size": 256}))
        command = _args(output, source, resources, **limits)
    else:
        command = _args(output, source, None, model=_CONFIG["_name_or_path"], model_kind="dense", **limits)
        command.remove(str(source))
        command.remove("--model-config")
        if route == "profile":
            if not limits:
                profile["deployments"][0]["resources"]["max_num_tokens"] = 128
            profile_path = tmp_path / "profile.json"
            profile_path.write_text(json.dumps(profile))
            command.extend(["--fpm-profile", str(profile_path)])

    with pytest.raises(SystemExit) as error:
        cli.main(command)
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "max_num_tokens" in message and "max_batch_size" in message
    assert not output.exists()

    payload["collection"] = limits
    if route == "identifier":
        payload.pop("fpm_profile")
    elif not limits:
        payload["fpm_profile"]["deployments"][0]["resources"]["max_num_tokens"] = 128
    request_path = tmp_path / "invalid-saved.yaml"
    request_path.write_text(yaml.safe_dump(payload))
    original = request_path.read_bytes()
    plan = tmp_path / "invalid-plan"
    with pytest.raises(SystemExit) as error:
        cli.main(["onboard", "plan", "--config", str(request_path), "--output-dir", str(plan)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "max_num_tokens" in message and "max_batch_size" in message
    assert request_path.read_bytes() == original
    assert not plan.exists()


@pytest.mark.parametrize("suggest_parallel", [False, True])
def test_config_small_token_budget_uses_reviewed_sequence_bound(tmp_path, capsys, suggest_parallel):
    overrides = {**_OVERRIDES, "max_num_tokens": 128}
    if suggest_parallel:
        # Automatic topology selection requires shared inputs rather than rank-local byte overrides.
        overrides = {key: value for key, value in overrides.items() if not key.endswith("_bytes")}
        overrides.pop("kv_bytes_per_token")
    source, resources = _files(tmp_path, overrides=overrides)
    output = tmp_path / "request.yaml"
    command = _args(output, source, resources, max_num_tokens=128, tensor_parallel=None if suggest_parallel else 1)
    if suggest_parallel:
        command.append("--suggest-parallel")

    assert cli.main(command) == 0

    if suggest_parallel:
        assert json.loads(capsys.readouterr().out)["default"] is not None
        assert not output.exists()
    else:
        request = SupportRequest.from_yaml(output)
        assert request.scheduler_limits() == {"max_batched_tokens": 128, "max_sequences": 8}
        root = tmp_path / "plan"
        assert cli.main(["onboard", "plan", "--config", str(output), "--output-dir", str(root)]) == 0
        command = fpm_cli_args(request, output_dir=root, plan_only=True)
        assert command[command.index("--fpm-max-num-batched-tokens") + 1] == "128"
        assert command[command.index("--fpm-max-num-seqs") + 1] == "8"


def test_guided_config_intake_needs_no_validation_lengths_concurrency_or_slas(tmp_path, monkeypatch):
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["accept"])
    omitted = dict.fromkeys(("input_tokens", "output_tokens", "concurrency", "ttft_ms", "tpot_ms"))
    assert cli.main(_args(output, source, resources, **omitted) + ["--interactive"]) == 0
    assert prompts == ["Review action (accept/edit/cancel): "]
    saved = SupportRequest.from_yaml(output)
    assert saved.workload.concurrency == 1
    assert saved.scheduler_limits()["max_sequences"] == 8


def test_grouped_review_collects_runtime_blocks_then_edits_pages_before_save(tmp_path, monkeypatch, capsys):
    config = {**_CONFIG, "sliding_window": 128, "layer_types": ["full_attention", "sliding_attention"]}
    overrides = {key: value for key, value in _OVERRIDES.items() if key not in {"kv_bytes_per_token", "cache_layout"}}
    source, resources = _files(tmp_path, config=config, overrides=overrides)
    output = tmp_path / "new" / "request.yaml"
    edited_groups = [
        {
            "name": "full_attention",
            "kind": "attention",
            "num_layers": 1,
            "block_size_tokens": 16,
            "page_size_bytes": 8192,
        },
        {
            "name": "sliding_attention",
            "kind": "attention",
            "num_layers": 1,
            "block_size_tokens": 8,
            "page_size_bytes": 2048,
            "sliding_window": 128,
        },
    ]
    answers = iter(
        [
            '{"full_attention": 16}',
            '{"sliding_attention": 8}',
            "edit",
            "cache_groups",
            json.dumps(edited_groups),
            "accept",
        ]
    )
    prompts = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def answer(prompt):
        assert not output.parent.exists(), "grouped profile was saved before acceptance"
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(builtins, "input", answer)
    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0
    saved = SupportRequest.from_yaml(output)
    cache = saved.profile_deployment().resources
    assert cache.cache_layout == "grouped"
    assert cache.kv_bytes_per_token is None
    assert cache.cache_groups[0].page_size_bytes == 8192
    assert cache.cache_groups[1].sliding_window == 128
    provenance = json.loads(cache.provenance)["fields"]
    assert provenance["cache_block_sizes"]["value"] == {"full_attention": 16, "sliding_attention": 8}
    assert "user override" in provenance["cache_groups"]["source"]
    assert "config geometry reference" in provenance["cache_groups"]["source"]
    assert sum(prompt.startswith("cache_block_sizes") for prompt in prompts) == 2
    assert "minimum packed tensor estimates" in capsys.readouterr().out
    source.unlink()
    resources.unlink()
    plan = tmp_path / "plan"
    assert cli.main(["onboard", "plan", "--config", str(output), "--output-dir", str(plan)]) == 0
    prediction = CorePredictionConfig.from_yaml(plan / "predict/pilot.yaml")
    recommendation = CoreRecommendationConfig.from_yaml(plan / "recommend/pilot.yaml")
    assert prediction.engine.fpm_profile == recommendation.engine.fpm_profile == saved.fpm_profile
    assert not prediction.engine.workers.aggregated.kv_cache.prefix_caching
    assert not recommendation.engine.workers.aggregated.kv_cache.prefix_caching
    planned = json.loads((plan / "support-plan.json").read_text())
    assert planned["resources"]["total_kv_size_tokens"] is None
    assert planned["resources"]["request_peak_cache_bytes"] > 0


def test_grouped_automatic_intake_defers_rank_resources_until_after_topology(tmp_path, monkeypatch):
    source, resources = _files(
        tmp_path,
        config={**_CONFIG, "sliding_window": 128},
        overrides={"fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", "kv_cache_dtype": "bfloat16"},
    )
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["1", '{"sliding_attention": 16}', "accept"])
    assert cli.main(_args(output, source, resources, tensor_parallel=None) + ["--interactive"]) == 0
    assert prompts[0].startswith("Choose topology")
    assert prompts[1].startswith("cache_block_sizes")
    assert SupportRequest.from_yaml(output).profile_deployment().resources.cache_layout == "grouped"


def _files(tmp_path, *, config=None, overrides=None, suffix="yaml"):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(_CONFIG if config is None else config))
    resource = tmp_path / f"resources.{suffix}"
    values = _OVERRIDES if overrides is None else overrides
    resource.write_text(json.dumps(values) if suffix == "json" else yaml.safe_dump(values))
    return source, resource


def _multimodal_config(text=None):
    return {
        "_name_or_path": "example/multimodal-checkpoint",
        "architectures": ["ExampleMultimodalForConditionalGeneration"],
        "model_type": "example_multimodal",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "text_config": dict(_CONFIG if text is None else text),
        "vision_config": {"hidden_size": 512, "num_hidden_layers": 8},
        "audio_config": {"hidden_size": 256, "num_hidden_layers": 4},
    }


def _args(output: Path, source: Path, resource: Path | None = None, **changes) -> list[str]:
    options = {**_IDENTITY, **_PILOT, **changes}
    command = ["onboard", "init", "--model-config", str(source), "--output", str(output)]
    if resource is not None:
        command += ["--resource-overrides", str(resource)]
    return command + [
        part
        for name, value in options.items()
        if value is not None
        for part in ("--" + name.replace("_", "-"), str(value))
    ]


def _terminal(monkeypatch, answers=()):
    prompts = []
    remaining = iter(answers)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def answer(prompt):
        prompts.append(prompt)
        try:
            value = next(remaining)
        except StopIteration:
            pytest.fail(f"unexpected prompt: {prompt}")
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(builtins, "input", answer)
    return prompts


def test_help_documents_local_config_and_override_options(capsys):
    with pytest.raises(SystemExit) as result:
        cli.main(["onboard", "init", "--help"])
    assert result.value.code == 0
    text = capsys.readouterr().out
    assert "--model-config" in text
    assert "--resource-overrides" in text


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_scripted_config_embeds_complete_profile_and_sources(tmp_path, monkeypatch, capsys, suffix):
    _terminal(monkeypatch)
    source, resources = _files(tmp_path, suffix=suffix)
    output = tmp_path / "request.yaml"

    assert cli.main(_args(output, source, resources)) == 0

    request = SupportRequest.from_yaml(output)
    profile = request.fpm_profile
    assert request.identity.model == _CONFIG["_name_or_path"]
    assert request.identity.model_kind == "dense"
    assert request.identity.model_revision == _IDENTITY["model_revision"]
    assert profile.architecture == "LlamaForCausalLM"
    assert profile.context_length == 32768
    assert request.search.context_length == 32768
    assert profile.num_experts == 0
    assert profile.deployments[0].resources.weights_bytes == _OVERRIDES["weights_bytes"]
    provenance = profile.provenance + profile.deployments[0].resources.provenance
    assert hashlib.sha256(source.read_bytes()).hexdigest() in provenance
    assert "weights_bytes" in provenance
    assert "override" in provenance.lower()
    assert _OVERRIDES["provenance"] in provenance
    transcript = capsys.readouterr().out
    assert "weights_bytes" in transcript
    assert "source" in transcript.lower()
    assert "estimate" in transcript.lower()


def test_explicit_identity_and_pilot_context_override_config_hints(tmp_path, monkeypatch):
    _terminal(monkeypatch)
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"

    assert cli.main(_args(output, source, resources, model="deployment/model-alias", context_length=2048)) == 0

    request = SupportRequest.from_yaml(output)
    assert request.identity.model == request.fpm_profile.model == "deployment/model-alias"
    assert request.search.context_length == 2048
    assert request.fpm_profile.context_length == 32768


@pytest.mark.parametrize("multimodal", [False, True])
def test_guided_review_requires_explicit_accept_before_creating_output(tmp_path, monkeypatch, capsys, multimodal):
    source, resources = _files(tmp_path, config=_multimodal_config() if multimodal else None)
    output = tmp_path / "new" / "request.yaml"
    prompts = []
    answers = iter(["", "save", "accept"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def respond(prompt):
        prompts.append(prompt)
        assert not output.parent.exists(), "setup wrote output before explicit acceptance"
        return next(answers)

    monkeypatch.setattr(builtins, "input", respond)

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert prompts == ["Review action (accept/edit/cancel): "] * 3
    transcript = capsys.readouterr().out
    assert "Review FPM profile before saving:" in transcript
    review = transcript.split("Review FPM profile before saving:", 1)[1]
    assert "Text decoder only" in review
    assert "encoders, projectors, preprocessing" in review
    assert "Full multimodal deployment memory and latency are not modeled" in review
    assert "weights_bytes: 2147483648" in transcript
    assert "source: user override" in transcript
    assert "per rank" in transcript
    assert "checkpoint-revision-123" in transcript
    assert "h200_sxm" in transcript
    assert SupportRequest.from_yaml(output).profile_deployment().resources.weights_bytes == 2 * 1024**3


def test_guided_review_edits_file_values_and_provenance_then_replays_saved_profile(tmp_path, monkeypatch, capsys):
    source, resources = _files(tmp_path)
    original_overrides = resources.read_bytes()
    output = tmp_path / "request.yaml"
    note = "Synthetic replacement bounds for CLI validation; not measured."
    prompts = _terminal(
        monkeypatch,
        [
            "edit",
            "not_a_field",
            "weights_bytes",
            "many",
            "0.1 B",
            "-1",
            "3 GiB",
            "edit",
            "weights_bytes",
            "4 GiB",
            "edit",
            "kv_cache_dtype",
            "auto",
            "fp8",
            "edit",
            "max_num_tokens",
            "16384",
            "edit",
            "provenance",
            note,
            "accept",
        ],
    )

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    request = SupportRequest.from_yaml(output)
    deployment = request.profile_deployment()
    assert deployment.resources.weights_bytes == 4 * 1024**3
    assert deployment.kv_cache_dtype == "fp8"
    assert deployment.resources.max_num_tokens == 16384
    # Explicit bounds remain user declarations when their dependencies change.
    assert deployment.resources.kv_bytes_per_token == _OVERRIDES["kv_bytes_per_token"]
    assert deployment.resources.activations_bytes == _OVERRIDES["activations_bytes"]
    provenance = json.loads(request.fpm_profile.provenance)["fields"]
    assert provenance["weights_bytes"]["value"] == 4 * 1024**3
    assert provenance["provenance"]["value"] == note
    assert all("user override" in provenance[field]["source"] for field in ("weights_bytes", "kv_bytes_per_token"))
    assert resources.read_bytes() == original_overrides
    assert sum(prompt.startswith("weights_bytes") for prompt in prompts) == 5
    transcript = capsys.readouterr().out
    assert "Choose an editable field" in transcript
    assert "whole number of bytes" in transcript
    review = transcript.rsplit("Review FPM profile before saving:", 1)[1]
    assert "weights_bytes: 4294967296" in review
    assert "kv_cache_dtype: fp8" in review
    assert note in review
    assert _OVERRIDES["provenance"] not in review

    source.unlink()
    resources.unlink()
    plan = tmp_path / "plan"
    assert cli.main(["onboard", "plan", "--config", str(output), "--output-dir", str(plan)]) == 0
    prediction = CorePredictionConfig.from_yaml(plan / "predict/pilot.yaml")
    recommendation = CoreRecommendationConfig.from_yaml(plan / "recommend/pilot.yaml")
    assert prediction.engine.fpm_profile == recommendation.engine.fpm_profile == request.fpm_profile
    assert json.loads((plan / "fpm-model-profile.json").read_text()) == request.fpm_profile.model_dump(mode="json")


def test_guided_review_recomputes_inferred_kv_and_activation_bounds(tmp_path, monkeypatch, capsys):
    config = {
        **_CONFIG,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
    }
    source, resources = _files(
        tmp_path,
        config=config,
        overrides={"fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", "kv_cache_dtype": "bfloat16"},
    )
    output = tmp_path / "request.yaml"
    _terminal(
        monkeypatch,
        ["edit", "kv_cache_dtype", "fp8", "edit", "max_num_tokens", "4096", "edit", "max_batch_size", "8", "accept"],
    )

    assert cli.main(_args(output, source, resources, tensor_parallel=4, concurrency=4) + ["--interactive"]) == 0

    request = SupportRequest.from_yaml(output)
    deployment = request.profile_deployment()
    assert deployment.parallel_tuple == (4, 1, 1, 1, 1, 1)
    assert deployment.resources.kv_bytes_per_token == 16384  # 2 K/V * 32 layers * 2 local heads * 128 * 1 byte
    assert deployment.resources.activations_bytes == 167772160  # 2 bytes * 4096 tokens * 4096 width * 5
    assert deployment.resources.max_num_tokens == 4096
    assert deployment.resources.max_batch_size == 8
    provenance = json.loads(request.fpm_profile.provenance)["fields"]
    assert "exact linear tensor geometry" in provenance["kv_bytes_per_token"]["source"]
    assert "estimate:" in provenance["activations_bytes"]["source"]
    assert "max_num_tokens=4096" in provenance["activations_bytes"]["source"]
    assert "max_batch_size=8" in provenance["activations_bytes"]["source"]
    transcript = capsys.readouterr().out
    assert "kv_bytes_per_token: 32768" in transcript
    assert "activations_bytes: 335544320" in transcript
    review = transcript.rsplit("Review FPM profile before saving:", 1)[1]
    assert "kv_bytes_per_token: 16384" in review
    assert "activations_bytes: 167772160" in review


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("weights_bytes", str(2**53), "total non-KV resource bytes"),
        ("architecture", "UnknownForCausalLM", "conflicts with the source config"),
        ("num_experts", "4", "conflicts with the source config"),
        ("context_length", "65536", "exceeds the source config context limit"),
        ("context_length", "8192", "search.context_length exceeds"),
    ],
)
def test_guided_review_rejects_inconsistent_edits_and_retains_complete_prior_request(
    tmp_path, monkeypatch, capsys, field, value, error
):
    source, resources = _files(tmp_path)
    baseline = tmp_path / "baseline.yaml"
    assert cli.main(_args(baseline, source, resources)) == 0
    output = tmp_path / "request.yaml"
    _terminal(monkeypatch, ["edit", field, value, "accept"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert SupportRequest.from_yaml(output) == SupportRequest.from_yaml(baseline)
    transcript = capsys.readouterr().out
    assert "Edit rejected:" in transcript
    assert error in transcript
    assert "Previous profile values retained" in transcript


def test_guided_review_collects_new_dependency_then_reviews_it_before_accepting(tmp_path, monkeypatch, capsys):
    source, resources = _files(
        tmp_path, overrides={name: value for name, value in _OVERRIDES.items() if name != "activations_bytes"}
    )
    output = tmp_path / "new" / "request.yaml"
    prompts = []
    answers = iter(["edit", "fmha_quant_mode", "fp8", "2 GiB", "accept"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def respond(prompt):
        assert not output.parent.exists(), "staged dependency must not write a partial request"
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(builtins, "input", respond)

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert len(prompts) == 5
    assert prompts[3].startswith("activations_bytes")
    assert prompts[-1] == "Review action (accept/edit/cancel): "
    deployment = SupportRequest.from_yaml(output).profile_deployment()
    assert deployment.fmha_quant_mode == "fp8"
    assert deployment.resources.activations_bytes == 2 * 1024**3
    review = capsys.readouterr().out.rsplit("Review FPM profile before saving:", 1)[1]
    assert "fmha_quant_mode: fp8" in review
    assert "activations_bytes: 2147483648 (source: user override" in review


def test_guided_review_rolls_back_edit_and_new_dependency_together_when_invalid(tmp_path, monkeypatch, capsys):
    overrides = {name: value for name, value in _OVERRIDES.items() if name != "activations_bytes"}
    overrides["weights_bytes"] = (
        2**53 - 2 * 1024**3 - overrides["runtime_overhead_bytes"] - overrides["comm_overhead_bytes"]
    )
    source, resources = _files(tmp_path, overrides=overrides)
    baseline = tmp_path / "baseline.yaml"
    assert cli.main(_args(baseline, source, resources)) == 0
    output = tmp_path / "request.yaml"
    _terminal(monkeypatch, ["edit", "fmha_quant_mode", "fp8", "3 GiB", "accept"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert SupportRequest.from_yaml(output) == SupportRequest.from_yaml(baseline)
    assert "total non-KV resource bytes" in capsys.readouterr().out


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "answers",
    [["cancel"], ["edit", "weights_bytes", "3 GiB", "cancel"]],
    ids=["review", "after-edit"],
)
def test_guided_review_explicit_cancel_leaves_output_untouched(tmp_path, monkeypatch, existing, answers):
    source, resources = _files(tmp_path)
    output = tmp_path / "new" / "request.yaml"
    if existing:
        output.parent.mkdir()
        output.write_text("preserve previous request\n")
    _terminal(monkeypatch, answers)

    assert cli.main(_args(output, source, resources) + ["--interactive", "--overwrite"]) == 130

    if existing:
        assert output.read_text() == "preserve previous request\n"
        assert list(output.parent.iterdir()) == [output]
    else:
        assert not output.parent.exists()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "answers",
    [[], ["edit"], ["edit", "weights_bytes"], ["edit", "fmha_quant_mode", "fp8"]],
    ids=["review", "field", "value", "dependency"],
)
def test_guided_review_interruption_leaves_output_untouched(tmp_path, monkeypatch, interruption, existing, answers):
    source, resources = _files(
        tmp_path, overrides={name: value for name, value in _OVERRIDES.items() if name != "activations_bytes"}
    )
    output = tmp_path / "new" / "request.yaml"
    if existing:
        output.parent.mkdir()
        output.write_text("preserve previous request\n")
    _terminal(monkeypatch, [*answers, interruption()])

    assert cli.main(_args(output, source, resources) + ["--interactive", "--overwrite"]) == 130

    if existing:
        assert output.read_text() == "preserve previous request\n"
        assert list(output.parent.iterdir()) == [output]
    else:
        assert not output.parent.exists()


def test_scripted_identity_failure_lists_all_missing_identity_without_input(tmp_path, monkeypatch, capsys):
    _terminal(monkeypatch)
    source, _ = _files(tmp_path)
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(["onboard", "init", "--model-config", str(source), "--output", str(output)])

    assert result.value.code == 2
    errors = capsys.readouterr().err
    for option in ("--model-revision", "--framework-version", "--gpu", "--interconnect"):
        assert option in errors
    assert "--resource-overrides" in errors
    assert "identity" in errors.lower()
    assert not output.parent.exists()


def test_scripted_profile_failure_lists_every_unresolved_field_without_input(tmp_path, monkeypatch, capsys):
    _terminal(monkeypatch)
    unknown = {"_name_or_path": "example/unknown", "architectures": ["UnknownForCausalLM"]}
    source, _ = _files(tmp_path, config=unknown)
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(_args(output, source, model_kind="dense", context_length=4096))

    assert result.value.code == 2
    errors = capsys.readouterr().err
    for field in (
        "context_length",
        "num_experts",
        "weights_bytes",
        "activations_bytes",
        "kv_bytes_per_token",
        "gemm_quant_mode",
        "moe_quant_mode",
        "fmha_quant_mode",
        "kv_cache_dtype",
    ):
        assert field in errors
    assert "--resource-overrides" in errors
    assert "JSON" in errors and "YAML" in errors
    assert not output.parent.exists()


def test_guided_config_asks_only_unresolved_inputs_and_recovers_invalid_answers(tmp_path, monkeypatch, capsys):
    overrides = {key: value for key, value in _OVERRIDES.items() if key != "fmha_quant_mode"}
    source, resources = _files(tmp_path, overrides=overrides)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["invalid-precision", "bfloat16", "accept"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert len(prompts) == 3
    assert all("fmha_quant_mode" in prompt for prompt in prompts[:-1])
    assert SupportRequest.from_yaml(output).fpm_profile.deployments[0].fmha_quant_mode == "bfloat16"
    assert "invalid-precision" in capsys.readouterr().out


@pytest.mark.parametrize("option", ["model_revision", "framework_version"])
@pytest.mark.parametrize("missing_precision", [False, True])
def test_guided_final_profile_identity_correction_preserves_resource_answers(
    tmp_path, monkeypatch, option, missing_precision
):
    overrides = {key: value for key, value in _OVERRIDES.items() if not missing_precision or key != "fmha_quant_mode"}
    source, resources = _files(tmp_path, overrides=overrides)
    output = tmp_path / "request.yaml"
    answers = (["bfloat16"] if missing_precision else []) + [_IDENTITY[option], "accept"]
    prompts = _terminal(monkeypatch, answers)

    assert cli.main(_args(output, source, resources, **{option: "unknown"}) + ["--interactive"]) == 0

    assert len(prompts) == len(answers)
    if missing_precision:
        assert "fmha_quant_mode" in prompts[0]
    assert ("Pinned model revision" if option == "model_revision" else "Pinned vLLM version") in prompts[-2]
    request = SupportRequest.from_yaml(output)
    assert getattr(request.identity, option) == _IDENTITY[option]
    assert request.profile_deployment().fmha_quant_mode == "bfloat16"
    assert request.profile_deployment().resources.weights_bytes == _OVERRIDES["weights_bytes"]


@pytest.mark.parametrize("missing_precision", [False, True])
def test_guided_composite_resource_error_does_not_repeat_unrelated_precision_prompt(
    tmp_path, monkeypatch, capsys, missing_precision
):
    overrides = {
        key: value
        for key, value in _OVERRIDES.items()
        if key != "activations_bytes" and (not missing_precision or key != "fmha_quant_mode")
    }
    overrides["weights_bytes"] = 2**53 - overrides["runtime_overhead_bytes"] - overrides["comm_overhead_bytes"]
    source, resources = _files(tmp_path, overrides=overrides)
    output = tmp_path / "new" / "request.yaml"
    prompts = _terminal(monkeypatch, ["bfloat16"] if missing_precision else [])

    with pytest.raises(SystemExit) as result:
        cli.main(_args(output, source, resources) + ["--interactive"])

    assert result.value.code == 2
    assert "total non-KV resource bytes must not exceed 2**53" in capsys.readouterr().err
    assert len(prompts) == int(missing_precision)
    if missing_precision:
        assert "fmha_quant_mode" in prompts[0]
    assert not output.parent.exists()


def test_scripted_late_dense_architecture_rejects_moe_config_without_output(tmp_path, monkeypatch, capsys):
    _terminal(monkeypatch)
    config = {key: value for key, value in _CONFIG.items() if key not in ("architectures", "model_type")}
    config["num_experts"] = 4
    source, resources = _files(tmp_path, config=config, overrides={**_OVERRIDES, "architecture": "LlamaForCausalLM"})
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(_args(output, source, resources))

    assert result.value.code == 2
    assert "dense architecture" in capsys.readouterr().err
    assert not output.parent.exists()


@pytest.mark.parametrize("field", ["num_experts_per_tok", "first_k_dense_replace"])
def test_real_cli_accepts_optional_null_moe_metadata_with_late_expert_count(tmp_path, field):
    config = {**_CONFIG, "architectures": ["MixtralForCausalLM"], "model_type": "mixtral", field: None}
    source, resources = _files(tmp_path, config=config, overrides={**_OVERRIDES, "num_experts": 4})
    output = tmp_path / "request.yaml"

    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *_args(output, source, resources)],
        input="",
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    request = SupportRequest.from_yaml(output)
    assert request.identity.model_kind == "moe"
    assert request.fpm_profile.num_experts == 4
    assert request.profile_deployment().resources.weights_bytes == _OVERRIDES["weights_bytes"]


def test_real_cli_multimodal_intake_preserves_text_scope_and_original_source(tmp_path):
    source, resources = _files(tmp_path, config=_multimodal_config())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "request.yaml"
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *_args(output, source, resources)],
        input="",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Text decoder only" in result.stdout
    assert "encoders, projectors, preprocessing" in result.stdout
    assert "Full multimodal deployment memory and latency are not modeled" in result.stdout
    assert "Review action" not in result.stdout
    request = SupportRequest.from_yaml(output)
    assert request.identity.model == "example/multimodal-checkpoint"
    assert request.fpm_profile.architecture == "LlamaForCausalLM"
    provenance = json.loads(request.fpm_profile.provenance)
    assert provenance["config_sha256"] == source_hash
    assert "Text decoder only" in provenance["config_notes"]["modeling_scope"]
    assert "text_config" in provenance["config_notes"]["decoder_config"]
    assert provenance["config_notes"]["hidden_size"] == "config hidden_size=128"


def test_real_cli_unknown_multimodal_decoder_lists_missing_resource_bounds_without_writing(tmp_path):
    text = {key: value for key, value in _CONFIG.items() if key not in ("architectures", "model_type")}
    text["num_experts"] = 0
    source, _ = _files(tmp_path, config=_multimodal_config(text))
    output = tmp_path / "new" / "request.yaml"
    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *_args(output, source)],
        input="",
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "Text decoder only" in result.stdout
    for field in ("weights_bytes", "activations_bytes", "kv_bytes_per_token", "cache_layout"):
        assert field in result.stderr
    assert "--resource-overrides" in result.stderr
    assert "unsupported encoder, multimodal" not in result.stderr
    assert not output.parent.exists()


def test_bundled_modelopt_string_cache_config_creates_complete_request(tmp_path, monkeypatch):
    _terminal(monkeypatch)
    config_path = (
        Path(__file__).parents[2] / "src/aisimulate_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_config.json"
    )
    source, resources = _files(
        tmp_path,
        config=json.loads(config_path.read_text()),
        overrides={"weights_bytes": 40 * 1024**3, "fmha_quant_mode": "bfloat16", "comm_quant_mode": "half"},
    )
    output = tmp_path / "request.yaml"

    assert cli.main(_args(output, source, resources, model="review/qwen3-config")) == 0

    profile = SupportRequest.from_yaml(output).fpm_profile
    assert profile.architecture == "Qwen3ForCausalLM"
    assert profile.num_experts == 0
    deployment = profile.deployments[0]
    assert deployment.gemm_quant_mode == "fp8_static"
    assert deployment.moe_quant_mode == deployment.kv_cache_dtype == "fp8"
    assert deployment.fmha_quant_mode == "bfloat16"
    assert deployment.resources.kv_bytes_per_token == 2 * 64 * 8 * 128


def test_guided_missing_identity_is_asked_before_profile_fields(tmp_path, monkeypatch):
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["checkpoint-revision-123", "0.25.1", "h200_sxm", "nvswitch", "accept"])

    assert cli.main(_args(output, source, resources, **dict.fromkeys(_IDENTITY)) + ["--interactive"]) == 0

    assert len(prompts) == 5
    assert "Pinned model revision" in prompts[0]
    assert not any("Model name" in prompt or "Model kind" in prompt for prompt in prompts)


def test_guided_unknown_model_metadata_bounds_pilot_before_ordinary_prompts(tmp_path, monkeypatch):
    config = {**_CONFIG, "architectures": ["UnknownForCausalLM"], "model_type": "unknown"}
    del config["max_position_embeddings"]
    source, resources = _files(tmp_path, config=config)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["invalid", "2048", "-1", "0", "accept"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert len(prompts) == 5
    assert all("context_length" in prompt for prompt in prompts[:2])
    assert all("num_experts" in prompt for prompt in prompts[2:-1])
    request = SupportRequest.from_yaml(output)
    assert request.search.context_length == request.fpm_profile.context_length == 2048
    assert request.identity.model_kind == "dense"


@pytest.mark.parametrize(
    "changes,answers",
    [
        ({"model_kind": "moe"}, ["dense"]),
        ({"context_length": 65536}, ["8192"]),
        ({"tensor_parallel": 8}, ["4"]),
        ({"model_revision": "unknown"}, ["checkpoint-revision-123"]),
    ],
)
def test_guided_setup_can_correct_request_conflicts_with_config_metadata(tmp_path, monkeypatch, changes, answers):
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, [*answers, "accept"])

    assert cli.main(_args(output, source, resources, **changes) + ["--interactive"]) == 0

    assert len(prompts) == len(answers) + 1
    request = SupportRequest.from_yaml(output)
    assert request.identity.model_kind == "dense"
    assert request.identity.model_revision == "checkpoint-revision-123"
    assert request.search.context_length <= request.fpm_profile.context_length
    assert request.search.tensor_parallel <= 4


def test_guided_precision_answers_resolve_supported_memory_without_redundant_prompts(tmp_path, monkeypatch):
    source, _ = _files(tmp_path)
    output = tmp_path / "request.yaml"
    prompts = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = {"fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", "kv_cache_dtype": "bfloat16"}

    def respond(prompt):
        prompts.append(prompt)
        if prompt == "Review action (accept/edit/cancel): ":
            return "accept"
        name = prompt.split(" ")[0]
        assert name in answers, f"unexpected prompt for a derived input: {prompt}"
        return answers[name]

    monkeypatch.setattr(builtins, "input", respond)

    assert cli.main(_args(output, source) + ["--interactive"]) == 0

    assert len(prompts) == 3
    resources = SupportRequest.from_yaml(output).profile_deployment().resources
    assert resources.kv_bytes_per_token == 512
    assert resources.max_num_tokens == 8192
    assert resources.max_batch_size == 256


def test_guided_memory_accepts_exact_units_and_reprompts_invalid_quantities(tmp_path, monkeypatch, capsys):
    config = {**_CONFIG, "architectures": ["UnknownForCausalLM"], "model_type": "unknown", "num_local_experts": 0}
    overrides = {key: value for key, value in _OVERRIDES.items() if key != "weights_bytes"}
    source, resources = _files(tmp_path, config=config, overrides=overrides)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["many", "0.1 B", "-1", "2 GiB", "accept"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert len(prompts) == 5
    assert all("weights_bytes" in prompt and "per rank" in prompt for prompt in prompts[:-1])
    assert SupportRequest.from_yaml(output).profile_deployment().resources.weights_bytes == 2 * 1024**3
    assert "whole number of bytes" in capsys.readouterr().out


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
@pytest.mark.parametrize("existing", [False, True])
def test_resource_prompt_cancellation_writes_no_partial_request(tmp_path, monkeypatch, interruption, existing):
    overrides = {key: value for key, value in _OVERRIDES.items() if key != "fmha_quant_mode"}
    source, resources = _files(tmp_path, overrides=overrides)
    output = tmp_path / "new" / "request.yaml"
    if existing:
        output.parent.mkdir()
        output.write_text("preserve previous request\n")
    _terminal(monkeypatch, [interruption()])

    assert cli.main(_args(output, source, resources) + ["--interactive", "--overwrite"]) == 130

    if existing:
        assert output.read_text() == "preserve previous request\n"
        assert list(output.parent.iterdir()) == [output]
    else:
        assert not output.parent.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"weights_bytes": -1},
        {"weights_bytes": True},
        {"weights_bytes": 1.5},
        {"weights_bytes": "2 GiB"},
        {"fmha_quant_mode": "auto"},
        {"not_a_profile_field": 1},
        {"resources": {"weights_bytes": 1}},
        {"max_num_tokens": 0},
        {"cache_layout": "recurrent"},
    ],
)
def test_invalid_overrides_fail_before_prompts_or_output(tmp_path, monkeypatch, overrides):
    _terminal(monkeypatch)
    source, resources = _files(tmp_path, overrides=overrides)
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(_args(output, source, resources) + ["--interactive"])

    assert result.value.code == 2
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "contents", ['{"weights_bytes": 1, "weights_bytes": 2}', "weights_bytes: 1\nweights_bytes: 2\n"]
)
def test_duplicate_override_fields_are_rejected_instead_of_discarded(tmp_path, monkeypatch, capsys, contents):
    _terminal(monkeypatch)
    source, resources = _files(tmp_path)
    resources.write_text(contents)
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(_args(output, source, resources) + ["--interactive"])

    assert result.value.code == 2
    assert "duplicate resource override" in capsys.readouterr().err
    assert not output.parent.exists()


def test_config_interactive_requires_a_terminal_without_reading_input(tmp_path, monkeypatch, capsys):
    _terminal(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    source, resources = _files(tmp_path)
    output = tmp_path / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(_args(output, source, resources) + ["--interactive"])

    assert result.value.code == 2
    assert "requires a terminal" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("contents", ["{broken", "[]", "model_type: llama\n"])
def test_model_config_requires_a_local_json_mapping(tmp_path, monkeypatch, contents):
    _terminal(monkeypatch)
    source = tmp_path / "config.json"
    source.write_text(contents)
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(_args(output, source) + ["--interactive"])

    assert result.value.code == 2
    assert not output.parent.exists()


@pytest.mark.parametrize("cache_type", [[], {}, True, 8, 1.5])
@pytest.mark.parametrize("existing", [False, True])
def test_real_cli_rejects_malformed_cache_type_without_writing_request(tmp_path, cache_type, existing):
    config = {**_CONFIG, "quantization_config": {"kv_cache_scheme": {"num_bits": 8, "type": cache_type}}}
    source, resources = _files(tmp_path, config=config)
    output = tmp_path / "new" / "request.yaml"
    if existing:
        output.parent.mkdir()
        output.write_text("preserve previous request\n")

    result = subprocess.run(
        [sys.executable, "-m", "aisimulate", *_args(output, source, resources), "--overwrite"],
        input="",
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 2, result.stderr
    assert "quantization_config.kv_cache_scheme.type" in result.stderr
    assert "Traceback" not in result.stderr
    if existing:
        assert output.read_text() == "preserve previous request\n"
        assert list(output.parent.iterdir()) == [output]
    else:
        assert not output.parent.exists()


@pytest.mark.parametrize(
    "options,expected",
    [
        (["--resource-overrides", "missing.yaml"], "requires --model-config"),
        (["--model-config", "missing.json", "--fpm-profile", "missing.yaml"], "not allowed"),
    ],
)
def test_config_route_option_relationships_are_rejected_before_reading_files(
    tmp_path, monkeypatch, capsys, options, expected
):
    _terminal(monkeypatch)
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(["onboard", "init", *options, "--output", str(output), "--interactive"])

    assert result.value.code == 2
    assert expected in capsys.readouterr().err
    assert not output.parent.exists()


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize(
    "tp,dp,moe_tp,moe_ep,preset",
    [(4, 1, 4, 1, "pure_tp"), (1, 8, 1, 8, "dep"), (8, 1, 1, 8, "tep")],
)
def test_config_init_to_plan_replay_preserves_full_topology_without_source_files(
    tmp_path, monkeypatch, capsys, tp, dp, moe_tp, moe_ep, preset, multimodal, interactive
):
    config = {
        **_CONFIG,
        "architectures": ["ExampleMoeForCausalLM"],
        "model_type": "example_moe",
        "num_local_experts": 8,
    }
    if multimodal:
        config = _multimodal_config(config)
    source, resources = _files(tmp_path, config=config)
    output = tmp_path / "request.yaml"
    real_import = builtins.__import__

    def no_model_or_download_import(name, *args, **kwargs):
        assert not name.startswith(("collector", "huggingface_hub", "transformers")), name
        assert ".sdk.models" not in name, name
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_or_download_import)
    prompts = _terminal(monkeypatch, ["accept"] if interactive else [])
    assert (
        cli.main(
            _args(
                output,
                source,
                resources,
                tensor_parallel=tp,
                attention_data_parallel=dp,
                moe_tensor_parallel=moe_tp,
                moe_expert_parallel=moe_ep,
            )
            + (["--interactive"] if interactive else [])
        )
        == 0
    )
    source.unlink()
    resources.unlink()
    request = SupportRequest.from_yaml(output)
    assert prompts == (["Review action (accept/edit/cancel): "] if interactive else [])
    assert "Collection GPUs required: " + str(tp * dp) in capsys.readouterr().out
    assert not {"gpu_count", "node_count", "gpus_per_node"} & request.identity.model_dump().keys()
    assert "max_candidates" not in request.search.model_dump()
    assert "Text decoder only" in json.loads(request.fpm_profile.provenance)["config_notes"]["modeling_scope"]
    assert request.parallel_preset == preset
    assert request.profile_deployment().parallel_tuple == (tp, 1, dp, moe_tp, moe_ep, 1)
    plan = tmp_path / "plan"
    assert cli.main(["onboard", "plan", "--config", str(output), "--output-dir", str(plan)]) == 0
    saved_plan = json.loads((plan / "support-plan.json").read_text())
    assert saved_plan["fpm"]["collection_gpus_required"] == tp * dp
    command = saved_plan["fpm"]["plan_command"]
    assert command[command.index("--fpm-gpu-counts") + 1] == str(tp * dp)
    assert command[command.index("--fpm-parallel-presets") + 1] == preset
    assert SupportRequest.from_yaml(plan / "request.yaml") == request
    prediction = CorePredictionConfig.from_yaml(plan / "predict/pilot.yaml")
    recommendation = CoreRecommendationConfig.from_yaml(plan / "recommend/pilot.yaml")
    assert recommendation.optimization.constraints.max_candidate_gpus == tp * dp
    assert len(recommendation.engine.workers.aggregated.parallelism.preset) == 1
    assert recommendation.engine.workers.aggregated.parallelism.preset[0].replicas == 1
    assert prediction.engine.fpm_profile == recommendation.engine.fpm_profile == request.fpm_profile
    for generated, path in (
        (prediction, plan / "predict/pilot.yaml"),
        (recommendation, plan / "recommend/pilot.yaml"),
    ):
        assert generated.engine.systems_paths == [str(plan / "systems")]
        timing = generated.engine.workers.aggregated.timing
        assert timing.estimation_mode == "fpm_interpolation"
        assert timing.fallback_policy == "deny"
        assert timing.estimator_config["fpm_interpolation"]["method"] == "direct"
        saved_engine = yaml.safe_load(path.read_text())["engine"]
        assert saved_engine["systems_paths"] == generated.engine.systems_paths
        assert "systems_path" not in saved_engine
        assert saved_engine["fpm_profile"] == request.fpm_profile.model_dump(mode="json")
        saved_timing = saved_engine["workers"]["aggregated"]["timing"]
        assert saved_timing == {
            "type": "default",
            "estimation_mode": "fpm_interpolation",
            "fallback_policy": "deny",
            "estimator_config": {"fpm_interpolation": {"method": "direct"}},
        }
    assert prediction.engine.workers.aggregated.parallelism.tensor == tp
    assert prediction.engine.workers.aggregated.parallelism.attention_data == dp
    assert prediction.engine.workers.aggregated.parallelism.moe_tensor == moe_tp
    assert prediction.engine.workers.aggregated.parallelism.moe_expert == moe_ep
    assert json.loads((plan / "fpm-model-profile.json").read_text()) == request.fpm_profile.model_dump(mode="json")
