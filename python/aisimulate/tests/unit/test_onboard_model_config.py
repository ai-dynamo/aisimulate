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
    "gpu_count": 8,
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


def _files(tmp_path, *, config=None, overrides=None, suffix="yaml"):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(_CONFIG if config is None else config))
    resource = tmp_path / f"resources.{suffix}"
    values = _OVERRIDES if overrides is None else overrides
    resource.write_text(json.dumps(values) if suffix == "json" else yaml.safe_dump(values))
    return source, resource


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
    assert request.search.context_length == 16384
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


def test_scripted_identity_failure_lists_all_missing_identity_without_input(tmp_path, monkeypatch, capsys):
    _terminal(monkeypatch)
    source, _ = _files(tmp_path)
    output = tmp_path / "new" / "request.yaml"

    with pytest.raises(SystemExit) as result:
        cli.main(["onboard", "init", "--model-config", str(source), "--output", str(output)])

    assert result.value.code == 2
    errors = capsys.readouterr().err
    for option in ("--model-revision", "--framework-version", "--gpu", "--gpu-count", "--interconnect"):
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
        "comm_quant_mode",
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
    prompts = _terminal(monkeypatch, ["invalid-precision", "bfloat16"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert len(prompts) == 2
    assert all("fmha_quant_mode" in prompt for prompt in prompts)
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
    answers = (["bfloat16"] if missing_precision else []) + [_IDENTITY[option]]
    prompts = _terminal(monkeypatch, answers)

    assert cli.main(_args(output, source, resources, **{option: "unknown"}) + ["--interactive"]) == 0

    assert len(prompts) == len(answers)
    if missing_precision:
        assert "fmha_quant_mode" in prompts[0]
    assert ("Pinned model revision" if option == "model_revision" else "Pinned vLLM version") in prompts[-1]
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


def test_bundled_modelopt_string_cache_config_creates_complete_request(tmp_path, monkeypatch):
    _terminal(monkeypatch)
    config_path = (
        Path(__file__).parents[2]
        / "src/aiconfigurator_core/model_configs/Qwen--Qwen3-32B-FP8-Static-PerTensor_config.json"
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
    prompts = _terminal(monkeypatch, ["checkpoint-revision-123", "0.25.1", "h200_sxm", "8", "nvswitch"])

    assert cli.main(_args(output, source, resources, **dict.fromkeys(_IDENTITY)) + ["--interactive"]) == 0

    assert len(prompts) == 5
    assert "Pinned model revision" in prompts[0]
    assert not any("Model name" in prompt or "Model kind" in prompt for prompt in prompts)


def test_guided_unknown_model_metadata_bounds_pilot_before_ordinary_prompts(tmp_path, monkeypatch):
    config = {**_CONFIG, "architectures": ["UnknownForCausalLM"], "model_type": "unknown"}
    del config["max_position_embeddings"]
    source, resources = _files(tmp_path, config=config)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["invalid", "2048", "-1", "0"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert len(prompts) == 4
    assert all("context_length" in prompt for prompt in prompts[:2])
    assert all("num_experts" in prompt for prompt in prompts[2:])
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
    prompts = _terminal(monkeypatch, answers)

    assert cli.main(_args(output, source, resources, **changes) + ["--interactive"]) == 0

    assert len(prompts) == len(answers)
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
        name = prompt.split(" ")[0]
        assert name in answers, f"unexpected prompt for a derived input: {prompt}"
        return answers[name]

    monkeypatch.setattr(builtins, "input", respond)

    assert cli.main(_args(output, source) + ["--interactive"]) == 0

    assert len(prompts) == 3
    resources = SupportRequest.from_yaml(output).profile_deployment().resources
    assert resources.kv_bytes_per_token == 512
    assert resources.max_num_tokens == 8192
    assert resources.max_batch_size == 1


def test_guided_memory_accepts_exact_units_and_reprompts_invalid_quantities(tmp_path, monkeypatch, capsys):
    config = {**_CONFIG, "architectures": ["UnknownForCausalLM"], "model_type": "unknown", "num_local_experts": 0}
    overrides = {key: value for key, value in _OVERRIDES.items() if key != "weights_bytes"}
    source, resources = _files(tmp_path, config=config, overrides=overrides)
    output = tmp_path / "request.yaml"
    prompts = _terminal(monkeypatch, ["many", "0.1 B", "-1", "2 GiB"])

    assert cli.main(_args(output, source, resources) + ["--interactive"]) == 0

    assert len(prompts) == 4
    assert all("weights_bytes" in prompt and "per rank" in prompt for prompt in prompts)
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


@pytest.mark.parametrize(
    "tp,dp,moe_tp,moe_ep,preset",
    [(4, 1, 4, 1, "pure_tp"), (1, 8, 1, 8, "dep"), (8, 1, 1, 8, "tep")],
)
def test_config_init_to_plan_replay_preserves_full_topology_without_source_files(
    tmp_path, monkeypatch, tp, dp, moe_tp, moe_ep, preset
):
    config = {
        **_CONFIG,
        "architectures": ["ExampleMoeForCausalLM"],
        "model_type": "example_moe",
        "num_local_experts": 8,
    }
    source, resources = _files(tmp_path, config=config)
    output = tmp_path / "request.yaml"
    real_import = builtins.__import__

    def no_model_or_download_import(name, *args, **kwargs):
        assert not name.startswith(("collector", "huggingface_hub", "transformers")), name
        assert ".sdk.models" not in name, name
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_or_download_import)
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
        )
        == 0
    )
    source.unlink()
    resources.unlink()
    request = SupportRequest.from_yaml(output)
    assert request.parallel_preset == preset
    assert request.profile_deployment().parallel_tuple == (tp, 1, dp, moe_tp, moe_ep, 1)
    plan = tmp_path / "plan"
    assert cli.main(["onboard", "plan", "--config", str(output), "--output-dir", str(plan)]) == 0
    assert SupportRequest.from_yaml(plan / "request.yaml") == request
    prediction = CorePredictionConfig.from_yaml(plan / "predict/pilot.yaml")
    recommendation = CoreRecommendationConfig.from_yaml(plan / "recommend/pilot.yaml")
    assert prediction.engine.fpm_profile == recommendation.engine.fpm_profile == request.fpm_profile
    assert prediction.engine.workers.aggregated.timing.fpm_interpolation == "direct"
    assert prediction.engine.workers.aggregated.parallelism.tensor == tp
    assert prediction.engine.workers.aggregated.parallelism.attention_data == dp
    assert prediction.engine.workers.aggregated.parallelism.moe_tensor == moe_tp
    assert prediction.engine.workers.aggregated.parallelism.moe_expert == moe_ep
    assert json.loads((plan / "fpm-model-profile.json").read_text()) == request.fpm_profile.model_dump(mode="json")
