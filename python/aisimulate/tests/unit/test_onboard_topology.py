# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology suggestions through onboarding's actual CLI and saved-plan consumers."""

from __future__ import annotations

import builtins
import hashlib
import json
import shlex
import subprocess
import sys

import pytest
import yaml

import aisimulate.main as cli
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit

# Original synthetic decoder geometries, not downloaded checkpoints or measurements.
_SMALL = {
    "_name_or_path": "example/synthetic-decoder",
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "vocab_size": 64,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
}
_LARGE = {
    **_SMALL,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 32768,
}
_MOE = {**_SMALL, "architectures": ["MixtralForCausalLM"], "model_type": "mixtral", "num_local_experts": 4}
_PRECISION = {"fmha_quant_mode": "bfloat16", "comm_quant_mode": "half", "kv_cache_dtype": "bfloat16"}
_OPTIONS = {
    "model_revision": "synthetic-checkpoint-123",
    "framework_version": "0.25.1",
    "gpu": "h200_sxm",
    "interconnect": "nvswitch",
    "tokenizer_revision": "synthetic-tokenizer-456",
    "chat_template_revision": "synthetic-template-789",
    "input_tokens": 1024,
    "output_tokens": 128,
    "concurrency": 2,
    "request_count": 3,
    "context_length": 4096,
    "ttft_ms": 1000,
    "tpot_ms": 100,
    "seed": 123,
}


def _inputs(tmp_path, config=None, overrides=None):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(_SMALL if config is None else config))
    resources = tmp_path / "resources.yaml"
    resources.write_text(yaml.safe_dump(_PRECISION if overrides is None else overrides))
    return source, resources


def _args(source, resources, output, **changes):
    options = {**_OPTIONS, **changes}
    command = ["onboard", "init", "--model-config", str(source), "--output", str(output)]
    if resources is not None:
        command += ["--resource-overrides", str(resources)]
    return command + [
        part
        for name, value in options.items()
        if value is not None
        for part in ("--" + name.replace("_", "-"), str(value))
    ]


def _real_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "aisimulate", *args], input="", text=True, capture_output=True, timeout=30
    )


def _terminal(monkeypatch, output, answers):
    remaining = iter(answers)
    prompts = []
    original = output.read_bytes() if output.exists() else None
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def respond(prompt):
        prompts.append(prompt)
        if original is None:
            assert not output.parent.exists(), "interactive setup wrote before acceptance"
        else:
            assert output.read_bytes() == original, "interactive setup replaced output before acceptance"
        try:
            answer = next(remaining)
        except StopIteration:
            pytest.fail(f"unexpected prompt: {prompt}")
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", respond)
    return prompts


@pytest.mark.parametrize("config,gpu,width", [(_SMALL, "h200_sxm", 1), (_LARGE, "h200_sxm", 2), (_LARGE, "gb200", 1)])
def test_real_cli_default_uses_model_geometry_and_hardware_then_embeds_exact_profile(tmp_path, config, gpu, width):
    source, resources = _inputs(tmp_path, config)
    output = tmp_path / "new" / "request.yaml"
    result = _real_cli(_args(source, resources, output, gpu=gpu))
    assert result.returncode == 0, result.stderr
    request = SupportRequest.from_yaml(output)
    assert request.worker_gpus == width
    assert request.profile_deployment().parallel_tuple == (width, 1, 1, 1, 1, 1)
    assert request.identity.model_revision == _OPTIONS["model_revision"]
    assert request.identity.tokenizer_revision == _OPTIONS["tokenizer_revision"]
    assert request.identity.chat_template_revision == _OPTIONS["chat_template_revision"]
    assert request.workload.concurrency == 2
    assert request.workload.request_count == 3
    assert request.search.seed == 123
    assert request.search.context_length == 4096
    assert f"Selected tp with {width} collection GPUs" in result.stdout
    assert "estimated_fit" in result.stdout
    assert "not a performance ranking" in result.stdout
    assert not {"gpu_count", "node_count", "gpus_per_node"} & request.identity.model_dump().keys()
    assert "max_candidates" not in request.search.model_dump()
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("output_kind", ["existing", "directory", "missing", "file_parent"])
def test_real_cli_preview_is_json_and_never_validates_or_writes_output_target(tmp_path, output_kind):
    source, resources = _inputs(tmp_path, _LARGE)
    existing = tmp_path / "existing-request.yaml"
    existing.write_text("preserve previous request\n")
    targets = {
        "existing": existing,
        "directory": tmp_path,
        "missing": tmp_path / "new" / "request.yaml",
        "file_parent": existing / "request.yaml",
    }
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = _real_cli(
        _args(source, resources, targets[output_kind], model="deployment/actual-model") + ["--suggest-parallel"]
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["default"]["topology"]["tensor_parallel"] == 2
    assert report["identity"]["model"] == "deployment/actual-model"
    assert report["identity"]["model_revision"] == _OPTIONS["model_revision"]
    assert report["identity"]["framework_version"] == _OPTIONS["framework_version"]
    assert "workload" not in report
    assert report["collection"]["max_sequences"] == 256
    assert report["context_length"] == 4096
    assert report["config_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("gpu,domain,width", [("h200_sxm", "node", 8), ("gb200", "rack", 72)])
def test_real_cli_preview_exposes_architecture_domain_and_moe_alternatives(tmp_path, gpu, domain, width):
    source, resources = _inputs(tmp_path, _MOE)
    output = tmp_path / "new" / "request.yaml"
    result = _real_cli(_args(source, resources, output, gpu=gpu) + ["--suggest-parallel"])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["hardware"]["fast_domain"] == domain
    assert report["hardware"]["fast_domain_gpus"] == width
    assert {candidate["family"] for candidate in report["candidates"]} == {"pure_tp", "dep", "tep"}
    assert len(report["candidates"]) == 6
    for candidate in report["candidates"]:
        if candidate["family"] in {"dep", "tep"}:
            assert candidate["status"] == "needs_inputs"
            assert "comm_overhead_bytes" in candidate["missing"]
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "config,resources,missing",
    [
        (_SMALL, False, "fmha_quant_mode"),
        ({**_SMALL, "quantization_config": {"quant_method": "fp8"}}, True, "weights_bytes"),
    ],
)
def test_real_cli_missing_facts_preview_has_no_default_and_normal_init_fails_before_write(
    tmp_path, config, resources, missing
):
    source, overrides = _inputs(tmp_path, config)
    output = tmp_path / "new" / "request.yaml"
    command = _args(source, overrides if resources else None, output)
    preview = _real_cli([*command, "--suggest-parallel"])
    assert preview.returncode == 0, preview.stderr
    report = json.loads(preview.stdout)
    assert report["default"] is None
    assert all(item["status"] == "needs_inputs" and missing in item["missing"] for item in report["candidates"])
    result = _real_cli(command)
    assert result.returncode == 2
    assert "No fully assessed topology default" in result.stderr
    assert missing in result.stderr
    assert "--tensor-parallel" in result.stderr
    assert "--resource-overrides" in result.stderr
    assert "Selected" not in result.stdout
    assert not output.parent.exists()


@pytest.mark.parametrize("preview", [False, True])
def test_flat_rank_bounds_require_an_explicit_topology_without_transferring_bytes(tmp_path, preview):
    source, resources = _inputs(tmp_path, overrides={**_PRECISION, "weights_bytes": 9 * 1024**3})
    output = tmp_path / "new" / "request.yaml"
    command = _args(source, resources, output)
    result = _real_cli(command + (["--suggest-parallel"] if preview else []))
    assert result.returncode == 2
    assert "rank-local byte overrides" in result.stderr
    assert "weights_bytes" in result.stderr
    assert "exact topology" in result.stderr
    assert not output.parent.exists()
    explicit = _real_cli(command + ["--tensor-parallel", "2"])
    assert explicit.returncode == 0, explicit.stderr
    request = SupportRequest.from_yaml(output)
    assert request.worker_gpus == 2
    assert request.profile_deployment().resources.weights_bytes == 9 * 1024**3
    assert "Suggested worker topologies" not in explicit.stdout


@pytest.mark.parametrize(
    "option", ["--tensor-parallel", "--attention-data-parallel", "--moe-tensor-parallel", "--moe-expert-parallel"]
)
def test_any_explicit_topology_option_preserves_existing_route(tmp_path, option):
    source, resources = _inputs(tmp_path, overrides={**_PRECISION, "weights_bytes": 2 * 1024**3})
    output = tmp_path / "request.yaml"
    result = _real_cli(_args(source, resources, output) + [option, "1"])
    assert result.returncode == 0, result.stderr
    assert SupportRequest.from_yaml(output).profile_deployment().resources.weights_bytes == 2 * 1024**3
    assert "Suggested worker topologies" not in result.stdout


def test_explicit_topology_may_exceed_automatic_hardware_domain(tmp_path):
    source, resources = _inputs(
        tmp_path,
        _LARGE,
        {**_PRECISION, "activations_bytes": 70 * 1024**2, "comm_overhead_bytes": 256 * 1024**2},
    )
    output = tmp_path / "request.yaml"
    result = _real_cli(_args(source, resources, output, tensor_parallel=16))
    assert result.returncode == 0, result.stderr
    request = SupportRequest.from_yaml(output)
    assert request.worker_gpus == request.profile_deployment().tp == 16
    assert request.profile_deployment().resources.comm_overhead_bytes == 256 * 1024**2
    assert "Suggested worker topologies" not in result.stdout


def test_no_candidate_in_declared_domain_fails_without_a_tp1_fallback(tmp_path):
    source, resources = _inputs(tmp_path, _LARGE)
    output = tmp_path / "new" / "request.yaml"
    result = _real_cli(_args(source, resources, output, interconnect="none"))
    assert result.returncode == 2
    assert "No eligible topology" in result.stderr
    assert "Rejected tp, 1 GPUs" in result.stderr
    assert "lower bound" in result.stderr
    assert "Selected" not in result.stdout
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "extra,expected",
    [
        (["--interactive"], "cannot be combined"),
        (["--profile", "onboarding"], "cannot be combined"),
        (["--tensor-parallel", "1"], "cannot be combined"),
        (["--attention-data-parallel", "1"], "cannot be combined"),
        (["--moe-tensor-parallel", "1"], "cannot be combined"),
        (["--moe-expert-parallel", "1"], "cannot be combined"),
        (["--fpm-profile", "missing.yaml"], "not allowed"),
    ],
)
def test_preview_rejects_incompatible_options_before_loading_files(tmp_path, extra, expected):
    output = tmp_path / "request.yaml"
    output.write_text("keep existing\n")
    result = _real_cli(_args(tmp_path / "absent.json", None, output) + ["--suggest-parallel", "--overwrite", *extra])
    assert result.returncode == 2
    assert expected in result.stderr
    assert output.read_text() == "keep existing\n"
    assert list(tmp_path.iterdir()) == [output]


def test_preview_requires_a_model_config_and_real_target_identity(tmp_path):
    output = tmp_path / "new" / "request.yaml"
    result = _real_cli(["onboard", "init", "--suggest-parallel", "--output", str(output)])
    assert result.returncode == 2
    assert "requires --model-config" in result.stderr
    source, resources = _inputs(tmp_path)
    result = _real_cli(
        _args(source, resources, output, model_revision=None, framework_version=None) + ["--suggest-parallel"]
    )
    assert result.returncode == 2
    assert "--model-revision" in result.stderr
    assert "--framework-version" in result.stderr
    assert not result.stdout
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "updates,extra,expected",
    [
        ({}, ["--context-length", "8192"], "context_length"),
        ({"weights_bytes": True}, [], "weights_bytes"),
        ({"unknown_field": 3}, [], "unknown"),
        ({}, ["--tensor-parallel", "3"], "num_attention_heads"),
    ],
)
def test_invalid_inputs_preserve_output_before_any_write(tmp_path, updates, extra, expected):
    source, resources = _inputs(tmp_path, overrides={**_PRECISION, **updates})
    output = tmp_path / "request.yaml"
    output.write_text("original\n")
    result = _real_cli(_args(source, resources, output) + ["--overwrite", *extra])
    assert result.returncode == 2
    assert expected in result.stderr
    assert output.read_text() == "original\n"


def test_guided_shared_precision_then_default_skips_blind_tp_prompt_and_retains_review(monkeypatch, tmp_path, capsys):
    source, _ = _inputs(tmp_path)
    output = tmp_path / "new" / "request.yaml"
    prompts = _terminal(monkeypatch, output, ["invalid", "bfloat16", "half", "bfloat16", "", "accept"])
    assert cli.main(_args(source, None, output) + ["--interactive"]) == 0
    assert prompts[0].startswith("fmha_quant_mode")
    assert prompts[2].startswith("comm_quant_mode")
    assert prompts[3].startswith("kv_cache_dtype")
    assert prompts[4] == "Choose topology 1-2 [1] (or cancel): "
    assert prompts[-1] == "Review action (accept/edit/cancel): "
    assert not any("Attention tensor-parallel" in prompt or "available" in prompt for prompt in prompts)
    assert SupportRequest.from_yaml(output).worker_gpus == 1
    assert "not checkpoint metadata" in capsys.readouterr().out


@pytest.mark.parametrize(
    "choice,preset,parallel",
    [("2", "pure_tp", (2, 1, 1, 2, 1, 1)), ("3", "dep", (1, 1, 2, 1, 2, 1)), ("5", "tep", (2, 1, 1, 1, 2, 1))],
)
def test_selected_moe_tuple_reaches_plan_collector_and_ordinary_runtime_configs(
    monkeypatch, tmp_path, capsys, choice, preset, parallel
):
    source, resources = _inputs(tmp_path, _MOE)
    output = tmp_path / "new" / "request.yaml"
    answers = [choice, *(["16 MiB"] if preset != "pure_tp" else []), "accept"]
    prompts = _terminal(monkeypatch, output, answers)
    assert cli.main(_args(source, resources, output) + ["--interactive"]) == 0
    request = SupportRequest.from_yaml(output)
    assert request.profile_deployment().parallel_tuple == parallel
    assert request.parallel_preset == preset
    assert prompts[0].startswith("Choose topology")
    if preset != "pure_tp":
        assert prompts[1].startswith("comm_overhead_bytes")
        assert request.profile_deployment().resources.comm_overhead_bytes == 16 * 1024**2
    source.unlink()
    resources.unlink()
    plan_root = tmp_path / "plan"
    result = _real_cli(["onboard", "plan", "--config", str(output), "--output-dir", str(plan_root), "--format", "json"])
    assert result.returncode == 0, result.stderr
    plan = json.loads((plan_root / "support-plan.json").read_text())
    assert plan["search"]["candidate_count"] == 1
    assert plan["fpm"]["collection_gpus_required"] == request.worker_gpus == 2
    command = plan["fpm"]["plan_command"]
    assert command[command.index("--fpm-gpu-counts") + 1] == "2"
    assert command[command.index("--fpm-parallel-presets") + 1] == preset
    assert command[command.index("--fpm-model-profile") + 1] == str(plan_root / "fpm-model-profile.json")
    collector = _real_cli(["onboard", "collect-fpm", "--config", str(output), "--output-dir", str(plan_root)])
    assert collector.returncode == 0, collector.stderr
    assert shlex.split(collector.stdout) == command
    prediction = CorePredictionConfig.from_yaml(plan_root / "predict/pilot.yaml")
    recommendation = CoreRecommendationConfig.from_yaml(plan_root / "recommend/pilot.yaml")
    assert prediction.engine.fpm_profile == recommendation.engine.fpm_profile == request.fpm_profile
    for config in (prediction, recommendation):
        worker = config.engine.workers.aggregated
        assert worker.timing.estimation_mode == "fpm_interpolation"
        assert worker.timing.fallback_policy == "deny"
        assert worker.timing.estimator_config["fpm_interpolation"]["method"] == "direct"
    assert prediction.engine.workers.aggregated.parallelism.model_dump() == request.parallelism()
    assert recommendation.engine.workers.aggregated.parallelism.preset[0].model_dump() == request.parallelism()
    assert recommendation.optimization.constraints.max_candidate_gpus == 2
    assert not (plan_root / "fpm-checkpoint").exists()
    assert not list((plan_root / "systems/data").glob("**/*.parquet"))


def test_guided_no_default_requires_number_then_bounds_for_that_tuple(monkeypatch, tmp_path, capsys):
    source, resources = _inputs(tmp_path, {**_SMALL, "quantization_config": {"quant_method": "fp8"}})
    output = tmp_path / "new" / "request.yaml"
    prompts = _terminal(monkeypatch, output, ["", "0", "2", "2 GiB", "accept"])
    assert cli.main(_args(source, resources, output) + ["--interactive"]) == 0
    assert prompts[:3] == ["Choose topology 1-2 (or cancel): "] * 3
    assert prompts[3].startswith("weights_bytes")
    request = SupportRequest.from_yaml(output)
    assert request.worker_gpus == 2
    assert request.profile_deployment().resources.weights_bytes == 2 * 1024**3
    assert "Memory fit is unresolved" in capsys.readouterr().out


@pytest.mark.parametrize("interruption", ["cancel", EOFError(), KeyboardInterrupt()])
@pytest.mark.parametrize("existing", [False, True])
def test_guided_selection_cancel_and_interrupt_preserve_output(monkeypatch, tmp_path, interruption, existing):
    source, resources = _inputs(tmp_path)
    output = tmp_path / "new" / "request.yaml"
    if existing:
        output.parent.mkdir()
        output.write_text("keep existing\n")
    _terminal(monkeypatch, output, [interruption])
    assert cli.main(_args(source, resources, output) + ["--interactive", "--overwrite"]) == 130
    if existing:
        assert output.read_text() == "keep existing\n"
        assert list(output.parent.iterdir()) == [output]
    else:
        assert not output.parent.exists()


def test_selected_topology_survives_late_identity_correction(monkeypatch, tmp_path):
    # Missing weights defer strict profile identity validation until after selection.
    source, resources = _inputs(tmp_path, {**_SMALL, "quantization_config": {"quant_method": "fp8"}})
    output = tmp_path / "new" / "request.yaml"
    _terminal(monkeypatch, output, ["2", "2 GiB", _OPTIONS["framework_version"], "accept"])
    assert cli.main(_args(source, resources, output, framework_version="unknown") + ["--interactive"]) == 0
    request = SupportRequest.from_yaml(output)
    assert request.worker_gpus == request.profile_deployment().tp == 2
    assert request.identity.framework_version == _OPTIONS["framework_version"]


def test_cold_supervised_preview_does_not_import_execution_or_model_code(tmp_path):
    source, resources = _inputs(tmp_path)
    output = tmp_path / "new" / "request.yaml"
    command = _args(source, resources, output) + ["--suggest-parallel"]
    script = """
import importlib.abc
import sys
blocked = ('aiconfigurator', 'aiconfigurator_core', 'aisimulate.sdk', 'aisimulate_core.sdk',
           'aisimulate_core._native', 'aisimulate._native', 'aisimulate._runtime',
           'collector', 'huggingface_hub', 'transformers', 'numpy', 'pandas', 'pyarrow', 'torch')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('unexpected execution import: ' + fullname)
sys.meta_path.insert(0, Guard())
from aisimulate.supervision import main
code = main(sys.argv[1:])
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in blocked)
core = {name for name in sys.modules if name == 'aisimulate_core' or name.startswith('aisimulate_core.')}
assert core <= {'aisimulate_core', 'aisimulate_core.fpm_profile', 'aisimulate_core.quantization'}, core
raise SystemExit(code)
"""
    result = subprocess.run([sys.executable, "-c", script, *command], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["default"]["required_gpus"] == 1
    assert not output.parent.exists()
