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
from pathlib import Path

import pytest
import yaml

import aisimulate.main as cli
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.fpm_profile import load_fpm_profile
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


def _directory_args(source, resources, root, **changes):
    args = _args(source, resources, root, **changes)
    args[args.index("--output")] = "--output-dir"
    return args


def _directory_terminal(monkeypatch, root, answers):
    remaining = iter(answers)
    prompts = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def respond(prompt):
        prompts.append(prompt)
        assert not root.exists() or not list(root.iterdir()), "configuration published before all acceptances"
        try:
            answer = next(remaining)
        except StopIteration:
            pytest.fail(f"unexpected prompt: {prompt}")
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(builtins, "input", respond)
    return prompts


def _saved_configurations(root):
    index = json.loads((root / "onboarding.json").read_text())
    assert index["schema_version"] == "aisimulate-onboarding/v1"
    entries = index["configurations"]
    assert len({entry["request_id"] for entry in entries}) == len(entries)
    requests = []
    for entry in entries:
        path = Path(entry["request"])
        assert path.parent.parent == root
        request = SupportRequest.from_yaml(path)
        assert request.fpm_profile == load_fpm_profile(Path(entry["fpm_profile"]).read_text())
        assert len(request.fpm_profile.deployments) == 1
        assert entry["collection_gpus_required"] == request.worker_gpus
        assert shlex.split(entry["plan_command"]) == [
            "aisimulate",
            "onboard",
            "plan",
            "--config",
            str(path),
            "--output-dir",
            entry["collection_dir"],
        ]
        requests.append(request)
    return entries, requests


def _consume_configurations(root):
    entries, requests = _saved_configurations(root)
    for entry, request in zip(entries, requests, strict=True):
        plan_root = Path(entry["collection_dir"])
        assert plan_root == Path(entry["request"]).parent / "collection"
        assert cli.main(shlex.split(entry["plan_command"])[1:]) == 0
        plan = json.loads((plan_root / "support-plan.json").read_text())
        assert plan["request_id"] == entry["request_id"]
        assert plan["search"]["candidate_count"] == 1
        assert plan["fpm"]["collection_gpus_required"] == request.worker_gpus
        assert plan["fpm"]["runtime_limits"]["context_length"] == request.search.context_length
        command = plan["fpm"]["plan_command"]
        for flag, expected in (
            ("--fpm-gpu-counts", str(request.worker_gpus)),
            ("--fpm-parallel-presets", request.parallel_preset),
            ("--fpm-max-model-len", str(request.search.context_length)),
            ("--fpm-model-profile", str(plan_root / "fpm-model-profile.json")),
            ("--fpm-database-root", str(plan_root / "systems/data")),
            ("--checkpoint-dir", str(plan_root / "fpm-checkpoint")),
        ):
            assert command[command.index(flag) + 1] == expected
        preview = _real_cli(["onboard", "collect-fpm", "--config", entry["request"], "--output-dir", str(plan_root)])
        assert preview.returncode == 0, preview.stderr
        assert shlex.split(preview.stdout) == command
        prediction = CorePredictionConfig.from_yaml(plan_root / "predict/pilot.yaml")
        recommendation = CoreRecommendationConfig.from_yaml(plan_root / "recommend/pilot.yaml")
        for config in (prediction, recommendation):
            assert config.engine.fpm_profile == request.fpm_profile
            assert config.engine.context_length == request.search.context_length
            assert [str(path) for path in config.engine.systems_paths] == [str(plan_root / "systems")]
            worker = config.engine.workers.aggregated
            assert worker.timing.estimation_mode == "fpm_interpolation"
            assert worker.timing.fallback_policy == "deny"
            assert worker.timing.estimator_config["fpm_interpolation"]["method"] == "direct"
            if request.profile_deployment().resources.cache_layout == "grouped":
                assert not worker.kv_cache.prefix_caching
        assert prediction.engine.workers.aggregated.parallelism.model_dump() == request.parallelism()
        assert recommendation.engine.workers.aggregated.parallelism.preset[0].model_dump() == request.parallelism()
        assert recommendation.optimization.constraints.max_candidate_gpus == request.worker_gpus
        assert not (plan_root / "fpm-checkpoint").exists()
        assert not list((plan_root / "systems/data").glob("**/*.parquet"))
    return requests


def test_guided_multiple_tp_dep_tep_profiles_reach_independent_consumers(monkeypatch, tmp_path):
    source, resources = _inputs(tmp_path, _MOE)
    root = tmp_path / "new collection" / "profiles"
    prompts = _directory_terminal(
        monkeypatch,
        root,
        [
            "bfloat16",
            "half",
            "bfloat16",
            "1,1",
            "7",
            "x",
            "2,,3",
            "2, 3,5",
            "accept",
            "16 MiB",
            "accept",
            "32 MiB",
            "accept",
        ],
    )
    assert cli.main(_directory_args(source, None, root) + ["--interactive"]) == 0
    assert sum(prompt.startswith("Choose topologies") for prompt in prompts) == 5
    assert sum(prompt.startswith("fmha_quant_mode") for prompt in prompts) == 1
    assert sum(prompt.startswith("Review action") for prompt in prompts) == 3
    source.unlink()
    resources.unlink()
    requests = _consume_configurations(root)
    assert [request.parallel_preset for request in requests] == ["pure_tp", "dep", "tep"]
    assert [request.profile_deployment().parallel_tuple for request in requests] == [
        (2, 1, 1, 2, 1, 1),
        (1, 1, 2, 1, 2, 1),
        (2, 1, 1, 1, 2, 1),
    ]
    assert requests[1].profile_deployment().resources.comm_overhead_bytes == 16 * 1024**2
    assert requests[2].profile_deployment().resources.comm_overhead_bytes == 32 * 1024**2
    assert (
        requests[0].profile_deployment().resources.kv_bytes_per_token
        < requests[1].profile_deployment().resources.kv_bytes_per_token
    )


@pytest.mark.parametrize("file_format", ["json", "yaml"])
def test_parallel_configuration_file_preserves_rank_bounds_and_precision(tmp_path, file_format):
    source, resources = _inputs(tmp_path, _MOE)
    entries = [
        {"tensor_parallel": 2, "resource_overrides": {"weights_bytes": 1024**3}},
        {
            "tensor_parallel": 1,
            "attention_data_parallel": 2,
            "moe_expert_parallel": 2,
            "resource_overrides": {"weights_bytes": 2 * 1024**3, "comm_overhead_bytes": 16 * 1024**2},
        },
        {
            "tensor_parallel": 2,
            "moe_tensor_parallel": 1,
            "moe_expert_parallel": 2,
            "resource_overrides": {
                "weights_bytes": 3 * 1024**3,
                "comm_overhead_bytes": 32 * 1024**2,
                "kv_cache_dtype": "fp8",
            },
        },
    ]
    choices = tmp_path / f"parallel.{file_format}"
    choices.write_text(json.dumps(entries) if file_format == "json" else yaml.safe_dump(entries))
    root = tmp_path / "profiles"
    root.mkdir()  # A new empty directory is accepted, but never a collected-data directory.
    result = _real_cli(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)])
    assert result.returncode == 0, result.stderr
    assert "Suggested worker topologies" not in result.stdout
    source.unlink()
    resources.unlink()
    choices.unlink()
    requests = _consume_configurations(root)
    assert [request.profile_deployment().resources.weights_bytes for request in requests] == [
        1024**3,
        2 * 1024**3,
        3 * 1024**3,
    ]
    assert [request.profile_deployment().kv_cache_dtype for request in requests] == ["bfloat16", "bfloat16", "fp8"]


@pytest.mark.parametrize(
    "model_context,shared_context,entry_contexts,runtime_context,expected",
    [
        (4096, None, [None, 2048], None, [4096, 2048]),
        (8192, 6144, [None, 7168], None, [6144, 7168]),
        (524288, None, [262144, 131072], None, [256000, 131072]),
        (4096, None, [None, 2048], 2048, [2048, 2048]),
        (524288, None, [None, 400000], 300000, [300000, 300000]),
    ],
    ids=["model-default", "merged-default", "agentx-cap", "explicit-fitting", "explicit-above-default-cap"],
)
def test_parallel_context_limits_reach_saved_plan_consumers(
    tmp_path, model_context, shared_context, entry_contexts, runtime_context, expected
):
    shared = {**_PRECISION}
    if shared_context is not None:
        shared["context_length"] = shared_context
    source, resources = _inputs(tmp_path, {**_SMALL, "max_position_embeddings": model_context}, shared)
    choices = tmp_path / "parallel.json"
    choices.write_text(
        json.dumps(
            [
                {
                    "tensor_parallel": tp,
                    "resource_overrides": {} if context is None else {"context_length": context},
                }
                for tp, context in enumerate(entry_contexts, 1)
            ]
        )
    )
    root = tmp_path / "profiles"
    result = _real_cli(
        _directory_args(source, resources, root, context_length=runtime_context) + ["--parallel-configs", str(choices)]
    )
    assert result.returncode == 0, result.stderr
    source.unlink()
    resources.unlink()
    choices.unlink()
    requests = _consume_configurations(root)
    assert [request.search.context_length for request in requests] == expected
    assert [request.fpm_profile.context_length for request in requests] == [
        context if context is not None else shared_context or model_context for context in entry_contexts
    ]


def test_explicit_context_exceeding_later_profile_cap_publishes_nothing(tmp_path, capsys):
    source, resources = _inputs(tmp_path)
    choices = tmp_path / "parallel.json"
    choices.write_text(
        json.dumps([{"tensor_parallel": 1}, {"tensor_parallel": 2, "resource_overrides": {"context_length": 2048}}])
    )
    root = tmp_path / "new" / "profiles"
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, root, context_length=4096) + ["--parallel-configs", str(choices)])
    assert error.value.code == 2
    assert "search.context_length exceeds the configured profile context_length" in capsys.readouterr().err
    assert not root.parent.exists()


def test_implicit_context_defaults_survive_candidate_identity_corrections(monkeypatch, tmp_path):
    source, resources = _inputs(tmp_path)
    choices = tmp_path / "parallel.json"
    choices.write_text(
        json.dumps([{"tensor_parallel": 1, "resource_overrides": {"context_length": 2048}}, {"tensor_parallel": 2}])
    )
    root = tmp_path / "profiles"
    prompts = _directory_terminal(monkeypatch, root, ["0.25.1", "accept", "0.25.2", "accept"])
    assert (
        cli.main(
            _directory_args(source, resources, root, context_length=None, framework_version="unknown")
            + ["--parallel-configs", str(choices), "--interactive"]
        )
        == 0
    )
    assert sum(prompt.startswith("Pinned vLLM version") for prompt in prompts) == 2
    assert not any(prompt.startswith("Runtime per-request context") for prompt in prompts)
    source.unlink()
    resources.unlink()
    choices.unlink()
    requests = _consume_configurations(root)
    assert [request.search.context_length for request in requests] == [2048, 4096]
    assert [request.identity.framework_version for request in requests] == ["0.25.1", "0.25.2"]


def test_explicit_context_correction_stays_with_its_candidate(monkeypatch, tmp_path):
    source, resources = _inputs(tmp_path)
    choices = tmp_path / "parallel.json"
    choices.write_text(
        json.dumps([{"tensor_parallel": 1, "resource_overrides": {"context_length": 2048}}, {"tensor_parallel": 2}])
    )
    root = tmp_path / "profiles"
    prompts = _directory_terminal(monkeypatch, root, ["2048", "accept", "accept"])
    assert (
        cli.main(
            _directory_args(source, resources, root, context_length=4096)
            + ["--parallel-configs", str(choices), "--interactive"]
        )
        == 0
    )
    assert sum(prompt.startswith("Runtime per-request context") for prompt in prompts) == 1
    source.unlink()
    resources.unlink()
    choices.unlink()
    requests = _consume_configurations(root)
    assert [request.search.context_length for request in requests] == [2048, 4096]


def test_parallel_profiles_reach_real_collector_plan_only_without_gpu_work(tmp_path):
    source, resources = _inputs(tmp_path, _MOE)
    choices = tmp_path / "parallel.json"
    choices.write_text(
        json.dumps(
            [
                {"tensor_parallel": 2},
                {
                    "tensor_parallel": 1,
                    "attention_data_parallel": 2,
                    "moe_expert_parallel": 2,
                    "resource_overrides": {"comm_overhead_bytes": 16 * 1024**2},
                },
                {
                    "tensor_parallel": 2,
                    "moe_tensor_parallel": 1,
                    "moe_expert_parallel": 2,
                    "resource_overrides": {"comm_overhead_bytes": 32 * 1024**2},
                },
            ]
        )
    )
    root = tmp_path / "profiles"
    assert cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)]) == 0
    entries, requests = _saved_configurations(root)
    collector_ids = set()
    for entry, request in zip(entries, requests, strict=True):
        assert cli.main(shlex.split(entry["plan_command"])[1:]) == 0
        plan_root = Path(entry["collection_dir"])
        command = json.loads((plan_root / "support-plan.json").read_text())["fpm"]["plan_command"]
        # Runtime collector metadata still comes from the checkpoint/config;
        # the emitted inline profile supplies the reviewed resource bounds.
        result = subprocess.run(
            [sys.executable, *command[1:], "--fpm-model-config", str(source)],
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        collected_plan = json.loads(result.stdout)
        assert collected_plan["fpm_profile"] == request.fpm_profile.model_dump(mode="json")
        assert collected_plan["counts"]["topologies"] == 1
        assert {cell["parallel_strategy"] for cell in collected_plan["cells"]} == {request.parallel_preset}
        assert {item["source"] for item in collected_plan["topology_memory_admission"]} == {"fpm_profile_declared"}
        collector_ids.add(collected_plan["sha256"])
        assert not (plan_root / "fpm-checkpoint").exists()
        assert not (plan_root / "fpm-artifacts").exists()
        assert not list((plan_root / "systems/data").glob("**/*.parquet"))
    assert len(collector_ids) == 3


def test_interactive_file_resolves_shared_fields_once_and_retains_entry_precision(monkeypatch, tmp_path):
    source, _ = _inputs(tmp_path)
    choices = tmp_path / "parallel.json"
    choices.write_text(
        json.dumps(
            [
                {"tensor_parallel": 1, "resource_overrides": {"kv_cache_dtype": "fp8"}},
                {"tensor_parallel": 2},
            ]
        )
    )
    root = tmp_path / "profiles"
    prompts = _directory_terminal(monkeypatch, root, ["bfloat16", "half", "bfloat16", "accept", "accept"])
    assert cli.main(_directory_args(source, None, root) + ["--parallel-configs", str(choices), "--interactive"]) == 0
    assert [prompt.split(" (")[0] for prompt in prompts[:3]] == ["fmha_quant_mode", "comm_quant_mode", "kv_cache_dtype"]
    assert len(prompts) == 5
    _, requests = _saved_configurations(root)
    assert [request.profile_deployment().kv_cache_dtype for request in requests] == ["fp8", "bfloat16"]


@pytest.mark.parametrize("explicit", [False, True])
def test_directory_singleton_preserves_default_or_explicit_topology(monkeypatch, tmp_path, explicit):
    source, resources = _inputs(tmp_path)
    root = tmp_path / "profiles"
    prompts = _directory_terminal(monkeypatch, root, ["accept"] if explicit else ["", "accept"])
    flags = ["--tensor-parallel", "2"] if explicit else []
    assert cli.main(_directory_args(source, resources, root) + ["--interactive", *flags]) == 0
    _, requests = _saved_configurations(root)
    assert len(requests) == 1
    assert requests[0].worker_gpus == (2 if explicit else 1)
    assert sum(prompt.startswith("Choose topologies") for prompt in prompts) == (0 if explicit else 1)


def test_guided_directory_without_default_requires_selection_and_separate_bounds(monkeypatch, tmp_path):
    source, resources = _inputs(tmp_path, {**_SMALL, "quantization_config": {"quant_method": "fp8"}})
    root = tmp_path / "profiles"
    prompts = _directory_terminal(monkeypatch, root, ["", "1,2", "1 GiB", "accept", "2 GiB", "accept"])
    assert cli.main(_directory_args(source, resources, root) + ["--interactive"]) == 0
    assert sum(prompt.startswith("Choose topologies") for prompt in prompts) == 2
    assert sum(prompt.startswith("weights_bytes") for prompt in prompts) == 2
    _, requests = _saved_configurations(root)
    assert [request.profile_deployment().resources.weights_bytes for request in requests] == [1024**3, 2 * 1024**3]


def test_parallel_file_accepts_explicit_topology_outside_shortlist(tmp_path):
    source, resources = _inputs(tmp_path, _LARGE)
    choices = tmp_path / "parallel.yaml"
    choices.write_text(
        yaml.safe_dump(
            [
                {
                    "tensor_parallel": 16,
                    "resource_overrides": {"activations_bytes": 70 * 1024**2, "comm_overhead_bytes": 256 * 1024**2},
                }
            ]
        )
    )
    root = tmp_path / "profiles"
    assert cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)]) == 0
    assert _saved_configurations(root)[1][0].worker_gpus == 16


def test_grouped_profile_edits_and_collection_limits_stay_with_their_candidate(monkeypatch, tmp_path):
    source, resources = _inputs(
        tmp_path,
        {**_SMALL, "sliding_window": 128, "layer_types": ["full_attention", "sliding_attention"]},
        {**_PRECISION, "cache_block_sizes": {"full_attention": 16, "sliding_attention": 8}},
    )
    choices = tmp_path / "parallel.json"
    choices.write_text(json.dumps([{"tensor_parallel": 1}, {"tensor_parallel": 2}]))
    root = tmp_path / "profiles"
    _directory_terminal(
        monkeypatch,
        root,
        [
            "edit",
            "cache_block_sizes",
            '{"full_attention": 32, "sliding_attention": 16}',
            "edit",
            "max_num_tokens",
            "2048",
            "edit",
            "runtime_context_length",
            "2048",
            "edit",
            "max_prefill_cudagraph_size",
            "512",
            "accept",
            "accept",
        ],
    )
    assert (
        cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices), "--interactive"]) == 0
    )
    source.unlink()
    resources.unlink()
    choices.unlink()
    first, second = _consume_configurations(root)
    assert [request.search.context_length for request in (first, second)] == [2048, 4096]
    assert [request.scheduler_limits()["max_batched_tokens"] for request in (first, second)] == [2048, 8192]
    assert [request.collection_settings()["max_prefill_cudagraph_size"] for request in (first, second)] == [512, 2048]
    first_cache, second_cache = (request.profile_deployment().resources for request in (first, second))
    assert [group.block_size_tokens for group in first_cache.cache_groups] == [32, 16]
    assert [group.block_size_tokens for group in second_cache.cache_groups] == [16, 8]
    assert first_cache.cache_groups[0].page_size_bytes != second_cache.cache_groups[0].page_size_bytes
    assert "user override" not in json.loads(second_cache.provenance)["fields"]["cache_groups"]["source"]


def test_grouped_per_entry_pages_do_not_replace_other_topology_estimates(tmp_path):
    source, resources = _inputs(
        tmp_path,
        {**_SMALL, "sliding_window": 128},
        {**_PRECISION, "cache_block_sizes": {"sliding_attention": 16}},
    )
    groups = [
        {
            "name": "sliding_attention",
            "kind": "attention",
            "num_layers": 2,
            "block_size_tokens": 32,
            "page_size_bytes": 8192,
            "sliding_window": 128,
        }
    ]
    choices = tmp_path / "parallel.yaml"
    choices.write_text(
        yaml.safe_dump(
            [
                {"tensor_parallel": 1, "resource_overrides": {"cache_groups": groups}},
                {"tensor_parallel": 2},
            ]
        )
    )
    root = tmp_path / "profiles"
    assert cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)]) == 0
    source.unlink()
    first, second = _consume_configurations(root)
    assert first.profile_deployment().resources.cache_groups[0].page_size_bytes == 8192
    assert first.profile_deployment().resources.cache_groups[0].block_size_tokens == 32
    assert second.profile_deployment().resources.cache_groups[0].block_size_tokens == 16
    assert second.profile_deployment().resources.cache_groups[0].page_size_bytes < 8192


@pytest.mark.parametrize("interruption", ["cancel", EOFError(), KeyboardInterrupt()])
@pytest.mark.parametrize("existing", [False, True])
def test_later_profile_cancellation_publishes_nothing(monkeypatch, tmp_path, interruption, existing):
    source, resources = _inputs(tmp_path)
    root = tmp_path / "new" / "profiles"
    if existing:
        root.mkdir(parents=True)
    prompts = _directory_terminal(monkeypatch, root, ["1,2", "accept", interruption])
    assert cli.main(_directory_args(source, resources, root) + ["--interactive"]) == 130
    assert sum(prompt.startswith("Review action") for prompt in prompts) == 2
    if existing:
        assert not list(root.iterdir())
    else:
        assert not root.parent.exists()


def test_late_identity_corrections_and_rank_bounds_are_independent(monkeypatch, tmp_path):
    source, resources = _inputs(tmp_path, {**_SMALL, "quantization_config": {"quant_method": "fp8"}})
    choices = tmp_path / "parallel.json"
    choices.write_text(json.dumps([{"tensor_parallel": 1}, {"tensor_parallel": 2}]))
    root = tmp_path / "profiles"
    prompts = _directory_terminal(monkeypatch, root, ["1 GiB", "0.25.1", "accept", "2 GiB", "0.25.2", "accept"])
    assert (
        cli.main(
            _directory_args(source, resources, root, framework_version="unknown")
            + ["--parallel-configs", str(choices), "--interactive"]
        )
        == 0
    )
    _, requests = _saved_configurations(root)
    assert [request.worker_gpus for request in requests] == [1, 2]
    assert [request.identity.framework_version for request in requests] == ["0.25.1", "0.25.2"]
    assert [request.profile_deployment().resources.weights_bytes for request in requests] == [1024**3, 2 * 1024**3]
    assert sum(prompt.startswith("Pinned vLLM version") for prompt in prompts) == 2


def test_topology_correction_cannot_publish_duplicate_configurations(monkeypatch, tmp_path, capsys):
    source, resources = _inputs(tmp_path)
    choices = tmp_path / "parallel.yaml"
    choices.write_text(yaml.safe_dump([{"tensor_parallel": 1}, {"tensor_parallel": 3}]))
    root = tmp_path / "new" / "profiles"
    _directory_terminal(monkeypatch, root, ["accept", "1", "accept"])
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices), "--interactive"])
    assert error.value.code == 2
    assert "duplicate resolved parallel configuration" in capsys.readouterr().err
    assert not root.parent.exists()


@pytest.mark.parametrize(
    "payload,expected",
    [
        ([], "nonempty"),
        ({"tensor_parallel": 1}, "list"),
        ([1], "object"),
        ([{}], "requires tensor_parallel"),
        ([{"tensor_parallel": True}], "positive integer"),
        ([{"tensor_parallel": "2"}], "positive integer"),
        ([{"tensor_parallel": 2.0}], "positive integer"),
        ([{"tensor_parallel": 0}], "positive integer"),
        ([{"tensor_parallel": 1, "attention_data_parallel": None}], "positive integer"),
        ([{"tensor_parallel": 1, "name": "../escape"}], "unknown fields"),
        ([{"tensor_parallel": 1, "resource_overrides": None}], "must be a mapping"),
        ([{"tensor_parallel": 1, "resource_overrides": {"weights_bytes": True}}], "weights_bytes"),
        ([{"tensor_parallel": 1, "resource_overrides": {"unknown": 4}}], "unknown resource"),
        (
            [
                {"tensor_parallel": 2},
                {
                    "tensor_parallel": 2,
                    "attention_data_parallel": 1,
                    "moe_tensor_parallel": 2,
                    "moe_expert_parallel": 1,
                },
            ],
            "duplicate resolved",
        ),
        ([{"tensor_parallel": 2, "attention_data_parallel": 2}], "complete TP, DEP, or TEP"),
        (
            [{"tensor_parallel": 2}, {"tensor_parallel": 1, "attention_data_parallel": 2, "moe_expert_parallel": 2}],
            "comm_overhead_bytes",
        ),
        ([{"tensor_parallel": 1, 7: "invalid", "unknown": 3}], "field names must be strings"),
    ],
)
def test_invalid_parallel_files_leave_no_partial_output(tmp_path, capsys, payload, expected):
    source, resources = _inputs(tmp_path, _MOE)
    choices = tmp_path / "parallel.yaml"
    choices.write_text(yaml.safe_dump(payload))
    root = tmp_path / "new" / "profiles"
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)])
    assert error.value.code == 2
    assert expected in capsys.readouterr().err
    assert not root.parent.exists()


@pytest.mark.parametrize("payload", ["- tensor_parallel: 1\n  tensor_parallel: 2\n", "[{"])
def test_duplicate_keys_and_malformed_parallel_files_fail_cleanly(tmp_path, capsys, payload):
    source, resources = _inputs(tmp_path)
    choices = tmp_path / "parallel.yaml"
    choices.write_text(payload)
    root = tmp_path / "profiles"
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)])
    assert error.value.code == 2
    expected = "duplicate" if "tensor_parallel" in payload else "malformed"
    assert expected in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights_bytes", 1024**3),
        ("activations_bytes", 1024),
        ("runtime_overhead_bytes", 1024),
        ("comm_overhead_bytes", 1024),
        ("kv_bytes_per_token", 1024),
        (
            "cache_groups",
            [
                {
                    "name": "full_attention",
                    "kind": "attention",
                    "num_layers": 2,
                    "block_size_tokens": 16,
                    "page_size_bytes": 4096,
                }
            ],
        ),
    ],
)
def test_parallel_configurations_reject_shared_rank_local_resources(tmp_path, capsys, field, value):
    source, resources = _inputs(tmp_path, overrides={**_PRECISION, field: value})
    choices = tmp_path / "parallel.json"
    choices.write_text(json.dumps([{"tensor_parallel": 1}, {"tensor_parallel": 2}]))
    root = tmp_path / "profiles"
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)])
    assert error.value.code == 2
    assert "cannot share rank-local resource overrides" in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--output", "request.yaml"],
        ["--overwrite"],
        ["--suggest-parallel"],
        ["--parallel-configs", "absent.yaml", "--tensor-parallel", "1"],
        ["--parallel-configs", "absent.yaml", "--attention-data-parallel", "1"],
        ["--parallel-configs", "absent.yaml", "--moe-tensor-parallel", "1"],
        ["--parallel-configs", "absent.yaml", "--moe-expert-parallel", "1"],
    ],
)
def test_directory_option_conflicts_fail_before_loading_inputs(tmp_path, capsys, extra):
    root = tmp_path / "profiles"
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(tmp_path / "absent.json", None, root) + extra)
    assert error.value.code == 2
    expected = "not allowed" if "--output" in extra else "cannot be combined"
    assert expected in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize("extra", [[], ["--model-config", "absent.json"], ["--output-dir", "absent"]])
def test_parallel_file_requires_both_config_and_directory(capsys, extra):
    with pytest.raises(SystemExit) as error:
        cli.main(["onboard", "init", "--parallel-configs", "absent.yaml", *extra])
    assert error.value.code == 2
    assert "requires --model-config and --output-dir" in capsys.readouterr().err


def test_empty_directory_argument_cannot_fall_back_to_single_file_output(monkeypatch, tmp_path, capsys):
    source, resources = _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, ""))
    assert error.value.code == 2
    assert "nonempty path" in capsys.readouterr().err
    assert not (tmp_path / "support-request.yaml").exists()


@pytest.mark.parametrize("kind", ["nonempty", "file", "symlink", "dangling", "parent_symlink"])
def test_directory_path_conflicts_preserve_existing_contents(tmp_path, capsys, kind):
    source, resources = _inputs(tmp_path)
    protected = tmp_path / "collected"
    protected.mkdir()
    (protected / "fpm_forward_perf.parquet").write_bytes(b"existing collected data")
    root = tmp_path / "profiles"
    if kind == "nonempty":
        root = protected
    elif kind == "file":
        root.write_text("existing file")
    elif kind in {"symlink", "dangling"}:
        root.symlink_to(protected if kind == "symlink" else tmp_path / "absent", target_is_directory=True)
    else:
        link = tmp_path / "linked"
        link.symlink_to(protected, target_is_directory=True)
        root = link / "profiles"
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, root))
    assert error.value.code == 2
    expected = "symlinks" if "symlink" in kind or kind == "dangling" else "new or empty"
    assert expected in capsys.readouterr().err
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ["occupied", "symlink", "serialization"])
def test_directory_publication_failure_leaves_no_partial_profiles(monkeypatch, tmp_path, capsys, failure):
    source, resources = _inputs(tmp_path)
    choices = tmp_path / "parallel.json"
    choices.write_text(json.dumps([{"tensor_parallel": 1}, {"tensor_parallel": 2}]))
    root = tmp_path / "profiles"
    protected = tmp_path / "collected"
    protected.mkdir()
    marker = protected / "fpm_forward_perf.parquet"
    marker.write_bytes(b"existing collected data")
    write_text = Path.write_text

    def interrupt_publication(path, *args, **kwargs):
        if path.name == "onboarding.json":
            if failure == "occupied":
                root.mkdir()
                (root / "collected.txt").write_bytes(b"new external data")
            elif failure == "symlink":
                root.symlink_to(protected, target_is_directory=True)
        if failure == "serialization" and path.name == "fpm-profile.json" and path.parent.name.startswith("tp2-"):
            raise OSError("simulated profile write failure")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", interrupt_publication)
    with pytest.raises(SystemExit) as error:
        cli.main(_directory_args(source, resources, root) + ["--parallel-configs", str(choices)])
    assert error.value.code == 2
    assert marker.read_bytes() == b"existing collected data"
    if failure == "occupied":
        assert [(path.name, path.read_bytes()) for path in root.iterdir()] == [("collected.txt", b"new external data")]
    elif failure == "symlink":
        assert root.is_symlink()
    else:
        assert not root.exists()
    assert not list(tmp_path.glob(".profiles-*"))
