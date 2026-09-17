# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill CP / decode CP rendering through the generator.

Covers the single decision point (``generator/context_parallel.py``), the
rule + mapping + versioned-template chain on both request paths (Sweeper
candidate and SDK result bridge), and the FPM world-size check.
"""

from __future__ import annotations

import copy
import shlex
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader

from aiconfigurator.fpm_contract import FPM_RUN_SCRIPT_FILENAME
from aiconfigurator.generator.api import generate_backend_artifacts, generate_from_request
from aiconfigurator.generator.context_parallel import (
    ContextParallelUnsupportedError,
    context_parallel_gpu_multiplier,
    context_parallel_params,
    cp_strategy_for,
)
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.generator.rendering.engine import render_backend_templates
from aiconfigurator.generator.request import ModelFacts, SweeperCandidateError, from_sweeper_candidate

from .test_fpm_artifacts import _params as _fpm_params
from .test_sweeper_request import _agg_candidate, _cli_flag_value, _disagg_candidate

pytestmark = pytest.mark.unit

_TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "aiconfigurator" / "generator" / "config" / "backend_templates"
)
_WORKLOAD = {"isl": 4096, "osl": 512}


def _flags(cli_args: str) -> list[str]:
    return shlex.split(cli_args)


def _render_template(backend: str, name: str, **context) -> str:
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_ROOT / backend)))
    return env.get_template(name).render(**context)


# --------------------------------------------------------------------------- #
# The decision point
# --------------------------------------------------------------------------- #


def test_both_knobs_at_one_render_nothing():
    assert context_parallel_params(backend="vllm", backend_version="0.20.1") == {}
    assert context_parallel_params(backend="trtllm", backend_version="1.3.0rc14") == {}
    assert context_parallel_gpu_multiplier(1) == 1


def test_prefill_cp_grows_the_worker_and_decode_cp_does_not():
    assert context_parallel_gpu_multiplier(4) == 4
    params = context_parallel_params(backend="vllm", backend_version="0.20.1", decode_context_parallel_size=8)
    assert params == {"decode_context_parallel_size": 8}


@pytest.mark.parametrize(
    ("backend", "version", "kwargs", "expected"),
    [
        ("vllm", "0.12.0", {"context_parallel_size": 2}, {"context_parallel_size": 2}),
        ("vllm", "0.10.2", {"decode_context_parallel_size": 4}, {"decode_context_parallel_size": 4}),
        (
            "vllm",
            "0.18.0",
            {"decode_context_parallel_size": 4, "dcp_comm_backend": "a2a"},
            {"decode_context_parallel_size": 4, "dcp_comm_backend": "a2a"},
        ),
        (
            "sglang",
            "0.5.15",
            {"context_parallel_size": 2, "architecture": "Qwen3ForCausalLM"},
            {"context_parallel_size": 2, "cp_strategy": "zigzag"},
        ),
        ("sglang", "0.5.15", {"decode_context_parallel_size": 2}, {"decode_context_parallel_size": 2}),
        (
            "sglang",
            "0.5.17",
            {"decode_context_parallel_size": 2, "dcp_comm_backend": "fi_a2a"},
            {"decode_context_parallel_size": 2, "dcp_comm_backend": "fi_a2a"},
        ),
    ],
)
def test_minimum_versions_are_inclusive(backend, version, kwargs, expected):
    assert context_parallel_params(backend=backend, backend_version=version, **kwargs) == expected


@pytest.mark.parametrize(
    ("backend", "version", "kwargs", "needle"),
    [
        ("vllm", "0.11.0", {"context_parallel_size": 2}, ">= 0.12.0"),
        ("vllm", "0.10.1", {"decode_context_parallel_size": 2}, ">= 0.10.2"),
        ("vllm", "0.17.0", {"decode_context_parallel_size": 2, "dcp_comm_backend": "a2a"}, ">= 0.18.0"),
        ("sglang", "0.5.14", {"context_parallel_size": 2}, ">= 0.5.15"),
        ("sglang", "0.5.11", {"decode_context_parallel_size": 2}, ">= 0.5.15"),
        ("sglang", "0.5.16", {"decode_context_parallel_size": 2, "dcp_comm_backend": "a2a"}, ">= 0.5.17"),
        ("trtllm", "1.3.0rc14", {"decode_context_parallel_size": 2}, "Helix"),
        ("trtllm", "1.3.0rc14", {"context_parallel_size": 2}, "no launch flag"),
        ("vllm", None, {"decode_context_parallel_size": 2}, "parseable"),
    ],
)
def test_unsupported_backend_or_version_fails_loudly(backend, version, kwargs, needle):
    with pytest.raises(ContextParallelUnsupportedError, match=needle):
        context_parallel_params(backend=backend, backend_version=version, **kwargs)


def test_dcp_comm_backend_is_validated_per_backend():
    with pytest.raises(ContextParallelUnsupportedError, match="not a vllm choice"):
        context_parallel_params(
            backend="vllm", backend_version="0.20.1", decode_context_parallel_size=2, dcp_comm_backend="fi_a2a"
        )
    with pytest.raises(ContextParallelUnsupportedError, match="needs decode_context_parallel_size > 1"):
        context_parallel_params(backend="vllm", backend_version="0.20.1", dcp_comm_backend="a2a")


@pytest.mark.parametrize(
    ("architecture", "family", "expected"),
    [
        ("Qwen3ForCausalLM", None, "zigzag"),
        ("DeepseekV3ForCausalLM", "DEEPSEEK", "zigzag"),
        ("DeepseekV32ForCausalLM", None, "interleave"),
        ("GlmMoeDsaForCausalLM", None, "interleave"),
        (None, "DEEPSEEKV32", "interleave"),
    ],
)
def test_sglang_cp_strategy_follows_dsa(architecture, family, expected):
    assert cp_strategy_for(architecture, family) == expected


# --------------------------------------------------------------------------- #
# Versioned templates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("version", ["0.14.1", "0.16.0", "0.19.0", "0.20.1"])
def test_vllm_templates_render_both_cp_flags(version):
    out = _render_template(
        "vllm",
        f"cli_args.{version}.j2",
        vllm={"tensor-parallel-size": 8, "prefill-context-parallel-size": 2, "decode-context-parallel-size": 8},
        speculative_config={},
    )
    tokens = _flags(out)
    assert _cli_flag_value(out, "--prefill-context-parallel-size") == "2"
    assert _cli_flag_value(out, "--decode-context-parallel-size") == "8"
    assert "--dcp-comm-backend" not in tokens


@pytest.mark.parametrize(("version", "rendered"), [("0.16.0", False), ("0.19.0", True), ("0.20.1", True)])
def test_vllm_dcp_comm_backend_only_from_0_18(version, rendered):
    out = _render_template(
        "vllm",
        f"cli_args.{version}.j2",
        vllm={"decode-context-parallel-size": 8, "dcp-comm-backend": "a2a"},
        speculative_config={},
    )
    assert ("--dcp-comm-backend" in _flags(out)) is rendered


def test_vllm_template_is_silent_without_cp():
    out = _render_template("vllm", "cli_args.0.20.1.j2", vllm={"tensor-parallel-size": 2}, speculative_config={})
    for flag in ("--prefill-context-parallel-size", "--decode-context-parallel-size", "--dcp-comm-backend"):
        assert flag not in out


@pytest.mark.parametrize("version", ["0.5.15", "0.5.17"])
def test_sglang_templates_render_prefill_and_decode_cp(version):
    out = _render_template(
        "sglang",
        f"cli_args.{version}.j2",
        sglang={
            "tensor-parallel-size": 8,
            "attn-cp-size": 4,
            "cp-strategy": "zigzag",
            "dcp-size": 2,
            "dcp-comm-backend": "a2a",
        },
    )
    tokens = _flags(out)
    assert "--enable-prefill-cp" in tokens
    assert _cli_flag_value(out, "--attn-cp-size") == "4"
    assert _cli_flag_value(out, "--cp-strategy") == "zigzag"
    assert _cli_flag_value(out, "--dcp-size") == "2"
    assert ("--dcp-comm-backend" in tokens) is (version == "0.5.17")


def test_sglang_0_5_11_template_has_no_cp_lines():
    # The pre-0.5.15 template must stay untouched: the decision point refuses
    # the knob there instead of the template silently dropping it.
    out = _render_template(
        "sglang", "cli_args.0.5.11.j2", sglang={"tensor-parallel-size": 8, "attn-cp-size": 4, "dcp-size": 2}
    )
    assert "--attn-cp-size" not in out
    assert "--dcp-size" not in out


# --------------------------------------------------------------------------- #
# Sweeper candidate path (rule + mapping + template, per backend)
# --------------------------------------------------------------------------- #


def test_vllm_agg_decode_cp_renders_flag_without_extra_gpus():
    candidate = _agg_candidate(tp=8, replicas=1, used_gpus=8, dcp=8)
    request = from_sweeper_candidate(
        candidate,
        workload=_WORKLOAD,
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
    )
    assert request.topology.roles["agg"].extra["gpus_per_worker"] == 8
    assert request.topology.roles["agg"].extra["decode_context_parallel_size"] == 8

    cli = generate_from_request(request)["cli_args_agg"]
    assert _cli_flag_value(cli, "--tensor-parallel-size") == "8"
    assert _cli_flag_value(cli, "--decode-context-parallel-size") == "8"
    assert "--prefill-context-parallel-size" not in _flags(cli)


def test_vllm_agg_prefill_cp_multiplies_worker_gpus():
    candidate = _agg_candidate(tp=2, replicas=4, used_gpus=16, cp=2)
    request = from_sweeper_candidate(
        candidate,
        workload=_WORKLOAD,
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
    )
    assert request.topology.roles["agg"].extra["gpus_per_worker"] == 4

    artifacts = generate_from_request(request)
    cli = artifacts["cli_args_agg"]
    # vLLM's PCP expands the world size; TP stays the weight shard.
    assert _cli_flag_value(cli, "--tensor-parallel-size") == "2"
    assert _cli_flag_value(cli, "--prefill-context-parallel-size") == "2"
    assert "--decode-context-parallel-size" not in _flags(cli)


def test_vllm_prefill_cp_gpu_mismatch_is_rejected():
    # used_gpus still counts tp*replicas only: the candidate is inconsistent.
    candidate = _agg_candidate(tp=2, replicas=4, used_gpus=8, cp=2)
    with pytest.raises(SweeperCandidateError):
        from_sweeper_candidate(candidate, workload=_WORKLOAD, model_facts=ModelFacts(is_moe=False))


def test_vllm_too_old_for_prefill_cp_is_rejected():
    candidate = _agg_candidate(backend_version="0.11.0", tp=2, replicas=4, used_gpus=16, cp=2)
    with pytest.raises(SweeperCandidateError, match=">= 0.12.0"):
        from_sweeper_candidate(candidate, workload=_WORKLOAD, model_facts=ModelFacts(is_moe=False))


def _sglang_dense_disagg(**overrides):
    base = {
        "model_name": "Qwen/Qwen3-32B-FP8",
        "backend": "sglang",
        "backend_version": "0.5.16",
        "prefill_tp": 2,
        "prefill_moe_tp": 1,
        "prefill_moe_ep": 1,
        "prefill_replicas": 1,
        "prefill_cp": 4,
        "decode_tp": 4,
        "decode_moe_tp": 1,
        "decode_moe_ep": 1,
        "decode_replicas": 1,
        "decode_dcp": 4,
        "used_gpus": 12,
        # The shared disagg fixture carries a TRT-LLM-only KVBM adapter.
        "adapters": {"dynamo.router": {"mode": "kv"}},
    }
    base.update(overrides)
    return _disagg_candidate(**base)


def test_sglang_disagg_folds_prefill_cp_into_tp_and_stripes_decode():
    request = from_sweeper_candidate(
        _sglang_dense_disagg(),
        workload=_WORKLOAD,
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
    )
    prefill = request.topology.roles["prefill"].extra
    decode = request.topology.roles["decode"].extra
    assert prefill["gpus_per_worker"] == 8
    assert prefill["context_parallel_size"] == 4
    assert prefill["cp_strategy"] == "zigzag"
    assert decode["gpus_per_worker"] == 4
    assert decode["decode_context_parallel_size"] == 4

    artifacts = generate_from_request(request)
    p_cli = artifacts["cli_args_prefill"]
    d_cli = artifacts["cli_args_decode"]
    # SGLang's attention-CP ranks live inside --tp: tp(2) x cp(4) -> 8.
    assert _cli_flag_value(p_cli, "--tensor-parallel-size") == "8"
    assert "--enable-prefill-cp" in _flags(p_cli)
    assert _cli_flag_value(p_cli, "--attn-cp-size") == "4"
    assert _cli_flag_value(p_cli, "--cp-strategy") == "zigzag"
    assert "--dcp-size" not in _flags(p_cli)

    assert _cli_flag_value(d_cli, "--tensor-parallel-size") == "4"
    assert _cli_flag_value(d_cli, "--dcp-size") == "4"
    for flag in ("--enable-prefill-cp", "--attn-cp-size", "--cp-strategy"):
        assert flag not in _flags(d_cli)


def test_sglang_moe_prefill_cp_uses_moe_fold_and_interleave_for_dsa():
    candidate = _agg_candidate(
        model_name="deepseek-ai/DeepSeek-V3.2",
        backend="sglang",
        backend_version="0.5.16",
        hardware_sku="gb200",
        tp=1,
        moe_tp=1,
        moe_ep=8,
        cp=8,
        replicas=1,
        used_gpus=8,
    )
    request = from_sweeper_candidate(
        candidate,
        workload=_WORKLOAD,
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV32ForCausalLM"),
    )
    assert request.topology.roles["agg"].extra["cp_strategy"] == "interleave"

    cli = generate_from_request(request)["cli_args_agg"]
    # MoE: TP = moe_tp * moe_ep already equals the attention-group width; the
    # dense-only CP fold must not double it.
    assert _cli_flag_value(cli, "--tensor-parallel-size") == "8"
    assert _cli_flag_value(cli, "--attn-cp-size") == "8"
    assert _cli_flag_value(cli, "--cp-strategy") == "interleave"
    assert "--enable-prefill-cp" in _flags(cli)


def test_sglang_too_old_for_decode_cp_is_rejected():
    with pytest.raises(SweeperCandidateError, match=">= 0.5.15"):
        from_sweeper_candidate(
            _sglang_dense_disagg(backend_version="0.5.11"),
            workload=_WORKLOAD,
            model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        )


def test_trtllm_decode_cp_is_rejected():
    candidate = _disagg_candidate(decode_dcp=4)
    with pytest.raises(SweeperCandidateError, match="Helix"):
        from_sweeper_candidate(
            candidate,
            workload=_WORKLOAD,
            model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
        )


def test_candidates_without_cp_columns_are_unchanged():
    request = from_sweeper_candidate(
        _agg_candidate(),
        workload=_WORKLOAD,
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
    )
    extra = request.topology.roles["agg"].extra
    assert extra["gpus_per_worker"] == 2
    for key in ("context_parallel_size", "decode_context_parallel_size", "cp_strategy", "dcp_comm_backend"):
        assert key not in extra


# --------------------------------------------------------------------------- #
# SDK result bridge path
# --------------------------------------------------------------------------- #


def _task(**overrides) -> SimpleNamespace:
    fields = {
        "primary_backend_name": "sglang",
        "primary_system_name": "gb200",
        "primary_backend_version": "0.5.16",
        "primary_model_path": "Qwen/Qwen3-32B-FP8",
        "prefix": 0,
        "is_moe": False,
        "nextn": 0,
        "nextn_accepted": None,
        "serving_mode": "agg",
        "total_gpus": 0,
        "system_name": "gb200",
        "prefill_system_name": "gb200",
        "decode_system_name": "gb200",
        "isl": 1024,
        "osl": 256,
        "ttft": 2000.0,
        "tpot": 50.0,
        "attention_backend": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_bridge_reads_prefill_cp_from_result_row():
    row = pd.Series({"workers": 1, "tp": 2, "pp": 1, "dp": 1, "cp": 4, "bs": 64})

    result = task_config_to_generator_config(_task(), row, num_gpus_per_node=8)
    agg = result["params"]["agg"]
    assert agg["context_parallel_size"] == 4
    assert agg["cp_strategy"] == "zigzag"
    assert agg["gpus_per_worker"] == 8
    assert "decode_context_parallel_size" not in agg

    artifacts = generate_backend_artifacts(result, "sglang", backend_version="0.5.16", deployment_target="dynamo-j2")
    cli = artifacts["cli_args_agg"]
    assert _cli_flag_value(cli, "--tensor-parallel-size") == "8"
    assert _cli_flag_value(cli, "--attn-cp-size") == "4"
    assert "--enable-prefill-cp" in _flags(cli)


def test_bridge_reads_decode_cp_from_task():
    row = pd.Series({"workers": 1, "tp": 4, "pp": 1, "dp": 1, "bs": 64})

    task = _task(primary_backend_version="0.5.17", dcp_size=4, dcp_comm="a2a")
    result = task_config_to_generator_config(task, row, num_gpus_per_node=8)
    agg = result["params"]["agg"]
    assert agg["decode_context_parallel_size"] == 4
    assert agg["dcp_comm_backend"] == "a2a"
    assert agg["gpus_per_worker"] == 4
    assert "context_parallel_size" not in agg

    artifacts = generate_backend_artifacts(result, "sglang", backend_version="0.5.17", deployment_target="dynamo-j2")
    cli = artifacts["cli_args_agg"]
    assert _cli_flag_value(cli, "--tensor-parallel-size") == "4"
    assert _cli_flag_value(cli, "--dcp-size") == "4"
    assert _cli_flag_value(cli, "--dcp-comm-backend") == "a2a"


def test_bridge_disagg_uses_per_role_decode_cp_and_prefill_cp_column():
    row = pd.Series(
        {
            "(p)workers": 1,
            "(p)tp": 1,
            "(p)pp": 1,
            "(p)dp": 1,
            "(p)cp": 2,
            "(p)bs": 8,
            "(d)workers": 1,
            "(d)tp": 4,
            "(d)pp": 1,
            "(d)dp": 1,
            "(d)bs": 64,
        }
    )
    task = _task(
        primary_backend_name="vllm",
        primary_backend_version="0.20.1",
        serving_mode="disagg",
        prefill_dcp_size=1,
        decode_dcp_size=4,
    )

    result = task_config_to_generator_config(task, row, num_gpus_per_node=8)
    prefill = result["params"]["prefill"]
    decode = result["params"]["decode"]
    assert prefill["context_parallel_size"] == 2
    assert prefill["gpus_per_worker"] == 2
    assert "decode_context_parallel_size" not in prefill
    assert decode["decode_context_parallel_size"] == 4
    assert decode["gpus_per_worker"] == 4
    assert "context_parallel_size" not in decode


def test_bridge_without_cp_is_unchanged():
    row = pd.Series({"workers": 1, "tp": 2, "pp": 1, "dp": 1, "bs": 64})
    agg = task_config_to_generator_config(_task(), row, num_gpus_per_node=8)["params"]["agg"]
    assert agg["gpus_per_worker"] == 2
    for key in ("context_parallel_size", "decode_context_parallel_size", "cp_strategy", "dcp_comm_backend"):
        assert key not in agg


def test_bridge_rejects_cp_on_trtllm():
    row = pd.Series({"workers": 1, "tp": 2, "pp": 1, "dp": 1, "bs": 64})
    task = _task(primary_backend_name="trtllm", primary_backend_version="1.3.0rc14", dcp_size=2)
    with pytest.raises(ContextParallelUnsupportedError, match="Helix"):
        task_config_to_generator_config(task, row, num_gpus_per_node=8)


# --------------------------------------------------------------------------- #
# FPM target world-size check
# --------------------------------------------------------------------------- #


def _fpm_with(**agg_overrides) -> dict:
    params = copy.deepcopy(_fpm_params())
    params["params"]["agg"].update(agg_overrides)
    params["WorkerConfig"]["agg_gpus_per_worker"] = params["params"]["agg"]["gpus_per_worker"]
    return params


def _fpm_engine_command(run_sh: str) -> list[str]:
    line = next(line for line in run_sh.splitlines() if line.startswith("engine_command=("))
    return shlex.split(line[len("engine_command=(") : -1])


def test_fpm_counts_prefill_cp_in_world_size():
    params = _fpm_with(tensor_parallel_size=2, gpus_per_worker=4, context_parallel_size=2)
    artifacts = render_backend_templates(params, "vllm", version="0.20.1", deployment_target="fpm")
    command = _fpm_engine_command(artifacts[FPM_RUN_SCRIPT_FILENAME])
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--prefill-context-parallel-size") + 1] == "2"


def test_fpm_decode_cp_adds_no_gpus():
    params = _fpm_with(tensor_parallel_size=4, gpus_per_worker=4, decode_context_parallel_size=4)
    artifacts = render_backend_templates(params, "vllm", version="0.20.1", deployment_target="fpm")
    command = _fpm_engine_command(artifacts[FPM_RUN_SCRIPT_FILENAME])
    assert command[command.index("--decode-context-parallel-size") + 1] == "4"
    assert "--prefill-context-parallel-size" not in command


def test_fpm_topology_check_multiplies_prefill_cp():
    from aiconfigurator.generator.builders.fpm_builder import _resolve_topology

    worker = SimpleNamespace(resources={"limits": {"gpu": "4"}}, multinode=None)
    context = {"NodeConfig": {"num_gpus_per_node": 8}}

    topology = _resolve_topology(
        context, worker, ["--tensor-parallel-size", "2", "--prefill-context-parallel-size", "2"]
    )
    assert topology["total_gpus"] == 4

    with pytest.raises(ValueError, match=r"pcp\(1\)"):
        _resolve_topology(context, worker, ["--tensor-parallel-size", "2"])

    # Decode CP reuses the TP ranks: it must not be counted.
    with pytest.raises(ValueError, match="does not match"):
        _resolve_topology(context, worker, ["--tensor-parallel-size", "2", "--decode-context-parallel-size", "2"])
