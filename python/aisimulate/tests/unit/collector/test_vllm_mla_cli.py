# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real CLI parsing, YAML model selection and worker dispatch on CPU."""

import argparse
import ast
import gc
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.case_generator import get_mla_module_model_specs
from collector.op_plans import plan
from collector.registry_types import PerfFile

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_mla_module.py"


@pytest.fixture
def cli(monkeypatch):
    # The CLI targets --model the way collect.py does, through the environment; restore it.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "")
    calls = []
    inference_active = False

    @contextmanager
    def inference_mode():
        nonlocal inference_active
        assert not inference_active
        inference_active = True
        try:
            yield
        finally:
            inference_active = False

    def measure(**kwargs):
        # CLI must use the same worker scope as collect.py. This assertion
        # covers initialization as well as the later forward/timing callback.
        assert inference_active
        calls.append(kwargs)

    namespace = {
        "argparse": argparse,
        "gc": gc,
        "os": os,
        "traceback": traceback,
        "PerfFile": PerfFile,
        "get_mla_module_model_specs": get_mla_module_model_specs,
        "plan": plan,
        "get_sm_version": lambda: 100,
        "run_mla_module": measure,
        "torch": SimpleNamespace(
            inference_mode=inference_mode,
            cuda=SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=lambda: None),
        ),
    }
    # vLLM is absent in CPU CI. Execute the actual entrypoint and its actual
    # helpers; only the native GPU call is replaced. Model, precision and point
    # selection read the real op plans and case YAML.
    names = {"_supported_model_map", "_perf_file_for", "run_mla_module_worker", "main"}
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)

    def execute(*args):
        monkeypatch.setattr(sys, "argv", [str(SOURCE), *args])
        namespace["main"]()
        assert inference_active is False
        return calls

    return execute


def test_default_cli_collects_only_canonical_mla_and_dsa_references(cli):
    calls = cli("--mode", "context", "--quick")
    assert {(row["model_path"], row["attn_type"]) for row in calls} == {
        ("deepseek-ai/DeepSeek-V3", "mla"),
        ("deepseek-ai/DeepSeek-V3.2", "dsa"),
        ("zai-org/GLM-5.2", "dsa"),
    }
    assert len(calls) == 3


def test_explicit_checkpoint_alias_is_preserved(cli):
    calls = cli("--mode", "context", "--model", "nvidia/GLM-5.2-NVFP4", "--quick", "--gemm-type", "nvfp4")
    assert len(calls) == 1
    assert calls[0]["model_path"] == "nvidia/GLM-5.2-NVFP4"
    assert calls[0]["gemm_type"] == "nvfp4"


def test_dsa_context_cli_runs_the_plan_rows_of_the_targeted_checkpoint(cli):
    calls = cli(
        "--mode", "context", "--model", "zai-org/GLM-5.2", "--gemm-type", "fp8_block", "--kv-cache-dtype", "fp8"
    )
    assert calls
    assert {(row["model_path"], row["gemm_type"], row["kv_cache_dtype"]) for row in calls} == {
        ("zai-org/GLM-5.2", "fp8_block", "fp8")
    }
    assert all(row["perf_filename"] == PerfFile.DSA_CONTEXT_MODULE for row in calls)
    # Each row lists its (isl, prefix) points; the cached prefix reaches GLM-5.2's context.
    points = [tuple(point) for row in calls for point in row["seq_len"]]
    assert all(row["prefix_len"] == 0 for row in calls)
    assert (1, 1048575) in points and max(s + p for s, p in points) == 1048576


@pytest.mark.parametrize(
    "phase,model",
    [("context", "deepseek-ai/DeepSeek-V3"), ("generation", "zai-org/GLM-5.2")],
)
def test_rows_without_a_prefix_list_their_lengths(cli, phase, model):
    calls = cli("--mode", phase, "--model", model, "--gemm-type", "bfloat16", "--kv-cache-dtype", "bfloat16")
    assert calls
    assert all(isinstance(length, int) for row in calls for length in row["seq_len"])
    assert all(row["prefix_len"] == 0 and row["model_path"] == model for row in calls)


def test_msa_model_is_not_routed_into_an_mla_dsa_entrypoint(cli):
    with pytest.raises(SystemExit) as result:
        cli("--mode", "context", "--model", "MiniMaxAI/MiniMax-M3", "--quick")
    assert result.value.code == 2
