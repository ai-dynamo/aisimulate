# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real CLI parsing, YAML model selection and worker dispatch on CPU."""

import argparse
import ast
import gc
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.case_generator import get_mla_module_model_specs, get_mla_module_precision_specs
from collector.registry_types import PerfFile

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector/vllm/collect_mla_module.py"


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
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

    sweep = SimpleNamespace(
        inner_sweep_head_counts=[8],
        context_batch_sizes=[1],
        context_sequence_lengths=[128],
        context_max_tokens=8192,
        context_prefix_lengths=[0, 131072],
        generation_batch_sizes=[1],
        generation_sequence_lengths=[8193],
        generation_max_tokens=2**25,
    )
    namespace = {
        "argparse": argparse,
        "gc": gc,
        "traceback": traceback,
        "PerfFile": PerfFile,
        "get_mla_module_model_specs": get_mla_module_model_specs,
        "get_mla_module_precision_specs": get_mla_module_precision_specs,
        "get_mla_module_sweep_spec": lambda backend: sweep,
        "get_sm_version": lambda: 100,
        "_device_total_memory_bytes": lambda: None,
        "run_mla_module": measure,
        "torch": SimpleNamespace(
            inference_mode=inference_mode,
            cuda=SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=lambda: None),
        ),
    }
    # vLLM is absent in CPU CI. Execute the actual entrypoint and its actual
    # helpers; only the native GPU call and sweep size are replaced. Model and
    # precision selection still read the real case YAML.
    names = {
        "_supported_model_map",
        "_get_precision_combos",
        "get_context_test_cases",
        "get_generation_test_cases",
        "run_mla_module_worker",
        "main",
    }
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


def test_dsa_context_cli_passes_declared_cached_prefix_to_native_worker(cli):
    calls = cli("--mode", "context", "--model", "zai-org/GLM-5.2", "--gemm-type", "fp8", "--kv-cache-dtype", "fp8")
    assert [(row["seq_len"], row["prefix_len"]) for row in calls] == [(128, 0), (128, 131072)]
    assert all(row["batch_size"] == 1 and row["num_heads"] == 8 for row in calls)
    assert all(row["gemm_type"] == "fp8" and row["perf_filename"] == PerfFile.DSA_CONTEXT_MODULE for row in calls)


@pytest.mark.parametrize(
    "phase,model,expected_length",
    [("context", "deepseek-ai/DeepSeek-V3", 128), ("generation", "zai-org/GLM-5.2", 8193)],
)
def test_six_field_cases_keep_zero_prefix_and_their_full_coordinate(cli, phase, model, expected_length):
    calls = cli("--mode", phase, "--model", model, "--gemm-type", "bfloat16", "--kv-cache-dtype", "bfloat16")
    assert len(calls) == 1
    assert calls[0]["seq_len"] == expected_length
    assert calls[0]["prefix_len"] == 0
    assert calls[0]["model_path"] == model


def test_msa_model_is_not_routed_into_an_mla_dsa_entrypoint(cli):
    with pytest.raises(SystemExit) as result:
        cli("--mode", "context", "--model", "MiniMaxAI/MiniMax-M3", "--quick")
    assert result.value.code == 2
