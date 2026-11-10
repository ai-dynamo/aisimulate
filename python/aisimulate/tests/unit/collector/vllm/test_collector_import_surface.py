# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-surface smoke tests for the three vllm collector modules bumped
0.24.0 -> 0.27.1 (AIC-1782 Task V1), mirroring
tests/unit/collector/sglang/test_collector_import_surface.py's strict-fake
pattern and its origin incident (a module-level surface that broke on the
first real 0.5.17 run, invisible to a lenient-mock test).

These tests use STRICT fakes: every ``vllm.*`` module is a plain
``types.ModuleType`` populated with only the exact names the collector
touches -- ``types.ModuleType`` raises ``AttributeError`` on anything not
explicitly set (converted to ``ImportError`` by ``from X import Y`` for a
missing name), the same failure mode the real framework produces for a
genuinely absent attribute. Two shapes are modeled per module, derived from
the sweep documented in each collect_*.py file's module-level comment block:
one matching vLLM 0.24.0's real surface, one matching 0.27.1's. Only names
that actually differ between versions are called out per-shape; everything
unchanged between versions is shared.

``torch``/``collector.vllm.utils`` are faked permissively (a bare
``types.SimpleNamespace`` for the latter, ``MagicMock`` for the former) --
neither is the subject of this bug class. ``collector/vllm/utils.py``'s own
module-level vllm imports (CacheConfig, ModelConfig, VllmConfig,
init_distributed_environment, ensure_model_parallel_initialized, cdiv,
kv_cache_dtype_str_to_dtype, AttentionBackendEnum, CommonAttentionMetadata,
FullAttentionSpec, SlidingWindowSpec, get_kv_quant_mode, ...) were
independently verified unchanged at v0.27.1 as part of this task's research
(collect_gemm.py imports from it at module level, collect_moe.py inside
run_moe_torch) -- stubbing it out here keeps each test focused on the
collector file actually under test, not on re-proving that separately-
verified surface every time.
"""

import contextlib
import functools
import importlib.util
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[4]
COLLECTOR_DIR = REPO_ROOT / "collector"


# ---------------------------------------------------------------------------
# Shared strict-fake-tree infrastructure (mirrors the sglang test file)
# ---------------------------------------------------------------------------
def _build_module_tree(specs: dict[str, dict]) -> dict[str, types.ModuleType]:
    """Build a strict fake module tree: every dotted path in `specs`, plus
    every implied parent prefix, becomes a plain ``types.ModuleType``.
    Pure ``sys.modules`` injection (no ``__path__``, no parent-attribute
    wiring) is sufficient for both ``import a.b.c as x`` and
    ``from a.b.c import y`` to resolve (verified for the sglang round,
    2026-08-18; reused unchanged here)."""
    modules: dict[str, types.ModuleType] = {}

    def ensure(path: str) -> types.ModuleType:
        if path not in modules:
            modules[path] = types.ModuleType(path)
        return modules[path]

    for path, attrs in specs.items():
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            ensure(".".join(parts[:i]))
        leaf = modules[path]
        for name, value in attrs.items():
            setattr(leaf, name, value)
    return modules


class _Dummy:
    """Stand-in for a vllm class the collector only constructs/references,
    never subclasses or introspects at import time."""

    def __init__(self, *args, **kwargs):
        pass


def _dummy(*args, **kwargs):
    return None


class _NullContext:
    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_null_context = _NullContext()


def _install_fake_vllm(monkeypatch, specs: dict[str, dict]) -> None:
    for path, module in _build_module_tree(specs).items():
        monkeypatch.setitem(sys.modules, path, module)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    # See module docstring: collector/vllm/utils.py's own vllm surface is
    # verified separately; stub it so each test here stays focused.
    monkeypatch.setitem(
        sys.modules,
        "collector.vllm.utils",
        types.SimpleNamespace(setup_distributed=_dummy, with_exit_stack=lambda f: f),
    )


def _import_fresh(monkeypatch, dotted_name: str, source_path: Path) -> types.ModuleType:
    """Exec `source_path` under module name `dotted_name`, bypassing any
    cached entry, and let import-time exceptions propagate to the caller."""
    monkeypatch.delitem(sys.modules, dotted_name, raising=False)
    spec = importlib.util.spec_from_file_location(dotted_name, source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, dotted_name, module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# collect_moe.py -- the one surface that actually moved (FusedMoE ->
# FusedMoEFactory rename inside run_moe_torch())
# ---------------------------------------------------------------------------
_MOE_COMMON_SURFACE = {
    "vllm.version": {"__version__": "0.0.0"},
    "vllm.config": {"VllmConfig": _Dummy, "set_current_vllm_config": _null_context},
    "vllm.forward_context": {"get_forward_context": _dummy, "set_forward_context": _null_context},
    "vllm.model_executor.layers.fused_moe.experts.fallback": {"FallbackExperts": _Dummy},
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors": {
        "CompressedTensorsConfig": _Dummy
    },
    "vllm.model_executor.layers.quantization.fp8": {"Fp8Config": _Dummy},
    "vllm.model_executor.layers.quantization.modelopt": {"ModelOptFp8Config": _Dummy},
    "vllm.model_executor.layers.quantization.mxfp4": {"Mxfp4Config": _Dummy},
    "vllm.v1.worker.workspace": {"init_workspace_manager": _dummy},
}

# vllm.model_executor.layers.fused_moe.layer: the module-level FusedMoE
# factory FUNCTION at 0.24.0 (layer.py:102-103, flagged "# TODO: rename
# this" immediately above its def), renamed to FusedMoEFactory at 0.27.1
# (layer.py:99, TODO removed; fused_moe/__init__.py exports FusedMoEFactory,
# not FusedMoE -- no back-compat alias at either version).
VLLM_0240_MOE_SURFACE = {
    **_MOE_COMMON_SURFACE,
    "vllm.model_executor.layers.fused_moe.layer": {"FusedMoE": _dummy},
}
VLLM_0271_MOE_SURFACE = {
    **_MOE_COMMON_SURFACE,
    "vllm.model_executor.layers.fused_moe.layer": {"FusedMoEFactory": _dummy},
}
# The surface collect_moe.py targets (vLLM 0.32.0): FusedMoEFactory with no
# FusedMoE fallback, MoEActivation for the non-gated ckpt_names
# (fused_moe/activation.py:21,47-54,72), and the platform/FlashInfer helpers
# its warmup-autotune parity reads (platforms/interface.py:531,
# utils/flashinfer.py:88).
VLLM_0320_MOE_SURFACE = {
    **VLLM_0271_MOE_SURFACE,
    "vllm.model_executor.layers.fused_moe.activation": {"MoEActivation": _Dummy},
    "vllm.platforms": {"current_platform": _Dummy()},
    "vllm.utils.flashinfer": {"has_flashinfer": _dummy},
}

_COLLECT_MOE_PATH = COLLECTOR_DIR / "vllm" / "collect_moe.py"
_COLLECT_MOE_DOTTED = "collector.vllm.collect_moe"

# run_moe_torch() runs its whole vllm import preamble (10 statements, 9 of
# them vllm-rooted) as the first lines of the function body, strictly before
# any other code -- including this collector-owned, vllm-independent guard.
# Requesting moe_tp_size>1 AND moe_ep_size>1 together makes the FIRST line
# after the imports raise this exact, deterministic ValueError, so reaching
# it (rather than an ImportError/AttributeError from a fake module) proves
# every import in the preamble resolved.
_SENTINEL_KWARGS = dict(moe_tp_size=2, moe_ep_size=2, model_name="x", perf_filename="x")
_SENTINEL_ARGS = ("bfloat16", [1], 1, 1, 1, 1)
_SENTINEL_MATCH = "does not combine logical TP and EP"

# Verbatim pre-fix excerpt: collector/vllm/collect_moe.py lines 322-354 at
# commit 86f07df2dd3df12179238e62f3480c7ac6ad83ee (HEAD at the start of
# AIC-1782 Task V1, before run_moe_torch()'s unconditional FusedMoE import
# gained its version-conditional FusedMoEFactory fallback), truncated right
# after the TP/EP sentinel guard -- nothing past the guard is reachable in
# these tests. "vs 0.24.0-shaped" on this exact pre-fix code proves the fix
# is version-specific, not a general break. Embedded as a string constant
# instead of a `git show` read: that commit is a branch-head object that
# squash-merge-only main and base-branch rebases leave unreachable on fresh
# clones, which turned the sibling sglang red-proof's `git show` into a hard
# CalledProcessError in CI.
_PRE_FIX_MOE_SNIPPET = '''\
def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0.0,
    *,
    perf_filename,
    device="cuda:0",
):
    """Benchmark the vLLM 0.24.0 model-execution MoE path."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
    from vllm.v1.worker.workspace import init_workspace_manager

    from collector.vllm.utils import setup_distributed

    if moe_tp_size > 1 and moe_ep_size > 1:
        raise ValueError("vLLM MoE collector does not combine logical TP and EP")
'''


class TestCollectMoeImportSurface:
    def test_imports_cleanly_against_0320_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, VLLM_0320_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)
        with pytest.raises(ValueError, match=_SENTINEL_MATCH):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)

    def test_one_release_per_file_drops_the_fusedmoe_fallback(self, monkeypatch):
        """collect_moe.py targets one release: a FusedMoE-only (0.24.0-shaped)
        layer module no longer resolves, instead of silently binding the
        pre-rename name."""
        _install_fake_vllm(
            monkeypatch,
            {**VLLM_0320_MOE_SURFACE, "vllm.model_executor.layers.fused_moe.layer": {"FusedMoE": _dummy}},
        )
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)
        with pytest.raises(ImportError, match="FusedMoEFactory"):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)

    def test_0271_shaped_fake_reproduces_a_break_on_prefix_code(self, monkeypatch, tmp_path):
        """Red-proof, per the sglang round's precedent: exec the embedded
        verbatim pre-fix excerpt (_PRE_FIX_MOE_SNIPPET, unconditional
        ``from vllm...import FusedMoE``) against the 0.27.1-shaped fake
        (which has no ``FusedMoE`` name) and confirm it fails on import,
        not an assumption about what "should" happen."""
        scratch_file = tmp_path / "collect_moe.py"
        scratch_file.write_text(_PRE_FIX_MOE_SNIPPET)

        _install_fake_vllm(monkeypatch, VLLM_0271_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, scratch_file)

        with pytest.raises(ImportError, match="FusedMoE"):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)

    def test_prefix_code_was_fine_against_0240_shaped_vllm(self, monkeypatch, tmp_path):
        """Companion to the test above: the pre-fix code was never broken in
        general, only against 0.27.1's shape -- confirms the break (and this
        regression test) is version-specific."""
        scratch_file = tmp_path / "collect_moe.py"
        scratch_file.write_text(_PRE_FIX_MOE_SNIPPET)

        _install_fake_vllm(monkeypatch, VLLM_0240_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, scratch_file)

        with pytest.raises(ValueError, match=_SENTINEL_MATCH):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)

    def test_load_model_moe_config_raises_for_radixark_nvfp4_missing_config(self, monkeypatch):
        """Executable proof for the RadixArk vllm nvfp4 gate decision
        (AIC-1782 Task V2, case YAML comment on that row): no packaged HF
        config exists for this model_path anywhere in the repo (unlike the
        base/-FP8 ids, both of which have one under
        src/aisimulate_core/model_configs/), so ``_load_model_moe_config``
        raises before the row could ever be benchmarked. Opening the vllm
        nvfp4 gate on this row without also adding that (cross-module,
        human-approved) config would crash vLLM MoE case generation for
        every model on the run, not just this one:
        ``get_moe_test_cases()`` reaches this function via
        ``_moe_execution_key() -> _resolve_moe_runtime_config()``
        (collect_moe.py:321 -> :251 -> :113) DURING CASE ENUMERATION,
        before any per-case classified-failure handling applies."""
        _install_fake_vllm(monkeypatch, VLLM_0320_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)
        with pytest.raises(FileNotFoundError, match=re.escape("RadixArk/Qwen3.8-2.4T-A95B-NVFP4")):
            module._load_model_moe_config("RadixArk/Qwen3.8-2.4T-A95B-NVFP4")

    def test_load_model_moe_config_succeeds_for_base_and_fp8_qwen38_max_ids(self, monkeypatch):
        """Companion: the base bf16 id and its packaged -FP8 alias id both
        DO have a packaged config (proves the RadixArk gap above is a real,
        specific absence, not a general fixture problem)."""
        _install_fake_vllm(monkeypatch, VLLM_0320_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)
        for model_name in ("Qwen/Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-2.4T-A95B-FP8"):
            config = module._load_model_moe_config(model_name)
            assert config.get("model_type") == "qwen3_5_moe_text"


# ---------------------------------------------------------------------------
# collect_gemm.py -- byte-identical framework surface at both pinned
# versions (no version branching needed); forward-looking regression guard
# ---------------------------------------------------------------------------
_GEMM_SURFACE = {
    "vllm.envs": {"VLLM_BATCH_INVARIANT": False},
    "vllm.version": {"__version__": "0.0.0"},
    "vllm._custom_ops": {"scaled_fp4_quant": _dummy},
    "vllm.config": {"VllmConfig": _Dummy, "set_current_vllm_config": _null_context},
    "vllm.model_executor.kernels.linear.scaled_mm.flashinfer": {
        "FlashInferFp8DeepGEMMDynamicBlockScaledKernel": _Dummy
    },
    "vllm.model_executor.layers.linear": {"RowParallelLinear": _Dummy},
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors": {
        "CompressedTensorsConfig": _Dummy
    },
    "vllm.model_executor.layers.quantization.fp8": {"Fp8Config": _Dummy},
    "vllm.platforms": {"current_platform": _Dummy()},
    "vllm.utils.deep_gemm": {"per_block_cast_to_fp8": _dummy},
    "vllm.utils.flashinfer": {"has_flashinfer": _dummy},
}

_COLLECT_GEMM_PATH = COLLECTOR_DIR / "vllm" / "collect_gemm.py"
_COLLECT_GEMM_DOTTED = "collector.vllm.collect_gemm"


class TestCollectGemmImportSurface:
    """Every surface collect_gemm.py touches -- directly and via
    collector/vllm/utils.py -- was found byte-identical or line-shift-only
    between v0.24.0 and v0.27.1 (this task's research); one fake tree
    therefore covers both pinned versions. These are the forward-looking
    regression guard, not a red/pre-fix reproduction -- there is no fix to
    prove here, unlike collect_moe.py above."""

    def test_imports_cleanly_against_0240_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, _GEMM_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)

    def test_imports_cleanly_against_0271_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, _GEMM_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)


# ---------------------------------------------------------------------------
# collect_gdn.py -- vllm.model_executor.layers.fla relocated wholesale to
# the new top-level vllm.third_party.flash_linear_attention package
# ---------------------------------------------------------------------------
_GDN_COMMON_SURFACE = {
    "vllm.version": {"__version__": "0.0.0"},
    "vllm.config": {"VllmConfig": _Dummy, "set_current_vllm_config": _null_context},
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn": {"ChunkGatedDeltaRule": _Dummy},
    "vllm.model_executor.layers.mamba.ops.causal_conv1d": {
        "causal_conv1d_fn": _dummy,
        "causal_conv1d_update": _dummy,
    },
    "vllm.v1.attention.backends.gdn_attn": {"GDNAttentionMetadata": _Dummy},
    "vllm.v1.attention.backends.utils": {"compute_causal_conv1d_metadata": _dummy},
}

# vllm.model_executor.layers.fla (0.24.0) relocated wholesale to the new
# top-level vllm.third_party.flash_linear_attention (0.27.1, verified via
# source clone: vllm/third_party/flash_linear_attention/ops/__init__.py);
# the only symbol this collector touches there,
# fused_recurrent_gated_delta_rule_packed_decode, is byte-identical at the
# new location and re-exported from ops/__init__.py at both versions.
VLLM_0240_GDN_SURFACE = {
    **_GDN_COMMON_SURFACE,
    "vllm.model_executor.layers.fla.ops": {"fused_recurrent_gated_delta_rule_packed_decode": _dummy},
}
VLLM_0271_GDN_SURFACE = {
    **_GDN_COMMON_SURFACE,
    "vllm.third_party.flash_linear_attention.ops": {"fused_recurrent_gated_delta_rule_packed_decode": _dummy},
}

_COLLECT_GDN_PATH = COLLECTOR_DIR / "vllm" / "collect_gdn.py"
_COLLECT_GDN_DOTTED = "collector.vllm.collect_gdn"

# Verbatim pre-fix excerpt: collector/vllm/collect_gdn.py lines 13-29 at
# commit 86f07df2dd3df12179238e62f3480c7ac6ad83ee (HEAD at the start of
# AIC-1782 Task V1, before the module-level fla import gained its
# version-conditional third_party.flash_linear_attention fallback) -- the
# complete module-level import block, i.e. the import surface under test;
# the file below this point is the aic_debug env read and function
# definitions, none reachable by these tests. Embedded as a string constant
# instead of a `git show` read (see _PRE_FIX_MOE_SNIPPET above for why
# history reads are banned here).
_PRE_FIX_GDN_SNIPPET = """\
__compat__ = "vllm==0.24.0"

import gc
import os
from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.version import __version__ as vllm_version

from collector.case_generator import get_common_gdn_test_cases
from collector.helper import benchmark_with_power, get_sm_version, log_perf
"""


class TestCollectGdnImportSurface:
    def test_imports_cleanly_against_0271_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, VLLM_0271_GDN_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, _COLLECT_GDN_PATH)

    def test_0271_shaped_fake_reproduces_a_break_on_prefix_code(self, monkeypatch, tmp_path):
        """Red-proof: the embedded verbatim pre-fix excerpt
        (_PRE_FIX_GDN_SNIPPET, unconditional import from the old
        vllm.model_executor.layers.fla.ops location) must fail on import
        against the 0.27.1-shaped fake, which only has the new
        vllm.third_party.flash_linear_attention.ops path."""
        scratch_file = tmp_path / "collect_gdn.py"
        scratch_file.write_text(_PRE_FIX_GDN_SNIPPET)

        _install_fake_vllm(monkeypatch, VLLM_0271_GDN_SURFACE)

        with pytest.raises(ImportError, match="fla"):
            _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, scratch_file)

    def test_prefix_code_was_fine_against_0240_shaped_vllm(self, monkeypatch, tmp_path):
        """Companion to the test above: the pre-fix code was never broken in
        general, only against 0.27.1's shape."""
        scratch_file = tmp_path / "collect_gdn.py"
        scratch_file.write_text(_PRE_FIX_GDN_SNIPPET)

        _install_fake_vllm(monkeypatch, VLLM_0240_GDN_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, scratch_file)


# ---------------------------------------------------------------------------
# collect_attn.py / collect_attn_encoder.py -- the vLLM 0.32.0 surface
# ---------------------------------------------------------------------------
class _StopRun(Exception):
    """Raised by a fake to end run_*_torch() at the call under test."""


class _FakeBackendEnum:
    FLASH_ATTN = "FLASH_ATTN"
    TRITON_ATTN = "TRITON_ATTN"
    TORCH_SDPA = "TORCH_SDPA"


def _with_exit_stack(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            return func(stack, *args, **kwargs)

    return wrapper


def _batch_spec(seq_lens, query_lens):
    return types.SimpleNamespace(seq_lens=seq_lens, query_lens=query_lens, compute_num_tokens=lambda: sum(query_lens))


# The names collect_attn.py takes from its vLLM-only helper module.
_ATTN_UTILS = {
    "BatchSpec": _batch_spec,
    "create_common_attn_metadata": _dummy,
    "create_kv_cache_and_block_mappings": _dummy,
    "create_standard_kv_cache_spec": _dummy,
    "create_vllm_config": _dummy,
    "framework_kv_block_size": _dummy,
    "get_attention_backend": _dummy,
    "set_attention_layer_of_model": _dummy,
    "with_exit_stack": _with_exit_stack,
}

# vLLM 0.32.0: selection goes through serving's get_attn_backend
# (v1/attention/selector.py:105), not a hand-built AttentionSelectorConfig.
VLLM_0320_ATTN_SURFACE = {
    "vllm.version": {"__version__": "0.0.0"},
    "vllm.config": {"set_current_vllm_config": _null_context},
    # The per-architecture config hooks serving applies (config/vllm.py:2629-2658);
    # the collector builds a model's config only for an architecture with one.
    "vllm.model_executor.models.config": {"MODELS_CONFIG_MAP": {}},
    "vllm.platforms": {"current_platform": _Dummy()},
    "vllm.utils.torch_utils": {"set_random_seed": _dummy},
    "vllm.v1.attention.backends.registry": {"AttentionBackendEnum": _FakeBackendEnum},
    "vllm.v1.attention.backends.utils": {
        "get_supported_kv_cache_layouts": _dummy,
        "resolve_kv_cache_layout": _dummy,
    },
    # _cached_get_attn_backend is the per-process memo behind get_attn_backend
    # (v1/attention/selector.py:203-225); the collector clears it per selection.
    "vllm.v1.attention.selector": {
        "get_attn_backend": _dummy,
        "_cached_get_attn_backend": functools.cache(_dummy),
    },
}

# vLLM 0.32.0 ViT surface. vit_flash_attn_wrapper is deliberately absent: the
# collector calls flash_attn_varlen_func as the wrapper's eager branch does.
VLLM_0320_ENCODER_SURFACE = {
    "vllm.version": {"__version__": "0.0.0"},
    "vllm.model_executor.models.vision": {"get_vit_attn_backend": _dummy},
    "vllm.v1.attention.backends.fa_utils": {
        "flash_attn_varlen_func": _dummy,
        "get_flash_attn_version": _dummy,
    },
    "vllm.v1.attention.backends.registry": {"AttentionBackendEnum": _FakeBackendEnum},
    "vllm.v1.attention.ops.vit_attn_wrappers": {
        "vit_torch_sdpa_wrapper": _dummy,
        "vit_triton_attn_wrapper": _dummy,
    },
}

_COLLECT_ATTN_PATH = COLLECTOR_DIR / "vllm" / "collect_attn.py"
_COLLECT_ATTN_DOTTED = "collector.vllm.collect_attn"
_COLLECT_ATTN_ENCODER_PATH = COLLECTOR_DIR / "vllm" / "collect_attn_encoder.py"
_COLLECT_ATTN_ENCODER_DOTTED = "collector.vllm.collect_attn_encoder"


def _import_collect_attn(monkeypatch, surface=VLLM_0320_ATTN_SURFACE, **utils_overrides):
    _install_fake_vllm(monkeypatch, surface)
    utils = types.SimpleNamespace(**{**_ATTN_UTILS, **utils_overrides})
    monkeypatch.setitem(sys.modules, "collector.vllm.utils", utils)
    return _import_fresh(monkeypatch, _COLLECT_ATTN_DOTTED, _COLLECT_ATTN_PATH)


class TestCollectAttnImportSurface:
    def test_imports_cleanly_against_0320_shaped_vllm(self, monkeypatch):
        _import_collect_attn(monkeypatch)

    @pytest.mark.parametrize(
        ("batch_size", "window_size", "use_fp8_kv_cache", "max_num_seqs", "kv_cache_dtype"),
        [(1, 0, False, 1024, "auto"), (64, 128, True, 1024, "fp8"), (2048, 0, False, 2048, "auto")],
    )
    def test_selects_like_serving_under_serving_scheduler_bound(
        self, monkeypatch, batch_size, window_size, use_fp8_kv_cache, max_num_seqs, kv_cache_dtype
    ):
        """The builders read CUDA-graph capture sizes derived from max_num_seqs,
        so the config carries serving's bound (or the case's larger batch), and
        the backend comes from serving's get_attn_backend entry point."""
        config_kwargs = {}
        selector_calls = []

        def create_vllm_config(**kwargs):
            config_kwargs.update(kwargs)
            return types.SimpleNamespace()

        def get_attn_backend(*args, **kwargs):
            selector_calls.append((args, kwargs))
            raise _StopRun

        module = _import_collect_attn(monkeypatch, create_vllm_config=create_vllm_config)
        monkeypatch.setattr(module, "get_attn_backend", get_attn_backend)
        with pytest.raises(_StopRun):
            module.run_attention_torch(
                batch_size, 64, 8, 8, 128, use_fp8_kv_cache, False, window_size, perf_filename="x"
            )

        assert config_kwargs["max_num_seqs"] == max_num_seqs
        assert config_kwargs["block_size"] is None
        ((args, kwargs),) = selector_calls
        assert args[0] == 128 and args[2] == kv_cache_dtype
        assert kwargs == {"has_sliding_window": window_size > 0}

    @pytest.mark.parametrize(
        ("window_size", "expected"),
        [(0, (None, True)), (128, ((-1, -1), False))],
    )
    def test_fa_aot_schedule_matches_hybrid_sliding_window_models(self, monkeypatch, window_size, expected):
        """Serving turns FA3's AOT scheduling off when the model's FlashAttention
        layers carry more than one window config (flash_attn.py:834-848); every
        planned sliding-window profile is such a model."""
        module = _import_collect_attn(monkeypatch)
        builder = types.SimpleNamespace(aot_sliding_window=None, aot_schedule=True)
        module.match_serving_fa_aot_schedule(builder, window_size)
        assert (builder.aot_sliding_window, builder.aot_schedule) == expected

    def test_fa_aot_schedule_leaves_other_builders_alone(self, monkeypatch):
        module = _import_collect_attn(monkeypatch)
        builder = types.SimpleNamespace(aot_schedule=True)
        module.match_serving_fa_aot_schedule(builder, 128)
        assert vars(builder) == {"aot_schedule": True}


class TestCollectAttnEncoderImportSurface:
    def test_imports_cleanly_against_0320_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, VLLM_0320_ENCODER_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_ATTN_ENCODER_DOTTED, _COLLECT_ATTN_ENCODER_PATH)

    def test_flash_attn_rows_time_the_eager_wrapper_call(self, monkeypatch):
        """vit_flash_attn_wrapper zero-fills its output under CUDA-graph capture
        (vit_attn_wrappers.py:63-67); serving's eager encoder hands
        flash_attn_varlen_func an empty preallocated buffer, and so must the
        timed call."""
        _install_fake_vllm(monkeypatch, VLLM_0320_ENCODER_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_ATTN_ENCODER_DOTTED, _COLLECT_ATTN_ENCODER_PATH)
        fa_calls = []
        logged = {}

        @contextlib.contextmanager
        def benchmark_with_power(*, kernel_func, **_kwargs):
            kernel_func()
            yield {"latency_ms": 1.0, "power_stats": None}

        monkeypatch.setattr(module, "get_vit_attn_backend", lambda **_kwargs: _FakeBackendEnum.FLASH_ATTN)
        monkeypatch.setattr(module, "get_flash_attn_version", lambda **_kwargs: 3)
        monkeypatch.setattr(module, "flash_attn_varlen_func", lambda *args, **kwargs: fa_calls.append((args, kwargs)))
        monkeypatch.setattr(module, "benchmark_with_power", benchmark_with_power)
        monkeypatch.setattr(module, "log_perf", lambda **kwargs: logged.update(kwargs))

        module.run_encoder_attention_torch(2, 1024, 16, 64, perf_filename="x")

        ((args, kwargs),) = fa_calls
        assert len(args) == 3
        assert kwargs["out"] is module.torch.empty_like.return_value
        assert kwargs["causal"] is False and kwargs["dropout_p"] == 0.0
        assert kwargs["fa_version"] == 3
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 1024
        assert logged["kernel_source"] == "vllm_vit_flash_attn_fa3"
