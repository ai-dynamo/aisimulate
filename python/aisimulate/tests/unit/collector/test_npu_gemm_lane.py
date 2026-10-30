# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU gemm lane — manifest pin, registry entry, cases, module.

The three wiring surfaces only work together (an op entry without its manifest
pin is unresolvable, a manifest entry without ops fails ``validate_resolution``),
so they are asserted together here. Neither an Ascend device, ``torch_npu`` nor
``vllm_ascend`` exists on CI or dev machines, so the collector module itself is
imported against stubbed ``torch``/``vllm``/``vllm_ascend`` modules; the point is
its declared shape (bf16 only, eager, npu:0, no power), not its timings.
"""

import functools
import importlib
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from collector.case_generator import get_gemm_case_specs, get_gemm_type_specs
from collector.framework_manifest import (
    get_collector_runtime,
    require_collector_runtime,
    resolve_op_runtime,
    validate_resolution,
)
from collector.registry_types import PerfFile
from collector.vllm.registry import REGISTRY_NPU, REGISTRY_XPU

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = "collector.vllm.collect_gemm_npu"

# Resolved from quay.io and cross-checked against the m.daocloud.io mirror on
# 2026-10-29.
IMAGE = (
    "quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler"
    "@sha256:2fb630483fee3ee74c882ad490e2f2586e683a507b8365306f8d25e4df3e33ea"
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_pins_the_vllm_ascend_runtime_identity():
    runtime = get_collector_runtime("vllm_npu")

    assert runtime.framework == "vllm_npu"
    # The version axis is the vllm-ascend release, not base vLLM.
    assert runtime.version == "0.30.0rc2"
    assert runtime.images == {"default": IMAGE}
    assert runtime.image() == IMAGE
    # The data backend directory stays "vllm", like the XPU lane.
    assert runtime.data_backend == "vllm"
    assert runtime.workload == "default"


def test_manifest_records_the_image_abi_read_from_the_image_layers():
    """The A5 image's daily-deps step overrides requirements.txt and the
    published versioning-policy matrix; abi: must describe what is
    installed."""
    abi = get_collector_runtime("vllm_npu").abi

    assert abi["cann"] == "9.2.0_20260930220649"
    assert abi["torch"] == "2.10.0"
    assert abi["torch_npu"] == "2.10.0.post7"  # not the matrix's 2.10.0.post4
    assert abi["triton_ascend"] == "3.6.0+dev20260923230212"  # not 3.2.2
    assert abi["triton"] == "3.6.0"  # uninstalled by the Dockerfile, pulled back by triton-ascend
    assert abi["vllm"] == "0.30.0"
    assert abi["soc_version"] == "ascend950dt_9582"


def test_every_registry_lane_including_npu_resolves():
    assert validate_resolution() == []


def test_gemm_resolves_through_the_npu_lane_and_enforces_the_pin():
    assert resolve_op_runtime("vllm_npu", "gemm").version == "0.30.0rc2"

    runtime = require_collector_runtime("vllm_npu", "0.30.0rc2", requested_ops={"gemm"}, wideep_ops=set())
    assert runtime.framework == "vllm_npu"

    with pytest.raises(RuntimeError, match=r"vllm_npu stock collector requires exactly 0\.30\.0rc2"):
        require_collector_runtime("vllm_npu", "0.30.0", requested_ops={"gemm"}, wideep_ops=set())


def test_npu_collectors_are_exactly_pinned_to_the_manifest_version():
    """Mirrors the XPU guard: no version routing, so the literal must match."""
    expected = f'__compat__ = "vllm_ascend=={get_collector_runtime("vllm_npu").version}"'
    assert all(not entry.versions for entry in REGISTRY_NPU)

    for module in sorted({entry.module for entry in REGISTRY_NPU}):
        source = (REPO_ROOT / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8")
        declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
        assert declarations == [expected], module


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_npu_carries_only_the_gemm_op():
    assert [entry.op for entry in REGISTRY_NPU] == ["gemm"]

    gemm = REGISTRY_NPU[0]
    assert gemm.module == MODULE
    assert (gemm.get_func, gemm.run_func) == ("get_gemm_test_cases", "run_gemm")
    # Reuses the shared table name: the NPU lane adds no new perf file.
    assert gemm.perf_filename is PerfFile.GEMM


def test_npu_gemm_entry_mirrors_the_xpu_one_except_for_its_module():
    xpu_gemm = next(entry for entry in REGISTRY_XPU if entry.op == "gemm")
    npu_gemm = REGISTRY_NPU[0]

    assert (npu_gemm.get_func, npu_gemm.run_func, npu_gemm.perf_filename) == (
        xpu_gemm.get_func,
        xpu_gemm.run_func,
        xpu_gemm.perf_filename,
    )
    assert npu_gemm.module != xpu_gemm.module


# ---------------------------------------------------------------------------
# Case partition
# ---------------------------------------------------------------------------


def test_npu_gemm_cases_are_bf16_only():
    assert get_gemm_type_specs("vllm_npu") == ["bfloat16"]
    # The XPU partition keeps its fp8 lane; the partitions are independent.
    assert get_gemm_type_specs("vllm_xpu") == ["bfloat16", "fp8"]


def test_npu_gemm_shape_sweep_expands():
    specs = get_gemm_case_specs("vllm_npu")

    assert specs, "cases/base_ops/gemm.yaml must define a vllm_npu shape sweep"
    assert all(spec.x > 0 and spec.n > 0 and spec.k > 0 for spec in specs)
    # Narrowed grid shared with the XPU partition, not the full base sweep.
    assert len(specs) == len(get_gemm_case_specs("vllm_xpu"))


# ---------------------------------------------------------------------------
# Collector module
# ---------------------------------------------------------------------------


def _with_exit_stack(func):
    """Stand-in for utils_xpu.with_exit_stack (same contract: injects a stack)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with ExitStack() as stack:
            return func(stack, *args, **kwargs)

    return wrapper


@pytest.fixture
def collect_gemm_npu(monkeypatch):
    """Import the collector against stubbed torch / vLLM / vllm-ascend."""
    for name in ("torch", "vllm", "vllm.config", "vllm.model_executor", "vllm.model_executor.layers"):
        monkeypatch.setitem(sys.modules, name, MagicMock(name=name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", MagicMock(name="linear"))
    monkeypatch.setitem(sys.modules, "vllm_ascend", SimpleNamespace(__version__="0.30.0rc2"))
    # utils pulls in the full vLLM attention stack; stub it, but keep a real
    # with_exit_stack so run_gemm keeps its decorated signature.
    utils = MagicMock(name="utils")
    utils.with_exit_stack = _with_exit_stack
    monkeypatch.setitem(sys.modules, "collector.vllm.utils", utils)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    module = importlib.import_module(MODULE)
    try:
        yield module
    finally:
        sys.modules.pop(MODULE, None)


def test_test_cases_are_all_bfloat16(collect_gemm_npu):
    cases = collect_gemm_npu.get_gemm_test_cases()

    assert cases
    assert {gemm_type for gemm_type, *_ in cases} == {"bfloat16"}
    assert len(cases) == len(get_gemm_case_specs("vllm_npu"))


def test_version_logged_is_the_vllm_ascend_version(collect_gemm_npu):
    """Rows and the parquet <version> directory share the manifest's axis."""
    assert collect_gemm_npu.vllm_ascend_version == "0.30.0rc2"


def test_run_gemm_defaults_to_npu_0_and_refuses_other_dtypes(collect_gemm_npu):
    import inspect

    signature = inspect.signature(collect_gemm_npu.run_gemm.__wrapped__)
    assert signature.parameters["device"].default == "npu:0"

    with pytest.raises(RuntimeError, match="collects bfloat16 only"):
        collect_gemm_npu.run_gemm("fp8", 16, 16, 16, perf_filename=PerfFile.GEMM)


def test_module_declares_the_mvp_measurement_contract():
    """Eager, no power sampling, untraced kernel label, XPU-compatible columns."""
    source = (REPO_ROOT / f"{MODULE.replace('.', '/')}.py").read_text(encoding="utf-8")

    assert "use_cuda_graph=False" in source
    assert "power_stats=None" in source
    assert 'kernel_source="vllm_default"' in source
    # Same columns as the XPU gemm table, so the two can share a schema.
    assert '"used_cuda_graph": results["used_cuda_graph"]' in source
    assert "xpu_graph_measure_enabled" not in source
    # torch_npu's get_device_capability returns None (or a configured fake) —
    # the lane must never call it.
    assert "get_device_capability(" not in source
