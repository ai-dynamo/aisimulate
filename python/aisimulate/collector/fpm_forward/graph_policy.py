# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native CUDA-graph settings for GLM same-request FPM campaigns.

Both backends keep their native graph defaults unless a campaign declares an
explicit setting. A declared setting is frozen plan/cell identity, rendered
once into the engine argv and verified against the native resolved settings
and per-rank artifacts. It is never inferred from collected data.

- vLLM: ``--cudagraph-capture-sizes``. The GLM runtime default captures up to
  64 tokens; larger prefills run eager. The sizes travel as a backend policy
  with the resolved-config marker ``config.engine_args.cudagraph_capture_sizes``.
- SGLang 0.5.20: ``--cuda-graph-backend-prefill breakable`` and
  ``--cuda-graph-max-bs-prefill``. Without the explicit opt-in, SGLang disables
  the breakable prefill graph for KDA hybrid linear-attention models
  ("Breakable CUDA graph is incompatible with KDA hybrid linear attention;
  disabling prefill CUDA graph.").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aisimulate.fpm_contract import FPM_RESOLVED_CONFIG_GLOB

VLLM_CAPTURE_SIZES_MARKER = "config.engine_args.cudagraph_capture_sizes"
VLLM_CAPTURE_SIZES_FLAG = "--cudagraph-capture-sizes"
SGLANG_PREFILL_GRAPH_BACKENDS = ("breakable",)
SGLANG_PREFILL_GRAPH_FLAGS = ("--cuda-graph-backend-prefill", "--cuda-graph-max-bs-prefill")


def validate_capture_sizes(sizes: tuple[int, ...] | None) -> None:
    """Accept only an increasing tuple of distinct positive integer capture sizes."""

    if sizes is None:
        return
    if (
        not isinstance(sizes, tuple)
        or not sizes
        or any(type(value) is not int or value < 1 for value in sizes)
        or tuple(sorted(set(sizes))) != sizes
    ):
        raise ValueError("--vllm-cudagraph-capture-sizes must be increasing, distinct positive integers")


def capture_sizes_label(sizes: tuple[int, ...]) -> str:
    """Stable, compact backend-policy label for one explicit capture list."""

    validate_capture_sizes(sizes)
    digest = hashlib.sha256(json.dumps(list(sizes), separators=(",", ":")).encode()).hexdigest()
    return f"n{len(sizes)}-max{sizes[-1]}-{digest[:12]}"


def vllm_capture_policy(sizes: tuple[int, ...]) -> tuple[list[str], dict[str, str]]:
    """Return the engine arguments and resolved-config marker for explicit sizes."""

    validate_capture_sizes(sizes)
    return [VLLM_CAPTURE_SIZES_FLAG, *map(str, sizes)], {VLLM_CAPTURE_SIZES_MARKER: str(list(sizes))}


def declared_capture_sizes(expected_markers: dict[str, str]) -> tuple[int, ...] | None:
    """Recover the frozen explicit capture list from a backend-policy marker."""

    raw = expected_markers.get(VLLM_CAPTURE_SIZES_MARKER)
    if raw is None:
        return None
    try:
        sizes = tuple(json.loads(raw))
    except (TypeError, ValueError) as error:
        raise ValueError("frozen vLLM capture-size marker is malformed") from error
    validate_capture_sizes(sizes)
    if str(list(sizes)) != raw:
        raise ValueError("frozen vLLM capture-size marker is not canonical")
    return sizes


def validate_resolved_markers(expected: dict[str, str], policy_id: str, raw_root: Path) -> None:
    """Compare every resolved-config file below ``raw_root`` with the frozen markers."""

    if not expected:
        return
    paths = sorted(Path(raw_root).glob(f"**/{FPM_RESOLVED_CONFIG_GLOB}"))
    if not paths:
        raise ValueError(f"backend policy {policy_id} requires resolved-config evidence")
    for path in paths:
        payload = json.loads(path.read_text())
        mismatches = {}
        for marker_path, marker_value in expected.items():
            actual: object = payload
            for part in marker_path.split("."):
                if not isinstance(actual, dict) or part not in actual:
                    actual = "<missing>"
                    break
                actual = actual[part]
            # Markers are declared as strings while the resolved config keeps
            # native JSON types (enable_eplb: true vs expected "True");
            # compare canonical string forms so a type gap is not a mismatch.
            if str(actual) != str(marker_value):
                mismatches[marker_path] = {"actual": actual, "expected": marker_value}
        if mismatches:
            raise ValueError(f"backend marker mismatch in {path}: {mismatches}")


def validate_vllm_native_capture(sizes: tuple[int, ...] | None, payload: dict, path: Path) -> None:
    """Bind one native vLLM rank artifact to the frozen explicit capture list."""

    if sizes is None:
        return
    graph = payload.get("cudagraph")
    if not isinstance(graph, dict):
        raise ValueError(f"native vLLM artifact lacks its CUDA-graph settings: {path}")
    expected = list(sizes)
    if (
        graph.get("capture_sizes") != expected
        or graph.get("prefill_capture_sizes") != expected
        or graph.get("max_capture_size") != sizes[-1]
    ):
        raise ValueError(f"native vLLM CUDA-graph capture sizes differ from the frozen policy: {path}")


def validate_sglang_prefill_graph(backend: str | None, max_bs: int | None) -> None:
    """Validate the pair of explicit SGLang prefill-graph settings."""

    if (backend is None) != (max_bs is None):
        raise ValueError(
            "--sglang-cuda-graph-backend-prefill and --sglang-cuda-graph-max-bs-prefill must be supplied together"
        )
    if backend is None:
        return
    if backend not in SGLANG_PREFILL_GRAPH_BACKENDS:
        raise ValueError(f"--sglang-cuda-graph-backend-prefill must be one of {SGLANG_PREFILL_GRAPH_BACKENDS}")
    if type(max_bs) is not int or max_bs < 1:
        raise ValueError("--sglang-cuda-graph-max-bs-prefill must be a positive integer")


def sglang_prefill_graph_args(backend: str | None, max_bs: int | None) -> list[str]:
    """Render the explicit SGLang prefill-graph request; empty keeps the native default."""

    validate_sglang_prefill_graph(backend, max_bs)
    if backend is None:
        return []
    return [SGLANG_PREFILL_GRAPH_FLAGS[0], backend, SGLANG_PREFILL_GRAPH_FLAGS[1], str(max_bs)]


def validate_sglang_native_prefill_graph(
    backend: str | None, max_bs: int | None, *, declared: dict, resolved: dict
) -> None:
    """Bind declared and resolved native ServerArgs to the frozen prefill-graph request.

    Without an explicit request the native default, including the KDA
    auto-disable, remains whatever the runtime resolved, so legacy campaigns
    are unchanged.
    """

    validate_sglang_prefill_graph(backend, max_bs)
    if backend is None:
        return
    for label, config in (("declared", declared), ("resolved", resolved)):
        if config.get("cuda_graph_backend_prefill") != backend or config.get("cuda_graph_max_bs_prefill") != max_bs:
            raise ValueError(f"SGLang {label} prefill CUDA graph differs from the frozen request")
        if config.get("disable_cuda_graph") or config.get("disable_prefill_cuda_graph"):
            raise ValueError(f"SGLang {label} configuration disables the requested prefill CUDA graph")
    graph = resolved.get("cuda_graph_config")
    prefill = graph.get("prefill") if isinstance(graph, dict) else None
    if (
        not isinstance(prefill, dict)
        or prefill.get("backend") != backend
        or prefill.get("max_bs") != max_bs
        or not isinstance(prefill.get("bs"), list)
        or not prefill["bs"]
        or max(prefill["bs"]) != max_bs
    ):
        raise ValueError("SGLang resolved prefill CUDA graph differs from the frozen request")
