# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel knobs for deployment rendering: the single decision point
for which backend/version can launch prefill CP and decode CP, and how.

The simulator carries two orthogonal per-worker knobs:

* ``context_parallel_size`` -- prefill context parallelism (SGLang
  ``--attn-cp-size`` + ``--enable-prefill-cp``, vLLM ``-pcp``). Extra
  attention ranks that split the prefill tokens; decode stays replicated on
  them, so the worker grows by ``cp`` GPUs.
* ``decode_context_parallel_size`` -- decode context parallelism (vLLM
  ``-dcp``, SGLang ``--dcp-size``). Stripes the decode KV across ranks that
  already belong to the attention group; adds no GPUs.

Both entry points into the generator (the SDK result bridge and the Sweeper
candidate request) funnel through :func:`context_parallel_params` so a knob a
backend/version cannot launch fails here, loudly, instead of being dropped by
a template that has no line for it (the "silently wrong deployment" pattern).

# Guard: the minimum versions below are the FIRST release tag containing the
# flag in the framework's own history (verified 2026-09-17 against the vLLM and
# SGLang git logs), floored to the versioned cli_args template that renders
# the flag. Re-verify on every backend version bump.
"""

from __future__ import annotations

from typing import Any

from packaging.version import InvalidVersion, Version

# DeepSeek Sparse Attention architectures: SGLang's prefill CP requires the
# ``interleave`` layout for them (zigzag is rejected at startup); dense and
# plain-MLA models use ``zigzag``.
DSA_ARCHITECTURES: frozenset[str] = frozenset({"DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"})
DSA_MODEL_FAMILIES: frozenset[str] = frozenset({"DEEPSEEKV32"})

_MIN_VERSIONS: dict[tuple[str, str], Version] = {
    # vLLM: DCP landed in v0.10.2 (#23734) and PCP in v0.12.0 (#28718), but the
    # generator only renders the flags from cli_args.0.14.1.j2 onwards (the
    # 0.10.2 / 0.11.0 / 0.12.0 templates predate them and are frozen), and
    # template selection is a floor match. The floor is therefore the first
    # template that emits the flags, not the framework release, so a request
    # can never resolve to a template that silently drops the knob. The DCP
    # communication backend selector arrived in v0.18.0 (#34883).
    ("vllm", "decode_context_parallel_size"): Version("0.14.1"),
    ("vllm", "context_parallel_size"): Version("0.14.1"),
    ("vllm", "dcp_comm_backend"): Version("0.18.0"),
    # SGLang: --enable-prefill-cp/--cp-strategy in v0.5.14 (#27312) and
    # --dcp-size in v0.5.15 (#25090); the generator renders both from the
    # cli_args.0.5.15 template, so the floor is 0.5.15 for either knob.
    # --dcp-comm-backend arrived in v0.5.17 (cli_args.0.5.17 template).
    ("sglang", "context_parallel_size"): Version("0.5.15"),
    ("sglang", "decode_context_parallel_size"): Version("0.5.15"),
    ("sglang", "dcp_comm_backend"): Version("0.5.17"),
}

_DCP_COMM_BACKENDS: dict[str, frozenset[str]] = {
    "vllm": frozenset({"ag_rs", "a2a"}),
    "sglang": frozenset({"ag_rs", "a2a", "fi_a2a"}),
}


class ContextParallelUnsupportedError(ValueError):
    """The requested CP knob cannot be launched on this backend/version."""


def _parse_version(backend_version: str | None) -> Version | None:
    if backend_version is None:
        return None
    text = str(backend_version).strip()
    if text.lower().startswith("v") and len(text) > 1:
        text = text[1:]
    try:
        return Version(text)
    except InvalidVersion:
        return None


def _positive(value: Any, name: str) -> int:
    if value is None:
        return 1
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ContextParallelUnsupportedError(f"{name} must be a positive integer, got {value!r}") from exc
    if number != value or number < 1:
        raise ContextParallelUnsupportedError(f"{name} must be a positive integer, got {value!r}")
    return number


def cp_strategy_for(architecture: str | None, model_family: str | None = None) -> str:
    """SGLang ``--cp-strategy`` for a model: ``interleave`` for DSA, else ``zigzag``."""
    if architecture in DSA_ARCHITECTURES or (model_family or "").upper() in DSA_MODEL_FAMILIES:
        return "interleave"
    return "zigzag"


def _require_version(backend: str, knob: str, backend_version: str | None, value: Any) -> None:
    minimum = _MIN_VERSIONS.get((backend, knob))
    if minimum is None:
        raise ContextParallelUnsupportedError(
            f"{knob}={value!r} cannot be rendered for backend {backend!r}: the generator has no "
            "launch flag for it on this backend"
            + (
                " (TensorRT-LLM's Helix context parallelism is a disaggregated-decode-only layout "
                "that the simulator does not model)"
                if backend == "trtllm"
                else ""
            )
        )
    parsed = _parse_version(backend_version)
    if parsed is None:
        raise ContextParallelUnsupportedError(
            f"{knob}={value!r} requires a parseable {backend} backend_version to check support "
            f"(need >= {minimum}), got {backend_version!r}"
        )
    if parsed < minimum:
        raise ContextParallelUnsupportedError(
            f"{knob}={value!r} requires {backend} >= {minimum} (flag not available in {backend_version})"
        )


def context_parallel_params(
    *,
    backend: str,
    backend_version: str | None,
    context_parallel_size: Any = 1,
    decode_context_parallel_size: Any = 1,
    dcp_comm_backend: str | None = None,
    architecture: str | None = None,
    model_family: str | None = None,
) -> dict[str, Any]:
    """Per-worker generator params for the CP knobs, or ``{}`` when both are 1.

    Raises :class:`ContextParallelUnsupportedError` when the backend/version
    has no launch flag for a knob that is set, so an evaluated CP deployment
    can never degrade into an un-CP'd launch.
    """
    cp = _positive(context_parallel_size, "context_parallel_size")
    dcp = _positive(decode_context_parallel_size, "decode_context_parallel_size")
    params: dict[str, Any] = {}
    if cp > 1:
        _require_version(backend, "context_parallel_size", backend_version, cp)
        params["context_parallel_size"] = cp
        if backend == "sglang":
            params["cp_strategy"] = cp_strategy_for(architecture, model_family)
    if dcp > 1:
        _require_version(backend, "decode_context_parallel_size", backend_version, dcp)
        params["decode_context_parallel_size"] = dcp
    if dcp_comm_backend:
        if dcp <= 1:
            raise ContextParallelUnsupportedError(
                f"dcp_comm_backend={dcp_comm_backend!r} needs decode_context_parallel_size > 1"
            )
        allowed = _DCP_COMM_BACKENDS.get(backend, frozenset())
        if dcp_comm_backend not in allowed:
            raise ContextParallelUnsupportedError(
                f"dcp_comm_backend={dcp_comm_backend!r} is not a {backend} choice; expected one of {sorted(allowed)}"
            )
        _require_version(backend, "dcp_comm_backend", backend_version, dcp_comm_backend)
        params["dcp_comm_backend"] = dcp_comm_backend
    return params


def context_parallel_gpu_multiplier(context_parallel_size: Any) -> int:
    """How prefill CP grows the worker: ``cp`` extra attention ranks on every
    backend (SGLang folds them into ``--tp``, vLLM's PCP expands the world size).
    Decode CP never changes the GPU count."""
    return _positive(context_parallel_size, "context_parallel_size")
