# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared whole-forward execution identity; schema 6 has the legacy defaults."""

from __future__ import annotations

import copy
import hashlib
import json

EXECUTION_COLUMNS = ("model_config_sha256", "execution_profile", "engram_residency", "input_modality")
LEGACY_EXECUTION_IDENTITY = ("", "full", "none", "text")


def execution_identity(
    raw_config: dict,
    *,
    decoder_replay: bool = False,
    backend: str = "vllm",
    engram_cpu_offload: bool | None = None,
    input_modality: str | None = None,
) -> tuple[str, ...]:
    """Bind V4.1 curves to config and execution; leave existing model keys intact.

    Both SDK and Collector normalize inferred quantization fields before hashing,
    so loading the same config via HF, a local directory or AIC cache agrees.
    V4.1 callers must supply the supported Engram/text execution facts explicitly.
    The SDK supplies its prediction contract; producers must pass their checked
    runtime configuration. A checkpoint hash does not prove runtime residency.
    """
    architectures = raw_config.get("architectures") or []
    if "DeepseekV41ForCausalLM" not in architectures:
        if decoder_replay:
            raise ValueError("FPM decoder replay requires a DeepSeek-V4.1 model")
        return LEGACY_EXECUTION_IDENTITY
    if engram_cpu_offload is not False:
        raise ValueError("DeepSeek-V4.1 FPM requires explicit engram_cpu_offload=False")
    if input_modality != "text":
        raise ValueError("DeepSeek-V4.1 FPM requires explicit input_modality='text'")
    from .deepseek_v41 import resolve_execution_profile
    from .utils import _attach_inferred_quant_fields

    payload = _attach_inferred_quant_fields(copy.deepcopy(raw_config))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return (digest, resolve_execution_profile(decoder_replay, backend).value, "hbm_tp_sharded", "text")
