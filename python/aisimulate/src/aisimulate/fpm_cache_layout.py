# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config markers of a cache layout that config.json cannot describe.

Config intake and the FPM collector detect these layouts with this one check. Intake leaves their cache
geometry to the runtime probe; the collector refuses an FPM attention template for them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

RECURRENT_STATE_FIELDS = (
    "ssm_cfg",
    "mamba_d_state",
    "linear_conv_kernel_dim",
    "linear_attn_config",
    "ssm_state_size",
    "mamba_num_heads",
)
RECURRENT_ARCHITECTURES = ("mamba", "jamba", "rwkv")


def recognized_layer_types(architecture: str | None) -> frozenset[str]:
    names = {"full_attention", "sliding_attention"}
    if architecture in {"GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM"}:
        names.add("deepseek_sparse_attention")
    return frozenset(names)


def cache_layout_markers(raw: Mapping[str, Any], architecture: str | None) -> tuple[str, ...]:
    """Sorted reasons the config declares a cache layout it cannot describe; empty when it declares none.

    Layer types, attention-type lists and recurrent-state markers count independently, so recognized values
    never hide declared state. ``architecture`` is the decoder architecture the layer types are checked against.
    Malformed lists are not markers: each caller raises its own structure errors for them.
    """
    reasons = []
    layer_types = raw.get("layer_types")
    if isinstance(layer_types, list):
        recognized = recognized_layer_types(architecture)
        unrecognized = Counter(
            value for value in layer_types if isinstance(value, str) and value.strip() and value not in recognized
        )
        reasons.extend(f"layer_types {name} x{count}" for name, count in unrecognized.items())
    attn_types = raw.get("attn_type_list")
    if isinstance(attn_types, list):
        # 1 is full attention; any other integer is a layer type the config does not describe.
        other = Counter(value for value in attn_types if type(value) is int and value != 1)
        reasons.extend(f"attn_type_list {value} x{count}" for value, count in other.items())
    reasons.extend(f"recurrent-state field {key}" for key in RECURRENT_STATE_FIELDS if raw.get(key) is not None)
    if any(name in (architecture or "").lower() for name in RECURRENT_ARCHITECTURES):
        reasons.append(f"recurrent architecture {architecture}")
    return tuple(sorted(reasons))
