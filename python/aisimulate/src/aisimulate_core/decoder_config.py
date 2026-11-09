# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose the key of a checkpoint config that nests the text decoder.

FPM config intake, the FPM collector and the FPM profile check call this function, so all three read one decoder.
The module imports only the standard library. Config intake must stay usable in a process that does not import
aisimulate_core.sdk. The cold-process tests in tests/unit/test_support_topology.py and
tests/unit/test_onboard_topology.py check that. The one exception is a document that holds both text_config and a
nonempty llm_config: for that shape the function reads the registry from aisimulate_core.sdk.common.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def decoder_config_key(document: Mapping[str, Any]) -> str | None:
    """Return the key that nests the text decoder in a checkpoint config, or None for a flat config.

    The candidates, in order, are the key that MULTIMODAL_TEXT_CONFIG_KEY registers for the wrapper's first
    architecture, then text_config, then llm_config. The first candidate that the document holds wins.
    text_config is held whenever the key is present, so a malformed one is still chosen and intake reports it.
    llm_config is held only as a nonempty object, so a null, empty or non-object value leaves the config flat.
    The function never raises.

    The registry maps every wrapper to text_config or llm_config. It changes the answer only when the document holds
    both keys and the wrapper maps to llm_config, so the registry is imported only in that case.
    tests/unit/sdk/test_fpm_model_metadata.py checks that the registry holds no other key.
    """
    text_held = "text_config" in document
    llm_config = document.get("llm_config")
    llm_held = isinstance(llm_config, dict) and bool(llm_config)
    if text_held and llm_held:
        from aisimulate_core.sdk.common import MULTIMODAL_TEXT_CONFIG_KEY

        architectures = document.get("architectures")
        wrapper = architectures[0] if isinstance(architectures, list) and architectures else None
        if isinstance(wrapper, str) and MULTIMODAL_TEXT_CONFIG_KEY.get(wrapper) == "llm_config":
            return "llm_config"
    if text_held:
        return "text_config"
    return "llm_config" if llm_held else None
