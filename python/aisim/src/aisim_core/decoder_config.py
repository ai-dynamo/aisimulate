# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose the key of a checkpoint config that nests the text decoder.

FPM config intake, the FPM collector and the FPM profile check call decoder_config_key, so all three read one decoder.
The module imports only the standard library. Config intake must stay usable in a process that does not import
aisim_core.sdk. The cold-process tests in tests/unit/test_support_topology.py and
tests/unit/test_onboard_topology.py check that, and tests/unit/sdk/test_fpm_model_metadata.py checks it for a wrapper
in LLM_CONFIG_WRAPPERS.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# The wrappers whose text decoder is under llm_config. aisim_core.sdk.common builds the llm_config entries of
# MULTIMODAL_TEXT_CONFIG_KEY from this set, so the op-level parser and decoder_config_key unwrap the same wrappers.
LLM_CONFIG_WRAPPERS = frozenset({"NemotronH_Omni_Reasoning_V3"})


def decoder_config_key(document: Mapping[str, Any]) -> str | None:
    """Return the key that nests the text decoder in a checkpoint config, or None for a flat config.

    A wrapper in LLM_CONFIG_WRAPPERS, named by the first architecture, holds llm_config whenever the key is present,
    as the op-level parser does. Otherwise text_config is held whenever the key is present, and then llm_config only
    as a nonempty object, so a null, empty or non-object llm_config leaves the config flat. A held key is chosen even
    when its value is malformed, so the caller reports it. The function never raises.

    MULTIMODAL_TEXT_CONFIG_KEY maps every other wrapper to text_config. tests/unit/sdk/test_fpm_model_metadata.py
    checks that the registry holds no other key and that it maps exactly LLM_CONFIG_WRAPPERS to llm_config.
    """
    architectures = document.get("architectures")
    wrapper = architectures[0] if isinstance(architectures, list) and architectures else None
    if isinstance(wrapper, str) and wrapper in LLM_CONFIG_WRAPPERS and "llm_config" in document:
        return "llm_config"
    if "text_config" in document:
        return "text_config"
    llm_config = document.get("llm_config")
    return "llm_config" if isinstance(llm_config, dict) and llm_config else None


def nested_decoder_ambiguity(key: str, decoder: Mapping[str, Any]) -> str | None:
    """Name the decoder key that a decoder config holds, or return None when it holds none.

    key is the result of decoder_config_key and decoder is the object under it. A decoder that holds a decoder key
    nests a second decoder, so no caller can tell which one to read. The held rule is that of decoder_config_key for a
    document that names no wrapper in LLM_CONFIG_WRAPPERS: text_config is held whenever the key is present, and
    llm_config only as a nonempty object. The decoder's own key comes first, then the other decoder key.

    The text is the diagnosis only: "ambiguous nested <key>" for the own key and "ambiguous nested <inner> in <key>"
    for the other key. Each caller adds its remedy after "; ", because intake accepts a complete --fpm-profile and
    the collector does not. The function never raises and never reads the registry.
    """
    llm_config = decoder.get("llm_config")
    held = {"text_config": "text_config" in decoder, "llm_config": isinstance(llm_config, dict) and bool(llm_config)}
    if held.get(key):
        return f"ambiguous nested {key}"
    inner = "llm_config" if key == "text_config" else "text_config"
    return f"ambiguous nested {inner} in {key}" if held[inner] else None


def decoder_config_problem(key: str, decoder: object) -> str | None:
    """Name what keeps the value under a decoder key from describing one text decoder, or return None.

    key is the result of decoder_config_key and decoder is the value under it. The value must be a nonempty object
    that holds no decoder key. text_config, and the llm_config of a wrapper in LLM_CONFIG_WRAPPERS, are held whenever
    the key is present, so a null, empty or non-object value reaches this function. Any other llm_config is held only
    as a nonempty object, so it never fails the first check.

    The text is the diagnosis only: "<key> must be a nonempty object describing one text decoder", or the text of
    nested_decoder_ambiguity. The collector and the FPM profile check add their remedy after "; ". Config intake keeps
    its own two checks, because it words the remedy of each differently. The function never raises.
    """
    if not isinstance(decoder, dict) or not decoder:
        return f"{key} must be a nonempty object describing one text decoder"
    return nested_decoder_ambiguity(key, decoder)
