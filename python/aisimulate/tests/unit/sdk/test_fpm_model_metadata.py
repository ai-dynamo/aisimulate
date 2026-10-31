# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder key that FPM config intake, the FPM collector and the FPM profile check share."""

from __future__ import annotations

import pytest

from aisimulate_core.sdk.common import MULTIMODAL_TEXT_CONFIG_KEY
from aisimulate_core.sdk.fpm_model_metadata import decoder_config_key

pytestmark = pytest.mark.unit

_DECODER = {"hidden_size": 16}
_BOTH_KEYS = {"text_config": _DECODER, "llm_config": _DECODER}
# MULTIMODAL_TEXT_CONFIG_KEY registers llm_config for this wrapper (#562).
_VL = "NemotronH_Omni_Reasoning_V3"
_WRAPPER = "ExampleForConditionalGeneration"


@pytest.mark.parametrize(
    "document,key",
    [
        ({"architectures": [_VL], "llm_config": _DECODER}, "llm_config"),
        ({"architectures": [_VL], "llm_config": _DECODER, "text_config": _DECODER}, "llm_config"),
        ({"architectures": [_VL], "llm_config": None, "text_config": _DECODER}, "text_config"),
        ({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": _DECODER}, "text_config"),
        ({"architectures": [_WRAPPER], "text_config": _DECODER, "llm_config": _DECODER}, "text_config"),
        # A malformed text_config is still chosen, so intake keeps reporting it.
        ({"architectures": [_WRAPPER], "text_config": None}, "text_config"),
        ({"architectures": ["InternVLChatModel"], "llm_config": _DECODER}, "llm_config"),
        # Another key counts only as a nonempty object, so it adds no error.
        ({"architectures": [_VL], "llm_config": None}, None),
        ({"architectures": [_VL], "llm_config": {}}, None),
        ({"llm_config": [_DECODER]}, None),
        ({"llm_config": "decoder"}, None),
        # A malformed wrapper architecture never raises here.
        ({"architectures": _VL, "llm_config": _DECODER}, "llm_config"),
        ({"architectures": [{"name": _VL}], "llm_config": _DECODER}, "llm_config"),
        ({"architectures": [], "text_config": _DECODER}, "text_config"),
        # With both keys the helper reads the registry. A malformed architectures value must neither raise nor
        # select llm_config, and a wrapper registered to text_config keeps text_config.
        ({"architectures": _VL, **_BOTH_KEYS}, "text_config"),
        ({"architectures": [{"name": _VL}], **_BOTH_KEYS}, "text_config"),
        ({"architectures": [], **_BOTH_KEYS}, "text_config"),
        ({"architectures": {"0": _VL}, **_BOTH_KEYS}, "text_config"),
        ({"architectures": ["Qwen3_5ForConditionalGeneration"], **_BOTH_KEYS}, "text_config"),
        ({"architectures": ["NemotronHForCausalLM"], "mamba_num_heads": 128}, None),
        ({}, None),
    ],
    ids=[
        "registered-llm-config",
        "registered-key-before-text-config",
        "registered-key-not-an-object",
        "registered-text-config",
        "text-config-before-llm-config",
        "malformed-text-config",
        "unregistered-llm-config",
        "null-llm-config",
        "empty-llm-config",
        "list-llm-config",
        "string-llm-config",
        "string-architectures",
        "object-architecture-entry",
        "empty-architectures",
        "both-keys-string-architectures",
        "both-keys-object-architecture-entry",
        "both-keys-empty-architectures",
        "both-keys-dict-architectures",
        "both-keys-wrapper-registered-to-text-config",
        "flat",
        "empty",
    ],
)
def test_decoder_config_key(document, key):
    assert decoder_config_key(document) == key


def test_the_registry_maps_wrappers_only_to_keys_that_the_helper_orders():
    # The helper reads the registry only for a document that holds both keys, so a third key would go unseen.
    assert set(MULTIMODAL_TEXT_CONFIG_KEY.values()) <= {"text_config", "llm_config"}
