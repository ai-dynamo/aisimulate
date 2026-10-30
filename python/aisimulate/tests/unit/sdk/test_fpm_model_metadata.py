# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder key that FPM config intake, the FPM collector and the FPM profile check share.

The file also pins the nested-decoder rule: a decoder that holds a decoder key is ambiguous.
"""

from __future__ import annotations

import pytest

from aisimulate_core.decoder_config import nested_decoder_ambiguity
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


@pytest.mark.parametrize(
    "key,decoder,ambiguity",
    [
        ("text_config", {"text_config": _DECODER}, "ambiguous nested text_config"),
        ("text_config", {"text_config": None}, "ambiguous nested text_config"),
        ("llm_config", {"llm_config": _DECODER}, "ambiguous nested llm_config"),
        ("text_config", {"llm_config": _DECODER}, "ambiguous nested llm_config in text_config"),
        ("llm_config", {"text_config": _DECODER}, "ambiguous nested text_config in llm_config"),
        ("llm_config", {"text_config": None}, "ambiguous nested text_config in llm_config"),
        # The decoder's own key comes first, so intake keeps its text of today.
        ("text_config", _BOTH_KEYS, "ambiguous nested text_config"),
        ("llm_config", _BOTH_KEYS, "ambiguous nested llm_config"),
        # llm_config is held only as a nonempty object, as decoder_config_key holds it.
        ("llm_config", {"llm_config": None}, None),
        ("llm_config", {"llm_config": {}}, None),
        ("text_config", {"llm_config": [_DECODER]}, None),
        ("text_config", {"llm_config": "decoder"}, None),
        ("text_config", _DECODER, None),
        ("llm_config", _DECODER, None),
    ],
    ids=[
        "own-text-config",
        "own-null-text-config",
        "own-llm-config",
        "llm-config-in-text-config",
        "text-config-in-llm-config",
        "null-text-config-in-llm-config",
        "both-keys-in-text-config",
        "both-keys-in-llm-config",
        "own-null-llm-config",
        "own-empty-llm-config",
        "list-llm-config-in-text-config",
        "string-llm-config-in-text-config",
        "flat-text-config",
        "flat-llm-config",
    ],
)
def test_nested_decoder_ambiguity(key, decoder, ambiguity):
    assert nested_decoder_ambiguity(key, decoder) == ambiguity


# decoder_config_key and nested_decoder_ambiguity each decide whether a value under llm_config or text_config holds a
# decoder. The two tests below fail when those two rules drift apart.
_KEY_VALUES = [
    pytest.param(None, id="null"),
    pytest.param({}, id="empty-object"),
    pytest.param([], id="empty-list"),
    pytest.param("decoder", id="string"),
    pytest.param(0, id="number"),
    pytest.param(_DECODER, id="decoder"),
]


@pytest.mark.parametrize("value", _KEY_VALUES)
def test_both_helpers_hold_an_llm_config_by_one_rule(value):
    picked = decoder_config_key({"llm_config": value}) == "llm_config"
    assert (nested_decoder_ambiguity("text_config", {"llm_config": value}) is not None) == picked


@pytest.mark.parametrize("value", _KEY_VALUES)
def test_both_helpers_hold_a_text_config_by_one_rule(value):
    picked = decoder_config_key({"text_config": value}) == "text_config"
    assert (nested_decoder_ambiguity("llm_config", {"text_config": value}) is not None) == picked
