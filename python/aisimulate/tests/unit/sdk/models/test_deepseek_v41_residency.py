# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target residency survives speculative and whole-forward graph rewrites."""

from types import SimpleNamespace

import pytest

from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.deepseek_v41 import MODEL_PATH
from aisimulate_core.sdk.models import _apply_forward_model_fpm, get_model
from aisimulate_core.sdk.speculation import NullScheme, SpeculationConfig
from aisimulate_core.sdk.speculation.draft_model import DraftModelScheme
from aisimulate_core.sdk.speculation.ngram import NgramScheme

pytestmark = pytest.mark.unit


def _model(*, model_path=MODEL_PATH, speculation=None, forward_model="op_level", pp=1, backend="sglang", replay=False):
    return get_model(
        model_path,
        ModelConfig(
            tp_size=4,
            pp_size=pp,
            attention_dp_size=1,
            moe_tp_size=4,
            moe_ep_size=1,
            speculation=speculation,
            forward_model=forward_model,
            decoder_replay=replay,
        ),
        backend,
    )


def _weights_bytes(model, backend):
    # Zero runtime overhead isolates model/scheme residency from device capacity.
    database = SimpleNamespace(system_spec={"misc": {"nccl_mem": {4: 0.0}, "other_mem": 0.0}})
    memory = get_backend(backend)._get_memory_usage(model, database, batch_size=1, beam_width=1, isl=128, osl=1)
    return memory["weights"] * (1 << 30)


@pytest.mark.parametrize("forward_model", ["op_level", "fpm"])
def test_ngram_keeps_complete_native_mxfp4_target_inventory(forward_model):
    plain = _model()
    ngram = _model(
        forward_model=forward_model,
        speculation=SpeculationConfig(kind="ngram", params={"num_speculative_tokens": 3}),
    )
    assert type(plain.spec_scheme) is NullScheme
    assert isinstance(ngram.spec_scheme, NgramScheme)
    # Checkpoint geometry: 40 layers, three matrices per expert, 384 experts,
    # TP4, with one native MXFP4 scale byte for every 32 expert elements.
    expected_scales = 40 * 3 * 5120 * 2304 * 384 // 4 // 32
    assert expected_scales == 4_246_732_800
    target_op_bytes = sum(op.get_weights() for op in plain.context_ops)
    expected_target = target_op_bytes + expected_scales
    assert plain.get_resident_weights_bytes() == expected_target
    assert _weights_bytes(plain, "sglang") == expected_target
    assert ngram.spec_scheme.draft_weights_bytes(ngram) == 0
    assert ngram.get_resident_weights_bytes() == expected_target
    assert _weights_bytes(ngram, "sglang") == expected_target
    if forward_model == "fpm":
        assert ngram.context_ops[0].get_weights() == expected_target


@pytest.mark.parametrize("model_path,pp", [(MODEL_PATH, 1), ("Qwen/Qwen3-8B", 2)])
@pytest.mark.parametrize("forward_model", ["op_level", "fpm"])
def test_owned_draft_weights_are_added_once_after_target_pp_division(model_path, pp, forward_model):
    plain = _model(model_path=model_path, pp=pp, backend="vllm")
    drafted = _model(
        model_path=model_path,
        pp=pp,
        backend="vllm",
        forward_model=forward_model,
        speculation=SpeculationConfig(
            kind="draft_model",
            draft_model_path="Qwen/Qwen3-0.6B",
            params={"num_speculative_tokens": 3},
        ),
    )
    scheme = drafted.spec_scheme
    assert isinstance(scheme, DraftModelScheme)
    owned_draft = scheme._draft_model
    expected_draft = owned_draft.get_resident_weights_bytes()
    assert expected_draft > 0
    assert sum(op.get_weights() for op in drafted.context_ops if op._name.startswith("draft_")) > 0
    assert scheme.draft_weights_bytes(drafted) == expected_draft
    assert drafted.get_resident_weights_bytes() == plain.get_resident_weights_bytes()
    expected_target_per_stage = plain.get_resident_weights_bytes() / pp
    assert _weights_bytes(plain, "vllm") == expected_target_per_stage
    assert _weights_bytes(drafted, "vllm") == expected_target_per_stage + expected_draft
    if forward_model == "fpm":
        assert drafted.context_ops[0].get_weights() == plain.get_resident_weights_bytes()


def test_legacy_fpm_rejects_decoder_replay_before_rewriting_graph(monkeypatch):
    from aisimulate_core.sdk.operations import fpm_forward

    # A legacy selector has no way to distinguish OFF and ON measurements.
    # Retain this boundary regression when newer FPM schemas add that axis.
    monkeypatch.setattr(
        fpm_forward,
        "_CELL_MATCH_COLUMNS",
        tuple(column for column in fpm_forward._CELL_MATCH_COLUMNS if column != "execution_profile"),
    )
    model = _model(replay=True)
    original_context = list(model.context_ops)
    original_generation = list(model.generation_ops)
    with pytest.raises(NotImplementedError, match="execution_profile identity"):
        _apply_forward_model_fpm(model)
    assert model.context_ops == original_context
    assert model.generation_ops == original_generation
