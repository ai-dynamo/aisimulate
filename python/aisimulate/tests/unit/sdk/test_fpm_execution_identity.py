# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pytest

from aiconfigurator_core.sdk.fpm_identity import LEGACY_EXECUTION_IDENTITY, execution_identity
from aiconfigurator_core.sdk.utils import _attach_inferred_quant_fields, _get_model_config_path

pytestmark = pytest.mark.unit


def config():
    return json.loads((Path(_get_model_config_path()) / "deepseek-ai--DeepSeek-V4.1-Flash_config.json").read_text())


def test_v41_config_and_execution_cannot_borrow_a_table():
    raw = config()
    off = execution_identity(raw)
    on = execution_identity(raw, decoder_replay=True, backend="sglang")
    assert off[1:] == ("full", "hbm_tp_sharded", "text")
    assert on[0] == off[0] and on[1] == "decoder_bounded"
    altered = copy.deepcopy(raw)
    altered["text_config"]["kv_source_layer_ids"] = [2, 8, 14]
    assert execution_identity(altered)[0] != off[0]
    assert execution_identity(_attach_inferred_quant_fields(copy.deepcopy(raw))) == off
    assert raw == config()
    with pytest.raises(NotImplementedError, match="not verified for vllm"):
        execution_identity(raw, decoder_replay=True)


def test_existing_model_identity_stays_legacy():
    assert execution_identity({"architectures": ["LlamaForCausalLM"]}) == LEGACY_EXECUTION_IDENTITY


def test_native_v41_requires_measured_execution_and_text_evidence():
    from types import SimpleNamespace

    from collector.fpm_forward.native_artifact import _validate_execution_provenance

    identity = execution_identity(config())
    cell = SimpleNamespace(execution_identity=identity, input_text_sha256="a" * 64)
    fields = ("model_config_sha256", "execution_profile", "engram_residency", "input_modality")
    payload = {
        "execution_mode": "eager",
        "execution_identity": dict(zip(fields, identity, strict=True)),
        "input_provenance": {
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        },
    }
    assert _validate_execution_provenance(cell, payload, Path("artifact")) == payload["input_provenance"]
    for mode in (None, "cuda_graph"):
        corrupt = copy.deepcopy(payload)
        corrupt["execution_mode"] = mode
        with pytest.raises(ValueError, match="verified eager execution"):
            _validate_execution_provenance(cell, corrupt, Path("artifact"))
    corrupt = copy.deepcopy(payload)
    corrupt["execution_identity"]["execution_profile"] = "decoder_bounded"
    with pytest.raises(ValueError, match="execution identity"):
        _validate_execution_provenance(cell, corrupt, Path("artifact"))
    corrupt = copy.deepcopy(payload)
    corrupt["input_provenance"]["unique_token_count"] = 1
    with pytest.raises(ValueError, match="multiple tokenizer-generated"):
        _validate_execution_provenance(cell, corrupt, Path("artifact"))
    assert (
        _validate_execution_provenance(
            SimpleNamespace(execution_identity=LEGACY_EXECUTION_IDENTITY, input_text_sha256=""), {}, Path("legacy")
        )
        is None
    )


@pytest.mark.parametrize("replay,backend", [(False, "vllm"), (True, "sglang")])
def test_v41_fpm_wrap_retains_resident_inventory_and_serialized_stages(replay, backend):
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.deepseek_v41 import MODEL_PATH
    from aiconfigurator_core.sdk.engine import build_ops_json
    from aiconfigurator_core.sdk.models import get_model

    kwargs = dict(tp_size=4, pp_size=1, attention_dp_size=1, moe_tp_size=4, moe_ep_size=1, decoder_replay=replay)
    granular = get_model(MODEL_PATH, ModelConfig(**kwargs), backend)
    wrapped = get_model(MODEL_PATH, ModelConfig(**kwargs, forward_model="fpm"), backend)
    for ops in (wrapped.context_ops, wrapped.generation_ops):
        assert len(ops) == 1
        assert ops[0].get_weights() == granular.get_resident_weights_bytes()
        assert ops[0]._match_identity[-4:] == execution_identity(config(), decoder_replay=replay, backend=backend)
        native = json.loads(build_ops_json(ops))[0]["FpmForward"]
        assert len(native["match_identity"]) == 19
        stages = [item["Dsv41Stage"] for item in native["sol_ops"] if "Dsv41Stage" in item]
        assert len(stages) == 40
        assert all(stage["decoder_replay"] == replay for stage in stages)
        assert '"Dsv41Linear"' in json.dumps(native["sol_ops"])
