# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for disaggregated MoE communication resolution."""

import pytest

pytestmark = pytest.mark.unit


def test_dense_gemma4_never_resolves_moe_communication():
    """A dense checkpoint must stop at the classification gate, before any database."""
    from aisim.sdk.config import ModelConfig
    from aisim.sdk.moe_comm_resolver import resolve_model_config_moe_comm

    resolved = resolve_model_config_moe_comm(
        ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=2),
        model_path="nvidia/Gemma-4-31B-IT-NVFP4",
        backend_name="vllm",
        database=None,
        required_phases=("generation",),
    )

    assert resolved is None


def test_disagg_resolves_each_role_moe_comm_before_session_run(monkeypatch):
    import aisim.legacy_cli.api as api
    import aisim.sdk.inference_session as inference_session

    databases = {"prefill": object(), "decode": object()}
    resolver_calls = []
    captured = {}

    def fake_resolve(model_config, **kwargs):
        resolver_calls.append((model_config, kwargs))
        phase = kwargs["required_phases"][0]
        model_config.moe_comm_backend = {phase: "deepep_ht" if phase == "context" else "deepep_ll"}
        model_config.num_gpus_per_node = 4

    class FakeSummary:
        def check_oom(self):
            return False

        def get_result_dict(self):
            return {"ttft": 1.0, "tpot": 1.0}

        def get_power_data_coverage(self):
            return 1.0

        def get_per_ops_data(self):
            return {}

        def get_per_ops_source(self):
            return {}

        def get_moe_comm_fallbacks(self):
            return ()

    class FakeSession:
        def __init__(self, *, prefill_database, prefill_backend, decode_database, decode_backend):
            assert prefill_database is databases["prefill"]
            assert decode_database is databases["decode"]

        def set_latency_correction_scales(self, *_args):
            pass

        def run_disagg(self, **kwargs):
            captured.update(kwargs)
            return FakeSummary()

    monkeypatch.setattr(api, "check_is_moe", lambda _model_path: True)
    monkeypatch.setattr(api, "_resolve_moe_parallelism", lambda *args, **kwargs: (1, 64))
    monkeypatch.setattr(api, "resolve_model_config_moe_comm", fake_resolve)
    monkeypatch.setattr(
        api,
        "resolve_context_fmha_by_data",
        lambda *args, **kwargs: pytest.fail("resolved WideEP config must bypass generic FMHA fallback"),
    )
    monkeypatch.setattr(api, "resolve_dsv4_moe_arch", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_nvfp4_for_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_session, "DisaggInferenceSession", FakeSession)

    api._run_disagg_estimate(
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="prefill",
        decode_system_name="decode",
        backend_name="vllm",
        resolved_version="0.24.0",
        isl=256,
        osl=256,
        image_height=0,
        image_width=0,
        num_images=1,
        video_height=0,
        video_width=0,
        video_frames=0,
        num_videos=0,
        num_video_tokens=0,
        enable_encoder_dp=False,
        prefill_tp_size=8,
        prefill_pp_size=1,
        prefill_attention_dp_size=1,
        prefill_moe_tp_size=8,
        prefill_moe_ep_size=1,
        prefill_batch_size=32,
        prefill_num_workers=1,
        decode_tp_size=1,
        decode_pp_size=1,
        decode_attention_dp_size=64,
        decode_moe_tp_size=1,
        decode_moe_ep_size=64,
        decode_batch_size=512,
        decode_num_workers=1,
        gemm_quant_mode=None,
        kvcache_quant_mode=None,
        fmha_quant_mode=None,
        moe_quant_mode=None,
        comm_quant_mode=None,
        load_database=lambda system: databases[system],
        get_backend=lambda _backend: object(),
        get_model=lambda *_args: object(),
    )

    assert [call[1]["required_phases"] for call in resolver_calls] == [("context",), ("generation",)]
    assert resolver_calls[0][1]["database"] is databases["prefill"]
    assert resolver_calls[1][1]["database"] is databases["decode"]
    assert captured["prefill_model_config"].moe_comm_backend == {"context": "deepep_ht"}
    assert captured["decode_model_config"].moe_comm_backend == {"generation": "deepep_ll"}


_DENSE_GEMMA4 = "nvidia/Gemma-4-31B-IT-NVFP4"
_ROUTED_GEMMA4 = "nvidia/Gemma-4-26B-A4B-NVFP4"


def test_dense_gemma4_moe_parallelism_is_passed_through_untouched():
    """A dense checkpoint has no MoE width to resolve; a routed one still does.

    ``_resolve_moe_parallelism`` short-circuits on ``check_is_moe``, so dense
    Gemma 4 never reaches ``ModelConfig.resolve_moe_parallelism`` (which would
    reject or invent an expert layout for a checkpoint without routed experts).
    """
    import aisim.legacy_cli.api as api

    assert api._resolve_moe_parallelism(1, 1, 2, 4, model_path=_DENSE_GEMMA4) == (2, 4)
    assert api._resolve_moe_parallelism(4, 1, None, 4, model_path=_ROUTED_GEMMA4) == (1, 4)


class _FakeStaticSummary:
    def check_oom(self):
        return False

    def get_result_dict(self):
        return {"ttft": 0.0, "tpot": 1.0, "power_w": 1.0}

    def get_power_data_coverage(self):
        return 1.0

    def get_moe_comm_fallbacks(self):
        return ()

    def get_per_ops_data(self):
        return {}

    def get_per_ops_source(self):
        return None


def _run_static(api, model_path, *, tp, moe_tp, moe_ep):
    return api._run_static_estimate(
        static_mode="static",
        model_path=model_path,
        system_name="gb200",
        backend_name="trtllm",
        resolved_version="test",
        isl=8192,
        osl=1024,
        image_height=0,
        image_width=0,
        num_images=1,
        video_height=0,
        video_width=0,
        video_frames=0,
        num_videos=0,
        num_video_tokens=0,
        enable_encoder_dp=True,
        batch_size=10,
        prefix=0,
        tp_size=tp,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        gemm_quant_mode=None,
        kvcache_quant_mode=None,
        fmha_quant_mode=None,
        moe_quant_mode=None,
        comm_quant_mode=None,
        nextn=0,
        nextn_accepted=None,
        stride=32,
        engine_step_backend="rust",
        load_database=lambda _system: object(),
        get_backend=lambda _backend, *_, **_kw: object(),
        get_model=lambda *_args: object(),
    )


def test_dense_gemma4_static_estimate_skips_moe_comm_resolution(monkeypatch):
    import aisim.legacy_cli.api as api
    import aisim.sdk.inference_session as inference_session

    class FakeSession:
        def __init__(self, model, database, backend):
            pass

        def run_static(self, **kwargs):
            return _FakeStaticSummary()

    monkeypatch.setattr(
        api,
        "resolve_model_config_moe_comm",
        lambda *args, **kwargs: pytest.fail("a dense checkpoint must not resolve MoE communication"),
    )
    monkeypatch.setattr(api, "resolve_context_fmha_by_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_dsv4_moe_arch", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_nvfp4_for_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_session, "InferenceSession", FakeSession)

    result = _run_static(api, _DENSE_GEMMA4, tp=1, moe_tp=None, moe_ep=None)

    assert result.ttft == 0.0


def test_routed_gemma4_static_estimate_still_resolves_moe_comm(monkeypatch):
    import aisim.legacy_cli.api as api
    import aisim.sdk.inference_session as inference_session

    calls = []

    def fake_resolve(model_config, **kwargs):
        calls.append(kwargs)
        model_config.moe_comm_backend = {"context": "nvlink_two_sided", "generation": "nvlink_two_sided"}
        model_config.num_gpus_per_node = 4

    class FakeSession:
        def __init__(self, model, database, backend):
            pass

        def run_static(self, **kwargs):
            return _FakeStaticSummary()

    monkeypatch.setattr(api, "resolve_model_config_moe_comm", fake_resolve)
    monkeypatch.setattr(api, "resolve_context_fmha_by_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_dsv4_moe_arch", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_nvfp4_for_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_session, "InferenceSession", FakeSession)

    result = _run_static(api, _ROUTED_GEMMA4, tp=4, moe_tp=None, moe_ep=4)

    assert result.ttft == 0.0
    assert [call["required_phases"] for call in calls] == [("context", "generation")]


def test_dense_gemma4_disagg_estimate_skips_moe_comm_resolution(monkeypatch):
    """The disagg gate must stop before either role resolves MoE communication."""
    import aisim.legacy_cli.api as api
    import aisim.sdk.inference_session as inference_session

    class FakeSession:
        def __init__(self, *, prefill_database, prefill_backend, decode_database, decode_backend):
            pass

        def set_latency_correction_scales(self, *_args):
            pass

        def run_disagg(self, **kwargs):
            return _FakeStaticSummary()

    monkeypatch.setattr(
        api,
        "resolve_model_config_moe_comm",
        lambda *args, **kwargs: pytest.fail("a dense checkpoint must not resolve MoE communication"),
    )
    monkeypatch.setattr(api, "resolve_context_fmha_by_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_dsv4_moe_arch", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_nvfp4_for_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_session, "DisaggInferenceSession", FakeSession)

    api._run_disagg_estimate(
        model_path=_DENSE_GEMMA4,
        system_name="prefill",
        decode_system_name="decode",
        backend_name="vllm",
        resolved_version="0.24.0",
        isl=256,
        osl=256,
        image_height=0,
        image_width=0,
        num_images=1,
        video_height=0,
        video_width=0,
        video_frames=0,
        num_videos=0,
        num_video_tokens=0,
        enable_encoder_dp=False,
        prefill_tp_size=1,
        prefill_pp_size=1,
        prefill_attention_dp_size=1,
        prefill_moe_tp_size=None,
        prefill_moe_ep_size=None,
        prefill_batch_size=32,
        prefill_num_workers=1,
        decode_tp_size=1,
        decode_pp_size=1,
        decode_attention_dp_size=1,
        decode_moe_tp_size=None,
        decode_moe_ep_size=None,
        decode_batch_size=512,
        decode_num_workers=1,
        gemm_quant_mode=None,
        kvcache_quant_mode=None,
        fmha_quant_mode=None,
        moe_quant_mode=None,
        comm_quant_mode=None,
        load_database=lambda _system: object(),
        get_backend=lambda _backend: object(),
        get_model=lambda *_args: object(),
    )
