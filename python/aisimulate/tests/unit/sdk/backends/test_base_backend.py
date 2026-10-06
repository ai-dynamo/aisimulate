# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aisimulate.sdk import common
from aisimulate.sdk.backends.base_backend import BaseBackend
from aisimulate.sdk.config import ModelConfig, RuntimeConfig
from aisimulate.sdk.performance_result import MoECommFallback
from aisimulate.sdk.step_estimate import MixedStepInput, StepEstimate

pytestmark = pytest.mark.unit

_CONTEXT_FALLBACK = MoECommFallback("context", "deepep_ht", 32, 8, 8, 1)
_GENERATION_FALLBACK = MoECommFallback("generation", "deepep_ll", 32, 8, 8, 1)


def _decode_step(latency, energy, per_op_latency, per_op_source, fallbacks):
    return StepEstimate(
        latency_ms=latency,
        energy_wms=energy,
        per_op_latency_ms=per_op_latency,
        per_op_source=per_op_source,
        moe_comm_fallbacks=fallbacks,
    )


class _LatencyResult:
    def __init__(self, latency_ms: float, energy_wms: float) -> None:
        self._latency_ms = latency_ms
        self.energy = energy_wms

    def __float__(self) -> float:
        return self._latency_ms


class _StaticOp:
    def __init__(self, name: str, latency_ms: float, energy_wms: float) -> None:
        self._name = name
        self._latency_ms = latency_ms
        self._energy_wms = energy_wms

    def query(self, *args, **kwargs) -> _LatencyResult:
        return _LatencyResult(self._latency_ms, self._energy_wms)


class _TestBackend(BaseBackend):
    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError

    def _get_memory_usage(
        self,
        model,
        database,
        batch_size,
        beam_width,
        isl,
        osl,
        num_tokens=0,
        prefix=0,
        encoder_memory=None,
        mtp_scaled_tokens=None,
    ) -> dict[str, float]:
        return {"total": 1.0}


@pytest.fixture
def backend() -> BaseBackend:
    return _TestBackend()


@pytest.fixture
def database():
    return SimpleNamespace(
        backend="test-backend",
        version="test-version",
        system="test-system",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
    )


@pytest.fixture
def model():
    model = MagicMock()
    model.model_path = "test-model"
    model.model_name = "test-model"
    model.forward_model = "op_level"
    model._nextn = 0
    model.get_resident_weights_bytes.return_value = 0.0
    model.get_additional_activation_bytes.return_value = 0.0
    model.encoder_ops = []
    model.context_ops = [
        _StaticOp("context_attention", latency_ms=11.0, energy_wms=110.0),
        _StaticOp("logits_gemm", latency_ms=3.0, energy_wms=30.0),
    ]
    model.generation_ops = [
        _StaticOp("generation_attention", latency_ms=2.0, energy_wms=20.0),
        _StaticOp("generation_mlp", latency_ms=1.0, energy_wms=10.0),
    ]
    model.config = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )
    return model


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)


class TestMTPActivationMemoryScaling:
    """MTP activation scaling applies only to the decode-token share."""

    @staticmethod
    def _model():
        return SimpleNamespace(
            context_ops=[SimpleNamespace(get_weights=lambda: 0.0)],
            get_resident_weights_bytes=lambda: 0.0,
            get_additional_activation_bytes=lambda _tokens: 0.0,
            config=ModelConfig(
                tp_size=1,
                pp_size=1,
                attention_dp_size=1,
                moe_tp_size=1,
                moe_ep_size=1,
            ),
            _num_heads=32,
            _head_size=128,
            _num_experts=0,
            model_family="test",
            get_kvcache_bytes_per_sequence=lambda _seq_len: 0.0,
            get_kvcache_rank_bytes_per_sequence=lambda _seq_len: 0.0,
        )

    @staticmethod
    def _database():
        return SimpleNamespace(system_spec={"misc": {"nccl_mem": {1: 0}, "other_mem": 0}})

    def _activations(self, *, nextn: int, num_tokens: int, mtp_scaled_tokens: int | None) -> float:
        model = self._model()
        model.config.nextn = nextn
        return BaseBackend()._get_memory_usage(
            model,
            self._database(),
            batch_size=4,
            beam_width=1,
            isl=65_536,
            osl=400,
            num_tokens=num_tokens,
            mtp_scaled_tokens=mtp_scaled_tokens,
        )["activations"]

    def test_mixed_step_scales_only_decode_share(self):
        num_tokens = 65_536 + 3
        base = self._activations(nextn=0, num_tokens=num_tokens, mtp_scaled_tokens=3)
        spec = self._activations(nextn=3, num_tokens=num_tokens, mtp_scaled_tokens=3)
        assert spec / base == pytest.approx((65_536 + 3 * 4) / num_tokens, rel=1e-6)

    def test_decode_only_step_keeps_full_multiplier(self):
        base = self._activations(nextn=0, num_tokens=512, mtp_scaled_tokens=None)
        spec = self._activations(nextn=3, num_tokens=512, mtp_scaled_tokens=None)
        assert spec / base == pytest.approx(4.0, rel=1e-6)

    def test_prefill_only_step_does_not_scale(self):
        base = self._activations(nextn=0, num_tokens=65_536, mtp_scaled_tokens=0)
        spec = self._activations(nextn=3, num_tokens=65_536, mtp_scaled_tokens=0)
        assert spec == pytest.approx(base, rel=1e-9)


class TestAFDPartitionActivationScaling:
    """``get_partition_memory_usage`` must honor the AFD ``num_tokens`` contract.

    ``InferenceSession._estimate_{a,f}_memory_dict`` passes ``num_tokens=0`` for
    ``phase == "prefill"`` -- the sentinel that makes ``_get_memory_usage`` derive
    the count from ``(isl - prefix) * batch_size``, a prefill-only footprint --
    and ``num_tokens=batch_size`` (one token per request) for decode.
    """

    MODEL = "openai/gpt-oss-120b"

    @classmethod
    def _model(cls, *, nextn: int):
        from aisimulate.sdk.models import get_model

        model = get_model(cls.MODEL, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
        model.config.nextn = nextn
        return model

    @staticmethod
    def _database():
        return SimpleNamespace(
            backend="vllm",
            version="test-version",
            system="b200_sxm",
            system_spec={
                "gpu": {"mem_capacity": 180 * (1 << 30)},
                "misc": {"nccl_mem": {1: 0, 2: 0, 4: 0, 8: 0}, "other_mem": 0},
            },
        )

    def _partition_activations(self, *, nextn: int, num_tokens: int) -> float:
        from aisimulate.sdk.backends.factory import get_backend

        return get_backend("vllm").get_partition_memory_usage(
            self._model(nextn=nextn),
            self._database(),
            partition_ops=[],
            batch_size=4,
            beam_width=1,
            isl=65_536,
            osl=400,
            num_tokens=num_tokens,
            max_seq_len=65_536,
        )["activations"]

    def test_prefill_partition_does_not_scale(self):
        """AFD prefill workers size activations from the derived context tokens.

        Pins the measured shape (gpt-oss-120b, ISL 65,536, batch 4, TP1): the
        context footprint is 44.0 GiB, which the full ``(nextn+1)`` multiplier
        turned into 176.0 GiB -- a 4x over-count that shrinks the A/F worker KV
        budget and over-prunes AFD batch sizes.
        """
        base = self._partition_activations(nextn=0, num_tokens=0)
        spec = self._partition_activations(nextn=3, num_tokens=0)
        assert base == pytest.approx(44.0, abs=0.1)
        assert spec == pytest.approx(base, rel=1e-9)

    def test_decode_partition_keeps_full_multiplier(self):
        """AFD decode workers pass ``num_tokens=batch_size``: every token is verified."""
        base = self._partition_activations(nextn=0, num_tokens=4)
        spec = self._partition_activations(nextn=3, num_tokens=4)
        assert spec / base == pytest.approx(4.0, rel=1e-6)


@pytest.mark.parametrize("mode", ["static", "static_ctx", "static_gen"])
@pytest.mark.parametrize("latency_correction_scale", [1.0, 1.25])
def test_run_static_latency_only_matches_run_static_latency(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    mode: str,
    latency_correction_scale: float,
) -> None:
    from aisimulate.sdk.backends import base_backend as base_backend_module

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        # Mode- and scale-aware like the real bridge: the engine applies the
        # flat correction scale itself and only fills the phases the mode ran.
        ctx = {"context_attention": 11.0 * scale_arg, "logits_gemm": 3.0 * scale_arg}
        gen = {"generation_attention": 2.0 * scale_arg, "generation_mlp": 1.0 * scale_arg}
        if mode_arg == "static_ctx":
            gen = {}
        elif mode_arg == "static_gen":
            ctx = {}
        return (
            ctx,
            gen,
            {name: latency * 10.0 for name, latency in ctx.items()},
            {name: latency * 10.0 for name, latency in gen.items()},
            dict.fromkeys(ctx, "silicon"),
            dict.fromkeys(gen, "silicon"),
            (),
        )

    monkeypatch.setattr(base_backend_module, "estimate_static_latency_breakdown_with_rust", _fake_rust_breakdown)
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust")

    summary = backend.run_static(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )
    latency_only = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )

    summary_latency = sum(summary.get_context_latency_dict().values()) + sum(
        summary.get_generation_latency_dict().values()
    )
    request_latency = float(summary.get_summary_df().iloc[0]["request_latency"])

    assert latency_only == pytest.approx(summary_latency)
    assert latency_only == pytest.approx(request_latency, abs=1e-3)


def test_run_static_can_route_to_rust_engine_step_backend(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aisimulate.sdk.backends import base_backend as base_backend_module

    calls = []

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        calls.append((model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg))
        return (
            {"context_qkv_gemm": 4.0, "context_attention": 3.0},
            {"generation_qkv_gemm": 2.0, "generation_attention": 1.0},
            {"context_qkv_gemm": 40.0, "context_attention": 8.0},
            {"generation_qkv_gemm": 12.0, "generation_attention": 5.0},
            {"context_qkv_gemm": "silicon", "context_attention": "empirical"},
            {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"},
            (_CONTEXT_FALLBACK,),
        )

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    summary = backend.run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode="static",
        stride=2,
        latency_correction_scale=1.25,
    )

    assert len(calls) == 1
    assert calls[0][3:] == ("static", 2, 1.25)
    # Real op-name keys folded from the compiled engine's per-op results.
    assert summary.get_context_latency_dict() == {"context_qkv_gemm": 4.0, "context_attention": 3.0}
    assert summary.get_generation_latency_dict() == {"generation_qkv_gemm": 2.0, "generation_attention": 1.0}
    # Per-op energies are real and DISTINCT per key — not one phase total
    # smeared onto every key, and not recomputed by a Python double-eval.
    assert summary.get_context_energy_wms_dict() == {"context_qkv_gemm": 40.0, "context_attention": 8.0}
    assert summary.get_generation_energy_wms_dict() == {"generation_qkv_gemm": 12.0, "generation_attention": 5.0}
    assert summary.get_context_source_dict() == {"context_qkv_gemm": "silicon", "context_attention": "empirical"}
    assert summary.get_generation_source_dict() == {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"}
    assert summary.get_moe_comm_fallbacks() == (_CONTEXT_FALLBACK,)


@pytest.mark.parametrize(
    ("mode", "expected_share"),
    [
        # "static" and "static_ctx" both size activations from the DERIVED
        # context token count ((isl - prefix) * batch_size), so no share of that
        # footprint verifies nextn+1 draft tokens.
        ("static", 0),
        ("static_ctx", 0),
        # Decode-only step (num_tokens = batch_size * beam_width): every token is
        # verified, so the full (nextn+1) multiplier stays (no decode-share cap).
        ("static_gen", None),
    ],
)
def test_run_static_declares_mtp_decode_share_per_mode(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
    mode: str,
    expected_share: int | None,
) -> None:
    """Each static mode must declare its decode-token share to ``_get_memory_usage``.

    The latency side is stubbed at the engine bridge (the compiled engine is the
    only static engine-step executor); memory sizing stays a Python
    ``_get_memory_usage`` call, which is the surface under test here.
    """
    from aisimulate.sdk.backends import base_backend as base_backend_module

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        ctx = {"context_attention": 1.0}
        gen = {"generation_attention": 1.0}
        if mode_arg == "static_ctx":
            gen = {}
        elif mode_arg == "static_gen":
            ctx = {}
        return (ctx, gen, dict(ctx), dict(gen), dict.fromkeys(ctx, "silicon"), dict.fromkeys(gen, "silicon"), ())

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    captured: dict = {}

    def _record(*args, **kwargs):
        captured.update(kwargs)
        return {"total": 1.0}

    monkeypatch.setattr(backend, "_get_memory_usage", _record)

    backend.run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode=mode,
        stride=2,
    )

    assert captured.get("mtp_scaled_tokens") == expected_share


def test_run_agg_declares_mtp_decode_share_at_call_site(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """run_agg must pass the decode-request share of ``num_tokens`` as
    ``mtp_scaled_tokens`` to ``_get_memory_usage``.

    The run_agg counterpart of test_run_static_declares_mtp_decode_share_per_mode
    above: pins the CALL SITE for a mixed step (b > 1: only the num_gen_requests
    decode requests verify nextn+1 draft tokens; the context share is processed
    once). The b == 1 context-only step and its separate decode peak are pinned
    by test_run_agg_b1_uses_scheduled_activation_peak below.
    """
    memory_calls: list[dict] = []

    def _record(*args, **kwargs):
        memory_calls.append(kwargs)
        return {"total": 1.0}

    monkeypatch.setattr(backend, "_get_memory_usage", _record)
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {}, {}, ()),
    )

    backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=4, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    # ctx_tokens budgets uncached tokens: 8 new tokens pack ceil(8 / (8 - 2)) = 2
    # prefilling requests, leaving b - 2 = 2 decode requests.
    assert len(memory_calls) == 1
    assert memory_calls[0].get("mtp_scaled_tokens") == 2


def test_trtllm_budget_path_ignores_the_decode_share() -> None:
    """TRT-LLM's agg override intentionally drops ``mtp_scaled_tokens``.

    Its ``num_tokens`` is ``BuildConfig.max_num_tokens`` (a per-iteration budget),
    not the agg-derived mixed-step count, so the decode-share correction is not
    forwarded and the legacy full ``(nextn+1)`` multiplier is retained on that
    path pending its own analysis (tracked in AIC-1755).
    """
    from aisimulate.sdk.backends.factory import get_backend

    agg_extra = {"max_num_tokens": 8192, "max_seq_len": 4096, "free_gpu_memory_fraction": 0.9}
    kwargs = get_backend("trtllm")._memory_usage_kwargs_for_agg(
        num_tokens=65_539,
        agg_extra=agg_extra,
        mtp_scaled_tokens=3,
    )
    assert kwargs == {"num_tokens": 8192, "max_seq_len": 4096}


def test_run_agg_with_osl_one_does_not_divide_by_zero(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """Regression: osl=1 (no-decode) must not raise and tokens/s/user must be 0.0."""
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(0.0, 0.0, {}, {}, ()),
    )
    monkeypatch.setattr(
        backend,
        "_get_memory_usage",
        lambda *args, **kwargs: {"total": 1.0},
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=1, prefix=2, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    row = summary.get_summary_df().iloc[0]
    assert row["tpot"] > 0.0
    assert row["tokens/s/user"] == 0.0


@pytest.mark.parametrize(
    ("osl", "expected_memory_calls", "expected_activations"),
    [
        (1, [(8, 0)], 0.068359375),
        (5, [(8, 0), (1, None)], 0.2734375),
    ],
)
def test_run_agg_b1_uses_scheduled_activation_peak(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    osl: int,
    expected_memory_calls: list[tuple[int, int | None]],
    expected_activations: float,
) -> None:
    """A b=1 agg run retains a decode peak only when decode is scheduled."""
    model._nextn = 3
    model.config.nextn = 3
    model.context_ops = [SimpleNamespace(get_weights=lambda: 0.0)]
    model._num_heads = 32
    model._head_size = 128
    model._num_experts = 0
    model.model_family = "test"
    model.get_kvcache_bytes_per_sequence = lambda _seq_len: 0.0
    model.get_kvcache_rank_bytes_per_sequence = lambda _seq_len: 0.0
    database.system_spec["misc"] = {"nccl_mem": {1: 0}, "other_mem": 0}
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {"decode": 1.0}, {"decode": "silicon"}, ()),
    )

    memory_calls: list[dict] = []

    def _get_memory_usage(*args, **kwargs):
        memory_calls.append(kwargs)
        return BaseBackend._get_memory_usage(backend, *args, **kwargs)

    monkeypatch.setattr(backend, "_get_memory_usage", _get_memory_usage)

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=osl, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert [(call["num_tokens"], call["mtp_scaled_tokens"]) for call in memory_calls] == expected_memory_calls
    assert summary.get_memory()["activations"] == pytest.approx(expected_activations)


def test_run_mixed_returns_components_and_counts_speculative_query_tokens(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aisimulate.sdk.backends import base_backend as base_backend_module

    calls: list[dict] = []

    def _fake_mixed_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return {
            "latency_ms": 8.5,
            "energy_wms": 85.0,
            "component_latency_ms": {"shared_non_attention": 5.0, "context_attention": 2.0, "decode_attention": 1.5},
            "component_energy_wms": {"shared_non_attention": 50.0, "context_attention": 20.0, "decode_attention": 15.0},
            "per_op_latency_ms": {"context_mlp": 5.0},
            "per_op_source": {"context_mlp": "silicon"},
            "moe_comm_fallbacks": (),
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", _fake_mixed_breakdown)
    model._nextn = 2

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        MixedStepInput(
            context_tokens=8,
            num_decode_requests=2,
        ),
    )

    assert estimate.num_decode_requests == 2
    # Every decode request verifies one target token plus the two scheduled
    # drafts (nextn=2); scheduling metadata is computed before the bridge call.
    assert estimate.num_decode_query_tokens == 6
    assert estimate.latency_ms == pytest.approx(sum(estimate.component_latency_ms.values()))
    assert set(estimate.component_latency_ms) == {
        "shared_non_attention",
        "context_attention",
        "decode_attention",
    }
    # The bridge receives the RAW decode-request count; the engine applies the
    # (nextn + 1) scaling internally, exactly like the deleted Python passes.
    assert calls[0]["gen_tokens"] == 2
    assert calls[0]["ctx_tokens"] == 8


def test_run_mixed_rust_path_returns_the_same_structured_contract(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The rust branch maps the bridge's rich dict 1:1 onto ``StepEstimate``:
    real energy, per-component splits, and the Python branch's per-op keys
    (raw non-attention names plus the two literal attention keys) — the
    synthetic "rust_engine_step_mixed" key no longer exists."""
    from aisimulate.sdk.backends import base_backend as base_backend_module

    model._nextn = 2
    monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args: True)
    components = {
        "latency_ms": 8.5,
        "energy_wms": 85.0,
        "component_latency_ms": {"shared_non_attention": 5.0, "context_attention": 2.0, "decode_attention": 1.5},
        "component_energy_wms": {"shared_non_attention": 50.0, "context_attention": 20.0, "decode_attention": 15.0},
        "per_op_latency_ms": {"context_mlp": 5.0, "context_attention (scaled)": 2.0, "generation_attention": 1.5},
        "per_op_source": {
            "context_mlp": "silicon",
            "context_attention (scaled)": "empirical",
            "generation_attention": "silicon",
        },
        "moe_comm_fallbacks": (),
    }
    monkeypatch.setattr(
        base_backend_module,
        "estimate_mixed_step_breakdown_with_rust",
        lambda *args, **kwargs: components,
    )

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, engine_step_backend="rust"),
        MixedStepInput(context_tokens=8, num_decode_requests=2),
    )

    assert estimate.latency_ms == 8.5
    assert estimate.energy_wms == 85.0  # real energy, not a 0.0 placeholder
    assert estimate.component_latency_ms == components["component_latency_ms"]
    assert estimate.component_energy_wms == components["component_energy_wms"]
    assert estimate.per_op_latency_ms == components["per_op_latency_ms"]
    assert "context_attention (scaled)" in estimate.per_op_latency_ms
    assert estimate.per_op_source == components["per_op_source"]
    assert estimate.context_tokens == 8
    assert estimate.num_decode_requests == 2
    assert estimate.num_decode_query_tokens == 6


def test_get_genonly_step_latency_rust_path_returns_decode_breakdown_verbatim(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The rust branch returns ``estimate_decode_step_breakdown_with_rust``
    verbatim — real op names, real per-op energy (no synthetic
    "rust_engine_step_generation" key) — without running the Python static
    step."""
    from aisimulate.sdk.backends import base_backend as base_backend_module

    calls = []
    breakdown = (
        3.0,
        24.0,
        {"generation_attention": 2.0, "generation_mlp": 1.0},
        {"generation_attention": "silicon", "generation_mlp": "empirical"},
        (),
    )

    def _fake_decode_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return breakdown

    monkeypatch.setattr(base_backend_module, "estimate_decode_step_breakdown_with_rust", _fake_decode_breakdown)

    def _python_step_trap(*args, **kwargs):
        raise AssertionError("the rust path must not run the Python static step")

    monkeypatch.setattr(backend, "run_static", _python_step_trap)

    result = backend._get_genonly_step_latency(
        model,
        database,
        RuntimeConfig(isl=8, osl=6, gen_seq_imbalance_correction_scale=1.25, engine_step_backend="rust"),
        gen_tokens=4,
        isl=8,
        osl=6,
    )

    assert result == breakdown
    assert calls == [{"gen_tokens": 4, "isl": 8, "osl": 6, "gen_seq_imbalance_correction_scale": 1.25}]


def test_run_agg_applies_speculative_progress_in_scheduler(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model._nextn = 1
    seen_steps: list[MixedStepInput] = []

    def _run_mixed(*args, **kwargs):
        step = args[-1]
        seen_steps.append(step)
        return StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            component_latency_ms={"shared_non_attention": 10.0},
            component_energy_wms={"shared_non_attention": 100.0},
            num_decode_requests=step.num_decode_requests,
            num_decode_query_tokens=step.num_decode_requests * 2,
        )

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}, ()),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
        decode_tokens_per_iteration=2.0,
    )
    row = summary.get_summary_df().iloc[0]
    scheduling = summary.get_step_estimates()["scheduling"]

    assert seen_steps[0].num_decode_requests == 1
    assert scheduling["decode_tokens_per_iteration"] == 2.0
    assert scheduling["decode_iterations"] == 3.0
    assert scheduling["num_mix_steps"] == 2.0
    assert scheduling["num_genonly_steps"] == 1.0
    assert row["tpot"] == pytest.approx(4.167)
    assert row["tokens/s"] == pytest.approx(320.0)


def test_run_agg_records_progress_only_when_explicitly_supplied(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """An omitted decode_tokens_per_iteration must leave no scheduler-progress
    marker on the summary: its presence is what tells the upper-layer
    projection to stand down, so recording the 1.0 default would silently
    disable the legacy post-hoc projection flow."""
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=10.0, energy_wms=100.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}, ()),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert "decode_tokens_per_iteration" not in summary.get_step_estimates()["scheduling"]

    explicit = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
        decode_tokens_per_iteration=1.0,
    )
    assert explicit.get_step_estimates()["scheduling"]["decode_tokens_per_iteration"] == 1.0
    # The two calls schedule identically but carry different projection
    # eligibility; they must not have shared a cache entry.
    assert explicit is not summary


def test_run_agg_preserves_executed_moe_fallbacks_from_mixed_and_decode_steps(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            moe_comm_fallbacks=(_CONTEXT_FALLBACK,),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(
            5.0,
            50.0,
            {"decode": 5.0},
            {"decode": "estimated"},
            (_CONTEXT_FALLBACK, _GENERATION_FALLBACK),
        ),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert summary.get_moe_comm_fallbacks() == (_CONTEXT_FALLBACK, _GENERATION_FALLBACK)


def test_run_agg_omits_decode_fallback_when_decode_step_has_zero_weight(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            moe_comm_fallbacks=(_CONTEXT_FALLBACK,),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(
            5.0,
            50.0,
            {"decode": 5.0},
            {"decode": "estimated"},
            (_GENERATION_FALLBACK,),
        ),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=1, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert summary.get_moe_comm_fallbacks() == (_CONTEXT_FALLBACK,)
    assert summary.get_per_ops_data()["scheduling"]["num_genonly_steps"] == 0.0


def test_run_agg_cache_separates_video_workloads(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model.encoder_config = _vision_encoder_config()
    mixed_calls: list[int] = []

    def _run_mixed(*args, **kwargs):
        mixed_calls.append(args[2].video_frames)
        return StepEstimate(latency_ms=1.0, energy_wms=1.0)

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {}, {}, ()),
    )

    common_kwargs = dict(
        batch_size=2,
        beam_width=1,
        isl=8,
        osl=5,
        video_height=448,
        video_width=448,
        num_videos_per_request=1,
        engine_step_backend="rust",
    )
    backend.run_agg(model, database, RuntimeConfig(**common_kwargs, video_frames=8), ctx_tokens=8)
    backend.run_agg(model, database, RuntimeConfig(**common_kwargs, video_frames=10), ctx_tokens=8)

    assert mixed_calls == [8, 10]
    assert len(backend._agg_cache) == 2


def test_run_agg_cache_separates_prefix_on_a_reused_backend(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """Two calls differing only in prefix must not share a cached summary."""
    mixed_calls: list[int] = []

    def _run_mixed(*args, **kwargs):
        prefix = args[2].prefix
        mixed_calls.append(prefix)
        # Make the answer depend on prefix so a stale hit is observable.
        return StepEstimate(latency_ms=10.0 - prefix, energy_wms=100.0)

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {}, {}, ()),
    )

    common_kwargs = dict(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust")
    no_prefix = backend.run_agg(model, database, RuntimeConfig(**common_kwargs, prefix=0), ctx_tokens=8)
    with_prefix = backend.run_agg(model, database, RuntimeConfig(**common_kwargs, prefix=4), ctx_tokens=8)

    assert mixed_calls == [0, 4]
    assert len(backend._agg_cache) == 2
    assert no_prefix is not with_prefix
    assert no_prefix.get_result_dict()["prefix"] == 0
    assert with_prefix.get_result_dict()["prefix"] == 4
    assert no_prefix.get_result_dict()["ttft"] != with_prefix.get_result_dict()["ttft"]

    # An identical request still hits the cache without re-estimating.
    repeat = backend.run_agg(model, database, RuntimeConfig(**common_kwargs, prefix=4), ctx_tokens=8)
    assert repeat is with_prefix
    assert mixed_calls == [0, 4]
    assert len(backend._agg_cache) == 2

    # prefix=None normalizes like run_mixed (int(prefix or 0)): it shares the
    # prefix=0 entry and echoes 0, not None.
    assert backend.run_agg(model, database, RuntimeConfig(**common_kwargs, prefix=None), ctx_tokens=8) is no_prefix
    assert mixed_calls == [0, 4]
    assert len(backend._agg_cache) == 2


@pytest.mark.parametrize("field", ["seq_imbalance_correction_scale", "gen_seq_imbalance_correction_scale"])
def test_run_agg_cache_separates_seq_imbalance_correction_scales(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    field: str,
) -> None:
    seen: list[float] = []

    def _run_mixed(*args, **kwargs):
        seen.append(getattr(args[2], field))
        return StepEstimate(latency_ms=1.0, energy_wms=1.0)

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {}, {}, ()),
    )

    common_kwargs = dict(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust")
    backend.run_agg(model, database, RuntimeConfig(**common_kwargs, **{field: 1.0}), ctx_tokens=8)
    backend.run_agg(model, database, RuntimeConfig(**common_kwargs, **{field: 1.5}), ctx_tokens=8)

    assert seen == [1.0, 1.5]
    assert len(backend._agg_cache) == 2


@pytest.mark.parametrize(
    ("engine_step_backend", "error", "match"),
    [
        ("auto", ValueError, "unknown engine_step_backend 'auto'"),
        (None, TypeError, "compiled engine is the only aggregate engine-step executor"),
    ],
)
def test_run_agg_validates_backend_before_cache_hit(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    engine_step_backend: str | None,
    error: type[Exception],
    match: str,
) -> None:
    """A cached Rust summary must not bypass request/database validation."""
    monkeypatch.delenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", raising=False)
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=10.0, energy_wms=100.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}, ()),
    )

    cached = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert (
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
            ctx_tokens=8,
        )
        is cached
    )

    with pytest.raises(error, match=match):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend=engine_step_backend),
            ctx_tokens=8,
        )


@pytest.mark.parametrize("progress", [0.0, 3.0, float("inf"), float("nan")])
def test_run_agg_rejects_invalid_speculative_progress(
    progress,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model._nextn = 1

    with pytest.raises(ValueError, match="decode_tokens_per_iteration"):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5),
            ctx_tokens=8,
            decode_tokens_per_iteration=progress,
        )


def _vision_encoder_config() -> common.VisionEncoderConfig:
    return common.VisionEncoderConfig(
        depth=1,
        hidden_size=8,
        num_heads=1,
        intermediate_size=8,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=8,
    )


def test_run_mixed_derives_effective_multimodal_isl_for_direct_calls(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """Regression: a direct run_mixed call with a plain (text-only isl) config
    must hand the bridge the image-augmented effective isl, exactly as
    run_static / run_agg model it. Text isl=8 plus 16 visual tokens gives
    effective isl 24 — the visual adjustment happens before the bridge call,
    not inside the engine."""
    from aisimulate.sdk.backends import base_backend as base_backend_module

    model.encoder_config = _vision_encoder_config()

    calls: list[dict] = []

    def _fake_mixed_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return {
            "latency_ms": 1.0,
            "energy_wms": 1.0,
            "component_latency_ms": {"shared_non_attention": 1.0},
            "component_energy_wms": {"shared_non_attention": 1.0},
            "per_op_latency_ms": {},
            "per_op_source": {},
            "moe_comm_fallbacks": (),
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", _fake_mixed_breakdown)

    backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=6, num_images_per_request=1, num_image_tokens=16, engine_step_backend="rust"),
        MixedStepInput(context_tokens=24, num_decode_requests=1),
    )

    assert calls[-1]["isl"] == 24
    assert calls[-1]["ctx_tokens"] == 24


def test_run_agg_does_not_double_count_visual_tokens_in_run_mixed(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """run_mixed owns the visual-token adjustment, so run_agg must hand it the
    unmodified runtime_config (a pre-adjusted copy would count images twice),
    while the genonly step keeps receiving run_agg's effective isl."""
    model.encoder_config = _vision_encoder_config()

    mixed_configs: list[RuntimeConfig] = []

    def _run_mixed(model_arg, database_arg, runtime_config_arg, step):
        mixed_configs.append(runtime_config_arg)
        return StepEstimate(latency_ms=1.0, energy_wms=1.0)

    genonly_isl: list[int] = []

    def _genonly(model_arg, database_arg, runtime_config_arg, num_tokens, isl, osl):
        genonly_isl.append(isl)
        return _decode_step(1.0, 1.0, {}, {}, ())

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", _genonly)

    runtime_config = RuntimeConfig(
        batch_size=2,
        beam_width=1,
        isl=8,
        osl=5,
        num_images_per_request=1,
        num_image_tokens=16,
        engine_step_backend="rust",
    )
    backend.run_agg(model, database, runtime_config, ctx_tokens=8)

    assert mixed_configs[0] is runtime_config
    assert genonly_isl == [8 + 16]


def test_mixed_step_requires_context_tokens() -> None:
    with pytest.raises(ValueError, match="context_tokens"):
        MixedStepInput(context_tokens=0, num_decode_requests=1)


def test_mix_step_efficiency_base_default_is_one(backend: BaseBackend) -> None:
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=16) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=0) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=0, gen_tokens=0) == 1.0


def test_run_static_latency_only_zeroes_energy_with_paired_keys(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """include_energy=False must zero the energy dicts while keeping their
    key sets identical to the latency dicts (the power coverage gate pairs
    latency and energy by name)."""
    from aisimulate.sdk.backends import base_backend as base_backend_module

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        lambda *args, **kwargs: (
            {"context_qkv_gemm": 4.0, "context_attention": 3.0},
            {"generation_qkv_gemm": 2.0, "generation_attention": 1.0},
            {"context_qkv_gemm": 40.0, "context_attention": 8.0},
            {"generation_qkv_gemm": 12.0, "generation_attention": 5.0},
            {"context_qkv_gemm": "silicon", "context_attention": "empirical"},
            {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"},
            (),
        ),
    )

    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust")
    latency = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode="static",
        stride=2,
    )
    assert latency == pytest.approx(10.0)

    (
        context_latency,
        context_energy,
        generation_latency,
        generation_energy,
        _,
        _,
        _,
    ) = backend._run_static_breakdown(model, database, runtime_config, mode="static", stride=2, include_energy=False)
    # Latency-only callers must not observe energy, but the key sets stay
    # paired with the latency dicts (the power coverage gate pairs by name).
    assert context_energy == {"context_qkv_gemm": 0.0, "context_attention": 0.0}
    assert generation_energy == {"generation_qkv_gemm": 0.0, "generation_attention": 0.0}
    assert context_energy.keys() == context_latency.keys()
    assert generation_energy.keys() == generation_latency.keys()


def test_step_requires_a_real_perf_database(
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The compiled engine resolves perf data from disk by identity, so a
    step over a synthetic database is a hard TypeError now that the Python
    step branch is gone (op-level and FPM models alike)."""
    with pytest.raises(TypeError, match="on-disk identity"):
        backend.run_static(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2),
            mode="static",
            stride=2,
        )
    with pytest.raises(TypeError, match="on-disk identity"):
        backend.run_mixed(
            model,
            database,
            RuntimeConfig(isl=8, osl=5),
            MixedStepInput(context_tokens=8, num_decode_requests=1),
        )
    with pytest.raises(TypeError, match="on-disk identity"):
        backend._get_genonly_step_latency(
            model,
            database,
            RuntimeConfig(isl=8, osl=5),
            gen_tokens=2,
            isl=8,
            osl=5,
        )


@pytest.mark.parametrize("encoder_energy,expected_coverage,published", [(1000.0, 0.9125, True), (0.0, 0.7875, False)])
def test_run_agg_retains_schedule_weighted_energy_for_publication(
    backend, model, database, monkeypatch, encoder_energy, expected_coverage, published
):
    from aisimulate.legacy_cli.api import _apply_power_coverage_gate

    monkeypatch.setattr(backend, "_mix_step_efficiency", lambda *_: 0.5)
    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_: ({"encoder": 10.0}, {"encoder": encoder_energy}, {"encoder": "silicon"}, 0),
    )
    mixed = StepEstimate(
        latency_ms=10.0,
        energy_wms=9000.0,
        covered_latency_ms=9.0,
        per_op_latency_ms={"gemm": 9.0, "missing": 1.0},
        per_op_energy_wms={"gemm": 9000.0, "missing": 0.0},
    )
    decode = StepEstimate(
        latency_ms=20.0,
        energy_wms=18000.0,
        covered_latency_ms=18.0,
        per_op_latency_ms={"gemm": 18.0, "missing": 2.0},
        per_op_energy_wms={"gemm": 18000.0, "missing": 0.0},
    )
    monkeypatch.setattr(backend, "run_mixed", lambda *_: mixed)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: decode)
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    groups = summary.get_aggregate_energy_breakdown()
    assert groups["mix_step"].latency_ms == 10.0  # two mixed steps, each scaled by 0.5
    assert groups["genonly_step"].latency_ms == 60.0  # three decode-only steps
    assert groups["encoder"].latency_ms == 10.0
    assert groups["mix_step"].per_op_energy_wms == {"gemm": 9000.0, "missing": 0.0}
    assert groups["genonly_step"].per_op_energy_wms == {"gemm": 54000.0, "missing": 0.0}
    assert summary.get_power_data_coverage() == pytest.approx(expected_coverage)
    raw = summary.get_result_dict()
    assert raw["power_w"] == pytest.approx((63000.0 + encoder_energy) / 80.0)
    gated = _apply_power_coverage_gate(summary, raw)
    assert (gated["power_w"] is not None) is published
    assert {k: v for k, v in gated.items() if not k.startswith("power_")} == {
        k: v for k, v in raw.items() if not k.startswith("power_")
    }


@pytest.mark.parametrize("covered,expected,published", [(9.0, 0.9, True), (8.9, 0.89, False), (0.0, 0.0, False)])
def test_run_agg_zero_decode_steps_do_not_change_energy_coverage(
    backend, model, database, monkeypatch, covered, expected, published
):
    from aisimulate.legacy_cli.api import _apply_power_coverage_gate

    mixed = StepEstimate(latency_ms=10.0, energy_wms=1000.0 if covered else 0.0, covered_latency_ms=covered)
    unused_decode = StepEstimate(latency_ms=1000.0, energy_wms=1000000.0, covered_latency_ms=1000.0)
    monkeypatch.setattr(backend, "run_mixed", lambda *_: mixed)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: unused_decode)
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, isl=8, osl=1, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert summary.get_aggregate_energy_breakdown()["genonly_step"].latency_ms == 0
    assert summary.get_power_data_coverage() == pytest.approx(expected)
    assert (_apply_power_coverage_gate(summary, summary.get_result_dict())["power_w"] is not None) is published


# ---------------------------------------------------------------------------
# Prefix-aware scheduling: ctx_tokens budgets UNCACHED prefill tokens
# ---------------------------------------------------------------------------

# A long-prompt shape one token past a multiple of the 16384-token per-step
# budget (SGLang --chunked-prefill-size), so every ceil in the schedule rounds
# up: ISL 32769 / OSL 512 / bs 16.
_ISL, _OSL, _BS, _CTX = 32769, 512, 16, 16384


def _fixed_step_costs(monkeypatch, backend: BaseBackend, mix_latency_ms: float = 10.0) -> list[RuntimeConfig]:
    """Pin the per-step costs so run_agg exposes only its scheduling; return
    the runtime configs handed to run_mixed."""
    seen: list[RuntimeConfig] = []

    def _run_mixed(model_arg, database_arg, runtime_config_arg, step):
        seen.append(runtime_config_arg)
        return StepEstimate(latency_ms=mix_latency_ms, energy_wms=1.0)

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {"decode": 1.0}, {"decode": "silicon"}, ()),
    )
    return seen


def _prefix_sweep_config(prefix: int) -> RuntimeConfig:
    return RuntimeConfig(batch_size=_BS, beam_width=1, isl=_ISL, osl=_OSL, prefix=prefix, engine_step_backend="rust")


@pytest.mark.parametrize(
    ("prefix", "expected_mix_steps"),
    [
        # ceil((isl - prefix) * b / ctx_tokens), hand-derived:
        (0, 33),  # ceil(32769 * 16 / 16384) = ceil(32.001)
        (8192, 25),  # ceil(24577 * 16 / 16384) = ceil(24.001)
        (16384, 17),  # ceil(16385 * 16 / 16384) = ceil(16.001)
        (24576, 9),  # ceil(8193 * 16 / 16384) = ceil(8.001)
        (29491, 4),  # ceil(3278 * 16 / 16384) = ceil(3.201)
    ],
)
def test_run_agg_mix_step_count_follows_uncached_isl(
    monkeypatch, backend: BaseBackend, model, database, prefix: int, expected_mix_steps: int
) -> None:
    """The ctx_tokens budget counts uncached tokens (what the engines'
    chunked-prefill knobs cap), so the mixed steps needed to drain a batch's
    prefill shrink with the cached prefix. The engine still receives the full
    isl and prefix through the unmodified runtime config, and the scheduling
    hook receives the uncached length."""
    seen = _fixed_step_costs(monkeypatch, backend)
    hook_isl: list[int] = []
    base_hook = backend._mix_step_gen_tokens

    def _hook(b, ctx_tokens, isl, decode_iterations):
        hook_isl.append(isl)
        return base_hook(b, ctx_tokens, isl, decode_iterations)

    monkeypatch.setattr(backend, "_mix_step_gen_tokens", _hook)

    summary = backend.run_agg(model, database, _prefix_sweep_config(prefix), ctx_tokens=_CTX)

    scheduling = summary.get_step_estimates()["scheduling"]
    assert scheduling["num_mix_steps"] == expected_mix_steps
    assert scheduling["num_genonly_steps"] == _OSL - expected_mix_steps
    result = summary.get_result_dict()
    assert result["num_ctx_reqs"] == math.ceil(_CTX / (_ISL - prefix))
    assert result["num_gen_reqs"] == _BS - result["num_ctx_reqs"]
    assert hook_isl == [_ISL - prefix]
    assert seen[0].isl == _ISL and seen[0].prefix == prefix


def test_run_agg_prefix_zero_schedule_is_unchanged(monkeypatch, backend: BaseBackend, model, database) -> None:
    """prefix=0 keeps the legacy schedule: ceil(isl*b/ctx) mixed steps,
    ceil(ctx/isl) prefilling requests, ceil(isl/ctx) TTFT chunks and the
    isl*b/ctx/osl balance score."""
    _fixed_step_costs(monkeypatch, backend, mix_latency_ms=10.0)
    summary = backend.run_agg(model, database, _prefix_sweep_config(0), ctx_tokens=_CTX)
    scheduling = summary.get_step_estimates()["scheduling"]
    result = summary.get_result_dict()
    assert scheduling["num_mix_steps"] == 33
    assert scheduling["num_genonly_steps"] == 479
    assert result["num_ctx_reqs"] == 1
    assert result["num_gen_reqs"] == 15
    assert result["balance_score"] == pytest.approx(_ISL * _BS / _CTX / _OSL)
    # 3 chunks of 16384 for a 32769-token prefill, times the queuing factor.
    assert result["ttft"] == pytest.approx(10.0 * 3 * backend._ttft_queuing_factor(_BS, 33))


def test_run_agg_ttft_chunk_count_follows_uncached_isl(monkeypatch, model, database) -> None:
    """A request's TTFT multiplies the prefill step by ceil((isl - prefix) /
    ctx_tokens): a cold 32769-token request takes 3 chunks of 16384, one with
    16384 cached tokens takes 2 and one with 24576 cached takes 1."""
    ttfts = []
    for prefix, chunks, steps in ((0, 3, 33), (16384, 2, 17), (24576, 1, 9)):
        # A fresh backend per prefix: the run_agg result cache is keyed on
        # the prefix by a separate fix, not by this change.
        backend = _TestBackend()
        _fixed_step_costs(monkeypatch, backend, mix_latency_ms=10.0)
        result = backend.run_agg(model, database, _prefix_sweep_config(prefix), ctx_tokens=_CTX).get_result_dict()
        assert result["ttft"] / backend._ttft_queuing_factor(_BS, steps) == pytest.approx(10.0 * chunks)
        ttfts.append(result["ttft"])
    assert ttfts[0] > ttfts[1] > ttfts[2]


@pytest.mark.parametrize("prefix", [8, 9])
def test_run_agg_rejects_prefix_at_or_beyond_isl(monkeypatch, backend: BaseBackend, model, database, prefix) -> None:
    """The engine rejects a mixed step whose requests carry no uncached
    token; run_agg applies the same contract before scheduling."""
    _fixed_step_costs(monkeypatch, backend)
    with pytest.raises(ValueError, match="prefix"):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=prefix, engine_step_backend="rust"),
            ctx_tokens=8,
        )


@pytest.mark.parametrize(
    ("b", "prefix", "isl_new"),
    [
        # ceil(16384 / 3278) = 5 prefilling requests would exceed the batch of 4.
        (4, 29491, 3278),
        # Every request carries a single uncached token: 16384 of them would
        # be 16384 requests.
        (16, _ISL - 1, 1),
    ],
)
def test_run_agg_caps_prefilling_requests_at_the_batch(
    monkeypatch, backend: BaseBackend, model, database, b: int, prefix: int, isl_new: int
) -> None:
    """A mixed step cannot hold more uncached tokens than the batch owns: the
    schedule runs on min(ctx_tokens, b * isl_new), so the request counts and
    the activation token count handed to the memory model stay non-negative
    and the step is priced at the tokens the batch can actually supply. The
    published ctx_tokens column keeps the requested engine knob."""
    steps: list[MixedStepInput] = []

    def _run_mixed(model_arg, database_arg, runtime_config_arg, step):
        steps.append(step)
        return StepEstimate(latency_ms=10.0, energy_wms=1.0)

    memory_calls: list[dict] = []

    def _memory(model_arg, database_arg, batch_size, beam_width, isl, osl, **kwargs):
        memory_calls.append(kwargs)
        return {"total": 1.0}

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {"decode": 1.0}, {"decode": "silicon"}, ()),
    )
    monkeypatch.setattr(backend, "_get_memory_usage", _memory)

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=b, beam_width=1, isl=_ISL, osl=_OSL, prefix=prefix, engine_step_backend="rust"),
        ctx_tokens=_CTX,
    )

    result = summary.get_result_dict()
    assert result["num_ctx_reqs"] == b
    assert result["num_gen_reqs"] == 0
    assert result["gen_tokens"] == 0
    assert result["ctx_tokens"] == _CTX
    assert result["num_tokens"] == b * isl_new
    assert steps[0].context_tokens == b * isl_new
    # Every request prefills in this step: no decode request rides along,
    # and the step does not enter the TPOT average (decode steps cost 1 ms).
    assert steps[0].num_decode_requests == 0
    assert result["tpot"] == pytest.approx(1.0)
    assert summary.get_step_estimates()["scheduling"]["num_mix_steps"] == 1
    assert memory_calls[0]["num_tokens"] == b * isl_new
    assert memory_calls[0]["mtp_scaled_tokens"] == 0
    # The whole batch prefills in one step of one chunk each.
    assert result["ttft"] == pytest.approx(10.0 * backend._ttft_queuing_factor(b, 1))


@pytest.mark.parametrize(
    ("b", "prefix", "ctx_tokens", "capped", "osl"),
    [
        # b == 1 with an explicit budget above its 16385 uncached tokens (e.g.
        # the pre-fix default --ctx-tokens <ISL>): no phantom second request.
        (1, 16384, _ISL, _ISL - 16384, _OSL),
        # prefix 0, b == 1, a budget above the isl: one request's prefill.
        (1, 0, 2 * _ISL, _ISL, _OSL),
        # prefix 0, b == 2, a budget above b * isl: the old schedule packed
        # ceil(ctx/isl) = 3 prefilling requests out of 2 (negative decode).
        (2, 0, 2 * _ISL + 100, 2 * _ISL, _OSL),
        # Exactly the batch's uncached tokens: the whole batch prefills in the
        # one step, at prefix 0 and with a prefix, for osl > 1 and osl == 1.
        (2, 0, 2 * _ISL, 2 * _ISL, _OSL),
        (2, 16384, 2 * (_ISL - 16384), 2 * (_ISL - 16384), _OSL),
        (2, 16384, 2 * (_ISL - 16384), 2 * (_ISL - 16384), 1),
    ],
)
def test_run_agg_caps_the_budget_for_every_batch_size(
    monkeypatch, backend: BaseBackend, model, database, b: int, prefix: int, ctx_tokens: int, capped: int, osl: int
) -> None:
    """The b * isl_new cap applies to every batch size and prefix: the mixed
    step and the activation token count are priced at the tokens the batch
    actually owns, while the published ctx_tokens column keeps the knob. At
    prefix 0 the schedule only changes in this regime (ctx > b * isl), where
    the old one was not physical."""
    steps: list[MixedStepInput] = []

    def _run_mixed(model_arg, database_arg, runtime_config_arg, step):
        steps.append(step)
        return StepEstimate(latency_ms=10.0, energy_wms=1.0)

    memory_calls: list[dict] = []

    def _memory(model_arg, database_arg, batch_size, beam_width, isl, osl, **kwargs):
        memory_calls.append(kwargs)
        return {"total": 1.0}

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {"decode": 1.0}, {"decode": "silicon"}, ()),
    )
    monkeypatch.setattr(backend, "_get_memory_usage", _memory)

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=b, beam_width=1, isl=_ISL, osl=osl, prefix=prefix, engine_step_backend="rust"),
        ctx_tokens=ctx_tokens,
    )

    result = summary.get_result_dict()
    assert steps[0].context_tokens == capped
    assert steps[0].num_decode_requests == 0
    if b > 1 and osl > 2:
        # The prefill-only step belongs to TTFT; TPOT is the decode step (1 ms).
        assert result["tpot"] == pytest.approx(1.0)
    assert result["ctx_tokens"] == ctx_tokens
    assert result["num_tokens"] == capped
    assert memory_calls[0]["num_tokens"] == capped
    assert result["num_gen_reqs"] >= 0
    assert result["num_ctx_reqs"] == (1 if b == 1 else b)
    if osl == 1:
        return
    # One chunk per request: the whole uncached prefill fits the budget.
    assert result["ttft"] / backend._ttft_queuing_factor(
        b, summary.get_step_estimates()["scheduling"]["num_mix_steps"]
    ) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("ctx_tokens", "prefilling", "decoding", "expected_tpot"),
    [
        # One complete request (1920 new tokens) plus a 128-token partial one:
        # both requests are prefilling in the first of ceil(3840/2048) = 2
        # mixed steps, which leaves TPOT; the second one (the partial request
        # finishing while the other decodes) stays: (10 + 62 * 1) / 63.
        (2048, 2, 0, (10.0 + 62 * 1.0) / 63),
        # Exactly one request: the other one decodes alongside it, and both of
        # the ceil(3840/1920) = 2 mixed steps count: (2 * 10 + 62 * 1) / 64.
        (1920, 1, 1, (2 * 10.0 + 62 * 1.0) / 64),
    ],
)
def test_run_agg_partial_last_request_prefills_the_whole_batch(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    ctx_tokens: int,
    prefilling: int,
    decoding: int,
    expected_tpot: float,
) -> None:
    """bs 2, isl 2048, prefix 128: once the budget reaches into the last
    request, every request of the batch is prefilling, so the mixed step
    prices no decode request (never b + 1 requests), and only the decode-free
    first mixed step stays out of the TPOT average (mixed step 10 ms, decode
    step 1 ms, 64 decode iterations)."""
    steps: list[MixedStepInput] = []

    def _run_mixed(model_arg, database_arg, runtime_config_arg, step):
        steps.append(step)
        return StepEstimate(latency_ms=10.0, energy_wms=1.0)

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {"decode": 1.0}, {"decode": "silicon"}, ()),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=2048, osl=64, prefix=128, engine_step_backend="rust"),
        ctx_tokens=ctx_tokens,
    )

    result = summary.get_result_dict()
    assert (result["num_ctx_reqs"], result["num_gen_reqs"]) == (prefilling, decoding)
    assert steps[0].num_decode_requests == decoding
    assert steps[0].context_tokens == ctx_tokens
    assert result["tpot"] == pytest.approx(expected_tpot)


def test_run_agg_budget_cap_is_inert_when_the_batch_owns_more_tokens(
    monkeypatch, backend: BaseBackend, model, database
) -> None:
    """bs 16 with 3278 uncached tokens each owns 52448 tokens, more than the
    16384 budget: nothing is capped and the schedule is the plain isl_new
    packing (5 prefilling + 11 decoding requests, 4 mixed steps)."""
    seen = _fixed_step_costs(monkeypatch, backend)
    summary = backend.run_agg(model, database, _prefix_sweep_config(29491), ctx_tokens=_CTX)
    result = summary.get_result_dict()
    assert result["num_ctx_reqs"] == 5
    assert result["num_gen_reqs"] == 11
    assert result["num_tokens"] == 11 + _CTX
    assert summary.get_step_estimates()["scheduling"]["num_mix_steps"] == 4
    assert seen[0].prefix == 29491


@pytest.mark.parametrize(
    ("ctx_tokens", "expected_batch", "expected_scale"),
    [
        # 60 uncached tokens hold ceil(60 / 20) = 3 requests in one chunk.
        (60, 3, 1.0),
        # 10 uncached tokens hold one request over ceil(20 / 10) = 2 chunks.
        (10, 1, 2.0),
    ],
)
def test_run_mixed_visual_context_batch_follows_uncached_isl(
    monkeypatch, backend: BaseBackend, model, database, ctx_tokens: int, expected_batch: int, expected_scale: float
) -> None:
    """Text isl 8 + 16 visual tokens = 24 effective tokens, 4 of them cached:
    the visual context phase runs for ceil(ctx_tokens / 20) requests and its
    cost is amortized over ceil(20 / ctx_tokens) chunks, like the engine's
    context-attention pass."""
    from aisimulate.sdk.backends import base_backend as base_backend_module

    model.encoder_config = _vision_encoder_config()
    model.visual_context_ops = [_StaticOp("visual_attention", latency_ms=1.0, energy_wms=1.0)]

    monkeypatch.setattr(
        base_backend_module,
        "estimate_mixed_step_breakdown_with_rust",
        lambda *args, **kwargs: {
            "latency_ms": 1.0,
            "energy_wms": 1.0,
            "component_latency_ms": {"shared_non_attention": 1.0},
            "component_energy_wms": {"shared_non_attention": 1.0},
            "per_op_latency_ms": {},
            "per_op_source": {},
            "moe_comm_fallbacks": (),
        },
    )
    batches: list[int] = []

    def _visual(model_arg, database_arg, runtime_config_arg, batch_size, *, include_energy=True):
        batches.append(batch_size)
        return {"visual_attention": 8.0}, {"visual_attention": 4.0}, {"visual_attention": "silicon"}

    monkeypatch.setattr(backend, "_run_visual_context_phase", _visual)

    step = backend.run_mixed(
        model,
        database,
        RuntimeConfig(
            isl=8, osl=6, prefix=4, num_images_per_request=1, num_image_tokens=16, engine_step_backend="rust"
        ),
        MixedStepInput(context_tokens=ctx_tokens, num_decode_requests=1),
    )

    assert batches == [expected_batch]
    assert step.per_op_latency_ms["visual_attention"] == pytest.approx(8.0 / expected_scale)
    assert step.component_latency_ms["context_attention"] == pytest.approx(8.0 / expected_scale)
    assert step.latency_ms == pytest.approx(1.0 + 8.0 / expected_scale)
    assert step.energy_wms == pytest.approx(1.0 + 4.0 / expected_scale)
