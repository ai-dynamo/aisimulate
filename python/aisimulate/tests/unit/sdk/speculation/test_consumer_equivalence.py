# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_consumer_equivalence.py

"""Consumer-contract tests for the compile-time materialization design.

The compiled engine is the only step executor, so a scheme reaches the
engine exclusively through what ``materialize_spec_scheme`` folds into the
model at ``get_model`` time. Three guarantees are pinned here:

1. Golden equivalence — legacy ``nextn`` construction and an explicit mtp
   SpeculationConfig produce models with identical engine-relevant state
   (same ``_nextn``, same engine identity), so MTP predictions stay
   bit-identical.
2. Materialization — a scheme with its own verify width / draft ops / bytes
   lands in the op lists (``draft_`` prefix, ``scale_num_tokens`` width
   folding, ``_nextn`` width channel) without mutating scheme-owned ops,
   and idempotently.
3. Memory accounting + engine routing — draft weights/KV enter
   ``_get_memory_usage`` through the scheme hooks, and the engine identity
   is keyed by the speculation config so same-width schemes never share a
   cached engine handle.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend
from aiconfigurator_core.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator_core.sdk.rust_engine_step import _engine_config_json, should_use_rust_engine_step
from aiconfigurator_core.sdk.speculation import DraftOpSpec, NullScheme, SpeculationConfig
from aiconfigurator_core.sdk.speculation.materialize import materialize_spec_scheme
from aiconfigurator_core.sdk.speculation.mtp import MTPScheme

pytestmark = pytest.mark.unit


class _RecordingOp:
    """Fake op carrying the attributes materialization touches."""

    def __init__(self, name: str, latency_ms: float = 1.0) -> None:
        self._name = name
        self._latency_ms = latency_ms
        self._scale_num_tokens = 1
        self._scale_factor = 1.0

    def query(self, *args, **kwargs) -> float:
        return self._latency_ms

    def get_weights(self) -> float:
        return 0.0


class _MemoryBackend(BaseBackend):
    """Memory-path backend: uses the REAL BaseBackend._get_memory_usage."""

    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError


def _model_config(speculation=None) -> ModelConfig:
    return ModelConfig(
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
        speculation=speculation,
    )


def _fake_model(nextn: int = 0, spec_scheme=None, speculation=None):
    model = SimpleNamespace()
    model.model_path = "test-model"
    model._nextn = nextn
    model.encoder_ops = []
    model.context_ops = [_RecordingOp("context_attention", 11.0)]
    model.generation_ops = [_RecordingOp("generation_attention", 2.0)]
    model.config = _model_config(speculation=speculation)
    model.config.nextn = nextn
    if spec_scheme is None:
        spec_scheme = NullScheme() if nextn == 0 else MTPScheme(depth=nextn)
    model.spec_scheme = spec_scheme
    return model


def _database():
    return SimpleNamespace(
        backend="test-backend",
        version="test-version",
        system="test-system",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
    )


class _FakeDraftScheme(NullScheme):
    """Verify width 6; one gen draft op at 5 tokens/request (non-divisible —
    mapped by 5/6) and one at 3 (divisible — maps by 1/2); one context
    draft op; nonzero weight/KV bytes."""

    kind = "fake_draft"

    def __init__(self) -> None:
        self.gen_op = _RecordingOp("dspark_backbone", 3.0)
        self.gen_op_divisible = _RecordingOp("dspark_head", 1.0)
        self.ctx_op = _RecordingOp("dspark_precompute", 5.0)

    def verify_width(self) -> int:
        return 6

    def build_draft_generation_ops(self, model):
        return [
            DraftOpSpec(op=self.gen_op, tokens_per_request=5, query_overrides={"s": 133}),
            DraftOpSpec(op=self.gen_op_divisible, tokens_per_request=3),
        ]

    def build_draft_context_ops(self, model):
        return [DraftOpSpec(op=self.ctx_op, tokens_per_request=1)]

    def draft_weights_bytes(self, model) -> float:
        return 4.0 * (1 << 30)

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        return 1024.0

    def verify_attention_sequence_basis(self) -> bool:
        return True  # block verify shares one KV pass (all real draft schemes)


class TestGoldenEquivalence:
    def test_legacy_and_explicit_mtp_share_engine_state(self):
        legacy = _fake_model(nextn=2)  # spec_scheme derived: MTPScheme(2)
        explicit = _fake_model(
            nextn=2,
            spec_scheme=MTPScheme(depth=2),
            speculation=SpeculationConfig(kind="mtp", params={"depth": 2}),
        )

        # Materialization must be a no-op for MTP: the legacy nextn contract
        # (model families + engine nextn scalar) stays authoritative.
        for model in (legacy, explicit):
            materialize_spec_scheme(model)
            assert model._nextn == 2
            assert [op._name for op in model.generation_ops] == ["generation_attention"]
            assert not getattr(model, "_spec_scheme_materialized", False)

        # Same engine identity: an explicit mtp SpeculationConfig rides the
        # nextn key, never the speculation content key.
        db = _database()
        assert _engine_config_json(legacy, db) == _engine_config_json(explicit, db)


class TestMaterialization:
    def test_width_channel_and_draft_ops_land_in_op_lists(self):
        scheme = _FakeDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        materialize_spec_scheme(model)

        # verify width 6 -> engine decode-batch multiplier (_nextn + 1) = 6.
        assert model._nextn == 5

        gen_names = [op._name for op in model.generation_ops]
        assert gen_names == ["generation_attention", "draft_dspark_backbone", "draft_dspark_head"]
        ctx_names = [op._name for op in model.context_ops]
        assert ctx_names == ["context_attention", "draft_dspark_precompute"]

        by_name = {op._name: op for op in model.generation_ops}
        # Both ratios remap query dimensions in Rust, preserving nonlinear costs.
        assert by_name["draft_dspark_backbone"]._draft_token_width == (5, 6)
        assert by_name["draft_dspark_backbone"]._scale_factor == 1.0
        assert by_name["draft_dspark_head"]._draft_token_width == (3, 6)

    def test_scheme_owned_ops_are_not_mutated(self):
        scheme = _FakeDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        materialize_spec_scheme(model)

        # The scheme's cached ops keep their identity: schemes that cache a
        # built draft model stay re-inspectable after materialization.
        assert scheme.gen_op._name == "dspark_backbone"
        assert scheme.gen_op_divisible._scale_num_tokens == 1
        assert scheme.ctx_op._name == "dspark_precompute"
        assert model.generation_ops[1] is not scheme.gen_op

    def test_materialization_is_idempotent(self):
        model = _fake_model(nextn=0, spec_scheme=_FakeDraftScheme())
        materialize_spec_scheme(model)
        once = [op._name for op in model.generation_ops]
        materialize_spec_scheme(model)
        assert [op._name for op in model.generation_ops] == once


class TestAttentionWidthChannel:
    """Sequence-basis fold + roofline guard on dense decode attention."""

    @staticmethod
    def _gen_attention(name="generation_attention"):
        from aiconfigurator_core.sdk.operations.attention import GenerationAttention

        return GenerationAttention(name, 1.0, n=32, n_kv=8, kv_cache_dtype=common.KVCacheQuantMode.bfloat16)

    def test_target_verify_attention_folds_to_sequence_basis(self):
        scheme = _FakeDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        target_attn = self._gen_attention()
        model.generation_ops = [target_attn]
        materialize_spec_scheme(model)

        # verify width 6: batch divisor 6, real query width 6 for the guard.
        assert target_attn._scale_num_tokens == 6
        assert target_attn._verify_query_tokens == 6

    def test_draft_block_attention_folds_full_width(self):
        class _BlockDraftScheme(_FakeDraftScheme):
            def __init__(self) -> None:
                super().__init__()
                self.attn_op = TestAttentionWidthChannel._gen_attention("dspark_attention")

            def build_draft_generation_ops(self, model):
                # 5 drafted tokens per request inside a width-6 phase: gemms
                # use the rational query fold, but attention folds by
                # the FULL width (one KV read per request) and carries the
                # real query width for the guard.
                return [DraftOpSpec(op=self.attn_op, tokens_per_request=5)]

        scheme = _BlockDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        materialize_spec_scheme(model)

        draft_attn = next(op for op in model.generation_ops if op._name == "draft_dspark_attention")
        assert draft_attn._scale_num_tokens == 6
        assert draft_attn._verify_query_tokens == 5
        # scheme-owned op untouched (copy semantics)
        assert scheme.attn_op._scale_num_tokens == 1

    # Query-time behavior (sequence-basis batch fold, roofline guard) is
    # priced only by the Rust oracle — anchored by the
    # `generation_attention_*` tests in `operators/attention.rs` (single-oracle
    # rule: Python carries the fields onto the wire; it never prices them).

    def test_engine_spec_carries_the_width_fields(self):
        import json

        op = self._gen_attention()
        wire = json.loads(op._spec_json())["GenerationAttention"]
        assert wire["scale_num_tokens"] == 1
        assert wire["verify_query_tokens"] == 0
        op._scale_num_tokens = 6
        op._verify_query_tokens = 6
        wire = json.loads(op._spec_json())["GenerationAttention"]
        assert wire["scale_num_tokens"] == 6
        assert wire["verify_query_tokens"] == 6

    def test_copy_preserves_the_width_fields(self):
        # materialize copy.copy()s scheme-owned ops before folding; the
        # pyo3 pickle protocol must round-trip the width channel.
        import copy

        op = self._gen_attention()
        op._scale_num_tokens = 6
        op._verify_query_tokens = 5
        dup = copy.copy(op)
        assert dup._scale_num_tokens == 6
        assert dup._verify_query_tokens == 5


class TestMemoryAndRouting:
    def test_memory_accounting_includes_draft_bytes(self):
        base = _fake_model(nextn=0)
        drafted = _fake_model(nextn=0, spec_scheme=_FakeDraftScheme())
        materialize_spec_scheme(drafted)
        database = _database()
        database.system_spec["misc"] = {"nccl_mem": {1: 0.0}, "other_mem": 0.0}

        backend = _MemoryBackend()
        kwargs = dict(batch_size=4, beam_width=1, isl=512, osl=64)
        for model in (base, drafted):
            model._num_heads = 8
            model._head_size = 128
            model._num_experts = 0
            model.model_family = "GPT"
            model.get_kvcache_bytes_per_sequence = lambda seq_len: 2048.0
            model._cp_kv_memory_divisor = lambda: 1

        m_base = backend._get_memory_usage(base, database, **kwargs)
        m_draft = backend._get_memory_usage(drafted, database, **kwargs)

        one_gib = 1 << 30
        # weights: +4 GiB from the scheme (materialized draft_ ops excluded
        # from the op-list sum; the scheme hook is the single source of truth)
        assert m_draft["weights"] - m_base["weights"] == pytest.approx(4.0, rel=1e-6)
        # kv: +batch * 1024 bytes
        assert (m_draft["kvcache"] - m_base["kvcache"]) * one_gib == pytest.approx(4 * 1024.0, rel=1e-6)
        # activations: the materialized _nextn drives the (nextn+1) width factor
        assert m_draft["activations"] == pytest.approx(m_base["activations"] * 6, rel=1e-6)

    def test_engine_identity_keyed_by_speculation_config(self):
        db = _database()
        plain = _fake_model(nextn=0)
        eagle = _fake_model(
            nextn=0,
            spec_scheme=NullScheme(),  # identity comes from the config, not the scheme object
            speculation=SpeculationConfig(kind="eagle3", params={"num_speculative_tokens": 5}),
        )
        # Same widths, different speculation content -> different engines.
        assert _engine_config_json(plain, db) != _engine_config_json(eagle, db)

        # The key is the config's content hash, so equal configs share.
        eagle2 = _fake_model(
            nextn=0,
            spec_scheme=NullScheme(),
            speculation=SpeculationConfig(kind="eagle3", params={"num_speculative_tokens": 5}),
        )
        assert _engine_config_json(eagle, db) == _engine_config_json(eagle2, db)
        payload = json.loads(_engine_config_json(eagle, db))
        assert payload["speculation"] is not None
        assert json.loads(_engine_config_json(plain, db))["speculation"] is None

    def test_rust_engine_routing_has_no_scheme_bypass(self):
        # Explicit "rust" routes to the compiled engine regardless of scheme;
        # the one remaining delegation is the synthetic-database default.
        explicit = RuntimeConfig(engine_step_backend="rust")
        default = RuntimeConfig()
        synthetic_db = _database()
        assert should_use_rust_engine_step(explicit, synthetic_db) is True
        assert should_use_rust_engine_step(default, synthetic_db) is False


def test_materialize_draft_width_larger_than_verify_budget():
    class WideDraftScheme(_FakeDraftScheme):
        def build_draft_generation_ops(self, model):
            return [DraftOpSpec(op=self.gen_op, tokens_per_request=8)]

    model = _fake_model(spec_scheme=WideDraftScheme())
    materialize_spec_scheme(model)
    assert model.generation_ops[-1]._draft_token_width == (8, 6)


@pytest.mark.parametrize("round_trip", ["copy", "deepcopy", "pickle"])
def test_native_attention_preserves_positional_options_and_keyword_widths(round_trip):
    import copy
    import pickle

    from aiconfigurator_core.sdk.operations.attention import GenerationAttention

    op = GenerationAttention(
        "draft_attention",
        2.0,
        16,
        8,
        common.KVCacheQuantMode.bfloat16,
        256,
        64,
        True,
        ["fa3", "default"],
        scale_num_tokens=6,
        verify_query_tokens=5,
    )
    if round_trip == "pickle":
        duplicate = pickle.loads(pickle.dumps(op))
    else:
        duplicate = getattr(copy, round_trip)(op)
    wire = json.loads(duplicate._spec_json())["GenerationAttention"]
    assert wire["window_size"] == 256
    assert wire["head_size"] == 64
    assert wire["use_qk_norm"] is True
    assert wire["lane_order"] == ["fa3", "default"]
    assert wire["scale_num_tokens"] == 6
    assert wire["verify_query_tokens"] == 5
    assert duplicate is not op


@pytest.fixture(scope="module")
def real_database():
    from aiconfigurator_core.sdk.perf_database import get_database_view

    return get_database_view("h100_sxm", "vllm", "0.24.0")


@pytest.mark.parametrize("draft_count", [1, 3, 7])
@pytest.mark.parametrize("batch_size", [1, 512])
def test_every_standalone_draft_op_matches_independent_decode(real_database, draft_count, batch_size):
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle

    cfg = _model_config()
    cfg.tp_size = 2  # Includes communication ops with nonlinear size costs.
    independent = get_model("Qwen/Qwen3-0.6B", cfg, "vllm")
    cfg = _model_config(
        SpeculationConfig(
            kind="draft_model", params={"num_speculative_tokens": draft_count}, draft_model_path="Qwen/Qwen3-0.6B"
        )
    )
    cfg.tp_size = 2
    target = get_model("Qwen/Qwen3-8B", cfg, "vllm")

    def breakdown(model):
        _, rows = _cached_engine_handle(model, real_database).run_static_per_op(
            batch_size=batch_size, isl=64, osl=2, mode="static_gen", stride=1
        )
        return {row[0]: row[1] for row in rows}

    expected, actual = breakdown(independent), breakdown(target)
    draft_names = {name.removeprefix("draft_") for name in actual if name.startswith("draft_")}
    assert draft_names == expected.keys()
    for name, latency in expected.items():
        assert actual[f"draft_{name}"] == pytest.approx(draft_count * latency, rel=1e-10), name


@pytest.mark.parametrize("tree_shape", [[1, 1], [2, 3], [4, 8]])
@pytest.mark.parametrize("batch_size", [1, 512])
def test_every_tree_draft_op_queries_its_own_width(real_database, tree_shape, batch_size):
    import copy

    from aiconfigurator_core.sdk.engine import _evaluate_single_op
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.operations.attention import GenerationAttention
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle

    from .test_dense_draft_schemes import EAGLE3_CONFIG

    cfg = _model_config(SpeculationConfig(kind="eagle3", params={"tree_shape": tree_shape}, draft_config=EAGLE3_CONFIG))
    cfg.tp_size = 2
    model = get_model("Qwen/Qwen3-8B", cfg, "vllm")
    specs = model.spec_scheme.build_draft_generation_ops(model)
    indices = [i for i, op in enumerate(model.generation_ops) if op._name.startswith("draft_")]
    handle = _cached_engine_handle(model, real_database)
    assert len(indices) == len(specs)
    for spec, index in zip(specs, indices, strict=True):
        # Repeated tree-level names are aggregated by the breakdown API;
        # evaluate each compiled index to check every lookup independently.
        [actual] = handle.evaluate_generation_ops(
            [index], batch_size=batch_size * model.verify_width, s=65, x=batch_size * model.verify_width
        )
        op = copy.copy(spec.op)
        if isinstance(op, GenerationAttention):
            # Physical block attention: one KV request per batch member,
            # with the block's actual query width in the roofline guard.
            expected_batch = batch_size
            op._verify_query_tokens = spec.tokens_per_request
        else:
            expected_batch = batch_size * spec.tokens_per_request
        expected = _evaluate_single_op(
            real_database,
            op,
            is_context=False,
            batch_size=expected_batch,
            s=65,
            x=batch_size * spec.tokens_per_request,
        )
        assert actual[0] == f"draft_{spec.op._name}"
        assert actual[1] == pytest.approx(float(expected), rel=1e-10), actual[0]
        assert actual[2] == pytest.approx(expected.energy, rel=1e-10), actual[0]


@pytest.mark.parametrize("mutation", ["params", "draft_config"])
def test_materialized_config_snapshot_keeps_cache_and_graph_consistent(real_database, mutation):
    import copy
    import dataclasses

    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle, _engine_handle_cache_clear

    from .test_dense_draft_schemes import EAGLE3_CONFIG

    params = {"tree_shape": [1, 1]}
    draft_config = copy.deepcopy(EAGLE3_CONFIG)
    cfg = _model_config(SpeculationConfig(kind="eagle3", params=params, draft_config=draft_config))
    first = get_model("Qwen/Qwen3-8B", cfg, "vllm")
    identity = _engine_config_json(first, real_database)
    graph = [op._spec_json() for op in first.generation_ops]
    if mutation == "params":
        params["tree_shape"][:] = [2]
    else:
        draft_config["num_hidden_layers"] = 2
    second = get_model("Qwen/Qwen3-8B", dataclasses.replace(cfg), "vllm")
    assert _engine_config_json(first, real_database) == identity
    assert [op._spec_json() for op in first.generation_ops] == graph
    assert _engine_config_json(second, real_database) != identity

    def latency(model):
        _, rows = _cached_engine_handle(model, real_database).run_static_per_op(
            batch_size=8, isl=64, osl=32, mode="static_gen", stride=1
        )
        return sum(row[1] for row in rows)

    _engine_handle_cache_clear()
    try:
        original = latency(first)
        warm = latency(second)
        assert _cached_engine_handle(first, real_database) is not _cached_engine_handle(second, real_database)
        assert latency(first) == original
        _engine_handle_cache_clear()
        assert latency(second) == warm
        assert latency(first) == original
        assert original != warm  # The changed input actually changes a native prediction.
    finally:
        _engine_handle_cache_clear()


@pytest.mark.parametrize("round_trip", ["copy", "deepcopy", "pickle"])
def test_draft_query_width_survives_copy_and_standalone_consumer(real_database, round_trip):
    import copy
    import pickle

    from aiconfigurator_core.sdk.engine import build_ops_json
    from aiconfigurator_core.sdk.operations.gemm import GEMM
    from aiconfigurator_core.sdk.speculation.materialize import _fold_width

    baseline = GEMM("draft_gemm", 1.0, 1024, 1024, common.GEMMQuantMode.bfloat16)
    folded = copy.copy(baseline)
    _fold_width(folded, 5, 6)
    duplicate = pickle.loads(pickle.dumps(folded)) if round_trip == "pickle" else getattr(copy, round_trip)(folded)
    assert duplicate is not folded
    assert json.loads(build_ops_json([duplicate]))[0]["TokenScale"]["numerator"] == 5
    assert duplicate._name == folded._name
    assert duplicate.get_weights() == baseline.get_weights()
    assert float(duplicate._engine_query(real_database, x=6)) == float(baseline._engine_query(real_database, x=5))


@pytest.mark.parametrize("tokens,width", [(0, 6), (5, 0), (-1, 6), (5, 1.5), (True, 6)])
def test_materialize_rejects_invalid_draft_width(tokens, width):
    from aiconfigurator_core.sdk.speculation.materialize import _fold_width

    with pytest.raises(ValueError, match="positive integer"):
        _fold_width(_RecordingOp("draft"), tokens, width)


def test_draft_query_wrapper_does_not_hide_retired_native_ops():
    from aiconfigurator_core.sdk.engine import OpConversionError, build_ops_json
    from aiconfigurator_core.sdk.operations.moe import MoEDispatch
    from aiconfigurator_core.sdk.speculation.materialize import _fold_width

    op = MoEDispatch("draft_retired", 1.0, 7168, 8, 256, 1, 16, 1, False, backend="sglang", moe_backend="deepep_moe")
    _fold_width(op, 1, 4)
    with pytest.raises(OpConversionError, match="no native variant"):
        build_ops_json([op])


def test_draft_width_does_not_silently_admit_unsupported_python_graphs():
    from aiconfigurator_core.sdk.engine import OpConversionError, build_ops_json
    from aiconfigurator_core.sdk.speculation.materialize import _fold_width

    op = _RecordingOp("draft_unsupported")
    _fold_width(op, 1, 4)
    with pytest.raises(OpConversionError, match="no OpSpec conversion"):
        build_ops_json([op])


@pytest.mark.parametrize("draft_path", ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3.5-35B-A3B"])
@pytest.mark.parametrize("draft_count", [1, 3])
def test_moe_draft_native_forward_repetition_preserves_context_and_metadata(real_database, draft_path, draft_count):
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.operations.overlap import OverlapOp
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle

    independent = get_model(draft_path, _model_config(), "vllm")
    target = get_model(
        "Qwen/Qwen3.5-397B-A17B",
        _model_config(
            SpeculationConfig(
                kind="draft_model",
                params={"num_speculative_tokens": draft_count},
                draft_model_path=draft_path,
            )
        ),
        "vllm",
    )
    draft = target.spec_scheme._draft_model
    assert (draft.config.moe_tp_size, draft.config.moe_ep_size) == (1, 1)
    if "3.5" in draft_path:
        assert sum(isinstance(op, OverlapOp) for op in draft.generation_ops) == 2
    # The cached draft remains an independent checkpoint, including context
    # work and unique weights, even after materialization repeats its graph.
    for phase in ("context_ops", "generation_ops"):
        assert [op._spec_json() for op in getattr(draft, phase)] == [
            op._spec_json() for op in getattr(independent, phase)
        ]
    assert target.spec_scheme.draft_weights_bytes(target) == sum(op.get_weights() for op in independent.generation_ops)

    def run(model):
        return _cached_engine_handle(model, real_database)._run_static_per_op_with_metadata(
            batch_size=8, isl=64, osl=2, mode="static", stride=1
        )

    expected, actual = run(independent), run(target)
    for expected_rows, actual_rows, repetitions in zip(expected, actual, (1, draft_count), strict=True):
        draft_rows = {row[0].removeprefix("draft_"): row for row in actual_rows if row[0].startswith("draft_")}
        assert draft_rows.keys() == {row[0] for row in expected_rows}
        for name, latency, energy, source, metadata in expected_rows:
            row = draft_rows[name]
            assert row[1] == pytest.approx(repetitions * latency, rel=1e-10), name
            assert row[2] == pytest.approx(repetitions * energy, rel=1e-10), name
            assert row[3:] == (source, metadata), name


@pytest.mark.parametrize("ctx_tokens,gen_requests,prefix", [(0, 7, 0), (128, 0, 64), (128, 7, 64), (8000, 7, 64)])
def test_mixed_draft_native_phases_match_independent_queries(real_database, ctx_tokens, gen_requests, prefix):
    import math

    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle

    from .test_dense_draft_schemes import EAGLE3_CONFIG

    model = get_model(
        "Qwen/Qwen3-8B",
        _model_config(SpeculationConfig(kind="eagle3", params={"tree_shape": [1, 1, 1]}, draft_config=EAGLE3_CONFIG)),
        "vllm",
    )
    handle = _cached_engine_handle(model, real_database)
    shared, context, generation = handle.mixed_step_breakdown_per_op(ctx_tokens, gen_requests, 4000, 64, prefix)
    assert not any(row[0].startswith("draft_") for row in shared)
    ctx_indices = [i for i, op in enumerate(model.context_ops) if op._name.startswith("draft_")]
    gen_indices = [i for i, op in enumerate(model.generation_ops) if op._name.startswith("draft_")]
    expected_context = (
        handle.evaluate_context_ops(
            ctx_indices, batch_size=math.ceil(ctx_tokens / 4000), s=4000 - prefix, prefix=prefix
        )
        if ctx_tokens
        else []
    )
    expected_generation = (
        handle.evaluate_generation_ops(gen_indices, batch_size=gen_requests * 4, s=4033) if gen_requests else []
    )
    for actual, expected, divisor in (
        (context, expected_context, math.ceil(4000 / ctx_tokens) if ctx_tokens else 1),
        (generation, expected_generation, 1),
    ):
        drafts = {r[0]: r for r in actual if r[0].startswith("draft_")}
        assert drafts.keys() == {r[0] for r in expected}
        for name, latency, energy, source in expected:
            assert drafts[name][1] == pytest.approx(latency / divisor, rel=1e-10)
            assert drafts[name][2] == pytest.approx(energy / divisor, rel=1e-10)
            assert drafts[name][3] == source
    assert sum(r[1] for group in (shared, context, generation) for r in group) == pytest.approx(
        handle.mixed_step_latency(ctx_tokens, gen_requests, 4000, 64, prefix), rel=1e-10
    )


def _assert_public_mixed_rows(estimate, native):
    """Compare the public report with executed native rows, without repricing."""
    rows = [row for group in native for row in group]
    names = {row[0] for row in rows} | {"context_attention", "generation_attention"}
    public_names = {"context_attention (scaled)" if name == "context_attention" else name for name in names}
    assert estimate.per_op_latency_ms.keys() == public_names
    assert estimate.per_op_source.keys() == public_names
    for name in names:
        matches = [row for row in rows if row[0] == name]
        public_name = "context_attention (scaled)" if name == "context_attention" else name
        sources = {row[3] for row in matches} or {"silicon"}
        assert estimate.per_op_latency_ms[public_name] == pytest.approx(sum(row[1] for row in matches)), name
        assert estimate.per_op_source[public_name] == (sources.pop() if len(sources) == 1 else "mixed"), name
    for component, group in zip(("shared_non_attention", "context_attention", "decode_attention"), native, strict=True):
        assert estimate.component_latency_ms[component] == pytest.approx(sum(row[1] for row in group))
        assert estimate.component_energy_wms[component] == pytest.approx(sum(row[2] for row in group))
    assert estimate.latency_ms == pytest.approx(sum(row[1] for row in rows))
    assert estimate.energy_wms == pytest.approx(sum(row[2] for row in rows))
    assert sum(estimate.per_op_latency_ms.values()) == pytest.approx(estimate.latency_ms)


@pytest.mark.parametrize("gen_requests", [0, 7])
@pytest.mark.parametrize("database_mode", ["SILICON", "SOL"])
@pytest.mark.parametrize("draft", ["eagle3", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3.5-35B-A3B"])
def test_public_mixed_draft_names_values_and_sources_match_native(draft, database_mode, gen_requests):
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator_core.sdk.backends.factory import get_backend
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.perf_database import get_database_view
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle
    from aiconfigurator_core.sdk.step_estimate import MixedStepInput

    from .test_dense_draft_schemes import EAGLE3_CONFIG

    speculation = (
        SpeculationConfig(kind="eagle3", params={"tree_shape": [1, 1, 1]}, draft_config=EAGLE3_CONFIG)
        if draft == "eagle3"
        else SpeculationConfig(kind="draft_model", params={"num_speculative_tokens": 3}, draft_model_path=draft)
    )
    target = "Qwen/Qwen3.5-35B-A3B" if draft == "Qwen/Qwen3-0.6B" else "Qwen/Qwen3-8B"
    model = get_model(target, _model_config(speculation), "vllm")
    database = get_database_view("h100_sxm", "vllm", "0.24.0", database_mode=database_mode)
    native = _cached_engine_handle(model, database)._mixed_step_breakdown_per_op_with_metadata(
        128, gen_requests, 4000, 64, 64
    )
    estimate = InferenceSession(model, database, get_backend("vllm")).run_mixed(
        RuntimeConfig(isl=4000, osl=64, prefix=64), MixedStepInput(128, gen_requests)
    )
    assert any(row[0].startswith("draft_") for row in native[1])
    assert any(row[0].startswith("draft_") for row in native[2]) == bool(gen_requests)
    if draft == "eagle3" and gen_requests:
        # EAGLE's feature projection executes in both phases under one name.
        duplicates = {row[0] for row in native[1]} & {row[0] for row in native[2]}
        assert any(name.startswith("draft_") and "fc" in name for name in duplicates)
    _assert_public_mixed_rows(estimate, native)


@pytest.mark.parametrize("gen_requests", [0, 7])
@pytest.mark.parametrize("depth", [0, 2])
def test_public_mixed_ar_and_mtp_keep_legacy_names_and_defaults(real_database, gen_requests, depth):
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator_core.sdk.backends.factory import get_backend
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle
    from aiconfigurator_core.sdk.step_estimate import MixedStepInput

    config = _model_config()
    config.nextn = depth
    model = get_model("Qwen/Qwen3-8B", config, "vllm")
    native = _cached_engine_handle(model, real_database)._mixed_step_breakdown_per_op_with_metadata(
        128, gen_requests, 4000, 64, 0
    )
    estimate = InferenceSession(model, real_database, get_backend("vllm")).run_mixed(
        RuntimeConfig(isl=4000, osl=64), MixedStepInput(128, gen_requests)
    )
    _assert_public_mixed_rows(estimate, native)
    assert estimate.num_decode_query_tokens == gen_requests * (depth + 1)
    if not gen_requests:
        assert estimate.per_op_latency_ms["generation_attention"] == 0.0
        assert estimate.per_op_source["generation_attention"] == "silicon"


def test_public_mixed_duplicate_draft_names_merge_sources_and_metadata(real_database, monkeypatch):
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator_core.sdk import rust_engine_step
    from aiconfigurator_core.sdk.backends.factory import get_backend
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.performance_result import MoECommFallback
    from aiconfigurator_core.sdk.step_estimate import MixedStepInput

    context_fallback = ("context", "deepep_ht", 32, 8, 8, 1)
    generation_fallback = ("generation", "deepep_ll", 32, 8, 4, 1)
    # Synthetic phase rows isolate an exact source collision and nonzero
    # energy; real native executions are checked separately above.
    native = (
        [("context_mlp", 5.0, 50.0, "sol")],
        [
            ("context_attention", 2.0, 20.0, "sol"),
            ("draft_eagle_fc", 3.0, 30.0, "empirical", (context_fallback, [])),
        ],
        [
            ("generation_attention", 7.0, 70.0, "silicon"),
            ("draft_eagle_fc", 11.0, 110.0, "sol", (generation_fallback, [])),
            ("draft_eagle_fc", 13.0, 130.0, "silicon", (generation_fallback, [])),
        ],
    )
    handle = SimpleNamespace(
        _mixed_step_breakdown_per_op_with_metadata=lambda *args, **kwargs: native, last_provenance=lambda: None
    )
    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda *args: handle)
    model = get_model("Qwen/Qwen3-8B", _model_config(), "vllm")
    estimate = InferenceSession(model, real_database, get_backend("vllm")).run_mixed(
        RuntimeConfig(isl=4000, osl=64), MixedStepInput(128, 7)
    )
    _assert_public_mixed_rows(estimate, native)
    assert estimate.per_op_latency_ms["draft_eagle_fc"] == 27.0
    assert estimate.per_op_source["draft_eagle_fc"] == "mixed"
    assert estimate.moe_comm_fallbacks == (MoECommFallback(*context_fallback), MoECommFallback(*generation_fallback))


@pytest.mark.parametrize("gen_requests", [0, 1])
def test_public_mixed_draft_reports_only_executed_native_fallbacks(gen_requests):
    import copy

    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator_core.sdk.backends.factory import get_backend
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.operations import MoEAllToAll
    from aiconfigurator_core.sdk.perf_database import get_database_view
    from aiconfigurator_core.sdk.performance_result import MoECommFallback
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle, _engine_handle_cache_clear
    from aiconfigurator_core.sdk.step_estimate import MixedStepInput

    config = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=32,
        moe_tp_size=1,
        moe_ep_size=32,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8_block,
        moe_comm_backend={"context": "deepep_ht", "generation": "deepep_ll"},
        num_gpus_per_node=4,
    )
    model = get_model("deepseek-ai/DeepSeek-R1", config, "sglang")
    # Isolate both draft-phase fallback paths with a synthetic materialized
    # graph. Native communication queries use real checked-in GB200 donors.
    for phase in ("context_ops", "generation_ops"):
        ops = getattr(model, phase)
        for op in list(ops):
            if isinstance(op, MoEAllToAll):
                draft = copy.copy(op)
                draft._name = f"draft_{op._name}"
                ops.append(draft)
    database = get_database_view("gb200", "sglang", "0.5.14")
    _engine_handle_cache_clear()
    try:
        native = _cached_engine_handle(model, database)._mixed_step_breakdown_per_op_with_metadata(
            128, gen_requests, 1024, 32, 0
        )
        estimate = InferenceSession(model, database, get_backend("sglang")).run_mixed(
            RuntimeConfig(isl=1024, osl=32), MixedStepInput(128, gen_requests)
        )
        _assert_public_mixed_rows(estimate, native)
        context_fallback = ("context", "deepep_ht", 32, 8, 8, 1)
        generation_fallback = ("generation", "deepep_ll", 32, 8, 4, 1)
        for phase, payload in ((1, context_fallback), (2, generation_fallback)):
            drafts = [row for row in native[phase] if row[0].startswith("draft_")]
            assert len(drafts) == (2 if phase == 1 or gen_requests else 0)
            assert all(row[3:] == ("estimated", (payload, [])) for row in drafts)
        expected = (MoECommFallback(*context_fallback),)
        if gen_requests:
            expected += (MoECommFallback(*generation_fallback),)
        assert estimate.moe_comm_fallbacks == expected
    finally:
        _engine_handle_cache_clear()


@pytest.mark.parametrize("round_trip", ["copy", "deepcopy", "pickle"])
def test_standalone_repeated_nested_composites_keep_native_costs_and_weights(real_database, round_trip):
    import copy
    import pickle

    from aiconfigurator_core.sdk.engine import _evaluate_single_op, build_ops_json
    from aiconfigurator_core.sdk.errors import SolNotImplementedError
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.operations.gemm import GEMM
    from aiconfigurator_core.sdk.operations.overlap import FallbackOp, OverlapOp
    from aiconfigurator_core.sdk.rust_engine_step import _cached_engine_handle
    from aiconfigurator_core.sdk.speculation.draft_model import DraftModelScheme

    leaf = GEMM("gemm", 2.0, 1024, 1024, common.GEMMQuantMode.bfloat16)
    composite = FallbackOp(
        "nested",
        primary=OverlapOp("overlap", group_a=[leaf], group_b=[FallbackOp("child", primary=leaf, fallback=[leaf])]),
        fallback=[leaf],
    )
    draft = SimpleNamespace(generation_ops=[composite], context_ops=[composite])
    scheme = DraftModelScheme("Qwen/Qwen3-0.6B", 3)
    scheme._draft_model = draft
    target = get_model("Qwen/Qwen3-8B", _model_config(), "vllm")
    target.spec_scheme = scheme
    original_wire = composite._spec_json()
    original_weights = composite.get_weights()
    materialize_spec_scheme(target)
    folded = [op for op in target.generation_ops if op._name.startswith("draft_")]
    assert len(folded) == 3
    folded = [
        pickle.loads(pickle.dumps(op)) if round_trip == "pickle" else getattr(copy, round_trip)(op) for op in folded
    ]
    assert composite._spec_json() == original_wire
    assert composite.get_weights() == original_weights
    assert all(op.get_weights() == original_weights for op in folded)
    expected = _evaluate_single_op(real_database, composite, is_context=False, batch_size=512, s=65, x=512)
    handle = _cached_engine_handle(target, real_database)
    [actual] = handle.evaluate_ops_json(build_ops_json(folded), is_context=False, batch_size=2048, s=65, x=2048)
    assert actual[1] == pytest.approx(3 * float(expected), rel=1e-10)
    assert actual[2] == pytest.approx(3 * expected.energy, rel=1e-10)
    assert actual[3] == expected.source
    # Overlap does not yet export a complete SOL decomposition. Repetition
    # must preserve that explicit absence rather than fabricate components.
    for ops, batch in [([composite], 512), (folded, 2048)]:
        with pytest.raises(SolNotImplementedError, match="no SOL decomposition"):
            handle.evaluate_ops_sol_json(build_ops_json(ops), is_context=False, batch_size=batch, s=65, x=batch)
