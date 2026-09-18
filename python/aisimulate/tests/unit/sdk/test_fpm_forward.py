# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/test_fpm_forward.py

"""Unit tests for forward_model="fpm": FPMForwardOp construction, the
centralized model rewrite, and static/mixed-step routing through the compiled
Rust engine (``Op::FpmForward``; the loader/query machinery lives in
``perf_database/fpm_forward.rs`` with its own tests).

Synthetic parquet/metadata pairs are written directly from the documented
``aic_fpm_forward_perf`` schema — deliberately NOT via collector code, so
this suite doubles as the producer/consumer contract test on the modeling
side of the module boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate.sdk import common, models
from aisimulate.sdk import config as sdk_config
from aisimulate.sdk.backends.factory import get_backend
from aisimulate.sdk.operations import FPMForwardOp
from aisimulate.sdk.perf_database import PerfDatabase
from aisimulate_core.sdk.operations.fpm_forward import _CELL_MATCH_COLUMNS

pytestmark = pytest.mark.unit

SYSTEM = "h200_sxm"
BACKEND = "vllm"
VERSION = "test-fpm-version"
MODEL_PATH = "test-org/test-model"

import aisimulate_core

_CORE_SYSTEMS = os.path.join(os.path.dirname(aisimulate_core.__file__), "systems")


def _row(
    workload_kind: str,
    batch_size: int,
    total_prefill_tokens: int,
    total_kv_read_tokens: int,
    latency_ms: float,
    *,
    model_path: str = MODEL_PATH,
    identity: dict | None = None,
) -> dict:
    base_identity = {
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "",
        "fmha_quant_mode": "",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "bfloat16",
        "tp": "1",
        "pp": "1",
        "dp": "1",
        "moe_tp": "1",
        "moe_ep": "1",
        "cp": "1",
        "moe_backend": "auto",
        "attention_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
    }
    if identity:
        base_identity.update(identity)
    # Identity dicts built from an op's _match_identity carry str(bool)
    # ("True"/"False"); the parquet columns are REAL booleans.
    for knob in ("enable_wideep", "enable_eplb"):
        if isinstance(base_identity[knob], str):
            base_identity[knob] = base_identity[knob] == "True"
    return {
        "cell_id": f"fpm-test-{workload_kind}",
        "model_path": model_path,
        "system": SYSTEM,
        "backend": BACKEND,
        "backend_version": VERSION,
        "weight_quantization": base_identity["gemm_quant_mode"],
        "gemm_quant_mode": base_identity["gemm_quant_mode"],
        "moe_quant_mode": base_identity["moe_quant_mode"] or None,
        "fmha_quant_mode": base_identity["fmha_quant_mode"] or None,
        "comm_quant_mode": base_identity["comm_quant_mode"] or None,
        "kv_cache_dtype": base_identity["kv_cache_dtype"],
        "tp": int(base_identity["tp"]),
        "pp": int(base_identity["pp"]),
        "dp": int(base_identity["dp"]),
        "moe_tp": int(base_identity["moe_tp"]),
        "moe_ep": int(base_identity["moe_ep"]),
        "cp": int(base_identity["cp"]),
        "moe_backend": base_identity["moe_backend"],
        "attention_backend": base_identity["attention_backend"],
        "enable_wideep": base_identity["enable_wideep"],
        "enable_eplb": base_identity["enable_eplb"],
        "workload_kind": workload_kind,
        "batch_size": batch_size,
        "total_prefill_tokens": total_prefill_tokens,
        "total_kv_read_tokens": total_kv_read_tokens,
        "partition_policy": "balanced_v1",
        "latency_ms": latency_ms,
    }


def _write_pair(data_dir: str, rows: list[dict], *, sidecar_overrides: dict | None = None) -> str:
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, "fpm_forward_perf.parquet")
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    with open(parquet_path, "rb") as handle:
        parquet_sha = hashlib.sha256(handle.read()).hexdigest()
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "coordinate_system": "iteration_totals_balanced_v1",
        "measurement_policy": "dynamo_native_single_sample_v1",
        "row_count": len(rows),
        "parquet_sha256": parquet_sha,
        "system": SYSTEM,
        "backend": BACKEND,
        "backend_version": VERSION,
    }
    metadata.update(sidecar_overrides or {})
    with open(os.path.join(data_dir, "fpm_forward_perf.metadata.json"), "w") as handle:
        json.dump(metadata, handle)
    return parquet_path


def _model_config(**overrides) -> sdk_config.ModelConfig:
    defaults = dict(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    )
    defaults.update(overrides)
    return sdk_config.ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# Centralized model rewrite
# ---------------------------------------------------------------------------


class TestForwardModelRewrite:
    def test_default_keeps_op_level_lists(self):
        model = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        assert model.forward_model == "op_level"
        assert len(model.context_ops) > 1
        assert not any(isinstance(op, FPMForwardOp) for op in model.context_ops)

    def test_fpm_rewrite_yields_exactly_one_op_per_phase(self):
        baseline = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        expected_weights = float(sum(op.get_weights() for op in baseline.context_ops))

        model = models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="fpm"), "vllm")
        assert model.forward_model == "fpm"
        assert [op._name for op in model.context_ops] == ["fpm_forward_prefill"]
        assert [op._name for op in model.generation_ops] == ["fpm_forward_decode"]
        assert all(isinstance(op, FPMForwardOp) for op in (*model.context_ops, *model.generation_ops))
        # Weight bytes captured from the original lists keep memory estimation intact.
        assert model.context_ops[0].get_weights() == pytest.approx(expected_weights)

    def test_fpm_spec_preserves_explicit_attention_backend_for_nested_ops(self):
        from aisimulate.sdk.engine import build_engine_spec_json
        from aisimulate.sdk.perf_database import get_database

        attention_backend = "trtllm_mha"
        model = models.get_model(
            "Qwen/Qwen3-0.6B",
            _model_config(forward_model="fpm", attention_backend=attention_backend),
            "sglang",
        )
        database = get_database("b200_sxm", "sglang", "0.5.14")

        spec = json.loads(
            build_engine_spec_json(
                model,
                model_path="Qwen/Qwen3-0.6B",
                system="b200_sxm",
                backend="sglang",
                backend_version="0.5.14",
                kv_block_size=None,
                systems_path=None,
                nextn=0,
                database=database,
            )
        )
        context_lane_order = next(
            op["ContextAttention"]["lane_order"]
            for op in spec["context_ops"][0]["FpmForward"]["sol_ops"]
            if "ContextAttention" in op
        )
        generation_lane_order = next(
            op["GenerationAttention"]["lane_order"]
            for op in spec["generation_ops"][0]["FpmForward"]["sol_ops"]
            if "GenerationAttention" in op
        )

        assert context_lane_order[0] == attention_backend
        assert generation_lane_order[0] == attention_backend

    def test_fpm_rejects_construction_without_sol_ops(self):
        # Legacy "exactly one of sol_fn/sol_ops" contract, minus the retired
        # half: omitting sol_ops keeps raising (main's ValueError), with the
        # message pointing at the surviving parameter.
        with pytest.raises(ValueError, match="provide sol_ops"):
            FPMForwardOp("prefill", _model_config(), MODEL_PATH, weight_bytes=1.0)

    def test_fpm_sol_fn_raises_targeted_migration_error(self):
        # The legacy sol_fn slot stays in the signature (so positional
        # weight_bytes/sol_ops callers keep their meaning) but a callback
        # cannot cross the compiled boundary: passing one must fail with
        # migration guidance, not silently rebind or get ignored.
        with pytest.raises(TypeError, match="sol_ops"):
            FPMForwardOp("prefill", _model_config(), MODEL_PATH, lambda **kwargs: 1.0)

    def test_fpm_legacy_positional_layout_preserved(self):
        # main exposed (phase, model_config, model_path, sol_fn=None,
        # weight_bytes=0.0, sol_ops=None): the 5th positional is
        # weight_bytes and the 6th is sol_ops.
        op = FPMForwardOp("prefill", _model_config(), MODEL_PATH, None, 123.0, [])
        assert op.get_weights() == 123.0
        assert op._sol_ops == []

    def test_fpm_rejects_unknown_phase(self):
        with pytest.raises(ValueError, match="unknown FPM phase"):
            FPMForwardOp("mixed", _model_config(), MODEL_PATH, weight_bytes=1.0, sol_ops=[])

    def test_unknown_forward_model_rejected(self):
        with pytest.raises(ValueError, match="Unknown forward_model"):
            models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="banana"), "vllm")

    def test_encoder_model_rejected(self):
        cfg = _model_config(forward_model="fpm")
        with pytest.raises(NotImplementedError, match="encoder"):
            models.get_model("Qwen/Qwen3-VL-2B-Instruct", cfg, "vllm")

    def test_mtp_rejected(self):
        cfg = _model_config(forward_model="fpm", nextn=1)
        with pytest.raises(NotImplementedError, match="MTP"):
            models.get_model("Qwen/Qwen3-0.6B", cfg, "vllm")

    # -- Hybrid speculative shape (verify-on-FPM) --

    _EAGLE3_CONFIG: ClassVar[dict] = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "num_hidden_layers": 1,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "draft_vocab_size": 32000,
        "sliding_window": None,
        "use_sliding_window": False,
    }

    def test_fpm_hybrid_ngram_sets_verify_width_without_draft_ops(self):
        from aisimulate_core.sdk.speculation import SpeculationConfig

        cfg = _model_config(
            forward_model="fpm",
            speculation=SpeculationConfig(kind="ngram", params={"num_speculative_tokens": 3}),
        )
        model = models.get_model("Qwen/Qwen3-0.6B", cfg, "vllm")
        # ngram drafts on the host: no draft ops, pure verify-width change.
        assert [op._name for op in model.generation_ops] == ["fpm_forward_decode"]
        assert model.generation_ops[0]._verify_width == 4
        assert model.context_ops[0]._verify_width == 1  # prefill untouched
        assert model._nextn == 3  # engine widening channel stays consistent

    def test_fpm_standalone_draft_keeps_native_ops(self):
        from aisimulate_core.sdk.speculation import SpeculationConfig

        cfg = _model_config(
            forward_model="fpm",
            speculation=SpeculationConfig(
                kind="draft_model", params={"num_speculative_tokens": 3}, draft_model_path="Qwen/Qwen3-0.6B"
            ),
        )
        model = models.get_model("Qwen/Qwen3-8B", cfg, "vllm")
        assert model.generation_ops[0]._verify_width == 4
        assert model.generation_ops[1:]
        assert all(not isinstance(op, FPMForwardOp) for op in model.generation_ops[1:])
        assert all(not isinstance(op, FPMForwardOp) for op in model.context_ops[1:])
        draft = model.spec_scheme._draft_model
        assert draft.config.forward_model == "op_level"
        assert draft._nextn == 0

    def test_fpm_hybrid_eagle3_keeps_draft_ops_op_level(self):
        from aisimulate_core.sdk.engine import _fpm_spec_dict
        from aisimulate_core.sdk.speculation import SpeculationConfig

        cfg = _model_config(
            forward_model="fpm",
            speculation=SpeculationConfig(
                kind="eagle3",
                params={"num_speculative_tokens": 3},
                draft_config=self._EAGLE3_CONFIG,
            ),
        )
        baseline = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        target_weights = float(sum(op.get_weights() for op in baseline.context_ops))
        model = models.get_model("Qwen/Qwen3-0.6B", cfg, "vllm")

        # Phase lists LEAD with the whole-model op; draft ops follow op-level.
        assert isinstance(model.generation_ops[0], FPMForwardOp)
        gen_tail = model.generation_ops[1:]
        assert gen_tail and all(op._name.startswith("draft_") for op in gen_tail)
        assert model.generation_ops[0]._verify_width == model.verify_width == 4

        # The whole-model fold covers the TARGET only: sol_ops and the weight
        # inventory exclude draft ops (draft memory rides the scheme hooks).
        assert not any(op._name.startswith("draft_") for op in model.generation_ops[0]._sol_ops)
        assert model.context_ops[0].get_weights() == pytest.approx(target_weights)

        # Wire: verify_width rides the FpmForward opspec.
        spec = _fpm_spec_dict(model.generation_ops[0])
        assert spec["FpmForward"]["verify_width"] == 4


# ---------------------------------------------------------------------------
# Static + mixed-step integration through a real PerfDatabase/backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def fpm_session(tmp_path):
    """A real PerfDatabase over a temp systems root holding ONLY fpm data,
    plus an fpm-mode model whose identity the rows are written to match."""
    systems_root = tmp_path / "systems"
    os.makedirs(systems_root, exist_ok=True)
    shutil.copy(os.path.join(_CORE_SYSTEMS, f"{SYSTEM}.yaml"), systems_root / f"{SYSTEM}.yaml")

    model = models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="fpm"), BACKEND)
    identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))

    isl, osl = 512, 2
    rows = [
        # static_ctx: batch=2 x isl new tokens, and the mixed ctx component (batch=1 x isl)
        _row("prefill", 2, 2 * isl, 0, 40.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, isl, 0, 22.0, model_path=model.model_path, identity=identity),
        # static_gen with osl=2 runs one decode step at s = isl+1.
        _row("decode", 2, 0, 2 * (isl + 1), 6.0, model_path=model.model_path, identity=identity),
        # mixed/genonly route through run_static(isl=isl+osl//2, osl=2), whose
        # decode step lands at s = isl + osl//2 + 1.
        _row("decode", 2, 0, 2 * (isl + osl // 2 + 1), 7.0, model_path=model.model_path, identity=identity),
        # Totals-coordinate rows for the mixed-step composition: the prefill
        # component queries (batch, chunk + gen_tokens, past_kv).
        _row("prefill", 1, isl + 2, 0, 23.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 258, 0, 11.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 258, 256, 13.0, model_path=model.model_path, identity=identity),
        # Prefill-only chunk averages used by the hybrid attribution probe.
        _row("prefill", 1, 256, 0, 10.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 256, 256, 12.0, model_path=model.model_path, identity=identity),
        # CUDA-graph cliff pair at capture=2048 plus the eager plateau: the
        # regime is encoded in the data, the formula only addresses it.
        _row("prefill", 1, 2048, 0, 47.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 2049, 0, 99.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 4096, 0, 99.0, model_path=model.model_path, identity=identity),
        # Decode coverage for the cliff test (gen=8 at isl=2048, osl=2).
        _row("decode", 8, 0, 1026, 6.5, model_path=model.model_path, identity=identity),
        _row("decode", 8, 0, 16400, 9.5, model_path=model.model_path, identity=identity),
    ]
    # data_dir comes from the system yaml ("data/h200_sxm").
    data_dir = os.path.join(systems_root, "data", SYSTEM, BACKEND, VERSION)
    _write_pair(data_dir, rows)

    database = PerfDatabase(SYSTEM, BACKEND, VERSION, systems_root=str(systems_root))
    backend = get_backend(BACKEND)
    return model, database, backend, isl, osl


class TestFPMStaticAndMixed:
    @pytest.mark.parametrize("ctx_tokens", [256, 512])
    @pytest.mark.parametrize("gen_requests", [0, 2])
    def test_public_mixed_hybrid_keeps_native_component_sources(self, fpm_session, ctx_tokens, gen_requests):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession
        from aisimulate_core.sdk.operations.elementwise import ElementWise
        from aisimulate_core.sdk.rust_engine_step import _cached_engine_handle
        from aisimulate_core.sdk.speculation import SpeculationConfig
        from aisimulate_core.sdk.speculation.materialize import _fold_width
        from aisimulate_core.sdk.step_estimate import MixedStepInput

        baseline, database, backend, isl, osl = fpm_session
        model = models.get_model(
            baseline.model_path,
            _model_config(
                forward_model="fpm",
                speculation=SpeculationConfig(kind="ngram", params={"num_speculative_tokens": 3}),
            ),
            BACKEND,
        )
        # A synthetic materialized draft graph runs real native empirical ops
        # alongside the fixture's silicon-tagged FPM table components.
        model.context_ops.append(ElementWise("draft_context", 1.0, 4096, 4096, 0.8))
        generation = ElementWise("draft_generation", 1.0, 4096, 4096, 0.8)
        _fold_width(generation, 1, 4)
        model.generation_ops.append(generation)
        handle = _cached_engine_handle(model, database)
        native = handle._mixed_step_breakdown_per_op_with_metadata(ctx_tokens, gen_requests, isl, osl, 0)
        estimate = InferenceSession(model, database, backend).run_mixed(
            RuntimeConfig(isl=isl, osl=osl), MixedStepInput(ctx_tokens, gen_requests)
        )
        shared, context, decode = native
        assert {row[0] for row in shared} == {"fpm_forward_prefill", "draft_context"}
        assert {row[0] for row in decode} == ({"fpm_forward_decode", "draft_generation"} if gen_requests else set())
        rows = {row[0]: row for group in native for row in group}
        # Independent phase queries establish draft values, energy and source;
        # the reporting path must not absorb these into target FPM rows.
        expected_context = handle.evaluate_context_ops([1], batch_size=1, s=ctx_tokens)[0]
        assert rows["draft_context"][1:3] == pytest.approx(expected_context[1:3])
        assert rows["draft_context"][3] == expected_context[3] == "empirical"
        if gen_requests:
            expected_generation = handle.evaluate_generation_ops(
                [1], batch_size=gen_requests * 4, s=isl + osl // 2 + 1
            )[0]
            assert rows["draft_generation"][1:3] == pytest.approx(expected_generation[1:3])
            assert rows["draft_generation"][3] == expected_generation[3] == "empirical"
        assert rows["fpm_forward_prefill"][3] == "silicon"
        public_rows = {
            "generation_attention" if name == "fpm_forward_decode" else name: row for name, row in rows.items()
        }
        expected_latency = {name: pytest.approx(row[1]) for name, row in public_rows.items()}
        expected_latency.setdefault("generation_attention", 0.0)
        expected_latency["context_attention (scaled)"] = 0.0
        assert estimate.per_op_latency_ms == expected_latency
        expected_source = {name: row[3] for name, row in public_rows.items()}
        expected_source.setdefault("generation_attention", "silicon")
        expected_source["context_attention (scaled)"] = "silicon"
        assert estimate.per_op_source == expected_source
        for name, group in zip(("shared_non_attention", "context_attention", "decode_attention"), native, strict=True):
            assert estimate.component_latency_ms[name] == pytest.approx(sum(row[1] for row in group))
            assert estimate.component_energy_wms[name] == pytest.approx(sum(row[2] for row in group))
        assert estimate.latency_ms == pytest.approx(handle.mixed_step_latency(ctx_tokens, gen_requests, isl, osl, 0))
        assert estimate.latency_ms == pytest.approx(sum(row[1] for row in rows.values()))
        assert estimate.energy_wms == pytest.approx(sum(row[2] for row in rows.values()))

    def test_ngram_verify_width_reaches_native_fpm_query(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession
        from aisimulate_core.sdk.speculation import SpeculationConfig

        baseline, database, backend, isl, osl = fpm_session
        model = models.get_model(
            baseline.model_path,
            _model_config(
                forward_model="fpm",
                speculation=SpeculationConfig(kind="ngram", params={"num_speculative_tokens": 3}),
            ),
            BACKEND,
        )
        summary = InferenceSession(model, database, backend).run_static(
            RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl), mode="static_gen"
        )
        # Two requests verify four queries each; shared KV remains 2 * 513.
        # The synthetic fixture has an exact (batch=8, KV=1026) row at 6.5 ms.
        assert summary.get_generation_latency_dict() == {"fpm_forward_decode": pytest.approx(6.5)}

    def test_static_ctx_uses_fpm_row(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession

        model, database, backend, isl, osl = fpm_session
        session = InferenceSession(model, database, backend)
        summary = session.run_static(
            runtime_config=RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl),
            mode="static_ctx",
        )
        latency_dict = summary.get_context_latency_dict()
        assert list(latency_dict) == ["fpm_forward_prefill"]
        assert latency_dict["fpm_forward_prefill"] == pytest.approx(40.0)

    def test_static_gen_uses_fpm_row(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig
        from aisimulate.sdk.inference_session import InferenceSession

        model, database, backend, isl, osl = fpm_session
        session = InferenceSession(model, database, backend)
        summary = session.run_static(
            runtime_config=RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl),
            mode="static_gen",
        )
        latency_dict = summary.get_generation_latency_dict()
        assert list(latency_dict) == ["fpm_forward_decode"]
        # osl=2 -> one decode step at s=isl+1, repeat_count 1.
        assert latency_dict["fpm_forward_decode"] == pytest.approx(6.0)

    def test_mixed_step_is_prefill_plus_marginal_decode(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, energy, per_op, per_src = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=isl, gen_tokens=2, isl=isl, osl=osl, prefix=0
        )
        # ctx component prices the step's SCHEDULED TOTAL: one whole prefill
        # (isl tokens) plus 2 decode riders -> totals (1, isl+2, 0) = 23.0.
        # gen component rides the prefill pass, so only its marginal counts:
        # full decode at s=isl+osl//2+1 (7.0) minus the pass baseline at the
        # KV-domain floor (the 2*(isl+1)=1026 row, 6.0) -> 1.0. The compiled
        # engine reports the decode marginal under the mixed breakdown's
        # uniform "generation_attention" component key.
        assert per_op["fpm_forward_prefill"] == pytest.approx(23.0)
        assert per_op["generation_attention"] == pytest.approx(1.0)
        assert total == pytest.approx(24.0)
        assert energy == 0.0
        assert set(per_src.values()) == {"silicon"}

    def test_mixed_step_total_crosses_the_graph_cliff(self, fpm_session):
        # Spec tests 1+2: the engine picks its regime from the step's TOTAL
        # scheduled tokens. ctx=2048 alone sits ON the capture boundary
        # (graph side, 47 ms); the same chunk with 8 decode riders crosses
        # it and must price on the eager plateau (99 ms).
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=2048, osl=osl)
        _, _, graph_ops, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=2048, gen_tokens=0, isl=2048, osl=osl, prefix=0
        )
        assert graph_ops["fpm_forward_prefill"] == pytest.approx(47.0)
        _, _, eager_ops, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=2048, gen_tokens=8, isl=2048, osl=osl, prefix=0
        )
        assert eager_ops["fpm_forward_prefill"] == pytest.approx(99.0)
        assert eager_ops["fpm_forward_prefill"] > 2 * graph_ops["fpm_forward_prefill"]

    def test_mixed_step_chunks_average_exact_coordinates(self, fpm_session):
        # Spec tests 3+4: a chunked request prices each chunk at its own
        # (chunk + gen, past_kv) coordinates — the exact rows (1, 258, 0)=11.0
        # and (1, 258, 256)=13.0 for ctx=256 of isl=512 — and the component is
        # their per-iteration average, identical to pricing the chunks
        # independently (no double billing, no averaging artifacts).
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, _, per_op, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=256, gen_tokens=2, isl=isl, osl=osl, prefix=0
        )
        assert per_op["fpm_forward_prefill"] == pytest.approx((11.0 + 13.0) / 2) == pytest.approx(12.0)
        assert total == pytest.approx(12.0 + 1.0)

    def test_mixed_step_gen_zero_prices_pure_chunk(self, fpm_session):
        # Spec test 5 (gen=0 degenerate): a pure-prefill step prices its own
        # totals with no decode marginal term.
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, _, per_op, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=isl, gen_tokens=0, isl=isl, osl=osl, prefix=0
        )
        assert per_op == {
            "fpm_forward_prefill": pytest.approx(22.0),
            # The mixed breakdown's uniform component keys are always present
            # (zero when the pass contributed nothing).
            "context_attention (scaled)": 0,
            "generation_attention": 0,
        }
        assert total == pytest.approx(22.0)

    def test_genonly_step_keeps_full_decode_pass(self, fpm_session):
        # With no prefill work in the step there is no pass to ride on: the
        # decode component must keep its full standalone latency. A step
        # without prefill work is a GENONLY step now — `MixedStepInput`
        # requires context_tokens > 0, so the gen-only contract lives behind
        # `_get_genonly_step_latency` (the mixed entry raises instead of
        # silently rerouting).
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, _energy, per_op, _src, _fallbacks = backend._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens=2, isl=isl, osl=osl
        )
        assert per_op["fpm_forward_decode"] == pytest.approx(7.0)
        assert total == pytest.approx(7.0)
        with pytest.raises(ValueError, match="context_tokens must be positive"):
            backend._get_mix_step_latency(
                model, database, runtime_config, ctx_tokens=0, gen_tokens=2, isl=isl, osl=osl, prefix=0
            )

    def test_genonly_step_works_with_single_op(self, fpm_session):
        from aisimulate.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl)
        total, energy, per_op, _, _fallbacks = backend._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens=2, isl=isl, osl=osl
        )
        assert per_op["fpm_forward_decode"] == pytest.approx(7.0)
        assert total == pytest.approx(7.0)


def test_explicit_selector_emits_matched_cell_warning_once(fpm_session, capfd):
    from aisimulate_core.sdk.rust_engine_step import _cached_engine_handle

    baseline, database, _backend, _isl, _osl = fpm_session
    selected = models.get_model(
        baseline.model_path,
        _model_config(forward_model="fpm", fpm_fmha_quant_mode=baseline.config.fmha_quant_mode),
        BACKEND,
    )
    original = baseline.config.fmha_quant_mode.name
    capfd.readouterr()
    handle = _cached_engine_handle(selected, database)
    first = handle.evaluate_context_ops([0], batch_size=1, s=512)
    warning = capfd.readouterr().err
    assert first[0][1] == 22.0
    assert "WARNING: FPM table FMHA selector" in warning
    assert f'original_model_mode="{original}"' in warning
    assert f'selector="{original}"' in warning
    assert "matched_cell_ids=" in warning
    assert "fpm-test-prefill" in warning and "fpm-test-decode" in warning
    assert "does not independently verify runtime attention precision" in warning
    assert handle.evaluate_context_ops([0], batch_size=1, s=512) == first
    assert "FPM table FMHA selector" not in capfd.readouterr().err
