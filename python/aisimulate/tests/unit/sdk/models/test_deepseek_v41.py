# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from dataclasses import replace

import pytest

from aisimulate_core.sdk import common
from aisimulate_core.sdk.deepseek_v41 import (
    MODEL_PATH,
    DeepSeekV41Config,
    DeepSeekV41KVCacheLayout,
    V41RequestWorkload,
    resolve_execution_profile,
    resolve_kv_cache_layout,
    stage_workloads,
)
from aisimulate_core.sdk.models.helpers import _infer_quant_modes_from_raw_config
from aisimulate_core.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit


@pytest.fixture
def descriptor():
    return get_model_config_from_model_path(MODEL_PATH)["extra_params"]


def test_real_v41_config_and_quant(descriptor):
    info = get_model_config_from_model_path(MODEL_PATH)
    assert isinstance(descriptor, DeepSeekV41Config)
    assert (info["layers"], info["hidden_size"], info["n"]) == (40, 5120, 64)
    assert len(descriptor.compress_ratios) == 40
    assert Counter(descriptor.layer_role(i) for i in range(40)) == {"swa": 2, "full": 4, "reindex": 4, "reuse": 30}
    quant = _infer_quant_modes_from_raw_config(info["raw_config"], info["architecture"])
    assert quant["moe_quant_mode"] == common.MoEQuantMode.w4a8_mxfp4_mxfp8


@pytest.mark.parametrize("system", ["gb200", "gb300"])
def test_sglang_blackwell_experts_select_measured_kernel_lane(system):
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models.helpers import resolve_dsv4_moe_arch, resolve_dsv4_moe_arch_mode

    assert resolve_dsv4_moe_arch_mode(MODEL_PATH, system, "sglang") == common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm
    assert resolve_dsv4_moe_arch_mode(MODEL_PATH, system, "vllm") is None
    explicit = ModelConfig(moe_quant_mode=common.MoEQuantMode.bfloat16)
    resolve_dsv4_moe_arch(explicit, MODEL_PATH, system_name=system, backend_name="sglang")
    assert explicit.moe_quant_mode == common.MoEQuantMode.bfloat16


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
@pytest.mark.parametrize("total_gpus", [4, 32])
def test_default_task_enumerates_matching_moe_parallel_widths(backend, total_gpus):
    from aisimulate.sdk.task_v2 import Task

    task = Task(
        model_path=MODEL_PATH,
        system_name="gb300",
        backend_name=backend,
        total_gpus=total_gpus,
        database_mode="SOL",
        isl=1024,
        osl=128,
        nextn=0,
    )
    assert task.model_family == "DEEPSEEKV41"
    assert task.is_moe
    parallel = [tuple(choice) for choice in task.iter_parallel("agg")]
    assert (4, 1, 1, 1, 4, 1) in parallel
    assert all(tp * dp * cp == moe_tp * moe_ep for tp, _, dp, moe_tp, moe_ep, cp in parallel)


def test_shared_pool_memory_slope_and_odd_decode(descriptor):
    assert descriptor.compressed_entry_bytes == 288
    assert descriptor.index_entry_bytes == 68
    assert descriptor.kvcache_bytes(130) - descriptor.kvcache_bytes(128) == 2 * 890
    # Only the ratio-one owner grows on odd tokens; the three ratio-two owners
    # publish their next compressed entry on the following even token.
    assert descriptor.kvcache_bytes(129) - descriptor.kvcache_bytes(128) == 356
    assert descriptor.engram_table_bytes(1) == 202758032400


def test_sglang_physical_payload_owners_and_publication_boundaries(descriptor):
    layout = DeepSeekV41KVCacheLayout.SGLANG_FP8_BF16
    assert descriptor.kv_entry_bytes(layout) == (584, 584, 68)
    # 40 SWA pools; one full-rate and three half-rate shared KV/index owners.
    # Reindex layers share both physical pools, rather than allocating copies.
    assert descriptor.kvcache_bytes(131072, layout) == 216662016 == 206.625 * 2**20
    assert descriptor.kvcache_bytes(129, layout) - descriptor.kvcache_bytes(128, layout) == 652
    assert descriptor.kvcache_bytes(130, layout) - descriptor.kvcache_bytes(129, layout) == 4 * 652
    assert descriptor.kvcache_bytes(130, layout) - descriptor.kvcache_bytes(128, layout) == 3260
    for invalid in (replace(descriptor, head_dim=256), replace(descriptor, qk_rope_head_dim=32)):
        with pytest.raises(ValueError, match="pinned SGLang"):
            invalid.kv_entry_bytes(layout)
    with pytest.raises(ValueError, match="unknown"):
        descriptor.kv_entry_bytes("unknown")
    with pytest.raises(ValueError, match="unknown"):
        resolve_kv_cache_layout("unknown")


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
@pytest.mark.parametrize("tokens", [127, 128, 129, 130, 16384, 131072])
def test_physical_and_theoretical_capacity_inverse(backend, tokens, descriptor):
    model = _build_model(backend=backend)
    expected = DeepSeekV41KVCacheLayout.SGLANG_FP8_BF16 if backend == "sglang" else DeepSeekV41KVCacheLayout.LOGICAL_FP4
    assert model.kv_cache_layout == expected
    capacity = descriptor.kvcache_bytes(tokens, expected)
    assert model.get_kvcache_bytes_per_sequence(tokens) == capacity
    assert model.get_kvcache_max_tokens(capacity) == tokens
    assert model.get_kvcache_max_tokens(capacity - 1) < tokens


def test_replay_tail_is_bounded_per_actual_request(descriptor):
    requests = (V41RequestWorkload(256, 1024), V41RequestWorkload(3, 4096))
    full = stage_workloads(descriptor, "full", requests, 39)
    assert [(r.query_tokens, r.prefix_tokens) for r in full] == [(256, 1024), (3, 4096)]
    tail = stage_workloads(descriptor, "decoder_bounded", requests, 21)
    assert [(r.query_tokens, r.prefix_tokens) for r in tail] == [(128, 1152), (3, 4096)]
    assert stage_workloads(descriptor, "decoder_bounded", requests, 20) == full


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_unverified_replay_backend_fails(backend):
    assert resolve_execution_profile(False, backend) == "full"
    with pytest.raises(NotImplementedError, match="not verified"):
        resolve_execution_profile(True, backend)


def _build_model(*, replay=False, backend="sglang", tp=4):
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model

    return get_model(
        MODEL_PATH,
        ModelConfig(tp_size=tp, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=tp, decoder_replay=replay),
        backend,
    )


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
def test_v41_text_graph_full_profile_all_backends(backend):
    import json

    model = _build_model(backend=backend)
    assert model.execution_profile == "full"
    assert not model.encoder_ops
    specs = [json.loads(op._spec_json()) for op in model.context_ops]
    stages = [op["Dsv41Stage"] for op in specs if "Dsv41Stage" in op]
    assert len(stages) == 40
    layouts = {
        child["Dsv41Attention"]["kv_cache_layout"]
        for stage in stages
        for child in stage["children"]
        if "Dsv41Attention" in child
    }
    assert layouts == {resolve_kv_cache_layout(backend).value}
    assert sum(stage["bounded"] for stage in stages) == 19
    assert all(not stage["decoder_replay"] for stage in stages)
    roles = Counter(
        next(c["Dsv41Attention"]["role"] for c in stage["children"] if "Dsv41Attention" in c) for stage in stages
    )
    assert roles == {"swa": 2, "full": 4, "reindex": 4, "reuse": 30}


def test_replay_keeps_inventory_and_kv_pool_capacity():
    full, replay = _build_model(), _build_model(replay=True)
    assert replay.execution_profile == "decoder_bounded"
    assert full.get_resident_weights_bytes() == replay.get_resident_weights_bytes()
    assert full.get_resident_weights_bytes() > full.extra_params.engram_table_bytes(4)
    assert full.get_kvcache_bytes_per_sequence(4096) == replay.get_kvcache_bytes_per_sequence(4096)
    assert replay.get_kvcache_max_tokens(replay.get_kvcache_bytes_per_sequence(4096)) == 4096


def test_batch_capacity_reserves_each_request_window(descriptor):
    model = _build_model()
    fixed = 40 * 128 * 584 + 3 * 2 * 2 * 512 * 4
    assert model.get_kvcache_batch_capacity(4 * fixed + 4096 * 1630, 4) == 4096
    assert model.get_kvcache_batch_capacity(4 * fixed - 1, 4) == 0
    assert model.get_additional_activation_bytes(1024) > 1024 * 4 * 5120 * 2


@pytest.mark.parametrize(("dp", "pp"), [(2, 1), (1, 2)])
def test_unmodeled_parallel_cache_ownership_fails(dp, pp):
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model

    with pytest.raises(NotImplementedError, match="DP Engram collectives and PP cache ownership"):
        get_model(
            MODEL_PATH,
            ModelConfig(tp_size=4, pp_size=pp, attention_dp_size=dp, moe_tp_size=1, moe_ep_size=4 * dp),
            "sglang",
        )


def test_native_sol_replay_decode_and_short_extend_contract():
    from aisimulate_core.sdk.engine import EngineHandle, compile_engine

    def engine(replay):
        return EngineHandle(
            compile_engine(
                MODEL_PATH,
                "gb300",
                "sglang",
                tp_size=4,
                moe_tp_size=1,
                moe_ep_size=4,
                decoder_replay=replay,
                database_mode="SOL",
            )
        )

    full, replay = engine(False), engine(True)
    assert replay.predict_prefill_latency(1, 1024) < full.predict_prefill_latency(1, 1024)
    assert replay.predict_decode_latency(1, 1024) == full.predict_decode_latency(1, 1024)
    # Both modes process the same three actual query tokens; the bounded SWA
    # intentionally does not read the pre-existing ring prefix.
    assert replay.predict_prefill_latency(1, 4099, 4096) > 0
    assert replay.mixed_step_latency(2048, 1, 1024, 2) > 0


def test_native_block32_shared_projections_have_distinct_perf_identity():
    import json

    model = _build_model()
    first_layer = next(
        json.loads(op._spec_json())["Dsv41Stage"]
        for op in model.context_ops
        if "Dsv41Stage" in json.loads(op._spec_json())
    )
    linears = [c["Dsv41Linear"] for c in first_layer["children"] if "Dsv41Linear" in c]
    assert {(op["n"], op["k"]) for op in linears} == {(1152, 5120), (5120, 576)}
    assert all(c["Gemm"]["quant_mode"] == "bfloat16" for c in first_layer["children"] if "Gemm" in c)


@pytest.mark.parametrize("first_replay", [False, True])
def test_engine_cache_keeps_replay_profiles_separate(first_replay):
    from aisimulate_core.sdk import rust_engine_step
    from aisimulate_core.sdk.perf_database import get_database_view

    database = get_database_view("gb300", "sglang", "current", allow_missing_data=True, database_mode="SOL")
    rust_engine_step._engine_handle_cache_clear()
    try:
        models = {replay: _build_model(replay=replay) for replay in (first_replay, not first_replay)}
        handles = {replay: rust_engine_step._cached_engine_handle(model, database) for replay, model in models.items()}
        assert handles[False] is not handles[True]
        assert handles[True].predict_prefill_latency(1, 1024) < handles[False].predict_prefill_latency(1, 1024)
        assert handles[True].predict_decode_latency(1, 1024) == handles[False].predict_decode_latency(1, 1024)
        assert (
            rust_engine_step._cached_engine_handle(_build_model(replay=first_replay), database) is handles[first_replay]
        )
    finally:
        rust_engine_step._engine_handle_cache_clear()


def test_sglang_pure_tp_eager_shared_and_routed_costs_are_sequential():
    import json

    import aisimulate_core._native as native
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.models import get_model
    from aisimulate_core.sdk.perf_database import get_database_view

    model = get_model(MODEL_PATH, ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1), "sglang")
    stages = [
        json.loads(op._spec_json())["Dsv41Stage"]
        for op in model.generation_ops
        if "Dsv41Stage" in json.loads(op._spec_json())
    ]
    assert len(stages) == 40
    assert all(not any("Overlap" in child for child in stage["children"]) for stage in stages)
    children = stages[0]["children"]
    shared = [child for child in children if "shared_" in next(iter(child.values())).get("name", "")]
    routed = [
        child
        for child in children
        if next(iter(child.values())).get("name", "")
        in ("generation_router_gemm", "generation_moe_pre_dispatch", "generation_moe")
    ]
    assert len(shared) == 3 and len(routed) == 2
    db = get_database_view("gb300", "sglang", "current", allow_missing_data=True, database_mode="SOL")

    def cost(children):
        stage = stages[0] | {"children": children}
        op = native.op_from_spec_json(json.dumps({"Dsv41Stage": stage}))
        return float(_evaluate_single_op(db, op, is_context=False, batch_size=2, s=129, prefix=0, x=2))

    assert cost(shared + routed) == pytest.approx(cost(shared) + cost(routed))
    assert cost(shared + routed) > max(cost(shared), cost(routed))


def test_unqualified_ep_generation_overlap_is_not_changed_by_tp_eager_fix():
    import json

    model = _build_model()
    stages = [
        json.loads(op._spec_json())["Dsv41Stage"]
        for op in model.generation_ops
        if "Dsv41Stage" in json.loads(op._spec_json())
    ]
    assert all(any("Overlap" in child for child in stage["children"]) for stage in stages)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"compress_ratios": (0,)}, "one backbone compression ratio"),
        ({"compress_ratios": (3,) * 40}, "one backbone compression ratio"),
        ({"kv_source_layer_ids": (8, 2)}, "unique, ascending"),
        ({"index_source_layer_ids": (2, 8, 14)}, "must also own an indexer"),
        ({"candidate_source_layer_id": 3}, "candidate source must own"),
        ({"engram_num_embeddings": (1,)}, "layer/table counts differ"),
        (
            {"kv_source_layer_ids": (0, 2, 8, 14, 20), "index_source_layer_ids": (0, 2, 8, 14, 20, 24, 28, 32, 36)},
            "positive compression ratio",
        ),
    ],
)
def test_descriptor_rejects_malformed_ownership(descriptor, changes, message):
    with pytest.raises(ValueError, match=message):
        replace(descriptor, **changes).validate()


def test_descriptor_rejects_ownerless_and_incompatible_compression(descriptor):
    for layer, ratio in [(0, 1), (3, 1)]:
        ratios = list(descriptor.compress_ratios)
        ratios[layer] = ratio
        with pytest.raises(ValueError, match="no compatible preceding KV owner"):
            replace(descriptor, compress_ratios=tuple(ratios)).validate()


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"nextn": 1}, NotImplementedError, "nextn=0"),
        ({"overwrite_num_layers": 2}, ValueError, "layer overrides"),
        ({"moe_backend": "megamoe"}, NotImplementedError, "decomposed MoE"),
        ({"cp_size": 2, "moe_ep_size": 8}, NotImplementedError, "Context parallelism"),
        ({"tp_size": 16, "moe_ep_size": 16}, ValueError, "TP must divide"),
    ],
)
def test_model_constructor_rejects_unmodeled_configuration(changes, error, message):
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model

    config = ModelConfig(**({"tp_size": 4, "moe_tp_size": 1, "moe_ep_size": 4} | changes))
    with pytest.raises(error, match=message):
        get_model(MODEL_PATH, config, "sglang")


@pytest.mark.parametrize("tp", [1, 4, 8])
def test_replicated_indexer_preserves_weights_and_score_cost_across_tp(tp):
    import json

    import aisimulate_core._native as native
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.perf_database import get_database_view

    model = _build_model(tp=tp)
    specs = [json.loads(op._spec_json()) for op in model.context_ops]
    attention = next(
        child["Dsv41Attention"]
        for stage in specs
        if "Dsv41Stage" in stage
        for child in stage["Dsv41Stage"]["children"]
        if "Dsv41Attention" in child and child["Dsv41Attention"]["role"] == "reindex"
    )
    assert attention["index_n_heads"] == 32
    indexed = native.op_from_spec_json(json.dumps({"Dsv41Attention": attention}))
    reused = native.op_from_spec_json(json.dumps({"Dsv41Attention": attention | {"role": "reuse"}}))
    # 1280*32*128 FP8 weights plus 32x32 block scales, and 5120*32 BF16 gates.
    assert indexed.get_weights() - reused.get_weights() == 5_575_680
    db = get_database_view("gb300", "sglang", "current", allow_missing_data=True, database_mode="SOL")

    def cost(op):
        return float(_evaluate_single_op(db, op, is_context=True, batch_size=1, s=4096, prefix=0, x=4096))

    index_cost = cost(indexed) - cost(reused)
    # Every TP uses the identical replicated indexer. Changing main attention
    # heads must not divide this complete scoring/selection contribution.
    reference = attention | {"num_heads": 64, "o_groups": 8}
    full = native.op_from_spec_json(json.dumps({"Dsv41Attention": reference}))
    full_reuse = native.op_from_spec_json(json.dumps({"Dsv41Attention": reference | {"role": "reuse"}}))
    assert index_cost == pytest.approx(cost(full) - cost(full_reuse), rel=1e-12)
    assert index_cost > 0


@pytest.mark.parametrize(
    ("backend_name", "coefficient", "overhead"), [("sglang", 13, 1.15), ("vllm", 10, 1.0), ("trtllm", 10, 1.0)]
)
def test_v41_actual_activation_memory_uses_moe_coefficient(backend_name, coefficient, overhead):
    from aisimulate_core.sdk.backends.factory import get_backend
    from aisimulate_core.sdk.perf_database import get_database_view

    model = _build_model(backend=backend_name)
    backend = get_backend(backend_name)
    db = get_database_view("gb300", backend_name, "current", allow_missing_data=True, database_mode="SOL")
    memory = backend._get_memory_usage(model, db, 1, 1, 8192, 1, num_tokens=8192)
    # Generic MoE workspace plus the separately owned mHC/Engram buffers.
    workspace_width = 5120 if backend_name == "sglang" else 64 * 512
    workspace = 8192 * workspace_width * 384 * 6 / 4 / 128 * 4
    expanded = model.get_additional_activation_bytes(8192)
    expected = (2 * 8192 * 64 * 512 * coefficient + workspace + expanded) * overhead
    assert memory["activations"] * (1 << 30) == pytest.approx(expected)
    assert memory["weights"] * (1 << 30) == model.get_resident_weights_bytes()


def test_nested_stage_rejects_retired_dispatch_at_serialization():
    import json

    import aisimulate_core._native as native

    model = _build_model()
    stage = next(json.loads(op._spec_json()) for op in model.context_ops if "Dsv41Stage" in json.loads(op._spec_json()))
    dispatch = next(child for child in stage["Dsv41Stage"]["children"] if "MoeDispatch" in child)
    dispatch["MoeDispatch"]["flavor"] = "RetiredDeepEp"
    inner = stage | {"Dsv41Stage": stage["Dsv41Stage"] | {"children": [dispatch]}}
    outer = stage | {"Dsv41Stage": stage["Dsv41Stage"] | {"children": [inner]}}
    op = native.op_from_spec_json(json.dumps(outer))
    with pytest.raises(ValueError, match="retired"):
        native.ops_json_from_ops([op])


@pytest.mark.parametrize("system", ["b200_sxm", "b300_sxm"])
def test_hgx_blackwell_has_published_scalar_fp32_rate(system):
    from aisimulate_core.sdk.engine import EngineHandle, compile_engine
    from aisimulate_core.sdk.perf_database import get_database_view

    db = get_database_view(system, "sglang", "current", allow_missing_data=True, database_mode="SOL")
    assert db.system_spec["gpu"]["fp32_flops"] == 75e12
    engine = EngineHandle(
        compile_engine(MODEL_PATH, system, "sglang", tp_size=4, moe_tp_size=4, moe_ep_size=1, database_mode="SOL")
    )
    assert engine.predict_prefill_latency(1, 128) > 0


@pytest.mark.parametrize("decoder_replay", [False, True])
def test_afd_rejects_v41_before_search_or_session_construction(decoder_replay):
    from aisimulate.sdk.inference_session import AFDInferenceSession
    from aisimulate.sdk.task_v2 import Task
    from aisimulate_core.sdk.config import ModelConfig

    with pytest.raises(NotImplementedError, match="AFD does not support DeepSeek-V4.1"):
        Task(
            model_path=MODEL_PATH,
            system_name="gb300",
            backend_name="sglang",
            total_gpus=8,
            database_mode="SOL",
            serving_mode="afd",
            isl=1024,
            osl=128,
            nextn=0,
        )
    with pytest.raises(NotImplementedError, match="AFD does not support DeepSeek-V4.1"):
        AFDInferenceSession(
            MODEL_PATH,
            ModelConfig(decoder_replay=decoder_replay),
            ModelConfig(decoder_replay=decoder_replay),
            None,
            None,
            None,
        )


@pytest.mark.parametrize("is_context", [False, True])
@pytest.mark.parametrize("batch,seq", [(1, 4), (0, 4), (1, 0)])
@pytest.mark.parametrize("database_mode", ["SOL", "SILICON", "HYBRID", "EMPIRICAL"])
def test_native_attention_rejects_unknown_role_before_zero_work(is_context, batch, seq, database_mode):
    import json

    import aisimulate_core._native as native
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.perf_database import get_database_view

    model = _build_model()
    specs = [json.loads(op._spec_json()) for op in model.context_ops]
    attention = next(
        child["Dsv41Attention"]
        for stage in specs
        if "Dsv41Stage" in stage
        for child in stage["Dsv41Stage"]["children"]
        if "Dsv41Attention" in child
    )
    op = native.op_from_spec_json(json.dumps({"Dsv41Attention": attention | {"role": "ful", "is_context": is_context}}))
    db = get_database_view("gb300", "sglang", "current", allow_missing_data=True, database_mode=database_mode)
    with pytest.raises(ValueError, match="attention role must be"):
        _evaluate_single_op(db, op, is_context=is_context, batch_size=batch, s=seq, prefix=0, x=batch * seq)
