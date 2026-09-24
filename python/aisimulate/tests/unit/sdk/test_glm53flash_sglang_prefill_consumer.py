# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY complete native SG model tables; no GPU measurement/admission claim."""

import copy
import hashlib
import json
import shutil
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit
SHA = "a" * 64
MODEL = "zai-org/GLM-5.3-Flash"
BASENAME = "glm53flash_sglang_prefill_perf.parquet"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def policy(checkpoint="fp8", tp=2):
    return dict(
        schema_version=4,
        backend="sglang",
        backend_version="0.5.20",
        backend_revision="94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
        checkpoint_format=checkpoint,
        checkpoint_revision=(
            "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
            if checkpoint == "fp8"
            else "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
        ),
        config_sha256=SHA,
        tp_size=tp,
        runtime_digest="sha256:" + SHA,
        source_sha256="401b762a863931720b2b5cdc7b64246fac11cb215dbf6ea0fd19f3db24ce7e49",
        source_pins={
            "srt/models/glm5_next.py": "12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2",
            "srt/utils/common.py": "52eedc9338c5d2565434d265858b5d47bcc67156d38353a5b218e91df7620e45",
            "srt/managers/mm_utils.py": "a5e34a1af72faadf610feb7bf20ae41f3a50b2bbb821e38e6b091e43e4ad46c2",
        },
        timing_boundary="embedding_to_logits_gpu_v1",
        execution_policy_sha256=SHA,
        prefill_backend="disabled",
        decode_backend="full",
        native_model_contract=dict(
            model_class="sglang.srt.models.glm5_next.Glm5NextForConditionalGeneration",
            language_model_class="sglang.srt.models.glm5_next.Glm5NextModel",
            start_layer=0,
            end_layer=45,
            pp_size=1,
            dp_size=1,
            ep_size=1,
            text_only=True,
            can_run_tbo=False,
            dflash_capture=False,
            layers_to_capture=[],
            capture_aux_hidden_states=False,
            input_embeds_buffer=False,
            gemm_output_zero_allocator_size=0,
            bump_allocator_calls=1,
            bump_allocator_elements=90,
            bump_allocator_dtype="torch.float32",
        ),
    )


@pytest.fixture
def prefill(tmp_path, request):
    checkpoint, tp = getattr(request, "param", ("fp8", 2))
    model_path = MODEL if checkpoint == "fp8" else "nvidia/GLM-5.3-Flash-NVFP4"
    systems = tmp_path / "systems"
    data = systems / "data/gb300/sglang/0.5.20"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    model = get_model(model_path, ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1), "sglang")
    p = policy(checkpoint, tp)
    components = dict(
        Glm53Attention="attention", Glm53Mhc="mhc", Glm53Ffn="ffn", Glm53Primitive="primitive", Glm53Runtime="runtime"
    )
    rows = []
    # Complete exact geometries; distinct P118/P126 are intentionally not vLLM aligned.
    for b, q, past in [(1, 32, 118), (1, 32, 126), (2, 32, 118), (1, 32, 131040), (1, 32, 0)]:
        assert len(model.context_ops) == 367
        for op in model.context_ops:
            kind, shape = next(iter(json.loads(op._spec_json()).items()))
            name = shape.pop("name")
            shape.pop("children", None)
            rows.append(
                dict(
                    component=components[kind],
                    operation_name=name,
                    geometry=canonical(shape),
                    phase="context",
                    runtime_mode="NONE",
                    batch_size=b,
                    query_length=q,
                    prefix=past,
                    physical_num_tokens=b * q,
                    physical_num_requests=b,
                    latency=0.002 if kind == "Glm53Runtime" else 0.001,
                    contribution_count=1,
                    activity_count=1,
                    sample_count=10,
                    dispatch_fingerprint=SHA,
                    measurement_method="native_sglang_prefill_events_v1",
                    prefill_policy=canonical(p),
                    prefill_policy_sha256=digest(p),
                    dataset_role="calibration",
                    aggregation_policy="whole_forward_slowest_rank_v1",
                    rank_selection_sha256=SHA,
                    evidence_sha256=SHA,
                    policy_evidence_sha256=SHA,
                    measurement_scope="native_sglang_prefill_units_v1",
                )
            )
    config = ForwardPassPerfModelConfig(
        model=model_path,
        system="gb300",
        backend="sglang",
        backend_version="0.5.20",
        worker_type="aggregated",
        tp=tp,
        pp=1,
        attention_dp=1,
        moe_tp_size=tp,
        moe_ep_size=1,
        kvcache_quant_mode="fp8",
        estimation_mode="op_level",
        database_mode="SILICON",
        systems_paths=(str(systems),),
        fallback_policy="deny",
        strict_provenance=True,
        enable_shared_layer=False,
    )
    return dict(rows=rows, path=data / BASENAME, config=config, model=model, systems=systems)


def build(case):
    pq.write_table(pa.Table.from_pylist(case["rows"]), case["path"])
    return RustForwardPassPerfModel.best_available(case["config"])


def handle(case, spec=None):
    import aisimulate_core
    from aisimulate_core.sdk.engine import EngineHandle, build_engine_spec_json

    if spec is None:
        spec = json.loads(
            build_engine_spec_json(
                case["model"],
                model_path=case["config"].model,
                system="gb300",
                backend="sglang",
                backend_version="0.5.20",
                kv_block_size=None,
                systems_path=str(case["systems"]),
                nextn=0,
                database_mode="SILICON",
                shared_layer=False,
                strict_provenance=True,
            )
        )
    return EngineHandle(
        aisimulate_core.engine_spec_bincode_from_json(json.dumps(spec)), systems_path=str(case["systems"])
    ), spec


def metrics(b, q, p):
    return dict(
        version=1,
        wall_time=1.0,
        scheduled_requests=dict(
            num_prefill_requests=b,
            num_decode_requests=0,
            sum_prefill_tokens=b * q,
            sum_prefill_kv_tokens=b * p,
            sum_decode_kv_tokens=0,
            var_prefill_length=0.0,
            var_decode_kv_tokens=0.0,
        ),
    )


@pytest.mark.parametrize("prefill", [("fp8", 2), ("fp8", 4), ("nvfp4", 2), ("nvfp4", 4)], indirect=True)
def test_public_model_adds_all366_units_and_actual_setup_once(prefill):
    predictor = build(prefill)
    # Hand derived: 366 independently named units * .001ms + .002ms setup.
    assert predictor.estimate_forward_pass_time_ms(metrics(1, 32, 118)) == pytest.approx(0.368)
    engine, _ = handle(prefill)
    for b, p in [(1, 118), (1, 126), (2, 118), (1, 131040)]:
        assert engine.predict_prefill_latency(b, p + 32, p) == pytest.approx(0.368)
    with pytest.raises(ValueError, match="homogeneous dispatch"):
        predictor.estimate_forward_pass_time_ms(metrics(2, 32, 118))


@pytest.mark.parametrize("q,p", [(32, 122), (31, 118), (33, 131040), (0, 118)])
def test_missing_exact_and_context_overflow_never_interpolate_or_fallback(prefill, q, p):
    build(prefill)
    engine, _ = handle(prefill)
    with pytest.raises(ValueError):
        engine.predict_prefill_latency(1, p + q, p)


@pytest.mark.parametrize(
    "defect",
    [
        "setup",
        "unit",
        "duplicate",
        "name",
        "zero_without_activity_proof",
        "fingerprint",
        "count",
        "setup_count",
        "sample",
        "policy",
        "mixed_policy",
        "evidence",
        "physical_tokens",
        "holdout",
    ],
)
def test_selected_malformed_table_rejects_before_prediction(prefill, defect):
    rows = prefill["rows"]
    if defect in {"setup", "unit"}:
        rows.pop(366 if defect == "setup" else 0)
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "name":
        rows[0]["operation_name"] = "wrong_embedding"
    elif defect == "zero_without_activity_proof":
        rows[0]["latency"] = 0.0
    elif defect == "fingerprint":
        rows[0]["dispatch_fingerprint"] = ""
    elif defect == "count":
        rows[0]["contribution_count"] = 0
    elif defect == "setup_count":
        rows[366]["contribution_count"] = 2
    elif defect == "sample":
        rows[0]["sample_count"] = 9
    elif defect in {"policy", "mixed_policy"}:
        p = json.loads(rows[0]["prefill_policy"])
        if defect == "policy":
            p["native_model_contract"]["input_embeds_buffer"] = True
        else:
            p["execution_policy_sha256"] = "b" * 64
        rows[0].update(prefill_policy=canonical(p), prefill_policy_sha256=digest(p))
    elif defect == "evidence":
        rows[0]["policy_evidence_sha256"] = "b" * 64
    elif defect == "physical_tokens":
        rows[0]["physical_num_tokens"] += 1
    elif defect == "holdout":
        rows[0]["dataset_role"] = "holdout"
    with pytest.raises(ValueError):
        build(prefill)


def test_completed_empty_unit_and_disjoint_multiple_calls_are_explicit(prefill):
    prefill["rows"][0].update(latency=0.0, activity_count=0, dispatch_fingerprint="")
    prefill["rows"][4]["contribution_count"] = 2  # two actually completed disjoint physical intervals
    predictor = build(prefill)
    assert predictor.estimate_forward_pass_time_ms(metrics(1, 32, 118)) == pytest.approx(0.367)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_setup",
        "duplicate_setup",
        "wrong_setup",
        "erased_context",
        "erased_generation",
        "erased_both",
        "nested",
        "wrong_geometry",
    ],
)
def test_old_or_modified_compiled_specs_cannot_drop_setup_or_phase(prefill, defect):
    build(prefill)
    _, spec = handle(prefill)
    ops = spec["context_ops"]
    if defect == "missing_setup":
        ops.pop()
    elif defect == "duplicate_setup":
        ops.append(copy.deepcopy(ops[-1]))
    elif defect == "wrong_setup":
        ops[-1]["Glm53Runtime"]["tp_size"] = 4
    elif defect == "erased_context":
        spec["context_ops"] = []
    elif defect == "erased_generation":
        spec["generation_ops"] = []
    elif defect == "erased_both":
        spec["context_ops"] = spec["generation_ops"] = []
    elif defect == "nested":
        ops[0] = {"TokenScale": {"op": ops[0], "token_scale": 1.0}}
    elif defect == "wrong_geometry":
        ops[0]["Glm53Primitive"]["token_selection"] = "last_per_request"
    with pytest.raises(ValueError):
        handle(prefill, spec)


def test_context_numtokens_and_phase_cannot_be_silently_reinterpreted(prefill):
    build(prefill)
    engine, spec = handle(prefill)
    # Explicit full RuntimeContext remains the per-unit public diagnostic API.
    encoded = json.dumps([spec["context_ops"][-1]])
    values = engine.evaluate_ops_json(encoded, is_context=True, batch_size=2, s=32, prefix=118)
    assert values[0][1] == pytest.approx(0.002)
    for params in [dict(x=32), dict(imbalance_correction_scale=2.0)]:
        with pytest.raises(ValueError):
            engine.evaluate_ops_json(encoded, is_context=True, batch_size=2, s=32, prefix=118, **params)


def test_sol_probes_and_whole_forward_fpm_keep_their_contract(prefill):
    build(prefill)
    _, spec = handle(prefill)
    sol = copy.deepcopy(spec)
    sol["engine"]["database_mode"] = "SOL"
    engine, _ = handle(prefill, sol)
    assert engine.predict_prefill_latency(1, 150, 118) > 0
    probe = copy.deepcopy(spec)
    probe["engine"]["model_name"] = "__database_probe__"
    probe["context_ops"] = probe["generation_ops"] = []
    engine, _ = handle(prefill, probe)
    assert engine.predict_prefill_latency(1, 32) == 0
    case = dict(prefill)
    case["model"] = get_model(
        MODEL, ModelConfig(tp_size=2, moe_tp_size=2, moe_ep_size=1, forward_model="fpm"), "sglang"
    )
    _, spec = handle(case)
    assert spec["engine"]["forward_model"] == "fpm"
    assert all("FpmForward" in spec[phase + "_ops"][0] for phase in ("context", "generation"))


def test_schema4_context_coexists_with_unchanged_schema1_decode(prefill):
    from collector.glm53flash_graph_policy import DIRECT_FLAGS, EXTRA_FLAGS, SOURCE_PINS

    p = policy()
    graph_policy = {
        k: p[k]
        for k in (
            "backend",
            "backend_version",
            "backend_revision",
            "checkpoint_format",
            "checkpoint_revision",
            "tp_size",
            "source_sha256",
            "config_sha256",
            "runtime_digest",
        )
    }
    graph_policy.update(
        schema_version=1,
        phase="generation",
        runtime_mode="FULL",
        capture_sizes=[1, 2, 4],
        disable_padding=False,
        captured_req_width=1,
        native_flags=dict.fromkeys(DIRECT_FLAGS + EXTRA_FLAGS, False),
        source_pins=SOURCE_PINS,
        resolved_config_sha256="b" * 64,
        native_policy_receipt_sha256=SHA,
        state_layout_sha256={str(i): SHA for i in range(2)},
        capture_registry_sha256={str(i): [SHA] * 3 for i in range(2)},
    )
    # Legacy raw-config SHA intentionally differs from normalized schema4 hash.
    # Legacy graph schema deduplicates repeated unit geometries by design.
    unique = {}
    components = dict(
        Glm53Attention="attention", Glm53Mhc="mhc", Glm53Ffn="ffn", Glm53Primitive="primitive", Glm53Runtime="runtime"
    )
    for op in prefill["model"].generation_ops:
        kind, shape = next(iter(json.loads(op._spec_json()).items()))
        shape.pop("name")
        shape.pop("children", None)
        geometry = canonical(shape)
        component = components[kind]
        unique[component, geometry] = dict(
            component=component,
            geometry=geometry,
            batch_size=1,
            prefix=150,
            padded_batch_size=1,
            latency=0.002 if kind == "Glm53Runtime" else 0.001,
            activity_count=1,
            sample_count=10,
            dispatch_fingerprint=SHA,
            graph_policy=canonical(graph_policy),
            graph_policy_sha256=digest(graph_policy),
            dataset_role="calibration",
            aggregation_policy="whole_forward_slowest_rank_v1",
            rank_selection_sha256=SHA,
            evidence_sha256=SHA,
            measurement_scope="disjoint_native_node_activity_union_v1",
        )
    pq.write_table(
        pa.Table.from_pylist(list(unique.values())), prefill["path"].with_name("glm53flash_graph_perf.parquet")
    )
    predictor = build(prefill)
    assert predictor.estimate_forward_pass_time_ms(metrics(1, 32, 118)) == pytest.approx(0.368)
    engine, _ = handle(prefill)
    assert engine.predict_decode_latency(1, 150, 2) == pytest.approx(0.368)

    spec = handle(prefill)[1]
    spec["generation_ops"] = []
    with pytest.raises(ValueError, match="complete 366"):
        handle(prefill, spec)
