# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY schema3 tables over the actual 277-unit model, no GPU data claims."""

import copy
import hashlib
import json
import shutil
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"
BASENAME = "glm53flash_graph_perf.parquet"
SHA = "a" * 64


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def policy():
    # Independently authored exact native snapshot: four homogeneous requests,
    # FULL/PW token buckets 1/2/4. These are test coordinates, not observations.
    pins = {
        "v1/worker/gpu/cudagraph_utils.py": "6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff",
        "v1/worker/gpu/model_runner.py": "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
        "config/compilation.py": "c9cec5c7200e8e559810ec8c30113ad61dab6780f9f7fb3c116bd0d9b5a43065",
        "compilation/breakable_cudagraph.py": "3cc427612a08e2b9b3fee47548026400c1d0776e2d4747535e59ef5512bdf1e8",
    }
    full = [
        dict(
            cg_mode="FULL",
            num_tokens=n,
            num_reqs=n,
            uniform_token_count=1,
            max_query_len=None,
            num_active_loras=0,
            num_ubatches=1,
        )
        for n in (1, 2, 4)
    ]
    pw = [dict(row, cg_mode="PIECEWISE", num_reqs=None, uniform_token_count=None) for row in full]
    revision = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
    snapshot = dict(
        backend="vllm",
        backend_version="0.30.0",
        backend_revision=revision,
        source_pins=pins,
        native_flags=dict.fromkeys(
            (
                "compiled_model",
                "varlen_decode",
                "microbatch_runner",
                "speculative",
                "lora",
                "encoder_decoder",
                "async_scheduling",
                "expert_parallel",
                "prefix_caching",
                "kda_recoverssm",
            ),
            False,
        ),
        capture_sizes=[1, 2, 4],
        max_num_reqs=4,
        max_capture_tokens=4,
        decode_query_len=1,
        graphs_captured=True,
        lora_capture_cases=[0],
        dp_size=1,
        tp_size=2,
        resolved_mode="FULL_AND_PIECEWISE",
        use_breakable_cg=True,
        capture_descriptors=dict(FULL=full[::-1], PIECEWISE=pw[::-1]),
        full_graphs=full,
        candidates=[
            dict(num_tokens=n, num_active_loras=0, descriptors=[full[i], pw[i]])
            for n, i in ((0, 0), (1, 0), (2, 1), (3, 2), (4, 2))
        ],
        piecewise_entries=[
            dict(
                num_tokens=n,
                num_reqs=None,
                uniform=False,
                has_lora=False,
                num_active_loras=0,
                completed=True,
                num_graphs=46,
                num_eager_breaks=45,
            )
            for n in (1, 2, 4)
        ],
    )
    return dict(
        schema_version=3,
        backend="vllm",
        backend_version="0.30.0",
        backend_revision=revision,
        checkpoint_format="fp8",
        checkpoint_revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        tp_size=2,
        config_sha256=SHA,
        execution_policy_sha256=SHA,
        native_policy=snapshot,
        native_policy_sha256=digest(snapshot),
        runtime_digest="sha256:" + SHA,
        source_pins=pins,
        source_sha256="46cb601e49c399143db029d3cce33c2ee5216b8cdb6bf62385820b25fc67cba8",
        timing_boundary="native_metadata_to_logits_gpu_v1",
    )


@pytest.fixture
def serving(tmp_path, request):
    checkpoint, tp = getattr(request, "param", ("fp8", 2))
    model_path = MODEL if checkpoint == "fp8" else "nvidia/GLM-5.3-Flash-NVFP4"
    systems = tmp_path / "systems"
    data = systems / "data/gb300/vllm/0.30.0"
    data.mkdir(parents=True)
    shutil.copyfile(str(files("aisimulate_core.systems") / "gb300.yaml"), systems / "gb300.yaml")
    model = get_model(model_path, ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1), "vllm")
    p = policy()
    p["tp_size"] = p["native_policy"]["tp_size"] = tp
    p["native_policy_sha256"] = digest(p["native_policy"])
    p["checkpoint_format"] = checkpoint
    if checkpoint == "nvfp4":
        p["checkpoint_revision"] = "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
    components = dict(
        Glm53Attention="attention", Glm53Mhc="mhc", Glm53Ffn="ffn", Glm53Primitive="primitive", Glm53Runtime="runtime"
    )
    # phase,B,Q,P,mode,padded_tokens,padded_requests,multiplier,fingerprint
    points = [
        ("context", 1, 8, 0, "NONE", 8, 1, 1, SHA),
        ("context", 1, 8, 4, "NONE", 8, 1, 2, SHA),
        ("context", 1, 8, 12, "NONE", 8, 1, 6, SHA),
        ("context", 1, 9, 4, "NONE", 9, 1, 2, ""),
        ("context", 1, 9, 12, "NONE", 9, 1, 6, ""),
        ("context", 2, 2, 0, "PIECEWISE", 4, 2, 2, SHA),
        ("context", 3, 1, 0, "PIECEWISE", 4, 3, 2, SHA),
        ("generation", 1, 1, 8, "FULL", 1, 1, 3, SHA),
        ("generation", 3, 1, 8, "FULL", 4, 4, 3, SHA),
        ("generation", 1, 1, 131071, "FULL", 1, 1, 3, SHA),
    ]
    rows = []
    for phase, b, q, past, mode, tokens, requests, multiple, fingerprint in points:
        ops = model.context_ops if phase == "context" else model.generation_ops
        assert len(ops) == 278
        for op in ops:
            kind, shape = next(iter(json.loads(op._spec_json()).items()))
            name = shape.pop("name")
            shape.pop("children", None)
            setup = kind == "Glm53Runtime"
            method = (
                "native_cupti_unit_union_v1"
                if mode != "NONE"
                else ("native_runtime_cuda_events_v1" if setup else "native_module_cuda_events_v1")
            )
            rows.append(
                dict(
                    component=components[kind],
                    operation_name=name,
                    geometry=canonical(shape),
                    phase=phase,
                    runtime_mode=mode,
                    batch_size=b,
                    query_length=q,
                    prefix=past,
                    physical_num_tokens=tokens,
                    physical_num_requests=requests,
                    latency=float((10 if setup else 1) * multiple),
                    contribution_count=2 if mode == "NONE" and setup else 1,
                    sample_count=10,
                    dispatch_fingerprint=fingerprint,
                    measurement_method=method,
                    graph_policy=canonical(p),
                    graph_policy_sha256=digest(p),
                    dataset_role="calibration",
                    aggregation_policy="whole_forward_slowest_rank_v1",
                    rank_selection_sha256=SHA,
                    evidence_sha256=SHA,
                    policy_evidence_sha256=SHA,
                    measurement_scope="native_vllm_serving_units_v1",
                )
            )
    config = ForwardPassPerfModelConfig(
        model=model_path,
        system="gb300",
        backend="vllm",
        backend_version="0.30.0",
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
    path = data / BASENAME
    return dict(rows=rows, path=path, config=config, model=model, systems=systems)


def build(case):
    pq.write_table(pa.Table.from_pylist(case["rows"]), case["path"])
    return RustForwardPassPerfModel.best_available(case["config"])


def metrics(context, b, q, past):
    return dict(
        version=1,
        wall_time=1.0,
        scheduled_requests=dict(
            num_prefill_requests=b if context else 0,
            num_decode_requests=0 if context else b,
            sum_prefill_tokens=b * q if context else 0,
            sum_prefill_kv_tokens=b * past if context else 0,
            sum_decode_kv_tokens=0 if context else b * past,
            var_prefill_length=0.0,
            var_decode_kv_tokens=0.0,
        ),
    )


def handle(case, spec=None):
    import aisimulate_core
    from aisimulate_core.sdk.engine import EngineHandle, build_engine_spec_json

    if spec is None:
        spec = json.loads(
            build_engine_spec_json(
                case["model"],
                model_path=case["config"].model,
                system="gb300",
                backend="vllm",
                backend_version="0.30.0",
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


def test_canonical_best_available_composes_original_names_and_setup_once(serving):
    predictor = build(serving)
    assert predictor.estimate_forward_pass_time_ms(metrics(True, 1, 8, 0)) == 287.0
    assert predictor.estimate_forward_pass_time_ms(metrics(True, 1, 8, 8)) == 1148.0
    assert predictor.estimate_forward_pass_time_ms(metrics(False, 1, 1, 8)) == 861.0
    assert predictor.estimate_forward_pass_time_ms(metrics(False, 1, 1, 131071)) == 861.0
    engine, _ = handle(serving)
    assert engine.predict_prefill_latency(2, 2) == 574.0  # logits receives RuntimeContext B*Q=4
    assert engine.predict_prefill_latency(3, 1) == 574.0  # contextQ1 selects PW, never FULL
    assert engine.predict_decode_latency(3, 8, 2) == 861.0
    for observed in [metrics(True, 2, 2, 0), metrics(False, 3, 1, 8)]:
        with pytest.raises(ValueError, match="homogeneous dispatch"):
            predictor.estimate_forward_pass_time_ms(observed)


def test_missing_region_empty_fingerprint_and_bounds_never_fallback(serving):
    predictor = build(serving)
    for observed in [
        metrics(True, 1, 9, 8),
        metrics(True, 1, 7, 0),
        metrics(True, 1, 8, 16),
        metrics(False, 1, 1, 131072),
        metrics(False, 1, 1, 0),
    ]:
        with pytest.raises(ValueError):
            predictor.estimate_forward_pass_time_ms(observed)
    assert predictor.estimate_forward_pass_time_ms(metrics(True, 1, 9, 4)) == 574.0


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "wrong_name",
        "bad_padding",
        "holdout",
        "bad_method",
        "zero_none",
        "wrong_evidence",
        "unadmitted",
    ],
)
def test_malformed_serving_rows_cannot_be_admitted(serving, defect):
    rows = serving["rows"]
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "wrong_name":
        original_name = rows[0]["operation_name"]
        for row in rows:
            if row["operation_name"] == original_name:
                row["operation_name"] = "TEST_ONLY_UNKNOWN_NATIVE_NAME"
    elif defect == "bad_padding":
        rows[0]["physical_num_tokens"] += 1
    elif defect == "holdout":
        rows[0]["dataset_role"] = "holdout"
    elif defect == "bad_method":
        rows[0]["measurement_method"] = "native_cupti_unit_union_v1"
    elif defect == "zero_none":
        rows[0]["latency"] = 0.0
    elif defect == "wrong_evidence":
        rows[0]["policy_evidence_sha256"] = "b" * 64
    else:
        for row in rows:
            p = json.loads(row["graph_policy"])
            p["backend_version"] = "0.30.0+glm53tail.eb4704514fdf"
            row["graph_policy"], row["graph_policy_sha256"] = canonical(p), digest(p)
    with pytest.raises(ValueError):
        build(serving)


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "mismatch"])
def test_current_bincode_spec_requires_exact_one_matching_setup(serving, phase, defect):
    build(serving)
    _, original = handle(serving)
    spec = copy.deepcopy(original)
    ops = spec[phase + "_ops"]
    marker = next(op for op in ops if "Glm53Runtime" in op)
    if defect == "missing":
        ops.remove(marker)
    elif defect == "duplicate":
        ops.append(copy.deepcopy(marker))
    else:
        marker["Glm53Runtime"]["tp_size"] = 4
    with pytest.raises(ValueError):
        handle(serving, spec)


def test_policy_may_describe_unmeasured_phase_but_queries_fail(serving):
    serving["rows"] = [row for row in serving["rows"] if row["phase"] == "generation"]
    predictor = build(serving)
    assert predictor.estimate_forward_pass_time_ms(metrics(False, 1, 1, 8)) == 861.0
    with pytest.raises(ValueError, match="phase or exact native operation is unmeasured"):
        predictor.estimate_forward_pass_time_ms(metrics(True, 1, 8, 0))


@pytest.mark.parametrize("same_identity", [True, False])
def test_legacy_schema2_coexists_only_under_a_different_checkpoint_tp(serving, same_identity):
    native = policy()
    tp = 2 if same_identity else 4
    legacy = {
        k: native[k]
        for k in (
            "backend",
            "backend_version",
            "backend_revision",
            "checkpoint_format",
            "checkpoint_revision",
            "config_sha256",
            "runtime_digest",
            "source_pins",
            "source_sha256",
        )
    }
    legacy.update(
        schema_version=2,
        tp_size=tp,
        phase="generation",
        runtime_mode="FULL",
        capture_sizes=[1, 2, 4],
        disable_padding=False,
        captured_req_width=1,
        native_flags=native["native_policy"]["native_flags"],
        resolved_config_sha256=SHA,
        native_policy_receipt_sha256=SHA,
        state_layout_sha256={str(rank): SHA for rank in range(tp)},
        capture_registry_sha256={str(rank): [SHA] * 3 for rank in range(tp)},
    )
    for row in serving["rows"]:
        row["padded_batch_size"], row["activity_count"] = None, None
    for component in ("runtime", "mhc"):
        original = next(
            row for row in serving["rows"] if row["phase"] == "generation" and row["component"] == component
        )
        row = dict.fromkeys(original)
        shape = json.loads(original["geometry"])
        shape["tp_size"] = tp
        row.update(
            component=component,
            geometry=canonical(shape),
            batch_size=1,
            prefix=8,
            padded_batch_size=1,
            latency=1.0,
            activity_count=1,
            sample_count=10,
            dispatch_fingerprint=SHA,
            graph_policy=canonical(legacy),
            graph_policy_sha256=digest(legacy),
            dataset_role="calibration",
            aggregation_policy="whole_forward_slowest_rank_v1",
            rank_selection_sha256=SHA,
            evidence_sha256=SHA,
            measurement_scope="disjoint_native_node_activity_union_v1",
        )
        serving["rows"].append(row)
    if same_identity:
        with pytest.raises(ValueError, match="compete for the same checkpoint/TP"):
            build(serving)
    else:
        predictor = build(serving)
        assert predictor.estimate_forward_pass_time_ms(metrics(True, 1, 8, 0)) == 287.0
        assert predictor.estimate_forward_pass_time_ms(metrics(False, 1, 1, 8)) == 861.0


def test_explicit_empty_graph_unit_is_zero_but_missing_setup_is_never_zero(serving):
    # One witnessed-empty PW unit can be zero. It remains an explicit named row;
    # removing any unit or setup already fails the complete-point tests above.
    row = next(
        row
        for row in serving["rows"]
        if row["runtime_mode"] == "PIECEWISE" and row["batch_size"] == 2 and row["component"] == "mhc"
    )
    row["latency"], row["contribution_count"] = 0.0, 0
    build(serving)
    engine, _ = handle(serving)
    assert engine.predict_prefill_latency(2, 2) == 572.0  # 574 minus this original 2ms unit


@pytest.mark.parametrize("serving", [("fp8", 2), ("fp8", 4), ("nvfp4", 2), ("nvfp4", 4)], indirect=True)
def test_actual_full_model_geometry_for_each_vllm_deployment(serving):
    predictor = build(serving)
    assert predictor.estimate_forward_pass_time_ms(metrics(True, 1, 8, 0)) == 287.0
    assert predictor.estimate_forward_pass_time_ms(metrics(False, 1, 1, 8)) == 861.0


@pytest.mark.parametrize("phase", ["context", "generation", "both"])
@pytest.mark.parametrize("serving", [("fp8", 2), ("nvfp4", 4)], indirect=True)
def test_current_spec_cannot_erase_complete_serving_phases(serving, phase):
    build(serving)
    _, spec = handle(serving)
    for name in ("context", "generation"):
        if phase in (name, "both"):
            spec[name + "_ops"] = []
    with pytest.raises(ValueError, match="both complete context and generation"):
        handle(serving, spec)


def test_phase_guard_preserves_sol_empty_probes_and_other_model_identities(serving):
    build(serving)
    _, spec = handle(serving)
    sol = copy.deepcopy(spec)
    sol["engine"]["database_mode"] = "SOL"
    engine, _ = handle(serving, sol)
    assert engine.predict_prefill_latency(1, 8) > 0
    assert engine.predict_decode_latency(1, 8, 2) > 0
    for name in ("__database_probe__", "TEST_ONLY_OTHER_MODEL"):
        probe = copy.deepcopy(spec)
        probe["engine"]["model_name"] = name
        probe["context_ops"] = probe["generation_ops"] = []
        engine, _ = handle(serving, probe)
        assert engine.predict_prefill_latency(1, 8) == 0
        assert engine.predict_decode_latency(1, 8, 2) == 0
    sol["context_ops"] = sol["generation_ops"] = []
    handle(serving, sol)  # SOL preserves its historical empty-spec contract.


def test_serving_phase_guard_preserves_whole_forward_fpm_rewrite(serving):
    build(serving)
    case = dict(serving)
    case["model"] = get_model(MODEL, ModelConfig(tp_size=2, moe_tp_size=2, moe_ep_size=1, forward_model="fpm"), "vllm")
    engine, spec = handle(case)
    assert spec["engine"]["forward_model"] == "fpm"
    assert all(
        len(spec[phase + "_ops"]) == 1 and "FpmForward" in spec[phase + "_ops"][0]
        for phase in ("context", "generation")
    )
    # No FPM data exists in this TEST_ONLY database. Construction is allowed,
    # but a graph serving table cannot substitute a whole-forward measurement.
    with pytest.raises(PerfDataNotAvailableError, match="No fpm_forward data collected"):
        engine.predict_prefill_latency(1, 8)


def test_empty_legacy_profile_without_schema3_retains_existing_contract(serving):
    # This isolated TEST_ONLY database never contained a serving policy. Avoid
    # mutating a loaded table: database caching correctly preserves its identity.
    assert not serving["path"].exists()
    _, spec = handle(serving)
    spec["context_ops"] = spec["generation_ops"] = []
    engine, _ = handle(serving, spec)
    assert engine.predict_prefill_latency(1, 8) == 0
    assert engine.predict_decode_latency(1, 8, 2) == 0
