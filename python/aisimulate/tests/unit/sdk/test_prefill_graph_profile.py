# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact direct-prefill contract, using the qualified public bundle.

Latency oracles are the separately reviewed frozen graph-prediction-v5 ledger,
not recomputed by Python. Native full-forward timings never enter prediction.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

import aisimulate_core._native as core
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.engine import EngineHandle, build_engine_spec_json, build_ops_json
from aisimulate_core.sdk.errors import PrefillGraphProfileError
from aisimulate_core.sdk.models import get_model

pytestmark = pytest.mark.unit
VERSION = "0.5.18+nvinternal.rubin.0.8full.66997102"
PROFILE = "sglang_glm52_nvfp4_vr200_tp4_graph_v1"
PUBLIC_CALLS = [
    (1, 1024, 0),
    (2, 1024, 0),
    (1, 2048, 1024),
    (1, 8192, 0),
    (2, 8192, 0),
    (1, 16384, 0),
    (1, 32768, 16384),
]
# Frozen independent v5 comparison; each cost is the unchanged accepted formula.
# Filled from that ledger, with no native-forward fitting.
PREDICTED_MS = [
    34.90489051212317,
    48.38665649293463,
    37.859508044217485,
    196.09922835363085,
    364.91345761325437,
    378.53291416754246,
    442.6694767692677,
]


def systems_root():
    import aisimulate_core

    return Path(os.environ.get("AISIMULATE_PREFILL_GRAPH_SYSTEMS", Path(aisimulate_core.__file__).parent / "systems"))


def model(selected=True, **overrides):
    kwargs = dict(
        tp_size=4,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=4,
        moe_ep_size=1,
        gemm_quant_mode="bfloat16",
        moe_quant_mode="nvfp4",
        kvcache_quant_mode="fp8",
        fmha_quant_mode="bfloat16",
        comm_quant_mode="half",
        prefill_graph_profile=PROFILE if selected else None,
    )
    kwargs.update(overrides)
    return get_model("nvidia/GLM-5.2-NVFP4", build_model_config(**kwargs), "sglang")


def spec_json(selected=True, root=None):
    return build_engine_spec_json(
        model(selected),
        model_path="nvidia/GLM-5.2-NVFP4",
        system="vr200_hecate",
        backend="sglang",
        backend_version=VERSION,
        kv_block_size=None,
        systems_path=str(root or systems_root()),
        nextn=0,
        database_mode="SILICON",
        shared_layer=False,
    )


def handle(selected=True, root=None):
    return EngineHandle(core.engine_spec_bincode_from_json(spec_json(selected, root)))


def copy_bundle(tmp_path):
    import shutil

    root = systems_root()
    sidecar = root / "data/vr200_hecate/sparse_attention/sglang" / VERSION / f"{PROFILE}.profile.json"
    profile = json.loads(sidecar.read_text())
    paths = {item["path"] for item in profile["retained_files"]}
    for table in profile["tables"].values():
        path = Path(table["relative_path"])
        paths.update((str(path), str(path.with_name(sidecar.name))))
        metadata = path.with_name("collection_meta.yaml")
        if (root / metadata).exists():
            paths.add(str(metadata))
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    return tmp_path


def test_all_seven_public_coordinates_match_reviewed_forward_ledger():
    engine = handle()
    assert engine.prefill_graph_profile["profile_name"] == PROFILE
    assert len(PREDICTED_MS) == 7
    for call, expected in zip(PUBLIC_CALLS, PREDICTED_MS, strict=True):
        assert engine.predict_prefill_latency(*call) == pytest.approx(expected, abs=1e-10, rel=1e-12)


def multiply_collected_latencies(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = pq.read_table(path)
    rows = data.to_pylist()
    for row in rows:
        row["latency"] *= 10
    pq.write_table(pa.Table.from_pylist(rows, schema=data.schema), path)


def test_selected_engine_ignores_stale_default_engine_tables(tmp_path):
    root = copy_bundle(tmp_path)
    gemm = next(root.rglob("gemm_perf.parquet"))
    approved = gemm.read_bytes()
    multiply_collected_latencies(gemm)
    default_engine = handle(False, root)
    # Keep the default engine alive after populating the process-wide shared tables.
    default_engine.predict_prefill_latency(*PUBLIC_CALLS[0])
    gemm.write_bytes(approved)

    selected = handle(root=root)
    for call, expected in zip(PUBLIC_CALLS, PREDICTED_MS, strict=True):
        assert selected.predict_prefill_latency(*call) == pytest.approx(expected, abs=1e-10, rel=1e-12)


@pytest.mark.parametrize(
    "filename",
    [
        "gemm_perf.parquet",
        "moe_perf.parquet",
        "custom_allreduce_perf.parquet",
        "dsa_context_module_perf.parquet",
        "dsa_generation_module_perf.parquet",
        "sglang_prefill_attention_sequence_perf.parquet",
        "sglang_prefill_comm_norm_boundary_perf.parquet",
        "vr200_hecate.yaml",
        f"{PROFILE}.profile.json",
    ],
)
def test_selected_engine_keeps_admitted_snapshot_before_first_prediction(tmp_path, filename):
    root = copy_bundle(tmp_path)
    selected = handle(root=root)
    source = next(root.rglob(filename))
    if source.suffix == ".parquet":
        multiply_collected_latencies(source)
    else:
        source.write_bytes(source.read_bytes() + b"\n")

    # Rebuilding must reject changed sources, while an admitted engine keeps its
    # verified snapshot even when the changed family has never been queried.
    with pytest.raises(PrefillGraphProfileError):
        handle(root=root)
    for call, expected in zip(PUBLIC_CALLS, PREDICTED_MS, strict=True):
        assert selected.predict_prefill_latency(*call) == pytest.approx(expected, abs=1e-10, rel=1e-12)


def test_selected_engine_snapshot_survives_source_directory_removal(tmp_path):
    import shutil

    root = copy_bundle(tmp_path / "systems")
    selected = handle(root=root)
    profile = selected.prefill_graph_profile
    shutil.rmtree(root)
    for call, expected in zip(PUBLIC_CALLS, PREDICTED_MS, strict=True):
        assert selected.predict_prefill_latency(*call) == pytest.approx(expected, abs=1e-10, rel=1e-12)
    assert selected.prefill_graph_profile == profile


def test_composition_preserves_full_model_weight_inventory_and_fixed_counts():
    baseline, selected = model(False), model(True)
    assert sum(op.get_weights() for op in baseline.context_ops) == sum(op.get_weights() for op in selected.context_ops)
    assert build_ops_json(baseline.generation_ops) == build_ops_json(selected.generation_ops)
    serialized = json.loads(build_ops_json(selected.context_ops))
    names = [next(iter(op.values()))["name"] for op in serialized]
    assert len(names) == len(set(names)) == 13
    assert names[:3] == [
        "context_attention_sequence",
        "context_post_attention_boundary",
        "context_following_mlp_boundary",
    ]
    assert not {
        "context_attention",
        "context_add_norm_1",
        "context_add_norm_2",
        "context_moe_pre_dispatch",
        "context_moe_post_dispatch",
    } & set(names)
    overlap = serialized[3]["Overlap"]
    assert [next(iter(op.values()))["name"] for op in overlap["group_a"]] == ["context_router_gemm", "context_moe"]
    assert [next(iter(op.values()))["name"] for op in overlap["group_b"]] == [
        "context_shared_gate_up_gemm",
        "context_shared_act_gate",
        "context_shared_ffn2_gemm",
    ]
    assert serialized[4]["MoeDispatch"]["scale_factor"] == 1
    assert serialized[5]["Elementwise"]["scale_factor"] == 2
    assert serialized[6]["Elementwise"]["scale_factor"] == 75
    assert serialized[6]["Elementwise"]["bytes_per_token"] == 6144 * 6


@pytest.mark.parametrize(
    "call",
    [
        (0, 1024, 0),
        (1, 1024, 1024),
        (1, 16384, 16384),
        (1, 1024, 1025),
        (1, 2048, 0),
        (2, 1536, 512),
        (4294967295, 2, 0),
    ],
)
def test_native_shape_admission_rejects_same_tokens_and_overflow(call):
    with pytest.raises(PrefillGraphProfileError):
        handle()._engine.predict_prefill_latency(*call)


@pytest.mark.parametrize("value", [True, False, -1, 2**32, 1.5, float("nan"), float("inf"), "1", None])
def test_public_input_never_coerces_selected_profile(value):
    engine = handle()
    for call in [(value, 1024, 0), (1, value, 0), (1, 1024, value)]:
        with pytest.raises(PrefillGraphProfileError):
            engine.predict_prefill_latency(*call)


def test_raw_native_binding_rejects_boolean_and_int_subclass():
    class IntSubclass(int):
        pass

    engine = handle()._engine
    for value in (True, IntSubclass(1)):
        with pytest.raises(PrefillGraphProfileError):
            engine.predict_prefill_latency(value, 1024, 0)


def test_selected_profile_rejects_all_unsupported_compute_surfaces():
    engine = handle()
    calls = [
        lambda: engine.run_static(batch_size=1, isl=1024, osl=1, mode="static_ctx"),
        lambda: engine.predict_decode_latency(1, 1024),
        lambda: engine.mixed_step_latency(1024, 0, 1024, 1),
        lambda: engine.run_static_per_op(batch_size=1, isl=1024, osl=1, mode="static_ctx"),
        lambda: engine._engine.evaluate_ops_json("[]", True, 1, 1024, 0, 1.0, None),
        lambda: engine._engine.evaluate_ops_sol_json("[]", True, 1, 1024, 0, 1.0, None),
    ]
    for call in calls:
        with pytest.raises(PrefillGraphProfileError):
            call()


def test_default_engine_cannot_query_composites_as_energy_bearing_ad_hoc_ops():
    engine = handle(False)
    op_json = build_ops_json(model(True).context_ops[:1])
    for method in (engine._engine.evaluate_ops_json, engine._engine.evaluate_ops_sol_json):
        with pytest.raises(PrefillGraphProfileError):
            method(op_json, True, 1, 1024, 0, 1.0, None)


@pytest.mark.parametrize(
    "change",
    ["selector", "id", "mode", "model", "tp", "scale", "weight", "nested", "generation", "role", "decoder_replay"],
)
def test_serialized_specs_cannot_bypass_admission(change):
    spec = json.loads(spec_json())
    if change == "selector":
        del spec["engine"]["prefill_graph_profile"]
    elif change == "id":
        spec["engine"]["prefill_graph_profile_id"] = "0" * 64
    elif change == "mode":
        spec["engine"]["database_mode"] = "HYBRID"
    elif change == "model":
        spec["engine"]["model_name"] = "other"
    elif change == "tp":
        spec["engine"]["tp_size"] = 8
    elif change == "decoder_replay":
        spec["engine"]["decoder_replay"] = True
    elif change == "scale":
        spec["context_ops"][5]["Elementwise"]["scale_factor"] = 3
    elif change == "weight":
        spec["context_ops"][0]["SglangPrefillAttentionSequence"]["weight_bytes"] = 1.0
    elif change == "nested":
        spec["context_ops"][3]["Overlap"]["group_a"].pop()
    elif change == "generation":
        spec["generation_ops"].append(copy.deepcopy(spec["context_ops"][0]))
    else:
        spec["context_ops"][1]["SglangPrefillCommNormBoundary"]["boundary_role"] = "following_mlp"
    with pytest.raises(PrefillGraphProfileError):
        EngineHandle(core.engine_spec_bincode_from_json(json.dumps(spec)))


def test_new_main_stage_cannot_hide_graph_ops_on_general_or_energy_routes():
    child = json.loads(build_ops_json(model(True).context_ops[:1]))[0]
    stage = {
        "Dsv41Stage": {
            "name": "nested_composite",
            "is_context": True,
            "decoder_replay": False,
            "bounded": False,
            "window_size": 128,
            "children": [child],
        }
    }
    spec = json.loads(spec_json(False))
    spec["context_ops"] = [stage]
    with pytest.raises(PrefillGraphProfileError, match="explicitly selected"):
        EngineHandle(core.engine_spec_bincode_from_json(json.dumps(spec)))
    engine = handle(False)._engine
    for method in (engine.evaluate_ops_json, engine.evaluate_ops_sol_json):
        with pytest.raises(PrefillGraphProfileError):
            method(json.dumps([stage]), True, 1, 1024, 0, 1.0, None)


@pytest.mark.parametrize("table", ["attention", "communication"])
@pytest.mark.parametrize("mutation", ["duplicate", "missing", "value", "null", "extra", "nullable"])
def test_new_tables_reject_incomplete_or_changed_payload(tmp_path, table, mutation):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = copy_bundle(tmp_path)
    profile = json.loads(next(root.rglob(f"{PROFILE}.profile.json")).read_text())
    path = root / profile["tables"][table]["relative_path"]
    data = pq.read_table(path)
    rows = data.to_pylist()
    schema = data.schema
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "missing":
        rows.pop()
    elif mutation == "value":
        rows[0]["latency"] *= 1.01
    elif mutation == "null":
        rows[0]["kernel_source"] = None
        schema = pa.schema([pa.field(field.name, field.type, nullable=True) for field in schema])
    elif mutation == "extra":
        schema = schema.append(pa.field("unexpected", pa.int64(), nullable=False))
        for row in rows:
            row["unexpected"] = 1
    else:
        schema = pa.schema([pa.field(field.name, field.type, nullable=True) for field in schema])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    with pytest.raises(PrefillGraphProfileError):
        handle(root=root)


@pytest.mark.parametrize("mutation", ["profile", "sidecar", "retained", "symlink"])
def test_profile_and_retained_identities_cannot_change(tmp_path, mutation):
    root = copy_bundle(tmp_path)
    profiles = list(root.rglob(f"{PROFILE}.profile.json"))
    data = json.loads(profiles[0].read_text())
    if mutation == "profile":
        for path in profiles:
            path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "sidecar":
        profiles[0].unlink()
    elif mutation == "retained":
        target = root / data["retained_files"][1]["path"]
        target.write_bytes(target.read_bytes() + b"changed")
    else:
        target = profiles[0]
        saved = target.with_suffix(".saved")
        target.rename(saved)
        target.symlink_to(saved.name)
    with pytest.raises(PrefillGraphProfileError):
        handle(root=root)


def test_previous_binary_schema_requires_recompilation():
    encoded = bytearray(core.engine_spec_bincode_from_json(spec_json()))
    encoded[:4] = (19).to_bytes(4, "little")
    with pytest.raises(ValueError, match="recompile"):
        EngineHandle(encoded)


def test_ops_json_and_pickle_preserve_new_variants():
    import pickle

    selected = model()
    for operation in selected.context_ops[:3]:
        assert pickle.loads(pickle.dumps(operation))._spec_json() == operation._spec_json()
        with pytest.raises(TypeError, match="no scale_factor"):
            operation._scale_factor = 2


def test_raw_views_preserve_all_exact_rows_without_energy():
    engine = handle(False)
    for attribute, count in [
        ("_sglang_prefill_attention_sequence_data", 7),
        ("_sglang_prefill_comm_norm_boundary_data", 8),
    ]:
        raw = json.loads(engine._engine.table_view_json(attribute))
        rows = list(raw.values()) if count == 7 else [row for roles in raw.values() for row in roles.values()]
        assert len(rows) == count
        assert all("energy" not in row and "power" not in row for row in rows)
        if count == 7:
            assert raw["2|1024|0"]["latency"] == 18.303354517618814
