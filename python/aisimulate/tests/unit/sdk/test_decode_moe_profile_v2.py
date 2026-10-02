# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite decode profile: exact occupancies, independent caches and provenance."""

import copy
import dataclasses
import json
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import aisimulate_core as core
from aisimulate_core.sdk.engine import build_ops_json, compile_engine
from aisimulate_core.sdk.errors import DecodeMoeProfileError
from aisimulate_core.sdk.models import get_model

from . import test_decode_moe_profile as v1

pytestmark = pytest.mark.unit
PROFILE = "observed_glm52_nvfp4_decode_composite_v2"
COLLECTOR_HASH = "sha256:bbe3ebf2d450053f524d38b0a8ef97f0e55df000b6c6f61430b0207afc622eaf"
SUPERSEDED_COLLECTOR_HASH = "sha256:4197366d8e3f2e3b97f0efde108557eea1d5169a893ab4cf6d041f0550e2c7b8"
CASE_PLAN_HASH = "sha256:8a1c8ca3136fcf2289ff49e7f07d1d95e3d23dea9301db0eafeb07436d529519"
# Hand-picked oracle, not hardware values: N4 is deliberately slower than N8.
VALUES = {1: 10.0, 4: 30.0, 8: 20.0, 32: 40.0}
EXPECTED = {1: 10.0, 3: 30.0, 8: 20.0, 29: 40.0, 31: 40.0, 32: 40.0}


def table_dir(root):
    return root / "data" / v1.SYSTEM / "moe/sglang" / v1.VERSION


def dataset(root, *, nodes=(1, 4, 8, 32), include_v1=True, event_change=None, row_change=None):
    v1.dataset(root, points=(1, 8, 32) if include_v1 else ())
    directory = table_dir(root)
    path = directory / "moe_perf.parquet"
    rows = pq.read_table(path).to_pylist()
    template = rows[0]
    for node in nodes:
        row = dict(template, distribution=PROFILE, num_tokens=node, latency=VALUES.get(node, 99.0))
        if row_change:
            row_change(row)
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path)
    metadata = yaml.safe_load((directory / "collection_meta.yaml").read_text())
    event = copy.deepcopy(metadata["tables"]["moe_perf"]["collections"][0])
    event.update(
        collector_ref="collector.sglang_rubin.publish_observed_moe_v2",
        collector_hash=COLLECTOR_HASH,
        case_plan_hash=CASE_PLAN_HASH,
        rows=4,
        source_campaign_rows=1152,
    )
    if event_change:
        event_change(event)
    metadata["tables"]["moe_perf"]["rows"] = len(rows)
    if not include_v1:
        metadata["tables"]["moe_perf"]["collections"] = []
    metadata["tables"]["moe_perf"]["collections"].append(event)
    (directory / "collection_meta.yaml").write_text(yaml.safe_dump(metadata))
    return root


def load(root, profile=PROFILE):
    return v1.load(v1.spec(root, selected=profile))


def test_exact_nonmonotonic_mapping_and_all_other_logical_sizes_reject(tmp_path):
    handle = load(dataset(tmp_path / "valid"))
    for logical, expected in EXPECTED.items():
        assert handle.predict_decode_latency(logical, 32768, 2) == expected
    for logical in {*range(34), 128, 2**32 - 1} - EXPECTED.keys():
        with pytest.raises(DecodeMoeProfileError, match=PROFILE):
            handle.predict_decode_latency(logical, 32768, 2)


@pytest.mark.parametrize("order", [(v1.PROFILE, PROFILE), (PROFILE, v1.PROFILE)])
def test_shared_database_keeps_both_profile_caches_distinct(tmp_path, order):
    root = dataset(tmp_path / "both")
    handles = {profile: load(root, profile) for profile in order}
    for _ in range(2):
        assert handles[v1.PROFILE].predict_decode_latency(20, 1024, 2) == 2.5
        assert handles[PROFILE].predict_decode_latency(3, 32768, 2) == 30.0
        assert handles[PROFILE].predict_decode_latency(8, 32768, 2) == 20.0


@pytest.mark.parametrize("missing", [v1.PROFILE, PROFILE])
def test_cached_missing_profile_does_not_poison_the_other(tmp_path, missing):
    root = dataset(
        tmp_path / "one-profile", nodes=() if missing == PROFILE else tuple(VALUES), include_v1=missing != v1.PROFILE
    )
    # Keep an ordinary engine alive so all loads share the same native PerfTables.
    default = v1.load(v1.spec(root, selected="uniform", exact=False))
    assert default.predict_decode_latency(1, 1024, 2) == 50.0
    with pytest.raises(DecodeMoeProfileError, match="absent"):
        load(root, missing)
    available = v1.PROFILE if missing == PROFILE else PROFILE
    handle = load(root, available)
    assert handle.predict_decode_latency(1, 1024, 2) == (1.0 if available == v1.PROFILE else 10.0)
    with pytest.raises(DecodeMoeProfileError, match="absent"):
        load(root, missing)


@pytest.mark.parametrize("nodes", [(), (1, 8, 32), (1, 4, 8), (1, 4, 8, 32, 32), (1, 4, 8, 31)])
def test_missing_duplicate_or_wrong_node_rejects_despite_v1_and_uniform(tmp_path, nodes):
    with pytest.raises(DecodeMoeProfileError):
        load(dataset(tmp_path / "bad-nodes", nodes=nodes))


@pytest.mark.parametrize(
    "field,value",
    [
        ("collector_ref", "collector.sglang_rubin.publish_observed_moe"),
        ("collector_hash", v1.COLLECTOR_HASH),
        ("collector_hash", SUPERSEDED_COLLECTOR_HASH),
        ("collector_hash", "sha256:" + "0" * 64),
        ("case_plan_hash", v1.CASE_PLAN_HASH),
        ("rows", 3),
        ("source_campaign_rows", 960),
        ("source_campaign_rows", 1151),
        ("status", "partial"),
        ("source_campaign_status", "partial"),
        ("collected_at", ""),
    ],
)
def test_exact_v2_event_cannot_borrow_v1_identity(tmp_path, field, value):
    root = dataset(tmp_path / "bad-event", event_change=lambda event: event.update({field: value}))
    with pytest.raises(DecodeMoeProfileError, match=PROFILE):
        load(root)
    assert load(root, v1.PROFILE).predict_decode_latency(1, 1024, 2) == 1.0


@pytest.mark.parametrize("change", ["absent", "duplicate", "bad-table-key", "schema", "table-status"])
def test_event_coverage_is_exact(tmp_path, change):
    root = dataset(tmp_path / "metadata")
    path = table_dir(root) / "collection_meta.yaml"
    meta = yaml.safe_load(path.read_text())
    table = meta["tables"]["moe_perf"]
    if change == "absent":
        table["collections"].pop()
    elif change == "duplicate":
        table["collections"].append(copy.deepcopy(table["collections"][-1]))
    elif change == "bad-table-key":
        meta["tables"]["moe_perf.parquet"] = meta["tables"].pop("moe_perf")
    elif change == "schema":
        meta["schema_version"] = 1
    else:
        table["status"] = "partial"
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(DecodeMoeProfileError):
        load(root)


@pytest.mark.parametrize("location", ["runtime", "event"])
@pytest.mark.parametrize("field", ["framework", "version", "image", "image_digest", "source_commit"])
def test_both_runtime_identities_remain_exact(tmp_path, location, field):
    root = dataset(tmp_path / "runtime")
    path = table_dir(root) / "collection_meta.yaml"
    meta = yaml.safe_load(path.read_text())
    target = meta["runtime"] if location == "runtime" else meta["tables"]["moe_perf"]["collections"][-1]["runtime"]
    target[field] = "wrong"
    path.write_text(yaml.safe_dump(meta))
    with pytest.raises(DecodeMoeProfileError):
        load(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("framework", "vLLM"),
        ("version", "other"),
        ("device", "other"),
        ("op_name", "other"),
        ("kernel_source", "other"),
        ("moe_dtype", "fp8"),
        ("hidden_size", 6143),
        ("inter_size", 1024),
        ("topk", 4),
        ("num_experts", 128),
        ("moe_tp_size", 2),
        ("moe_ep_size", 2),
        ("latency", 0.0),
        ("latency", -1.0),
        ("latency", float("nan")),
        ("latency", float("inf")),
    ],
)
def test_all_profile_row_fields_are_strict(tmp_path, field, value):
    with pytest.raises(DecodeMoeProfileError):
        load(dataset(tmp_path / "bad-row", row_change=lambda row: row.update({field: value})))


@pytest.mark.parametrize("kind", ["no-profile", "missing-root", "bad-yaml", "missing-node", "superseded-collector"])
def test_serialized_reload_checks_effective_root_before_uniform_fallback(tmp_path, kind):
    valid = dataset(tmp_path / "valid")
    if kind == "missing-root":
        bad = tmp_path / "absent"
    else:
        bad = dataset(
            tmp_path / "bad",
            nodes=() if kind == "no-profile" else (1, 8, 32) if kind == "missing-node" else tuple(VALUES),
        )
        if kind == "bad-yaml":
            (table_dir(bad) / "collection_meta.yaml").write_text("[broken yaml")
        elif kind == "superseded-collector":
            path = table_dir(bad) / "collection_meta.yaml"
            metadata = yaml.safe_load(path.read_text())
            metadata["tables"]["moe_perf"]["collections"][-1]["collector_hash"] = SUPERSEDED_COLLECTOR_HASH
            path.write_text(yaml.safe_dump(metadata))
    blob = bytes(core.engine_spec_bincode_from_json(json.dumps(v1.spec(selected=PROFILE))))
    assert core.AicEngine.from_spec(blob, systems_path=str(valid)).predict_decode_latency(3, 32768, 2) == 30.0
    with pytest.raises(DecodeMoeProfileError):
        core.AicEngine.from_spec(blob, systems_path=str(bad))
    with pytest.raises(DecodeMoeProfileError):
        v1.load(v1.spec(bad, selected=PROFILE), valid)
    assert v1.load(v1.spec(valid, selected=PROFILE), bad).predict_decode_latency(29, 32768, 2) == 40.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 2),
        ("attention_dp_size", 2),
        ("weight_dtype", "float16"),
        ("activation_dtype", "fp8"),
        ("kv_cache_dtype", "bfloat16"),
        ("database_mode", "HYBRID"),
        ("nextn", 1),
        ("forward_model", "fpm"),
    ],
)
def test_v2_native_scope_is_unchanged(tmp_path, field, value):
    wire = v1.spec(dataset(tmp_path / "valid"), selected=PROFILE)
    wire["engine"][field] = value
    with pytest.raises(DecodeMoeProfileError):
        v1.load(wire)


@pytest.mark.parametrize("profile", [v1.PROFILE, PROFILE])
@pytest.mark.parametrize("phase", ["context_ops", "generation_ops"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["CustomAllReduce", "Nccl", "MoeDispatch"])
@pytest.mark.parametrize("quant", ["half", "fp8", "int8"])
def test_serialized_communication_precision_is_checked_through_composites(
    tmp_path, profile, phase, nested, kind, quant
):
    wire = v1.spec(dataset(tmp_path / "valid"), selected=profile)
    if kind == "CustomAllReduce":
        fields = dict(name="communication", scale_factor=1.0, hidden_size=6144, tp_size=4, quant=quant)
    elif kind == "Nccl":
        fields = dict(
            name="communication", scale_factor=1.0, hidden_size=6144.0, num_gpus=4, dtype=quant, operation="allreduce"
        )
    else:
        # Use the model's real dispatch operator, changing only its wire precision.
        model = get_model(v1.MODEL, v1.build_model_config(**v1.KWARGS), "sglang")
        fields = next(op[kind] for op in json.loads(build_ops_json(model.context_ops)) if kind in op)
        fields["is_context"] = phase == "context_ops"
        fields["comm_quant"] = quant
    op = {kind: fields}
    if nested:
        op = {
            "Dsv41Stage": dict(
                name="nested_communication",
                is_context=phase == "context_ops",
                decoder_replay=False,
                bounded=False,
                window_size=128,
                children=[op],
            )
        }
        op = {"TokenScale": dict(op=op, numerator=1, denominator=1)}
        op = {"Fallback": dict(name="communication_fallback", primary=op, fallback=[copy.deepcopy(op)])}
        op = {"Overlap": dict(name="communication_overlap", group_a=[], group_b=[op])}
    wire[phase].append(op)
    if quant == "half":
        assert v1.load(wire) is not None
    else:
        with pytest.raises(DecodeMoeProfileError, match="requires half communication"):
            v1.load(wire)


@pytest.mark.parametrize("profile", [v1.PROFILE, PROFILE])
@pytest.mark.parametrize("location", ["engine", "operation"])
@pytest.mark.parametrize("source", [None, "sglang_flashinfer_trtllm_moe", "unavailable_kernel_source"])
def test_serialized_decode_profiles_reject_active_moe_source(tmp_path, profile, location, source):
    wire = v1.spec(dataset(tmp_path / "valid"), selected=profile)
    target = wire["engine"] if location == "engine" else wire["generation_ops"][0]["Overlap"]["group_a"][0]["Moe"]
    target["moe_kernel_source"] = source
    if source is not None:
        with pytest.raises(DecodeMoeProfileError, match="moe_kernel_source cannot override"):
            v1.load(wire)
    else:
        assert v1.load(wire).predict_decode_latency(1, 1024, 2) > 0


@pytest.mark.parametrize("profile", [v1.PROFILE, PROFILE])
@pytest.mark.parametrize("source", ["sglang_flashinfer_trtllm_moe", "unavailable_kernel_source"])
def test_direct_model_rejects_moe_source_override(profile, source):
    config = v1.build_model_config(**v1.KWARGS, decode_workload_distribution=profile, moe_kernel_source=source)
    with pytest.raises(DecodeMoeProfileError, match="moe_kernel_source cannot override"):
        get_model(v1.MODEL, config, "sglang")


def test_selector_preserves_context_and_unrelated_generation_ops():
    base = v1.build_model_config(**v1.KWARGS)
    normal = get_model(v1.MODEL, base, "sglang")
    selected = dataclasses.replace(base, decode_workload_distribution=PROFILE)
    assert pickle.loads(pickle.dumps(selected)) == selected
    changed = get_model(v1.MODEL, selected, "sglang")
    assert build_ops_json(changed.context_ops) == build_ops_json(normal.context_ops)
    prior = json.loads(
        build_ops_json(
            get_model(
                v1.MODEL, dataclasses.replace(base, decode_workload_distribution=v1.PROFILE), "sglang"
            ).generation_ops
        )
    )
    current = json.loads(build_ops_json(changed.generation_ops))
    # The v1 and v2 model graphs differ by one literal only.
    assert json.dumps(current).replace(PROFILE, v1.PROFILE) == json.dumps(prior)


def test_public_compile_preflight_and_real_timing_config(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_STRICT_PROVENANCE", "1")
    root = dataset(tmp_path / "valid")
    blob = compile_engine(
        v1.MODEL,
        v1.SYSTEM,
        "sglang",
        v1.VERSION,
        systems_path=str(root),
        decode_workload_distribution=PROFILE,
        **v1.KWARGS,
    )
    assert core.AicEngine.from_spec(blob) is not None
    v1.test_actual_public_native_timing_config_forwards_selector(monkeypatch, PROFILE)


@pytest.mark.parametrize("primary_nodes", [(1,), (1, 4, 8)])
def test_partial_primary_is_not_unioned_with_complete_lower_source(tmp_path, primary_nodes):
    import shutil

    root = dataset(tmp_path / "primary", nodes=primary_nodes)
    donor = dataset(tmp_path / "donor")
    lower = table_dir(root).with_name("0.5.17")
    shutil.copytree(table_dir(donor), lower)
    (root / "perf_data_reuse_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "groups": [
                    {
                        "op_file": "moe_perf.parquet",
                        "kernel_source": "sglang_flashinfer_trtllm_moe",
                        "tier": "shared",
                        "frameworks": ["sglang"],
                        "systems": [v1.SYSTEM],
                    }
                ]
            }
        )
    )
    # Establish that the same resolver really sees the lower complete source
    # when the primary has no selected rows, rather than merely being absent.
    visible = dataset(tmp_path / "visible", nodes=())
    shutil.copytree(lower, table_dir(visible).with_name("0.5.17"))
    shutil.copyfile(root / "perf_data_reuse_manifest.yaml", visible / "perf_data_reuse_manifest.yaml")
    assert load(visible).predict_decode_latency(3, 32768, 2) == 30.0
    with pytest.raises(DecodeMoeProfileError, match="incomplete"):
        load(root)


def test_scheduler_empty_step_retains_zero_work_short_circuit(tmp_path):
    root = dataset(tmp_path / "valid")
    for profile in (v1.PROFILE, PROFILE):
        handle = load(root, profile)
        assert handle.decode_step_latency(0, 1024, 2) == 0.0
        assert handle.mixed_step_latency(0, 0, 1024, 2) == 0.0
