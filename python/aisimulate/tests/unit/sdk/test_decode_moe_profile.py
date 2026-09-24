# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit profiles survive native execution/reload without a uniform fallback."""

import copy
import dataclasses
import json
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import aisimulate_core as core
from aisimulate_core.sdk import ForwardPassPerfModelConfig, common, rust_engine_step
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.engine import build_ops_json, compile_engine
from aisimulate_core.sdk.errors import DecodeMoeProfileError, PerfDataNotAvailableError
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit
PROFILE = "observed_glm52_nvfp4_decode_1ab2c747975e_v1"
VERSION = "0.5.18+nvinternal.rubin.0.8full.66997102"
COLLECTOR_HASH = "sha256:067ad7797474518eab028911d4f0d6f314e1dd0456b08be5bae45de267f3e332"
CASE_PLAN_HASH = "sha256:dd11934097a3bc664e82c7081af5ceac3161666eb689e736b47765d7ea955cde"
MODEL = "nvidia/GLM-5.2-NVFP4"
SYSTEM = "vr200_hecate"
KWARGS = dict(
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
    forward_model="op_level",
)


def dataset(root, *, points=(1, 8, 32), metadata_change=None, row_change=None):
    """Synthetic latencies intentionally differ from actual qualified data."""
    root.mkdir(parents=True)
    system = {
        "data_dir": "data/vr200_hecate",
        "gpu": {
            "mem_bw": 1e12,
            "mem_capacity": 10**12,
            "bfloat16_tc_flops": 1e15,
            "fp4_tc_flops": 4e15,
            "sm_version": 107,
        },
        "node": {"num_gpus_per_node": 4, "intra_node_bw": 1e12, "inter_node_bw": 1e11},
    }
    (root / (SYSTEM + ".yaml")).write_text(yaml.safe_dump(system))
    (root / "query_versions.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "defaults": {},
                "overrides": {SYSTEM: {"sglang": {"current": VERSION, "previous": None}}},
            }
        )
    )
    directory = root / "data" / SYSTEM / "moe/sglang" / VERSION
    directory.mkdir(parents=True)
    runtime = {
        "framework": "sglang",
        "version": VERSION,
        "image": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci",
        "image_digest": "sha256:53299500a280c8de34bd484507a45b2f83b4d5e7c999b77284fa31930f7e63ab",
    }
    meta = {
        "schema_version": 2,
        "runtime": runtime,
        "tables": {
            "moe_perf": {
                "rows": len(points) + 3,
                "status": "complete",
                "collections": [
                    {
                        "collector_ref": "collector.sglang_rubin.publish_observed_moe",
                        "collector_hash": COLLECTOR_HASH,
                        "case_plan_hash": CASE_PLAN_HASH,
                        "collected_at": "2026-09-19",
                        "rows": 3,
                        "status": "complete",
                        "source_campaign_rows": 864,
                        "source_campaign_status": "complete",
                        "runtime": copy.deepcopy(runtime),
                    }
                ],
            }
        },
    }
    if metadata_change:
        metadata_change(meta)
    (directory / "collection_meta.yaml").write_text(yaml.safe_dump(meta))
    rows = []
    for distribution, nodes, base in [(PROFILE, points, 1.0), ("uniform", (1, 8, 32), 50.0)]:
        for index, node in enumerate(nodes):
            row = dict(
                framework="SGLang",
                version=VERSION,
                device="NVIDIA Graphics Device",
                op_name="moe",
                kernel_source="sglang_flashinfer_trtllm_moe",
                moe_dtype="nvfp4",
                num_tokens=node,
                hidden_size=6144,
                inter_size=2048,
                topk=8,
                num_experts=256,
                moe_tp_size=4,
                moe_ep_size=1,
                distribution=distribution,
                latency=base + index,
            )
            if row_change and distribution == PROFILE:
                row_change(row)
            rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), directory / "moe_perf.parquet")
    return root


def spec(root=None, *, selected=PROFILE, exact=True):
    op = dict(
        name="generation_moe",
        scale_factor=1.0,
        hidden_size=6144,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_tp_size=4,
        moe_ep_size=1,
        attention_dp_size=1,
        quant_mode="nvfp4",
        workload_distribution=selected,
        is_gated=True,
        moe_backend=None,
        enable_eplb=False,
        is_context=True,
        require_exact_workload_distribution=exact,
    )
    return {
        "schema_version": core.engine_spec_schema_version(),
        "engine": dict(
            schema_version=1,
            model_name=MODEL,
            system_name=SYSTEM,
            backend="sglang",
            backend_version=VERSION,
            systems_path=None if root is None else str(root),
            tp_size=4,
            pp_size=1,
            moe_tp_size=4,
            moe_ep_size=1,
            attention_dp_size=1,
            cp_size=1,
            weight_dtype="bfloat16",
            activation_dtype="bfloat16",
            moe_dtype="nvfp4",
            kv_cache_dtype="fp8",
            kv_block_size=64,
            nextn=0,
            database_mode="SILICON",
            forward_model="op_level",
        ),
        "context_ops": [],
        "generation_ops": [{"Overlap": {"name": "generation_moe_overlap", "group_a": [{"Moe": op}], "group_b": []}}],
    }


def load(value, fallback_root=None):
    return core.AicEngine.from_spec(
        bytes(core.engine_spec_bincode_from_json(json.dumps(value))),
        systems_path=None if fallback_root is None else str(fallback_root),
    )


def test_exact_profile_wins_over_uniform_decoy_and_default_fallback_survives(tmp_path):
    root = dataset(tmp_path / "valid")
    handle = load(spec(root))
    assert [handle.predict_decode_latency(n, 1024, 2) for n in (1, 8, 32)] == [1.0, 2.0, 3.0]
    default = load(spec(root, selected="missing", exact=False))
    assert default.predict_decode_latency(1, 1024, 2) == 50.0


@pytest.mark.parametrize("power", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_decode_profile_loader_rejects_invalid_power(tmp_path, power):
    root = dataset(tmp_path / "invalid-power", row_change=lambda row: row.update(power=power))
    with pytest.raises(DecodeMoeProfileError, match="invalid profile power at node 1"):
        load(spec(root))


@pytest.mark.parametrize("power", [None, 0.0, 125.0])
def test_decode_profile_loader_accepts_optional_nonnegative_power(tmp_path, power):
    root = dataset(tmp_path / "valid-power", row_change=lambda row: row.update(power=power))
    engine = load(spec(root))
    result = engine.evaluate_generation_ops([0], 1, 1024)
    # The synthetic N=1 profile has 1 ms latency; Rust carries W*ms energy.
    assert result[0][1:3] == (1.0, power or 0.0)


@pytest.mark.parametrize("distribution", [PROFILE, "uniform", ""])
def test_public_legacy_migration_rejects_decode_distribution(distribution):
    legacy = spec()["engine"]
    legacy["decode_workload_distribution"] = distribution
    with pytest.raises(DecodeMoeProfileError, match="cannot be migrated"):
        ForwardPassPerfModelConfig.from_legacy_engine_config(legacy, "decode")


@pytest.mark.parametrize("selected_first", [False, True])
def test_warm_engine_cache_preserves_decode_distribution_and_exact_profile_limits(selected_first):
    from pathlib import Path

    root = Path(core.__file__).parent / "systems"
    database = PerfDatabase(SYSTEM, "sglang", VERSION, str(root), shared_layer=False, strict_provenance=True)
    ordinary = get_model(MODEL, build_model_config(**KWARGS), "sglang")
    selected = get_model(MODEL, build_model_config(**KWARGS, decode_workload_distribution=PROFILE), "sglang")
    rust_engine_step._engine_handle_cache_clear()
    try:
        for chosen in [selected, ordinary] if selected_first else [ordinary, selected]:
            if chosen is selected:
                with pytest.raises(DecodeMoeProfileError, match="1..=32"):
                    rust_engine_step.estimate_decode_step_latency_with_rust(
                        chosen, database, gen_tokens=33, isl=1024, osl=2
                    )
            else:
                assert (
                    rust_engine_step.estimate_decode_step_latency_with_rust(
                        chosen, database, gen_tokens=33, isl=1024, osl=2
                    )
                    > 0
                )
        default_handle = rust_engine_step._cached_engine_handle(ordinary, database)
        selected_handle = rust_engine_step._cached_engine_handle(selected, database)
        assert default_handle is not selected_handle
        assert default_handle.predict_decode_latency(1, 1024) != selected_handle.predict_decode_latency(1, 1024)
    finally:
        rust_engine_step._engine_handle_cache_clear()


@pytest.mark.parametrize("selected", ["typo", "", "uniform"])
def test_explicit_missing_or_unapproved_profile_never_falls_back(tmp_path, selected):
    root = dataset(tmp_path / "valid")
    with pytest.raises(DecodeMoeProfileError):
        load(spec(root, selected=selected))
    assert not issubclass(DecodeMoeProfileError, PerfDataNotAvailableError)


def test_serialized_reload_validates_different_effective_root_with_uniform_decoy(tmp_path):
    valid = dataset(tmp_path / "valid")
    missing = dataset(tmp_path / "missing", points=())
    blob = bytes(core.engine_spec_bincode_from_json(json.dumps(spec())))
    assert core.AicEngine.from_spec(blob, systems_path=str(valid)).predict_decode_latency(1, 1024, 2) == 1.0
    with pytest.raises(DecodeMoeProfileError, match="absent"):
        core.AicEngine.from_spec(blob, systems_path=str(missing))
    # The serialized systems_path wins over this fallback argument.
    assert load(spec(valid), missing).predict_decode_latency(1, 1024, 2) == 1.0
    with pytest.raises(DecodeMoeProfileError):
        load(spec(missing), valid)
    with pytest.raises(DecodeMoeProfileError, match="cannot load resolved profile database"):
        load(spec(tmp_path / "nonexistent-root"), valid)


def test_malformed_sidecar_cannot_fall_back_to_uniform_or_other_root(tmp_path):
    valid = dataset(tmp_path / "valid")
    malformed = dataset(tmp_path / "malformed")
    (malformed / "data" / SYSTEM / "moe/sglang" / VERSION / "collection_meta.yaml").write_text("[invalid yaml")
    with pytest.raises(DecodeMoeProfileError):
        load(spec(malformed), valid)
    # Only explicit intent converts load failures to the dedicated error.
    with pytest.raises(Exception) as failure:
        load(spec(tmp_path / "missing-default", exact=False), valid)
    assert not isinstance(failure.value, DecodeMoeProfileError)


def test_missing_required_middle_anchor_is_not_satisfied_by_interpolation(tmp_path):
    root = dataset(tmp_path / "missing-eight", points=(1, 32))
    with pytest.raises(DecodeMoeProfileError, match="N1/8/32"):
        load(spec(root))


@pytest.mark.parametrize(
    "field,value",
    [
        ("collector_hash", "sha256:" + "0" * 64),
        ("collector_hash", "sha256:3224e66909a24dbf519310b855346caa583dc8ffdac2fe54fd723edd2c36b56b"),
        ("case_plan_hash", "sha256:" + "0" * 64),
        ("collector_ref", "wrong"),
        ("rows", 2),
        ("source_campaign_rows", 863),
        ("status", "partial"),
        ("source_campaign_status", "partial"),
    ],
)
def test_profile_source_event_mismatch_rejects(tmp_path, field, value):
    root = dataset(
        tmp_path / "bad-event",
        metadata_change=lambda meta: meta["tables"]["moe_perf"]["collections"][0].update({field: value}),
    )
    with pytest.raises(DecodeMoeProfileError):
        load(spec(root))


@pytest.mark.parametrize("table_key", ["moe_perf.parquet", "different_perf"])
def test_metadata_requires_canonical_table_stem(tmp_path, table_key):
    def mutate(meta):
        meta["tables"][table_key] = meta["tables"].pop("moe_perf")

    root = dataset(tmp_path / "wrong-table-key", metadata_change=mutate)
    with pytest.raises(DecodeMoeProfileError):
        load(spec(root))


@pytest.mark.parametrize("location", ["runtime", "event"])
@pytest.mark.parametrize("field", ["framework", "version", "image", "image_digest", "source_commit"])
def test_runtime_mismatch_and_unsupported_schema2_runtime_key_reject(tmp_path, location, field):
    def mutate(meta):
        target = meta["runtime"] if location == "runtime" else meta["tables"]["moe_perf"]["collections"][0]["runtime"]
        target[field] = "wrong"

    root = dataset(tmp_path / "bad-runtime", metadata_change=mutate)
    with pytest.raises(DecodeMoeProfileError):
        load(spec(root))


@pytest.mark.parametrize(
    "field,value",
    [
        ("moe_dtype", "fp8"),
        ("hidden_size", 6143),
        ("moe_tp_size", 2),
        ("moe_ep_size", 2),
        ("kernel_source", "moe_torch_flow_min_latency"),
        ("version", "different"),
        ("num_tokens", 7),
    ],
)
def test_profile_row_shape_or_source_mismatch_rejects(tmp_path, field, value):
    root = dataset(tmp_path / "bad-row", row_change=lambda row: row.update({field: value}))
    with pytest.raises(DecodeMoeProfileError):
        load(spec(root))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "Qwen/Qwen3-32B"),
        ("tp_size", 2),
        ("moe_ep_size", 2),
        ("moe_tp_size", 2),
        ("pp_size", 2),
        ("cp_size", 2),
        ("attention_dp_size", 2),
        ("moe_dtype", "fp8"),
        ("nextn", 1),
        ("forward_model", "fpm"),
        ("weight_dtype", "float16"),
        ("activation_dtype", "fp8"),
        ("kv_cache_dtype", "bfloat16"),
        ("database_mode", "HYBRID"),
        ("database_mode", "EMPIRICAL"),
        ("database_mode", "SOL"),
        ("system_name", "different-system"),
        ("backend", "vllm"),
        ("backend_version", "different-version"),
    ],
)
def test_native_reload_scope_guard(tmp_path, field, value):
    root = dataset(tmp_path / "valid")
    value_spec = spec(root)
    value_spec["engine"][field] = value
    with pytest.raises(DecodeMoeProfileError):
        load(value_spec)


def test_explicit_profile_inside_fallback_cannot_be_replaced(tmp_path):
    root = dataset(tmp_path / "missing", points=())
    value = spec(root)
    selected = value["generation_ops"][0]["Overlap"]["group_a"][0]
    fallback = copy.deepcopy(selected)
    fallback["Moe"].update(require_exact_workload_distribution=False, workload_distribution="uniform")
    value["generation_ops"] = [
        {"Fallback": {"name": "cannot-mask-profile", "primary": selected, "fallback": [fallback]}}
    ]
    with pytest.raises(DecodeMoeProfileError):
        load(value)


def test_native_policy_roundtrip_and_stale_binary_rejection(tmp_path):
    root = dataset(tmp_path / "valid")
    value = spec(root)
    native_op = core.op_from_spec_json(json.dumps(value["generation_ops"][0]))
    assert json.loads(native_op._spec_json()) == json.loads(pickle.loads(pickle.dumps(native_op))._spec_json())
    blob = bytes(core.engine_spec_bincode_from_json(json.dumps(value)))
    assert core.AicEngine.from_spec(blob).predict_decode_latency(8, 1024, 2) == 2.0
    stale = (core.engine_spec_schema_version() - 1).to_bytes(4, "little") + blob[4:]
    with pytest.raises(ValueError, match="unsupported schema version"):
        core.AicEngine.from_spec(stale)


def test_model_selector_changes_only_generation_moe_and_survives_clone_pickle():
    base = build_model_config(**KWARGS)
    selected = dataclasses.replace(base, decode_workload_distribution=PROFILE)
    assert pickle.loads(pickle.dumps(selected)).decode_workload_distribution == PROFILE
    normal = get_model(MODEL, base, "sglang")
    changed = get_model(MODEL, selected, "sglang")
    assert build_ops_json(normal.context_ops) == build_ops_json(changed.context_ops)
    old, new = json.loads(build_ops_json(normal.generation_ops)), json.loads(build_ops_json(changed.generation_ops))
    found = []

    def normalize(ops):
        for op in ops:
            if "Moe" in op:
                found.append(op["Moe"].copy())
                op["Moe"]["workload_distribution"] = "power_law_1.01"
                op["Moe"]["require_exact_workload_distribution"] = False
            elif "Overlap" in op:
                normalize(op["Overlap"]["group_a"])
                normalize(op["Overlap"]["group_b"])

    normalize(new)
    assert len(found) == 1 and found[0]["name"] == "generation_moe"
    assert found[0]["workload_distribution"] == PROFILE and found[0]["require_exact_workload_distribution"] is True
    assert new == old
    assert build_ops_json(
        get_model(MODEL, dataclasses.replace(base, decode_workload_distribution=None), "sglang").generation_ops
    ) == build_ops_json(normal.generation_ops)


@pytest.mark.parametrize(
    "changes",
    [
        {"decode_workload_distribution": ""},
        {"decode_workload_distribution": 7},
        {"decode_workload_distribution": " profile "},
        {"enable_eplb": True},
        {"nextn": 1},
        {"forward_model": "fpm"},
        {"overwrite_num_layers": 4},
        {"pp_size": 2},
        {"cp_size": 2, "moe_tp_size": 8},
        {"tp_size": 2, "moe_tp_size": 2},
        {"attention_dp_size": 2, "moe_tp_size": 8},
        {"moe_ep_size": 2, "moe_tp_size": 2},
        {"moe_quant_mode": common.MoEQuantMode.fp8},
        {"gemm_quant_mode": common.GEMMQuantMode.fp8},
        {"fmha_quant_mode": common.FMHAQuantMode.fp8},
        {"kvcache_quant_mode": common.KVCacheQuantMode.bfloat16},
        {"comm_quant_mode": common.CommQuantMode.fp8},
    ],
)
def test_direct_model_config_cannot_silently_ignore_selector(changes):
    cfg = dataclasses.replace(build_model_config(**KWARGS), decode_workload_distribution=PROFILE)
    cfg = dataclasses.replace(cfg, **changes)
    with pytest.raises(ValueError):
        get_model(MODEL, cfg, "sglang")


def test_compile_engine_preflight_uses_native_missing_profile_guard(tmp_path):
    root = dataset(tmp_path / "missing", points=())
    with pytest.raises(DecodeMoeProfileError):
        compile_engine(
            MODEL, SYSTEM, "sglang", VERSION, systems_path=str(root), decode_workload_distribution=PROFILE, **KWARGS
        )
    with pytest.raises(DecodeMoeProfileError):
        get_model(
            "Qwen/Qwen3-1.7B",
            dataclasses.replace(build_model_config(**KWARGS), decode_workload_distribution=PROFILE),
            "sglang",
        )


@pytest.mark.parametrize("tokens", [0, 33, 128])
def test_explicit_profile_rejects_zero_and_unmeasured_tail(tmp_path, tokens):
    handle = load(spec(dataset(tmp_path / "valid")))
    with pytest.raises(DecodeMoeProfileError, match="1..=32"):
        handle.predict_decode_latency(tokens, 1024, 2)


def test_intermediate_query_uses_existing_linear_interpolation_only(tmp_path):
    handle = load(spec(dataset(tmp_path / "valid")))
    assert handle.predict_decode_latency(4, 1024, 2) == pytest.approx(1 + 3 / 7)
    assert handle.predict_decode_latency(20, 1024, 2) == pytest.approx(2.5)


def test_serialized_communication_scope_cannot_be_changed(tmp_path):
    value = spec(dataset(tmp_path / "valid"))
    value["context_ops"] = [
        {
            "CustomAllReduce": dict(
                name="context_ar",
                scale_factor=1.0,
                hidden_size=6144,
                tp_size=4,
                quant="fp8",
                seq_split=1,
            )
        }
    ]
    with pytest.raises(DecodeMoeProfileError, match="half communication"):
        load(value)


def test_public_compile_engine_accepts_exact_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_STRICT_PROVENANCE", "1")
    root = dataset(tmp_path / "valid")
    blob = compile_engine(
        MODEL, SYSTEM, "sglang", VERSION, systems_path=str(root), decode_workload_distribution=PROFILE, **KWARGS
    )
    assert core.AicEngine.from_spec(blob) is not None


@pytest.mark.parametrize("selected", [None, PROFILE])
def test_actual_public_native_timing_config_forwards_selector(monkeypatch, selected):
    from aisimulate import _runtime
    from aisimulate_core.sdk import engine

    seen = []

    def capture(*args, **kwargs):
        seen.append((args, kwargs))
        raise RuntimeError("stop-after-real-AicTimingConfig")

    monkeypatch.setattr(engine, "compile_engine", capture)
    timing = dict(
        model=MODEL,
        backend="sglang",
        system=SYSTEM,
        tp=4,
        backend_version=VERSION,
        decode_workload_distribution=selected,
    )
    wire = {
        "version": 1,
        "topology": {"kind": "aggregated", "workers": {"initial_workers": 1, "startup_delay_ms": 0.0}},
        "engine": {
            "tensor_parallel_size": 4,
            "num_gpu_blocks_is_explicit": True,
            "rank": {
                "backend": "sglang",
                "num_gpu_blocks": 100,
                "timing_model": {"type": "external", "provider": "aic", "config": timing},
            },
        },
        "requests": [],
    }
    with pytest.raises(RuntimeError, match="stop-after-real-AicTimingConfig"):
        _runtime.run_replay_json(json.dumps(wire))
    assert len(seen) == 1
    assert seen[0][1]["decode_workload_distribution"] == selected
