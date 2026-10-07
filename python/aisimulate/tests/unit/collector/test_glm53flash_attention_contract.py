# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from collector.glm53flash_attention_contract import (
    BASENAME,
    COLUMNS,
    KERNEL_DECODE,
    KERNEL_PREFILL,
    RUNTIME_IMAGES,
    RUNTIME_VERSIONS,
    TIMING_METHODS,
    aggregate_rank_samples,
    attention_body,
    build_plan,
    context_class_sets,
    geometry,
    geometry_key,
    indexer_regime,
    load_attempt,
    projection_quant_mode,
    representative_layer_is_uniform,
    seed_chunk,
    sha256_json,
    target_keys,
    unaligned_targets,
    validate_row,
    write_parquet,
)
from collector.glm53flash_attention_launch import SMOKE_SWEEPS
from collector.glm53flash_attention_tokens import spec as input_token_spec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "fp8": ROOT / "src/aisimulate_core/model_configs/zai-org--GLM-5.3-Flash_config.json",
    "nvfp4": ROOT / "src/aisimulate_core/model_configs/nvidia--GLM-5.3-Flash-NVFP4_config.json",
}
SWEEP = ROOT / "collector/cases/base_ops/glm53flash_attention.yaml"
SHA = "a" * 64


def config(checkpoint):
    return json.loads(CONFIGS[checkpoint].read_text())


def sweep():
    return yaml.safe_load(SWEEP.read_text())["common_case_values"]["glm53flash_attention"]


def test_projection_precision_follows_the_pinned_model_builders():
    assert projection_quant_mode("sglang", "fp8") == "fp8_block"
    for backend, checkpoint in (("sglang", "nvfp4"), ("vllm", "fp8"), ("vllm", "nvfp4")):
        assert projection_quant_mode(backend, checkpoint) == "bfloat16"
    with pytest.raises(ValueError):
        projection_quant_mode("trtllm", "fp8")


@pytest.mark.parametrize("checkpoint", ["fp8", "nvfp4"])
def test_layer3_represents_every_sparse_mla_layer(checkpoint):
    cfg = config(checkpoint)
    representative_layer_is_uniform(cfg, 3, checkpoint)
    with pytest.raises(ValueError, match="not a NoPE sparse-MLA layer"):
        representative_layer_is_uniform(cfg, 4, checkpoint)
    reused = copy.deepcopy(cfg)
    reused["text_config"]["indexer_types"][7] = "shared"
    with pytest.raises(ValueError, match="reuses"):
        representative_layer_is_uniform(reused, 3, checkpoint)


def test_fp8_representative_layer_rejects_a_differently_quantized_layer():
    cfg = copy.deepcopy(config("fp8"))
    cfg["quantization_config"]["modules_to_not_convert"].append("model.layers.43.self_attn.o_proj")
    with pytest.raises(ValueError, match="layer 43 quantization"):
        representative_layer_is_uniform(cfg, 3, "fp8")


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("checkpoint", ["fp8", "nvfp4"])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_geometry_key_is_the_model_operator_body(backend, checkpoint, tp):
    from aisimulate_core.sdk import common
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.models import get_model

    quant = common.GEMMQuantMode.fp8_block if checkpoint == "fp8" else common.GEMMQuantMode.nvfp4
    moe = common.MoEQuantMode.fp8_block if checkpoint == "fp8" else common.MoEQuantMode.nvfp4
    model_path = {"fp8": "zai-org/GLM-5.3-Flash", "nvfp4": "nvidia/GLM-5.3-Flash-NVFP4"}[checkpoint]
    model = get_model(
        model_path,
        ModelConfig(
            tp_size=tp,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=tp,
            moe_ep_size=1,
            gemm_quant_mode=quant,
            moe_quant_mode=moe,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        ),
        backend,
    )
    flat = geometry(config(checkpoint), backend, checkpoint, tp)
    for is_context, ops in ((True, model.context_ops), (False, model.generation_ops)):
        emitted = {
            geometry_key({k: v for k, v in spec["Glm53Attention"].items() if k not in ("name", "measured")})
            for spec in (json.loads(op._spec_json()) for op in ops)
            if "Glm53Attention" in spec and spec["Glm53Attention"]["layer_kind"] == "sparse_mla"
        }
        assert emitted == {geometry_key(attention_body(flat, is_context))}


def test_indexer_regimes_split_at_topk():
    assert indexer_regime("context", 0, 2048, 2048) == "short"
    assert indexer_regime("context", 0, 2052, 2048) == "pooled"
    assert indexer_regime("context", 1024, 1024, 2048) == "short"
    assert indexer_regime("context", 1024, 1028, 2048) == "pooled"
    assert indexer_regime("generation", 0, 2048, 2048) == "short"
    assert indexer_regime("generation", 0, 2049, 2048) == "pooled"
    with pytest.raises(ValueError):
        indexer_regime("generation", 1, 10, 2048)


def test_repository_plan_covers_both_regimes_inside_the_serving_budget():
    plan = build_plan(sweep())
    assert plan["layer_id"] == 3
    keys = target_keys(plan)
    assert len(keys) == len(set(keys)) == 463
    regular = {s["set_id"] for s in plan["sets"] if s["context_class"] == "regular"}
    regular_plan = {**plan, "sets": [s for s in plan["sets"] if s["set_id"] in regular]}
    regular_keys = target_keys(regular_plan)
    assert len(regular_keys) == 454
    regimes = {(phase, indexer_regime(phase, prefix, x, 2048)) for phase, _, prefix, x in keys}
    assert regimes == {(p, r) for p in ("context", "generation") for r in ("short", "pooled")}
    for request_set in plan["sets"]:
        batch = request_set["batch_size"]
        assert batch * request_set["seed_chunk"] <= plan["max_step_tokens"]
        assert request_set["seed_chunk"] % 4 == 0
        limit = request_set["max_model_len"]
        if request_set["phase"] == "context":
            query = request_set["query"]
            assert batch * query <= plan["max_step_tokens"]
            assert request_set["targets"][-1] + query < limit
            gaps = [b - a for a, b in zip(request_set["targets"], request_set["targets"][1:], strict=False)]
            assert all(gap >= query for gap in gaps)
        else:
            assert request_set["targets"] == sorted(request_set["targets"])
            assert request_set["targets"][-1] <= limit - 1
    # Stock vLLM 0.31.0 runs every repository prefill chunk on the pool grid.
    assert unaligned_targets(plan) == []
    batches = {(phase, batch) for phase, batch, _, _ in regular_keys}
    assert {b for p, b in batches if p == "context"} == {1, 2, 4, 8, 16, 32}
    assert {b for p, b in batches if p == "generation"} == {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}
    assert max(x for phase, _, _, x in regular_keys if phase == "generation") == 131072
    assert max(prefix + x for phase, _, prefix, x in regular_keys if phase == "context") == 106496
    assert {s["max_model_len"] for s in regular_plan["sets"]} == {131079}


def test_long_context_rows_are_b1_and_use_the_model_position_limit():
    plan = build_plan(sweep())
    long_sets = [s for s in plan["sets"] if s["set_id"] in context_class_sets(plan, "long")]
    assert {s["max_model_len"] for s in long_sets} == {1048576}
    long_plan = {**plan, "sets": long_sets}
    assert sorted(target_keys(long_plan)) == sorted(
        [("context", 1, p, q) for p in (262144, 524288, 1032192) for q in (1024, 8192)]
        + [("generation", 1, 0, x) for x in (262144, 524288, 1048575)]
    )
    assert all(s["set_id"].startswith("long-") for s in long_sets)


def test_seed_chunks_are_floored_to_the_index_pool():
    assert [seed_chunk(8192, b) for b in (1, 3, 6, 12, 24, 32)] == [8192, 2728, 1364, 680, 340, 256]
    with pytest.raises(ValueError):
        seed_chunk(8, 3)


def test_plan_rejects_budget_overflow_and_reports_unaligned_targets():
    bad = copy.deepcopy(SMOKE_SWEEPS["validation-vllm"])
    bad["prefill"]["query_lengths"]["4"] = [4096]
    with pytest.raises(ValueError, match="step budget"):
        build_plan(bad)
    assert unaligned_targets(build_plan(SMOKE_SWEEPS["validation-vllm"])) == []
    sglang = unaligned_targets(build_plan(SMOKE_SWEEPS["validation-sglang"]))
    assert "prefill-b1-q2042-c0:prefix=0" in sglang and "prefill-b32-q10-c0:prefix=0" in sglang
    long = copy.deepcopy(SMOKE_SWEEPS["long"])
    long["long_context"]["decode"]["sequence_lengths"]["1"] = [1048576]
    with pytest.raises(ValueError, match="must fit"):
        build_plan(long)


def test_explicit_prefix_grids_select_exact_validation_geometries():
    keys = set(target_keys(build_plan(SMOKE_SWEEPS["validation-sglang"])))
    assert {k for k in keys if k[0] == "context"} == {
        ("context", 1, 128, 32),
        ("context", 1, 0, 2042),
        ("context", 1, 0, 8189),
        ("context", 4, 12032, 256),
        ("context", 32, 0, 10),
        ("context", 32, 4352, 32),
        ("context", 32, 0, 250),
        ("context", 32, 98048, 256),
    }


def _flat(backend="vllm", checkpoint="fp8", tp=2):
    return geometry(config(checkpoint), backend, checkpoint, tp)


def _records(flat, phase, batch, prefix, x, latencies_by_rank, source=SHA, extra=None, method=None):
    method = method or {"context": KERNEL_PREFILL, "generation": KERNEL_DECODE}[phase]
    graph = TIMING_METHODS[phase][method]
    key = {
        "geometry": geometry_key(attention_body(flat, phase == "context")),
        "batch_size": batch,
        "prefix": prefix,
        "x": x,
        "indexer_regime": indexer_regime(phase, prefix, x, 2048),
    }
    provenance = {
        "framework_version": RUNTIME_VERSIONS[flat["backend"]],
        "source_sha256": source,
        "config_sha256": sha256_json(config(flat["checkpoint_format"])),
        "checkpoint_revision": "rev",
        "runtime_digest": RUNTIME_IMAGES[flat["backend"]],
        "layer_id": 3,
    }
    target = f"{phase}-b{batch}-p{prefix}-x{x}"
    return [
        {
            "record": "sample",
            "target_id": target,
            "tp_rank": rank,
            "repetition": repetition,
            "latency_ms": latency,
            "key": key,
            "provenance": provenance,
            "kernel_source": f"{flat['backend']}|witness|{phase}",
            "timing_method": method,
            "used_cuda_graph": graph,
            "extra": extra or {"finite": True},
        }
        for rank, latencies in enumerate(latencies_by_rank)
        for repetition, latency in enumerate(latencies)
    ]


def test_rank_maximum_then_median_and_published_schema():
    flat = _flat()
    records = _records(flat, "context", 2, 4096, 256, [[1.0, 5.0, 2.0], [3.0, 1.0, 1.0]])
    rows, evidence = aggregate_rank_samples(records, 2)
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == COLUMNS
    # Rank maxima per repetition are 3, 5, 2; median 3.
    assert row["latency"] == 3.0 and row["sample_count"] == 3
    assert row["measurement_scope"] == "local_compute" and row["execution_profile"] == "full"
    assert row["used_cuda_graph"] is True and row["kv_seed_regime"] == "real_kv"
    assert evidence[0]["rank_max_ms"] == [3.0, 5.0, 2.0]
    assert evidence[0]["timing_method"] == KERNEL_PREFILL
    assert evidence[0]["latency_cv"] == pytest.approx(0.3742, abs=1e-4)
    assert evidence[0]["indexer_regime"] == "pooled"
    decode, _ = aggregate_rank_samples(_records(flat, "generation", 4, 0, 4096, [[1.0], [2.0]]), 2)
    assert decode[0]["used_cuda_graph"] is True and decode[0]["prefix"] == 0


def test_incomplete_duplicate_mixed_or_nonfinite_samples_fail():
    flat = _flat()
    with pytest.raises(ValueError, match="missing a TP rank"):
        aggregate_rank_samples(_records(flat, "context", 1, 0, 16, [[1.0, 1.0]]), 2)
    doubled = _records(flat, "context", 1, 0, 16, [[1.0], [1.0]])
    with pytest.raises(ValueError, match="repeats rank"):
        aggregate_rank_samples(doubled + doubled[:1], 2)
    mixed = _records(flat, "context", 1, 0, 16, [[1.0], [1.0]])
    mixed[1]["kernel_source"] = "another kernel"
    with pytest.raises(ValueError, match="mixes invocation identities"):
        aggregate_rank_samples(mixed, 2)
    with pytest.raises(ValueError, match="nonfinite"):
        aggregate_rank_samples(_records(flat, "context", 1, 0, 16, [[1.0], [1.0]], extra={"finite": False}), 2)


def _table(tmp_path, rows):
    path = tmp_path / BASENAME
    write_parquet(rows, path)
    return pq.read_table(path)


def test_writer_emits_the_reader_schema_and_rejects_duplicates_and_mixed_provenance(tmp_path):
    rows = []
    for checkpoint in ("fp8", "nvfp4"):
        for tp in (2, 4):
            flat = _flat(checkpoint=checkpoint, tp=tp)
            rows += aggregate_rank_samples(_records(flat, "context", 1, 0, 2048, [[1.0]] * tp), tp)[0]
            rows += aggregate_rank_samples(_records(flat, "generation", 1, 0, 2049, [[0.1]] * tp), tp)[0]
    table = _table(tmp_path, rows)
    assert table.schema.names == list(COLUMNS)
    assert table.schema.field("batch_size").type == pa.int64()
    assert table.schema.field("latency").type == pa.float64()
    assert table.schema.field("used_cuda_graph").type == pa.bool_()
    assert table.num_rows == 8
    with pytest.raises(ValueError, match="duplicate physical"):
        write_parquet(rows + rows[:1], tmp_path / "dup.parquet")
    other_source = aggregate_rank_samples(
        _records(_flat(tp=2), "context", 2, 0, 16, [[1.0], [1.0]], source="b" * 64), 2
    )[0]
    with pytest.raises(ValueError, match="one backend/runtime/source"):
        write_parquet(rows + other_source, tmp_path / "mixed.parquet")
    for row in rows[:2]:
        # Both phases are timed under the serving CUDA graphs.
        with pytest.raises(ValueError, match="CUDA graph use"):
            validate_row(dict(row, used_cuda_graph=False))
    other_config = dict(rows[0], batch_size=3, config_sha256="c" * 64)
    with pytest.raises(ValueError, match="configuration identities"):
        write_parquet(rows + [other_config], tmp_path / "config.parquet")


def test_rows_the_reader_would_reject_fail_before_publication():
    flat = _flat()
    row = aggregate_rank_samples(_records(flat, "generation", 1, 0, 4096, [[1.0], [1.0]]), 2)[0][0]
    validate_row(row)
    for change in (
        {"prefix": 7},
        {"kv_seed_regime": "n/a"},
        {"measurement_scope": "includes_all_reduce"},
        {"execution_profile": "decoder_bounded"},
        {"runtime_digest": "sha256:" + "0" * 64},
        {"latency": float("nan")},
        {"x": 0},
        {"component": "mhc"},
        {"geometry": row["geometry"].replace(":", ": ")},
    ):
        with pytest.raises(ValueError):
            validate_row(dict(row, **change))
    body = json.loads(row["geometry"])
    for change in ({"projection_quant_mode": "fp8_block"}, {"layer_kind": "kda"}, {"num_heads": 64}):
        with pytest.raises(ValueError):
            validate_row(dict(row, geometry=geometry_key(dict(body, **change))))


def _attempt(tmp_path, role="full", only_sets=None, name=None, sweep_spec=None, max_model_len=None):
    flat = _flat()
    sweep_spec = sweep_spec or SMOKE_SWEEPS["validation-vllm"]
    plan = build_plan(sweep_spec)
    body = {
        "schema_version": 2,
        "op": "glm53flash_attention",
        "role": role,
        "geometry": flat,
        "checkpoint_revision": "rev",
        "framework_version": RUNTIME_VERSIONS["vllm"],
        "runtime_digest": RUNTIME_IMAGES["vllm"],
        "layer_id": 3,
        "sweep": sweep_spec,
        "plan": plan,
        "input_tokens": input_token_spec(plan),
        "source_commit": "c" * 40,
    }
    if only_sets is not None:
        body["only_sets"] = sorted(only_sets)
        plan = {**plan, "sets": [s for s in plan["sets"] if s["set_id"] in only_sets]}
    body["max_model_len"] = max_model_len or plan["sets"][0]["max_model_len"]
    attempt = tmp_path / (name or role)
    raw = attempt / "raw"
    raw.mkdir(parents=True)
    (attempt / "manifest.json").write_text(json.dumps({**body, "manifest_sha256": sha256_json(body)}))
    records = []
    for phase, batch, prefix, x in target_keys(plan):
        records += _records(flat, phase, batch, prefix, x, [[1.0, 2.0], [1.5, 1.0]])
    for rank in (0, 1):
        lines = [json.dumps(r) for r in records if r["tp_rank"] == rank]
        (raw / f"rank-{rank}.jsonl").write_text("\n".join(lines) + "\n")
    (raw / "COMPLETE").write_text("done\n")
    return attempt, records


def test_attempt_admission_requires_every_planned_target(tmp_path):
    attempt, records = _attempt(tmp_path)
    manifest, rows, evidence = load_attempt(attempt)
    assert len(rows) == len(evidence) == len(target_keys(manifest["plan"]))
    smoke, _ = _attempt(tmp_path, role="smoke")
    with pytest.raises(ValueError, match="only full attempts publish"):
        load_attempt(smoke)
    lines = (attempt / "raw/rank-1.jsonl").read_text().splitlines()
    missing_target = records[0]["target_id"]
    kept = [line for line in lines if json.loads(line)["target_id"] != missing_target]
    (attempt / "raw/rank-1.jsonl").write_text("\n".join(kept) + "\n")
    with pytest.raises(ValueError):
        load_attempt(attempt)


def test_finalize_writes_table_evidence_and_collection_sidecar(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    attempt, _ = _attempt(tmp_path)
    output = tmp_path / "data/gb300/glm53_attention/vllm" / RUNTIME_VERSIONS["vllm"] / BASENAME
    evidence = tmp_path / "evidence.json"
    monkeypatch.setattr(
        "sys.argv", ["x", "finalize", str(attempt), "--output", str(output), "--evidence", str(evidence)]
    )
    contract.main()
    assert pq.read_table(output).num_rows == len(target_keys(build_plan(SMOKE_SWEEPS["validation-vllm"])))
    meta = yaml.safe_load((output.parent / "collection_meta.yaml").read_text())
    table = meta["tables"]["glm53_attention_module_perf"]
    assert meta["runtime"]["image_digest"] == RUNTIME_IMAGES["vllm"]
    assert table["rows"] == pq.read_table(output).num_rows and len(table["data_sha256"]) == 64
    assert json.loads(evidence.read_text())["attempts"][0]["deployment"] == "fp8-tp2"
    assert table["input_tokens"] == {
        "source": "seeded_random_tokens",
        **input_token_spec(build_plan(SMOKE_SWEEPS["validation-vllm"])),
    }
    assert table["timing_method"] == {"context": [KERNEL_PREFILL], "generation": [KERNEL_DECODE]}
    assert table["attempts"][0]["max_model_len"] == 131079


def test_regular_and_long_context_attempts_finalize_into_one_table(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    spec = copy.deepcopy(SMOKE_SWEEPS["validation-vllm"])
    spec["long_context"] = copy.deepcopy(SMOKE_SWEEPS["long"]["long_context"])
    spec["long_context"]["decode"]["sequence_lengths"]["1"] = [1048575]
    plan = build_plan(spec)
    regular, long = context_class_sets(plan, "regular"), context_class_sets(plan, "long")
    first, _ = _attempt(tmp_path, only_sets=regular, name="regular", sweep_spec=spec)
    second, _ = _attempt(tmp_path, only_sets=long, name="long", sweep_spec=spec)
    assert load_attempt(second)[0]["max_model_len"] == 1048576
    out = tmp_path / "t" / BASENAME
    argv = ["x", "finalize", str(first), str(second), "--output", str(out), "--evidence", str(tmp_path / "e.json")]
    monkeypatch.setattr("sys.argv", argv)
    contract.main()
    assert pq.read_table(out).num_rows == len(target_keys(plan))
    meta = yaml.safe_load((out.parent / "collection_meta.yaml").read_text())
    limits = [a["max_model_len"] for a in meta["tables"]["glm53_attention_module_perf"]["attempts"]]
    assert limits == [131079, 1048576]
    wrong, _ = _attempt(tmp_path, only_sets=long, name="wrong", sweep_spec=spec, max_model_len=131079)
    with pytest.raises(ValueError, match="max_model_len"):
        load_attempt(wrong)
    mixed, _ = _attempt(tmp_path, only_sets=regular[:1] + long, name="mixed", sweep_spec=spec)
    with pytest.raises(ValueError, match="one context class"):
        load_attempt(mixed)


def test_split_attempts_must_cover_the_plan_exactly_once(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    plan = build_plan(SMOKE_SWEEPS["validation-vllm"])
    sets = [s["set_id"] for s in plan["sets"]]
    assert len(sets) >= 2
    first, _ = _attempt(tmp_path, only_sets=sets[:-1], name="a")
    last, _ = _attempt(tmp_path, only_sets=sets[-1:], name="b")
    assert load_attempt(last)[1]
    out = tmp_path / "t" / BASENAME
    base = ["x", "finalize", "--output", str(out), "--evidence", str(tmp_path / "e.json")]
    monkeypatch.setattr("sys.argv", base[:2] + [str(first), str(last)] + base[2:])
    contract.main()
    rows = pq.read_table(out).to_pylist()
    assert len(rows) == len(target_keys(plan))
    meta = yaml.safe_load((out.parent / "collection_meta.yaml").read_text())
    assert [a.get("only_sets") for a in meta["tables"]["glm53_attention_module_perf"]["attempts"]] == [
        sorted(sets[:-1]),
        sets[-1:],
    ]
    monkeypatch.setattr("sys.argv", base[:2] + [str(first)] + base[2:])
    with pytest.raises(ValueError, match="cover"):
        contract.main()
    twice, _ = _attempt(tmp_path, only_sets=sets, name="c")
    monkeypatch.setattr("sys.argv", base[:2] + [str(first), str(twice)] + base[2:])
    with pytest.raises(ValueError, match="twice"):
        contract.main()


def _launch_args(tmp_path, name, **overrides):
    from argparse import Namespace

    values = {
        "attempt": tmp_path / name,
        "config": CONFIGS["nvfp4"],
        "smoke": None,
        "sweep": SWEEP,
        "context_class": "regular",
        "tag": "",
        "layer_id": None,
        "backend": "vllm",
        "checkpoint": "nvfp4",
        "tp": 1,
        "source_commit": "c" * 40,
        "allocator_max_split_mb": None,
        "only_sets": None,
        "skip_sets": None,
        "sglang_mem_fraction": None,
        "vllm_gpu_memory_utilization": None,
        "remote_attempt": "/remote/attempt",
        "remote_source": "/remote/src",
        "remote_model": "/remote/model",
        "image": "image.sqsh",
        "account": "acct",
        "partition": "batch",
        "time": "01:00:00",
    }
    return Namespace(**{**values, **overrides})


def test_launch_selects_one_context_class_and_guards_vllm_alignment(tmp_path):
    from collector.glm53flash_attention_launch import prepare

    regular = json.loads((prepare(_launch_args(tmp_path, "r")) / "manifest.json").read_text())
    assert regular["max_model_len"] == 131079 and regular["geometry"]["tp_size"] == 1
    assert len(regular["only_sets"]) == len(context_class_sets(regular["plan"], "regular"))
    run = (tmp_path / "r" / "run.sbatch").read_text()
    assert "glm53-v031c-attn-vllm-nvfp4-tp1" in run and "glm53flash-candidate" not in run
    long = json.loads((prepare(_launch_args(tmp_path, "l", context_class="long")) / "manifest.json").read_text())
    assert long["max_model_len"] == 1048576 and all(s.startswith("long-") for s in long["only_sets"])
    sglang = prepare(_launch_args(tmp_path, "s", backend="sglang", context_class="long"))
    assert "--context-length 1048576" in (sglang / "run.sbatch").read_text()
    with pytest.raises(SystemExit, match="kpool_align4"):
        prepare(_launch_args(tmp_path, "v", smoke="validation-sglang", sweep=None))
    smoke = prepare(_launch_args(tmp_path, "g", smoke="validation-sglang", sweep=None, backend="sglang"))
    assert json.loads((smoke / "manifest.json").read_text())["role"] == "smoke"
