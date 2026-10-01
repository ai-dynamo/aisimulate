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
    RUNTIME_IMAGES,
    RUNTIME_VERSIONS,
    TIMING_METHODS,
    aggregate_rank_samples,
    attention_body,
    build_plan,
    geometry,
    geometry_key,
    indexer_regime,
    load_attempt,
    projection_quant_mode,
    representative_layer_is_uniform,
    sha256_json,
    target_keys,
    validate_row,
    write_parquet,
)
from collector.glm53flash_attention_launch import SMOKE_SWEEP

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
@pytest.mark.parametrize("tp", [2, 4])
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
    assert len(keys) == len(set(keys)) == 454
    regimes = {(phase, indexer_regime(phase, prefix, x, 2048)) for phase, _, prefix, x in keys}
    assert regimes == {(p, r) for p in ("context", "generation") for r in ("short", "pooled")}
    for request_set in plan["sets"]:
        batch = request_set["batch_size"]
        assert batch * request_set["seed_chunk"] <= plan["max_step_tokens"]
        if request_set["phase"] == "context":
            query = request_set["query"]
            assert batch * query <= plan["max_step_tokens"]
            assert request_set["targets"][-1] + query <= plan["max_context"]
            gaps = [b - a for a, b in zip(request_set["targets"], request_set["targets"][1:], strict=False)]
            assert all(gap >= query for gap in gaps)
        else:
            assert request_set["targets"] == sorted(request_set["targets"])
            assert request_set["targets"][-1] <= plan["max_context"]
    batches = {(phase, batch) for phase, batch, _, _ in keys}
    assert {b for p, b in batches if p == "context"} == {1, 2, 4, 8, 16, 32}
    assert {b for p, b in batches if p == "generation"} == {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}
    assert max(x for phase, _, _, x in keys if phase == "generation") == 131072
    assert max(prefix + x for phase, _, prefix, x in keys if phase == "context") == 106496


def test_plan_rejects_budget_overflow_and_unaligned_targets():
    bad = copy.deepcopy(SMOKE_SWEEP)
    bad["prefill"]["query_lengths"]["4"] = [4096]
    with pytest.raises(ValueError, match="step budget"):
        build_plan(bad)
    bad = copy.deepcopy(SMOKE_SWEEP)
    bad["prefill"]["prefix_lengths"] = [0, 1023]
    with pytest.raises(ValueError, match="aligned"):
        build_plan(bad)


def _flat(backend="vllm", checkpoint="fp8", tp=2):
    return geometry(config(checkpoint), backend, checkpoint, tp)


def _records(flat, phase, batch, prefix, x, latencies_by_rank, source=SHA, extra=None):
    method, graph = TIMING_METHODS[phase]
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
    assert row["used_cuda_graph"] is False and row["kv_seed_regime"] == "real_kv"
    assert evidence[0]["rank_max_ms"] == [3.0, 5.0, 2.0]
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
    wrong_graph = dict(rows[0], batch_size=3, used_cuda_graph=True)
    with pytest.raises(ValueError, match="CUDA graph use"):
        write_parquet(rows + [wrong_graph], tmp_path / "graph.parquet")
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


def _attempt(tmp_path, role="full"):
    flat = _flat()
    body = {
        "schema_version": 1,
        "op": "glm53flash_attention",
        "role": role,
        "geometry": flat,
        "checkpoint_revision": "rev",
        "framework_version": RUNTIME_VERSIONS["vllm"],
        "runtime_digest": RUNTIME_IMAGES["vllm"],
        "layer_id": 3,
        "sweep": SMOKE_SWEEP,
        "plan": build_plan(SMOKE_SWEEP),
        "source_commit": "c" * 40,
    }
    attempt = tmp_path / role
    raw = attempt / "raw"
    raw.mkdir(parents=True)
    (attempt / "manifest.json").write_text(json.dumps({**body, "manifest_sha256": sha256_json(body)}))
    records = []
    for phase, batch, prefix, x in target_keys(body["plan"]):
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
    assert pq.read_table(output).num_rows == len(target_keys(build_plan(SMOKE_SWEEP)))
    meta = yaml.safe_load((output.parent / "collection_meta.yaml").read_text())
    table = meta["tables"]["glm53_attention_module_perf"]
    assert meta["runtime"]["image_digest"] == RUNTIME_IMAGES["vllm"]
    assert table["rows"] == pq.read_table(output).num_rows and len(table["data_sha256"]) == 64
    assert json.loads(evidence.read_text())["attempts"][0]["deployment"] == "fp8-tp2"
