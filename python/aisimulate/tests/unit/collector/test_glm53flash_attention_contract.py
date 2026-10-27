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
    CHECKPOINT_CONFIG_SHA256,
    COLUMNS,
    KERNEL_DECODE,
    KERNEL_PREFILL,
    RUNTIME_IMAGES,
    RUNTIME_VERSIONS,
    TIMING_METHODS,
    aggregate_rank_samples,
    attention_body,
    build_plan,
    check_runtime_arch,
    context_class_sets,
    geometry,
    geometry_key,
    load_attempt,
    projection_quant_mode,
    representative_layer_is_uniform,
    runtime_arch,
    seed_chunk,
    sha256_json,
    target_keys,
    unaligned_targets,
    validate_row,
    write_parquet,
)
from collector.glm53flash_attention_tokens import spec as input_token_spec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "fp8": ROOT / "src/aisimulate_core/model_configs/zai-org--GLM-5.3-Flash_config.json",
    "nvfp4": ROOT / "src/aisimulate_core/model_configs/nvidia--GLM-5.3-Flash-NVFP4_config.json",
}
SWEEP = ROOT / "collector/cases/base_ops/glm53flash_attention.yaml"
SHA = "a" * 64
# A small aligned sweep with both IndexPool regimes and a long-context class.
SMALL = {
    "layer_id": 3,
    "warmup": 3,
    "iterations": 10,
    "max_step_tokens": 8192,
    "max_context": 131072,
    "max_model_len": 131079,
    "prefill": {"batch_sizes": [1, 4], "query_lengths": {"1": [32, 2048], "4": [256]}, "prefix_lengths": [0, 4096]},
    "decode": {"batch_sizes": [1, 32], "sequence_lengths": {"1": [2048, 16384], "32": [4096]}},
}
LONG = {
    "max_context": 1048575,
    "max_model_len": 1048576,
    "prefill": {"batch_sizes": [1], "query_lengths": {"1": [1024]}, "prefix_lengths": [1032192]},
    "decode": {"batch_sizes": [1], "sequence_lengths": {"1": [1048575]}},
}


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
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
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


def test_repository_plan_covers_both_regimes_inside_the_serving_budget():
    plan = build_plan(sweep())
    assert plan["layer_id"] == 3
    keys = target_keys(plan)
    assert len(keys) == len(set(keys)) == 463
    regular = {s["set_id"] for s in plan["sets"] if s["context_class"] == "regular"}
    regular_plan = {**plan, "sets": [s for s in plan["sets"] if s["set_id"] in regular]}
    regular_keys = target_keys(regular_plan)
    assert len(regular_keys) == 454
    # Both IndexPool regimes (prefix + x <= index_topk selects every pool).
    for phase in ("context", "generation"):
        ends = {prefix + x for p, _, prefix, x in keys if p == phase}
        assert min(ends) <= 2048 < max(ends)
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
    bad = copy.deepcopy(SMALL)
    bad["prefill"]["query_lengths"]["4"] = [4096]
    with pytest.raises(ValueError, match="step budget"):
        build_plan(bad)
    assert unaligned_targets(build_plan(SMALL)) == []
    unaligned = copy.deepcopy(SMALL)
    unaligned["prefill"]["query_lengths"]["1"] = [2042]
    assert "prefill-b1-q2042-c0:prefix=0" in unaligned_targets(build_plan(unaligned))
    long = dict(copy.deepcopy(SMALL), long_context=copy.deepcopy(LONG))
    long["long_context"]["decode"]["sequence_lengths"]["1"] = [1048576]
    with pytest.raises(ValueError, match="must fit"):
        build_plan(long)


def _flat(backend="vllm", checkpoint="fp8", tp=2, kv_cache_dtype="fp8"):
    return geometry(config(checkpoint), backend, checkpoint, tp, kv_cache_dtype)


def _records(flat, phase, batch, prefix, x, latencies_by_rank, source=SHA, extra=None, method=None, arch="arm64"):
    method = method or {"context": KERNEL_PREFILL, "generation": KERNEL_DECODE}[phase]
    graph = TIMING_METHODS[phase][method]
    key = {
        "geometry": geometry_key(attention_body(flat, phase == "context")),
        "batch_size": batch,
        "prefix": prefix,
        "x": x,
    }
    provenance = {
        "framework_version": RUNTIME_VERSIONS[flat["backend"]],
        "source_sha256": source,
        "config_sha256": sha256_json(config(flat["checkpoint_format"])),
        "checkpoint_revision": "rev",
        "runtime_digest": RUNTIME_IMAGES[(flat["backend"], arch)],
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


def _aggregate(records, tp_size):
    """Aggregate a fixture whose every target carries its full repetition set."""
    return aggregate_rank_samples(records, tp_size, 1 + max(r["repetition"] for r in records))


def test_rank_maximum_then_median_and_published_schema():
    flat = _flat()
    records = _records(flat, "context", 2, 4096, 256, [[1.0, 5.0, 2.0], [3.0, 1.0, 1.0]])
    rows = _aggregate(records, 2)
    assert len(rows) == 1
    row = rows[0]
    assert tuple(row) == COLUMNS
    # Rank maxima per repetition are 3, 5, 2; median 3.
    assert row["latency"] == 3.0 and row["sample_count"] == 3
    assert row["measurement_scope"] == "local_compute" and row["execution_profile"] == "full"
    assert row["used_cuda_graph"] is True and row["kv_seed_regime"] == "real_kv"
    decode = _aggregate(_records(flat, "generation", 4, 0, 4096, [[1.0], [2.0]]), 2)
    assert decode[0]["used_cuda_graph"] is True and decode[0]["prefix"] == 0


def test_incomplete_duplicate_mixed_or_nonfinite_samples_fail():
    flat = _flat()
    with pytest.raises(ValueError, match="missing a TP rank"):
        _aggregate(_records(flat, "context", 1, 0, 16, [[1.0, 1.0]]), 2)
    doubled = _records(flat, "context", 1, 0, 16, [[1.0], [1.0]])
    with pytest.raises(ValueError, match="repeats rank"):
        _aggregate(doubled + doubled[:1], 2)
    mixed = _records(flat, "context", 1, 0, 16, [[1.0], [1.0]])
    mixed[1]["kernel_source"] = "another kernel"
    with pytest.raises(ValueError, match="mixes invocation identities"):
        _aggregate(mixed, 2)
    # Every rank holding the same short subset is still below the plan.
    with pytest.raises(ValueError, match="the plan requires 3"):
        aggregate_rank_samples(_records(flat, "context", 1, 0, 16, [[1.0, 1.0], [1.0, 1.0]]), 2, 3)
    with pytest.raises(ValueError, match="nonfinite"):
        _aggregate(_records(flat, "context", 1, 0, 16, [[1.0], [1.0]], extra={"finite": False}), 2)


def _table(tmp_path, rows):
    path = tmp_path / BASENAME
    write_parquet(rows, path)
    return pq.read_table(path)


def test_writer_emits_the_reader_schema_and_rejects_duplicates_and_mixed_provenance(tmp_path):
    rows = []
    for checkpoint in ("fp8", "nvfp4"):
        for tp in (2, 4):
            flat = _flat(checkpoint=checkpoint, tp=tp)
            rows += _aggregate(_records(flat, "context", 1, 0, 2048, [[1.0]] * tp), tp)
            rows += _aggregate(_records(flat, "generation", 1, 0, 2049, [[0.1]] * tp), tp)
    table = _table(tmp_path, rows)
    assert table.schema.names == list(COLUMNS)
    assert table.schema.field("batch_size").type == pa.int64()
    assert table.schema.field("latency").type == pa.float64()
    assert table.schema.field("used_cuda_graph").type == pa.bool_()
    assert table.num_rows == 8
    with pytest.raises(ValueError, match="duplicate physical"):
        write_parquet(rows + rows[:1], tmp_path / "dup.parquet")
    other_source = _aggregate(_records(_flat(tp=2), "context", 2, 0, 16, [[1.0], [1.0]], source="b" * 64), 2)
    with pytest.raises(ValueError, match="one backend/runtime/source"):
        write_parquet(rows + other_source, tmp_path / "mixed.parquet")
    for row in rows[:2]:
        # Both phases are timed under the serving CUDA graphs.
        with pytest.raises(ValueError, match="CUDA graph use"):
            validate_row(dict(row, used_cuda_graph=False))
    other_config = dict(rows[0], batch_size=3, config_sha256="c" * 64)
    with pytest.raises(ValueError, match="pinned checkpoint revision"):
        write_parquet(rows + [other_config], tmp_path / "config.parquet")


def test_pinned_config_hashes_match_the_bundled_checkpoint_configs():
    for checkpoint, path in CONFIGS.items():
        assert CHECKPOINT_CONFIG_SHA256[checkpoint] == sha256_json(json.loads(path.read_text()))


def test_rows_the_reader_would_reject_fail_before_publication():
    flat = _flat()
    row = _aggregate(_records(flat, "generation", 1, 0, 4096, [[1.0], [1.0]]), 2)[0]
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


def test_runtime_digest_is_pinned_per_backend_and_architecture(tmp_path):
    assert {arch for _, arch in RUNTIME_IMAGES} == {"arm64", "amd64"}
    assert len(set(RUNTIME_IMAGES.values())) == 4
    flat = _flat()
    arm = _aggregate(_records(flat, "generation", 1, 0, 4096, [[1.0], [1.0]]), 2)[0]
    amd = _aggregate(_records(flat, "generation", 1, 0, 4096, [[1.0], [1.0]], arch="amd64"), 2)[0]
    validate_row(arm)
    validate_row(amd)
    assert runtime_arch("vllm", amd["runtime_digest"]) == "amd64"
    # Another backend's digest is not this backend's pinned image.
    with pytest.raises(ValueError, match="pinned"):
        validate_row(dict(arm, runtime_digest=RUNTIME_IMAGES[("sglang", "arm64")]))
    # One table is one image: arm64 and amd64 rows never mix.
    with pytest.raises(ValueError, match="one backend/runtime"):
        write_parquet([arm, dict(amd, batch_size=2)], tmp_path / "arch.parquet")
    manifest = {"geometry": flat, "runtime_digest": amd["runtime_digest"], "runtime_arch": "amd64"}
    assert check_runtime_arch(manifest, "x86_64") == "amd64"
    with pytest.raises(RuntimeError, match="aarch64"):
        check_runtime_arch(manifest, "aarch64")
    with pytest.raises(RuntimeError):
        check_runtime_arch(dict(manifest, runtime_arch="arm64"), "x86_64")


def test_kv_cache_dtype_is_a_key_and_never_mixes_within_a_table(tmp_path):
    with pytest.raises(ValueError, match="does not serve"):
        _flat(backend="vllm", kv_cache_dtype="int8")
    fp8 = _flat(backend="sglang", tp=8)
    bf16 = _flat(backend="sglang", tp=8, kv_cache_dtype="bfloat16")
    assert bf16["num_heads"] == 8 and bf16["kv_cache_dtype"] == "bfloat16"
    assert attention_body(fp8, True) != attention_body(bf16, True)
    rows = {
        kv: _aggregate(_records(flat, "generation", 1, 0, 4096, [[1.0]] * 8, arch="amd64"), 8)
        for kv, flat in (("fp8", fp8), ("bfloat16", bf16))
    }
    assert validate_row(rows["bfloat16"][0])["kv_cache_dtype"] == "bfloat16"
    assert _table(tmp_path, rows["bfloat16"]).num_rows == 1
    with pytest.raises(ValueError, match="KV-dtype"):
        write_parquet(rows["fp8"] + rows["bfloat16"], tmp_path / "mixed-kv.parquet")
    vllm = _aggregate(_records(_flat(), "generation", 1, 0, 4096, [[1.0], [1.0]]), 2)[0]
    body = dict(json.loads(vllm["geometry"]), kv_cache_dtype="int8")
    with pytest.raises(ValueError, match="KV dtype"):
        validate_row(dict(vllm, geometry=geometry_key(body)))


def test_vllm_hopper_rows_are_bf16_kv_and_never_mix_with_fp8(tmp_path):
    # Stock vLLM 0.31.0 serves only BF16 KV on SM90 (FlashInfer SM90 sparse MLA
    # rejects the uint8 FP8 storage), so H100/H200 rows carry kv bfloat16.
    fp8 = _flat(backend="vllm", checkpoint="fp8", tp=8)
    bf16 = _flat(backend="vllm", checkpoint="fp8", tp=8, kv_cache_dtype="bfloat16")
    rows = {
        kv: _aggregate(_records(flat, "context", 1, 0, 2048, [[1.0]] * 8, arch="amd64"), 8)
        for kv, flat in (("fp8", fp8), ("bfloat16", bf16))
    }
    assert validate_row(rows["bfloat16"][0])["kv_cache_dtype"] == "bfloat16"
    assert _table(tmp_path, rows["bfloat16"]).num_rows == 1
    with pytest.raises(ValueError, match="KV-dtype"):
        write_parquet(rows["fp8"] + rows["bfloat16"], tmp_path / "mixed.parquet")


def _attempt(tmp_path, only_sets=None, name="full", sweep_spec=None, max_model_len=None, system=None):
    flat = _flat()
    sweep_spec = sweep_spec or SMALL
    plan = build_plan(sweep_spec)
    body = {
        "schema_version": 2,
        "op": "glm53flash_attention",
        "geometry": flat,
        "checkpoint_revision": "rev",
        "framework_version": RUNTIME_VERSIONS["vllm"],
        "runtime_digest": RUNTIME_IMAGES[("vllm", "arm64")],
        "layer_id": 3,
        "sweep": sweep_spec,
        "plan": plan,
        "input_tokens": input_token_spec(plan),
        "source_commit": "c" * 40,
    }
    if system is not None:
        body["system"] = system
    if only_sets is not None:
        body["only_sets"] = sorted(only_sets)
        plan = {**plan, "sets": [s for s in plan["sets"] if s["set_id"] in only_sets]}
    body["max_model_len"] = max_model_len or plan["sets"][0]["max_model_len"]
    attempt = tmp_path / name
    raw = attempt / "raw"
    raw.mkdir(parents=True)
    (attempt / "manifest.json").write_text(json.dumps({**body, "manifest_sha256": sha256_json(body)}))
    records = []
    for phase, batch, prefix, x in target_keys(plan):
        reps = range(sweep_spec["iterations"])
        records += _records(
            flat, phase, batch, prefix, x, [[1.0 + r % 2 for r in reps], [1.5 - r % 2 / 2 for r in reps]]
        )
    for rank in (0, 1):
        lines = [json.dumps(r) for r in records if r["tp_rank"] == rank]
        (raw / f"rank-{rank}.jsonl").write_text("\n".join(lines) + "\n")
    (raw / "COMPLETE").write_text("done\n")
    return attempt, records


def test_finalized_table_is_bound_to_the_attempts_system_directory(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    def finalize(attempt, system_dir):
        out = tmp_path / system_dir / "data" / system_dir / "glm53_attention/vllm" / RUNTIME_VERSIONS["vllm"] / BASENAME
        monkeypatch.setattr("sys.argv", ["x", "finalize", str(attempt), "--output", str(out)])
        contract.main()
        return out

    b300, _ = _attempt(tmp_path, name="b300", system="b300_sxm")
    # The Rust reader keys the system by directory: a B300 table under gb200
    # would load as GB200 silicon, so it is refused.
    with pytest.raises(ValueError, match="b300_sxm/glm53_attention/vllm"):
        finalize(b300, "gb200")
    assert pq.read_table(finalize(b300, "b300_sxm")).num_rows > 0
    # Manifests without a recorded system come from the GB300-only launcher.
    legacy, _ = _attempt(tmp_path, name="legacy")
    with pytest.raises(ValueError, match="gb300/glm53_attention/vllm"):
        finalize(legacy, "b300_sxm")
    assert pq.read_table(finalize(legacy, "gb300")).num_rows > 0
    # One table never mixes a recorded system with an unrecorded one.
    sets = [s["set_id"] for s in build_plan(SMALL)["sets"]]
    first, _ = _attempt(tmp_path, only_sets=sets[:-1], name="first")
    last, _ = _attempt(tmp_path, only_sets=sets[-1:], name="last", system="gb300")
    out = tmp_path / "mixed/data/gb300/glm53_attention/vllm" / RUNTIME_VERSIONS["vllm"] / BASENAME
    monkeypatch.setattr("sys.argv", ["x", "finalize", str(first), str(last), "--output", str(out)])
    with pytest.raises(ValueError, match="one system"):
        contract.main()


def test_attempt_admission_requires_every_planned_target(tmp_path):
    attempt, records = _attempt(tmp_path)
    manifest, rows = load_attempt(attempt)
    assert len(rows) == len(target_keys(manifest["plan"]))
    lines = (attempt / "raw/rank-1.jsonl").read_text().splitlines()
    missing_target = records[0]["target_id"]
    kept = [line for line in lines if json.loads(line)["target_id"] != missing_target]
    (attempt / "raw/rank-1.jsonl").write_text("\n".join(kept) + "\n")
    with pytest.raises(ValueError):
        load_attempt(attempt)


def test_regular_and_long_context_attempts_finalize_into_one_table(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    spec = dict(copy.deepcopy(SMALL), long_context=copy.deepcopy(LONG))
    plan = build_plan(spec)
    regular, long = context_class_sets(plan, "regular"), context_class_sets(plan, "long")
    first, _ = _attempt(tmp_path, only_sets=regular, name="regular", sweep_spec=spec)
    second, _ = _attempt(tmp_path, only_sets=long, name="long", sweep_spec=spec)
    assert load_attempt(second)[0]["max_model_len"] == 1048576
    out = tmp_path / "data/gb300/glm53_attention/vllm" / RUNTIME_VERSIONS["vllm"] / BASENAME
    monkeypatch.setattr("sys.argv", ["x", "finalize", str(first), str(second), "--output", str(out)])
    contract.main()
    assert pq.read_table(out).num_rows == len(target_keys(plan))
    meta = yaml.safe_load((out.parent / "collection_meta.yaml").read_text())
    table = meta["tables"]["glm53_attention_module_perf"]
    assert meta["runtime"]["image_digest"] == RUNTIME_IMAGES[("vllm", "arm64")]
    assert meta["runtime"]["image_arch"] == "arm64"
    assert meta["tables"]["glm53_attention_module_perf"]["kv_cache_dtype"] == "fp8"
    assert table["rows"] == len(target_keys(plan)) and len(table["data_sha256"]) == 64
    assert table["input_tokens"] == {"source": "seeded_random_tokens", **input_token_spec(plan)}
    assert table["timing_method"] == {"context": [KERNEL_PREFILL], "generation": [KERNEL_DECODE]}
    assert [a["max_model_len"] for a in table["attempts"]] == [131079, 1048576]
    assert table["attempts"][0]["deployment"] == "fp8-tp2"
    wrong, _ = _attempt(tmp_path, only_sets=long, name="wrong", sweep_spec=spec, max_model_len=131079)
    with pytest.raises(ValueError, match="max_model_len"):
        load_attempt(wrong)
    mixed, _ = _attempt(tmp_path, only_sets=regular[:1] + long, name="mixed", sweep_spec=spec)
    with pytest.raises(ValueError, match="one context class"):
        load_attempt(mixed)


def test_split_attempts_must_cover_the_plan_exactly_once(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    plan = build_plan(SMALL)
    sets = [s["set_id"] for s in plan["sets"]]
    assert len(sets) >= 2
    first, _ = _attempt(tmp_path, only_sets=sets[:-1], name="a")
    last, _ = _attempt(tmp_path, only_sets=sets[-1:], name="b")
    assert load_attempt(last)[1]
    out = tmp_path / "t/data/gb300/glm53_attention/vllm" / RUNTIME_VERSIONS["vllm"] / BASENAME
    base = ["x", "finalize", "--output", str(out)]
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
        "sweep": SWEEP,
        "context_class": "regular",
        "tag": "",
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
        "system": "gb300",
        "systems_dir": None,
        "arch": "arm64",
        "gpus_per_node": None,
        "kv_cache_dtype": None,
    }
    return Namespace(**{**values, **overrides})


def test_launch_selects_one_context_class_and_guards_vllm_alignment(tmp_path):
    from collector.glm53flash_attention_launch import prepare

    regular = json.loads((prepare(_launch_args(tmp_path, "r")) / "manifest.json").read_text())
    assert regular["max_model_len"] == 131079 and regular["geometry"]["tp_size"] == 1
    assert len(regular["only_sets"]) == len(context_class_sets(regular["plan"], "regular"))
    run = (tmp_path / "r" / "run.sbatch").read_text()
    assert "glm53-attn-vllm-nvfp4-tp1" in run
    long = json.loads((prepare(_launch_args(tmp_path, "l", context_class="long")) / "manifest.json").read_text())
    assert long["max_model_len"] == 1048576 and all(s.startswith("long-") for s in long["only_sets"])
    sglang = prepare(_launch_args(tmp_path, "s", backend="sglang", tp=2, context_class="long"))
    assert "--context-length 1048576" in (sglang / "run.sbatch").read_text()
    unaligned = tmp_path / "unaligned.yaml"
    spec = copy.deepcopy(SMALL)
    spec["prefill"]["query_lengths"]["1"] = [2042]
    unaligned.write_text(yaml.safe_dump({"common_case_values": {"glm53flash_attention": spec}}))
    with pytest.raises(SystemExit, match="kpool_align4"):
        prepare(_launch_args(tmp_path, "v", sweep=unaligned))


def test_launch_follows_the_system_yaml_and_image_architecture(tmp_path):
    from collector.glm53flash_attention_launch import prepare

    def launch(name, **overrides):
        attempt = prepare(_launch_args(tmp_path, name, **overrides))
        return json.loads((attempt / "manifest.json").read_text()), (attempt / "run.sbatch").read_text()

    # GB300 (NVL72, 4 GPUs/node, arm64): the NVL72 vLLM serving environment.
    manifest, run = launch("gb300")
    assert "#SBATCH --gres=gpu:4" in run and "NCCL_MNNVL_ENABLE=1" in run
    assert "PYTHONNOUSERSITE=1" in run and "--no-container-mount-home" in run
    assert manifest["serving_env"]["VLLM_USE_NCCL_SYMM_MEM"] == "1"
    assert manifest["serving_env"]["NCCL_MNNVL_ENABLE"] == "1"
    assert manifest["runtime_digest"] == RUNTIME_IMAGES[("vllm", "arm64")] and manifest["runtime_arch"] == "arm64"
    assert (manifest["system"], manifest["gpus_per_node"]) == ("gb300", 4)
    # HGX H200 (8 GPUs/node, x86): TP8 FP8; NVLS and symmetric memory stay on,
    # MNNVL is off (the NVLink domain is one node).
    hopper = {"system": "h200_sxm", "arch": "amd64", "kv_cache_dtype": "bfloat16"}
    manifest, run = launch("h200", checkpoint="fp8", config=CONFIGS["fp8"], tp=8, **hopper)
    assert "#SBATCH --gres=gpu:8" in run and "--account=acct" in run
    assert manifest["serving_env"] == {
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NCCL_P2P_LEVEL": "NVL",
        "VLLM_USE_NCCL_SYMM_MEM": "1",
    }
    assert "NCCL_MNNVL_ENABLE" not in run and "NCCL_NVLS_ENABLE=1" in run and "VLLM_USE_NCCL_SYMM_MEM=1" in run
    assert manifest["runtime_digest"] == RUNTIME_IMAGES[("vllm", "amd64")]
    assert manifest["geometry"]["tp_size"] == 8 and manifest["geometry"]["num_heads"] == 8
    assert launch("gpn", gpus_per_node=4, tp=4, **hopper)[1].count("--gres=gpu:4") == 1
    # Without --kv-cache-dtype a Hopper attempt is refused (its runtimes serve
    # only BF16 KV); elsewhere the default is FP8 (the GB300 attempt above).
    with pytest.raises(SystemExit, match="explicit --kv-cache-dtype"):
        launch("default", system="h100_sxm", arch="amd64", checkpoint="fp8", config=CONFIGS["fp8"], tp=8)
    assert launch("b300", system="b300_sxm", arch="amd64", tp=2)[0]["geometry"]["kv_cache_dtype"] == "fp8"
    # SGLang on Hopper serves BF16 KV: the flag and the key follow the manifest.
    manifest, run = launch(
        "sgl",
        backend="sglang",
        system="h100_sxm",
        arch="amd64",
        checkpoint="fp8",
        config=CONFIGS["fp8"],
        tp=8,
        kv_cache_dtype="bfloat16",
    )
    assert "--kv-cache-dtype bfloat16" in run and "fp8_e4m3" not in run
    assert "PYTHONNOUSERSITE=1" in run
    assert manifest["geometry"]["kv_cache_dtype"] == "bfloat16" and manifest["serving_env"] == {}
    assert manifest["runtime_digest"] == RUNTIME_IMAGES[("sglang", "amd64")]
    assert "--kv-cache-dtype fp8_e4m3" in launch("sgl-gb", backend="sglang", tp=2)[1]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # FP8 weights (305.8 GiB) exceed one GB300, B300 or GB200 GPU.
        ({"checkpoint": "fp8", "tp": 1}, "exceed one gb300 GPU"),
        ({"checkpoint": "fp8", "tp": 1, "system": "b300_sxm", "arch": "amd64"}, "exceed one b300_sxm"),
        # NVFP4 (190.4 GiB) exceeds one GB200 (185 GiB); FP8 TP2 exceeds H100.
        ({"system": "gb200"}, "exceed one gb200"),
        (
            {"checkpoint": "fp8", "tp": 2, "system": "h100_sxm", "arch": "amd64", "kv_cache_dtype": "bfloat16"},
            "exceed one h100_sxm",
        ),
        # One attempt is one node: TP8 needs two 4-GPU NVL72 trays.
        ({"tp": 8, "system": "gb200"}, "exceeds the 4 GPUs"),
    ],
)
def test_launch_rejects_capacity_and_topology_impossibilities(tmp_path, overrides, message):
    from collector.glm53flash_attention_launch import prepare

    with pytest.raises(SystemExit, match=message):
        prepare(_launch_args(tmp_path, "x", **overrides))


def test_vllm_hopper_launch_serves_bf16_kv(tmp_path):
    from collector.glm53flash_attention_launch import prepare

    attempt = prepare(
        _launch_args(
            tmp_path,
            "vllm-h100",
            system="h100_sxm",
            arch="amd64",
            checkpoint="fp8",
            config=CONFIGS["fp8"],
            tp=8,
            kv_cache_dtype="bfloat16",
        )
    )
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["geometry"]["kv_cache_dtype"] == "bfloat16"
    assert manifest["runtime_digest"] == RUNTIME_IMAGES[("vllm", "amd64")]


def test_vllm_runner_and_worker_pass_the_key_kv_dtype_to_the_engine():
    # The runner builds engine_args from the manifest key; the worker refuses
    # an engine whose resolved cache_dtype differs from it.
    runner = (ROOT / "collector/vllm/glm53flash_attention_runner.py").read_text()
    worker = (ROOT / "collector/vllm/glm53flash_attention_worker.py").read_text()
    assert '"kv_cache_dtype": SERVING_KV_CACHE_DTYPE[expected["kv_cache_dtype"]]' in runner
    assert '"fp8_e4m3"' not in runner
    assert "cache_config.cache_dtype" in worker and 'SERVING_KV_CACHE_DTYPE[expected["kv_cache_dtype"]]' in worker
    from collector.glm53flash_attention_contract import SERVING_KV_CACHE_DTYPE

    assert SERVING_KV_CACHE_DTYPE == {"fp8": "fp8_e4m3", "bfloat16": "bfloat16"}


def test_tp1_admission_is_per_system_capacity(tmp_path):
    from collector.glm53flash_attention_launch import prepare

    # NVFP4 fits one B300 or GB300 GPU on either backend; no backend rule.
    for name, overrides in (("b300", {"system": "b300_sxm", "arch": "amd64"}), ("sgl", {"backend": "sglang"})):
        manifest = json.loads((prepare(_launch_args(tmp_path, name, **overrides)) / "manifest.json").read_text())
        assert manifest["geometry"]["tp_size"] == 1


def test_partial_attempt_admits_only_its_complete_sets(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    plan = build_plan(SMALL)
    sets = [s["set_id"] for s in plan["sets"]]
    failed_set = sets[-1]
    broken, _ = _attempt(tmp_path, name="broken")
    (broken / "raw/COMPLETE").unlink()
    with pytest.raises(ValueError, match="no completion receipt"):
        load_attempt(broken)
    lost = {f"{p}-b{b}-p{q}-x{x}" for p, b, q, x in target_keys({"sets": plan["sets"][-1:]})}
    rank1 = (broken / "raw/rank-1.jsonl").read_text().splitlines()
    kept = [line for line in rank1 if json.loads(line)["target_id"] not in lost]
    (broken / "raw/rank-1.jsonl").write_text("\n".join(kept) + "\n")
    with pytest.raises(ValueError, match="no failed set"):
        load_attempt(broken, partial=True)
    (broken / "raw/progress.jsonl").write_text(json.dumps({"set_id": failed_set, "status": "failed"}) + "\n")
    manifest, rows = load_attempt(broken, partial=True)
    assert manifest["failed_sets"] == [failed_set] and failed_set not in manifest["admitted_sets"]
    assert len(rows) == len(target_keys(plan)) - len(lost)
    # The rerun covers the failed set; the partial copy of its other targets is superseded.
    rerun, _ = _attempt(tmp_path, only_sets=[failed_set, sets[0]], name="rerun")
    out = tmp_path / "p/data/gb300/glm53_attention/vllm" / RUNTIME_VERSIONS["vllm"] / BASENAME
    argv = ["x", "finalize", str(rerun), "--partial", str(broken), "--output", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    contract.main()
    assert pq.read_table(out).num_rows == len(target_keys(plan))
    meta = yaml.safe_load((out.parent / "collection_meta.yaml").read_text())
    partial = meta["tables"]["glm53_attention_module_perf"]["attempts"][1]["partial"]
    assert partial["failed_sets"] == [failed_set] and partial["superseded_by_complete_attempts"] >= 1


def test_classified_failures_are_explicit_gaps(tmp_path, monkeypatch):
    from collector import glm53flash_attention_contract as contract

    plan = build_plan(SMALL)
    sets = [s["set_id"] for s in plan["sets"]]
    first, _ = _attempt(tmp_path, only_sets=sets[:-1], name="a")
    gap = [list(k) for k in target_keys({"sets": plan["sets"][-1:]})]
    classified = tmp_path / "classified.json"
    out = tmp_path / "c/data/gb300/glm53_attention/vllm" / RUNTIME_VERSIONS["vllm"] / BASENAME
    argv = ["x", "finalize", str(first), "--output", str(out)]
    entries = [{"deployment": "fp8-tp2", "key": k, "reason": "OOM", "evidence": ["log"]} for k in gap]
    classified.write_text(json.dumps(entries))
    monkeypatch.setattr("sys.argv", argv + ["--classified-failures", str(classified)])
    contract.main()
    meta = yaml.safe_load((out.parent / "collection_meta.yaml").read_text())
    assert len(meta["tables"]["glm53_attention_module_perf"]["classified_failures"]) == len(gap)
    classified.write_text(json.dumps([{**entries[0], "evidence": []}]))
    with pytest.raises(ValueError, match="evidence"):
        contract.main()
    second, _ = _attempt(tmp_path, only_sets=sets[-1:], name="b")
    classified.write_text(json.dumps(entries))
    monkeypatch.setattr("sys.argv", argv[:3] + [str(second)] + argv[3:] + ["--classified-failures", str(classified)])
    with pytest.raises(ValueError, match="were measured"):
        contract.main()
