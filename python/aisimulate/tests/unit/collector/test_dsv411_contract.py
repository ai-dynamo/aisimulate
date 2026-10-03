# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dsv411 contract: SDK manifest -> grid -> rows -> admission -> parquet (CPU only)."""

from __future__ import annotations

import json

import pytest

from collector.dsv411 import contract
from collector.sglang import collect_dsv411_module as sglang_producer
from collector.vllm import collect_dsv411_module as vllm_producer

pytestmark = pytest.mark.unit

PRODUCERS = {"sglang": sglang_producer, "vllm": vllm_producer}
SMOKE = {
    "context": {"query_lengths": [128, 2048], "past_kv_lengths": [0, 65536], "batch_sizes": [1, 4]},
    "generation": {"past_kv_lengths": [1024, 1048574], "batch_sizes": [1, 16]},
    "tokens": {"context_tokens": [16, 262144], "generation_tokens": [1, 1024]},
}


@pytest.fixture(scope="module")
def manifests():
    return {backend: contract.build_manifest(2, backend) for backend in PRODUCERS}


def test_manifest_structures_match_rust_keys(manifests):
    for backend, manifest in manifests.items():
        contract.validate_manifest(manifest)
        reps = manifest["representatives"]
        # 6 attention roles x ratios, 3 indexer geometries, 2 engram tables, 1 mhc, 2 shared projections
        assert {c: len(s) for c, s in reps["context"].items()} == {
            "attention_core": 6,
            "indexer": 3,
            "engram": 2,
            "mhc": 1,
            "shared_linear": 2,
        }
        core = next(e for e in manifest["entries"] if e["component"] == "attention_core" and e["layer"] == 2)
        assert core["structure_key"].startswith("role=full|compress_ratio=2|num_heads=32|head_dim=512|")
        assert core["structure_key"].endswith("|window_size=128|index_topk=512|quant_mode=fp8_block")
        indexer = next(e for e in manifest["entries"] if e["component"] == "indexer" and e["layer"] == 20)
        assert "|is_candidate_source=true|candidate_limit=0|" in indexer["structure_key"]
        engram = next(e for e in manifest["entries"] if e["component"] == "engram")
        assert engram["structure"]["sharding"] == ("row" if backend == "sglang" else "head")
        assert manifest["runtime_facts"]["index_entry_bytes"] == (68.0 if backend == "sglang" else 132.0)
        # representatives are the first layer carrying a structure
        for phase, by_component in reps.items():
            for component, structures in by_component.items():
                for key, layer in structures.items():
                    layers = [
                        e["layer"]
                        for e in manifest["entries"]
                        if e["phase"] == phase and e["component"] == component and e["structure_key"] == key
                    ]
                    assert layer == min(layers)


def test_grid_expansion_honors_budgets_and_counts_drops():
    grid = contract.load_grid()
    cases, drops = contract.expand_cases(grid, list(contract.STRUCTURE_FIELDS))
    kinds = {}
    for case in cases:
        kinds[(case["kind"], case["phase"])] = kinds.get((case["kind"], case["phase"]), 0) + 1
    assert kinds[("tokens", "context")] == len(grid["tokens"]["context_tokens"])
    assert kinds[("tokens", "generation")] == len(grid["tokens"]["generation_tokens"])
    ctx = [c for c in cases if c["kind"] == "attention" and c["phase"] == "context"]
    assert all(c["batch_size"] * c["query"] <= grid["context"]["max_new_tokens"] for c in ctx)
    assert all(
        c["batch_size"] <= grid["context"]["long_kv_max_batch"]
        for c in ctx
        if c["past_kv"] >= grid["context"]["long_kv_min"]
    )
    assert max(c["past_kv"] for c in ctx) == 1048575 and max(c["query"] for c in ctx) == 8192
    gen = [c for c in cases if c["kind"] == "attention" and c["phase"] == "generation"]
    assert all(c["batch_size"] * (c["past_kv"] + 1) <= grid["generation"]["max_tokens"] for c in gen)
    # a decode forward at seq_len == context_len never happens in serving (its token would not fit)
    assert max(c["past_kv"] + 1 for c in gen) == 1048575 < grid["context"]["max_sequence_length"]
    for floor, max_batch in grid["generation"]["decode_batch_ladder"]:
        assert all(c["batch_size"] <= max_batch for c in gen if c["past_kv"] >= floor)
    assert {"context.max_new_tokens", "context.long_kv_max_batch", "generation.max_tokens"} <= set(drops)
    assert len({c["case_id"] for c in cases}) == len(cases)
    sharded = [contract.shard_cases(cases, (i, 4)) for i in range(4)]
    assert sum(len(s) for s in sharded) == len(cases)
    owners = {}
    for i, shard in enumerate(sharded):
        for c in shard:
            assert owners.setdefault(contract.seed_group(c), i) == i, "a seed group must not straddle shards"
    assert min(len(s) for s in sharded) > 0.8 * max(len(s) for s in sharded)


def test_row_coordinates_follow_the_engine_convention():
    assert contract.coordinates(dict(kind="attention", phase="context", batch_size=4, query=2048, past_kv=65536)) == (
        4,
        2048,
        65536,
    )
    assert contract.coordinates(dict(kind="attention", phase="generation", batch_size=16, query=1, past_kv=32768)) == (
        16,
        1,
        32769,
    )
    assert contract.coordinates(dict(kind="tokens", phase="generation", tokens=64)) == (1, 64, 0)


def _plan(backend, manifest, purpose="calibration", **overrides):
    producer = PRODUCERS[backend]
    grid = contract.load_grid()
    cases, _ = contract.expand_cases(grid, list(contract.STRUCTURE_FIELDS), overrides=SMOKE)
    plan = dict(
        schema=contract.PLAN_SCHEMA,
        purpose=purpose,
        backend=backend,
        tp_size=2,
        components=list(contract.STRUCTURE_FIELDS),
        cases=cases,
        grid_sha256=contract.sha256_file(contract.GRID_PATH),
        chunk_prefill_size=8192,
        warmup=2,
        iterations=5,
        seed=1,
        regimes=dict(grid["regimes"]),
        regime_exceptions=[],
        expected_gpu="H20",
        expected_sm=90,
        framework_commit=producer.FRAMEWORK_COMMIT,
        framework_version=producer.FRAMEWORK_VERSION,
        runtime_digest="sha256:" + "b" * 64,
        image_sha256="b" * 64,
        source_pins=dict.fromkeys(producer.REQUIRED_SOURCES, "c" * 64),
        metadata_pins=dict.fromkeys(("config.json", "tokenizer.json", "tokenizer_config.json"), "d" * 64),
        collector_revision="e" * 40,
        weight_initializer=producer.WEIGHT_INITIALIZER,
        pool={},
    )
    plan.update(overrides)
    return plan


def _fake_run(tmp_path, backend, manifest, plan, *, skip=None):
    producer = PRODUCERS[backend]
    raw = tmp_path / f"run-{backend}"
    raw.mkdir(parents=True)
    (raw / "plan.json").write_text(json.dumps(plan, sort_keys=True))
    (raw / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    plan_sha = contract.sha256_file(raw / "plan.json")
    sources = dict.fromkeys(producer.REQUIRED_SOURCES, "c" * 64)
    provenance = dict(
        source_sha256=contract.sha256_json(sources),
        config_sha256=manifest["config_sha256"],
        runtime_digest=plan["runtime_digest"],
        case_plan_sha256=plan_sha,
    )
    for rank in range(plan["tp_size"]):
        receipt = dict(
            schema=contract.RECEIPT_SCHEMA,
            state="complete_pending_admission",
            tp_rank=rank,
            plan_sha256=plan_sha,
            manifest_sha256=contract.sha256_file(raw / "manifest.json"),
            runtime_digest=plan["runtime_digest"],
            image_sha256=plan["image_sha256"],
            purpose=plan["purpose"],
            framework_version=producer.FRAMEWORK_VERSION,
            collector_revision=plan["collector_revision"],
            checkpoint_weights_loaded=False,
            allocated_device_witness=dict(returncode=0, sm=90),
            source_hashes=sources,
            regime_violations=[],
        )
        (raw / f"rank-{rank}.json").write_text(json.dumps(receipt))
        with (raw / f"rank-{rank}.jsonl").open("w") as out:
            for case in plan["cases"]:
                components = contract.ATTENTION_COMPONENTS if case["kind"] == "attention" else contract.TOKEN_COMPONENTS
                for entry in contract.representative_entries(manifest, case["phase"], components):
                    if skip and skip(case, entry):
                        continue
                    regime, graph = contract.regime_for(plan, entry["component"], case["phase"])
                    for sample in range(plan["warmup"], plan["warmup"] + plan["iterations"]):
                        latency = 1.0 + rank + 0.1 * sample
                        row = contract.make_row(
                            entry,
                            case,
                            tp_size=plan["tp_size"],
                            latency=latency,
                            kernel_source="fake",
                            regime=regime,
                            used_cuda_graph=graph,
                        )
                        row.update(
                            provenance,
                            sample=sample,
                            invocation=case["index"],
                            tp_rank=rank,
                            case_id=case["case_id"],
                            collection_purpose=plan["purpose"],
                        )
                        out.write(json.dumps(row) + "\n")
    return raw


def _admit(backend, raw):
    producer = PRODUCERS[backend]
    return contract.aggregate_run(
        raw,
        framework_commit=producer.FRAMEWORK_COMMIT,
        framework_version=producer.FRAMEWORK_VERSION,
        expected_sm=producer.EXPECTED_SM,
        required_sources=producer.REQUIRED_SOURCES,
    )


@pytest.mark.parametrize("backend", sorted(PRODUCERS))
def test_fake_run_is_admitted_and_written(tmp_path, backend, manifests):
    manifest = manifests[backend]
    plan = _plan(backend, manifest)
    raw = _fake_run(tmp_path, backend, manifest, plan)
    rows, meta = _admit(backend, raw)
    expected = contract.expected_keys(plan, manifest)
    assert len(rows) == len(expected)
    # rank maximum (rank 1 = +1.0) then median over the 5 samples (sample 4 -> +0.4)
    assert all(abs(row["latency"] - (1.0 + 1 + 0.4)) < 1e-9 and row["sample_count"] == 5 for row in rows)
    gen = [r for r in rows if r["phase"] == "generation"]
    assert gen and all(r["measurement_regime"] == "cuda_graph" and r["used_cuda_graph"] for r in gen)
    ctx = [r for r in rows if r["phase"] == "context"]
    assert all(r["measurement_regime"] == "eager_drained" and not r["used_cuda_graph"] for r in ctx)
    assert all(
        r["kv_seed_regime"] == "real_kv"
        for r in rows
        if r["component"] in contract.ATTENTION_COMPONENTS and (r["phase"] == "generation" or r["kv_len"])
    )
    tokens = [r for r in rows if r["component"] in contract.TOKEN_COMPONENTS]
    assert tokens and all((r["batch_size"], r["kv_len"], r["kv_seed_regime"]) == (1, 0, "n/a") for r in tokens)
    import pyarrow.parquet as pq

    path = tmp_path / "dsv411_module_perf.parquet"
    contract.write_parquet(rows, path)
    table = pq.read_table(path)
    assert set(contract.KEY_COLUMNS) | set(contract.STRUCTURE_COLUMNS) | set(contract.IDENTITY_COLUMNS) <= set(
        table.column_names
    )
    assert table.num_rows == len(rows)
    assert meta["plan_sha256"] == contract.sha256_file(raw / "plan.json")


def test_incomplete_coverage_and_regime_violations_fail_closed(tmp_path, manifests):
    manifest = manifests["sglang"]
    plan = _plan("sglang", manifest)
    raw = _fake_run(
        tmp_path,
        "sglang",
        manifest,
        plan,
        skip=lambda case, entry: entry["component"] == "indexer" and case["phase"] == "generation",
    )
    with pytest.raises(ValueError, match="incomplete coverage"):
        _admit("sglang", raw)
    raw2 = _fake_run(tmp_path / "v", "sglang", manifest, plan)
    receipt = json.loads((raw2 / "rank-0.json").read_text())
    receipt["regime_violations"] = ["decode ran eagerly"]
    (raw2 / "rank-0.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="regime violations"):
        _admit("sglang", raw2)


def test_generation_rows_must_be_cuda_graph_unless_declared():
    entry = dict(
        component="mhc",
        structure=dict(hidden_size=5120, hc_mult=4, sinkhorn_iters=20),
        structure_key="x",
        phase="generation",
        layer=2,
        name="generation_mhc",
    )
    case = dict(kind="tokens", phase="generation", tokens=8, index=0, case_id="t")
    row = contract.make_row(
        entry, case, tp_size=2, latency=1.0, kernel_source="k", regime="cuda_graph", used_cuda_graph=False
    )
    row.update(source_sha256="a" * 64, config_sha256="a" * 64, runtime_digest="sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="cuda_graph"):
        contract.validate_row(row)
    row.update(measurement_regime="eager_exception")
    contract.validate_row(row)
    plan = dict(regime_exceptions=[dict(component="mhc", phase="generation", reason="x")])
    assert contract.regime_for(plan, "mhc", "generation") == ("eager_exception", False)
    assert contract.regime_for(plan, "engram", "generation") == ("cuda_graph", True)
    assert contract.regime_for(plan, "mhc", "context") == ("eager_drained", False)


def test_pool_runs_rejects_overlapping_keys(tmp_path, manifests):
    manifest = manifests["vllm"]
    plan = _plan("vllm", manifest)
    rows, _ = _admit("vllm", _fake_run(tmp_path, "vllm", manifest, plan))
    with pytest.raises(ValueError, match="two runs"):
        contract.pool_runs([rows, rows])
    assert len(contract.pool_runs([rows])) == len(rows)


def test_plan_pool_must_hold_every_attention_case(manifests):
    from collector.dsv411.plan import DEFAULT_POOL

    manifest = manifests["sglang"]
    producer = PRODUCERS["sglang"]
    kwargs = dict(
        framework_commit=producer.FRAMEWORK_COMMIT,
        framework_version=producer.FRAMEWORK_VERSION,
        expected_sm=producer.EXPECTED_SM,
        required_sources=producer.REQUIRED_SOURCES,
    )
    # the full grid's largest resident context case: batch 8 x (1048575 + 1) full tokens
    grid = contract.load_grid()
    cases, _ = contract.expand_cases(grid, list(contract.STRUCTURE_FIELDS))
    largest = max(
        (c for c in cases if c["kind"] == "attention"),
        key=lambda c: c["batch_size"] * (c["past_kv"] + (c["query"] if c["phase"] == "context" else 1)),
    )
    assert largest["batch_size"] * (largest["past_kv"] + largest["query"]) == 8 * 1048576
    contract.validate_plan(
        _plan("sglang", manifest, cases=[largest], pool=dict(DEFAULT_POOL["sglang"])), manifest, **kwargs
    )
    too_small = dict(DEFAULT_POOL["sglang"], max_total_tokens=2359296)
    with pytest.raises(ValueError, match="exceed the pool"):
        contract.validate_plan(_plan("sglang", manifest, cases=[largest], pool=too_small), manifest, **kwargs)
    # no pool declared (captures, vllm): nothing to check
    contract.validate_plan(_plan("sglang", manifest, cases=[largest], pool={}), manifest, **kwargs)


def test_resume_keeps_only_finished_cases(tmp_path, manifests):
    import json as _json

    from collector.dsv411.runtime import RowStream, completed_cases

    manifest = manifests["vllm"]
    plan = _plan("vllm", manifest)
    attention = [c for c in plan["cases"] if c["kind"] == "attention"]
    finished, partial = attention[0], attention[1]
    for rank in range(2):
        rows = [dict(case_id=finished["case_id"], v=rank), dict(case_id=partial["case_id"], v=rank)]
        if rank == 0:
            rows.append(dict(case_id="tokens-context-t16", v=0))
        (tmp_path / f"rank-{rank}.jsonl").write_text("".join(_json.dumps(r) + "\n" for r in rows))
    # rank 1 also finished the second case, rank 0 did not: only the intersection counts
    (tmp_path / "progress-rank-0.jsonl").write_text(_json.dumps(dict(case_id=finished["case_id"])) + "\n")
    (tmp_path / "progress-rank-1.jsonl").write_text(
        "".join(_json.dumps(dict(case_id=c["case_id"])) + "\n" for c in (finished, partial))
    )
    done = completed_cases(tmp_path, plan)
    assert done == {finished["case_id"]}
    stream = RowStream(tmp_path / "rank-0.jsonl", plan=plan, provenance={}, rank=0, keep_cases=done)
    assert stream.rows == 1
    stream.close()
    kept = [_json.loads(x) for x in (tmp_path / "rank-0.jsonl").read_text().splitlines()]
    assert [r["case_id"] for r in kept] == [finished["case_id"]]
    with pytest.raises(FileExistsError):
        RowStream(tmp_path / "rank-1.jsonl", plan=plan, provenance={}, rank=1)
