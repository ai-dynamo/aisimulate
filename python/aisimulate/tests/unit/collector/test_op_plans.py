# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every op plan is one expansion shared by every CUDA backend.

The tests run over every entry in ``OP_PLANS``, so moving an op onto a plan
needs no new test: each backend's getter must be the one-line plan call, the
backends may differ only by skips.py entries that name them, every left-out
case is counted under its entry, rows may not drop a case, batch rows time
every kept case exactly once, and case strings keep the list form that case
IDs use.
"""

from __future__ import annotations

import ast
import csv
import importlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from collector import skips
from collector.case_generator import without_models
from collector.op_plans import OP_PLANS, GemmCase, ModuleCase, OpPlan, _Run, _runners, plan, summarize
from collector.provenance import case_plan_hash

pytestmark = pytest.mark.unit

PKG_ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("sglang", "trtllm", "vllm")
SMS = (80, 89, 90, 100, 103, 120)


@pytest.fixture(autouse=True)
def _full_plan(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)


def _entry(**keys) -> dict:
    """The skips.py entry with these keys and values."""
    return next(entry for entry in skips.SKIPS if all(entry.get(key) == value for key, value in keys.items()))


def _registry(backend: str) -> list:
    """The backend's registry entries, with those of its wide-EP registry."""
    return [
        *importlib.import_module(f"collector.{backend}.registry").REGISTRY,
        *importlib.import_module(f"collector.wideep.{backend}.registry").REGISTRY,
    ]


def _registered_ops(backend: str) -> list[str]:
    """The ops with an op plan that this backend's registries collect."""
    return sorted({entry.op for entry in _registry(backend)} & set(OP_PLANS))


def _getters():
    for backend in BACKENDS:
        for entry in _registry(backend):
            if entry.op in OP_PLANS:
                yield pytest.param(backend, entry, id=f"{entry.op}-{backend}")


@pytest.mark.parametrize(("backend", "entry"), list(_getters()))
def test_getter_is_the_one_line_plan_call(backend, entry):
    modules = [entry.module] if entry.module else [route.module for route in entry.versions]
    for module in modules:
        source = (PKG_ROOT / Path(*module.split("."))).with_suffix(".py").read_text()
        getter = next(
            node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == entry.get_func
        )
        assert ast.unparse(getter.body) == f"return plan_cases('{entry.op}', '{backend}', get_sm_version())"


@pytest.mark.parametrize("op", sorted(OP_PLANS))
@pytest.mark.parametrize("sm", SMS)
def test_backends_differ_only_by_entries_that_name_them(op, sm):
    spec = OP_PLANS[op]
    plans = {backend: plan(op, backend, sm) for backend in BACKENDS}
    union = set().union(*(set(op_plan.cases) for op_plan in plans.values()))
    for backend, op_plan in plans.items():
        missing = union - set(op_plan.cases)
        if not missing:
            continue
        with without_models(skips.models(backend, op, sm)):
            expanded = set(spec.expand())
        # A model entry leaves the case out before cases merge ...
        if missing - expanded:
            assert skips.model_entries(backend, op, sm), (backend, sorted(missing - expanded, key=str)[:3])

        # ... a case entry leaves out more ...
        dtypes_of = spec.dtypes
        _kept, skipped = skips.split(missing & expanded, backend, op, sm, dtypes_of, dispatch_entries=False)
        assert all("backend" in entry for _case, entry, _region in skipped), backend
        # ... and entries that name a dispatch leave out the rest on every kernel it runs on.
        left = (missing & expanded) - {case for case, _entry, _region in skipped}
        if left:
            dispatch = spec.dispatch[backend]
            runs = [_Run(case, runner) for case in left for runner in _runners(dispatch(case, sm))]
            kept_runs, skipped_runs = skips.split(
                runs, backend, op, sm, lambda run: dtypes_of(run.case), dispatch_entries=True
            )
            assert not kept_runs, (backend, kept_runs[:3])
            assert all("backend" in entry for _run, entry, _region in skipped_runs), backend


@pytest.mark.parametrize("op", sorted(OP_PLANS))
@pytest.mark.parametrize("backend", BACKENDS)
def test_every_left_out_case_is_counted_under_its_entry(op, backend):
    expanded = OP_PLANS[op].expand()
    entries = {repr(entry) for entry in skips.SKIPS}
    for sm in SMS:
        op_plan = plan(op, backend, sm)
        assert len(op_plan.cases) + sum(region.cases for region in op_plan.removed) == len(expanded)
        for region in [*op_plan.removed, *op_plan.removed_runs]:
            assert region.reason in entries, region


@pytest.mark.parametrize("op", sorted(OP_PLANS))
def test_case_string_keeps_the_list_form_of_case_ids(op):
    case = OP_PLANS[op].expand()[0]
    listed = ast.literal_eval(str(case))

    assert str(case) == repr(case)
    assert isinstance(listed, list)
    assert [tuple(value) if isinstance(value, list) else value for value in listed] == list(case)


@pytest.mark.parametrize("op", sorted(OP_PLANS))
def test_cases_sharing_a_module_build_are_contiguous(op):
    spec = OP_PLANS[op]
    cases = plan(op, "vllm", 100).cases
    assert len(cases) == len(set(cases))
    seen, previous = set(), None
    for case in cases:
        group = tuple(getattr(case, field) for field in spec.group_fields)
        if group != previous:
            assert group not in seen, f"group {group} is split"
            seen.add(group)
            previous = group


def test_every_planned_model_ships_its_config(monkeypatch):
    """A model a plan reads has a packaged config.json; the planner raises on a missing one
    rather than planning from an empty config (case_authoring: unresolvable declarations fail)."""
    from collector import op_plans

    requested: dict[str, set[str]] = defaultdict(set)
    real = op_plans._packaged_config

    def spy(model_path, kind="config"):
        requested[kind].add(model_path)
        return real(model_path, kind)

    monkeypatch.setattr(op_plans, "_packaged_config", spy)
    for name in OP_PLANS:
        for backend in ("vllm", "sglang", "trtllm"):
            plan(name, backend, 100)
    config_root = Path(op_plans._AIC_MODEL_CONFIG_DIR)
    assert requested["config"]
    for model_path in sorted(requested["config"]):
        assert (config_root / f"{model_path.replace('/', '--')}_config.json").is_file(), model_path


def test_packaged_config_fails_loudly_only_for_the_config(monkeypatch, tmp_path):
    from collector import op_plans

    (tmp_path / "org--model_config.json").write_text('{"max_position_embeddings": 4096}')
    monkeypatch.setattr(op_plans, "_AIC_MODEL_CONFIG_DIR", str(tmp_path))
    assert op_plans._max_position("org/model") == 4096
    assert op_plans._packaged_config("org/model", "hf_quant_config") == {}
    with pytest.raises(FileNotFoundError, match="org/missing has no packaged config.json"):
        op_plans._packaged_config("org/missing")
    (tmp_path / "org--broken_config.json").write_text("{not json")
    with pytest.raises(ValueError):
        op_plans._expand_deepep()


def test_op_plan_defaults_come_from_factories():
    """A mappingproxy default is rejected by dataclasses on Python 3.11, where it is unhashable."""
    import dataclasses

    for field in dataclasses.fields(OpPlan):
        assert field.default is dataclasses.MISSING, field.name
    assert dict(OpPlan.__dataclass_fields__["dispatch"].default_factory()) == {}


def test_rows_may_not_drop_a_case(monkeypatch):
    dropping = OpPlan(
        GemmCase,
        lambda: [GemmCase("bfloat16", 1, 128, 128)],
        dtype_field="gemm_type",
        group_fields=("n", "k"),
        rows={"vllm": lambda case, sm, runner: []},
    )
    monkeypatch.setitem(OP_PLANS, "probe", dropping)

    with pytest.raises(ValueError, match="rows dropped case"):
        _ = plan("probe", "vllm", 100).rows
    assert plan("probe", "sglang", 100).rows == [GemmCase("bfloat16", 1, 128, 128)]


def test_batch_rows_must_time_every_kept_case_exactly_once(monkeypatch):
    cases = [GemmCase("bfloat16", 1, 128, 128), GemmCase("bfloat16", 2, 128, 128)]

    def probe(batch_rows):
        op_plan = OpPlan(
            GemmCase, lambda: cases, dtype_field="gemm_type", group_fields=("n", "k"), batch_rows={"vllm": batch_rows}
        )
        monkeypatch.setitem(OP_PLANS, "probe", op_plan)
        return plan("probe", "vllm", 100)

    assert probe(lambda kept, sm: [([[case.m for case in kept], 128, 128], tuple(kept))]).rows == [[[1, 2], 128, 128]]
    for wrong in (
        lambda kept, sm: [(["m=1"], (kept[0],))],
        lambda kept, sm: [(["all"], tuple(kept)), (["m=1 again"], (kept[0],))],
        lambda kept, sm: [(["all"], (*kept, GemmCase("fp8", 1, 128, 128)))],
    ):
        with pytest.raises(ValueError, match="exactly once"):
            _ = probe(wrong).rows


def test_a_backend_declares_batch_rows_or_rows_not_both():
    with pytest.raises(ValueError, match="not both"):
        OpPlan(
            GemmCase,
            list,
            dtype_field="gemm_type",
            group_fields=("n", "k"),
            rows={"vllm": lambda case, sm, runner: [case]},
            batch_rows={"vllm": lambda cases, sm: []},
        )


@pytest.mark.parametrize(
    "op", ["mla_context_module", "mla_generation_module", "dsa_context_module", "dsa_generation_module"]
)
@pytest.mark.parametrize("backend", ["trtllm", "vllm"])
def test_module_rows_list_the_points_of_their_cases(op, backend):
    op_plan = plan(op, backend, 100)
    points = []
    for series, *key, attn_type in op_plan.rows:
        assert attn_type == op.split("_")[0]
        # TRT-LLM rows list (seq_len, prefix_len) points; vLLM rows list lengths, or the
        # (seq_len, prefix_len) points of a DSA context module.
        pairs = series if backend == "trtllm" or op == "dsa_context_module" else [[length, 0] for length in series]
        assert pairs == sorted(pairs, reverse=True)
        points += [ModuleCase(length, *key, prefix_len) for length, prefix_len in pairs]
    assert len(points) == len(set(points)) == len(op_plan.cases)
    assert set(points) == set(op_plan.cases)


# The checkpoint each (architecture, projection GEMM) is timed on.
DSA_MODELS = {
    "deepseek-ai/DeepSeek-V3.2": ("bfloat16", "fp8_block"),
    "nvidia/DeepSeek-V3.2-NVFP4": ("nvfp4",),
    "zai-org/GLM-5.2": ("bfloat16", "nvfp4"),
    "zai-org/GLM-5.2-FP8": ("fp8_block",),
}


def test_dsa_plan_times_the_longest_context_checkpoint_of_each_projection_gemm():
    from collector.op_plans import _sparse_module_models

    assert {spec.model_path: gemms for spec, gemms in _sparse_module_models("dsa")} == DSA_MODELS
    by_model = defaultdict(set)
    for case in OP_PLANS["dsa_context_module"].expand():
        by_model[case.model_path].add(case.gemm_type)
    assert {model: tuple(sorted(gemms)) for model, gemms in by_model.items()} == DSA_MODELS


def test_dsa_context_points_follow_chunked_prefill():
    from collector.case_generator import get_mla_module_sweep_spec

    sweep = get_mla_module_sweep_spec("plan")
    anchors = [
        0, 1, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288,
        16384, 32768, 65536, 131072, 262144, 524288, 1048575,
    ]  # fmt: skip
    assert sweep.context_prefix_lengths == anchors
    max_position = {"deepseek-ai/DeepSeek-V3.2": 163840, "zai-org/GLM-5.2": 1048576}
    points = defaultdict(set)
    for case in OP_PLANS["dsa_context_module"].expand():
        if case.model_path in max_position and case.num_heads == 8:
            points[case.model_path].add((case.batch_size, case.seq_len, case.prefix_len))
    for model, ceiling in max_position.items():
        expected = {
            (b, s, p)
            for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)
            for s in (1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384)
            for p in anchors
            # new tokens per forward, KV tokens per forward, the model's context
            if b * s <= 16384 and b * (s + p) <= 1048576 and p + s <= ceiling
        }
        assert points[model] == expected, model
    assert (1, 1, 1048575) in points["zai-org/GLM-5.2"]
    assert max(p + s for _b, s, p in points["deepseek-ai/DeepSeek-V3.2"]) <= 163840


def test_dsa_decode_reaches_the_long_context_at_small_batch():
    lengths = defaultdict(set)
    for case in OP_PLANS["dsa_generation_module"].expand():
        lengths[case.model_path].add((case.batch_size, case.seq_len))
    long_kv = {(b, s) for b in (1, 2, 4) for s in (262144, 524288, 1048575)}
    assert long_kv <= lengths["zai-org/GLM-5.2"]
    # A KV length leaves room for the decoded token within the model's context.
    assert max(s for _b, s in lengths["deepseek-ai/DeepSeek-V3.2"]) == 131072
    assert lengths["zai-org/GLM-5.2"] - long_kv == lengths["deepseek-ai/DeepSeek-V3.2"]


@pytest.mark.parametrize("op", ["dsa_context_module", "dsa_generation_module"])
def test_dsa_skips_keep_each_backends_precisions(op):
    def combos(backend, sm):
        return {(case.kv_cache_dtype, case.gemm_type) for case in plan(op, backend, sm).cases}

    gemms = {"bfloat16", "fp8_block", "nvfp4"}
    both_kv = {(kv, gemm) for kv in ("bfloat16", "fp8") for gemm in gemms}
    # TRT-LLM: FP8 KV from the SM100 family on, and nothing on SM120, where it needs the
    # fp8_ds_mla pool; vLLM: both KV dtypes; NVFP4 from SM100.
    assert combos("trtllm", 90) == {("bfloat16", "bfloat16"), ("bfloat16", "fp8_block")}
    assert combos("vllm", 90) == {pair for pair in both_kv if pair[1] != "nvfp4"}
    for sm in (100, 103):
        assert combos("trtllm", sm) == combos("vllm", sm) == both_kv
    assert combos("vllm", 120) == both_kv
    assert combos("trtllm", 120) == set()
    # SGLang times each checkpoint on its own projection GEMM, with no NVFP4 projections.
    for sm in (90, 100):
        assert {(case.model_path, case.gemm_type) for case in plan(op, "sglang", sm).cases} == {
            ("deepseek-ai/DeepSeek-V3.2", "fp8_block"),
            ("zai-org/GLM-5.2", "bfloat16"),
            ("zai-org/GLM-5.2-FP8", "fp8_block"),
        }
        assert {case.kv_cache_dtype for case in plan(op, "sglang", sm).cases} == {"bfloat16", "fp8"}


@pytest.mark.parametrize("op", ["msa_context_module", "msa_generation_module"])
def test_msa_skips_keep_each_backends_precisions(op):
    def combos(backend, sm):
        return {(case.kv_cache_dtype, case.gemm_type) for case in plan(op, backend, sm).cases}

    def gemms(*names):
        return {(kv, gemm) for kv in ("bfloat16", "fp8") for gemm in names}

    # TRT-LLM: BF16 KV only. vLLM: both KV dtypes. fp8_block from SM89, NVFP4 from SM100.
    # SGLang: BF16 projections only, FP8 KV from SM90.
    expected = {
        80: ({("bfloat16", "bfloat16")}, {("bfloat16", "bfloat16")}, {("bfloat16", "bfloat16")}),
        89: (
            {("bfloat16", "bfloat16"), ("bfloat16", "fp8_block")},
            gemms("bfloat16", "fp8_block"),
            {("bfloat16", "bfloat16")},
        ),
        90: (
            {("bfloat16", "bfloat16"), ("bfloat16", "fp8_block")},
            gemms("bfloat16", "fp8_block"),
            gemms("bfloat16"),
        ),
    }
    for sm in (100, 103, 120):
        expected[sm] = (
            {("bfloat16", gemm) for gemm in ("bfloat16", "fp8_block", "nvfp4")},
            gemms("bfloat16", "fp8_block", "nvfp4"),
            gemms("bfloat16"),
        )
    for sm, (trtllm, vllm, sglang) in expected.items():
        assert (combos("trtllm", sm), combos("vllm", sm), combos("sglang", sm)) == (trtllm, vllm, sglang), sm


def test_msa_plan_times_the_model_shards_on_the_long_context_grid():
    cases = OP_PLANS["msa_context_module"].expand()
    assert {case.model_path for case in cases} == {"MiniMaxAI/MiniMax-M3"}
    # MiniMax-M3's 64 heads at TP 1 to 16.
    assert {case.num_heads for case in cases} == {64, 32, 16, 8, 4}
    dsa = {
        (case.batch_size, case.seq_len, case.prefix_len)
        for case in OP_PLANS["dsa_context_module"].expand()
        if case.model_path == "zai-org/GLM-5.2"
    }
    # Both 1M-context models share the long-context grid.
    assert {(case.batch_size, case.seq_len, case.prefix_len) for case in cases} == dsa
    decode = {(case.batch_size, case.seq_len) for case in OP_PLANS["msa_generation_module"].expand()}
    assert {(b, s) for b in (1, 2, 4) for s in (262144, 524288, 1048575)} <= decode


def test_trtllm_msa_stops_at_a_32k_prefix_where_it_runs_the_triton_reference():
    def prefixes(sm):
        return {case.prefix_len for case in plan("msa_context_module", "trtllm", sm).cases}

    for sm in (89, 90, 120):
        assert max(prefixes(sm)) == 32768
    for sm in (100, 103):
        assert max(prefixes(sm)) == 1048575
    assert max(case.prefix_len for case in plan("msa_context_module", "vllm", 90).cases) == 1048575


def test_msa_rows_carry_their_points():
    trtllm = plan("msa_context_module", "trtllm", 100)
    # run_msa_module_series_worker: points, then run_msa_module_worker's other fields.
    assert {len(row) for row in trtllm.rows} == {7}
    assert sum(len(row[0]) for row in trtllm.rows) == len(trtllm.cases)
    # vLLM times one point per task: the case is run_msa_module_worker's fields.
    vllm = plan("msa_generation_module", "vllm", 100)
    assert vllm.rows == vllm.cases
    # SGLang: one ModelRunner per context batch, or one for every decode point, with the
    # target TP of the heads and the [batch, seq_len, prefix_len] points.
    for op, batches in (("msa_context_module", {1, 2, 4, 8, 16, 32, 64, 128, 256}), ("msa_generation_module", {0})):
        sglang = plan(op, "sglang", 100)
        assert {row[1] for row in sglang.rows} == batches
        assert {(row[2], row[7]) for row in sglang.rows} == {(64, 1), (32, 2), (16, 4), (8, 8), (4, 16)}
        points = [tuple(point) for row in sglang.rows for point in row[8]]
        assert sorted(points) == sorted((c.batch_size, c.seq_len, c.prefix_len) for c in sglang.cases)
        for row in sglang.rows:
            assert row[8] == sorted(row[8], key=lambda point: (point[2], point[0], point[1]))


def test_sglang_dsa_dispatch_is_the_backend_serving_picks():
    assert plan("dsa_context_module", "sglang", 90).by_dispatch.keys() == {"flashmla_kv", "flashmla_sparse"}
    assert plan("dsa_context_module", "sglang", 100).by_dispatch.keys() == {"trtllm", "flashmla_sparse"}
    assert plan("dsa_generation_module", "sglang", 90).by_dispatch.keys() == {"flashmla_kv", "fa3"}
    assert plan("dsa_generation_module", "sglang", 100).by_dispatch.keys() == {"trtllm"}


def test_batch_rows_time_every_dispatch_of_a_case_once(monkeypatch):
    cases = [GemmCase("bfloat16", 1, 128, 128), GemmCase("bfloat16", 2, 128, 128)]
    op_plan = OpPlan(
        GemmCase,
        lambda: cases,
        dtype_field="gemm_type",
        group_fields=("n", "k"),
        dispatch={"sglang": lambda case, sm: ("a", "b") if case.m == 1 else "a"},
        batch_rows={"sglang": lambda runs, sm: [([run.dispatch, run.m], (run,)) for run in runs]},
    )
    monkeypatch.setitem(OP_PLANS, "probe", op_plan)
    assert plan("probe", "sglang", 100).rows == [["a", 1], ["b", 1], ["a", 2]]


def test_module_skips_remove_points_per_backend():
    def rows(op, backend, sm):
        return plan(op, backend, sm).rows

    # FP8 prefill queries: vLLM on the SM100 family only, never TRT-LLM.
    assert {row[4] for row in rows("mla_context_module", "vllm", 100)} == {"bfloat16", "fp8"}
    assert {row[4] for row in rows("mla_context_module", "vllm", 90)} == {"bfloat16"}
    assert {row[4] for row in rows("mla_context_module", "trtllm", 100)} == {"bfloat16"}
    # SM89: vLLM keeps BF16 KV only, and TRT-LLM has no MLA kernel below Hopper.
    for op in ("mla_context_module", "mla_generation_module"):
        assert {row[3] for row in rows(op, "vllm", 89)} == {"bfloat16"}
        assert rows(op, "trtllm", 89) == []
    # Where neither backend leaves anything out, both time the same points.
    assert plan("mla_generation_module", "vllm", 100).cases == plan("mla_generation_module", "trtllm", 100).cases


@pytest.mark.parametrize(
    ("sm", "dtypes"),
    [
        (80, ["bfloat16"]),
        (89, ["bfloat16", "fp8", "fp8_block"]),
        (90, ["bfloat16", "fp8", "fp8_block"]),
        (100, ["bfloat16", "fp8", "fp8_block", "nvfp4"]),
        (120, ["bfloat16", "fp8", "fp8_block", "nvfp4"]),
    ],
)
def test_gemm_floors_remove_whole_dtypes(sm, dtypes):
    assert sorted({case.gemm_type for case in plan("gemm", "vllm", sm).cases}) == dtypes


def test_gemm_sglang_plans_fp8_block_on_every_fp8_sm():
    # SGLang times fp8_block through serving's dispatch (Triton on SM89, CUTLASS
    # on SM120, DeepGEMM on SM90/SM10x), so no SM leaves the dtype out.
    for sm in (89, 90, 100, 103, 120):
        op_plan = plan("gemm", "sglang", sm)
        assert "fp8_block" in {case.gemm_type for case in op_plan.cases}
        assert not [region for region in op_plan.removed if region.region == "dtype=fp8_block"]


def test_summary_counts_per_dtype_and_hashes_the_planned_ids():
    op_plan = plan("gemm", "sglang", 89)
    summary = summarize("gemm", op_plan, case_id=lambda row: f"sglang.gemm:run_gemm:{row}")
    shapes = summary["by_dtype"]["bfloat16"]

    assert summary["cases"] == 3 * shapes
    assert summary["by_dtype"] == {"bfloat16": shapes, "fp8": shapes, "fp8_block": shapes}
    assert summary["groups"] == len({(case.gemm_type, case.n, case.k) for case in op_plan.cases})
    assert summary["removed"] == [
        {"region": "dtype=nvfp4", "cases": shapes, "reason": repr(_entry(dtype=["nvfp4", "w4a8_mxfp4_mxfp8"]))}
    ]
    assert summary["plan_hash"] == case_plan_hash([f"sglang.gemm:run_gemm:{case}" for case in op_plan.cases])
    assert "rows" not in summary


def test_plan_only_reports_op_plans_and_writes_the_table(tmp_path, monkeypatch):
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PKG_ROOT / "collector" / "collect.py"),
            *("--backend", "vllm", "--model-cases-full", "--sm", "89", "--ops", *_registered_ops("vllm")),
            *("--plan-only", "--plan-table", str(tmp_path)),
        ],
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    plans = json.loads(result.stdout)["plans"]
    assert set(plans) == set(_registered_ops("vllm"))
    gemm = plans["gemm"]
    shapes = gemm["by_dtype"]["bfloat16"]
    assert gemm["by_dtype"] == {"bfloat16": shapes, "fp8": shapes, "fp8_block": shapes}
    nvfp4 = repr(_entry(dtype=["nvfp4", "w4a8_mxfp4_mxfp8"]))
    assert gemm["removed"] == [{"region": "dtype=nvfp4", "cases": shapes, "reason": nvfp4}]
    rows = (tmp_path / "vllm_sm89_gemm.csv").read_text().splitlines()
    assert rows[0] == "gemm_type,m,n,k"
    assert len(rows) == 1 + gemm["cases"]
    for op in plans:
        assert len((tmp_path / f"vllm_sm89_{op}.csv").read_text().splitlines()) == 1 + plans[op]["cases"], op

    # A run records an {id, params} row (mHC) under its own id, so the plan hash covers those ids.
    from collector.op_plans import plan_cases
    from collector.provenance import case_plan_hash

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    mhc_ids = [row["id"] for row in plan_cases("mhc_module", "vllm", 89)]
    assert plans["mhc_module"]["plan_hash"] == case_plan_hash(mhc_ids)


def test_planning_imports_no_framework():
    """--plan-only runs on any CPU: no op plan imports SGLang, vLLM, TensorRT-LLM or torch."""
    script = """
import sys
for name in ("sglang", "vllm", "tensorrt_llm", "torch"):
    sys.modules[name] = None
from collector.op_plans import OP_PLANS, plan
for op in OP_PLANS:
    for backend in ("sglang", "trtllm", "vllm"):
        assert plan(op, backend, 100).cases, (op, backend)
"""
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PKG_ROOT, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_moe_plan_reports_the_runner_sglang_picks_on_each_sm():
    sm100 = plan("moe", "sglang", 100)
    assert sum(sm100.by_dispatch.values()) == len(sm100.cases)
    assert {"triton", "flashinfer_trtllm", "flashinfer_mxfp4"} <= set(sm100.by_dispatch)
    assert "marlin" in plan("moe", "sglang", 90).by_dispatch
    assert plan("moe", "vllm", 100).by_dispatch == {}


def test_moe_plan_runs_gpt_oss_w4a16_on_the_runner_sglang_picks_by_ep():
    # SGLang's GPT-OSS override serves triton_kernel at EP1 and keeps auto
    # (Triton) at EP>1 on SM89/90 (model_overrides/gpt_oss.py:102-120). From SM100
    # it serves flashinfer_mxfp4 with MXFP8 activations at every EP (:64-75), so
    # SM100/103/120 plan no GPT-OSS W4A16 case (collector/skips.py) and the W4A8
    # cases run flashinfer_mxfp4.
    def gpt_oss_runners(sm):
        sglang = plan("moe", "sglang", sm)
        return {
            (case.ep == 1, runners)
            for case, runners in zip(sglang.cases, sglang.dispatches, strict=True)
            if case.model_name.startswith("openai/gpt-oss") and case.moe_type == "w4a16_mxfp4"
        }

    for sm in (89, 90):
        assert gpt_oss_runners(sm) == {(True, ("triton_kernel",)), (False, ("triton",))}
        sglang = plan("moe", "sglang", sm)
        kernel_cases = [
            case for case, runners in zip(sglang.cases, sglang.dispatches, strict=True) if "triton_kernel" in runners
        ]
        assert kernel_cases and all(case.ep == 1 and "gpt-oss" in case.model_name for case in kernel_cases)
    for sm in (100, 103, 120):
        assert gpt_oss_runners(sm) == set()
        assert "triton_kernel" not in plan("moe", "sglang", sm).by_dispatch
    for sm in (100, 103):
        sglang = plan("moe", "sglang", sm)
        w4a8 = {
            runners
            for case, runners in zip(sglang.cases, sglang.dispatches, strict=True)
            if case.model_name.startswith("openai/gpt-oss") and case.moe_type == "w4a8_mxfp4_mxfp8"
        }
        assert w4a8 == {("flashinfer_mxfp4",)}


@pytest.mark.parametrize("sm", SMS)
def test_moe_ep_skips_keep_each_backends_builds(sm):
    sglang, trtllm = plan("moe_ep", "sglang", sm), plan("moe_ep", "trtllm", sm)
    # SGLang times its checkpoint's fp8_block MoE without an EPLB layout.
    assert all(case.moe_type == "fp8_block" and case.own_quant for case in sglang.cases)
    assert all(not case.use_eplb and case.num_slots == case.num_experts for case in sglang.cases)
    assert bool(sglang.cases) == (sm >= 89)
    # TRT-LLM times fp8_block on Hopper (DeepGEMM's JIT) and nvfp4 from SM100.
    modes = {case.moe_type for case in trtllm.cases}
    assert modes == ({"nvfp4"} if sm >= 100 else {"fp8_block"} if sm == 90 else set())


def test_moe_ep_keeps_one_case_per_table_key_owned_by_a_checkpoint_that_plans_it():
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model, moe_model_allows_quantization

    cases = OP_PLANS["moe_ep"].expand()
    keys = [case._replace(model_name="", own_quant=False) for case in cases]
    assert len(keys) == len(set(keys))
    models_of = defaultdict(set)
    for layer in get_common_moe_test_cases():
        if is_wideep_moe_model(layer.model_name) and layer.tp == 1 and layer.ep > 1:
            models_of[(layer.hidden_size, layer.inter_size, layer.topk, layer.num_experts)].add(layer.model_name)
    shared = [case for case in cases if len(models_of[case[1:5]]) > 1]
    assert shared
    for case in shared:
        if not case.own_quant:
            planning = [
                model for model in models_of[case[1:5]] if moe_model_allows_quantization("sglang", model, case.moe_type)
            ]
            assert not planning, (case, planning)


def test_moe_ep_rows_keep_each_backends_run_arguments():
    sglang = plan("moe_ep", "sglang", 90)
    case = sglang.cases[0]
    assert sglang.rows[0] == [
        case.num_experts // case.ep,
        case.ep,
        case.hidden_size,
        case.inter_size,
        case.topk,
        case.num_experts,
        case.num_experts,
        "fp8_block",
        case.model_name,
    ]
    trtllm = plan("moe_ep", "trtllm", 100)
    case = trtllm.cases[0]
    assert len(case.power_law_alphas) > 1
    assert trtllm.rows[: len(case.power_law_alphas)] == [
        [case.moe_type, "cutlass", list(case.num_tokens), *case[1:5], 1, case.ep, False, case.model_name]
        + ["power_law", alpha, case.use_eplb, case.num_slots, True]
        for alpha in case.power_law_alphas
    ]


def test_deepep_plans_every_packaged_moe_shape_once():
    for op in ("deepep_ll", "deepep_normal"):
        op_plan = plan(op, "sglang", 100)
        assert op_plan.cases == sorted(set(op_plan.cases))
        assert len(op_plan.cases) == 25
        assert op_plan.rows == [case._asdict() for case in op_plan.cases]
        assert all(list(row) == ["hidden_size", "num_experts", "topk"] for row in op_plan.rows)
        # The shapes run on every SM: a kernel limit fails its shape at run time.
        assert plan(op, "sglang", 80).cases == op_plan.cases


def test_plan_table_carries_the_dispatch_of_each_case(tmp_path):
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PKG_ROOT / "collector" / "collect.py"),
            *("--backend", "sglang", "--model-cases-full", "--sm", "90", "--ops", "moe"),
            *("--plan-only", "--plan-table", str(tmp_path)),
        ],
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    with (tmp_path / "sglang_sm90_moe.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    plans = json.loads(result.stdout)["plans"]["moe"]
    assert len(table) == plans["cases"]
    assert Counter(row["dispatch"] for row in table) == Counter(plans["by_dispatch"])
    # SGLang serves Kimi-K2.5 INT4 on SM90 through Marlin.
    assert {row["dispatch"] for row in table if row["model_name"] == "moonshotai/Kimi-K2.5"} == {"marlin"}


def test_plan_only_report_includes_the_wideep_ops():
    """The WideEP ops have op plans but live in the wideep registries; the report merges them like a run does."""
    script = """
import json
from collector.collect import _op_plan_report
report = _op_plan_report("sglang", ["gemm", "moe_ep", "deepep_ll", "deepep_normal"], 100, None)
print(json.dumps({op: summary["cases"] for op, summary in report.items()}))
"""
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    env["PYTHONPATH"] = os.pathsep.join([str(PKG_ROOT), str(PKG_ROOT / "collector"), str(PKG_ROOT / "src")])
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PKG_ROOT, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    cases = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(cases) == {"gemm", "moe_ep", "deepep_ll", "deepep_normal"}
    assert cases["moe_ep"] > 0
