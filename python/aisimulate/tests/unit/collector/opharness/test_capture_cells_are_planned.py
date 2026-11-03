# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A capture that names its cell through the op plan (op_smoke.planned_row,
run_planned or --select) must name exactly one planned case on the SMs it is graded
on: a capture of a cell the collector never collects grades nothing (the GLM-5
DSA module capture used to time 128 heads on GLM-5, a cell no plan holds)."""

import ast
import importlib.util
from pathlib import Path

import pytest
from collector.op_plans import plan

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"
CAPTURES = COMPONENTS / "captures"
GRADED_SMS = (90, 100)


def _selections(path: Path) -> list[tuple[str, str, dict]]:
    """(backend, op, select) of every planned_row / run_planned call and op_smoke --select argv in a capture."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("planned_row", "run_planned"):
            backend, op = (ast.literal_eval(arg) for arg in node.args[:2])
            found.append((backend, op, {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}))
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.List)
            and any(getattr(t, "attr", None) == "argv" for t in node.targets)
        ):
            argv = [ast.literal_eval(elt) for elt in node.value.elts]
            if "--select" in argv:
                select = dict(pair.split("=", 1) for pair in argv[argv.index("--select") + 1].split(","))
                found.append((argv[argv.index("--backend") + 1], argv[argv.index("--op") + 1], select))
    return found


PLANNED_CAPTURES = sorted((path.name, selection) for path in CAPTURES.glob("*.py") for selection in _selections(path))


def test_the_plan_derived_captures_are_found():
    assert len(PLANNED_CAPTURES) >= 44


@pytest.mark.parametrize("sm", GRADED_SMS)
@pytest.mark.parametrize(("name", "selection"), PLANNED_CAPTURES, ids=[name for name, _ in PLANNED_CAPTURES])
def test_capture_names_one_planned_case(name, selection, sm):
    backend, op, select = selection
    cases = plan(op, backend, sm).cases
    hits = [case for case in cases if all(str(getattr(case, f)) == str(v) for f, v in select.items())]
    assert len(hits) == 1, f"{name}: {backend} {op} sm{sm} select {select} matches {len(hits)} planned cases"


def test_planned_row_restricts_a_batch_row_to_the_selected_point():
    spec = importlib.util.spec_from_file_location("op_smoke_t", COMPONENTS / "op_smoke.py")
    op_smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op_smoke)
    select = dict(model_path="deepseek-ai/DeepSeek-V3.2", kv_cache_dtype="fp8", compute_dtype="bfloat16")
    select.update(gemm_type="fp8_block", num_heads=128, batch_size=1, seq_len=4096, prefix_len=0)
    hopper = op_smoke.planned_row("sglang", "dsa_context_module", 90, **select)
    blackwell = op_smoke.planned_row("sglang", "dsa_context_module", 100, **select)
    assert hopper[-1] == blackwell[-1] == [[1, 4096, 0]]
    # the FP8-KV backend the plan pins per SM reaches the capture
    assert (hopper[10], blackwell[10]) == ("flashmla_kv", "trtllm")
    with pytest.raises(SystemExit, match="matches 5 planned cases"):
        op_smoke.planned_row(
            "sglang", "dsa_context_module", 90, **{k: v for k, v in select.items() if k != "num_heads"}
        )
