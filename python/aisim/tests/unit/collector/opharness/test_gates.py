# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gates.yaml names every path gate as one planned collector case: the probe.trace_case
shard it schedules must select exactly one planned case and fit the pipeline's case-filter
charset. A gate whose case is planned on no
SM grades nothing (the GLM-5 DSA module capture once timed 128 heads, a cell no plan holds)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture(scope="module")
def gates():
    spec = importlib.util.spec_from_file_location("gates_t", COMPONENTS / "gates.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gates_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ids():
    import yaml

    return [g["gate"] + "@" + g["backend"] for g in yaml.safe_load((COMPONENTS / "gates.yaml").read_text())["gates"]]


@pytest.mark.parametrize("gate_id", _ids())
def test_every_gate_names_one_planned_case(gates, gate_id):
    name, backend = gate_id.split("@")
    (gate,) = [g for g in gates.load()["gates"] if g["gate"] == name and g["backend"] == backend]
    cases = gates._cases(gate)
    planned_on = [sm for sm in (90, 100) if gates._matches(backend, gate["op"], sm, gate["select"])]
    assert planned_on, f"{gate_id}: select {gate['select']} matches no single planned case on SM90 or SM100"
    for _name, select in cases:
        op = select.pop("op", gate["op"])
        # raises when it does not fit the pipeline
        gates.case_filter(backend, op, select, gate.get("env"), gate.get("row"))
    # the traced case resolves to one row on every SM it is planned on (a case the plan runs
    # as several rows needs the gate's `row`); rows that build framework objects (SGLang and
    # TRT-LLM MLA) cannot be built on a CPU and are checked when the probe runs
    from collector.probes import trace_case

    for sm in planned_on:
        row = {} if gate.get("row") is None else {"row": gate["row"]}
        try:
            trace_case.planned_row(backend, gate["op"], sm, **dict(gate["select"]), **row)
        except ModuleNotFoundError:
            continue


def test_shard_plans_parse_as_pipeline_plans(gates, tmp_path):
    plans = gates.shard_plans("sm100", "sglang")
    shards = [s for plan in plans.values() for s in plan["shards"]]
    assert {plan["backend"] for plan in plans.values()} == {"sglang"} and shards
    assert all(s["op"] == "probe.trace_case" and s["case_filter"].startswith("backend=sglang,op=") for s in shards)
    assert len({s["name"] for s in shards}) == len(shards)


def test_a_gate_runs_at_the_release_its_op_is_collected_with(gates):
    """vLLM families pinned apart from the 0.30.0 default (gemm, moe, attention, mla, ... at
    0.32.0) would be traced in the default image and graded against default-release serving
    records if a gate ran at the default. Each (backend, release) is its own job; an also case
    shares its gate's release."""
    plans = gates.shard_plans("sm90", "vllm")
    gemm_release = gates.op_version("vllm", "gemm")
    assert set(plans) == {("vllm", gates.gate_version(g)) for g in gates.gates_in_scope("vllm", "sm90")}
    gemm = {s["name"] for s in plans[("vllm", gemm_release)]["shards"]}
    assert "gemm_fp8block_DeepSeek-V3.2" in gemm
    for g in gates.load()["gates"]:
        assert {gates.op_version(g["backend"], op) for op in gates._ops(g)} == {gates.gate_version(g)}, g["gate"]
    # the pipeline runs a shard plan on the default image unless told otherwise: every release
    # a gate runs at has a pinned image to name at trigger time
    assert all("@sha256:" in gates.release_image(b, v) for b, v in plans)


def test_one_op_is_planned_across_the_frameworks(gates, tmp_path):
    plans = gates.shard_plans("sm100", op="dsa_context_module")
    assert {backend for backend, _ in plans} == {"vllm", "sglang", "trtllm"}
    assert all("op=dsa_context_module," in s["case_filter"] for p in plans.values() for s in p["shards"])
    # a campaign's serving shards of the same (backend, release) join the job's plan
    camp = tmp_path / "camp" / "shards" / "tp1"
    camp.mkdir(parents=True)
    serving = {"op": "probe.serving", "name": "abc", "case_filter": "spec=http://share/abc.json"}
    (camp / "sglang-0.5.21.yaml").write_text(yaml.safe_dump({"backend": "sglang", "shards": [serving]}))
    assert gates.serving_shards(tmp_path / "camp", "sglang", "0.5.21") == [serving]
    assert gates.serving_shards(tmp_path / "camp", "vllm", "0.30.0") == []


@pytest.mark.parametrize(
    "op, select, roles, phase",
    [
        ("gemm", {}, ("gemm",), None),
        ("mla_bmm_gen_post", {}, ("gemm",), None),
        ("moe", {}, ("moe_gemm", "routing"), None),
        ("gdn", {"phase": "generation"}, ("linear_attention",), "decode"),
        # the gate name used to carry this; glm5_* and mamba2_* names matched no pattern and
        # graded unscoped
        ("mamba2", {"phase": "context"}, ("linear_attention",), "prefill"),
        ("glm5_topk_module", {}, ("attention", "dsa_indexer"), None),
        ("dsv4_csa_generation_module", {}, ("attention", "dsa_indexer"), "decode"),
        ("mla_context", {}, ("attention", "dsa_indexer"), "prefill"),
        ("encoder_attention", {}, ("attention",), "profile_run"),
    ],
)
def test_the_op_decides_what_a_gate_proves(gates, op, select, roles, phase):
    assert gates.gate_target(op, select) == (roles, phase)


def test_plan_narrows_to_a_repo_or_family(gates):
    by_repo = [s for p in gates.shard_plans("sm90", "vllm", repo="Qwen/Qwen3.5-0.8B").values() for s in p["shards"]]
    assert by_repo and all("Qwen3.5-0.8B" in s["name"] for s in by_repo)
    (by_family,) = gates.shard_plans("sm90", "vllm", family="gdn_").values()
    assert {s["name"] for s in by_family["shards"]} == {"gdn_ctx_Qwen3.5-0.8B", "gdn_gen_Qwen3.5-0.8B"}


def test_a_prefill_gate_reads_the_serving_record_of_its_prompt_length(gates):
    """vLLM picks the DSA prefill kernel by length on SM100 (sparse at 512, dense at 4096): a
    seq_len-512 case graded against the default-length record diverged, and passed before
    only because the 512 campaign was the newest import."""
    sys.path.insert(0, str(COMPONENTS))
    import probe_driver

    wrong = []
    for g in gates.load()["gates"]:
        _, phase = gates.gate_target(g["op"], g["select"])
        length = g["select"].get("seq_len") or g["select"].get("input_len")
        if phase == "prefill" and length and length != g.get("isl", probe_driver.DEFAULT_ISL):
            wrong.append(f"{g['gate']}@{g['backend']}: case length {length}, serving isl {g.get('isl')}")
    assert wrong == []


def test_a_no_serving_floor_names_an_sm_where_the_gate_exists(gates):
    """36 of 59 floors named an SM whose plan holds no such case (the gate is not there, so the
    floor was never read); 8 more stood for a probe nobody had re-run yet."""
    dead = [
        f"{g['gate']}@{g['backend']}:{sm}"
        for g in gates.load()["gates"]
        for sm in g.get("no_serving") or {}
        if not gates._matches(g["backend"], g["op"], int(sm.removeprefix("sm")), g["select"])
    ]
    assert dead == []


def test_a_gate_reads_only_its_own_backends_trace(gates, tmp_path):
    """Shard names repeat across backends: grading vLLM once read a TRT-LLM trace of the same
    shard from another result dir."""
    name = "gdn_ctx_Qwen3.5-0.8B"
    for backend, version in (("trtllm", "1.4.0rc2"), ("vllm", "0.31.0"), ("vllm", "0.32.0")):
        shard = tmp_path / f"{backend}-{version}" / f"{gates.SHARD_OP}__{name}"
        shard.mkdir(parents=True)
        doc = {"args": {"backend": backend}, "result": {"framework_version": version}}
        (shard / "probe_trace_case.json").write_text(json.dumps(doc))
    dirs = sorted(tmp_path.iterdir())
    assert gates._trace(dirs, name, "vllm", "0.32.0").parent.parent.name == "vllm-0.32.0"
    assert gates._trace(dirs, name, "vllm", "0.33.0").parent.parent.name.startswith("vllm-")
    assert gates._trace(dirs[:1], name, "vllm", "0.32.0") is None


def test_a_gate_reads_only_traces_from_its_own_sm(gates, tmp_path):
    """Shard names repeat across systems: grading sm90 once read the B200 trace of the same shard."""
    name = "gemm_bf16_Llama-3.1-8B"
    for system, sm in (("b200_sxm", 100), ("h100_sxm", 90), ("h100_failed", None)):
        shard = tmp_path / system / f"{gates.SHARD_OP}__{name}"
        shard.mkdir(parents=True)
        result = {"framework_version": "0.32.0", **({"sm": sm} if sm is not None else {})}
        (shard / "probe_trace_case.json").write_text(json.dumps({"args": {"backend": "vllm"}, "result": result}))
    dirs = sorted(tmp_path.iterdir())
    assert gates._trace(dirs, name, "vllm", "0.32.0", 90).parent.parent.name == "h100_sxm"
    assert gates._trace(dirs, name, "vllm", "0.32.0", 100).parent.parent.name == "b200_sxm"
    # a trace that died before recording its SM is used only when no attributed one exists
    assert gates._trace(dirs, name, "vllm", "0.32.0", 103).parent.parent.name == "h100_failed"
    assert gates._trace([tmp_path / "b200_sxm"], name, "vllm", "0.32.0", 90) is None


def test_every_registry_op_has_a_gate_or_a_reason(gates):
    """Coverage is read from the registries, not from the table: an op (or a planned phase of
    it) with no gate needs a reason under ``ungated``."""
    assert {"gemm", "moe"} <= set(gates.registry_ops("vllm")) and "dsv41_module" in gates.registry_ops("sglang")
    assert {backend: gates.uncovered(backend) for backend in ("vllm", "sglang", "trtllm")} == {
        "vllm": [],
        "sglang": [],
        "trtllm": [],
    }


def test_an_op_or_phase_without_a_gate_is_uncovered(gates, tmp_path):
    """A rebuilt gate set once kept 19 of 32 gates (kda decode among the lost) and reported 19/19."""
    table = tmp_path / "gates.yaml"
    row = lambda name, op, **select: {"gate": name, "backend": "vllm", "op": op, "select": select}
    doc = {"ungated": {"compute_scale": "no serving instance", "kda:verify": "no speculative decoding"}}
    doc["gates"] = [row("gemm_bf16", "gemm"), row("kda_ctx", "kda", phase="context")]
    table.write_text(yaml.safe_dump(doc))
    ops = ["gemm", "kda", "compute_scale", "brand_new_op"]
    assert gates.uncovered("vllm", table, ops) == ["kda:generation", "brand_new_op"]
    doc["gates"].append(row("kda_gen", "kda", phase="generation"))
    table.write_text(yaml.safe_dump(doc))
    assert gates.uncovered("vllm", table, ops) == ["brand_new_op"]
    # another backend's gates cover nothing here
    assert gates.uncovered("sglang", table, ["gemm"]) == ["gemm"]


def test_a_missing_trace_removes_the_verdict_of_an_earlier_run(gates, tmp_path, monkeypatch):
    """A gate without a trace is graded missing-trace; the verdict file an earlier run left must
    not keep saying aligned."""
    sys.path.insert(0, str(COMPONENTS))
    import probe_driver

    monkeypatch.setattr(probe_driver, "ROOT", tmp_path)
    gate = next(g for g in gates.gates_in_scope("vllm", "sm100") if not (g.get("no_serving") or {}).get("sm100"))
    stale = gates.verdict_dir("sm100", "vllm", gates.gate_version(gate)) / f"{gate['gate']}.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"verdict": "aligned"}))
    monkeypatch.setattr(gates, "gates_in_scope", lambda *a, **k: [gate])
    assert gates.grade("sm100", [tmp_path / "no_results"]) == {("vllm", gate["gate"]): "missing-trace"}
    assert not stale.exists()


def test_paths_are_read_from_the_op_plan(gates):
    """A path is the cases of one op that share a phase, the plan's dtype fields and its
    dispatch; a gate covers the path its case is on, and nothing else."""
    rows = gates.paths("sm100", "sglang", "moe,kda")
    moe = {(r["dtypes"], r["dispatch"]): r for r in rows if r["op"] == "moe"}
    assert len({dispatch for _dtypes, dispatch in moe}) > 1  # SGLang's MoE plan picks the runner
    assert all(r["cases"] > 0 for r in rows)
    gated = [r for r in moe.values() if r["gates"]]
    assert gated and all("derived" not in r for r in gated)
    kda = {r["phase"]: r for r in rows if r["op"] == "kda"}
    assert kda["verify"]["ungated"] and "derived" not in kda["verify"]
    assert kda["context"]["gates"] and kda["generation"]["gates"]


def test_a_derived_gate_names_one_planned_case_of_its_path(gates):
    """A path no row names gets a gate when its cases name a roster checkpoint: one planned
    case, on that path, whose filter fits the pipeline; rows are in scope only on request."""
    from collector.op_plans import OP_PLANS

    rows = gates.paths("sm100", "vllm")
    derived = [r for r in rows if r.get("derived")]
    assert derived
    names = [r["derived"]["gate"] for r in derived]
    assert len(set(names)) == len(names) and not set(names) & {g["gate"] for g in gates.load()["gates"]}
    for r in derived:
        g = r["derived"]
        (i,) = gates._matching("vllm", g["op"], 100, g["select"])
        case = gates._planned("vllm", g["op"], 100).cases[i]
        assert (getattr(case, "phase", None), OP_PLANS[g["op"]].dtypes(case)) == (r["phase"], r["dtypes"])
        assert g["repo"] in (getattr(case, "model_path", None), getattr(case, "model_name", None))
        gates.case_filter("vllm", g["op"], g["select"])
    # a case that names no model has no derived gate: its path needs a row or a reason
    assert not any(r.get("derived") for r in rows if r["op"] in ("gemm", "attention_context"))
    in_scope = {g["gate"] for g in gates.gates_in_scope("vllm", "sm100")}
    with_derived = {g["gate"] for g in gates.gates_in_scope("vllm", "sm100", derive=True)}
    assert not in_scope & set(names) and with_derived == in_scope | set(names)
