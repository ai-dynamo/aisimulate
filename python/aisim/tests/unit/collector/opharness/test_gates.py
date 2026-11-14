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
        gates.case_filter(backend, op, select, gate.get("env"))  # raises when it does not fit the pipeline


def test_shard_plans_parse_as_pipeline_plans(gates, tmp_path):
    plans = gates.shard_plans("sm100", "sglang")
    shards = [s for plan in plans.values() for s in plan["shards"]]
    assert {plan["backend"] for plan in plans.values()} == {"sglang"} and shards
    assert all(s["op"] == "probe.trace_case" and s["case_filter"].startswith("backend=sglang,op=") for s in shards)
    assert len({s["name"] for s in shards}) == len(shards)


def test_a_gate_runs_at_the_release_its_op_is_collected_with(gates):
    """vLLM gemm/moe/kda/attention are pinned to 0.31.0 next to the 0.30.0 default: their gates
    were traced in the default image and graded against default-release serving records. Each
    (backend, release) is its own job; an also case shares its gate's release."""
    plans = gates.shard_plans("sm90", "vllm")
    gemm_release = gates.op_version("vllm", "gemm")
    assert gemm_release != gates.op_version("vllm", "dsa_context_module")  # true while the families differ
    assert set(plans) == {("vllm", gates.gate_version(g)) for g in gates.gates_in_scope("vllm", "sm90")}
    gemm = {s["name"] for s in plans[("vllm", gemm_release)]["shards"]}
    assert "gemm_fp8block_DeepSeek-V3.2" in gemm and "dsa_ctx_bf16_DeepSeek-V3.2" not in gemm
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
