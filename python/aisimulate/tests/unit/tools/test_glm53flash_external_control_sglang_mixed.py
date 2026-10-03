# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY mixed source/control fixtures; actual source54 checked separately."""

import copy
import json
from pathlib import Path

import pytest
from tools.glm53flash_hf import external_control as control
from tools.glm53flash_hf import external_control_current as current
from tools.glm53flash_hf import external_control_sglang_mixed as mixed
from tools.glm53flash_hf import glm53flash as policy

from . import test_glm53flash_external_control_current as fixture

pytestmark = pytest.mark.unit


def mixed_fixture():
    f = fixture.sg_fixture()
    original = json.loads(f.files["launch/admission.json"])
    default = copy.deepcopy(original)
    default["qualifications"]["fp8-tp2"] = None
    fp8 = copy.deepcopy(original)
    for deployment in current.DEPLOYMENTS - {"fp8-tp2"}:
        fp8["qualifications"][deployment] = None
    definitions = {}
    for label, value in [("launch-default", default), ("launch-fp8", fp8)]:
        f.put(label + "/admission.json", value)
        f.manifest(label, [label + "/admission.json"])
        names = dict(
            f.document["anchors"], launcher_manifest=label + "/manifest.sha256", admission=label + "/admission.json"
        )
        definitions[label] = {k: {"path": n, "sha256": fixture.sha(f.files[n])} for k, n in names.items()}
    f.document = {
        "schema": current.MIXED_SCHEMA,
        "adapter": current.MIXED_SGLANG,
        "original_task_root": str(fixture.TASK),
        "deployment_controls": {
            d: copy.deepcopy(definitions["launch-fp8" if d == "fp8-tp2" else "launch-default"])
            for d in current.DEPLOYMENTS
        },
        "runs": [],
    }
    return f


def add_runs(f):
    context = current.frozen_contract(f.document, f.files.__getitem__)
    for i, (cid, child) in enumerate(context["children"].items()):
        job = str(1000 + i)
        root = f"runs/{cid}/{job}"
        raw = root + "/raw/node0000"
        d = child["deployment"]
        anchors = context["deployment_anchors"][d]
        start = {
            "state": "RUNNING",
            "job": job,
            "deployment": d,
            "mode": "formal",
            "source_commit": context["producer"][0],
            "wheel_sha256": context["producer"][1],
            "host_source_commit": context["host"][0],
            "actual_cpu_job": 100,
            "admission_sha256": context["files"][anchors["admission"]],
            "launcher_manifest_sha256": context["files"][anchors["launcher_manifest"]],
            "selected": {"kind": "formal", "role": child["role"], "child_identity": child["original_identity"]},
            "plan_sha256": child["child_plan_sha256"],
            "qualification": context["qualifications"][d],
            "requested_allocator_policy": {
                "schema": "sglang_native_allocator_policy_v1",
                "backend": "native",
                "max_split_size_mb": 16384 if d == "fp8-tp2" else None,
            },
        }
        f.put(root + "/started.json", start)
        for kind in ("host", "producer"):
            value = {"wheel_sha256": context[kind][1], "runtime_sha256": "7" * 64, "files": {"TEST_ONLY.so": "8" * 64}}
            value.update(
                {"status": "ACTUAL_HOST_RECORD_GIT_COLLECTOR_RENDERER_AND_ARM_ELF_PASS", "head": context[kind][0]}
                if kind == "host"
                else {"state": "EXACT_INSTALLED_WHEEL_RECORD_SOURCE_ELF_PASS", "source": context[kind][0]}
            )
            f.put(root + "/actual-" + kind + "-wheel.json", value)
        f.put(
            raw + "/collector-provenance.json",
            {
                "cell_id": cid,
                "plan_sha256": child["child_plan_sha256"],
                "attempt_id": "TEST_ONLY_attempt",
                "runtime": {"backend": "sglang", "backend_version": "0.5.20"},
            },
        )
        f.document["runs"].append(
            {
                "cell_id": cid,
                "started": root + "/started.json",
                "raw_root": str(fixture.TASK / raw),
                "collector_provenance": raw + "/collector-provenance.json",
                "host_wheel_verification": root + "/actual-host-wheel.json",
                "producer_wheel_verification": root + "/actual-producer-wheel.json",
            }
        )
    return context


def test_two_original_admissions_cover72_without_rewriting_either():
    f = mixed_fixture()
    before = {n: v for n, v in f.files.items() if n.endswith("admission.json")}
    context = add_runs(f)
    files, admissions = current.closure(f.document, f.files.__getitem__)
    assert len(context["children"]) == 72 and len(admissions["deployments"]) == 4
    assert {n: f.files[n] for n in before} == before
    assert "launch-default/admission.json" in files and "launch-fp8/admission.json" in files
    assert json.loads(f.files["launch-default/admission.json"])["qualifications"]["fp8-tp2"] is None


def test_partial_source54_is_preparation_only_and_full_control_rejects():
    f = mixed_fixture()
    f.document["deployment_controls"].pop("fp8-tp2")
    context = mixed.inspect_partial(f.document, f.files.__getitem__)
    assert len(context["children"]) == 54 and len(context["source_children"]) == 72 and context["preparation_only"]
    with pytest.raises(ValueError, match="all four"):
        current.closure(f.document, f.files.__getitem__)


def test_old_v2_still_rejects_partial_admission():
    f = mixed_fixture()
    d = f.document["deployment_controls"]["fp8-tp4"]
    old = {
        "schema": current.SCHEMA,
        "adapter": current.SGLANG,
        "original_task_root": str(fixture.TASK),
        "anchors": {k: v["path"] for k, v in d.items()},
        "runs": [],
    }
    with pytest.raises(ValueError, match="qualification missing"):
        current.frozen_contract(old, f.files.__getitem__)


@pytest.mark.parametrize(
    "corruption",
    ["wrong_admission", "wrong_deployment", "wrong_policy", "wrong_qualification", "changed_child", "host_revision"],
)
def test_started_remains_bound_to_its_own_deployment_original(corruption):
    f = mixed_fixture()
    context = add_runs(f)
    run = next(r for r in f.document["runs"] if context["children"][r["cell_id"]]["deployment"] == "fp8-tp4")
    value = json.loads(f.files[run["started"]])
    if corruption == "wrong_admission":
        value["admission_sha256"] = context["files"]["launch-fp8/admission.json"]
    elif corruption == "wrong_deployment":
        value["deployment"] = "fp8-tp2"
    elif corruption == "wrong_policy":
        value["requested_allocator_policy"]["max_split_size_mb"] = 16384
    elif corruption == "wrong_qualification":
        value["qualification"] = context["qualifications"]["fp8-tp2"]
    elif corruption == "changed_child":
        value["selected"]["child_identity"]["point_map"].pop()
    else:
        value["host_source_commit"] = "f" * 40
    f.put(run["started"], value)
    with pytest.raises(ValueError):
        current.closure(f.document, f.files.__getitem__)


def test_missing_fourth_qualification_cannot_be_filled_by_source72():
    f = mixed_fixture()
    f.document["deployment_controls"]["fp8-tp2"] = copy.deepcopy(f.document["deployment_controls"]["fp8-tp4"])
    with pytest.raises(ValueError, match="qualification missing"):
        current.frozen_contract(f.document, f.files.__getitem__)


def test_full_original_attachment_prepare_validate_roundtrip(tmp_path):
    f = mixed_fixture()
    add_runs(f)
    source = tmp_path / "source"
    source.mkdir()
    for name, raw in f.files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    target = tmp_path / "attachment"
    result = control.prepare(
        source,
        str(fixture.TASK),
        f.document["deployment_controls"],
        f.document["runs"],
        target,
        adapter=current.MIXED_SGLANG,
    )
    assert result["schema"] == current.MIXED_SCHEMA and "anchors" not in result
    index, get, admission = control.validate(target, result)
    assert set(index) == set(current.closure(f.document, f.files.__getitem__)[0])
    assert len(admission["deployments"]) == 4
    revisions = current.configuration_revisions(
        result, get, admission, "sglang", "fp8", 4, "installed:aisimulate==0.13.0:record-sha256:" + "9" * 64
    )
    assert revisions["planner_source_commit"] != revisions["native_producer_revision"]
    context = current.frozen_contract(result, get)
    run = result["runs"][0]
    cid = run["cell_id"]
    child = context["children"][cid]
    plan_path = str(
        (Path(child["native_directory"]).parent.parent / "plans" / (cid + ".json")).relative_to(fixture.TASK)
    )
    plan = json.loads(f.files[plan_path])
    spec = dict(cell_id=cid, raw_root=run["raw_root"], attempt_id="TEST_ONLY_attempt", plan={"path": plan_path})
    evidence = {
        "receipts": [{"path": "collector-provenance.json", "sha256": index[run["collector_provenance"]]["sha256"]}]
    }
    inventory = [dict(kind="file", path=name, sha256=item["sha256"]) for name, item in index.items()]
    control.bind_role(
        result, get, admission, [(spec, evidence)], {plan_path: plan}, str(fixture.TASK), inventory, fixture.TASK
    )
    with pytest.raises(ValueError, match="archive changed"):
        control.bind_role(
            result, get, admission, [(spec, evidence)], {plan_path: plan}, str(fixture.TASK), [], fixture.TASK
        )
    fixture._check_publication_revisions(target, context)


def test_canonical_policy_includes_and_imports_new_mixed_module():
    assert "external_control_sglang_mixed.py" in policy.POLICY_MODULES
    assert Path(mixed.__file__).resolve().parent == Path(policy.__file__).resolve().parent
