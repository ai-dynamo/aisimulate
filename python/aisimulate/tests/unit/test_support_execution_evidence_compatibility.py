# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical prepared campaigns retain their frozen observation semantics."""

import json

import pytest
from collector.fpm_forward import repeatability

from aisimulate import main as cli
from aisimulate.support import validation_workflow as workflow
from aisimulate.support.plan import create_plan
from aisimulate.support.schema import SupportRequest

from .collector.test_fpm_measurement_evidence import _remove_execution_protocol
from .test_onboard_finalization_quality import assessed_collection  # noqa: F401

pytestmark = pytest.mark.unit


def _legacy_execution(case):
    case["execution_evidence"] = False
    for path in case["root"].rglob("benchmark-dp*.json"):
        payload = json.loads(path.read_text())
        _remove_execution_protocol(payload)
        path.write_text(json.dumps(payload))


def test_fresh_v2_public_quality_cannot_pass_without_execution_observations(assessed_collection):  # noqa: F811
    case = assessed_collection
    _legacy_execution(case)
    assert cli.main([*case["validation_args"], "--execute"]) == 1
    report = json.loads(case["report"].read_text())
    selection = json.loads((case["report"].parent / "repeatability-selection.json").read_text())
    assert selection["observation_evidence_version"] == 2
    assert all(
        "execution_evidence" not in cell["source_measurement"]["measurement_protocol"] for cell in selection["cells"]
    )
    assert all(cell["source_measurement"]["status"] == "unestablished" for cell in selection["cells"])
    assert report["status"] == report["gates"]["execution"]["status"] == "incomplete"


@pytest.mark.parametrize("policy", ["runtime", "explicit"])
@pytest.mark.parametrize("execute", [False, True])
def test_ordinary_collection_reports_graph_advisory_before_launch(
    assessed_collection,  # noqa: F811
    tmp_path,
    monkeypatch,
    capsys,
    policy,
    execute,
):
    from collector.fpm_forward import capabilities, entry

    case = assessed_collection
    payload = case["request"].model_dump(mode="json")
    payload["collection"].update(
        prefill_cudagraph_policy=policy, max_prefill_cudagraph_size=2048 if policy == "explicit" else None
    )
    request = SupportRequest.model_validate(payload)
    root = tmp_path / "graph-advisory"
    create_plan(request, root)
    capsys.readouterr()
    calls = []
    monkeypatch.setattr(capabilities, "load_model_config", lambda *args, **kwargs: case["plan"].capability.model_config)

    def launch(*args):
        output = capsys.readouterr()
        assert ("prefill only" in output.err) == (policy == "explicit")
        calls.append(args)
        return []

    monkeypatch.setattr(entry, "run_resolved", launch)
    args = ["onboard", "collect-fpm", "--config", str(root / "request.yaml"), "--output-dir", str(root)]
    result = cli.main([*args, *(["--execute", "--smoke"] if execute else [])])
    if execute:
        # The launch was reached; this synthetic launcher does not publish readiness artifacts.
        assert calls
        assert result == 1
    else:
        output = capsys.readouterr()
        assert result == 0
        assert ("prefill only" in output.err) == (policy == "explicit")
        assert "collector.fpm_forward" in output.out


@pytest.mark.parametrize("mutation", ["running", "future_order"])
def test_public_quality_cannot_pass_impossible_warmup_history(assessed_collection, monkeypatch, mutation):  # noqa: F811
    case = assessed_collection

    def mutate(root):
        for path in root.rglob("benchmark-dp*.json"):
            payload = json.loads(path.read_text())
            record = payload["warmup_evidence"]["records"][0]
            if mutation == "running":
                record.update(status="running", forward_index_end=None)
            else:
                record["completed_points_before"] = 99
            path.write_text(json.dumps(payload))

    mutate(case["root"])
    collect = repeatability.run_collection

    def mutated_repeat(plan, **kwargs):
        from pathlib import Path

        result = collect(plan, **kwargs)
        mutate(Path(kwargs["artifact_root"]))
        return result

    monkeypatch.setattr(repeatability, "run_collection", mutated_repeat)
    if mutation == "future_order":
        with pytest.raises(SystemExit, match="2"):
            cli.main([*case["validation_args"], "--execute"])
        assert not case["report"].exists()
    else:
        assert cli.main([*case["validation_args"], "--execute"]) == 1
    if case["report"].exists():
        report = json.loads(case["report"].read_text())
        assert report["status"] != "passed"
        if mutation == "running":
            assert report["status"] == report["gates"]["execution"]["status"] == "incomplete"


def test_legacy_preparation_can_resume_first_repeats_and_reuse_them(assessed_collection, tmp_path, monkeypatch):  # noqa: F811
    case = assessed_collection
    _legacy_execution(case)
    current = workflow._repeat_plan

    def historical(*args, **kwargs):
        return current(*args, **{**kwargs, "observation_evidence_version": 1})

    monkeypatch.setattr(workflow, "_repeat_plan", historical)
    assert cli.main(case["validation_args"]) == 0
    selection_path = case["report"].parent / "repeatability-selection.json"
    selection = selection_path.read_bytes()
    assert "observation_evidence_version" not in json.loads(selection)
    monkeypatch.setattr(workflow, "_repeat_plan", current)
    assert cli.main([*case["validation_args"], "--resume", "--execute"]) == 0
    assert json.loads(case["report"].read_text())["status"] == "passed"
    assert selection_path.read_bytes() == selection
    repeats = case["report"].parent / "repeatability"
    assert "observation_evidence_version" not in json.loads((repeats / repeatability.PLAN_FILENAME).read_text())
    workflow.check_collection_report(case["report"])
    calls = len(case["calls"])
    new_output = tmp_path / "reused"
    args = list(case["validation_args"])
    args[args.index("--validation-output-dir") + 1] = str(new_output)
    assert cli.main([*args, "--repeatability-dir", str(repeats)]) == 0
    assert len(case["calls"]) == calls
    assert json.loads((new_output / "collection-validation.json").read_text())["status"] == "passed"
    workflow.check_collection_report(new_output / "collection-validation.json")
