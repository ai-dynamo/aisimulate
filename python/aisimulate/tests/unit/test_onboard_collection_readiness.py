# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness checks consume actual native artifacts without claiming silicon accuracy."""

from __future__ import annotations

# ruff: noqa: F811
import hashlib
import json
import shutil
from dataclasses import replace

import pytest
from collector.fpm_forward import entry, planner, runner

from aisimulate import main as cli
from aisimulate.support import fpm
from aisimulate.support.collection_readiness import assess_readiness, resume_without_workers

from .test_support_serving_validation import materialized_workload  # noqa: F401
from .test_support_validation import validation_case  # noqa: F401
from .test_support_validation_workflow import quality_case  # noqa: F401

pytestmark = pytest.mark.unit


def _read(case):
    return assess_readiness(case["request"], case["root"], case["checkpoint"].parent)


def _decode(report):
    return next(item for item in report["selected_cells"] if item["cell"]["workload_kind"] == "decode")


def _change_decode(case, *, fake=None, kvwarm="unchanged"):
    for cell in case["plan"].cells:
        if cell.workload_kind != "decode":
            continue
        for path in (case["campaign"] / "cells" / cell.cell_id / "raw").rglob("benchmark-dp*.json"):
            payload = json.loads(path.read_text())
            if kvwarm is None:
                payload.pop("kvwarm", None)
            elif kvwarm != "unchanged":
                payload["kvwarm"] = kvwarm
            if fake is not None:
                for index, row in enumerate(payload["results"]):
                    if fake == "all" or index == 0:
                        row["kv_seed_regime"] = "fake_fallback"
                        row["point"]["sample_reasons"] = ["kvwarm_fake_fallback"]
                        payload["iteration_groups"][index]["point"]["sample_reasons"] = ["kvwarm_fake_fallback"]
            path.write_text(json.dumps(payload))


def test_public_readiness_revalidates_without_model_resolution_or_writes(quality_case, monkeypatch, capsys):
    case = quality_case
    before = {path: path.read_bytes() for path in case["root"].rglob("*") if path.is_file()}
    monkeypatch.setattr(planner, "build_collection_plan", lambda **kwargs: pytest.fail("must not resolve models"))
    assert (
        cli.main(
            [
                "onboard",
                "collect-fpm",
                "-c",
                str(case["root"] / "request.yaml"),
                "--output-dir",
                str(case["root"]),
                "--check-readiness",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["ready_for_full_collection"]
    assert {item["cell"]["workload_kind"] for item in report["selected_cells"]} == {"prefill", "decode"}
    assert _decode(report)["formal_regime_counts"] == {"real_kv": 12}
    assert report["campaigns"][0]["runtime_identity"]["collector_revision"] == "synthetic-workflow-revision"
    assert {path: path.read_bytes() for path in case["root"].rglob("*") if path.is_file()} == before
    _change_decode(case, fake="all")
    assert not _read(case)["ready_for_full_collection"]
    decode = _decode(_read(case))
    assert decode["direct_eligible_points"] == 0
    assert decode["native_regime_counts"] == {"fake_fallback": 12}
    assert decode["formal_regime_counts"] == {"fake_fallback": 12}
    assert "unchanged" in " ".join(decode["blockers"])


def test_mixed_coverage_and_optional_observation_gaps_do_not_claim_complete_qualification(quality_case):
    case = quality_case
    _change_decode(case, fake="one")
    for path in case["campaign"].rglob("fpm-execution-worker-*.json"):
        path.unlink()
    report = _read(case)
    assert report["ready_for_full_collection"]
    decode = _decode(report)
    assert decode["coverage"] == "partial"
    assert decode["direct_eligible_points"] == 11
    assert decode["execution"]["status"] == "incomplete"
    assert decode["execution"]["missing_evidence"]
    assert "not complete query coverage" in report["scope"]


@pytest.mark.parametrize(
    "kvwarm",
    [
        None,
        {"enabled": True, "warm_eligible": True, "skip_reason": "contradiction"},
        {"enabled": True, "warm_eligible": False, "skip_reason": "hybrid_state_layers_unsupported"},
    ],
)
def test_missing_or_contradictory_warm_protocol_blocks_with_raw_diagnostics(quality_case, kvwarm):
    case = quality_case
    _change_decode(case, fake="all", kvwarm=kvwarm)
    report = _read(case)
    assert not report["ready_for_full_collection"]
    decode = _decode(report)
    assert decode["status"] == "blocked"
    assert decode["rank_diagnostics"][0]["kvwarm"] == kvwarm
    if kvwarm is not None and kvwarm["warm_eligible"]:
        assert not decode["rank_diagnostics"][0]["verified"]
    assert decode["blockers"]


def test_execution_contradiction_blocks_but_does_not_discard_measurements(quality_case):
    case = quality_case
    path = next(case["campaign"].rglob("fpm-execution-worker-*.json"))
    payload = json.loads(path.read_text())
    payload["resolved_config"]["model_config"]["revision"] = "wrong-revision"
    path.write_text(json.dumps(payload))
    report = _read(case)
    assert not report["ready_for_full_collection"]
    failed = next(item for item in report["selected_cells"] if item["status"] == "blocked")
    assert failed["native_regime_counts"]
    assert failed["execution"]["failures"]


@pytest.mark.parametrize("value", [None, [], ["bad"], 42])
@pytest.mark.parametrize(
    ("filename", "field", "diagnostic"),
    [
        ("fpm-execution-worker-*.json", None, "runtime execution evidence must be an object"),
        ("fpm-execution-worker-*.json", "collector_provenance", "collector_provenance must be an object"),
        ("generator-request.json", None, "generated request must be an object"),
        ("generator-request.json", "ServiceConfig", "generated ServiceConfig must be an object"),
        ("generator-overrides.json", "K8sConfig", "archived K8sConfig must be an object"),
    ],
)
def test_public_readiness_blocks_malformed_optional_evidence_without_writes(
    quality_case, capsys, filename, field, diagnostic, value
):
    case = quality_case
    assert _read(case)["ready_for_full_collection"]
    decode = next(cell for cell in case["plan"].cells if cell.workload_kind == "decode")
    directory = (
        case["campaign"] if filename == "generator-overrides.json" else case["campaign"] / "cells" / decode.cell_id
    )
    path = next(directory.rglob(filename))
    payload = json.loads(path.read_text())
    if field is None:
        payload = value
    else:
        payload[field] = value
    path.write_text(json.dumps(payload))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in case["root"].rglob("*") if path.is_file()}
    assert (
        cli.main(
            [
                "onboard",
                "collect-fpm",
                "-c",
                str(case["root"] / "request.yaml"),
                "--output-dir",
                str(case["root"]),
                "--check-readiness",
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert not report["ready_for_full_collection"]
    assert diagnostic in " ".join(report["blockers"])
    if filename != "generator-overrides.json":
        assert _decode(report)["status"] == "blocked"
        assert _decode(report)["native_regime_counts"] == {"real_kv": 12}
        assert (
            next(item for item in report["selected_cells"] if item["cell"]["workload_kind"] == "prefill")["status"]
            == "ready"
        )
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in case["root"].rglob("*") if path.is_file()
    } == before


def test_later_formal_attempt_cannot_reuse_older_good_smoke(quality_case):
    case = quality_case
    shutil.copytree(case["campaign"], case["campaign"] / "smoke", ignore=shutil.ignore_patterns("smoke"))
    smoke_checkpoint = case["checkpoint"].with_name("fpm_forward_smoke.json")
    smoke_checkpoint.write_bytes(case["checkpoint"].read_bytes())
    checkpoint = json.loads(case["checkpoint"].read_text())
    for value in checkpoint["cells"].values():
        value.update(status="failed", attempt_id="later-failing-attempt")
    case["checkpoint"].write_text(json.dumps(checkpoint))
    report = _read(case)
    assert not report["ready_for_full_collection"]
    assert all(item["campaign"] == "formal" for item in report["selected_cells"])
    assert all("attempt mismatch" in " ".join(item["blockers"]) for item in report["selected_cells"])


def test_resolved_smoke_defaults_cover_all_cells_without_assuming_order(quality_case, monkeypatch):
    case = quality_case
    # A generic plan can have multiple prefill cells before any decode cell.
    plan = replace(case["plan"], cells=case["plan"].cells + case["plan"].cells)
    monkeypatch.setattr(entry, "resolve_run_inputs", lambda *_: (plan, {}))
    command = fpm.fpm_cli_args(case["request"], output_dir=case["root"], plan_only=False, smoke=True)
    args, _ = fpm._resolve_execution(command)
    assert args.limit == 4
    args, _ = fpm._resolve_execution([*command, "--limit", "1"])
    assert args.limit == 1


@pytest.mark.parametrize("quality_case", [False, True], indirect=True, ids=["ordinary", "instrumented"])
@pytest.mark.parametrize("state", ["passed", "failed", "cleanup_failed"])
def test_recovery_bypass_uses_real_runner_with_every_resource_launch_forbidden(quality_case, monkeypatch, state):
    case = quality_case
    _change_decode(case, fake="all")
    checkpoint = json.loads(case["checkpoint"].read_text())
    checkpoint.pop("database", None)
    # Model an unpublished attempt, so recovery can commit its native rows.
    shutil.rmtree(case["root"] / "systems/data")
    (case["root"] / "systems/data").mkdir()
    for value in checkpoint["cells"].values():
        value["status"] = state
        if state == "cleanup_failed":
            value["cleanup_error"] = "preserved unresolved teardown"
    case["checkpoint"].write_text(json.dumps(checkpoint))
    assert resume_without_workers(case["plan"], case["checkpoint"])
    monkeypatch.setattr(runner, "_cell_runner", lambda *_: pytest.fail("recovery must not touch resources"))
    monkeypatch.setattr(runner, "_render_cell", lambda *_: pytest.fail("recovery must not launch a cell"))
    assert not _read(case)["ready_for_full_collection"]
    args = (
        __import__("collector.fpm_forward.cli", fromlist=["_parser"])
        ._parser()
        .parse_args(fpm.fpm_cli_args(case["request"], output_dir=case["root"], plan_only=False, resume=True)[3:])
    )
    monkeypatch.setattr(
        fpm,
        "_resolve_execution",
        lambda _: (args, (case["plan"], {"K8sConfig": {"k8s_image": "example/runtime@sha256:" + "a" * 64}})),
    )
    assert fpm.run_fpm(case["request"], output_dir=case["root"], execute=True, resume=True) == 1
    report = json.loads((case["root"] / "fpm-readiness.json").read_text())
    assert not report["ready_for_full_collection"]
    assert report["recovery_only"]
    if not case["plan"].runtime_instrumentation and state != "cleanup_failed":
        assert report["collector_exit_status"] == 0
    assert _decode(report)["formal_regime_counts"] == {"fake_fallback": 12}


@pytest.mark.parametrize("state", ["running", "interrupted", "pending", "missing", "malformed"])
def test_unsafe_or_incomplete_resume_does_not_bypass_readiness(quality_case, monkeypatch, state):
    case = quality_case
    _change_decode(case, fake="all")
    checkpoint = json.loads(case["checkpoint"].read_text())
    key = next(iter(checkpoint["cells"]))
    if state == "missing":
        del checkpoint["cells"][key]
    elif state == "malformed":
        checkpoint["cells"][key] = "passed"
    else:
        checkpoint["cells"][key]["status"] = state
    case["checkpoint"].write_text(json.dumps(checkpoint))
    assert not resume_without_workers(case["plan"], case["checkpoint"])
    args = (
        __import__("collector.fpm_forward.cli", fromlist=["_parser"])
        ._parser()
        .parse_args(fpm.fpm_cli_args(case["request"], output_dir=case["root"], plan_only=False, resume=True)[3:])
    )
    monkeypatch.setattr(fpm, "_resolve_execution", lambda _: (args, (case["plan"], {})))
    monkeypatch.setattr(entry, "run_resolved", lambda *_: pytest.fail("blocked before runner"))
    assert fpm.run_fpm(case["request"], output_dir=case["root"], execute=True, resume=True) == 1


def test_fresh_full_collection_requires_saved_readiness(quality_case, monkeypatch):
    case = quality_case
    shutil.rmtree(case["campaign"])
    case["checkpoint"].unlink()
    shutil.rmtree(case["root"] / "systems/data")
    (case["root"] / "systems/data").mkdir()
    args = (
        __import__("collector.fpm_forward.cli", fromlist=["_parser"])
        ._parser()
        .parse_args(fpm.fpm_cli_args(case["request"], output_dir=case["root"], plan_only=False)[3:])
    )
    monkeypatch.setattr(fpm, "_resolve_execution", lambda _: (args, (case["plan"], {})))
    monkeypatch.setattr(entry, "run_resolved", lambda *_: pytest.fail("must smoke before full collection"))
    assert fpm.run_fpm(case["request"], output_dir=case["root"], execute=True) == 1


def test_approved_legacy_tp_skip_retains_formal_consumer_eligibility(quality_case):
    case = quality_case
    _change_decode(
        case,
        fake="all",
        kvwarm={"enabled": True, "warm_eligible": False, "skip_reason": "moe_tp_balanced_by_construction"},
    )
    report = _read(case)
    assert report["ready_for_full_collection"]
    decode = _decode(report)
    assert decode["native_regime_counts"] == {"fake_fallback": 12}
    assert decode["formal_regime_counts"] == {"skip:moe_tp_balanced_by_construction": 12}
    assert decode["direct_eligible_points"] == 12


@pytest.mark.parametrize("value", [[], 42, None, {"cells": []}])
def test_malformed_checkpoint_reports_blocker_without_crashing(quality_case, value):
    case = quality_case
    case["checkpoint"].write_text(json.dumps(value))
    report = _read(case)
    assert not report["ready_for_full_collection"]
    assert report["blockers"]
    assert not resume_without_workers(case["plan"], case["checkpoint"])


def test_malformed_native_artifact_preserves_independent_phase_result(quality_case):
    case = quality_case
    decode = next(cell for cell in case["plan"].cells if cell.workload_kind == "decode")
    path = next((case["campaign"] / "cells" / decode.cell_id).rglob("benchmark-dp*.json"))
    path.write_text("[]")
    report = _read(case)
    assert not report["ready_for_full_collection"]
    assert (
        next(item for item in report["selected_cells"] if item["cell"]["workload_kind"] == "prefill")["status"]
        == "ready"
    )


def test_consistent_but_wrong_observed_runtime_version_is_blocked(quality_case):
    case = quality_case
    for path in case["campaign"].rglob("collector-provenance.json"):
        payload = json.loads(path.read_text())
        payload["runtime"]["backend_version"] = "0.99.0"
        path.write_text(json.dumps(payload))
    report = _read(case)
    assert not report["ready_for_full_collection"]
    assert all(item["status"] == "blocked" for item in report["selected_cells"])


def _bounded_source(case, tmp_path, *, explicit=False):
    from aisimulate.support.plan import create_plan
    from aisimulate.support.schema import SupportRequest

    request_payload = case["request"].model_dump(mode="json")
    if explicit:
        request_payload["collection"].update(prefill_cudagraph_policy="explicit", max_prefill_cudagraph_size=16)
    request = SupportRequest.model_validate(request_payload)
    root = tmp_path / "bounded"
    create_plan(request, root)
    model_config = tmp_path / "bounded-model-config.json"
    model_config.write_text(json.dumps(case["plan"].capability.model_config.payload))
    overrides = {"K8sConfig": {"k8s_image": "example/runtime@sha256:" + "a" * 64}}
    plan = planner.build_collection_plan(
        backend="vllm",
        model_path=request.identity.model,
        system=request.identity.gpu,
        selected_ops=set(),
        model_architecture=request.fpm_profile.architecture,
        model_config_path=str(model_config),
        fpm_profile=request.fpm_profile,
        generator_overrides=overrides,
        options=replace(
            case["plan"].options,
            benchmark_points_json=None,
            benchmark_points_sha256=None,
            prefill_cudagraph_policy=request.collection.prefill_cudagraph_policy,
            max_prefill_cudagraph_size=request.collection.max_prefill_cudagraph_size,
        ),
    )
    return request, root, plan, overrides


def _write_bounded_native(plan, root, overrides, *, smoke):
    from collector.fpm_forward.config import with_kv_warmup_defaults

    from .collector.test_fpm_runner import _native_payload, _write_provenance

    campaign = root / "fpm-artifacts" / plan.sha256[:16]
    if smoke:
        campaign /= "smoke"
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "collection-plan.json").write_text(json.dumps(plan.to_dict()))
    (campaign / "generator-overrides.json").write_text(json.dumps(with_kv_warmup_defaults(overrides)))
    entries = {}
    for cell in plan.cells:
        directory = campaign / "cells" / cell.cell_id
        raw = directory / "raw/pod"
        raw.mkdir(parents=True)
        runner._render_cell(plan, cell, directory, overrides, smoke=smoke)
        provenance = raw / "collector-provenance.json"
        _write_provenance(provenance, cell_id=cell.cell_id, plan_sha256=plan.sha256, attempt_id="bounded")
        payload = json.loads(provenance.read_text())
        payload["runtime"]["backend_version"] = plan.capability.aic_database_version
        provenance.write_text(json.dumps(payload))
        for rank in range(cell.topology.dp):
            native = _native_payload(phase=cell.workload_kind, rank=rank, dp=cell.topology.dp)
            regime, stamp = (
                ("real_kv", "kvwarm_real_kv")
                if cell.workload_kind == "decode"
                else ("real_prefix", "prefill_real_seed")
            )
            native["results"][0]["kv_seed_regime"] = regime
            native["results"][0]["point"]["sample_reasons"] = [stamp]
            (raw / f"benchmark-dp{rank}.json").write_text(json.dumps(native))
        entries[cell.cell_id] = {"status": "passed", "attempt_id": "bounded"}
    checkpoint = root / "fpm-checkpoint" / ("fpm_forward_smoke.json" if smoke else "fpm_forward.json")
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_text(
        json.dumps({"schema": runner.CHECKPOINT_SCHEMA, "plan_sha256": plan.sha256, "cells": entries})
    )
    return campaign


def test_public_smoke_covers_both_phases_then_matching_full_collection_can_start(quality_case, tmp_path, monkeypatch):
    request, root, plan, overrides = _bounded_source(quality_case, tmp_path)
    calls = []
    monkeypatch.setattr(entry, "resolve_run_inputs", lambda *_: (plan, overrides))

    def synthetic_collection(args, resolved):
        calls.append((args.smoke, args.limit))
        _write_bounded_native(resolved[0], root, resolved[1], smoke=args.smoke)
        return []

    monkeypatch.setattr(entry, "run_resolved", synthetic_collection)
    command = ["onboard", "collect-fpm", "-c", str(root / "request.yaml"), "--output-dir", str(root), "--execute"]
    assert cli.main([*command, "--smoke"]) == 0
    assert calls == [(True, len(plan.cells))]
    report = json.loads((root / "fpm-readiness.json").read_text())
    assert report["ready_for_full_collection"]
    assert {item["cell"]["workload_kind"] for item in report["selected_cells"]} == {"prefill", "decode"}
    assert cli.main(command) == 0
    assert calls == [(True, len(plan.cells)), (False, None)]
    assert all(
        item["campaign"] == "formal" for item in json.loads((root / "fpm-readiness.json").read_text())["selected_cells"]
    )


def test_old_explicit_smoke_with_missing_or_changed_capture_flags_is_blocked(quality_case, tmp_path):
    import re
    import shlex

    request, root, plan, overrides = _bounded_source(quality_case, tmp_path, explicit=True)
    campaign = _write_bounded_native(plan, root, overrides, smoke=True)
    assert assess_readiness(request, root, root / "fpm-checkpoint")["ready_for_full_collection"]
    prefill = next(cell for cell in plan.cells if cell.workload_kind == "prefill")
    path = campaign / "cells" / prefill.cell_id / "run.sh"
    script = path.read_text()
    command = re.search(r"^engine_command=\((.*)\)$", script, re.MULTILINE)
    argv = shlex.split(command[1])
    index = argv.index("--compilation-config")
    for config in (None, json.dumps({"cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": 1})):
        edited = list(argv)
        if config is None:
            del edited[index : index + 2]
        else:
            edited[index + 1] = config
        path.write_text(script.replace(command[0], "engine_command=(" + shlex.join(edited) + ")"))
        report = assess_readiness(request, root, root / "fpm-checkpoint")
        assert not report["ready_for_full_collection"]
        assert "explicit graph configuration" in " ".join(report["blockers"])


@pytest.mark.parametrize(
    ("explicit", "phase", "graph", "ready"),
    [
        pytest.param(
            True, "prefill", {"capture_sizes": [1, 2, 4, 8], "max_capture_size": 8}, False, id="contradictory-native"
        ),
        pytest.param(True, "prefill", {"capture_sizes": [1, 2, 4, 8]}, False, id="capture-list-only"),
        pytest.param(True, "prefill", {"max_capture_size": 8}, False, id="maximum-only"),
        pytest.param(
            True,
            "prefill",
            {
                "capture_sizes": [1, 2, 4, 8, 16],
                "max_capture_size": 16,
                "prefill_capture_sizes": [],
                "decode_capture_sizes": [1, 2],
            },
            True,
            id="matching-full-captures-with-filtered-phase-lists",
        ),
        pytest.param(True, "prefill", None, True, id="missing-optional"),
        pytest.param(
            False, "prefill", {"capture_sizes": [1, 2, 4, 8], "max_capture_size": 8}, True, id="runtime-policy"
        ),
        pytest.param(True, "decode", {"capture_sizes": [1, 2, 4, 8], "max_capture_size": 8}, True, id="decode-default"),
    ],
)
def test_public_readiness_checks_native_explicit_prefill_graph_evidence(
    quality_case, tmp_path, capsys, explicit, phase, graph, ready
):
    request, root, plan, overrides = _bounded_source(quality_case, tmp_path, explicit=explicit)
    campaign = _write_bounded_native(plan, root, overrides, smoke=True)
    assert assess_readiness(request, root, root / "fpm-checkpoint")["ready_for_full_collection"]
    cell = next(cell for cell in plan.cells if cell.workload_kind == phase)
    path = next((campaign / "cells" / cell.cell_id / "raw").rglob("benchmark-dp*.json"))
    payload = json.loads(path.read_text())
    if graph is not None:
        payload["cudagraph"] = graph
    else:
        payload.pop("cudagraph", None)
    path.write_text(json.dumps(payload))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    assert cli.main(
        [
            "onboard",
            "collect-fpm",
            "-c",
            str(root / "request.yaml"),
            "--output-dir",
            str(root),
            "--check-readiness",
        ]
    ) == (0 if ready else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["ready_for_full_collection"] == ready
    prefill = next(item for item in report["selected_cells"] if item["cell"]["workload_kind"] == "prefill")
    assert prefill["execution"]["observed_workers"] == []
    if not ready:
        assert prefill["status"] == "blocked"
        assert "native prefill graph configuration differs" in " ".join(prefill["blockers"])
        assert _decode(report)["status"] == "ready"
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()} == before


def test_missing_evidence_cannot_be_replaced_by_a_cached_ready_report(quality_case, monkeypatch):
    case = quality_case
    (case["root"] / "fpm-readiness.json").write_text(json.dumps({"status": "ready", "ready_for_full_collection": True}))
    decode = next(cell for cell in case["plan"].cells if cell.workload_kind == "decode")
    shutil.rmtree(case["campaign"] / "cells" / decode.cell_id / "raw")
    report = _read(case)
    assert not report["ready_for_full_collection"]
    assert _decode(report)["status"] == "incomplete"


def test_malformed_formal_cell_overrides_good_smoke_even_when_null(quality_case):
    case = quality_case
    shutil.copytree(case["campaign"], case["campaign"] / "smoke", ignore=shutil.ignore_patterns("smoke"))
    case["checkpoint"].with_name("fpm_forward_smoke.json").write_bytes(case["checkpoint"].read_bytes())
    checkpoint = json.loads(case["checkpoint"].read_text())
    decode = next(cell for cell in case["plan"].cells if cell.workload_kind == "decode")
    checkpoint["cells"][decode.cell_id] = None
    case["checkpoint"].write_text(json.dumps(checkpoint))
    report = _read(case)
    assert not report["ready_for_full_collection"]
    assert _decode(report)["campaign"] == "formal"


@pytest.mark.parametrize("regime", [42, ["real_kv"], {"regime": "real_kv"}])
def test_malformed_native_regime_remains_json_reportable(quality_case, regime):
    case = quality_case
    decode = next(cell for cell in case["plan"].cells if cell.workload_kind == "decode")
    path = next((case["campaign"] / "cells" / decode.cell_id).rglob("benchmark-dp*.json"))
    payload = json.loads(path.read_text())
    payload["results"][0]["kv_seed_regime"] = regime
    path.write_text(json.dumps(payload))
    report = _read(case)
    assert not report["ready_for_full_collection"]
    assert _decode(report)["rank_diagnostics"][0]["regime_counts"]["invalid"] == 1
    assert json.loads(json.dumps(report, sort_keys=True))["status"] == "blocked"
