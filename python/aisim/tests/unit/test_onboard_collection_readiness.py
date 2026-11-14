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

from aisim import main as cli
from aisim.support import fpm
from aisim.support.collection_readiness import assess_readiness, resume_without_workers

from .collector.test_fpm_glm53flash_native_collection import _glm_producer_shape
from .collector.test_fpm_measurement_evidence import _add_measurement_protocol
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
            _add_measurement_protocol(payload)
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


def _bounded_source(case, tmp_path, *, explicit=False, warmup_iterations=None):
    from aisim.support.plan import create_plan
    from aisim.support.schema import SupportRequest

    request_payload = case["request"].model_dump(mode="json")
    if explicit:
        request_payload["collection"].update(prefill_cudagraph_policy="explicit", max_prefill_cudagraph_size=16)
    if warmup_iterations is not None:
        request_payload["collection"]["warmup_iterations"] = warmup_iterations
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
        collector_config={"runtime_backend_version": request.identity.runtime_framework_version},
        generator_overrides=overrides,
        options=replace(
            case["plan"].options,
            benchmark_points_json=None,
            benchmark_points_sha256=None,
            prefill_cudagraph_policy=request.collection.prefill_cudagraph_policy,
            max_prefill_cudagraph_size=request.collection.max_prefill_cudagraph_size,
            **({"warmup_iterations": warmup_iterations} if warmup_iterations is not None else {}),
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
        if not args.smoke:
            return runner.run_collection(
                resolved[0],
                generator_overrides=resolved[1],
                checkpoint_dir=str(root / "fpm-checkpoint"),
                artifact_root=str(root / "fpm-artifacts"),
                database_root=str(root / "systems/data"),
                resume=True,
                retry_failed=False,
            )
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


def test_readiness_matches_the_request_warmup_iterations(quality_case, tmp_path):
    request, root, plan, overrides = _bounded_source(quality_case, tmp_path, warmup_iterations=0)
    assert plan.options.warmup_iterations == 0
    _write_bounded_native(plan, root, overrides, smoke=True)
    assert assess_readiness(request, root, root / "fpm-checkpoint")["ready_for_full_collection"]
    # A request without the field asks for the default count, and a request with a count asks for that count, so
    # neither one matches the saved plan of the other.
    differs = "saved collection differs from the reviewed onboarding request"
    report = assess_readiness(quality_case["request"], root, root / "fpm-checkpoint")
    assert not report["ready_for_full_collection"]
    assert f"smoke: {differs}" in report["blockers"]
    report = assess_readiness(request, quality_case["root"], quality_case["checkpoint"].parent)
    assert not report["ready_for_full_collection"]
    assert f"formal: {differs}" in report["blockers"]


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


def _glm_smoke(tmp_path, monkeypatch, shape=None):
    """Write the smoke campaign of a probe-backed GLM-5.3-Flash collection plan, as a vLLM 0.33.0 runtime fills it.

    Each cell has its generated launch, the producer's rank artifact of the smoke points and one execution record per
    TP rank, which the execution observer writes from a stand-in vLLM 0.33.0 configuration of that launch. The rank
    artifacts have no measurement protocol, benchmark measurement or warm-up evidence. The producer's scheduler base,
    ai-dynamo 1.7.0.dev20261109, writes them, and ``_glm_producer_shape`` adds them. The profile names no probe
    provenance, so readiness does not bind the runtime memory to probe observations here. ``shape``, if given, rewrites
    each rank artifact before it is written. Return the request and the onboarding root.
    """
    from types import SimpleNamespace

    from collector.fpm_forward.config import with_kv_warmup_defaults
    from collector.fpm_forward.runtime import fpm_memory_observer as observer

    from aisim.support.schema import SupportRequest

    from .collector.test_fpm_glm53flash_native_collection import (
        DECODE_POINTS,
        PREFILL_CACHED,
        PREFILL_COLD,
        glm_native_payload,
        write_glm_rank,
    )
    from .collector.test_fpm_runner import _write_provenance
    from .collector.test_fpm_runtime_memory import (
        _mamba_group,
        _probe_backed_glm_plan,
        _typed,
        _uniform_mla_group,
        _vllm_config,
    )

    plan, overrides = _probe_backed_glm_plan(tmp_path)
    revision = plan.runtime_launch["identity"]["model_revision"]
    root = tmp_path / "onboarding"
    campaign = root / "fpm-artifacts" / plan.sha256[:16] / "smoke"
    runner._atomic_json(campaign / "collection-plan.json", plan.to_dict())
    runner._atomic_json(campaign / "generator-overrides.json", with_kv_warmup_defaults(overrides))
    config = _vllm_config(tp=4, dp=1)
    config.model_config = SimpleNamespace(
        model=plan.model_path,
        revision=revision,
        dtype="torch.bfloat16",
        quantization="fp8",
        max_model_len=plan.options.vllm_max_model_len,
        enforce_eager=False,
        # transformers stores the commit of the loaded configuration on the configuration object.
        hf_config=SimpleNamespace(_commit_hash=revision),
    )
    config.quant_config = _typed("vllm.model_executor.layers.quantization.fp8", "Fp8Config")
    vars(config.quant_config).update(activation_scheme="dynamic", weight_block_size=[128, 128])
    vars(config.cache_config).update(
        cache_dtype="fp8_e4m3",
        num_gpu_blocks_override=None,
        block_size=2176,
        mamba_cache_mode="align",
        prefix_match_unit=None,
    )
    vars(config.scheduler_config).update(max_num_batched_tokens=8192, max_num_seqs=256)
    vars(config.parallel_config).update(
        enable_expert_parallel=False, enable_eplb=False, data_parallel_size_local=1, data_parallel_external_lb=False
    )
    group = SimpleNamespace(
        backend=type("ResolvedBackend", (), {"__module__": "actual.runtime"}),
        layer_names=["model.layers.3.self_attn.attn"],
        kv_cache_group_id=0,
    )
    # The packed GLM-5.3-Flash groups: sparse attention and KDA state, each a UniformTypeKVCacheSpecs group.
    cache = SimpleNamespace(kv_cache_groups=[_uniform_mla_group(), _mamba_group()])
    worker = SimpleNamespace(
        vllm_config=config, model_runner=SimpleNamespace(attn_groups=[[group]], kv_cache_config=cache)
    )
    points = {"prefill": [PREFILL_COLD, PREFILL_CACHED], "decode": DECODE_POINTS}
    checkpoint = {"schema": runner.CHECKPOINT_SCHEMA, "plan_sha256": plan.sha256, "cells": {}}
    for cell in plan.cells:
        directory = campaign / "cells" / cell.cell_id
        pod = directory / "raw" / "pod-0"
        pod.mkdir(parents=True)
        runner._render_cell(plan, cell, directory, overrides, smoke=True)
        attempt = f"smoke-{cell.workload_kind}"
        _write_provenance(
            pod / "collector-provenance.json", cell_id=cell.cell_id, plan_sha256=plan.sha256, attempt_id=attempt
        )
        provenance = json.loads((pod / "collector-provenance.json").read_text())
        provenance["runtime"]["backend_version"] = "0.33.0"
        (pod / "collector-provenance.json").write_text(json.dumps(provenance))
        payload = glm_native_payload(cell.workload_kind, points=points[cell.workload_kind])
        # The producer records the revision that the engine loaded and the runtime context of the launch.
        payload["input_provenance"]["tokenizer_revision"] = revision
        payload["context_policy"]["runtime_context_length"] = plan.options.vllm_max_model_len
        if shape is not None:
            shape(cell.workload_kind, payload)
        write_glm_rank(pod, payload)
        with monkeypatch.context() as patch:
            patch.setattr(observer, "RESULTS_DIR", pod)
            patch.setattr(observer.importlib.metadata, "version", lambda _name: "0.33.0")
            for name in ("FPM_EXECUTION_OUTPUT_DIR", "FPM_EXECUTION_PROVENANCE_FILE"):
                patch.delenv(name, raising=False)
            for tp_rank in range(cell.topology.tp):
                observer.observe_execution(worker, dp_rank=0, tp_rank=tp_rank, pp_rank=0)
        checkpoint["cells"][cell.cell_id] = {"status": "passed", "attempt_id": attempt}
    runner._atomic_json(root / "fpm-checkpoint" / "fpm_forward_smoke.json", checkpoint)
    request = SupportRequest.model_validate(
        {
            "identity": plan.runtime_launch["identity"],
            "search": {
                "tensor_parallel": 4,
                "attention_data_parallel": 1,
                "moe_tensor_parallel": 4,
                "moe_expert_parallel": 1,
                "context_length": 1_048_576,
            },
            "collection": {
                "max_num_tokens": 8192,
                "max_batch_size": 256,
                "prefill_cudagraph_policy": "runtime",
                "gpu_memory_utilization": 0.9,
                "warmup_iterations": 0,
            },
            "fpm_profile": plan.to_dict()["fpm_profile"],
        }
    )
    return request, root


def test_readiness_blocks_a_glm_smoke_without_the_producer_protocol(tmp_path, monkeypatch):
    # The producer's scheduler base writes the measurement protocol into every rank artifact, so an artifact without it
    # is not the producer's.
    request, root = _glm_smoke(tmp_path, monkeypatch)
    report = assess_readiness(request, root, root / "fpm-checkpoint")
    assert not report["ready_for_full_collection"]
    message = "measurement_protocol of a GLM-5.3-Flash producer artifact must be present"
    assert {item["cell"]["workload_kind"] for item in report["selected_cells"]} == {"prefill", "decode"}
    for item in report["selected_cells"]:
        assert item["status"] != "ready"
        assert any(message in blocker for blocker in item["blockers"]), item["blockers"]


def test_readiness_qualifies_a_glm_smoke_of_dynamo_17_producer_artifacts(tmp_path, monkeypatch):
    request, root = _glm_smoke(tmp_path, monkeypatch, shape=_glm_producer_shape(tmp_path))
    report = assess_readiness(request, root, root / "fpm-checkpoint")
    assert report["blockers"] == []
    assert report["ready_for_full_collection"]
    cells = {item["cell"]["workload_kind"]: item for item in report["selected_cells"]}
    assert set(cells) == {"prefill", "decode"}
    # The cold prefill point has no KV to label. The cached prefill and the decode points were seeded with real KV.
    assert cells["prefill"]["formal_regime_counts"] == {"n/a": 1, "real_kv": 1}
    assert cells["decode"]["formal_regime_counts"] == {"real_kv": 2}
    for item in cells.values():
        assert (item["campaign"], item["status"], item["blockers"]) == ("smoke", "ready", [])
        assert item["kvwarm"]["method"] == "native_prefix_cache_real_seed"
        # vLLM 0.33.0 execution is audited, and the observed configuration agrees with the plan and its launch.
        assert item["execution"]["failures"] == []
        assert [reason for reason in item["execution"]["missing_evidence"] if "unaudited" in reason] == []
        # The measurement record of every point is read, so no legacy note remains.
        missing = item["execution"]["missing_evidence"]
        assert not [reason for reason in missing if "legacy" in reason or "warmup evidence is unreported" in reason]


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        ("interval", "benchmark_sample contradicts its local timing interval"),
        # The artifact's timing totals stay consistent, so the reduction check of the measurement record refuses it.
        ("median", "benchmark_measurement reduction disagrees with retained timing"),
    ],
)
def test_readiness_blocks_a_glm_smoke_with_a_broken_producer_record(tmp_path, monkeypatch, breakage, message):
    request, root = _glm_smoke(tmp_path, monkeypatch, shape=_glm_producer_shape(tmp_path, breakage))
    report = assess_readiness(request, root, root / "fpm-checkpoint")
    assert not report["ready_for_full_collection"]
    for item in report["selected_cells"]:
        assert item["status"] != "ready"
        assert any(message in blocker for blocker in item["blockers"]), item["blockers"]


def _then(shape, change):
    """``shape``, then ``change`` on the same rank artifact."""

    def both(phase, payload):
        shape(phase, payload)
        change(phase, payload)

    return both


def _other_vllm(_phase, payload):
    # A patch release of vLLM is another runtime. The producer runs only on vLLM 0.33.0.
    payload["producer"]["vllm_package_version"] = "0.33.1"


def test_readiness_blocks_a_glm_smoke_whose_producer_names_another_vllm(tmp_path, monkeypatch):
    request, root = _glm_smoke(tmp_path, monkeypatch, shape=_then(_glm_producer_shape(tmp_path), _other_vllm))
    report = assess_readiness(request, root, root / "fpm-checkpoint")
    assert not report["ready_for_full_collection"]
    message = "lacks valid producer.vllm_package_version: expected '0.33.0', found '0.33.1'"
    assert {item["cell"]["workload_kind"] for item in report["selected_cells"]} == {"prefill", "decode"}
    for item in report["selected_cells"]:
        assert item["status"] != "ready"
        assert any(message in blocker for blocker in item["blockers"]), item["blockers"]


def _nine_repetitions(_phase, payload):
    """Rewrite the producer's record of each point into a consistent record of nine measured repetitions.

    The protocol names nine repetitions and random content, and each point keeps its first nine raw samples, the
    indices of samples 1 to 8 and their upper median. The record still expects ten internal samples.
    """
    protocol = payload["measurement_protocol"]
    protocol["producer"]["measurement_repeats"] = 9
    protocol["synthetic_content"] = "random"
    for row, group in zip(payload["results"], payload["iteration_groups"], strict=True):
        fpm = row["fpms"][0]
        measurement = fpm["benchmark_measurement"]
        del measurement["raw_fpms"][9:]
        measurement["estimate"]["raw_sample_indices"] = list(range(1, 9))
        walls = sorted(sample["wall_time"] for sample in measurement["raw_fpms"][1:])
        fpm["wall_time"] = group["wall_time"] = walls[len(walls) // 2]
        assert measurement["expected_internal_samples"] == 10
    payload["timing"]["measured_iteration_seconds"] = sum(group["wall_time"] for group in payload["iteration_groups"])


def test_readiness_blocks_a_glm_smoke_whose_protocol_names_nine_repetitions(tmp_path, monkeypatch):
    request, root = _glm_smoke(tmp_path, monkeypatch, shape=_then(_glm_producer_shape(tmp_path), _nine_repetitions))
    report = assess_readiness(request, root, root / "fpm-checkpoint")
    assert not report["ready_for_full_collection"]
    message = "measurement_protocol.producer must be the GLM-5.3-Flash producer's record: expected "
    assert {item["cell"]["workload_kind"] for item in report["selected_cells"]} == {"prefill", "decode"}
    for item in report["selected_cells"]:
        assert item["status"] != "ready"
        assert any(message in blocker and "'measurement_repeats': 9" in blocker for blocker in item["blockers"]), item[
            "blockers"
        ]


def test_measurement_reader_holds_each_glm_cell_to_the_producer_protocol(tmp_path, monkeypatch):
    # The readers of a frozen GLM-5.3-Flash plan, without readiness around them.
    from collector.fpm_forward.measurement_evidence import extract_measurement_evidence
    from collector.fpm_forward.native_artifact import validate_native_collection
    from collector.fpm_forward.repeatability import load_repeatability_source

    _request, root = _glm_smoke(tmp_path, monkeypatch, shape=_then(_glm_producer_shape(tmp_path), _nine_repetitions))
    (campaign,) = [path / "smoke" for path in (root / "fpm-artifacts").iterdir()]
    plan = load_repeatability_source(campaign)
    assert {cell.workload_kind for cell in plan.cells} == {"prefill", "decode"}
    for cell in plan.cells:
        raw = campaign / "cells" / cell.cell_id / "raw"
        collection = validate_native_collection(
            cell, raw, expected_plan_sha256=plan.sha256, expected_attempt_id=f"smoke-{cell.workload_kind}"
        )
        with pytest.raises(
            ValueError,
            match=r"^measurement_protocol\.producer must be the GLM-5\.3-Flash producer's record: expected \{.+\}, "
            r"found \{.+'measurement_repeats': 9.+\}: .+benchmark\.json$",
        ):
            extract_measurement_evidence(raw, collection)
