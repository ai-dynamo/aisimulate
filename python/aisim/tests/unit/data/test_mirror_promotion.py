# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirror validator, redaction rules and promotion procedure, on synthetic data only."""

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from aisim_core.data import Identity
from aisim_core.data.catalog import build_catalog
from aisim_core.data.e2e import (
    E2eManifestBody,
    Setting,
    aiperf_to_configuration,
    inferencex_to_configurations,
    write_configuration,
)
from aisim_core.data.eval import (
    EvalRecord,
    PredictionIdentity,
    compute_results,
    record_dir,
    write_record,
    write_truth_table,
)
from aisim_core.data.mirror import (
    INFX_CACHE_REPO,
    INTERNAL_REPO,
    validate_tree,
    validator_fingerprint,
)
from aisim_core.data.mirror import (
    main as mirror_main,
)
from aisim_core.data.promote import (
    Approval,
    PromotionError,
    PromotionPolicy,
    PromotionRequest,
    apply_promotion,
    plan_promotion,
    register_manifests,
)
from aisim_core.data.redaction import (
    VisibilityEntry,
    VisibilityLedger,
    check_disclosure,
    check_publishable,
    write_ledger,
)

pytestmark = pytest.mark.unit

PUBLIC = "nvidia/aisim-data-test"
COMMIT = "a" * 40
CLEAN = {"producer_id": None, "image": None, "raw_artifact_id": None, "log_artifact_id": None, "evidence_refs": []}


def _body(model="synthetic/model-a", source="developer"):
    return E2eManifestBody(
        model_id=model,
        deployment="aggregated",
        hardware="h100",
        framework="vllm",
        framework_version="0.11.0",
        parallelism={"aggregated": {"tp": 8}},
        parallelism_complete=True,
        precision={"weights": "fp8"},
        settings={"max_num_seqs": Setting(value=256, authority="observed")},
        source_kind=source,
    )


def _summary():
    stat = {"unit": "ms", "avg": 100.0, "p50": 90.0, "p90": 150.0, "p99": 300.0}
    return {
        "is_complete": True,
        "was_cancelled": False,
        "request_count": {"avg": 2},
        "error_request_count": {"avg": 0},
        "time_to_first_token": stat,
        "inter_token_latency": {"unit": "ms", "avg": 10.0, "p50": 9.0, "p90": 14.0},
        "request_latency": {"unit": "seconds", "avg": 1.5, "p50": 1.4, "p90": 2.0},
        "output_token_throughput": {"unit": "tokens/second", "avg": 9600.0},
    }


def _records():
    metrics = {
        "input_sequence_length": {"value": 1024, "unit": "tokens"},
        "output_sequence_length": {"value": 128, "unit": "tokens"},
        "time_to_first_token": {"value": 0.1, "unit": "seconds"},
        "inter_token_latency": {"value": 10, "unit": "ms"},
    }
    return [{"request_id": i, "error": None, "metrics": metrics} for i in range(2)]


def _config(model="synthetic/model-a", *, clean=True, **kw):
    cfg = aiperf_to_configuration(
        _summary(), _records(), _body(model), observation_id=f"dev:{model}", device_count=8, **kw
    )
    if clean:  # the redacted, aggregate-only form that may be public
        cfg = dataclasses.replace(
            cfg,
            observations=[o.model_copy(update=CLEAN) for o in cfg.observations],
            requests=[],
            provenance=[],
        )
    return cfg


def _eval(root, configuration_ids, *, case_ids=5):
    rows = [(f"s{i}", {"tokens": 10 * i}, 10.0 * i) for i in range(1, case_ids + 1)]
    art = write_truth_table(root / "truth.parquet", rows)
    ident = Identity(
        hardware="h100", framework="vllm", framework_version="0.11.0", operation_family=None,
        model="synthetic/model-a", effective_settings={}, unresolved_identity=None,
        parallelism={"aggregated": {"tp": 8}}, parallelism_complete=True, deployment="aggregated",
    )  # fmt: skip
    results = compute_results([(c, t, t * 1.1) for c, _, t in rows])
    record = EvalRecord.model_validate(
        {
            "schema_version": "aisim-eval/v1",
            "prediction": PredictionIdentity.build("fpm_interpolation", {"p": "b" * 64}, "a" * 64).model_dump(),
            "truth": {
                "layer": "e2e",
                "configuration_ids": configuration_ids,
                "identity": ident.model_dump(),
                "artifacts": [art.model_dump(by_alias=True)],
                "truth_table_artifact_id": "truth-table",
            },
            "results": results.model_dump(),
            "assumptions": [],
            "simulation_cost": {},
            "producer": {"kind": "agent", "name": "t", "evaluator_revision": "r1"},
            "tier": "submitted",
            "created_at": "2026-11-07T00:00:00Z",
        }
    )
    # The truth table sits next to the record, under the record's identity directory.
    directory = record_dir(root, ident)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.move(root / "truth.parquet", directory / "truth.parquet")
    return write_record(root, record)


def _mirror(tmp_path, *, repo=INTERNAL_REPO, dev=("synthetic/model-a",)):
    root = tmp_path / "mirror"
    root.mkdir()
    manifests = [write_configuration(root, _config(m), prefix="data") for m in dev]
    register_manifests(root, manifests, repo)
    build_catalog(root)
    return root, manifests


def _public(root, *prefixes, **entry):
    write_ledger(root, VisibilityLedger(entries=[VisibilityEntry(path=p, label="public", **entry) for p in prefixes]))


def _reviewers():
    return PromotionPolicy(reviewers=frozenset({"alice", "bob"}))


def _request(directories, plan=None, **kw):
    approvals = [Approval(reviewer="alice", approved_at="2026-11-09", content_digest=plan.digest)] if plan else []
    return PromotionRequest(
        target_repo=PUBLIC,
        source_commit=COMMIT,
        directories=directories,
        requested_by="carol",
        approvals=approvals,
        **kw,
    )


# ---------------------------------------------------------------- 2343: mirror validator


def test_clean_mirror_passes_with_repo_id_and_fingerprint(tmp_path):
    root, _ = _mirror(tmp_path)
    report = validate_tree(root, repo=INTERNAL_REPO)
    assert report.ok, report.errors
    assert report.configurations == 1 and report.fingerprint == validator_fingerprint()
    # The expected repository is configurable; a different id is refused.
    assert "dataset-id-mismatch" in {f.code for f in validate_tree(root, repo="ai-dynamo/other").errors}
    assert mirror_main(["validate", "--root", str(root)]) == 0


def test_infx_cache_is_validated_with_its_own_repo_id(tmp_path):
    root, _ = _mirror(tmp_path, repo=INFX_CACHE_REPO)
    assert validate_tree(root, repo=INFX_CACHE_REPO).ok


def test_empty_mirror_needs_explicit_bootstrap(tmp_path):
    (tmp_path / "e").mkdir()
    assert not validate_tree(tmp_path / "e").ok
    assert validate_tree(tmp_path / "e", allow_empty=True).ok


@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_internal_fails_on_everything_the_public_validator_rejects(tmp_path, profile):
    """Exit gate: one function judges both repositories, so every defect below fails both."""
    root, manifests = _mirror(tmp_path, repo=PUBLIC if profile == "public" else INTERNAL_REPO)
    repo = PUBLIC if profile == "public" else INTERNAL_REPO
    assert validate_tree(root, repo=repo, profile=profile).ok
    directory = (root / manifests[0]).parent
    metrics = directory / "metrics.parquet"
    original = metrics.read_bytes()

    metrics.write_bytes(original[:-9] + b"corrupt!!")  # tampered table
    assert "manifest-invalid" in {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    metrics.write_bytes(original)

    (root / "eval").mkdir()
    (root / "eval" / "bad.json").write_text("{not json")  # malformed record
    assert "eval-record-invalid" in {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    (root / "eval" / "bad.json").unlink()

    (root / "data" / "tool.py").write_text("print(1)")  # executable content
    codes = {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    assert {"suffix"} <= codes
    (root / "data" / "tool.py").unlink()

    (root / "data" / "link.json").symlink_to(metrics)
    assert "symlink" in {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    (root / "data" / "link.json").unlink()
    assert validate_tree(root, repo=repo, profile=profile).ok


def test_catalog_drift_and_schema_drift_fail(tmp_path):
    root, _ = _mirror(tmp_path)
    shard = next((root / "catalog").rglob("*.parquet"))
    shard.write_bytes(shard.read_bytes()[:-5] + b"xxxxx")
    assert "catalog-drift" in {f.code for f in validate_tree(root, repo=INTERNAL_REPO).errors}
    build_catalog(root)
    schemas = root / "catalog" / "schemas"
    schemas.mkdir()
    (schemas / "aisim-eval.v1.json").write_text("{}\n")
    assert "schema-drift" in {f.code for f in validate_tree(root, repo=INTERNAL_REPO).errors}


def test_mirror_control_files_are_private_only(tmp_path):
    root, _ = _mirror(tmp_path)
    _public(root, "data/")
    assert validate_tree(root, repo=INTERNAL_REPO).ok
    report = validate_tree(root, repo=INTERNAL_REPO, profile="public")
    assert "control-file" in {f.code for f in report.errors}


# ---------------------------------------------------------------- 2344: redaction rules


def test_unredacted_run_url_image_digest_patch_and_logs_are_refused(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    dirty = _config(clean=False, log=b"server log\n" * 4, log_redacted=True)
    obs = [
        o.model_copy(
            update={"producer_id": "https://github.com/o/r/actions/runs/123", "image": "nvcr.io/x/y@sha256:" + "d" * 64}
        )
        for o in dirty.observations
    ]
    dirty = dataclasses.replace(dirty, observations=obs)
    path = write_configuration(root, dirty, prefix="data")
    files = [p.relative_to(root).as_posix() for p in (root / path).parent.iterdir()]
    codes = {f.code for f in check_publishable(root, files)}
    assert {"run-url", "provenance-withheld", "request-rows"} <= codes
    assert "image-digest" in codes or "image-name" in codes


def test_text_rules_cover_patch_image_and_secret_values():
    from aisim_core.data.redaction import DEFAULT_POLICY, scan_value

    hits = scan_value(
        {"patch_ref": "fix.patch", "container_image": "nvcr.io/nvidia/vllm:25.01", "note": "token hf_" + "a" * 30,
         "u": "https://gitlab-master.nvidia.com/a/b/-/jobs/42", "dispatch_mode": "fast", "ok": "[REDACTED]"},
        DEFAULT_POLICY, "p", "t",
    )  # fmt: skip
    codes = {f.code for f in hits}
    assert {"patch-reference", "image-name", "secret", "run-url", "internal-host"} <= codes
    assert not [f for f in hits if "dispatch" in f.message]  # "dispatch" is not a patch key


def test_inferencex_derived_is_aggregates_only_by_default(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (infx,) = inferencex_to_configurations([_infx()])
    infx = dataclasses.replace(
        infx, observations=[o.model_copy(update=CLEAN) for o in infx.observations], provenance=[], requests=[]
    )
    path = write_configuration(root, infx, prefix="data")
    files = [p.relative_to(root).as_posix() for p in (root / path).parent.iterdir()]
    assert "infx-aggregates-only" in {f.code for f in check_publishable(root, files)}
    from aisim_core.data.redaction import RedactionPolicy

    relaxed = RedactionPolicy(inferencex_disclosure="point-level")
    assert "infx-aggregates-only" not in {f.code for f in check_publishable(root, files, policy=relaxed)}


def _infx():
    return dict(
        model="synthetic/model-a", hardware="h100", framework="vllm", precision="fp8", spec_method="none",
        disagg=False, is_multinode=False, prefill_tp=1, prefill_ep=1, prefill_dp_attention=False,
        prefill_num_workers=0, decode_tp=8, decode_ep=1, decode_dp_attention=False, decode_num_workers=1,
        num_prefill_gpu=0, num_decode_gpu=8, id=1, date="2026-10-17", isl=1024, osl=128, conc=32,
        metrics={"mean_ttft": 0.1, "mean_tpot": 0.01, "tput_per_gpu": 1200.0},
        image="img", benchmark_type="single_turn", run_url="https://github.com/o/r/actions/runs/9",
    )  # fmt: skip


def test_aggregate_disclosure_allows_only_aggregate_fields_and_large_cells():
    ok = {"hardware": "h100", "model": "m", "framework": "vllm", "n_scored": 12, "mape_pct": 4.2, "coverage": 0.9}
    assert check_disclosure([ok]) == []
    leaky = {**ok, "case_id": "s1", "truth": 1.0, "run_url": "https://x/actions/runs/1"}
    assert {"not-aggregate"} <= {f.code for f in check_disclosure([leaky])}
    assert "small-cell" in {f.code for f in check_disclosure([{**ok, "n_scored": 2}])}


# ---------------------------------------------------------------- 2344: promotion


def _promotable(tmp_path):
    root, manifests = _mirror(tmp_path)
    directory = str(Path(manifests[0]).parent)
    _public(root, directory)
    return root, directory


def _target(tmp_path):
    target = tmp_path / "public"
    target.mkdir()
    return target


def test_sample_configuration_is_promoted_by_copy_only_and_passes_public_validator(tmp_path):
    """Exit gate for AISIM-2344."""
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request([directory]), _reviewers())
    assert draft.content_ok and not draft.ok  # needs approval
    plan = plan_promotion(root, _request([directory], draft), _reviewers())
    assert plan.ok, plan.findings

    target = _target(tmp_path)
    ledger = apply_promotion(plan, root, target)
    for rel, digest in plan.files.items():  # path-identical and byte-identical
        assert (target / rel).read_bytes() == (root / rel).read_bytes()
    assert not (target / "mirror").exists()
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok, report.errors
    assert report.configurations == 1
    entry = json.loads(ledger.read_text())
    assert entry["approved_by"] == ["alice"] and entry["source_commit"] == COMMIT
    # The source tree still validates as a mirror with its new ledger entry.
    assert validate_tree(root, repo=INTERNAL_REPO).ok


def test_unlabeled_and_restricted_paths_are_not_promoted(tmp_path):
    root, manifests = _mirror(tmp_path)
    directory = str(Path(manifests[0]).parent)
    plan = plan_promotion(root, _request([directory]), _reviewers())
    assert "not-public" in {f.code for f in plan.findings}
    write_ledger(root, VisibilityLedger(entries=[VisibilityEntry(path=directory, label="restricted")]))
    assert "not-public" in {f.code for f in plan_promotion(root, _request([directory]), _reviewers()).findings}


def test_approval_rules(tmp_path):
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request([directory]), _reviewers())
    stale = _request([directory], draft)
    stale = stale.model_copy(
        update={"approvals": [Approval(reviewer="alice", approved_at="d", content_digest="0" * 64)]}
    )
    assert "approval-stale" in {f.code for f in plan_promotion(root, stale, _reviewers()).findings}
    self_approved = _request([directory], draft).model_copy(update={"requested_by": "alice"})
    assert "approval-self" in {f.code for f in plan_promotion(root, self_approved, _reviewers()).findings}
    stranger = _request([directory], draft).model_copy(
        update={"approvals": [Approval(reviewer="mallory", approved_at="d", content_digest=draft.digest)]}
    )
    assert "approval-reviewer" in {f.code for f in plan_promotion(root, stranger, _reviewers()).findings}
    # No reviewers configured: fail closed.
    assert not plan_promotion(root, _request([directory], draft), PromotionPolicy()).ok


def test_changed_bytes_invalidate_approval_and_apply(tmp_path):
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request([directory]), _reviewers())
    plan = plan_promotion(root, _request([directory], draft), _reviewers())
    (root / directory / "metrics.parquet").write_bytes(b"changed")
    with pytest.raises(PromotionError, match="source changed"):
        apply_promotion(plan, root, _target(tmp_path))


def test_dirty_content_blocks_promotion_even_if_labeled_public(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    dirty = _config(clean=False)
    path = write_configuration(root, dirty, prefix="data")
    register_manifests(root, [path], INTERNAL_REPO)
    build_catalog(root)
    directory = str(Path(path).parent)
    _public(root, directory)
    plan = plan_promotion(root, _request([directory]), _reviewers())
    assert {"provenance-withheld", "request-rows"} <= {f.code for f in plan.findings}
    assert not plan.content_ok


def test_infx_cache_and_same_repo_are_never_sources(tmp_path):
    root, directory = _promotable(tmp_path)
    cache = _request([directory], source_repo=INFX_CACHE_REPO)
    assert "never-promoted" in {f.code for f in plan_promotion(root, cache, _reviewers()).findings}
    same = _request([directory], source_repo=PUBLIC)
    assert "same-repo" in {f.code for f in plan_promotion(root, same, _reviewers()).findings}


def test_selection_must_be_inside_data_or_eval_and_exist(tmp_path):
    root, directory = _promotable(tmp_path)
    codes = {f.code for f in plan_promotion(root, _request(["catalog", "../x", "data/nope"]), _reviewers()).findings}
    assert {"selection-root", "selection-path", "selection-missing"} <= codes


def test_target_is_restored_when_public_validator_rejects(tmp_path):
    root, directory = _promotable(tmp_path)
    target = _target(tmp_path)
    (target / "data").mkdir()
    (target / "data" / "stray.py").write_text("x")  # public tree already invalid: the result must be refused
    draft = plan_promotion(root, _request([directory]), _reviewers())
    plan = plan_promotion(root, _request([directory], draft), _reviewers())
    with pytest.raises(PromotionError, match="public validator rejects"):
        apply_promotion(plan, root, target)
    assert [p.name for p in (target / "data").iterdir()] == ["stray.py"] and not (target / "catalog").exists()


def test_snapshot_never_spans_private_and_public(tmp_path):
    from aisim_core.data.e2e import SnapshotMember, write_snapshot

    root, manifests = _mirror(tmp_path, dev=("synthetic/model-a", "synthetic/model-b"))
    public_dir, private_dir = (str(Path(m).parent) for m in manifests)
    from aisim_core.data import Manifest

    members = [
        SnapshotMember(
            configuration_id=Manifest.model_validate_json((root / m).read_bytes()).configuration_id,
            observation_id=f"o{i}", record_sha256="b" * 64, manifest_path=m,
        )
        for i, m in enumerate(manifests)
    ]  # fmt: skip
    snap_dir = root / "data" / "e2e-snapshots" / "s1"
    snap_dir.mkdir(parents=True)
    write_snapshot(
        root, "data/e2e-snapshots/s1/snapshot.json", members,
        source_scope={"kind": "synthetic"}, selection_policy="p", completeness="complete",
    )  # fmt: skip
    _public(root, public_dir, "data/e2e-snapshots/s1")
    plan = plan_promotion(root, _request([public_dir, "data/e2e-snapshots/s1"]), _reviewers())
    assert "snapshot-spans" in {f.code for f in plan.findings}
    _ = private_dir


def test_eval_record_is_promoted_with_its_configuration(tmp_path):
    root, manifests = _mirror(tmp_path)
    from aisim_core.data import Manifest

    cid = Manifest.model_validate_json((root / manifests[0]).read_bytes()).configuration_id
    record = _eval(root, [cid])
    build_catalog(root)
    eval_dir = str(record.parent.relative_to(root))
    cfg_dir = str(Path(manifests[0]).parent)
    _public(root, cfg_dir, eval_dir)
    alone = plan_promotion(root, _request([eval_dir]), _reviewers())
    assert "eval-truth-not-selected" in {f.code for f in alone.findings}
    draft = plan_promotion(root, _request([cfg_dir, eval_dir]), _reviewers())
    plan = plan_promotion(root, _request([cfg_dir, eval_dir], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok and report.eval_records == 1, report.errors
    # An eval record scoring InferenceX truth is not promoted by default.
    shutil.rmtree(target)


def test_lineage_of_redacted_originals_is_recorded_privately(tmp_path):
    root, directory = _promotable(tmp_path)
    original = "e" * 64
    path = f"{directory}/metrics.parquet"
    _public(root, directory, original_sha256={path: original})
    draft = plan_promotion(root, _request([directory]), _reviewers())
    plan = plan_promotion(root, _request([directory], draft), _reviewers())
    ledger = apply_promotion(plan, root, _target(tmp_path))
    assert json.loads(ledger.read_text())["original_sha256"] == {path: original}
