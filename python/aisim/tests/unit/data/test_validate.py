# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR validation with a fake transport; no network, GitHub or credentials."""

import hashlib
import json

import pytest

from aisim_core.data import Identity, configuration_id
from aisim_core.data.eval import EvalRecord, GpuTime, PredictionIdentity, compute_results, write_truth_table
from aisim_core.data.fetch import Cache
from aisim_core.data.validate import (
    PrHead,
    PrValidationError,
    PrValidationReport,
    StaleHeadError,
    ValidationRequest,
    check_current,
    evaluator_revision,
    main,
    parse_request,
    refuse_credentials,
    resolve_pr_head,
    run_validation,
)

pytestmark = pytest.mark.unit

REPO = "org/dataset"
DSHA = "d" * 40
HEAD = "1" * 40
BASE = "2" * 40
TRUSTED = "3" * 40
SPEC = "a" * 64
PIN = "b" * 64
RECORD_PATH = "eval/record.json"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _art(artifact_id, path, data, fmt, schema, **extra):
    return {
        "artifact_id": artifact_id,
        "path": path,
        "role": "x",
        "format": fmt,
        "schema": schema,
        "sha256": _sha(data),
        "bytes": len(data),
        **extra,
    }


def _predict(inputs):
    return inputs["tokens"] * 1.1


def _identity():
    return Identity(
        hardware="b200_sxm",
        framework="vllm",
        framework_version="0.11.0",
        operation_family=None,
        model="Qwen/Qwen3-32B",
        effective_settings={},
        unresolved_identity=None,
    )


def build_dataset(tmp_path, claim_scale=1.0):
    rows = [(f"s{i}", {"tokens": 10 * i}, 10.0 * i) for i in range(1, 5)] + [("bad", {"tokens": 1}, None)]
    art = write_truth_table(tmp_path / "t.parquet", rows)
    truth_bytes = (tmp_path / "t.parquet").read_bytes()
    art = art.model_copy(update={"path": "eval/truth.parquet"})
    cases = [(c, t, _predict(i) * claim_scale) for c, i, t in rows]
    pred = PredictionIdentity.build("fpm_interpolation", {"p": PIN}, SPEC)
    ident = _identity()
    cfg = configuration_id("eval", "evaluation-record", ident)
    record = EvalRecord.model_validate(
        {
            "schema_version": "aisim-eval/v1",
            "prediction": pred.model_dump(),
            "truth": {
                "layer": "fpm",
                "configuration_ids": [cfg],
                "identity": ident.model_dump(),
                "artifacts": [art.model_dump(by_alias=True)],
                "truth_table_artifact_id": "truth-table",
            },
            "results": compute_results(cases).model_dump(),
            "assumptions": [],
            "gpu_time": GpuTime(authority="unknown").model_dump(),
            "simulation_cost": {"wall_seconds": 1.0, "gpu_seconds": 0.0},
            "producer": {"kind": "human", "name": "t", "evaluator_revision": "r"},
            "tier": "submitted",
            "created_at": "2026-11-07T00:00:00Z",
        }
    )
    rec_name = f"eval/{record.record_id}.json"
    rec_bytes = json.dumps(record.model_dump(by_alias=True, mode="json"), indent=2, sort_keys=True).encode() + b"\n"
    manifest = {
        "schema_version": "aisim-artifact-manifest/v1",
        "category": "eval",
        "purpose": "evaluation-record",
        "configuration_id": cfg,
        "identity": ident.model_dump(),
        "requested_settings": None,
        "artifacts": [
            _art("r", rec_name, rec_bytes, "json", "aisim-eval/v1"),
            _art("truth-table", "eval/truth.parquet", truth_bytes, "parquet", "aisim-eval-truth/v1", rows=len(rows)),
        ],
        "provenance": {},
        "evidence_gaps": [],
    }
    files = {rec_name: rec_bytes, "eval/truth.parquet": truth_bytes}
    files["eval/manifest.json"] = json.dumps(manifest).encode()
    index = {
        "schema_version": "aisim-category-index/v1",
        "category": "eval",
        "entries": [
            {
                "configuration_id": cfg,
                "purpose": "evaluation-record",
                "identity": ident.model_dump(),
                "manifest": _art(
                    "m", "eval/manifest.json", files["eval/manifest.json"], "json", "aisim-artifact-manifest/v1"
                ),
            }
        ],
    }
    files["catalog/eval.json"] = json.dumps(index).encode()
    dataset = {
        "schema_version": "aisim-dataset-index/v1",
        "dataset_id": REPO,
        "categories": {
            "eval": {
                "root": "eval",
                "layout": "category",
                "index": _art("i", "catalog/eval.json", files["catalog/eval.json"], "json", "aisim-category-index/v1"),
            }
        },
    }
    files["catalog/dataset.json"] = json.dumps(dataset).encode()
    return files, rec_name, cfg, record


class FakeTransport:
    def __init__(self, files, fail=None):
        self.files, self.fail, self.opened = files, fail, []

    def resolve_revision(self, repo, revision):
        return DSHA

    def open_file(self, repo, sha, path):
        if self.fail:
            raise self.fail
        self.opened.append(path)
        yield self.files[path]


def request(rec_name, cfg, **kw):
    base = {
        "schema_version": "aisim-pr-validation/v1",
        "dataset_repo": REPO,
        "dataset_revision": DSHA,
        "record_path": rec_name,
        "configuration_ids": [cfg],
        "hardware": "b200_sxm",
        "model": "Qwen/Qwen3-32B",
        "required_cases": ["s1", "s2"],
    }
    return ValidationRequest.model_validate({**base, **kw})


def pr(head=HEAD):
    return PrHead(REPO, 7, head, BASE, "open")


def factory(record):
    return _predict, record.prediction.fingerprint


def run(tmp_path, files, req, factory=factory, transport=None):
    return run_validation(
        req,
        pr(),
        trusted_sha=TRUSTED,
        transport=transport or FakeTransport(files),
        predictor_factory=factory,
        cache=Cache(tmp_path / "cache"),
    )


def test_pass_binds_all_identities(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    req = request(rec, cfg)
    r = run(tmp_path, files, req)
    assert r.verdict == "pass", r.problems
    assert (r.code_sha, r.dataset_sha, r.request_digest) == (HEAD, DSHA, req.digest())
    assert r.evaluator_revision == evaluator_revision(TRUSTED)
    assert [o.status for o in r.required_cases] == ["pass", "pass"]
    assert r.accuracy is not None and r.accuracy.n_scored == 4
    assert PrValidationReport.model_validate_json(r.model_dump_json()) == r


def test_wrong_claims_fail(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path, claim_scale=1.5)
    r = run(tmp_path, files, request(rec, cfg))
    assert r.verdict == "fail" and any("mape_pct" in p for p in r.problems)
    assert all(o.status == "fail" for o in r.required_cases)


def test_required_case_unpredicted_or_missing_or_excluded(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg, required_cases=["bad", "nope"]))
    reasons = {o.case_id: o.reason for o in r.required_cases}
    assert r.verdict == "fail"
    assert reasons["bad"] == "excluded by producer" and reasons["nope"] == "not in truth table"
    r = run(tmp_path, files, request(rec, cfg), factory=lambda rec_: (lambda i: None, rec_.prediction.fingerprint))
    assert r.verdict == "fail"


def test_fingerprint_mismatch_fails(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg), factory=lambda rec_: (_predict, "f" * 64))
    assert r.verdict == "fail" and "fingerprint" in r.problems[0]


def test_record_not_selected_fails(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    assert run(tmp_path, files, request("eval/other.json", cfg)).verdict == "fail"


def test_outage_is_error_not_substitution(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg), transport=FakeTransport(files, fail=ConnectionError("hf down")))
    assert r.verdict == "error" and "hf down" in r.problems[0]


def test_budget_exceeded_is_error(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg, max_bytes=10))
    assert r.verdict == "error" and "BudgetExceededError" in r.problems[0]


def test_changed_head_request_or_evaluator_invalidates(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    req = request(rec, cfg)
    r = run(tmp_path, files, req)
    rev = evaluator_revision(TRUSTED)
    assert check_current(r, head_sha=HEAD, request_digest=req.digest(), evaluator_rev=rev, dataset_sha=DSHA) == []
    assert check_current(r, head_sha="9" * 40, request_digest=req.digest(), evaluator_rev=rev)
    edited = request(rec, cfg, required_cases=["s1"])
    assert edited.digest() != req.digest()
    assert check_current(r, head_sha=HEAD, request_digest=edited.digest(), evaluator_rev=rev)
    assert check_current(r, head_sha=HEAD, request_digest=req.digest(), evaluator_rev=evaluator_revision("4" * 40))
    assert check_current(r, head_sha=HEAD, request_digest=req.digest(), evaluator_rev=rev, dataset_sha="e" * 40)


def test_resolve_pr_head_rules():
    assert resolve_pr_head(pr(), HEAD).head_sha == HEAD
    with pytest.raises(StaleHeadError):
        resolve_pr_head(pr("9" * 40), HEAD)
    with pytest.raises(PrValidationError):
        resolve_pr_head(PrHead(REPO, 1, "abc", BASE, "open"), None)
    with pytest.raises(PrValidationError):
        resolve_pr_head(PrHead(REPO, 1, HEAD, BASE, "closed"), None)
    payload = {"number": 3, "state": "open", "merged": False, "head": {"sha": HEAD}, "base": {"sha": BASE}}
    assert PrHead.from_github(payload, REPO).number == 3
    with pytest.raises(PrValidationError):
        PrHead.from_github({}, REPO)


def test_request_parsing_is_strict():
    ok = {
        "schema_version": "aisim-pr-validation/v1",
        "dataset_repo": REPO,
        "dataset_revision": DSHA,
        "record_path": "a",
        "configuration_ids": ["c"],
        "required_cases": ["x"],
    }
    assert parse_request(json.dumps(ok)).max_bytes > 0
    for bad in (
        {**ok, "dataset_revision": "main"},
        {**ok, "extra": 1},
        {**ok, "max_bytes": 1 << 40},
        {**ok, "required_cases": []},
    ):
        with pytest.raises(PrValidationError):
            parse_request(json.dumps(bad))
    with pytest.raises(PrValidationError):
        parse_request("not json")


def test_refuse_credentials():
    refuse_credentials({"PATH": "/bin"})
    with pytest.raises(PrValidationError):
        refuse_credentials({"HF_TOKEN": "x"})


def test_cli_run_and_recheck(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    files, rec, cfg, _ = build_dataset(tmp_path)
    req = request(rec, cfg)
    (tmp_path / "req.json").write_text(req.model_dump_json())
    payload = {"number": 7, "state": "open", "head": {"sha": HEAD}, "base": {"sha": BASE}}
    (tmp_path / "pr.json").write_text(json.dumps(payload))
    common = [
        "--pr-json",
        str(tmp_path / "pr.json"),
        "--repo",
        REPO,
        "--request-file",
        str(tmp_path / "req.json"),
        "--trusted-sha",
        TRUSTED,
    ]
    out = tmp_path / "report.json"
    run_args = [
        "run",
        *common,
        "--expected-head",
        HEAD,
        "--predictor-factory",
        "aisim_core.data.validate:nope",
        "--cache-dir",
        str(tmp_path / "c"),
        "--output",
        str(out),
        "--summary",
        str(tmp_path / "sum.md"),
    ]
    assert main(run_args, transport=FakeTransport(files)) == 2  # unknown factory is bad input
    i = run_args.index("--predictor-factory") + 1
    bad_factory = [*run_args[:i], "os:path", *run_args[i + 1 :]]
    assert main(bad_factory) == 2  # outside the trusted prefixes

    monkeypatch.setattr("aisim_core.data.validate._load_factory", lambda spec: factory)
    assert main(run_args, transport=FakeTransport(files)) == 0
    assert "PR validation: PASS" in (tmp_path / "sum.md").read_text()
    report = PrValidationReport.model_validate_json(out.read_text())
    assert report.verdict == "pass"

    recheck = ["recheck", *common, "--report", str(out)]
    assert main(recheck) == 0
    payload["head"]["sha"] = "9" * 40  # the PR head moved
    (tmp_path / "pr.json").write_text(json.dumps(payload))
    assert main(recheck) == 1
    assert "PR head changed" in capsys.readouterr().err
    # a dispatch for the old head is refused outright
    assert main(run_args, transport=FakeTransport(files)) == 2
    monkeypatch.setenv("HF_TOKEN", "x")
    assert main(recheck) == 2


def test_default_predictor_fails_closed(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    from aisim_core.data.validate import no_predictor

    with pytest.raises(PrValidationError):
        no_predictor(build_dataset(tmp_path)[3])
