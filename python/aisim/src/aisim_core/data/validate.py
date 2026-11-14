# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PR validation (DEP #435, "CI and public evaluation").

A contributor publishes a Hugging Face candidate and adds a small validation
request (``aisim-pr-validation/v1``) to the PR. A maintainer dispatches the
trusted workflow ``.github/workflows/pr-validation.yml``, which uses this module
to:

1. resolve the PR head to a full commit SHA (:func:`resolve_pr_head`),
2. fetch only the evidence the request selects, at a pinned dataset commit,
   without credentials (:func:`run_validation` uses ``data.fetch``),
3. recompute the evaluation from recorded truth (``data.eval.verify_record``),
4. emit a report bound to the PR head SHA, evaluator revision, dataset SHA and
   request digest (:class:`PrValidationReport`).

The PR's code is data here and is never imported or executed: the predictor
comes from the trusted checkout. Required cases must pass; accuracy is advisory.
A changed PR head, request, evaluator or dataset commit invalidates a report
(:func:`check_current`). Failures such as a Hugging Face outage yield an
``error`` verdict with the reason; no other data is ever substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from aisim_core.data.eval import (
    TRUTH_SCHEMA_ID,
    VERIFIER_VERSION,
    EvalRecord,
    Results,
    load_record,
    verify_record,
)
from aisim_core.data.fetch import (
    Cache,
    DataFetchError,
    HfTransport,
    Selection,
    Transport,
    fetch,
)
from aisim_core.data.parquet import read_parquet

REQUEST_SCHEMA_ID = "aisim-pr-validation/v1"
REPORT_SCHEMA_ID = "aisim-pr-validation-report/v1"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
MAX_BYTES_CAP = 1024 * 1024 * 1024
_SHA = re.compile(r"^[0-9a-f]{40}$")
_CREDENTIAL_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME_TOKEN")
_PREDICTOR_PREFIXES = ("aisim_core.", "aisim.")

Predict = Callable[[Mapping[str, JsonValue]], float | None]
# Builds (predict, fingerprint) for a record. Lives in trusted code only.
PredictorFactory = Callable[[EvalRecord], tuple[Predict, str]]


class PrValidationError(Exception):
    """A request or PR head that cannot be validated (as opposed to a failed check)."""


class StaleHeadError(PrValidationError):
    """The PR head moved since the maintainer dispatched validation."""


# --------------------------------------------------------------------------- request


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ValidationRequest(_Model):
    """The small request file a contributor adds to the PR."""

    schema_version: Literal["aisim-pr-validation/v1"]
    dataset_repo: str = Field(pattern=r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
    # Full commit SHA of the candidate revision. A branch or `refs/pr/N` would move.
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    record_path: str = Field(min_length=1)  # eval record path inside the dataset
    configuration_ids: list[str] = Field(min_length=1)
    hardware: str | None = None
    model: str | None = None
    required_cases: list[str] = Field(min_length=1)
    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, gt=0, le=MAX_BYTES_CAP)

    def digest(self) -> str:
        """SHA-256 of the canonical request JSON; any edit changes it."""
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()

    def selection(self) -> Selection:
        return Selection(
            category="eval",
            configuration_ids=tuple(self.configuration_ids),
            hardware=self.hardware,
            model=self.model,
        )


def parse_request(raw: bytes | str) -> ValidationRequest:
    """Parse request bytes strictly. Raises :class:`PrValidationError` on any defect."""
    try:
        return ValidationRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise PrValidationError(
            f"invalid validation request: {exc.error_count()} error(s): {exc.errors()[0]['msg']}"
        ) from exc


# --------------------------------------------------------------------------- PR head


@dataclass(frozen=True)
class PrHead:
    repo: str  # base repository "owner/name"
    number: int
    head_sha: str
    base_sha: str
    state: str  # open | closed | merged

    @classmethod
    def from_github(cls, payload: Mapping, repo: str) -> PrHead:
        """Build from a GitHub ``pulls/{n}`` API payload."""
        try:
            return cls(
                repo=repo,
                number=int(payload["number"]),
                head_sha=str(payload["head"]["sha"]),
                base_sha=str(payload["base"]["sha"]),
                state="merged" if payload.get("merged") else str(payload["state"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PrValidationError(f"unexpected pull request payload: {exc!r}") from exc


def resolve_pr_head(head: PrHead, expected_head_sha: str | None) -> PrHead:
    """Check a freshly read PR head. ``expected_head_sha`` is what the maintainer dispatched."""
    if not _SHA.fullmatch(head.head_sha) or not _SHA.fullmatch(head.base_sha):
        raise PrValidationError("pull request head or base is not a full commit SHA")
    if head.state != "open":
        raise PrValidationError(f"pull request #{head.number} is {head.state}, not open")
    if expected_head_sha is not None:
        if not _SHA.fullmatch(expected_head_sha):
            raise PrValidationError("expected head must be a full commit SHA")
        if expected_head_sha != head.head_sha:
            raise StaleHeadError(f"PR head is {head.head_sha}, dispatched for {expected_head_sha}")
    return head


# --------------------------------------------------------------------------- report


class CaseOutcome(_Model):
    case_id: str
    status: Literal["pass", "fail"]
    reason: str | None = None


class Accuracy(_Model):
    """Advisory until a threshold is agreed; never part of the verdict."""

    mape_pct: float | None
    coverage: float | None
    n_scored: int
    n_total: int


class PrValidationReport(_Model):
    schema_version: Literal["aisim-pr-validation-report/v1"] = REPORT_SCHEMA_ID
    verdict: Literal["pass", "fail", "error"]
    # Bindings: a change to any of these invalidates the report.
    code_sha: str  # PR head SHA that was validated
    pr_number: int
    base_sha: str
    evaluator_revision: str  # trusted checkout SHA + verifier version
    dataset_repo: str
    dataset_sha: str | None  # None only when the candidate could not be resolved
    request_digest: str
    required_cases: list[CaseOutcome] = Field(default_factory=list)
    accuracy: Accuracy | None = None
    problems: list[str] = Field(default_factory=list)
    downloaded_bytes: int = 0

    def binding(self) -> str:
        """One digest over every bound identity, for display and comparison."""
        parts = [self.code_sha, self.evaluator_revision, self.dataset_sha or "", self.request_digest]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def evaluator_revision(trusted_sha: str) -> str:
    if not _SHA.fullmatch(trusted_sha):
        raise PrValidationError("trusted checkout SHA must be a full commit SHA")
    return f"{trusted_sha}+verifier-{VERIFIER_VERSION}"


def check_current(
    report: PrValidationReport,
    *,
    head_sha: str,
    request_digest: str,
    evaluator_rev: str,
    dataset_sha: str | None = None,
) -> list[str]:
    """Why ``report`` no longer applies; empty means it is still current."""
    stale: list[str] = []
    if report.code_sha != head_sha:
        stale.append(f"PR head changed: validated {report.code_sha}, now {head_sha}")
    if report.request_digest != request_digest:
        stale.append("validation request changed")
    if report.evaluator_revision != evaluator_rev:
        stale.append("evaluator revision changed")
    if dataset_sha is not None and report.dataset_sha != dataset_sha:
        stale.append("dataset commit changed")
    return stale


# --------------------------------------------------------------------------- run


def refuse_credentials(env: Mapping[str, str]) -> None:
    """Evaluation runs without credentials; refuse to start when any is present."""
    present = [name for name in _CREDENTIAL_ENV if env.get(name)]
    if present:
        raise PrValidationError(f"credentials must not be present during validation: {present}")


def _canonical_inputs(inputs: Mapping) -> str:
    return json.dumps(inputs, sort_keys=True, separators=(",", ":"))


def run_validation(
    request: ValidationRequest,
    pr: PrHead,
    *,
    trusted_sha: str,
    transport: Transport,
    predictor_factory: PredictorFactory,
    cache: Cache,
) -> PrValidationReport:
    """Fetch the selected evidence, verify the record, and report. Never raises for data defects."""
    rev = evaluator_revision(trusted_sha)
    base = {
        "code_sha": pr.head_sha,
        "pr_number": pr.number,
        "base_sha": pr.base_sha,
        "evaluator_revision": rev,
        "dataset_repo": request.dataset_repo,
        "request_digest": request.digest(),
    }

    def failed(verdict, problems, **extra) -> PrValidationReport:
        return PrValidationReport(
            verdict=verdict, dataset_sha=extra.pop("dataset_sha", None), problems=problems, **base, **extra
        )

    try:
        result = fetch(
            request.selection(),
            repo=request.dataset_repo,
            revision=request.dataset_revision,
            transport=transport,
            cache=cache,
            max_bytes=request.max_bytes,
        )
    except (DataFetchError, OSError) as exc:
        return failed("error", [f"fetch failed: {type(exc).__name__}: {exc}"])
    sha = result.plan.revision
    if sha != request.dataset_revision:
        return failed("error", ["dataset revision resolved to a different commit"], dataset_sha=sha)
    local = result.paths.get(request.record_path)
    if local is None:
        return failed("fail", [f"record {request.record_path!r} is not in the selected evidence"], dataset_sha=sha)
    try:
        record = load_record(local)
        predict, fingerprint = predictor_factory(record)
        table_art = next(a for a in record.truth.artifacts if a.artifact_id == record.truth.truth_table_artifact_id)
        seen: dict[str, float | None] = {}

        def recording(inputs):
            value = predict(inputs)
            seen[_canonical_inputs(inputs)] = value
            return value

        report = verify_record(record, result.root, recording, fingerprint)
        table = read_parquet(result.root / table_art.path, TRUTH_SCHEMA_ID) if report.recomputed else None
    except (PrValidationError, ValueError, KeyError, OSError, ValidationError, StopIteration) as exc:
        return failed("fail", [f"record could not be verified: {type(exc).__name__}: {exc}"], dataset_sha=sha)

    outcomes = _required_outcomes(request.required_cases, record.results, table, seen, report.ok)
    accuracy = None
    if report.recomputed is not None:
        r: Results = report.recomputed
        accuracy = Accuracy(mape_pct=r.mape_pct, coverage=r.coverage, n_scored=r.n_scored, n_total=r.n_total)
    ok = report.ok and all(o.status == "pass" for o in outcomes)
    return failed(
        "pass" if ok else "fail",
        list(report.problems),
        dataset_sha=sha,
        required_cases=outcomes,
        accuracy=accuracy,
        downloaded_bytes=result.downloaded_bytes,
    )


def _required_outcomes(required, claimed: Results, table, seen, verified: bool) -> list[CaseOutcome]:
    excluded = {e.case_id for e in claimed.exclusions}
    inputs_by_case: dict[str, str] = {}
    if table is not None:
        for case_id, inputs_json in zip(
            table.column("case_id").to_pylist(), table.column("inputs_json").to_pylist(), strict=True
        ):
            inputs_by_case[case_id] = _canonical_inputs(json.loads(inputs_json))
    out = []
    for case_id in required:
        if not verified:
            out.append(CaseOutcome(case_id=case_id, status="fail", reason="record did not verify"))
        elif case_id not in inputs_by_case:
            out.append(CaseOutcome(case_id=case_id, status="fail", reason="not in truth table"))
        elif case_id in excluded:
            out.append(CaseOutcome(case_id=case_id, status="fail", reason="excluded by producer"))
        elif seen.get(inputs_by_case[case_id]) is None:
            out.append(CaseOutcome(case_id=case_id, status="fail", reason="no prediction"))
        else:
            out.append(CaseOutcome(case_id=case_id, status="pass"))
    return out


def render_summary(report: PrValidationReport) -> str:
    """Markdown for the workflow step summary. Contains only public identifiers."""
    lines = [
        f"## PR validation: {report.verdict.upper()}",
        "",
        f"- PR: #{report.pr_number} at `{report.code_sha}`",
        f"- Evaluator: `{report.evaluator_revision}`",
        f"- Dataset: `{report.dataset_repo}` at `{report.dataset_sha}`",
        f"- Request digest: `{report.request_digest}`",
        f"- Binding: `{report.binding()}`",
    ]
    if report.accuracy:
        a = report.accuracy
        lines.append(
            f"- Accuracy (advisory): MAPE {a.mape_pct}, coverage {a.coverage}, {a.n_scored}/{a.n_total} scored"
        )
    for o in report.required_cases:
        lines.append(f"- Required `{o.case_id}`: {o.status}" + (f" ({o.reason})" if o.reason else ""))
    lines += [f"- Problem: {p}" for p in report.problems]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def no_predictor(record: EvalRecord) -> tuple[Predict, str]:
    """Fail-closed default: no trusted predictor is registered for this record yet."""
    raise PrValidationError("no trusted predictor factory is registered for this evaluation layer")


def _load_factory(spec: str) -> PredictorFactory:
    module, _, attr = spec.partition(":")
    if not module.startswith(_PREDICTOR_PREFIXES) or not attr:
        raise PrValidationError(f"predictor factory must be 'aisim[_core].module:callable', got {spec!r}")
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise PrValidationError(f"cannot load predictor factory {spec!r}: {exc}") from exc


def main(argv: list[str] | None = None, *, transport: Transport | None = None) -> int:
    """``python -m aisim_core.data.validate {run,recheck}``. Exit 0 pass, 1 fail/error/stale, 2 bad input."""
    p = argparse.ArgumentParser(prog="aisim_core.data.validate")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "recheck"):
        s = sub.add_parser(name)
        s.add_argument("--pr-json", required=True, help="GitHub pulls/{n} payload, read as data")
        s.add_argument("--repo", required=True, help="GitHub repository owner/name")
        s.add_argument("--request-file", required=True, help="request JSON read from the PR head, as data")
        s.add_argument("--trusted-sha", required=True)
    run = sub.choices["run"]
    run.add_argument("--expected-head", default=None)
    run.add_argument("--predictor-factory", required=True)
    run.add_argument("--cache-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--summary", default=None)
    sub.choices["recheck"].add_argument("--report", required=True)
    args = p.parse_args(argv)
    try:
        refuse_credentials(os.environ)
        request = parse_request(Path(args.request_file).read_bytes())
        pr = PrHead.from_github(json.loads(Path(args.pr_json).read_text()), args.repo)
        if args.cmd == "run":
            pr = resolve_pr_head(pr, args.expected_head)
            report = run_validation(
                request,
                pr,
                trusted_sha=args.trusted_sha,
                transport=transport or HfTransport(token=None),
                predictor_factory=_load_factory(args.predictor_factory),
                cache=Cache(Path(args.cache_dir)),
            )
            Path(args.output).write_text(report.model_dump_json(indent=2) + "\n")
            summary = render_summary(report)
            if args.summary:
                with open(args.summary, "a", encoding="utf-8") as handle:
                    handle.write(summary)
            print(summary)
            return 0 if report.verdict == "pass" else 1
        report = PrValidationReport.model_validate_json(Path(args.report).read_text())
        stale = check_current(
            report,
            head_sha=pr.head_sha,
            request_digest=request.digest(),
            evaluator_rev=evaluator_revision(args.trusted_sha),
        )
        for reason in stale:
            print(f"STALE: {reason}", file=sys.stderr)
        return 1 if stale else 0
    except (PrValidationError, ValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
