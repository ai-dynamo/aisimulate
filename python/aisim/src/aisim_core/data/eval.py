# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim-eval/v1`` evaluation records, prediction fingerprint and verifier.

An eval record stores a prediction's accuracy against recorded truth. It is
immutable and content-addressed: the record ID is the SHA-256 of its canonical
JSON, the file is named by that ID, and a newer record *supersedes* an older one
by naming its ID; nothing is overwritten. Prediction is deterministic and needs
no GPU, so :func:`verify_record` recomputes the prediction side from the
recorded truth table and checks the claimed numbers. Only ``verified`` and
``qualified`` records should feed defaults and gates.

The verifier takes a ``predict`` callable so this module never constructs a
performance model; callers build one through the canonical
``best_available`` interface and pass it in. Offline only.
"""

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from aisim_core.data import Artifact, Digest, Identity
from aisim_core.data.gpu_time import Finding, GpuTime, check_gpu_time, gpu_time_ok
from aisim_core.data.parquet import read_parquet, write_parquet

EVAL_SCHEMA_ID = "aisim-eval/v1"
TRUTH_SCHEMA_ID = "aisim-eval-truth/v1"
TRUTH_COLUMNS = ("case_id", "inputs_json", "truth_value")
FINGERPRINT_VERSION = 1
VERIFIER_VERSION = "1"
QUANTILES = (0.5, 0.9, 0.99)
WORST_N = 10
# Metrics are rounded before comparison so platform float noise never decides.
ROUND_DIGITS = 9
REL_TOL = 1e-6

Layer = Literal["op", "fpm", "e2e"]
Tier = Literal["submitted", "verified", "qualified"]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


# --- prediction identity ----------------------------------------------------


class PredictionIdentity(_Record):
    """What produced the predictions. ``fingerprint`` is derived, never trusted."""

    code_path: str = Field(min_length=1)  # e.g. "fpm_interpolation"; never a PR head SHA
    code_version: str | None = None  # release or tag of the perf-model code path
    data_pins: dict[str, Digest]  # input name -> SHA-256 (profiles, tables)
    spec_sha256: Digest
    fingerprint: Digest

    @classmethod
    def build(
        cls, code_path: str, data_pins: Mapping[str, str], spec_sha256: str, code_version: str | None = None
    ) -> "PredictionIdentity":
        fp = prediction_fingerprint(code_path, data_pins, spec_sha256, code_version)
        return cls(
            code_path=code_path,
            code_version=code_version,
            data_pins=dict(sorted(data_pins.items())),
            spec_sha256=spec_sha256,
            fingerprint=fp,
        )

    @model_validator(mode="after")
    def fingerprint_matches(self):
        expected = prediction_fingerprint(self.code_path, self.data_pins, self.spec_sha256, self.code_version)
        if self.fingerprint != expected:
            raise ValueError("prediction fingerprint does not match its components")
        return self


def prediction_fingerprint(
    code_path: str, data_pins: Mapping[str, str], spec_sha256: str, code_version: str | None = None
) -> str:
    """SHA-256 over code path, data pins and system spec hash (not a commit SHA).

    Unrelated commits leave it unchanged, so recorded evidence stays valid and
    an agent can skip a comparison that already exists.
    """
    payload = {
        "version": FINGERPRINT_VERSION,
        "code_path": code_path,
        "code_version": code_version,
        "data_pins": dict(sorted(data_pins.items())),
        "spec_sha256": spec_sha256,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


# --- truth identity ---------------------------------------------------------


class TruthIdentity(_Record):
    layer: Layer
    snapshot_id: str | None = None
    configuration_ids: list[str] = Field(min_length=1)
    identity: Identity
    artifacts: list[Artifact] = Field(min_length=1)  # truth inputs, incl. the truth table
    truth_table_artifact_id: str  # which artifact is the Parquet truth table

    @model_validator(mode="after")
    def table_listed(self):
        if self.truth_table_artifact_id not in {a.artifact_id for a in self.artifacts}:
            raise ValueError("truth_table_artifact_id is not among artifacts")
        return self


# --- results, assumptions, cost ----------------------------------------------


class Outlier(_Record):
    case_id: str
    truth: float
    predicted: float
    rel_error_pct: float  # signed (predicted - truth) / truth * 100


class Exclusion(_Record):
    case_id: str
    reason: str = Field(min_length=1)


class Results(_Record):
    n_total: int = Field(ge=0)
    n_scored: int = Field(ge=0)
    mape_pct: float | None  # null when nothing was scored
    residual_quantiles_pct: dict[str, float] | None  # of signed relative error
    worst_n: list[Outlier]
    coverage: float | None  # n_scored / n_total
    exclusions: list[Exclusion]
    # Kept apart on purpose; unknown stays null.
    collection_quality: dict[str, JsonValue] | None = None
    resolved_memory: dict[str, JsonValue] | None = None
    replay_coverage: dict[str, JsonValue] | None = None


class Assumption(_Record):
    name: str = Field(min_length=1)
    value: JsonValue
    status: Literal["verified", "estimated"]
    kind: Literal["truth", "calibration"]


class SimulationCost(_Record):
    wall_seconds: float | None = Field(default=None, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)  # prediction needs no GPU


class Producer(_Record):
    kind: Literal["agent", "human", "ci"]
    name: str = Field(min_length=1)
    skill: str | None = None
    skill_version: str | None = None
    evaluator_revision: str = Field(min_length=1)


class Verification(_Record):
    verifier_version: str
    recomputed_fingerprint: Digest
    verified_record_id: Digest  # the submitted record this attests


class EvalRecord(_Record):
    schema_version: Literal["aisim-eval/v1"]
    prediction: PredictionIdentity
    truth: TruthIdentity
    results: Results
    assumptions: list[Assumption]
    gpu_time: GpuTime | None = None  # absent is missing, not zero; see check_record_gpu_time
    simulation_cost: SimulationCost
    producer: Producer
    tier: Tier
    status: Literal["complete", "partial", "failed"] = "complete"
    status_reason: str | None = None  # required for partial/failed; kept, never dropped
    supersedes: Digest | None = None
    verification: Verification | None = None
    created_at: str  # RFC 3339 UTC, supplied by the caller for reproducible tests
    findings: list[Artifact] = Field(default_factory=list)  # linked artifacts

    @model_validator(mode="after")
    def consistent(self):
        if self.status != "complete" and not self.status_reason:
            raise ValueError("partial and failed records need status_reason")
        if self.tier != "submitted" and self.verification is None:
            raise ValueError(f"tier {self.tier!r} requires a verification block")
        if self.verification and self.verification.recomputed_fingerprint != self.prediction.fingerprint:
            raise ValueError("verification fingerprint differs from the record's prediction fingerprint")
        errors = [f.message for f in check_record_gpu_time(self) if f.severity == "error"]
        if errors:
            raise ValueError(f"qualified record needs gpu_time: {errors[0]}")
        return self

    @property
    def record_id(self) -> str:
        return record_id(self)


def check_record_gpu_time(record: EvalRecord) -> list[Finding]:
    """``gpu_time`` findings: errors for qualified records, warnings otherwise (archival)."""
    return check_gpu_time(record.gpu_time, "qualification" if record.tier == "qualified" else "archival")


# --- content addressing and storage -----------------------------------------


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def record_id(record: EvalRecord) -> str:
    """SHA-256 of the canonical JSON of the whole record."""
    return hashlib.sha256(_canonical(record.model_dump(by_alias=True, mode="json")).encode()).hexdigest()


def _label(value: str) -> str:
    return value.replace("/", "--")


def parallelism_label(identity: Identity) -> str:
    """Readable path label, e.g. ``tp4-ep8``; ``unspecified`` when unknown."""
    if not identity.parallelism:
        return "unspecified"
    roles = identity.parallelism
    if len(roles) == 1:
        dims = next(iter(roles.values()))
        return "-".join(f"{k}{v}" for k, v in dims.items()) or "tp1"
    return "_".join(f"{r}-" + ("-".join(f"{k}{v}" for k, v in d.items()) or "tp1") for r, d in roles.items())


def record_dir(root: str | Path, identity: Identity) -> Path:
    """``eval/<model>/<hardware>/<framework>/<version>/<parallelism>/`` (op: ``<hardware>/<family>/``)."""
    base = Path(root) / "eval"
    if identity.model is None:
        return base / identity.hardware / (identity.operation_family or "unspecified")
    return (
        base
        / _label(identity.model)
        / identity.hardware
        / identity.framework
        / identity.framework_version
        / parallelism_label(identity)
    )


def record_path(root: str | Path, record: EvalRecord) -> Path:
    return record_dir(root, record.truth.identity) / f"{record.record_id}.json"


def write_record(root: str | Path, record: EvalRecord) -> Path:
    """Write a record by content address. Never overwrites different content.

    A superseding record must name an existing record in the same directory.
    Writing identical content again is a no-op.
    """
    path = record_path(root, record)
    if record.supersedes is not None and not (path.parent / f"{record.supersedes}.json").is_file():
        raise ValueError(f"superseded record {record.supersedes} not found next to the new record")
    text = json.dumps(record.model_dump(by_alias=True, mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if load_record(path) != record:
            raise FileExistsError(f"refusing to overwrite {path} with different content")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.link(tmp, path)  # fails if another writer created it first
    except FileExistsError:
        if load_record(path) != record:
            raise
    finally:
        Path(tmp).unlink(missing_ok=True)
    return path


def load_record(path: str | Path) -> EvalRecord:
    """Load a record and check its file name equals its content hash."""
    path = Path(path)
    record = EvalRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if path.stem != record.record_id:
        raise ValueError(f"{path.name}: file name does not match record content hash {record.record_id}")
    return record


def superseded_ids(directory: str | Path) -> set[str]:
    return {r.supersedes for r in _load_all(directory) if r.supersedes}


def current_records(directory: str | Path) -> list[EvalRecord]:
    """Records no other record supersedes (the current view)."""
    records = _load_all(directory)
    gone = {r.supersedes for r in records if r.supersedes}
    return [r for r in records if r.record_id not in gone]


def _load_all(directory: str | Path) -> list[EvalRecord]:
    return [load_record(p) for p in sorted(Path(directory).glob("*.json"))]


# --- truth table and metrics ------------------------------------------------


def write_truth_table(path: str | Path, rows: list[tuple[str, Mapping[str, JsonValue], float | None]]) -> Artifact:
    """Write the Parquet truth table and return its Artifact (path = file name)."""
    table = pa.table(
        {
            "case_id": pa.array([r[0] for r in rows], pa.string()),
            "inputs_json": pa.array([_canonical(dict(r[1])) for r in rows], pa.string()),
            "truth_value": pa.array([r[2] for r in rows], pa.float64()),
        }
    )
    path = write_parquet(table, path, TRUTH_SCHEMA_ID)
    data = path.read_bytes()
    return Artifact(
        artifact_id="truth-table",
        path=path.name,
        role="truth-table",
        format="parquet",
        schema=TRUTH_SCHEMA_ID,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        rows=len(rows),
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def compute_results(
    cases: list[tuple[str, float | None, float | None]],
    exclusions: list[Exclusion] | None = None,
    **separate_fields: dict[str, JsonValue] | None,
) -> Results:
    """Score ``(case_id, truth, predicted)`` rows. Rows lacking either value are excluded.

    Pure and deterministic; used by both the producer and the verifier.
    """
    excluded = {e.case_id: e for e in exclusions or []}
    errors: list[tuple[str, float, float, float]] = []
    for case_id, truth, predicted in cases:
        if case_id in excluded:
            continue
        if truth is None or predicted is None or not math.isfinite(predicted) or not truth > 0:
            reason = "missing truth" if truth is None else "missing prediction"
            if truth is not None and predicted is not None and math.isfinite(predicted):
                reason = "non-positive truth"
            excluded[case_id] = Exclusion(case_id=case_id, reason=reason)
            continue
        errors.append((case_id, truth, predicted, (predicted - truth) / truth * 100.0))
    n_total = len(cases)
    n_scored = len(errors)
    if errors:
        mape = round(sum(abs(e[3]) for e in errors) / n_scored, ROUND_DIGITS)
        signed = sorted(e[3] for e in errors)
        quantiles = {f"p{round(q * 100)}": round(_quantile(signed, q), ROUND_DIGITS) for q in QUANTILES}
    else:
        mape, quantiles = None, None
    worst = sorted(errors, key=lambda e: (-abs(e[3]), e[0]))[:WORST_N]
    return Results(
        n_total=n_total,
        n_scored=n_scored,
        mape_pct=mape,
        residual_quantiles_pct=quantiles,
        worst_n=[
            Outlier(case_id=c, truth=t, predicted=p, rel_error_pct=round(r, ROUND_DIGITS)) for c, t, p, r in worst
        ],
        coverage=round(n_scored / n_total, ROUND_DIGITS) if n_total else None,
        exclusions=sorted(excluded.values(), key=lambda e: e.case_id),
        **separate_fields,
    )


# --- verifier ---------------------------------------------------------------


class VerificationReport(BaseModel):
    ok: bool
    problems: list[str]
    recomputed: Results | None = None


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-6)


def verify_record(
    record: EvalRecord,
    truth_root: str | Path,
    predict: Callable[[Mapping[str, JsonValue]], float | None],
    current_fingerprint: str,
) -> VerificationReport:
    """Recompute the prediction side from recorded truth and compare claimed numbers.

    ``predict`` maps one truth row's inputs to a predicted value (None when the
    model cannot predict). ``current_fingerprint`` is the fingerprint of the
    predictor the caller built; recomputation is only meaningful when it equals
    the record's. GPU-free and deterministic. ``truth_root`` is the directory
    that holds the truth artifacts by their ``path``.
    """
    problems: list[str] = []
    if current_fingerprint != record.prediction.fingerprint:
        return VerificationReport(ok=False, problems=["predictor fingerprint differs from the record's"])
    table_art = next(a for a in record.truth.artifacts if a.artifact_id == record.truth.truth_table_artifact_id)
    path = Path(truth_root) / table_art.path
    if not path.is_file():
        return VerificationReport(ok=False, problems=[f"truth table missing: {table_art.path}"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != table_art.sha256 or len(data) != table_art.bytes:
        return VerificationReport(ok=False, problems=["truth table hash or size differs from record"])
    table = read_parquet(path, TRUTH_SCHEMA_ID)
    if tuple(table.column_names) != TRUTH_COLUMNS:
        return VerificationReport(ok=False, problems=[f"truth table columns {table.column_names}"])
    ids = table.column("case_id").to_pylist()
    if len(set(ids)) != len(ids):
        return VerificationReport(ok=False, problems=["duplicate case_id in truth table"])
    truths = table.column("truth_value").to_pylist()
    cases = []
    for case_id, inputs_json, truth in zip(ids, table.column("inputs_json").to_pylist(), truths, strict=True):
        cases.append((case_id, truth, predict(json.loads(inputs_json))))
    # Exclusions recorded by the producer are honored only if they name real rows.
    unknown = [e.case_id for e in record.results.exclusions if e.case_id not in set(ids)]
    if unknown:
        problems.append(f"exclusions name unknown cases: {unknown[:3]}")
    recorded_excl = [e for e in record.results.exclusions if e.case_id in set(ids)]
    got = compute_results(cases, recorded_excl)
    claimed = record.results
    for name in ("n_total", "n_scored"):
        if getattr(got, name) != getattr(claimed, name):
            problems.append(f"{name}: claimed {getattr(claimed, name)}, recomputed {getattr(got, name)}")
    for name in ("mape_pct", "coverage"):
        if not _close(getattr(got, name), getattr(claimed, name)):
            problems.append(f"{name}: claimed {getattr(claimed, name)}, recomputed {getattr(got, name)}")
    gq, cq = got.residual_quantiles_pct, claimed.residual_quantiles_pct
    if (gq is None) != (cq is None) or (gq and (gq.keys() != cq.keys() or not all(_close(gq[k], cq[k]) for k in gq))):
        problems.append(f"residual_quantiles_pct: claimed {cq}, recomputed {gq}")
    if [o.case_id for o in got.worst_n] != [o.case_id for o in claimed.worst_n]:
        problems.append("worst_n case ordering differs from recomputation")
    elif not all(
        _close(a.predicted, b.predicted) and _close(a.rel_error_pct, b.rel_error_pct)
        for a, b in zip(got.worst_n, claimed.worst_n, strict=True)
    ):
        problems.append("worst_n values differ from recomputation")
    return VerificationReport(ok=not problems, problems=problems, recomputed=got)


def attest(record: EvalRecord, report: VerificationReport, tier: Tier = "verified") -> EvalRecord:
    """Build the superseding record that promotes a successfully verified one.

    The original is untouched; the new record has its own content hash.
    """
    if not report.ok:
        raise ValueError(f"cannot promote a record that failed verification: {report.problems}")
    if tier == "submitted":
        raise ValueError("attest promotes to verified or qualified")
    if record.status != "complete":
        raise ValueError("only complete records can be promoted")
    doc = record.model_dump(by_alias=True)
    doc.update(
        tier=tier,
        supersedes=record.record_id,
        verification={
            "verifier_version": VERIFIER_VERSION,
            "recomputed_fingerprint": record.prediction.fingerprint,
            "verified_record_id": record.record_id,
        },
    )
    return EvalRecord.model_validate(doc)


__all__ = [
    "EVAL_SCHEMA_ID",
    "EvalRecord",
    "GpuTime",
    "attest",
    "check_record_gpu_time",
    "compute_results",
    "current_records",
    "gpu_time_ok",
    "load_record",
    "prediction_fingerprint",
    "record_id",
    "verify_record",
    "write_record",
    "write_truth_table",
]
