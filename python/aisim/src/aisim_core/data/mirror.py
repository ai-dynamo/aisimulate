# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One dataset-tree validator for the public dataset and its private mirror (DEP #435, Phase 7).

``ai-dynamo/aisim-data-internal`` has the same layout and schemas as the public
dataset, so its CI runs exactly this function, from a pinned checkout of this
repository. There is no separate "internal" rule set: a tree the public CI would
reject fails the mirror CI for the same reason.

Two profiles exist and differ only in what the *public* repository additionally
forbids:

* ``mirror``  structure, identity, integrity, schema and catalog checks. Sensitive
  content (unredacted logs, run URLs, image digests) is allowed.
* ``public``  everything in ``mirror`` plus the publication rules in
  :mod:`aisim_core.data.redaction`, and no ``mirror/`` control directory.

``aisim-infx-cache`` is a separate private cache that is never promoted
(:data:`NEVER_PROMOTED`). It uses the same E2E layout, so it can be validated with
``repo=INFX_CACHE_REPO``.

Nothing here touches the network. The report carries a validator fingerprint (the
hash of every generated schema snapshot plus :data:`VALIDATOR_VERSION`), so a CI
log shows which rule set judged the tree.

CLI: ``python -m aisim_core.data.mirror validate --root DIR [--repo ID]
[--profile mirror|public]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from aisim_core.data import CategoryIndex, DatasetIndex, Manifest, _verify, check_manifest, configuration_id
from aisim_core.data.catalog import CATALOG_DIR, _write, build_catalog, catalog_files, known_system_specs
from aisim_core.data.e2e import load_e2e, read_snapshot
from aisim_core.data.eval import load_record, record_dir
from aisim_core.data.hygiene import (
    CandidateFile,
    CandidatePolicy,
    Finding,
    check_content,
    check_listing,
)
from aisim_core.data.loaders import LoaderError
from aisim_core.data.schemas import check_snapshots, render_all

VALIDATOR_VERSION = "1"
INTERNAL_REPO = "ai-dynamo/aisim-data-internal"
INFX_CACHE_REPO = "ai-dynamo/aisim-infx-cache"
# Never a promotion source: private caches whose terms forbid public export.
NEVER_PROMOTED = frozenset({INFX_CACHE_REPO})
MIRROR_CONTROL_DIR = "mirror"  # private-only control files; absent from the public layout
DATA_ROOTS = ("catalog/", "eval/", "data/")
Profile = Literal["mirror", "public"]


def validator_fingerprint() -> str:
    """Hash of the rule set: validator version plus every generated schema snapshot."""
    digest = hashlib.sha256(VALIDATOR_VERSION.encode())
    for name, text in sorted(render_all().items()):
        digest.update(name.encode())
        digest.update(text.encode())
    return digest.hexdigest()


@dataclass
class TreeReport:
    repo: str | None
    profile: str
    fingerprint: str = field(default_factory=validator_fingerprint)
    files: int = 0
    configurations: int = 0
    eval_records: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, code: str, message: str, path: str | None = None, severity: str = "error") -> None:
        self.findings.append(Finding(code, message, path, severity))  # type: ignore[arg-type]

    def to_json(self) -> dict:
        return {
            "repo": self.repo,
            "profile": self.profile,
            "validator_fingerprint": self.fingerprint,
            "ok": self.ok,
            "files": self.files,
            "configurations": self.configurations,
            "eval_records": self.eval_records,
            "findings": [vars(f) for f in self.findings],
        }


def _tree_policy(extra_roots: tuple[str, ...]) -> CandidatePolicy:
    # Whole-tree limits are the host's business; candidate size limits stay in hygiene.
    return CandidatePolicy(
        max_files=10**9,
        max_total_bytes=1 << 62,
        max_file_bytes=1 << 40,
        allowed_roots=DATA_ROOTS + extra_roots,
        allowed_suffixes=frozenset({".parquet", ".json", ".gz"}),
    )


def list_tree(root: Path) -> list[CandidateFile]:
    """Every file under ``root`` (symlinks are listed, never followed), skipping ``.git``."""
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel == ".git" or rel.startswith(".git/") or (path.is_dir() and not path.is_symlink()):
            continue
        out.append(CandidateFile(rel, 0 if path.is_symlink() else path.stat().st_size, is_symlink=path.is_symlink()))
    return out


def _read_json(root: Path, rel: str, model, report: TreeReport):
    try:
        return model.model_validate_json((root / rel).read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        report.add("document-invalid", f"{model.__name__}: {str(exc).splitlines()[0]}", rel)
        return None


def _check_manifest_any(root: Path, rel: str, category: str):
    """Category-specific loader: the same code the evaluators use."""
    if category == "e2e":
        return load_e2e(root, rel).manifest
    if category in ("op-latency", "fpm"):
        return check_manifest(root, rel)
    manifest = Manifest.model_validate_json((root / rel).read_bytes())
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        raise ValueError("configuration identity hash mismatch")
    for artifact in manifest.artifacts:
        _verify(root, artifact)
    return manifest


def _check_indexes(root: Path, report: TreeReport, declared: set[str]) -> dict[str, Manifest]:
    index_rel = f"{CATALOG_DIR}/dataset.json"
    if not (root / index_rel).is_file():
        report.add("dataset-index-missing", "catalog/dataset.json is missing", index_rel)
        return {}
    dataset = _read_json(root, index_rel, DatasetIndex, report)
    if dataset is None:
        return {}
    if report.repo and dataset.dataset_id != report.repo:
        report.add("dataset-id-mismatch", f"dataset_id {dataset.dataset_id!r} is not {report.repo!r}", index_rel)
    declared.add(index_rel)
    manifests: dict[str, Manifest] = {}
    for category, ref in dataset.categories.items():
        try:
            _verify(root, ref.index)
        except ValueError as exc:
            report.add("index-integrity", str(exc), ref.index.path)
            continue
        declared.add(ref.index.path)
        entries = _read_json(root, ref.index.path, CategoryIndex, report)
        if entries is None:
            continue
        if entries.category != category:
            report.add("index-category", "category index identity mismatch", ref.index.path)
        seen: set[str] = set()
        for entry in entries.entries:
            path = entry.manifest.path
            if entry.configuration_id in seen:
                report.add("duplicate-configuration", entry.configuration_id, path)
            seen.add(entry.configuration_id)
            try:
                _verify(root, entry.manifest)
                manifest = _check_manifest_any(root, path, category)
                if (manifest.category, manifest.configuration_id, manifest.identity, manifest.purpose) != (
                    category,
                    entry.configuration_id,
                    entry.identity,
                    entry.purpose,
                ):
                    raise ValueError("catalog and manifest identity mismatch")
            except (ValueError, LoaderError, ValidationError, OSError) as exc:
                report.add("manifest-invalid", str(exc).splitlines()[0], path)
                continue
            manifests[path] = manifest
            report.configurations += 1
            declared.add(path)
            declared.update(a.path for a in manifest.artifacts)
    return manifests


def _check_eval(root: Path, report: TreeReport, declared: set[str]) -> None:
    eval_root = root / "eval"
    if not eval_root.is_dir():
        return
    for path in sorted(eval_root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            record = load_record(path)
            expected = record_dir(root, record.truth.identity)
            if path.parent != expected:
                raise ValueError(f"record is not at its identity path {expected.relative_to(root).as_posix()}")
            for artifact in record.truth.artifacts:
                _verify(path.parent, artifact)
                declared.add((path.parent / artifact.path).relative_to(root).as_posix())
            for artifact in record.findings:
                _verify(path.parent, artifact)
                declared.add((path.parent / artifact.path).relative_to(root).as_posix())
        except (ValueError, ValidationError, OSError) as exc:
            report.add("eval-record-invalid", str(exc).splitlines()[0], rel)
            continue
        declared.add(rel)
        report.eval_records += 1


def _check_snapshots(root: Path, report: TreeReport, declared: set[str]) -> None:
    for path in sorted((root / "data").rglob("*snapshot*.json")) if (root / "data").is_dir() else []:
        rel = path.relative_to(root).as_posix()
        try:
            snapshot, _ = read_snapshot(root, rel)
        except (ValueError, ValidationError, OSError) as exc:
            report.add("snapshot-invalid", str(exc).splitlines()[0], rel)
            continue
        declared.update({rel, snapshot.members.path} | ({snapshot.acquisition.path} if snapshot.acquisition else set()))


def _check_schema_snapshots(root: Path, report: TreeReport, declared: set[str]) -> None:
    directory = root / CATALOG_DIR / "schemas"
    if not directory.is_dir():
        return
    for problem in check_snapshots(directory):
        report.add("schema-drift", problem, f"{CATALOG_DIR}/schemas")
    declared.update(p.relative_to(root).as_posix() for p in directory.glob("*.json"))


def _check_catalog_drift(root: Path, report: TreeReport, declared: set[str]) -> None:
    existing = catalog_files(root)
    declared.update(p.relative_to(root).as_posix() for p in existing)
    if not existing:
        report.add("catalog-missing", "no catalog shards; regenerate with build_catalog", f"{CATALOG_DIR}/")
        return
    try:
        rows = build_catalog(root, write=False)
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp)
            _write(fresh, rows, known_system_specs())
            want = {p.relative_to(fresh).as_posix(): p.read_bytes() for p in catalog_files(fresh)}
    except (ValueError, ValidationError, OSError) as exc:
        report.add("catalog-unbuildable", str(exc).splitlines()[0], f"{CATALOG_DIR}/")
        return
    have = {p.relative_to(root).as_posix(): p.read_bytes() for p in existing}
    for name in sorted(set(want) | set(have)):
        if name not in have:
            report.add("catalog-drift", "catalog shard is missing; regenerate the catalog", name)
        elif name not in want:
            report.add("catalog-drift", "stale catalog shard; regenerate the catalog", name)
        elif want[name] != have[name]:
            report.add("catalog-drift", "catalog shard differs from the data; regenerate the catalog", name)


def validate_tree(
    root: str | Path,
    *,
    repo: str | None = None,
    profile: Profile = "mirror",
    allow_empty: bool = False,
) -> TreeReport:
    """Validate a dataset checkout. ``repo`` (when given) must equal ``dataset.json``'s ``dataset_id``.

    ``allow_empty`` accepts a tree with no files at all, for a freshly created mirror.
    """
    root = Path(root)
    report = TreeReport(repo=repo, profile=profile)
    if not root.is_dir():
        report.add("root-missing", f"{root} is not a directory")
        return report
    listing = [f for f in list_tree(root) if not (profile == "mirror" and f.path.startswith(f"{MIRROR_CONTROL_DIR}/"))]
    if profile == "mirror":
        control = [f for f in list_tree(root) if f.path.startswith(f"{MIRROR_CONTROL_DIR}/")]
        for f in control:
            if f.is_symlink or not f.path.endswith(".json"):
                report.add("control-file", "mirror/ may hold plain .json files only", f.path)
    report.files = len(listing)
    if not listing and allow_empty:
        return report

    declared: set[str] = set()
    manifests = _check_indexes(root, report, declared)
    _check_eval(root, report, declared)
    _check_snapshots(root, report, declared)
    _check_schema_snapshots(root, report, declared)
    _check_catalog_drift(root, report, declared)

    gz_declared = {p for p in declared if p.endswith(".gz")}
    for finding in check_listing(_tree_policy(()), listing):
        report.findings.append(finding)
    skip = report_paths_with_errors(report, "path-", "symlink")
    for f in listing:
        if f.is_symlink or f.path in skip:
            continue
        data = (root / f.path).read_bytes() if f.size <= 4 * 1024 * 1024 else None
        head = data if data is not None else (root / f.path).open("rb").read(16)
        tail = data[-4:] if data is not None else b""
        if data is None and f.path.endswith(".parquet"):
            with (root / f.path).open("rb") as handle:
                handle.seek(-4, 2)
                tail = handle.read(4)
        if f.path.endswith(".gz"):
            continue  # declared provenance artifact; archive magic is expected
        report.findings.extend(check_content(_tree_policy(()), f, head, tail))
    for f in listing:
        if f.path.endswith(".gz") and f.path not in gz_declared:
            report.add("undeclared-archive", "gzip file is not a declared provenance artifact", f.path)
        elif f.path not in declared and not f.path.endswith(".gz"):
            report.add("orphan-file", "file is not referenced by any manifest, index or record", f.path, "warning")

    if profile == "public":
        from aisim_core.data.redaction import check_publishable

        for f in listing:
            if f.path.startswith(f"{MIRROR_CONTROL_DIR}/"):
                report.add("control-file", "mirror/ control files must not exist in the public dataset", f.path)
        report.findings.extend(check_publishable(root, [f.path for f in listing], manifests=manifests))
    return report


def report_paths_with_errors(report: TreeReport, *prefixes: str) -> set[str]:
    return {f.path for f in report.errors if f.path and f.code.startswith(prefixes)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="validate a dataset checkout")
    val.add_argument("--root", type=Path, required=True)
    val.add_argument("--repo", default=INTERNAL_REPO, help="expected dataset_id (default: the internal mirror)")
    val.add_argument("--profile", choices=("mirror", "public"), default="mirror")
    val.add_argument("--allow-empty", action="store_true", help="accept a tree with no files (new mirror)")
    val.add_argument("--output", type=Path, help="write the JSON report here")
    sub.add_parser("fingerprint", help="print the validator fingerprint")
    args = parser.parse_args(argv)
    if args.command == "fingerprint":
        print(validator_fingerprint())
        return 0
    report = validate_tree(args.root, repo=args.repo, profile=args.profile, allow_empty=args.allow_empty)
    if args.output:
        args.output.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n")
    print(f"validator {report.fingerprint[:12]} profile={report.profile} repo={report.repo}")
    for f in report.findings:
        print(f"{f.severity.upper()} {f.code} {f.path or ''}: {f.message}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
