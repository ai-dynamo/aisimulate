# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed check of the collected perf-database tree and the collector notes.

Full CI runs it in the `collector-data` job when a change touches the
collector, the perf-database tools or the data tree, and
`tests/unit/tools/test_check_collector_data.py` runs it against the committed
tree with the unit tests. Exits non-zero on ANY failure and prints every
failure it finds — never stops at the first one.

Eight rules:

- **R1 sidecar coverage**: every family version dir
  (`<system>/<family>/<backend>/<version>/`) holding >=1 parquet table must
  have a `collection_meta.yaml` whose `tables` key set covers every parquet
  stem present. The legacy tier (`provenance: legacy`, migrated pre-V3 data)
  satisfies coverage with status-only entries — the check is key presence,
  not field completeness, so a legacy entry with only `status` is enough.
- **R2 reuse validity**: every `reuse.yaml`
  must parse via the real loader parser
  (`aisimulate_core.sdk.perf_database._parse_reuse_yaml`); each entry's
  donor (same family, same backend, `from_version`, `table`) must exist as
  real parquet data, not merely another declared-reuse dir. "Same backend
  only" is a structural guarantee here: a `reuse.yaml` entry carries no
  backend field, so its donor is always resolved under the same
  `<family>/<backend>/` subtree it lives in. `from_version` may be newer or
  older than the declaring dir's own version — that asymmetry is exactly what
  a declaration is for; this check does not re-derive direction.
- **R3 comm declared-reuse exclusion**: no `reuse.yaml`
  may exist under `comm`; validated framework namespaces reuse earlier
  versions implicitly, while NCCL/oneCCL and unknown backends are primary-only.
- **R4 family placement** (catalog-driven): every parquet table's
  stem must map, via the op catalog (`collector/op_backend_catalog.yaml`), to
  the family directory it is actually filed under.
- **R5 identity**: `collector.framework_manifest.validate_resolution()`
  must return no errors — every registry op resolves to exactly one pinned
  runtime.
- **R6 no legacy markers**: zero `SHARED_LAYER_REUSE.txt` / `INCOMPLETE.txt`
  files remain anywhere under the data root (both are fully superseded by
  `reuse.yaml` / `collection_meta.yaml`).
- **R7 attested collection events**: every schema-v2 collection event records
  a valid status and a non-empty attempted-case plan; schema-v1 tables retain
  the original single-event empty-plan guard.
- **R8 notes**: every `collector/notes/<backend>.yaml` matches the closed note
  schema (unique `D-`/`G-` ids, catalog tables, known systems, a `direction`
  for a deviation and a `reason` for a guard), and no note that covers the
  pinned version has a `verified_on` behind the pin of a table it covers. R8
  reads the notes, the catalog and the manifest, not the data tree under
  check, so it runs even when the data root is missing.

Reuses (imports, does not duplicate): `collector.op_catalog.load_family_map`,
`collector.framework_manifest.validate_resolution`, and the loader's real
parsers `aisimulate_core.sdk.perf_database._parse_reuse_yaml` /
`_load_collection_meta_yaml` (via the `aisimulate.sdk` compatibility
alias, which is the same module object — see
`src/aisimulate/sdk/_compat.py`). The tree-walking shape mirrors
`perf_data_layout.py`'s `iter_data_files`, specialized to the
post-migration family-first layout (`<system>/<family>/<backend>/<version>/`)
that is now the only layout on disk.

Usage:
    python3 tools/perf_database/check_collector_data.py
    python3 tools/perf_database/check_collector_data.py \\
        --data-root src/aisimulate_core/systems/data \\
        --catalog collector/op_backend_catalog.yaml

Exit codes: 0 = every rule OK. 1 = at least one failure (printed, grouped by
rule).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

# `collector/` is a standalone top-level package (not part of the installed
# wheel — see .claude/rules/repo-guide.md), so it is only importable when the
# repo root is on sys.path. Running this file directly (`python3
# tools/perf_database/check_collector_data.py`) puts the script's own
# directory on sys.path[0], not the repo root, so the insert below is
# required (mirrors collect.py:1808 / changed_ops.py's REPO_ROOT). Harmless
# under pytest, where pytest.ini's `pythonpath = .` already covers it.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# aisimulate_core is pip-installed editable (pyproject.toml
# [tool.uv.sources]), so it needs no sys.path help. Imported via the
# `aisimulate.sdk` compatibility alias, as the SDK-layer test suite does
# (e.g. tests/unit/sdk/database/test_dual_layout_discovery.py) — it is the
# same module object as aisimulate_core.sdk.perf_database
# (src/aisimulate/sdk/_compat.py:alias_module).
from collector.framework_manifest import MANIFEST_PATH, load_manifest, validate_resolution
from collector.op_catalog import (
    CATALOG_PATH,
    load_family_map,
)

from aisimulate.sdk.perf_database import (
    _load_collection_meta_yaml,
    _parse_reuse_yaml,
)

DATA_ROOT_DEFAULT = REPO_ROOT / "src" / "aisimulate_core" / "systems" / "data"

REUSE_YAML = "reuse.yaml"
COLLECTION_META_YAML = "collection_meta.yaml"
LEGACY_MARKERS = ("SHARED_LAYER_REUSE.txt", "INCOMPLETE.txt")
COMM_FAMILY = "comm"

RULES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")
RULE_TITLES = {
    "R1": "sidecar coverage",
    "R2": "reuse validity",
    "R3": "comm exclusion",
    "R4": "family placement",
    "R5": "identity (manifest v2 resolution)",
    "R6": "no legacy markers",
    "R7": "attested collection events",
    "R8": "notes",
}


# --------------------------------------------------------------------------
# tree walking: <data_root>/<system>/<family>/<backend>/<version>/
# --------------------------------------------------------------------------


def _subdirs(path: Path) -> list[Path]:
    """Sorted, non-dot-prefixed subdirectories (dot-dirs are never perf-data
    dirs; guards against `.git` when a test points data_root at a repo root).
    """
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith("."))


def _parquet_stems(version_dir: Path) -> set[str]:
    return {p.stem for p in version_dir.iterdir() if p.is_file() and p.suffix == ".parquet"}


def iter_version_dirs(data_root: Path) -> list[tuple[str, str, str, str, Path]]:
    """Every `(system, family, backend, version, version_dir)` in the
    family-first layout (Collector V3 §3) — the only layout left on disk
    after the PR2 migration.
    """
    version_dirs: list[tuple[str, str, str, str, Path]] = []
    for system_dir in _subdirs(data_root):
        for family_dir in _subdirs(system_dir):
            for backend_dir in _subdirs(family_dir):
                for version_dir in _subdirs(backend_dir):
                    version_dirs.append(
                        (system_dir.name, family_dir.name, backend_dir.name, version_dir.name, version_dir)
                    )
    return version_dirs


# --------------------------------------------------------------------------
# R1: sidecar coverage
# --------------------------------------------------------------------------


def check_r1_sidecar_coverage(data_root: Path, version_dirs: list[tuple[str, str, str, str, Path]]) -> list[str]:
    failures: list[str] = []
    for _system, _family, _backend, _version, version_dir in version_dirs:
        parquet_stems = _parquet_stems(version_dir)
        if not parquet_stems:
            continue
        rel_dir = version_dir.relative_to(data_root)
        meta_path = version_dir / COLLECTION_META_YAML
        if not meta_path.is_file():
            failures.append(f"{rel_dir}: missing {COLLECTION_META_YAML} for parquet table(s) {sorted(parquet_stems)}")
            continue
        try:
            meta = _load_collection_meta_yaml(str(meta_path))
        except ValueError as exc:
            failures.append(f"{rel_dir}/{COLLECTION_META_YAML}: {exc}")
            continue
        tables = meta.get("tables")
        covered = set(tables) if isinstance(tables, dict) else set()
        for table in sorted(parquet_stems - covered):
            failures.append(f"{rel_dir}: parquet table '{table}' has no entry in {COLLECTION_META_YAML} tables")
    return failures


# --------------------------------------------------------------------------
# R2: reuse validity
# --------------------------------------------------------------------------


def check_r2_reuse_validity(data_root: Path, version_dirs: list[tuple[str, str, str, str, Path]]) -> list[str]:
    failures: list[str] = []
    # (system, family, backend) -> {version: version_dir}, for donor lookup.
    # A reuse.yaml entry carries no backend field, so donors are always
    # resolved within the SAME (system, family, backend) subtree the
    # declaring dir lives in -- "same backend only" is structural, not a
    # separate check.
    by_line: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for system, family, backend, version, version_dir in version_dirs:
        by_line[(system, family, backend)][version] = version_dir

    for system, family, backend, _version, version_dir in version_dirs:
        reuse_path = version_dir / REUSE_YAML
        if not reuse_path.is_file():
            continue
        rel_reuse = reuse_path.relative_to(data_root)
        try:
            parsed = _parse_reuse_yaml(str(reuse_path))
        except ValueError as exc:
            failures.append(f"{rel_reuse}: {exc}")
            continue

        siblings = by_line[(system, family, backend)]
        for entry in parsed["entries"]:
            table = entry["table"]
            from_version = entry["from_version"]
            donor_dir = siblings.get(from_version)
            if donor_dir is None:
                failures.append(
                    f"{rel_reuse}: table '{table}' declares from_version '{from_version}', which does not "
                    f"exist under {system}/{family}/{backend}"
                )
                continue
            donor_table = donor_dir / f"{table}.parquet"
            if not donor_table.is_file():
                failures.append(
                    f"{rel_reuse}: table '{table}' from_version '{from_version}' has no real parquet data at "
                    f"{donor_table.relative_to(data_root)} (donor must hold actual data, not itself be a "
                    "declared-reuse-only dir)"
                )
    return failures


# --------------------------------------------------------------------------
# R3: comm declared-reuse exclusion (design §6.5 rule 5)
# --------------------------------------------------------------------------


def check_r3_comm_exclusion(data_root: Path) -> list[str]:
    failures: list[str] = []
    for system_dir in _subdirs(data_root):
        comm_dir = system_dir / COMM_FAMILY
        if not comm_dir.is_dir():
            continue
        for reuse_path in sorted(comm_dir.rglob(REUSE_YAML)):
            failures.append(
                f"{reuse_path.relative_to(data_root)}: reuse.yaml is not allowed under the comm family "
                "(design §6.5 rule 5 -- framework backends reuse earlier versions implicitly; "
                "NCCL/oneCCL and unknown backends are primary-only)"
            )
    return failures


# --------------------------------------------------------------------------
# R4: family placement
# --------------------------------------------------------------------------


def check_r4_family_placement(
    data_root: Path,
    version_dirs: list[tuple[str, str, str, str, Path]],
    family_map: dict[str, str] | None,
) -> list[str]:
    if family_map is None:
        return [f"op catalog not found; cannot validate family placement (fail-closed, expected at {CATALOG_PATH})"]

    failures: list[str] = []
    for _system, family, _backend, _version, version_dir in version_dirs:
        for parquet_path in sorted(p for p in version_dir.iterdir() if p.is_file() and p.suffix == ".parquet"):
            stem = parquet_path.stem
            expected_family = family_map.get(stem)
            rel = parquet_path.relative_to(data_root)
            if expected_family is None:
                failures.append(f"{rel}: table '{stem}' is not mapped to any family in the op catalog")
            elif expected_family != family:
                failures.append(
                    f"{rel}: table '{stem}' belongs to family '{expected_family}' per the op catalog, "
                    f"but is filed under family '{family}'"
                )
    return failures


# --------------------------------------------------------------------------
# R5: identity (manifest v2 resolution)
# --------------------------------------------------------------------------


def check_r5_identity(catalog_path: Path) -> list[str]:
    return list(validate_resolution(catalog_path=catalog_path))


# --------------------------------------------------------------------------
# R6: no legacy markers
# --------------------------------------------------------------------------


def check_r6_no_legacy_markers(data_root: Path) -> list[str]:
    failures: list[str] = []
    for marker_name in LEGACY_MARKERS:
        for marker_path in sorted(data_root.rglob(marker_name)):
            failures.append(
                f"{marker_path.relative_to(data_root)}: legacy marker file must be migrated and removed "
                "(superseded by reuse.yaml / collection_meta.yaml)"
            )
    return failures


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# R7: attested case plan
# --------------------------------------------------------------------------

# provenance.case_plan_hash of an EMPTY attempted-case set. A sidecar entry
# carrying this hash while its parquet holds rows attests a plan that was
# never observed -- collect.py refuses to finalize such a table
# (zero-checkpoint-evidence guard), so an entry like this can only come from
# out-of-band sidecar authoring that skipped the checkpoint (the rc23 moe
# sidecars shipped this way once; review finding on PR #1486).
EMPTY_CASE_PLAN_HASH = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def check_r7_attested_case_plan(data_root: Path, version_dirs: list[tuple[str, str, str, str, Path]]) -> list[str]:
    failures: list[str] = []
    for _system, _family, _backend, _version, version_dir in version_dirs:
        meta_path = version_dir / COLLECTION_META_YAML
        if not meta_path.is_file():
            continue  # R1 owns missing sidecars
        try:
            meta = _load_collection_meta_yaml(str(meta_path))
        except ValueError:
            continue  # R1 owns malformed sidecars
        tables = meta.get("tables")
        if not isinstance(tables, dict):
            continue
        rel_dir = version_dir.relative_to(data_root)
        for table, entry in tables.items():
            if not isinstance(entry, dict):
                continue
            collections = entry.get("collections")
            if isinstance(collections, list):
                for index, event in enumerate(collections):
                    if not isinstance(event, dict):
                        continue  # schema-v2 parser reports this under R1
                    event_label = f"table '{table}' collections[{index}]"
                    status = event.get("status")
                    if status not in ("complete", "partial"):
                        failures.append(
                            f"{rel_dir}/{COLLECTION_META_YAML}: {event_label} must record status: complete|partial"
                        )
                    plan_hash = event.get("case_plan_hash")
                    if not isinstance(plan_hash, str) or not plan_hash.strip():
                        failures.append(
                            f"{rel_dir}/{COLLECTION_META_YAML}: {event_label} must record a non-empty case_plan_hash"
                        )
                    elif plan_hash == EMPTY_CASE_PLAN_HASH:
                        failures.append(
                            f"{rel_dir}/{COLLECTION_META_YAML}: {event_label} case_plan_hash attests an EMPTY "
                            "attempted-case set (sha256 of nothing) -- regenerate the hash from the campaign "
                            "checkpoint's done+failed case ids"
                        )
                continue

            rows = entry.get("rows")
            plan_hash = entry.get("case_plan_hash")
            if isinstance(rows, int) and rows > 0 and plan_hash == EMPTY_CASE_PLAN_HASH:
                failures.append(
                    f"{rel_dir}/{COLLECTION_META_YAML}: table '{table}' has {rows} rows but case_plan_hash "
                    "attests an EMPTY attempted-case set (sha256 of nothing) -- regenerate the hash from the "
                    "campaign checkpoint's done+failed case ids"
                )
    return failures


# --------------------------------------------------------------------------
# R8: notes (collector/notes/<backend>.yaml)
# --------------------------------------------------------------------------

NOTES_DIR_DEFAULT = REPO_ROOT / "collector" / "notes"
NOTES_SCHEMA_VERSION = 1
NOTE_ID_PREFIX = {"deviation": "D-", "guard": "G-"}
NOTE_DIRECTIONS = ("over", "under", "unknown")
NOTE_GUARD_REASONS = ("mismeasure", "hang", "expensive")
_NOTE_KEYS = frozenset(
    ("id", "kind", "summary", "direction", "reason", "scope", "verified_on", "sources", "retire_via", "approved_by")
)
_NOTE_REQUIRED = ("id", "kind", "summary", "scope", "verified_on", "sources")
_NOTE_SCOPE_KEYS = {"tables", "systems", "versions"}
_NOTE_ID_RE = re.compile(r"[DG]-[a-z0-9][a-z0-9-]*")


def _version(value: object) -> Version | None:
    try:
        return Version(str(value))
    except InvalidVersion:
        return None


def _family_pin(framework_spec: dict, family: str) -> str:
    """The version a family is collected on: its family pin, else the framework default."""
    return ((framework_spec.get("families") or {}).get(family) or framework_spec["default"])["version"]


def _check_note(
    note: object,
    where: str,
    framework_spec: dict,
    family_map: dict[str, str],
    known_systems: set[str],
    seen_ids: dict[str, str],
) -> list[str]:
    if not isinstance(note, dict):
        return [f"{where}: a note must be a mapping"]
    failures: list[str] = []
    unknown = sorted(str(key) for key in set(note) - _NOTE_KEYS)
    if unknown:
        failures.append(f"{where}: unsupported key(s) {', '.join(unknown)}")
    missing = [key for key in _NOTE_REQUIRED if note.get(key) in (None, "", [], {})]
    if missing:
        failures.append(f"{where}: missing {', '.join(missing)}")
        return failures

    note_id, kind = note["id"], note["kind"]
    label = f"{where} {note_id}"
    if not isinstance(kind, str) or kind not in NOTE_ID_PREFIX:
        failures.append(f"{label}: kind must be deviation or guard, got {kind!r}")
    elif not (isinstance(note_id, str) and _NOTE_ID_RE.fullmatch(note_id) and note_id.startswith(NOTE_ID_PREFIX[kind])):
        failures.append(f"{label}: a {kind} id is {NOTE_ID_PREFIX[kind]}<backend>-<slug> in lowercase, digits, hyphens")
    if isinstance(note_id, str):
        if note_id in seen_ids:
            failures.append(f"{label}: duplicate id, first defined at {seen_ids[note_id]}")
        else:
            seen_ids[note_id] = where
    if kind == "deviation" and note.get("direction") not in NOTE_DIRECTIONS:
        failures.append(f"{label}: a deviation needs direction: {' | '.join(NOTE_DIRECTIONS)}")
    if kind == "guard" and note.get("reason") not in NOTE_GUARD_REASONS:
        failures.append(f"{label}: a guard needs reason: {' | '.join(NOTE_GUARD_REASONS)}")
    if kind != "deviation" and "direction" in note:
        failures.append(f"{label}: direction applies only to a deviation")
    if kind != "guard" and "reason" in note:
        failures.append(f"{label}: reason applies only to a guard")
    for key in ("summary", "retire_via", "approved_by"):
        if key in note and not (isinstance(note[key], str) and note[key].strip()):
            failures.append(f"{label}: {key} must be a non-empty string")
    sources = note["sources"]
    if not (isinstance(sources, list) and all(isinstance(source, str) and source.strip() for source in sources)):
        failures.append(f"{label}: sources must be a list of non-empty strings")

    scope = note["scope"]
    if not isinstance(scope, dict):
        failures.append(f"{label}: scope must be a mapping")
        return failures
    unknown_scope = sorted(str(key) for key in set(scope) - _NOTE_SCOPE_KEYS)
    if unknown_scope:
        failures.append(f"{label}: unsupported scope key(s) {', '.join(unknown_scope)}")
    tables = scope.get("tables")
    if not (isinstance(tables, list) and tables and all(isinstance(table, str) for table in tables)):
        failures.append(f"{label}: scope.tables must be a non-empty list of table names")
        tables = []
    systems = scope.get("systems")
    if systems != "all":
        if not (isinstance(systems, list) and systems):
            failures.append(f"{label}: scope.systems must be 'all' or a non-empty list")
        else:
            unknown_systems = sorted(str(s) for s in systems if not isinstance(s, str) or s not in known_systems)
            if unknown_systems:
                failures.append(f"{label}: unknown system(s) {', '.join(unknown_systems)}")
    versions = scope.get("versions", "all")
    if versions != "all" and not (isinstance(versions, list) and versions):
        failures.append(f"{label}: scope.versions must be 'all' or a non-empty list")
        return failures

    verified_on = _version(note["verified_on"])
    if verified_on is None:
        failures.append(f"{label}: verified_on {note['verified_on']!r} is not a version")
        return failures
    for table in tables:
        family = family_map.get(table)
        if family is None:
            failures.append(f"{label}: unknown table {table} (not in the op catalog)")
            continue
        pin = _family_pin(framework_spec, family)
        if versions != "all" and pin not in {str(version) for version in versions}:
            continue  # the note says nothing about the version collected now
        pinned = _version(pin)
        if pinned is not None and verified_on < pinned:
            failures.append(
                f"{label}: verified_on {note['verified_on']} is behind the {family} pin {pin} ({table}); "
                "re-check the note on the pinned version and move verified_on, or delete the note"
            )
    return failures


def check_r8_notes(
    notes_dir: Path,
    family_map: dict[str, str],
    frameworks: dict[str, dict],
    known_systems: set[str],
) -> list[str]:
    if not notes_dir.is_dir():
        return []
    failures: list[str] = []
    seen_ids: dict[str, str] = {}
    for path in sorted(notes_dir.glob("*.yaml")):
        name = f"notes/{path.name}"
        framework_spec = frameworks.get(path.stem)
        if framework_spec is None:
            failures.append(f"{name}: {path.stem!r} is not a framework in framework_manifest.yaml")
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            failures.append(f"{name}: invalid YAML ({exc})")
            continue
        if not (
            isinstance(doc, dict)
            and doc.get("schema_version") == NOTES_SCHEMA_VERSION
            and isinstance(doc.get("notes"), list)
        ):
            failures.append(f"{name}: expected schema_version: {NOTES_SCHEMA_VERSION} and a notes list")
            continue
        extra = sorted(str(key) for key in set(doc) - {"schema_version", "notes"})
        if extra:
            failures.append(f"{name}: unsupported top-level key(s) {', '.join(extra)}")
        for index, note in enumerate(doc["notes"]):
            failures.extend(_check_note(note, f"{name}[{index}]", framework_spec, family_map, known_systems, seen_ids))
    return failures


def run_r8_notes(notes_dir: Path, catalog_path: Path, manifest_path: Path) -> list[str]:
    """R8 with its inputs loaded; a load error becomes a failure, never an exception."""
    if not notes_dir.is_dir():
        return []
    try:
        family_map = load_family_map(catalog_path)
        frameworks = load_manifest(manifest_path)["frameworks"]
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return [f"cannot check notes: {exc}"]
    if family_map is None:
        return [f"cannot check notes: op catalog {catalog_path} does not exist"]
    known_systems = {path.name for path in _subdirs(DATA_ROOT_DEFAULT)}
    return check_r8_notes(notes_dir, family_map, frameworks, known_systems)


def run_checks(
    data_root: Path,
    catalog_path: Path,
    notes_dir: Path = NOTES_DIR_DEFAULT,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, list[str]]:
    """Run all rules and return {rule: [failure messages]} (empty list = pass).

    Pure function of its arguments; never raises on data problems -- every
    problem becomes a failure message so the caller can print all of them,
    never stopping at the first.
    """
    notes = run_r8_notes(notes_dir, catalog_path, manifest_path)
    if not data_root.is_dir():
        no_root = [f"data root {data_root} does not exist"]
        results = dict.fromkeys(RULES, no_root)
        results["R5"] = check_r5_identity(catalog_path)
        results["R8"] = notes
        return results

    version_dirs = iter_version_dirs(data_root)
    family_map = load_family_map(catalog_path)

    return {
        "R1": check_r1_sidecar_coverage(data_root, version_dirs),
        "R2": check_r2_reuse_validity(data_root, version_dirs),
        "R3": check_r3_comm_exclusion(data_root),
        "R4": check_r4_family_placement(data_root, version_dirs, family_map),
        "R5": check_r5_identity(catalog_path),
        "R6": check_r6_no_legacy_markers(data_root),
        "R7": check_r7_attested_case_plan(data_root, version_dirs),
        "R8": notes,
    }


def render_report(results: dict[str, list[str]]) -> tuple[str, bool]:
    lines: list[str] = []
    any_failures = False
    for rule in RULES:
        failures = results[rule]
        title = RULE_TITLES[rule]
        if failures:
            any_failures = True
            lines.append(f"[FAIL] {rule} {title}: {len(failures)} failure(s)")
            lines.extend(f"    - {msg}" for msg in failures)
        else:
            lines.append(f"[OK]   {rule} {title}")

    total = sum(len(v) for v in results.values())
    if any_failures:
        failed_rules = [rule for rule in RULES if results[rule]]
        lines.append(f"\ncollector data check FAILED: {total} failure(s) across rule(s) {', '.join(failed_rules)}")
    else:
        lines.append("\ncollector data check OK: all rules passed")
    return "\n".join(lines), any_failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT, help="Root of the systems/data tree.")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="Path to the op backend catalog yaml.")
    parser.add_argument("--notes-dir", type=Path, default=NOTES_DIR_DEFAULT, help="Directory of collector notes.")
    args = parser.parse_args(argv)

    results = run_checks(args.data_root, args.catalog, notes_dir=args.notes_dir)
    report, failed = render_report(results)
    print(report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
