#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity gate for the FPM move to ``data/fpm/`` (AISIM-2338).

Runs the *old* resolution (legacy ``catalog/index.json``, ``data/<model>/...`` paths, the whole tree) and the
*new* resolution (``catalog/dataset.json``, envelopes, selection before download through the library fetch
with ``fpm-provenance`` excluded) on the same migrated tree, then compares what the accuracy pipeline
consumes: selected configurations and their order, truth/helper membership and hashes, FPM artifacts, observation
identities, membership digests and (optionally) the evaluation rows (scores, coverage). It also checks that every
pinned profile and recipe file resolves at its new path with the same SHA-256.

The tree is only read. The old layout is a symlink view (``data/fpm/<rest>`` appears as ``data/<rest>``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from aisim_core.data import Manifest
from aisim_core.data.fpm_layout import logical_path
from scripts.fpm_accuracy.exceptions import DataError
from scripts.fpm_accuracy.hf import HfDataset
from scripts.fpm_accuracy.hf import dataset as dataset_module
from scripts.fpm_accuracy.hf.overrides import load_overrides

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ROOT / "python/aisim/src/aisim_core/systems"
RECIPES = ROOT / "python/aisim/collector/fpm_forward/recipes"
NEW_ONLY = ("catalog/dataset.json", "catalog/systems.parquet")
FAKE_REVISION = "0" * 39 + "1"


def _walk(tree: Path) -> Iterator[str]:
    for base, _, names in os.walk(tree):
        for name in names:
            yield (Path(base) / name).relative_to(tree).as_posix()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def old_layout_view(tree: Path, view: Path) -> None:
    """Symlink view in the layout of the legacy repository (the current pipeline's input)."""
    for rel in _walk(tree):
        if rel in NEW_ONLY or rel.startswith("catalog/fpm/") or rel.endswith("/aisim-manifest.json"):
            continue
        target = "data/" + rel[len("data/fpm/") :] if rel.startswith("data/fpm/") else rel
        (view / target).parent.mkdir(parents=True, exist_ok=True)
        (view / target).symlink_to(tree / rel)


class TreeTransport:
    """Serves the migrated tree as if it were the Hub; records every transfer."""

    def __init__(self, tree: Path, revision: str) -> None:
        self.tree, self.revision, self.opened = tree, revision, []

    def __call__(self, token: str | None = None) -> TreeTransport:  # replaces the HfTransport class
        return self

    def resolve_revision(self, repo: str, revision: str) -> str:
        return self.revision

    def open_file(self, repo: str, sha: str, path: str) -> Iterator[bytes]:
        self.opened.append(path)
        with (self.tree / path).open("rb") as stream:
            while chunk := stream.read(1 << 20):
                yield chunk


def _signature(case) -> dict:
    configuration = case.configuration
    return {
        "case_id": case.case_id,
        "membership_sha256": case.measurement_membership_sha256,
        "status": str(case.status),
        "configuration_id": configuration.configuration_id,
        "configuration_path": configuration.configuration_path,
        "manifest_sha256": configuration.manifest_sha256,
        "truth": [(f.measurement_file_id, f.path, f.sha256) for f in case.truth_files],
        "helpers": [(f.measurement_file_id, f.path, f.sha256) for f in case.helper_files],
        "fpm": [(a.artifact_id, a.path, a.sha256, a.metadata_sha256, a.row_count) for a in case.fpm_artifacts],
        "observations": [(o.observation_id, o.source_path, o.source_row, o.actual_ms) for o in case.observations],
        "worker_role": case.worker_role,
        "ordering": str(case.ordering),
        "warnings": list(case.warnings),
    }


def _urls(case) -> list[str]:
    return list(case.provenance_urls)


def _normalize_old_url(url: str, repo: str, revision: str) -> str:
    prefix = f"https://huggingface.co/datasets/{repo}/blob/{revision}/data/"
    return prefix + "fpm/" + url[len(prefix) :] if url.startswith(prefix) else url


def pins_report(tree: Path, old: Path) -> dict:
    """Every pinned file (profiles, recipes) must exist at its physical path with the same SHA-256."""
    checked, problems = 0, []

    def check(source: str, repo: str, path: str, sha256: str) -> None:
        nonlocal checked
        checked += 1
        if repo != dataset_module.DEFAULT_REPO_ID:
            return  # a deliberate legacy pin resolves in the retired repository, not in this tree
        physical = tree / path
        if not physical.is_file():
            problems.append({"source": source, "path": path, "problem": "missing"})
        elif _sha256(physical) != sha256:
            problems.append({"source": source, "path": path, "problem": "sha256 differs"})
        elif path.startswith("data/fpm/") and not (old / logical_path(path)).is_file():
            problems.append({"source": source, "path": path, "problem": "absent from the old-layout view"})

    legacy = 0
    for manifest in [*SYSTEMS.glob("profiles/*/*hf_dataset.json"), SYSTEMS / "dsv41_fpm_hf.json"]:
        document = json.loads(manifest.read_text())
        if document["repo_id"] != dataset_module.DEFAULT_REPO_ID:
            legacy += sum(len(entry["files"]) for entry in document["profiles"].values())
        for entry in document["profiles"].values():
            for item in entry["files"]:
                check(manifest.name, document["repo_id"], item["path"], item["sha256"])
    for recipe in sorted(RECIPES.glob("*.json")):
        pin = json.loads(recipe.read_text())
        check(recipe.name, pin["repo_id"], pin["filename"], pin["sha256"])
    return {"checked": checked, "legacy_pinned_files_not_checked": legacy, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True, help="migrated tree (read only)")
    parser.add_argument("--repo-id", default=dataset_module.DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=FAKE_REVISION, help="label used by both sides (default: placeholder)")
    parser.add_argument("--configuration", action="append", help="limit both sides to these configuration paths")
    parser.add_argument(
        "--evaluate",
        default="0",
        help="number of configurations to score on both sides, or 'all' (needs the native forward-pass SDK)",
    )
    parser.add_argument("--skip-cases", action="store_true", help="skip the per-configuration case comparison")
    parser.add_argument("--skip-pins", action="store_true", help="do not check profile and recipe pins (test trees)")
    parser.add_argument("--workdir", type=Path, help="scratch directory (default: a temporary one)")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tree = args.tree.resolve()
    scratch = Path(args.workdir or tempfile.mkdtemp(prefix="fpm-parity-"))
    scratch.mkdir(parents=True, exist_ok=True)
    report: dict = {"tree": str(tree), "repo_id": args.repo_id, "revision_label": args.revision, "checks": {}}
    failures: list[str] = []

    def record(name: str, ok: bool, **detail) -> None:
        report["checks"][name] = {"ok": ok, **detail}
        if not ok:
            failures.append(name)
        print(f"{'PASS' if ok else 'FAIL'} {name}", flush=True)

    # -- old: whole legacy layout, legacy catalog ----------------------------------------------------------
    view = scratch / "old-layout"
    old_layout_view(tree, view)
    overrides = load_overrides(None)
    old = HfDataset(
        view,
        repo_id=args.repo_id,
        revision=args.revision,
        overrides=overrides,
        overrides_sha256=None,
        allowed_file_roots=[tree],
    )
    # -- new: select from the catalog, fetch only evaluation artifacts through the library -----------------
    transport = TreeTransport(tree, args.revision)
    original_transport = dataset_module.HfTransport
    dataset_module.HfTransport = transport
    try:
        new = HfDataset.from_hub(
            args.repo_id,
            revision="main",
            cache_dir=scratch / "cache",
            configuration_paths=args.configuration,
        )
    finally:
        dataset_module.HfTransport = original_transport
    tree_bytes = sum((tree / rel).stat().st_size for rel in _walk(tree) if rel.startswith("data/fpm/"))
    downloaded = sum((tree / rel).stat().st_size for rel in transport.opened if rel.startswith("data/fpm/"))
    envelopes = [rel for rel in _walk(tree) if rel.endswith("/aisim-manifest.json")]
    declared_provenance = sum(
        a.bytes
        for rel in envelopes
        for a in Manifest.model_validate_json((tree / rel).read_bytes()).artifacts
        if a.role == "fpm-provenance"
    )
    report["download"] = {
        "layout": {"old": old.layout, "new": new.layout},
        "tree_data_bytes": tree_bytes,
        "new_pipeline_bytes": downloaded,
        "excluded_fpm_provenance_bytes": declared_provenance,
        "files_opened": len(transport.opened),
        "fpm_provenance_opened": sum("/fpm/provenance/" in p for p in transport.opened),
    }
    record(
        "layouts",
        (old.layout, new.layout) == ("legacy", "unified"),
        old=old.layout,
        new=new.layout,
    )
    record("no fpm-provenance fetched", report["download"]["fpm_provenance_opened"] == 0)

    selected = set(args.configuration or [])
    old_configs = [c for c in old.configurations() if not selected or c.configuration_path in selected]
    new_configs = list(new.configurations())
    record(
        "selected libraries (configuration identity, order, manifest hash)",
        [(c.configuration_id, c.configuration_path, c.snapshot_id, c.manifest_sha256) for c in old_configs]
        == [(c.configuration_id, c.configuration_path, c.snapshot_id, c.manifest_sha256) for c in new_configs],
        old=len(old_configs),
        new=len(new_configs),
    )
    record(
        "worker configuration records",
        [c.worker_config_record for c in old_configs] == [c.worker_config_record for c in new_configs],
    )

    counts = {"configurations": len(new_configs), "truth": 0, "helpers": 0, "fpm": 0, "observations": 0}
    mismatches: list[dict] = []
    evaluate_limit = len(new_configs) if args.evaluate == "all" else int(args.evaluate)
    evaluated, differing = [], []
    started = time.time()
    for old_configuration, new_configuration in zip(old_configs, new_configs, strict=True):
        if args.skip_cases:
            break
        kwargs = {"snapshot_id": old_configuration.snapshot_id}
        old_case = old.measurement_case(old_configuration.configuration_path, **kwargs)
        new_case = new.measurement_case(new_configuration.configuration_path, **kwargs)
        a, b = _signature(old_case), _signature(new_case)
        urls_equal = [_normalize_old_url(u, args.repo_id, args.revision) for u in _urls(old_case)] == _urls(new_case)
        if a != b or not urls_equal:
            mismatches.append(
                {
                    "configuration_path": a["configuration_path"],
                    "fields": [k for k in a if a[k] != b[k]],
                    "urls": urls_equal,
                }
            )
        counts["truth"] += len(a["truth"])
        counts["helpers"] += len(a["helpers"])
        counts["fpm"] += len(a["fpm"])
        counts["observations"] += len(a["observations"])
        if len(evaluated) < evaluate_limit:
            from scripts.fpm_accuracy.evaluate import evaluate_case

            t0 = time.time()
            old_row, new_row = evaluate_case(old_case), evaluate_case(new_case)
            same = json.dumps(old_row, sort_keys=True) == json.dumps(new_row, sort_keys=True)
            predicted = {
                method: [result["status"], result["metrics"]["all"]["predicted_count"]]
                for method, result in old_row["results"].items()
            }
            evaluated.append(
                {
                    "configuration_path": a["configuration_path"],
                    "seconds": round(time.time() - t0, 1),
                    "measured": len(a["observations"]),
                    "status_and_predicted_count_by_method": predicted,
                }
            )
            if not same:
                differing.append(a["configuration_path"])
            print(f"  scored {a['configuration_path']} {evaluated[-1]['seconds']}s identical={same}", flush=True)
    report["counts"] = counts
    report["case_resolution_seconds"] = round(time.time() - started, 1)
    if not args.skip_cases:
        record(
            "truth membership, hashes, FPM artifacts, case IDs, provenance links",
            not mismatches,
            mismatches=mismatches,
            **counts,
        )

    # History: not selected by the new default (finding 2 of AISIM-2337). The old pipeline lists the catalog's
    # history manifests; the new one derives them from the pointers inside the current manifests.
    def outcome(fn):
        try:
            return "ok", fn()
        except DataError as exc:
            return "error", str(exc)

    legacy_index = json.loads((tree / "catalog/index.json").read_text())
    old_kind, old_all = outcome(lambda: old.configurations(include_history=True))
    new_kind, new_all = outcome(
        lambda: HfDataset(
            tree, repo_id=args.repo_id, revision=args.revision, overrides=overrides, overrides_sha256=None
        ).configurations(include_history=True)
    )
    reachable = sorted(
        {
            h["manifest_path"]
            for c in new_configs
            for h in json.loads(old._safe_path(c.manifest_path).read_text()).get("history") or []
        }
    )
    catalog_history = legacy_index["history_manifests"]
    detail = {
        "old": old_kind if old_kind == "error" else len(old_all),
        "old_error": old_all if old_kind == "error" else None,
        "new": new_kind if new_kind == "error" else len(new_all),
        "catalog_history_manifests": len(catalog_history),
        "pointer_reachable_history": len(reachable),
        "orphans_in_catalog": sorted(set(catalog_history) - set(reachable)),
    }
    if old_kind == "ok" and new_kind == "ok":
        ok = [(c.configuration_id, c.manifest_sha256) for c in old_all] == [
            (c.configuration_id, c.manifest_sha256) for c in new_all
        ]
    else:
        # The legacy catalog lists history manifests that no current manifest points to, so the old read fails
        # closed. The new read must list exactly the pointer-reachable snapshots, each from the same bytes.
        ok = (
            new_kind == "ok"
            and len(new_all) - len(new_configs) == len(reachable)
            and set(reachable) <= set(catalog_history)
        )
    if not selected:
        record("history snapshots", ok, **detail)

    legacy_counts = legacy_index.get("counts")
    if legacy_counts and not selected:
        record(
            "legacy catalog counts",
            legacy_counts["configurations"] == len(old_configs)
            and legacy_counts["history_snapshots"] == len(catalog_history),
            legacy=legacy_counts,
        )

    if not args.skip_pins:
        pins = pins_report(tree, view)
        report["pins"] = pins
        record("profile and recipe pins resolve at their new paths", not pins["problems"], **pins)

    if evaluate_limit:
        report["evaluated"] = evaluated
        record(
            "evaluation rows (scores, coverage counts) identical",
            not differing,
            differing=differing,
            count=len(evaluated),
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"report: {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
