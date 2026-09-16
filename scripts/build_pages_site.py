#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the explicitly allowlisted public AISimulate GitHub Pages artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = Path("python/aisimulate/docs")
SYSTEMS_ROOT = Path("python/aisimulate/src/aiconfigurator_core/systems")

# Directories are opt-in so adding internal documentation under docs/ never
# publishes it accidentally. Every listed page is now part of the required
# public surface and its absence must fail the build.
PUBLIC_PAGE_DIRECTORIES = {
    "support-matrix": True,
    "e2e-accuracy": True,
    "fpe-support-matrix": True,
}
PUBLIC_ASSET_SUFFIXES = {".css", ".html", ".js", ".json", ".png", ".svg", ".webp"}
PUBLIC_DATASETS = {
    "support-matrix": "support_matrix",
    "fpe-support-matrix": "fpe_support_matrix",
}


class PagesBuildError(RuntimeError):
    """Raised when the public Pages artifact contract is invalid."""


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise PagesBuildError(f"public artifact source cannot be a symlink: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_page_directory(source: Path, destination: Path) -> None:
    if not (source / "index.html").is_file():
        raise PagesBuildError(f"public page has no index.html: {source}")
    for asset in sorted(source.rglob("*")):
        if not asset.is_file() or asset.suffix.lower() not in PUBLIC_ASSET_SUFFIXES:
            continue
        _copy_file(asset, destination / asset.relative_to(source))


def _copy_dataset(source: Path, destination: Path) -> None:
    index_path = source / "index.json"
    if not index_path.is_file():
        raise PagesBuildError(f"public dataset index is missing: {index_path}")

    try:
        index = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise PagesBuildError(f"cannot read public dataset index: {index_path}") from exc

    files = index.get("files") if isinstance(index, dict) else None
    if not isinstance(files, list) or not files:
        raise PagesBuildError(f"public dataset index has no files: {index_path}")

    _copy_file(index_path, destination / "index.json")
    for filename in files:
        if not isinstance(filename, str) or Path(filename).name != filename or Path(filename).suffix.lower() != ".csv":
            raise PagesBuildError(f"unsafe public dataset entry in {index_path}: {filename!r}")
        csv_path = source / filename
        if not csv_path.is_file():
            raise PagesBuildError(f"public dataset file is missing: {csv_path}")
        _copy_file(csv_path, destination / filename)


def _record_legacy_snapshot(repo_root: Path, destination: Path) -> None:
    """Describe the copied legacy data without treating a commit date as a test run."""
    index_path = destination / "index.json"
    index = json.loads(index_path.read_text())
    snapshot = {"kind": "historical", "qualification": "not_recorded"}
    dataset = SYSTEMS_ROOT / "support_matrix"
    paths = [str(dataset / name) for name in ["index.json", *index["files"]]]

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()

    try:
        # Archives, shallow clones, and locally changed data cannot establish
        # the last committed update of the exact dataset being displayed.
        if Path(git("rev-parse", "--show-toplevel")).resolve() != repo_root.resolve():
            raise ValueError("dataset is outside the repository root")
        if git("rev-parse", "--is-shallow-repository") != "false":
            raise ValueError("dataset history is incomplete")
        git("ls-files", "--error-unmatch", "--", *paths)
        git("diff", "--quiet", "HEAD", "--", *paths)
        commit, updated_at = git("log", "-1", "--format=%H%n%cI", "HEAD", "--", *paths).splitlines()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("invalid data commit")
        updated = datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            raise ValueError("data commit timestamp requires a time zone")
        snapshot.update(
            data_commit=commit,
            data_updated_at=updated.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        # Keep the historical matrix useful without inventing a date or
        # inheriting unverified metadata from its source index.
        pass
    index["snapshot"] = snapshot
    index_path.write_text(json.dumps(index, indent=2) + "\n")


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as exc:
        raise PagesBuildError(f"cannot read accuracy branch evidence: {exc.stderr.strip()}") from exc


def _accuracy_summary(text: str) -> dict:
    """Reject incomplete branch artifacts before replacing the deployed site."""

    def require(condition: bool, field: str) -> None:
        if not condition:
            raise ValueError(f"invalid or missing {field}")

    def number(value: object) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    def strings(value: object) -> bool:
        return isinstance(value, list) and bool(value) and all(isinstance(item, str) for item in value)

    def branch_name(value: object) -> bool:
        return (
            isinstance(value, str)
            and not value.endswith("/")
            and (value == "main" or bool(re.fullmatch(r"release/[A-Za-z0-9][A-Za-z0-9._/-]*", value)))
        )

    def aggregate(item: dict) -> None:
        require(isinstance(item, dict) and type(item.get("rows")) is int and item["rows"] > 0, "rows")
        for name in ("aic", "aisimulate"):
            metrics = item.get(name)
            require(isinstance(metrics, dict), name)
            require(type(metrics.get("points")) is int and 0 <= metrics["points"] <= item["rows"], f"{name}.points")
            for metric in ("ttft_mape_pct", "tpot_mape_pct", "ttft_shape_error_pct", "tpot_shape_error_pct"):
                require(metric in metrics and (metrics[metric] is None or number(metrics[metric])), f"{name}.{metric}")
        counts = item["aisimulate"].get("status_counts")
        require(isinstance(counts, dict), "status_counts")
        require(
            all(
                type(counts.get(key)) is int and counts[key] >= 0
                for key in ("success", "unsupported", "failed", "unknown")
            ),
            "status counts",
        )
        require(
            sum(counts.values()) == item["rows"] and counts["success"] == item["aisimulate"]["points"],
            "coverage totals",
        )

    def topology(item: dict) -> None:
        aggregate(item)
        require(isinstance(item.get("id"), str) and bool(re.fullmatch(r"[0-9a-f]{16}", item["id"])), "topology id")
        require(
            all(isinstance(item.get(key), str) for key in ("framework", "precision", "serving", "spec_method")),
            "topology dimensions",
        )
        require(isinstance(item.get("parallelism"), dict), "parallelism")
        points = item.get("points")
        require(isinstance(points, list) and len(points) == item["rows"], "topology points")
        previous = 0
        counts = {"success": 0, "unsupported": 0, "failed": 0, "unknown": 0}
        for point in points:
            require(isinstance(point, dict), "point")
            concurrency = point.get("concurrency")
            require(number(concurrency) and concurrency > 0 and concurrency >= previous, "concurrency")
            previous = concurrency
            status = point.get("status")
            require(status in ("success", "unsupported", "failed"), "point status")
            counts[status] += 1
            for name in ("measured", "aic", "aisimulate"):
                series = point.get(name)
                require(isinstance(series, dict), "point series")
                for metric in ("ttft", "tpot"):
                    keys = [f"{metric}_relative"] + ([f"{metric}_error_pct"] if name != "measured" else [])
                    missing = name == "aisimulate" and status != "success"
                    require(
                        all(
                            key in series and (series[key] is None if missing else number(series[key])) for key in keys
                        ),
                        "point metric",
                    )
        require(counts == item["aisimulate"]["status_counts"], "topology status counts")

    try:
        summary = json.loads(text)
        require(isinstance(summary, dict) and summary.get("schema_version") == 1, "summary schema")
        snapshot = summary.get("snapshot")
        require(isinstance(snapshot, dict), "snapshot")
        require(isinstance(snapshot.get("release_tag"), str), "measurement release")
        require(
            snapshot.get("measurement_source_url")
            == "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/" + snapshot["release_tag"],
            "measurement source URL",
        )
        require(isinstance(snapshot.get("aisimulate_packages"), dict), "AISimulate packages")
        for date in ("measurement_date_through", "aisimulate_completed_at"):
            require(snapshot.get(date) is None or isinstance(snapshot[date], str), date)
        revision = snapshot.get("evaluated_revision")
        if revision is not None:
            require(
                isinstance(revision, dict)
                and isinstance(revision.get("commit_sha"), str)
                and bool(re.fullmatch(r"[0-9a-f]{40}", revision["commit_sha"]))
                and branch_name(revision.get("branch")),
                "evaluated revision",
            )
        if revision is not None or "aic_source" in snapshot:
            aic_source = snapshot.get("aic_source")
            require(
                isinstance(aic_source, dict)
                and aic_source.get("repository") == "https://github.com/ai-dynamo/aisimulate"
                and isinstance(aic_source.get("branch"), str)
                and isinstance(aic_source.get("commit_sha"), str)
                and bool(re.fullmatch(r"[0-9a-f]{40}", aic_source["commit_sha"]))
                and aic_source["commit_sha"] == snapshot.get("aic_commit_sha")
                and (revision is None or all(aic_source[key] == revision[key] for key in ("branch", "commit_sha"))),
                "legacy AIC CLI source",
            )
        scope = summary.get("scope")
        require(isinstance(scope, dict) and scope.get("multinode") in ("included", "excluded"), "scope")
        require(
            type(scope.get("excluded_multinode_rows")) is int and scope["excluded_multinode_rows"] >= 0, "excluded rows"
        )
        require(isinstance(scope.get("claim"), str), "scope claim")
        totals = summary.get("totals")
        aggregate(totals)
        require(strings(totals.get("gpu_skus")) and strings(totals.get("precisions")), "total dimensions")
        models = summary.get("models")
        require(isinstance(models, list) and bool(models) and len(models) == totals.get("models"), "models")
        for model in models:
            aggregate(model)
            require(isinstance(model.get("model"), str), "model name")
            require(strings(model.get("gpu_skus")) and strings(model.get("precisions")), "model dimensions")
            workloads = model.get("workloads")
            require(isinstance(workloads, list) and bool(workloads), "workloads")
            for workload in workloads:
                aggregate(workload)
                require(
                    isinstance(workload.get("identity"), str) and isinstance(workload.get("label"), str),
                    "workload identity",
                )
                require(
                    strings(workload.get("gpu_skus")) and strings(workload.get("precisions")), "workload dimensions"
                )
                gpus = workload.get("gpus")
                require(isinstance(gpus, list) and bool(gpus), "GPUs")
                for gpu in gpus:
                    aggregate(gpu)
                    require(isinstance(gpu.get("gpu"), str) and strings(gpu.get("precisions")), "GPU dimensions")
                    if "topologies" in gpu:
                        require(isinstance(gpu["topologies"], list) and bool(gpu["topologies"]), "topologies")
                        for item in gpu["topologies"]:
                            topology(item)
                        require(sum(item["rows"] for item in gpu["topologies"]) == gpu["rows"], "topology coverage")
                require(sum(item["rows"] for item in gpus) == workload["rows"], "GPU coverage")
            require(sum(item["rows"] for item in workloads) == model["rows"], "workload coverage")
        require(sum(item["rows"] for item in models) == totals["rows"], "model coverage")
        return summary
    except (ValueError, AttributeError, TypeError, KeyError) as exc:
        raise PagesBuildError(f"invalid accuracy summary: {exc}") from exc


def _build_accuracy_catalog(repo_root: Path, output_dir: Path, include_refs: bool) -> None:
    """Package data only from release refs; all branches share the reviewed UI.

    A branch's tree is a publication location, never evidence that its current
    commit was evaluated. Legacy package-only results retain that distinction.
    """
    relative_path = DOCS_ROOT / "e2e-accuracy" / "summary.json"
    entries = []
    sources = [("main", None)]
    if include_refs:
        refs = _git(repo_root, "for-each-ref", "--format=%(refname)", "refs/remotes/origin/release/")
        sources.extend((ref.removeprefix("refs/remotes/origin/"), ref) for ref in refs.splitlines())
    for branch, ref in sources:
        entry = {"branch": branch, "summary_path": None, "published_from_commit": None}
        if ref is None:
            source = repo_root / relative_path
            if source.is_symlink():
                raise PagesBuildError("accuracy summary cannot be a symlink")
            content = source.read_text()
            if include_refs:
                entry["published_from_commit"] = _git(repo_root, "rev-parse", "HEAD")
        else:
            commit = _git(repo_root, "rev-parse", ref)
            entry["published_from_commit"] = commit
            tree_entry = _git(repo_root, "ls-tree", commit, "--", relative_path.as_posix())
            if not tree_entry:
                entry["status"] = "unavailable"
                entries.append(entry)
                continue
            if not tree_entry.startswith("100644 blob "):
                raise PagesBuildError(f"accuracy summary must be a regular file on {branch}")
            content = _git(repo_root, "show", f"{commit}:{relative_path.as_posix()}")
        summary = _accuracy_summary(content)
        revision = summary["snapshot"].get("evaluated_revision")
        if revision and revision["branch"] == branch:
            entry["status"] = "evaluated"
        elif revision:
            entry["status"] = "inherited"
        else:
            entry["status"] = "historical"
        if revision:
            entry["evaluated_revision"] = revision
        # Fixed hex paths avoid path traversal, slash encoding, and collisions
        # between release/foo and release-foo. Never copy branch HTML or JS.
        key = hashlib.sha256(branch.encode()).hexdigest()[:16]
        entry["summary_path"] = f"branches/{key}/summary.json"
        destination = output_dir / "e2e-accuracy" / entry["summary_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        entries.append(entry)
    catalog = {"schema_version": 1, "default_branch": "main", "branches": entries}
    (output_dir / "e2e-accuracy" / "branches.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")


def build_site(
    repo_root: Path, output_dir: Path, *, fpe_data_dir: Path | None = None, accuracy_refs: bool = False
) -> set[Path]:
    """Build the public site and return its files relative to ``output_dir``."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == repo_root:
        raise PagesBuildError("the Pages output directory cannot be the repository root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PagesBuildError(f"the Pages output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_root = repo_root / DOCS_ROOT
    index_path = docs_root / "index.html"
    if not index_path.is_file():
        raise PagesBuildError(f"public landing page is missing: {index_path}")
    _copy_file(index_path, output_dir / "index.html")
    (output_dir / ".nojekyll").touch()

    for public_name, required in PUBLIC_PAGE_DIRECTORIES.items():
        page_source = docs_root / public_name
        if not page_source.is_dir():
            if required:
                raise PagesBuildError(f"required public page is missing: {page_source}")
            continue
        _copy_page_directory(page_source, output_dir / public_name)

        dataset_name = PUBLIC_DATASETS.get(public_name)
        if dataset_name:
            _copy_dataset(
                fpe_data_dir
                if public_name == "fpe-support-matrix" and fpe_data_dir is not None
                else repo_root / SYSTEMS_ROOT / dataset_name,
                output_dir / "data" / public_name,
            )
            if public_name == "support-matrix":
                _record_legacy_snapshot(repo_root, output_dir / "data" / public_name)

    _build_accuracy_catalog(repo_root, output_dir, accuracy_refs)

    return {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fpe-data-dir", type=Path, help="Qualified FPE data prepared for this deployment")
    parser.add_argument(
        "--accuracy-refs", action="store_true", help="Include fetched origin/release/* accuracy snapshots"
    )
    args = parser.parse_args()

    files = build_site(
        args.repo_root, args.output_dir, fpe_data_dir=args.fpe_data_dir, accuracy_refs=args.accuracy_refs
    )
    print(f"Built {len(files)} public files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
