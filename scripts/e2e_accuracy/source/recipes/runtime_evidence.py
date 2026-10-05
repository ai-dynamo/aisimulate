# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore pinned public runtime artifacts and reviewed, row-bound archived facts."""

import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import requests

from scripts.e2e_accuracy.source.recipes.runtime_recipe import inspect_cached_runtime_workload
from scripts.e2e_accuracy.source.schema import SiliconRow
from scripts.e2e_accuracy.source.sources import load_manifest

REPOSITORY = "SemiAnalysisAI/InferenceX"
MAX_BYTES = 512 * 1024 * 1024


def row_digest(row):
    return hashlib.sha256(json.dumps(asdict(row), sort_keys=True, allow_nan=False).encode()).hexdigest()


def archived_recipe(row, part="parsed"):
    manifest = load_manifest("runtime_observations.json")
    if manifest["schema_version"] != "reviewed-runtime-observations/1":
        raise ValueError("unsupported runtime observation schema")
    records = manifest["records"]
    key = row_digest(row)
    record = records.get(key)
    if record is None or part not in record:
        return None
    if record["benchmark_id"] != row.bench_id:
        raise ValueError("runtime observation measurement identity mismatch")
    parsed = deepcopy(record[part])
    parsed[-1]["archived_runtime"] = {
        "source": "reviewed runtime observations",
        "measurement_sha256": key,
        "manifest": "runtime_observations.json",
        "historical_artifacts_revalidated": False,
    }
    return tuple(parsed)


def _filename(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in {".", ".."}:
        raise ValueError("runtime evidence requires a simple cache filename")
    return name


def entry_files(entry, aggregated):
    """Use only fixed-repository API routes; manifests cannot redirect credentials."""
    files = {}
    archives = [entry["server"], entry["benchmark"]] if aggregated else [entry]
    for artifact in archives:
        artifact_id = artifact["id"]
        if type(artifact_id) is not int or artifact_id <= 0:
            raise ValueError("invalid runtime artifact ID")
        files[_filename(artifact["path"])] = (
            f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip",
            artifact["sha256"],
        )
    log_path = entry.get("logs_path" if aggregated else "run_log_path")
    if log_path:
        run, attempt = str(entry["github_run_id"]), entry["run_attempt"]
        if not run.isdecimal() or type(attempt) is not int or attempt <= 0:
            raise ValueError("invalid runtime run/attempt")
        proof = entry["run_attempt_evidence"]
        endpoint = f"repos/{REPOSITORY}/actions/runs/{run}/attempts/{attempt}/logs"
        if proof["logs_endpoint"] != endpoint:
            raise ValueError("runtime evidence log endpoint disagrees with run/attempt")
        files[_filename(log_path)] = (endpoint, proof["logs_archive_sha256"])
    for _, digest in files.values():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid runtime evidence checksum")
    return files


def _download(root, name, endpoint, expected):
    path = root / name
    if path.exists():
        if path.stat().st_size > MAX_BYTES or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"runtime evidence checksum mismatch: {name}")
        return "cached"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(
            f"https://api.github.com/{endpoint}",
            headers=headers,
            stream=True,
            timeout=60,
        ) as response:
            if response.status_code in {403, 404, 410}:
                return "unavailable"
            response.raise_for_status()
            size, digest = 0, hashlib.sha256()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError(f"runtime evidence exceeds size limit: {name}")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"runtime evidence checksum mismatch: {name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def prepare_runtime_evidence(points, cache_dir, *, workers=4):
    root = Path(cache_dir) / "runtime-evidence"
    root.mkdir(parents=True, exist_ok=True)
    runs = {str(point["source_row"]["github_run_id"]) for point in points}
    manifests, files = {}, {}
    for kind in ("agg", "disagg"):
        manifest = load_manifest(f"runtime_{kind}_index.json")
        entries = [entry for entry in manifest["artifacts"] if str(entry["github_run_id"]) in runs]
        manifests[kind] = (manifest, entries)
        for entry in entries:
            for name, identity in entry_files(entry, kind == "agg").items():
                if name in files and files[name] != identity:
                    raise ValueError("conflicting runtime evidence identities")
                files[name] = identity
    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = dict(
            zip(
                files,
                pool.map(lambda name: _download(root, name, *files[name]), files),
                strict=True,
            )
        )
    for kind, (manifest, entries) in manifests.items():
        available = [
            entry
            for entry in entries
            if all(statuses[name] != "unavailable" for name in entry_files(entry, kind == "agg"))
        ]
        (root / f"{kind}_index.json").write_text(
            json.dumps({**manifest, "artifacts": available}, sort_keys=True) + "\n"
        )
    return {"files": statuses, "counts": dict(Counter(statuses.values()))}


def freeze_records(points, source, read_recipe):
    """Generate reviewed facts only through the raw artifact parsers' identity checks.

    The caller supplies read_deployment_recipe to avoid a parser import cycle.
    Latency values stay out of the archive; the complete input row is hash-bound.
    """
    if source.archived_runtime:
        raise ValueError("archive generation requires raw runtime evidence")
    records = {}
    for point in points:
        row = SiliconRow(**point["source_row"])
        try:
            parsed = read_recipe(row, source)
        except ValueError:
            # A verified workload can identify the actual checkout even when
            # its server launcher is unsupported. Preserve only that evidence.
            if row.disagg:
                workload = inspect_cached_runtime_workload(row, source._cache_dir)["parsed"]
                if workload is not None:
                    records[row_digest(row)] = {"benchmark_id": row.bench_id, "workload": workload}
            continue
        evidence = parsed[-1]
        if not any(evidence.get(key) for key in ("artifact", "single_node_runtime", "runtime_workload")):
            continue
        records[row_digest(row)] = {"benchmark_id": row.bench_id, "parsed": parsed}
    return {"schema_version": "reviewed-runtime-observations/1", "records": records}
