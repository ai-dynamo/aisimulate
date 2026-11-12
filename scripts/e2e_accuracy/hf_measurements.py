# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read pinned private HF observations and resolve their exact producer attempts.

Dataset access (revision resolution, verified cached transfer, the E2E loader and
snapshot identity) is delegated to the shared ``aisim_core.data`` library;
this module keeps only the campaign-specific selection of the InferenceX
cohort and the GitHub producer lookup. The library is imported lazily so that
shard jobs, which only need the policy constants, do not require it.

The API current view is a different cohort from a historical database dump.
Configuration IDs below are local content identities, never upstream database IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
import requests

REPO_ID = "ai-dynamo/aisim-infx-cache"
INDEX_PATH = "catalog/e2e/inferencex.json"
POLICY = "hf-current-view-config-v1"
MTP_POLICY = "hf-current-view-config-v2"
# Pin-file layouts. "legacy-inferencex" (the default, so existing pin files keep
# working) is the source-index layout; "aisim-e2e/v1" is the unified dataset
# catalog read through the library's E2E loader.
LEGACY_LAYOUT = "legacy-inferencex"
UNIFIED_LAYOUT = "aisim-e2e/v1"
LAYOUTS = (LEGACY_LAYOUT, UNIFIED_LAYOUT)
SOURCE_SCOPE = {"kind": "api-current-model-view"}
DATASET_FILE = "dataset.json"
PRODUCER_URL = re.compile(
    r"https://github\.com/SemiAnalysisAI/InferenceX/actions/runs/([1-9][0-9]*)/attempts/([1-9][0-9]*)"
)
_LEGACY_PREFIXES = ("catalog/e2e/", "data/e2e/inferencex/")
_REQUIRED_FIELDS = {"schema_version", "repo_id", "revision", "release_tag", "selection_policy", "max_age_days"}
_OPTIONAL_FIELDS = {"layout", "snapshot_id"}


def _library():
    """Import the shared data library, falling back to this checkout's source tree."""
    try:
        import aisim_core.data.fetch  # noqa: F401
    except ImportError:
        # An older installed package may lack the library; prefer this checkout's source tree.
        for name in [n for n in sys.modules if n.split(".")[0] in {"aisim", "aisim_core"}]:
            del sys.modules[name]
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "aisim" / "src"))
    from aisim_core.data import e2e, fetch

    return fetch, e2e


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def validate_manifest(manifest):
    if (
        not isinstance(manifest, dict)
        or not set(manifest) >= _REQUIRED_FIELDS
        or set(manifest) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    ):
        raise ValueError("invalid HF measurement manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 2:
        raise ValueError("unsupported HF measurement manifest schema")
    if manifest["repo_id"] != REPO_ID or manifest["selection_policy"] not in {POLICY, MTP_POLICY}:
        raise ValueError("unsupported HF measurement source or cohort policy")
    if not isinstance(manifest["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("HF measurements require a full commit SHA")
    if manifest["release_tag"] != "hf/" + manifest["revision"]:
        raise ValueError("HF measurement label does not match revision")
    if type(manifest["max_age_days"]) is not int or manifest["max_age_days"] < 0:
        raise ValueError("invalid HF measurement max_age_days")
    if manifest.get("layout", LEGACY_LAYOUT) not in LAYOUTS:
        raise ValueError("unsupported HF measurement layout")
    if "snapshot_id" in manifest and (
        not isinstance(manifest["snapshot_id"], str)
        or not re.fullmatch(r"[0-9A-Za-z._:-]{1,128}", manifest["snapshot_id"])
    ):
        raise ValueError("invalid HF measurement snapshot_id")


def default_transport():
    """The library's HF transport, refusing a repository that is no longer private."""
    fetch, _ = _library()

    class PrivateTransport(fetch.HfTransport):
        def resolve_revision(self, repo, revision):
            from huggingface_hub import HfApi

            info = HfApi(token=self._token).repo_info(repo, repo_type="dataset", revision=revision)
            if not info.private:
                raise ValueError("InferenceX cache must remain private")
            return info.sha

    return PrivateTransport()


def _get(cache, transport, sha, path, ref=None):
    """Verified local path of one dataset file at the pinned commit (cached by commit)."""
    fetch, _ = _library()
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path:
        raise ValueError("unsafe HF measurement artifact path")
    expected = None
    if ref is not None:
        expected = fetch.PlannedFile(
            path,
            ref["bytes"],
            ref["sha256"],
            "artifact",
            False,
            format=ref.get("format"),
            schema=ref.get("schema"),
            rows=ref.get("rows"),
        )
    if not cache.valid(REPO_ID, sha, path, expected):
        cache.discard(REPO_ID, sha, path)
        cache.store(REPO_ID, sha, path, transport.open_file(REPO_ID, sha, path), expected)
    return cache.path_for(REPO_ID, sha, path)


def _legacy_snapshot_id(cache, transport, sha):
    return json.loads(_get(cache, transport, sha, INDEX_PATH).read_bytes())["snapshot_id"]


def resolve_latest_manifest(template, transport=None, cache=None):
    """Freeze ``main`` to one commit and record that commit's snapshot ID when it is cheap to read."""
    validate_manifest(template)
    fetch, _ = _library()
    transport = transport or default_transport()
    sha = fetch.resolve_revision(transport, REPO_ID, "main")
    resolved = {k: v for k, v in template.items() if k != "snapshot_id"}
    resolved |= {"revision": sha, "release_tag": "hf/" + sha}
    if resolved.get("layout", LEGACY_LAYOUT) == LEGACY_LAYOUT:
        try:
            resolved["snapshot_id"] = _legacy_snapshot_id(cache or fetch.Cache(), transport, sha)
        except fetch.DataFetchError as exc:
            raise ValueError(f"HF source index unavailable at {sha}: {exc}") from exc
    validate_manifest(resolved)
    return resolved


def producer_metadata(url):
    """Never substitute the latest attempt or curve owner for the measured producer."""
    match = PRODUCER_URL.fullmatch(url or "")
    if match is None:
        return {}
    run_id, attempt = match.groups()
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
        headers["Authorization"] = "Bearer " + token
    endpoint = f"https://api.github.com/repos/SemiAnalysisAI/InferenceX/actions/runs/{run_id}/attempts/{attempt}"
    for retry in range(3):
        delay = 2**retry
        try:
            response = requests.get(endpoint, headers=headers, timeout=60)
        except requests.RequestException:
            if retry == 2:
                raise
        else:
            if retry == 2:
                break
            if response.status_code in {403, 429} and "Retry-After" in response.headers:
                value = response.headers["Retry-After"]
                try:
                    seconds = int(value) if value.isdigit() else parsedate_to_datetime(value).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    response.raise_for_status()
                # Fail instead of retrying before a long server-requested delay.
                if seconds > 60:
                    response.raise_for_status()
                delay = max(delay, seconds)
            elif not 500 <= response.status_code < 600:
                break
            response.close()
        time.sleep(delay)
    if response.status_code in {404, 410}:
        return {"github_run_id": run_id, "run_attempt": int(attempt), "html_url": url}
    response.raise_for_status()
    run = response.json()
    if str(run.get("id")) != run_id or run.get("run_attempt") != int(attempt):
        raise ValueError("producer workflow attempt identity mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha", "")):
        raise ValueError("producer workflow has no immutable source SHA")
    return {
        "github_run_id": run_id,
        "run_attempt": int(attempt),
        "html_url": url,
        **{key: run.get(key) for key in ("head_sha", "head_branch", "run_started_at", "status", "conclusion")},
    }


def to_tables(records, runs):
    _, e2e = _library()
    configs, benches, workflows = {}, {}, {}
    for raw in records:
        if not isinstance(raw, dict) or not re.fullmatch(r"[0-9]+", str(raw.get("id", ""))):
            raise ValueError("invalid HF benchmark identity")
        config = {key: raw[key] for key in e2e.INFERENCEX_CONFIG_FIELDS}
        config.update({key: raw[key] for key in ("offload_mode", "workers") if key in raw})
        # Full hashes avoid inventing or colliding with numeric upstream config IDs.
        config_id = "hf:" + hashlib.sha256(canonical(config)).hexdigest()
        config = {"id": config_id, **config}
        if configs.setdefault(config_id, config) != config:
            raise ValueError("conflicting HF configuration identity")
        benchmark_id = int(raw["id"])
        url = raw.get("run_url")
        run_id = url if PRODUCER_URL.fullmatch(url or "") else f"unknown-producer:{benchmark_id}"
        run = {"id": run_id, **runs.get(url, {})}
        if workflows.setdefault(run_id, run) != run:
            raise ValueError("conflicting HF producer identity")
        bench = {
            **{key: raw[key] for key in ("date", "isl", "osl", "conc", "metrics", "image", "benchmark_type")},
            "id": benchmark_id,
            "config_id": config_id,
            "workflow_run_id": run_id,
            "recipe_fingerprint": raw.get("recipe_fingerprint"),
        }
        if benchmark_id in benches:
            raise ValueError("duplicate HF benchmark identity")
        benches[benchmark_id] = bench
    if not benches:
        raise ValueError("HF cache contains no observations")
    return {
        "configs": [configs[key] for key in sorted(configs)],
        "workflow_runs": [workflows[key] for key in sorted(workflows)],
        "benchmark_results": [benches[key] for key in sorted(benches)],
    }


def _legacy_records(manifest, cache, transport):
    """Verified raw records and snapshot ID from the source-index layout."""
    sha = manifest["revision"]
    index = json.loads(_get(cache, transport, sha, INDEX_PATH).read_bytes())
    if index.get("schema_version") != "e2e-source-index/v1" or index.get("source_kind") != "inferencex":
        raise ValueError("unsupported HF source index")
    snapshot_ref = index["snapshot"]
    snapshot = json.loads(_get(cache, transport, sha, snapshot_ref["path"], snapshot_ref).read_bytes())
    if (
        snapshot.get("schema_version") != "inferencex-input-snapshot/v1"
        or snapshot.get("snapshot_id") != index["snapshot_id"]
        or snapshot.get("completeness") != "complete"
        or snapshot.get("partitions") != index["partitions"]
        or snapshot.get("source_scope", {}).get("kind") != "api-current-model-view"
    ):
        raise ValueError("HF snapshot is incomplete or inconsistent")
    records = []
    for partition in index["partitions"]:
        artifacts = [a for a in partition["artifacts"] if a["schema"] == "inferencex-observations-v1"]
        if len(artifacts) != 1:
            raise ValueError("HF partition requires exactly one observation artifact")
        entry = artifacts[0]
        # The library cache verifies size and SHA-256. Legacy files stamp ``schema_id`` rather than
        # the library's ``aisim.schema`` key, so their schema and row count are checked here.
        table = pq.ParquetFile(_get(cache, transport, sha, entry["path"], entry)).read()
        if (
            table.num_rows != entry["rows"]
            or (table.schema.metadata or {}).get(b"schema_id") != b"inferencex-observations-v1"
        ):
            raise ValueError("HF observation schema or row count mismatch")
        for row in table.to_pylist():
            raw = json.loads(row["raw_record_json"])
            if (
                hashlib.sha256(canonical(raw)).hexdigest() != row["record_sha256"]
                or row["observation_id"] != "inferencex:" + str(raw["id"])
                or any(
                    raw[key] != partition[field]
                    for key, field in (("model", "source_model"), ("hardware", "hardware"), ("framework", "framework"))
                )
            ):
                raise ValueError("HF observation identity mismatch")
            records.append(raw)
    if len(records) != snapshot["counts"]["observations_distinct"]:
        raise ValueError("HF snapshot observation count mismatch")
    return records, index["snapshot_id"]


def _unified_records(manifest, cache, transport):
    """Raw records and snapshot ID from the unified aisim-e2e/v1 catalog via the library loaders."""
    from pydantic import ValidationError

    fetch, e2e = _library()
    sha = manifest["revision"]
    dataset = fetch.DatasetIndex.model_validate_json(_get(cache, transport, sha, fetch.DATASET_INDEX_PATH).read_bytes())
    reference = dataset.categories.get("e2e")
    if reference is None:
        raise ValueError("HF dataset has no e2e category")
    ref = reference.index
    index_ref = {"bytes": ref.bytes, "sha256": ref.sha256, "format": ref.format, "schema": ref.artifact_schema}
    try:
        index = fetch.CategoryIndex.model_validate_json(_get(cache, transport, sha, ref.path, index_ref).read_bytes())
    except ValidationError as exc:
        raise ValueError("invalid HF e2e catalog index") from exc
    entries = [entry for entry in index.entries if entry.purpose == e2e.PURPOSE]
    if not entries:
        raise ValueError("HF e2e catalog contains no configurations")
    result = fetch.fetch(
        fetch.Selection(
            category="e2e",
            purpose=e2e.PURPOSE,
            configuration_ids=tuple(entry.configuration_id for entry in entries),
        ),
        repo=REPO_ID,
        revision=sha,
        transport=transport,
        cache=cache,
    )
    records, members = [], []
    for entry in entries:
        data = e2e.load_e2e(result.root, entry.manifest.path)
        for obs in data.observations:
            if obs.source_kind != "inferencex" or obs.raw_artifact_id is None:
                raise ValueError("HF e2e observation is not a raw InferenceX record")
        # Raw records come from the per-configuration bundle (each one hash-checked against its row).
        raw_records = e2e.read_raw_records(result.root, data.manifest, data.observations)
        for obs in data.observations:
            raw = json.loads(raw_records[obs.observation_id])
            if hashlib.sha256(
                canonical(raw)
            ).hexdigest() != obs.record_sha256 or obs.observation_id != "inferencex:" + str(raw["id"]):
                raise ValueError("HF observation identity mismatch")
            records.append(raw)
            members.append(
                e2e.SnapshotMember(
                    configuration_id=obs.configuration_id,
                    observation_id=obs.observation_id,
                    record_sha256=obs.record_sha256,
                    manifest_path=entry.manifest.path,
                )
            )
    return records, e2e.snapshot_id(members, SOURCE_SCOPE, manifest["selection_policy"])


def fetch(manifest, output, transport=None, cache=None):
    """Write ``tables.json`` plus ``dataset.json`` (repo, commit, snapshot ID) under ``output``."""
    validate_manifest(manifest)
    lib, _ = _library()
    transport = transport or default_transport()
    cache = cache or lib.Cache()
    layout = manifest.get("layout", LEGACY_LAYOUT)
    try:
        reader = _unified_records if layout == UNIFIED_LAYOUT else _legacy_records
        records, snapshot_id = reader(manifest, cache, transport)
    except lib.DataFetchError as exc:
        raise ValueError(f"HF measurement fetch failed: {exc}") from exc
    if manifest.get("snapshot_id", snapshot_id) != snapshot_id:
        raise ValueError("HF snapshot ID does not match the pinned snapshot")
    urls = sorted({r["run_url"] for r in records if PRODUCER_URL.fullmatch(r.get("run_url") or "")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = dict(zip(urls, pool.map(producer_metadata, urls), strict=True))
    tables = to_tables(records, runs)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "tables.json"
    target.write_bytes(canonical(tables) + b"\n")
    source = {"repo_id": REPO_ID, "revision": manifest["revision"], "snapshot_id": snapshot_id, "layout": layout}
    (output / DATASET_FILE).write_bytes(canonical(source) + b"\n")
    return target


def read_dataset(tables_path):
    """The dataset provenance written next to ``tables.json``, or ``None`` for older inputs."""
    path = Path(tables_path).with_name(DATASET_FILE)
    return json.loads(path.read_text()) if path.exists() else None
