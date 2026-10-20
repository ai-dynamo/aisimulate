# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read pinned private HF observations and resolve their exact producer attempts.

The API current view is a different cohort from a historical database dump.
Configuration IDs below are local content identities, never upstream database IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "ai-dynamo/aisim-infx-cache"
INDEX_PATH = "catalog/e2e/inferencex.json"
POLICY = "hf-current-view-config-v1"
MTP_POLICY = "hf-current-view-config-v2"
CONFIG_FIELDS = (
    "model",
    "hardware",
    "framework",
    "precision",
    "spec_method",
    "disagg",
    "is_multinode",
    "prefill_tp",
    "prefill_ep",
    "prefill_dp_attention",
    "prefill_num_workers",
    "decode_tp",
    "decode_ep",
    "decode_dp_attention",
    "decode_num_workers",
    "num_prefill_gpu",
    "num_decode_gpu",
)
PRODUCER_URL = re.compile(
    r"https://github\.com/SemiAnalysisAI/InferenceX/actions/runs/([1-9][0-9]*)/attempts/([1-9][0-9]*)"
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def validate_manifest(manifest):
    fields = {"schema_version", "repo_id", "revision", "release_tag", "selection_policy", "max_age_days"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
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


def resolve_latest_manifest(template):
    validate_manifest(template)
    info = HfApi().repo_info(REPO_ID, repo_type="dataset", revision="main")
    if not info.private:
        raise ValueError("InferenceX cache must remain private")
    resolved = {**template, "revision": info.sha, "release_tag": "hf/" + info.sha}
    validate_manifest(resolved)
    return resolved


def download(manifest, path, expected=None):
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in path
        or not path.startswith(("catalog/e2e/", "data/e2e/inferencex/"))
    ):
        raise ValueError("unsafe HF measurement artifact path")
    local = Path(hf_hub_download(REPO_ID, path, repo_type="dataset", revision=manifest["revision"]))
    if expected is not None:
        with local.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if local.stat().st_size != expected["bytes"] or checksum != expected["sha256"]:
            raise ValueError("HF measurement artifact checksum or size mismatch")
    return local


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
    configs, benches, workflows = {}, {}, {}
    for raw in records:
        if not isinstance(raw, dict) or not re.fullmatch(r"[0-9]+", str(raw.get("id", ""))):
            raise ValueError("invalid HF benchmark identity")
        config = {key: raw[key] for key in CONFIG_FIELDS}
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


def fetch(manifest, output):
    validate_manifest(manifest)
    index = json.loads(download(manifest, INDEX_PATH).read_bytes())
    if index.get("schema_version") != "e2e-source-index/v1" or index.get("source_kind") != "inferencex":
        raise ValueError("unsupported HF source index")
    snapshot_ref = index["snapshot"]
    snapshot = json.loads(download(manifest, snapshot_ref["path"], snapshot_ref).read_bytes())
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
        table = pq.ParquetFile(download(manifest, entry["path"], entry)).read()
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
    urls = sorted({r["run_url"] for r in records if PRODUCER_URL.fullmatch(r.get("run_url") or "")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = dict(zip(urls, pool.map(producer_metadata, urls), strict=True))
    tables = to_tables(records, runs)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "tables.json"
    target.write_bytes(canonical(tables) + b"\n")
    return target
