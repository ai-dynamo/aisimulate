# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive InferenceX API evidence as immutable, selectively downloadable files.

This intake layer does not select an accuracy cohort or certify Replay readiness.
It retains full source records so normalization can evolve without reacquisition.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlencode

import pyarrow as pa
import pyarrow.parquet as pq
import requests

API_ORIGIN = "https://inferencex.semianalysis.com/api/v1"
NORMALIZER_VERSION = "inferencex-intake-v1"
INDEX_PATH = "catalog/e2e/inferencex.json"
LATENCIES = {"ttft": "ttft", "tpot": "tpot", "itl": "itl", "e2el": "e2e_latency"}
THROUGHPUTS = {
    "input_tput_per_gpu": "input_throughput",
    "output_tput_per_gpu": "output_throughput",
    "tput_per_gpu": "total_throughput",
}
# A source-intake contract, deliberately separate from the DEP's Replay-ready v2
# configuration/observation contract. Unknown effective settings remain in raw data.
OBSERVATIONS = pa.schema(
    [
        ("observation_id", pa.string()),
        ("record_sha256", pa.string()),
        ("source_kind", pa.string()),
        ("source_benchmark_id", pa.string()),
        ("source_model", pa.string()),
        ("display_model", pa.string()),
        ("hardware", pa.string()),
        ("framework", pa.string()),
        ("measurement_date", pa.string()),
        ("producer_run_url", pa.string()),
        ("source_config_id", pa.string()),
        ("raw_record_json", pa.string()),
    ],
    metadata={b"schema_id": b"inferencex-observations-v1"},
)
METRICS = pa.schema(
    [
        ("observation_id", pa.string()),
        ("record_sha256", pa.string()),
        ("name", pa.string()),
        ("statistic", pa.string()),
        ("percentile", pa.float64()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("normalization", pa.string()),
        ("timing_boundary", pa.string()),
        ("status", pa.string()),
        ("reason", pa.string()),
        ("source_metric", pa.string()),
    ],
    metadata={b"schema_id": b"inferencex-metrics-v1"},
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"Unsafe artifact path: {relative}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes cache: {relative}")
    return target


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value) + b"\n"
    if path.exists() and path.read_bytes() == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def component(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Empty source identity")
    readable = re.sub(r"[^A-Za-z0-9._-]", "-", value)[:80].strip(".-") or "source"
    return readable + "-" + digest(value.encode())[:12]


@dataclass(frozen=True)
class Acquisition:
    url: str
    body: bytes
    retrieved_at: str
    etag: str | None = None
    last_modified: str | None = None

    def json(self) -> Any:
        return json.loads(self.body)


class InferenceXClient:
    """Bounded acquisition with a local conditional-response cache."""

    def __init__(self, cache: Path, session: requests.Session | None = None):
        self.cache = cache
        self.session = session or requests.Session()

    def get(self, endpoint: str, **parameters: Any) -> Acquisition:
        if endpoint not in {"views/options", "availability", "benchmarks", "workflow-info", "benchmarks/history"}:
            raise ValueError("Unsupported InferenceX endpoint")
        if "runId" in parameters and not re.fullmatch(r"[1-9][0-9]*", str(parameters["runId"])):
            raise ValueError("runId must be a positive integer")
        query = urlencode(sorted(parameters.items()))
        url = f"{API_ORIGIN}/{endpoint}" + (f"?{query}" if query else "")
        key = digest(url.encode())
        metadata = self.cache / f"{key}.json"
        payload = self.cache / f"{key}.body"
        previous = json.loads(metadata.read_bytes()) if metadata.exists() else {}
        headers = {}
        if payload.exists() and digest(payload.read_bytes()) == previous.get("body_sha256"):
            if previous.get("etag"):
                headers["If-None-Match"] = previous["etag"]
            elif previous.get("last_modified"):
                headers["If-Modified-Since"] = previous["last_modified"]
        for attempt in range(4):
            try:
                response = self.session.get(url, headers=headers, timeout=(10, 90))
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == 3:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After", "")
                delay = min(float(retry_after), 60) if retry_after.isdigit() else 2**attempt
                time.sleep(delay)
                continue
            break
        if response.status_code == 304:
            if not headers or not payload.exists():
                raise ValueError("304 without a verified cached response")
            body = payload.read_bytes()
            etag, modified = previous.get("etag"), previous.get("last_modified")
        else:
            response.raise_for_status()
            body = response.content
            if body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            json.loads(body)  # Reject errors/HTML before modifying the cache.
            etag = response.headers.get("ETag")
            modified = response.headers.get("Last-Modified")
        retrieved = datetime.now(UTC).isoformat()
        self.cache.mkdir(parents=True, exist_ok=True)
        temporary = payload.with_suffix(".tmp")
        temporary.write_bytes(body)
        temporary.replace(payload)
        write_json(metadata, {"url": url, "body_sha256": digest(body), "etag": etag, "last_modified": modified})
        return Acquisition(url, body, retrieved, etag, modified)


def normalize(model: str, rows: Any) -> tuple[list[dict], list[dict]]:
    """Preserve source facts; map only known metric names and documented units."""
    if not isinstance(rows, list):
        raise ValueError(f"Expected benchmark list for {model}")
    observations, metrics, seen = [], [], {}
    for raw in rows:
        if not isinstance(raw, dict) or not re.fullmatch(r"[0-9]+", str(raw.get("id", ""))):
            raise ValueError(f"Missing benchmark identity for {model}")
        if not isinstance(raw.get("metrics"), dict):
            raise ValueError(f"Missing metrics object for benchmark {raw['id']}")
        # Curve fields describe the queried logical snapshot, not the producer.
        source = {k: v for k, v in raw.items() if not k.startswith("curve_")}
        content_hash = digest(canonical(source))
        observation_id = f"inferencex:{raw['id']}"
        if observation_id in seen:
            if seen[observation_id] != content_hash:
                raise ValueError(f"Conflicting benchmark versions: {observation_id}")
            continue
        seen[observation_id] = content_hash
        for field in ("model", "hardware", "framework"):
            if not isinstance(raw.get(field), str) or not raw[field]:
                raise ValueError(f"Missing {field} for {observation_id}")
        observations.append(
            {
                "observation_id": observation_id,
                "record_sha256": content_hash,
                "source_kind": "inferencex",
                "source_benchmark_id": str(raw["id"]),
                "source_model": raw["model"],
                "display_model": model,
                "hardware": raw["hardware"],
                "framework": raw["framework"],
                "measurement_date": raw.get("date"),
                "producer_run_url": raw.get("run_url"),
                "source_config_id": str(raw["config_id"]) if raw.get("config_id") is not None else None,
                "raw_record_json": canonical(source).decode(),
            }
        )
        for key, value in sorted(raw["metrics"].items()):
            percentile = None
            if key in THROUGHPUTS:
                name, statistic, unit, normalization, scale = THROUGHPUTS[key], "rate", "tokens/s", "per_device", 1
            else:
                match = re.fullmatch(r"(mean|median|std|p[0-9]+(?:\.[0-9]+)?)_(ttft|tpot|itl|e2el)", key)
                if not match:
                    continue  # Unknown fields remain in the complete source record.
                stat, suffix = match.groups()
                name, unit, normalization, scale = LATENCIES[suffix], "ms", "none", 1000
                statistic = {"std": "stddev"}.get(stat, stat)
                if stat.startswith("p"):
                    statistic, percentile = "percentile", float(stat[1:])
                    if not 0 < percentile <= 100:
                        raise ValueError(f"Invalid percentile: {key}")
            valid = (
                isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
            )
            converted = float(value) * scale if valid else None
            if converted is not None and not math.isfinite(converted):
                raise ValueError(f"Metric conversion overflow: {observation_id}/{key}")
            metrics.append(
                {
                    "observation_id": observation_id,
                    "record_sha256": content_hash,
                    "name": name,
                    "statistic": statistic,
                    "percentile": percentile,
                    "value": converted,
                    "unit": unit,
                    "normalization": normalization,
                    "timing_boundary": "unknown",
                    "status": "valid" if valid else "missing" if value is None else "invalid",
                    "reason": "Source API does not declare timing boundary"
                    if valid
                    else "Source value missing or invalid",
                    "source_metric": key,
                }
            )
    return sorted(observations, key=lambda r: r["observation_id"]), sorted(
        metrics, key=lambda r: (r["observation_id"], r["source_metric"])
    )


def artifact(root: Path, relative: str, schema: str, rows: int | None = None) -> dict:
    path = safe_path(root, relative)
    result = {"path": relative, "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size, "schema": schema}
    if rows is not None:
        result["rows"] = rows
    return result


def write_table(root: Path, folder: str, role: str, records: list[dict], schema: pa.Schema) -> dict:
    identity = digest(canonical({"normalizer": NORMALIZER_VERSION, "role": role, "records": records}))
    relative = f"{folder}/{role}-{identity}.parquet"
    path = safe_path(root, relative)
    expected = pa.Table.from_pylist(records, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not pq.ParquetFile(path).read().equals(expected, check_metadata=True):
            raise ValueError(f"Existing immutable shard differs: {relative}")
    else:
        temporary = path.with_suffix(".tmp")
        pq.write_table(expected, temporary, compression="zstd")
        temporary.replace(path)
    return artifact(root, relative, schema.metadata[b"schema_id"].decode(), len(records))


def prepare_snapshot(
    root: Path, options: Acquisition, responses: dict[str, Acquisition], *, normalizer_revision: str, publication: dict
) -> dict:
    """Freeze an explicitly scoped intake; leave the previous index on failure.

    Data is partitioned by source model, hardware and framework. Updates reuse
    unchanged shards. Content-addressed files from failed preparations are safe
    to retry and are not visible until the discovery index is replaced.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", normalizer_revision):
        raise ValueError("Pin the normalizer to a full Git revision")
    if not responses:
        raise ValueError("Empty acquisition plan")
    discovered = options.json()
    if not isinstance(discovered, dict) or not isinstance(discovered.get("models"), list):
        raise ValueError("Invalid model discovery response")
    available = {entry["name"] for entry in discovered["models"]}
    if not responses.keys() <= available:
        raise ValueError("Acquired models do not match discovery")
    manifests, acquisitions, body_hashes = [], [], []
    returned = 0
    global_ids = {}
    for model, response in sorted(responses.items()):
        raw = response.json()
        observations, metrics = normalize(model, raw)
        returned += len(raw)
        for row in observations:
            oid = row["observation_id"]
            if oid in global_ids:
                raise ValueError(f"Overlapping source model acquisitions: {oid}")
            global_ids[oid] = row["record_sha256"]
        raw_hash = digest(response.body)
        raw_path = f"data/e2e/inferencex/responses/{raw_hash}.json.gz"
        target = safe_path(root, raw_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(gzip.decompress(target.read_bytes())) != raw_hash:
                raise ValueError("Corrupt response archive")
        else:
            target.write_bytes(gzip.compress(response.body, mtime=0))
        acquisition = {
            "model": model,
            "url": response.url,
            "retrieved_at": response.retrieved_at,
            "etag": response.etag,
            "last_modified": response.last_modified,
            "body_sha256": raw_hash,
            "artifact": artifact(root, raw_path, "inferencex-api-json/v1"),
        }
        acquisitions.append(acquisition)
        body_hashes.append([model, raw_hash])
        groups: dict[tuple, list[dict]] = {}
        metric_groups: dict[str, list[dict]] = {}
        for metric in metrics:
            metric_groups.setdefault(metric["observation_id"], []).append(metric)
        for row in observations:
            groups.setdefault((row["source_model"], row["hardware"], row["framework"]), []).append(row)
        for (source_model, hardware, framework), group in sorted(groups.items()):
            folder = "data/e2e/inferencex/" + "/".join(component(v) for v in (source_model, hardware, framework))
            artifacts = [
                write_table(root, folder, "observations", group, OBSERVATIONS),
                write_table(
                    root,
                    folder,
                    "metrics",
                    [m for r in group for m in metric_groups.get(r["observation_id"], [])],
                    METRICS,
                ),
            ]
            manifests.append(
                {
                    "display_model": model,
                    "source_model": source_model,
                    "hardware": hardware,
                    "framework": framework,
                    "artifacts": artifacts,
                }
            )
    model_map = discovered["models"]
    identity = {
        "normalizer_version": NORMALIZER_VERSION,
        "normalizer_code_sha256": digest(Path(__file__).read_bytes()),
        "model_mapping": model_map,
        "responses": body_hashes,
        "partitions": manifests,
        "publication": publication,
    }
    snapshot_id = digest(canonical(identity))
    relative = f"catalog/e2e/snapshots/{snapshot_id}.json"
    path = safe_path(root, relative)
    if not path.exists():
        write_json(
            path,
            {
                "schema_version": "inferencex-input-snapshot/v1",
                "snapshot_id": snapshot_id,
                "normalizer_revision": normalizer_revision,
                **identity,
                "acquisitions": acquisitions,
                "source_scope": {"kind": "api-current-model-view", "models": sorted(responses)},
                "selection_policy": "archive-all-returned-records-v1",
                "completeness": "complete",
                "counts": {
                    "acquisition_requests_planned": len(responses),
                    "acquisition_requests_completed": len(responses),
                    "records_returned": returned,
                    "observations_distinct": len(global_ids),
                },
                "qualification": "archival-only; cohort parity and effective-configuration resolution pending",
            },
        )
    else:
        old = json.loads(path.read_bytes())
        if any(old.get(k) != v for k, v in identity.items()):
            raise ValueError("Immutable snapshot identity conflict")
    index = {
        "schema_version": "e2e-source-index/v1",
        "source_kind": "inferencex",
        "snapshot": artifact(root, relative, "inferencex-input-snapshot/v1"),
        "snapshot_id": snapshot_id,
        "partitions": manifests,
    }
    validate_snapshot(root, index)
    category_path = root / "catalog/e2e/index.json"
    category = (
        json.loads(category_path.read_bytes())
        if category_path.exists()
        else {"schema_version": "e2e-index/v1", "sources": {}}
    )
    if category.get("schema_version") != "e2e-index/v1" or not isinstance(category.get("sources"), dict):
        raise ValueError("Unsupported E2E category index")
    write_json(root / INDEX_PATH, index)
    category["sources"]["inferencex"] = artifact(root, INDEX_PATH, "e2e-source-index/v1")
    write_json(category_path, category)
    return index


def validate_snapshot(root: Path, index: dict | None = None) -> None:
    index = index or json.loads((root / INDEX_PATH).read_bytes())
    snapshot = json.loads(safe_path(root, index["snapshot"]["path"]).read_bytes())
    if snapshot["snapshot_id"] != index["snapshot_id"] or snapshot["completeness"] != "complete":
        raise ValueError("Incomplete or inconsistent snapshot")
    if snapshot["partitions"] != index["partitions"]:
        raise ValueError("Snapshot/catalog partition mismatch")
    artifacts = [index["snapshot"], *(a["artifact"] for a in snapshot["acquisitions"])]
    identities = set()
    for partition in index["partitions"]:
        artifacts.extend(partition["artifacts"])
        by_schema = {a["schema"]: a for a in partition["artifacts"]}
        tables = {}
        for schema in (OBSERVATIONS, METRICS):
            entry = by_schema[schema.metadata[b"schema_id"].decode()]
            table = pq.ParquetFile(safe_path(root, entry["path"])).read()
            if not table.schema.equals(schema, check_metadata=True) or table.num_rows != entry["rows"]:
                raise ValueError("Parquet schema/row-count mismatch")
            tables[schema.metadata[b"schema_id"].decode()] = table.to_pylist()
        keys = set()
        for row in tables["inferencex-observations-v1"]:
            if row["observation_id"] in identities:
                raise ValueError("Duplicate observation")
            identities.add(row["observation_id"])
            keys.add((row["observation_id"], row["record_sha256"]))
            if digest(canonical(json.loads(row["raw_record_json"]))) != row["record_sha256"]:
                raise ValueError("Observation content hash mismatch")
        for metric in tables["inferencex-metrics-v1"]:
            if (metric["observation_id"], metric["record_sha256"]) not in keys:
                raise ValueError("Metric references an unknown observation version")
    for entry in artifacts:
        data = safe_path(root, entry["path"]).read_bytes()
        if digest(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise ValueError(f"Artifact integrity mismatch: {entry['path']}")
