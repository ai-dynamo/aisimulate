# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InferenceX REST ingestion adapter: acquisition cache -> immutable snapshot (``aisim-e2e/v1``).

This is an *ingestion source*, not an evaluator input. Evaluators read a frozen
snapshot through :func:`aisim_core.data.e2e.load_e2e` /
:func:`~aisim_core.data.e2e.read_snapshot` and never import this module.

Layers
------
1. **Client** (:class:`InfxClient`): documented public GET endpoints
   (``views/options``, ``availability``, ``benchmarks?exact=true``,
   ``workflow-info``) with bounded retries for 429/5xx/transport errors/truncated
   bodies, honoring ``Retry-After`` (capped; a longer request fails rather than
   waits). The transport and sleep are injected, so tests use fakes.
2. **Acquisition cache** (:class:`AcquisitionCache`): mutable, raw bytes + body
   hash + ETag per URL; conditional requests make an unchanged refresh cheap.
3. **Snapshot** (:func:`write_candidate`): converts through
   :func:`inferencex_to_configurations`, writes stable shard paths
   (``e2e/<model>/.../<configuration-id>``) and an immutable
   ``snapshots/<snapshot_id>/`` document. Corrections keep ``observation_id`` and
   change ``record_sha256``, which changes the snapshot ID; old snapshot
   documents are never rewritten.

Acquisition fails closed: a required page that is incomplete (exhausted retries,
truncated or non-JSON body, configurations declared by ``availability`` but absent
from ``benchmarks``) or contradictory raises, so no snapshot is produced. A valid
empty response (``[]`` where nothing is declared) is a distinct, non-failure outcome.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from aisim_core.data.e2e import (
    INFERENCEX_CONFIG_FIELDS,
    SnapshotAcquisition,
    SnapshotMember,
    configuration_dir,
    inferencex_to_configurations,
    snapshot_id,
    write_configuration,
    write_snapshot,
)

BASE_URL = "https://inferencex.semianalysis.com/api/v1/"
USER_AGENT = "aisim-infx-ingest/1"
SELECTION_POLICY = "inferencex-all-observations-in-scope/v1"
REQUIRED_ROW_KEYS = (*INFERENCEX_CONFIG_FIELDS, "id", "metrics")
# ``exact=true&date=D`` returns the logical snapshot as of D, so one benchmark id can reappear on later
# dates. Only these per-view context fields differ between appearances (verified against the live API);
# they are not measured content and are excluded from the stored record and from contradiction checks.
VIEW_CONTEXT_PREFIX = "curve_"
_AVAILABILITY_KEY = ("model", "hardware", "framework", "precision", "spec_method", "disagg", "benchmark_type")


class InfxError(Exception):
    """Base class; any subclass means the snapshot must not be produced."""


class TransientExhaustedError(InfxError):
    """Retries for a 429/5xx/transport failure were used up."""


class RateLimitedError(InfxError):
    """``Retry-After`` exceeds the cap or the request budget is exhausted."""


class HttpStatusError(InfxError):
    """A non-retryable status (for example 400/404)."""


class IncompleteAcquisitionError(InfxError):
    """A required page is truncated, malformed or missing declared configurations."""


class ContradictionError(InfxError):
    """Sources disagree (duplicate ids with different content, wrong model, suspicious empty)."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # lower-case names
    body: bytes


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """Stdlib transport; HTTP error statuses are responses, network failures raise ``OSError``."""
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, {k.lower(): v for k, v in response.headers.items()}, response.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read())
    except urllib.error.URLError as exc:
        raise ConnectionError(str(exc.reason)) from exc


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    max_retry_after: float = 120.0
    timeout: float = 60.0
    max_requests: int = 5000  # budget across a run, retries included


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Seconds from a ``Retry-After`` value (delta-seconds or HTTP-date); ``None`` if absent or invalid."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


@dataclass(frozen=True)
class Fetched:
    url: str
    status: int  # 200, or 304 when the cached body is still current
    body: bytes
    etag: str | None
    attempts: int


class InfxClient:
    def __init__(
        self,
        transport: Transport = urllib_transport,
        policy: RetryPolicy = RetryPolicy(),  # noqa: B008 - frozen
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = BASE_URL,
    ):
        self.transport, self.policy, self.sleep, self.base_url = transport, policy, sleep, base_url
        self.requests_made = 0

    def url(self, route: str, params: Mapping[str, Any] | None = None) -> str:
        query = urlencode(sorted((params or {}).items()))
        return self.base_url + route + (f"?{query}" if query else "")

    def get(self, route: str, params: Mapping[str, Any] | None = None, *, etag: str | None = None) -> Fetched:
        url = self.url(route, params)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "X-InferenceX-Traffic": "ci"}
        if etag:
            headers["If-None-Match"] = etag
        last = "no attempt"
        for attempt in range(1, self.policy.max_attempts + 1):
            if self.requests_made >= self.policy.max_requests:
                raise RateLimitedError(f"request budget of {self.policy.max_requests} exhausted before {url}")
            self.requests_made += 1
            delay = min(self.policy.max_delay, self.policy.base_delay * 2 ** (attempt - 1))
            try:
                response = self.transport(url, headers, self.policy.timeout)
            except (OSError, TimeoutError) as exc:
                last = f"transport error: {exc}"
            else:
                status = response.status
                if status in (200, 304):
                    declared = response.headers.get("content-length")
                    if (
                        status == 200
                        and declared is not None
                        and declared.isdigit()
                        and int(declared) != len(response.body)
                    ):
                        last = f"truncated body ({len(response.body)} of {declared} bytes)"
                    else:
                        return Fetched(url, status, response.body, response.headers.get("etag"), attempt)
                elif status == 429 or 500 <= status < 600:
                    last = f"HTTP {status}"
                    asked = parse_retry_after(response.headers.get("retry-after"))
                    if asked is not None:
                        if asked > self.policy.max_retry_after:
                            raise RateLimitedError(
                                f"{url}: Retry-After {asked:.0f}s exceeds the {self.policy.max_retry_after:.0f}s cap"
                            )
                        delay = asked
                else:
                    raise HttpStatusError(f"{url}: HTTP {status}")
            if attempt < self.policy.max_attempts:
                self.sleep(delay)
        raise TransientExhaustedError(f"{url}: {self.policy.max_attempts} attempts failed ({last})")


# --------------------------------------------------------------------------- acquisition cache


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AcquisitionCache:
    """Mutable cache of raw response bytes keyed by request URL; never part of a snapshot."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = _sha(url.encode())
        return self.root / key[:2] / f"{key}.body", self.root / key[:2] / f"{key}.meta.json"

    def get(self, url: str) -> tuple[dict[str, Any], bytes] | None:
        body_path, meta_path = self._paths(url)
        if not (body_path.exists() and meta_path.exists()):
            return None
        meta, body = json.loads(meta_path.read_text()), body_path.read_bytes()
        if meta.get("body_sha256") != _sha(body):  # corrupt entry: refetch instead of trusting it
            return None
        return meta, body

    def put(self, url: str, body: bytes, etag: str | None, fetched_at: str) -> dict[str, Any]:
        body_path, meta_path = self._paths(url)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"url": url, "etag": etag, "body_sha256": _sha(body), "bytes": len(body), "fetched_at": fetched_at}
        for path, data in ((body_path, body), (meta_path, (json.dumps(meta, sort_keys=True) + "\n").encode())):
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                handle.write(data)
            os.replace(handle.name, path)  # body first, meta last: a torn write fails the hash check
        return meta


# --------------------------------------------------------------------------- acquisition


@dataclass
class Acquisition:
    records: list[dict[str, Any]]
    runs: dict[str, dict[str, Any]]
    scope: dict[str, Any]
    acquisitions: list[SnapshotAcquisition]
    empty_scope: bool = False  # valid-empty: nothing declared for the scope (not a failure)
    undeclared: list[str] = field(default_factory=list)  # benchmark rows availability did not list (warning)
    requests_made: int = 0


def _json(fetched: Fetched, body: bytes, expect: type) -> Any:
    try:
        value = json.loads(body)
    except ValueError as exc:
        raise IncompleteAcquisitionError(f"{fetched.url}: body is not valid JSON ({exc})") from exc
    if not isinstance(value, expect):
        raise IncompleteAcquisitionError(f"{fetched.url}: expected {expect.__name__}, got {type(value).__name__}")
    return value


class Acquirer:
    def __init__(self, client: InfxClient, cache: AcquisitionCache, now: Callable[[], datetime] | None = None):
        self.client, self.cache = client, cache
        self.now = now or (lambda: datetime.now(UTC))

    def _page(self, route: str, params: Mapping[str, Any] | None, expect: type) -> tuple[Any, str, str]:
        """Fetch with a conditional request; returns (value, acquired_at, status 'acquired'|'inherited')."""
        url = self.client.url(route, params)
        cached = self.cache.get(url)
        fetched = self.client.get(route, params, etag=cached[0].get("etag") if cached else None)
        if fetched.status == 304:
            if cached is None:
                raise IncompleteAcquisitionError(f"{url}: 304 without a cached body")
            return _json(fetched, cached[1], expect), cached[0]["fetched_at"], "inherited"
        value = _json(fetched, fetched.body, expect)  # validate before it can enter the cache
        stamp = _utc(self.now())
        self.cache.put(url, fetched.body, fetched.etag, stamp)
        return value, stamp, "acquired"

    def acquire(self, models: Sequence[str] | None = None, prior_observation_ids: Sequence[str] = ()) -> Acquisition:
        options, _, _ = self._page("views/options", None, dict)
        by_key: dict[str, str] = {}
        for entry in options.get("models", []):
            for key in entry.get("dbKeys", []):
                by_key[key] = entry["name"]
        availability, _, _ = self._page("availability", None, list)
        wanted = set(models) if models else None
        declared: dict[tuple[str, str], set[tuple]] = {}
        for row in availability:
            if not isinstance(row, dict) or not {*_AVAILABILITY_KEY, "date"} <= row.keys():
                raise IncompleteAcquisitionError(f"availability row lacks required fields: {row!r:.200}")
            if wanted is not None and row["model"] not in wanted and by_key.get(row["model"]) not in wanted:
                continue
            if row["model"] not in by_key:
                raise IncompleteAcquisitionError(f"availability model {row['model']!r} is not in views/options")
            declared.setdefault((by_key[row["model"]], row["date"]), set()).add(
                tuple(row[k] for k in _AVAILABILITY_KEY)
            )
        scope = {"source": "inferencex", "models": sorted(wanted) if wanted else "all"}
        if not declared:
            if prior_observation_ids:
                raise ContradictionError(
                    f"empty availability for scope but {len(prior_observation_ids)} observations were ingested before"
                )
            return Acquisition([], {}, scope, [], empty_scope=True, requests_made=self.client.requests_made)

        records: dict[str, dict[str, Any]] = {}
        acquired: dict[str, SnapshotAcquisition] = {}
        undeclared: list[str] = []
        for (name, date), expected in sorted(declared.items()):
            rows, stamp, status = self._page("benchmarks", {"model": name, "date": date, "exact": "true"}, list)
            url = self.client.url("benchmarks", {"model": name, "date": date, "exact": "true"})
            valid_keys = {k for k, n in by_key.items() if n == name}
            seen: set[tuple] = set()
            for row in rows:
                if not isinstance(row, dict) or any(k not in row for k in REQUIRED_ROW_KEYS):
                    raise IncompleteAcquisitionError(f"{url}: row lacks required fields: {row!r:.200}")
                if row["model"] not in valid_keys:
                    raise ContradictionError(f"{url}: row {row['id']} is for model {row['model']!r}, not {name!r}")
                row = {k: v for k, v in row.items() if not k.startswith(VIEW_CONTEXT_PREFIX)}
                rid = str(row["id"])
                if rid in records and _canonical(records[rid]) != _canonical(row):
                    raise ContradictionError(f"{url}: benchmark id {rid} has two different bodies in one acquisition")
                records[rid] = row
                acquired[rid] = SnapshotAcquisition(
                    observation_id=f"inferencex:{rid}", acquired_at=stamp, source_uri=url, status=status
                )
                seen.add(tuple(row[k] for k in _AVAILABILITY_KEY))
            missing = expected - seen
            if missing:
                raise IncompleteAcquisitionError(
                    f"{url}: availability declares {len(missing)} configuration(s) absent from benchmarks "
                    f"(returned {len(rows)} rows), for example {sorted(map(str, missing))[0]}"
                )
            if seen - expected:  # availability is a cached view and may lag benchmarks: warn, do not fail
                undeclared.append(f"{name}/{date}/{sorted(map(str, seen - expected))[0]}")

        runs = self._runs(sorted({date for _, date in declared}))
        return Acquisition(
            records=[records[k] for k in sorted(records, key=int)],
            runs=runs,
            scope=scope,
            acquisitions=[acquired[k] for k in sorted(acquired, key=int)],
            undeclared=undeclared,
            requests_made=self.client.requests_made,
        )

    def _runs(self, dates: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Producer metadata keyed by run URL; a run whose ``head_sha`` differs between pages is a contradiction."""
        runs: dict[str, dict[str, Any]] = {}
        for date in dates:
            info, _, _ = self._page("workflow-info", {"date": date}, dict)
            for run in info.get("runs", []):
                runs.setdefault(run["html_url"], {})["run_attempt"] = run.get("run_attempt")
            for config in info.get("runConfigs", []):
                entry = runs.setdefault(config["html_url"], {})
                sha = config.get("head_sha")
                if sha and entry.get("head_sha") not in (None, sha):
                    raise ContradictionError(f"workflow-info {date}: run {config['html_url']} reports two head_sha")
                if sha:
                    entry["head_sha"] = sha
        return runs


# --------------------------------------------------------------------------- snapshot layer


@dataclass(frozen=True)
class SnapshotPlan:
    """The snapshot an acquisition would produce, computed without touching disk."""

    snapshot_id: str
    members: list[SnapshotMember]
    configurations: list[Any]


def plan_snapshot(acq: Acquisition) -> SnapshotPlan:
    configurations = inferencex_to_configurations(acq.records, acq.runs)
    members = [
        SnapshotMember(
            configuration_id=o.configuration_id,
            observation_id=o.observation_id,
            record_sha256=o.record_sha256,
            manifest_path=f"{configuration_dir(data.body)}/manifest.json",
        )
        for data in configurations
        for o in data.observations
    ]
    return SnapshotPlan(snapshot_id(members, acq.scope, SELECTION_POLICY), members, configurations)


def snapshot_path(sid: str) -> str:
    return f"snapshots/{sid}/snapshot.json"


def write_candidate(root: Path, acq: Acquisition, plan: SnapshotPlan) -> str:
    """Write shards and the immutable snapshot document; returns its path. An existing snapshot is never rewritten."""
    root = Path(root)
    path = snapshot_path(plan.snapshot_id)
    if (root / path).exists():
        return path
    for data in plan.configurations:
        write_configuration(root, data)
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    snapshot = write_snapshot(
        root,
        path,
        plan.members,
        source_scope=acq.scope,
        selection_policy=SELECTION_POLICY,
        completeness="complete",
        acquisition=acq.acquisitions,
    )
    assert snapshot.snapshot_id == plan.snapshot_id
    return path


def diff_members(before: Mapping[str, str], after: Sequence[SnapshotMember]) -> dict[str, list[str]]:
    """Change summary against ``{observation_id: record_sha256}``: added, corrected, removed."""
    now = {m.observation_id: m.record_sha256 for m in after}
    return {
        "added": sorted(set(now) - set(before)),
        "corrected": sorted(o for o in now if o in before and now[o] != before[o]),
        "removed": sorted(set(before) - set(now)),
    }
