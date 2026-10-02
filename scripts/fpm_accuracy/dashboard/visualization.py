# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from AISim FPM Gym f934c030afc3a03cb04d8f3ff4709194f7445c98.
# See ../README.md for upstream paths and modifications.
"""Static, predictor-independent 3D points from verified measurement cases.

The score equations are independently implemented from the definitions in
AISimulate 99d6acb722bf75e2b16119c59c79e1bad73b4efd, model.rs (Apache-2.0):
https://github.com/ai-dynamo/aisimulate/blob/99d6acb722bf75e2b16119c59c79e1bad73b4efd/crates/core/src/perfmodel/fpm/model.rs
Unit feature weights are intentional. Workload selection uses Gym's native
ForwardPassIteration.representative_rank, never the attention-score maximum.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from fpm_accuracy.dashboard.visualization_diagnostics import diagnostic_observations
from fpm_accuracy.hf.models import MeasurementCase, MeasurementObservation

POLICY = "native-online-rank-unit-features-v1"
FEATURE_REVISION = "99d6acb722bf75e2b16119c59c79e1bad73b4efd"
FIELDS = ("points", "rank_details", "axis_values", "iteration_ids")
AXES = [
    dict(
        id="attention",
        label="Attn Score",
        short="Attn<br>Score",
        point_index=0,
        definition="max_dp_rank(attn_score)",
        unit="score",
        rank_columns=[2],
    ),
    dict(
        id="moe",
        label="MOE Score",
        short="MOE<br>Score",
        point_index=1,
        definition="sum_dp_rank(prefill_tokens + decode_batch_size)",
        unit="tokens",
        reduction="sum",
        rank_columns=[3],
    ),
    dict(
        id="batch",
        label="Max batch size",
        short="Max batch<br>size",
        feature_index=0,
        definition="max_dp_rank(prefill_requests + decode_batch_size)",
        unit="requests",
        rank_columns=[5, 8],
    ),
    dict(
        id="prefill",
        label="Max prefill tokens",
        short="Max prefill<br>tokens",
        feature_index=1,
        definition="max_dp_rank(prefill_tokens)",
        unit="tokens",
        rank_columns=[6],
    ),
    dict(
        id="decode_batch",
        label="Max decode batch size",
        short="Max decode<br>batch size",
        feature_index=2,
        definition="max_dp_rank(decode_batch_size)",
        unit="requests",
        rank_columns=[8],
    ),
    dict(
        id="prefill_kv",
        label="Max prefill KV read",
        short="Max prefill<br>KV read",
        feature_index=3,
        definition="max_dp_rank(prefill_kv_read)",
        unit="KV tokens",
        rank_columns=[7],
    ),
    dict(
        id="decode_kv",
        label="Max decode KV read",
        short="Max decode<br>KV read",
        feature_index=4,
        definition="max_dp_rank(decode_kv_read)",
        unit="KV tokens",
        rank_columns=[9],
    ),
]


def _encoded(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def point(observation: MeasurementObservation) -> dict:
    """Keep all ranks, independent axis contributors, one ID and one latency."""
    iteration = observation.iteration
    rows = []
    for rank in iteration.ranks:
        s = rank.scheduled
        n, p, h, b, k = (
            s.num_prefill_requests,
            s.sum_prefill_tokens,
            s.sum_prefill_kv_tokens,
            s.num_decode_requests,
            s.sum_decode_kv_tokens,
        )
        attention = k + (h + h * p / n + p * p / (2 * n) + p / 2 if p > 0 else 0)
        rows.append(
            [
                rank.dp_rank,
                str(rank.workload_kind),
                attention,
                p + b,
                rank.wall_time_s * 1000,
                n,
                p,
                h,
                b,
                k,
                rank.extra_metadata.get("hf_source_row", observation.source_row),
            ]
        )
    a = max(row[2] for row in rows)
    latency = observation.actual_ms
    coordinates = [
        a,
        sum(row[3] for row in rows),
        latency,
        iteration.ranks[0].counter_id,
        sum(row[5] for row in rows),
        sum(row[8] for row in rows),
    ]
    if not all(math.isfinite(v) for v in coordinates):
        raise ValueError("non-finite visualization coordinate")
    return dict(
        points=coordinates,
        rank_details=[
            [row[0] for row in rows if row[2] == a],
            [row[0] for row in rows if row[4] == latency],
            iteration.representative_rank.dp_rank,
            rows,
        ],
        axis_values=[max(row[5] + row[8] for row in rows), *(max(row[k] for row in rows) for k in (6, 8, 7, 9))],
        iteration_ids=observation.observation_id,
    )


class VisualizationWriter:
    """Write one configuration at a time; publish the catalog only on success."""

    def __init__(
        self, directory: Path, *, repo_id: str, revision: str, sample_size: int = 1000, chunk_size: int = 20000
    ):
        if sample_size < 1 or chunk_size < 1:
            raise ValueError("sample and chunk sizes must be positive")
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError("visualization output must be new or empty")
        self.repo_id, self.revision = repo_id, revision
        self.sample_size, self.chunk_size = sample_size, chunk_size
        self.catalog, self.groups, self.files = [], [], {}

    def _asset(self, value: object, *, compressed: bool = False) -> str:
        data = _encoded(value)
        if compressed:
            data = gzip.compress(data, compresslevel=6, mtime=0)
        digest = hashlib.sha256(data).hexdigest()
        name = digest + (".json.gz" if compressed else ".json")
        (self.directory / name).write_bytes(data)
        self.files[name] = digest
        return name

    def add_case(self, case: MeasurementCase) -> None:
        snapshot = case.configuration
        key = _digest([snapshot.configuration_id, snapshot.snapshot_id])
        if any(c["id"] == key for c in self.catalog):
            raise ValueError("duplicate visualization configuration")
        grouped = defaultdict(list)
        diagnostic, diagnostic_issues = diagnostic_observations(case)
        seen = set()
        for observation in (*case.observations, *diagnostic):
            if observation.observation_id in seen:
                raise ValueError("duplicate visualization observation")
            seen.add(observation.observation_id)
            first = observation.iteration.ranks[0]
            # File identity keeps independent runs with reused worker/counter IDs apart.
            worker = _digest([observation.source_file_id, first.worker_id, first.stream_id])
            grouped[(worker, str(observation.workload_kind), observation.source_file_id, first.worker_id)].append(
                point(observation)
            )
        sources = {f.measurement_file_id: f for f in case.truth_files}
        group_ids = []
        for (worker, phase, file_id, worker_name), points in sorted(grouped.items()):
            group_id = _digest([key, worker, phase, POLICY])
            group_ids.append(group_id)
            # Stable hash sampling plus extremes for every axis. Selection never
            # depends on UI choices and is reused when workload pools are combined.
            chosen = set(
                sorted(range(len(points)), key=lambda i: _digest(points[i]["iteration_ids"]))[: self.sample_size]
            )
            for field, columns in (("points", range(3)), ("axis_values", range(5))):
                for column in columns:
                    chosen.add(min(range(len(points)), key=lambda i: points[i][field][column]))
                    chosen.add(max(range(len(points)), key=lambda i: points[i][field][column]))
            sample_indices = sorted(chosen)
            sample = {"sample_" + field: [points[i][field] for i in sample_indices] for field in FIELDS}
            chunks = [
                self._asset(
                    {f: [p[f] for p in points[start : start + self.chunk_size]] for f in FIELDS}, compressed=True
                )
                for start in range(0, len(points), self.chunk_size)
            ]
            latencies = sorted(p["points"][2] for p in points)
            zoom = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))] * 1.08
            source = sources[file_id]
            self.groups.append(
                dict(
                    id=group_id,
                    worker=worker,
                    phase=phase,
                    n=len(points),
                    role=case.worker_role,
                    label=f"{Path(source.path).name} · {worker_name}",
                    source=source.provenance_url,
                    source_sha256=source.sha256,
                    ranges=[
                        [min(p["points"][k] for p in points), max(p["points"][k] for p in points)] for k in range(3)
                    ],
                    axis_ranges=[
                        [min(p["axis_values"][k] for p in points), max(p["axis_values"][k] for p in points)]
                        for k in range(5)
                    ],
                    zoom=zoom,
                    above_zoom=sum(v > zoom for v in latencies),
                    sample_indices=sample_indices,
                    sample_file=self._asset(sample),
                    all_files=chunks,
                )
            )
        status = "diagnostic" if diagnostic else "ready" if grouped else "no_truth"
        reason = "; ".join(case.warnings) or "No accepted measurement iterations are declared."
        note = (
            "Exploratory grouping by source file, worker and counter; exactly one record per expected DP rank. "
            "Cross-rank synchronization is unverified. Recorded rank latencies are preserved. "
            "These points are excluded from evaluation."
        )
        self.catalog.append(
            dict(
                id=key,
                configuration_id=snapshot.configuration_id,
                snapshot_id=snapshot.snapshot_id,
                model=snapshot.model_id,
                snapshot=snapshot.snapshot_status,
                label=" · ".join(
                    [
                        snapshot.gpu_family,
                        snapshot.framework + " " + snapshot.framework_version,
                        snapshot.parallelism,
                        snapshot.snapshot_id,
                        snapshot.snapshot_status.title(),
                    ]
                ),
                source=snapshot.manifest_provenance_url,
                groups=group_ids,
                availability=status,
                reason=reason,
                diagnostic_note=note if diagnostic else "",
                measured=len(case.observations),
                diagnostic_count=len(diagnostic),
                parser_policy_id=case.parser_policy_id,
                membership_digest=case.measurement_membership_sha256,
                case_id=case.case_id,
                protocol_id=case.protocol_id,
                coordinate_scope="reduced_record" if case.protocol_id == "forward-pass-record-v1" else "dp_iteration",
                warnings=list(case.warnings),
                issues=[asdict(issue) for issue in case.issues],
                diagnostic_issues=diagnostic_issues,
                provenance=[
                    dict(path=f.path, sha256=f.sha256, url=f.provenance_url)
                    for f in (*case.truth_files, *case.helper_files)
                ],
                override_sha256=case.override_sha256,
                override_effects=case.override_effects,
            )
        )

    def finish(self) -> dict:
        available = [c for c in self.catalog if c["groups"]] or self.catalog
        defaults = {}
        if available:
            left = available[0]
            right = next((c for c in available if c["model"] != left["model"]), left)
            for pane, config in (("left", left), ("right", right)):
                group = next((g for g in self.groups if g["id"] in config["groups"]), None)
                defaults[pane] = dict(
                    model=config["model"], configuration=config["id"], worker=group["worker"] if group else ""
                )
        catalog = dict(
            schema_version=1,
            policy=POLICY,
            source_feature_commit=FEATURE_REVISION,
            repo_id=self.repo_id,
            hf_revision=self.revision,
            pointsRevision=self.revision,
            catalogRevision=self.revision,
            axes=AXES,
            catalog=self.catalog,
            groups=self.groups,
            defaults=defaults,
            full=True,
            sampling="stable identity hash plus each axis and latency extremes per worker/workload",
        )
        content = _encoded(catalog)
        (self.directory / "catalog.json").write_bytes(content)
        self.files["catalog.json"] = hashlib.sha256(content).hexdigest()
        manifest = dict(
            schema_version=1,
            policy=POLICY,
            hf_revision=self.revision,
            repo_id=self.repo_id,
            files=self.files,
            observations=sum(g["n"] for g in self.groups),
        )
        (self.directory / "manifest.json").write_bytes(_encoded(manifest))
        return manifest
