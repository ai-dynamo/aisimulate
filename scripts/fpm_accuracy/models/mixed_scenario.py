# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified original-A inputs for native mixed-workload composition.

This adapter stages immutable sources only. The canonical model owns eligibility,
interpolation, quality checks, calibration, selection, and all prediction routes.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.fpm_accuracy.contract import strict_json
from scripts.fpm_accuracy.exceptions import DataError, DependencyError
from scripts.fpm_accuracy.models.aic_fpm_database import prepare_aic_fpm_database


class MixedOriginalA:
    def __init__(self, engine_config: dict, artifact: Any, scenario: str):
        if scenario not in {"a", "ac"} or artifact is None:
            raise ValueError("Mixed composition requires A or A+C with an original-A artifact.")
        for path, digest in (
            (artifact.local_path, artifact.sha256),
            (artifact.local_metadata_path, artifact.metadata_sha256),
        ):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise DataError("Original-A source changed after dataset validation.")
        metadata = strict_json(Path(artifact.local_metadata_path).read_text())
        self.provenance = {
            "source_sha256": artifact.sha256,
            "metadata_sha256": artifact.metadata_sha256,
            "artifact_id": artifact.artifact_id,
            "sources": [
                {"path": str(Path(artifact.local_path).resolve()), "sha256": artifact.sha256},
                {"path": str(Path(artifact.local_metadata_path).resolve()), "sha256": artifact.metadata_sha256},
            ],
            "grouping": {
                "coordinate_system": metadata.get("coordinate_system"),
                "partition_policy": "balanced_v1",
                "rank_workload_reconstruction": "replicated_table_coordinates",
                "target_scope": "recorded_aggregate_latency",
            },
        }
        self.scenario = scenario
        self.artifact = artifact
        self.manifests: dict[str, Path] = {}
        self.prepared = prepare_aic_fpm_database(engine_config, artifact)
        engine_config["systems_path"] = str(self.prepared.systems_root)

    def configure_request(self, request):
        from scripts.fpm_accuracy.models.scenario_interpolation import configure_scenario_interpolation

        return configure_scenario_interpolation(request, self.artifact, self.prepared)

    def bind(self, perf_model, trainer, config, rows, seed, minimum, directory: Path, *, quality_gate=False):
        for name in ("estimate_forward_pass_with_route", "mixed_composition_diagnostics"):
            if not callable(getattr(perf_model, name, None)):
                raise DependencyError(f"Mixed composition requires native {name}.")
        writer = getattr(trainer, "write_original_a_manifest", None)
        if not callable(writer):
            raise DependencyError("Mixed composition requires the original-A manifest exporter.")
        path = directory / f"original-a-{len(self.manifests)}.json"
        source = writer(
            path,
            rows,
            config["worker_type"],
            identity=config,
            min_store_rows=minimum,
            seed_artifact_paths=[seed],
            provenance=self.provenance,
        )
        self.manifests[source["source_sha256"]] = path
        authored = deepcopy(config)
        policy = {
            "scenario": "a" if self.scenario == "a" else "a_c",
            "mode": "compose" if self.scenario == "a" else "compose_calibrated",
            "original_a": source,
        }
        if quality_gate:
            # An empty object enables the product defaults; Rust owns their values.
            policy["quality_gate"] = {}
        authored["estimator_config"]["temporary_mixed_composition"] = policy
        from scripts.fpm_accuracy.models.learned_scenario import _canonical_normalize

        return _canonical_normalize(perf_model, authored)

    def close(self):
        self.prepared.close()
