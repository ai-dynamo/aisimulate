# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

"""Pinned, hash-verified access to the AISimulate FPM Hugging Face dataset."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

from fpm_accuracy.contract import strict_json
from fpm_accuracy.exceptions import ConfigurationError, DataError
from fpm_accuracy.hf.models import (
    CaseStatus,
    ConfigurationSnapshot,
    FpmArtifact,
    MeasurementCase,
    MeasurementFile,
    MeasurementObservation,
    MeasurementReference,
    OrderingKind,
)
from fpm_accuracy.hf.overrides import HfCaseOverride, HfOverrides, load_overrides
from fpm_accuracy.hf.protocols import (
    SUPPORTED_EVIDENCE_FORMAT_IDS,
    ParsedObservation,
    adapter_for,
)
from fpm_accuracy.types.forward_pass import WorkloadKind
from fpm_accuracy.types.worker_config import WorkerConfig, WorkerConfigRecord

DEFAULT_REPO_ID = "nvidia/aisimulate-fpm-dataset"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRUTH_ROLES = frozenset({"truth", "derived_truth"})
_MEASUREMENT_ROLES = frozenset({*_TRUTH_ROLES, "supporting_evidence", "window", "configuration", "manifest"})
_FPM_ROLES = frozenset({"primary", "comparator", "reference", "historical", "quarantined"})
_FPM_PHASES = frozenset({"prefill", "decode"})
_SNAPSHOT_STATUSES = frozenset({"current", "historical"})
_FPM_SCHEMA_NAMES = frozenset({"aic_fpm_forward_perf", "fpm_forward_perf"})
_FPM_SCHEMA_VERSION = 6
_FPM_CONFIGURATION_FIELDS = (
    "model_path",
    "system",
    "backend",
    "backend_version",
    "weight_quantization",
    "kv_cache_dtype",
    "parallel_strategy",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    "dcp",
)


class HfDataset:
    """One immutable view of the private HF source-evidence repository."""

    def __init__(
        self,
        root: Path,
        *,
        repo_id: str,
        revision: str,
        overrides: HfOverrides,
        overrides_sha256: str | None,
        allowed_file_roots: Sequence[Path] = (),
    ) -> None:
        if _FULL_SHA.fullmatch(revision) is None:
            raise ConfigurationError("HF dataset revision must resolve to a full 40-character commit SHA")
        self.root = root.resolve()
        self._allowed_file_roots = tuple(dict.fromkeys((self.root, *(path.resolve() for path in allowed_file_roots))))
        self.repo_id = repo_id
        self.revision = revision
        self._overrides = overrides
        self._overrides_sha256 = overrides_sha256
        self._index = self._read_json("catalog/index.json")
        if self._index.get("catalog_version") != 5:
            raise DataError(f"unsupported HF catalog version: {self._index.get('catalog_version')!r}")
        if self._index.get("dataset_id") != repo_id:
            raise DataError(
                f"HF catalog dataset_id {self._index.get('dataset_id')!r} does not match requested repo {repo_id!r}"
            )
        self._configuration_cache: tuple[ConfigurationSnapshot, ...] | None = None

    @classmethod
    def from_local(
        cls,
        root: str | Path,
        *,
        revision: str | None = None,
        overrides_path: str | Path | None = None,
        repo_id: str = DEFAULT_REPO_ID,
    ) -> HfDataset:
        local_root = Path(root).expanduser().resolve()
        if not local_root.is_dir():
            raise ConfigurationError(f"HF dataset root is not a directory: {local_root}")
        resolved_revision = _resolve_local_revision(local_root, revision)
        override_file = Path(overrides_path).expanduser() if overrides_path is not None else None
        overrides = load_overrides(override_file)
        overrides_sha256 = _file_sha256(override_file) if override_file is not None else None
        return cls(
            local_root,
            repo_id=repo_id,
            revision=resolved_revision,
            overrides=overrides,
            overrides_sha256=overrides_sha256,
        )

    @classmethod
    def from_hub(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        *,
        revision: str = "main",
        overrides_path: str | Path | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> HfDataset:
        info = HfApi(token=token).repo_info(repo_id, repo_type="dataset", revision=revision)
        resolved_revision = info.sha
        if not isinstance(resolved_revision, str) or _FULL_SHA.fullmatch(resolved_revision) is None:
            raise DataError(f"Hugging Face did not resolve {repo_id}@{revision} to a full commit SHA")
        root = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        local_root = Path(root).expanduser().resolve()
        override_file = Path(overrides_path).expanduser() if overrides_path is not None else None
        overrides = load_overrides(override_file)
        overrides_sha256 = _file_sha256(override_file) if override_file is not None else None
        return cls(
            local_root,
            repo_id=repo_id,
            revision=resolved_revision,
            overrides=overrides,
            overrides_sha256=overrides_sha256,
            allowed_file_roots=_hub_cache_blob_roots(local_root, resolved_revision),
        )

    def configurations(self, *, include_history: bool = False) -> tuple[ConfigurationSnapshot, ...]:
        if self._configuration_cache is None:
            current = self._manifest_paths("configuration_manifests")
            history = self._manifest_paths("history_manifests")
            history_hashes: dict[str, str] = {}
            for path in current:
                manifest = self._read_json(path)
                for value in manifest.get("history") or ():
                    if not isinstance(value, Mapping):
                        raise DataError(f"configuration manifest {path} contains a non-object history entry")
                    history_path = _required_str(value, "manifest_path", path)
                    history_sha256 = _required_sha256(value, "manifest_sha256", path)
                    previous = history_hashes.setdefault(history_path, history_sha256)
                    if previous != history_sha256:
                        raise DataError(f"conflicting recorded hashes for history manifest {history_path}")
            missing_hashes = set(history) - set(history_hashes)
            if missing_hashes:
                raise DataError(f"historical manifests lack a current-manifest hash pointer: {sorted(missing_hashes)}")
            extra_hashes = set(history_hashes) - set(history)
            if extra_hashes:
                raise DataError(f"current manifests reference uncataloged history: {sorted(extra_hashes)}")
            for path in history:
                self._verify_file(self._safe_path(path), history_hashes[path], description=path)
            snapshots: list[ConfigurationSnapshot] = []
            for expected_status, paths in (("current", current), ("historical", history)):
                for path in paths:
                    snapshot = self._load_configuration(path)
                    if snapshot.snapshot_status != expected_status:
                        raise DataError(
                            f"HF catalog lists {path} as {expected_status}, but its manifest says "
                            f"{snapshot.snapshot_status!r}"
                        )
                    snapshots.append(snapshot)
            identities = [snapshot.configuration_id for snapshot in snapshots]
            if len(identities) != len(set(identities)):
                raise DataError("HF catalog contains duplicate configuration/snapshot identities")
            self._configuration_cache = tuple(
                sorted(snapshots, key=lambda item: (item.configuration_path, item.snapshot_id))
            )
            self._validate_override_selectors(self._configuration_cache)
        if include_history:
            return self._configuration_cache
        return tuple(snapshot for snapshot in self._configuration_cache if snapshot.snapshot_status == "current")

    def measurement_case(
        self,
        configuration_path: str,
        *,
        snapshot_id: str | None = None,
        fpm_artifact_ids: Sequence[str] | None = None,
    ) -> MeasurementCase:
        configuration = self._select_configuration(configuration_path, snapshot_id)
        override = self._overrides.find(configuration.configuration_path, configuration.snapshot_id)
        manifest = self._load_measurement_manifest(configuration)
        files = self._measurement_files(configuration, manifest)
        truth_files, helper_files = self._select_measurement_files(files, override)
        fpm_artifacts = self._select_fpm_artifacts(configuration, override, fpm_artifact_ids)

        for file in (*truth_files, *helper_files):
            self._verify_file(file.local_path, file.sha256, description=file.path)
        for artifact in fpm_artifacts:
            self._verify_fpm_artifact(configuration, artifact)

        ordering = override.ordering if override and override.ordering else OrderingKind.FILE_ORDER_FALLBACK
        worker_role: Literal["prefill", "decode", "aggregated"] = (
            override.worker_role if override and override.worker_role else "aggregated"
        )
        override_effects = _override_effects(override)
        applied_override_sha256 = self._overrides_sha256 if override is not None else None
        adapter = adapter_for(configuration.measurements.protocol_id)
        parser_policy_id = adapter.policy_id if adapter is not None else None
        if configuration.measurements.evidence_format_id is not None and truth_files:
            raise DataError("supporting-evidence formats may not declare truth or derived_truth files")
        if truth_files and configuration.measurements.protocol_id is None:
            raise DataError("measurement truth files require a measurement protocol")
        unsupported = configuration.measurements.protocol_id is not None and adapter is None
        if not truth_files or unsupported:
            membership_sha256 = _membership_sha256(())
            supporting_evidence = bool(helper_files) and (configuration.measurements.evidence_format_id is not None)
            status = CaseStatus.SUPPORTING_EVIDENCE_ONLY if supporting_evidence else CaseStatus.NO_MEASUREMENTS
            if unsupported:
                status = CaseStatus.UNSUPPORTED_PROTOCOL
            warning = (
                "supporting evidence does not contain the scheduler inputs and observed latency required for evaluation"
                if supporting_evidence
                else "no truth or derived_truth files are declared for this snapshot"
            )
            return MeasurementCase(
                case_id=self._case_id(
                    configuration,
                    truth_files,
                    helper_files,
                    ordering,
                    worker_role,
                    override_effects,
                    membership_sha256,
                    parser_policy_id,
                ),
                measurement_membership_sha256=membership_sha256,
                configuration=configuration,
                protocol_id=configuration.measurements.protocol_id,
                status=status,
                observations=(),
                truth_files=truth_files,
                helper_files=helper_files,
                fpm_artifacts=fpm_artifacts,
                worker_role=worker_role,
                ordering=ordering,
                override_applied=override is not None,
                override_sha256=applied_override_sha256,
                override_effects=override_effects,
                parser_policy_id=parser_policy_id,
                warnings=(warning,),
            )

        assert adapter is not None

        parsed = adapter.parse(
            configuration.configuration_id,
            truth_files,
            configuration.worker_config_record.config.parallelism.attention_dp_size,
        )
        ordered, ordering, ordering_warnings = self._order_observations(parsed.observations, override)
        observations = self._materialize_observations(configuration.configuration_id, ordered)
        if not (override and override.worker_role):
            worker_role = _infer_worker_role(observations)
        warnings = (*parsed.warnings, *ordering_warnings)
        status = CaseStatus.READY if observations else CaseStatus.NO_MEASUREMENTS
        if not observations:
            warnings = (*warnings, "declared truth files contain no usable measured observations")
        membership_sha256 = _membership_sha256(observations)
        return MeasurementCase(
            case_id=self._case_id(
                configuration,
                truth_files,
                helper_files,
                ordering,
                worker_role,
                override_effects,
                membership_sha256,
                parser_policy_id,
            ),
            measurement_membership_sha256=membership_sha256,
            configuration=configuration,
            protocol_id=configuration.measurements.protocol_id,
            status=status,
            observations=observations,
            truth_files=truth_files,
            helper_files=helper_files,
            fpm_artifacts=fpm_artifacts,
            worker_role=worker_role,
            ordering=ordering,
            override_applied=override is not None,
            override_sha256=applied_override_sha256,
            override_effects=override_effects,
            parser_policy_id=parser_policy_id,
            warnings=tuple(dict.fromkeys(warnings)),
            issues=parsed.issues,
        )

    def _case_id(
        self,
        configuration: ConfigurationSnapshot,
        truth_files: Sequence[MeasurementFile],
        helper_files: Sequence[MeasurementFile],
        ordering: OrderingKind,
        worker_role: str,
        override_effects: Sequence[str],
        membership_sha256: str,
        parser_policy_id: str | None,
    ) -> str:
        payload = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "configuration_path": configuration.configuration_path,
            "snapshot_id": configuration.snapshot_id,
            "protocol_id": configuration.measurements.protocol_id,
            "truth_files": [(file.measurement_file_id, file.sha256) for file in truth_files],
            "helper_files": [(file.measurement_file_id, file.sha256) for file in helper_files],
            "ordering": ordering.value,
            "worker_role": worker_role,
            "override_effects": list(override_effects),
            "override_sha256": self._overrides_sha256 if override_effects else None,
            "parser_policy_id": parser_policy_id,
            "measurement_membership_sha256": membership_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"case-{hashlib.sha256(encoded).hexdigest()[:24]}"

    def _manifest_paths(self, key: str) -> tuple[str, ...]:
        values = self._index.get(key, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise DataError(f"HF catalog field {key!r} must be a list of paths")
        if len(values) != len(set(values)):
            raise DataError(f"HF catalog field {key!r} contains duplicate paths")
        return tuple(values)

    def _validate_override_selectors(self, snapshots: Sequence[ConfigurationSnapshot]) -> None:
        identities = {(item.configuration_path, item.snapshot_id) for item in snapshots}
        paths = {item.configuration_path for item in snapshots}
        for override in self._overrides.overrides:
            if override.snapshot_id is None:
                known = override.configuration_path in paths
            else:
                known = (override.configuration_path, override.snapshot_id) in identities
            if not known:
                selector = override.configuration_path
                if override.snapshot_id is not None:
                    selector = f"{selector}@{override.snapshot_id}"
                raise ConfigurationError(f"HF override selects an unknown configuration snapshot: {selector}")

    def _load_configuration(self, manifest_path: str) -> ConfigurationSnapshot:
        manifest = self._read_json(manifest_path)
        if manifest.get("manifest_version") != 3:
            raise DataError(f"unsupported configuration manifest version in {manifest_path}")
        configuration_path = _required_str(manifest, "configuration_path", manifest_path)
        snapshot_id = _required_str(manifest, "snapshot_id", manifest_path)
        snapshot_status = _required_str(manifest, "snapshot_status", manifest_path)
        if snapshot_status not in _SNAPSHOT_STATUSES:
            raise DataError(f"configuration manifest {manifest_path} has invalid snapshot status {snapshot_status!r}")
        identity = f"{configuration_path}@{snapshot_id}"
        fpm_values = manifest.get("fpm")
        if not isinstance(fpm_values, list):
            raise DataError(f"configuration manifest {manifest_path} field 'fpm' must be a list")
        fpm_artifacts = tuple(self._fpm_artifact(value, manifest_path) for value in fpm_values)
        fpm_ids = [artifact.artifact_id for artifact in fpm_artifacts]
        if len(fpm_ids) != len(set(fpm_ids)):
            raise DataError(f"configuration manifest {manifest_path} contains duplicate FPM artifact IDs")
        measurement_value = manifest.get("measurements")
        if not isinstance(measurement_value, Mapping):
            raise DataError(f"configuration manifest {manifest_path} is missing measurements")
        protocol_id = _optional_str(measurement_value.get("protocol_id"))
        evidence_format_id = _optional_str(measurement_value.get("evidence_format_id"))
        if protocol_id is not None and evidence_format_id is not None:
            raise DataError(
                f"configuration manifest {manifest_path} may not declare both a measurement protocol "
                "and an evidence format"
            )
        if evidence_format_id is not None and evidence_format_id not in SUPPORTED_EVIDENCE_FORMAT_IDS:
            raise DataError(f"unsupported evidence format: {evidence_format_id}")
        measurement = MeasurementReference(
            artifact_id=_required_str(measurement_value, "artifact_id", manifest_path),
            protocol_id=protocol_id,
            evidence_format_id=evidence_format_id,
            manifest_path=_required_str(measurement_value, "manifest_path", manifest_path),
            manifest_sha256=_required_sha256(measurement_value, "manifest_sha256", manifest_path),
            file_count=_required_nonnegative_int(measurement_value, "file_count", manifest_path),
            provenance_url=self._provenance_url(_required_str(measurement_value, "manifest_path", manifest_path)),
        )
        self._verify_file(
            self._safe_path(measurement.manifest_path),
            measurement.manifest_sha256,
            description=measurement.manifest_path,
        )
        return ConfigurationSnapshot(
            configuration_id=identity,
            configuration_path=configuration_path,
            snapshot_id=snapshot_id,
            snapshot_status=snapshot_status,
            manifest_path=manifest_path,
            manifest_sha256=_content_sha256(self._safe_path(manifest_path)),
            manifest_provenance_url=self._provenance_url(manifest_path),
            model_id=_required_str(manifest, "model_id", manifest_path),
            model_revision=_optional_str(manifest.get("model_revision")),
            system=_required_str(manifest, "system", manifest_path),
            gpu_family=_required_str(manifest, "gpu_family", manifest_path),
            framework=_required_str(manifest, "framework", manifest_path),
            framework_version=_required_str(manifest, "framework_version", manifest_path),
            parallelism=_required_str(manifest, "parallelism", manifest_path),
            parallel_strategy=_required_str(manifest, "parallel_strategy", manifest_path),
            fpm_artifacts=fpm_artifacts,
            measurements=measurement,
            worker_config_record=_worker_config_record(identity, manifest),
            aisim_commit=_optional_str(manifest.get("aisim_commit")),
        )

    def _fpm_artifact(self, value: Any, manifest_path: str) -> FpmArtifact:
        if not isinstance(value, Mapping):
            raise DataError(f"configuration manifest {manifest_path} contains a non-object FPM entry")
        path = _required_str(value, "path", manifest_path)
        metadata_path = _required_str(value, "metadata_path", manifest_path)
        local_metadata_path = self._safe_path(metadata_path)
        phases = value.get("phases")
        if not isinstance(phases, list) or not phases or not all(isinstance(phase, str) for phase in phases):
            raise DataError(f"configuration manifest {manifest_path} contains invalid FPM phases")
        if len(phases) != len(set(phases)) or not set(phases).issubset(_FPM_PHASES):
            raise DataError(f"configuration manifest {manifest_path} contains invalid FPM phases: {phases!r}")
        role = _required_str(value, "role", manifest_path)
        if role not in _FPM_ROLES:
            raise DataError(f"configuration manifest {manifest_path} contains unknown FPM role {role!r}")
        return FpmArtifact(
            artifact_id=_required_str(value, "artifact_id", manifest_path),
            path=path,
            metadata_path=metadata_path,
            sha256=_required_sha256(value, "sha256", manifest_path),
            metadata_sha256=_content_sha256(local_metadata_path),
            role=role,
            phases=tuple(phases),
            row_count=_required_positive_int(value, "row_count", manifest_path),
            local_path=self._safe_path(path),
            local_metadata_path=local_metadata_path,
            provenance_url=self._provenance_url(path),
            metadata_provenance_url=self._provenance_url(metadata_path),
        )

    def _load_measurement_manifest(self, configuration: ConfigurationSnapshot) -> Mapping[str, Any]:
        manifest = self._read_json(
            configuration.measurements.manifest_path,
            expected_sha256=configuration.measurements.manifest_sha256,
        )
        expected = {
            "manifest_version": 4,
            "measurement_artifact_id": configuration.measurements.artifact_id,
            "measurement_protocol_id": configuration.measurements.protocol_id,
            "evidence_format_id": configuration.measurements.evidence_format_id,
            "configuration_path": configuration.configuration_path,
            "snapshot_id": configuration.snapshot_id,
            "snapshot_status": configuration.snapshot_status,
            "model_id": configuration.model_id,
            "system": configuration.system,
            "framework": configuration.framework,
            "framework_version": configuration.framework_version,
            "parallelism": configuration.parallelism,
        }
        mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if mismatches:
            raise DataError(
                f"measurement manifest {configuration.measurements.manifest_path} disagrees with its configuration: "
                f"{mismatches}"
            )
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != configuration.measurements.file_count:
            raise DataError(
                f"measurement manifest {configuration.measurements.manifest_path} file count does not match "
                f"the configuration manifest"
            )
        return manifest

    def _measurement_files(
        self,
        configuration: ConfigurationSnapshot,
        manifest: Mapping[str, Any],
    ) -> tuple[MeasurementFile, ...]:
        result: list[MeasurementFile] = []
        for value in manifest["files"]:
            if not isinstance(value, Mapping):
                raise DataError(
                    f"measurement manifest {configuration.measurements.manifest_path} has a non-object file"
                )
            if value.get("configuration_path") not in (None, configuration.configuration_path):
                raise DataError("measurement file belongs to a different configuration")
            if value.get("snapshot_id") not in (None, configuration.snapshot_id):
                raise DataError("measurement file belongs to a different snapshot")
            if value.get("snapshot_status") not in (None, configuration.snapshot_status):
                raise DataError("measurement file belongs to a different snapshot status")
            path = _required_str(value, "path", configuration.measurements.manifest_path)
            role = _required_str(value, "role", configuration.measurements.manifest_path)
            if role not in _MEASUREMENT_ROLES:
                raise DataError(
                    f"measurement manifest {configuration.measurements.manifest_path} has unknown role {role!r}"
                )
            source_sha256 = _optional_str(value.get("source_sha256"))
            if source_sha256 is not None and _SHA256.fullmatch(source_sha256) is None:
                raise DataError(
                    f"measurement manifest {configuration.measurements.manifest_path} has an invalid source SHA-256"
                )
            derived = value.get("derived", False)
            if not isinstance(derived, bool):
                raise DataError(
                    f"measurement manifest {configuration.measurements.manifest_path} field 'derived' must be boolean"
                )
            representation = _optional_str(value.get("representation"))
            if representation not in {None, "pre_grouped_rank_lists"}:
                raise DataError(
                    f"measurement manifest {configuration.measurements.manifest_path} has unsupported "
                    f"representation {representation!r}"
                )
            iteration_count = value.get("iteration_count")
            rank_record_count = value.get("rank_record_count")
            grouping = value.get("grouping")
            if representation == "pre_grouped_rank_lists":
                if role not in _TRUTH_ROLES:
                    raise DataError("pre_grouped_rank_lists representation requires a truth file role")
                if not isinstance(iteration_count, int) or isinstance(iteration_count, bool) or iteration_count <= 0:
                    raise DataError("pre_grouped_rank_lists representation requires a positive iteration_count")
                if (
                    not isinstance(rank_record_count, int)
                    or isinstance(rank_record_count, bool)
                    or rank_record_count <= 0
                ):
                    raise DataError("pre_grouped_rank_lists representation requires a positive rank_record_count")
                if not isinstance(grouping, Mapping):
                    raise DataError("pre_grouped_rank_lists representation requires grouping metadata")
            elif any(item is not None for item in (iteration_count, rank_record_count, grouping)):
                raise DataError("grouping metadata requires the pre_grouped_rank_lists representation")
            result.append(
                MeasurementFile(
                    measurement_file_id=_required_str(
                        value, "measurement_file_id", configuration.measurements.manifest_path
                    ),
                    path=path,
                    sha256=_required_sha256(value, "sha256", configuration.measurements.manifest_path),
                    role=role,
                    local_path=self._safe_path(path),
                    provenance_url=self._provenance_url(path),
                    source_path=_optional_str(value.get("source_path")),
                    source_sha256=source_sha256,
                    derived=derived,
                    representation=representation,
                    iteration_count=iteration_count,
                    rank_record_count=rank_record_count,
                    grouping=grouping,
                )
            )
        ids = [file.measurement_file_id for file in result]
        if len(ids) != len(set(ids)):
            raise DataError(f"measurement manifest {configuration.measurements.manifest_path} has duplicate file IDs")
        self._validate_pre_grouped_metadata(configuration, tuple(result))
        return tuple(result)

    def _validate_pre_grouped_metadata(
        self,
        configuration: ConfigurationSnapshot,
        files: tuple[MeasurementFile, ...],
    ) -> None:
        by_path = {file.path: file for file in files}
        expected_ranks = list(range(configuration.worker_config_record.config.parallelism.attention_dp_size))
        for file in files:
            if file.representation != "pre_grouped_rank_lists":
                continue
            grouping = file.grouping
            if grouping is None:
                raise DataError(f"pre-grouped measurement file {file.path} is missing grouping metadata")
            if grouping.get("method") != "pre_grouped" or grouping.get("authority") not in {"producer", "derived"}:
                raise DataError(f"pre-grouped measurement file {file.path} has invalid grouping authority")
            if grouping.get("expected_dp_ranks") != expected_ranks:
                raise DataError(f"pre-grouped measurement file {file.path} ranks disagree with its configuration")
            producer = grouping.get("producer")
            if (
                not isinstance(producer, Mapping)
                or not isinstance(producer.get("component"), str)
                or not producer.get("component")
            ):
                raise DataError(f"pre-grouped measurement file {file.path} is missing its producer")
            provenance = grouping.get("provenance_files")
            if not isinstance(provenance, list) or not provenance:
                raise DataError(f"pre-grouped measurement file {file.path} is missing grouping provenance")
            for reference in provenance:
                if not isinstance(reference, Mapping):
                    raise DataError(f"pre-grouped measurement file {file.path} has invalid grouping provenance")
                reference_path = reference.get("path")
                reference_sha256 = reference.get("sha256")
                declared = by_path.get(reference_path) if isinstance(reference_path, str) else None
                if declared is None or declared.sha256 != reference_sha256:
                    raise DataError(
                        f"pre-grouped measurement file {file.path} references undeclared grouping provenance"
                    )

    def _select_measurement_files(
        self,
        files: tuple[MeasurementFile, ...],
        override: HfCaseOverride | None,
    ) -> tuple[tuple[MeasurementFile, ...], tuple[MeasurementFile, ...]]:
        by_id = {file.measurement_file_id: file for file in files}
        truth = tuple(file for file in files if file.role in _TRUTH_ROLES)
        helpers = tuple(file for file in files if file.role not in _TRUTH_ROLES)
        if override and override.truth_file_ids is not None:
            truth = tuple(_select_ids(by_id, override.truth_file_ids, "truth_file_ids"))
            invalid = [file.measurement_file_id for file in truth if file.role not in _TRUTH_ROLES]
            if invalid:
                raise ConfigurationError(f"HF truth_file_ids select non-truth files: {invalid}")
        if override and override.helper_file_ids is not None:
            helpers = tuple(_select_ids(by_id, override.helper_file_ids, "helper_file_ids"))
            invalid = [file.measurement_file_id for file in helpers if file.role in _TRUTH_ROLES]
            if invalid:
                raise ConfigurationError(f"HF helper_file_ids select truth files: {invalid}")
        overlap = {file.measurement_file_id for file in truth} & {file.measurement_file_id for file in helpers}
        if overlap:
            raise ConfigurationError(f"HF overrides select files as both truth and helper: {sorted(overlap)}")
        return truth, helpers

    def _select_fpm_artifacts(
        self,
        configuration: ConfigurationSnapshot,
        override: HfCaseOverride | None,
        requested_ids: Sequence[str] | None,
    ) -> tuple[FpmArtifact, ...]:
        by_id = {artifact.artifact_id: artifact for artifact in configuration.fpm_artifacts}
        allowed_ids = override.fpm_artifact_ids if override and override.fpm_artifact_ids is not None else tuple(by_id)
        allowed = tuple(_select_ids(by_id, allowed_ids, "fpm_artifact_ids"))
        if requested_ids is None:
            if configuration.snapshot_status == "current":
                return tuple(artifact for artifact in allowed if artifact.role not in {"historical", "quarantined"})
            return tuple(artifact for artifact in allowed if artifact.role != "quarantined")
        requested = tuple(_select_ids(by_id, requested_ids, "fpm_artifact_ids"))
        disallowed = {artifact.artifact_id for artifact in requested} - {artifact.artifact_id for artifact in allowed}
        if disallowed:
            raise ConfigurationError(f"requested FPM artifacts are excluded by HF overrides: {sorted(disallowed)}")
        quarantined = [artifact.artifact_id for artifact in requested if artifact.role == "quarantined"]
        if quarantined:
            raise ConfigurationError(f"quarantined FPM artifacts cannot be evaluated: {quarantined}")
        return requested

    def _verify_fpm_artifact(self, configuration: ConfigurationSnapshot, artifact: FpmArtifact) -> None:
        self._verify_file(artifact.local_path, artifact.sha256, description=artifact.path)
        metadata = self._read_json(artifact.metadata_path, expected_sha256=artifact.metadata_sha256)
        if metadata.get("parquet_sha256") != artifact.sha256:
            raise DataError(f"FPM sidecar {artifact.metadata_path} does not bind the selected parquet hash")
        if metadata.get("row_count") != artifact.row_count:
            raise DataError(f"FPM sidecar {artifact.metadata_path} does not bind the selected parquet row count")
        if (
            metadata.get("schema_name") not in _FPM_SCHEMA_NAMES
            or metadata.get("schema_version") != _FPM_SCHEMA_VERSION
        ):
            raise DataError(f"FPM sidecar {artifact.metadata_path} declares an unsupported schema")
        expected_identity = _configuration_identity(configuration)
        selector = metadata.get("configuration_selector")
        if not isinstance(selector, Mapping):
            raise DataError(f"FPM sidecar {artifact.metadata_path} is missing configuration_selector")
        # Legacy libraries omit DCP, which means the unsharded value of one.
        selector = {"dcp": 1, **selector}
        selector_fields = set(selector)
        expected_fields = set(expected_identity)
        if selector_fields != expected_fields:
            missing = sorted(expected_fields - selector_fields)
            unknown = sorted(selector_fields - expected_fields)
            raise DataError(
                f"FPM sidecar {artifact.metadata_path} has invalid configuration_selector fields; "
                f"missing={missing}, unknown={unknown}"
            )
        mismatches = {
            field: (selector[field], expected)
            for field, expected in expected_identity.items()
            if not _same_identity_value(selector[field], expected)
        }
        if mismatches:
            raise DataError(f"FPM sidecar {artifact.metadata_path} selects a different configuration: {mismatches}")
        try:
            parquet = pq.ParquetFile(artifact.local_path)
            physical_rows = parquet.metadata.num_rows
        except (OSError, pa.ArrowException) as exc:
            raise DataError(f"cannot read selected FPM Parquet {artifact.path}: {exc}") from exc
        if physical_rows != artifact.row_count:
            raise DataError(
                f"selected FPM Parquet {artifact.path} has {physical_rows} rows; expected {artifact.row_count}"
            )
        parquet_fields = set(parquet.schema_arrow.names)
        parquet_identity = dict(expected_identity)
        if "dcp" not in parquet_fields and expected_identity["dcp"] == 1:
            parquet_identity.pop("dcp")
        missing_identity_fields = sorted(set(parquet_identity) - parquet_fields)
        if missing_identity_fields:
            raise DataError(
                f"selected FPM Parquet {artifact.path} is missing configuration identity columns: "
                f"{missing_identity_fields}"
            )
        try:
            identity_table = parquet.read(columns=list(parquet_identity))
        except (OSError, pa.ArrowException) as exc:
            raise DataError(f"cannot read selected FPM Parquet identity columns {artifact.path}: {exc}") from exc
        parquet_mismatches: dict[str, list[Any]] = {}
        for field, expected in parquet_identity.items():
            values = identity_table[field].to_pylist()
            invalid = {repr(value) for value in values if not _same_identity_value(value, expected)}
            if invalid:
                parquet_mismatches[field] = sorted(invalid)
        if parquet_mismatches:
            raise DataError(
                f"selected FPM Parquet {artifact.path} contains rows for a different configuration: "
                f"{parquet_mismatches}"
            )

    def _select_configuration(self, configuration_path: str, snapshot_id: str | None) -> ConfigurationSnapshot:
        candidates = [
            item
            for item in self.configurations(include_history=True)
            if item.configuration_path == configuration_path
            and ((snapshot_id is None and item.snapshot_status == "current") or item.snapshot_id == snapshot_id)
        ]
        if not candidates:
            suffix = f" at snapshot {snapshot_id}" if snapshot_id else ""
            raise ConfigurationError(f"HF configuration not found: {configuration_path}{suffix}")
        if len(candidates) != 1:
            raise DataError(f"HF configuration selection is ambiguous: {configuration_path}")
        return candidates[0]

    def _order_observations(
        self,
        observations: tuple[ParsedObservation, ...],
        override: HfCaseOverride | None,
    ) -> tuple[tuple[ParsedObservation, ...], OrderingKind, tuple[str, ...]]:
        requested = override.ordering if override else None
        all_chronological = bool(observations) and all(item.chronology_key is not None for item in observations)
        if requested is OrderingKind.CHRONOLOGICAL:
            if not all_chronological:
                raise ConfigurationError("ordering: chronological requires a timestamp on every selected observation")
            return (
                tuple(
                    sorted(
                        observations,
                        key=lambda item: (item.chronology_key, item.source_file.path, item.source_row),
                    )
                ),
                OrderingKind.CHRONOLOGICAL,
                (),
            )
        if requested is OrderingKind.INTENTIONAL_SWEEP:
            return observations, OrderingKind.INTENTIONAL_SWEEP, ()
        if requested is OrderingKind.FILE_ORDER_FALLBACK:
            return observations, OrderingKind.FILE_ORDER_FALLBACK, (_fallback_warning(),)
        if all_chronological:
            return (
                tuple(
                    sorted(
                        observations,
                        key=lambda item: (item.chronology_key, item.source_file.path, item.source_row),
                    )
                ),
                OrderingKind.CHRONOLOGICAL,
                (),
            )
        return observations, OrderingKind.FILE_ORDER_FALLBACK, (_fallback_warning(),)

    @staticmethod
    def _materialize_observations(
        configuration_id: str,
        observations: Iterable[ParsedObservation],
    ) -> tuple[MeasurementObservation, ...]:
        result: list[MeasurementObservation] = []
        seen: set[str] = set()
        for order, parsed in enumerate(observations):
            identity = {
                "configuration_id": configuration_id,
                "source_file_id": parsed.source_file.measurement_file_id,
                "source_path": parsed.source_file.path,
                "source_row": parsed.source_row,
                "identity_hint": parsed.identity_hint,
                "event_time": parsed.event_time,
                "ranks": [
                    {
                        "target_blind_payload": rank.to_aic_dict(include_observation=False),
                        "wall_time_s": rank.wall_time_s,
                    }
                    for rank in parsed.iteration.ranks
                ],
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()[:24]
            observation_id = f"observation-{digest}"
            if observation_id in seen:
                raise DataError(f"duplicate HF observation identity: {observation_id}")
            seen.add(observation_id)
            result.append(
                MeasurementObservation(
                    observation_id=observation_id,
                    configuration_id=configuration_id,
                    order=order,
                    source_file_id=parsed.source_file.measurement_file_id,
                    source_path=parsed.source_file.path,
                    source_row=parsed.source_row,
                    iteration=parsed.iteration,
                    event_time=parsed.event_time,
                )
            )
        return tuple(result)

    def _read_json(self, relative: str, *, expected_sha256: str | None = None) -> Mapping[str, Any]:
        path = self._safe_path(relative)
        if expected_sha256 is not None:
            self._verify_file(path, expected_sha256, description=relative)
        try:
            value = strict_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError(f"cannot parse HF JSON {relative}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise DataError(f"HF JSON {relative} must contain an object")
        return value

    def _safe_path(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise DataError(f"unsafe HF path: {relative!r}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise DataError(f"unsafe HF path: {relative!r}")
        path = self.root.joinpath(*pure.parts)
        if not path.is_file():
            raise DataError(f"HF dataset file is missing: {relative}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise DataError(f"HF dataset path cannot be resolved: {relative}") from exc
        if not any(_is_relative_to(resolved, root) for root in self._allowed_file_roots):
            raise DataError(f"HF dataset path escapes its pinned root: {relative}")
        return path

    @staticmethod
    def _verify_file(path: Path, expected_sha256: str, *, description: str) -> None:
        if _SHA256.fullmatch(expected_sha256) is None:
            raise DataError(f"invalid recorded SHA-256 for {description}")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise DataError(f"cannot read HF dataset file {description}: {exc}") from exc
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise DataError(f"HF dataset hash mismatch for {description}: expected {expected_sha256}, got {actual}")

    def _provenance_url(self, relative: str) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}/blob/{self.revision}/{quote(relative, safe='/')}"


def _resolve_local_revision(root: Path, requested: str | None) -> str:
    explicit_revision = requested if requested is not None and _FULL_SHA.fullmatch(requested) else None
    try:
        top_level_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        if explicit_revision is None:
            raise ConfigurationError("a non-Git local HF dataset requires an explicit full commit SHA") from exc
        return explicit_revision

    top_level = Path(top_level_result.stdout.strip()).resolve()
    if top_level != root:
        raise ConfigurationError(f"local HF dataset root must be the Git repository root: {top_level}")
    ref = requested or "HEAD"
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"{ref}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        head_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(f"cannot validate local HF revision {requested!r}") from exc
    revision = revision_result.stdout.strip()
    if _FULL_SHA.fullmatch(revision) is None:
        raise ConfigurationError(f"local HF revision {requested!r} did not resolve to a full commit SHA")
    if head_result.stdout.strip() != revision:
        raise ConfigurationError(f"local HF checkout HEAD does not match requested revision {revision}")
    if status_result.stdout.strip():
        raise ConfigurationError("local HF checkout must be clean so its pinned revision describes the bytes read")
    return revision


def _hub_cache_blob_roots(root: Path, revision: str) -> tuple[Path, ...]:
    """Allow the Hub blob stores backing this exact cache snapshot.

    ``snapshot_download`` normally returns ``.../snapshots/<sha>`` where files
    are symlinks into the same repository cache's ``blobs`` directory. Local
    Hub 1.32 can link those blobs into the cache-wide shared store. Local
    dataset checkouts do not receive either exception; content hashes are
    still verified independently of these storage locations.
    """

    if root.name != revision or root.parent.name != "snapshots":
        return ()
    blobs = root.parent.parent / "blobs"
    roots = [blobs.resolve()] if blobs.is_dir() else []
    shared = root.parents[2] / "blobs"
    marker = shared / ".huggingface-shared-blobs"
    if (
        shared.is_dir()
        and not shared.is_symlink()
        and marker.is_file()
        and not marker.is_symlink()
        and marker.read_text() == "1\n"
    ):
        roots.append(shared.resolve())
    return tuple(roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _worker_config_record(configuration_id: str, manifest: Mapping[str, Any]) -> WorkerConfigRecord:
    moe_ep = _required_positive_int(manifest, "moe_ep", configuration_id) if "moe_ep" in manifest else 1
    moe_tp = _required_positive_int(manifest, "moe_tp", configuration_id) if "moe_tp" in manifest else 1
    dcp = _required_positive_int(manifest, "dcp", configuration_id) if "dcp" in manifest else 1
    model_kind = "moe" if moe_ep > 1 or moe_tp > 1 else None
    engine = {
        "schema_version": 1,
        "model_name": manifest.get("model_id"),
        "system_name": manifest.get("system"),
        "backend": manifest.get("framework"),
        "backend_version": manifest.get("framework_version"),
        "tp_size": manifest.get("tp"),
        "pp_size": manifest.get("pp"),
        "attention_dp_size": manifest.get("dp"),
        "moe_tp_size": moe_tp,
        "moe_ep_size": moe_ep,
        "cp_size": manifest.get("cp"),
        "weight_dtype": manifest.get("weight_quantization"),
        "kv_cache_dtype": manifest.get("kv_cache_dtype"),
        "extra": {"parallel_strategy": str(manifest.get("parallel_strategy", ""))},
    }
    config = WorkerConfig.model_validate(
        {
            "backend": {"name": manifest.get("framework"), "version": manifest.get("framework_version")},
            "hardware": {"gpu_sku": manifest.get("system"), "system_name": manifest.get("system")},
            "model": {"id": manifest.get("model_id"), "revision": manifest.get("model_revision"), "kind": model_kind},
            "parallelism": {
                "tensor_parallel_size": manifest.get("tp"),
                "pipeline_parallel_size": manifest.get("pp"),
                "attention_dp_size": manifest.get("dp"),
                "context_parallel_size": manifest.get("cp"),
                "decode_context_parallel_size": dcp,
            },
            "precision": {
                "weights": manifest.get("weight_quantization"),
                "kv_cache": manifest.get("kv_cache_dtype"),
            },
            "aic_engine_config": {key: value for key, value in engine.items() if value is not None},
        }
    )
    return WorkerConfigRecord(configuration_id=configuration_id, schema_version=1, config=config)


def _select_ids(values: Mapping[str, Any], selected: Sequence[str], field: str) -> list[Any]:
    if len(selected) != len(set(selected)):
        raise ConfigurationError(f"{field} may not contain duplicate IDs")
    unknown = [value for value in selected if value not in values]
    if unknown:
        raise ConfigurationError(f"{field} references unknown stable artifact IDs: {unknown}")
    return [values[value] for value in selected]


def _required_str(value: Mapping[str, Any], key: str, source: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise DataError(f"{source} field {key!r} must be a non-empty string")
    return result


def _required_sha256(value: Mapping[str, Any], key: str, source: str) -> str:
    result = _required_str(value, key, source)
    if _SHA256.fullmatch(result) is None:
        raise DataError(f"{source} field {key!r} must be a lowercase SHA-256")
    return result


def _required_nonnegative_int(value: Mapping[str, Any], key: str, source: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise DataError(f"{source} field {key!r} must be a non-negative integer")
    return result


def _required_positive_int(value: Mapping[str, Any], key: str, source: str) -> int:
    result = _required_nonnegative_int(value, key, source)
    if result == 0:
        raise DataError(f"{source} field {key!r} must be positive")
    return result


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _infer_worker_role(
    observations: Sequence[MeasurementObservation],
) -> Literal["prefill", "decode", "aggregated"]:
    kinds = {observation.workload_kind for observation in observations}
    if kinds == {WorkloadKind.PREFILL}:
        return "prefill"
    if kinds == {WorkloadKind.DECODE}:
        return "decode"
    return "aggregated"


def _fallback_warning() -> str:
    return (
        "measurement chronology is not declared; preserving manifest file order and source row order "
        "(file_order_fallback)"
    )


def _override_effects(override: HfCaseOverride | None) -> tuple[str, ...]:
    if override is None:
        return ()
    values: list[str] = []
    for field in ("truth_file_ids", "helper_file_ids", "fpm_artifact_ids", "worker_role", "ordering"):
        value = getattr(override, field)
        if value is None:
            continue
        if isinstance(value, OrderingKind):
            normalized: object = value.value
        elif isinstance(value, tuple):
            normalized = list(value)
        else:
            normalized = value
        values.append(f"{field}={json.dumps(normalized, sort_keys=True, separators=(',', ':'))}")
    return tuple(values)


def _membership_sha256(observations: Sequence[MeasurementObservation]) -> str:
    digest = hashlib.sha256()
    for observation in observations:
        prediction_input = observation.prediction_input()
        payload = {
            "observation_id": observation.observation_id,
            "order": observation.order,
            "source_file_id": observation.source_file_id,
            "source_path": observation.source_path,
            "source_row": observation.source_row,
            "event_time": observation.event_time,
            "actual_ms": observation.actual_ms,
            "workload_kind": observation.workload_kind.value,
            "prediction_input": {
                "configuration_id": prediction_input.configuration_id,
                "iteration_id": prediction_input.iteration_id,
                "fpm_ids": list(prediction_input.fpm_ids),
                "workload_kind": prediction_input.workload_kind.value,
                "rank_payloads": [dict(rank) for rank in prediction_input.rank_payloads],
            },
            "rank_wall_time_s": [rank.wall_time_s for rank in observation.iteration.ranks],
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _configuration_identity(configuration: ConfigurationSnapshot) -> dict[str, str | int]:
    config = configuration.worker_config_record.config
    embedded = config.aic_engine_config or {}
    values: dict[str, Any] = {
        "model_path": configuration.model_id,
        "system": configuration.system,
        "backend": configuration.framework,
        "backend_version": configuration.framework_version,
        "weight_quantization": config.precision.weights,
        "kv_cache_dtype": config.precision.kv_cache,
        "parallel_strategy": configuration.parallel_strategy,
        "tp": config.parallelism.tensor_parallel_size,
        "pp": config.parallelism.pipeline_parallel_size,
        "dp": config.parallelism.attention_dp_size,
        "moe_tp": embedded.get("moe_tp_size"),
        "moe_ep": embedded.get("moe_ep_size"),
        "cp": config.parallelism.context_parallel_size,
        "dcp": config.parallelism.decode_context_parallel_size,
    }
    identity: dict[str, str | int] = {}
    for fpm_field in _FPM_CONFIGURATION_FIELDS:
        value = values[fpm_field]
        if isinstance(value, bool) or not isinstance(value, (str, int)) or (isinstance(value, str) and not value):
            raise DataError(
                f"configuration {configuration.configuration_id} lacks a valid FPM identity field {fpm_field!r}"
            )
        identity[fpm_field] = value
    return identity


def _same_identity_value(value: Any, expected: str | int) -> bool:
    return type(value) is type(expected) and value == expected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConfigurationError(f"cannot hash HF overrides {path}: {exc}") from exc
    return digest.hexdigest()


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataError(f"cannot hash HF dataset file {path}: {exc}") from exc
    return digest.hexdigest()
