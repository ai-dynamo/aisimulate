# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY fixtures ensure runtime proofs cannot relabel another native run."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector import glm53flash_runtime_identity as identity
from collector.fpm_forward import database, glm53flash_validation, native_artifact, runner
from collector.fpm_forward.hybrid_artifact import PROTOCOL, validate_vllm_hardware_receipts
from tests.unit.collector.test_fpm_glm53flash_hardware import artifact

pytestmark = pytest.mark.unit
CANDIDATE = identity.VLLM_KPOOL_CANDIDATE


def chain(tmp_path, monkeypatch, outer, actual):
    # Process-local test admission only; no source or published data is modified.
    # TEST_ONLY: this test isolates downstream runtime binding; summary validation has its own suite.
    monkeypatch.setattr(identity, "_validate_qualification_summary", lambda _: {})
    monkeypatch.setitem(identity.ADMITTED_VLLM_REPAIRS, CANDIDATE, "a" * 64)
    pod = tmp_path / "pod"
    pod.mkdir()
    cell, payload, path = artifact(pod)
    cell.cell_id, cell.backend, cell.state_protocol = "TEST_ONLY-cell", "vllm", PROTOCOL
    provenance = {
        "schema_name": "aic_fpm_collector_provenance",
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "plan_sha256": "b" * 64,
        "attempt_id": "TEST_ONLY-attempt",
        "runtime": {"backend": "vllm", "backend_version": outer},
    }
    provenance_path = pod / "collector-provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    provenance_sha = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    manifest = Path(identity.__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"
    payload["producer"].update(
        vllm_package_version=actual,
        runtime_source_manifest_sha256=identity.vllm_source_manifest_sha256(actual, manifest),
    )
    closure = identity.vllm_runtime_closure(actual, manifest)
    for entry in payload["input_provenance"]["native_hardware_manifest"]:
        hardware_path = pod / entry["file"]
        receipt = json.loads(hardware_path.read_bytes())
        receipt.update(backend_version=actual, collector_provenance_sha256=provenance_sha)
        if closure:
            receipt["runtime_closure"] = {
                "contract_sha256": identity._canonical_sha256(closure),
                "observed_files": closure["files"],
            }
        hardware_path.write_text(json.dumps(receipt))
        entry["sha256"] = hashlib.sha256(hardware_path.read_bytes()).hexdigest()
    return cell, payload, path


def read_provenance(cell, payload, path, expected=None):
    return native_artifact._validate_collector_provenance(
        cell,
        path.parent.parent,
        [(path, payload)],
        expected_plan_sha256="b" * 64,
        expected_attempt_id="TEST_ONLY-attempt",
        expected_backend_version=expected,
    )


@pytest.mark.parametrize("version", ["0.30.0", CANDIDATE])
@pytest.mark.parametrize("has_plan", [False, True])
def test_matching_actual_runtime_chain(tmp_path, monkeypatch, version, has_plan):
    cell, payload, path = chain(tmp_path, monkeypatch, version, version)
    assert read_provenance(cell, payload, path, version if has_plan else None) == (version, "TEST_ONLY-attempt")
    validate_vllm_hardware_receipts(cell, payload, path)


@pytest.mark.parametrize("outer,actual", [("0.30.0", CANDIDATE), (CANDIDATE, "0.30.0")])
def test_rehashed_collector_provenance_cannot_relabel_actual_workers(tmp_path, monkeypatch, outer, actual):
    cell, payload, path = chain(tmp_path, monkeypatch, outer, actual)
    with pytest.raises(ValueError, match="producer differs from Collector runtime"):
        read_provenance(cell, payload, path)
    with pytest.raises(ValueError, match="hardware producer differs from Collector runtime"):
        validate_vllm_hardware_receipts(cell, payload, path)
    # Public reader must enforce the crossbind even without an external plan.
    monkeypatch.setattr(native_artifact, "_rank_artifacts", lambda _: [(path, payload)])
    with pytest.raises(ValueError, match="producer differs from Collector runtime"):
        native_artifact.validate_native_collection(cell, path.parent.parent)


@pytest.mark.parametrize("actual,expected", [("0.30.0", CANDIDATE), (CANDIDATE, "0.30.0")])
def test_consistent_actual_runtime_cannot_replace_frozen_plan_version(tmp_path, monkeypatch, actual, expected):
    cell, payload, path = chain(tmp_path, monkeypatch, actual, actual)
    monkeypatch.setattr(native_artifact, "_rank_artifacts", lambda _: [(path, payload)])
    with pytest.raises(ValueError, match="frozen plan backend version"):
        native_artifact.validate_native_collection(cell, path.parent.parent, expected_backend_version=expected)


@pytest.mark.parametrize("producer", [None, {}, {"vllm_package_version": "0.30.0+unknown"}])
def test_missing_or_unqualified_producer_is_not_aliased(tmp_path, monkeypatch, producer):
    cell, payload, path = chain(tmp_path, monkeypatch, "0.30.0", "0.30.0")
    payload["producer"] = producer
    with pytest.raises(ValueError, match="producer differs"):
        read_provenance(cell, payload, path)


@pytest.mark.parametrize("outer,actual", [("0.5.20", "0.5.20"), ("0.5.20", "0.5.21"), ("0.5.21", "0.5.21")])
def test_sglang_uses_same_exact_runtime_binding(tmp_path, outer, actual):
    pod = tmp_path / "pod"
    pod.mkdir()
    cell = SimpleNamespace(cell_id="TEST_ONLY-cell", backend="sglang", state_protocol=PROTOCOL)
    (pod / "collector-provenance.json").write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "plan_sha256": "b" * 64,
                "attempt_id": "TEST_ONLY-attempt",
                "runtime": {"backend": "sglang", "backend_version": outer},
            }
        )
    )
    payload = {"producer": {"backend": "sglang", "backend_version": actual}}
    if outer == actual == "0.5.20":
        assert read_provenance(cell, payload, pod / "benchmark.json", "0.5.20")[0] == actual
    else:
        with pytest.raises(ValueError, match="unqualified|producer differs"):
            read_provenance(cell, payload, pod / "benchmark.json")


@pytest.mark.parametrize("entrypoint", ["aggregate", "runner", "acceptance"])
def test_formal_entrypoints_supply_frozen_runtime_version(tmp_path, monkeypatch, entrypoint):
    class ReachedReader(Exception):
        pass

    def capture(*args, **kwargs):
        assert kwargs["expected_backend_version"] == CANDIDATE
        raise ReachedReader

    cell = SimpleNamespace(cell_id="TEST_ONLY-cell", state_protocol=PROTOCOL)
    plan = SimpleNamespace(sha256="b" * 64, capability=SimpleNamespace(aic_database_version=CANDIDATE))
    with pytest.raises(ReachedReader):
        if entrypoint == "aggregate":
            monkeypatch.setattr(database, "_validate_backend_markers", lambda *_: None)
            monkeypatch.setattr(database, "validate_native_collection", capture)
            database.aggregate_cell(plan, cell, tmp_path, expected_attempt_id="TEST_ONLY-attempt")
        elif entrypoint == "runner":
            monkeypatch.setattr(runner, "validate_native_collection", capture)
            runner._runtime_collection_summary(cell, tmp_path, expected_backend_version=CANDIDATE)
        else:
            monkeypatch.setattr(glm53flash_validation, "validate_native_collection", capture)
            glm53flash_validation._native_run(
                {
                    "spec": {"raw_root": ".", "attempt_id": "TEST_ONLY-attempt"},
                    "plan": {"sha256": "b" * 64, "capability": {"aic_database_version": CANDIDATE}},
                    "runtime_cell": cell,
                },
                tmp_path,
            )


@pytest.mark.parametrize("version", [None, "", False])
def test_formal_acceptance_requires_plan_runtime_identity(tmp_path, version):
    run = {
        "spec": {"raw_root": ".", "attempt_id": "TEST_ONLY-attempt"},
        "plan": {"sha256": "b" * 64, "capability": {"aic_database_version": version}},
    }
    with pytest.raises(ValueError, match="formal acceptance requires the frozen plan"):
        glm53flash_validation._native_run(run, tmp_path)
