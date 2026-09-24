# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown repair suffixes and cross-runtime evidence must fail admission."""

import hashlib
import json
from pathlib import Path

import pytest
from collector import glm53flash_runtime_identity as identity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", [None, "0.30", "0.30.0+unknown", "0.30.0+glm53kpool.bf5f6b0e689d.other", "0.31.0"])
def test_candidate_versions_are_not_implicitly_qualified(version):
    with pytest.raises(ValueError, match="unqualified"):
        identity.validate_backend_version("vllm", version)


def test_actual_source_closure_is_required_even_for_stock_version():
    manifest = Path(identity.__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"
    producer = {
        "vllm_package_version": "0.30.0",
        "runtime_source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    assert "vllm/model_executor/layers/sparse_attn_indexer_kpool.py" in identity.validate_vllm_source_identity(
        producer, manifest
    )
    producer["runtime_source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source manifest"):
        identity.validate_vllm_source_identity(producer, manifest)


def test_holdout_runtime_must_equal_calibration_runtime():
    calibration = {"backend_version": "0.30.0"}
    assert identity.validate_runtime_pair("vllm", calibration, calibration) == "0.30.0"
    with pytest.raises(ValueError, match="different native runtime"):
        identity.validate_runtime_pair("vllm", calibration, {"backend_version": "0.30.0+other"})


def source_manifest():
    return Path(identity.__file__).parent / "fpm_forward/runtime/glm53flash/runtime-source-sha256.json"


def test_closed_candidate_cannot_generate_a_source_or_binary_admission(monkeypatch):
    monkeypatch.setattr(identity, "ADMITTED_VLLM_REPAIRS", {})
    for function in (identity.vllm_source_pins, identity.vllm_runtime_closure, identity.vllm_source_manifest_sha256):
        with pytest.raises(ValueError, match="unqualified"):
            function(identity.VLLM_KPOOL_CANDIDATE, source_manifest())
    assert identity.vllm_unaligned_prefill_admitted("0.30.0") is False
    with pytest.raises(ValueError, match="unqualified"):
        identity.vllm_unaligned_prefill_admitted(identity.VLLM_KPOOL_CANDIDATE)


def test_reviewed_repair_contract_binds_distinct_sources_and_all_native_binaries(monkeypatch):
    # TEST_ONLY: isolate downstream binding from the separately tested receipt gate.
    monkeypatch.setattr(identity, "_validate_qualification_summary", lambda _: {})
    monkeypatch.setitem(identity.ADMITTED_VLLM_REPAIRS, identity.VLLM_KPOOL_CANDIDATE, "a" * 64)
    contract = identity.vllm_runtime_closure(identity.VLLM_KPOOL_CANDIDATE, source_manifest())
    stock = identity.vllm_source_pins("0.30.0", source_manifest())
    repaired = identity.vllm_source_pins(identity.VLLM_KPOOL_CANDIDATE, source_manifest())
    assert [name for name in stock if stock[name] != repaired[name]] == [
        "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
    ]
    assert contract["files"]["vllm/vllm-rs"] == "e918038efcc755733f3ffacbe35cd3c133c9d0b146f9ec05a88bcdc2519ff1a3"
    for path in ["model_runner.py", "cudagraph_utils.py", "model_states/mamba_hybrid.py", "warmup.py"]:
        assert "vllm/v1/worker/gpu/" + path in contract["files"]
    assert len(contract["files"]) == len([name for name in repaired if name.startswith("vllm/")]) + 19
    assert contract["runtime_source_manifest_sha256"] != identity.vllm_source_manifest_sha256(
        "0.30.0", source_manifest()
    )
    observation = {"contract_sha256": identity._canonical_sha256(contract), "observed_files": contract["files"]}
    identity.validate_vllm_runtime_closure(identity.VLLM_KPOOL_CANDIDATE, source_manifest(), observation)
    for bad in (None, {**observation, "contract_sha256": "0" * 64}, {**observation, "observed_files": {}}):
        with pytest.raises(ValueError, match="closure"):
            identity.validate_vllm_runtime_closure(identity.VLLM_KPOOL_CANDIDATE, source_manifest(), bad)
    with pytest.raises(ValueError, match="closure"):
        identity.validate_vllm_runtime_closure("0.30.0", source_manifest(), observation)
    with pytest.raises(ValueError, match="source manifest"):
        identity.validate_vllm_source_identity(
            {
                "vllm_package_version": identity.VLLM_KPOOL_CANDIDATE,
                "runtime_source_manifest_sha256": identity.vllm_source_manifest_sha256("0.30.0", source_manifest()),
            },
            source_manifest(),
        )


@pytest.mark.parametrize("corruption", [None, "bytes", "escape", "version", "metadata", "missing"])
def test_worker_closure_reads_real_file_bytes_and_rejects_substitution(tmp_path, monkeypatch, corruption):
    import sys
    from types import SimpleNamespace

    package = tmp_path / "vllm"
    package.mkdir()
    module = package / "__init__.py"
    module.write_text("# TEST ONLY package\n")
    binary = package / "kernel.so"
    binary.write_bytes(b"TEST ONLY native binary bytes")
    contract = {"files": {"vllm/kernel.so": hashlib.sha256(binary.read_bytes()).hexdigest()}}
    monkeypatch.setattr(identity, "vllm_runtime_closure", lambda *_: contract)
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(__file__=str(module), __version__=identity.VLLM_KPOOL_CANDIDATE)
    )
    monkeypatch.setattr(identity.importlib.metadata, "version", lambda _: identity.VLLM_KPOOL_CANDIDATE)
    if corruption == "bytes":
        binary.write_bytes(b"different native binary")
    elif corruption == "escape":
        outside = tmp_path / "external.so"
        outside.write_bytes(binary.read_bytes())
        binary.unlink()
        binary.symlink_to(outside)
    elif corruption == "missing":
        binary.unlink()
    elif corruption == "version":
        sys.modules["vllm"].__version__ = "0.30.0"
    elif corruption == "metadata":
        monkeypatch.setattr(identity.importlib.metadata, "version", lambda _: "0.30.0")
    if corruption is None:
        assert identity.observe_vllm_runtime_closure(identity.VLLM_KPOOL_CANDIDATE, source_manifest()) == {
            "contract_sha256": identity._canonical_sha256(contract),
            "observed_files": contract["files"],
        }
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            identity.observe_vllm_runtime_closure(identity.VLLM_KPOOL_CANDIDATE, source_manifest())


@pytest.mark.parametrize(
    "version",
    [
        identity.VLLM_TAIL_CANDIDATE,
        identity.tail_qualification.VERSIONS["tail_reference"],
        identity.VLLM_TAIL_CANDIDATE + ".other",
    ],
)
def test_new_tail_runtime_stays_closed_at_every_public_identity_entrypoint(version):
    # The real registry is empty; packaged build/CPU/partial functional evidence
    # cannot authorize either the candidate or its diagnostic reference.
    assert identity.ADMITTED_VLLM_REPAIRS == {}
    for function in (identity.vllm_source_pins, identity.vllm_runtime_closure, identity.vllm_source_manifest_sha256):
        with pytest.raises(ValueError, match="unqualified"):
            function(version, source_manifest())
    with pytest.raises(ValueError, match="unqualified"):
        identity.vllm_unaligned_prefill_admitted(version)


def test_tail_registry_entry_still_requires_its_actual_complete_qualification(monkeypatch):
    # TEST_ONLY registry insertion cannot substitute for the missing actual
    # four-cell evidence package. No validator is replaced in this test.
    monkeypatch.setitem(identity.ADMITTED_VLLM_REPAIRS, identity.VLLM_TAIL_CANDIDATE, "a" * 64)
    with pytest.raises((ValueError, FileNotFoundError)):
        identity.validate_backend_version("vllm", identity.VLLM_TAIL_CANDIDATE)


def tail_binding_only(monkeypatch):
    # TEST_ONLY: exercise downstream source/binary binding with the separate
    # qualification dependency isolated. This does not qualify a real runtime.
    seen = []

    def validate(root, *, expected_summary_sha256):
        seen.append((root, expected_summary_sha256))
        return {}

    monkeypatch.setattr(identity.tail_qualification, "validate_tail_qualification", validate)
    monkeypatch.setitem(identity.ADMITTED_VLLM_REPAIRS, identity.VLLM_TAIL_CANDIDATE, "b" * 64)
    return seen


def test_tail_binding_uses_new_build_both_repairs_and_original_v2_sources(monkeypatch):
    seen = tail_binding_only(monkeypatch)
    version = identity.VLLM_TAIL_CANDIDATE
    contract = identity.vllm_runtime_closure(version, source_manifest())
    pins = identity.vllm_source_pins(version, source_manifest())
    assert seen and all(item == (identity._tail_root(), "b" * 64) for item in seen)
    assert contract["wheel_sha256"] == "538d440757a3bbf21b1b6a9f4348fee631789654e10ec8b8f4b13b8aa99b53e2"
    assert contract["build_receipt_sha256"] == "59b012dcd92831405847b91d1fcfdc9f373d01cd8d1b829b54ce96dfcf0da07e"
    assert contract["qualification_receipt_sha256"] == "b" * 64
    assert contract["backend_version"] == version
    assert pins["vllm/v1/kv_cache_interface.py"] == "76fdecf31c8cd93479ebe7c788d699498e83957d18037eb5b39ccfe46f57c138"
    assert pins["vllm/model_executor/layers/sparse_attn_indexer_kpool.py"] == (
        "2aa61ce832e530f07a33ee2884a92bc30fca4f693b83174546020b70dca211e8"
    )
    actual_sources = json.loads((identity._tail_root() / "candidate/expected-source-sha256.json").read_bytes())
    actual_binaries = json.loads((identity._tail_root() / "candidate/expected-native-binaries.json").read_bytes())
    assert len(actual_sources) == 31 and len(actual_binaries) == 19
    assert contract["files"] == actual_sources | identity._v2_source_pins() | actual_binaries
    assert len(contract["files"]) == 52
    assert (
        pins["dynamo/vllm/instrumented_scheduler.py"]
        == identity.vllm_source_pins("0.30.0", source_manifest())["dynamo/vllm/instrumented_scheduler.py"]
    )
    observation = {"contract_sha256": identity._canonical_sha256(contract), "observed_files": contract["files"]}
    identity.validate_vllm_runtime_closure(version, source_manifest(), observation)
    assert identity.vllm_unaligned_prefill_admitted(version) is True
    assert identity.validate_runtime_pair("vllm", {"backend_version": version}, {"backend_version": version}) == version
    with pytest.raises(ValueError, match="different native runtime"):
        identity.validate_runtime_pair("vllm", {"backend_version": version}, {"backend_version": "0.30.0"})
    assert contract["runtime_source_manifest_sha256"] != identity.vllm_source_manifest_sha256(
        "0.30.0", source_manifest()
    )
    # A valid closure for the new wheel cannot qualify the old quarantined wheel.
    with pytest.raises(ValueError, match="unqualified"):
        identity.validate_vllm_runtime_closure(identity.VLLM_KPOOL_CANDIDATE, source_manifest(), observation)


@pytest.mark.parametrize(
    "filename", ["actual-build-receipt.json", "expected-source-sha256.json", "expected-native-binaries.json"]
)
def test_tail_build_and_source_bytes_are_immutable(tmp_path, monkeypatch, filename):
    import shutil

    tail_binding_only(monkeypatch)
    shutil.copytree(identity._tail_root() / "candidate", tmp_path / "candidate")
    changed = tmp_path / "candidate" / filename
    changed.write_bytes(changed.read_bytes() + b"\n")
    monkeypatch.setattr(identity, "_tail_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="SHA256 differs"):
        identity.vllm_runtime_closure(identity.VLLM_TAIL_CANDIDATE, source_manifest())


@pytest.mark.parametrize(
    "path", ["vllm/model_executor/layers/sparse_attn_indexer_kpool.py", "vllm/models/glm5next/nvidia/kda.py"]
)
def test_tail_source_merge_rejects_changed_baseline_instead_of_overwriting_it(tmp_path, monkeypatch, path):
    tail_binding_only(monkeypatch)
    manifest = tmp_path / "runtime-source-sha256.json"
    pins = json.loads(source_manifest().read_bytes())
    assert path in pins
    pins[path] = "0" * 64
    manifest.write_text(json.dumps(pins))
    with pytest.raises(ValueError, match="source base differs|conflicting identities"):
        identity.vllm_source_pins(identity.VLLM_TAIL_CANDIDATE, manifest)


def test_tail_reference_cannot_be_admitted_as_the_production_candidate(monkeypatch):
    reference = identity.tail_qualification.VERSIONS["tail_reference"]
    monkeypatch.setitem(identity.ADMITTED_VLLM_REPAIRS, reference, "b" * 64)
    with pytest.raises(ValueError, match="unqualified"):
        identity.validate_backend_version("vllm", reference)
