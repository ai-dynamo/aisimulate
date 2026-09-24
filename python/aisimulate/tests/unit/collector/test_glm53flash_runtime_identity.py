# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown repair suffixes and cross-runtime evidence must fail admission."""

import hashlib
from pathlib import Path

import pytest
from collector import glm53flash_runtime_identity as identity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", [None, "0.30", "0.30.0+unknown", "0.30.0+glm53kpool.bf5f6b0e689d", "0.31.0"])
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


def test_closed_candidate_cannot_generate_a_source_or_binary_admission():
    assert identity.ADMITTED_VLLM_REPAIRS == {}
    for function in (identity.vllm_source_pins, identity.vllm_runtime_closure, identity.vllm_source_manifest_sha256):
        with pytest.raises(ValueError, match="unqualified"):
            function(identity.VLLM_KPOOL_CANDIDATE, source_manifest())
    assert identity.vllm_unaligned_prefill_admitted("0.30.0") is False
    with pytest.raises(ValueError, match="unqualified"):
        identity.vllm_unaligned_prefill_admitted(identity.VLLM_KPOOL_CANDIDATE)


def test_reviewed_repair_contract_binds_distinct_sources_and_all_native_binaries(monkeypatch):
    # TEST ONLY: exercise the dormant contract. This does not qualify the runtime.
    # TEST_ONLY: summary admission has its own suite; isolate downstream binding.
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
