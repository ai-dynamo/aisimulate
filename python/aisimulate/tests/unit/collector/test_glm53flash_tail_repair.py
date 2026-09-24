# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Package provenance checks; these are not native model qualification tests."""

import base64
import hashlib
import json
from pathlib import Path

import pytest
from collector.glm53flash_runtime_identity import ADMITTED_VLLM_REPAIRS

ROOT = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime/glm53flash_vllm_tail_repair"
TAIL_PATH = "vllm/v1/kv_cache_interface.py"
HELPER_PATH = "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
BASE_REVISION = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", ["candidate", "reference"])
def test_packaged_tail_patch_and_actual_install_lineage(kind):
    root = ROOT / kind
    identity = load(root / "patch-identity.json")
    build = load(root / "actual-build-receipt.json")
    install = load(root / "actual-install-receipt.json")
    (patch_file,) = root.glob("*.patch.b64")
    patch = base64.b64decode(patch_file.read_bytes(), validate=False)
    assert hashlib.sha256(patch).hexdigest() == identity["patch_sha256"] == build["patch_sha256"]
    assert build["patch"] == identity
    assert identity["base_revision"] == BASE_REVISION
    assert install["build_receipt_sha256"] == digest(root / "actual-build-receipt.json")
    assert install["version"] == build["version"] == identity["version"]
    assert install["wheel_sha256"] == build["wheel_sha256"]
    assert install["status"] == "ACTUAL_CPU_INSTALL_SOURCE_AND_NATIVE_SPEC_PASS"
    assert install["pytorch_cuda_initialized"] is False
    assert install["tail_uses_slot_mapping"] is False
    assert install["ordinary_sliding_window_uses_slot_mapping"] is True
    sources = load(root / "expected-source-sha256.json")
    binaries = load(root / "expected-native-binaries.json")
    assert len(sources) == 31 and len(binaries) == 19
    assert install["observed_files"] == sources | binaries
    assert build["unchanged_native_binaries"] == binaries
    assert build["preserved_binary_base_legal_files"]
    assert sources[TAIL_PATH] == "76fdecf31c8cd93479ebe7c788d699498e83957d18037eb5b39ccfe46f57c138"
    assert {entry["source_path"] for entry in identity["sources"]} == {TAIL_PATH, HELPER_PATH}
    assert {
        entry["source_path"] for entry in identity["sources"] if entry["base_sha256"] != entry["patched_sha256"]
    } == ({TAIL_PATH, HELPER_PATH} if kind == "candidate" else {TAIL_PATH})
    for entry in identity["sources"]:
        assert sources[entry["source_path"]] == entry["patched_sha256"]
    assert b"contributors to the vLLM project" in patch
    assert b"Apache-2.0" in patch
    assert b"NVIDIA" in patch


def test_reference_changes_no_pooling_helper_or_native_binary():
    candidate = load(ROOT / "candidate/actual-install-receipt.json")["observed_files"]
    reference = load(ROOT / "reference/actual-install-receipt.json")["observed_files"]
    assert candidate.keys() == reference.keys()
    assert [name for name in candidate if candidate[name] != reference[name]] == [HELPER_PATH]
    assert reference[HELPER_PATH] == "625d30d17ad3c66164f83fea59fc86e0e5edc53b4642e8a3a724aa90276f98ab"
    assert candidate[HELPER_PATH] == "2aa61ce832e530f07a33ee2884a92bc30fca4f693b83174546020b70dca211e8"


@pytest.mark.parametrize("kind", ["candidate", "reference"])
def test_packaged_inputs_retain_original_executed_bytes_or_explicit_formatting_lineage(kind):
    root = ROOT / kind
    original = load(root / "original-build-inputs.json")
    lineage = load(root / "packaging-lineage.json")
    assert lineage["original_input_inventory_sha256"] == digest(root / "original-build-inputs.json")
    assert lineage["runtime_patch_bytes_changed"] is False
    assert len(lineage["formatted_files"]) == 1
    (formatted,) = lineage["formatted_files"]
    assert formatted["file"] == "verify_install.py"
    assert formatted["original_sha256"] == original["files"][formatted["file"]]
    assert formatted["packaged_sha256"] == digest(root / formatted["file"])
    for name, expected in original["files"].items():
        if name != formatted["file"]:
            assert digest(root / name) == expected


def test_cpu_build_receipts_do_not_open_production_admission():
    status = load(ROOT / "qualification-status.json")
    assert status["production_admission"] == "CLOSED"
    assert status["model_correctness"] == "NOT_EVALUATED"
    assert status["formal_fpm_accuracy"] == status["formal_ops_accuracy"] == "NOT_EVALUATED"
    for kind in ("candidate", "reference"):
        identity = load(ROOT / kind / "patch-identity.json")
        assert identity["formal_runtime_admission"] == "CLOSED"
        assert identity["version"] not in ADMITTED_VLLM_REPAIRS
