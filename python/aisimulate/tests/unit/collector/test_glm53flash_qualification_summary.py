# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY temporary receipts exercise admission without any accepted GPU data."""

import copy
import hashlib
import json
import shutil

import pytest
from collector import glm53flash_runtime_identity as identity

pytestmark = pytest.mark.unit


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(root, name, value):
    path = root / name
    path.write_text(json.dumps(value, indent=2))
    return {"path": name, "sha256": sha(path.read_bytes())}


def load(root, ref):
    return json.loads((root / ref["path"]).read_bytes())


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Never copied into a source tree, publication stage, or HF repository."""
    root = tmp_path / "TEST_ONLY-qualification"
    root.mkdir()
    original = identity._qualification_root()
    for name in (*identity._QUALIFICATION_SOURCES, "expected-runtime.json"):
        shutil.copyfile(original / name, root / name)
    expected = json.loads((root / "expected-runtime.json").read_bytes())
    summary = {
        "schema_version": 1,
        "status": "native_engine_qualification_passed",
        "scope": identity._QUALIFICATION_SCOPE,
        "backend_version": identity.VLLM_KPOOL_CANDIDATE,
        "build_receipt_sha256": identity._BUILD_SHA256,
        "wheel_sha256": identity._WHEEL_SHA256,
        "expected_runtime_sha256": identity._ENGINE_IDENTITY_SHA256,
        "accuracy_acceptance": "NOT_EVALUATED",
        "formal_8_cell_coverage": "NOT_EVALUATED",
        "source_commit": "1" * 40,
        "validator_sources": {name: sha((root / name).read_bytes()) for name in identity._QUALIFICATION_SOURCES},
        "cells": [],
        "fixture_notice": "TEST_ONLY fabricated contracts; never native qualification evidence",
    }
    for checkpoint in ("fp8", "nvfp4"):
        for tp in (2, 4):
            label = f"TEST_ONLY-{checkpoint}-tp{tp}"
            result = {
                "status": "passed",
                "scope": identity._QUALIFICATION_SCOPE,
                "checkpoint": checkpoint,
                "tp": tp,
                "policy": "production",
                "requests_per_mode": 20,
                "greedy_output_tokens_per_request": 32,
                "comparisons": 40,
                "differences": [],
                "native_receipts": [],
                "accuracy_acceptance": "NOT_EVALUATED",
                "formal_8_cell_coverage": "NOT_EVALUATED",
            }
            cell = {"checkpoint": checkpoint, "tp": tp, "profiles": []}
            for runtime_kind, mode in identity._QUALIFICATION_PROFILES:
                profile_name = f"{label}-{runtime_kind}-{mode}"
                args = {
                    "revision": expected["checkpoints"][checkpoint]["revision"],
                    "tokenizer_revision": expected["checkpoints"][checkpoint]["revision"],
                    "tensor_parallel_size": tp,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                    "enable_expert_parallel": False,
                    "enable_prefix_caching": False,
                    "async_scheduling": False,
                    "mamba_cache_mode": "none",
                    "kv_cache_dtype": "fp8_e4m3",
                    "max_model_len": 131079,
                    "max_num_batched_tokens": 16398,
                    "long_prefill_token_threshold": 4097 if mode == "split" else 0,
                    "enforce_eager": False,
                }
                preflight = {
                    "status": "passed",
                    "checkpoint_config_sha256": expected["checkpoints"][checkpoint]["config_sha256"],
                    "public_engine_args": args,
                    "runtime": {
                        "version": expected["versions"][runtime_kind],
                        "loaded_files": {
                            **expected["source_pins"],
                            **expected["native_binaries"],
                            expected["helper_path"]: expected["helper_sha256"][runtime_kind],
                        },
                        "candidate_wheel_sha256": identity._WHEEL_SHA256 if runtime_kind == "candidate" else None,
                    },
                }
                for field, value, source_name in (
                    ("request_identity", "native_assign_request_id_v1", "request-id-source.json"),
                    ("cohort_admission", "native_scheduling_pause_enqueue_v1", "cohort-source.json"),
                ):
                    preflight[field + "_protocol"] = value
                    preflight[field + "_source_sha256"] = {
                        item["path"]: item["sha256"]
                        for item in json.loads((root / source_name).read_bytes())["sources"]
                    }
                profile = {
                    "runtime_kind": runtime_kind,
                    "mode": mode,
                    "raw_uri": f"ssh://test.invalid/TEST_ONLY/{profile_name}/native",
                    "preflight": save(root, profile_name + "-preflight.json", preflight),
                    "original_validator": {
                        "sha256": sha(b"TEST_ONLY historical validator identity"),
                        "source_uri": "https://test.invalid/TEST_ONLY/historical/validate.py",
                    },
                    "revalidation_added_fields": ["checkpoint_identity", "files.native-receipt.json"],
                }
                mapping = {str(i): f"{i}-{i:08x}" for i in range(20)}
                names = [
                    "outputs.jsonl",
                    "effective-native-config.json",
                    "worker-installation.json",
                    "worker-completion.json",
                    "request-id-map.jsonl",
                    "cohort-admission.jsonl",
                    *(
                        f"{prefix}-rank-{rank}.{suffix}"
                        for rank in range(tp)
                        for prefix, suffix in (("worker", "json"), ("forward", "jsonl"), ("prompts", "jsonl"))
                    ),
                ]
                evidence = {
                    "requests": 20,
                    "native_modes": ["FULL", "NONE"],
                    "all_tp_trace_digest": sha(profile_name.encode()),
                    "external_to_native_request_ids": mapping,
                    "request_identity_protocol": "native_assign_request_id_v1",
                    "cohort_admission_protocol": "native_scheduling_pause_enqueue_v1",
                    "actual_prefill_splits": {
                        str(rank): {rid: [[0, 4100]] for rid in mapping.values()} for rank in range(tp)
                    },
                    "files": [{"path": name, "sha256": sha((profile_name + name).encode())} for name in names]
                    + [{"path": "preflight.json", "sha256": profile["preflight"]["sha256"]}],
                }
                receipt = {
                    "status": "native_requests_and_histories_verified",
                    "runtime_kind": runtime_kind,
                    "mode": mode,
                    "checkpoint": checkpoint,
                    "tp": tp,
                    "policy": "production",
                    "evidence": copy.deepcopy(evidence),
                    "correctness_comparison": "NOT_EVALUATED",
                    "accuracy_acceptance": "NOT_EVALUATED",
                }
                profile["native_receipt"] = save(root, profile_name + "-native.json", receipt)
                evidence["checkpoint_identity"] = {"checkpoint": checkpoint, **expected["checkpoints"][checkpoint]}
                evidence["files"].append({"path": "native-receipt.json", "sha256": profile["native_receipt"]["sha256"]})
                result["native_receipts"].append(evidence)
                cell["profiles"].append(profile)
            cell["comparison"] = {
                **save(root, label + "-comparison.json", result),
                "source_uri": f"ssh://test.invalid/TEST_ONLY/{label}/comparison.json",
            }
            summary["cells"].append(cell)
    monkeypatch.setattr(identity, "_qualification_root", lambda: root)
    return root, summary


def admit(staged, monkeypatch):
    root, summary = staged
    ref = save(root, "admission-summary.json", summary)
    monkeypatch.setitem(identity.ADMITTED_VLLM_REPAIRS, identity.VLLM_KPOOL_CANDIDATE, ref["sha256"])
    return identity.validate_backend_version("vllm", identity.VLLM_KPOOL_CANDIDATE)


def rebind(staged, index=0, profile_index=0, *, preflight=None, receipt=None, evidence=None):
    """Rehash the entire TEST_ONLY chain so semantic checks, not stale SHA, fail."""
    root, summary = staged
    cell = summary["cells"][index]
    profile = cell["profiles"][profile_index]
    result = load(root, cell["comparison"])
    proof = result["native_receipts"][profile_index] if evidence is None else evidence
    original = load(root, profile["native_receipt"]) if receipt is None else receipt
    if preflight is not None:
        profile["preflight"] = save(root, profile["preflight"]["path"], preflight)
        for rows in (original["evidence"]["files"], proof["files"]):
            next(r for r in rows if r["path"] == "preflight.json")["sha256"] = profile["preflight"]["sha256"]
    profile["native_receipt"] = save(root, profile["native_receipt"]["path"], original)
    next(r for r in proof["files"] if r["path"] == "native-receipt.json")["sha256"] = profile["native_receipt"][
        "sha256"
    ]
    result["native_receipts"][profile_index] = proof
    cell["comparison"].update(save(root, cell["comparison"]["path"], result))


def test_only_four_complete_profiles_may_reach_test_admission(staged, monkeypatch):
    assert identity.ADMITTED_VLLM_REPAIRS == {}
    assert admit(staged, monkeypatch) == identity.VLLM_KPOOL_CANDIDATE


def test_registry_hash_does_not_admit_a_missing_packaged_summary(monkeypatch):
    monkeypatch.setitem(identity.ADMITTED_VLLM_REPAIRS, identity.VLLM_KPOOL_CANDIDATE, "a" * 64)
    with pytest.raises(ValueError, match="regular JSON"):
        identity.validate_backend_version("vllm", identity.VLLM_KPOOL_CANDIDATE)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "wrong_tp", "wrong_precision"])
def test_exact_four_cells_required(staged, monkeypatch, bad):
    _, summary = staged
    if bad == "missing":
        summary["cells"].pop()
    elif bad == "duplicate":
        summary["cells"][3] = copy.deepcopy(summary["cells"][0])
    elif bad == "wrong_tp":
        summary["cells"][0]["tp"] = 1
    else:
        summary["cells"][0]["checkpoint"] = "bf16"
    with pytest.raises(ValueError, match="exactly four"):
        admit(staged, monkeypatch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wheel_sha256", "f" * 64),
        ("build_receipt_sha256", "f" * 64),
        ("expected_runtime_sha256", "f" * 64),
        ("backend_version", "0.30.0"),
        ("source_commit", "main"),
        ("accuracy_acceptance", "PASSED"),
    ],
)
def test_fixed_summary_identity(staged, monkeypatch, field, value):
    staged[1][field] = value
    with pytest.raises(ValueError):
        admit(staged, monkeypatch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("policy", "eager"),
        ("requests_per_mode", 19),
        ("greedy_output_tokens_per_request", 31),
        ("comparisons", 39),
        ("differences", [{"first_differing_token": 7}]),
        ("tp", 4),
    ],
)
def test_rehashed_comparison_cannot_weaken_qualification(staged, monkeypatch, field, value):
    root, summary = staged
    ref = summary["cells"][0]["comparison"]
    result = load(root, ref)
    result[field] = value
    ref.update(save(root, ref["path"], result))
    with pytest.raises(ValueError, match="production 20x32/40"):
        admit(staged, monkeypatch)


@pytest.mark.parametrize(
    "bad", ["checkpoint", "revision", "tokenizer", "version", "binary", "wheel", "eager", "tp", "protocol"]
)
def test_actual_original_preflight_not_just_receipt_label(staged, monkeypatch, bad):
    root, summary = staged
    profile = summary["cells"][0]["profiles"][0]
    preflight = load(root, profile["preflight"])
    if bad == "checkpoint":
        preflight["checkpoint_config_sha256"] = "f" * 64
    elif bad in {"revision", "tokenizer"}:
        preflight["public_engine_args"]["revision" if bad == "revision" else "tokenizer_revision"] = "f" * 40
    elif bad == "version":
        preflight["runtime"]["version"] = identity.VLLM_KPOOL_CANDIDATE
    elif bad == "binary":
        preflight["runtime"]["loaded_files"]["vllm/vllm-rs"] = "f" * 64
    elif bad == "wheel":
        preflight["runtime"]["candidate_wheel_sha256"] = identity._WHEEL_SHA256
    elif bad == "protocol":
        preflight["cohort_admission_protocol"] = "legacy_generate_sequential_admission"
    else:
        preflight["public_engine_args"]["enforce_eager" if bad == "eager" else "tensor_parallel_size"] = (
            True if bad == "eager" else 4
        )
    rebind(staged, preflight=preflight)
    with pytest.raises(ValueError):
        admit(staged, monkeypatch)


@pytest.mark.parametrize(
    "bad",
    [
        "checkpoint",
        "no_full",
        "missing_rank",
        "duplicate_request",
        "original_trace",
        "extra_rank",
        "missing_raw",
        "duplicate_file",
        "profile_order",
        "history",
    ],
)
def test_rehashed_native_proof_must_preserve_original_chain(staged, monkeypatch, bad):
    root, summary = staged
    cell = summary["cells"][0]
    result = load(root, cell["comparison"])
    proof = result["native_receipts"][0]
    if bad == "checkpoint":
        proof["checkpoint_identity"]["revision"] = "f" * 40
    elif bad == "no_full":
        proof["native_modes"] = ["NONE"]
    elif bad == "missing_rank":
        proof["actual_prefill_splits"].pop("1")
    elif bad == "duplicate_request":
        proof["external_to_native_request_ids"]["0"] = proof["external_to_native_request_ids"]["1"]
    elif bad == "original_trace":
        proof["all_tp_trace_digest"] = "f" * 64
    elif bad == "extra_rank":
        proof["files"].append({"path": "worker-rank-2.json", "sha256": "f" * 64})
    elif bad == "missing_raw":
        proof["files"] = [r for r in proof["files"] if r["path"] != "outputs.jsonl"]
    elif bad == "duplicate_file":
        proof["files"].append(copy.deepcopy(proof["files"][0]))
    elif bad == "profile_order":
        cell["profiles"].reverse()
    else:
        cell["profiles"][0]["revalidation_added_fields"] = []
    cell["comparison"].update(save(root, cell["comparison"]["path"], result))
    with pytest.raises(ValueError):
        admit(staged, monkeypatch)


@pytest.mark.parametrize("which", ["summary", "comparison", "preflight", "native_receipt", "source"])
def test_original_packaged_bytes_are_rehashed(staged, monkeypatch, which):
    root, summary = staged
    admit(staged, monkeypatch)
    cell = summary["cells"][0]
    path = root / (
        "admission-summary.json"
        if which == "summary"
        else "validate.py"
        if which == "source"
        else cell["comparison"]["path"]
        if which == "comparison"
        else cell["profiles"][0][which]["path"]
    )
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256|source bytes"):
        identity.validate_backend_version("vllm", identity.VLLM_KPOOL_CANDIDATE)


@pytest.mark.parametrize(
    "bad", ["traversal", "symlink", "credential", "query", "raw_alias", "receipt_alias", "historical_source"]
)
def test_unsafe_or_ambiguous_source_identity_rejected(staged, monkeypatch, bad):
    root, summary = staged
    profile = summary["cells"][0]["profiles"][0]
    if bad == "traversal":
        profile["preflight"]["path"] = "../outside.json"
    elif bad == "symlink":
        path = root / profile["preflight"]["path"]
        outside = root.parent / "TEST_ONLY-outside.json"
        path.rename(outside)
        path.symlink_to(outside)
    elif bad in {"credential", "query"}:
        profile["raw_uri"] = (
            "ssh://secret@test.invalid/path" if bad == "credential" else "https://test.invalid/path?token=secret"
        )
    elif bad == "raw_alias":
        summary["cells"][0]["profiles"][1]["raw_uri"] = profile["raw_uri"]
    elif bad == "receipt_alias":
        # Duplicate references alone fail before an inconsistent later profile can replace them.
        profile["native_receipt"]["path"] = profile["preflight"]["path"]
    else:
        profile["original_validator"]["sha256"] = "unrecorded"
    with pytest.raises(ValueError):
        admit(staged, monkeypatch)


def test_current_original_receipts_need_no_retrospective_checkpoint_addition(staged, monkeypatch):
    root, summary = staged
    cell = summary["cells"][0]
    proof = load(root, cell["comparison"])["native_receipts"][0]
    receipt = load(root, cell["profiles"][0]["native_receipt"])
    receipt["evidence"]["checkpoint_identity"] = proof["checkpoint_identity"]
    cell["profiles"][0]["revalidation_added_fields"] = ["files.native-receipt.json"]
    rebind(staged, receipt=receipt)
    assert admit(staged, monkeypatch) == identity.VLLM_KPOOL_CANDIDATE
