# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY proposed packages; no synthetic receipt is production evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from collector import glm53flash_tail_qualification as gate
from collector.glm53flash_runtime_identity import ADMITTED_VLLM_REPAIRS, validate_backend_version

pytestmark = pytest.mark.unit
PACKAGED = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime/glm53flash_vllm_tail_repair"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Package:
    def __init__(self, root):
        self.root = root

    def put(self, name, value, *, uri=None, raw=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw if raw is not None else (json.dumps(value, indent=2) + "\n").encode())
        return {"path": name, "sha256": digest(path), "source_uri": uri or "https://test.invalid/TEST_ONLY/" + name}

    def read(self, ref):
        return json.loads((self.root / ref["path"]).read_bytes())

    def replace(self, ref, value):
        ref.update(self.put(ref["path"], value, uri=ref["source_uri"]))

    def sources(self, pins):
        return {
            name: {"sha256": sha, "source_uri": "https://test.invalid/TEST_ONLY/source/" + name}
            for name, sha in pins.items()
        }

    def derive(self, name, scope, original, result, inputs, sources):
        return self.put(
            name,
            {
                "schema": "glm53flash_tail_revalidation_v1",
                "status": "passed",
                "scope": scope,
                "original_result_sha256": original["sha256"],
                "derived_result_sha256": result["sha256"],
                "input_files": inputs,
                "consumer_reread_external_raw": False,
                "validator_sources": self.sources(sources),
                "assembler_source": {"sha256": "a" * 64, "source_uri": "https://test.invalid/TEST_ONLY/assembler.py"},
            },
        )

    def run(self):
        ref = self.put("qualification/admission-summary.json", self.summary)
        return gate.validate_tail_qualification(self.root, expected_summary_sha256=ref["sha256"])


def runtime(kind, closures):
    return {
        "version": gate.VERSIONS[kind],
        "runtime_wheel_sha256": gate.WHEELS[kind],
        "package_root": "/TEST_ONLY",
        "loaded_files": closures[kind],
    }


def group(rank, closure):
    names = [f"language_model.model.layers.{layer}.self_attn.indexer.tail_cache" for layer in gate.TAIL_LAYERS]
    spec = {
        "class": "vllm.v1.kv_cache_interface.KpoolTailSpec",
        "uses_slot_mapping": False,
        "fields": {"block_size": 4, "sliding_window": 4},
    }
    paths = (
        gate.TAIL_PATH,
        "vllm/v1/attention/backends/mla/indexer.py",
        "vllm/v1/worker/gpu/model_states/default.py",
        "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
        "vllm/v1/worker/gpu/attn_utils.py",
    )

    def method(path):
        return {"path": "/TEST_ONLY/" + path, "relative_path": path, "sha256": closure[path]}

    return {
        "tp_rank": rank,
        "source_pins": {name: closure[name] for name in paths},
        "model_state_class": "vllm.v1.worker.gpu.model_states.mamba_hybrid.MambaHybridModelState",
        "model_state_prepare_source": method("vllm/v1/worker/gpu/model_states/mamba_hybrid.py"),
        "slot_mapping_enabled_host": [False],
        "slot_mapping_enabled_device": [False],
        "groups": [
            {
                "group_index": 0,
                "per_layer_specs": dict.fromkeys(names, spec),
                "tail_builders": [
                    {
                        "group_index": 0,
                        "builder_index": 0,
                        "builder_class": "vllm.v1.attention.backends.mla.indexer.KpoolTailMetadataBuilder",
                        "builder_spec": spec,
                        "layer_names": names,
                        "build_source": method("vllm/v1/attention/backends/mla/indexer.py"),
                    }
                ],
            }
        ],
    }


def tail(rank, group_sha, addresses, forwards=320):
    return {
        "tp_rank": rank,
        "forwards": forwards,
        "tail_layers": gate.TAIL_LAYERS,
        "checked_actual_token_addresses": addresses,
        "cache_groups_sha256": group_sha,
        "status": "ACTUAL_NATIVE_TAIL_ADDRESSES_AND_POSITIONS_VERIFIED",
        "latency_admission": False,
    }


def inventory(mapping):
    return [{"path": name, "sha256": sha} for name, sha in sorted(mapping.items())]


@pytest.fixture
def package(tmp_path, monkeypatch):
    p = Package(tmp_path)
    builds = {}
    for kind in gate.VERSIONS:
        dirname = "reference" if kind == "tail_reference" else kind
        names = {
            "build": "actual-build-receipt.json",
            "install": "actual-install-receipt.json",
            "packaged_verifier": "actual-packaged-verifier-receipt.json",
            "packaged_launch": "actual-packaged-verifier-launch.json",
            "sources": "expected-source-sha256.json",
            "binaries": "expected-native-binaries.json",
        }
        builds[kind] = {
            key: p.put(dirname + "/" + name, None, raw=(PACKAGED / dirname / name).read_bytes())
            for key, name in names.items()
        }
        builds[kind]["executed_inputs"] = p.put(
            "packaged-verifier-executed-inputs.json",
            None,
            raw=(PACKAGED / "packaged-verifier-executed-inputs.json").read_bytes(),
        )
    closures = gate._build_profiles(tmp_path, builds)
    expected_ref = p.put(
        "qualification/expected-runtime.json", None, raw=(PACKAGED / "qualification/expected-runtime.json").read_bytes()
    )
    expected = p.read(expected_ref)
    summary = {
        "schema_name": "glm53flash_tail_runtime_qualification",
        "schema_version": 1,
        "status": "bounded_native_qualification_passed",
        "scope": gate.SCOPE,
        "candidate_version": gate.VERSIONS["candidate"],
        "reference_version": gate.VERSIONS["tail_reference"],
        "formal_fpm_accuracy": "NOT_EVALUATED",
        "formal_ops_accuracy": "NOT_EVALUATED",
        "formal_8_cell_coverage": "NOT_EVALUATED",
        "http_acceptance": "NOT_EVALUATED",
        "per_cell_capacity_admission": "SEPARATE_NATIVE_NINE_POINT_QUALIFICATION_REQUIRED",
        "build_profiles": builds,
        "validator_identity": {
            "functional_sources": p.sources(gate.FUNCTIONAL_SOURCES),
            "expected_runtime": expected_ref,
            "historical_unused_field": {
                "field": "expected-runtime.json.build_receipt_sha256",
                "value": gate.HISTORICAL_BUILD,
                "meaning": "inherited_old_kpool_build_unused_for_tail_admission",
            },
        },
        "functional_cells": [],
    }
    for checkpoint in ("fp8", "nvfp4"):
        for tp in (2, 4):
            cell = {"checkpoint": checkpoint, "tp": tp, "profiles": []}
            comparison = {
                "status": "passed",
                "scope": "native_Engine_functional_correctness_for_frozen_geometry_suite_only",
                "checkpoint": checkpoint,
                "tp": tp,
                "policy": "production",
                "requests_per_mode": 20,
                "greedy_output_tokens_per_request": 32,
                "comparisons": 40,
                "differences": [],
                "accuracy_acceptance": "NOT_EVALUATED",
                "formal_8_cell_coverage": "NOT_EVALUATED",
                "native_receipts": [],
            }
            for kind, mode in gate.PROFILES:
                name = f"qualification/TEST_ONLY-{checkpoint}-{tp}-{kind}-{mode}"
                raw_uri = "https://test.invalid/" + name
                pin = expected["checkpoints"][checkpoint]
                args = {
                    "revision": pin["revision"],
                    "tokenizer_revision": pin["revision"],
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
                pre = p.put(
                    name + "/preflight.json",
                    {
                        "status": "passed",
                        "runtime": runtime(kind, closures),
                        "checkpoint_config_sha256": pin["config_sha256"],
                        "public_engine_args": args,
                    },
                )
                raw = {
                    "preflight.json": pre["sha256"],
                    **dict.fromkeys(
                        (
                            "outputs.jsonl",
                            "effective-native-config.json",
                            "worker-installation.json",
                            "worker-completion.json",
                            "request-id-map.jsonl",
                            "cohort-admission.jsonl",
                        ),
                        "b" * 64,
                    ),
                }
                workers, groups, tails, splits = {}, {}, [], {}
                mapping = {f"TEST_ONLY-external-{index}": f"TEST_ONLY-native-{index}" for index in range(20)}
                for rank in range(tp):
                    worker = p.put(
                        f"{name}/worker-rank-{rank}.json",
                        {
                            "tp_rank": rank,
                            "runtime": runtime(kind, closures),
                            "hardware": {
                                "compute_capability": [10, 3],
                                "name": "TEST_ONLY GB300",
                                "uuid": f"TEST_ONLY-{rank}",
                            },
                        },
                    )
                    grp = p.put(f"{name}/cache-groups-rank-{rank}.json", group(rank, closures[kind]))
                    workers[str(rank)], groups[str(rank)] = worker, grp
                    raw[f"worker-rank-{rank}.json"], raw[f"cache-groups-rank-{rank}.json"] = (
                        worker["sha256"],
                        grp["sha256"],
                    )
                    raw.update(
                        {f"{prefix}-rank-{rank}.jsonl": "c" * 64 for prefix in ("prompts", "forward", "tail-metadata")}
                    )
                    tails.append(tail(rank, grp["sha256"], sum(gate.PROMPT_LENGTHS) + 620))
                    splits[str(rank)] = {
                        f"TEST_ONLY-native-{index}": [[0, n]]
                        if mode == "reference"
                        else (
                            [[0, 4097], [4097, n - 4097]] if n < 8194 else [[0, 4097], [4097, 4097], [8194, n - 8194]]
                        )
                        for index, n in enumerate(gate.PROMPT_LENGTHS)
                    }
                evidence = {
                    "requests": 20,
                    "checkpoint_identity": {"checkpoint": checkpoint, **pin},
                    "native_modes": ["FULL", "NONE"],
                    "all_tp_trace_digest": "d" * 64,
                    "external_to_native_request_ids": mapping,
                    "request_identity_protocol": "native_assign_request_id_v1",
                    "cohort_admission_protocol": "native_scheduling_pause_enqueue_v1",
                    "actual_prefill_splits": splits,
                    "native_tail_evidence": tails,
                    "files": inventory(raw),
                }
                receipt = p.put(
                    name + "/native-receipt.json",
                    {
                        "status": "native_requests_and_histories_verified",
                        "runtime_kind": kind,
                        "mode": mode,
                        "checkpoint": checkpoint,
                        "tp": tp,
                        "policy": "production",
                        "correctness_comparison": "NOT_EVALUATED",
                        "accuracy_acceptance": "NOT_EVALUATED",
                        "evidence": evidence,
                    },
                )
                derived = copy.deepcopy(evidence)
                derived["files"] = inventory(raw | {"native-receipt.json": receipt["sha256"]})
                cell["profiles"].append(
                    {
                        "runtime_kind": kind,
                        "mode": mode,
                        "raw_uri": raw_uri,
                        "original_validator": p.sources({"validate.py": gate.FUNCTIONAL_SOURCES["validate.py"]})[
                            "validate.py"
                        ],
                        "preflight": pre,
                        "native_receipt": receipt,
                        "workers": workers,
                        "cache_groups": groups,
                        "revalidation_added_fields": ["files.native-receipt.json"],
                    }
                )
                comparison["native_receipts"].append(derived)
            base = f"qualification/TEST_ONLY-{checkpoint}-{tp}"
            cell["original_comparison"] = p.put(base + "-original.json", comparison)
            cell["comparison"] = p.put(base + "-derived.json", comparison)
            inputs = [
                {"raw_uri": profile["raw_uri"], "files": proof["files"]}
                for profile, proof in zip(cell["profiles"], comparison["native_receipts"], strict=True)
            ]
            cell["revalidation"] = p.derive(
                base + "-revalidation.json",
                "four_cell_functional_profile_comparison",
                cell["original_comparison"],
                cell["comparison"],
                inputs,
                gate.FUNCTIONAL_SOURCES,
            )
            summary["functional_cells"].append(cell)
    oracle_uri = "https://test.invalid/TEST_ONLY-oracle"
    rt = p.put("qualification/TEST_ONLY-oracle-runtime.json", runtime("candidate", closures))
    raw = {"actual-runtime.json": rt["sha256"]}
    cases = []
    for index, (prefix, query) in enumerate(gate.ORACLE_CASES):
        for flag in (False, True):
            sha = hashlib.sha256(f"TEST_ONLY-{index}-{flag}".encode()).hexdigest()
            cases.append(
                {
                    "prefixes": list(prefix),
                    "queries": list(query),
                    "nonuniform_gate": flag,
                    "one_shot_matches_oracle": True,
                    "split_cache_matches_one_shot": True,
                    "tail_matches_one_shot": True,
                    "snapshot_sha256": sha,
                }
            )
            raw[f"case-{index}-gate-{int(flag)}.pt"] = sha
    result = p.put(
        "qualification/TEST_ONLY-oracle-result.json",
        {
            "actual_runtime": runtime("candidate", closures),
            "formal_admission": False,
            "performance_data": False,
            "model_correctness": "NOT_EVALUATED",
            "device": "TEST_ONLY GB300",
            "cases": cases,
        },
    )
    raw["probe-result.json"] = result["sha256"]
    rows = inventory(raw)
    summary["cache_oracle"] = {
        "raw_uri": oracle_uri,
        "result": result,
        "runtime": rt,
        "raw_inventory": rows,
        "revalidation": p.derive(
            "qualification/TEST_ONLY-oracle-revalidation.json",
            "original_16_case_cache_oracle",
            result,
            result,
            [{"raw_uri": oracle_uri, "files": rows}],
            gate.ORACLE_SOURCES,
        ),
    }
    # TEST_ONLY source data replaces only these two pinned long-input files;
    # real four-cell/profile/build validation remains enabled throughout.
    lineage = p.put("qualification/TEST_ONLY-original-lineage.json", {"TEST_ONLY": "lineage"})
    args = p.put("qualification/TEST_ONLY-original-args.json", {"TEST_ONLY": "args"})
    monkeypatch.setitem(gate.LONG_SOURCES, "receipt.json", lineage["sha256"])
    monkeypatch.setitem(gate.LONG_SOURCES, "actual-engine-args.json", args["sha256"])
    pre = p.put(
        "qualification/TEST_ONLY-long-preflight.json",
        {
            "runtime": runtime("candidate", closures),
            "diagnostic_environment": {"CUDA_LAUNCH_BLOCKING": "1"},
            "input_lineage": p.read(lineage),
            "engine_args": {
                **p.read(args),
                "scheduler_cls": None,
                "worker_extension_cls": "worker_probe.QualificationWorker",
            },
        },
    )
    output = p.put("qualification/TEST_ONLY-long-output.json", {"token_ids": [123]})
    raw = {
        "preflight.json": pre["sha256"],
        "native-output.json": output["sha256"],
        "native-request-id-mapping.jsonl": "e" * 64,
    }
    tails, workers, groups = [], {}, {}
    for rank in range(2):
        raw.update(
            {
                f"witness/{prefix}-rank-{rank}.{suffix}": "f" * 64
                for prefix, suffix in (
                    ("forward", "jsonl"),
                    ("worker", "json"),
                    ("cache-groups", "json"),
                    ("tail-metadata", "jsonl"),
                )
            }
        )
        worker = p.put(
            f"qualification/TEST_ONLY-long-worker-{rank}.json",
            {"tp_rank": rank, "runtime": runtime("candidate", closures)},
        )
        grp = p.put(f"qualification/TEST_ONLY-long-group-{rank}.json", group(rank, closures["candidate"]))
        workers[str(rank)], groups[str(rank)] = worker, grp
        raw[f"witness/worker-rank-{rank}.json"] = worker["sha256"]
        raw[f"witness/cache-groups-rank-{rank}.json"] = grp["sha256"]
        tails.append(tail(rank, grp["sha256"], 131072, 16))
    result = p.put(
        "qualification/TEST_ONLY-long-result.json",
        {
            "status": "FRESH_NATIVE_REQUEST_COMPLETED_WITH_ACTUAL_LONG_SEED",
            "formal_admission": False,
            "accuracy_acceptance": "NOT_EVALUATED",
            "latency_admission": False,
            "actual_tail_evidence": tails,
            "history": [
                {
                    "tp_rank": rank,
                    "completed_forwards": 16,
                    "prefill_tokens": 131072,
                    "seed_16384_8192_completed": True,
                    "original_failure_24576_8192_completed": True,
                }
                for rank in range(2)
            ],
        },
    )
    raw["result.json"] = result["sha256"]
    rows = inventory(raw)
    uri = "https://test.invalid/TEST_ONLY-long"
    summary["ordinary_128k"] = {
        "raw_uri": uri,
        "checkpoint": "nvfp4",
        "tp": 2,
        "original_token_file_sha256": gate.LONG_SOURCES["reconstructed-point4-request0.tokens.json"],
        "original_lineage": lineage,
        "original_engine_args": args,
        "preflight": pre,
        "result": result,
        "workers": workers,
        "cache_groups": groups,
        "output": output,
        "raw_inventory": rows,
        "revalidation": p.derive(
            "qualification/TEST_ONLY-long-revalidation.json",
            "original_nvfp4_tp2_131072_request",
            result,
            result,
            [{"raw_uri": uri, "files": rows}],
            gate.LONG_SOURCES,
        ),
    }
    p.summary = summary
    return p


def test_complete_test_only_package_does_not_open_runtime_registry(package):
    before = dict(ADMITTED_VLLM_REPAIRS)
    assert package.run()["scope"] == gate.SCOPE
    assert ADMITTED_VLLM_REPAIRS == before == {}
    for version in gate.VERSIONS.values():
        with pytest.raises(ValueError, match="unqualified"):
            validate_backend_version("vllm", version)


@pytest.mark.parametrize("replacement", [False, 0.0])
def test_oracle_zero_prefix_requires_integer_even_with_consistent_hashes(package, replacement):
    section = package.summary["cache_oracle"]
    result = package.read(section["result"])
    for row in result["cases"]:
        row["prefixes"] = [replacement if value == 0 else value for value in row["prefixes"]]
    package.replace(section["result"], result)
    for row in section["raw_inventory"]:
        if row["path"] == "probe-result.json":
            row["sha256"] = section["result"]["sha256"]
    derived = package.read(section["revalidation"])
    derived["original_result_sha256"] = derived["derived_result_sha256"] = section["result"]["sha256"]
    derived["input_files"] = [{"raw_uri": section["raw_uri"], "files": section["raw_inventory"]}]
    package.replace(section["revalidation"], derived)
    with pytest.raises(ValueError, match="oracle geometry must contain exact integer"):
        package.run()


@pytest.mark.parametrize(
    "name,target", [("cache_oracle", "profile"), ("ordinary_128k", "profile"), ("ordinary_128k", "cache_oracle")]
)
def test_regression_raw_directories_cannot_alias_other_original_runs(package, name, target):
    section = package.summary[name]
    section["raw_uri"] = (
        package.summary["functional_cells"][0]["profiles"][0]["raw_uri"]
        if target == "profile"
        else package.summary[target]["raw_uri"]
    )
    derived = package.read(section["revalidation"])
    derived["input_files"] = [{"raw_uri": section["raw_uri"], "files": section["raw_inventory"]}]
    package.replace(section["revalidation"], derived)
    with pytest.raises(ValueError, match="aliases another original raw directory"):
        package.run()


@pytest.mark.parametrize(
    "defect",
    [
        "missing_cell",
        "duplicate_cell",
        "old_version",
        "old_build",
        "wrong_worker",
        "worker_missing_binary",
        "worker_extra_source",
        "missing_rank",
        "tail_enabled",
        "missing_tail_layer",
        "wrong_checkpoint",
        "missing_derivation",
        "changed_raw_hash",
        "alias_raw",
        "old_validator",
        "one_shot_split",
        "missing_oracle_case",
        "missing_snapshot",
        "long_missing_rank",
        "long_missing_original_failure",
        "long_wrong_tokens",
        "historical_field",
        "scope",
        "bool_rank",
        "changed_verifier",
    ],
)
def test_rejects_incomplete_or_mislabeled_proposed_evidence(package, defect):
    p, s = package, package.summary
    cell = s["functional_cells"][0]
    profile = cell["profiles"][0]
    if defect == "missing_cell":
        s["functional_cells"].pop()
    elif defect == "duplicate_cell":
        s["functional_cells"][-1] = copy.deepcopy(cell)
    elif defect == "old_version":
        s["candidate_version"] = "0.30.0+glm53kpool.bf5f6b0e689d"
    elif defect == "old_build":
        s["build_profiles"]["candidate"]["build"]["sha256"] = gate.HISTORICAL_BUILD
    elif defect in {"wrong_worker", "worker_missing_binary", "worker_extra_source", "bool_rank"}:
        ref = profile["workers"]["0"]
        obj = p.read(ref)
        if defect == "wrong_worker":
            obj["runtime"]["version"] = gate.VERSIONS["candidate"]
        elif defect == "worker_missing_binary":
            obj["runtime"]["loaded_files"].pop("vllm/_flashkda_C.abi3.so")
        elif defect == "worker_extra_source":
            obj["runtime"]["loaded_files"]["vllm/TEST_ONLY-unexpected.py"] = "f" * 64
        else:
            obj["tp_rank"] = False
        p.replace(ref, obj)
    elif defect == "missing_rank":
        profile["workers"].pop("1")
    elif defect in {"tail_enabled", "missing_tail_layer"}:
        ref = profile["cache_groups"]["0"]
        obj = p.read(ref)
        if defect == "tail_enabled":
            obj["slot_mapping_enabled_device"][0] = True
        else:
            obj["groups"][0]["tail_builders"][0]["layer_names"].pop()
        p.replace(ref, obj)
    elif defect == "wrong_checkpoint":
        ref = profile["preflight"]
        obj = p.read(ref)
        obj["checkpoint_config_sha256"] = "f" * 64
        p.replace(ref, obj)
    elif defect == "missing_derivation":
        (p.root / cell["revalidation"]["path"]).unlink()
    elif defect == "changed_raw_hash":
        ref = cell["revalidation"]
        obj = p.read(ref)
        obj["input_files"][0]["files"][0]["sha256"] = "f" * 64
        p.replace(ref, obj)
    elif defect == "alias_raw":
        cell["profiles"][1]["raw_uri"] = profile["raw_uri"]
    elif defect == "old_validator":
        profile["original_validator"]["sha256"] = "f" * 64
    elif defect == "one_shot_split":
        ref = cell["comparison"]
        obj = p.read(ref)
        proof = obj["native_receipts"][0]
        key = next(iter(proof["actual_prefill_splits"]["0"]))
        proof["actual_prefill_splits"]["0"][key] = [[0, 4097], [4097, 3]]
        p.replace(ref, obj)
    elif defect in {"missing_oracle_case", "missing_snapshot"}:
        o = s["cache_oracle"]
        if defect == "missing_snapshot":
            o["raw_inventory"] = [row for row in o["raw_inventory"] if row["path"] != "case-0-gate-0.pt"]
        else:
            obj = p.read(o["result"])
            obj["cases"].pop()
            p.replace(o["result"], obj)
    elif defect in {"long_missing_rank", "long_missing_original_failure"}:
        ref = s["ordinary_128k"]["result"]
        obj = p.read(ref)
        if defect == "long_missing_rank":
            obj["history"].pop()
        else:
            obj["history"][0]["original_failure_24576_8192_completed"] = False
        p.replace(ref, obj)
    elif defect == "long_wrong_tokens":
        s["ordinary_128k"]["original_token_file_sha256"] = "f" * 64
    elif defect == "historical_field":
        s["validator_identity"]["historical_unused_field"]["meaning"] = "current_build"
    elif defect == "changed_verifier":
        s["build_profiles"]["candidate"]["packaged_verifier"]["sha256"] = "f" * 64
    else:
        s["scope"] = "all_four_cells_128k_capacity_passed"
    with pytest.raises(ValueError):
        p.run()


@pytest.mark.parametrize("defect", ["symlink", "traversal", "unsafe_uri", "summary_bytes"])
def test_receipt_files_require_safe_exact_original_bytes(package, defect):
    p = package
    ref = p.summary["functional_cells"][0]["profiles"][0]["preflight"]
    if defect == "symlink":
        path = p.root / ref["path"]
        dest = path.with_name("TEST_ONLY-original.json")
        path.rename(dest)
        path.symlink_to(dest.name)
    elif defect == "traversal":
        ref["path"] = "../outside.json"
    elif defect == "unsafe_uri":
        ref["source_uri"] = "https://test.invalid/a/../b?credential=TEST_ONLY"
    else:
        p.put("qualification/admission-summary.json", p.summary)
        with pytest.raises(ValueError, match="summary bytes"):
            gate.validate_tail_qualification(p.root, expected_summary_sha256="f" * 64)
        return
    with pytest.raises(ValueError):
        p.run()


@pytest.mark.parametrize("defect", ["extra_source", "missing_binary", "wrong_version", "wrong_wheel"])
def test_runtime_map_semantics_reject_self_consistent_bad_receipt(package, defect):
    closures = gate._build_profiles(package.root, package.summary["build_profiles"])
    actual = copy.deepcopy(runtime("candidate", closures))
    if defect == "extra_source":
        actual["loaded_files"]["vllm/TEST_ONLY-extra.py"] = "f" * 64
    elif defect == "missing_binary":
        actual["loaded_files"].pop("vllm/_flashkda_C.abi3.so")
    elif defect == "wrong_version":
        actual["version"] = gate.VERSIONS["tail_reference"]
    else:
        actual["runtime_wheel_sha256"] = gate.WHEELS["tail_reference"]
    with pytest.raises(ValueError):
        gate._runtime(actual, "candidate", closures)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_layer",
        "extra_layer",
        "host_device_disagree",
        "enabled",
        "builder_spec",
        "wrong_source",
        "method_source",
        "duplicate_builder",
    ],
)
def test_native_group_semantics_reject_self_consistent_bad_receipt(package, defect):
    closures = gate._build_profiles(package.root, package.summary["build_profiles"])
    original = group(0, closures["candidate"])
    row = copy.deepcopy(original)
    builder = row["groups"][0]["tail_builders"][0]
    if defect == "missing_layer":
        builder["layer_names"].pop()
    elif defect == "extra_layer":
        builder["layer_names"].append("language_model.model.layers.0.self_attn.indexer.tail_cache")
    elif defect == "host_device_disagree":
        row["slot_mapping_enabled_host"][0] = True
    elif defect == "enabled":
        row["slot_mapping_enabled_host"][0] = row["slot_mapping_enabled_device"][0] = True
    elif defect == "builder_spec":
        builder["builder_spec"] = {**builder["builder_spec"], "uses_slot_mapping": True}
    elif defect == "wrong_source":
        row["source_pins"][gate.TAIL_PATH] = "f" * 64
    elif defect == "method_source":
        builder["build_source"]["sha256"] = "f" * 64
    else:
        row["groups"][0]["tail_builders"].append(copy.deepcopy(builder))
    with pytest.raises((ValueError, KeyError)):
        gate._group(row, closures["candidate"])


def test_boolean_rank_cannot_satisfy_integer_tail_history():
    row = tail(False, "a" * 64, 131072, 16)
    with pytest.raises(ValueError, match="rank"):
        gate._tail_rows([row], 1, {"cache-groups-rank-0.json": "a" * 64})
