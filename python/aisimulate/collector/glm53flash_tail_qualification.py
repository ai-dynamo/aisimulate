# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit a proposed tail-runtime evidence package without granting admission.

This original reader checks immutable small receipts. Full token/metadata and
oracle snapshots remain external and must be revalidated before the reviewed
summary SHA is pinned. It never modifies runtime registries or qualifies FPM,
Ops, HTTP, or per-deployment capacity. Frozen native API provenance is carried
by each original validator source reference, independently of this reader.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

SCOPE = "four_cell_native_Engine_functional_correctness_and_original_nvfp4_tp2_128k_regression_only"
VERSIONS = {
    "candidate": "0.30.0+glm53tail.eb4704514fdf",
    "tail_reference": "0.30.0+glm53tailref.4e4a40c2a838",
}
WHEELS = {
    "candidate": "538d440757a3bbf21b1b6a9f4348fee631789654e10ec8b8f4b13b8aa99b53e2",
    "tail_reference": "60655091e1d15fdfcbdfba12c1b47cbbebbdcca84bce551129ef172553882a94",
}
BUILDS = {
    "candidate": "59b012dcd92831405847b91d1fcfdc9f373d01cd8d1b829b54ce96dfcf0da07e",
    "tail_reference": "0a76cd7a4e9a5bf945be1e79add0310ed9c2142b87b31063423c58603afd4f95",
}
BUILD_RECEIPTS = {
    "candidate": {
        "sources": "591517b8379f6d79fafb386983fbd90b0c1613b4b61fd15b0e32fade66f453bf",
        "install": "e4f660da8c7b358ccdff66c64e79f057df2782f816c95c33a96567c76322ce79",
        "packaged_verifier": "71e5813b1871c90ffb2e4b0865b3d69b5652c48cff5dd860a46d0e9843898cc7",
        "packaged_launch": "b0114bbdd817ef91314af2210b5fde4dd16d256847b0dca27ee920d26f401aa1",
    },
    "tail_reference": {
        "sources": "f63b434a40d4833621e2d6d860cdbe8f26e2b24f2652fbb65b6c20664ead79b3",
        "install": "138f42aa855fbc8627d41705ed319601eb368c61bf5725f26bc00e00809cc956",
        "packaged_verifier": "edff93fad8929b4959a15ea94413fbdba2991df0a0b013c304399cb42f9c531c",
        "packaged_launch": "5d9d9edcc91351db18d95b653354882b32bb05595b42c9a8b535f0626b544ef0",
    },
}
EXECUTED_INPUTS_SHA = "30d9cf0c342453088783d541c68f6b98c7e02edab169078da0cafa225650d09b"
BINARIES_SHA = "2ffbf12cd5590804c821328510224bef9c2b5b25b64b7dbbda072037c68b541b"
VERIFIER_SHA = "181e9c1dbb8ae1c4c1afb839f620181dd78952c041617a0215b846328172aa86"
FUNCTIONAL_SOURCES = {
    "expected-runtime.json": "cf7ab8b815ef16e41e178a17cb24effe1f31f54785f0c61b1931d6e3a134e910",
    "validate.py": "885258e35358c07278e2f7e55c93b28c507ea678fd822c8b004640adeacd503f",
    "validate_tail.py": "751457541eb3a2c16797b15e6a2c12913d7625cc96c116d1035ebb2e8e1bb44e",
    "tail_metadata_probe.py": "a8ccfd7866301cb838a179dc90688260617593ee4c91d7604ed15049b1af6704",
    "probe.py": "9b26d0f0ccd9ef08f9944aca1c59744a8810fd8d07d23eb9bbb8f0fa55bac31d",
    "worker_probe.py": "6a8ca1b92d37119f21475f425499a66a7acada4879a48376e1488b370dcdf1ca",
    "request-id-source.json": "e2ceda1bb628586e3b5fafc448681669e089fcb8c0567a89fab4d73853902749",
    "cohort-source.json": "d0bd7102c44e00babb7ab503415f01115cd91cc50166f599658452051c841ead",
}
LONG_SOURCES = {
    **{
        name: FUNCTIONAL_SOURCES[name]
        for name in ("expected-runtime.json", "validate_tail.py", "tail_metadata_probe.py", "worker_probe.py")
    },
    "probe.py": "f48b930f07e62b02064bcf2de243bf601303ce2609b0bc53801792c18b46fc91",
    "actual-engine-args.json": "13108e457d2dec12f067934a34d6f72ed215593ae60f5e8eb33e446d5ca6761c",
    "restore_args.py": "d7e177ca79dcf9d3604daa69466660de8442e26ba2b16721ca247bc17d7f8731",
    "lineage.json": "95bceef9e36dc84877332c1ed081d5f3dbecbba4ca15a58aa3eb3078fe787d17",
    "receipt.json": "8980b8490c41385fc43613fdfb467b88e3a3ea9a9a51a504aeb1f96b37b58daf",
    "reconstructed-point4-request0.tokens.json": "b39e2fde811f638e5e3e9a7c3a27c697fcf185e9f1f5a3d476d2092fa1fdb57e",
}
ORACLE_SOURCES = {
    "expected-runtime.json": FUNCTIONAL_SOURCES["expected-runtime.json"],
    "probe.py": "f5c5de90bd22233f825ae3a4b90921975363c4576009c0886f694e110a801c28",
    "runtime_identity.py": "b1d9196a2bc3acb7b8f12f56b926db893b93ca8c4321f2a2b44d65b9a5e99854",
    "sources.json": "e36af47a4be7c80aaf028fdfe8478584c58561932692ee535920e314ae00925f",
}
HISTORICAL_BUILD = "3b72d70800e2ea244944580c1ce6a4faaa3dedf68af41b323aa690999abd9444"
TAIL_PATH = "vllm/v1/kv_cache_interface.py"
HELPER_PATH = "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
TAIL_SHA = "76fdecf31c8cd93479ebe7c788d699498e83957d18037eb5b39ccfe46f57c138"
TAIL_LAYERS = list(range(3, 44, 4))
PROFILES = (("tail_reference", "reference"), ("candidate", "reference"), ("candidate", "split"))
PROMPT_LENGTHS = [4100, 4101, 4100, 4101, 4098, 4099, 4100, 4101, 8196, 8197] * 2
ORACLE_CASES = (
    ((4096,), (3,)),
    ((4096,), (4,)),
    ((4097,), (3,)),
    ((4097, 4097), (3, 3)),
    ((4097,), (4,)),
    ((0, 0), (3, 5)),
    ((1, 2, 3, 7), (2, 2, 2, 9)),
    ((4097, 4098, 4099, 4096), (7, 10, 3, 8)),
)


def _require(condition, message):
    if not condition:
        raise ValueError("tail qualification: " + message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def _same(actual, expected, label):
    # In particular, True must not satisfy a rank/count/version integer.
    _require(type(actual) is type(expected) and actual == expected, label)
    if isinstance(expected, dict):
        for key in expected:
            _same(actual[key], expected[key], label)
    elif isinstance(expected, (list, tuple)):
        for left, right in zip(actual, expected, strict=True):
            _same(left, right, label)


def _uri(value):
    _require(isinstance(value, str), "external URI is not a string")
    parsed = urlsplit(value)
    decoded = unquote(parsed.path)
    _require(
        parsed.scheme in {"ssh", "https", "s3", "gs"}
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and decoded.startswith("/")
        and decoded != "/"
        and decoded == parsed.path
        and not any(c.isspace() or ord(c) < 32 or c in "\\?#%" for c in value)
        and not any(part in {".", "..", ""} for part in decoded.split("/")[1:]),
        "unsafe or noncanonical external URI",
    )
    return value


def _path(root, name):
    _require(isinstance(name, str), "file path is not a string")
    parts = PurePosixPath(name)
    _require(
        not parts.is_absolute()
        and str(parts) == name
        and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) and part not in {".", ".."} for part in parts.parts),
        "unsafe file path",
    )
    path = root
    _require(not path.is_symlink(), "symlink package root")
    for part in parts.parts:
        path /= part
        _require(not path.is_symlink(), "symlink receipt path")
    _require(path.is_file(), "missing regular receipt: " + name)
    return path


def _read(root, ref):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "source_uri"}, "receipt reference shape")
    _uri(ref["source_uri"])
    raw = _path(root, ref["path"]).read_bytes()
    _same(hashlib.sha256(raw).hexdigest(), _sha(ref["sha256"]), "receipt bytes changed")
    result = json.loads(raw)
    _require(isinstance(result, dict), "receipt is not an object")
    return result


def _inventory(rows, *, nested=False):
    _require(isinstance(rows, list) and rows, "missing raw inventory")
    result = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"path", "sha256"}, "raw inventory shape")
        name = row["path"]
        _require(
            isinstance(name, str)
            and all(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) and part not in {".", ".."}
                for part in name.split("/")
            )
            and (nested or "/" not in name),
            "unsafe raw member",
        )
        _require(name not in result, "duplicate raw inventory member")
        result[name] = _sha(row["sha256"])
    return result


def _bind_inventory(ref, inventory, original_name):
    _same(ref["sha256"], inventory.get(original_name), "small receipt differs from original raw inventory")


def _source_refs(refs, expected):
    _require(isinstance(refs, dict) and refs.keys() == expected.keys(), "validator source set differs")
    for name, sha in expected.items():
        _require(set(refs[name]) == {"sha256", "source_uri"}, "validator source reference shape")
        _same(refs[name]["sha256"], sha, "validator source identity differs")
        _uri(refs[name]["source_uri"])


def _build_profiles(root, refs):
    _require(isinstance(refs, dict) and set(refs) == set(VERSIONS), "exact two build profiles required")
    closures = {}
    for kind, version in VERSIONS.items():
        records = refs[kind]
        _require(
            set(records)
            == {"build", "install", "packaged_verifier", "packaged_launch", "executed_inputs", "sources", "binaries"},
            "build evidence set differs",
        )
        data = {key: _read(root, value) for key, value in records.items()}
        for key, sha in {
            **BUILD_RECEIPTS[kind],
            "binaries": BINARIES_SHA,
            "executed_inputs": EXECUTED_INPUTS_SHA,
        }.items():
            _same(records[key]["sha256"], sha, "actual build evidence bytes differ: " + key)
        build, install, actual, launch = (
            data[key] for key in ("build", "install", "packaged_verifier", "packaged_launch")
        )
        _same(records["build"]["sha256"], BUILDS[kind], "new runtime requires its own actual build")
        _same(build["version"], version, "build version differs")
        _same(build["wheel_sha256"], WHEELS[kind], "wheel differs")
        sources, binaries = data["sources"], data["binaries"]
        _require(
            len(sources) == 31 and len(binaries) == 19 and not sources.keys() & binaries.keys(),
            "source/binary inventory differs",
        )
        for sha in [*sources.values(), *binaries.values()]:
            _sha(sha)
        _same(sources[TAIL_PATH], TAIL_SHA, "tail source differs")
        _same(build["unchanged_native_binaries"], binaries, "native binary build closure differs")
        _same(
            {entry["source_path"]: entry["patched_sha256"] for entry in build["patch"]["sources"]},
            {name: sources[name] for name in (TAIL_PATH, HELPER_PATH)},
            "patch closure differs",
        )
        for receipt in (install, actual):
            for key, wanted in {
                "status": "ACTUAL_CPU_INSTALL_SOURCE_AND_NATIVE_SPEC_PASS",
                "version": version,
                "wheel_sha256": WHEELS[kind],
                "build_receipt_sha256": BUILDS[kind],
                "observed_files": sources | binaries,
                "native_binary_names": sorted(binaries),
                "tail_uses_slot_mapping": False,
                "ordinary_sliding_window_uses_slot_mapping": True,
                "pytorch_cuda_initialized": False,
                "formal_admission": False,
                "model_correctness": "NOT_EVALUATED",
            }.items():
                _same(receipt.get(key), wanted, "actual build/install/verifier identity differs: " + key)
        _same(launch["job"], 614605, "packaged verifier job differs")
        _same(launch["exit"]["returncode"], 0, "packaged verifier failed")
        _same(launch["exit"]["argv"], launch["started"]["argv"], "packaged verifier launch differs")
        _same(
            launch["actual_receipt_sha256"],
            records["packaged_verifier"]["sha256"],
            "packaged verifier receipt binding differs",
        )
        _same(
            launch["executed_inventory_sha256"],
            records["executed_inputs"]["sha256"],
            "executed input inventory differs",
        )
        prefix = "reference" if kind == "tail_reference" else kind
        operative = {
            "actual-build-receipt.json": "build",
            "expected-source-sha256.json": "sources",
            "expected-native-binaries.json": "binaries",
        }
        for filename, key in operative.items():
            _same(
                data["executed_inputs"]["files"][prefix + "/" + filename],
                records[key]["sha256"],
                "executed operative input differs",
            )
        _require(
            f"/probe/{prefix}/verify_install.py" in launch["started"]["argv"],
            "actual packaged verifier invocation absent",
        )
        # Its immutable executed inventory pins the verifier and before-execution
        # metadata. Later packaging-lineage.json is intentionally not substituted.
        _same(
            data["executed_inputs"]["files"][prefix + "/verify_install.py"],
            VERIFIER_SHA,
            "executed packaged verifier differs",
        )
        _sha(data["executed_inputs"]["files"][prefix + "/packaging-lineage.json"])
        closures[kind] = sources | binaries
    _same(closures["candidate"].keys(), closures["tail_reference"].keys(), "reference file set differs")
    _same(
        [name for name in closures["candidate"] if closures["candidate"][name] != closures["tail_reference"][name]],
        [HELPER_PATH],
        "reference changes additional sources/binaries",
    )
    return closures


def _derivation(root, ref, *, scope, original, result, inputs, sources):
    """Bind a later full-raw revalidation without relabeling its originals."""
    receipt = _read(root, ref)
    for key, value in {
        "schema": "glm53flash_tail_revalidation_v1",
        "status": "passed",
        "scope": scope,
        "original_result_sha256": original["sha256"],
        "derived_result_sha256": result["sha256"],
        "input_files": inputs,
        "consumer_reread_external_raw": False,
    }.items():
        _same(receipt.get(key), value, "fresh raw revalidation differs: " + key)
    _source_refs(receipt["validator_sources"], sources)
    _require(set(receipt["assembler_source"]) == {"sha256", "source_uri"}, "assembler source reference shape")
    _sha(receipt["assembler_source"]["sha256"])
    _uri(receipt["assembler_source"]["source_uri"])


def _runtime(actual, kind, closures):
    _same(actual.get("version"), VERSIONS[kind], "actual runtime version differs")
    _same(actual.get("runtime_wheel_sha256"), WHEELS[kind], "actual runtime wheel differs")
    _same(actual.get("loaded_files"), closures[kind], "actual source and19-binary closure differs")


def _tail_rows(rows, tp, inventory, *, expected_addresses=None, expected_forwards=None):
    _require(isinstance(rows, list) and len(rows) == tp, "tail proof rank count differs")
    _same([row.get("tp_rank") for row in rows], list(range(tp)), "tail rank order/coverage differs")
    for rank, row in enumerate(rows):
        _same(row.get("status"), "ACTUAL_NATIVE_TAIL_ADDRESSES_AND_POSITIONS_VERIFIED", "native tail proof failed")
        _same(row.get("tail_layers"), TAIL_LAYERS, "tail layer set differs")
        _same(row.get("latency_admission"), False, "diagnostic tail proof cannot admit timing")
        _same(
            row.get("cache_groups_sha256"),
            inventory.get(f"cache-groups-rank-{rank}.json"),
            "tail owner group bytes differ",
        )
        for key, expected in (("checked_actual_token_addresses", expected_addresses), ("forwards", expected_forwards)):
            _require(type(row.get(key)) is int and row[key] > 0, "missing actual tail coverage")
            if expected is not None:
                _same(row[key], expected, "tail coverage count differs")


def _profile(root, profile, evidence, cell, expected, closures, unique):
    kind, mode = profile["runtime_kind"], profile["mode"]
    raw_uri = _uri(profile["raw_uri"])
    _require(raw_uri not in unique, "profiles alias one original raw directory")
    unique.add(raw_uri)
    tp, checkpoint = cell["tp"], cell["checkpoint"]
    original = _read(root, profile["native_receipt"])
    preflight = _read(root, profile["preflight"])
    for key, wanted in {
        "status": "native_requests_and_histories_verified",
        "runtime_kind": kind,
        "mode": mode,
        "checkpoint": checkpoint,
        "tp": tp,
        "policy": "production",
        "correctness_comparison": "NOT_EVALUATED",
        "accuracy_acceptance": "NOT_EVALUATED",
    }.items():
        _same(original.get(key), wanted, "original native profile differs: " + key)
    _runtime(preflight["runtime"], kind, closures)
    _same(preflight["status"], "passed", "native preflight failed")
    identity = expected["checkpoints"][checkpoint]
    actual_identity = evidence["checkpoint_identity"]
    _same(
        actual_identity, original["evidence"]["checkpoint_identity"], "checkpoint revalidation changed original proof"
    )
    for key, value in identity.items():
        _same(actual_identity.get(key), value, "checkpoint identity differs")
    _same(actual_identity.get("checkpoint"), checkpoint, "checkpoint label differs")
    _same(
        preflight.get("checkpoint_config_sha256"), identity["config_sha256"], "actual preflight checkpoint bytes differ"
    )
    args = preflight["public_engine_args"]
    for key, value in {
        "revision": identity["revision"],
        "tokenizer_revision": identity["revision"],
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
    }.items():
        _same(args.get(key), value, "actual native functional Engine argument differs: " + key)
    _source_refs({"validate.py": profile["original_validator"]}, {"validate.py": FUNCTIONAL_SOURCES["validate.py"]})
    inventory = _inventory(evidence["files"])
    old_inventory = _inventory(original["evidence"]["files"])
    _require("native-receipt.json" not in old_inventory, "original receipt contains its future hash")
    _same(
        inventory,
        old_inventory | {"native-receipt.json": profile["native_receipt"]["sha256"]},
        "revalidation raw inventory differs",
    )
    _same(
        {k: v for k, v in evidence.items() if k != "files"},
        {k: v for k, v in original["evidence"].items() if k != "files"},
        "revalidation changed original native evidence",
    )
    _same(profile["revalidation_added_fields"], ["files.native-receipt.json"], "derived additions must be explicit")
    for field, name in (("preflight", "preflight.json"), ("native_receipt", "native-receipt.json")):
        _bind_inventory(profile[field], inventory, name)
    _same(evidence["requests"], 20, "incomplete native request suite")
    _same(evidence["request_identity_protocol"], "native_assign_request_id_v1", "request identity protocol differs")
    _same(
        evidence["cohort_admission_protocol"], "native_scheduling_pause_enqueue_v1", "cohort admission protocol differs"
    )
    _sha(evidence["all_tp_trace_digest"])
    modes = evidence["native_modes"]
    _require(
        isinstance(modes, list)
        and modes == sorted(set(modes))
        and "FULL" in modes
        and set(modes) <= {"NONE", "PIECEWISE", "FULL"},
        "observed production modes differ",
    )
    mapping = evidence["external_to_native_request_ids"]
    _require(len(mapping) == len(set(mapping.values())) == 20, "request ID bijection differs")
    splits = evidence["actual_prefill_splits"]
    _same(set(splits), {str(rank) for rank in range(tp)}, "prefill rank coverage differs")
    for rows in splits.values():
        _same(set(rows), set(mapping.values()), "prefill request coverage differs")
        lengths = []
        for chain in rows.values():
            _require(
                isinstance(chain, list)
                and chain
                and all(
                    isinstance(pair, list) and len(pair) == 2 and all(type(n) is int for n in pair) for pair in chain
                ),
                "prefill chain shape differs",
            )
            n = sum(pair[1] for pair in chain)
            wanted = (
                [[0, n]]
                if mode == "reference"
                else ([[0, 4097], [4097, n - 4097]] if n < 8194 else [[0, 4097], [4097, 4097], [8194, n - 8194]])
            )
            _same(chain, wanted, "actual one-shot/split partition differs")
            lengths.append(n)
        _same(sorted(lengths), sorted(PROMPT_LENGTHS), "frozen prompt geometry differs")
    _tail_rows(evidence["native_tail_evidence"], tp, inventory, expected_addresses=sum(PROMPT_LENGTHS) + 20 * 31)
    for field, prefix in (("workers", "worker"), ("cache_groups", "cache-groups")):
        refs = profile[field]
        _same(set(refs), {str(rank) for rank in range(tp)}, "packaged worker/group coverage differs")
        for rank in range(tp):
            ref = refs[str(rank)]
            row = _read(root, ref)
            _bind_inventory(ref, inventory, f"{prefix}-rank-{rank}.json")
            _same(row["tp_rank"], rank, "original worker/group rank differs")
            if field == "workers":
                _runtime(row["runtime"], kind, closures)
                _same(row["runtime"], preflight["runtime"], "worker and preflight runtime differ")
                _same(row["hardware"]["compute_capability"], [10, 3], "native GPU architecture differs")
                _require("GB300" in row["hardware"]["name"].upper(), "native GPU identity differs")
            else:
                _group(row, closures[kind])
    required = {
        "outputs.jsonl",
        "effective-native-config.json",
        "worker-installation.json",
        "worker-completion.json",
        "request-id-map.jsonl",
        "cohort-admission.jsonl",
    }
    required |= {
        f"{prefix}-rank-{rank}.{suffix}"
        for rank in range(tp)
        for prefix, suffix in (
            ("worker", "json"),
            ("cache-groups", "json"),
            ("prompts", "jsonl"),
            ("forward", "jsonl"),
            ("tail-metadata", "jsonl"),
        )
    }
    _require(required <= inventory.keys(), "missing original native raw members")
    _require(
        all(
            name in required
            for name in inventory
            if re.fullmatch(r"(worker|cache-groups|prompts|forward|tail-metadata)-rank-\d+\.(json|jsonl)", name)
        ),
        "unexpected TP rank raw member",
    )


def _group(row, closure):
    paths = (
        TAIL_PATH,
        "vllm/v1/attention/backends/mla/indexer.py",
        "vllm/v1/worker/gpu/model_states/default.py",
        "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
        "vllm/v1/worker/gpu/attn_utils.py",
    )
    _same(row["source_pins"], {name: closure[name] for name in paths}, "tail owner sources differ")

    def method(receipt, source):
        _same(receipt["relative_path"], source, "native method source path differs")
        _same(receipt["sha256"], closure[source], "native method source bytes differ")
        _require(
            isinstance(receipt["path"], str) and receipt["path"].endswith("/" + source), "native method origin differs"
        )

    method(row["model_state_prepare_source"], "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
    _same(
        row["model_state_class"],
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.MambaHybridModelState",
        "native model-state class differs",
    )
    _same(row["slot_mapping_enabled_host"], row["slot_mapping_enabled_device"], "tail host/device flags differ")
    groups = row["groups"]
    _same(len(groups), len(row["slot_mapping_enabled_device"]), "cache group count differs")
    layers, builders = set(), set()
    for index, group in enumerate(groups):
        _same(group["group_index"], index, "cache group order differs")
        if not group["tail_builders"]:
            continue
        _same(row["slot_mapping_enabled_device"][index], False, "generic tail mapping enabled")
        for builder in group["tail_builders"]:
            key = builder["group_index"], builder["builder_index"]
            _require(key not in builders and key[0] == index, "tail builder ownership repeated/changed")
            builders.add(key)
            _same(
                builder["builder_class"],
                "vllm.v1.attention.backends.mla.indexer.KpoolTailMetadataBuilder",
                "tail builder class differs",
            )
            method(builder["build_source"], "vllm/v1/attention/backends/mla/indexer.py")
            for name in builder["layer_names"]:
                match = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.self_attn\.indexer\.tail_cache", name)
                _require(match is not None and int(match[1]) not in layers, "tail layer ownership differs")
                layers.add(int(match[1]))
                spec = group["per_layer_specs"][name]
                _same(spec["class"], "vllm.v1.kv_cache_interface.KpoolTailSpec", "tail spec class differs")
                _same(spec["uses_slot_mapping"], False, "actual tail spec mapping enabled")
                _same(spec["fields"]["block_size"], 4, "tail block size differs")
                _same(spec["fields"]["sliding_window"], 4, "tail window differs")
                _same(spec, builder["builder_spec"], "actual builder spec differs from owner")
    _same(sorted(layers), TAIL_LAYERS, "eleven tail owners are not exactly covered")


def _validate_oracle(root, section, closures):
    _uri(section["raw_uri"])
    result = _read(root, section["result"])
    runtime = _read(root, section["runtime"])
    _runtime(runtime, "candidate", closures)
    _same(result["actual_runtime"], runtime, "oracle runtime differs")
    _same(result["formal_admission"], False, "oracle cannot grant formal admission")
    _same(result["performance_data"], False, "kernel oracle cannot admit latency")
    _same(result["model_correctness"], "NOT_EVALUATED", "kernel oracle cannot replace Engine correctness")
    _require(isinstance(result["device"], str) and "GB300" in result["device"].upper(), "oracle GPU differs")
    cases = result["cases"]
    _require(len(cases) == 16, "sixteen actual oracle cases required")
    for row in cases:
        prefixes, queries = row["prefixes"], row["queries"]
        _require(
            isinstance(prefixes, list)
            and isinstance(queries, list)
            and len(prefixes) == len(queries) > 0
            and all(type(value) is int and value >= 0 for value in prefixes)
            and all(type(value) is int and value > 0 for value in queries),
            "oracle geometry must contain exact integer prefix/query counts",
        )
    wanted = {(p, q, gate) for p, q in ORACLE_CASES for gate in (False, True)}
    actual = {(tuple(row["prefixes"]), tuple(row["queries"]), row["nonuniform_gate"]) for row in cases}
    _same(actual, wanted, "oracle geometry/gate coverage differs")
    inventory = _inventory(section["raw_inventory"])
    _bind_inventory(section["result"], inventory, "probe-result.json")
    _bind_inventory(section["runtime"], inventory, "actual-runtime.json")
    for row in cases:
        _require(type(row["nonuniform_gate"]) is bool, "oracle gate type differs")
        for key in ("one_shot_matches_oracle", "split_cache_matches_one_shot", "tail_matches_one_shot"):
            _same(row[key], True, "oracle comparison failed")
        index = ORACLE_CASES.index((tuple(row["prefixes"]), tuple(row["queries"])))
        _same(
            row["snapshot_sha256"],
            inventory.get(f"case-{index}-gate-{int(row['nonuniform_gate'])}.pt"),
            "oracle snapshot bytes differ",
        )
    _derivation(
        root,
        section["revalidation"],
        scope="original_16_case_cache_oracle",
        original=section["result"],
        result=section["result"],
        inputs=[{"raw_uri": section["raw_uri"], "files": section["raw_inventory"]}],
        sources=ORACLE_SOURCES,
    )


def _validate_long(root, section, closures):
    _uri(section["raw_uri"])
    result = _read(root, section["result"])
    preflight = _read(root, section["preflight"])
    _runtime(preflight["runtime"], "candidate", closures)
    lineage = _read(root, section["original_lineage"])
    original_args = _read(root, section["original_engine_args"])
    _same(
        section["original_lineage"]["sha256"],
        LONG_SOURCES["receipt.json"],
        "original request reconstruction receipt differs",
    )
    _same(
        section["original_engine_args"]["sha256"],
        LONG_SOURCES["actual-engine-args.json"],
        "original244 Engine arguments differ",
    )
    _same(preflight["input_lineage"], lineage, "original request/checkpoint/tokenizer lineage differs")
    _same(
        preflight["engine_args"],
        {**original_args, "scheduler_cls": None, "worker_extension_cls": "worker_probe.QualificationWorker"},
        "ordinary reference changed original native arguments",
    )
    _same(
        result["status"],
        "FRESH_NATIVE_REQUEST_COMPLETED_WITH_ACTUAL_LONG_SEED",
        "ordinary128K request did not complete",
    )
    for key, wanted in {
        "formal_admission": False,
        "accuracy_acceptance": "NOT_EVALUATED",
        "latency_admission": False,
    }.items():
        _same(result[key], wanted, "long regression scope differs")
    _same(preflight["diagnostic_environment"], {"CUDA_LAUNCH_BLOCKING": "1"}, "original diagnostic environment differs")
    _same(section["checkpoint"], "nvfp4", "original long checkpoint differs")
    _same(section["tp"], 2, "original long TP differs")
    _same(
        section["original_token_file_sha256"],
        LONG_SOURCES["reconstructed-point4-request0.tokens.json"],
        "original131072 token file differs",
    )
    inventory = _inventory(section["raw_inventory"], nested=True)
    _bind_inventory(section["result"], inventory, "result.json")
    _bind_inventory(section["preflight"], inventory, "preflight.json")
    expected_history = [
        {
            "tp_rank": rank,
            "completed_forwards": 16,
            "prefill_tokens": 131072,
            "seed_16384_8192_completed": True,
            "original_failure_24576_8192_completed": True,
        }
        for rank in range(2)
    ]
    _same(result["history"], expected_history, "original failed boundaries/full128K history omitted")
    tail_inventory = {
        name.removeprefix("witness/"): sha for name, sha in inventory.items() if name.startswith("witness/")
    }
    _tail_rows(result["actual_tail_evidence"], 2, tail_inventory, expected_addresses=131072, expected_forwards=16)
    for field, prefix in (("workers", "worker"), ("cache_groups", "cache-groups")):
        _same(set(section[field]), {"0", "1"}, "long packaged rank coverage differs")
        for rank in range(2):
            ref = section[field][str(rank)]
            row = _read(root, ref)
            _bind_inventory(ref, inventory, f"witness/{prefix}-rank-{rank}.json")
            _same(row["tp_rank"], rank, "long worker/group rank differs")
            if field == "workers":
                _same(row["runtime"], preflight["runtime"], "long actual worker runtime differs")
            else:
                _group(row, closures["candidate"])
    output = _read(root, section["output"])
    _bind_inventory(section["output"], inventory, "native-output.json")
    _require(
        isinstance(output.get("token_ids"), list)
        and len(output["token_ids"]) == 1
        and type(output["token_ids"][0]) is int
        and output["token_ids"][0] >= 0,
        "long final native sampled token missing",
    )
    required = {"native-request-id-mapping.jsonl", "native-output.json"} | {
        f"witness/{prefix}-rank-{rank}.{suffix}"
        for rank in range(2)
        for prefix, suffix in (
            ("forward", "jsonl"),
            ("worker", "json"),
            ("cache-groups", "json"),
            ("tail-metadata", "jsonl"),
        )
    }
    _require(required <= inventory.keys(), "long raw inventory lacks original output/rank evidence")
    _derivation(
        root,
        section["revalidation"],
        scope="original_nvfp4_tp2_131072_request",
        original=section["result"],
        result=section["result"],
        inputs=[{"raw_uri": section["raw_uri"], "files": section["raw_inventory"]}],
        sources=LONG_SOURCES,
    )


def _validate_tail_qualification(root: Path, *, expected_summary_sha256: str) -> dict:
    """Validate a proposed immutable package; runtime admission stays unchanged.

    The fixed summary filename is qualification/admission-summary.json. This API
    does not fetch raw evidence, run GPUs, or treat CPU/build success as correctness.
    """
    root = Path(root)
    path = _path(root, "qualification/admission-summary.json")
    raw = path.read_bytes()
    _same(hashlib.sha256(raw).hexdigest(), _sha(expected_summary_sha256), "summary bytes differ")
    summary = json.loads(raw)
    fixed = {
        "schema_name": "glm53flash_tail_runtime_qualification",
        "schema_version": 1,
        "status": "bounded_native_qualification_passed",
        "scope": SCOPE,
        "candidate_version": VERSIONS["candidate"],
        "reference_version": VERSIONS["tail_reference"],
        "formal_fpm_accuracy": "NOT_EVALUATED",
        "formal_ops_accuracy": "NOT_EVALUATED",
        "formal_8_cell_coverage": "NOT_EVALUATED",
        "http_acceptance": "NOT_EVALUATED",
        "per_cell_capacity_admission": "SEPARATE_NATIVE_NINE_POINT_QUALIFICATION_REQUIRED",
    }
    for key, value in fixed.items():
        _same(summary.get(key), value, "summary identity/scope differs: " + key)
    _same(
        set(summary),
        set(fixed) | {"build_profiles", "validator_identity", "functional_cells", "cache_oracle", "ordinary_128k"},
        "unknown or missing summary fields",
    )
    closures = _build_profiles(root, summary["build_profiles"])
    validator = summary["validator_identity"]
    _source_refs(validator["functional_sources"], FUNCTIONAL_SOURCES)
    expected = _read(root, validator["expected_runtime"])
    _same(
        validator["expected_runtime"]["sha256"],
        FUNCTIONAL_SOURCES["expected-runtime.json"],
        "frozen functional runtime manifest differs",
    )
    _same(expected["build_receipt_sha256"], HISTORICAL_BUILD, "historical unused build field differs")
    _same(
        validator["historical_unused_field"],
        {
            "field": "expected-runtime.json.build_receipt_sha256",
            "value": HISTORICAL_BUILD,
            "meaning": "inherited_old_kpool_build_unused_for_tail_admission",
        },
        "historical field cannot serve as new build proof",
    )
    _same(expected["versions"], VERSIONS, "functional runtime versions differ")
    _same(expected["wheel_sha256"], WHEELS, "functional wheel identities differ")
    cells = summary["functional_cells"]
    _require(isinstance(cells, list) and len(cells) == 4, "exact four functional cells required")
    _same(
        {(cell["checkpoint"], cell["tp"]) for cell in cells},
        {(fmt, tp) for fmt in ("fp8", "nvfp4") for tp in (2, 4)},
        "functional cell matrix differs",
    )
    unique = set()
    for cell in cells:
        _require(type(cell["tp"]) is int, "invalid TP type")
        comparison = _read(root, cell["comparison"])
        original_comparison = _read(root, cell["original_comparison"])
        _same(original_comparison, comparison, "fresh comparison changed original functional results")
        for key, wanted in {
            "status": "passed",
            "scope": "native_Engine_functional_correctness_for_frozen_geometry_suite_only",
            "checkpoint": cell["checkpoint"],
            "tp": cell["tp"],
            "policy": "production",
            "requests_per_mode": 20,
            "greedy_output_tokens_per_request": 32,
            "comparisons": 40,
            "differences": [],
            "accuracy_acceptance": "NOT_EVALUATED",
            "formal_8_cell_coverage": "NOT_EVALUATED",
        }.items():
            _same(comparison.get(key), wanted, "native comparison differs: " + key)
        profiles = cell["profiles"]
        _same(
            [(p["runtime_kind"], p["mode"]) for p in profiles], list(PROFILES), "three ordered native profiles required"
        )
        _require(len(comparison["native_receipts"]) == 3, "three native proof results required")
        for profile, evidence in zip(profiles, comparison["native_receipts"], strict=True):
            _profile(root, profile, evidence, cell, expected, closures, unique)
        inputs = [
            {"raw_uri": profile["raw_uri"], "files": proof["files"]}
            for profile, proof in zip(profiles, comparison["native_receipts"], strict=True)
        ]
        _derivation(
            root,
            cell["revalidation"],
            scope="four_cell_functional_profile_comparison",
            original=cell["original_comparison"],
            result=cell["comparison"],
            inputs=inputs,
            sources=FUNCTIONAL_SOURCES,
        )
    # Separate roots/evidence prevent a successful small geometry comparison
    # from silently substituting for the original long-request regression.
    for name in ("cache_oracle", "ordinary_128k"):
        raw_uri = _uri(summary[name]["raw_uri"])
        _require(raw_uri not in unique, "oracle/long regression aliases another original raw directory")
        unique.add(raw_uri)
    _validate_oracle(root, summary["cache_oracle"], closures)
    _validate_long(root, summary["ordinary_128k"], closures)
    return summary


def validate_tail_qualification(root: Path, *, expected_summary_sha256: str) -> dict:
    """Read a proposed SHA-pinned package without changing runtime admission.

    Missing, type-confused or incomplete receipt fields are explicit rejections.
    No external raw files are fetched or claimed revalidated by this consumer.
    """
    try:
        return _validate_tail_qualification(root, expected_summary_sha256=expected_summary_sha256)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("tail qualification: malformed or incomplete receipt") from error
