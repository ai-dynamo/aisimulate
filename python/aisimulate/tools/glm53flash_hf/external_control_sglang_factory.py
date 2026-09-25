# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original-byte contract for the public SGLang factory and sixteen CPU transports.

This fourth adapter is not an alias for the historical split-host launcher. It
checks evidence, never installs software, admits native rows, or launches work.
The prospective admission/started schemas are documented beside these tools.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

if __package__:
    from . import external_control_current as current
    from . import raw_archive as archive
else:
    import external_control_current as current
    import raw_archive as archive

SCHEMA = "glm53flash_external_control_v4"
ADAPTER = "sglang_public_factory_formal_v4"
ADMISSION = "sglang_public_factory_formal_admission_v1"
STARTED = "sglang_public_factory_formal_started_v1"
FINAL = "sglang_public_factory_formal_result_v1"
ADMISSION_FIELDS = {
    "schema",
    "status",
    "host",
    "producer",
    "cpu_controller_manifest",
    "cpu_source",
    "factory_manifest",
    "factory_source",
    "cache_hook",
    "storage_binding",
    "actual_cpu",
    "qualifications",
    "qualification_points",
}
STARTED_FIELDS = {
    "schema",
    "state",
    "job",
    "deployment",
    "mode",
    "source_commit",
    "wheel_sha256",
    "host_source_commit",
    "host_wheel_sha256",
    "admission_sha256",
    "launcher_manifest_sha256",
    "actual_cpu_job",
    "selected",
    "qualification",
    "requested_allocator_policy",
    "plan_sha256",
}
QUALIFICATION_FIELDS = {
    "reader",
    "aggregation",
    "rows",
    "plan",
    "started",
    "checkpoint",
    "installed_reader",
    "reader_manifest",
    "idle",
}
READER_SOURCES = {
    "fpm_forward/config.py",
    "fpm_forward/database.py",
    "fpm_forward/native_artifact.py",
    "fpm_forward/planner.py",
    "fpm_forward/sglang_allocator.py",
    "fpm_forward/sglang_artifact.py",
    "glm53flash_sglang_retained.py",
}
require = archive.require


def fields(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), label + " fields differ")


def digest_map(value):
    require(isinstance(value, dict) and value, "missing complete digest map")
    for name, digest in value.items():
        current._control().relative(name)
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid payload digest")
    return value


class Evidence(current.Frozen):
    """Translate only a public, hash-bound storage-root proof; keep JSON verbatim."""

    def __init__(self, get, task, storage):
        super().__init__(get)
        self.task = task
        self.storage = storage
        if storage is not None:
            _, canonical = archive.validate_storage_binding(storage)
            require(canonical == task, "attachment task is not the bound canonical storage root")

    def key(self, path):
        if Path(path).is_absolute():
            resolved = archive.storage_path(path, self.storage)
            require(resolved.is_relative_to(self.task), "original reference escapes task root")
            return resolved.relative_to(self.task).as_posix()
        return current._control().relative(path)

    def ref(self, value):
        fields(value, {"path", "sha256"}, "original reference")
        return self.bind(self.key(value["path"]), value["sha256"])

    def value(self, ref):
        return self.read(self.ref(ref))

    def same_path(self, left, right):
        return self.key(left) == self.key(right)


def installed(proof, identity, records, evidence, *, container=False):
    require(
        proof.get("state") == "EXACT_INSTALLED_WHEEL_RECORD_SOURCE_ELF_PASS"
        and proof.get("source") == identity["source_commit"]
        and proof.get("wheel_sha256") == identity["wheel_sha256"]
        and proof.get("files") == records
        and proof.get("runtime_sha256") == records.get("aisimulate/_runtime.abi3.so"),
        "complete installed wheel/source/runtime map differs",
    )
    require(
        proof.get("target") == "/opt/glm53flash-current"
        if container
        else evidence.same_path(proof["target"], identity["target"]),
        "installed target differs",
    )


def origins(values, target, records, evidence, *, required=()):
    require(isinstance(values, dict) and set(required) <= set(values), "installed module origins missing")
    for value in values.values():
        require(isinstance(value, dict), "invalid module origin")
        root = Path(evidence.key(target))
        path = Path(evidence.key(value["path"]))
        require(path.is_relative_to(root), "module is outside installed target")
        require(records.get(path.relative_to(root).as_posix()) == value.get("sha256"), "module source bytes differ")


def policy(options, deployment):
    require(
        options.get("sglang_mem_fraction_static") == 0.82
        and options.get("sglang_allocator_max_split_size_mb") == (16384 if deployment == "fp8-tp2" else None),
        "SGLang memory/allocator policy differs",
    )


def _sources(admission, frozen):
    for label in ("host", "producer"):
        fields(admission[label], {"source_commit", "wheel_sha256", "target"}, label)
        current._identity(admission[label]["source_commit"], admission[label]["wheel_sha256"])
    require(not frozen.same_path(admission["host"]["target"], admission["producer"]["target"]), "targets are shared")
    for key in ("cpu_controller_manifest", "factory_manifest"):
        ref = admission[key]
        frozen.manifest(frozen.key(ref["path"]), ref["sha256"])
    for key, manifest in (("cpu_source", "cpu_controller_manifest"), ("factory_source", "factory_manifest")):
        require(
            frozen.key(admission[key]["path"])
            == str(Path(frozen.key(admission[manifest]["path"])).parent / "source.json"),
            "source is outside its frozen manifest",
        )
        require(frozen.files.get(frozen.key(admission[key]["path"])) == admission[key]["sha256"], "source not frozen")
    source, factory = frozen.value(admission["cpu_source"]), frozen.value(admission["factory_source"])
    require(source.get("contract") == "sg2d51_formal_public_arm_cpu_envelope_v1", "wrong CPU envelope contract")
    require(factory.get("contract") == "sg2d51_formal_public_factory_v1", "wrong public factory contract")
    for key in ("factory_manifest", "factory_source"):
        require(source[key] == admission[key], "CPU/factory source reference differs")
    require(
        source["source_commit"]
        == factory["source_commit"]
        == admission["producer"]["source_commit"]
        == admission["host"]["source_commit"]
        and source["wheel"]["sha256"]
        == factory["wheel_sha256"]
        == admission["producer"]["wheel_sha256"]
        == admission["host"]["wheel_sha256"],
        "new factory host/producer identities differ",
    )
    for label, key in (("host", "consumer_target"), ("producer", "producer_target")):
        require(frozen.same_path(source[key], admission[label]["target"]), "CPU installed target differs")
    records = digest_map(frozen.value(source["record_payloads"]))
    record = factory["records"]["arm64"]
    require(
        source["record_payloads"]["sha256"] == factory["record_payloads_sha256"] == record["record_map_sha256"]
        and len(records) == record["record_payloads"]
        and record["wheel_sha256"] == factory["wheel_sha256"]
        and records.get("aisimulate/_runtime.abi3.so") == record["runtime_sha256"] == factory["runtime_sha256"]
        and {k: v for k, v in records.items() if k.startswith("collector/")} == factory["collector_sha256"],
        "factory complete ARM RECORD/collector map differs",
    )
    for label in ("host", "producer"):
        installed(frozen.value(source[label + "_proof"]), admission[label], records, frozen)
    require(source["producer_proof"]["sha256"] == factory["producer_proof_sha256"], "producer original proof differs")
    # Bind actual source dependencies, not arbitrary historical review receipts or wheel blobs.
    for key in ("host_import", "assets", "original_inventory"):
        frozen.ref(source[key])
    require(
        source["original_inventory"]["sha256"] == factory["original_prepared_inventory"], "original inventory differs"
    )
    copy_ref = factory["copied_sources"]["cache-hook/sitecustomize.py"]
    copied = str(Path(frozen.key(admission["factory_source"]["path"])).parent / "cache-hook/sitecustomize.py")
    require(
        frozen.key(admission["cache_hook"]["path"]) == copy_ref["original"]
        and admission["cache_hook"]["sha256"] == copy_ref["sha256"] == factory["cache_sha256"] == frozen.files[copied],
        "actual mounted cache hook differs from frozen factory copy",
    )
    frozen.ref(admission["cache_hook"])
    return source, factory, records


def _prepared(frozen, cpu_root, source, factory, records):
    root = cpu_root / "prepared"
    frozen.mapping(frozen.read(str(root / "inventory.json")), str(root))
    receipt = frozen.read(str(root / "receipt.json"))
    require(
        receipt.get("status") == "INSTALLED_PUBLIC2D51_SOURCE_RENDER_PASS_NOT_QUALIFIED"
        and receipt.get("host_arch") == "arm64"
        and receipt.get("source_commit") == factory["source_commit"]
        and receipt.get("host_record_map_sha256") == source["record_payloads"]["sha256"]
        and receipt.get("host_whole_payloads") == len(records)
        and receipt.get("producer_original_proof_sha256") == source["producer_proof"]["sha256"]
        and frozen.same_path(receipt["host_target"], source["consumer_target"])
        and frozen.same_path(receipt["producer_target"], source["producer_target"])
        and receipt.get("producer_rechecked_now") is False
        and receipt.get("actual_formal_cpu") is None
        and receipt.get("execution_release") is None
        and receipt.get("actual_gpu_qualifications") == dict.fromkeys(current.DEPLOYMENTS)
        and receipt.get("historical_rows_reused") is False
        and receipt.get("historical_token_replay_claimed") is False,
        "actual ARM source rendering differs",
    )
    origins(
        receipt["loaded_module_origins"],
        source["consumer_target"],
        records,
        frozen,
        required=("collector.fpm_forward.planner", "collector.fpm_forward.shards"),
    )
    original_root = frozen.key(source["original_prepared"])
    original_inventory = frozen.value(source["original_inventory"])
    require(isinstance(receipt.get("original_observed_files"), dict), "original geometry bytes absent")
    for name, digest in receipt["original_observed_files"].items():
        require(original_inventory.get(name) == digest, "original geometry is not in frozen inventory")
        frozen.bind(str(Path(original_root) / name), digest)
    groups, source_rows, old_ids, total = {}, {}, set(), 0
    crosswalk, parents = [], []
    for deployment in sorted(current.DEPLOYMENTS):
        group = []
        for role in ("calibration", "holdout"):
            part = root / "formal-inputs" / deployment / role
            old_part = Path(original_root) / "formal-inputs" / deployment / role
            old_parent, parent = (frozen.read(str(p / "collection-plan.json")) for p in (old_part, part))
            role_crosswalk = []
            old_shards, shards = (frozen.read(str(p / "shard-manifest.json"))["shards"] for p in (old_part, part))
            old_options, options = copy.deepcopy(old_parent["options"]), copy.deepcopy(parent["options"])
            old_mounts, mounts = old_options.pop("slurm_container_mounts"), options.pop("slurm_container_mounts")
            require(old_options == options and options["shard_token_budget"] == 100_000_000, "original options changed")
            policy(options, deployment)
            differences = [(a, b) for a, b in zip(old_mounts, mounts, strict=True) if a != b]
            require(
                len(differences) == 1
                and all(x.endswith(":/opt/glm53flash-current:ro") for x in differences[0])
                and differences[0][1] == source["producer_target"] + ":/opt/glm53flash-current:ro",
                "producer mount migration differs",
            )
            require(old_parent["sha256"] != parent["sha256"], "historical parent identity reused")
            require(parent["aic_revision"].startswith("installed:aisimulate=="), "installed planner identity missing")
            parents.append(
                dict(
                    deployment=deployment,
                    role=role,
                    original_plan_sha256=old_parent["sha256"],
                    new_plan_sha256=parent["sha256"],
                    original_planner=old_parent["aic_revision"],
                    new_planner=parent["aic_revision"],
                    policy_comparison={
                        "original_producer_mount": differences[0][0],
                        "new_producer_mount": differences[0][1],
                    },
                )
            )
            require(len(old_shards) == len(shards), "original whole-child partition changed")
            parent_points = old_parent["options"]["benchmark_points"]["payload"]
            require(
                all(len(parent_points[phase]) == current.POINTS[role, phase] for phase in ("prefill", "decode")),
                "parent phase denominator differs",
            )
            for old, new in zip(old_shards, shards, strict=True):
                require(
                    old["phase"] == new["phase"]
                    and old["point_map"] == new["point_map"]
                    and old["requested_real_tokens"] == new["requested_real_tokens"]
                    and old["child_cell_id"] != new["child_cell_id"]
                    and old["child_plan_sha256"] != new["child_plan_sha256"]
                    and old["parent_plan_sha256"] == old_parent["sha256"]
                    and new["parent_plan_sha256"] == parent["sha256"],
                    "original/new point crosswalk differs",
                )
                for index, point in enumerate(new["point_map"], 1):
                    number = point["original_point_id"]
                    require(
                        type(number) is int
                        and 1 <= number <= len(parent_points[new["phase"]])
                        and point["native_benchmark_id"] == index
                        and point["point"] == parent_points[new["phase"]][number - 1],
                        "child geometry is not the original parent point",
                    )
                require(old["child_cell_id"] not in old_ids, "duplicate historical child")
                old_ids.add(old["child_cell_id"])
                for point in new["point_map"]:
                    role_crosswalk.append(
                        dict(
                            phase=new["phase"],
                            original_point_id=point["original_point_id"],
                            geometry=point["point"],
                            original_child_id=old["child_cell_id"],
                            new_child_id=new["child_cell_id"],
                            original_native_benchmark_id=point["native_benchmark_id"],
                            new_native_benchmark_id=point["native_benchmark_id"],
                            original_parent_plan_sha256=old_parent["sha256"],
                            new_parent_plan_sha256=parent["sha256"],
                        )
                    )
                matches = [x for x in receipt["formal_children"] if x["child_cell_id"] == new["child_cell_id"]]
                require(len(matches) == 1, "new child receipt is missing or duplicated")
                row = matches[0]
                require(
                    row["new_identity"] == new
                    and row["original_identity"] == old
                    and row["deployment"] == deployment
                    and row["role"] == role
                    and row["phase"] == new["phase"]
                    and row["kind"] == "formal"
                    and row["child_plan_sha256"] == new["child_plan_sha256"]
                    and row["parent_plan_sha256"] == parent["sha256"]
                    and row["requested_allocator"] == allocator_policy(deployment)["max_split_size_mb"]
                    and row["new_attempt_id"] is None
                    and row["native_request_set"] is None,
                    "prepared child ownership differs",
                )
                native = part / "native" / new["child_cell_id"]
                require(frozen.read(str(native / "fixture.json")) == row, "native fixture differs from prepared child")
                child_plan = frozen.read(str(part / "plans" / (new["child_cell_id"] + ".json")))
                require(child_plan["aic_revision"] == parent["aic_revision"], "child/parent planner identity differs")
                precision, tp = deployment.rsplit("-tp", 1)
                require(
                    row["precision"] == precision
                    and row["tp"] == int(tp)
                    and child_plan["cells"][0]["weight_quantization"]
                    == {"fp8": "fp8_block", "nvfp4": "nvfp4"}[precision]
                    and child_plan["cells"][0]["topology"]["tp"] == int(tp),
                    "new child precision/topology differs",
                )
                require(frozen.key(row["native_directory"]) == str(native), "new native directory differs")
                for filename, digest_key in (
                    ("benchmark-points.json", "point_json_sha256"),
                    ("fpm_text.txt", "corpus_sha256"),
                ):
                    old_path = str(old_part / "native" / old["child_cell_id"] / filename)
                    new_path = str(native / filename)
                    require(
                        frozen.files[new_path] == frozen.files[old_path] == row[digest_key],
                        "original input bytes changed",
                    )
                old_argv = frozen.read(str(old_part / "native" / old["child_cell_id"] / "argv.json"))
                argv = frozen.read(str(native / "argv.json"))
                left, right = list(old_argv), list(argv)
                require(left.count("--run-id") == right.count("--run-id") == 1, "ambiguous native run identity")
                require(
                    left[left.index("--run-id") + 1] == old["child_cell_id"]
                    and right[right.index("--run-id") + 1] == new["child_cell_id"],
                    "derived native run ID differs",
                )
                left[left.index("--run-id") + 1] = right[right.index("--run-id") + 1] = None
                require(left == right, "native argv changed beyond derived identity")
                child = dict(row, original_point_ids=[p["original_point_id"] for p in new["point_map"]])
                group.append(child)
                source_rows[new["child_cell_id"]] = row
                total += len(new["point_map"])
            require(frozen.read(str(part / "point-crosswalk.json")) == role_crosswalk, "role point crosswalk differs")
            crosswalk.extend(dict(deployment=deployment, role=role, **p) for p in role_crosswalk)
        groups[deployment] = group
    require(receipt["parents"] == parents, "eight original/new parent identities differ")
    require(frozen.read(str(root / "point-crosswalk.json")) == crosswalk, "complete point crosswalk differs")
    require(
        len(receipt["formal_children"]) == 72 and total == receipt["point_crosswalk_count"] == 2472,
        "source point denominator differs",
    )
    return groups, current._children(groups), source_rows


def _cpu(admission, frozen, source, factory, records):
    cpu = admission["actual_cpu"]
    fields(cpu, {"job_id", "directory", "files"}, "actual CPU")
    require(type(cpu["job_id"]) is int and cpu["job_id"] > 0, "actual CPU job missing")
    root = Path(frozen.key(cpu["directory"]))
    require(root.name == str(cpu["job_id"]), "CPU job/root differs")
    frozen.mapping(cpu["files"], str(root))
    read = lambda suffix: frozen.read(str(root / suffix))
    result = read("result.json")
    public = read("transport/receipt.json")
    require(
        result.get("state") == "PUBLIC_ARM_FACTORY_AND_16_CPU_TRANSPORTS_PASS"
        and str(result.get("job")) == str(cpu["job_id"])
        and result.get("source") == source
        and result.get("transport_result") == public
        and result.get("GPU_execution") is False
        and result.get("formal_GPU_admission") is False,
        "actual CPU envelope is incomplete or failed",
    )
    bound = read("bound-inputs.json")
    require(bound == {"source": source, "release": result["release"]}, "CPU release/source chain differs")
    for mode in ("identity", "render", "transport"):
        entry = read("entry-" + mode + ".json")
        process = read("process-" + mode + "/process-result.json")
        require(
            entry.get("status") == "PASS"
            and entry.get("mode") == mode
            and entry.get("scope") == "CPU_SOURCE_RENDER_AND_TRANSPORT_ONLY"
            and entry.get("host_payloads") == entry.get("producer_payloads") == len(records)
            and entry.get("formal_GPU_admission") is False
            and process.get("returncode") == 0
            and not process.get("error")
            and not process.get("cleanup_error"),
            "CPU entry/process failed",
        )
        origins(entry["loaded_project_origins"], source["consumer_target"], records, frozen)
    for label in ("host", "producer"):
        installed(read("identity/" + label + "-installed-wheel.json"), admission[label], records, frozen)
    assets = frozen.value(source["assets"])
    require(read("identity/actual-image.json") == assets["images"]["sglang"], "CPU native image differs")
    release = read("transport-release.json")
    require(
        release.get("status") == "ROOT_REVIEWED_ARM_CPU_TRANSPORT"
        and str(release.get("job_id")) == str(cpu["job_id"])
        and release.get("source_sha256") == admission["factory_source"]["sha256"]
        and release.get("prepared_inventory_sha256") == frozen.files[str(root / "prepared/inventory.json")]
        and release.get("factory_manifest_sha256") == admission["factory_manifest"]["sha256"]
        and release.get("original_inventory_sha256") == source["original_inventory"]["sha256"],
        "dynamic sixteen-fixture release differs",
    )
    groups, children, source_rows = _prepared(frozen, root, source, factory, records)
    require(public.get("status") == "16_PUBLIC_CPU_TRANSPORT_FIXTURES_PASS", "sixteen CPU fixtures did not finish")
    require(
        public.get("formal_admission") is False and public.get("formal_native_requests") == "NOT_EVALUATED",
        "CPU fixture receipt claims native admission",
    )
    selected = {}
    for row in source_rows.values():
        selected.setdefault(tuple(row[k] for k in ("deployment", "role", "phase")), row)
    selection = read("prepared/cpu-transport-selection.json")
    require(
        selection.get("fixtures") == list(selected.values())
        and selection.get("execution_release") is None
        and selection.get("native_GPU_qualification") is False,
        "original CPU transport selection differs",
    )
    require(
        [row["child"] for row in public["rows"]] == selection["fixtures"],
        "original sixteen-fixture order/coverage differs",
    )
    observed, attempts, native_audits = set(), set(), []
    for row in public["rows"]:
        child = row["child"]
        key = tuple(child[k] for k in ("deployment", "role", "phase"))
        require(key not in observed and selected.get(key) == child, "CPU fixture selection/coverage differs")
        observed.add(key)
        directory = root / "transport" / "-".join(key)
        require(read(str(directory.relative_to(root) / "receipt.json")) == row, "CPU fixture original receipt differs")
        require(
            row.get("status") == "PASS"
            and row.get("scope") == "ACTUAL_CPU_TRANSPORT_ONLY_NO_ENGINE"
            and row.get("attempt_id")
            and row["attempt_id"] not in attempts,
            "CPU fixture failed/reused attempt",
        )
        attempts.add(row["attempt_id"])
        require(
            frozen.same_path(
                row["slurm_source"], str(Path(source["consumer_target"]) / "collector/fpm_forward/slurm.py")
            )
            and row["slurm_source_sha256"] == records["collector/fpm_forward/slurm.py"],
            "public CPU transport source differs",
        )
        candidates = [
            Path(p).parent
            for p in frozen.files
            if Path(p).is_relative_to(directory / "raw") and Path(p).name == "collector-provenance.json"
        ]
        require(len(candidates) == 1, "CPU raw pod is missing or ambiguous")
        raw = candidates[0]
        frozen.mapping(row["raw_files"], str(raw))
        raw_read = lambda suffix: frozen.read(str(raw / suffix))
        provenance = raw_read("collector-provenance.json")
        require(
            row["provenance_sha256"] == frozen.files[str(raw / "collector-provenance.json")]
            and (provenance.get("attempt_id"), provenance.get("cell_id"), provenance.get("plan_sha256"))
            == (row["attempt_id"], child["child_cell_id"], child["child_plan_sha256"])
            and provenance.get("runtime") == {"backend": "sglang", "backend_version": "0.5.20"},
            "CPU public collector attempt differs",
        )
        installed(
            raw_read("actual-container-installed-wheel.json"), admission["producer"], records, frozen, container=True
        )
        before, after = (raw_read(k + "-identity.json") for k in ("preparation", "execution"))
        strip = lambda item: {k: v for k, v in item.items() if k not in {"cache_receipt", "cache_receipt_sha256"}}
        require(strip(before) == strip(after), "CPU preparation/execution identity changed")
        caches = []
        for kind, value in (("preparation", before), ("execution", after)):
            require(
                value.get("status") == "CPU_PUBLIC_FIXTURE_PASS"
                and value.get("mode") == "identity"
                and value.get("fixture") == child
                and value.get("model_constructed") is False
                and value.get("cuda_initialized") is False
                and value.get("formal_admission") is False
                and value.get("source_commit") == factory["source_commit"]
                and value.get("collector_sha256") == factory["collector_sha256"]
                and value.get("driver_file") == "/opt/glm53flash-current/collector/fpm_forward/sglang_driver.py"
                and value.get("driver_sha256") == records["collector/fpm_forward/sglang_driver.py"]
                and value.get("backend_version") == "0.5.20",
                "CPU native import identity differs",
            )
            native = Path(frozen.key(child["native_directory"]))
            require(
                value["argv"] == frozen.read(str(native / "argv.json"))
                and value["argv_file_sha256"] == frozen.files[str(native / "argv.json")],
                "CPU native argv differs",
            )
            require(
                value["requested_allocator_policy"] == allocator_policy(child["deployment"]), "CPU allocator differs"
            )
            cache = value["cache_receipt"]
            cache_path = str(raw / ("cache-setup-" + str(cache["pid"]) + ".json"))
            require(
                type(cache["pid"]) is int
                and cache["pid"] > 0
                and str(cache["job_id"]) == str(cpu["job_id"])
                and frozen.read(cache_path) == cache
                and frozen.files[cache_path] == value["cache_receipt_sha256"],
                "CPU original process/cache witness differs",
            )
            caches.append(cache)
            audit = raw_read(kind + "-identity-native-source/runtime-preflight.json")
            require(
                value["native_runtime_source_audit"] == audit
                and audit.get("status") == "passed"
                and audit.get("backend") == "sglang"
                and audit.get("backend_version") == "0.5.20",
                "CPU native runtime source audit differs",
            )
            digest_map(audit["sources"])
            native_audits.append(audit)
        require(
            caches[0]["pid"] != caches[1]["pid"] and caches[0]["root"] != caches[1]["root"],
            "CPU processes/cache reused",
        )
    require(observed == set(selected) and len(observed) == 16, "sixteen role/phase transports incomplete")
    require(all(a == native_audits[0] for a in native_audits), "CPU native runtime source identities differ")
    return root, groups, children, source_rows


def allocator_policy(deployment):
    return {
        "schema": "sglang_native_allocator_policy_v1",
        "backend": "native",
        "max_split_size_mb": 16384 if deployment == "fp8-tp2" else None,
    }


def _qualification(frozen, refs, deployment, phase, host, producer, records, points):
    fields(refs, QUALIFICATION_FIELDS, "new SGLang qualification")
    values = {key: frozen.value(refs[key]) for key in QUALIFICATION_FIELDS - {"reader_manifest", "idle"}}
    reader, aggregate, rows, plan, start, checkpoint = (
        values[k] for k in ("reader", "aggregation", "rows", "plan", "started", "checkpoint")
    )
    manifest = refs["reader_manifest"]
    frozen.manifest(frozen.key(manifest["path"]), manifest["sha256"])
    require(reader["source_manifest_sha256"] == manifest["sha256"], "strict reader source manifest differs")
    installed(values["installed_reader"], host, records, frozen)
    precision, tp = deployment.rsplit("-tp", 1)
    tp = int(tp)
    cells = [c for c in plan["cells"] if c["workload_kind"] == phase]
    require(len(cells) == 1, "qualification phase is ambiguous")
    cell = cells[0]
    policy(plan["options"], deployment)
    require(
        plan.get("backend") == "sglang"
        and cell["topology"]["tp"] == tp
        and cell.get("weight_quantization") == {"fp8": "fp8_block", "nvfp4": "nvfp4"}[precision]
        and plan["options"]["benchmark_points"]["payload"] == points,
        "qualification original geometry differs",
    )
    require(
        (start.get("source_commit"), start.get("wheel_sha256")) == (producer["source_commit"], producer["wheel_sha256"])
        and start.get("deployment") == deployment
        and start.get("state") == "RUNNING"
        and reader.get("status") == "passed"
        and reader.get("returncode") == 0
        and reader.get("phase") == phase
        and reader.get("deployment") == deployment
        and str(reader.get("source_job")) == str(start["job"])
        and reader.get("started_sha256") == refs["started"]["sha256"]
        and reader.get("checkpoint_sha256") == refs["checkpoint"]["sha256"]
        and aggregate.get("status") == aggregate.get("validation") == aggregate.get("aggregation") == "passed"
        and (aggregate.get("reader_source_revision"), aggregate.get("installed_wheel_sha256"))
        == (producer["source_commit"], producer["wheel_sha256"])
        and aggregate.get("frozen_plan_file_sha256") == refs["plan"]["sha256"]
        and aggregate.get("raw_directory") == reader.get("raw_directory")
        and aggregate.get("cell") == cell
        and aggregate.get("row_count") == len(rows) == (6 if phase == "prefill" else 3),
        "new SGLang strict qualification identity/result differs",
    )
    require(
        set(aggregate["installed_source_hashes"]) == READER_SOURCES, "qualification reader source map is incomplete"
    )
    require(
        aggregate.get("validation_performed_by") == "unmodified_database.aggregate_cell -> validate_native_collection"
        and aggregate.get("original_point_coordinate_coverage") == "passed"
        and aggregate.get("native_hardware_contract_validation") == "passed",
        "qualification strict reader/hardware acceptance missing",
    )
    for name, digest in aggregate["installed_source_hashes"].items():
        require(records.get("collector/" + name) == digest, "qualification reader source bytes differ")
    require(
        frozen.same_path(
            aggregate["reader_source"], str(Path(host["target"]) / "collector/fpm_forward/native_artifact.py")
        )
        and aggregate["reader_source_sha256"] == records["collector/fpm_forward/native_artifact.py"],
        "qualification reader module origin differs",
    )
    entry = checkpoint["cells"][cell["cell_id"]]
    require(
        start["plan_sha256"]
        == reader["plan_sha256"]
        == aggregate["plan_sha256"]
        == checkpoint["plan_sha256"]
        == plan["sha256"]
        and entry["status"] == "passed"
        and entry.get("requested_point_count") == entry.get("measured_point_count") == len(rows)
        and reader["attempt_id"] == aggregate["attempt_id"] == entry["attempt_id"],
        "qualification original attempt differs",
    )
    names = ("batch_size", "total_kv_read_tokens") + (("total_prefill_tokens",) if phase == "prefill" else ())
    coordinates = lambda values: [tuple(v.get(k) for k in names) for v in values]
    require(
        all(
            all(type(v.get(k)) is int and v[k] >= 0 for k in names) and v["batch_size"] > 0
            for v in rows + points[phase]
        ),
        "invalid qualification coordinate",
    )
    actual = coordinates(rows)
    require(
        len(set(actual)) == len(actual) and set(actual) == set(coordinates(points[phase])),
        "qualification point union differs",
    )
    policies = set()
    for row in rows:
        observed = row.get("sglang_allocator_policy_sha256")
        require(
            isinstance(observed, str) and re.fullmatch(r"[0-9a-f]{64}", observed), "unknown actual allocator policy"
        )
        policies.add(observed)
        require(
            row.get("sglang_allocator_max_split_size_mb") == allocator_policy(deployment)["max_split_size_mb"]
            and row.get("backend") == "sglang"
            and row.get("tp") == tp
            and row.get("cell_id") == cell["cell_id"]
            and row.get("workload_kind") == phase
            and row.get("source_plan_sha256") == plan["sha256"]
            and row.get("collector_attempt_id") == entry["attempt_id"]
            and row.get("runtime_run_id") == aggregate["runtime_run_id"]
            and row.get("runtime_grid_digest") == aggregate["runtime_grid_digest"]
            and row.get("warmup_repeats") == 5
            and row.get("measurement_repeats") == 10,
            "qualification original row provenance differs",
        )
    require(len(policies) == 1, "qualification mixes allocator policies")
    if phase == "prefill":
        fields(refs["idle"], {"start", "observed", "return"}, "prefill idle evidence")
        idle = {k: frozen.value(v) for k, v in refs["idle"].items()}
        before, after, returned = (idle[k] for k in ("start", "observed", "return"))
        require(
            before.get("producer_commit")
            == after.get("producer_commit")
            == returned.get("producer_commit")
            == producer["source_commit"]
            and before.get("idle_seconds") == 660
            and after.get("idle_elapsed_seconds", 0) >= 660
            and before.get("native_watchdog_timeout_changed") is False
            and before.get("direct_marker_observation") is False
            and after.get("direct_marker_observation") is False
            and before.get("post_idle_health_budget_seconds") == after.get("post_idle_health_budget_seconds") == 900
            and before["workers"] == after["workers_before"] == after["workers_after"]
            and len(before["workers"]) == tp
            and after.get("native_health_response")
            and returned.get("original_reader_calls") == 1,
            "qualification post-release idle/health proof differs",
        )
        require(
            {w["rank"] for w in before["workers"]} == set(range(tp))
            and len({(w["pid"], w["start_ticks"]) for w in before["workers"]}) == tp
            and all(
                type(w["pid"]) is int and w["pid"] > 0 and type(w["start_ticks"]) is int and w["start_ticks"] > 0
                for w in before["workers"]
            ),
            "qualification worker lifetime identity differs",
        )
        require(
            reader.get("idle_proof") == {Path(v["path"]).name: v["sha256"] for v in refs["idle"].values()},
            "strict reader did not bind original idle evidence",
        )
    else:
        require(refs["idle"] is None, "decode must not invent an idle probe")
    return policies.pop()


def frozen_contract(document, get):
    require(document.get("schema") == SCHEMA and document.get("adapter") == ADAPTER, "unsupported factory v4 contract")
    anchors = document["anchors"]
    fields(anchors, {"launcher_manifest", "admission"}, "factory anchors")
    initial = current.Frozen(get)
    members = initial.manifest(anchors["launcher_manifest"])
    require(anchors["admission"] in members, "admission is outside launcher manifest")
    admission = initial.read(anchors["admission"])
    fields(admission, ADMISSION_FIELDS, "factory admission")
    require(
        admission["schema"] == ADMISSION and admission["status"] == "FROZEN_PUBLIC_FACTORY_FORMAL",
        "factory admission is not frozen",
    )
    task = current._control().absolute(document["original_task_root"])
    frozen = Evidence(get, task, admission["storage_binding"])
    frozen.files.update(initial.files)
    source, factory, records = _sources(admission, frozen)
    cpu_root, groups, children, source_rows = _cpu(admission, frozen, source, factory, records)
    points = frozen.value(admission["qualification_points"])
    require(set(admission["qualifications"]) == current.DEPLOYMENTS, "four deployment qualifications required")
    for deployment, phases in admission["qualifications"].items():
        fields(phases, {"prefill", "decode"}, "qualification phases")
        observed = {
            _qualification(
                frozen, phases[phase], deployment, phase, admission["host"], admission["producer"], records, points
            )
            for phase in ("prefill", "decode")
        }
        require(len(observed) == 1, "qualification phases have different allocator identities")
    context = dict(
        backend="sglang",
        version="0.5.20",
        children=children,
        groups=groups,
        host=current._identity(admission["host"]["source_commit"], admission["host"]["wheel_sha256"]),
        producer=current._identity(admission["producer"]["source_commit"], admission["producer"]["wheel_sha256"]),
        cpu_root=str(cpu_root),
        source=source,
        records=records,
        source_rows=source_rows,
        qualifications=admission["qualifications"],
        admission=admission,
        files=frozen.files,
    )
    # Existing literal environment/plan guards; only the explicit storage proof
    # resolves the original cache mount's root. No original bytes are rewritten.
    current._native_plans(
        context,
        frozen,
        task,
        {"cache_hook": frozen.key(admission["cache_hook"]["path"])},
        storage_root_binding=admission["storage_binding"],
    )
    return context


def closure(document, get):
    context = frozen_contract(document, get)
    admission, expected = context["admission"], dict(context["files"])
    task = current._control().absolute(document["original_task_root"])
    frozen = Evidence(get, task, admission["storage_binding"])
    runs = {r["cell_id"]: r for r in document["runs"]}
    require(
        len(runs) == len(document["runs"]) and set(runs) == set(context["children"]),
        "all seventy-two formal executions required",
    )
    attempts = set()
    for cid, run in runs.items():
        child = context["source_rows"][cid]
        start, provenance = (json.loads(get(run[k])) for k in ("started", "collector_provenance"))
        fields(start, STARTED_FIELDS, "factory started")
        job_root = Path(run["started"]).parent
        raw = current._control().absolute(run["raw_root"])
        require(
            raw.is_relative_to(task / job_root)
            and task / run["collector_provenance"] == raw / "collector-provenance.json",
            "formal original execution escaped job/raw root",
        )
        require(
            start["schema"] == STARTED
            and start["state"] == "RUNNING"
            and start["mode"] == "formal"
            and isinstance(start["job"], str)
            and start["job"].isdecimal()
            and start["job"] == job_root.name
            and start["deployment"] == child["deployment"]
            and start["selected"] == child
            and start["plan_sha256"] == child["child_plan_sha256"]
            and start["actual_cpu_job"] == admission["actual_cpu"]["job_id"]
            and start["qualification"] == admission["qualifications"][child["deployment"]]
            and start["requested_allocator_policy"] == allocator_policy(child["deployment"])
            and start["admission_sha256"] == expected[document["anchors"]["admission"]]
            and start["launcher_manifest_sha256"] == expected[document["anchors"]["launcher_manifest"]],
            "formal started ownership/qualification differs",
        )
        for label, prefix in (("host", "host_"), ("producer", "")):
            require(
                (start[prefix + "source_commit"], start[prefix + "wheel_sha256"]) == context[label],
                "formal source identity differs",
            )
            path = run[label + "_wheel_verification"]
            require(Path(path) == job_root / ("actual-" + label + "-wheel.json"), "formal installed proof escaped job")
            installed(json.loads(get(path)), admission[label], context["records"], frozen)
        require(
            provenance.get("cell_id") == cid
            and provenance.get("plan_sha256") == child["child_plan_sha256"]
            and provenance.get("attempt_id")
            and provenance.get("runtime") == {"backend": "sglang", "backend_version": "0.5.20"},
            "formal fresh collector attempt differs",
        )
        require(provenance["attempt_id"] not in attempts, "formal collector attempt was reused")
        attempts.add(provenance["attempt_id"])
        for path in current._control().execution_paths(document, run):
            current._control().relative(path)
            require(path not in expected, "execution control aliases another original")
            expected[path] = current._control().sha(get(path))
    return expected, admission


def history_identity(start, final):
    """New attempt history keeps the full new/historical crosswalk, not a legacy view."""
    fields(start, STARTED_FIELDS, "factory history started")
    require(
        start["schema"] == STARTED
        and start["state"] == "RUNNING"
        and final.get("schema") == FINAL
        and final.get("state") in {"COLLECTION_PASSED", "COLLECTION_FAILED_PRESERVED", "FAILED_PRESERVED"},
        "factory history schema/state differs",
    )
    require(
        all(final.get(k) == start[k] for k in STARTED_FIELDS - {"schema", "state"}),
        "factory history final identity changed",
    )
    selected = start["selected"]
    require(
        selected.get("kind") == "formal"
        and selected["new_identity"]["child_cell_id"] == selected["child_cell_id"]
        and selected["new_identity"]["child_plan_sha256"] == selected["child_plan_sha256"]
        and selected["original_identity"]["child_cell_id"] != selected["child_cell_id"],
        "factory history new child ownership differs",
    )
    return selected["new_identity"]


def live_prepare_storage(document, get):
    """prepare runs at the original storage host; portable validation does not."""
    admission = json.loads(get(document["anchors"]["admission"]))
    require(admission.get("schema") == ADMISSION, "v4 prepare requires its explicit original admission")
    binding = admission.get("storage_binding")
    if binding is not None:
        try:
            archive.validate_storage_binding(binding, live=True)
        except (OSError, ValueError) as error:
            raise ValueError(
                "v4 prepare requires the live original storage binding before and after copy; "
                "use offline validate for an already copied attachment"
            ) from error
