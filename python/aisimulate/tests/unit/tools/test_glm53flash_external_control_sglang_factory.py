# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY complete v4 controls. No fixture is measured data or GPU admission."""

import copy
import json
from pathlib import Path

import pytest
from tools.glm53flash_hf import external_control as control
from tools.glm53flash_hf import external_control_current as current
from tools.glm53flash_hf import external_control_sglang_factory as factory
from tools.glm53flash_hf import glm53flash as publication
from tools.glm53flash_hf import raw_archive as archive

from . import test_glm53flash_external_control_current as legacy

pytestmark = pytest.mark.unit
SHA = legacy.sha
COMMIT = "2" * 40
WHEEL = "3" * 64


class Fixture(legacy.Fixture):
    def __init__(self, mutate=None, task=legacy.TASK, original=None, storage=None):
        super().__init__()
        self.mutate = mutate or (lambda _path, _value: None)
        self.task = Path(task)
        self.original = Path(original or task)
        self.storage = storage

    def put(self, path, value):
        value = copy.deepcopy(value)
        self.mutate(path, value)
        return super().put(path, value)

    def read(self, path):
        return json.loads(self.files[path])

    def inventory(self, root):
        return {str(Path(k).relative_to(root)): SHA(v) for k, v in self.files.items() if Path(k).is_relative_to(root)}

    def absolute_ref(self, ref):
        return dict(ref, path=str(self.original / ref["path"]))


def fixture(mutate=None, **kwargs):
    f = Fixture(mutate, **kwargs)
    host = dict(source_commit=COMMIT, wheel_sha256=WHEEL, target=str(f.task / "host"))
    producer = dict(source_commit=COMMIT, wheel_sha256=WHEEL, target=str(f.original / "producer"))
    records = {
        name: SHA(name.encode())
        for name in (
            "aisimulate/_runtime.abi3.so",
            "collector/fpm_forward/slurm.py",
            "collector/fpm_forward/sglang_driver.py",
            "collector/fpm_forward/planner.py",
            "collector/fpm_forward/shards.py",
            "collector/fpm_forward/native_artifact.py",
        )
    }
    records.update({"collector/" + name: SHA(name.encode()) for name in factory.READER_SOURCES})
    records.update({f"TEST_ONLY/payload-{i}": SHA(str(i).encode()) for i in range(2384 - len(records))})
    record_ref = f.ref("wheel/record-payloads.json", records)
    collector = {k: v for k, v in records.items() if k.startswith("collector/")}

    def proof(identity, container=False):
        return dict(
            state="EXACT_INSTALLED_WHEEL_RECORD_SOURCE_ELF_PASS",
            source=COMMIT,
            wheel_sha256=WHEEL,
            runtime_sha256=records["aisimulate/_runtime.abi3.so"],
            files=records,
            target="/opt/glm53flash-current" if container else identity["target"],
        )

    def origins():
        return {
            "collector.fpm_forward." + n: {
                "path": str(f.task / "host/collector/fpm_forward" / (n + ".py")),
                "sha256": records["collector/fpm_forward/" + n + ".py"],
            }
            for n in ("planner", "shards")
        }

    host_proof = f.ref("installs/host.json", proof(host))
    producer_proof = f.ref("installs/producer.json", proof(producer))
    cache = f.ref("old-factory/cache-hook/sitecustomize.py", b"# TEST_ONLY original cache\n")
    f.put("factory/cache-hook/sitecustomize.py", f.files[cache["path"]])
    prepared, originals = "cpu/100/prepared", "old-prepared"
    children, parents, crosswalk = [], [], []
    for deployment, group in legacy.groups().items():
        for role in ("calibration", "holdout"):
            prefix = f"formal-inputs/{deployment}/{role}"
            options = dict(
                dataset_role=role,
                sglang_mem_fraction_static=0.82,
                shard_token_budget=100_000_000,
                sglang_allocator_max_split_size_mb=16384 if deployment == "fp8-tp2" else None,
                slurm_container_mounts=[
                    str(f.original / "old-factory/cache-hook") + ":/opt/glm53flash-cache:ro",
                    producer["target"] + ":/opt/glm53flash-current:ro",
                ],
            )
            options["benchmark_points"] = {
                "payload": {
                    phase: [
                        dict(
                            batch_size=1,
                            total_kv_read_tokens=i,
                            **({"total_prefill_tokens": 1} if phase == "prefill" else {}),
                        )
                        for i in range(1, current.POINTS[role, phase] + 1)
                    ]
                    for phase in ("prefill", "decode")
                }
            }
            old_options = copy.deepcopy(options)
            old_options["slurm_container_mounts"][-1] = str(f.original / "old-producer") + ":/opt/glm53flash-current:ro"
            parent_hash = SHA((deployment + role).encode())
            f.put(
                f"{prepared}/{prefix}/collection-plan.json",
                dict(sha256=parent_hash, options=options, aic_revision="installed:aisimulate==TEST_ONLY"),
            )
            f.put(
                f"{originals}/{prefix}/collection-plan.json",
                dict(sha256="8" * 64, options=old_options, aic_revision="installed:OLD_TEST_ONLY"),
            )
            parents.append(
                dict(
                    deployment=deployment,
                    role=role,
                    original_plan_sha256="8" * 64,
                    new_plan_sha256=parent_hash,
                    original_planner="installed:OLD_TEST_ONLY",
                    new_planner="installed:aisimulate==TEST_ONLY",
                    policy_comparison={
                        "original_producer_mount": old_options["slurm_container_mounts"][-1],
                        "new_producer_mount": options["slurm_container_mounts"][-1],
                    },
                )
            )
            role_crosswalk = []
            old_shards, shards = [], []
            for child in (c for c in group if c["role"] == role):
                cid, phase = child["child_cell_id"], child["phase"]
                points = [
                    {
                        "original_point_id": n,
                        "native_benchmark_id": i + 1,
                        "point": {
                            "batch_size": 1,
                            "total_kv_read_tokens": n,
                            **({"total_prefill_tokens": 1} if phase == "prefill" else {}),
                        },
                    }
                    for i, n in enumerate(child["original_point_ids"])
                ]
                new = {k: child[k] for k in ("child_cell_id", "child_plan_sha256", "parent_plan_sha256", "phase")}
                new.update(point_map=points, requested_real_tokens=len(points) * 100)
                old = dict(
                    new,
                    child_cell_id="OLD-" + cid,
                    child_plan_sha256=SHA(("OLD-" + cid).encode()),
                    parent_plan_sha256="8" * 64,
                )
                for point in points:
                    role_crosswalk.append(
                        dict(
                            phase=phase,
                            original_point_id=point["original_point_id"],
                            geometry=point["point"],
                            original_child_id=old["child_cell_id"],
                            new_child_id=cid,
                            original_native_benchmark_id=point["native_benchmark_id"],
                            new_native_benchmark_id=point["native_benchmark_id"],
                            original_parent_plan_sha256="8" * 64,
                            new_parent_plan_sha256=parent_hash,
                        )
                    )
                shards.append(new)
                old_shards.append(old)
                native, old_native = f"{prepared}/{prefix}/native/{cid}", f"{originals}/{prefix}/native/OLD-{cid}"
                args = [
                    "python3",
                    "-m",
                    "collector.fpm_forward.sglang_driver",
                    "--run-id",
                    cid,
                    "--warmup-iterations",
                    "5",
                    "--iterations",
                    "10",
                ]
                old_args = list(args)
                old_args[4] = "OLD-" + cid
                point_sha = f.put(native + "/benchmark-points.json", {phase: [p["point"] for p in points]})
                corpus = f.put(native + "/fpm_text.txt", ("TEST_ONLY_" + role).encode())
                f.put(old_native + "/benchmark-points.json", f.files[native + "/benchmark-points.json"])
                f.put(old_native + "/fpm_text.txt", f.files[native + "/fpm_text.txt"])
                f.put(old_native + "/argv.json", old_args)
                f.put(native + "/argv.json", args)
                f.put(
                    native + "/collector-runtime-env.sh",
                    b"export PYTHONPATH=/opt/glm53flash-cache:/opt/glm53flash-current:/opt/glm53flash-sg-deps\n",
                )
                f.put(
                    f"{prepared}/{prefix}/plans/{cid}.json",
                    dict(
                        sha256=child["child_plan_sha256"],
                        backend="sglang",
                        capability={"aic_database_version": "0.5.20"},
                        aic_revision="installed:aisimulate==TEST_ONLY",
                        options=options,
                        cells=[
                            dict(
                                cell_id=cid,
                                workload_kind=phase,
                                topology={"tp": int(deployment[-1])},
                                weight_quantization={"fp8": "fp8_block", "nvfp4": "nvfp4"}[deployment.split("-tp")[0]],
                            )
                        ],
                    ),
                )
                children.append(
                    dict(
                        child_cell_id=cid,
                        child_plan_sha256=child["child_plan_sha256"],
                        parent_plan_sha256=parent_hash,
                        deployment=deployment,
                        precision=deployment.split("-tp")[0],
                        tp=int(deployment[-1]),
                        role=role,
                        phase=phase,
                        kind="formal",
                        new_identity=new,
                        original_identity=old,
                        native_directory=str(f.task / native),
                        point_json_sha256=point_sha,
                        corpus_sha256=corpus,
                        requested_allocator=16384 if deployment == "fp8-tp2" else None,
                        new_attempt_id=None,
                        native_request_set=None,
                    )
                )
                f.put(native + "/fixture.json", children[-1])
            f.put(f"{prepared}/{prefix}/point-crosswalk.json", role_crosswalk)
            crosswalk.extend(dict(deployment=deployment, role=role, **p) for p in role_crosswalk)
            f.put(f"{prepared}/{prefix}/shard-manifest.json", {"shards": shards})
            f.put(f"{originals}/{prefix}/shard-manifest.json", {"shards": old_shards})
    observed = f.inventory(originals)
    original_inventory = f.ref(originals + "/inventory.json", observed)
    record = dict(
        record_map_sha256=record_ref["sha256"],
        record_payloads=len(records),
        wheel_sha256=WHEEL,
        runtime_sha256=records["aisimulate/_runtime.abi3.so"],
    )
    source_factory = dict(
        contract="sg2d51_formal_public_factory_v1",
        source_commit=COMMIT,
        wheel_sha256=WHEEL,
        records={"arm64": record},
        record_payloads_sha256=record_ref["sha256"],
        collector_sha256=collector,
        runtime_sha256=record["runtime_sha256"],
        producer_proof_sha256=producer_proof["sha256"],
        original_prepared_inventory=original_inventory["sha256"],
        cache_sha256=cache["sha256"],
        copied_sources={"cache-hook/sitecustomize.py": dict(original=cache["path"], sha256=cache["sha256"])},
    )
    factory_ref = f.absolute_ref(f.ref("factory/source.json", source_factory))
    factory_manifest = f.absolute_ref(
        {
            "path": "factory/manifest.sha256",
            "sha256": f.manifest("factory", ["factory/source.json", "factory/cache-hook/sitecustomize.py"]),
        }
    )
    assets = f.absolute_ref(
        f.ref("assets.json", {"images": {"sglang": {"path": "TEST_ONLY.sqsh", "sha256": "b" * 64}}})
    )
    source = dict(
        contract="sg2d51_formal_public_arm_cpu_envelope_v1",
        source_commit=COMMIT,
        wheel={"sha256": WHEEL},
        consumer_target=host["target"],
        producer_target=producer["target"],
        factory_manifest=factory_manifest,
        factory_source=factory_ref,
        record_payloads=f.absolute_ref(record_ref),
        host_proof=f.absolute_ref(host_proof),
        producer_proof=f.absolute_ref(producer_proof),
        original_inventory=f.absolute_ref(original_inventory),
        original_prepared=str(f.original / originals),
        assets=assets,
        host_import=f.absolute_ref(f.ref("installs/import.json", {"scope": "TEST_ONLY"})),
    )
    cpu_source = f.absolute_ref(f.ref("controller/source.json", source))
    cpu_manifest = f.absolute_ref(
        {"path": "controller/manifest.sha256", "sha256": f.manifest("controller", ["controller/source.json"])}
    )
    f.put(prepared + "/point-crosswalk.json", crosswalk)
    selected_source = {}
    for child in children:
        selected_source.setdefault(tuple(child[k] for k in ("deployment", "role", "phase")), child)
    f.put(
        prepared + "/cpu-transport-selection.json",
        dict(fixtures=list(selected_source.values()), execution_release=None, native_GPU_qualification=False),
    )
    f.put(
        prepared + "/receipt.json",
        dict(
            status="INSTALLED_PUBLIC2D51_SOURCE_RENDER_PASS_NOT_QUALIFIED",
            host_arch="arm64",
            source_commit=COMMIT,
            host_record_map_sha256=record_ref["sha256"],
            host_whole_payloads=len(records),
            producer_original_proof_sha256=producer_proof["sha256"],
            host_target=host["target"],
            producer_target=producer["target"],
            producer_rechecked_now=False,
            actual_formal_cpu=None,
            execution_release=None,
            actual_gpu_qualifications=dict.fromkeys(current.DEPLOYMENTS),
            historical_rows_reused=False,
            historical_token_replay_claimed=False,
            loaded_module_origins=origins(),
            original_observed_files=observed,
            parents=parents,
            formal_children=children,
            point_crosswalk_count=2472,
        ),
    )
    f.put(prepared + "/inventory.json", f.inventory(prepared))
    for mode in ("identity", "render", "transport"):
        f.put(
            "cpu/100/entry-" + mode + ".json",
            dict(
                status="PASS",
                mode=mode,
                scope="CPU_SOURCE_RENDER_AND_TRANSPORT_ONLY",
                host_payloads=len(records),
                producer_payloads=len(records),
                formal_GPU_admission=False,
                loaded_project_origins=origins(),
            ),
        )
        f.put("cpu/100/process-" + mode + "/process-result.json", {"returncode": 0})
    for label, identity in (("host", host), ("producer", producer)):
        f.put("cpu/100/identity/" + label + "-installed-wheel.json", proof(identity))
    f.put("cpu/100/identity/actual-image.json", {"path": "TEST_ONLY.sqsh", "sha256": "b" * 64})
    f.put(
        "cpu/100/transport-release.json",
        dict(
            status="ROOT_REVIEWED_ARM_CPU_TRANSPORT",
            job_id=100,
            source_sha256=factory_ref["sha256"],
            prepared_inventory_sha256=SHA(f.files[prepared + "/inventory.json"]),
            factory_manifest_sha256=factory_manifest["sha256"],
            original_inventory_sha256=original_inventory["sha256"],
        ),
    )
    selected = {}
    for child in children:
        selected.setdefault(tuple(child[k] for k in ("deployment", "role", "phase")), child)
    transport_rows = []
    for i, (key, child) in enumerate(selected.items()):
        directory = "cpu/100/transport/" + "-".join(key)
        raw = directory + "/raw/node0000"
        attempt = SHA(str(i).encode())[:32]
        provenance = dict(
            attempt_id=attempt,
            cell_id=child["child_cell_id"],
            plan_sha256=child["child_plan_sha256"],
            runtime={"backend": "sglang", "backend_version": "0.5.20"},
        )
        provenance_sha = f.put(raw + "/collector-provenance.json", provenance)
        f.put(raw + "/actual-container-installed-wheel.json", proof(producer, container=True))
        native = str(Path(child["native_directory"]).relative_to(f.task))
        for j, kind in enumerate(("preparation", "execution")):
            pid = i * 10 + j + 1
            cache_receipt = dict(pid=pid, job_id="100", root="/TEST_ONLY/cache/" + str(pid))
            cache_sha = f.put(raw + f"/cache-setup-{pid}.json", cache_receipt)
            audit = dict(
                status="passed", backend="sglang", backend_version="0.5.20", sources={"TEST_ONLY.py": "c" * 64}
            )
            f.put(raw + "/" + kind + "-identity-native-source/runtime-preflight.json", audit)
            f.put(
                raw + "/" + kind + "-identity.json",
                dict(
                    status="CPU_PUBLIC_FIXTURE_PASS",
                    mode="identity",
                    fixture=child,
                    model_constructed=False,
                    cuda_initialized=False,
                    formal_admission=False,
                    source_commit=COMMIT,
                    collector_sha256=collector,
                    driver_file="/opt/glm53flash-current/collector/fpm_forward/sglang_driver.py",
                    driver_sha256=records["collector/fpm_forward/sglang_driver.py"],
                    backend_version="0.5.20",
                    argv=f.read(native + "/argv.json"),
                    argv_file_sha256=SHA(f.files[native + "/argv.json"]),
                    requested_allocator_policy=factory.allocator_policy(key[0]),
                    cache_receipt=cache_receipt,
                    cache_receipt_sha256=cache_sha,
                    native_runtime_source_audit=audit,
                ),
            )
        row = dict(
            child=child,
            attempt_id=attempt,
            status="PASS",
            scope="ACTUAL_CPU_TRANSPORT_ONLY_NO_ENGINE",
            slurm_source=str(f.task / "host/collector/fpm_forward/slurm.py"),
            slurm_source_sha256=records["collector/fpm_forward/slurm.py"],
            provenance_sha256=provenance_sha,
            raw_files=f.inventory(raw),
        )
        f.put(directory + "/receipt.json", row)
        transport_rows.append(f.read(directory + "/receipt.json"))
    public = dict(
        status="16_PUBLIC_CPU_TRANSPORT_FIXTURES_PASS",
        rows=transport_rows,
        formal_admission=False,
        formal_native_requests="NOT_EVALUATED",
    )
    f.put("cpu/100/transport/receipt.json", public)
    f.put("cpu/100/bound-inputs.json", dict(source=source, release={"TEST_ONLY": True}))
    f.put(
        "cpu/100/result.json",
        dict(
            state="PUBLIC_ARM_FACTORY_AND_16_CPU_TRANSPORTS_PASS",
            job=100,
            source=source,
            transport_result=public,
            GPU_execution=False,
            formal_GPU_admission=False,
            release={"TEST_ONLY": True},
        ),
    )
    qualifications, points = qualification_fixture(f, host, producer, records, proof(host))
    admission = dict(
        schema=factory.ADMISSION,
        status="FROZEN_PUBLIC_FACTORY_FORMAL",
        host=host,
        producer=producer,
        cpu_controller_manifest=cpu_manifest,
        cpu_source=cpu_source,
        factory_manifest=factory_manifest,
        factory_source=factory_ref,
        cache_hook=f.absolute_ref(cache),
        storage_binding=f.storage,
        actual_cpu={"job_id": 100, "directory": "cpu/100", "files": f.inventory("cpu/100")},
        qualifications=qualifications,
        qualification_points=f.ref("qualification-points.json", points),
    )
    f.put("launcher/admission.json", admission)
    f.manifest("launcher", ["launcher/admission.json"])
    f.document = dict(
        schema=factory.SCHEMA,
        adapter=factory.ADAPTER,
        original_task_root=str(f.task),
        anchors={"launcher_manifest": "launcher/manifest.sha256", "admission": "launcher/admission.json"},
        runs=[],
    )
    f.records, f.proof, f.children = records, proof, children
    return f


def qualification_fixture(f, host, producer, records, installed):
    legacy_fixture = legacy.sg_fixture()
    old_admission = json.loads(legacy_fixture.files["launch/admission.json"])
    points = json.loads(legacy_fixture.files["factory/qualification-points.json"])
    result = {}
    for deployment, phases in old_admission["qualifications"].items():
        result[deployment] = {}
        for phase, original_refs in phases.items():
            values = {k: json.loads(legacy_fixture.files[v["path"]]) for k, v in original_refs.items()}
            prefix = "qual/" + deployment + "/" + phase
            for cell in values["plan"]["cells"]:
                cell["weight_quantization"] = {"fp8": "fp8_block", "nvfp4": "nvfp4"}[deployment.split("-tp")[0]]
            values["aggregation"]["cell"]["weight_quantization"] = {"fp8": "fp8_block", "nvfp4": "nvfp4"}[
                deployment.split("-tp")[0]
            ]
            start = values["started"]
            start.update(source_commit=COMMIT, wheel_sha256=WHEEL, deployment=deployment, state="RUNNING")
            reader, aggregate = values["reader"], values["aggregation"]
            for entry in values["checkpoint"]["cells"].values():
                entry.update(requested_point_count=len(values["rows"]), measured_point_count=len(values["rows"]))
            refs = {k: f.ref(prefix + "/" + k + ".json", values[k]) for k in ("rows", "plan", "started", "checkpoint")}
            aggregate.update(
                reader_source_revision=COMMIT,
                installed_wheel_sha256=WHEEL,
                frozen_plan_file_sha256=refs["plan"]["sha256"],
                installed_source_hashes={name: records["collector/" + name] for name in factory.READER_SOURCES},
                reader_source=str(f.task / "host/collector/fpm_forward/native_artifact.py"),
                reader_source_sha256=records["collector/fpm_forward/native_artifact.py"],
                validation_performed_by="unmodified_database.aggregate_cell -> validate_native_collection",
                original_point_coordinate_coverage="passed",
                native_hardware_contract_validation="passed",
            )
            reader.update(
                status="passed",
                returncode=0,
                deployment=deployment,
                phase=phase,
                started_sha256=refs["started"]["sha256"],
                checkpoint_sha256=refs["checkpoint"]["sha256"],
                raw_directory=aggregate["raw_directory"],
                attempt_id=aggregate["attempt_id"],
            )
            source_path = prefix + "/reader-source/reader.py"
            f.put(source_path, b"# TEST_ONLY reader\n")
            refs["reader_manifest"] = {
                "path": prefix + "/reader-source/manifest.sha256",
                "sha256": f.manifest(prefix + "/reader-source", [source_path]),
            }
            reader["source_manifest_sha256"] = refs["reader_manifest"]["sha256"]
            refs["installed_reader"] = f.ref(prefix + "/installed.json", installed)
            refs["idle"] = None
            if phase == "prefill":
                workers = [
                    dict(pid=10 + i, rank=i, start_ticks=100, receipt_sha256="d" * 64)
                    for i in range(int(deployment[-1]))
                ]
                idle = {
                    "start": dict(
                        producer_commit=COMMIT,
                        idle_seconds=660,
                        post_idle_health_budget_seconds=900,
                        native_watchdog_timeout_changed=False,
                        direct_marker_observation=False,
                        workers=workers,
                    ),
                    "observed": dict(
                        producer_commit=COMMIT,
                        idle_elapsed_seconds=660.1,
                        post_idle_health_budget_seconds=900,
                        direct_marker_observation=False,
                        workers_before=workers,
                        workers_after=workers,
                        native_health_response={"TEST_ONLY": True},
                    ),
                    "return": dict(producer_commit=COMMIT, original_reader_calls=1),
                }
                refs["idle"] = {k: f.ref(prefix + "/idle-" + k + ".json", v) for k, v in idle.items()}
                reader["idle_proof"] = {Path(v["path"]).name: v["sha256"] for v in refs["idle"].values()}
            refs["aggregation"] = f.ref(prefix + "/aggregation.json", aggregate)
            refs["reader"] = f.ref(prefix + "/reader.json", reader)
            result[deployment][phase] = refs
    return result, points


def add_runs(f):
    admission = f.read("launcher/admission.json")
    for i, child in enumerate(f.children):
        root = f"formal/{child['deployment']}/{1000 + i}"
        raw = root + "/raw/node0000"
        start = dict(
            schema=factory.STARTED,
            state="RUNNING",
            job=str(1000 + i),
            deployment=child["deployment"],
            mode="formal",
            source_commit=COMMIT,
            wheel_sha256=WHEEL,
            host_source_commit=COMMIT,
            host_wheel_sha256=WHEEL,
            admission_sha256=SHA(f.files["launcher/admission.json"]),
            launcher_manifest_sha256=SHA(f.files["launcher/manifest.sha256"]),
            actual_cpu_job=100,
            selected=child,
            qualification=admission["qualifications"][child["deployment"]],
            requested_allocator_policy=factory.allocator_policy(child["deployment"]),
            plan_sha256=child["child_plan_sha256"],
        )
        f.put(root + "/started.json", start)
        for label in ("host", "producer"):
            f.put(root + "/actual-" + label + "-wheel.json", f.proof(admission[label]))
        f.put(
            raw + "/collector-provenance.json",
            dict(
                cell_id=child["child_cell_id"],
                plan_sha256=child["child_plan_sha256"],
                attempt_id=f"TEST_ONLY-{i}",
                runtime={"backend": "sglang", "backend_version": "0.5.20"},
            ),
        )
        f.document["runs"].append(
            dict(
                cell_id=child["child_cell_id"],
                started=root + "/started.json",
                raw_root=str(f.task / raw),
                collector_provenance=raw + "/collector-provenance.json",
                host_wheel_verification=root + "/actual-host-wheel.json",
                producer_wheel_verification=root + "/actual-producer-wheel.json",
            )
        )


def test_complete_factory_separate_native_attempts_and_portable_attachment(tmp_path):
    f = fixture()
    context = current.frozen_contract(f.document, f.files.__getitem__)
    assert len(context["children"]) == 72
    assert sum(len(c["original_point_ids"]) for c in context["children"].values()) == 2472
    add_runs(f)
    source = tmp_path / "source"
    for name, raw in f.files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    out = tmp_path / "portable"
    document = control.prepare(
        source, str(f.task), f.document["anchors"], f.document["runs"], out, adapter=factory.ADAPTER
    )
    if not isinstance(document, dict):
        document = json.loads((out / "external-controls.json").read_bytes())
    assert document["schema"] == factory.SCHEMA
    control.validate(out, document)
    assert "external_control_sglang_factory.py" in publication.POLICY_MODULES


@pytest.mark.parametrize(
    ("path", "change", "message"),
    [
        ("cpu/100/result.json", lambda v: v.update(state="RUNNING"), "incomplete or failed"),
        ("cpu/100/prepared/receipt.json", lambda v: v.update(host_arch="amd64"), "ARM source rendering"),
        (
            "cpu/100/prepared/point-crosswalk.json",
            lambda v: v[0].update(original_point_id=999),
            "complete point crosswalk",
        ),
        ("cpu/100/prepared/cpu-transport-selection.json", lambda v: v["fixtures"].pop(), "transport selection"),
        (
            "qual/fp8-tp2/prefill/aggregation.json",
            lambda v: v["installed_source_hashes"].pop("fpm_forward/planner.py"),
            "source map is incomplete",
        ),
        ("qual/fp8-tp2/decode/rows.json", lambda v: v.__setitem__(0, v[1]), "point union differs"),
        ("cpu/100/entry-transport.json", lambda v: v.update(status="FAILED"), "entry/process failed"),
        (
            "cpu/100/process-render/process-result.json",
            lambda v: v.update(cleanup_error="TEST_ONLY failure"),
            "entry/process failed",
        ),
        (
            "installs/producer.json",
            lambda v: v["files"].pop("collector/fpm_forward/sglang_driver.py"),
            "complete installed",
        ),
        ("cpu/100/identity/host-installed-wheel.json", lambda v: v.update(source="9" * 40), "complete installed"),
        (
            "cpu/100/entry-render.json",
            lambda v: v["loaded_project_origins"]["collector.fpm_forward.planner"].update(path="/OUTSIDE/planner.py"),
            "escapes task",
        ),
        ("cpu/100/prepared/receipt.json", lambda v: v["formal_children"].pop(), "missing or duplicated"),
        (
            "cpu/100/prepared/receipt.json",
            lambda v: v["formal_children"][0]["original_identity"]["point_map"][0]["point"].update(batch_size=99),
            "ownership differs",
        ),
        (
            "cpu/100/prepared/formal-inputs/fp8-tp2/calibration/collection-plan.json",
            lambda v: v["options"].update(shard_token_budget=99),
            "original options changed",
        ),
        ("cpu/100/transport-release.json", lambda v: v.update(prepared_inventory_sha256="0" * 64), "dynamic sixteen"),
        (
            "cpu/100/transport/fp8-tp2-calibration-prefill/raw/node0000/preparation-identity.json",
            lambda v: v.update(cuda_initialized=True),
            "preparation/execution identity",
        ),
        (
            "cpu/100/transport/fp8-tp2-calibration-prefill/raw/node0000/collector-provenance.json",
            lambda v: v.update(attempt_id="WRONG"),
            "collector attempt differs",
        ),
        (
            "qual/fp8-tp2/prefill/reader.json",
            lambda v: v.update(started_sha256="0" * 64),
            "qualification identity/result",
        ),
        (
            "qual/fp8-tp2/prefill/rows.json",
            lambda v: v[0].update(sglang_allocator_max_split_size_mb=None),
            "row provenance differs",
        ),
        (
            "qual/fp8-tp2/decode/rows.json",
            lambda v: v[0].update(sglang_allocator_policy_sha256="UNKNOWN"),
            "unknown actual",
        ),
        (
            "qual/fp8-tp2/decode/checkpoint.json",
            lambda v: next(iter(v["cells"].values())).update(status="running"),
            "attempt differs",
        ),
        (
            "qual/fp8-tp2/prefill/idle-observed.json",
            lambda v: v["workers_after"][0].update(start_ticks=123),
            "idle/health",
        ),
        ("qual/fp8-tp2/prefill/idle-observed.json", lambda v: v.update(direct_marker_observation=True), "idle/health"),
        ("qual/fp8-tp2/prefill/idle-return.json", lambda v: v.update(original_reader_calls=0), "idle/health"),
        ("launcher/admission.json", lambda v: v["qualifications"]["fp8-tp4"].pop("decode"), "qualification phases"),
        (
            "launcher/admission.json",
            lambda v: v.update(status="FROZEN_REVIEWED_CURRENT_HOST636_GPU_LAUNCHER"),
            "not frozen",
        ),
        ("launcher/admission.json", lambda v: v.update(unknown_contract_field=True), "admission fields"),
    ],
)
def test_consistently_hash_bound_rejections(path, change, message):
    f = fixture(lambda name, value: change(value) if name == path else None)
    with pytest.raises(ValueError, match=message):
        current.frozen_contract(f.document, f.files.__getitem__)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda v: v.pop("schema"), "started fields"),
        (lambda v: v.update(schema="historical636"), "ownership/qualification"),
        (lambda v: v["selected"].update(new_identity=v["selected"]["original_identity"]), "ownership/qualification"),
        (lambda v: v.update(host_source_commit="9" * 40), "source identity"),
        (lambda v: v.update(actual_cpu_job=101), "ownership/qualification"),
        (lambda v: v["qualification"].pop("decode"), "ownership/qualification"),
    ],
)
def test_formal_started_rejects_old_identity_or_missing_evidence(change, message):
    f = fixture()
    add_runs(f)
    path = f.document["runs"][0]["started"]
    value = f.read(path)
    change(value)
    f.put(path, value)
    with pytest.raises(ValueError, match=message):
        control.closure(f.document, f.files.__getitem__)


def test_explicit_storage_proof_preserves_original_strings(tmp_path):
    canonical = tmp_path / "physical"
    canonical.mkdir()
    lexical = tmp_path / "lexical"
    lexical.symlink_to(canonical, target_is_directory=True)
    storage = archive.create_storage_binding(lexical, canonical)
    f = fixture(task=canonical, original=lexical, storage=storage)
    context = current.frozen_contract(f.document, f.files.__getitem__)
    assert context["source"]["producer_target"].startswith(str(lexical))
    assert all(c["native_directory"].startswith(str(canonical)) for c in context["children"].values())
    f = fixture(task=canonical, original=lexical)
    with pytest.raises(ValueError, match="escapes task"):
        current.frozen_contract(f.document, f.files.__getitem__)


def test_no_schema_alias_or_partial_execution_admission():
    f = fixture()
    with pytest.raises(ValueError, match="seventy-two"):
        control.closure(f.document, f.files.__getitem__)
    f.document["adapter"] = current.SGLANG
    with pytest.raises(ValueError, match="unsupported factory"):
        current.frozen_contract(f.document, f.files.__getitem__)
    f.document.update(schema=current.SCHEMA, adapter=factory.ADAPTER)
    with pytest.raises(ValueError, match="unsupported current"):
        current.frozen_contract(f.document, f.files.__getitem__)


def test_prepare_live_storage_twice_then_offline_without_originals(tmp_path, monkeypatch):
    import shutil

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    lexical = tmp_path / "lexical"
    lexical.symlink_to(canonical, target_is_directory=True)
    storage = archive.create_storage_binding(lexical, canonical)
    f = fixture(task=canonical, original=lexical, storage=storage)
    add_runs(f)
    for name, raw in f.files.items():
        p = canonical / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    calls = []
    original = archive.validate_storage_binding

    def checked(binding, *, live=False):
        calls.append(live)
        return original(binding, live=live)

    monkeypatch.setattr(archive, "validate_storage_binding", checked)
    out = tmp_path / "portable"
    document = control.prepare(
        canonical, str(canonical), f.document["anchors"], f.document["runs"], out, adapter=factory.ADAPTER
    )
    assert calls.count(True) == 2
    lexical.unlink()
    shutil.rmtree(canonical)
    calls.clear()
    control.validate(out, document)
    assert calls and not any(calls)


@pytest.mark.parametrize("retarget_after", [False, True])
def test_prepare_rejects_missing_or_retargeted_actual_storage(tmp_path, monkeypatch, retarget_after):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    lexical = tmp_path / "lexical"
    lexical.symlink_to(canonical, target_is_directory=True)
    storage = archive.create_storage_binding(lexical, canonical)
    f = fixture(task=canonical, original=lexical, storage=storage)
    add_runs(f)
    for name, raw in f.files.items():
        p = canonical / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    if retarget_after:
        original = control.validate

        def validate(*args):
            result = original(*args)
            lexical.unlink()
            lexical.symlink_to(tmp_path, target_is_directory=True)
            return result

        monkeypatch.setattr(control, "validate", validate)
    else:
        lexical.unlink()
    with pytest.raises(ValueError, match="live original storage"):
        control.prepare(
            canonical,
            str(canonical),
            f.document["anchors"],
            f.document["runs"],
            tmp_path / "out",
            adapter=factory.ADAPTER,
        )


def test_archive_role_binding_and_explicit_revision_semantics(tmp_path):
    f = fixture()
    add_runs(f)
    source = tmp_path / "source"
    for name, raw in f.files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    out = tmp_path / "out"
    document = control.prepare(
        source, str(f.task), f.document["anchors"], f.document["runs"], out, adapter=factory.ADAPTER
    )
    index, get, admission = control.validate(out, document)
    revisions = current.configuration_revisions(
        document, get, admission, "sglang", "fp8", 2, "installed:aisimulate==TEST_ONLY"
    )
    assert revisions["planner_source_commit"] == revisions["native_producer_revision"] == COMMIT
    child, run = f.children[0], f.document["runs"][0]
    cid = child["child_cell_id"]
    plan_path = str((Path(child["native_directory"]).parent.parent / "plans" / (cid + ".json")).relative_to(f.task))
    spec = dict(cell_id=cid, raw_root=run["raw_root"], attempt_id="TEST_ONLY-0", plan={"path": plan_path})
    evidence = {
        "receipts": [{"path": "collector-provenance.json", "sha256": index[run["collector_provenance"]]["sha256"]}]
    }
    inventory = [dict(kind="file", path=p, sha256=item["sha256"]) for p, item in index.items()]
    control.bind_role(
        document, get, admission, [(spec, evidence)], {plan_path: f.read(plan_path)}, str(f.task), inventory, f.task
    )
    with pytest.raises(ValueError, match="archive changed"):
        control.bind_role(
            document, get, admission, [(spec, evidence)], {plan_path: f.read(plan_path)}, str(f.task), [], f.task
        )


def v4_history_fixture():
    from tools.glm53flash_hf import closed_history as history

    from .test_glm53flash_portable_history import PortableTests

    value = PortableTests("runTest")
    value.setUp()
    request = value.requests["sglang"]
    request["external_control_request"]["adapter"] = factory.ADAPTER
    for attempt in value.ledgers["sglang"]["attempts"]:
        cid = attempt["cell_id"]
        child = {"child_cell_id": cid, "child_plan_sha256": "b" * 64}
        selected = dict(kind="formal", **child, new_identity=child, original_identity={"child_cell_id": "OLD-" + cid})
        start = dict.fromkeys(factory.STARTED_FIELDS, "TEST_ONLY")
        start.update(schema=factory.STARTED, state="RUNNING", job=attempt["job"], selected=selected)
        final = dict(start, schema=factory.FINAL, state=attempt["terminal_state"], TEST_ONLY_final_extension=True)
        for label, data in (("started", start), ("final", final)):
            path = value.source / attempt[label]["path"]
            path.write_bytes(history.canonical(data))
            attempt[label] = dict(history.reference(path), path=attempt[label]["path"])
    failed = value.ledgers["sglang"]["attempts"][-1]
    request["history_floor"] = [{k: failed[k] for k in ("started", "final", "checkpoint")}]
    value.ledgers["sglang"]["request_sha256"] = history.digest(request)
    for label, data in (("request", request), ("ledger", value.ledgers["sglang"])):
        path = Path(value.inputs["sglang"][label]["path"])
        path.write_bytes(history.canonical(data))
        value.inputs["sglang"][label] = history.reference(path)
    return value


def test_v4_complete_failed_history_portable_after_deleted_original_and_tar():
    import shutil

    value = v4_history_fixture()
    try:
        value.prepare()  # Real tiny tar validation; only outer accepted-stage orchestration is TEST_ONLY mocked.
        shutil.rmtree(value.source)
        shutil.rmtree(value.root / "bound")
        for bundle in set(value.bundles.values()):
            shutil.rmtree(bundle)
        assert value.verify()["files"]
    finally:
        value.doCleanups()


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["selected"]["new_identity"].update(child_cell_id="OTHER_CHILD"),
        lambda v: v.update(source_commit="DIFFERENT_SOURCE"),
        lambda v: v.pop("schema"),
    ],
)
def test_v4_history_changed_original_final_identity_rejected(change):
    from tools.glm53flash_hf import closed_history as history

    value = v4_history_fixture()
    try:
        attempt = value.ledgers["sglang"]["attempts"][0]
        path = value.source / attempt["final"]["path"]
        data = json.loads(path.read_bytes())
        change(data)
        path.write_bytes(history.canonical(data))
        attempt["final"] = dict(history.reference(path), path=attempt["final"]["path"])
        ledger_path = Path(value.inputs["sglang"]["ledger"]["path"])
        ledger_path.write_bytes(history.canonical(value.ledgers["sglang"]))
        value.inputs["sglang"]["ledger"] = history.reference(ledger_path)
        with pytest.raises(ValueError, match="factory history"):
            history.snapshot("sglang", value.inputs["sglang"], value.plan, archive)
    finally:
        value.doCleanups()
