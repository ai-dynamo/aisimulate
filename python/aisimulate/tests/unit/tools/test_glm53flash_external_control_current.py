# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY complete formal identities; no measured or accepted GPU data."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from tools.glm53flash_hf import external_control as control
from tools.glm53flash_hf import external_control_current as current
from tools.glm53flash_hf import native_roots

pytestmark = pytest.mark.unit
TASK = Path("/TEST_ONLY/task")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def groups():
    result = {}
    counts = [7, 4, 4, 3]
    for deployment in sorted(current.DEPLOYMENTS):
        children = []
        for ((role, phase), count), pieces in zip(current.POINTS.items(), counts, strict=True):
            ids = list(range(1, count + 1))
            for i in range(pieces):
                chunk = ids[i * count // pieces : (i + 1) * count // pieces]
                cid = f"TEST_ONLY-{deployment}-{role}-{phase}-{i}"
                children.append(
                    dict(
                        child_cell_id=cid,
                        child_plan_sha256=sha(cid.encode()),
                        parent_plan_sha256=sha((deployment + role).encode()),
                        deployment=deployment,
                        role=role,
                        phase=phase,
                        original_point_ids=chunk,
                    )
                )
        result[deployment] = children
    return result


class Fixture:
    def __init__(self):
        self.files = {}

    def put(self, path, value):
        self.files[path] = value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode()
        return sha(self.files[path])

    def ref(self, path, value):
        return {"path": path, "sha256": self.put(path, value)}

    def manifest(self, root, paths):
        return self.put(
            root + "/manifest.sha256",
            "".join(sha(self.files[path]) + "  " + str(Path(path).relative_to(root)) + "\n" for path in paths).encode(),
        )


def sg_fixture(mutate=None):
    f = Fixture()
    groups_value = groups()
    host, producer = ("1" * 40, "2" * 64), ("3" * 40, "4" * 64)
    points = {
        phase: [
            dict(batch_size=1, total_kv_read_tokens=i, **({"total_prefill_tokens": 1} if phase == "prefill" else {}))
            for i in range(6 if phase == "prefill" else 3)
        ]
        for phase in ("prefill", "decode")
    }
    factory = "factory"
    hook = factory + "/cache-hook/sitecustomize.py"
    f.put(hook, b"# TEST_ONLY cache hook\n")
    f.put(factory + "/qualification-points.json", points)
    factory_sha = f.manifest(factory, list(f.files))
    source = {
        "host": dict(source_commit=host[0], wheel_sha256=host[1], wheel="host.whl"),
        "producer": dict(source_commit=producer[0], wheel_sha256=producer[1], wheel="producer.whl"),
        "bindings": {hook: sha(f.files[hook]), "host.whl": host[1]},
    }
    for item in ("host", "producer"):
        source[item].update(record_payloads=1, runtime_sha256="7" * 64)
    if mutate:
        mutate("source", source)
    f.put("controller/source.json", source)
    controller_sha = f.manifest("controller", ["controller/source.json"])
    prepared = "cpu/100/prepared"
    public_rows = []
    for deployment, children in groups_value.items():
        tp = int(deployment[-1])
        for role in ("calibration", "holdout"):
            root = f"{prepared}/formal-inputs/{deployment}/{role}"
            shards = []
            for child in [c for c in children if c["role"] == role]:
                cid = child["child_cell_id"]
                child["native_directory"] = str(TASK / root / "native" / cid)
                identity = {k: child[k] for k in ("child_cell_id", "child_plan_sha256", "parent_plan_sha256", "phase")}
                identity["point_map"] = [dict(original_point_id=i) for i in child["original_point_ids"]]
                shards.append(identity)
                plan = dict(
                    aic_revision="installed:aisimulate==0.13.0:record-sha256:" + "9" * 64,
                    sha256=child["child_plan_sha256"],
                    backend="sglang",
                    capability={"aic_database_version": "0.5.20"},
                    options=dict(
                        dataset_role=role,
                        sglang_mem_fraction_static=0.82,
                        sglang_allocator_max_split_size_mb=16384 if deployment == "fp8-tp2" else None,
                        slurm_container_mounts=[str(TASK / "factory/cache-hook") + ":/opt/glm53flash-cache:ro"],
                    ),
                    cells=[dict(cell_id=cid, workload_kind=child["phase"], topology={"tp": tp})],
                )
                f.put(root + "/plans/" + cid + ".json", plan)
                f.put(
                    root + "/native/" + cid + "/collector-runtime-env.sh",
                    b"export PYTHONPATH=/opt/glm53flash-cache:/opt/glm53flash-current:/opt/glm53flash-sg-deps\n",
                )
            if mutate:
                mutate("shards", shards)
            f.put(root + "/shard-manifest.json", {"shards": shards})
            public_rows.extend(
                dict(status="PASS", child=dict(kind="formal", deployment=deployment, role=role, phase=phase))
                for phase in ("prefill", "decode")
            )
    public_rows.extend(
        dict(status="PASS", child=dict(kind="qualification", deployment="fp8-tp2", role="calibration", phase=p))
        for p in ("prefill", "decode")
    )
    public = dict(status="18_PUBLIC_CPU_FIXTURES_PASS", rows=public_rows)
    if mutate:
        mutate("public", public)
    public_ref = f.ref("cpu/100/public-cpu/receipt.json", public)
    inventory = {str(Path(p).relative_to(prepared)): sha(b) for p, b in f.files.items() if p.startswith(prepared + "/")}
    inventory_ref = f.ref(prepared + "/inventory.json", inventory)
    result = dict(
        status="CURRENT_HOST636_PRODUCER_PUBLIC_CPU_FACTORY_PASS",
        source=source,
        job="100",
        source_only_formal_children=72,
        actual_cpu_fixtures=18,
        original636_producer_unchanged=True,
        public_cpu_receipt_sha256=public_ref["sha256"],
        runs=[{"returncode": 0}],
    )
    if mutate:
        mutate("cpu", result)
    result_ref = f.ref("cpu/100/result.json", result)
    qualifications = {}
    for deployment in current.DEPLOYMENTS:
        precision, tp = deployment.rsplit("-tp", 1)
        allocator = 16384 if deployment == "fp8-tp2" else None
        phases = {}
        for phase in ("prefill", "decode"):
            root = "qualification/" + deployment + "/" + phase
            cid, plan_sha = "TEST_ONLY_qual_" + phase, sha(root.encode())
            cell = dict(cell_id=cid, workload_kind=phase, topology={"tp": int(tp)})
            plan = dict(
                backend="sglang",
                sha256=plan_sha,
                cells=[cell],
                options=dict(
                    benchmark_points={"payload": points},
                    sglang_mem_fraction_static=0.82,
                    sglang_allocator_max_split_size_mb=allocator,
                ),
            )
            plan_ref = f.ref(root + "/plan.json", plan)
            start_ref = f.ref(
                root + "/started.json",
                dict(
                    source_commit=producer[0],
                    wheel_sha256=producer[1],
                    precision=precision,
                    tp=int(tp),
                    job="200",
                    plan_sha256=plan_sha,
                ),
            )
            entry = dict(status="passed", attempt_id="TEST_ONLY_attempt")
            checkpoint_ref = f.ref(root + "/checkpoint.json", dict(plan_sha256=plan_sha, cells={cid: entry}))
            rows = [
                dict(
                    p,
                    sglang_allocator_policy_sha256="5" * 64,
                    sglang_allocator_max_split_size_mb=allocator,
                    backend="sglang",
                    tp=int(tp),
                    cell_id=cid,
                    workload_kind=phase,
                    source_plan_sha256=plan_sha,
                    collector_attempt_id="TEST_ONLY_attempt",
                    runtime_run_id="TEST_ONLY_run",
                    runtime_grid_digest="6" * 64,
                    warmup_repeats=5,
                    measurement_repeats=10,
                )
                for p in points[phase]
            ]
            if mutate:
                mutate("qualification_rows", rows)
            rows_ref = f.ref(root + "/rows.json", rows)
            raw = str(TASK / root / "raw/node0")
            aggregate = dict(
                status="passed",
                validation="passed",
                aggregation="passed",
                reader_source_revision=producer[0],
                installed_wheel_sha256=producer[1],
                frozen_plan_file_sha256=plan_ref["sha256"],
                raw_directory=raw,
                cell=cell,
                row_count=len(rows),
                plan_sha256=plan_sha,
                attempt_id="TEST_ONLY_attempt",
                runtime_run_id="TEST_ONLY_run",
                runtime_grid_digest="6" * 64,
            )
            reader = dict(
                status="passed",
                returncode=0,
                phase=phase,
                source_job="200",
                original_started_sha256=start_ref["sha256"],
                checkpoint_snapshot_sha256=checkpoint_ref["sha256"],
                frozen_plan_file_sha256=plan_ref["sha256"],
                original_raw_directory=raw,
                plan_sha256=plan_sha,
                checkpoint_phase=entry,
            )
            if mutate:
                mutate("qualification", reader)
            phases[phase] = dict(
                reader=f.ref(root + "/reader.json", reader),
                aggregation=f.ref(root + "/aggregation.json", aggregate),
                rows=rows_ref,
                plan=plan_ref,
                started=start_ref,
                checkpoint=checkpoint_ref,
            )
        qualifications[deployment] = phases
    admission = dict(
        status="FROZEN_REVIEWED_CURRENT_HOST636_GPU_LAUNCHER",
        host_commit=host[0],
        host_wheel_sha256=host[1],
        producer_commit=producer[0],
        producer_wheel_sha256=producer[1],
        cpu_controller_directory="controller",
        cpu_controller_manifest_sha256=controller_sha,
        factory_directory=factory,
        factory_manifest_sha256=factory_sha,
        actual_cpu=dict(
            directory="cpu/100",
            job_id=100,
            result=result_ref,
            public_receipt=public_ref,
            prepared_inventory=inventory_ref,
        ),
        qualifications=qualifications,
    )
    if mutate:
        mutate("admission", admission)
    f.put("launch/admission.json", admission)
    f.manifest("launch", ["launch/admission.json"])
    f.document = dict(
        schema=current.SCHEMA,
        adapter=current.SGLANG,
        original_task_root=str(TASK),
        runs=[],
        anchors=dict(
            launcher_manifest="launch/manifest.sha256",
            admission="launch/admission.json",
            source_identity="controller/source.json",
            cache_hook=hook,
        ),
    )
    return f


def test_complete_split_host_frozen_identity_is_not_a_gpu_result():
    f = sg_fixture()
    result = current.frozen_contract(f.document, f.files.__getitem__)
    assert len(result["children"]) == 72
    assert result["host"] != result["producer"]
    assert f.document["runs"] == []
    with pytest.raises(ValueError, match="every original formal child"):
        current.closure(f.document, f.files.__getitem__)


@pytest.mark.parametrize(
    ("scope", "mutation", "message"),
    [
        ("admission", lambda x: x.update(status="DRAFT_NOT_FOR_EXECUTION"), "still a draft"),
        ("admission", lambda x: x.update(host_commit=x["producer_commit"]), "host/producer"),
        ("admission", lambda x: x.update(producer_wheel_sha256=x["host_wheel_sha256"]), "host/producer"),
        ("admission", lambda x: x.update(actual_cpu=None), "actual CPU job"),
        ("admission", lambda x: x["qualifications"].update({"fp8-tp2": None}), "qualification missing"),
        ("source", lambda x: x["bindings"].update({"host.whl": "f" * 64}), "wheel pin"),
        ("cpu", lambda x: x["runs"][0].update(returncode=1), "CPU execution"),
        ("cpu", lambda x: x.update(original636_producer_unchanged=False), "actual CPU gate"),
        ("public", lambda x: x["rows"].pop(), "fixture coverage"),
        ("qualification", lambda x: x.update(status="failed_preserved"), "qualification source/result"),
        ("qualification", lambda x: x.update(source_job="201"), "qualification source/result"),
        ("qualification_rows", lambda x: x[0].update(sglang_allocator_policy_sha256=None), "unknown actual allocator"),
        ("qualification_rows", lambda x: x[0].update(batch_size=False), "coordinate"),
        ("qualification_rows", lambda x: x[0].update(collector_attempt_id="TEST_ONLY_other"), "row provenance"),
        ("shards", lambda x: x[0]["point_map"].pop(), "point union"),
    ],
)
def test_consistently_hashed_semantic_mutations_reject(scope, mutation, message):
    def mutate(key, value):
        if key == scope:
            mutation(value)

    f = sg_fixture(mutate)
    with pytest.raises(ValueError, match=message):
        current.frozen_contract(f.document, f.files.__getitem__)


@pytest.mark.parametrize("mode", ["missing_child", "duplicate_child", "cross_deployment", "false_id", "wrong_phase"])
def test_nested_original_union_is_not_flattened_or_truncated(mode):
    value = groups()
    first = value["fp8-tp2"][0]
    if mode == "missing_child":
        value["fp8-tp2"].pop()
    elif mode == "duplicate_child":
        value["fp8-tp2"][1] = first
    elif mode == "cross_deployment":
        first["deployment"] = "fp8-tp4"
    elif mode == "false_id":
        first["original_point_ids"][0] = True
    else:
        first["phase"] = "decode"
    with pytest.raises(ValueError):
        current._children(value)


@pytest.mark.parametrize(
    "corruption", [None, "host_wheel", "producer_wheel", "host_source", "cpu_job", "selected_child", "allocator"]
)
@pytest.mark.parametrize("collection_scope", [False, True])
def test_actual_original_execution_bytes_are_portable_and_role_bound(tmp_path, corruption, collection_scope):
    f = sg_fixture()
    context = current.frozen_contract(f.document, f.files.__getitem__)
    for index, (cid, child) in enumerate(context["children"].items()):
        job = str(1000 + index)
        root = f"runs/{cid}/{job}"
        raw = (
            root + "/artifacts/" + child["child_plan_sha256"][:16] + "/cells/" + cid + "/raw/node0000"
            if collection_scope
            else root + "/raw/node0000"
        )
        selected = dict(kind="formal", role=child["role"], child_identity=child["original_identity"])
        start = dict(
            state="RUNNING",
            job=job,
            deployment=child["deployment"],
            mode="formal",
            source_commit=context["producer"][0],
            wheel_sha256=context["producer"][1],
            host_source_commit=context["host"][0],
            actual_cpu_job=100,
            admission_sha256=context["files"]["launch/admission.json"],
            launcher_manifest_sha256=context["files"]["launch/manifest.sha256"],
            selected=selected,
            plan_sha256=child["child_plan_sha256"],
            qualification=context["qualifications"][child["deployment"]],
            requested_allocator_policy=dict(
                schema="sglang_native_allocator_policy_v1",
                backend="native",
                max_split_size_mb=16384 if child["deployment"] == "fp8-tp2" else None,
            ),
        )
        f.put(root + "/started.json", start)
        f.put(
            root + "/actual-host-wheel.json",
            dict(
                status="ACTUAL_HOST_RECORD_GIT_COLLECTOR_RENDERER_AND_ARM_ELF_PASS",
                head=context["host"][0],
                wheel_sha256=context["host"][1],
                runtime_sha256="7" * 64,
                files={"TEST_ONLY.so": "8" * 64},
            ),
        )
        f.put(
            root + "/actual-producer-wheel.json",
            dict(
                state="EXACT_INSTALLED_WHEEL_RECORD_SOURCE_ELF_PASS",
                source=context["producer"][0],
                wheel_sha256=context["producer"][1],
                runtime_sha256="7" * 64,
                files={"TEST_ONLY.so": "8" * 64},
            ),
        )
        f.put(
            raw + "/collector-provenance.json",
            dict(
                cell_id=cid,
                plan_sha256=child["child_plan_sha256"],
                attempt_id="TEST_ONLY_attempt",
                runtime={"backend": "sglang", "backend_version": "0.5.20"},
            ),
        )
        f.document["runs"].append(
            dict(
                cell_id=cid,
                started=root + "/started.json",
                raw_root=str(TASK / raw),
                collector_provenance=raw + "/collector-provenance.json",
                host_wheel_verification=root + "/actual-host-wheel.json",
                producer_wheel_verification=root + "/actual-producer-wheel.json",
            )
        )
    if corruption:
        run = f.document["runs"][0]
        path = run["started"]
        if corruption in {"host_wheel", "producer_wheel"}:
            path = run[corruption.replace("wheel", "wheel_verification")]
        value = json.loads(f.files[path])
        if corruption in {"host_wheel", "producer_wheel"}:
            value["wheel_sha256"] = "0" * 64
        elif corruption == "host_source":
            value["host_source_commit"] = context["producer"][0]
        elif corruption == "cpu_job":
            value["actual_cpu_job"] = 101
        elif corruption == "selected_child":
            value["selected"]["child_identity"]["child_plan_sha256"] = "0" * 64
        else:
            value["requested_allocator_policy"]["max_split_size_mb"] = 42
        f.put(path, value)
    source = tmp_path / "TEST_ONLY_original"
    for name, data in f.files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    output = tmp_path / "TEST_ONLY_portable"
    if corruption:
        with pytest.raises(ValueError):
            control.prepare(
                source, str(TASK), f.document["anchors"], f.document["runs"], output, adapter=current.SGLANG
            )
        assert not output.exists()
        return
    document = control.prepare(
        source, str(TASK), f.document["anchors"], f.document["runs"], output, adapter=current.SGLANG
    )
    index, get, admission = control.validate(output, document)
    assert len(document["runs"]) == 72
    assert all(get(path) == f.files[path] for path in index)
    # The returned admission is the actual split document, never a forged legacy view.
    assert "host_and_producer_source" not in admission and "children" not in admission
    run = document["runs"][0]
    cid = run["cell_id"]
    child = context["children"][cid]
    plan_path = str((Path(child["native_directory"]).parent.parent / "plans" / (cid + ".json")).relative_to(TASK))
    plan = json.loads(f.files[plan_path])
    spec = dict(cell_id=cid, raw_root=run["raw_root"], attempt_id="TEST_ONLY_attempt", plan={"path": plan_path})
    evidence = {
        "receipts": [{"path": "collector-provenance.json", "sha256": index[run["collector_provenance"]]["sha256"]}]
    }
    if collection_scope:
        spec["raw_root"] = str(Path(run["raw_root"]).parent)
        spec[native_roots.FIELD] = native_roots.SCOPE
        evidence["receipts"][0]["path"] = "node0000/collector-provenance.json"
    inventory = [dict(kind="file", path=p, sha256=item["sha256"]) for p, item in index.items()]
    control.bind_role(document, get, admission, [(spec, evidence)], {plan_path: plan}, str(TASK), inventory, TASK)
    if collection_scope:
        for bad in (
            dict(spec, native_root_scope="unknown"),
            dict(spec, raw_root=spec["raw_root"].replace(cid, "foreign")),
            {k: v for k, v in spec.items() if k != native_roots.FIELD},
        ):
            with pytest.raises(ValueError):
                control.bind_role(
                    document, get, admission, [(bad, evidence)], {plan_path: plan}, str(TASK), inventory, TASK
                )
        for bad_path in ("collector-provenance.json", "node0001/collector-provenance.json"):
            bad_evidence = copy.deepcopy(evidence)
            bad_evidence["receipts"][0]["path"] = bad_path
            with pytest.raises(ValueError):
                control.bind_role(
                    document, get, admission, [(spec, bad_evidence)], {plan_path: plan}, str(TASK), inventory, TASK
                )
    altered = copy.deepcopy(plan)
    altered["options"]["dataset_role"] = "holdout"
    with pytest.raises(ValueError, match="original frozen child"):
        control.bind_role(
            document, get, admission, [(spec, evidence)], {plan_path: altered}, str(TASK), inventory, TASK
        )
    with pytest.raises(ValueError, match="archive changed"):
        control.bind_role(document, get, admission, [(spec, evidence)], {plan_path: plan}, str(TASK), [], TASK)
    _check_publication_revisions(output, context)


def _check_publication_revisions(output, context):
    """Use the complete 72-child attachment, not mocked producer dictionaries."""
    from tools.glm53flash_hf import glm53flash as policy

    revision = "installed:aisimulate==0.13.0:record-sha256:" + "9" * 64
    meta = dict(
        revision_identity_schema=policy.REVISION_SCHEMA,
        aic_revision=revision,
        planner_revision=revision,
        producer_revision=revision,
        producer_revision_semantics=policy.PLANNER_ALIAS,
    )
    part = dict(backend="sglang", weight_quantization="fp8", tp=2)
    ref = dict(path="external-control.json", sha256=sha((output / "external-control.json").read_bytes()))
    records = [dict(part, phase=p, role=r, external_control=ref) for r, p in current.POINTS]
    report = {
        "consumer": dict(
            distribution="aisimulate",
            version="0.13.0",
            api="RustForwardPassPerfModel.best_available",
            payload_sha256="a" * 64,
        )
    }
    result = policy.publication_revisions(meta, report, records, output, "b" * 40, part)
    assert result["planner_revision"] == result["producer_revision"] == revision
    assert result["planner_source_commit"] == context["host"][0]
    assert result["native_producer_revision"] == context["producer"][0]
    assert result["native_producer_revision"] != result["planner_source_commit"]
    assert result["native_producer_wheel_sha256"] == context["producer"][1]
    assert result["analysis_revision"] == dict(
        installed_consumer=report["consumer"], publication_tool_revision="b" * 40
    )
    assert result["external_control_sha256"] == [ref["sha256"]]
    for field in ("native_producer_revision", "analysis_revision"):
        assert field not in meta  # Native/analysis evidence never rewrites source metadata.
    for mutation, message in (
        (lambda m, r, rows: m.update(planner_revision=context["producer"][0]), "planner revision"),
        (
            lambda m, r, rows: m.update(aic_revision="wrong", planner_revision="wrong", producer_revision="wrong"),
            "original plan renderer",
        ),
        (lambda m, r, rows: rows.pop(), "four original"),
        (lambda m, r, rows: rows[0].pop("external_control"), "external controls"),
        (lambda m, r, rows: r["consumer"].pop("version"), "analysis installed"),
        (lambda m, r, rows: r["consumer"].update(payload_sha256="unknown"), "analysis installed"),
        (
            lambda m, r, rows: [
                m.pop(k) for k in ("revision_identity_schema", "planner_revision", "producer_revision_semantics")
            ],
            "explicit revision",
        ),
    ):
        changed_meta, changed_report, changed_records = copy.deepcopy((meta, report, records))
        mutation(changed_meta, changed_report, changed_records)
        with pytest.raises(ValueError, match=message):
            policy.publication_revisions(changed_meta, changed_report, changed_records, output, "b" * 40, part)


def test_truncated_frozen_inventory_is_not_filled_from_disk():
    f = sg_fixture()
    path = next(p for p in f.files if p.endswith("collector-runtime-env.sh"))
    del f.files[path]
    with pytest.raises(KeyError):
        current.frozen_contract(f.document, f.files.__getitem__)


def vllm_fixture(mutate=None):
    """Current nested manifest shape, generated entirely as TEST_ONLY data."""
    base = sg_fixture()
    context = current.frozen_contract(base.document, base.files.__getitem__)
    f = Fixture()
    groups_value = copy.deepcopy(context["groups"])
    for path, data in base.files.items():
        if path.endswith("/shard-manifest.json"):
            f.put(path, data)
    version = "0.30.0+TEST_ONLY_tail"
    identity = "a" * 40, "b" * 64
    cpu_root = "vllm-cpu/300"
    hook, user = "vllm-cache/cache_hook.py", "vllm-cache/usercustomize.py"
    hook_sha, user_sha = f.put(hook, b"# TEST_ONLY hook\n"), f.put(user, b"# TEST_ONLY usercustomize\n")
    source = dict(source_commit=identity[0], wheel_sha256=identity[1], runtime_sha256="c" * 64)
    if mutate:
        mutate("source", source)
    f.put("vllm-source.json", source)
    worker_sha = "d" * 64
    expected = dict(
        versions={"candidate": version},
        source_pins={current.vllm.WORKER_SOURCE: worker_sha},
        native_binaries={"vllm/TEST_ONLY.so": "e" * 64},
    )
    f.put("vllm-expected.json", expected)
    closure = dict(contract_sha256="f" * 64, observed_files={**expected["source_pins"], **expected["native_binaries"]})
    public_rows = []
    for deployment, children in groups_value.items():
        for child in children:
            native = str(Path(child["native_directory"]).relative_to(TASK))
            plan_path = str(Path(native).parent.parent / "plans" / (child["child_cell_id"] + ".json"))
            plan = json.loads(base.files[plan_path])
            plan["backend"] = "vllm"
            plan["capability"]["aic_database_version"] = version
            plan["options"]["slurm_container_mounts"] = [str(TASK / "vllm-cache") + ":/opt/glm53flash-cache:ro"]
            f.put(plan_path, plan)
            f.put(
                native + "/collector-runtime-env.sh",
                (
                    "export PYTHONPATH=" + ":".join(current.vllm.PYTHONPATH) + "\nexport DYN_FPM_GLM53FLASH_REAL_KV=1\n"
                ).encode(),
            )
        for role, phase in current.POINTS:
            child = next(c for c in children if (c["role"], c["phase"]) == (role, phase))
            root = f"{cpu_root}/public-cpu/{deployment}-{role}-{phase}"
            observer_sha = f.put(root + "/slurm-runtime/sitecustomize.py", b"# TEST_ONLY packaged observer\n")
            values = []
            for offset, name in enumerate(("preparation", "execution")):
                pid = 1000 + 2 * len(public_rows) + offset
                cache_root = f"/tmp/glm53-fpm-0-300/{pid}"
                cache = dict(
                    pid=pid,
                    uid=0,
                    job_id="300",
                    root=cache_root,
                    cubin_owned_write_read_delete="passed",
                    cache_variables={
                        "FLASHINFER_WORKSPACE_BASE": cache_root + "/flashinfer",
                        "FLASHINFER_CUBIN_DIR": cache_root + "/flashinfer-cubin",
                        "TRITON_CACHE_DIR": cache_root + "/triton",
                    },
                )
                order = dict(
                    pid=pid,
                    before_framework_modules=[],
                    cache_hook_sha256=hook_sha,
                    usercustomize_sha256=user_sha,
                    sitecustomize_file="/tmp/fpm-bench/sitecustomize.py",
                    usercustomize_file="/opt/glm53flash-cache/usercustomize.py",
                )
                value = dict(
                    status="passed",
                    cuda_initialized=False,
                    vllm_import_version=version,
                    vllm_metadata_version=version,
                    vllm_import_file="/opt/glm53flash-candidate/vllm/__init__.py",
                    collector_import_file="/opt/glm53flash-current/collector/__init__.py",
                    startup_evidence={"pid": pid},
                    source_pins=expected["source_pins"],
                    native_binaries=expected["native_binaries"],
                    actual_producer_binding=dict(
                        scheduler="glm53flash_scheduler.Glm53FlashRealKVScheduler",
                        runtime_closure=closure,
                        native_worker_source_sha256=worker_sha,
                    ),
                    bootstrap_identity=dict(
                        sitecustomize_sha256=observer_sha, packaged_sitecustomize_sha256=observer_sha
                    ),
                )
                if mutate:
                    mutate("cpu_identity", value)
                raw = root + "/raw/node0000"
                f.put(raw + f"/cache-setup-{pid}.json", cache)
                f.put(raw + f"/cache-startup-order-{pid}.json", order)
                f.put(raw + f"/{name}-identity.json", value)
                values.append(value)
            public_rows.append(dict(status="passed", child=child, actual_import_identity=values[0]))
    public = dict(state="SIXTEEN_ROLE_PHASE_PUBLIC_CPU_ENVIRONMENTS_PASS", rows=public_rows)
    result = dict(status="passed", source=source, runs=[{"returncode": 0}])
    if mutate:
        mutate("cpu", result)
        mutate("public", public)
    f.put(cpu_root + "/result.json", result)
    f.put(cpu_root + "/public-environment-receipt.json", public)
    for label in ("host", "producer"):
        f.put(
            cpu_root + f"/{label}-installed-wheel.json",
            dict(
                state="EXACT_INSTALLED_WHEEL_RECORD_SOURCE_ELF_PASS",
                source=identity[0],
                wheel_sha256=identity[1],
                files={"TEST_ONLY.py": "c" * 64},
                runtime_sha256=source["runtime_sha256"],
            ),
        )
    deployments = {}
    for deployment, children in groups_value.items():
        phases = []
        for phase, count in (("prefill", 6), ("decode", 3)):
            reader = dict(status="passed", phase=phase, source_job="400", row_count=count, rows_sha256="d" * 64)
            if mutate:
                mutate("qualification", reader)
            digest = f.put(f"vllm-qual/{deployment}/{phase}/result.json", reader)
            phases.append(
                dict(phase=phase, row_count=count, aggregated_rows_sha256="d" * 64, reader_result_sha256=digest)
            )
        deployments[deployment] = dict(
            qualification_status="ORIGINAL_NINE_POINTS_STRICT_PASS",
            children=children,
            qualification=dict(
                deployment=deployment,
                original_collection_status="COLLECTION_PASSED",
                original_gpu_job=400,
                phase_readers=phases,
            ),
        )
    admission = dict(
        status="FROZEN_CURRENT_TAIL_FORMAL_CHILDREN_WITH_CURRENT_QUALIFICATION",
        source_commit=identity[0],
        wheel_sha256=identity[1],
        actual_formal_cpu_job=300,
        deployments=deployments,
        bindings={p: sha(data) for p, data in f.files.items()},
    )
    if mutate:
        mutate("admission", admission)
    f.put("launch/admission.json", admission)
    f.manifest("launch", ["launch/admission.json"])
    f.document = dict(
        schema=current.SCHEMA,
        adapter=current.VLLM,
        original_task_root=str(TASK),
        runs=[],
        anchors=dict(
            launcher_manifest="launch/manifest.sha256",
            admission="launch/admission.json",
            source_identity="vllm-source.json",
            cpu_result=cpu_root + "/result.json",
            cpu_public=cpu_root + "/public-environment-receipt.json",
            cache_hook=hook,
            usercustomize=user,
            runtime_expected="vllm-expected.json",
        ),
    )
    return f


def test_nested_vllm_frozen_source_and_cpu_contract():
    f = vllm_fixture()
    result = current.frozen_contract(f.document, f.files.__getitem__)
    assert len(result["children"]) == 72
    assert result["host"] == result["producer"]
    assert result["admission"]["deployments"]["fp8-tp2"]["children"][0]["original_point_ids"][0] == 1
    identities = current.configuration_revisions(
        f.document,
        f.files.__getitem__,
        result["admission"],
        "vllm",
        "fp8",
        2,
        "installed:aisimulate==0.13.0:record-sha256:" + "9" * 64,
    )
    assert identities["planner_source_commit"] == identities["native_producer_revision"] == result["producer"][0]
    assert identities["planner_wheel_sha256"] == identities["native_producer_wheel_sha256"] == result["producer"][1]
    with pytest.raises(ValueError, match="every original formal child"):
        current.closure(f.document, f.files.__getitem__)


@pytest.mark.parametrize(
    ("scope", "mutation", "message"),
    [
        ("admission", lambda x: x["deployments"]["fp8-tp2"]["children"].pop(), "eighteen"),
        (
            "admission",
            lambda x: x["deployments"]["fp8-tp2"].update(qualification_status="UNKNOWN"),
            "qualification missing",
        ),
        ("admission", lambda x: x.update(source_commit="9" * 40), "source/wheel"),
        ("admission", lambda x: x.update(actual_formal_cpu_job=301), "CPU job/root"),
        (
            "admission",
            lambda x: x["bindings"].pop(next(p for p in x["bindings"] if p.endswith("collector-runtime-env.sh"))),
            "environment missing",
        ),
        ("public", lambda x: x["rows"].pop(), "fixture coverage"),
        ("cpu", lambda x: x.update(status="failed_preserved"), "CPU source/wheel"),
        ("cpu_identity", lambda x: x.update(cuda_initialized=True), "CPU import identity"),
        ("qualification", lambda x: x.update(row_count=0), "strict result differs"),
    ],
)
def test_nested_vllm_consistent_hash_negative_contract(scope, mutation, message):
    f = vllm_fixture(lambda key, value: mutation(value) if key == scope else None)
    with pytest.raises(ValueError, match=message):
        current.frozen_contract(f.document, f.files.__getitem__)
