# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable original-byte bindings for nested and split-host formal launches.

This is an analysis adapter, not a native runtime qualification. The original
admission, plans, CPU receipts and started files are never rewritten. Native
measurement acceptance remains the independent publication stage's job.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

if __package__:
    from . import external_control_vllm as vllm
    from . import raw_archive as archive
else:
    import external_control_vllm as vllm
    import raw_archive as archive

SCHEMA = "glm53flash_external_control_v2"
MIXED_SCHEMA = "glm53flash_external_control_v3"
SCHEMAS = {SCHEMA, MIXED_SCHEMA}
MIXED_SGLANG = "sglang_split_host_formal_mixed_v3"
VLLM = "vllm_nested_formal_v2"
SGLANG = "sglang_split_host_formal_v2"
ADAPTERS = {VLLM, SGLANG, MIXED_SGLANG}
DEPLOYMENTS = {f"{precision}-tp{tp}" for precision in ("fp8", "nvfp4") for tp in (2, 4)}
POINTS = {
    ("calibration", "prefill"): 250,
    ("calibration", "decode"): 147,
    ("holdout", "prefill"): 144,
    ("holdout", "decode"): 77,
}
require = archive.require


def _control():
    # The dispatcher imports this module; defer the reverse reference.
    if __package__:
        from . import external_control
    else:
        import external_control
    return external_control


class Frozen:
    """Only explicitly hash-bound original files enter the frozen closure."""

    def __init__(self, get):
        self.get = get
        self.files = {}

    def bind(self, path, digest):
        control = _control()
        control.relative(path)
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid original file digest")
        require(path not in self.files or self.files[path] == digest, "conflicting original file identity")
        require(control.sha(self.get(path)) == digest, "original frozen file SHA256 mismatch: " + path)
        self.files[path] = digest
        return path

    def ref(self, value):
        require(isinstance(value, dict), "missing original file reference")
        return self.bind(value["path"], value["sha256"])

    def read(self, path):
        require(path in self.files, "unbound original file: " + path)
        return json.loads(self.get(path))

    def mapping(self, values, parent=""):
        require(isinstance(values, dict) and values, "missing original file inventory")
        for path, digest in values.items():
            _control().relative(path)
            self.bind(str(Path(parent) / path), digest)

    def manifest(self, path, digest=None):
        if digest is None:
            digest = _control().sha(self.get(path))
        self.bind(path, digest)
        members = _control().sums(self.get(path), Path(path).parent)
        self.mapping(members)
        return members


def _identity(commit, wheel):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "invalid original source commit")
    require(isinstance(wheel, str) and re.fullmatch(r"[0-9a-f]{64}", wheel), "invalid original wheel SHA256")
    return commit, wheel


def _cpu_pass(result):
    require(
        result.get("runs") and all(row.get("returncode") == 0 for row in result["runs"]),
        "original CPU execution did not pass",
    )


def _children(groups, *, expected_deployments=None):
    """Preserve phase-local IDs: prefill 1 is distinct from decode 1."""
    require(
        set(groups) == (DEPLOYMENTS if expected_deployments is None else set(expected_deployments)),
        "formal deployment set differs",
    )
    indexed = {}
    for deployment, children in groups.items():
        require(len(children) == 18, "formal deployment must contain eighteen original children")
        seen = {key: set() for key in POINTS}
        for child in children:
            cid = child["child_cell_id"]
            require(cid not in indexed and child["deployment"] == deployment, "duplicate or cross-deployment child")
            key = child["role"], child["phase"]
            require(key in POINTS, "invalid original child role/phase")
            ids = child["original_point_ids"]
            require(ids and all(type(i) is int and 1 <= i <= POINTS[key] for i in ids), "invalid original point IDs")
            require(len(set(ids)) == len(ids) and not seen[key].intersection(ids), "duplicate original phase point")
            seen[key].update(ids)
            indexed[cid] = child
        require(
            all(seen[key] == set(range(1, count + 1)) for key, count in POINTS.items()),
            "formal original phase-point union is incomplete",
        )
    return indexed


def _public_rows(public, backend):
    expected = {(d, role, phase) for d in DEPLOYMENTS for role, phase in POINTS}
    if backend == "sglang":
        expected.update({("fp8-tp2", "calibration", phase) for phase in ("prefill", "decode")})
        # Qualification shares coordinates with formal fixtures, but is a distinct kind.
        expected = {("formal", *key) for key in expected}
        expected.update({("qualification", "fp8-tp2", "calibration", phase) for phase in ("prefill", "decode")})
    rows = public.get("rows", [])
    actual = []
    for row in rows:
        require(row.get("status") == ("PASS" if backend == "sglang" else "passed"), "CPU role fixture failed")
        child = row["child"]
        key = child["deployment"], child["role"], child["phase"]
        actual.append((child["kind"], *key) if backend == "sglang" else key)
    require(len(actual) == len(set(actual)) and set(actual) == expected, "CPU role/phase fixture coverage differs")
    return rows


def _native_plans(context, frozen, task, anchors):
    """Bind each original child to the literal framework startup route."""
    for child in context["children"].values():
        directory = _control().absolute(child["native_directory"])
        require(directory.is_relative_to(task), "native preparation is outside original task")
        prefix = str(directory.relative_to(task))
        partition = frozen.read(str(directory.parent.parent.relative_to(task) / "shard-manifest.json"))
        shards = [item for item in partition["shards"] if item["child_cell_id"] == child["child_cell_id"]]
        require(
            len(shards) == 1
            and shards[0]["child_plan_sha256"] == child["child_plan_sha256"]
            and shards[0]["parent_plan_sha256"] == child["parent_plan_sha256"]
            and shards[0]["phase"] == child["phase"]
            and [p["original_point_id"] for p in shards[0]["point_map"]] == child["original_point_ids"],
            "nested child differs from original phase partition",
        )
        plan = frozen.read(
            str(directory.parent.parent.relative_to(task) / "plans" / (child["child_cell_id"] + ".json"))
        )
        require(
            plan.get("sha256") == child["child_plan_sha256"]
            and plan.get("backend") == context["backend"]
            and plan.get("capability", {}).get("aic_database_version") == context["version"]
            and plan["options"].get("dataset_role", "calibration") == child["role"],
            "original child plan identity differs",
        )
        require(
            len(plan["cells"]) == 1
            and plan["cells"][0]["cell_id"] == child["child_cell_id"]
            and plan["cells"][0]["workload_kind"] == child["phase"],
            "original child phase differs",
        )
        env_path = prefix + "/collector-runtime-env.sh"
        require(env_path in frozen.files, "original runtime environment missing")
        env = {}
        for line in frozen.get(env_path).decode().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            require(
                line.startswith("export ") and not any(c in line for c in "$`\\"),
                "nonliteral native startup environment",
            )
            tokens = shlex.split(line)
            require(len(tokens) == 2 and "=" in tokens[1], "invalid native startup export")
            key, value = tokens[1].split("=", 1)
            require(key not in env, "duplicate native startup export")
            env[key] = value
        expected_path = (
            ":".join(vllm.PYTHONPATH)
            if context["backend"] == "vllm"
            else ("/opt/glm53flash-cache:/opt/glm53flash-current:/opt/glm53flash-sg-deps")
        )
        require(
            env.get("PYTHONPATH") == expected_path
            and env.get("AISIM_GLM53_PURPOSE", "fpm") == "fpm"
            and not env.get("PYTHONNOUSERSITE")
            and not any("OPS" in key and value for key, value in env.items()),
            "native cache order/purpose differs",
        )
        hook = str(task / Path(anchors["cache_hook"]).parent) + ":/opt/glm53flash-cache:ro"
        require(
            plan["options"].get("slurm_container_mounts", []).count(hook) == 1, "original read-only cache mount differs"
        )
        if context["backend"] == "vllm":
            require(env.get("DYN_FPM_GLM53FLASH_REAL_KV") == "1", "native real-state observer disabled")
        else:
            allocator = 16384 if child["deployment"] == "fp8-tp2" else None
            require(
                plan["options"].get("sglang_mem_fraction_static") == 0.82
                and plan["options"].get("sglang_allocator_max_split_size_mb") == allocator,
                "original SGLang memory/allocator policy differs",
            )


def _vllm_frozen(document, frozen, admission, task):
    anchors = document["anchors"]
    require(
        set(anchors)
        == {
            "launcher_manifest",
            "admission",
            "source_identity",
            "cpu_result",
            "cpu_public",
            "cache_hook",
            "usercustomize",
            "runtime_expected",
        },
        "current vLLM anchors differ",
    )
    require(
        admission.get("status") == "FROZEN_CURRENT_TAIL_FORMAL_CHILDREN_WITH_CURRENT_QUALIFICATION",
        "current vLLM launch was not qualified",
    )
    frozen.mapping(admission["bindings"])
    source, result, public = (frozen.read(anchors[k]) for k in ("source_identity", "cpu_result", "cpu_public"))
    identity = _identity(admission["source_commit"], admission["wheel_sha256"])
    require(
        _identity(source["source_commit"], source["wheel_sha256"]) == identity
        and result.get("source") == source
        and result.get("status") == "passed",
        "actual CPU source/wheel mismatch",
    )
    _cpu_pass(result)
    cpu_root = Path(anchors["cpu_result"]).parent
    require(
        cpu_root.name == str(admission["actual_formal_cpu_job"])
        and anchors["cpu_public"] == str(cpu_root / "public-environment-receipt.json"),
        "actual CPU job/root mismatch",
    )
    require(public.get("state") == "SIXTEEN_ROLE_PHASE_PUBLIC_CPU_ENVIRONMENTS_PASS", "current public CPU gate missing")
    rows = _public_rows(public, "vllm")
    for label in ("host", "producer"):
        wheel = frozen.read(str(cpu_root / (label + "-installed-wheel.json")))
        require(
            wheel.get("state") == "EXACT_INSTALLED_WHEEL_RECORD_SOURCE_ELF_PASS"
            and (wheel.get("source"), wheel.get("wheel_sha256")) == identity
            and wheel.get("files")
            and wheel.get("runtime_sha256") == source["runtime_sha256"],
            "actual CPU installed wheel differs",
        )
    runtime = frozen.read(anchors["runtime_expected"])
    version = runtime["versions"]["candidate"]
    expected_runtime = {**runtime["source_pins"], **runtime["native_binaries"]}
    groups = {}
    for deployment, value in admission["deployments"].items():
        require(
            value.get("qualification_status") == "ORIGINAL_NINE_POINTS_STRICT_PASS", "deployment qualification missing"
        )
        qualification = value["qualification"]
        require(
            qualification.get("deployment") == deployment
            and qualification.get("original_collection_status") == "COLLECTION_PASSED",
            "qualification identity differs",
        )
        phases = qualification.get("phase_readers", [])
        require(
            len(phases) == 2 and {r["phase"] for r in phases} == {"prefill", "decode"}, "qualification phases missing"
        )
        for phase in phases:
            path = next(
                (
                    p
                    for p, digest in frozen.files.items()
                    if digest == phase["reader_result_sha256"] and p.endswith("/result.json")
                ),
                None,
            )
            require(path is not None, "qualification strict receipt is not frozen")
            reader = frozen.read(path)
            require(
                reader.get("status") == "passed"
                and reader.get("phase") == phase["phase"]
                and str(reader.get("source_job")) == str(qualification["original_gpu_job"])
                and reader.get("row_count") == phase["row_count"]
                and reader.get("rows_sha256") == phase["aggregated_rows_sha256"],
                "qualification strict result differs",
            )
            require(
                phase["row_count"] == (6 if phase["phase"] == "prefill" else 3), "qualification point count differs"
            )
        groups[deployment] = value["children"]
    children = _children(groups)
    closures = []
    for row in rows:
        child = row["child"]
        require(children.get(child["child_cell_id"]) == child, "CPU fixture selects a different child")
        root = cpu_root / "public-cpu" / f"{child['deployment']}-{child['role']}-{child['phase']}"
        raw = root / "raw/node0000"
        ids = [frozen.read(str(raw / (name + "-identity.json"))) for name in ("preparation", "execution")]
        require(row.get("actual_import_identity") == ids[0], "public CPU import identity differs")
        pids = []
        for value in ids:
            require(
                value.get("status") == "passed"
                and value.get("cuda_initialized") is False
                and value.get("vllm_import_version") == value.get("vllm_metadata_version") == version
                and value.get("vllm_import_file") == "/opt/glm53flash-candidate/vllm/__init__.py"
                and value.get("collector_import_file") == "/opt/glm53flash-current/collector/__init__.py",
                "native CPU import identity mismatch",
            )
            pid = value["startup_evidence"]["pid"]
            cache = frozen.read(str(raw / f"cache-setup-{pid}.json"))
            order = frozen.read(str(raw / f"cache-startup-order-{pid}.json"))
            vllm.cache_identity(
                cache,
                order,
                hook_sha=frozen.files[anchors["cache_hook"]],
                user_sha=frozen.files[anchors["usercustomize"]],
                job=cpu_root.name,
                pid=pid,
            )
            binding = value["actual_producer_binding"]
            closure = binding["runtime_closure"]
            require(
                binding.get("scheduler") == "glm53flash_scheduler.Glm53FlashRealKVScheduler"
                and all(closure["observed_files"].get(k) == v for k, v in expected_runtime.items())
                and value["source_pins"] == runtime["source_pins"]
                and value["native_binaries"] == runtime["native_binaries"],
                "actual CPU runtime closure differs",
            )
            require(
                re.fullmatch(r"[0-9a-f]{64}", closure.get("contract_sha256", ""))
                and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in closure["observed_files"].values())
                and binding["native_worker_source_sha256"] == closure["observed_files"][vllm.WORKER_SOURCE],
                "invalid native runtime/worker source closure",
            )
            observer = str(root / "slurm-runtime/sitecustomize.py")
            require(
                value["bootstrap_identity"]["sitecustomize_sha256"]
                == frozen.files[observer]
                == value["bootstrap_identity"]["packaged_sitecustomize_sha256"],
                "actual CPU observer differs",
            )
            closures.append(closure)
            pids.append(pid)
        require(pids[0] != pids[1], "CPU prepare/execute reused a process")
    require(all(item == closures[0] for item in closures), "CPU native runtime closures differ")
    return dict(
        backend="vllm",
        version=version,
        children=children,
        groups=groups,
        host=identity,
        producer=identity,
        runtime_closure=closures[0],
        cpu_root=str(cpu_root),
    )


def _sglang_frozen(document, frozen, admission, task, *, deployments=None):
    anchors = document["anchors"]
    require(
        set(anchors) == {"launcher_manifest", "admission", "source_identity", "cache_hook"},
        "current SGLang anchors differ",
    )
    require(admission.get("status") == "FROZEN_REVIEWED_CURRENT_HOST636_GPU_LAUNCHER", "SGLang launch is still a draft")
    host = _identity(admission["host_commit"], admission["host_wheel_sha256"])
    producer = _identity(admission["producer_commit"], admission["producer_wheel_sha256"])
    controller = Path(admission["cpu_controller_directory"])
    factory = Path(admission["factory_directory"])
    frozen.manifest(str(controller / "manifest.sha256"), admission["cpu_controller_manifest_sha256"])
    frozen.manifest(str(factory / "manifest.sha256"), admission["factory_manifest_sha256"])
    require(
        anchors["source_identity"] == str(controller / "source.json")
        and anchors["cache_hook"] == str(factory / "cache-hook/sitecustomize.py"),
        "SGLang source/cache path differs",
    )
    source = frozen.read(anchors["source_identity"])
    require(
        _identity(source["host"]["source_commit"], source["host"]["wheel_sha256"]) == host
        and _identity(source["producer"]["source_commit"], source["producer"]["wheel_sha256"]) == producer,
        "SGLang host/producer source identities differ",
    )
    # Wheel blobs remain external immutable artifacts. Their original pins and
    # installed RECORD receipts are preserved; this small attachment is not a
    # second wheel archive or an independent installation verification.
    wheel_pins = {source[k]["wheel"]: source[k]["wheel_sha256"] for k in ("host", "producer")}
    require(isinstance(source.get("bindings"), dict) and source["bindings"], "SGLang source bindings missing")
    for path, digest in source["bindings"].items():
        if path.endswith(".whl"):
            require(wheel_pins.get(path) == digest, "unrecognized external wheel pin")
        else:
            frozen.bind(path, digest)
    cpu = admission["actual_cpu"]
    require(isinstance(cpu, dict) and type(cpu.get("job_id")) is int and cpu["job_id"] > 0, "actual CPU job missing")
    cpu_root = Path(cpu["directory"])
    require(cpu_root.name == str(cpu["job_id"]), "actual CPU job/root differs")
    for key, suffix in (
        ("result", "result.json"),
        ("public_receipt", "public-cpu/receipt.json"),
        ("prepared_inventory", "prepared/inventory.json"),
    ):
        require(cpu[key]["path"] == str(cpu_root / suffix), "CPU reference is outside original job")
        frozen.ref(cpu[key])
    result = frozen.read(cpu["result"]["path"])
    public = frozen.read(cpu["public_receipt"]["path"])
    require(
        result.get("status") == "CURRENT_HOST636_PRODUCER_PUBLIC_CPU_FACTORY_PASS"
        and result.get("source") == source
        and str(result.get("job")) == str(cpu["job_id"])
        and result.get("source_only_formal_children") == 72
        and result.get("actual_cpu_fixtures") == 18
        and result.get("original636_producer_unchanged") is True
        and result.get("public_cpu_receipt_sha256") == cpu["public_receipt"]["sha256"],
        "SGLang actual CPU gate differs",
    )
    _cpu_pass(result)
    require(public.get("status") == "18_PUBLIC_CPU_FIXTURES_PASS", "SGLang public CPU gate missing")
    _public_rows(public, "sglang")
    prepared = cpu_root / "prepared"
    frozen.mapping(frozen.read(cpu["prepared_inventory"]["path"]), str(prepared))
    groups = {}
    qualifications = {}
    points = frozen.read(str(factory / "qualification-points.json"))
    for deployment in DEPLOYMENTS if deployments is None else set(deployments):
        qualification = admission["qualifications"].get(deployment)
        require(
            isinstance(qualification, dict) and set(qualification) == {"prefill", "decode"},
            "SGLang strict deployment qualification missing",
        )
        qualifications[deployment] = [
            _sg_qualification(frozen, qualification[phase], deployment, phase, producer, points)
            for phase in ("prefill", "decode")
        ]
        require(
            len({p["actual_allocator_policy_sha256"] for p in qualifications[deployment]}) == 1,
            "SGLang qualification phases use different allocator policies",
        )
        children = []
        for role in ("calibration", "holdout"):
            root = prepared / "formal-inputs" / deployment / role
            manifest = frozen.read(str(root / "shard-manifest.json"))
            for item in manifest["shards"]:
                children.append(
                    dict(
                        child_cell_id=item["child_cell_id"],
                        child_plan_sha256=item["child_plan_sha256"],
                        parent_plan_sha256=item["parent_plan_sha256"],
                        role=role,
                        phase=item["phase"],
                        deployment=deployment,
                        original_point_ids=[p["original_point_id"] for p in item["point_map"]],
                        original_identity=item,
                        native_directory=str(task / root / "native" / item["child_cell_id"]),
                    )
                )
        groups[deployment] = children
    return dict(
        backend="sglang",
        version="0.5.20",
        children=_children(groups, expected_deployments=deployments),
        groups=groups,
        host=host,
        producer=producer,
        cpu_root=str(cpu_root),
        qualifications=qualifications,
        source=source,
    )


def _sg_qualification(frozen, refs, deployment, phase, producer, points):
    require(
        set(refs) == {"reader", "aggregation", "rows", "plan", "started", "checkpoint"},
        "SGLang qualification original files missing",
    )
    items = {k: frozen.read(frozen.ref(v)) for k, v in refs.items()}
    reader, aggregate, rows, plan, start, checkpoint = (
        items[k] for k in ("reader", "aggregation", "rows", "plan", "started", "checkpoint")
    )
    precision, tp = deployment.rsplit("-tp", 1)
    tp = int(tp)
    allocator = 16384 if deployment == "fp8-tp2" else None
    cells = [cell for cell in plan["cells"] if cell["workload_kind"] == phase]
    require(len(cells) == 1, "SGLang qualification phase is ambiguous")
    cell = cells[0]
    require(
        plan["backend"] == "sglang"
        and cell["topology"]["tp"] == tp
        and plan["options"]["benchmark_points"]["payload"] == points
        and plan["options"].get("sglang_allocator_max_split_size_mb") == allocator
        and plan["options"].get("sglang_mem_fraction_static") == 0.82,
        "SGLang qualification geometry/policy differs",
    )
    require(
        (start.get("source_commit"), start.get("wheel_sha256")) == producer
        and (start.get("precision"), start.get("tp")) == (precision, tp)
        and reader.get("status") == "passed"
        and reader.get("returncode") == 0
        and reader.get("phase") == phase
        and str(reader.get("source_job")) == str(start["job"])
        and reader["original_started_sha256"] == refs["started"]["sha256"]
        and reader["checkpoint_snapshot_sha256"] == refs["checkpoint"]["sha256"]
        and reader["frozen_plan_file_sha256"] == refs["plan"]["sha256"]
        and aggregate.get("status") == aggregate.get("validation") == aggregate.get("aggregation") == "passed"
        and (aggregate.get("reader_source_revision"), aggregate.get("installed_wheel_sha256")) == producer
        and aggregate["frozen_plan_file_sha256"] == refs["plan"]["sha256"]
        and aggregate["raw_directory"] == reader["original_raw_directory"]
        and aggregate["cell"] == cell
        and aggregate["row_count"] == len(rows) == (6 if phase == "prefill" else 3),
        "SGLang strict qualification source/result differs",
    )
    entry = checkpoint["cells"][cell["cell_id"]]
    require(
        start["plan_sha256"]
        == reader["plan_sha256"]
        == aggregate["plan_sha256"]
        == checkpoint["plan_sha256"]
        == plan["sha256"]
        and entry["status"] == "passed"
        and aggregate["attempt_id"] == entry["attempt_id"]
        and reader["checkpoint_phase"] == entry,
        "SGLang qualification original attempt differs",
    )

    def coordinates(values):
        result = []
        for value in values:
            names = ("batch_size", "total_kv_read_tokens") + (("total_prefill_tokens",) if phase == "prefill" else ())
            require(
                all(type(value.get(k)) is int and value[k] >= 0 for k in names) and value["batch_size"] > 0,
                "invalid qualification coordinate",
            )
            result.append(tuple(value[k] for k in names))
        require(len(set(result)) == len(result), "duplicate qualification coordinate")
        return set(result)

    require(coordinates(rows) == coordinates(points[phase]), "SGLang qualification point union differs")
    policies = set()
    for row in rows:
        policy = row.get("sglang_allocator_policy_sha256")
        require(isinstance(policy, str) and re.fullmatch(r"[0-9a-f]{64}", policy), "unknown actual allocator policy")
        policies.add(policy)
        require(
            row.get("sglang_allocator_max_split_size_mb") == allocator
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
            "SGLang qualification row provenance differs",
        )
    require(len(policies) == 1, "SGLang qualification mixes actual allocator policies")
    return dict(
        phase=phase,
        points=len(rows),
        source_job=reader["source_job"],
        actual_allocator_policy_sha256=policies.pop(),
        original_refs=refs,
    )


def frozen_contract(document, get):
    """Source-only gate. This does not require or manufacture GPU attempts."""
    if document.get("schema") == MIXED_SCHEMA:
        if __package__:
            from . import external_control_sglang_mixed as mixed
        else:
            import external_control_sglang_mixed as mixed
        return mixed.frozen_contract(document, get)
    require(
        document.get("schema") == SCHEMA and document.get("adapter") in {VLLM, SGLANG},
        "unsupported current external-control contract",
    )
    task = _control().absolute(document["original_task_root"])
    frozen = Frozen(get)
    anchors = document["anchors"]
    members = frozen.manifest(anchors["launcher_manifest"])
    require(anchors["admission"] in members, "admission is outside original launcher manifest")
    admission = frozen.read(anchors["admission"])
    context = (_vllm_frozen if document["adapter"] == VLLM else _sglang_frozen)(document, frozen, admission, task)
    _native_plans(context, frozen, task, anchors)
    context.update(files=frozen.files, admission=admission)
    return context


def execution_paths(document, run):
    return (
        vllm.execution_paths(run)
        if document["adapter"] == VLLM
        else [run["host_wheel_verification"], run["producer_wheel_verification"]]
    )


def closure(document, get):
    control = _control()
    context = frozen_contract(document, get)
    task = control.absolute(document["original_task_root"])
    admission = context["admission"]
    anchors = document.get("anchors", {})
    expected = dict(context["files"])
    runs = {r["cell_id"]: r for r in document["runs"]}
    require(
        len(runs) == len(document["runs"]) and runs.keys() == context["children"].keys(),
        "current external controls must cover every original formal child",
    )
    for cid, run in runs.items():
        child = context["children"][cid]
        if document["schema"] == MIXED_SCHEMA:
            admission = context["deployment_admissions"][child["deployment"]]
            anchors = context["deployment_anchors"][child["deployment"]]
        start = json.loads(get(run["started"]))
        provenance = json.loads(get(run["collector_provenance"]))
        raw = control.absolute(run["raw_root"])
        require(
            raw.is_relative_to(task / Path(run["started"]).parent)
            and task / run["collector_provenance"] == raw / "collector-provenance.json",
            "original native controls are outside their job/raw root",
        )
        require(
            start.get("state") == "RUNNING"
            and str(start["job"]) == Path(run["started"]).parent.name
            and start.get("deployment") == child["deployment"],
            "original started job/deployment differs",
        )
        require(
            start["admission_sha256"] == expected[anchors["admission"]]
            and start["launcher_manifest_sha256"] == expected[anchors["launcher_manifest"]],
            "original started launch hash chain differs",
        )
        require(
            (start["source_commit"], start["wheel_sha256"]) == context["producer"], "original producer identity differs"
        )
        if context["backend"] == "vllm":
            require(
                start.get("child") == child
                and start.get("runtime_version") == context["version"]
                and start.get("actual_formal_cpu_job") == admission["actual_formal_cpu_job"]
                and start.get("qualification") == admission["deployments"][child["deployment"]]["qualification"],
                "original nested child/CPU/qualification differs",
            )
            vllm.worker_evidence(
                run,
                get,
                task=task,
                provenance_sha=control.sha(get(run["collector_provenance"])),
                version=context["version"],
                closure=context["runtime_closure"],
                hook_sha=expected[anchors["cache_hook"]],
                user_sha=expected[anchors["usercustomize"]],
            )
        else:
            selected = start["selected"]
            require(
                start.get("mode") == "formal"
                and start.get("host_source_commit") == context["host"][0]
                and start.get("actual_cpu_job") == admission["actual_cpu"]["job_id"]
                and selected.get("kind") == "formal"
                and selected.get("role") == child["role"]
                and selected.get("child_identity") == child["original_identity"]
                and start.get("plan_sha256") == child["child_plan_sha256"],
                "original split-host child identity differs",
            )
            require(
                start.get("qualification") == context["qualifications"][child["deployment"]]
                and start.get("requested_allocator_policy")
                == {
                    "schema": "sglang_native_allocator_policy_v1",
                    "backend": "native",
                    "max_split_size_mb": 16384 if child["deployment"] == "fp8-tp2" else None,
                },
                "original SGLang qualification/allocator differs",
            )
            for label, expected_name in (
                ("host", "actual-host-wheel.json"),
                ("producer", "actual-producer-wheel.json"),
            ):
                path = run[label + "_wheel_verification"]
                require(
                    Path(path) == Path(run["started"]).parent / expected_name,
                    "installed wheel evidence is outside its original job",
                )
                proof = json.loads(get(path))
                valid = (
                    (
                        proof.get("status") == "ACTUAL_HOST_RECORD_GIT_COLLECTOR_RENDERER_AND_ARM_ELF_PASS"
                        and proof.get("head") == context["host"][0]
                    )
                    if label == "host"
                    else (
                        proof.get("state") == "EXACT_INSTALLED_WHEEL_RECORD_SOURCE_ELF_PASS"
                        and proof.get("source") == context["producer"][0]
                    )
                )
                require(
                    valid
                    and proof.get("wheel_sha256") == context[label][1]
                    and proof.get("runtime_sha256") == context["source"][label]["runtime_sha256"]
                    and isinstance(proof.get("files"), dict)
                    and type(context["source"][label]["record_payloads"]) is int
                    and context["source"][label]["record_payloads"] > 0
                    and len(proof["files"]) == context["source"][label]["record_payloads"]
                    and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in proof["files"].values()),
                    "actual split-host installed wheel evidence differs",
                )
        require(
            provenance.get("cell_id") == cid
            and provenance.get("plan_sha256") == child["child_plan_sha256"]
            and provenance.get("attempt_id")
            and provenance.get("runtime") == {"backend": context["backend"], "backend_version": context["version"]},
            "original native provenance/runtime differs",
        )
        for path in control.execution_paths(document, run):
            control.relative(path)
            require(path not in expected, "execution receipt aliases frozen or another execution file")
            expected[path] = control.sha(get(path))
    return expected, context["admission"]


def configuration_revisions(document, get, admission, backend, quant, tp, planner_revision):
    """Derive revision names only after validate() has checked the full attachment.

    Keep the plan's original aic_revision verbatim: an installed RECORD identity
    is not a Git SHA. Frozen source/installed-wheel proofs establish the separate
    host and native worker Git/wheel identities.
    """
    context = frozen_contract(document, get)
    require(context["admission"] == admission and context["backend"] == backend, "revision backend/admission differs")
    deployment = f"{quant}-tp{tp}"
    require(deployment in context["groups"], "revision deployment is absent")
    task = _control().absolute(document["original_task_root"])
    for child in context["groups"][deployment]:
        native = Path(child["native_directory"])
        path = str((native.parent.parent / "plans" / (child["child_cell_id"] + ".json")).relative_to(task))
        require(
            path in context["files"] and json.loads(get(path)).get("aic_revision") == planner_revision,
            "original plan renderer revision differs from table planner revision",
        )
    return {
        "planner_source_commit": context["host"][0],
        "planner_wheel_sha256": context["host"][1],
        "native_producer_revision": context["producer"][0],
        "native_producer_wheel_sha256": context["producer"][1],
    }


def bind_role(
    document, get, admission, pairs, plans, manifest_base, inventory, archive_source, *, storage_root_binding=None
):
    """Crossbind the original child bytes already validated by the strict stage."""
    control = _control()
    context = frozen_contract(document, get)
    require(context["admission"] == admission, "original admission changed during binding")
    task = control.absolute(document["original_task_root"])
    runs = {run["cell_id"]: run for run in document["runs"]}
    files = {row["original_path"]: row for row in document["files"]}
    required = {}
    for spec, evidence in pairs:
        cid = spec["cell_id"]
        require(cid in runs, "accepted child lacks original controls")
        run, child = runs[cid], context["children"][cid]
        plan = plans[spec["plan"]["path"]]
        raw = Path(spec["raw_root"])
        raw = raw if raw.is_absolute() else Path(manifest_base) / raw
        require(
            str(raw) == run["raw_root"]
            and plan["backend"] == context["backend"]
            and plan["sha256"] == child["child_plan_sha256"],
            "accepted child plan/raw identity differs",
        )
        native = Path(child["native_directory"])
        original_plan = native.parent.parent / "plans" / (cid + ".json")
        original_key = str(original_plan.relative_to(task))
        require(
            original_key in context["files"] and json.loads(get(original_key)) == plan,
            "accepted plan is not the original frozen child bytes",
        )
        require(plan["options"].get("dataset_role", "calibration") == child["role"], "accepted role differs")
        cells = [c for c in plan["cells"] if c["cell_id"] == cid]
        require(len(cells) == 1 and cells[0]["workload_kind"] == child["phase"], "accepted phase/cell differs")
        provenance = json.loads(get(run["collector_provenance"]))
        require(provenance["attempt_id"] == spec["attempt_id"], "accepted original attempt differs")
        receipts = {r["path"]: r["sha256"] for r in evidence["receipts"]}
        require(
            receipts.get("collector-provenance.json") == files[run["collector_provenance"]]["sha256"],
            "accepted native provenance bytes differ",
        )
        workers = [p for worker in run["workers"] for p in worker.values()] if context["backend"] == "vllm" else []
        if workers:
            require(len(run["workers"]) == cells[0]["topology"]["tp"], "native worker count differs from original TP")
        for path in workers:
            require(
                receipts.get(str((task / path).relative_to(raw))) == files[path]["sha256"],
                "accepted native worker bytes differ",
            )
        for path in control.execution_paths(document, run):
            original = archive.storage_path(task / path, storage_root_binding)
            require(original.is_relative_to(archive_source), "archive omits original external execution controls")
            required[original.relative_to(archive_source).as_posix()] = files[path]["sha256"]
    observed = {row["path"]: row["sha256"] for row in inventory if row["kind"] == "file" and row["path"] in required}
    require(observed == required, "archive changed or omitted original external controls")
