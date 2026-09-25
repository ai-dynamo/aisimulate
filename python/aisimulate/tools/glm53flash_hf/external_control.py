# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve a frozen Slurm launch's finite file closure without changing raw data.

The attachment is assembled after execution. Its historical authority is the
original started.json -> launcher/admission hash chain, never its creation time.
Only standard-library code; source paths are provenance, not tar member names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

if __package__:
    from . import external_control_current as current
    from . import external_control_vllm as vllm
    from . import raw_archive as archive
else:
    import external_control_current as current
    import external_control_vllm as vllm
    import raw_archive as archive

SCHEMA = "glm53flash_external_control_v1"
ADAPTER = "sglang_pid_private_sitecustomize_v1"
require = archive.require


def relative(value):
    require(archive.relative_parts(value), "external control file path is empty")
    return value


def absolute(value):
    require(isinstance(value, str) and value.startswith("/"), "original path must be absolute")
    relative(value[1:])
    return Path(value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(root, name):
    """Refuse symlinks and source mutation while reading a small control file."""
    root = archive.absolute_safe(root)
    parts = archive.relative_parts(relative(name))
    fd = os.open(root, archive.FLAGS | os.O_DIRECTORY)
    try:
        with archive.open_at(fd, parts) as source:
            before = os.fstat(source)
            require(stat.S_ISREG(before.st_mode), "external control must be a regular file")
            with os.fdopen(os.dup(source), "rb") as stream:
                data = stream.read()
            archive.assert_stat(os.fstat(source), archive.stat_identity(before), name)
        path = archive.absolute_safe(root.joinpath(*parts))
        archive.assert_stat(path.stat(), archive.stat_identity(before), name)
    finally:
        os.close(fd)
    return data


def sums(data, parent):
    result = {}
    for line in data.decode("utf-8").splitlines():
        require(len(line) > 66 and line[64:66] == "  ", "invalid frozen launcher SHA256 manifest")
        digest, name = line[:64], relative(line[66:])
        require(re.fullmatch(r"[0-9a-f]{64}", digest), "invalid frozen launcher digest")
        name = (Path(parent) / name).as_posix()
        require(name not in result, "duplicate launcher manifest path")
        result[name] = digest
    require(result, "empty frozen launcher manifest")
    return result


def closure(document, get):
    """Validate original launch semantics using a caller's bytes-only resolver."""
    if document.get("schema") == current.SCHEMA:
        return current.closure(document, get)
    require(document.get("schema") == SCHEMA, "invalid external control schema")
    require(document.get("adapter") in (ADAPTER, vllm.ADAPTER), "unsupported external startup adapter")
    is_vllm = document["adapter"] == vllm.ADAPTER
    task = absolute(document["original_task_root"])
    anchors = document["anchors"]
    require(
        set(anchors)
        == (
            vllm.ANCHORS
            if is_vllm
            else {"launcher_manifest", "admission", "cache_hook", "source_identity", "cache_cpu_result"}
        ),
        "external control anchors differ",
    )
    for path in anchors.values():
        relative(path)
    manifest = get(anchors["launcher_manifest"])
    members = sums(manifest, Path(anchors["launcher_manifest"]).parent)
    require(anchors["admission"] in members, "admission is not in frozen launcher manifest")
    admission_bytes = get(anchors["admission"])
    admission = json.loads(admission_bytes)
    if not is_vllm:
        require(admission.get("framework") == "sglang0.5.20", "startup adapter requires frozen SGLang 0.5.20")
    require(
        admission.get("status") == "CONFIGURATION_QUALIFIED_FOR_FROZEN_FORMAL_COLLECTION", "launch was not qualified"
    )
    bindings = {}
    for item in admission["bindings"]:
        path = relative(item["path"])
        require(
            path not in bindings and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
            "invalid or duplicate admission binding",
        )
        bindings[path] = item["sha256"]
    for key in set(anchors) - {"launcher_manifest", "admission"}:
        require(anchors[key] in bindings, f"{key} was not frozen by admission")
    if not is_vllm:
        require(Path(anchors["cache_hook"]).name == "sitecustomize.py", "unsupported cache hook entrypoint")
    expected = {anchors["launcher_manifest"]: sha(manifest)}
    for paths in (members, bindings):
        for path, digest in paths.items():
            require(path not in expected or expected[path] == digest, "conflicting launch file identity")
            expected[path] = digest
            require(sha(get(path)) == digest, "frozen launch file SHA256 mismatch")
    identity = json.loads(get(anchors["source_identity"]))
    require(
        identity["source_commit"] == admission["source_commit"]
        and ("wheel_sha256" not in identity or identity["wheel_sha256"] == admission["wheel_sha256"]),
        "source/wheel identity differs from frozen admission",
    )
    # Some original preparation receipts predate the optional wheel field.
    # The actual installed producer and original started receipts always bind it.
    launch = Path(anchors["launcher_manifest"]).parent
    producer_path = (launch / "cpu-public-host-producer.json").as_posix()
    wheel_path = (launch / "installed-wheel-source-record.json").as_posix()
    require(
        producer_path in members and wheel_path in members,
        "frozen actual producer/wheel receipts missing from launcher manifest",
    )
    producer = json.loads(get(producer_path))
    wheel = json.loads(get(wheel_path))
    require(
        producer.get("state") == "passed"
        and producer.get("source_commit") == wheel.get("head_sha") == admission["source_commit"]
        and producer.get("wheel_sha256") == wheel.get("wheel_sha256") == admission["wheel_sha256"],
        "actual producer/wheel identity differs from frozen admission",
    )
    require(
        identity["cache_hook_sha256"] == bindings[anchors["cache_hook"]], "source identity names a different cache hook"
    )
    if is_vllm:
        runtime_closure = vllm.frozen_startup(anchors, get, bindings, identity, admission)
    else:
        cpu = json.loads(get(anchors["cache_cpu_result"]))
        require(
            cpu.get("status") == "passed"
            and cpu.get("cache_hook_sha256") == bindings[anchors["cache_hook"]]
            and cpu.get("host_and_producer_source") == admission["source_commit"]
            and cpu.get("runs")
            and all(run.get("returncode") == 0 for run in cpu["runs"]),
            "cache CPU evidence is not passed for the admitted hook/source",
        )
    children = {child["child_cell_id"]: child for child in admission["children"]}
    require(len(children) == len(admission["children"]) == admission["child_count"], "duplicate/missing admitted child")
    runs = {run["cell_id"]: run for run in document["runs"]}
    require(
        len(runs) == len(document["runs"]) and runs.keys() == children.keys(),
        "external controls must bind exactly all admitted children",
    )
    for cell_id, run in runs.items():
        start_path = relative(run["started"])
        provenance_path = relative(run["collector_provenance"])
        start = json.loads(get(start_path))
        provenance = json.loads(get(provenance_path))
        raw = absolute(run["raw_root"])
        require(raw.is_relative_to(task / Path(start_path).parent), "native raw is outside its original job output")
        require(
            (task / provenance_path).parent == raw and Path(provenance_path).name == "collector-provenance.json",
            "native provenance is outside the selected raw root",
        )
        require(
            start.get("state") == "RUNNING" and start.get("child") == children[cell_id],
            "original started receipt selects a different child",
        )
        require(str(start["job"]) == Path(start_path).parent.name, "job identity differs from original output path")
        require(
            start["admission_sha256"] == sha(admission_bytes) and start["launcher_manifest_sha256"] == sha(manifest),
            "started receipt is not bound to this frozen launch",
        )
        require(
            start["host_and_producer_source"] == admission["source_commit"]
            and start["host_and_producer_wheel_sha256"] == admission["wheel_sha256"],
            "started source/wheel differs from admission",
        )
        require(
            provenance["cell_id"] == cell_id
            and provenance["plan_sha256"] == children[cell_id]["child_plan_sha256"]
            and provenance.get("attempt_id"),
            "native attempt differs from launched child plan",
        )
        if is_vllm:
            require(
                provenance.get("runtime") == {"backend": "vllm", "backend_version": admission["framework_version"]},
                "native collector runtime differs from frozen vLLM admission",
            )
            vllm.worker_evidence(
                run,
                get,
                task=task,
                provenance_sha=sha(get(provenance_path)),
                version=admission["framework_version"],
                closure=runtime_closure,
                hook_sha=bindings[anchors["cache_hook"]],
                user_sha=bindings[anchors["usercustomize"]],
            )
        for path in execution_paths(document, run):
            require(path not in expected, "execution receipt aliases a frozen launch file")
            expected[path] = sha(get(path))
    return expected, admission


def execution_paths(document, run):
    paths = [run["started"], run["collector_provenance"]]
    if document.get("schema") == current.SCHEMA:
        paths.extend(current.execution_paths(document, run))
    elif document["adapter"] == vllm.ADAPTER:
        paths.extend(vllm.execution_paths(run))
    require(len(paths) == len(set(paths)), "duplicate external execution path")
    return paths


def validate(root, document):
    """Revalidate a portable attachment, returning its original-path byte resolver."""
    files = document["files"]
    index = {item["original_path"]: item for item in files}
    require(len(index) == len(files), "duplicate original external control path")
    for path, item in index.items():
        relative(path)
        require(item["path"] == "files/" + item["sha256"], "external control storage is not content-addressed")
        require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "invalid external control digest")
        require(item["scope"] in {"frozen_launch", "original_execution"}, "invalid external control scope")

    def get(path):
        require(path in index, "missing external control file")
        item = index[path]
        data = read_bytes(root, item["path"])
        require(
            sha(data) == item["sha256"] and len(data) == item["bytes"], "external control content identity mismatch"
        )
        return data

    expected, admission = closure(document, get)
    require(expected == {p: i["sha256"] for p, i in index.items()}, "external control file closure differs")
    execution = {p for run in document["runs"] for p in execution_paths(document, run)}
    require(
        all(i["scope"] == ("original_execution" if p in execution else "frozen_launch") for p, i in index.items()),
        "external control scope differs from original role",
    )
    return index, get, admission


def prepare(source_root, original_task_root, anchors, runs, output, *, adapter=ADAPTER):
    """Map an explicit source root to original absolute provenance; no discovery."""
    source_root = archive.absolute_safe(source_root)
    output = archive.absolute_safe(output, must_exist=False)
    require(
        not output.exists() and not output.is_relative_to(source_root), "attachment output exists or is inside source"
    )
    document = dict(
        schema=current.SCHEMA if adapter in current.ADAPTERS else SCHEMA,
        adapter=adapter,
        original_task_root=str(absolute(original_task_root)),
        anchors=anchors,
        runs=runs,
    )
    observed = {}

    def get(path):
        data = read_bytes(source_root, path)
        require(path not in observed or observed[path] == data, "external control source changed")
        observed[path] = data
        return data

    expected, _ = closure(document, get)
    output.mkdir(mode=0o700)
    (output / "files").mkdir()
    execution = {p for run in runs for p in execution_paths(document, run)}
    files = []
    for path, digest in sorted(expected.items()):
        data = get(path)
        target = output / "files" / digest
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
        files.append(
            dict(
                original_path=path,
                path="files/" + digest,
                sha256=digest,
                bytes=len(data),
                scope="original_execution" if path in execution else "frozen_launch",
            )
        )
    document["files"] = files
    validate(output, document)
    for path in expected:
        get(path)  # Preserve the original byte identity through the entire copy.
    archive.write_json(output / "external-control.json", document)
    return document


def bind_role(document, get, admission, pairs, plans, manifest_base, inventory, archive_source):
    """Join selected accepted attempts to the archived original started bytes."""
    if document.get("schema") == current.SCHEMA:
        return current.bind_role(document, get, admission, pairs, plans, manifest_base, inventory, archive_source)
    task = absolute(document["original_task_root"])
    runs = {run["cell_id"]: run for run in document["runs"]}
    children = {child["child_cell_id"]: child for child in admission["children"]}
    files = {item["original_path"]: item for item in document["files"]}
    required = {}
    for spec, evidence in pairs:
        cid = spec["cell_id"]
        require(cid in runs, "accepted child missing external execution controls")
        run, child = runs[cid], children[cid]
        raw = Path(spec["raw_root"])
        raw = raw if raw.is_absolute() else Path(manifest_base) / raw
        plan = plans[spec["plan"]["path"]]
        is_vllm = document["adapter"] == vllm.ADAPTER
        require(plan["backend"] == ("vllm" if is_vllm else "sglang"), "startup adapter does not cover this backend")
        require(
            str(raw) == run["raw_root"] and plan["sha256"] == child["child_plan_sha256"],
            "external control selects a different raw root or plan",
        )
        cell = next(c for c in plan["cells"] if c["cell_id"] == cid)
        require(
            child["role"] == plan["options"].get("dataset_role", "calibration")
            and child["phase"] == cell["workload_kind"],
            "external control child phase/role differs",
        )
        if is_vllm:
            vllm.native_environment(plan, cid, admission, document["anchors"], get, task=task)
            require(len(run["workers"]) == cell["topology"]["tp"], "native worker set differs from frozen TP")
        else:
            hook_mount = str(task / Path(document["anchors"]["cache_hook"]).parent) + ":/opt/glm53flash-cache:ro"
            require(
                hook_mount in plan["options"].get("slurm_container_mounts", []),
                "frozen plan does not mount the admitted hook read-only",
            )
            env_paths = [
                item["path"]
                for item in admission["bindings"]
                if item["path"].endswith(f"/native/{cid}/collector-runtime-env.sh")
            ]
            require(len(env_paths) == 1, "admitted child runtime environment missing or ambiguous")
            require(
                any(
                    line.startswith("export PYTHONPATH=/opt/glm53flash-cache:")
                    for line in get(env_paths[0]).decode().splitlines()
                ),
                "admitted hook is not first on native PYTHONPATH",
            )
        provenance = json.loads(get(run["collector_provenance"]))
        require(provenance["attempt_id"] == spec["attempt_id"], "external control native attempt differs")
        native_receipts = {item["path"]: item["sha256"] for item in evidence["receipts"]}
        require(
            native_receipts.get("collector-provenance.json") == files[run["collector_provenance"]]["sha256"],
            "external provenance differs from accepted native bytes",
        )
        worker_paths = [path for item in run["workers"] for path in item.values()] if is_vllm else []
        for path in worker_paths:
            name = str((task / path).relative_to(raw))
            require(
                native_receipts.get(name) == files[path]["sha256"],
                "external worker evidence differs from accepted native bytes",
            )
        for path in [run["started"], run["collector_provenance"], *worker_paths]:
            original = task / path
            require(original.is_relative_to(archive_source), "archive must retain original execution controls")
            required[original.relative_to(archive_source).as_posix()] = files[path]["sha256"]
    observed = {
        item["path"]: item["sha256"] for item in inventory if item["kind"] == "file" and item["path"] in required
    }
    require(observed == required, "archived original started/provenance SHA differs from external controls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--request", type=Path, required=True, help="JSON: original_task_root, anchors and all admitted runs"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    prepare(
        args.source_root,
        request["original_task_root"],
        request["anchors"],
        request["runs"],
        args.output,
        adapter=request.get("adapter", ADAPTER),
    )


if __name__ == "__main__":
    main()
