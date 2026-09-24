# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original validators for the packaged observer plus native usercustomize route.

This attachment crossbinds original evidence; it does not qualify a runtime or
replace the strict native reader. Formal data requires its own original launch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
from pathlib import Path

if __package__:
    from . import raw_archive as archive
else:
    import raw_archive as archive

require = archive.require
ADAPTER = "vllm_packaged_observer_usercustomize_v1"
ANCHORS = {
    "launcher_manifest",
    "admission",
    "source_identity",
    "cache_cpu_result",
    "cache_hook",
    "usercustomize",
    "observer_entry",
    "runtime_manifest",
}
PYTHONPATH = [
    "/tmp/fpm-bench",
    "/opt/glm53flash-cache",
    "/opt/glm53flash-current",
    "/opt/glm53flash-candidate",
    "/opt/glm53flash-dynamo",
]
WORKER_SOURCE = "vllm/v1/worker/gpu_worker.py"
REPAIR_SOURCE = "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"


def execution_paths(run):
    return [run["nvml_receipt"], run["nvml_stdout"], *(p for worker in run["workers"] for p in worker.values())]


def cache_identity(cache, order, *, hook_sha, user_sha, job, pid):
    require(type(pid) is int and pid > 0, "invalid original cache PID")
    require(
        cache.get("pid") == order.get("pid") == pid and cache.get("job_id") == str(job),
        "cache/order PID or allocation mismatch",
    )
    require(
        order.get("before_framework_modules") == []
        and order.get("cache_hook_sha256") == hook_sha
        and order.get("usercustomize_sha256") == user_sha
        and order.get("sitecustomize_file") == "/tmp/fpm-bench/sitecustomize.py"
        and order.get("usercustomize_file") == "/opt/glm53flash-cache/usercustomize.py",
        "original cache startup order or hook identity mismatch",
    )
    uid = cache.get("uid")
    require(type(uid) is int and uid >= 0, "cache UID missing")
    root = f"/tmp/glm53-fpm-{uid}-{job}/{pid}"
    require(
        cache.get("root") == root and cache.get("cubin_owned_write_read_delete") == "passed",
        "cache root/write proof mismatch",
    )
    variables = cache.get("cache_variables", {})
    for name, child in {
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "FLASHINFER_CUBIN_DIR": "flashinfer-cubin",
        "TRITON_CACHE_DIR": "triton",
    }.items():
        require(variables.get(name) == root + "/" + child, "cache directories are not allocation/PID private")
    return root


def cpu_identity(value, get, root, *, hook_sha, user_sha, observer_sha, version, manifest):
    """Validate one original CPU startup; returns its exact worker closure."""
    require(
        value.get("status") == "passed" and value.get("cuda_initialized") is False,
        "CPU import qualification did not pass",
    )
    require(
        value.get("vllm_metadata_version") == value.get("vllm_import_version") == version,
        "actual CPU runtime version mismatch",
    )
    require(
        value.get("vllm_import_file") == "/opt/glm53flash-candidate/vllm/__init__.py"
        and value.get("collector_import_file") == "/opt/glm53flash-current/collector/__init__.py",
        "CPU import paths differ from startup route",
    )
    bootstrap = value["bootstrap_identity"]
    require(
        bootstrap.get("loaded_sitecustomize") == "/tmp/fpm-bench/sitecustomize.py"
        and bootstrap.get("sitecustomize_sha256") == bootstrap.get("packaged_sitecustomize_sha256") == observer_sha
        and bootstrap.get("cache_hook_sha256") == hook_sha,
        "CPU observer is not the frozen packaged entrypoint",
    )
    startup = value["startup_evidence"]
    pid = startup["pid"]
    cache_path, order_path = str(root / f"cache-setup-{pid}.json"), str(root / f"cache-startup-order-{pid}.json")
    cache, order = json.loads(get(cache_path)), json.loads(get(order_path))
    require(
        value["startup_order_identity"] == {k: v for k, v in order.items() if k != "pid"},
        "CPU identity differs from original startup-order receipt",
    )
    cache_root = cache_identity(cache, order, hook_sha=hook_sha, user_sha=user_sha, job=cache["job_id"], pid=pid)
    require(
        startup.get("cache_root") == cache_root
        and startup.get("actual_flashinfer_cubin_dir") == cache["cache_variables"]["FLASHINFER_CUBIN_DIR"]
        and startup.get("owned_write_read_delete") == "passed",
        "actual FlashInfer CPU path/write mismatch",
    )
    binding = value["actual_producer_binding"]
    require(
        binding.get("scheduler") == "glm53flash_scheduler.Glm53FlashRealKVScheduler",
        "CPU real-state scheduler was not active",
    )
    closure = binding["runtime_closure"]
    observed, binaries, sources = closure["observed_files"], value["native_binaries"], value["source_pins"]
    require(
        re.fullmatch(r"[0-9a-f]{64}", closure.get("contract_sha256", ""))
        and len(observed) == 47
        and len(binaries) == 19
        and len(sources) == 23,
        "CPU runtime closure inventory incomplete",
    )
    require(all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in observed.values()), "invalid runtime file digest")
    require(
        {k: v for k, v in observed.items() if not k.endswith(".py")} == binaries, "CPU native binary closure differs"
    )
    require(
        all(observed.get(k) == v for k, v in sources.items())
        and observed.get(REPAIR_SOURCE) == value.get("helper_sha256")
        and observed.get(WORKER_SOURCE) == binding.get("native_worker_source_sha256"),
        "CPU source/worker/repair closure differs",
    )
    require(
        all(observed.get(k) == v for k, v in manifest.items() if k.startswith("vllm/") and k != REPAIR_SOURCE),
        "CPU sources differ from packaged observer runtime manifest",
    )
    return closure, [cache_path, order_path]


def frozen_startup(anchors, get, bindings, identity, admission):
    from_adapter = admission.get("vllm_startup", {})
    require(
        admission.get("backend") == "vllm" and admission.get("framework_version") == identity.get("candidate_version"),
        "vLLM admission runtime mismatch",
    )
    for key, name in {
        "cache_hook": "cache_hook.py",
        "usercustomize": "usercustomize.py",
        "observer_entry": "sitecustomize.py",
        "runtime_manifest": "runtime-source-sha256.json",
    }.items():
        require(Path(anchors[key]).name == name, "unsupported vLLM startup entrypoint")
    require(
        Path(anchors["cache_hook"]).parent == Path(anchors["usercustomize"]).parent,
        "usercustomize and cache hook must be adjacent",
    )
    require(
        identity.get("cache_hook_sha256") == bindings[anchors["cache_hook"]]
        and identity.get("original_sitecustomize_sha256") == bindings[anchors["observer_entry"]],
        "source identity selects a different cache/observer",
    )
    for key in ("cpu_result", "cpu_source_identity", "cpu_bundle_manifest"):
        require(from_adapter.get(key) in bindings, "original CPU source/result/manifest not frozen")
    result = json.loads(get(from_adapter["cpu_result"]))
    require(
        result.get("status") == "passed"
        and result.get("wheel_sha256") == admission["wheel_sha256"]
        and result.get("runs")
        and all(run.get("returncode") == 0 for run in result["runs"]),
        "actual CPU qualification wheel/result mismatch",
    )
    cpu_source = json.loads(get(from_adapter["cpu_source_identity"]))
    require(
        all(
            cpu_source.get(k) == identity.get(k)
            for k in (
                "source_commit",
                "wheel_sha256",
                "candidate_version",
                "candidate_wheel_sha256",
                "cache_hook_sha256",
                "original_sitecustomize_sha256",
            )
        ),
        "original CPU source identity differs from formal source",
    )
    bundle_parent = Path(from_adapter["cpu_bundle_manifest"]).parent
    members = set()
    for line in get(from_adapter["cpu_bundle_manifest"]).decode().splitlines():
        require(len(line) > 66 and line[64:66] == "  ", "invalid original CPU source manifest")
        parts = archive.relative_parts(line[66:])
        path = str(bundle_parent.joinpath(*parts))
        require(
            path not in members and bindings.get(path) == line[:64], "CPU source manifest closure missing/conflicting"
        )
        members.add(path)
    require(from_adapter["cpu_source_identity"] in members, "CPU source identity was not in original source manifest")
    manifest = json.loads(get(anchors["runtime_manifest"]))
    public = json.loads(get(anchors["cache_cpu_result"]))
    require(
        public.get("status") == "passed"
        and public["provenance"]["runtime"] == {"backend": "vllm", "backend_version": admission["framework_version"]},
        "public prepare CPU provenance mismatch",
    )
    paths = from_adapter.get("cpu_identity_paths", [])
    require(
        len(paths) == 2
        and [Path(p).name for p in paths] == ["preparation-identity.json", "execution-identity.json"]
        and len(set(paths)) == 2,
        "two original CPU startup identities required",
    )
    closures, pids = [], []
    for path in paths:
        require(path in bindings, "CPU startup identity was not frozen by admission")
        value = json.loads(get(path))
        closure, originals = cpu_identity(
            value,
            get,
            Path(path).parent,
            hook_sha=bindings[anchors["cache_hook"]],
            user_sha=bindings[anchors["usercustomize"]],
            observer_sha=bindings[anchors["observer_entry"]],
            version=admission["framework_version"],
            manifest=manifest,
        )
        require(all(p in bindings for p in originals), "CPU original cache/order files were not frozen")
        closures.append(closure)
        pids.append(value["startup_evidence"]["pid"])
    require(closures[0] == closures[1] and pids[0] != pids[1], "CPU preparation and execution runtime/PID mismatch")
    require(
        public["actual_import_identity"] == json.loads(get(paths[0])),
        "public prepare does not bind its original CPU identity",
    )
    return closures[0]


def worker_evidence(run, get, *, task, provenance_sha, version, closure, hook_sha, user_sha):
    """Rederive worker rank -> NVML UUID/PID -> original startup receipts."""
    witness = json.loads(get(run["nvml_receipt"]))
    job = Path(run["started"]).parent.name
    raw = Path(run["raw_root"])
    text = get(run["nvml_stdout"])
    require(
        witness.get("job") == job
        and witness.get("returncode") == 0
        and type(witness.get("observed_ns")) is int
        and witness["observed_ns"] > 0
        and witness.get("nvml_sha256") == hashlib.sha256(text).hexdigest(),
        "original NVML job/output binding mismatch",
    )
    argv = witness.get("argv", [])
    require(
        argv
        == [
            "srun",
            f"--jobid={job}",
            "--overlap",
            "--ntasks=1",
            "--nodes=1",
            "--cpus-per-task=1",
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        "unqualified original NVML command",
    )
    rows = [[part.strip() for part in row] for row in csv.reader(text.decode().splitlines()) if row]
    require(all(len(row) == 4 for row in rows), "malformed original NVML output")
    workers = run["workers"]
    require(len(workers) in (2, 4), "missing complete native worker set")
    used_pids, used_uuids = set(), set()
    for rank, paths in enumerate(workers):
        require(set(paths) == {"device", "cache", "order"}, "worker evidence fields differ")
        require(
            all(task / Path(p).parent == raw for p in paths.values()),
            "native worker evidence is outside selected raw root",
        )
        device, cache, order = (json.loads(get(paths[k])) for k in ("device", "cache", "order"))
        require(
            Path(paths["device"]).name == f"native-device-rank-{rank}.json"
            and device.get("tp_rank") == rank
            and device.get("tp_size") == len(workers),
            "native worker rank set mismatch",
        )
        require(
            device.get("status") == "passed"
            and device.get("backend") == "vllm"
            and device.get("backend_version") == version
            and device.get("collector_provenance_sha256") == provenance_sha
            and device.get("runtime_closure") == closure
            and device.get("worker_source_sha256") == closure["observed_files"][WORKER_SOURCE],
            "actual worker source/library/runtime closure mismatch",
        )
        uuid = device["hardware"]["uuid"]
        require(
            isinstance(uuid, str) and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid),
            "native GPU UUID is not in the recorded torch format",
        )
        # torch.cuda's recorded UUID is bare; original nvidia-smi CSV prefixes GPU-.
        matches = [row for row in rows if row[1] == f"GPU-{uuid}" and row[2] == f"VLLM::Worker_TP{rank}"]
        require(len(matches) == 1 and matches[0][0].isdigit(), "missing/ambiguous original worker NVML join")
        pid = int(matches[0][0])
        require(pid not in used_pids and uuid not in used_uuids, "duplicate native worker PID/UUID")
        used_pids.add(pid)
        used_uuids.add(uuid)
        require(
            Path(paths["cache"]).name == f"cache-setup-{pid}.json"
            and Path(paths["order"]).name == f"cache-startup-order-{pid}.json",
            "worker PID selects different startup receipts",
        )
        cache_identity(cache, order, hook_sha=hook_sha, user_sha=user_sha, job=job, pid=pid)
    return workers


def native_environment(plan, cid, admission, anchors, get, *, task):
    """Check only the explicit frozen route, never infer it from substrings."""
    require(
        plan["backend"] == "vllm" and plan["capability"]["aic_database_version"] == admission["framework_version"],
        "frozen vLLM plan/runtime mismatch",
    )
    mounts = admission["vllm_startup"].get("mounts", {})
    require(set(mounts) == set(PYTHONPATH[1:]), "frozen vLLM runtime mount mapping incomplete")
    require(
        mounts["/opt/glm53flash-cache"] == str(task / Path(anchors["cache_hook"]).parent),
        "cache mount source differs from original hook",
    )
    actual = plan["options"].get("slurm_container_mounts", [])
    for target, source in mounts.items():
        require(
            Path(source).is_absolute() and str(Path(source)) == source and ".." not in Path(source).parts,
            "unsafe native mount source",
        )
        require(actual.count(f"{source}:{target}:ro") == 1, "frozen read-only runtime mount missing/duplicated")
    for mount in actual:
        parts = mount.split(":")
        require(len(parts) in (2, 3), "ambiguous native container mount")
        target = Path(parts[1])
        for protected in map(Path, PYTHONPATH):
            if target.is_relative_to(protected) or protected.is_relative_to(target):
                require(
                    str(target) in mounts and mount == f"{mounts[str(target)]}:{target}:ro",
                    "mount shadows native observer/runtime",
                )
    bindings = {item["path"]: item["sha256"] for item in admission["bindings"]}
    candidates = [p for p in bindings if p.endswith(f"/native/{cid}/collector-runtime-env.sh")]
    require(len(candidates) == 1, "admitted child runtime environment missing or ambiguous")
    parent = Path(candidates[0]).parent
    env = {}
    for line in get(candidates[0]).decode().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        require(
            line.startswith("export ") and not any(c in line for c in "$`\\"),
            "native environment must use literal exports",
        )
        tokens = shlex.split(line)
        require(len(tokens) == 2 and "=" in tokens[1], "unsupported native export syntax")
        name, value = tokens[1].split("=", 1)
        require(name not in env, "duplicate native environment field")
        env[name] = value
    require(
        env.get("PYTHONPATH") == ":".join(PYTHONPATH)
        and env.get("DYN_FPM_GLM53FLASH_REAL_KV") == "1"
        # The source-bound native observer and scheduler both default an absent
        # purpose to FPM. Explicit empty/other values must still be rejected.
        and env.get("AISIM_GLM53_PURPOSE", "fpm") == "fpm",
        "native observer/cache order or FPM activation differs",
    )
    require(
        not env.get("PYTHONNOUSERSITE") and not env.get("AISIM_GLM53_OPS_MANIFEST"),
        "native usercustomize disabled or Ops observer enabled",
    )
    for name, anchor in [("sitecustomize.py", "observer_entry"), ("runtime-source-sha256.json", "runtime_manifest")]:
        path = str(parent / name)
        require(
            path in bindings and get(path) == get(anchors[anchor]),
            "child observer/runtime manifest differs from frozen anchor",
        )
    cpu = json.loads(get(admission["vllm_startup"]["cpu_identity_paths"][0]))
    worker_hook = str(parent / "glm53flash_worker_hardware.py")
    require(
        worker_hook in bindings
        and bindings[worker_hook] == cpu["actual_producer_binding"]["worker_hook_source_sha256"],
        "child worker observer differs from actual CPU source",
    )
    scheduler = str(parent / "glm53flash_scheduler.py")
    cpu_scheduler = str(Path(anchors["observer_entry"]).parent / "glm53flash_scheduler.py")
    require(
        scheduler in bindings and cpu_scheduler in bindings and get(scheduler) == get(cpu_scheduler),
        "child scheduler differs from frozen original CPU stage",
    )
    run_path = str(parent / "run.sh")
    require(run_path in bindings, "child native command missing")
    commands = [line for line in get(run_path).decode().splitlines() if line.startswith("engine_command=(")]
    require(len(commands) == 1 and commands[0].endswith(")"), "native command missing/ambiguous")
    argv = shlex.split(commands[0][len("engine_command=(") : -1])
    require(
        len(argv) > 2 and Path(argv[0]).name in {"python", "python3"} and argv[1] == "-m" and argv[2] == "dynamo.vllm",
        "native command disables or bypasses normal Python startup",
    )
