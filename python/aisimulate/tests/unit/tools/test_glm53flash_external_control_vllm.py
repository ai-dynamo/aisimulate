# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY: original offline fixtures, never production admission evidence."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.tools.test_glm53flash_external_control import (
    inventory,
    refreeze_test_launch,
    write,
)
from tests.unit.tools.test_glm53flash_external_control import (
    launch as sg_launch,
)
from tools.glm53flash_hf import external_control as control
from tools.glm53flash_hf import external_control_vllm as adapter

pytestmark = pytest.mark.unit


def cache(root, job, pid, hook, user):
    cache_root = f"/tmp/glm53-fpm-0-{job}/{pid}"
    values = {
        "pid": pid,
        "job_id": str(job),
        "uid": 0,
        "root": cache_root,
        "cubin_owned_write_read_delete": "passed",
        "cache_variables": {
            name: cache_root + "/" + sub
            for name, sub in (
                ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
                ("FLASHINFER_CUBIN_DIR", "flashinfer-cubin"),
                ("TRITON_CACHE_DIR", "triton"),
            )
        },
    }
    order = dict(
        pid=pid,
        before_framework_modules=[],
        cache_hook_sha256=hook,
        usercustomize_sha256=user,
        sitecustomize_file="/tmp/fpm-bench/sitecustomize.py",
        usercustomize_file="/opt/glm53flash-cache/usercustomize.py",
    )
    paths = {"cache": str(root / f"cache-setup-{pid}.json"), "order": str(root / f"cache-startup-order-{pid}.json")}
    return paths, values, order


@pytest.fixture
def launch(tmp_path):
    result = sg_launch.__wrapped__(tmp_path)
    source, original, anchors = result["source"], result["original"], result["anchors"]
    old_hook = anchors["cache_hook"]
    anchors.update(
        cache_hook="cpu-bundle/cache-hook/cache_hook.py",
        usercustomize="cpu-bundle/cache-hook/usercustomize.py",
        observer_entry="cpu-stage/sitecustomize.py",
        runtime_manifest="cpu-stage/runtime-source-sha256.json",
    )
    hook = write(source, anchors["cache_hook"], b"# TEST_ONLY cache original\n")
    user = write(source, anchors["usercustomize"], b"# TEST_ONLY usercustomize original\n")
    observer = write(source, anchors["observer_entry"], b"# TEST_ONLY packaged observer original\n")
    worker_hook = write(source, "cpu-stage/glm53flash_worker_hardware.py", b"# TEST_ONLY worker observer\n")
    write(source, "cpu-stage/glm53flash_scheduler.py", b"# TEST_ONLY scheduler\n")
    version = "0.30.0+glm53kpool.bf5f6b0e689d"
    runtime_manifest = {adapter.WORKER_SOURCE: "3" * 64}
    write(source, anchors["runtime_manifest"], runtime_manifest)
    identity = json.loads((source / anchors["source_identity"]).read_bytes())
    identity.update(
        cache_hook_sha256=hook,
        original_sitecustomize_sha256=observer,
        candidate_version=version,
        candidate_wheel_sha256="4" * 64,
    )
    write(source, anchors["source_identity"], identity)
    write(source, "cpu-bundle/source.json", identity)
    write(
        source,
        "cpu-bundle/manifest.sha256",
        (control.sha((source / "cpu-bundle/source.json").read_bytes()) + "  source.json\n").encode(),
    )
    write(
        source,
        "cpu/result.json",
        dict(status="passed", wheel_sha256=identity["wheel_sha256"], runs=[dict(returncode=0)]),
    )
    anchors["cache_cpu_result"] = "cpu/public.json"
    binaries = {f"vllm/TEST_ONLY_{i}.so": "5" * 64 for i in range(19)}
    sources = {adapter.WORKER_SOURCE: "3" * 64, **{f"vllm/TEST_ONLY_{i}.py": "6" * 64 for i in range(22)}}
    observed = {
        **sources,
        **binaries,
        adapter.REPAIR_SOURCE: "7" * 64,
        **{f"vllm/TEST_ONLY_graph_{i}.py": "9" * 64 for i in range(4)},
    }
    closure = {"contract_sha256": "8" * 64, "observed_files": observed}
    cpu_paths, frozen = [], {}
    for label, pid in (("preparation", 101), ("execution", 102)):
        parent = Path("cpu/native")
        paths, caches, order = cache(parent, "600", pid, hook, user)
        for key, data in (("cache", caches), ("order", order)):
            frozen[paths[key]] = write(source, paths[key], data)
        cpu = dict(
            status="passed",
            cuda_initialized=False,
            vllm_metadata_version=version,
            vllm_import_version=version,
            vllm_import_file="/opt/glm53flash-candidate/vllm/__init__.py",
            collector_import_file="/opt/glm53flash-current/collector/__init__.py",
            bootstrap_identity=dict(
                loaded_sitecustomize="/tmp/fpm-bench/sitecustomize.py",
                sitecustomize_sha256=observer,
                packaged_sitecustomize_sha256=observer,
                cache_hook_sha256=hook,
            ),
            startup_evidence=dict(
                pid=pid,
                cache_root=caches["root"],
                actual_flashinfer_cubin_dir=caches["cache_variables"]["FLASHINFER_CUBIN_DIR"],
                owned_write_read_delete="passed",
            ),
            startup_order_identity={k: v for k, v in order.items() if k != "pid"},
            actual_producer_binding=dict(
                scheduler="glm53flash_scheduler.Glm53FlashRealKVScheduler",
                worker_hook_source_sha256=worker_hook,
                native_worker_source_sha256="3" * 64,
                runtime_closure=closure,
            ),
            source_pins=sources,
            native_binaries=binaries,
            helper_sha256="7" * 64,
        )
        path = str(parent / f"{label}-identity.json")
        cpu_paths.append(path)
        frozen[path] = write(source, path, cpu)
    write(
        source,
        anchors["cache_cpu_result"],
        dict(
            status="passed",
            provenance={"runtime": {"backend": "vllm", "backend_version": version}},
            actual_import_identity=json.loads((source / cpu_paths[0]).read_bytes()),
        ),
    )
    admission = json.loads((source / anchors["admission"]).read_bytes())
    mounts = {p: str(original / f"TEST_ONLY_packages/{Path(p).name}") for p in adapter.PYTHONPATH[1:]}
    mounts["/opt/glm53flash-cache"] = str(original / Path(anchors["cache_hook"]).parent)
    admission.update(
        backend="vllm",
        framework_version=version,
        vllm_startup=dict(
            cpu_result="cpu/result.json",
            cpu_source_identity="cpu-bundle/source.json",
            cpu_bundle_manifest="cpu-bundle/manifest.sha256",
            cpu_identity_paths=cpu_paths,
            mounts=mounts,
        ),
    )
    admission.pop("framework")
    frozen.update({i["path"]: i["sha256"] for i in admission["bindings"] if i["path"] != old_hook})
    for i, (run, (spec, evidence)) in enumerate(zip(result["runs"], result["pairs"], strict=True)):
        cid, job = run["cell_id"], str(700 + i)
        parent = Path(f"inputs/native/{cid}")
        for name, data in {
            "collector-runtime-env.sh": (
                "export PYTHONPATH="
                + ":".join(adapter.PYTHONPATH)
                + "\nexport DYN_FPM_GLM53FLASH_REAL_KV=1\nexport AISIM_GLM53_PURPOSE=fpm\n"
            ).encode(),
            "sitecustomize.py": (source / anchors["observer_entry"]).read_bytes(),
            "runtime-source-sha256.json": (source / anchors["runtime_manifest"]).read_bytes(),
            "glm53flash_worker_hardware.py": (source / "cpu-stage/glm53flash_worker_hardware.py").read_bytes(),
            "glm53flash_scheduler.py": b"# TEST_ONLY scheduler\n",
            "run.sh": b"engine_command=(python3 -m dynamo.vllm --model TEST_ONLY)\n",
        }.items():
            frozen[str(parent / name)] = write(source, str(parent / name), data)
        plan = result["plans"][spec["plan"]["path"]]
        plan.update(backend="vllm", capability={"aic_database_version": version})
        plan["options"]["slurm_container_mounts"] = [f"{src}:{dst}:ro" for dst, src in mounts.items()]
        plan["cells"][0]["topology"] = {"tp": 2}
        raw = Path(run["collector_provenance"]).parent
        provenance = json.loads((source / run["collector_provenance"]).read_bytes())
        provenance["runtime"] = {"backend": "vllm", "backend_version": version}
        prov_sha = write(source, run["collector_provenance"], provenance)
        evidence["receipts"][0]["sha256"] = prov_sha
        workers, nvml = [], []
        for rank in range(2):
            pid = 1000 + i * 2 + rank
            uuid = f"00000000-0000-0000-0000-{rank:012d}"
            paths, caches, order = cache(raw, job, pid, hook, user)
            paths["device"] = str(raw / f"native-device-rank-{rank}.json")
            device = dict(
                status="passed",
                backend="vllm",
                backend_version=version,
                tp_rank=rank,
                tp_size=2,
                collector_provenance_sha256=prov_sha,
                runtime_closure=closure,
                worker_source_sha256="3" * 64,
                hardware={"uuid": uuid},
            )
            for key, data in (("cache", caches), ("order", order), ("device", device)):
                digest = write(source, paths[key], data)
                evidence["receipts"].append(dict(path=Path(paths[key]).name, sha256=digest))
            workers.append(paths)
            nvml.append(f"{pid}, GPU-{uuid}, VLLM::Worker_TP{rank}, 100 MiB\n")
        run["workers"] = workers
        run["nvml_stdout"] = f"witness/{job}/nvml.txt"
        run["nvml_receipt"] = f"witness/{job}/receipt.json"
        nvml_sha = write(source, run["nvml_stdout"], "".join(nvml).encode())
        write(
            source,
            run["nvml_receipt"],
            dict(
                job=job,
                returncode=0,
                observed_ns=1,
                nvml_sha256=nvml_sha,
                argv=[
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
            ),
        )
    for path in list(anchors.values())[2:] + [
        "cpu/result.json",
        "cpu-bundle/source.json",
        "cpu-bundle/manifest.sha256",
        "cpu-stage/glm53flash_scheduler.py",
    ]:
        frozen[path] = control.sha((source / path).read_bytes())
    admission["bindings"] = [dict(path=p, sha256=s) for p, s in frozen.items()]
    write(source, anchors["admission"], admission)
    refreeze_test_launch(result)
    return result


def prepare(fixture):
    return control.prepare(
        fixture["source"],
        str(fixture["original"]),
        fixture["anchors"],
        fixture["runs"],
        fixture["output"],
        adapter=adapter.ADAPTER,
    )


def bind(fixture, document):
    _, get, admission = control.validate(fixture["output"], document)
    control.bind_role(
        document,
        get,
        admission,
        fixture["pairs"],
        fixture["plans"],
        "/",
        inventory(fixture),
        fixture["original"] / "runs",
    )


def test_complete_vllm_attachment_replays_offline_and_binds_original_worker_bytes(launch):
    document = prepare(launch)
    bind(launch, document)
    assert document["adapter"] == adapter.ADAPTER
    assert not launch["original"].exists()
    assert all(not Path(item["path"]).is_absolute() for item in document["files"])


@pytest.mark.parametrize("purpose", [None, "fpm", "", "ops", "FPM", "unknown"])
def test_native_frozen_fpm_default_requires_absence_or_exact_explicit_value(launch, purpose):
    # a590's actual native producer omits this variable; its original observer
    # and scheduler use the FPM default. Keep the same source/worker closure and
    # exercise archive reconstruction rather than only parsing the environment.
    for run in launch["runs"]:
        path = launch["source"] / "inputs/native" / run["cell_id"] / "collector-runtime-env.sh"
        text = path.read_text().replace("export AISIM_GLM53_PURPOSE=fpm\n", "")
        if purpose is not None:
            text += f"export AISIM_GLM53_PURPOSE={purpose}\n"
        path.write_text(text)
    refreeze_test_launch(launch)
    document = prepare(launch)
    if purpose in (None, "fpm"):
        bind(launch, document)
    else:
        with pytest.raises(ValueError, match="FPM activation differs"):
            bind(launch, document)


def test_standalone_bundle_executes_without_repository_pythonpath(launch, tmp_path):
    bundle = tmp_path / "TEST_ONLY_standalone"
    bundle.mkdir()
    for name in (
        "raw_archive.py",
        "raw_campaign.py",
        "external_control.py",
        "external_control_vllm.py",
        "external_control_current.py",
    ):
        shutil.copyfile(Path(control.__file__).with_name(name), bundle / name)
    request = tmp_path / "TEST_ONLY_request.json"
    request.write_text(
        json.dumps(
            dict(
                original_task_root=str(launch["original"]),
                adapter=adapter.ADAPTER,
                anchors=launch["anchors"],
                runs=launch["runs"],
            )
        )
    )
    # -E/-s suppress inherited Python configuration while keeping the script's
    # directory available for this explicitly standalone standard-library bundle.
    result = subprocess.run(
        [
            sys.executable,
            "-E",
            "-s",
            str(bundle / "external_control.py"),
            "--source-root",
            str(launch["source"]),
            "--request",
            str(request),
            "--output",
            str(launch["output"]),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    bind(launch, json.loads((launch["output"] / "external-control.json").read_bytes()))


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("cpu", "vllm_metadata_version", "0.30.0"),
        ("cpu", "cuda_initialized", True),
        ("cpu", "native_binaries", {}),
        ("cpu", "source_pins", {}),
        ("cpu", "actual_producer_binding", {}),
        ("worker", "runtime_closure", {}),
        ("worker", "collector_provenance_sha256", "f" * 64),
        ("worker", "tp_rank", 7),
        ("worker", "backend_version", "0.30.0"),
        ("order", "before_framework_modules", ["torch"]),
        ("order", "usercustomize_sha256", "f" * 64),
        ("cache", "pid", 999),
        ("cache", "job_id", "other"),
    ],
)
def test_corrupt_original_startup_or_worker_evidence_rejects(launch, target, field, value):
    path = (
        "cpu/native/execution-identity.json"
        if target == "cpu"
        else launch["runs"][0]["workers"][0][{"worker": "device", "order": "order", "cache": "cache"}[target]]
    )
    record = json.loads((launch["source"] / path).read_bytes())
    record[field] = value
    write(launch["source"], path, record)
    refreeze_test_launch(launch)
    with pytest.raises((ValueError, KeyError)):
        prepare(launch)


@pytest.mark.parametrize(
    "mutation", ["observer", "scheduler", "path_order", "shadow", "user_site", "ops", "isolated_python", "worker_bytes"]
)
def test_native_route_or_accepted_worker_mismatch_rejects(launch, mutation):
    source = launch["source"]
    cid = launch["runs"][0]["cell_id"]
    parent = Path(f"inputs/native/{cid}")
    env = source / parent / "collector-runtime-env.sh"
    if mutation == "observer":
        (source / parent / "sitecustomize.py").write_text("TEST_ONLY historical bootstrap\n")
    elif mutation == "scheduler":
        (source / parent / "glm53flash_scheduler.py").write_text("TEST_ONLY different scheduler\n")
    elif mutation == "path_order":
        env.write_text(env.read_text().replace(":".join(adapter.PYTHONPATH), ":".join(reversed(adapter.PYTHONPATH))))
    elif mutation == "shadow":
        next(iter(launch["plans"].values()))["options"]["slurm_container_mounts"].append("/different:/tmp/fpm-bench:ro")
    elif mutation in ("user_site", "ops"):
        env.write_text(
            env.read_text()
            + (
                "export PYTHONNOUSERSITE=1\n"
                if mutation == "user_site"
                else "export AISIM_GLM53_OPS_MANIFEST=/other.json\n"
            )
        )
    elif mutation == "isolated_python":
        path = source / parent / "run.sh"
        path.write_text(path.read_text().replace("python3 -m", "python3 -I -m"))
    else:
        launch["pairs"][0][1]["receipts"][1]["sha256"] = "f" * 64
    refreeze_test_launch(launch)
    document = prepare(launch)
    with pytest.raises(ValueError):
        bind(launch, document)


def test_nvml_reused_helper_pid_is_not_worker_identity(launch):
    run = launch["runs"][0]
    write(launch["source"], run["nvml_stdout"], b"9999, TEST_ONLY_UUID0, python3, 100 MiB\n")
    witness = json.loads((launch["source"] / run["nvml_receipt"]).read_bytes())
    witness["nvml_sha256"] = control.sha((launch["source"] / run["nvml_stdout"]).read_bytes())
    write(launch["source"], run["nvml_receipt"], witness)
    with pytest.raises(ValueError, match="NVML join"):
        prepare(launch)


def test_unknown_adapter_does_not_fall_back_to_sglang(launch):
    with pytest.raises(ValueError, match="unsupported external startup adapter"):
        control.prepare(
            launch["source"],
            str(launch["original"]),
            launch["anchors"],
            launch["runs"],
            launch["output"],
            adapter="unknown",
        )


@pytest.mark.parametrize("mutation", ["wheel", "source", "manifest", "missing_start", "native_runtime", "timestamp"])
def test_original_cpu_or_launch_chain_cannot_be_replaced_with_archive_assertions(launch, mutation):
    source = launch["source"]
    if mutation in {"wheel", "source"}:
        path = "cpu/result.json" if mutation == "wheel" else "cpu-bundle/source.json"
        value = json.loads((source / path).read_bytes())
        value["wheel_sha256" if mutation == "wheel" else "source_commit"] = "f" * (64 if mutation == "wheel" else 40)
        write(source, path, value)
    elif mutation == "manifest":
        write(source, "cpu-bundle/manifest.sha256", b"f" * 64 + b"  source.json\n")
    elif mutation == "missing_start":
        (source / launch["runs"][0]["started"]).unlink()
    else:
        run = launch["runs"][0]
        path = run["collector_provenance"] if mutation == "native_runtime" else run["nvml_receipt"]
        value = json.loads((source / path).read_bytes())
        if mutation == "native_runtime":
            value["runtime"]["backend_version"] = "0.30.0"
        else:
            value.pop("observed_ns")
        write(source, path, value)
    if mutation != "missing_start":
        refreeze_test_launch(launch)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(launch)
