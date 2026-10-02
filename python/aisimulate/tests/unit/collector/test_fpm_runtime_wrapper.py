# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""etcd lifecycle and fail-closed environment tests for fpm_exec.sh.

The gate/checker/barrier behavior of the runtime lives in test_fpm_exec.py;
this file pins what the retired run_with_etcd.sh wrapper used to own: leader
etcd startup ordering, follower abstention, cleanup, the exported etcd
endpoint, and the fail-closed propagation of sourcing fpm_env.sh.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aisimulate.fpm_contract import (
    FPM_ENV_EXPORTED_VARS,
    FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FPM_EXEC = REPO_ROOT / "collector/fpm_forward/runtime/fpm_exec.sh"

_ETCD_STUB = """#!/bin/bash
printf '%s\\n' "$$" "$@" > "${FPM_ETCD_TRACE}"
exec python3 - <<'PY'
import socket
import time

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server.bind(("127.0.0.1", 2379))
    server.listen(8)
except OSError:
    pass
time.sleep(3600)
PY
"""

_PYTHON3_SHADOW = """#!/bin/bash
if [[ "${1:-}" == "-" ]]; then
  /bin/cat >/dev/null
fi
exit 0
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _valid_result() -> dict:
    return {
        "schema_version": FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
        "status": "complete",
        "valid": True,
        "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
        "config": {"mode": "prefill"},
        "results": [{"point": {"point_type": "prefill"}, "fpms": [{"dp_rank": 0}]}],
        "skipped_points": [],
        "errors": [],
    }


def _stage(
    tmp_path: Path,
    *,
    run_script: str,
    env_overrides: dict[str, object] | None = None,
    env_script: str | None = None,
    shadow_python3: bool = False,
) -> SimpleNamespace:
    workdir = tmp_path / "fpm-bench"
    results = tmp_path / "results"
    bin_dir = tmp_path / "bin"
    for directory in (workdir, results, bin_dir):
        directory.mkdir(exist_ok=True)

    script = FPM_EXEC.read_text()
    assert script.count("workdir=/tmp/fpm-bench") == 1
    script = script.replace("workdir=/tmp/fpm-bench", f"workdir={workdir}")
    assert script.count("/results/") == 4
    script = script.replace("/results/", f"{results}/")
    script_path = tmp_path / "fpm_exec.sh"
    script_path.write_text(script)

    if env_script is None:
        values: dict[str, object] = {
            "FPM_NODE_COUNT": 1,
            "FPM_DATA_PARALLEL_SIZE": 1,
            "FPM_LOCAL_DATA_PARALLEL_SIZE": 1,
            "FPM_BENCHMARK_MODE": "prefill",
            "FPM_BENCHMARK_OUTPUT_PATH": str(results / "benchmark.json"),
            "FPM_WAIT_TIMEOUT_SECONDS": 30,
            "FPM_RESULT_SCHEMA_VERSION": FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
            "FPM_NODE_RANK": 0,
            "FPM_MASTER_ADDR": "127.0.0.1",
        }
        values.update(env_overrides or {})
        exports = "\n".join(f"export {name}={shlex.quote(str(values[name]))}" for name in FPM_ENV_EXPORTED_VARS)
        env_script = f"#!/usr/bin/env bash\n{exports}\n"
    (workdir / "fpm_env.sh").write_text(env_script)
    (workdir / "collector-runtime-env.sh").write_text("export FPM_READINESS_TIMEOUT_SECONDS=900\n")
    (workdir / "preflight.py").write_text("")
    (workdir / "run.sh").write_text(run_script)
    _write_executable(bin_dir / "etcd", _ETCD_STUB)
    if shadow_python3:
        _write_executable(bin_dir / "python3", _PYTHON3_SHADOW)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FPM_ETCD_TRACE"] = str(tmp_path / "etcd.trace")
    return SimpleNamespace(
        script=script_path,
        workdir=workdir,
        results=results,
        env=env,
        etcd_trace=tmp_path / "etcd.trace",
    )


def _run(staged: SimpleNamespace, *, timeout: int = 60, extra_env: dict[str, str] | None = None):
    env = dict(staged.env)
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(staged.script)],
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
        check=False,
    )


@pytest.mark.parametrize("status,mask", [("observed", [0]), ("unavailable", []), ("partial", list(range(16)))])
def test_slurm_same_step_cpu_guard_preserves_failure_before_model_start(tmp_path, status, mask):
    staged = _stage(tmp_path, run_script="touch model-started\n")
    staged.script.write_text(staged.script.read_text().replace('Path("/results")', f"Path({str(staged.results)!r})"))
    # The observer's Linux API is tested separately. This seam verifies the
    # real wrapper enforces its result before etcd/preflight/model startup.
    payload = {"status": status, "main_thread_allowed_cpus": mask}
    (staged.workdir / "fpm_memory_observer.py").write_text(
        "import json\ndef observe_cpu(kind, **kwargs):\n"
        f"    payload = {payload!r}\n"
        "    payload.update(kind=kind,requested=kwargs['requested_cpus_per_task'])\n"
        "    (kwargs['directory']/'fpm-cpu-launcher.json').write_text(json.dumps(payload))\n"
        "    return payload\n"
    )
    result = _run(
        staged,
        extra_env={
            "FPM_SLURM_CPUS_PER_TASK": "16",
            "FPM_SLURM_CPU_BIND": "cores",
            "FPM_LOCAL_GPU_COUNT": "4",
        },
    )
    assert result.returncode != 0
    assert "CPU affinity" in result.stderr
    assert json.loads((staged.results / "fpm-cpu-launcher.json").read_text())["requested"] == 16
    assert not staged.etcd_trace.exists()
    assert not (staged.workdir / "model-started").exists()


def test_fpm_exec_starts_leader_etcd_before_preflight():
    """The follower readiness probe budget only covers pod-exec skew when the
    leader's etcd starts before the unbounded vLLM/torch preflight import."""

    script = FPM_EXEC.read_text()
    assert script.index("etcd_pid=$!") < script.index('python3 "${workdir}/preflight.py"')
    assert "time.monotonic() + float(sys.argv[2])" in script


@pytest.mark.parametrize("pending_memory", [False, True])
@pytest.mark.parametrize("existing", [None, "", "/other/modules"])
def test_memory_observer_pythonpath_is_available_before_engine_launch(tmp_path, pending_memory, existing):
    staged = _stage(tmp_path, run_script="#!/bin/bash\n")
    if pending_memory:
        (staged.workdir / "fpm_memory_worker.py").write_text("# staged observer\n")
    script = staged.script.read_text().split('etcd_endpoint="http://')[0]
    script += '\nprintf "%s" "${PYTHONPATH:-}"\n'
    env = dict(staged.env)
    if existing is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = existing
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, check=True)
    expected = existing or ""
    if pending_memory:
        expected = str(staged.workdir) + (":" + expected if expected else "")
    assert result.stdout == expected


def test_fpm_exec_leader_starts_etcd_and_cleanup_stops_it(tmp_path):
    output_path = tmp_path / "results" / "benchmark.json"
    staged = _stage(
        tmp_path,
        # Write the complete result, then idle: the engine must stay alive
        # while the gate validates so this pins the success path, not the
        # engine-early-exit path.
        run_script=(
            "#!/usr/bin/env bash\n"
            f"printf '%s' {shlex.quote(json.dumps(_valid_result()))} > {shlex.quote(str(output_path))}\n"
            "exec sleep 300\n"
        ),
    )

    completed = _run(staged)

    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    trace_lines = staged.etcd_trace.read_text().splitlines()
    etcd_pid = int(trace_lines[0])
    etcd_args = trace_lines[1:]
    assert "--data-dir" in etcd_args
    assert "--listen-client-urls" in etcd_args
    assert etcd_args[etcd_args.index("--advertise-client-urls") + 1] == "http://127.0.0.1:2379"
    assert (staged.results / "etcd.log").exists()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(etcd_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"etcd stub {etcd_pid} survived fpm_exec.sh cleanup")


def test_fpm_exec_follower_does_not_start_etcd(tmp_path):
    staged = _stage(
        tmp_path,
        run_script="#!/usr/bin/env bash\nexit 0\n",
        env_overrides={
            "FPM_NODE_COUNT": 2,
            "FPM_DATA_PARALLEL_SIZE": 1,
            "FPM_NODE_RANK": 1,
        },
        shadow_python3=True,
    )

    completed = _run(staged, timeout=30)

    assert completed.returncode == 0, completed.stderr
    assert not staged.etcd_trace.exists()


def test_fpm_exec_exports_the_leader_etcd_endpoint_to_the_engine(tmp_path):
    trace_file = tmp_path / "endpoint.txt"
    staged = _stage(
        tmp_path,
        run_script=('#!/usr/bin/env bash\nprintf \'%s\\n\' "${ETCD_ENDPOINTS}" > "${TRACE_FILE}"\nexit 0\n'),
        env_overrides={
            "FPM_NODE_COUNT": 2,
            "FPM_DATA_PARALLEL_SIZE": 1,
            "FPM_NODE_RANK": 1,
            "FPM_MASTER_ADDR": "explicit-leader",
        },
        shadow_python3=True,
    )

    completed = _run(staged, timeout=30, extra_env={"TRACE_FILE": str(trace_file)})

    assert completed.returncode == 0, completed.stderr
    assert trace_file.read_text().strip() == "http://explicit-leader:2379"


def test_preflight_writes_failed_audit_when_runtime_import_fails(tmp_path, monkeypatch):
    """An image predating PR11509 has no dynamo.vllm.instrumented_scheduler at
    all; the audit artifact exists to document exactly that rejection, so the
    import failure must still produce it before the pod fails."""

    import sys

    from collector.fpm_forward.runtime import preflight

    audit_path = tmp_path / "runtime-preflight.json"
    monkeypatch.setattr(preflight, "_AUDIT_PATH", audit_path)
    monkeypatch.setitem(sys.modules, "dynamo.vllm.instrumented_scheduler", None)

    with pytest.raises(RuntimeError, match="Provide a compatible Dynamo image"):
        preflight.main()

    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "failed"
    assert "dynamo" in audit["import_error"]
    assert audit["missing_fields"] == []
    assert audit["missing_methods"] == []
    assert audit["runtime_contract"] == "dynamo_pr11509_native_schema_v2_kvwarm_v1"


def test_preflight_rejects_runtime_without_kvwarm_capability(tmp_path, monkeypatch):
    """A native-schema image is still incompatible when it predates the
    strategy-aware KV warm-up predicate required by pure-TP decode."""

    import sys
    from types import ModuleType

    from collector.fpm_forward.runtime import preflight

    benchmark_point = type(
        "BenchmarkPoint",
        (),
        {"__dataclass_fields__": {name: object() for name in preflight.GRAPH_AWARE_FIELDS}},
    )
    scheduler = type(
        "InstrumentedScheduler",
        (),
        {name: lambda self: None for name in preflight.GRAPH_AWARE_METHODS if name != "_kvwarm_warm_eligible"},
    )
    module = ModuleType("dynamo.vllm.instrumented_scheduler")
    module.BenchmarkPoint = benchmark_point
    module.InstrumentedScheduler = scheduler
    audit_path = tmp_path / "runtime-preflight.json"
    monkeypatch.setattr(preflight, "_AUDIT_PATH", audit_path)
    monkeypatch.setitem(sys.modules, "dynamo.vllm.instrumented_scheduler", module)

    with pytest.raises(RuntimeError, match="_kvwarm_warm_eligible"):
        preflight.main()

    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "failed"
    assert audit["missing_methods"] == ["_kvwarm_warm_eligible"]


@pytest.mark.parametrize(
    ("phase", "setting", "native_capability", "adapter", "accepted"),
    [
        ("prefill", "on", False, False, False),
        ("agg", "true", False, False, False),
        ("prefill", "on", True, False, True),
        ("prefill", "off", False, False, True),
        ("prefill", "0", False, False, True),
        ("decode", "on", False, False, True),
        ("prefill", "on", False, True, True),
    ],
)
def test_preflight_checks_requested_prefill_seeding_before_model_load(
    tmp_path, monkeypatch, phase, setting, native_capability, adapter, accepted
):
    import sys
    from types import ModuleType

    from collector.fpm_forward.runtime import preflight

    def no_initialization(*_args, **_kwargs):
        pytest.fail("preflight must inspect capabilities without constructing the model or scheduler")

    methods = {name: lambda self: None for name in preflight.GRAPH_AWARE_METHODS}
    methods["__init__"] = no_initialization
    if native_capability:
        methods["_bench_realseed_on"] = lambda self: True
    scheduler = type("InstrumentedScheduler", (), methods)
    module = ModuleType("dynamo.vllm.instrumented_scheduler")
    module.BenchmarkPoint = type(
        "BenchmarkPoint", (), {"__dataclass_fields__": {name: object() for name in preflight.GRAPH_AWARE_FIELDS}}
    )
    module.InstrumentedScheduler = scheduler
    monkeypatch.setitem(sys.modules, "dynamo.vllm.instrumented_scheduler", module)
    monkeypatch.setenv("FPM_BENCHMARK_MODE", phase)
    monkeypatch.setenv("DYN_BENCH_PREFILL_REAL_SEED", setting)
    monkeypatch.delenv("DYN_FPM_DSV41_REAL_KV", raising=False)
    if adapter:
        adapter_module = ModuleType("dsv41_scheduler")
        adapter_module.DeepseekV41RealKVScheduler = scheduler
        monkeypatch.setitem(sys.modules, "dsv41_scheduler", adapter_module)
        monkeypatch.setenv("DYN_FPM_DSV41_REAL_KV", "1")
    audit_path = tmp_path / "runtime-preflight.json"
    monkeypatch.setattr(preflight, "_AUDIT_PATH", audit_path)

    if accepted:
        preflight.main()
    else:
        with pytest.raises(RuntimeError, match="_bench_realseed_on"):
            preflight.main()
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == ("passed" if accepted else "failed")
    assert audit["missing_methods"] == ([] if accepted else ["_bench_realseed_on"])
    assert audit["prefill_real_seed_requested"] == (phase in {"prefill", "agg"} and setting in {"on", "true"})


def test_fpm_exec_propagates_fail_closed_env_source(tmp_path):
    """fpm_env.sh exits 2 on incomplete multinode discovery; sourcing must
    terminate fpm_exec.sh with the same status before any resource starts."""

    staged = _stage(
        tmp_path,
        run_script='#!/usr/bin/env bash\ntouch "${ENGINE_TRACE}"\nexit 0\n',
        env_script=(
            "#!/usr/bin/env bash\n"
            'echo "Multinode FPM requires rank and leader discovery from FPM_NODE_*, LWS, or Grove" >&2\n'
            "exit 2\n"
        ),
    )
    engine_trace = tmp_path / "engine.trace"

    completed = _run(staged, timeout=30, extra_env={"ENGINE_TRACE": str(engine_trace)})

    assert completed.returncode == 2
    assert "requires rank and leader discovery" in completed.stderr
    assert not staged.etcd_trace.exists()
    assert not engine_trace.exists()


@pytest.mark.parametrize("failure", [RuntimeError("pinned source mismatch"), ValueError("adapter activation failed")])
def test_preflight_preserves_non_importerror_runtime_failure(tmp_path, monkeypatch, failure):
    import builtins

    from collector.fpm_forward.runtime import preflight

    original_import = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == "dynamo.vllm.instrumented_scheduler":
            raise failure
        return original_import(name, *args, **kwargs)

    audit = tmp_path / "runtime-preflight.json"
    monkeypatch.setattr(preflight, "_AUDIT_PATH", audit)
    monkeypatch.setattr(builtins, "__import__", fail)
    with pytest.raises(RuntimeError, match="runtime activation or identity preflight failed") as caught:
        preflight.main()
    assert caught.value.__cause__ is failure
    receipt = json.loads(audit.read_text())
    assert receipt["status"] == "failed" and receipt["import_error"] == str(failure)
