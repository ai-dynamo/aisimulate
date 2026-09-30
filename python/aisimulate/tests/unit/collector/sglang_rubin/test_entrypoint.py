# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pilot reuses real planning, execution accounting, and publication."""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow.parquet as pq
import pytest
import yaml
from collector.sglang_rubin import collect, runtime
from collector.sglang_rubin.registry import MODEL_PATH, REGISTRY, SGLANG_COMMIT, SGLANG_DISTRIBUTION_VERSION

pytestmark = pytest.mark.unit


def test_plan_only_uses_canonical_plan_without_gpu_imports(tmp_path):
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(collect._PACKAGE_ROOT)!r})\n"
        "from collector.sglang_rubin.collect import main\n"
        "assert main(['--plan-only']) == 0\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'sglang' not in sys.modules\n"
        "assert 'collector.collect' not in sys.modules\n"
        "assert 'collector.helper' not in sys.modules\n"
        "assert 'helper' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["model_path"] == MODEL_PATH
    assert plan["sm_version"] == 107
    assert plan["ops"] == sorted(entry.op for entry in REGISTRY)
    assert plan["qualification"].startswith("unqualified")
    assert plan["declared_serving_configuration"] == runtime.declared_serving_configuration()
    assert plan["observed_serving_environment"] == {name: os.environ.get(name) for name in runtime.REQUIRED_SERVING_ENV}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("import_order", ["pilot_first", "executor_first"])
def test_real_spawn_shares_helper_restart_signal(tmp_path, import_order):
    script = tmp_path / "spawn_probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import multiprocessing as mp
            import sys
            sys.path[:0] = [{str(collect._PACKAGE_ROOT)!r}, {str(collect._PACKAGE_ROOT / "collector")!r}]
            if sys.argv[1] == 'pilot_first':
                from collector.sglang_rubin import collect as pilot

            def worker(queue):
                import collector.collect as executor
                from collector.sglang_rubin import collect as pilot
                import collector.helper as qualified
                import helper as short
                queue.put((short is qualified, isinstance(qualified.WORKER_RESTART, executor.WorkerRestartSignal)))

            if __name__ == '__main__':
                from collector.sglang_rubin import collect as pilot
                pilot._load_executor()
                context = mp.get_context('spawn')
                queue = context.Queue()
                child = context.Process(target=worker, args=(queue,))
                with pilot._worker_helper_imports():
                    child.start()
                result = queue.get(timeout=30)
                child.join(timeout=30)
                assert child.exitcode == 0, child.exitcode
                assert result == (True, True), result
            """
        )
    )
    result = subprocess.run(
        [sys.executable, str(script), import_order], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_module_cli_restarts_real_spawn_workers(monkeypatch, tmp_path, checkpoint, inventory):
    # Exercise the actual `python -m collector.sglang_rubin` launch. The startup
    # hook imports the pilot before spawn.prepare(), while the child's name is
    # still MainProcess. Only the CUDA interface/kernel/inventory are CPU fakes;
    # worker restart detection, checkpoints, and publication are real.
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda device: None)\n"
        "device = lambda name: name\n"
    )
    (tmp_path / "fake_kernel.py").write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            def get_gemm_test_cases():
                return [['bfloat16', 1, 16, 16], ['bfloat16', 2, 16, 16]]
            def run_gemm(dtype, m, n, k, *, device, perf_filename):
                from collector.helper import WORKER_RESTART, log_perf
                Path(f'worker-{{m}}.pid').write_text(str(os.getpid()))
                assert log_perf([{{'gemm_dtype': dtype, 'm': m, 'n': n, 'k': k, 'latency': 0.125}}],
                    'sglang', {SGLANG_DISTRIBUTION_VERSION!r}, 'cpu-test', 'gemm', 'fake-kernel', perf_filename)
                return WORKER_RESTART
            """
        )
    )
    (tmp_path / "sitecustomize.py").write_text(
        textwrap.dedent(
            f"""
            import json
            import sys
            from pathlib import Path
            from collector.sglang_rubin import collect as pilot
            import fake_kernel
            sys.modules['collector.sglang_rubin.collect_gemm'] = fake_kernel
            inventory_path = Path({str(tmp_path / "inventory.json")!r})
            pilot.collect_inventory = lambda **kwargs: json.loads(inventory_path.read_text())
            """
        )
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(collect._PACKAGE_ROOT)))
    output = tmp_path / "output"
    arguments = _arguments(checkpoint, output)
    arguments.remove("--sequential")
    result = subprocess.run(
        [sys.executable, "-m", "collector.sglang_rubin", *arguments, "--processes", "1"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "worker-1.pid").read_text() != (output / "worker-2.pid").read_text()
    assert pq.read_metadata(output / "gemm_perf.parquet").num_rows == 2


@pytest.mark.parametrize(
    "args",
    [
        ["--plan-only", "--model-path", "zai-org/GLM-5.2-FP8"],
        ["--plan-only", "--ops", "attention"],
        ["--plan-only", "--ops", "gemm", "gemm"],
        ["--plan-only", "--limit", "0"],
        ["--plan-only", "--ops", "moe", "--case-filter", "4"],
        ["--plan-only", "--ops", "gemm", "moe", "--case-filter", "4"],
        ["--resume"],
        ["--resume-retry-failed"],
    ],
)
def test_invalid_scope_fails_before_execution(args):
    with pytest.raises(SystemExit) as error:
        collect.main(args)
    assert error.value.code == 2


def test_registry_preserves_standard_checkpoint_and_table_contracts():
    from collector.sglang.registry import REGISTRY as STOCK_REGISTRY

    stock = {entry.op: entry for entry in STOCK_REGISTRY}
    for entry in REGISTRY:
        assert entry.module.startswith("collector.sglang_rubin.")
        assert (entry.get_func, entry.run_func, entry.perf_filename, entry.worker_perf_filename) == (
            stock[entry.op].get_func,
            stock[entry.op].run_func,
            stock[entry.op].perf_filename,
            stock[entry.op].worker_perf_filename,
        )


def test_gemm_filter_is_recorded_and_selects_an_existing_shape(monkeypatch):
    from collector.sglang_rubin.collect_gemm import get_gemm_test_cases

    fragment = "['bfloat16', 128, 6144, 6144]"
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL_PATH)
    args = collect._parse_args(["--plan-only", "--ops", "gemm", "--case-filter", fragment])
    _plan, document = collect._build_plan(args)
    assert document["runtime_case_filters"]["gemm"] == [fragment]
    assert [case for case in get_gemm_test_cases() if fragment in str(case)] == [["bfloat16", 128, 6144, 6144]]


def test_real_populations_match_visible_tp4_scope(monkeypatch):
    from collector.sglang_rubin import collect_mla_module, collect_moe

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL_PATH)
    cases = collect_moe.get_moe_test_cases()
    selected = [case for case in cases if any(value in str(case) for value in collect._case_filters("moe"))]
    assert selected
    assert all(case[6:9] == [4, 1, MODEL_PATH] for case in selected)
    cases = collect_mla_module.get_dsa_context_module_test_cases()
    selected = [
        case for case in cases if any(value in str(case) for value in collect._case_filters("dsa_context_module"))
    ]
    assert selected
    assert all(case[2] == 16 and case[9] == 4 for case in selected)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    for filename, suffix in (("config.json", "config"), ("hf_quant_config.json", "hf_quant_config")):
        shutil.copyfile(collect._MODEL_CONFIG.with_name(f"nvidia--GLM-5.2-NVFP4_{suffix}.json"), path / filename)
    return path


@pytest.fixture
def inventory(checkpoint):
    files = {
        path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
        for path in checkpoint.iterdir()
    }
    return {
        "declared_image": {"reference": runtime.IMAGE_REF},
        "declared_serving_configuration": runtime.declared_serving_configuration(),
        "launcher_provenance": {"image": runtime.IMAGE_REF, "image_identity_verified": False},
        "observed": {
            "platform": {"system": "Linux", "machine": "aarch64"},
            "reported_build_environment": dict(runtime.EXPECTED_BUILD_ENV),
            "serving_environment": dict(runtime.REQUIRED_SERVING_ENV),
            "package_versions": {
                name: {"version": version, "error": None, "error_type": None}
                for name, version in (("sglang", SGLANG_DISTRIBUTION_VERSION), ("torch", "test-cpu-fake"))
            },
            "cuda": {"available": True, "devices": [{"index": 0, "capability": [10, 7]}], "torch_cuda_version": "13.5"},
            "checkpoint": {"directory": str(checkpoint), "files": files},
            "imports": {
                name: {"file": "test-fake", "error": None} for name in ("sglang", "aisimulate", "aisimulate._runtime")
            },
        },
    }


def test_checkpoint_policy_and_snapshot_are_checked(checkpoint, inventory):
    assert collect._checkpoint_errors(checkpoint, inventory) == []
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["quant_algo"] = "FP8"
    config_path.write_text(json.dumps(config))
    assert any("config.json SHA-256" in error for error in collect._checkpoint_errors(checkpoint, inventory))
    inventory["observed"]["checkpoint"]["files"]["hf_quant_config.json"]["sha256"] = "0" * 64
    assert any("hf_quant_config.json SHA-256" in error for error in collect._checkpoint_errors(checkpoint, inventory))


def _arguments(checkpoint, output):
    return [
        "--checkpoint-dir",
        str(checkpoint),
        "--launcher-image",
        runtime.IMAGE_REF,
        "--output-dir",
        str(output),
        "--ops",
        "gemm",
        "--sequential",
    ]


def test_failed_preflight_writes_evidence_and_never_loads_executor(monkeypatch, tmp_path, checkpoint, inventory):
    inventory["observed"]["imports"]["aisimulate._runtime"]["error"] = "ImportError: absent native extension"
    monkeypatch.setattr(collect, "collect_inventory", lambda **kwargs: inventory)
    monkeypatch.setattr(collect, "_load_executor", lambda: pytest.fail("executor entered after failed preflight"))
    output = tmp_path / "output"
    assert collect.main(_arguments(checkpoint, output)) == 1
    artifact = json.loads(next(output.glob("inventory-*.json")).read_text())
    assert any("absent native extension" in error for error in artifact["validation"]["errors"])
    assert not (output / "pilot_identity.json").exists()


@pytest.fixture
def cpu_executor(monkeypatch, tmp_path):
    # Existing collector tests can import the same file through both historical
    # names. Isolate this fixture's import state; production starts a fresh CLI.
    helper = sys.modules.get("helper") or sys.modules.get("collector.helper")
    if helper is not None:
        monkeypatch.setitem(sys.modules, "helper", helper)
        monkeypatch.setitem(sys.modules, "collector.helper", helper)
    executor = collect._load_executor()
    logger = logging.getLogger("rubin-entrypoint-test")

    def setup_logging(**kwargs):
        log_dir = Path.cwd() / "test-logs"
        log_dir.mkdir(exist_ok=True)
        monkeypatch.setenv("COLLECTOR_LOG_DIR", str(log_dir))
        return logger

    monkeypatch.setattr(executor, "setup_logging", setup_logging)
    monkeypatch.setattr(executor, "_require_torch", lambda: SimpleNamespace(device=lambda name: name))
    monkeypatch.setattr(executor, "get_device_module", lambda: SimpleNamespace(set_device=lambda device: None))
    monkeypatch.setattr(executor, "get_device_str", lambda: "cpu")
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL_PATH)
    return executor, sys.modules["helper"]


@pytest.mark.parametrize("fail_second", [False, True])
def test_real_executor_finalizes_partial_data_and_resumes(
    monkeypatch, tmp_path, checkpoint, inventory, cpu_executor, fail_second
):
    executor, helper = cpu_executor
    monkeypatch.setattr(collect, "collect_inventory", lambda **kwargs: inventory)
    module_name = "collector.sglang_rubin.collect_gemm"
    module = ModuleType(module_name)
    module.__compat__ = f"sglang=={SGLANG_DISTRIBUTION_VERSION}"
    module.get_gemm_test_cases = lambda: [["bfloat16", 1, 16, 16], ["bfloat16", 2, 16, 16]]
    executed = []

    def run_gemm(dtype, m, n, k, *, device, perf_filename):
        executed.append(m)
        if m == 2 and fail_second:
            raise RuntimeError("deliberate fake-kernel failure")
        assert helper.log_perf(
            [{"gemm_dtype": dtype, "m": m, "n": n, "k": k, "latency": 0.125}],
            "sglang",
            SGLANG_DISTRIBUTION_VERSION,
            "cpu-test",
            "gemm",
            "fake-kernel",
            perf_filename,
        )

    module.run_gemm = run_gemm
    monkeypatch.setitem(sys.modules, module_name, module)
    output = tmp_path / "output"
    arguments = _arguments(checkpoint, output)
    assert collect.main(arguments) == int(fail_second)
    assert executed == [1, 2]
    parquet_path = output / "gemm_perf.parquet"
    assert pq.read_metadata(parquet_path).num_rows == 2 - int(fail_second)
    metadata = yaml.safe_load((output / "collection_meta.yaml").read_text())
    assert metadata["runtime"]["version"] == SGLANG_DISTRIBUTION_VERSION
    assert metadata["runtime"]["source_commit"] == SGLANG_COMMIT
    assert metadata["runtime"]["image_digest"] == runtime.IMAGE_INDEX_DIGEST
    assert metadata["tables"]["gemm_perf"]["collector_hash"].startswith("sha256:")
    pilot_identity = json.loads((output / "pilot_identity.json").read_text())
    assert pilot_identity["declared_serving_configuration"] == inventory["declared_serving_configuration"]
    assert pilot_identity["observed_serving_environment"] == inventory["observed"]["serving_environment"]
    assert not (output / "gemm_perf.txt").exists()
    checkpoint_path = output / ".collector_checkpoint/sglang/sglang.gemm.json"
    recorded = json.loads(checkpoint_path.read_text())
    assert recorded["attempted"] == []
    assert len(recorded["done"]) == 2 - int(fail_second)
    assert len(recorded["failed"]) == int(fail_second)
    assert collect.main([*arguments, "--resume"]) == int(fail_second)
    assert executed == [1, 2]
    assert pq.read_metadata(parquet_path).num_rows == 2 - int(fail_second)
    assert len(list(output.glob("inventory-*.json"))) == 2
    assert sys.modules["helper"] is sys.modules["collector.helper"]
    identity = {key: recorded[key] for key in executor._CHECKPOINT_IDENTITY_FIELDS}
    assert executor._registered_checkpoint_table(identity, backend="sglang") == "gemm_perf"
    assert collect.main([*arguments, "--resume", "--case-filter", "['bfloat16', 1, 16, 16]"]) == 1
    assert executed == [1, 2]
    assert pq.read_metadata(parquet_path).num_rows == 2 - int(fail_second)


def test_empty_requested_collection_is_a_failure(monkeypatch, tmp_path, checkpoint, inventory, cpu_executor):
    monkeypatch.setattr(collect, "collect_inventory", lambda **kwargs: inventory)
    module = ModuleType("collector.sglang_rubin.collect_gemm")
    module.get_gemm_test_cases = lambda: []

    def run_gemm(*args, **kwargs):
        pytest.fail("empty population should never run a kernel")

    module.run_gemm = run_gemm
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "output"
    assert collect.main(_arguments(checkpoint, output)) == 1
    summary = json.loads((output / "test-logs/collection_summary_sglang.json").read_text())
    assert {error["error_type"] for error in summary["errors"]} == {"EmptyCollection", "MissingPerfOutput"}


@pytest.fixture
def dsa_executor(monkeypatch, inventory, cpu_executor):
    from collector.sglang_rubin import collect_mla_module as module

    monkeypatch.setattr(collect, "collect_inventory", lambda **kwargs: inventory)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda _: None, get_device_name=lambda _: "VR200"),
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setattr(module, "_validate_runtime", lambda: None)
    monkeypatch.setattr(module, "get_version", lambda _: SGLANG_DISTRIBUTION_VERSION)
    monkeypatch.setattr(module, "_dsa_context_derived_shapes", lambda _: [(4096, 128, 1)])
    monkeypatch.setattr(module, "_dsa_generation_derived_shapes", lambda _: [(4096, 1, 1)])
    monkeypatch.setattr(module, "load_model_runner", lambda *_, **__: object())
    monkeypatch.setattr(module, "_prepare_batch", lambda *_, **__: object())
    executed = []

    def measure_module(*_, is_prefill, skip_indexer, **__):
        executed.append((is_prefill, skip_indexer))
        return {"latency_ms": 0.125 if skip_indexer else 0.25, "power_stats": None}, "sglang_dsa_test_trtllm"

    monkeypatch.setattr(module, "_measure_module", measure_module)
    monkeypatch.setattr(module, "_run_subprocess", lambda name, arguments, device: module.run_mla_module(**arguments))
    return module, executed


def _dsa_arguments(checkpoint, output, ops):
    arguments = _arguments(checkpoint, output)
    arguments[arguments.index("gemm") : arguments.index("gemm") + 1] = ops
    arguments.extend(["--limit", "1"])
    return arguments


@pytest.mark.parametrize("variants", [(False,), (True,), (False, True)], ids=["full", "skip", "both"])
def test_real_executor_publishes_requested_dsa_variants_and_resumes(tmp_path, checkpoint, dsa_executor, variants):
    from collector.provenance import case_plan_hash

    _module, executed = dsa_executor
    ops = [
        f"dsa_{phase}_module{'_skip_indexer' if skip else ''}"
        for skip in variants
        for phase in ("context", "generation")
    ]
    output = tmp_path / "output"
    arguments = _dsa_arguments(checkpoint, output, ops)
    assert collect.main(arguments) == 0
    assert executed == [(prefill, skip) for skip in variants for prefill in (True, False)]
    assert {path.name for path in output.glob("*_perf.parquet")} == {
        "dsa_context_module_perf.parquet",
        "dsa_generation_module_perf.parquet",
    }
    for phase in ("context", "generation"):
        rows = pq.read_table(output / f"dsa_{phase}_module_perf.parquet").to_pylist()
        assert sorted((row["op_name"], row["latency"]) for row in rows) == [
            (f"dsa_{phase}_module{'_skip_indexer' if skip else ''}", 0.125 if skip else 0.25) for skip in variants
        ]
    done = {}
    for op in ops:
        record = json.loads((output / f".collector_checkpoint/sglang/sglang.{op}.json").read_text())
        assert len(record["done"]) == 1
        assert not record["failed"]
        assert not record["attempted"]
        done[op] = record["done"]
    metadata = yaml.safe_load((output / "collection_meta.yaml").read_text())
    for phase in ("context", "generation"):
        table = metadata["tables"][f"dsa_{phase}_module_perf"]
        assert table["status"] == "complete"
        assert table["case_plan_hash"] == case_plan_hash([case for op in ops if phase in op for case in done[op]])
    before_resume = (output / "collection_meta.yaml").read_bytes()
    assert collect.main([*arguments, "--resume"]) == 0
    assert len(executed) == len(ops)
    assert (output / "collection_meta.yaml").read_bytes() == before_resume


@pytest.mark.parametrize("missing_skip", [False, True], ids=["missing-full", "missing-skip"])
def test_real_executor_rejects_missing_requested_dsa_labels(
    monkeypatch, tmp_path, checkpoint, dsa_executor, missing_skip
):
    module, _executed = dsa_executor
    # Simulate a successful worker dispatching the wrong variant. Both
    # checkpoints and canonical files exist, but a requested op has no rows.
    monkeypatch.setattr(
        module,
        "_run_subprocess",
        lambda name, arguments, device: module.run_mla_module(**{**arguments, "skip_indexer": not missing_skip}),
    )
    ops = [entry.op for entry in REGISTRY if entry.op.startswith("dsa_")]
    output = tmp_path / "output"
    arguments = _dsa_arguments(checkpoint, output, ops)
    assert collect.main(arguments) == 1
    summary = json.loads((output / "test-logs/collection_summary_sglang.json").read_text())
    assert {(error["module"], error["error_type"]) for error in summary["errors"]} == {
        (f"sglang.dsa_{phase}_module{'_skip_indexer' if missing_skip else ''}", "MissingPerfOutput")
        for phase in ("context", "generation")
    }
    for op in ops:
        record = json.loads((output / f".collector_checkpoint/sglang/sglang.{op}.json").read_text())
        assert len(record["done"]) == 1
        assert not record["failed"]
        assert not record["attempted"]
    assert collect.main([*arguments, "--resume"]) == 1


@pytest.mark.parametrize("failing_skip", [False, True], ids=["full-failure", "skip-failure"])
def test_real_executor_records_dsa_producer_failure_and_resumes_retry(
    monkeypatch, tmp_path, checkpoint, dsa_executor, failing_skip
):
    from collector.provenance import case_plan_hash

    module, executed = dsa_executor
    measure = module._measure_module
    failed_calls = []

    def fail_variant(*args, skip_indexer, **kwargs):
        if skip_indexer == failing_skip:
            failed_calls.append(skip_indexer)
            raise RuntimeError("deliberate DSA producer failure")
        return measure(*args, skip_indexer=skip_indexer, **kwargs)

    monkeypatch.setattr(module, "_measure_module", fail_variant)
    full = "dsa_context_module"
    skip = f"{full}_skip_indexer"
    ops = [full, skip]
    output = tmp_path / "output"
    arguments = _dsa_arguments(checkpoint, output, ops)
    assert collect.main(arguments) == 1
    assert len(failed_calls) == len(executed) == 1
    rows = pq.read_table(output / f"{full}_perf.parquet").to_pylist()
    assert [row["op_name"] for row in rows] == [full if failing_skip else skip]
    attempted = []
    for op in ops:
        record = json.loads((output / f".collector_checkpoint/sglang/sglang.{op}.json").read_text())
        failed = (op == skip) == failing_skip
        assert len(record["done"]) == int(not failed)
        assert len(record["failed"]) == int(failed)
        assert not record["attempted"]
        attempted.extend(record["done"] + record["failed"])
    metadata = yaml.safe_load((output / "collection_meta.yaml").read_text())
    assert metadata["tables"][f"{full}_perf"]["case_plan_hash"] == case_plan_hash(attempted)
    summary = json.loads((output / "test-logs/collection_summary_sglang.json").read_text())
    assert any(error["error_type"] == "RuntimeError" for error in summary["errors"])
    assert any(error["error_type"] == "MissingPerfOutput" for error in summary["errors"])
    before_resume = (output / "collection_meta.yaml").read_bytes()
    assert collect.main([*arguments, "--resume"]) == 1
    assert len(failed_calls) == len(executed) == 1
    assert (output / "collection_meta.yaml").read_bytes() == before_resume
    completed_op = full if failing_skip else skip
    completed_checkpoint = output / f".collector_checkpoint/sglang/sglang.{completed_op}.json"
    completed_before = completed_checkpoint.read_bytes()
    monkeypatch.setattr(module, "_measure_module", measure)
    assert collect.main([*arguments, "--resume", "--resume-retry-failed"]) == 0
    assert len(executed) == 2
    assert sorted(row["op_name"] for row in pq.read_table(output / f"{full}_perf.parquet").to_pylist()) == ops
    assert completed_checkpoint.read_bytes() == completed_before
    for op in ops:
        record = json.loads((output / f".collector_checkpoint/sglang/sglang.{op}.json").read_text())
        assert len(record["done"]) == 1
        assert not record["failed"]
        assert not record["attempted"]
    metadata = yaml.safe_load((output / "collection_meta.yaml").read_text())
    events = metadata["tables"][f"{full}_perf"]["collections"]
    assert len(events) == 2
    assert events[0]["case_plan_hash"] == case_plan_hash(attempted)
    failed_op = skip if failing_skip else full
    retried = json.loads((output / f".collector_checkpoint/sglang/sglang.{failed_op}.json").read_text())
    assert events[1]["case_plan_hash"] == case_plan_hash(retried["done"])


@pytest.mark.parametrize("directory", ["", "relative/nested", "/absolute/nested"])
@pytest.mark.parametrize("worker_filename", [None, "dsa_context_module_skip_indexer_perf.txt"])
def test_executor_worker_filename_preserves_output_directory(monkeypatch, cpu_executor, directory, worker_filename):
    executor, _helper = cpu_executor
    module = ModuleType("dsa_binding_test")
    received = []
    module.get_cases = lambda: []
    module.run = lambda *, perf_filename: received.append(perf_filename)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def collect_module_safe(name, op, get_func, run_func, processes, **kwargs):
        run_func()
        return []

    monkeypatch.setattr(executor, "collect_module_safe", collect_module_safe)
    physical_filename = str(Path(directory) / "dsa_context_module_perf.txt")
    collection = {
        "name": "sglang",
        "type": "dsa_context_module_skip_indexer",
        "module": module.__name__,
        "get_func": "get_cases",
        "run_func": "run",
        "perf_filename": physical_filename,
    }
    if worker_filename is not None:
        collection["worker_perf_filename"] = worker_filename
    assert executor.collect_ops(0, [collection]) == []
    assert received == [str(Path(directory) / worker_filename) if worker_filename else physical_filename]
    assert collection["perf_filename"] == physical_filename


def test_existing_output_and_identity_change_are_rejected(monkeypatch, tmp_path, checkpoint, inventory):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "existing.parquet"
    sentinel.write_bytes(b"existing")
    monkeypatch.setattr(collect, "collect_inventory", lambda **kwargs: pytest.fail("existing output was accepted"))
    assert collect.main(_arguments(checkpoint, output)) == 1
    assert list(output.iterdir()) == [sentinel]
    sentinel.unlink()
    identity_path = output / "pilot_identity.json"
    identity_path.write_text(json.dumps({"schema": "different-runtime"}))
    monkeypatch.setattr(collect, "collect_inventory", lambda **kwargs: inventory)
    assert collect.main([*_arguments(checkpoint, output), "--resume"]) == 1
    assert json.loads(identity_path.read_text()) == {"schema": "different-runtime"}


@pytest.mark.parametrize("changed_setting", ["declared", "observed"])
def test_resume_rejects_changed_serving_configuration(monkeypatch, tmp_path, checkpoint, inventory, changed_setting):
    arguments = _arguments(checkpoint, tmp_path / "output")
    _plan, document = collect._build_plan(collect._parse_args(arguments))
    previous = collect._dataset_identity(inventory, document)
    if changed_setting == "declared":
        previous["declared_serving_configuration"]["server_args"]["disable_prefill_cuda_graph"] = False
    else:
        previous["observed_serving_environment"]["SGLANG_ENABLE_MOE_DEFERRED_FINALIZE"] = "1"
        # The stored prior run differs; the current runtime has the required value.
        inventory["observed"]["serving_environment"] = dict(runtime.REQUIRED_SERVING_ENV)
    output = tmp_path / "output"
    output.mkdir()
    identity_path = output / "pilot_identity.json"
    identity_path.write_text(json.dumps(previous))
    monkeypatch.setattr(collect, "collect_inventory", lambda **kwargs: inventory)
    monkeypatch.setattr(collect, "_load_executor", SimpleNamespace)

    assert collect.main([*arguments, "--resume"]) == 1
    assert json.loads(identity_path.read_text()) == previous
    assert not (output / ".collector_checkpoint").exists()


def test_pilot_provenance_closure_includes_declared_serving_configuration():
    from collector.provenance import load_closures

    closures = load_closures(collect._PACKAGE_ROOT / "collector/hash_closures.yaml")
    for module in {entry.module for entry in REGISTRY}:
        assert "collector/sglang_rubin/runtime.py" in closures[module]
