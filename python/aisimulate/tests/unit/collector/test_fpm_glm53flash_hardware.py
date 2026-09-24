# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.fpm_forward.hybrid_artifact import validate_vllm_hardware_receipts
from collector.fpm_forward.runtime.glm53flash import glm53flash_worker_hardware as hardware

pytestmark = pytest.mark.unit


def native_worker(monkeypatch, tmp_path, *, name="NVIDIA GB300", rank=0):
    source = tmp_path / "gpu_worker.py"
    source.write_text("# test-only native worker source\n")
    output = tmp_path / "benchmark.json"
    monkeypatch.setenv("DYN_FPM_BENCHMARK_OUTPUT_PATH", str(output))
    (tmp_path / "collector-provenance.json").write_text('{"attempt_id":"test-attempt"}\n')
    monkeypatch.setattr(hardware.importlib.metadata, "version", lambda _: "0.30.0")
    ready = []

    class Worker:
        def init_device(self, value):
            ready.append(value)
            self.device = f"cuda:{rank}"
            return "native-return"

    def current_device():
        assert ready, "device properties must be read after native initialization"
        return rank

    cuda = SimpleNamespace(
        current_device=current_device,
        get_device_name=lambda _: name,
        get_device_capability=lambda _: (10, 3),
        get_device_properties=lambda _: SimpleNamespace(total_memory=288 * 1024**3, uuid=f"GPU-{rank}"),
    )
    module = SimpleNamespace(Worker=Worker, torch=SimpleNamespace(cuda=cuda), __file__=str(source))
    hardware.install(module)
    hardware.install(module)  # One callback, even when imported again.
    worker = Worker()
    worker.rank = rank
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1)
    return worker, ready


def test_native_initializer_return_and_selected_device_are_preserved(monkeypatch, tmp_path):
    worker, calls = native_worker(monkeypatch, tmp_path)
    assert worker.init_device("native-input") == "native-return"
    assert calls == ["native-input"]
    receipt = json.loads((tmp_path / "native-device-rank-0.json").read_text())
    assert receipt["hardware"]["uuid"] == "GPU-0"
    assert receipt["status"] == "passed"
    assert (
        receipt["collector_provenance_sha256"]
        == hashlib.sha256((tmp_path / "collector-provenance.json").read_bytes()).hexdigest()
    )
    assert not list(tmp_path.glob("benchmark*.json")), "device receipts are not benchmark envelopes"
    with pytest.raises(FileExistsError):
        worker.init_device("repeat-attempt")


def test_rejected_native_device_preserves_failure_receipt(monkeypatch, tmp_path):
    worker, _ = native_worker(monkeypatch, tmp_path, name="NVIDIA B300")
    with pytest.raises(ValueError, match="GB300"):
        worker.init_device("native-input")
    receipt = json.loads((tmp_path / "native-device-rank-0.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["hardware"]["name"] == "NVIDIA B300"


def artifact(tmp_path):
    from collector.fpm_forward import hybrid_artifact

    source_path = Path(hybrid_artifact.__file__).parent / "runtime/glm53flash/runtime-source-sha256.json"
    source_manifest = source_path.read_bytes()
    source_pin = json.loads(source_manifest)["vllm/v1/worker/gpu_worker.py"]
    provenance = tmp_path / "collector-provenance.json"
    provenance.write_text('{"attempt_id":"test-attempt"}\n')
    entries = []
    for rank in range(2):
        receipt = {
            "schema_version": 1,
            "status": "passed",
            "backend": "vllm",
            "backend_version": "0.30.0",
            "collector_provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
            "tp_rank": rank,
            "tp_size": 2,
            "worker_source_sha256": source_pin,
            "hardware": {
                "schema": "glm53flash_gpu_identity_v1",
                "name": "NVIDIA GB300",
                "compute_capability": [10, 3],
                "total_memory_bytes": 288 * 1024**3,
                "cuda_device_index": rank,
                "uuid": f"GPU-{rank}",
            },
        }
        path = tmp_path / f"native-device-rank-{rank}.json"
        path.write_text(json.dumps(receipt))
        entries.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tp_rank": rank})
    payload = {
        "producer": {
            "hardware_contract_version": 1,
            "vllm_package_version": "0.30.0",
            "runtime_source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        },
        "input_provenance": {"native_hardware_manifest": entries},
    }
    return SimpleNamespace(topology=SimpleNamespace(tp=2)), payload, tmp_path / "benchmark.json"


def test_complete_actual_hardware_is_admitted(tmp_path):
    validate_vllm_hardware_receipts(*artifact(tmp_path))


@pytest.mark.parametrize(
    "corruption", ["missing", "coverage", "digest", "rank", "bool_rank", "source", "attempt", "version", "gpu", "uuid"]
)
def test_hardware_gate_rejects_rehashed_wrong_runtime_or_worker(tmp_path, corruption):
    cell, payload, path = artifact(tmp_path)
    entries = payload["input_provenance"]["native_hardware_manifest"]
    entry = entries[1]
    receipt_path = path.with_name(entry["file"])
    receipt = json.loads(receipt_path.read_text())
    if corruption == "missing":
        payload["producer"].pop("hardware_contract_version")
    elif corruption == "coverage":
        entries.pop()
    elif corruption == "digest":
        entry["sha256"] = "0" * 64
    elif corruption == "rank":
        receipt["tp_rank"] = 0
    elif corruption == "bool_rank":
        receipt["tp_rank"] = True
    elif corruption == "source":
        receipt["worker_source_sha256"] = "0" * 64
    elif corruption == "attempt":
        receipt["collector_provenance_sha256"] = "0" * 64
    elif corruption == "version":
        receipt["backend_version"] = "0.30.0+other"
    elif corruption == "gpu":
        receipt["hardware"]["name"] = "NVIDIA B300"
    else:
        receipt["hardware"]["uuid"] = "GPU-0"
    receipt_path.write_text(json.dumps(receipt))
    if corruption != "digest":
        entry["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="hardware|GPU"):
        validate_vllm_hardware_receipts(cell, payload, path)
