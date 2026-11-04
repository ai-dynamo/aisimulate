# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One probe invocation per run, two routes: the per-GPU docker queue on a probe box and
the probe.serving shard on the collection pipeline. Both run the same in-container probe
script with the same argv; the pipeline spec replaces the workspace mount with the
unpacked bundle and the raw path with the shard dir."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture()
def pd(tmp_path, monkeypatch):
    monkeypatch.setenv("AIS_PROBE_WORKSPACE", str(tmp_path / "ws"))
    spec = importlib.util.spec_from_file_location("probe_driver_routes", COMPONENTS / "probe_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe_driver_routes"] = mod
    spec.loader.exec_module(mod)
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "run.sh").write_text("engine_command=(vllm serve /m --tp 1)\n")
    (golden / "run_0.sh").write_text("python -m sglang.launch_server --model /m --tp 1\n")
    (golden / "agg_config.yaml").write_text("kv_cache_config: {}\n")
    monkeypatch.setattr(mod, "render_golden", lambda run: golden)
    monkeypatch.setattr(mod, "extract_sglang_cli_from_run_sh", lambda p: "--attention-backend fa3")
    monkeypatch.setattr(mod, "_generator_src_commit", lambda: {"generator_commit": "abc"})
    monkeypatch.setattr(mod, "SCRATCH_QUEUES", tmp_path / "queues")
    dummy = tmp_path / "ws" / "dummy_models" / "m__rep"
    dummy.mkdir(parents=True)
    (dummy / "config.json").write_text("{}")
    return mod


def _run(pd, backend, tp=1):
    return {
        "id": f"{backend}_tp{tp}",
        "backend": backend,
        "tp": tp,
        "repo": "org/m",
        "variant": "rep",
        "version": "1",
        "image": f"img/{backend}",
        "kv_dtype": "fp8",
        "model_dir": f"{pd.WORK}/dummy_models/m__rep",
    }


@pytest.mark.parametrize(
    "backend, tp, script, expect",
    [
        ("vllm", 1, "probe_vllm.py", "--entrypoint python3 img/vllm /harness"),
        ("sglang", 1, "probe_sglang.py", "img/sglang python3 /harness"),
        ("sglang", 2, "probe_sglang_server.py", "--tp 2 --kv-dtype fp8"),
        ("trtllm", 1, "probe_trtllm.py", "-e TLLM_WORKER_USE_SINGLE_PROCESS=1"),
        ("trtllm", 2, "probe_trtllm_server.py", "--model /work/dummy_models/m__rep --tp 2 --trust-remote-code"),
    ],
)
def test_the_docker_queue_runs_the_prepared_probe(pd, monkeypatch, backend, tp, script, expect):
    monkeypatch.setattr(pd, "PROBES_IN_CONTAINER", "/harness/probes")
    runs = [_run(pd, backend, tp)]
    pd.emit_queues(runs, [0, 1], "plan.json")
    cmd = (pd.SCRATCH_QUEUES / "gpu0.sh").read_text()
    assert f"/harness/probes/{script}" in cmd and expect in cmd
    assert f"--out {pd.WORK}/archive/raw/{runs[0]['id']}.json" in cmd
    assert runs[0]["probe_script"] == script and runs[0]["exec_fingerprint"]


def test_emit_shards_bundles_the_inputs_and_rewrites_the_paths(pd, tmp_path):
    runs = [_run(pd, "vllm"), _run(pd, "trtllm", tp=2)]
    pd.emit_shards(runs, "plan.json", tmp_path / "shards", "http://share/plans/x", upload=False)
    plan = yaml.safe_load((tmp_path / "shards" / "tp1" / "vllm.yaml").read_text())
    (shard,) = plan["shards"]
    assert shard["op"] == "probe.serving" and shard["case_filter"] == "spec=http://share/plans/x/vllm_tp1.json"
    assert yaml.safe_load((tmp_path / "shards" / "tp2" / "trtllm.yaml").read_text())["shards"]
    spec = json.loads((tmp_path / "shards" / "bundles" / "vllm_tp1.json").read_text())
    assert spec["script"] == "probe_vllm.py" and spec["bundle"] == "http://share/plans/x/vllm_tp1.tgz"
    assert "{BUNDLE}/dummy_models/m__rep" in spec["args"] and "{OUT}/raw_vllm_tp1.json" in spec["args"]
    assert not any(a.startswith(pd.WORK) for a in spec["args"])
    with tarfile.open(tmp_path / "shards" / "bundles" / "vllm_tp1.tgz") as tar:
        assert {"dummy_models/m__rep/config.json", "archive/run_sh/vllm_tp1.sh"} <= set(tar.getnames())
    saved = {r["id"]: r for r in json.loads((pd.ROOT / "archive" / "plan.json").read_text())}
    assert saved["vllm_tp1"]["exec_fingerprint"]["fingerprint"] == spec["fingerprint"]


def test_import_raws_files_the_job_raws_with_their_fingerprint(pd, tmp_path):
    runs = [_run(pd, "vllm")]
    pd.emit_shards(runs, "plan.json", tmp_path / "shards", "http://share/x", upload=False)
    job = tmp_path / "job" / "probe.serving__vllm_tp1"
    job.mkdir(parents=True)
    (job / "raw_vllm_tp1.json").write_text('{"errors": {}}')
    pd.import_raws(tmp_path / "job", "plan.json")
    assert (pd.ROOT / "archive" / "raw" / "vllm_tp1.json").read_text() == '{"errors": {}}'
    assert (pd.ROOT / "archive" / "raw" / "vllm_tp1.fp").read_text().strip() == runs[0]["exec_fingerprint"][
        "fingerprint"
    ]


def test_the_serving_probe_runs_the_spec_script_on_the_unpacked_bundle(monkeypatch, tmp_path):
    sys.path.insert(0, str(COMPONENTS.parents[2]))
    from collector.probes import serving

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"{}"
        info = tarfile.TarInfo("dummy_models/m__rep/config.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    spec = {"run_id": "r1", "backend": "vllm", "script": "probe_vllm.py", "env": {"X": "1"}, "bundle": "u/b.tgz"}
    spec["args"] = ["--model-override", "{BUNDLE}/dummy_models/m__rep", "--out", "{OUT}/raw_r1.json"]
    blobs = {"u/spec.json": json.dumps(spec).encode(), "u/b.tgz": buf.getvalue()}
    monkeypatch.setattr(serving, "_fetch", lambda url: blobs[url])
    seen = {}

    def fake_run(cmd, env, capture_output, text, timeout):
        seen.update(cmd=cmd, x=env.get("X"), offline=env.get("HF_HUB_OFFLINE"))
        model = Path(cmd[cmd.index("--model-override") + 1])
        assert (model / "config.json").read_text() == "{}"
        Path(cmd[cmd.index("--out") + 1]).write_text("{}")
        return types.SimpleNamespace(returncode=0, stdout="WROTE", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = serving.run({"spec": "u/spec.json"}, tmp_path)
    assert out["raw"] == "raw_r1.json" and out["failed"] is False
    assert seen["cmd"][1].endswith("opharness/components/probes/probe_vllm.py") and seen["x"] == "1"
    assert seen["offline"] == "1" and os.path.exists(tmp_path / "raw_r1.json")
