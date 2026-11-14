# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A serving-probe campaign end to end on the collection pipeline: emit (render, bundle,
spec, shard plan) -> the job's probe.serving shards -> --import-raws (only this emit's
fingerprint) -> records labeled with the SM the raw ran on."""

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
    monkeypatch.setattr(mod, "render_golden", lambda run, cdir: golden)
    monkeypatch.setattr(mod, "extract_sglang_cli_from_run_sh", lambda p: "--attention-backend fa3")
    monkeypatch.setattr(mod, "_generator_src_commit", lambda: {"generator_commit": "abc"})
    dummy = tmp_path / "ws" / "dummy_models" / "m__rep"
    dummy.mkdir(parents=True)
    (dummy / "config.json").write_text("{}")
    return mod


def _run(pd, backend, tp=1, system="b200_sxm"):
    return {
        "id": f"{backend}_tp{tp}",
        "backend": backend,
        "tp": tp,
        "repo": "org/m",
        "variant": "rep",
        "version": "1",
        "system": system,
        "sm": pd.system_sm(system),
        "image": f"img/{backend}",
        "kv_dtype": None,
        "model_dir": f"{pd.WORK}/dummy_models/m__rep",
    }


@pytest.mark.parametrize(
    "backend, tp, script, expect",
    [
        ("vllm", 1, "probe_vllm.py", "--run-sh"),
        ("sglang", 1, "probe_sglang.py", "--trace"),
        ("sglang", 2, "probe_sglang_server.py", "--tp"),
        ("trtllm", 1, "probe_trtllm.py", "--engine-yaml"),
        ("trtllm", 2, "probe_trtllm_server.py", "--tp"),
    ],
)
def test_emit_writes_a_spec_per_run_with_bundle_and_out_paths(pd, backend, tp, script, expect):
    cdir = pd.campaign_dir("c")
    pd.emit_shards([_run(pd, backend, tp)], cdir, "http://share/c", upload=False)
    (shard,) = yaml.safe_load((cdir / "shards" / f"tp{tp}" / f"{backend}-1.yaml").read_text())["shards"]
    rid = f"{backend}_tp{tp}"
    assert shard["op"] == "probe.serving" and shard["case_filter"] == f"spec=http://share/c/{rid}.json"
    spec = json.loads((cdir / "bundles" / f"{rid}.json").read_text())
    assert spec["script"] == script and expect in spec["args"]
    assert "{BUNDLE}/dummy_models/m__rep" in spec["args"] and f"{{OUT}}/raw_{rid}.json" in spec["args"]
    assert (spec["env"] == {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}) == (backend == "trtllm" and tp == 1)
    with tarfile.open(cdir / "bundles" / f"{rid}.tgz") as tar:
        names = set(tar.getnames())
    assert "dummy_models/m__rep/config.json" in names
    assert all(not a.startswith("/") for a in names)
    plan = json.loads((cdir / "plan.json").read_text())
    assert plan[0]["exec_fingerprint"] == spec["fingerprint"]


def _job_shard(job: Path, rid: str, fingerprint: str, raw: dict | None, tail: str = "") -> None:
    shard = job / f"probe.serving__{rid}"
    shard.mkdir(parents=True)
    result = {"run_id": rid, "fingerprint": fingerprint, "raw": f"raw_{rid}.json" if raw else None, "tail": tail}
    (shard / "probe_serving.json").write_text(json.dumps({"probe": "serving", "result": result}))
    if raw:
        (shard / f"raw_{rid}.json").write_text(json.dumps(raw))


RAW_SM100 = {
    "errors": {},
    "device_capability": "sm100",
    "probe_isl": 4096,
    "decode_kernels": [
        {"kernel": "fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvDenseP32VarSeqQ8Kv128StaticSwapsAbForGen", "us": 3}
    ],
}


def test_import_files_only_this_emit_and_records_label_by_the_raw_sm(pd, tmp_path, capsys):
    cdir = pd.campaign_dir("c")
    runs = [
        _run(pd, "vllm"),
        dict(_run(pd, "vllm"), id="vllm_fp8", kv_dtype="fp8"),
        dict(_run(pd, "vllm"), id="dead", repo="org/d"),
    ]
    pd.emit_shards(runs, cdir, "http://share/c", upload=False)
    fps = {r["id"]: r["exec_fingerprint"] for r in json.loads((cdir / "plan.json").read_text())}
    job = tmp_path / "job"
    _job_shard(job, "vllm_tp1", fps["vllm_tp1"], RAW_SM100)
    _job_shard(job, "vllm_fp8", "0" * 12, RAW_SM100)  # ran another emit of this run: not filed
    _job_shard(job, "dead", fps["dead"], None, tail="loading\nRuntimeError: CUDA error: no kernel image\n")
    pd.import_raws(job, cdir)
    assert (cdir / "raw" / "vllm_tp1.json").exists() and not (cdir / "raw" / "vllm_fp8.json").exists()
    assert "no kernel image" in (cdir / "raw" / "dead.tail").read_text()

    pd.build_records(cdir)
    out = capsys.readouterr().out
    (rec,) = pd.load_records(cdir).values()
    assert rec["runtime"]["sm"] == "sm100" and rec["runtime"]["system"] == "b200_sxm"
    # the Blackwell FMHA cubin is labeled by the sm100 rules (the sm90 rules leave it unlabeled)
    assert "trtllm_fmha" in pd.label_kernels(rec["orphan_kernels"], "sm100")[0]
    assert pd.label_kernels(rec["orphan_kernels"], "sm90")[0] == set()
    # every planned run says whether it is serving evidence; a probe that died before its raw
    # shows its own last line
    assert "vllm_tp1 vllm 1 org/m kv=auto: ok" in out
    assert "vllm_fp8 vllm 1 org/m kv=fp8: not run" in out
    assert "dead vllm 1 org/d kv=auto: no raw: RuntimeError: CUDA error: no kernel image" in out


def test_a_campaign_prompt_length_reaches_every_probe_and_the_run_id(pd):
    cdir = pd.campaign_dir("c")
    run = dict(_run(pd, "sglang", 2), isl=512)
    pd.emit_shards([run], cdir, "http://share/c", upload=False)
    spec = json.loads((cdir / "bundles" / "sglang_tp2.json").read_text())
    assert spec["args"][-2:] == ["--isl", "512"]
    ck = {"repo": "org/m", "profile": "fp8"}
    assert pd._run_id(ck, "v", "vllm", "1", 1, None, "h200_sxm", 512) != pd._run_id(
        ck, "v", "vllm", "1", 1, None, "h200_sxm"
    )


def test_reemitting_a_campaign_starts_it_over(pd, tmp_path):
    cdir = pd.campaign_dir("c")
    pd.emit_shards([_run(pd, "vllm")], cdir, "http://share/c", upload=False)
    (cdir / "raw").mkdir()
    (cdir / "raw" / "vllm_tp1.json").write_text("{}")
    pd.emit_shards([_run(pd, "vllm")], cdir, "http://share/c", upload=False)
    assert not (cdir / "raw").exists()


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


def test_a_gate_campaign_probes_what_each_gate_reads(pd, monkeypatch, capsys):
    """--gates: per gate, its repo at its kv class, prompt length and release. vLLM gemm is
    pinned to 0.31.0 apart from the 0.30.0 default; the DSA s512 gate reads a 512-token record;
    a no_serving floor needs no run; a repo the roster lacks is reported."""
    sys.path.insert(0, str(COMPONENTS))
    import gates

    def gate(name, op, repo, kv, **extra):
        return {"gate": name, "backend": "vllm", "op": op, "repo": repo, "kv": kv, "select": {}, **extra}

    rows = [
        gate("dsa_s512", "dsa_context_module", "org/m", "fp8", isl=512),
        gate("gemm", "gemm", "org/m", "auto"),
        gate("floor", "gemm", "org/m", "fp8", no_serving={"sm100": "cannot serve"}),
        gate("lost", "gemm", "org/x", "auto"),
    ]
    monkeypatch.setattr(gates, "gates_in_scope", lambda fw, sm, op=None, derive=False: rows)
    monkeypatch.setattr(gates, "gate_version", lambda g: "0.31.0" if g["op"] == "gemm" else "0.30.0")

    def roster(targets, system, full, backends, only, isl, version):
        return [
            dict(_run(pd, "vllm"), id=f"{kv}-{isl}-{version}", repo="org/m", kv_dtype=kv, isl=isl, version=version)
            for kv in (None, "fp8")
        ]

    monkeypatch.setattr(pd, "enumerate_runs", roster)
    runs = pd.gate_runs({}, "b200_sxm", ["vllm"])
    assert sorted(r["id"] for r in runs) == ["None-4096-0.31.0", "fp8-512-0.30.0"]
    assert "org/x kv=unquant (lost)" in capsys.readouterr().err
