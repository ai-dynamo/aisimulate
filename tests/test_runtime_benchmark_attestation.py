# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_migration_runtime", Path(__file__).resolve().parents[1] / "scripts/benchmark_migration_runtime.py"
)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


@pytest.mark.parametrize("tracked", [False, True])
def test_attestation_rejects_changed_source_before_import(tmp_path, tracked):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "src"
    source.mkdir()
    module = source / "shadow.py"
    module.write_text("value = 1\n")
    if tracked:
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Benchmark Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        module.write_text("value = 2\n")
    variant = {
        "label": "fixture",
        "kind": "aisimulate",
        "root": str(tmp_path),
        "sources": [str(source)],
        "python": "/must-not-run-python",
    }
    with pytest.raises(RuntimeError, match="uncommitted source changes"):
        BENCHMARK.attest(variant, {})


def test_attestation_rejects_an_extension_from_another_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    other = tmp_path / "other/_runtime.abi3.so"
    other.parent.mkdir()
    other.write_bytes(b"different build")

    def output(command, **kwargs):
        del kwargs
        if "status" in command:
            return ""
        if "-c" in command:
            return json.dumps({"native": str(other), "python": "test", "dependencies": {}})
        raise AssertionError(command)

    monkeypatch.setattr(BENCHMARK.subprocess, "check_output", output)
    variant = {"label": "fixture", "kind": "aisimulate", "root": str(root), "sources": [str(root)], "python": "python"}
    with pytest.raises(ValueError):
        BENCHMARK.attest(variant, {})


@pytest.mark.parametrize("failure", [None, "source", "prediction"])
def test_controller_publishes_only_after_integrity_checks(tmp_path, monkeypatch, failure):
    import hashlib
    from types import SimpleNamespace

    source = tmp_path / "source.py"
    source.write_text("original")
    native = tmp_path / "_runtime.so"
    native.write_bytes(b"fixture")
    variant = {
        "label": "fixture",
        "kind": "aiconfigurator",
        "root": str(tmp_path),
        "sources": [str(tmp_path)],
        "python": "fixture-python",
    }
    variants = tmp_path / "variants.json"
    variants.write_text(json.dumps([variant]))
    worker_calls = []

    def attest(_variant, _env):
        if source.read_text() != "original":
            raise RuntimeError("uncommitted source changes after preflight")
        return {"native": str(native), "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}

    def worker(command, **_kwargs):
        worker_calls.append(command)
        result = Path(command[command.index("--result") + 1])
        prediction = "changed" if failure == "prediction" and len(worker_calls) > 4 else "unchanged"
        result.write_text(json.dumps({"native_modules": [str(native)], "result_sha256": prediction}))
        if failure == "source":
            source.write_text("changed while measuring")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(BENCHMARK, "attest", attest)
    monkeypatch.setattr(BENCHMARK.platform, "platform", lambda: "fixture-platform")
    monkeypatch.setattr(BENCHMARK.subprocess, "check_output", lambda *_a, **_kw: "revision\n")
    monkeypatch.setattr(BENCHMARK.subprocess, "run", worker)
    args = SimpleNamespace(
        variants=str(variants),
        output=str(tmp_path / "results.json"),
        systems_path=str(tmp_path / "systems"),
        rounds=2 if failure == "prediction" else 1,
    )
    if failure:
        message = "source changes after preflight" if failure == "source" else "prediction mismatch"
        with pytest.raises(RuntimeError, match=message):
            BENCHMARK.controller(args)
        assert not (tmp_path / "results.json").exists()
        assert (tmp_path / "results.partial.json").exists()
    else:
        BENCHMARK.controller(args)
        assert len(json.loads((tmp_path / "results.json").read_text())["samples"]) == 2
        assert not (tmp_path / "results.partial.json").exists()
    assert len(worker_calls) == 2 * (args.rounds + 1)  # Two workers per warm-up/recorded round.


@pytest.mark.parametrize(
    "score,metric,mutate_source",
    [
        (float("nan"), 1, False),
        (float("inf"), 1, False),
        (-float("inf"), 1, False),
        (1, float("nan"), False),
        (1, 1, True),
        (1, 1, False),
    ],
)
def test_recommend_controller_only_publishes_valid_results(tmp_path, monkeypatch, score, metric, mutate_source):
    import math
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark_recommend_runtime as recommend

    variants = tmp_path / "variants.json"
    variants.write_text(
        json.dumps(
            [
                {
                    "label": "fixture",
                    "kind": "aisimulate",
                    "root": str(tmp_path),
                    "sources": [str(tmp_path)],
                    "python": "fixture-python",
                }
            ]
        )
    )
    output = tmp_path / "output"
    identities = iter([{"revision": "initial"}, {"revision": "changed" if mutate_source else "initial"}])
    monkeypatch.setattr(recommend, "attest", lambda *_args: next(identities))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recommend-benchmark",
            "--variants",
            str(variants),
            "--systems-path",
            str(tmp_path / "systems"),
            "--output-dir",
            str(output),
            "--rounds",
            "1",
            "--algorithms",
            "random",
        ],
    )

    def run(command, *_args):
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir()
        report = {
            "counts": {"failed": 0, "timed_out": 0},
            "views": {"top_n": [{}]},
            "candidates": [{"score": score, "metrics": {"duration_ms": metric}}],
        }
        (destination / "recommendation.json").write_text(json.dumps(report))
        return 1.0

    monkeypatch.setattr(recommend, "run", run)
    if not math.isfinite(score) or not math.isfinite(metric) or mutate_source:
        with pytest.raises((RuntimeError, ValueError)):
            recommend.main()
        assert not (output / "results.json").exists()
        if mutate_source:
            assert (output / "results.partial.json").exists()
    else:
        recommend.main()
        assert json.loads((output / "results.json").read_text())["samples"][0]["best_score"] == score
        assert not (output / "results.partial.json").exists()
