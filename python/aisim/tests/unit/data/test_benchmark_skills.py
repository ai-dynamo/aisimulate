# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run walkthrough of the benchmark skill helpers on fixtures (no GPUs, no network).

plan (catalog) -> record GPU time (sacct and Kubernetes documents) -> store, verify and
attest an eval record. This exercises the deterministic checks the skills call; it does
not replace a run on a real model and GPU.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from aisim_core.data.catalog import build_catalog
from aisim_core.data.eval import current_records, write_record
from tests.unit.data.test_catalog import dataset  # noqa: F401  (fixture)
from tests.unit.data.test_eval import _predict
from tests.unit.data.test_eval import built as built_record  # noqa: F401  (fixture)

pytestmark = pytest.mark.unit

SKILLS = Path(__file__).resolve().parents[5] / ".agents" / "skills"
QWEN = "Qwen/Qwen3-32B"


def _script(skill: str, name: str):
    path = SKILLS / skill / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_frontmatter_and_links():
    skills = sorted(SKILLS.glob("aisim-benchmark*"))
    assert [s.name for s in skills] == [
        "aisim-benchmark",
        "aisim-benchmark-collect",
        "aisim-benchmark-evaluate",
        "aisim-benchmark-plan",
        "aisim-benchmark-publish",
    ]
    for skill in skills:
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        assert re.match(rf"---\nname: {skill.name}\ndescription: .+\n", text)
        assert len(text.splitlines()) < 120
        assert (skill / "agents" / "openai.yaml").is_file()
        assert (SKILLS.parents[1] / ".claude" / "skills" / skill.name).is_symlink()


def test_plan_groups_cells_and_budget(built_catalog, capsys):
    root, systems = built_catalog
    plan = _script("aisim-benchmark-plan", "plan_catalog.py")
    assert (
        plan.main(
            ["--dataset-root", str(root), "--hardware", "b200_sxm", "--model", QWEN, "--gpus", "8", "--wall-hours", "2"]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    cells = out["cells"]
    assert {(c["layer"], c["model"]) for c in cells["reuse"]} == {("fpm", QWEN), ("e2e", QWEN)}
    assert out["budget"]["authority"] == "estimated" and out["budget"]["gpu_seconds"] == 8 * 2 * 3600 * 1.25
    assert out["prior_recorded_gpu_seconds"] == 2400.0
    assert plan.main(["--dataset-root", str(root / "missing"), "--hardware", "b200_sxm"]) == 2


@pytest.fixture
def built_catalog(dataset):  # noqa: F811
    root, systems = dataset
    build_catalog(root, systems_dir=systems)
    return root, systems


SACCT_HEADER = "JobID|JobName|State|Submit|Start|End|ElapsedRaw|Elapsed|NNodes|AllocTRES"
SACCT = "\n".join(
    [
        SACCT_HEADER,
        "1|b|FAILED|2026-01-01T00:00:00|2026-01-01T00:00:00|2026-01-01T00:10:00|600|00:10:00|1|gres/gpu=8",
        "2|b|COMPLETED|2026-01-01T00:20:00|2026-01-01T01:00:00|2026-01-01T02:00:00|3600|01:00:00|1|gres/gpu=8",
        "2.batch|batch|COMPLETED|2026-01-01T01:00:00|2026-01-01T01:00:00|2026-01-01T02:00:00|3600|01:00:00|1|gres/gpu=8",
    ]
)


def test_record_gpu_time_slurm_and_unfinished(tmp_path, capsys):
    tool = _script("aisim-benchmark-collect", "record_gpu_time.py")
    sacct = tmp_path / "sacct.txt"
    sacct.write_text(SACCT + "\n")
    assert tool.main(["slurm", "--sacct", str(sacct), "--gpu-type", "b200"]) == 0
    out = json.loads(capsys.readouterr().out)
    gt = out["gpu_time"]
    assert gt["authority"] == "measured" and gt["gpu_seconds"] == 3600 * 8
    assert gt["failed"]["gpu_seconds"] == 600 * 8 and gt["failed"]["count"] == 1
    assert out["findings"]["qualification"] == []

    sacct.write_text(SACCT_HEADER + "\n3|b|RUNNING|x|x|Unknown|10|00:00:10|1|gres/gpu=8\n")
    assert tool.main(["slurm", "--sacct", str(sacct)]) == 2  # not final: no number invented


def test_record_gpu_time_kubernetes(tmp_path, capsys):
    tool = _script("aisim-benchmark-collect", "record_gpu_time.py")
    job = {
        "metadata": {"name": "j", "namespace": "ns"},
        "status": {"succeeded": 1},
        "spec": {"template": {"spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "4"}}}]}}},
    }
    pod = {
        "metadata": {"name": "j-abc"},
        "spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "4"}}}]},
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {"state": {"terminated": {"startedAt": "2026-01-01T00:00:00Z", "finishedAt": "2026-01-01T00:30:00Z"}}}
            ],
        },
    }
    (tmp_path / "job.json").write_text(json.dumps(job))
    (tmp_path / "pods.json").write_text(json.dumps({"items": [pod]}))
    assert tool.main(["k8s", "--job", str(tmp_path / "job.json"), "--pods", str(tmp_path / "pods.json")]) == 0
    gt = json.loads(capsys.readouterr().out)["gpu_time"]
    assert gt["gpu_count"] == 4 and gt["gpu_seconds"] == 1800 * 4 and gt["gpu_type"] is None


def test_store_verify_attest_walkthrough(built_record, tmp_path, capsys):  # noqa: F811
    root, record = built_record
    path = write_record(root, record)
    tool = _script("aisim-benchmark-evaluate", "check_eval_record.py")

    assert tool.main([str(path)]) == 0  # load + findings only
    info = json.loads(capsys.readouterr().out)
    assert info["tier"] == "submitted" and info["gpu_time_findings"]["qualification"][0]["severity"] == "error"

    good = tmp_path / "good.py"
    good.write_text(
        "def predict(inputs):\n    return inputs['tokens'] * 1.1 + (5.0 if inputs['tokens'] == 40 else 0.0)\n"
        f"def fingerprint():\n    return {record.prediction.fingerprint!r}\n"
    )
    assert tool.main([str(path), "--truth-root", str(root), "--predictor", str(good), "--attest-root", str(root)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verification"]["ok"] and out["verified_record_id"]
    assert [r.tier for r in current_records(path.parent)] == ["verified"]

    bad = tmp_path / "bad.py"
    bad.write_text(good.read_text().replace("1.1", "1.5"))
    assert tool.main([str(path), "--truth-root", str(root), "--predictor", str(bad)]) == 1
    assert _predict({"tokens": 10}) == pytest.approx(11.0)
