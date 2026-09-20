# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session pause/resume through the public CLI without GPU execution."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import aisimulate.main as cli
from aisimulate.support import checkpoint
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit


@pytest.fixture
def request_payload():
    """Synthetic resources are declared test bounds, not measured silicon."""
    profile = {
        "schema_version": 1,
        "model": "example/checkpoint-test",
        "model_revision": "revision-123",
        "architecture": "SyntheticDecoder",
        "context_length": 16384,
        "num_experts": 0,
        "provenance": "Synthetic checkpoint test",
        "deployments": [
            {
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.27.0",
                "tp": 1,
                "dp": 1,
                "moe_tp": 1,
                "moe_ep": 1,
                "gemm_quant_mode": "bfloat16",
                "moe_quant_mode": "bfloat16",
                "fmha_quant_mode": "bfloat16",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "bfloat16",
                "resources": {
                    "weights_bytes": 1000,
                    "activations_bytes": 100,
                    "runtime_overhead_bytes": 100,
                    "comm_overhead_bytes": 0,
                    "kv_bytes_per_token": 128,
                    "cache_layout": "linear",
                    "max_num_tokens": 8192,
                    "max_batch_size": 256,
                    "provenance": "Synthetic declared bounds",
                },
            }
        ],
    }
    return SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "dense",
                "framework_version": "0.27.0",
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {"context_length": 16384},
            "collection": {"prefill_cudagraph_policy": "runtime"},
            "fpm_profile": profile,
        }
    ).model_dump(mode="json", exclude_none=True)


def _save(path, capsys, monkeypatch, patch=None, revision=None, accept=(), expected=0):
    args = ["onboard", "checkpoint", "--file", str(path)]
    if patch is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(patch)))
        args += ["--update", "-"]
    if revision is not None:
        args += ["--expect-revision", str(revision)]
    for name in accept:
        args += ["--accept-profile", name]
    assert cli.main(args) == expected
    return json.loads(capsys.readouterr().out)


def _resume(path, capsys, expected=0):
    assert cli.main(["onboard", "resume", "--checkpoint", str(path)]) == expected
    return json.loads(capsys.readouterr().out)


def _subprocess(*args, cwd, stdin=None):
    # The installed public console entry point bypasses supervision for onboarding.
    return subprocess.run(
        [str(Path(sys.executable).parent / "aisimulate"), "onboard", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).parents[1])},
    )


def test_empty_stage_one_and_research_resume_in_new_process_and_cwd(tmp_path):
    path = tmp_path / "session" / "onboarding-checkpoint.json"
    created = _subprocess("checkpoint", "--file", str(path), cwd=tmp_path)
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["state"]["revision"] == 1
    marker = tmp_path / "must-not-execute"
    patch = {
        "research": {"runtime": {"source": "pinned/source/path", "confidence": "unknown", "alternatives": ["a", "b"]}},
        "pending_questions": ["Choose framework version"],
        "progress": {"stage": 2, "next_action": f"touch {marker}"},
    }
    updated = _subprocess(
        "checkpoint",
        "--file",
        str(path),
        "--update",
        "-",
        "--expect-revision",
        "1",
        cwd=tmp_path,
        stdin=json.dumps(patch),
    )
    assert updated.returncode == 0, updated.stderr
    resumed = _subprocess("resume", "--checkpoint", str(path), cwd=tmp_path.parent)
    assert resumed.returncode == 0, resumed.stderr
    result = json.loads(resumed.stdout)
    assert result["state"]["research"] == patch["research"]
    assert result["state"]["pending_questions"] == patch["pending_questions"]
    assert result["verification"]["executed"] is False
    assert not marker.exists()
    assert sorted(p.name for p in path.parent.iterdir()) == [
        ".onboarding-checkpoint.json.lock",
        "onboarding-checkpoint.json",
    ]


def test_partial_approval_and_affected_changes(tmp_path, capsys, monkeypatch, request_payload):
    path = tmp_path / "checkpoint.json"
    first = {"inputs": {"precision": "bf16"}, "draft_request": request_payload}
    second = {"inputs": {"precision": "alternative"}, "draft_request": {"identity": {"model": "still investigating"}}}
    saved = _save(
        path, capsys, monkeypatch, {"configurations": {"bf16-tp1": first, "other-tp4": second}}, accept=["bf16-tp1"]
    )
    assert saved["configurations"]["bf16-tp1"]["profile_accepted"]
    assert not saved["configurations"]["other-tp4"]["profile_accepted"]
    original_acceptance = saved["state"]["configurations"]["bf16-tp1"]["acceptance"]
    notes = _save(path, capsys, monkeypatch, {"research": {"note": "Paused after reviewing first profile"}}, 1)
    assert notes["state"]["configurations"]["bf16-tp1"]["acceptance"] == original_acceptance
    changed_second = _save(
        path, capsys, monkeypatch, {"configurations": {"other-tp4": {"inputs": {"precision": "fp8"}}}}, 2
    )
    assert changed_second["configurations"]["bf16-tp1"]["profile_accepted"]
    changed_first = _save(
        path,
        capsys,
        monkeypatch,
        {"configurations": {"bf16-tp1": {"draft_request": {"collection": {"gpu_memory_utilization": 0.8}}}}},
        3,
    )
    assert not changed_first["configurations"]["bf16-tp1"]["profile_accepted"]
    assert changed_first["state"]["configurations"]["bf16-tp1"]["acceptance"] is None
    _save(path, capsys, monkeypatch, revision=4, accept=["bf16-tp1"])
    shared = _save(path, capsys, monkeypatch, {"inputs": {"runtime_build": "different"}}, 5)
    assert not shared["configurations"]["bf16-tp1"]["profile_accepted"]
    assert shared["state"]["configurations"]["bf16-tp1"]["progress"]["stage"] == 3
    assert not (tmp_path / "request.yaml").exists()


def test_validation_changes_preserve_collection_identity_and_approval(tmp_path, capsys, monkeypatch, request_payload):
    path = tmp_path / "checkpoint.json"
    for name in ("plan.json", "validation.json"):
        (tmp_path / name).write_text("{}")
    patch = {
        "configurations": {
            "one": {
                "draft_request": request_payload,
                "progress": {"stage": 6, "status": "complete"},
                "artifacts": {
                    "plan": {"path": "plan.json", "scope": "collection"},
                    "validation": {"path": "validation.json", "scope": "validation"},
                },
            }
        }
    }
    saved = _save(path, capsys, monkeypatch, patch, accept=["one"])
    assert saved["configurations"]["one"]["profile_accepted"]
    updated = _save(path, capsys, monkeypatch, {"validation_inputs": {"trace": "new trace"}}, 1, expected=2)
    assert updated["saved"]
    assert updated["configurations"]["one"]["profile_accepted"]
    assert [issue["artifact"] for issue in updated["integrity_issues"]] == ["validation"]
    assert updated["state"]["configurations"]["one"]["progress"]["status"] == "in_progress"
    assert (tmp_path / "plan.json").read_text() == "{}"
    (tmp_path / "validation-v2.json").write_text('{"trace":"new trace"}')
    recovered = _save(
        path,
        capsys,
        monkeypatch,
        {
            "configurations": {
                "one": {
                    "artifacts": {
                        "validation": {"archived": True},
                        "validation_v2": {"path": "validation-v2.json", "scope": "validation"},
                    }
                }
            }
        },
        2,
    )
    original = saved["state"]["configurations"]["one"]
    current = recovered["state"]["configurations"]["one"]
    assert current["acceptance"] == original["acceptance"]
    assert current["artifacts"]["plan"] == original["artifacts"]["plan"]
    assert current["artifacts"]["validation"] == {**original["artifacts"]["validation"], "archived": True}
    assert _resume(path, capsys)["configurations"]["one"]["effective_status"] == "accepted"


def test_input_and_immutable_drift_survive_note_save_while_collector_can_advance(
    tmp_path, capsys, monkeypatch, request_payload
):
    source = tmp_path / "config.json"
    source.write_text('{"model":1}')
    collector = tmp_path / "collector.json"
    collector.write_text('{"cells":{}}')
    path = tmp_path / "checkpoint.json"
    saved = _save(
        path,
        capsys,
        monkeypatch,
        {
            "artifacts": {"source": {"path": str(source)}},
            "configurations": {
                "one": {
                    "draft_request": request_payload,
                    "artifacts": {
                        "collector": {
                            "path": "collector.json",
                            "kind": "collector_checkpoint",
                            "scope": "collection",
                        }
                    },
                }
            },
        },
        accept=["one"],
    )
    digest = saved["state"]["artifacts"]["source"]["sha256"]
    collector.write_text('{"cells":{"one":"passed"}}')
    assert _resume(path, capsys)["configurations"]["one"]["profile_accepted"]
    source.write_text('{"model":2}')
    saved = _save(
        path,
        capsys,
        monkeypatch,
        {"research": {"note": "do not refresh hashes"}, "artifacts": {"source": {"path": "config.json"}}},
        1,
        expected=2,
    )
    assert saved["state"]["artifacts"]["source"]["sha256"] == digest
    assert not saved["configurations"]["one"]["profile_accepted"]
    assert saved["integrity_issues"][0]["artifact"] == "source"
    assert json.loads(collector.read_text()) == {"cells": {"one": "passed"}}
    collector.unlink()
    resumed = _resume(path, capsys, expected=2)
    assert {issue["artifact"] for issue in resumed["integrity_issues"]} == {"source", "collector"}


def test_exact_request_and_profile_references(tmp_path, capsys, monkeypatch, request_payload):
    path = tmp_path / "checkpoint.json"
    (tmp_path / "request.yaml").write_text(yaml.safe_dump(request_payload))
    (tmp_path / "profile.json").write_text(json.dumps(request_payload["fpm_profile"]))
    patch = {
        "configurations": {
            "one": {
                "draft_request": request_payload,
                "artifacts": {
                    "request": {"path": "request.yaml", "kind": "request", "scope": "collection"},
                    "profile": {"path": "profile.json", "kind": "profile", "scope": "collection"},
                },
            }
        }
    }
    assert _save(path, capsys, monkeypatch, patch, accept=["one"])["configurations"]["one"]["profile_accepted"]
    wrong = deepcopy(request_payload)
    wrong["collection"]["gpu_memory_utilization"] = 0.7
    (tmp_path / "wrong.yaml").write_text(yaml.safe_dump(wrong))
    result = _save(
        path,
        capsys,
        monkeypatch,
        {"configurations": {"one": {"artifacts": {"request": {"path": "wrong.yaml"}}}}},
        1,
        expected=2,
    )
    assert "does not match" in result["integrity_issues"][0]["detail"]
    assert "agent-reported" in result["verification"]["scope"]


def test_archive_superseded_outputs_retains_history_and_allows_current_work(
    tmp_path, capsys, monkeypatch, request_payload
):
    path = tmp_path / "checkpoint.json"
    (tmp_path / "request-v1.yaml").write_text(yaml.safe_dump(request_payload))
    (tmp_path / "plan-v1.json").write_text('{"plan":"old"}')
    (tmp_path / "collector-v1.json").write_text('{"cells":{"one":"passed"}}')
    old_refs = {
        "request_v1": {"path": "request-v1.yaml", "kind": "request", "scope": "collection"},
        "plan_v1": {"path": "plan-v1.json", "scope": "collection"},
        "collector_v1": {"path": "collector-v1.json", "kind": "collector_checkpoint", "scope": "collection"},
    }
    initial = _save(
        path,
        capsys,
        monkeypatch,
        {"configurations": {"one": {"draft_request": request_payload, "artifacts": old_refs}}},
        accept=["one"],
    )
    snapshots = initial["state"]["configurations"]["one"]["artifacts"]
    changed = deepcopy(request_payload)
    changed["collection"]["gpu_memory_utilization"] = 0.8
    (tmp_path / "request-v2.yaml").write_text(yaml.safe_dump(changed))
    (tmp_path / "plan-v2.json").write_text('{"plan":"current"}')
    current = _save(
        path,
        capsys,
        monkeypatch,
        {
            "configurations": {
                "one": {
                    "draft_request": changed,
                    "artifacts": {
                        "request_v2": {"path": "request-v2.yaml", "kind": "request", "scope": "collection"},
                        "plan_v2": {"path": "plan-v2.json", "scope": "collection"},
                    },
                }
            }
        },
        1,
        accept=["one"],
        expected=2,
    )
    assert current["configurations"]["one"]["profile_accepted"]
    assert current["configurations"]["one"]["effective_status"] == "needs_attention"
    assert {issue["artifact"] for issue in current["integrity_issues"]} == set(old_refs)
    archived = _save(
        path,
        capsys,
        monkeypatch,
        {"configurations": {"one": {"artifacts": {name: {"archived": True} for name in old_refs}}}},
        2,
    )
    assert archived["configurations"]["one"]["effective_status"] == "accepted"
    assert {ref["artifact"] for ref in archived["archived_artifacts"]} == set(old_refs)
    assert "Historical references only" in archived["verification"]["archived_artifacts"]
    for name, original in snapshots.items():
        assert archived["state"]["configurations"]["one"]["artifacts"][name] == {**original, "archived": True}
        assert (tmp_path / original["path"]).is_file()
    restored = _save(
        path,
        capsys,
        monkeypatch,
        {"configurations": {"one": {"artifacts": {"plan_v1": {"archived": False}}}}},
        3,
        expected=2,
    )
    assert [issue["artifact"] for issue in restored["integrity_issues"]] == ["plan_v1"]
    assert "stale" in restored["integrity_issues"][0]["detail"]
    _save(
        path,
        capsys,
        monkeypatch,
        {"configurations": {"one": {"artifacts": {"plan_v1": {"archived": True}}}}},
        4,
    )
    for original in snapshots.values():
        (tmp_path / original["path"]).unlink()
    resumed = _resume(path, capsys)
    assert resumed["configurations"]["one"]["effective_status"] == "accepted"
    assert resumed["archived_artifacts"] == archived["archived_artifacts"]
    assert (tmp_path / "request-v2.yaml").is_file()
    assert (tmp_path / "plan-v2.json").is_file()


@pytest.mark.parametrize("shared", [True, False])
def test_retiring_source_requires_new_acceptance(tmp_path, capsys, monkeypatch, request_payload, shared):
    path = tmp_path / "checkpoint.json"
    source = tmp_path / "config.json"
    source.write_text('{"version":1}')
    patch = {
        "configurations": {
            name: {"draft_request": request_payload, "progress": {"stage": 6, "status": "complete"}}
            for name in ("one", "two")
        }
    }
    owner = patch if shared else patch["configurations"]["one"]
    owner["artifacts"] = {"source": {"path": "config.json"}}
    initial = _save(path, capsys, monkeypatch, patch, accept=["one", "two"])
    original = initial["state"] if shared else initial["state"]["configurations"]["one"]
    retired_ref = {"artifacts": {"source": {"archived": True}}}
    update = retired_ref if shared else {"configurations": {"one": retired_ref}}
    source.unlink()
    retired = _save(path, capsys, monkeypatch, update, 1)
    assert not retired["configurations"]["one"]["profile_accepted"]
    assert retired["state"]["configurations"]["one"]["acceptance"] is None
    assert retired["state"]["configurations"]["one"]["progress"]["stage"] == 3
    assert retired["configurations"]["two"]["profile_accepted"] is not shared
    owner = retired["state"] if shared else retired["state"]["configurations"]["one"]
    assert owner["artifacts"]["source"] == {**original["artifacts"]["source"], "archived": True}
    assert retired["archived_artifacts"] == [
        {"configuration": None if shared else "one", "artifact": "source", "path": "config.json"}
    ]
    accepted = _save(path, capsys, monkeypatch, revision=2, accept=["one"])
    assert accepted["configurations"]["one"]["profile_accepted"]


@pytest.mark.parametrize("invalid", ["true", 1, [], {}])
def test_invalid_archive_field_preserves_checkpoint(tmp_path, capsys, monkeypatch, invalid):
    path = tmp_path / "checkpoint.json"
    (tmp_path / "config.json").write_text("{}")
    _save(path, capsys, monkeypatch, {"artifacts": {"source": {"path": "config.json"}}})
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        _save(path, capsys, monkeypatch, {"artifacts": {"source": {"archived": invalid}}}, 1)
    assert error.value.code == 2
    capsys.readouterr()
    assert path.read_bytes() == before
    malformed = json.loads(before)
    malformed["artifacts"]["source"]["archived"] = invalid
    path.write_text(json.dumps(malformed))
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        _resume(path, capsys)
    assert error.value.code == 2
    capsys.readouterr()
    assert path.read_bytes() == before


def test_saved_artifact_without_archive_field_remains_current(tmp_path, capsys, monkeypatch, request_payload):
    path = tmp_path / "checkpoint.json"
    source = tmp_path / "config.json"
    source.write_text("{}")
    _save(
        path,
        capsys,
        monkeypatch,
        {
            "artifacts": {"source": {"path": "config.json"}},
            "configurations": {"one": {"draft_request": request_payload}},
        },
        accept=["one"],
    )
    legacy = json.loads(path.read_text())
    del legacy["artifacts"]["source"]["archived"]
    path.write_text(json.dumps(legacy))
    before = path.read_bytes()
    assert _resume(path, capsys)["configurations"]["one"]["profile_accepted"]
    assert path.read_bytes() == before
    source.write_text('{"changed":true}')
    resumed = _resume(path, capsys, expected=2)
    assert not resumed["configurations"]["one"]["profile_accepted"]
    assert resumed["integrity_issues"][0]["artifact"] == "source"


@pytest.mark.parametrize(
    "patch,accept",
    [
        ({"schema_version": "unknown"}, []),
        ({"configurations": []}, []),
        ({"configurations": {"one": {"acceptance": {"sha256": "0" * 64, "revision": 1}}}}, []),
        ({"configurations": {"one": {"draft_request": {}}}}, ["one"]),
        ({"configurations": {"one": {"progress": {"stage": 7}}}}, []),
        ({"artifacts": {"missing": {"path": "missing.json"}}}, []),
        ({"artifacts": {"self": {"path": "checkpoint.json"}}}, []),
        ({"artifacts": {"self": {"path": ".checkpoint.json.lock"}}}, []),
        ({"artifacts": {"fake": {"path": "missing.json", "sha256": "0" * 64}}}, []),
    ],
)
def test_invalid_updates_preserve_previous_checkpoint(tmp_path, capsys, monkeypatch, patch, accept):
    path = tmp_path / "checkpoint.json"
    _save(path, capsys, monkeypatch)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        _save(path, capsys, monkeypatch, patch, 1, accept=accept)
    assert error.value.code == 2
    assert path.read_bytes() == before
    capsys.readouterr()


def test_stale_writer_and_interrupted_atomic_save_preserve_checkpoint(tmp_path, capsys, monkeypatch):
    path = tmp_path / "checkpoint.json"
    _save(path, capsys, monkeypatch)
    _save(path, capsys, monkeypatch, {"research": {"first": "saved"}}, 1)
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        _save(path, capsys, monkeypatch, {"research": {"lost": "stale"}}, 1)
    capsys.readouterr()
    assert path.read_bytes() == before

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint.os, "replace", interrupt)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"research":{"second":"never saved"}}'))
    assert cli.main(["onboard", "checkpoint", "--file", str(path), "--update", "-", "--expect-revision", "2"]) == 130
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [".checkpoint.json.lock", "checkpoint.json"]


def test_two_processes_cannot_overwrite_same_revision(tmp_path):
    path = tmp_path / "checkpoint.json"
    assert _subprocess("checkpoint", "--file", str(path), cwd=tmp_path).returncode == 0
    command = [
        str(Path(sys.executable).parent / "aisimulate"),
        "onboard",
        "checkpoint",
        "--file",
        str(path),
        "--update",
        "-",
        "--expect-revision",
        "1",
    ]
    env = {**os.environ, "PYTHONPATH": str(Path(cli.__file__).parents[1])}
    processes = [
        subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        for _ in range(2)
    ]
    for index, process in enumerate(processes):
        process.stdin.write(json.dumps({"research": {"writer": index}}))
        process.stdin.close()
        process.stdin = None
    results = [process.communicate(timeout=30) for process in processes]
    assert sorted(process.returncode for process in processes) == [0, 2], results
    assert json.loads(path.read_text())["revision"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        '{"revision":1}',
        '{"schema_version":"future","revision":1}',
        '{"schema_version":"aisimulate-onboarding-checkpoint/v1","revision":1,"revision":2}',
    ],
)
def test_invalid_saved_checkpoints_are_not_replaced(tmp_path, capsys, monkeypatch, payload):
    path = tmp_path / "checkpoint.json"
    path.write_text(payload)
    for action in (["resume", "--checkpoint"], ["checkpoint", "--file"]):
        with pytest.raises(SystemExit) as error:
            cli.main(["onboard", *action, str(path)])
        assert error.value.code == 2
        assert path.read_text() == payload
        capsys.readouterr()


def test_checkpoint_and_lock_symlink_collisions_do_not_replace_files(tmp_path, capsys, monkeypatch):
    victim = tmp_path / "victim.json"
    victim.write_text("keep me")
    path = tmp_path / "checkpoint.json"
    path.symlink_to(victim)
    with pytest.raises(SystemExit):
        _save(path, capsys, monkeypatch)
    capsys.readouterr()
    assert victim.read_text() == "keep me"
    path.unlink()
    (tmp_path / ".checkpoint.json.lock").symlink_to(victim)
    with pytest.raises(SystemExit):
        _save(path, capsys, monkeypatch)
    capsys.readouterr()
    assert victim.read_text() == "keep me"
    assert not path.exists()


def test_one_session_keeps_same_topology_precision_variants_independent(tmp_path, capsys, monkeypatch, request_payload):
    path = tmp_path / "checkpoint.json"
    alternate = deepcopy(request_payload)
    alternate["fpm_profile"]["deployments"][0]["kv_cache_dtype"] = "fp8"
    alternate["fpm_profile"]["deployments"][0]["resources"]["kv_bytes_per_token"] = 64
    configurations = {
        "bf16-tp1": {"draft_request": request_payload},
        "fp8-kv-tp1": {"draft_request": alternate},
    }
    saved = _save(path, capsys, monkeypatch, {"configurations": configurations}, accept=list(configurations))
    assert all(config["profile_accepted"] for config in saved["configurations"].values())
    assert saved["verification"]["executed"] is False
    # Recording a valid precision profile does not establish collector compatibility.
    updated = _save(
        path,
        capsys,
        monkeypatch,
        {
            "configurations": {
                "fp8-kv-tp1": {
                    "draft_request": {
                        "fpm_profile": {
                            "deployments": [
                                {
                                    **alternate["fpm_profile"]["deployments"][0],
                                    "resources": {
                                        **alternate["fpm_profile"]["deployments"][0]["resources"],
                                        "weights_bytes": 2000,
                                    },
                                }
                            ]
                        },
                    }
                }
            }
        },
        1,
    )
    assert updated["configurations"]["bf16-tp1"]["profile_accepted"]
    assert not updated["configurations"]["fp8-kv-tp1"]["profile_accepted"]


def test_locked_session_fails_promptly_and_preserves_previous_revision(tmp_path, capsys, monkeypatch):
    import fcntl

    path = tmp_path / "checkpoint.json"
    _save(path, capsys, monkeypatch)
    before = path.read_bytes()
    with (tmp_path / ".checkpoint.json.lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with pytest.raises(SystemExit) as error:
            _save(path, capsys, monkeypatch, {"research": {"note": "busy"}}, 1)
        assert error.value.code == 2
        assert "another checkpoint operation" in capsys.readouterr().err
    assert path.read_bytes() == before


def test_existing_update_requires_revision_and_nan_is_rejected(tmp_path, capsys, monkeypatch):
    path = tmp_path / "checkpoint.json"
    _save(path, capsys, monkeypatch)
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        _save(path, capsys, monkeypatch, {"research": {"note": "no revision"}})
    assert "--expect-revision 1" in capsys.readouterr().err
    for invalid in ('{"research":{"value":NaN}}', '{"research":{"value":1e1000}}'):
        monkeypatch.setattr(sys, "stdin", io.StringIO(invalid))
        with pytest.raises(SystemExit) as error:
            cli.main(["onboard", "checkpoint", "--file", str(path), "--update", "-", "--expect-revision", "1"])
        assert error.value.code == 2
        capsys.readouterr()
    assert path.read_bytes() == before
