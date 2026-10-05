# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from e2e_accuracy_source.defaults import workload_defaults as defaults


@pytest.fixture(autouse=True)
def verified_sources(monkeypatch):
    monkeypatch.setattr(defaults, "_verified_runner_sources", lambda *_: [{"sha256": "verified-source"}])


def identity(**kwargs):
    return {"repository": defaults.SRT_REPOSITORY, "git_sha": defaults.SRT_V1_0_29_SHA, **kwargs}


def test_exact_runner_defaults_follow_actual_entrypoint():
    resolved, evidence = defaults.resolve_workload_defaults({"type": "sa-bench"}, identity())
    assert resolved["random_range_ratio"] == 0.8  # Python CLI alone declares 1.0
    assert resolved["num_prompts_mult"] == 10
    assert resolved["num_warmup_mult"] == 2
    assert resolved["use_chat_template"] is True
    assert resolved["seed"] == 0
    assert all(e["runner_git_sha"] == defaults.SRT_V1_0_29_SHA for e in evidence)


@pytest.mark.parametrize(
    "sha", ["c1fb6989fc5aca803b4ca0f2d17d8be85fad9732", "c180328b98c3793ca84a1e24a030f90545eb7d5d"]
)
def test_additional_exact_runner_commits(sha):
    resolved, evidence = defaults.resolve_workload_defaults({"type": "sa-bench"}, identity(git_sha=sha))
    assert resolved["num_prompts_mult"] == 10
    assert resolved["random_range_ratio"] == 0.8
    assert all(item["runner_git_sha"] == sha for item in evidence)


@pytest.mark.parametrize("ref", ["main", "c1fb6989", "c180328b", "aflowers/vllm-gb200-v0.20.0"])
def test_additional_runner_defaults_require_full_historical_commit(ref):
    source = {"repository": defaults.SRT_REPOSITORY, "ref": ref}
    assert defaults.resolve_workload_defaults({"type": "sa-bench"}, source) == ({"type": "sa-bench"}, [])


@pytest.mark.parametrize("multiplier", [16, 20])
def test_explicit_recipe_or_runtime_multiplier_wins(multiplier):
    benchmark = {
        "type": "sa-bench",
        "num_prompts_mult": multiplier,
        "random_range_ratio": 1.0,
        "use_chat_template": False,
    }
    resolved, evidence = defaults.resolve_workload_defaults(benchmark, identity())
    assert resolved["num_prompts_mult"] == multiplier
    assert resolved["random_range_ratio"] == 1.0
    assert resolved["use_chat_template"] is False
    assert not any(e["knob"] in benchmark for e in evidence)


@pytest.mark.parametrize(
    "source",
    [
        None,
        {"repository": defaults.SRT_REPOSITORY, "ref": "sa-submission-q2-2026"},
        {"repository": defaults.SRT_REPOSITORY, "version": "1.0.29"},
        {"repository": "https://github.com/someone/srt-slurm", "git_sha": defaults.SRT_V1_0_29_SHA},
        {"repository": defaults.SRT_REPOSITORY, "git_sha": "a" * 40, "ref": "v1.0.29"},
    ],
)
def test_unverified_revision_or_fork_cannot_supply_defaults(source):
    assert defaults.resolve_workload_defaults({"type": "sa-bench"}, source) == ({"type": "sa-bench"}, [])


def test_executed_release_tag_has_reviewed_sha():
    source = {"repository": defaults.SRT_REPOSITORY, "ref": "v1.0.29", "launcher_path": "runners/launch_gb300-nv.sh"}
    resolved, evidence = defaults.resolve_workload_defaults({"type": "sa-bench"}, source)
    assert resolved["num_prompts_mult"] == 10
    assert evidence[0]["source_identity"] == source


def test_custom_benchmark_does_not_receive_sa_defaults():
    assert defaults.resolve_workload_defaults({"type": "custom"}, identity()) == ({"type": "custom"}, [])


def test_client_controls_remain_explicit_prediction_limitations():
    notes = defaults.unmodeled_workload_controls(
        {"use_chat_template": True, "num_warmup_mult": 2, "custom_tokenizer": "my.CustomTokenizer"}
    )
    assert {n["knob"] for n in notes} == {"use_chat_template", "num_warmup_mult", "custom_tokenizer"}
    assert defaults.unmodeled_workload_controls({"use_chat_template": False}) == []
