# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-gated SA-Bench defaults for the exact historical runner revision."""

from __future__ import annotations

from functools import cache
from typing import Any

import requests

from e2e_accuracy_source.inferencex_recipe import InferenceXRecipeError
from e2e_accuracy_source.sources import verify_sources

SRT_REPOSITORY = "https://github.com/NVIDIA/srt-slurm"
SRT_V1_0_29_SHA = "c1b6b5c97f323baefad577d70c4e8392b6f537d9"
_SA = "src/srtctl/benchmarks/scripts/sa-bench"
_SOURCE_HASHES = {
    "src/srtctl/core/schema.py": "1311cc1b1e8cf51192944d475317ad1eb62ab90c4bae9b948ad39cd546fb526e",
    "src/srtctl/benchmarks/sa_bench.py": "b4306db1f230b9ce8f0256367552c603737810c40e09e703a278adf3c0ad1b13",
    f"{_SA}/bench.sh": "12f11d8070210f7001b6b09c78cfc2beb5d0393fb2b9561a176b9339ca6770f4",
    f"{_SA}/benchmark_serving.py": "078d98fa0e1ce128b1cc88dac7f04067494b4925318339f9be7a8f4aedfb1ee2",
}
_SOURCES_BY_SHA = {
    SRT_V1_0_29_SHA: _SOURCE_HASHES,
    "c180328b98c3793ca84a1e24a030f90545eb7d5d": _SOURCE_HASHES
    | {"src/srtctl/core/schema.py": "8fcbc51e2ac8beaa27630b8b48c17e9cb7966cffc586fb57634bdd5c2a2595fb"},
    "c1fb6989fc5aca803b4ca0f2d17d8be85fad9732": {
        "src/srtctl/core/schema.py": "c3de67558cef2bf2768e830d0ea95fa39b5f3b3ba06e24b66290cbd0607ad06d",
        "src/srtctl/benchmarks/sa_bench.py": "487373592a416ac3b43d50847cc4ad52ae3fbf0f5c3fd7b5fc0604ac6fc01018",
        f"{_SA}/bench.sh": "e7f3c49dc452afc0ecfdca4c59ffda0339ea184cf2b803f81acfc038466aa0a0",
        f"{_SA}/benchmark_serving.py": "8af2d3328009d53e5fc0c640182c3d760f4dea8b2af25559aeabc7d2145b8af0",
    },
}


@cache
def _verified_runner_sources(git_sha: str = SRT_V1_0_29_SHA) -> list[dict]:
    records = [
        dict(path=path, url=f"https://raw.githubusercontent.com/NVIDIA/srt-slurm/{git_sha}/{path}", sha256=sha)
        for path, sha in _SOURCES_BY_SHA[git_sha].items()
    ]
    return verify_sources(records, kind="benchmark")


def resolve_workload_defaults(
    benchmark: dict[str, Any],
    source_identity: dict | None,
) -> tuple[dict[str, Any], list[dict]]:
    """Fill omissions only when the caller established this historical runner.

    The caller must trace ``source_identity`` to the actual run attempt (a
    runtime record or an executed launcher branch), not today's branch HEAD.
    Accepted identities contain the NVIDIA repository and exact ``git_sha`` or
    executed ``ref=v1.0.29``. A package version alone is not a source identity.
    Explicit benchmark values, including recorded runtime overrides, survive.
    """
    resolved = dict(benchmark)
    if benchmark.get("type") != "sa-bench" or not source_identity:
        return resolved, []
    repository = str(source_identity.get("repository", "")).removesuffix(".git").rstrip("/")
    if repository not in {SRT_REPOSITORY, "NVIDIA/srt-slurm"}:
        return resolved, []
    sha = source_identity.get("git_sha")
    if sha is not None and sha not in _SOURCES_BY_SHA:
        return resolved, []
    if sha is None and source_identity.get("ref") != "v1.0.29":
        return resolved, []
    sha = sha or SRT_V1_0_29_SHA
    try:
        sources = _verified_runner_sources(sha)
    except requests.RequestException as error:
        raise InferenceXRecipeError(f"cannot verify benchmark defaults: {error}") from error
    defaults = dict(
        random_range_ratio=0.8,
        num_prompts_mult=10,
        num_warmup_mult=2,
        use_chat_template=True,
        dataset_name="random",
        req_rate="inf",
        reuse_http_connections=False,
        seed=0,
    )
    evidence = []
    for knob, value in defaults.items():
        if resolved.get(knob) is None:
            resolved[knob] = value
            evidence.append(
                dict(
                    knob=knob,
                    value=value,
                    kind="verified_runner_default",
                    runner_git_sha=sha,
                    source_identity=source_identity,
                    sources=sources,
                )
            )
    return resolved, evidence


def unmodeled_workload_controls(benchmark: dict[str, Any]) -> list[dict]:
    """Keep client behavior visible even when scalar workload fields qualify.

    These are compatibility notes, not parser failures. They must be carried to
    prediction provenance; a caller requiring exact request replay can reject
    them separately instead of silently discarding the controls.
    """
    controls = []
    if benchmark.get("use_chat_template"):
        controls.append(
            dict(
                knob="use_chat_template",
                value=True,
                reason="native length-only replay does not apply tokenizer chat-template length adjustments",
            )
        )
    if benchmark.get("custom_tokenizer"):
        controls.append(
            dict(
                knob="custom_tokenizer",
                value=benchmark["custom_tokenizer"],
                reason="native replay does not execute the benchmark tokenizer",
            )
        )
    if benchmark.get("dataset_name") not in {None, "random"}:
        controls.append(
            dict(
                knob="dataset_name",
                value=benchmark["dataset_name"],
                reason="scalar ISL/OSL distribution does not reconstruct a custom dataset",
            )
        )
    if benchmark.get("num_warmup_mult", 0):
        controls.append(
            dict(
                knob="num_warmup_mult",
                value=benchmark["num_warmup_mult"],
                reason="warmup request cache state is not reconstructed by prediction replay",
            )
        )
    return controls
