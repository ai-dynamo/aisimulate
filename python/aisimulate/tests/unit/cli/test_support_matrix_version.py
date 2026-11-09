# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aisimulate.legacy_cli import cli_support
from aisimulate.legacy_cli.main import _latest_support_matrix_version, _run_support_mode
from aisimulate.sdk import perf_database

pytestmark = pytest.mark.unit


def _row(
    *,
    model: str = "model",
    architecture: str = "Arch",
    system: str = "b200_sxm",
    backend: str = "sglang",
    version: str,
) -> dict[str, str]:
    return {
        "HuggingFaceID": model,
        "Architecture": architecture,
        "System": system,
        "Backend": backend,
        "Version": version,
    }


@pytest.mark.parametrize(
    ("versions", "expected_version"),
    [
        pytest.param(
            ["0.5.9", "0.5.10"],
            "0.5.10",
            id="semantic-version-sort",
        ),
        pytest.param(
            ["", "bad-version", "0.5.10"],
            "0.5.10",
            id="ignore-invalid-versions",
        ),
        pytest.param(
            ["", "bad-version"],
            None,
            id="no-valid-versions",
        ),
    ],
)
def test_latest_support_matrix_version_selects_latest_valid_version(versions, expected_version):
    matrix = [_row(version=version) for version in versions]

    assert _latest_support_matrix_version(matrix, "b200_sxm", "sglang", model="model") == expected_version


def test_latest_support_matrix_version_matches_system_and_backend_case_insensitively():
    matrix = [_row(version="0.5.10")]

    assert _latest_support_matrix_version(matrix, "B200_SXM", "SGLang", model="model") == "0.5.10"


@pytest.mark.parametrize(
    ("matrix", "system", "model", "architecture", "expected_version"),
    [
        pytest.param(
            [
                _row(
                    model="Qwen/Qwen3-32B",
                    architecture="Qwen3ForCausalLM",
                    system="b300_sxm",
                    version="0.5.9",
                ),
                _row(
                    model="zai-org/GLM-5",
                    architecture="GlmMoeDsaForCausalLM",
                    system="b300_sxm",
                    version="0.5.10",
                ),
            ],
            "b300_sxm",
            "Qwen/Qwen3-32B",
            "Qwen3ForCausalLM",
            "0.5.9",
            id="prefer-exact-model",
        ),
        pytest.param(
            [
                _row(
                    model="deepseek-ai/DeepSeek-V3.2",
                    architecture="GlmMoeDsaForCausalLM",
                    version="0.5.9",
                ),
                _row(
                    model="zai-org/GLM-5-FP8",
                    architecture="GlmMoeDsaForCausalLM",
                    version="0.5.10",
                ),
            ],
            "b200_sxm",
            "local-glm5-variant",
            "GlmMoeDsaForCausalLM",
            "0.5.10",
            id="architecture-fallback",
        ),
    ],
)
def test_latest_support_matrix_version_scopes_rows_by_model_or_architecture(
    matrix,
    system,
    model,
    architecture,
    expected_version,
):
    assert (
        _latest_support_matrix_version(
            matrix,
            system,
            "sglang",
            model=model,
            architecture=architecture,
        )
        == expected_version
    )


def test_latest_support_matrix_version_does_not_fall_back_to_unrelated_rows():
    matrix = [
        _row(model="Qwen/Qwen3-32B", architecture="Qwen3ForCausalLM", version="0.5.9"),
        _row(model="zai-org/GLM-5-FP8", architecture="GlmMoeDsaForCausalLM", version="0.5.10"),
    ]

    assert (
        _latest_support_matrix_version(
            matrix,
            "b200_sxm",
            "sglang",
            model="local-unknown-model",
            architecture="UnknownArchitecture",
        )
        is None
    )


def test_latest_support_matrix_version_filters_queryable_versions_before_model_precedence():
    # An exact-model row at a stale version must not hide a queryable row for the same architecture.
    matrix = [
        _row(model="Qwen/Qwen3-32B", architecture="Qwen3ForCausalLM", version="0.5.16"),
        _row(model="Qwen/Qwen3-8B", architecture="Qwen3ForCausalLM", version="0.5.14"),
    ]

    assert (
        _latest_support_matrix_version(
            matrix,
            "b200_sxm",
            "sglang",
            model="Qwen/Qwen3-32B",
            architecture="Qwen3ForCausalLM",
            queryable_versions={"0.5.14", "0.5.17"},
        )
        == "0.5.14"
    )


def test_run_support_mode_stops_when_auto_version_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr("aisimulate.legacy_cli.main.perf_database.get_version_slots", lambda *_args: None)
    monkeypatch.setattr(
        "aisimulate.legacy_cli.main.get_model_config_from_model_path",
        lambda _model: {"architecture": "UnknownArchitecture"},
    )
    monkeypatch.setattr(
        "aisimulate.legacy_cli.main.common.get_support_matrix",
        lambda: [_row(model="Qwen/Qwen3-32B", architecture="Qwen3ForCausalLM", version="0.5.10")],
    )

    def fail_check_support(*_args, **_kwargs):
        raise AssertionError("check_support should not run without an auto-selected version")

    monkeypatch.setattr("aisimulate.legacy_cli.main.common.check_support", fail_check_support)

    _run_support_mode(
        SimpleNamespace(
            backend="sglang",
            backend_version=None,
            model_path="local-unknown-model",
            system="b200_sxm",
        )
    )

    assert "No valid support-matrix backend version found" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("system", "backend"),
    [("h200_sxm", "sglang"), ("a100_sxm", "sglang"), ("h200_sxm", "vllm"), ("h200_sxm", "trtllm")],
)
@pytest.mark.parametrize("current_model", [None, "model", "same-architecture-model"])
def test_run_support_mode_defaults_to_current_certification(monkeypatch, capsys, system, backend, current_model):
    slots = perf_database.get_version_slots(system, backend)
    assert slots is not None
    matrix = [
        dict(_row(system=system, backend=backend, version=version), Mode="agg", Status="PASS")
        for version in {"0.0.1", slots.get("previous", "0.0.1")}
    ]
    if current_model:
        matrix.append(
            dict(
                _row(model=current_model, system=system, backend=backend, version=slots["current"]),
                Mode="agg",
                Status="PASS",
            )
        )
    monkeypatch.setattr(
        "aisimulate.legacy_cli.main.get_model_config_from_model_path",
        lambda _model: {"architecture": "Arch"},
    )
    monkeypatch.setattr(
        "aisimulate.sdk.utils.get_model_config_from_model_path",
        lambda _model: {"architecture": "Arch"},
    )
    monkeypatch.setattr("aisimulate.legacy_cli.main.common.get_support_matrix", lambda: matrix)

    versions = [None, "current", slots["current"]]
    if "previous" in slots:
        versions.extend(["previous", slots["previous"]])
    for version in versions:
        _run_support_mode(SimpleNamespace(backend=backend, backend_version=version, model_path="model", system=system))
        output = capsys.readouterr().out
        supported = current_model is not None or version in {"previous", slots.get("previous", "")}
        assert f"Aggregated Support:    {'YES' if supported else 'NO'}" in output
        assert f"Version:         {version or slots['current']}" in output
        assert cli_support("model", system, backend=backend, backend_version=version).agg_supported == supported
