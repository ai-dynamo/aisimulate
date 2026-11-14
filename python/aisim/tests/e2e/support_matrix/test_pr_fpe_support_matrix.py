# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the nightly's required native FPE sentinels on the existing CI wheel."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest
from tools.support_matrix.fpe_support_matrix import (
    ProbeWorkload,
    build_probe_plans,
    run_probe_plans,
    write_outputs,
)
from tools.support_matrix.qualify_fpe_support_matrix import IDENTITY, qualify

pytestmark = [pytest.mark.e2e, pytest.mark.build, pytest.mark.support_matrix]

ROOT = Path(__file__).resolve().parents[5]


def _release_tooling():
    spec = importlib.util.spec_from_file_location("run_release_fpe", ROOT / "scripts/fpe/run_release_fpe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RELEASE = _release_tooling()
REQUIRED_SELECTORS = RELEASE.required_probe_selectors()

# These temporary reports test qualification behavior, not release provenance.
TEST_SOURCE_SHA = "0" * 40
TEST_SOURCE_VERSION = "ci-fpe-sentinel-test"


@pytest.fixture(scope="module")
def wheel_sha256() -> str:
    if wheel := (os.environ.get("AISIM_TEST_WHEEL") or os.environ.get("AISIMULATE_TEST_WHEEL")):
        with Path(wheel).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    # Editable local test runs have no wheel artifact to attest to.
    return hashlib.sha256(b"local-fpe-sentinel-test-without-wheel-provenance").hexdigest()


@pytest.fixture(scope="module")
def required_probes(wheel_sha256):
    from tools.support_matrix.support_matrix import SupportMatrix, perf_database

    receipt = {
        "schema_version": 1,
        "source_branch": "main",
        "source_sha": TEST_SOURCE_SHA,
        "tooling_sha": TEST_SOURCE_SHA,
        "wheel_sha256": wheel_sha256,
    }
    plan = RELEASE.discover_required_probes(
        SupportMatrix().generate_combinations(), receipt, perf_database.resolve_query_version
    )
    # Each case qualifies one probe; enforce the nightly's whole-list contract first.
    identities = {tuple(probe[key] for key in IDENTITY) for probe in plan["probes"]}
    assert len(identities) == len(plan["probes"]), "required FPE probe identities must be unique"
    return plan["probes"]


@pytest.mark.parametrize(
    "probe_index",
    range(len(REQUIRED_SELECTORS)),
    ids=[
        "-".join((probe["model"].rsplit("/", 1)[-1], probe["system"], probe["backend"], probe["backend_version"]))
        for probe in REQUIRED_SELECTORS
    ],
)
def test_required_native_fpe_sentinel(probe_index, required_probes, tmp_path, monkeypatch, wheel_sha256):
    probe = required_probes[probe_index]
    plans = build_probe_plans(
        models={probe["model"]},
        systems={probe["system"]},
        backends={probe["backend"]},
        backend_versions={probe["backend_version"]},
        forward_models=(probe["forward_model"],),
    )
    assert plans, f"No native FPE plans were generated for the resolved required sentinel: {probe}"

    workload = ProbeWorkload()
    results, metrics = run_probe_plans(
        plans,
        workload=workload,
        source_version=TEST_SOURCE_VERSION,
        source_sha=TEST_SOURCE_SHA,
        # pytest-xdist already parallelizes these cases; avoid nested CPU pools.
        max_workers=1,
    )
    monkeypatch.setenv("FPE_WHEEL_SHA256", wheel_sha256)
    write_outputs(
        results=results,
        metrics=metrics,
        output_dir=tmp_path,
        source_version=TEST_SOURCE_VERSION,
        source_sha=TEST_SOURCE_SHA,
        workload=workload,
    )
    # Explore every topology for this identity, as nightly does. Legitimately
    # unsupported candidates may coexist with one complete passing topology.
    qualify(
        tmp_path,
        expected_shards=[{"system": probe["system"], "backend": probe["backend"]}],
        expected_sha=TEST_SOURCE_SHA,
        expected_wheel_sha256=wheel_sha256,
        required_probes=[probe],
    )
