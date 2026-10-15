# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct FPM construction checks each required phase before selecting a root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import aisimulate_core
from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.fpm_profile import load_fpm_profile

pytestmark = pytest.mark.unit


@pytest.fixture
def phase_roots(tmp_path, monkeypatch):
    """Use real metadata, Parquet loading, compilation and native root selection."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    architecture = "UnregisteredDecoderForCausalLM"
    (checkpoint / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    profile = {
        "schema_version": 1,
        "model": str(checkpoint),
        "model_revision": "phase-readiness-fixture-v1",
        "architecture": architecture,
        "context_length": 4096,
        "num_experts": 64,
        "provenance": "Synthetic SDK readiness fixture, not silicon qualification.",
        "deployments": [
            {
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.25.1",
                "tp": 1,
                "dp": 1,
                "moe_tp": 1,
                "moe_ep": 1,
                "gemm_quant_mode": "nvfp4",
                "moe_quant_mode": "nvfp4",
                "fmha_quant_mode": "fp8",
                "comm_quant_mode": "half",
                "kv_cache_dtype": "fp8",
                "resources": {
                    "weights_bytes": 100,
                    "activations_bytes": 20,
                    "runtime_overhead_bytes": 30,
                    "comm_overhead_bytes": 50,
                    "kv_bytes_per_token": 10,
                    "cache_layout": "linear",
                    "max_num_tokens": 8192,
                    "max_batch_size": 256,
                    "provenance": "Synthetic per-rank resource declarations.",
                },
            }
        ],
    }
    config = {
        "model": profile["model"],
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "worker_type": "aggregated",
        "tp": 1,
        "attention_dp": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "estimation_mode": "fpm_interpolation",
        "fallback_policy": "deny",
        "fpm_profile": profile,
        "estimator_config": {"fpm_interpolation": {"method": "direct"}},
    }
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    def make_root(name, *, prefill="genuine", decode="genuine"):
        root = tmp_path / name
        root.mkdir()
        packaged = Path(aisimulate_core.__file__).parent / "systems/h200_sxm.yaml"
        (root / packaged.name).write_bytes(packaged.read_bytes())
        identity = load_fpm_profile(profile).deployments[0].model_dump(mode="json", exclude={"resources"})
        rows = [
            {
                **identity,
                "model_path": profile["model"],
                "cell_id": f"synthetic-{phase}",
                "weight_quantization": "synthetic",
                "workload_kind": phase,
                "partition_policy": "balanced_v1",
                "batch_size": 1,
                "total_prefill_tokens": 1 if phase == "prefill" else 0,
                "total_kv_read_tokens": 0 if phase == "prefill" else 1,
                "latency_ms": 2.0 if phase == "prefill" else 3.0,
                "kv_seed_regime": "real_kv" if kind == "genuine" else "fake_fallback",
            }
            for phase, kind in (("prefill", prefill), ("decode", decode))
            if kind != "absent"
        ]
        path = root / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "schema_name": "aic_fpm_forward_perf",
                    "schema_version": 6,
                    "coordinate_system": "iteration_totals_balanced_v1",
                    "measurement_policy": "dynamo_native_single_sample_v1",
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "backend_version": "0.25.1",
                    "row_count": len(rows),
                    "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        )
        return str(root)

    return config, make_root


@pytest.mark.parametrize("missing_phase", ["prefill", "decode"])
@pytest.mark.parametrize("rows", ["absent", "fake"])
def test_direct_constructor_rejects_missing_genuine_phase(phase_roots, missing_phase, rows):
    config, make_root = phase_roots
    root = make_root("incomplete", **{missing_phase: rows})
    config["systems_paths"] = [root]
    with pytest.raises(PerfDataNotAvailableError, match=f"direct FPM {missing_phase}") as exc:
        RustForwardPassPerfModel.best_available(config)
    message = str(exc.value)
    assert "no genuine measurements" in message
    assert f"Collect genuine {missing_phase} FPM rows" in message
    assert root in message


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
@pytest.mark.parametrize("missing_rows", ["absent", "fake"])
def test_single_role_requires_only_its_own_genuine_phase(phase_roots, worker_type, missing_rows):
    config, make_root = phase_roots
    other = "decode" if worker_type == "prefill" else "prefill"
    root = make_root("single-role", **{other: missing_rows})
    config.update(worker_type=worker_type, systems_paths=[root])
    config["fpm_profile"]["deployments"][0]["worker_type"] = worker_type
    queries = {
        "prefill": {"num_prefill_requests": 1, "sum_prefill_tokens": 1},
        "decode": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1},
    }
    for request in (config, json.loads(json.dumps(config))):
        model = RustForwardPassPerfModel.best_available(request)
        try:
            assert model.diagnostics()["readiness"] == "ready"
            assert model.estimate_forward_pass_time_ms({"scheduled_requests": queries[worker_type]}) == (
                2.0 if worker_type == "prefill" else 3.0
            )
            with pytest.raises(PerfDataNotAvailableError):
                model.estimate_forward_pass_time_ms({"scheduled_requests": queries[other]})
        finally:
            model.close()


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
def test_single_role_does_not_accept_other_roles_measurements(phase_roots, worker_type):
    config, make_root = phase_roots
    root = make_root("wrong-role", **{worker_type: "absent"})
    config.update(worker_type=worker_type, systems_paths=[root])
    config["fpm_profile"]["deployments"][0]["worker_type"] = worker_type
    with pytest.raises(PerfDataNotAvailableError, match=f"direct FPM {worker_type}"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
def test_single_role_pins_root_with_its_phase_without_merging_other_root(phase_roots, worker_type):
    config, make_root = phase_roots
    other = "decode" if worker_type == "prefill" else "prefill"
    wrong = make_root("wrong", **{worker_type: "absent"})
    matching = make_root("matching", **{other: "absent"})
    config.update(worker_type=worker_type, systems_paths=[wrong, matching])
    config["fpm_profile"]["deployments"][0]["worker_type"] = worker_type
    model = RustForwardPassPerfModel.best_available(config)
    try:
        provenance = model.diagnostics()["provenance"]
        assert provenance["selected_systems_root"] == matching
        assert provenance["config"]["systems_paths"] == [matching]
        absent_workload = (
            {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}
            if other == "decode"
            else {"num_prefill_requests": 1, "sum_prefill_tokens": 1}
        )
        with pytest.raises(PerfDataNotAvailableError):
            model.estimate_forward_pass_time_ms({"scheduled_requests": absent_workload})
    finally:
        model.close()


@pytest.mark.parametrize("missing_phase", ["prefill", "decode"])
@pytest.mark.parametrize("rows", ["absent", "fake"])
@pytest.mark.parametrize("fallback", ["deny", "allow"])
def test_direct_constructor_selects_later_root_with_both_required_phases(phase_roots, missing_phase, rows, fallback):
    config, make_root = phase_roots
    first = make_root("incomplete", **{missing_phase: rows})
    second = make_root("usable")
    config.update(systems_paths=[first, second], fallback_policy=fallback)
    model = RustForwardPassPerfModel.best_available(config)
    before = model.diagnostics()
    assert before["readiness"] == "ready"
    provenance = before["provenance"]
    assert provenance["selected_systems_root"] == second
    assert provenance["config"]["systems_paths"] == [second]
    assert provenance["selected_estimation_mode"] == "fpm_interpolation"
    assert provenance["config"]["estimator_config"]["fpm_interpolation"]["method"] == "direct"
    # The selected root's exact, hand-declared synthetic timings remain queryable.
    assert (
        model.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}
        )
        == 2.0
    )
    assert (
        model.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}
        )
        == 3.0
    )
    with pytest.raises(PerfDataNotAvailableError, match="direct"):
        model.estimate_forward_pass_time_ms(
            {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 64}}
        )
    assert model.diagnostics() == before
