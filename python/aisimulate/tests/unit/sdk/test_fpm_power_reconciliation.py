# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISIM-1973 V1: published FPM power reconciles with operation energy.

Synthetic contract tests, not measured accuracy. Each test writes a
schema-6/7/8 pair from the documented ``aic_fpm_forward_perf`` schema and
checks that the public outputs agree with the per-operation energy and
latency the engine returned:

- the legacy static path publishes ``power_w = sum(energy) / sum(latency)``
  at full coverage, and disaggregated sweep rows weight each worker's watts
  by its time;
- latency-only (v7, all-null v8) pairs publish ``None`` with latency unchanged;
- unpowered low-KV decode rows (D2/D8) reduce coverage by the latency of
  the queries that touch them;
- a draft-scheme static ``cli estimate`` withholds watts at full coverage;
- Replay and unified ``aisimulate predict``/``recommend`` publish no FPM
  power (D4), for one attention-DP rank, for several, and for a fitted
  regression role restored from a snapshot;
- aggregate ``cli exp`` sweeps, with and without a draft scheme, publish no
  watts and keep v7 latency.

The SDK publication gates are tested in ``test_fpm_forward.py``
(``TestFPMPowerPublication``, ``TestFPMStaticPowerCoverage``); this module
reaches them through the public legacy and unified entry points.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from aisimulate.main import main
from aisimulate_core.sdk.operations.fpm_forward import _CELL_MATCH_COLUMNS

from . import test_fpm_forward
from .test_fpm_forward import (
    _POWER_VERSION,
    BACKEND,
    SYSTEM,
    _agg_rows,
    _full_power,
    _power_rows,
    _qwen_fpm,
    _row,
    _run_static,
    _trajectory_rows,
    _write_pair,
)

pytestmark = pytest.mark.unit

# The shared ``write(model, power_w, rows=None) -> PerfDatabase`` fixture.
fpm_power_root = test_fpm_forward.fpm_power_root
_EAGLE3 = {
    "method": "eagle3",
    "params": {"num_speculative_tokens": 3},
    "draft_config": test_fpm_forward.TestForwardModelRewrite._EAGLE3_CONFIG,
    "accepted_tokens": 1.8,
}

_PREFILL_MS, _PREFILL_W = 40.0, 300.0  # the ``_trajectory_rows`` prefill anchor
# Static decode with isl 512 and the default stride 32 visits s = 513, 545,
# 577 (osl 66): repeats 32, 32 and 1. Each step queries total KV 2 * s at batch 2.
_ISL, _OSL, _STRIDE = 512, 66, 32
_STEPS = ((513, 32), (545, 32), (577, 1))


def _decode_pair(model, decode: dict[int, tuple[float, float | None]]):
    """``_trajectory_rows`` with decode rows at ``{total_kv: (latency_ms, watts)}``.

    Returns ``(rows, power_w)`` for ``fpm_power_root``.
    """
    rows, watts = _trajectory_rows(model, [(kv // 2, ms, w) for kv, (ms, w) in decode.items()])
    return rows, lambda row: watts[row["workload_kind"], row["total_kv_read_tokens"]]


def _static(model, database, mode: str = "static", *, isl: int = _ISL, osl: int = _OSL):
    return _run_static(model, database, mode=mode, isl=isl, osl=osl)


def _cli_static(database, **overrides):
    """``aiconfigurator cli estimate --estimate-mode static`` on ``database``'s root."""
    from aisimulate.legacy_cli.api import cli_estimate

    kwargs = {
        "mode": "static",
        "backend_name": BACKEND,
        "backend_version": _POWER_VERSION,
        "systems_paths": database.systems_root,
        "forward_model": "fpm",
        "isl": _ISL,
        "osl": _OSL,
        "batch_size": 2,
        "gemm_quant_mode": "bfloat16",
        "kvcache_quant_mode": "bfloat16",
        "stride": _STRIDE,
        **overrides,
    }
    return cli_estimate("Qwen/Qwen3-0.6B", SYSTEM, **kwargs)


def _energy_and_latency(summary) -> tuple[float, float]:
    energy = sum(summary.get_context_energy_wms_dict().values()) + sum(
        summary.get_generation_energy_wms_dict().values()
    )
    latency = sum(summary.get_context_latency_dict().values()) + sum(summary.get_generation_latency_dict().values())
    return energy, latency


# Exact decode anchors at each visited step, with different latency and power.
_EXACT = {1026: (6.0, 450.0), 1090: (7.0, 500.0), 1154: (8.0, 550.0)}


def test_static_watts_equal_operation_energy_over_latency(fpm_power_root):
    """At full coverage the published watts are summed energy over summed latency."""
    model = _qwen_fpm()
    rows, power_w = _decode_pair(model, _EXACT)
    database = fpm_power_root(model, power_w, rows=rows)
    summary = _static(model, database)

    # Hand-derived: decode = 32 x 6 + 32 x 7 + 1 x 8 = 424 ms and
    # 32 x 6 x 450 + 32 x 7 x 500 + 1 x 8 x 550 = 202,800 W-ms; prefill is
    # 40 ms at 300 W = 12,000 W-ms.
    assert summary.get_generation_latency_dict()["fpm_forward_decode"] == pytest.approx(424.0)
    assert summary.get_generation_energy_wms_dict()["fpm_forward_decode"] == pytest.approx(202_800.0)
    assert summary.get_context_energy_wms_dict()["fpm_forward_prefill"] == pytest.approx(12_000.0)
    expected = (12_000.0 + 202_800.0) / (40.0 + 424.0)
    energy, latency = _energy_and_latency(summary)
    assert summary.get_power_data_coverage() == 1.0
    assert summary.get_result_dict()["power_w"] == pytest.approx(energy / latency)
    assert summary.get_result_dict()["power_w"] == pytest.approx(expected)
    assert summary.get_e2e_power_avg() == pytest.approx(expected)
    assert summary.get_context_power_avg() == pytest.approx(_PREFILL_W)
    assert summary.get_generation_power_avg() == pytest.approx(202_800.0 / 424.0)
    # The public legacy CLI publishes the same value.
    assert _cli_static(database).power_w == pytest.approx(expected)


def test_draft_scheme_static_cli_estimate_withholds_watts_at_full_coverage(fpm_power_root):
    """Step 5b through the public legacy route, where coverage alone would publish watts.

    The aggregate sweeps below withhold for the aggregate rule (D10) first, so
    only a static or disaggregated route can show the draft-scheme rule.
    """
    rows = _power_rows(_qwen_fpm())
    rows[1]["batch_size"] = 8  # two requests verify four queries each
    database = fpm_power_root(_qwen_fpm(), _full_power, rows=rows)
    with pytest.warns(FutureWarning, match="Hybrid FPM/OP"):
        estimate = _cli_static(database, osl=2, speculative=_EAGLE3)

    assert estimate.raw["power_coverage"] >= 0.9
    energy = {**estimate.summary.get_context_energy_wms_dict(), **estimate.summary.get_generation_energy_wms_dict()}
    assert any(name.startswith("draft_") and value > 0 for name, value in energy.items())
    assert energy["fpm_forward_prefill"] == pytest.approx(40.0 * 300.0)
    assert energy["fpm_forward_decode"] == pytest.approx(6.0 * 450.0)
    assert estimate.power_w is None
    assert estimate.raw["power_w"] is None
    assert estimate.ttft > 0 and estimate.tpot > 0


def _rewrite_as_schema(database, rows: list[dict], schema_version: int) -> None:
    """Replace the pair under ``database``'s root before any query loads it."""
    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY
    from aisimulate_core.sdk.rust_engine_step import _engine_handle_cache_clear

    data_dir = Path(database.systems_root) / "data" / SYSTEM / BACKEND / _POWER_VERSION
    execution = dict(zip(EXECUTION_COLUMNS, LEGACY_EXECUTION_IDENTITY, strict=True))
    _write_pair(
        str(data_dir),
        [dict(row, **execution) for row in rows],
        sidecar_overrides={"schema_version": schema_version, "backend_version": _POWER_VERSION},
    )
    _engine_handle_cache_clear()


@pytest.mark.parametrize("schema", ["v7", "v8_all_null"])
def test_latency_only_pair_publishes_null_power_with_bit_identical_latency(fpm_power_root, schema):
    """v7 and all-null v8 pairs (v6 is in ``TestFPMPowerPublication``) through the public CLI."""
    model = _qwen_fpm()
    rows, power_w = _decode_pair(model, _EXACT)
    powered = _static(model, fpm_power_root(model, power_w, rows=rows))
    if schema == "v8_all_null":
        database = fpm_power_root(model, lambda row: None, rows=rows)
    else:
        database = fpm_power_root(model, None, rows=rows)
        _rewrite_as_schema(database, rows, 7)
    summary = _static(model, database)

    result = summary.get_result_dict()
    assert result["power_w"] is None
    assert summary.get_e2e_power_avg() is None
    assert summary.get_power_data_coverage() == 0.0
    assert _energy_and_latency(summary)[0] == 0.0
    # Power never feeds latency: every latency output is bit-identical
    # to the fully powered v8 pair with the same rows.
    for getter in ("get_context_latency_dict", "get_generation_latency_dict"):
        assert getattr(summary, getter)() == getattr(powered, getter)()
    for key in ("ttft", "tpot", "seq/s", "tokens/s/gpu"):
        assert result[key] == powered.get_result_dict()[key], key
    estimate = _cli_static(database)
    assert (estimate.power_w, estimate.raw["power_coverage"]) == (None, 0.0)
    assert (estimate.ttft, estimate.tpot) == (result["ttft"], result["tpot"])


# D2/D8: decode windows that would drift more than 10% of their KV read keep
# latency and publish no power, so the low-KV end of every decode curve,
# down to its floor of about 2 tokens per request, is unpowered.
_LOW_KV = {4: (5.0, None), 1000: (6.0, None), 1060: (6.5, 450.0), 1120: (7.0, 460.0), 1200: (7.5, 470.0)}


def _one_step(model, database, s: int):
    """A static decode of one step at sequence length ``s`` (total KV ``2 * s``)."""
    summary = _static(model, database, mode="static_gen", isl=s - 1, osl=2)
    return (
        summary.get_generation_latency_dict()["fpm_forward_decode"],
        summary.get_generation_energy_wms_dict()["fpm_forward_decode"],
    )


def test_low_kv_queries_that_touch_unpowered_rows_have_no_energy(fpm_power_root):
    """Per query (D8): any unpowered row in the interpolation zeroes energy, never latency."""
    model = _qwen_fpm()
    rows, power_w = _decode_pair(model, _LOW_KV)
    partial = fpm_power_root(model, power_w, rows=rows)
    all_powered = _decode_pair(model, {kv: (ms, 450.0) for kv, (ms, _) in _LOW_KV.items()})[1]
    full = fpm_power_root(model, all_powered, rows=rows)
    for s, _ in _STEPS:
        latency, energy = _one_step(model, partial, s)
        assert latency == _one_step(model, full, s)[0]
        if 2 * s < 1060:
            # 1026 interpolates between the unpowered 1000 row and the powered 1060 row.
            assert energy == 0.0
        else:
            # Both neighbours carry power; power follows the latency weights.
            assert 450.0 * latency < energy < 470.0 * latency


def test_low_kv_unpowered_rows_reduce_coverage_by_their_query_latency(fpm_power_root):
    """Coverage is the latency share of the queries that touch only powered rows.

    ``TestFPMStaticPowerCoverage`` uses exact anchors; here queries
    interpolate next to unpowered low-KV rows and the gate is also read
    through the public ``cli_estimate``. The decode steps query total KV
    1026, 1090 and 1154. 1026 lies between
    the unpowered 1000 row and the powered 1060 row, so its interpolation
    touches an unpowered row and the query is uncovered. 1090 (1060-1120)
    and 1154 (1120-1200) touch only powered rows. With per-step latencies
    L1, L2, L3 and repeats 32, 32, 1:

        coverage = (40 + 32 * L2 + L3) / (40 + 32 * (L1 + L2) + L3)

    Each L is the latency of a one-step static run at the same step, so the
    expected value follows from the rows alone, not from the folded total.
    """
    model = _qwen_fpm()
    rows, power_w = _decode_pair(model, _LOW_KV)
    database = fpm_power_root(model, power_w, rows=rows)

    l1, l2, l3 = (_one_step(model, database, s)[0] for s, _ in _STEPS)
    covered = _PREFILL_MS + 32 * l2 + l3
    total = _PREFILL_MS + 32 * (l1 + l2) + l3
    coverage = covered / total
    assert coverage < 0.9

    summary = _static(model, database)
    assert summary.get_generation_latency_dict()["fpm_forward_decode"] == pytest.approx(total - _PREFILL_MS)
    assert summary.get_power_data_coverage() == pytest.approx(coverage)
    # SDK gate (step 5c): watts withheld, energy and latency kept.
    assert summary.get_result_dict()["power_w"] is None
    assert summary.get_generation_power_avg() is None
    assert summary.get_generation_energy_wms_dict()["fpm_forward_decode"] > 0
    # Legacy CLI gate: latency still shown, power unavailable with the true coverage.
    estimate = _cli_static(database)
    assert estimate.power_w is None
    assert estimate.raw["power_coverage"] == pytest.approx(coverage)
    assert estimate.ttft == summary.get_result_dict()["ttft"]
    assert estimate.tpot == summary.get_result_dict()["tpot"]


# ---------------------------------------------------------------------------
# Replay and unified CLI: FPM power stays unsupported in V1 (D4, AISIM-2001)
# ---------------------------------------------------------------------------

_UNSUPPORTED_REASON = "whole-model FPM and latency-only providers are unsupported"


@pytest.fixture
def powered_engine_config(fpm_power_root, monkeypatch):
    """Unified ``engine``/``traffic`` mapping over a fully powered v8 pair.

    The pair holds one powered grid per attention-DP size. Returns
    ``build(attention_data=1) -> dict``.
    """

    def build(attention_data: int = 1) -> dict:
        rows = []
        for dp in (1, 2):
            grid = _agg_rows(_qwen_fpm(attention_dp_size=dp))
            rows += [dict(row, cell_id=f"{row['cell_id']}-dp{dp}") for row in grid]
        model = _qwen_fpm()
        database = fpm_power_root(model, lambda row: 450.0, rows=rows)
        root = Path(database.systems_root)
        monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(root))
        monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
        parquet = root / "data" / SYSTEM / BACKEND / _POWER_VERSION / "fpm_forward_perf.parquet"
        table = pq.read_table(parquet)
        assert json.loads(parquet.with_suffix(".metadata.json").read_text())["schema_version"] == 8
        assert table["power_w"].null_count == 0
        return {
            "engine": {
                "mode": "aggregated",
                "model": "Qwen/Qwen3-0.6B",
                "hardware": SYSTEM,
                "backend": BACKEND,
                "backend_version": _POWER_VERSION,
                "context_length": 1024,
                "workers": {
                    "aggregated": {
                        "parallelism": {"replicas": 1, "attention_data": attention_data},
                        "kv_cache": {"prefix_caching": False, "capacity": {"type": "fixed", "blocks": 128}},
                        "timing": {
                            "estimation_mode": "fpm_interpolation",
                            "estimator_config": {"fpm_interpolation": {"fpm_parquet_path": str(parquet)}},
                        },
                    }
                },
            },
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 512, "output_tokens": 3},
                "load": {"type": "concurrency", "concurrency": 2},
                "stop": {"requests": 2},
            },
        }

    return build


def _assert_replay_power_unsupported(diagnostics: dict) -> None:
    assert diagnostics["publication_status"] == "unsupported"
    assert diagnostics["power_w"] is None
    assert diagnostics["power_coverage"] is None
    assert _UNSUPPORTED_REASON in diagnostics["unavailable_reason"]


@pytest.mark.parametrize("attention_data", [1, 2], ids=["dp1", "attention_dp2"])
def test_unified_predict_keeps_powered_fpm_unsupported(powered_engine_config, tmp_path, capsys, attention_data):
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(powered_engine_config(attention_data)))
    out = tmp_path / "out"
    assert main(["predict", "-c", str(path), "--detail", "energy", "--format", "json", "--output-dir", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    summary = payload["summary"]
    assert summary["completed_requests"] == 2
    assert summary["mean_ttft_ms"] > 0 and summary["mean_tpot_ms"] > 0
    # Every row of the v8 pair carries power, yet the Replay summary keeps
    # both values null (never 0 W or 0%) and states why.
    assert summary["power_w"] is None
    assert summary["power_coverage"] is None
    energy = payload["details"]["sections"]["energy"]
    assert energy["status"] == "unsupported"
    _assert_replay_power_unsupported(energy["diagnostics"])
    [saved] = out.rglob("prediction.json")
    report = json.loads(saved.read_text())
    assert (report["power_w"], report["power_coverage"]) == (None, None)
    _assert_replay_power_unsupported(report["power_diagnostics"])


def test_unified_recommend_publishes_no_fpm_power(powered_engine_config):
    from aisimulate.config import CoreRecommendationConfig
    from aisimulate.recommend import run_recommendation
    from aisimulate.runner import EngineReplayRunnerFactory

    raw = powered_engine_config()
    worker = raw["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"preset": False, "replicas": 1, "tensor": 1, "pipeline": 1, "attention_data": 1}
    raw["optimization"] = {"target": "throughput", "constraints": {"max_candidate_gpus": 1}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1}
    result = run_recommendation(
        CoreRecommendationConfig.model_validate(raw),
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
        show_progress=False,
    )
    assert result.counts.feasible == 1, [record.reason for record in result.candidates]
    [candidate] = result.selected_candidates
    assert candidate.config["forward_pass_estimators"]["agg"]["config"]["estimation_mode"] == "fpm_interpolation"
    assert candidate.metrics["mean_tpot_ms"] > 0
    assert (candidate.metrics["power_w"], candidate.metrics["power_coverage"]) == (None, None)
    [record] = result.candidates
    power = record.provenance.power
    assert power["power_w"] is None
    assert power["publication_status"] == "unavailable"
    assert power["unavailable_reason"] == "energy provider or topology unsupported"


def test_replay_of_restored_regression_role_keeps_power_unsupported(tmp_path):
    from aisimulate import _runtime
    from aisimulate_core.sdk import RustForwardPassPerfModel

    from .test_fpm_state_snapshot import _regression_config, _stream

    model = RustForwardPassPerfModel.best_available(_regression_config())
    model.tune_with_fpms(_stream(7, 400))
    saved = model.save_state_snapshot(tmp_path / "regression")
    restored = RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "regression")
    assert restored.diagnostics()["provenance"]["selected_estimation_mode"] == "fpm_regression"
    model.close()
    restored.close()

    payload = {
        "spec": {
            "version": 1,
            "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
            "engine": {
                "rank": {
                    "backend": "vllm",
                    "block_size": 64,
                    "num_gpu_blocks": 4096,
                    "timing_model": {"type": "external", "provider": "aic", "config": saved},
                }
            },
            "requests": [{"id": "r0", "arrival_time_ms": 0.0, "input_tokens": 512, "output_tokens": 3}],
        },
    }
    report = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    assert report["completed_requests"] == 1
    assert report["mean_ttft_ms"] > 0
    assert report["power_w"] is None
    assert report["power_coverage"] is None
    _assert_replay_power_unsupported(report["power_diagnostics"])


# ---------------------------------------------------------------------------
# End-to-end legacy sweeps (`aiconfigurator cli exp`)
# ---------------------------------------------------------------------------


def _sweep_rows(model, batches) -> list[dict]:
    """The ``_agg_rows`` grid, extended to every decode batch a sweep can query."""
    identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))
    rows = _agg_rows(model)
    rows += [
        dict(
            _row("decode", batch, 0, kv, 5.0 + kv / 1024, model_path=model.model_path, identity=identity),
            backend_version=_POWER_VERSION,
            cell_id=f"fpm-sweep-{batch}-{kv}",
        )
        for batch in batches
        for kv in (2, 512, 1024, 2048, 4096)
        if batch > 2
    ]
    return rows


def _experiment(serving_mode: str, **overrides) -> dict:
    """One `cli exp` experiment for Qwen3-0.6B on the synthetic pair, bf16, tp 1."""
    identity = {
        "model_path": "Qwen/Qwen3-0.6B",
        "system_name": SYSTEM,
        "backend_name": BACKEND,
        "backend_version": _POWER_VERSION,
        "gemm_quant_mode": "bfloat16",
        "kvcache_quant_mode": "bfloat16",
    }
    if serving_mode == "disagg":
        identity = {f"{role}_{key}": value for role in ("prefill", "decode") for key, value in identity.items()}
    return {
        "serving_mode": serving_mode,
        **identity,
        "total_gpus": 1 if serving_mode == "agg" else 2,
        "isl": 512,
        "osl": 64,
        "ttft": 5000,
        "tpot": 200,
        "forward_model": "fpm",
        **overrides,
    }


def _sweep(tmp_path: Path, name: str, database, experiment: dict):
    """Run ``cli_exp`` against ``database``'s systems root and save its report."""
    from aisimulate.legacy_cli.api import cli_exp
    from aisimulate.sdk.perf_database import get_systems_paths, set_systems_paths

    previous = get_systems_paths()
    set_systems_paths(database.systems_root)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # hybrid FPM/OP deprecation
            result = cli_exp(config={"fpm": experiment}, save_dir=str(tmp_path / name))
    finally:
        set_systems_paths(previous)
    [saved] = (tmp_path / name).rglob("fpm")
    return result, saved


@pytest.mark.parametrize("scheme", ["plain", "eagle3"])
def test_legacy_aggregate_sweep_withholds_fpm_power_and_keeps_v7_latency(fpm_power_root, tmp_path, monkeypatch, scheme):
    """Aggregate FPM sweeps, with and without a draft scheme, finish with no watts anywhere.

    The aggregate rule (D10) withholds every row, so these sweeps cannot show
    the draft-scheme rule (step 5b): the legacy sweep accepts scheme-based
    speculation only for aggregated serving. Step 5b is pinned by
    ``test_draft_scheme_static_cli_estimate_withholds_watts_at_full_coverage``
    and, at the SDK level, by ``TestFPMPowerPublication``. The ``eagle3`` case
    pins that a draft-scheme sweep runs and keeps its v7 latency.
    """
    import pandas as pd

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    experiment = _experiment("agg")
    if scheme == "eagle3":
        experiment["speculative"] = _EAGLE3
    model = _qwen_fpm()
    rows = _sweep_rows(model, batches=(4, 8))
    powered, powered_dir = _sweep(tmp_path, "v8", fpm_power_root(model, lambda row: 450.0, rows=rows), experiment)
    latency_only = fpm_power_root(model, None, rows=rows)
    _rewrite_as_schema(latency_only, rows, 7)
    v7, _ = _sweep(tmp_path, "v7", latency_only, experiment)
    assert (powered.tasks["fpm"]._speculation_config is not None) is (scheme == "eagle3")

    for frames in (powered.pareto_fronts, powered.best_configs):
        assert not frames["fpm"].empty
        assert frames["fpm"]["power_w"].isna().all()
    for csv in ("pareto.csv", "best_config_topn.csv"):
        saved = pd.read_csv(powered_dir / csv)
        assert not saved.empty and saved["power_w"].isna().all(), csv
    # Power never feeds latency or throughput: every non-power column of the
    # v8 sweep matches the same sweep over a latency-only v7 pair.
    for frames, reference in ((powered.pareto_fronts, v7.pareto_fronts), (powered.best_configs, v7.best_configs)):
        columns = [column for column in frames["fpm"].columns if "power" not in column]
        assert {"ttft", "tpot", "tokens/s/gpu"} <= set(columns)
        pd.testing.assert_frame_equal(
            frames["fpm"][columns].reset_index(drop=True), reference["fpm"][columns].reset_index(drop=True)
        )
    assert powered.best_throughputs == v7.best_throughputs


def test_legacy_disagg_sweep_rows_reconcile_worker_watts(fpm_power_root, tmp_path, monkeypatch):
    """Disaggregated sweep rows weight 300 W prefill and 450 W decode by their time."""
    import pandas as pd

    from aisimulate.sdk.picking import disagg_power_w

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    model = _qwen_fpm()
    database = fpm_power_root(model, _watts_by_phase, rows=_sweep_rows(model, batches=(4, 8, 16, 32, 64)))
    result, saved_dir = _sweep(tmp_path, "v8", database, _experiment("disagg"))

    frame = result.pareto_fronts["fpm"]
    assert len(frame) > 1
    for _, row in frame.iterrows():
        decode_ms = row["tpot"] * (row["osl"] - 1)
        expected = (_PREFILL_W * row["ttft"] + 450.0 * decode_ms) / (row["ttft"] + decode_ms)
        assert row["power_w"] == pytest.approx(expected, rel=1e-4)
        assert row["power_w"] == pytest.approx(disagg_power_w(_PREFILL_W, 450.0, row["ttft"], decode_ms), rel=1e-4)
    saved = pd.read_csv(saved_dir / "pareto.csv")
    assert saved["power_w"].notna().all()


def _watts_by_phase(row: dict) -> float:
    return _PREFILL_W if row["workload_kind"] == "prefill" else 450.0
