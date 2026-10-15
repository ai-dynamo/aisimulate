# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE selection metadata through the public database and native query path."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate_core.sdk.common import MoEQuantMode
from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.engine_table_view import fetch_table_view
from aisimulate_core.sdk.errors import EmpiricalNotImplementedError, PerfDataNotAvailableError
from aisimulate_core.sdk.perf_database import get_database_view

pytestmark = pytest.mark.unit

SYSTEMS = Path(__file__).resolve().parents[4] / "src/aisimulate_core/systems"
SOURCE = "sglang_flashinfer_trtllm_moe"


@pytest.fixture
def systems_root(tmp_path):
    shutil.copyfile(SYSTEMS / "b200_sxm.yaml", tmp_path / "b200_sxm.yaml")
    return tmp_path


def _row(**overrides):
    return {
        "moe_dtype": "fp8_block",
        "num_tokens": 32,
        "hidden_size": 8192,
        "inter_size": 2048,
        "topk": 8,
        "num_experts": 256,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "distribution": "uniform",
        "kernel_source": SOURCE,
        "latency": 0.25,
        **overrides,
    }


def _write(root, rows, *, backend="sglang", version="0.5.17"):
    path = root / f"data/b200_sxm/moe/{backend}/{version}/moe_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _database(root, *, mode="SILICON", shared_layer=False):
    # Synthetic measurement rows intentionally omit campaign provenance.
    return get_database_view(
        "b200_sxm",
        "sglang",
        "0.5.17",
        systems_paths=str(root),
        database_mode=mode,
        shared_layer=shared_layer,
        strict_provenance=False,
    )


def _query(db, *, source=None, **overrides):
    fields = {
        "name": "eligibility fixture",
        "scale_factor": 1.0,
        "hidden_size": 8192,
        "inter_size": 2048,
        "topk": 8,
        "num_experts": 256,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        "attention_dp_size": 1,
        "quant_mode": "fp8_block",
        "workload_distribution": "uniform",
        "is_gated": True,
        "moe_backend": None,
        "moe_kernel_source": source,
        "enable_eplb": False,
        "is_context": False,
        **overrides,
    }
    return EngineHandle.for_database(db, systems_path=db.systems_root).evaluate_ops_json(
        json.dumps([{"Moe": fields}]),
        is_context=False,
        batch_size=1,
        s=1,
        x=32,
    )[0]


def test_opt_in_only_table_requires_exact_source(systems_root):
    _write(systems_root, [_row(default_eligible=False)])
    db = _database(systems_root)

    # Synthetic measured latency: the named lane must remain usable even
    # when the table has no automatic-selection rows at all.
    assert _query(db, source=SOURCE)[1] == 0.25
    with pytest.raises(PerfDataNotAvailableError):
        _query(db)


def test_default_views_exclude_opt_in_rows(systems_root):
    rows = [_row(default_eligible=False)]
    path = _write(systems_root, rows)
    db = _database(systems_root)

    assert fetch_table_view(db, "_moe_data") == {}
    assert db.supported_quant_mode["moe"] == []
    assert db.legacy_moe_compute_coverage(8192, 2048, 8, 256, MoEQuantMode.fp8_block) == set()
    # Raw parquet enumeration keeps the exact source, metadata and measurement.
    assert pq.read_table(path).to_pylist() == rows


@pytest.mark.parametrize("flag", [None, True, False])
@pytest.mark.parametrize("quant", ["fp8_block", "nvfp4"])
def test_absent_and_true_keep_legacy_default_sources(systems_root, flag, quant):
    row = _row(moe_dtype=quant)
    if flag is not None:
        row["default_eligible"] = flag
    _write(systems_root, [row])
    db = _database(systems_root)

    assert _query(db, source=SOURCE, quant_mode=quant)[1] == 0.25
    if flag is False:
        with pytest.raises(PerfDataNotAvailableError):
            _query(db, quant_mode=quant)
    else:
        assert _query(db, quant_mode=quant)[1] == 0.25


@pytest.mark.parametrize("mode", ["EMPIRICAL", "HYBRID"])
@pytest.mark.parametrize("overrides", [{}, {"hidden_size": 4096}, {"quant_mode": "bfloat16"}])
def test_empirical_transfers_cannot_use_opt_in_rows(systems_root, mode, overrides):
    _write(systems_root, [_row(default_eligible=False)])
    db = _database(systems_root, mode=mode)

    assert _query(db, source=SOURCE)[1] == pytest.approx(0.25)
    with pytest.raises(EmpiricalNotImplementedError):
        _query(db, **overrides)


@pytest.mark.parametrize("mode", ["SILICON", "EMPIRICAL", "HYBRID"])
@pytest.mark.parametrize("flag", [None, "false", 0, 1.0])
def test_malformed_eligibility_fails_closed(systems_root, mode, flag):
    _write(systems_root, [_row(default_eligible=flag)])
    db = _database(systems_root, mode=mode)

    # InvalidPerfData crosses the FFI as ValueError, never a coverage miss.
    for source in (None, SOURCE):
        with pytest.raises(ValueError, match="default_eligible.*non-null Boolean"):
            _query(db, source=source)
    with pytest.raises(ValueError, match="default_eligible.*non-null Boolean"):
        fetch_table_view(db, "_moe_data")


@pytest.mark.parametrize("source", [None, "", " \t", 7])
def test_opt_in_requires_an_exact_nonblank_source(systems_root, source):
    _write(systems_root, [_row(default_eligible=False, kernel_source=source)])
    db = _database(systems_root)
    with pytest.raises(ValueError, match="default_eligible=false.*kernel_source"):
        _query(db)


def test_opt_in_requires_a_source_column(systems_root):
    row = _row(default_eligible=False)
    del row["kernel_source"]
    _write(systems_root, [row])
    with pytest.raises(ValueError, match="default_eligible=false.*kernel_source"):
        _query(_database(systems_root))


@pytest.mark.parametrize("mode", ["SILICON", "EMPIRICAL", "HYBRID"])
@pytest.mark.parametrize("eligible", [False, True])
def test_low_latency_eligibility_keeps_the_named_lane(systems_root, mode, eligible):
    low_latency = "moe_torch_flow_min_latency"
    _write(
        systems_root,
        [
            _row(moe_dtype="nvfp4", kernel_source=low_latency, default_eligible=eligible, latency=0.125),
            _row(moe_dtype="nvfp4", kernel_source="moe_torch_flow", default_eligible=True),
        ],
    )
    db = _database(systems_root, mode=mode)
    assert _query(db, source=low_latency, quant_mode="nvfp4")[1] == pytest.approx(0.125)
    assert _query(db, quant_mode="nvfp4")[1] == pytest.approx(0.125 if eligible else 0.25)
    assert bool(fetch_table_view(db, "_moe_low_latency_data")) is eligible


@pytest.mark.parametrize("shared", [False, True])
def test_eligibility_preserves_source_priority(systems_root, shared):
    _write(
        systems_root,
        [
            _row(default_eligible=False, latency=0.125),
            _row(kernel_source="triton", default_eligible=True, moe_tp_size=2),
            # Two eligible sources may legitimately own disjoint keys.
            _row(kernel_source="other", default_eligible=True, moe_tp_size=4),
        ],
    )
    _write(
        systems_root,
        [
            _row(default_eligible=False, latency=0.1),
            _row(kernel_source="triton", default_eligible=True, latency=0.5),
            _row(kernel_source="triton", default_eligible=True, latency=0.75, moe_tp_size=2),
        ],
        version="0.5.16",
    )
    db = _database(systems_root, shared_layer=shared)
    assert _query(db, source=SOURCE)[1] == 0.125
    assert _query(db, moe_tp_size=2)[1] == 0.25
    assert _query(db, moe_tp_size=4)[1] == 0.25
    if shared:
        assert _query(db)[1] == 0.5
    else:
        with pytest.raises(PerfDataNotAvailableError):
            _query(db)


def test_opt_in_source_identity_is_not_trimmed(systems_root):
    source = f" {SOURCE} "
    _write(systems_root, [_row(default_eligible=False, kernel_source=source)])
    db = _database(systems_root)
    assert _query(db, source=source)[1] == 0.25
    with pytest.raises(PerfDataNotAvailableError):
        _query(db, source=SOURCE)


def test_shared_cross_backend_opt_in_rows_remain_named_only(systems_root):
    _write(systems_root, [_row(kernel_source="triton", default_eligible=True, moe_tp_size=2)])
    _write(
        systems_root,
        [
            _row(default_eligible=False),
            _row(kernel_source="not_shared", default_eligible=True),
        ],
        backend="vllm",
        version="0.24.0",
    )
    (systems_root / "perf_data_reuse_manifest.yaml").write_text(
        "groups:\n  - op_file: moe_perf.txt\n"
        f"    kernel_source: {SOURCE}\n    tier: shared\n    frameworks: [sglang, vllm]\n"
    )
    db = _database(systems_root, shared_layer=True)
    assert _query(db, source=SOURCE)[1] == 0.25
    with pytest.raises(PerfDataNotAvailableError):
        _query(db)
    assert db.legacy_moe_compute_coverage(8192, 2048, 8, 256, MoEQuantMode.fp8_block) == set()


@pytest.mark.parametrize("mode", ["SILICON", "EMPIRICAL", "HYBRID"])
def test_later_null_flag_invalidates_the_table(systems_root, mode):
    _write(systems_root, [_row(default_eligible=True), _row(default_eligible=None, num_tokens=64)])
    db = _database(systems_root, mode=mode)
    for _ in range(2):
        with pytest.raises(ValueError, match="default_eligible.*non-null Boolean"):
            _query(db)


@pytest.mark.parametrize("eligible", [None, True])
def test_legacy_unnamed_default_rows_remain_eligible(systems_root, eligible):
    row = _row()
    del row["kernel_source"]
    if eligible is not None:
        row["default_eligible"] = eligible
    _write(systems_root, [row])
    assert _query(_database(systems_root))[1] == 0.25
