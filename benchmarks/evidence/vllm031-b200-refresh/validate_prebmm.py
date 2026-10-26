# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check exact pre-BMM rows through Python orchestration and native op queries."""

import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from aisimulate_core.sdk import common
from aisimulate_core.sdk.engine import _probe_handle_for, build_ops_json
from aisimulate_core.sdk.operations.mla import MLABmm
from aisimulate_core.sdk.perf_database import PerfDatabase


def main():
    repo = Path(__file__).resolve().parents[3]
    systems = repo / "python/aisimulate/src/aisimulate_core/systems"
    table = systems / "data/b200_sxm/mla_bmm/vllm/0.31.0/mla_bmm_perf.parquet"
    rows = pq.read_table(table).to_pylist()
    database = PerfDatabase(
        "b200_sxm", "vllm", "0.31.0", systems_root=str(systems), shared_layer=False
    )
    handle = _probe_handle_for(database, None)
    identities = set()
    observations = []
    for row in rows:
        assert row["op_name"] == "mla_gen_pre" and row["bmm_dtype"] == "bfloat16", row
        identity = (
            row["op_name"],
            row["bmm_dtype"],
            row["num_tokens"],
            row["num_heads"],
        )
        assert identity not in identities, identity
        identities.add(identity)
        operation = MLABmm(
            "pre", 1.0, row["num_heads"], common.GEMMQuantMode.bfloat16, True
        )
        python_value = float(
            operation._engine_query(database, batch_size=row["num_tokens"])
        )
        native = handle.evaluate_ops_json(
            build_ops_json([operation]),
            is_context=False,
            batch_size=row["num_tokens"],
            s=1,
        )[0]
        native_value = float(native[1])
        assert math.isfinite(native_value) and native_value > 0
        assert math.isclose(python_value, row["latency"], rel_tol=1e-6), (
            identity,
            python_value,
            row,
        )
        assert math.isclose(native_value, row["latency"], rel_tol=1e-6), (
            identity,
            native,
            row,
        )
        observations.append(
            {"identity": identity, "latency_ms": native_value, "source": str(native[3])}
        )
    assert len(rows) == 300
    result = {
        "version": "0.31.0",
        "system": "b200_sxm",
        "shared_layer": False,
        "planned": 300,
        "passed": len(observations),
        "failed": 0,
        "interfaces": [
            "Python orchestration _engine_query",
            "native evaluate_ops_json",
        ],
        "legacy_note": "The public legacy query method has been removed from this main revision; both retained interfaces use the native engine.",
        "observations": observations,
    }
    Path(__file__).with_name("prebmm-consumer-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2)
    )


if __name__ == "__main__":
    main()
