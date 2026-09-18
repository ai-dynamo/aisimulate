# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression evidence for the B200 TRT-LLM 1.3.0rc20 import.

Source: ai-dynamo/aiconfigurator at 915f590680d8a79fe9c39f6f3a9ff13bc267fcce,
under aic-core/src/aiconfigurator_core/systems/data/b200_sxm/.
See the adjacent data README for attribution and the two attention merges.
The context MLA and MoE pins include the reviewed September 17-18 refresh
documented in the data README; other pins still describe the original import.
Version-independent power-field checks live in test_power_data_invariants.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.unit

DATA_ROOT = Path(__file__).resolve().parents[3] / "src/aisimulate_core/systems/data/b200_sxm"


@pytest.mark.parametrize(
    "relative, sha256",
    [
        (
            "attention/trtllm/1.3.0rc20/context_attention_perf.parquet",
            "bac5acf73e155d5dc08b56313c88f54fb4f5affe30b084ba93b2d1d94404d014",
        ),
        (
            "attention/trtllm/1.3.0rc20/generation_attention_perf.parquet",
            "4e8bd4bc9d60a81e01a94c91f3e0c289c3d4ec7d789716866b0e37aadaed4914",
        ),
        (
            "comm/trtllm/1.3.0rc20/custom_allreduce_perf.parquet",
            "b7ef301e0da921733bd70733fcaf75e450f5416148d1e5f43b75cb3e7da39257",
        ),
        (
            "encoder_attention/trtllm/1.3.0rc20/encoder_attention_perf.parquet",
            "ffcf7d187fa3b4a06c95350c407861840f6b3081e3eb491c4037a53ceca0f554",
        ),
        (
            "gemm/trtllm/1.3.0rc20/gemm_perf.parquet",
            "850893f77b63c52c25310165b3e5ec9893035993b692315621bbb9f5fabb6a94",
        ),
        (
            "linear_attention/trtllm/1.3.0rc20/gdn_perf.parquet",
            "f6745c86d0003d85566ed68efba56cff6b93d55a32de9d6da6372edfbd3bd6d8",
        ),
        (
            "linear_attention/trtllm/1.3.0rc20/mamba2_perf.parquet",
            "79f99c76f11ccabad90f738c300bab83136ea0ac6c4c1c1c6e1f1ce1ed6c8bca",
        ),
        (
            "mhc/trtllm/1.3.0rc20/mhc_module_perf.parquet",
            "b22d43712ece8e5701f8aace0d902543ceec7a72f1f06139ff8bf3e5bd38140a",
        ),
        (
            "mla/trtllm/1.3.0rc20/context_mla_perf.parquet",
            "905f09a335f56c3a4c37cc7eb74113c1c026e3c57a8ff3dfc40a7c98d1fab30a",
        ),
        (
            "mla/trtllm/1.3.0rc20/generation_mla_perf.parquet",
            "fd1225a36b158c024ac419be7f419bddc8ec31cfce1d2e27586557f02b529788",
        ),
        (
            "mla/trtllm/1.3.0rc20/mla_context_module_perf.parquet",
            "4e7932ce34c9e2ebc5732c0108dd4409bafe2cc712cf62039887d82aa86e96f2",
        ),
        (
            "mla/trtllm/1.3.0rc20/mla_generation_module_perf.parquet",
            "4df91c4a51a4dba00ad97d7b37d4429653f997d6cd0119d709b9dc108d4ba257",
        ),
        (
            "mla_bmm/trtllm/1.3.0rc20/mla_bmm_perf.parquet",
            "f714e04f688ec2d2b8022fd1c4a2f005d5894fe776d54779b86e62a9ee262f97",
        ),
        (
            "moe/trtllm/1.3.0rc20/moe_perf.parquet",
            "9c88ebdd67b97ded16310f18abaa5459d4959ff7b880bbd15c6fec43753af411",
        ),
        (
            "quantize/trtllm/1.3.0rc20/computescale_perf.parquet",
            "d28351cc04052ab3611b816b5bc713e952fd22e2f1f5b58d393ea441147e2d8f",
        ),
        (
            "quantize/trtllm/1.3.0rc20/scale_matrix_perf.parquet",
            "2fe4746f7f2f8b9e3994c09d20facd83dbdb6d216a63b535bd8e1622a8c50515",
        ),
        (
            "sparse_attention/trtllm/1.3.0rc20/dsa_context_module_perf.parquet",
            "affb173c311fd5eefdcb208b5fddf0b1aa4e5ddb00f4a7e0f80740d5849903db",
        ),
        (
            "sparse_attention/trtllm/1.3.0rc20/dsa_generation_module_perf.parquet",
            "5d3478300a992c4d4c97ba2f9edc7e9bc04730b4ca1d4282bcfd0172cd036883",
        ),
    ],
)
def test_b200_import_file(relative, sha256):
    # Fourteen pins equal upstream bytes; two pin attention merges and two the MLA/MoE refresh.
    assert hashlib.sha256((DATA_ROOT / relative).read_bytes()).hexdigest() == sha256


# Local identity/latency digests come from ffcb6576b3a60077ea1200788f1b784213f979cf,
# the parent of import commit 717f973bea4ebc07673192475b3a0f743d82c168. Filter that
# baseline to identities absent from the pinned upstream table, then serialize
# the ordered columns and sorted rows exactly as below; no Git access is needed
# when running the tests.
@pytest.mark.parametrize(
    "phase, upstream_sha256, retained_rows, local_sha256",
    [
        (
            "context",
            "2ec64496ee67e80343b89386ec692813d23c7a03eddc0872d232ef45e5e64b36",
            1428,
            "eb9d42806a5c5d0c5a3d6303f27135f09a7d1b7a5ea9ce9c60c48b581b3b2ca0",
        ),
        (
            "generation",
            "1456c5133fbf85031dfda81b9440b4d5c70b672ee5edb86174b334a12c8681e7",
            1334,
            "a60bdcf42bcfe6e2d2a2ff53ba73ef6cd1b85a1bd33dee260ac2e0df136274fe",
        ),
    ],
)
def test_attention_merge_preserves_upstream_measurements(phase, upstream_sha256, retained_rows, local_sha256):
    source = DATA_ROOT / "power_upstream" / f"{phase}_attention.parquet.source"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == upstream_sha256
    upstream = pq.read_table(source)
    packaged = pq.read_table(DATA_ROOT / "attention/trtllm/1.3.0rc20" / f"{phase}_attention_perf.parquet")
    assert upstream.schema.equals(packaged.schema, check_metadata=False)
    identities = [name for name in upstream.column_names if name not in {"latency", "power", "power_limit"}]
    imported = {tuple(row[name] for name in identities): row for row in upstream.to_pylist()}
    merged = {tuple(row[name] for name in identities): row for row in packaged.to_pylist()}
    assert len(imported) == upstream.num_rows, "duplicate upstream identities"
    assert len(merged) == packaged.num_rows, "duplicate packaged identities"
    assert imported.keys() <= merged.keys(), "dropped upstream identities"
    assert all(merged[key] == row for key, row in imported.items()), "changed upstream measurements"
    local = merged.keys() - imported.keys()
    assert len(local) == retained_rows
    assert all(merged[key]["power"] == merged[key]["power_limit"] == 0.0 for key in local)
    columns = [*identities, "latency"]
    pairs = sorted([merged[key][name] for name in columns] for key in local)
    evidence = json.dumps(
        {"columns": columns, "rows": pairs}, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(evidence).hexdigest() == local_sha256, "changed pre-import local identity/latency pairs"
