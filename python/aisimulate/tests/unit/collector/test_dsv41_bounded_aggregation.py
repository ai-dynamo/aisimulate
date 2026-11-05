# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import statistics

import pytest
from collector.sglang.collect_dsv41_module import aggregate_bounded_attention_records
from collector.sglang.dsv41_contract import build_manifest
from collector.sglang.dsv41_workloads import freeze_workloads, projected_keys

pytestmark = pytest.mark.unit


def fixture(tmp_path, kind="native_attention_isolated"):
    # Synthetic reducer witnesses, not GPU measurements or audit evidence.
    coordinates = [
        (1, 256, 256),
        (1, 512, 0),
        (1, 256, 512),
        (1, 512, 256),
        (1, 768, 0),
        (2, 128, 256),
        (2, 384, 0),
        (2, 256, 512),
        (2, 768, 0),
    ]
    workloads = freeze_workloads(
        {
            "schema_version": 3,
            "prefill": [
                {"batch_size": b, "total_prefill_tokens": b * q, "total_kv_read_tokens": b * p}
                for b, q, p in coordinates
            ],
            "decode": [],
        }
    )
    tp = 2 if kind == "native_attention_isolated" else 4
    manifest = build_manifest(tp, True)
    warmup = 2 if kind == "native_attention_isolated" else 1
    paths = []
    for rank in range(tp):
        rows = []
        for owner, case in enumerate(workloads["cases"]):
            for sample in range(warmup, warmup + 5):
                for key in sorted(projected_keys(manifest, case)):
                    if key[0] != "attention":
                        continue
                    point = dict(zip(("component", "geometry", "batch_size", "prefix", "x"), key, strict=True))
                    point.update(
                        latency=(owner + 1) ** 2 + (sample - warmup - 2) * 0.02 - (tp - 1 - rank) * 0.1,
                        kernel_source="sglang.srt.models.deepseek_v4.MQALayer.forward",
                        measurement_scope="local_compute",
                        used_cuda_graph=False,
                        sample_count=1,
                        kv_seed_regime="real_kv" if key[3] else "n/a",
                        source_sha256="d50217d8f78e4bd173774c36713650bbf44b058c9575ac8babba208a5c5173a2",
                        config_sha256=manifest["config_sha256"],
                        runtime_digest=(
                            "sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5"
                            if kind == "native_attention_isolated"
                            else "sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d"
                        ),
                        execution_profile="decoder_bounded",
                        tp_rank=rank,
                        sample=sample,
                        invocation=owner if kind == "native_attention_isolated" else owner * 6 + sample + 1,
                    )
                    if kind == "native_attention_isolated":
                        point.update(case_id=case["case_id"], producer_kind=kind, collection_purpose="calibration")
                    rows.append(point)
        path = tmp_path / f"rank-{rank}.jsonl"
        path.write_text("\n".join(map(json.dumps, rows)))
        paths.append(path)
    return paths, manifest, workloads


@pytest.mark.parametrize("kind", ["native_attention_isolated", "native_checkpoint"])
def test_equal_owner_mean_preserves_each_distribution_instead_of_pooling_samples(tmp_path, kind):
    paths, manifest, workloads = fixture(tmp_path, kind)
    originals = [p.read_bytes() for p in paths]
    evidence = tmp_path / "owners.json"
    rows = aggregate_bounded_attention_records(
        paths, manifest["tp_size"], manifest, workloads, producer_kind=kind, evidence_path=evidence
    )
    assert [p.read_bytes() for p in paths] == originals
    report = json.loads(evidence.read_text())
    assert report["collision_groups"] == 8 and not report["input_or_timing_equivalence_asserted"]
    assert report["physical_rows"] == len(rows)
    three_owner = [r for r in report["distributions"] if len(r["owners"]) == 3]
    assert len(three_owner) == 2
    for group in three_owner:
        assert [r["median_ms"] for r in group["owners"]] == [9.0, 16.0, 25.0]
        assert group["equal_owner_mean_ms"] == pytest.approx(50.0 / 3)
        assert statistics.median([v for r in group["owners"] for v in r["rank_maxima_ms"]]) == 16.0
        assert all(len(r["rank_samples_ms"]) == 5 for r in group["owners"])
        measured = next(
            row
            for row in rows
            if [row[k] for k in ("component", "geometry", "batch_size", "prefix", "x")] == group["key"]
        )
        assert measured["latency"] == pytest.approx(50.0 / 3)
        assert measured["sample_count"] == 15
        assert "case_id" not in measured
    with pytest.raises(ValueError, match="new destination"):
        aggregate_bounded_attention_records(
            paths, manifest["tp_size"], manifest, workloads, producer_kind=kind, evidence_path=evidence
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_rank",
        "missing_sample",
        "missing_owner",
        "duplicate",
        "wrong_rank",
        "boolean_rank",
        "boolean_latency",
        "wrong_owner",
        "source",
        "runtime",
        "mixed_runtime",
        "graph",
        "kernel",
        "profile",
        "unknown_collision",
        "partial_groups",
    ],
)
def test_bounded_policy_rejects_incomplete_or_unaudited_inputs(tmp_path, failure):
    paths, manifest, workloads = fixture(tmp_path)
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    if failure == "missing_rank":
        paths = paths[:1]
    elif failure == "missing_sample":
        rows.pop()
    elif failure == "missing_owner":
        rows = [r for r in rows if r["invocation"] != 4]
    elif failure == "duplicate":
        rows.append(rows[0].copy())
    elif failure == "wrong_rank":
        rows[0]["tp_rank"] = 1
    elif failure == "boolean_rank":
        rows[0]["tp_rank"] = False
    elif failure == "boolean_latency":
        rows[0]["latency"] = True
    elif failure == "wrong_owner":
        rows[0]["case_id"] = "prefill-9999"
    elif failure in ("source", "runtime", "mixed_runtime", "graph", "kernel", "profile"):
        field, value = {
            "source": ("source_sha256", "0" * 64),
            "runtime": ("runtime_digest", "sha256:" + "0" * 64),
            "mixed_runtime": (
                "runtime_digest",
                "sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d",
            ),
            "graph": ("used_cuda_graph", True),
            "kernel": ("kernel_source", "another_attention"),
            "profile": ("execution_profile", "full"),
        }[failure]
        rows[0][field] = value
    else:
        payload = workloads["source_payload"]
        if failure == "unknown_collision":
            payload["prefill"].append({"batch_size": 1, "total_prefill_tokens": 384, "total_kv_read_tokens": 128})
        else:
            payload["prefill"].pop()
        workloads = freeze_workloads(payload)
    paths[0].write_text("\n".join(map(json.dumps, rows)))
    evidence = tmp_path / "rejected.json"
    with pytest.raises(ValueError):
        aggregate_bounded_attention_records(
            paths, 2, manifest, workloads, producer_kind="native_attention_isolated", evidence_path=evidence
        )
    assert not evidence.exists()
