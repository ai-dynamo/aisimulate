# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from AISim FPM Gym f934c030afc3a03cb04d8f3ff4709194f7445c98,
# tests/test_visualization.py. See scripts/fpm_accuracy/README.md.
import gzip
import json
from dataclasses import replace

from test_hf_dataset import CONFIGURATION_PATH, _build_dataset, _fpm_payload

from scripts.fpm_accuracy.dashboard.visualization import VisualizationWriter, point
from scripts.fpm_accuracy.dashboard.visualization_diagnostics import diagnostic_observations
from scripts.fpm_accuracy.hf.models import MeasurementObservation
from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics


def observation(metrics: list[RequestMetrics]) -> MeasurementObservation:
    ranks = tuple(
        ForwardPassMetric(
            fpm_id=i,
            configuration_id="config",
            version=1,
            worker_id="worker",
            dp_rank=i,
            counter_id=1,
            wall_time_s=0.01 + i * 0.005,
            scheduled=m,
            queued=RequestMetrics(),
        )
        for i, m in enumerate(metrics)
    )
    return MeasurementObservation("one", "config", 0, "source", "source.jsonl", 1, ForwardPassIteration(ranks))


def test_coordinates_keep_native_workload_when_max_attention_and_max_batch_disagree():
    value = observation(
        [
            RequestMetrics(num_prefill_requests=1, sum_prefill_tokens=20),
            RequestMetrics(num_decode_requests=9, sum_decode_kv_tokens=21),
        ]
    )
    p = point(value)
    assert str(value.workload_kind) == "decode"  # Native 9 + 21 > 20; maximum attention is prefill.
    assert p["points"][:3] == [210, 29, 15]
    assert p["axis_values"] == [9, 20, 9, 0, 21]
    assert p["rank_details"][:3] == [[0], [1], 1]
    assert p["iteration_ids"] == "one"


def test_native_ties_and_idle_prefix_metadata_are_preserved():
    value = observation(
        [
            RequestMetrics(num_prefill_requests=1, sum_prefill_tokens=20),
            RequestMetrics(num_decode_requests=1, sum_decode_kv_tokens=19),
            RequestMetrics(num_prefill_requests=1, sum_prefill_kv_tokens=10000),
        ]
    )
    p = point(value)
    assert p["rank_details"][2] == 1  # Later active rank wins tie, idle rank cannot replace it.
    assert p["points"][:3] == [210, 21, 20]  # Slowest rank can be idle.
    assert p["axis_values"][3] == 10000
    assert p["rank_details"][3][2][1] == "empty"


def dataset(tmp_path, *, dp=1, duplicate=False, files=1):
    records = []
    for counter in range(1, 41):
        for rank in range(dp):
            payload = _fpm_payload(rank=rank, counter=counter, wall_time=0.01 + counter * 0.001)
            payload["scheduled_requests"]["sum_decode_kv_tokens"] = counter * (rank + 1)
            if rank == 0 and counter % 3 == 0:
                payload["scheduled_requests"] = dict(
                    num_prefill_requests=1,
                    sum_prefill_tokens=20,
                    sum_prefill_kv_tokens=4,
                    num_decode_requests=0,
                    sum_decode_kv_tokens=0,
                )
            records.append(payload)
    if duplicate:
        records.append(records[-1])
    raw = "\n".join(map(json.dumps, records)).encode()
    return _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        dp=dp,
        files=[("truth", f"run-{i}.jsonl.gz", gzip.compress(raw)) for i in range(files)],
    )


def read_group(root, group, mode):
    prefix = "sample_" if mode == "sample" else ""
    values = []
    for name in [group["sample_file"]] if mode == "sample" else group["all_files"]:
        data = (root / name).read_bytes()
        if name.endswith(".gz"):
            data = gzip.decompress(data)
        values.extend(json.loads(data)[prefix + "iteration_ids"])
    return values


def test_reduced_record_scope_and_no_truth_catalog(tmp_path):
    raw = (
        b"measurement_id,phase,batch_size,total_prefill_tokens,total_kv_read_tokens,truth_latency_ms\n"
        b"a,decode,2,0,100,7\n"
    )
    hf = _build_dataset(
        tmp_path / "hf", protocol_id="forward-pass-record-v1", files=[("derived_truth", "x.csv", raw)], dp=4
    )
    writer = VisualizationWriter(tmp_path / "out", repo_id=hf.repo_id, revision=hf.revision)
    case = hf.measurement_case(CONFIGURATION_PATH)
    writer.add_case(case)
    # New models/snapshots use metadata, without frontend model-specific dispatch.
    other = replace(
        case,
        case_id="new-case",
        observations=(),
        configuration=replace(
            case.configuration, configuration_id="future-config", model_id="Future/Model", snapshot_id="future"
        ),
    )
    writer.add_case(other)
    writer.finish()
    assert writer.catalog[0]["coordinate_scope"] == "reduced_record"
    assert writer.catalog[0]["availability"] == "ready"
    assert writer.catalog[1]["model"] == "Future/Model"
    assert writer.catalog[1]["availability"] == "no_truth"


def test_samples_chunks_and_diagnostics(tmp_path):
    for dp in (1, 2):
        hf = dataset(tmp_path / str(dp), dp=dp, files=2)
        case = hf.measurement_case(CONFIGURATION_PATH)
        outputs = []
        for name in ("first", "second"):
            output = tmp_path / f"{dp}-{name}"
            writer = VisualizationWriter(output, repo_id=hf.repo_id, revision=hf.revision, sample_size=3, chunk_size=7)
            writer.add_case(case)
            writer.finish()
            outputs.append((output / "manifest.json").read_bytes())
            ids = []
            for group in writer.groups:
                full = read_group(output, group, "all")
                sample = read_group(output, group, "sample")
                assert sample == [full[i] for i in group["sample_indices"]]
                assert len(full) == group["n"]
                assert len(sample) <= len(full)
                ids.extend(full)
            assert len(ids) == len(set(ids)) == 80
            assert len({g["worker"] for g in writer.groups}) == 2
        assert outputs[0] == outputs[1]
        if dp == 2:
            assert not case.observations
            assert writer.catalog[0]["availability"] == "diagnostic"
            assert writer.catalog[0]["measured"] == 0
            bad = dataset(tmp_path / "duplicate", dp=2, duplicate=True).measurement_case(CONFIGURATION_PATH)
            observations, issues = diagnostic_observations(bad)
            assert len(observations) == 39
            assert issues == {"incomplete_duplicate_or_unexpected_rank_group": 1}
