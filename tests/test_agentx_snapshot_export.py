# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace as Ns

import pytest

from scripts.prediction_regression.export_agentx_snapshots import SequentialCursor, export_snapshot, write_export


def metadata(strategy="sequential"):
    return Ns(
        sampling_strategy=strategy,
        conversations=[
            Ns(conversation_id="skip", is_root=True, turns=[], branches=[]),
            Ns(conversation_id="a", is_root=True, turns=[Ns(timestamp_ms=0), Ns(timestamp_ms=100)], branches=[]),
            Ns(conversation_id="child", is_root=False, turns=[], branches=[]),
            Ns(conversation_id="b", is_root=True, turns=[Ns(timestamp_ms=0), Ns(timestamp_ms=100)], branches=[]),
        ],
    )


def test_export_keeps_actual_draw_count_and_lane_order():
    def reference(**kwargs):
        assert kwargs["random_seed"] == 42
        assert (kwargs["start_min_ratio"], kwargs["start_max_ratio"]) == (0.25, 0.75)
        sampler = kwargs["dataset_sampler"]
        assert sampler.next_conversation_id() == "skip"
        assert sampler.next_conversation_id() == "a"
        assert sampler.next_conversation_id() == "b"
        return Ns(
            trajectories=[
                Ns(conversation_id="a", snapshot=Ns(t_star_ms=12.5)),
                Ns(conversation_id="b", snapshot=Ns(t_star_ms=40.5)),
            ]
        )

    result = export_snapshot(
        metadata(),
        {"skip": "s", "a": "A", "b": "B"},
        lanes=2,
        seed=42,
        trajectory_type=reference,
        trace_format="agentic_mooncake",
    )
    assert result == {
        "seed": 42,
        "initial_selections": [{"source_play_id": "A", "t_star_ms": 12.5}, {"source_play_id": "B", "t_star_ms": 40.5}],
        "corpus_cursor": 3,
    }


def test_skipped_timestamp_less_root_does_not_require_an_origin():
    source = metadata()
    source.conversations[0].turns = [Ns(timestamp_ms=None)]

    def reference(**kwargs):
        sampler = kwargs["dataset_sampler"]
        assert sampler.next_conversation_id() == "skip"
        assert sampler.next_conversation_id() == "a"
        return Ns(trajectories=[Ns(conversation_id="a", snapshot=Ns(t_star_ms=50))])

    evidence = []
    result = export_snapshot(
        source,
        {"skip": "S", "a": "A", "b": "B"},
        lanes=1,
        seed=42,
        trajectory_type=reference,
        trace_format="weka",
        clock_evidence=evidence,
    )
    assert result["corpus_cursor"] == 2
    assert [entry["reference_root_id"] for entry in evidence] == ["a"]


@pytest.mark.parametrize(
    "children,error", [(["missing"], "missing reference child"), (["a"], "cyclic"), (["b"], "sampleable root")]
)
def test_unmappable_child_metadata_fails_closed(children, error):
    source = metadata()
    source.conversations[1].branches = [Ns(child_conversation_ids=children)]
    with pytest.raises(ValueError, match=error):
        export_snapshot(
            source,
            {"skip": "S", "a": "A", "b": "B"},
            lanes=1,
            seed=42,
            trajectory_type=lambda **_: Ns(trajectories=[Ns(conversation_id="a", snapshot=Ns(t_star_ms=50))]),
            trace_format="weka",
        )


def test_sequential_cursor_wraps_without_resetting_consumed_draws():
    cursor = SequentialCursor(["a", "b"])
    assert [cursor.next_conversation_id() for _ in range(5)] == ["a", "b", "a", "b", "a"]
    assert cursor.cursor == 5


@pytest.mark.parametrize("strategy", ["random", "shuffle"])
def test_export_rejects_unimplemented_recycle_samplers(strategy):
    with pytest.raises(ValueError, match="sequential"):
        export_snapshot(metadata(strategy), {}, lanes=2, seed=42, trajectory_type=None, trace_format="weka")


@pytest.mark.parametrize("mapping", [{}, {"skip": "S", "a": "A", "b": "A"}, {"skip": "S", "a": "A", "b": ""}])
def test_export_requires_complete_unambiguous_normalized_play_mapping(mapping):
    with pytest.raises(ValueError, match="source.play|distinct"):
        export_snapshot(metadata(), mapping, lanes=2, seed=42, trajectory_type=None, trace_format="weka")


@pytest.mark.parametrize("trajectories", [[], [Ns(conversation_id="a", snapshot=None)]])
def test_export_fails_closed_for_missing_lanes_or_timestamp_less_source(trajectories):
    with pytest.raises(ValueError, match="lane|timestamp"):
        export_snapshot(
            metadata(),
            {"skip": "S", "a": "A", "b": "B"},
            lanes=1,
            seed=42,
            trajectory_type=lambda **_: Ns(trajectories=trajectories),
            trace_format="weka",
        )


def _export_cut(times, cut, *, trace_format="weka", child_times=()):
    parent = Ns(conversation_id="root", is_root=True, turns=[Ns(timestamp_ms=t) for t in times], branches=[])
    conversations = [parent]
    if child_times:
        parent.branches = [Ns(child_conversation_ids=["child"], start_timestamp_ms=1)]
        conversations.append(
            Ns(conversation_id="child", is_root=False, turns=[Ns(timestamp_ms=t) for t in child_times], branches=[])
        )
    source = Ns(sampling_strategy="sequential", conversations=conversations)

    def reference(**kwargs):
        kwargs["dataset_sampler"].next_conversation_id()
        return Ns(trajectories=[Ns(conversation_id="root", snapshot=Ns(t_star_ms=cut))])

    clock_evidence = []
    result = export_snapshot(
        source,
        {"root": "PLAY"},
        lanes=1,
        seed=42,
        trajectory_type=reference,
        trace_format=trace_format,
        clock_evidence=clock_evidence,
    )
    return result, clock_evidence


@pytest.mark.parametrize("trace_format,expected", [("weka", 594.0), ("agentic_mooncake", 10594.0)])
def test_export_explicitly_converts_weka_but_preserves_mooncake_clock(trace_format, expected):
    result, evidence = _export_cut([10000.0, 11000.0], 10594.0, trace_format=trace_format)
    assert result["initial_selections"][0]["t_star_ms"] == expected
    assert evidence[0]["source_origin_ms"] == 10000.0
    assert evidence[0]["reference_t_star_ms"] == 10594.0
    assert evidence[0]["imported_t_star_ms"] == expected


@pytest.mark.parametrize("parent,child", [([10000, 30000], [15000]), ([20000, 30000], [10000])])
def test_origin_uses_earliest_request_in_whole_tree_not_marker_or_root_only(parent, child):
    result, _ = _export_cut(parent, 21000, child_times=child)
    assert result["initial_selections"][0]["t_star_ms"] == 11000


@pytest.mark.parametrize("timestamp", [None, float("nan"), float("inf"), -1, True, "10000"])
def test_incomplete_or_invalid_source_clock_fails_closed(timestamp):
    with pytest.raises(ValueError, match="finite nonnegative timestamp"):
        _export_cut([timestamp, 30000], 21000)


def test_unknown_clock_format_fails_closed():
    with pytest.raises(ValueError, match="explicitly select"):
        _export_cut([10000, 30000], 21000, trace_format="auto")


@pytest.mark.parametrize("last_seconds,absolute_cut,expected_history", [(11, 10594.0, 1), (40, 21000.0, 2)])
def test_exported_cuts_reach_real_weka_loader_without_rejection_or_silent_frontier_shift(
    tmp_path, last_seconds, absolute_cut, expected_history
):
    from aisimulate import _runtime

    # Original fixture: in the long case, passing 21s straight to the normalized
    # [0,30]s graph is legal but wrongly makes the 30s source turn historical.
    times = [10, last_seconds] if last_seconds == 11 else [10, 20, 30, last_seconds]
    path = tmp_path / "nonzero-origin.json"
    path.write_text(
        json.dumps(
            {
                "id": "nonzero-origin",
                "models": ["model"],
                "block_size": 4,
                "hash_id_scope": "local",
                "requests": [
                    {
                        "t": t,
                        "type": "s",
                        "model": "model",
                        "in": 4 * (i + 1),
                        "out": 1,
                        "api_time": 0.1,
                        "hash_ids": list(range(i + 1)),
                    }
                    for i, t in enumerate(times)
                ],
            }
        )
    )
    payload = {
        "spec": {
            "version": 1,
            "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
            "engine": {
                "tensor_parallel_size": 1,
                "rank": {
                    "backend": "vllm",
                    "block_size": 4,
                    "num_gpu_blocks": 128,
                    "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
                },
            },
            "requests": [],
        },
        "traffic": {
            "source_type": "trace",
            "load_type": "trace_timestamps",
            "trace_format": "weka",
            "trace_path": str(path),
            "trace_block_size": 4,
            "execution_model": "model",
            "agentic_lanes": 1,
            "agentic_snapshot": {"seed": 42},
        },
    }

    def run():
        return json.loads(_runtime.run_replay_json(json.dumps(payload)))["agentic_snapshots"][0]

    play_id = run()["source_play_id"]
    options, _ = _export_cut([t * 1000 for t in times], absolute_cut)
    options["initial_selections"][0]["source_play_id"] = play_id
    payload["traffic"]["agentic_snapshot"] = options
    actual = run()
    assert actual["t_star_ms"] == absolute_cut - 10000
    assert sum(request["historical"] for request in actual["requests"]) == expected_history
    options["initial_selections"][0]["t_star_ms"] = absolute_cut
    if last_seconds == 11:
        with pytest.raises(RuntimeError, match="bounds"):
            run()
    else:
        assert sum(request["historical"] for request in run()["requests"]) == 3


def test_export_persists_sidecar_and_refuses_to_overwrite(tmp_path, capsys):
    output = tmp_path / "snapshot.json"
    result, evidence = _export_cut([10000, 11000], 10594)
    provenance = {"reference_revision": "pinned", "clock_conversions": evidence}
    write_export(result, provenance, output=output, provenance_out=None)
    assert json.loads(output.read_text()) == result
    sidecar = output.with_name("snapshot.json.provenance.json")
    saved = json.loads(sidecar.read_text())
    assert saved["clock_conversions"] == evidence
    assert len(saved["snapshot_options_sha256"]) == 64
    before = (output.read_bytes(), sidecar.read_bytes())
    with pytest.raises(FileExistsError):
        write_export(result, provenance, output=output, provenance_out=None)
    assert (output.read_bytes(), sidecar.read_bytes()) == before
    with pytest.raises(FileExistsError):
        write_export(result, provenance, output=None, provenance_out=sidecar)
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="requires --provenance-out"):
        write_export(result, provenance, output=None, provenance_out=None)


def test_stdout_remains_importable_options_with_durable_sidecar(tmp_path, capsys):
    result, evidence = _export_cut([10000, 11000], 10594)
    sidecar = tmp_path / "provenance.json"
    write_export(result, {"clock_conversions": evidence}, output=None, provenance_out=sidecar)
    assert json.loads(capsys.readouterr().out) == result
    assert json.loads(sidecar.read_text())["clock_conversions"] == evidence
