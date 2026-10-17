# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as Ns

import pytest

from scripts.prediction_regression.export_agentx_snapshots import SequentialCursor, export_snapshot


def metadata(strategy="sequential"):
    return Ns(
        sampling_strategy=strategy,
        conversations=[
            Ns(conversation_id="skip", is_root=True),
            Ns(conversation_id="a", is_root=True),
            Ns(conversation_id="child", is_root=False),
            Ns(conversation_id="b", is_root=True),
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

    result = export_snapshot(metadata(), {"skip": "s", "a": "A", "b": "B"}, lanes=2, seed=42, trajectory_type=reference)
    assert result == {
        "seed": 42,
        "initial_selections": [{"source_play_id": "A", "t_star_ms": 12.5}, {"source_play_id": "B", "t_star_ms": 40.5}],
        "corpus_cursor": 3,
    }


def test_sequential_cursor_wraps_without_resetting_consumed_draws():
    cursor = SequentialCursor(["a", "b"])
    assert [cursor.next_conversation_id() for _ in range(5)] == ["a", "b", "a", "b", "a"]
    assert cursor.cursor == 5


@pytest.mark.parametrize("strategy", ["random", "shuffle"])
def test_export_rejects_unimplemented_recycle_samplers(strategy):
    with pytest.raises(ValueError, match="sequential"):
        export_snapshot(metadata(strategy), {}, lanes=2, seed=42, trajectory_type=None)


@pytest.mark.parametrize("mapping", [{}, {"skip": "S", "a": "A", "b": "A"}, {"skip": "S", "a": "A", "b": ""}])
def test_export_requires_complete_unambiguous_normalized_play_mapping(mapping):
    with pytest.raises(ValueError, match="source.play|distinct"):
        export_snapshot(metadata(), mapping, lanes=2, seed=42, trajectory_type=None)


@pytest.mark.parametrize("trajectories", [[], [Ns(conversation_id="a", snapshot=None)]])
def test_export_fails_closed_for_missing_lanes_or_timestamp_less_source(trajectories):
    with pytest.raises(ValueError, match="lane|timestamp"):
        export_snapshot(
            metadata(),
            {"skip": "S", "a": "A", "b": "B"},
            lanes=1,
            seed=42,
            trajectory_type=lambda **_: Ns(trajectories=trajectories),
        )
