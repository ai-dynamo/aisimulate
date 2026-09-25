# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native completion is cohort membership, independent of scheduler ordering."""

import json

import pytest

from collector.fpm_forward.sglang_driver import wait_retained_release
from collector.glm53flash_sglang_retained import PRODUCER_PROTOCOL

pytestmark = pytest.mark.unit


def receipt(root, rank, ids, *, change=None):
    rows = [{"request_id": rid, "released": True, "parked": None} for rid in ids]
    if change is not None:
        change(rows)
    event = {"producer_protocol": PRODUCER_PROTOCOL, "tp_rank": rank, "requests": rows}
    path = root / f"retained-rank-{rank}.jsonl"
    path.write_text(json.dumps(event) + "\n")
    return path


@pytest.mark.parametrize("batch,tp", [(2, 2), (4, 4), (16, 2), (32, 4)])
def test_sorted_native_manifest_cohort_releases_numeric_api_submission(tmp_path, batch, tp):
    ids = [f"test-request-p1-r0-q{i}" for i in range(batch)]
    # write_json sorts the frozen manifest keys. The child reconstructs cohorts
    # in that order; the parent retains the API's numeric submission order.
    native_ids = list(json.loads(json.dumps(dict.fromkeys(ids), sort_keys=True)))
    for rank in range(tp):
        receipt(tmp_path, rank, native_ids)
    wait_retained_release(tmp_path, ids, tp, 0)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "extra", "wrong_id", "parked", "unreleased", "rank"])
def test_exact_all_rank_release_is_still_required(tmp_path, failure):
    ids = [f"test-q{i}" for i in range(32)]
    for rank in (0, 1):
        receipt(tmp_path, rank, sorted(ids))

    def corrupt(rows):
        if failure == "missing":
            rows.pop()
        elif failure == "duplicate":
            rows[-1] = dict(rows[0])
        elif failure == "extra":
            rows.append({"request_id": "unexpected", "released": True, "parked": None})
        elif failure == "wrong_id":
            rows[-1]["request_id"] = "different"
        elif failure == "parked":
            rows[-1]["parked"] = {"still_owns_state": True}
        elif failure == "unreleased":
            rows[-1]["released"] = False

    path = receipt(tmp_path, 1, sorted(ids), change=corrupt)
    if failure == "rank":
        path.unlink()
    with pytest.raises(TimeoutError, match="ranks \\[1\\]"):
        wait_retained_release(tmp_path, ids, 2, 0)


@pytest.mark.parametrize("ids", [[], ["same", "same"]])
def test_invalid_parent_cohort_is_rejected(tmp_path, ids):
    with pytest.raises(ValueError, match="unique request cohort"):
        wait_retained_release(tmp_path, ids, 2, 0)
