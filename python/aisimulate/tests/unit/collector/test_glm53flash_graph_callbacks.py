# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY synthetic callback integrity; never native qualification evidence."""

import copy
import ctypes
import json
import unittest
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
from collector.glm53flash_graph_callbacks import (
    EVENT_RECORD_CUDART_SHA256,
    QUALIFIED_CUPTI_SHA256,
    record_event_record_types,
    resolve_registry,
)


class CloneIntegrity(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "graph_id": 4,
            "nodes": [{"node_id": 900, "node_type": 0, "name": "a"}, {"node_id": 3, "node_type": 2, "name": "b"}],
            "edges": [{"from": 900, "to": 3, "type": 1}],
        }
        self.receipt = {
            "callback_errors": [],
            "actual_graph_exec_id": 19,
            "callbacks": [
                {
                    "kind": "graph_exec_created",
                    "graph_id": 4,
                    "graph_exec_id": 19,
                    "raw_fields": {"graph": 101, "graphExec": 909},
                },
                {
                    "kind": "node_cloned",
                    "original_node_id": 3,
                    "node_id": 808,
                    "node_type": 2,
                    "raw_fields": {"originalGraph": 101, "graph": 909, "node": 88, "originalNode": 33, "nodeType": 2},
                },
                {
                    "kind": "node_cloned",
                    "original_node_id": 900,
                    "node_id": 8,
                    "node_type": 0,
                    "raw_fields": {"originalGraph": 101, "graph": 909, "node": 89, "originalNode": 34, "nodeType": 0},
                },
            ],
        }

    def test_explicit_nonordered_opaque_ids_and_edge(self):
        out = resolve_registry(self.registry, self.receipt)
        self.assertEqual([row["node_id"] for row in out["nodes"]], [8, 808])
        self.assertEqual(out["edges"], [{"from": 8, "to": 808, "type": 1}])
        self.assertEqual(self.registry["nodes"][0]["node_id"], 900)
        self.assertEqual(out["graph_id"], 19)
        self.assertEqual(out["capture_graph_id"], 4)

    def test_extra_target_structural_clone_from_other_source_is_rejected(self):
        extra = copy.deepcopy(self.receipt["callbacks"][-1])
        extra.update(original_node_id=789, node_id=543, node_type=3)
        extra["raw_fields"]["originalGraph"] = 202
        self.receipt["callbacks"].append(extra)
        with self.assertRaisesRegex(ValueError, "unqualified source graph"):
            resolve_registry(self.registry, self.receipt)

    def test_missing_extra_ambiguous_or_wrong_type_cannot_be_inferred(self):
        variations = []
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"].pop()
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"].append(changed["callbacks"][-1])
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"][-1]["node_type"] = 2
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["actual_graph_exec_id"] = 23
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callback_errors"] = ["test native failure"]
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"][-1]["node_id"] = 808
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"][0]["graph_id"] = 71
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"][0]["raw_fields"]["graphExec"] = None
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"][0]["raw_fields"]["graphExec"] = 101
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"][-1]["raw_fields"]["graph"] = 818
        variations.append(changed)
        changed = copy.deepcopy(self.receipt)
        changed["callbacks"][-1]["raw_fields"]["originalGraph"] = 818
        variations.append(changed)
        for value in variations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_registry(self.registry, value)


if __name__ == "__main__":
    unittest.main()


def event_record_fixture():
    fixture = CloneIntegrity()
    fixture.setUp()
    source, receipt = fixture.registry, fixture.receipt
    source["nodes"][1]["node_type"] = 7
    source["native_api_libraries"] = {
        "cudart": {"sha256": EVENT_RECORD_CUDART_SHA256},
        "cupti": {"sha256": QUALIFIED_CUPTI_SHA256},
    }
    receipt["callbacks"][1]["node_type"] = 0
    receipt["callbacks"][1]["raw_fields"]["nodeType"] = 0
    receipt["callback_subscription_closed"] = True
    return source, receipt


def query_fixture(tmp_path, *, status=0, actual_type=7):
    source, receipt = event_record_fixture()
    path = tmp_path / "TEST_ONLY-event-record-types.json"
    calls = []

    def native_query(handle, pointer):
        before = json.loads(path.read_text())
        assert before["completed"] is False and before["callback_subscription_closed"] is True
        assert before["queries"][-1]["rc"] is None
        assert before["queries"][-1]["clone_node_handle"] == handle == 88
        calls.append(handle)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = actual_type
        return status

    api = SimpleNamespace(
        libraries=source["native_api_libraries"],
        runtime=SimpleNamespace(cudaGraphNodeGetType=native_query),
    )
    graph = SimpleNamespace(raw_cuda_graph_exec=lambda: 909)
    return source, receipt, api, graph, path, calls


def test_event_record_uses_deferred_query_without_changing_original_callback_or_source(tmp_path):
    source, receipt, api, graph, path, calls = query_fixture(tmp_path)
    original = copy.deepcopy((source, receipt))
    with pytest.raises(ValueError, match="exact deferred native type proof"):
        resolve_registry(source, receipt)
    proof = record_event_record_types(api, graph, source, receipt, path)
    assert proof == json.loads(path.read_text()) and calls == [88]
    derived = resolve_registry(source, receipt, proof)
    assert derived["nodes"][1]["node_type"] == 7
    assert derived["nodes"][1]["node_id"] == 808
    assert derived["native_instantiation"]["node_clones"][0]["node_type"] == 0
    assert (source, receipt) == original


@pytest.mark.parametrize(
    "defect",
    [
        "absent",
        "failed_rc",
        "bool_rc",
        "wrong_type",
        "wrong_handle",
        "wrong_original",
        "wrong_clone",
        "duplicate_query",
        "missing_query",
        "wrong_provider",
        "wrong_exec",
        "not_completed",
        "still_subscribed",
    ],
)
def test_event_record_deferred_type_evidence_cannot_be_guessed_or_cross_bound(tmp_path, defect):
    source, receipt, api, graph, path, _ = query_fixture(tmp_path)
    proof = record_event_record_types(api, graph, source, receipt, path)
    row = proof["queries"][0]
    if defect == "absent":
        proof = None
    elif defect == "failed_rc":
        row["rc"] = 1
    elif defect == "bool_rc":
        row["rc"] = False
    elif defect == "wrong_type":
        row["native_node_type"] = 0
    elif defect == "wrong_handle":
        row["clone_node_handle"] = 34
    elif defect == "wrong_original":
        row["original_node_id"] = 900
    elif defect == "wrong_clone":
        row["node_id"] = 8
    elif defect == "duplicate_query":
        proof["queries"].append(copy.deepcopy(row))
    elif defect == "missing_query":
        proof["queries"].clear()
    elif defect == "wrong_provider":
        proof["native_api_libraries"]["cudart"]["sha256"] = "0" * 64
    elif defect == "wrong_exec":
        proof["graph_exec_handle"] = 101
    elif defect == "not_completed":
        proof["completed"] = False
    elif defect == "still_subscribed":
        receipt["callback_subscription_closed"] = False
    with pytest.raises(ValueError, match="exact deferred native type proof"):
        resolve_registry(source, receipt, proof)


@pytest.mark.parametrize("status,actual_type", [(1, 7), (0, 0)])
def test_deferred_native_query_failure_is_preserved_and_never_normalized(tmp_path, status, actual_type):
    source, receipt, api, graph, path, _ = query_fixture(tmp_path, status=status, actual_type=actual_type)
    with pytest.raises(RuntimeError, match="failed or changed type"):
        record_event_record_types(api, graph, source, receipt, path)
    proof = json.loads(path.read_text())
    assert proof["completed"] is False
    assert proof["queries"][0]["rc"] == status and proof["queries"][0]["native_node_type"] == actual_type
    with pytest.raises(ValueError):
        resolve_registry(source, receipt, proof)


@pytest.mark.parametrize("defect", ["empty", "wait", "memset", "duplicate_handle", "foreign_provider", "subscription"])
def test_only_event_record_mismatch_can_reach_the_deferred_api(tmp_path, defect):
    source, receipt, api, graph, path, calls = query_fixture(tmp_path)
    if defect in ("empty", "wait", "memset"):
        source["nodes"][1]["node_type"] = {"empty": 5, "wait": 6, "memset": 2}[defect]
    elif defect == "duplicate_handle":
        receipt["callbacks"][1]["raw_fields"]["node"] = 89
    elif defect == "foreign_provider":
        api.libraries = copy.deepcopy(api.libraries)
        api.libraries["cudart"]["sha256"] = "0" * 64
    elif defect == "subscription":
        receipt["callback_subscription_closed"] = False
    with pytest.raises((ValueError, RuntimeError)):
        record_event_record_types(api, graph, source, receipt, path)
    assert not calls and not path.exists()
