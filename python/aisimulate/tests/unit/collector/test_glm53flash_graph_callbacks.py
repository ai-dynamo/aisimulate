# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY synthetic callback integrity; never native qualification evidence."""

import copy
import unittest

import pytest

pytestmark = pytest.mark.unit
from collector.glm53flash_graph_callbacks import resolve_registry


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
                    "raw_fields": {"originalGraph": 101, "graph": 909},
                },
                {
                    "kind": "node_cloned",
                    "original_node_id": 900,
                    "node_id": 8,
                    "node_type": 0,
                    "raw_fields": {"originalGraph": 101, "graph": 909},
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
