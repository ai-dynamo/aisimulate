# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY: synthetic ABI object lifetimes, not native CUDA qualification."""

import ctypes
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.unit
from pathlib import Path
from types import SimpleNamespace

from collector.glm53flash_graph_callbacks import CloneCallbacks, GraphData, ResourceData, ResourceHandle


class ResourceWrapperTests(unittest.TestCase):
    def make_callbacks(self, audit_path):
        api = SimpleNamespace(
            cupti=SimpleNamespace(),
            _bind=lambda *args: None,
            libraries={"cupti": {"sha256": "a55e03ccab21830f5b9d1ca7a02ecd59c557e0d54c769a181ad1140a3cff8ac1"}},
        )
        callbacks = CloneCallbacks(api, audit_path)
        queries = []

        def get_id(name, handle, kind=ctypes.c_uint32):
            queries.append((name, handle))
            return handle + 1000

        callbacks._id = get_id
        return callbacks, queries

    def test_resource_and_graph_are_separate_native_objects(self):
        self.assertEqual(ctypes.sizeof(ResourceData), 24)
        self.assertEqual(ResourceData.resourceDescriptor.offset, 16)
        for cbid, expected in [
            (11, [("cuptiGetGraphId", 101), ("cuptiGetGraphId", 102)]),
            (18, [("cuptiGetGraphId", 101), ("cuptiGetGraphExecId", 106)]),
            (20, [("cuptiGetGraphNodeId", 103), ("cuptiGetGraphNodeId", 104)]),
        ]:
            with self.subTest(cbid=cbid), tempfile.TemporaryDirectory() as tmp:
                callbacks, queries = self.make_callbacks(Path(tmp) / "progress.jsonl")
                graph = GraphData(101, 102, 103, 104, 0, 105, 106)
                wrapper = ResourceData(201, ResourceHandle(202), ctypes.addressof(graph))
                # Use the actual CFUNCTYPE boundary and keep both objects alive.
                callbacks.callback(None, 3, cbid, ctypes.addressof(wrapper))
                self.assertEqual(callbacks.errors, [])
                self.assertEqual(queries, expected)
                self.assertEqual(callbacks.rows[0]["raw_fields"]["graph"], 101)
                text = (Path(tmp) / "progress.jsonl").read_text()
                self.assertIn('"stage": "resource_wrapper"', text)
                self.assertIn('"stage": "graph_payload"', text)
                callbacks.close_audit()

    def test_missing_descriptor_rejects_before_any_handle_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            callbacks, queries = self.make_callbacks(Path(tmp) / "progress.jsonl")
            wrapper = ResourceData(201, ResourceHandle(202), None)
            callbacks.callback(None, 3, 20, ctypes.addressof(wrapper))
            self.assertEqual(queries, [])
            self.assertEqual(callbacks.rows, [])
            self.assertIn("no descriptor", callbacks.errors[0])
            callbacks.close_audit()


if __name__ == "__main__":
    unittest.main()
