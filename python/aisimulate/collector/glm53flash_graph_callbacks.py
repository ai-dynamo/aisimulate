# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original, read-only wrapper for documented CUPTI 13.0 graph callbacks.

No CUDA API calls or graph mutations occur inside callbacks. Only CUPTI's ID
queries and copies of callback scalars are used. See adjacent README for API
references and exact header/library identity in README.glm53flash.md. Native
callback proof establishes node ownership only, never performance admission.
"""

import copy
import ctypes
import json
import os


class ResourceHandle(ctypes.Union):
    _fields_ = (("stream", ctypes.c_void_p),)


class ResourceData(ctypes.Structure):
    _fields_ = (
        ("context", ctypes.c_void_p),
        ("resourceHandle", ResourceHandle),
        ("resourceDescriptor", ctypes.c_void_p),
    )


class GraphData(ctypes.Structure):
    _fields_ = (
        ("graph", ctypes.c_void_p),
        ("originalGraph", ctypes.c_void_p),
        ("node", ctypes.c_void_p),
        ("originalNode", ctypes.c_void_p),
        ("nodeType", ctypes.c_int),
        ("dependency", ctypes.c_void_p),
        ("graphExec", ctypes.c_void_p),
    )


CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p)
CALLBACKS = {11: "graph_cloned", 18: "graph_exec_created", 20: "node_cloned"}
QUALIFIED_CUPTI_SHA256 = "a55e03ccab21830f5b9d1ca7a02ecd59c557e0d54c769a181ad1140a3cff8ac1"


class CloneCallbacks:
    def __init__(self, api, audit_path=None):
        if (
            api.libraries.get("cupti", {}).get("sha256") != QUALIFIED_CUPTI_SHA256
            or ctypes.sizeof(ResourceData) != 24
            or ResourceData.resourceDescriptor.offset != 16
            or ctypes.sizeof(GraphData) != 56
        ):
            raise RuntimeError("native graph callbacks require the qualified CUPTI library and resource ABI")
        self.api = api
        self.audit_fd = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600) if audit_path else None
        self.rows, self.errors = [], []
        self.subscriber = ctypes.c_void_p()
        self.callback = CALLBACK(self._callback)
        api._bind(
            api.cupti,
            "cuptiSubscribe",
            [ctypes.POINTER(ctypes.c_void_p), CALLBACK, ctypes.c_void_p],
        )
        api._bind(api.cupti, "cuptiUnsubscribe", [ctypes.c_void_p])
        api._bind(
            api.cupti,
            "cuptiEnableCallback",
            [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32],
        )
        api._bind(
            api.cupti,
            "cuptiGetGraphExecId",
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
        )

    def _audit(self, stage, **fields):
        if self.audit_fd is not None:
            raw = (json.dumps({"stage": stage, **fields}, sort_keys=True) + "\n").encode()
            while raw:
                raw = raw[os.write(self.audit_fd, raw) :]

    def _id(self, name, handle, kind=ctypes.c_uint32):
        if not handle:
            raise RuntimeError(f"callback {name} required a nonnull native handle")
        value = kind()
        self._audit("before_cupti_id_query", name=name, handle=handle)
        self.api._call(getattr(self.api.cupti, name), handle, ctypes.byref(value))
        self._audit("after_cupti_id_query", name=name, handle=handle, result=value.value)
        return value.value

    def _callback(self, user, domain, cbid, data):
        try:
            if domain != 3 or cbid not in CALLBACKS or not data:
                raise RuntimeError("unexpected native callback domain/id/data")
            # CUPTI resource callbacks carry ResourceData. The graph payload is
            # its resourceDescriptor, as in NVIDIA CUDA 13.0.85 graph tracing.
            self._audit("callback_entry", domain=domain, cbid=cbid, data=data)
            resource = ctypes.cast(data, ctypes.POINTER(ResourceData)).contents
            self._audit(
                "resource_wrapper",
                cbid=cbid,
                context=resource.context,
                resource_handle=resource.resourceHandle.stream,
                descriptor=resource.resourceDescriptor,
            )
            if not resource.resourceDescriptor:
                raise RuntimeError("graph resource callback has no descriptor")
            value = ctypes.cast(resource.resourceDescriptor, ctypes.POINTER(GraphData)).contents
            row = {
                "kind": CALLBACKS[cbid],
                "cbid": cbid,
                "raw_fields": {name: getattr(value, name) for name, _ in GraphData._fields_},
            }
            self.rows.append(row)
            self._audit("graph_payload", **row)
            if cbid == 11:
                row.update(
                    graph_id=self._id("cuptiGetGraphId", value.graph),
                    original_graph_id=self._id("cuptiGetGraphId", value.originalGraph),
                )
            elif cbid == 20:
                row.update(
                    node_id=self._id("cuptiGetGraphNodeId", value.node, ctypes.c_uint64),
                    original_node_id=self._id("cuptiGetGraphNodeId", value.originalNode, ctypes.c_uint64),
                    node_type=value.nodeType,
                )
            elif cbid == 18:
                row.update(
                    graph_id=self._id("cuptiGetGraphId", value.graph),
                    graph_exec_id=self._id("cuptiGetGraphExecId", value.graphExec),
                )
        except BaseException as exc:
            # ctypes callbacks cannot propagate into native code. Persist the
            # failure and reject after capture, never silently admit partial data.
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self._audit("callback_error", cbid=cbid, error=self.errors[-1])

    def __enter__(self):
        try:
            self.api._call(
                self.api.cupti.cuptiSubscribe,
                ctypes.byref(self.subscriber),
                self.callback,
                None,
            )
        except BaseException:
            self.close_audit()
            raise
        try:
            for cbid in CALLBACKS:
                self.api._call(self.api.cupti.cuptiEnableCallback, 1, self.subscriber, 3, cbid)
        except BaseException:
            try:
                self.api._call(self.api.cupti.cuptiUnsubscribe, self.subscriber)
            finally:
                self.close_audit()
            raise
        return self

    def __exit__(self, *args):
        try:
            self.api._call(self.api.cupti.cuptiUnsubscribe, self.subscriber)
        finally:
            self.close_audit()

    def close_audit(self):
        if self.audit_fd is not None:
            os.close(self.audit_fd)
            self.audit_fd = None

    def receipt(self, graph):
        return {
            "callbacks": self.rows,
            "callback_errors": self.errors,
            "actual_graph_exec_id": self._id("cuptiGetGraphExecId", graph.raw_cuda_graph_exec()),
            "graph_mutations": False,
        }


def resolve_registry(registry, receipt):
    """Bind one directly instantiated source graph through native node callbacks.

    Graph creation reports the source graph ID separately from the executable
    graph ID. Instantiation emits node-cloned callbacks without a graph-cloned
    callback on the qualified CUPTI 13.0.85 runtime. Require that exact observed
    relation, and reject additional clone chains rather than infer opaque IDs.
    """
    if receipt["callback_errors"]:
        raise ValueError("native clone callback failed; inspect preserved receipt")
    creates = [
        row
        for row in receipt["callbacks"]
        if row["kind"] == "graph_exec_created" and row["graph_exec_id"] == receipt["actual_graph_exec_id"]
    ]
    if len(creates) != 1:
        raise ValueError("executable graph lacks one actual creation callback")
    create = creates[0]
    if create["graph_id"] != registry["graph_id"]:
        raise ValueError("executable graph is not directly instantiated from the capture")
    source_handle = create.get("raw_fields", {}).get("graph")
    exec_handle = create.get("raw_fields", {}).get("graphExec")
    if not source_handle or not exec_handle or source_handle == exec_handle:
        raise ValueError("creation callback lacks distinct source/executable handles")
    # Compare copied callback identities only. These opaque handles are never
    # reinterpreted, dereferenced, or passed to an API of a different handle type.
    clones = [
        row
        for row in receipt["callbacks"]
        if row["kind"] == "node_cloned" and row.get("raw_fields", {}).get("graph") == exec_handle
    ]
    if any(row.get("raw_fields", {}).get("originalGraph") != source_handle for row in clones):
        raise ValueError("executable graph includes nodes from an unqualified source graph")
    nodes = registry["nodes"]
    originals = {row["node_id"]: row for row in nodes}
    if len(originals) != len(nodes) or len(clones) != len(nodes):
        raise ValueError("native instantiation lacks complete unique node coverage")
    ids = {}
    for clone in clones:
        original = clone["original_node_id"]
        if original not in originals or original in ids or clone["node_type"] != originals[original]["node_type"]:
            raise ValueError("native instantiation has an unknown, duplicate, or changed node")
        ids[original] = clone["node_id"]
    if len(set(ids.values())) != len(nodes):
        raise ValueError("native instantiation node mapping is not one-to-one")
    result = copy.deepcopy(registry)
    for row in result["nodes"]:
        row["capture_node_id"] = row["node_id"]
        row["node_id"] = ids[row["node_id"]]
    for edge in result["edges"]:
        edge["from"], edge["to"] = ids[edge["from"]], ids[edge["to"]]
    result.update(
        capture_graph_id=registry["graph_id"],
        graph_id=receipt["actual_graph_exec_id"],
        graph_exec_id=receipt["actual_graph_exec_id"],
        native_instantiation={"creation": create, "node_clones": clones},
        identity_mapping="CUPTI_DIRECT_INSTANTIATION_NODE_CLONES_V1",
    )
    return result
