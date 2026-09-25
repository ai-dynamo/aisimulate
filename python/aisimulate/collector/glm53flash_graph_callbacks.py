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
EVENT_RECORD_CUDART_SHA256 = "7bdba2b5b08cbdc85203c41cc94598adedb1bcfea7cb574ca693ac73599e4e63"


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
        self.subscription_closed = False
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
            self.subscription_closed = True
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
            "callback_subscription_closed": self.subscription_closed,
        }


def _direct_instantiation(registry, receipt):
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
    if (
        len(originals) != len(nodes)
        or len(clones) != len(nodes)
        or any(
            type(row["node_id"]) is not int or row["node_id"] <= 0 or type(row["node_type"]) is not int for row in nodes
        )
    ):
        raise ValueError("native instantiation lacks complete unique node coverage")
    ids, handles, originals_by_handle = {}, set(), set()
    for clone in clones:
        original = clone["original_node_id"]
        raw = clone.get("raw_fields", {})
        if (
            original not in originals
            or original in ids
            or type(original) is not int
            or type(clone["node_id"]) is not int
            or clone["node_id"] <= 0
            or type(clone["node_type"]) is not int
            or raw.get("nodeType") != clone["node_type"]
            or any(type(raw.get(key)) is not int or raw[key] <= 0 for key in ("node", "originalNode"))
            or raw["node"] in handles
            or raw["originalNode"] in originals_by_handle
            or raw["node"] == raw["originalNode"]
        ):
            raise ValueError("native instantiation has an unknown or duplicate node identity/handle")
        handles.add(raw["node"])
        originals_by_handle.add(raw["originalNode"])
        ids[original] = clone["node_id"]
    if len(set(ids.values())) != len(nodes):
        raise ValueError("native instantiation node mapping is not one-to-one")
    return create, clones, originals, ids


def _event_record_mismatches(registry, receipt, *, allow_pending_memcpy=False, allow_memset_query=False):
    create, clones, originals, ids = _direct_instantiation(registry, receipt)
    mismatches = [row for row in clones if row["node_type"] != originals[row["original_node_id"]]["node_type"]]
    copies = [
        row for row in mismatches if (originals[row["original_node_id"]]["node_type"], row["node_type"]) == (1, 0)
    ]
    if copies and allow_pending_memcpy:
        from .glm53flash_graph_nodes import memcpy_activity_requirement

        libraries = registry.get("native_api_libraries", {})
        if (
            receipt.get("callback_subscription_closed") is not True
            or libraries.get("cudart", {}).get("sha256") != EVENT_RECORD_CUDART_SHA256
            or libraries.get("cupti", {}).get("sha256") != QUALIFIED_CUPTI_SHA256
        ):
            raise ValueError("pending memcpy mapping differs from its qualified native providers")
        for clone in copies:
            original = originals[clone["original_node_id"]]
            memcpy_activity_requirement(original)
            if original["memcpy_params"]["source_node_handle"] != clone["raw_fields"]["originalNode"]:
                raise ValueError("memcpy source query handle differs from its exact clone callback")
        mismatches = [row for row in mismatches if row not in copies]
    query_types = (2, 7) if allow_memset_query else (7,)
    if any(
        originals[row["original_node_id"]]["node_type"] not in query_types or row["node_type"] != 0
        for row in mismatches
    ):
        raise ValueError("native instantiation has an unqualified changed node type")
    return create, clones, originals, ids, mismatches


def _query_method(originals, mismatches):
    # Preserve the original EventRecord-only proof byte contract. The explicit
    # Memset opt-in gets a distinct method, with each exact source type retained.
    if any(originals[row["original_node_id"]]["node_type"] == 2 for row in mismatches):
        return "CUDA13_MEMSET_EVENT_RECORD_CLONE_QUERY_V1"
    return "CUDA13_EVENT_RECORD_CLONE_QUERY_V1"


def record_event_record_types(
    api, graph, registry, receipt, path, *, allow_pending_memcpy=False, allow_memset_query=False
):
    """Query qualified EventRecord / explicitly enabled Memset mismatches.

    The captured source types remain authoritative. Actual GPU probe 612570 on
    the exact libraries below found callback type0 versus native clone types7/2.
    Memset additionally requires complete positive replay activity, independently
    checked by bind_replay_kernels. Capture query success supplies no timing.
    Its Empty5→clone0 case is deliberately rejected. No CUDA API runs inside a
    callback, no source handle is dereferenced after Torch may have freed it,
    and the caller retains the real graph executable throughout this function.
    """
    create, _, originals, _, mismatches = _event_record_mismatches(
        registry, receipt, allow_pending_memcpy=allow_pending_memcpy, allow_memset_query=allow_memset_query
    )
    if not mismatches:
        return None
    if receipt.get("callback_subscription_closed") is not True:
        raise RuntimeError("deferred node type queries require a closed callback subscription")
    libraries = registry.get("native_api_libraries")
    if (
        api.libraries != libraries
        or libraries.get("cudart", {}).get("sha256") != EVENT_RECORD_CUDART_SHA256
        or libraries.get("cupti", {}).get("sha256") != QUALIFIED_CUPTI_SHA256
    ):
        raise RuntimeError("deferred native node query differs from its actual qualified provider")
    handle = graph.raw_cuda_graph_exec()
    if handle != create["raw_fields"]["graphExec"]:
        raise RuntimeError("deferred node query does not retain its original executable")
    proof = {
        "method": _query_method(originals, mismatches),
        "native_api_libraries": copy.deepcopy(libraries),
        "graph_exec_handle": handle,
        "graph_exec_id": receipt["actual_graph_exec_id"],
        "callback_subscription_closed": True,
        "completed": False,
        "queries": [],
    }
    with path.open("x") as stream:

        def persist():
            stream.seek(0)
            json.dump(proof, stream, indent=2)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())

        persist()
        for clone in mismatches:
            source_type = originals[clone["original_node_id"]]["node_type"]
            row = {
                "original_node_id": clone["original_node_id"],
                "node_id": clone["node_id"],
                "source_node_handle": clone["raw_fields"]["originalNode"],
                "clone_node_handle": clone["raw_fields"]["node"],
                "source_node_type": source_type,
                "callback_node_type": 0,
                "api": "cudaGraphNodeGetType",
                "rc": None,
                "native_node_type": None,
            }
            proof["queries"].append(row)
            persist()  # Preserve handles before entering the native API.
            actual = ctypes.c_int(-1)
            row["rc"] = api.runtime.cudaGraphNodeGetType(row["clone_node_handle"], ctypes.byref(actual))
            row["native_node_type"] = actual.value
            persist()
            if row["rc"] != 0 or actual.value != source_type:
                raise RuntimeError("deferred native node clone query failed or changed type")
        proof["completed"] = True
        persist()
    return proof


def resolve_registry(registry, receipt, node_type_proof=None, *, allow_pending_memcpy=False, allow_memset_query=False):
    """Derive executable ownership, retaining strict native type evidence."""
    create, clones, originals, ids, mismatches = _event_record_mismatches(
        registry, receipt, allow_pending_memcpy=allow_pending_memcpy, allow_memset_query=allow_memset_query
    )
    if mismatches:
        libraries = registry.get("native_api_libraries", {})
        expected = {
            "method": _query_method(originals, mismatches),
            "native_api_libraries": libraries,
            "graph_exec_handle": create["raw_fields"]["graphExec"],
            "graph_exec_id": receipt["actual_graph_exec_id"],
            "callback_subscription_closed": True,
            "completed": True,
            "queries": [
                {
                    "original_node_id": row["original_node_id"],
                    "node_id": row["node_id"],
                    "source_node_handle": row["raw_fields"]["originalNode"],
                    "clone_node_handle": row["raw_fields"]["node"],
                    "source_node_type": originals[row["original_node_id"]]["node_type"],
                    "callback_node_type": 0,
                    "api": "cudaGraphNodeGetType",
                    "rc": 0,
                    "native_node_type": originals[row["original_node_id"]]["node_type"],
                }
                for row in mismatches
            ],
        }
        if (
            node_type_proof != expected
            or receipt.get("callback_subscription_closed") is not True
            or libraries.get("cudart", {}).get("sha256") != EVENT_RECORD_CUDART_SHA256
            or libraries.get("cupti", {}).get("sha256") != QUALIFIED_CUPTI_SHA256
            or any(
                type(row[key]) is not int
                for row in node_type_proof["queries"]
                for key in (
                    "original_node_id",
                    "node_id",
                    "source_node_handle",
                    "clone_node_handle",
                    "source_node_type",
                    "callback_node_type",
                    "rc",
                    "native_node_type",
                )
            )
        ):
            raise ValueError("native node callback mismatch lacks its exact deferred native type proof")
    elif node_type_proof is not None:
        raise ValueError("unexpected node type evidence cannot alter matching native clone types")
    result = copy.deepcopy(registry)
    by_original = {row["original_node_id"]: row for row in clones}
    for row in result["nodes"]:
        row["capture_node_id"] = row["node_id"]
        clone = by_original[row["node_id"]]
        if row["node_type"] == 1 and clone["node_type"] == 0:
            from .glm53flash_graph_nodes import memcpy_activity_requirement

            row["memcpy_activity_requirement"] = memcpy_activity_requirement(row)
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
    if node_type_proof is not None:
        result["native_instantiation"]["event_record_type_proof"] = copy.deepcopy(node_type_proof)
    return result
