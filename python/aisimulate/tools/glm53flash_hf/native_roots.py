# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit collection/pod boundary for original single-node formal attempts.

This is metadata validation, not a new native read or an alias/symlink bypass.
Unmarked documents retain their historical pod-root contract.
"""

from pathlib import Path

SCOPE = "glm53flash_collection_node0000_v1"
FIELD = "native_root_scope"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scope(value):
    if FIELD not in value:
        require(
            "original_pod_root" not in value and "accepted_native_roots" not in value,
            "pod mapping requires explicit native root scope",
        )
        return None
    require(value[FIELD] == SCOPE, "unknown native root scope")
    return SCOPE


def uniform(values):
    scopes = {scope(value) for value in values}
    require(len(scopes) <= 1, "mixed native root scopes")
    return next(iter(scopes), None)


def collection(value, base):
    require(scope(value) == SCOPE, "collection root requires explicit scope")
    text = value["raw_root"]
    require(
        isinstance(text, str) and text and ".." not in Path(text).parts and "\\" not in text and "\x00" not in text,
        "unsafe native collection root",
    )
    root = Path(text)
    require(root.is_absolute(), "native collection root must be absolute")
    cid = value["cell_id"]
    require(
        isinstance(cid, str)
        and cid not in {"", ".", ".."}
        and Path(cid).parts == (cid,)
        and root.parts[-3:] == ("cells", cid, "raw"),
        "native collection root differs from child",
    )
    pod = root / "node0000"
    if "original_pod_root" in value:
        require(value["original_pod_root"] == str(pod), "original pod mapping differs")
    return root, pod


def fields(value, base):
    if scope(value) is None:
        return {}
    _root, pod = collection(value, base)
    return {FIELD: SCOPE, "original_pod_root": str(pod)}


def receipts(evidence):
    names = []
    for item in evidence["receipts"]:
        name = item["path"]
        parts = Path(name).parts
        require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in parts
            and len(parts) >= 2
            and parts[0] == "node0000"
            and str(Path(name)) == name,
            "native receipt is outside the original pod",
        )
        names.append(name)
    require(len(names) == len(set(names)), "duplicate native receipt path")
    require("node0000/collector-provenance.json" in names, "original pod provenance receipt missing")


def inventory_root(records, prefix):
    """Check exact single-pod membership including empty foreign directories."""
    root = records.get(prefix)
    require(root is not None and root["kind"] == "directory", "native collection directory missing")
    pod_name = prefix + "/node0000"
    pod = records.get(pod_name)
    require(pod is not None and pod["kind"] == "directory", "original native pod directory missing")
    for name, row in records.items():
        if name.startswith(prefix + "/"):
            require(
                (name == pod_name or name.startswith(pod_name + "/")) and row["kind"] in {"file", "directory"},
                "extra pod or unsafe member in native collection",
            )
