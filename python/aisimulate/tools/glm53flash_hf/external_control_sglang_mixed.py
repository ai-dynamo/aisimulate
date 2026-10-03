# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local proposed v3: combine exact original SG admissions by deployment.

Derived from this repository's external_control_current.py at bf5e1fbc.
No original admission, qualification, plan or started record is synthesized.
inspect_partial is preparation only; the publication entry requires all four.
"""

from pathlib import Path

if __package__:
    from . import external_control_current as current
else:
    import external_control_current as current

require = current.require
ANCHORS = {"launcher_manifest", "admission", "source_identity", "cache_hook"}


def inspect_partial(document, get, *, require_complete=False):
    require(
        document.get("schema") == current.MIXED_SCHEMA and document.get("adapter") == current.MIXED_SGLANG,
        "mixed admissions are explicit SG v3 only",
    )
    require("anchors" not in document, "mixed admission must not inherit an ambiguous global anchor")
    definitions = document["deployment_controls"]
    require(
        isinstance(definitions, dict) and definitions and set(definitions) <= current.DEPLOYMENTS,
        "unknown/missing deployment control",
    )
    if require_complete:
        require(set(definitions) == current.DEPLOYMENTS, "all four actual deployment admissions required")
    task = current._control().absolute(document["original_task_root"])
    frozen = current.Frozen(get)
    groups, qualifications, admissions, anchors = {}, {}, {}, {}
    contexts = []
    for deployment, refs in sorted(definitions.items()):
        require(
            isinstance(refs, dict) and set(refs) == ANCHORS,
            "exact deployment manifest/admission/source/cache references required",
        )
        for value in refs.values():
            require(
                isinstance(value, dict) and set(value) == {"path", "sha256"},
                "explicit original byte reference required",
            )
            frozen.ref(value)
        names = {key: ref["path"] for key, ref in refs.items()}
        members = frozen.manifest(names["launcher_manifest"], refs["launcher_manifest"]["sha256"])
        require(names["admission"] in members, "deployment admission outside original launcher manifest")
        admission = frozen.read(names["admission"])
        original_view = {"anchors": names}
        # The unchanged default v2 call still requires all four. This explicit
        # internal scope selects just the real qualified deployment from its
        # original admission; it does not add a missing qualification.
        context = current._sglang_frozen(original_view, frozen, admission, task, deployments={deployment})
        current._native_plans(context, frozen, task, names)
        require(context["groups"].keys() == {deployment}, "wrong deployment admission group")
        contexts.append(context)
        groups[deployment] = context["groups"][deployment]
        qualifications[deployment] = context["qualifications"][deployment]
        admissions[deployment], anchors[deployment] = admission, names
    first = contexts[0]
    for context in contexts[1:]:
        require(
            all(context[key] == first[key] for key in ("backend", "version", "host", "producer", "cpu_root", "source")),
            "mixed admissions changed original host/producer/factory identity",
        )
    children = current._children(groups, expected_deployments=definitions)
    # All 72 source children are separately visible even when only 54 have an
    # admission. They are never returned as admitted/executed children.
    source_groups = {}
    prepared = Path(first["cpu_root"]) / "prepared/formal-inputs"
    for deployment in sorted(current.DEPLOYMENTS):
        values = []
        for role in ("calibration", "holdout"):
            root = prepared / deployment / role
            manifest = frozen.read(str(root / "shard-manifest.json"))
            for item in manifest["shards"]:
                values.append(
                    dict(
                        child_cell_id=item["child_cell_id"],
                        child_plan_sha256=item["child_plan_sha256"],
                        parent_plan_sha256=item["parent_plan_sha256"],
                        role=role,
                        phase=item["phase"],
                        deployment=deployment,
                        original_point_ids=[p["original_point_id"] for p in item["point_map"]],
                        original_identity=item,
                        native_directory=str(task / root / "native" / item["child_cell_id"]),
                    )
                )
        source_groups[deployment] = values
    source_children = current._children(source_groups)
    require(
        all(source_children[cid] == child for cid, child in children.items()), "admission changed frozen source child"
    )
    return dict(
        first,
        groups=groups,
        children=children,
        qualifications=qualifications,
        source_groups=source_groups,
        source_children=source_children,
        deployment_admissions=admissions,
        deployment_anchors=anchors,
        admission={"schema": "sg_original_deployment_admissions_v1", "deployments": admissions},
        files=frozen.files,
        preparation_only=not require_complete,
    )


def frozen_contract(document, get):
    return inspect_partial(document, get, require_complete=True)
