# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish checked E2E source snapshots as reviewable HF deltas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from scripts.e2e_accuracy.hf_cache import INDEX_PATH, artifact, safe_path, validate_snapshot, write_json


def publish_snapshot(root: Path, repo_id: str, *, signoff: str, api: HfApi | None = None) -> dict:
    """Reuse remote file objects; never delete or auto-promote input evidence."""
    if "\n" in signoff or "<" not in signoff or not signoff.endswith(">"):
        raise ValueError("A configured Git identity is required for DCO sign-off")
    validate_snapshot(root)
    index = json.loads((root / INDEX_PATH).read_bytes())
    snapshot = json.loads(safe_path(root, index["snapshot"]["path"]).read_bytes())
    publication = snapshot["publication"]
    if (
        publication.get("redistribution_status") != "unresolved"
        or not publication.get("attribution")
        or not publication.get("terms_request")
    ):
        raise ValueError("Preserve attribution and the unresolved upstream data-license request")
    api = api or HfApi()
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    if not info.private:
        raise ValueError("InferenceX cache publication requires a private dataset")

    def remote_json(path, revision):
        try:
            local = hf_hub_download(repo_id, path, repo_type="dataset", revision=revision, token=api.token)
            return json.loads(Path(local).read_bytes())
        except EntryNotFoundError:
            return None

    previous = remote_json(INDEX_PATH, info.sha)
    if previous and previous.get("snapshot_id") == index["snapshot_id"]:
        return {"status": "unchanged", "snapshot_id": index["snapshot_id"], "revision": info.sha}
    title = f"Weekly InferenceX delta {index['snapshot_id']}"
    for discussion in api.get_repo_discussions(
        repo_id, repo_type="dataset", discussion_type="pull_request", discussion_status="open"
    ):
        if discussion.title == title:
            candidate = remote_json(INDEX_PATH, f"refs/pr/{discussion.num}")
            if candidate and candidate.get("snapshot_id") == index["snapshot_id"]:
                return {"status": "candidate_exists", "snapshot_id": index["snapshot_id"], "url": discussion.url}
    # Preserve developer/other source indexes when refreshing InferenceX.
    category = remote_json("catalog/e2e/index.json", info.sha) or {"schema_version": "e2e-index/v1", "sources": {}}
    if category.get("schema_version") != "e2e-index/v1" or not isinstance(category.get("sources"), dict):
        raise ValueError("Unsupported E2E category index")
    category["sources"]["inferencex"] = artifact(root, INDEX_PATH, "e2e-source-index/v1")
    write_json(root / "catalog/e2e/index.json", category)
    paths = {INDEX_PATH, "catalog/e2e/index.json", index["snapshot"]["path"]}
    paths.update(a["artifact"]["path"] for a in snapshot["acquisitions"])
    paths.update(a["path"] for p in index["partitions"] for a in p["artifacts"])
    remote_files = {entry.rfilename: entry for entry in info.siblings}
    operations = []
    transferred = 0
    for relative in sorted(paths):
        local = safe_path(root, relative)
        content = local.read_bytes()
        remote = remote_files.get(relative)
        if remote is not None:
            if remote.lfs is not None:
                same = hashlib.sha256(content).hexdigest() == remote.lfs.sha256
            else:
                same = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == remote.blob_id
            if same:
                continue
        operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=local))
        transferred += len(content)
    if not operations:
        return {"status": "unchanged", "snapshot_id": index["snapshot_id"], "revision": info.sha}
    commit = api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=operations,
        parent_commit=info.sha,
        commit_message=title,
        commit_description=f"Frozen source evidence; review before promotion.\n\nSigned-off-by: {signoff}",
        create_pr=True,
    )
    return {
        "status": "candidate_created",
        "snapshot_id": index["snapshot_id"],
        "revision": commit.oid,
        "url": commit.pr_url,
        "changed_files": len(operations),
        "upload_bytes": transferred,
    }
