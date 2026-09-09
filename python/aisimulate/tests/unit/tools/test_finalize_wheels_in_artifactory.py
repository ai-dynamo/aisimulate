# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from tools.finalize_wheels_in_artifactory import FinalizeError, build_manifest, finalize


class FakeArtifacts:
    def __init__(self) -> None:
        self.files = {
            "aisimulate-2.0.0-cp311-abi3-manylinux_2_28_x86_64.whl": (b"unified", "a" * 64),
        }
        self.uploads: list[tuple[str, bytes]] = []

    def list_files(self, subpath: str) -> list[dict]:
        return [
            {"filename": name, "sha256": checksum, "size": len(payload)}
            for name, (payload, checksum) in self.files.items()
        ]

    def upload(self, path: str, payload: bytes) -> None:
        self.uploads.append((path, payload))


def test_finalize_writes_manifest_before_completion_marker() -> None:
    artifacts = FakeArtifacts()

    manifest = finalize(
        artifacts,
        subpath="post-merge/sha/42/2",
        repository="ai-dynamo/aisimulate",
        commit_sha="sha",
        run_id=42,
        run_attempt=2,
    )

    assert [path for path, _ in artifacts.uploads] == [
        "post-merge/sha/42/2/_MANIFEST.json",
        "post-merge/sha/42/2/_COMPLETE.json",
    ]
    marker = json.loads(artifacts.uploads[1][1])
    assert marker["manifest_sha256"] == hashlib.sha256(artifacts.uploads[0][1]).hexdigest()
    assert marker["commit_sha"] == "sha"
    assert manifest["schemaVersion"] == "aisimulate-wheel-manifest/1.0.0"
    assert manifest["wheels"][0]["filename"] == "aisimulate-2.0.0-cp311-abi3-manylinux_2_28_x86_64.whl"


def test_manifest_requires_an_aisimulate_wheel() -> None:
    with pytest.raises(FinalizeError, match="at least one aisimulate wheel"):
        build_manifest(
            [],
            repository="ai-dynamo/aisimulate",
            commit_sha="sha",
            run_id=1,
            run_attempt=1,
        )

    with pytest.raises(FinalizeError, match="non-aisimulate wheels"):
        build_manifest(
            [{"filename": "aiconfigurator-2.0.0-py3-none-any.whl", "sha256": "a" * 64, "size": 1}],
            repository="ai-dynamo/aisimulate",
            commit_sha="sha",
            run_id=1,
            run_attempt=1,
        )


def test_manifest_accepts_one_unified_wheel_per_platform() -> None:
    wheels = [
        {
            "filename": f"aisimulate-2.0.0-cp311-abi3-{platform}.whl",
            "sha256": character * 64,
            "size": 1,
        }
        for platform, character in (
            ("manylinux_2_28_x86_64", "a"),
            ("manylinux_2_28_aarch64", "b"),
            ("macosx_11_0_arm64", "c"),
        )
    ]

    manifest = build_manifest(
        wheels,
        repository="ai-dynamo/aisimulate",
        commit_sha="sha",
        run_id=1,
        run_attempt=1,
    )

    assert [wheel["filename"] for wheel in manifest["wheels"]] == sorted(wheel["filename"] for wheel in wheels)
