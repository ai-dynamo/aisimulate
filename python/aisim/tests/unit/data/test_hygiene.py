# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure candidate-hygiene rules, tested with in-memory fixtures."""

import pytest

from aisim_core.data.hygiene import (
    CandidateFile,
    CandidatePolicy,
    check_candidate,
    check_content,
    check_creator,
    check_listing,
)

pytestmark = pytest.mark.unit

POLICY = CandidatePolicy(allowed_creators=frozenset({"Alice"}), max_files=3, max_total_bytes=1000, max_file_bytes=600)
GOOD = CandidateFile("data/eval/x/record.json", 10)
PQ = CandidateFile("data/eval/x/truth.parquet", 100)


def codes(findings):
    return {f.code for f in findings}


def test_creator_allowlist_fails_closed():
    assert check_creator(POLICY, "alice") == []  # case-insensitive
    assert codes(check_creator(POLICY, "mallory")) == {"creator-not-allowed"}
    assert codes(check_creator(POLICY, None)) == {"creator-unknown"}
    assert codes(check_creator(CandidatePolicy(), "alice")) == {"creator-not-allowed"}


def test_listing_accepts_valid():
    assert check_listing(POLICY, [GOOD, PQ]) == []


@pytest.mark.parametrize(
    ("file", "code"),
    [
        (CandidateFile("data/../etc/x.json", 1), "path-traversal"),
        (CandidateFile("/data/x.json", 1), "path-unsafe"),
        (CandidateFile("data\\x.json", 1), "path-unsafe"),
        (CandidateFile("data//x.json", 1), "path-traversal"),
        (CandidateFile("other/x.json", 1), "path-root"),
        (CandidateFile("data/x.pkl", 1), "suffix"),
        (CandidateFile("data/x.py", 1), "suffix"),
        (CandidateFile("data/x.txt", 1), "suffix"),
        (CandidateFile("data/.gitattributes", 1), "path-reserved"),
        (CandidateFile("data/big.parquet", 601), "file-too-large"),
        (CandidateFile("data/l.json", 1, is_symlink=True), "symlink"),
        (CandidateFile("data/l.parquet", 1, is_lfs=True), "lfs-pointer"),
    ],
)
def test_listing_rejects(file, code):
    assert code in codes(check_listing(POLICY, [file]))


def test_listing_limits_and_duplicates():
    many = [CandidateFile(f"data/{i}.json", 1) for i in range(4)]
    assert "too-many-files" in codes(check_listing(POLICY, many))
    assert "too-large" in codes(
        check_listing(POLICY, [CandidateFile("data/a.json", 500), CandidateFile("data/b.json", 600)])
    )
    assert "duplicate-path" in codes(
        check_listing(POLICY, [CandidateFile("data/A.json", 1), CandidateFile("data/a.json", 1)])
    )
    assert codes(check_listing(POLICY, [])) == {"empty"}


def test_content_parquet_and_json():
    assert check_content(POLICY, PQ, b"PAR1" + b"x" * 8, b"PAR1") == []
    assert "parquet-magic" in codes(check_content(POLICY, PQ, b"notparquet", b"PAR1"))
    assert "parquet-magic" in codes(check_content(POLICY, PQ, b"PAR1", b"xxxx"))
    ok = CandidateFile("data/a.json", 2)
    assert check_content(POLICY, ok, b"{}") == []
    assert "json-invalid" in codes(check_content(POLICY, CandidateFile("data/a.json", 4), b"{NaN"))
    assert "json-invalid" in codes(check_content(POLICY, CandidateFile("data/a.json", 6), b'{"a":NaN}'[:6] + b""))
    assert "json-incomplete" in codes(check_content(POLICY, CandidateFile("data/a.json", 50), b"{}"))
    assert "json-too-large" in codes(check_content(CandidatePolicy(max_json_bytes=1), ok, b"{}"))


@pytest.mark.parametrize("head", [b"\x7fELF....", b"#!/bin/sh", b"PK\x03\x04zz", b"MZ\x90\x00"])
def test_content_rejects_executables_even_when_renamed(head):
    assert "forbidden-content" in codes(check_content(POLICY, CandidateFile("data/a.json", len(head)), head))


def test_lfs_pointer_content():
    head = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n"
    assert "lfs-pointer" in codes(check_content(POLICY, CandidateFile("data/a.json", len(head)), head))


def test_candidate_aggregate_and_missing_sample():
    samples = {GOOD.path: (b"{" + b" " * 8 + b"}", b""), PQ.path: (b"PAR1" + b"x" * 8, b"PAR1")}
    assert check_candidate(POLICY, "alice", [GOOD, PQ], samples).ok
    report = check_candidate(POLICY, "alice", [GOOD, PQ], {GOOD.path: samples[GOOD.path]})
    assert not report.ok and codes(report.errors) == {"no-sample"}
    assert not check_candidate(POLICY, "mallory", [GOOD], {GOOD.path: samples[GOOD.path]}).ok
