# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY Ops allocator provenance across native, graph and schema4 readers."""

import json
from types import SimpleNamespace

import pytest

from collector import glm53flash_graph_export as graph
from collector import glm53flash_sglang_prefill_export as prefill
from collector import glm53flash_validation as native
from collector.fpm_forward.glm53flash_validation import _same_sglang_policy
from collector.fpm_forward.sglang_allocator import request_policy
from collector.glm53flash_graph_callbacks import resolve_registry
from collector.glm53flash_jsonl import file_sha256, iter_records

from .test_glm53flash_graph_export import control_fixture
from .test_glm53flash_graph_export import fixture as graph_fixture
from .test_glm53flash_ops_evidence import put, put_lines
from .test_glm53flash_sglang_allocator import worker
from .test_glm53flash_sglang_prefill_export import control_fixture as prefill_control
from .test_glm53flash_sglang_prefill_export import fixture as prefill_fixture

pytestmark = pytest.mark.unit


def known(root, run, requested=16384, torch_sha="b" * 64):
    run.setdefault("plan", {}).setdefault("options", {})["sglang_allocator_max_split_size_mb"] = requested
    run.setdefault("cell", {})["sglang_allocator_max_split_size_mb"] = requested
    policy = request_policy(requested)
    path = root / "sglang-provenance.json"
    provenance = json.loads(path.read_bytes())
    provenance["allocator_policy"] = policy
    put(path, provenance)
    for rank in range(run["key"][2]):
        hardware = json.loads((root / f"state-layout-rank-{rank}.json").read_bytes())["hardware"]
        receipt = worker(rank, policy, provenance["execution_identity"], hardware)
        receipt["run_id"] = provenance["run_id"]
        receipt["torch"]["files"]["version.py"]["sha256"] = torch_sha
        path = root / f"allocator-identity-rank-{rank}.json"
        put(path, receipt)
        digest = file_sha256(path)
        registry_sha = None
        source_path = root / f"capture-source-nodes-rank-{rank}.jsonl"
        if source_path.exists():
            source = next(iter_records(source_path))
            source["provenance"]["allocator_policy"] = policy
            put_lines(source_path, [source])
            callback = root / f"graph-clones-rank-{rank}-capture-0.json"
            registry = resolve_registry(source, json.loads(callback.read_bytes()))
            registry["instantiation_receipt"] = {"file": callback.name, "sha256": file_sha256(callback)}
            put_lines(root / f"capture-nodes-rank-{rank}.jsonl", [registry])
            registry_sha = graph._semantic_sha(registry)
        for stem in ("forward-rank", "graph-forward-rank", "rank"):
            path = root / f"{stem}-{rank}.jsonl"
            if not path.exists():
                continue
            rows = list(iter_records(path))
            for row in rows:
                row["allocator_policy"] = policy
                row["allocator_identity_sha256"] = digest
                if stem == "graph-forward-rank" and registry_sha is not None and run["role"] == "calibration":
                    row["capture_registry_sha256"] = registry_sha
            put_lines(path, rows)


@pytest.mark.parametrize("requested", [None, 16384, 32768])
def test_actual_policy_reaches_common_and_graph_reader_without_changing_raw_config(tmp_path, monkeypatch, requested):
    run, root = graph_fixture(tmp_path, monkeypatch, "holdout")
    before = file_sha256(root / "sglang-resolved-config.json")
    known(root, run, requested)
    common = native.load_native(run, root)
    proof = graph.read_graph_run(root, run)
    assert common["execution_policy"] == proof["execution_policy"]
    assert common["execution_policy"]["normalization"] == "resolved_server_args_and_native_allocator_v2"
    assert common["_allocator_policy"]["requested_policy"]["max_split_size_mb"] == requested
    assert "allocator-identity-rank-0.json" in proof["files"]
    assert before == file_sha256(root / "sglang-resolved-config.json")


@pytest.mark.parametrize("reader", ["common", "graph"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra_rank",
        "wrong_rank",
        "wrong_run",
        "hardware",
        "effective",
        "policy",
        "forward_sha",
        "seed_sha",
        "strip_provenance",
        "strip_files_and_provenance",
    ],
)
def test_each_original_rank_and_forward_is_required_even_with_unchanged_serverargs(
    tmp_path, monkeypatch, reader, damage
):
    run, root = graph_fixture(tmp_path, monkeypatch, "holdout")
    known(root, run, None)
    path = root / "allocator-identity-rank-1.json"
    value = json.loads(path.read_bytes())
    if damage == "missing":
        path.unlink()
    elif damage == "extra_rank":
        put(root / "allocator-identity-rank-2.json", value)
    elif damage in ("forward_sha", "seed_sha"):
        for stem in ("forward", "graph-forward"):
            path = root / f"{stem}-rank-1.jsonl"
            rows = list(iter_records(path))
            for row in rows:
                if row["stage"] == ("seed" if damage == "seed_sha" else "measure"):
                    row["allocator_identity_sha256"] = "f" * 64
            put_lines(path, rows)
    elif damage.startswith("strip"):
        path = root / "sglang-provenance.json"
        value = json.loads(path.read_bytes())
        del value["allocator_policy"]
        put(path, value)
        if damage == "strip_files_and_provenance":
            for path in root.glob("allocator-identity-rank-*.json"):
                path.unlink()
    else:
        if damage == "wrong_rank":
            value["rank"] = 0
        elif damage == "wrong_run":
            value["run_id"] = "another"
        elif damage == "hardware":
            value["hardware"]["uuid"] = "different-GPU"
        elif damage == "effective":
            value["max_split_size_bytes"] = 16384 * (1 << 20)
        else:
            value["requested_policy"]["max_split_size_mb"] = 16384
        put(path, value)
    with pytest.raises(ValueError, match="allocator"):
        if reader == "common":
            native.load_native(run, root)
        else:
            graph.read_graph_run(root, run)


@pytest.mark.parametrize(
    "damage", ["missing_cell", "missing_plan", "runtime_cell", "different", "boolean", "other_backend"]
)
def test_frozen_allocator_request_is_not_inferred_from_actual_policy(tmp_path, monkeypatch, damage):
    run, root = graph_fixture(tmp_path, monkeypatch, "holdout")
    known(root, run)
    key = "sglang_allocator_max_split_size_mb"
    if damage == "missing_cell":
        del run["cell"][key]
    elif damage == "missing_plan":
        del run["plan"]["options"][key]
    elif damage == "runtime_cell":
        run["runtime_cell"] = SimpleNamespace(sglang_allocator_max_split_size_mb=None)
    elif damage == "different":
        run["cell"][key] = 32768
    elif damage == "boolean":
        run["plan"]["options"][key] = run["cell"][key] = True
    else:
        run["key"] = ("vllm", *run["key"][1:])
    with pytest.raises(ValueError, match="allocator|max-split"):
        native.requested_sglang_allocator(run)


def test_legacy_unknown_is_unchanged_and_cannot_satisfy_explicit_request(tmp_path, monkeypatch):
    run, root = graph_fixture(tmp_path, monkeypatch, "holdout")
    before = native.load_native(run, root)
    assert "_allocator_policy" not in before
    assert before["execution_policy"]["normalization"] == "resolved_server_args_except_random_seed_v1"
    run["plan"].setdefault("options", {})["sglang_allocator_max_split_size_mb"] = 16384
    run.setdefault("cell", {})["sglang_allocator_max_split_size_mb"] = 16384
    with pytest.raises(ValueError, match="allocator original rank files"):
        native.load_native(run, root)


@pytest.mark.parametrize("other", ["legacy", "requested", "actual_torch"])
def test_graph_control_compares_observed_allocator_and_keeps_original_config_hash(tmp_path, monkeypatch, other):
    run, root = graph_fixture(tmp_path, monkeypatch)
    independent = control_fixture(tmp_path, monkeypatch)
    known(root, run)
    if other != "legacy":
        known(
            independent["control_root"],
            independent["control_run"],
            32768 if other == "requested" else 16384,
            "c" * 64 if other == "actual_torch" else "b" * 64,
        )
    proof = graph.read_graph_run(root, run)
    assert proof["policy"]["resolved_config_sha256"] == file_sha256(root / "sglang-resolved-config.json")
    with pytest.raises(ValueError, match="execution policies differ"):
        graph.profile_control(root, proof, **independent)


def test_schema4_policy_and_publication_bind_actual_allocator(tmp_path):
    run, root, _ = prefill_fixture(tmp_path / "cal")
    other_run, other_root = prefill_control(tmp_path / "control")
    independent = {"control_run": other_run, "control_root": other_root}
    known(root, run)
    known(independent["control_root"], independent["control_run"])
    proof = prefill.read_prefill_run(root, run)
    assert proof["policy"]["execution_policy_sha256"] == proof["execution_policy"]["sha256"]
    assert proof["execution_policy"]["normalization"] == "resolved_server_args_and_native_allocator_v2"
    assert {f"allocator-identity-rank-{rank}.json" for rank in range(2)} <= proof["files"]
    result = prefill.export_prefill(root, run, root / prefill.BASENAME, **independent)
    assert result["rows"] == 367
    common = native.load_native(run, root)
    assert common["execution_policy"] == proof["execution_policy"]
    native.bind_calibration([root / prefill.BASENAME], run, common)
    # Consistent replacement of both native rank receipts is still another
    # actual policy, even with identical ServerArgs and requested split limit.
    known(independent["control_root"], independent["control_run"], torch_sha="d" * 64)
    changed = prefill.read_prefill_run(independent["control_root"], independent["control_run"])
    assert changed["policy"]["execution_policy_sha256"] != proof["policy"]["execution_policy_sha256"]
    with pytest.raises(ValueError, match="execution policies differ"):
        _same_sglang_policy(proof, changed, "actual calibration/control")


@pytest.mark.parametrize("other", ["legacy", "actual_torch"])
def test_existing_ops_shard_union_rejects_known_unknown_or_actual_allocator_mixing(tmp_path, monkeypatch, other):
    run, root = graph_fixture(tmp_path, monkeypatch, "holdout")
    independent = control_fixture(tmp_path, monkeypatch)
    known(root, run)
    if other == "actual_torch":
        known(independent["control_root"], independent["control_run"], torch_sha="d" * 64)
    first = native.load_native(run, root)
    second = native.load_native(independent["control_run"], independent["control_root"])
    with pytest.raises(ValueError, match="Ops calibration shards"):
        native._sharded_calibration_rows([(run, first), (independent["control_run"], second)], {})


@pytest.mark.parametrize("mode,phase", [("eager", "decode"), ("native_eager_prefill", "prefill")])
def test_optional_native_command_forwards_allocator_and_preserves_omitted_default(tmp_path, mode, phase):
    from collector.collect_glm53flash import native_command
    from collector.fpm_forward.sglang_allocator import OPTION

    args = ("sglang", "/models/TEST_ONLY", "revision", 2, phase, tmp_path, tmp_path / "corpus")
    default = native_command(*args, ops_execution_mode=mode)
    explicit_none = native_command(*args, ops_execution_mode=mode, sglang_allocator_max_split_size_mb=None)
    actual = native_command(*args, ops_execution_mode=mode, sglang_allocator_max_split_size_mb=16384)
    assert default == explicit_none and OPTION not in default
    pos = actual.index(OPTION)
    assert actual[pos + 1] == "16384" and actual.count(OPTION) == 1
    del actual[pos : pos + 2]
    assert actual == default
    with pytest.raises(ValueError, match="allocator"):
        native_command("vllm", *args[1:], sglang_allocator_max_split_size_mb=16384)
    with pytest.raises(ValueError, match="integer"):
        native_command(*args, sglang_allocator_max_split_size_mb=True)
