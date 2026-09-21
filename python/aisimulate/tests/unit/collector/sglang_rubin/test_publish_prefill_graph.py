# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact timing reducers and all-or-nothing portable graph publication."""

import copy
import hashlib
import json
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from collector import provenance
from collector.sglang_rubin import publish_prefill_graph as publisher

pytestmark = pytest.mark.unit


@pytest.fixture
def identity():
    return publisher._read(publisher.IDENTITY_PATH)


def attention_data(identity):
    contexts = []
    for key in identity["admitted_contexts"]:
        contexts.append(
            {
                "shape": {
                    "new_tokens": [key["input_seq_len"]] * key["batch_size"],
                    "past_kv": [key["prefix_len"]] * key["batch_size"],
                },
                "arms": {
                    arm: {
                        "timing": {
                            "samples": [
                                {"fixture": f, "window": w, "sample": s, "gpu_ms": 10.0 + f + w / 10 + s / 100}
                                for f in (0, 1)
                                for w in range(3)
                                for s in range(5)
                            ]
                        }
                    }
                    for arm in publisher.ATTENTION_ARMS
                },
            }
        )
    return {"status": "passed", "primary_arm": "joint_attention_sequence", "contexts": contexts}


def communication_data(identity):
    boundaries = []
    contract = identity["table_contracts"]["communication"]
    for key in contract["key_inventory"]:
        for protocol, captured, repeats in (("single_replay", 1, 1), ("block_5x20", 5, 20)):
            count = captured * repeats
            samples = []
            for w in range(3):
                for s in range(10):
                    # Alternating the slowest rank distinguishes mean(max(ranks))
                    # from the incorrect max(mean(each rank)) reduction.
                    ranks = [float(count)] * 4
                    ranks[s % 4] *= 2 + w + s / 10
                    maximum = max(ranks)
                    samples.append(
                        {
                            "window": w,
                            "sample": s,
                            "rank_cuda_ms": ranks,
                            "max_rank_cuda_ms": maximum,
                            "per_call_max_rank_cuda_ms": maximum / count,
                        }
                    )
            boundaries.append(
                {
                    "tokens": key["num_tokens"],
                    "role": key["boundary_role"],
                    "mode": "graph",
                    "protocol": protocol,
                    "path": contract["kernel_source_by_tokens"][str(key["num_tokens"])],
                    "timing": {
                        "normalization": {
                            "captured_calls": captured,
                            "replays_per_sample": repeats,
                            "native_calls_per_sample": count,
                        },
                        "samples": samples,
                    },
                }
            )
    return {"status": "diagnostic_complete", "boundaries": boundaries}


def test_attention_retains_every_sample_and_excludes_control_cost(identity):
    data = attention_data(identity)
    before = publisher._attention_rows(data, identity)
    samples = data["contexts"][0]["arms"]["joint_attention_sequence"]["timing"]["samples"]
    samples[-1]["gpu_ms"] += 30
    data["contexts"][0]["arms"]["separate_intervals"]["timing"]["samples"][0]["gpu_ms"] += 9000
    after = publisher._attention_rows(data, identity)
    assert after[0]["latency"] - before[0]["latency"] == pytest.approx(1.0)
    assert after[1:] == before[1:]
    assert len(after) == 7


@pytest.mark.parametrize(
    "mutation", ["missing_primary", "missing_control", "duplicate_sample", "nan", "heterogeneous", "wrong_prefix"]
)
def test_attention_rejects_incomplete_or_mislabelled_matrix(identity, mutation):
    data = attention_data(identity)
    case = data["contexts"][0]
    samples = case["arms"]["joint_attention_sequence"]["timing"]["samples"]
    if mutation == "missing_primary":
        samples.pop()
    elif mutation == "missing_control":
        del case["arms"]["separate_intervals"]
    elif mutation == "duplicate_sample":
        samples[-1] = copy.deepcopy(samples[0])
    elif mutation == "nan":
        samples[0]["gpu_ms"] = float("nan")
    elif mutation == "heterogeneous":
        case["shape"]["new_tokens"].append(2048)
        case["shape"]["past_kv"].append(0)
    else:
        case["shape"]["past_kv"][0] = 1024
    with pytest.raises(ValueError):
        publisher._attention_rows(data, identity)


def test_communication_uses_aligned_max_then_normalization_then_all_sample_mean(identity):
    data = communication_data(identity)
    row = publisher._communication_rows(data, identity)[0]
    samples = data["boundaries"][1]["timing"]["samples"]
    expected = statistics.mean(max(s["rank_cuda_ms"]) / 100 for s in samples)
    incorrect = max(statistics.mean(s["rank_cuda_ms"][r] for s in samples) / 100 for r in range(4))
    assert row["latency"] == expected
    assert row["latency"] != incorrect
    primary = samples[-1]
    rank = primary["rank_cuda_ms"].index(max(primary["rank_cuda_ms"]))
    primary["rank_cuda_ms"][rank] += 3000
    primary["max_rank_cuda_ms"] += 3000
    primary["per_call_max_rank_cuda_ms"] += 30
    assert publisher._communication_rows(data, identity)[0]["latency"] - row["latency"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_rank",
        "wrong_max",
        "wrong_count",
        "missing_sample",
        "duplicate_boundary",
        "wrong_role",
        "wrong_kernel",
        "zero",
    ],
)
def test_communication_rejects_incomplete_or_wrong_reduction(identity, mutation):
    data = communication_data(identity)
    boundary = data["boundaries"][1]
    sample = boundary["timing"]["samples"][0]
    if mutation == "missing_rank":
        sample["rank_cuda_ms"].pop()
    elif mutation == "wrong_max":
        sample["max_rank_cuda_ms"] += 1
    elif mutation == "wrong_count":
        boundary["timing"]["normalization"]["native_calls_per_sample"] = 5
    elif mutation == "missing_sample":
        boundary["timing"]["samples"].pop()
    elif mutation == "duplicate_boundary":
        data["boundaries"].append(copy.deepcopy(boundary))
    elif mutation == "wrong_role":
        boundary["role"] = "other"
    elif mutation == "wrong_kernel":
        boundary["path"] = "native_allreduce_then_rmsnorm"
    else:
        sample["rank_cuda_ms"][0] = 0
    with pytest.raises(ValueError):
        publisher._communication_rows(data, identity)


def test_portable_paths_reject_escape_and_symlinks(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    target = tmp_path / "outside"
    target.write_text("external")
    (root / "link").symlink_to(target)
    for name in ("../outside", str(target), "./outside", "x/../outside", "link"):
        with pytest.raises(ValueError):
            publisher._relative(root, name)
    with pytest.raises(ValueError, match="Symlink"):
        publisher._inventory(root)


def test_real_identity_requires_complete_raw_bundle(tmp_path, identity):
    with pytest.raises(ValueError, match="incomplete or changed"):
        publisher.qualified_profile(tmp_path, tmp_path, identity)


@pytest.fixture
def publication(tmp_path, identity, monkeypatch):
    """Publication tests isolate staging; raw qualification is tested separately."""
    base, evidence, output = (tmp_path / name for name in ("base", "evidence", "published"))
    base.mkdir()
    evidence.mkdir()
    tables = {}
    for kind, rows in (
        ("attention", publisher._attention_rows(attention_data(identity), identity)),
        ("communication", publisher._communication_rows(communication_data(identity), identity)),
    ):
        contract = identity["table_contracts"][kind]
        folder = base / contract["relative_directory"]
        folder.mkdir(parents=True)
        (folder / "existing.parquet").write_bytes(b"untouched source bytes")
        runtime = {key: identity["runtime"][key] for key in ("framework", "version", "image_digest")}
        runtime["image"] = "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci"
        provenance.write_collection_meta(
            folder,
            runtime,
            {
                "existing": {
                    "collector_ref": "collector.original",
                    "collector_hash": "sha256:" + "1" * 64,
                    "case_plan_hash": "sha256:" + "2" * 64,
                    "collected_at": "2026-09-20",
                    "rows": 3,
                    "status": "complete",
                }
            },
        )
        tables[kind] = {
            "relative_path": str(Path(contract["relative_directory"]) / contract["canonical_filename"]),
            "rows": rows,
        }
    profile = {"schema_version": 1, "profile_name": publisher.PROFILE, "runtime": identity["runtime"], "tables": tables}
    monkeypatch.setattr(publisher, "qualified_profile", lambda *_: copy.deepcopy(profile))
    return {"base_systems": base, "evidence": evidence, "output_systems": output}, profile


def test_atomic_publication_exact_schema_history_and_idempotence(publication, identity):
    args, profile = publication
    original = publisher._inventory(args["base_systems"])
    result = publisher.publish(**args)
    expected_id = hashlib.sha256(publisher._bytes(profile)).hexdigest()
    assert result == {
        "status": "PUBLISHED",
        "profile_name": publisher.PROFILE,
        "profile_id": expected_id,
        "rows": {"attention": 7, "communication": 8},
    }
    for kind, table in profile["tables"].items():
        path = args["output_systems"] / table["relative_path"]
        actual = pq.read_table(path)
        assert all(not field.nullable for field in actual.schema)
        assert actual.schema.field("latency").type == pa.float64()
        assert actual.to_pylist() == [{**row, "profile_id": expected_id} for row in table["rows"]]
        assert (path.parent / identity["profile_filename"]).read_bytes() == publisher._bytes(profile)
        metadata = yaml.safe_load((path.parent / "collection_meta.yaml").read_text())
        assert metadata["tables"]["existing"]["collections"][0]["collector_ref"] == "collector.original"
        assert metadata["tables"][path.stem]["collections"][0]["collector_ref"] == publisher.MODULE
    assert publisher._inventory(args["base_systems"]) == original
    published = publisher._inventory(args["output_systems"])
    assert publisher.publish(**args)["status"] == "ALREADY_PUBLISHED"
    assert publisher._inventory(args["output_systems"]) == published


@pytest.mark.parametrize(
    ("protected_root", "location"),
    [
        ("evidence", "equal"),
        ("evidence", "nested"),
        ("evidence", "parent_component"),
        ("evidence", "output_symlink"),
        ("evidence", "input_symlink"),
        ("base_systems", "equal"),
        ("base_systems", "nested"),
        ("base_systems", "output_symlink"),
    ],
)
def test_output_cannot_mutate_inputs_before_qualification(publication, tmp_path, monkeypatch, protected_root, location):
    args, _ = publication
    root = args[protected_root]
    (args["evidence"] / "original.json").write_text("unchanged evidence")
    output = root if location == "equal" else root / "published"
    if location == "parent_component":
        output = root / ".." / root.name / "published"
    elif location in ("output_symlink", "input_symlink"):
        alias = tmp_path / "input-alias"
        alias.symlink_to(root, target_is_directory=True)
        if location == "output_symlink":
            output = alias / "published"
        else:
            args[protected_root] = alias
    args["output_systems"] = output

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): (
                ("link", str(path.readlink()))
                if path.is_symlink()
                else ("directory", None)
                if path.is_dir()
                else ("file", path.read_bytes())
            )
            for path in tmp_path.rglob("*")
        }

    def unexpected(*_args, **_kwargs):
        pytest.fail("Unsafe output reached qualification or staging")

    before = snapshot()
    monkeypatch.setattr(publisher, "qualified_profile", unexpected)
    monkeypatch.setattr(publisher.tempfile, "TemporaryDirectory", unexpected)
    with pytest.raises(ValueError, match="Unsafe output location"):
        publisher.publish(**args)
    assert snapshot() == before


@pytest.mark.parametrize("mutation", ["changed_row_and_receipt", "missing_profile", "missing_table", "extra_file"])
def test_repeat_rejects_changed_or_partial_bundle(publication, identity, mutation):
    args, profile = publication
    publisher.publish(**args)
    root = args["output_systems"]
    parquet = root / profile["tables"]["attention"]["relative_path"]
    if mutation == "changed_row_and_receipt":
        table = pq.read_table(parquet)
        rows = table.to_pylist()
        rows[0]["latency"] += 0.5
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), parquet)
        receipt = root / "prefill_graph_publications" / f"{publisher.PROFILE}.json"
        value = json.loads(receipt.read_text())
        value["published_files"][str(parquet.relative_to(root))] = publisher._record(parquet)
        receipt.write_bytes(publisher._bytes(value))
    elif mutation == "missing_profile":
        (parquet.parent / identity["profile_filename"]).unlink()
    elif mutation == "missing_table":
        parquet.unlink()
    else:
        (root / "unexpected").write_text("unowned")
    with pytest.raises(ValueError, match="Existing publication differs"):
        publisher.publish(**args)


def test_partial_write_failure_exposes_no_bundle(publication, monkeypatch):
    args, _ = publication
    original = publisher._write_table
    calls = 0

    def fail_second(*positional):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-family failure")
        return original(*positional)

    before = publisher._inventory(args["base_systems"])
    monkeypatch.setattr(publisher, "_write_table", fail_second)
    with pytest.raises(RuntimeError, match="second-family"):
        publisher.publish(**args)
    assert not args["output_systems"].exists()
    assert publisher._inventory(args["base_systems"]) == before
