# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""collection_meta.yaml provenance writer tests (Collector V3 design §5)."""

from pathlib import Path

import pytest
import yaml
from collector import provenance

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# collector_hash: deterministic, content-sensitive, rebase-stable
# --------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# One file of every kind collector_hash covers, and one of every kind it leaves out.
HASHED_FILES = (
    "collector/helper.py",
    "collector/vllm/collect_gemm_npu.py",
    "collector/cases/base_ops/gemm.yaml",
    "collector/sglang/dsv4_megamoe/run_torchrun.sh",
    "collector/wideep/vllm/patches/deepep.patch",
    "src/aisimulate_core/sdk/models/deepseek_v41.py",
    "src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V4.1-Flash_config.json",
)
UNHASHED_FILES = (
    "collector/trtllm/moe_tuned_cache_path/sm100_cache.json",
    "collector/README.md",
    "collector/__pycache__/helper.cpython-312.pyc",
    "src/aisimulate_core/systems/data/b200_sxm/gemm/sglang/0.5.21/gemm_perf.parquet",
)


def _build_fake_repo(root: Path) -> None:
    for relative in HASHED_FILES + UNHASHED_FILES:
        _write(root / relative, f"# {relative}\n")


def test_collector_hash_deterministic(tmp_path):
    _build_fake_repo(tmp_path)
    h1 = provenance.collector_hash(tmp_path)
    assert h1 == provenance.collector_hash(tmp_path)
    assert h1.startswith("sha256:")


@pytest.mark.parametrize("relative", HASHED_FILES)
def test_collector_hash_changes_when_a_hashed_file_changes(tmp_path, relative):
    _build_fake_repo(tmp_path)
    before = provenance.collector_hash(tmp_path)
    _write(tmp_path / relative, "# changed\n")
    assert provenance.collector_hash(tmp_path) != before


@pytest.mark.parametrize("relative", UNHASHED_FILES)
def test_collector_hash_ignores_caches_docs_and_data(tmp_path, relative):
    _build_fake_repo(tmp_path)
    before = provenance.collector_hash(tmp_path)
    _write(tmp_path / relative, "# changed\n")
    assert provenance.collector_hash(tmp_path) == before


def test_collector_hash_changes_when_a_file_is_added(tmp_path):
    _build_fake_repo(tmp_path)
    before = provenance.collector_hash(tmp_path)
    _write(tmp_path / "collector" / "vllm" / "utils.py", "# new\n")
    assert provenance.collector_hash(tmp_path) != before


def test_collector_hash_stable_across_repo_relocation(tmp_path_factory):
    # Rebase-stability: content hash must not depend on the absolute path the
    # repo happens to live at.
    root_a = tmp_path_factory.mktemp("repo_a")
    root_b = tmp_path_factory.mktemp("repo_b_somewhere_else")
    _build_fake_repo(root_a)
    _build_fake_repo(root_b)
    assert provenance.collector_hash(root_a) == provenance.collector_hash(root_b)


def test_real_repo_is_hashable():
    assert "collector/vllm/collect_gemm_npu.py" in provenance.hashed_files(REPO_ROOT)
    assert provenance.collector_hash(REPO_ROOT).startswith("sha256:")


# --------------------------------------------------------------------------
# case_plan_hash
# --------------------------------------------------------------------------


def test_case_plan_hash_deterministic():
    assert provenance.case_plan_hash(["a", "b", "c"]) == provenance.case_plan_hash(["a", "b", "c"])


def test_case_plan_hash_order_independent():
    assert provenance.case_plan_hash(["c", "a", "b"]) == provenance.case_plan_hash(["a", "b", "c"])


def test_case_plan_hash_deduplicates():
    assert provenance.case_plan_hash(["a", "a", "b"]) == provenance.case_plan_hash(["a", "b"])


def test_case_plan_hash_changes_with_different_set():
    assert provenance.case_plan_hash(["a", "b"]) != provenance.case_plan_hash(["a", "b", "c"])


def test_case_plan_hash_empty_set_is_stable():
    assert provenance.case_plan_hash([]) == provenance.case_plan_hash([])
    assert provenance.case_plan_hash([]).startswith("sha256:")


# --------------------------------------------------------------------------
# derive_table_status — table-driven
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unresolved_failed_count", "had_module_failure", "expected"),
    [
        (0, False, provenance.STATUS_COMPLETE),
        # Owner decision: recorded per-case failures
        # are DATA (failure_handling.md) and no longer demote a table —
        # only a module-level collection failure does.
        (1, False, provenance.STATUS_COMPLETE),
        (0, True, provenance.STATUS_PARTIAL),
        (3, True, provenance.STATUS_PARTIAL),
    ],
    ids=["all-passed", "recorded-failures-are-data", "module-collection-failure", "both"],
)
def test_derive_table_status(unresolved_failed_count, had_module_failure, expected):
    assert (
        provenance.derive_table_status(
            unresolved_failed_count=unresolved_failed_count,
            had_module_failure=had_module_failure,
        )
        == expected
    )


# --------------------------------------------------------------------------
# write_collection_meta: design §5 schema rendering
# --------------------------------------------------------------------------


RUNTIME_META = {
    "framework": "sglang",
    "version": "0.5.14",
    "image": "lmsysorg/sglang:v0.5.14",
    "image_variant": "linux/amd64",
    "image_digest": "sha256:" + "0" * 64,
}

TABLES = {
    "moe_perf": {
        "collector_ref": "0b077da5",
        "collector_hash": "sha256:" + "1" * 64,
        "case_plan_hash": "sha256:" + "2" * 64,
        "collected_at": "2026-07-20",
        "rows": 12345,
        "status": "complete",
    },
    "gemm_perf": {
        "collector_ref": "0b077da5",
        "collector_hash": "sha256:" + "3" * 64,
        "case_plan_hash": "sha256:" + "4" * 64,
        "collected_at": "2026-07-20",
        "rows": 42,
        "status": "partial",
    },
}


def test_write_collection_meta_returns_flat_sidecar_path(tmp_path):
    meta_path = provenance.write_collection_meta(tmp_path, RUNTIME_META, TABLES)
    assert meta_path == tmp_path / "collection_meta.yaml"
    assert meta_path.exists()


def test_write_collection_meta_schema_matches_design_5(tmp_path):
    meta_path = provenance.write_collection_meta(tmp_path, RUNTIME_META, TABLES)
    doc = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    assert doc["schema_version"] == 1
    assert doc["runtime"] == {
        "framework": "sglang",
        "version": "0.5.14",
        "image": "lmsysorg/sglang:v0.5.14",
        "image_variant": "linux/amd64",
        "image_digest": "sha256:" + "0" * 64,
    }
    assert set(doc["tables"]) == {"moe_perf", "gemm_perf"}
    for table_name, expected in TABLES.items():
        assert doc["tables"][table_name] == {
            "collector_ref": expected["collector_ref"],
            "collector_hash": expected["collector_hash"],
            "case_plan_hash": expected["case_plan_hash"],
            "collected_at": expected["collected_at"],
            "rows": expected["rows"],
            "status": expected["status"],
        }


def test_write_collection_meta_preserves_multi_event_history(tmp_path):
    event_a = TABLES["gemm_perf"]
    event_b = {
        **event_a,
        "case_plan_hash": "sha256:" + "5" * 64,
        "collected_at": "2026-08-26",
        "rows": 3,
        "classified_failures": 1,
        "status": "complete",
    }
    tables = {
        "gemm_perf": {
            "rows": 45,
            "status": "complete",
            "collections": [event_a, event_b],
        },
        "moe_perf": TABLES["moe_perf"],
    }

    runtime_meta = {
        **RUNTIME_META,
        "source_commit": "1" * 40,
        "abi": {"deep_ep": "d4f41e4e93", "nvshmem": "3.3.24"},
    }
    meta_path = provenance.write_collection_meta(tmp_path, runtime_meta, tables)
    doc = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    assert doc["schema_version"] == 2
    assert doc["runtime"] == runtime_meta
    assert doc["tables"]["gemm_perf"] == tables["gemm_perf"]
    assert doc["tables"]["moe_perf"] == {
        "rows": TABLES["moe_perf"]["rows"],
        "status": TABLES["moe_perf"]["status"],
        "collections": [TABLES["moe_perf"]],
    }


def test_write_collection_meta_omits_absent_optional_runtime_fields(tmp_path):
    runtime_meta = {"framework": "wideep_sglang", "version": "0.5.10", "image": "deepseek-v4-blackwell"}
    meta_path = provenance.write_collection_meta(tmp_path, runtime_meta, TABLES)
    doc = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert "image_digest" not in doc["runtime"]


def test_write_collection_meta_preserves_pinned_source_and_abi(tmp_path):
    runtime_meta = {
        **RUNTIME_META,
        "source_commit": "1" * 40,
        "abi": {
            "deep_ep": "d4f41e4e93",
            "nvshmem": "3.3.24",
            "nccl": ">=2.30.4",
        },
    }
    meta_path = provenance.write_collection_meta(tmp_path, runtime_meta, TABLES)
    runtime = yaml.safe_load(meta_path.read_text(encoding="utf-8"))["runtime"]
    assert runtime["source_commit"] == "1" * 40
    assert runtime["abi"] == runtime_meta["abi"]


def test_write_collection_meta_preserves_backend_runtime_evidence(tmp_path):
    runtime_meta = {
        **RUNTIME_META,
        "live_abi": {"deep_ep_api": "ElasticBuffer", "nccl": "2.30.4"},
        "transport": {"allow_mnnvl": True, "allow_nvlink": True},
        "backend_capability": {
            "backend": "deepep_v2",
            "topology_source": "nccl_lsa",
            "num_scaleout_ranks": "2",
            "num_scaleup_ranks": "4",
        },
    }
    meta_path = provenance.write_collection_meta(tmp_path, runtime_meta, TABLES)
    runtime = yaml.safe_load(meta_path.read_text(encoding="utf-8"))["runtime"]

    assert runtime["live_abi"] == runtime_meta["live_abi"]
    assert runtime["transport"] == runtime_meta["transport"]
    assert runtime["backend_capability"] == runtime_meta["backend_capability"]


def test_write_collection_meta_deterministic_key_order(tmp_path):
    meta_path = provenance.write_collection_meta(tmp_path, RUNTIME_META, TABLES)
    text = meta_path.read_text(encoding="utf-8")
    assert text.startswith("# SPDX-FileCopyrightText:")
    lines = [line for line in text.splitlines() if line and not line.startswith((" ", "#"))]
    assert lines == ["schema_version: 1", "runtime:", "tables:"]

    all_lines = text.splitlines()
    runtime_start = all_lines.index("runtime:") + 1
    runtime_lines = all_lines[runtime_start : runtime_start + 5]
    assert [line.split(":")[0].strip() for line in runtime_lines] == [
        "framework",
        "version",
        "image",
        "image_variant",
        "image_digest",
    ]

    # tables render in sorted (not insertion) order.
    tables_section = text.split("tables:", 1)[1]
    table_names_in_order = [
        line.strip().rstrip(":")
        for line in tables_section.splitlines()
        if line.startswith("  ") and not line.startswith("   ")
    ]
    assert table_names_in_order == sorted(TABLES)

    # field order within a table entry follows the design's declared order,
    # regardless of the insertion order of the input dict.
    shuffled_table = {
        "status": "complete",
        "rows": 1,
        "collected_at": "2026-07-20",
        "case_plan_hash": "sha256:" + "9" * 64,
        "collector_hash": "sha256:" + "8" * 64,
        "collector_ref": "deadbeef",
    }
    meta_path2 = provenance.write_collection_meta(tmp_path / "other", RUNTIME_META, {"only_perf": shuffled_table})
    text2 = meta_path2.read_text(encoding="utf-8")
    field_lines = text2.split("only_perf:", 1)[1].splitlines()[1:7]
    assert [line.split(":")[0].strip() for line in field_lines] == [
        "collector_ref",
        "collector_hash",
        "case_plan_hash",
        "collected_at",
        "rows",
        "status",
    ]


def test_write_collection_meta_creates_out_dir(tmp_path):
    out_dir = tmp_path / "nested" / "not_yet_created"
    meta_path = provenance.write_collection_meta(out_dir, RUNTIME_META, TABLES)
    assert meta_path.exists()
    assert meta_path.parent == out_dir
