# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace

import pytest
from e2e_accuracy_source.defaults.research_defaults import can_assume_recipe, fill_missing, resolve_auto_kv
from e2e_accuracy_source.recipes import runtime_evidence as runtime
from e2e_accuracy_source.schema import SiliconRow


def test_reviewed_runtime_is_bound_to_complete_measurement_and_not_mutable(monkeypatch):
    row = SiliconRow(
        config_id=1,
        isl=128,
        osl=128,
        conc=8,
        hardware="h200",
        framework="vllm",
        silicon_model="minimaxm2.7",
        precision="fp8",
        spec_method="none",
        disagg=False,
        is_multinode=False,
        prefill_tp=0,
        prefill_ep=0,
        prefill_dp_attention=False,
        prefill_num_workers=0,
        decode_tp=4,
        decode_ep=4,
        decode_dp_attention=False,
        decode_num_workers=1,
        num_prefill_gpu=0,
        num_decode_gpu=4,
        bench_id="7",
        workflow_run_id="1",
        date="2026-09-28",
        image=None,
        metrics={"mean_ttft": 100},
    )
    parsed = [{}, {}, {}, {}, {"artifact": {"sha256": "a" * 64}}]
    manifest = {
        "schema_version": "reviewed-runtime-observations/1",
        "records": {runtime.row_digest(row): {"benchmark_id": "7", "parsed": parsed}},
    }
    monkeypatch.setattr(runtime, "load_manifest", lambda _: manifest)
    restored = runtime.archived_recipe(row)
    assert restored[-1]["archived_runtime"]["historical_artifacts_revalidated"] is False
    restored[-1]["artifact"]["sha256"] = "changed"
    assert runtime.archived_recipe(row)[-1]["artifact"]["sha256"] == "a" * 64
    assert runtime.archived_recipe(replace(row, metrics={"mean_ttft": 101})) is None


@pytest.mark.parametrize("name", ["../secret", "/tmp/secret", "..", "a/b"])
def test_runtime_manifest_cannot_choose_download_destination(name):
    with pytest.raises(ValueError, match="filename"):
        runtime.entry_files({"id": 1, "path": name, "sha256": "a" * 64}, False)


def test_cached_evidence_corruption_is_fatal(tmp_path):
    (tmp_path / "1.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime._download(tmp_path, "1.zip", "unused", hashlib.sha256(b"valid").hexdigest())


@pytest.mark.parametrize("status", [403, 404, 410])
def test_expired_or_inaccessible_archives_are_explicit(monkeypatch, tmp_path, status):
    class Response:
        status_code = status

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(runtime.requests, "get", lambda *a, **kw: Response())
    assert runtime._download(tmp_path, "1.zip", "unused", "a" * 64) == "unavailable"
    assert list(tmp_path.iterdir()) == []


def test_assumptions_preserve_explicit_controls_and_reject_conflicts():
    values, assumptions = fill_missing(
        {"prefix": False, "limit": 0, "dtype": "auto"},
        {"prefix": True, "limit": 256, "dtype": "bfloat16", "block": 16},
        role="aggregated",
    )
    assert values == {"prefix": False, "limit": 0, "dtype": "auto", "block": 16}
    assert len(assumptions) == 1 and assumptions[0]["historical_value_verified"] is False
    assert can_assume_recipe("recipe does not exist at revision")
    assert not can_assume_recipe("runtime artifact checksum mismatch")
    assert not can_assume_recipe("conflicting model identity")
    values, assumptions = resolve_auto_kv({"kv_cache_dtype": "fp8_e4m3"}, None, role="aggregated")
    assert values["kv_cache_dtype"] == "fp8_e4m3" and not assumptions
