# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collector/skips.py: the one list of cases a collection leaves out."""

from dataclasses import dataclass

import pytest
import yaml
from collector import case_generator, skips
from collector.model_cases import _backend_registry_ops, build_collection_case_plan
from collector.op_plans import OP_PLANS, plan
from collector.skips import SKIPS, case_dtypes, filter_cases

pytestmark = pytest.mark.unit


@dataclass
class FakeCase:
    """A typed case exposing the attributes skips.py reads."""

    label: str = "case"
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    use_fp8_kv_cache: bool = False
    parallel: str | None = None


def _listed(value) -> list:
    return value if isinstance(value, list) else [value]


def test_fp8_is_skipped_below_sm89_and_bfloat16_runs():
    fp8, bf16 = FakeCase("fp8", dtype="fp8"), FakeCase("bf16", dtype="bfloat16")

    kept, skipped = filter_cases([fp8, bf16], "vllm", "gemm", 80)

    assert kept == [bf16]
    assert [(case.label, region) for case, _entry, region in skipped] == [("fp8", "dtype=fp8")]
    assert filter_cases([fp8, bf16], "vllm", "gemm", 89) == ([fp8, bf16], [])


def test_nvfp4_starts_at_sm100():
    nvfp4 = FakeCase(dtype="nvfp4")

    assert filter_cases([nvfp4], "trtllm", "moe", 100)[0] == [nvfp4]
    assert filter_cases([nvfp4], "trtllm", "moe", 90)[0] == []


def test_an_op_entry_skips_every_case_of_the_op_even_positional_ones():
    cases = [[1, 2, 3], FakeCase(dtype="bfloat16")]

    kept, skipped = filter_cases(cases, "sglang", "dsa_context_module", 89)

    assert kept == []
    assert {region for _case, _entry, region in skipped} == {"op=dsa_context_module"}
    assert filter_cases(cases, "sglang", "dsa_context_module", 90)[0] == cases


def test_unknown_dtypes_and_an_unknown_sm_run():
    future, nvfp4 = FakeCase(dtype="future_dtype"), FakeCase(dtype="nvfp4")

    assert filter_cases([future], "vllm", "gemm", 80)[0] == [future]
    # An entry with an sm list never applies when the SM is unknown (XPU).
    assert filter_cases([nvfp4], "vllm", "dsa_context_module", None)[0] == [nvfp4]


def test_fp8_kv_cache_flag_counts_as_fp8():
    flagged = FakeCase(use_fp8_kv_cache=True)

    assert case_dtypes(flagged) == ["fp8"]
    assert filter_cases([flagged], "trtllm", "attention_context", 80)[0] == []


def test_keys_match_a_value_any_of_a_list_or_a_predicate(monkeypatch):
    monkeypatch.setattr(
        skips,
        "SKIPS",
        [
            {"backend": "vllm", "op": "moe", "parallel": "tp_ep"},
            {"backend": "sglang", "dtype": ["a", "b"], "label": lambda value: value.startswith("big")},
        ],
    )
    tp_ep, ep = FakeCase(parallel="tp_ep"), FakeCase(parallel="ep")

    assert filter_cases([tp_ep, ep], "vllm", "moe", 100)[0] == [ep]
    # Another backend or op is untouched; without sm an entry applies on every SM.
    assert filter_cases([tp_ep], "trtllm", "moe", 100)[0] == [tp_ep]
    assert filter_cases([tp_ep], "vllm", "gemm", 100)[0] == [tp_ep]
    assert filter_cases([tp_ep], "vllm", "moe", None)[0] == []
    # A case without the named field is never skipped by that entry.
    assert filter_cases([[1, 2]], "vllm", "moe", 100)[0] == [[1, 2]]

    big_a, small_a, big_c = FakeCase("big1", dtype="a"), FakeCase("small", dtype="a"), FakeCase("big2", dtype="c")
    kept, skipped = filter_cases([big_a, small_a, big_c], "sglang", "gemm", 90)
    assert kept == [small_a, big_c]
    assert [region for _case, _entry, region in skipped] == ["dtype=a,label=big1"]


def test_model_entries_name_a_whole_model_or_one_op(monkeypatch):
    monkeypatch.setattr(
        skips,
        "SKIPS",
        [{"backend": "trtllm", "model": "A"}, {"backend": "vllm", "op": "mhc_module", "model": ["B", "C"]}],
    )

    assert skips.models("trtllm", "gemm", 100) == {"A"}
    assert skips.models("trtllm", None, None) == {"A"}
    assert skips.models("vllm", "mhc_module", 100) == {"B", "C"}
    assert skips.models("vllm", "gemm", 100) == frozenset()
    assert skips.models("vllm", None, 100) == frozenset()
    # A model entry acts before cases are merged, never on a queued case.
    assert filter_cases([FakeCase()], "trtllm", "gemm", 100)[0] == [FakeCase()]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([{}], "non-empty dict"),
        ([{"dtype": []}], "must not be empty"),
        ([{"dtype": "fp8", "sm": ["89"]}], "SM versions"),
        ([{"backend": "trtllm", "model": "A", "dtype": "fp8"}], "may name only backend, op and sm"),
        ([{"backend": "trtllm", "model": ["A", 1]}], "case-file architectures"),
    ],
)
def test_malformed_entries_are_rejected(entries, message):
    with pytest.raises(ValueError, match=message):
        skips._check(entries)


def test_entries_name_known_backends_ops_and_case_files():
    backends = ("sglang", "trtllm", "vllm")
    ops = set().union(*(_backend_registry_ops(backend) for backend in backends))
    architectures = {
        yaml.safe_load(path.read_text()).get("architecture")
        for path in case_generator.MODEL_CASES_DIR.glob("*_cases.yaml")
    }
    for entry in SKIPS:
        assert set(_listed(entry.get("backend", list(backends)))) <= set(backends), entry
        assert set(_listed(entry.get("op", []))) <= ops, entry
        assert set(_listed(entry.get("model", []))) <= architectures, entry


def test_case_keys_exist_on_the_op_plans_they_name():
    for entry in SKIPS:
        keys = set(entry) - {"backend", "op", "sm", "model", "dtype", "dispatch"}
        for op in _listed(entry.get("op", [])):
            case_type = OP_PLANS[op].case_type if op in OP_PLANS else None
            for key in keys if case_type is not None else ():
                assert key in case_type._fields or isinstance(getattr(case_type, key, None), property), (entry, key)
            if "dispatch" in entry:
                # A dispatch entry names a backend whose plan picks the kernel.
                assert op in OP_PLANS and entry["backend"] in OP_PLANS[op].dispatch, entry


def test_a_model_entry_keeps_a_shape_another_model_needs(monkeypatch):
    def row(path, heads, kv_heads, head_dim):
        return {
            "model_path": path,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "window_size": 0,
            "tensor_parallel_sizes": [1],
        }

    files = (
        {"architecture": "ModelA", "model_case_values": {"attention": [row("org/a", 8, 2, 128)]}},
        {
            "architecture": "ModelB",
            "model_case_values": {"attention": [row("org/b", 8, 2, 128), row("org/b", 16, 2, 64)]},
        },
    )
    monkeypatch.setattr(case_generator, "_load_model_cases_data", lambda: files)
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    def shapes():
        configs = case_generator.get_attention_head_configs({"head_profiles": []}, phase="context")
        return {(config.num_heads, config.num_kv_heads, config.head_dim) for config in configs}

    assert shapes() == {(8, 2, 128), (16, 2, 64)}
    with case_generator.without_models({"ModelA"}):
        assert shapes() == {(8, 2, 128), (16, 2, 64)}
    with case_generator.without_models({"ModelB"}):
        assert shapes() == {(8, 2, 128)}


def test_plan_reports_the_cases_a_model_entry_removes(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    entry = next(entry for entry in SKIPS if entry.get("model") == "MiMoV2FlashForCausalLM")

    trtllm = plan("moe", "trtllm", 100)
    removed = {region.region: region for region in trtllm.removed}

    assert removed["model=MiMoV2FlashForCausalLM"].cases > 0
    assert removed["model=MiMoV2FlashForCausalLM"].reason == repr(entry)
    assert not any("MiMo-V2" in case.model_name for case in trtllm.cases)
    assert any("MiMo-V2" in case.model_name for case in plan("moe", "sglang", 100).cases)


def test_a_model_run_plans_no_op_its_backend_skips(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    assert build_collection_case_plan(backend="trtllm", model_path="XiaomiMiMo/MiMo-V2-Flash", sm_version=100).ops == []
    assert (
        "moe" in build_collection_case_plan(backend="sglang", model_path="XiaomiMiMo/MiMo-V2-Flash", sm_version=100).ops
    )
    # An entry naming one op drops only that op.
    dsv4 = build_collection_case_plan(backend="vllm", model_path="deepseek-ai/DeepSeek-V4-Flash", sm_version=100).ops
    assert "mhc_module" not in dsv4
    assert "moe" in dsv4
