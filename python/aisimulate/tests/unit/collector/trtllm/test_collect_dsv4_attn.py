# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
from collector.model_cases import build_collection_case_plan
from collector.registry_types import PerfFile
from collector.trtllm.registry import REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]

DSV4_MODULE_OPS = {
    "dsv4_csa_context_module": (PerfFile.DSV4_CSA_CONTEXT_MODULE, "get_dsv4_csa_context_test_cases"),
    "dsv4_hca_context_module": (PerfFile.DSV4_HCA_CONTEXT_MODULE, "get_dsv4_hca_context_test_cases"),
    "dsv4_csa_generation_module": (PerfFile.DSV4_CSA_GENERATION_MODULE, "get_dsv4_csa_generation_test_cases"),
    "dsv4_hca_generation_module": (PerfFile.DSV4_HCA_GENERATION_MODULE, "get_dsv4_hca_generation_test_cases"),
}


def _load_module_with_torch_stub(monkeypatch):
    # exec_module runs the collector's import fallbacks, which append to
    # sys.path; snapshot it so the mutation does not leak into later tests.
    monkeypatch.setattr(sys, "path", list(sys.path))
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    module_path = REPO_ROOT / "collector" / "trtllm" / "collect_dsv4_attn.py"
    spec = importlib.util.spec_from_file_location("trtllm_dsv4_target", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # no GPU under unit tests: the population / serving-KV-dtype paths read the SM
    monkeypatch.setattr(module, "get_sm_version", lambda: 90, raising=False)
    return module


def test_registry_wires_all_four_dsv4_module_ops():
    entries = {entry.op: entry for entry in REGISTRY}
    for op, (perf_file, get_func) in DSV4_MODULE_OPS.items():
        entry = entries[op]
        assert entry.module == "collector.trtllm.collect_dsv4_attn"
        assert entry.run_func == "run_dsv4_attn_series_worker"
        # get_func <-> perf_filename pairing decides which population lands
        # in which table; a CSA getter wired to the HCA file would otherwise
        # pass and mislabel every row.
        assert entry.get_func == get_func
        assert entry.perf_filename == perf_file
        # Pre-Blackwell rejection comes from the framework at runtime
        # (classified failure). SM120 is parked with hardware probe evidence
        # (RTX PRO 6000 campaign 100% classified failures).
        assert entry.unverified is False
        assert entry.unverified_sms == (120,)


def test_trtllm_dsv4_plan_schedules_attention_modules():
    plan = build_collection_case_plan(backend="trtllm", model_path="sgl-project/DeepSeek-V4-Pro-FP8")
    assert set(DSV4_MODULE_OPS) <= plan.selected_ops
    assert "mhc_module" in plan.selected_ops


def test_dsv4_rows_are_the_plan_series(monkeypatch):
    from collector.op_plans import plan

    module = _load_module_with_torch_stub(monkeypatch)
    for mode, expected_kind, getter in (
        ("context", "csa", module.get_dsv4_csa_context_test_cases),
        ("context", "hca", module.get_dsv4_hca_context_test_cases),
        ("generation", "csa", module.get_dsv4_csa_generation_test_cases),
        ("generation", "hca", module.get_dsv4_hca_generation_test_cases),
    ):
        rows = getter()
        assert rows == plan(f"dsv4_{expected_kind}_{mode}_module", "trtllm", 90).rows
        keys = [tuple(row[1:]) for row in rows]
        assert len(keys) == len(set(keys)), f"two series share a module build in {mode}/{expected_kind}"
        for points, bs, tp, kv, comp, gemm, _model, kind in rows:
            # TRT-LLM times the FP8 checkpoints' own projections on an FP8 KV cache.
            assert (kv, comp, gemm) == ("fp8", "bfloat16", "fp8_block")
            assert kind == expected_kind
            assert tp in (1, 2, 4, 8)
            assert points == sorted(points, reverse=True)
            for sl, prefix in points:
                if mode == "context":
                    assert bs * sl <= 16384
                    assert bs * (prefix + sl) <= 1048576
                else:
                    assert prefix == 0
                    assert bs * sl <= 1048576


def test_dsv4_getters_return_empty_for_unrelated_model_filter(monkeypatch, capsys):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-32B")
    module = _load_module_with_torch_stub(monkeypatch)

    for getter in (
        module.get_dsv4_csa_context_test_cases,
        module.get_dsv4_hca_context_test_cases,
        module.get_dsv4_csa_generation_test_cases,
        module.get_dsv4_hca_generation_test_cases,
    ):
        assert getter() == []

    assert "is not a DSV4 model; generating no DSV4 cases" in capsys.readouterr().out


def test_dsv4_worker_infers_mode_from_perf_filename(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_dsv4_attn", fake_run)
    module.run_dsv4_attn_worker(
        64,
        1,
        2,
        "fp8",
        "bfloat16",
        "fp8_block",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "csa",
        128,
        perf_filename="/out/dsv4_csa_context_module_perf.txt",
    )
    assert captured["mode"] == "context"
    assert captured["prefix_len"] == 128
    assert captured["tp_size"] == 2

    module.run_dsv4_attn_worker(
        1024,
        4,
        8,
        "fp8",
        "bfloat16",
        "fp8_block",
        "sgl-project/DeepSeek-V4-Pro-FP8",
        "hca",
        perf_filename="/out/dsv4_hca_generation_module_perf.txt",
    )
    assert captured["mode"] == "generation"
    assert captured["attn_kind"] == "hca"


def test_module_cache_reuses_same_geometry_and_evicts_on_change(monkeypatch):
    """Size-1 cache: consecutive same-(model, kind, tp) cases share one build;
    a geometry change evicts and rebuilds (bounded memory)."""
    module = _load_module_with_torch_stub(monkeypatch)

    builds = []

    def fake_build(*, model_path, attn_kind, tp_size, device, kv_cache_dtype="fp8"):
        builds.append((model_path, attn_kind, tp_size))
        return (object(), object(), {"local_heads": 8, "native_heads": 64})

    monkeypatch.setattr(module, "create_dsv4_attention_module", fake_build)
    module._MODULE_CACHE.clear()

    a1 = module._cached_dsv4_attention_module("m/flash", "csa", 4, "cuda:0")
    a2 = module._cached_dsv4_attention_module("m/flash", "csa", 4, "cuda:0")
    assert a1 is a2
    assert len(builds) == 1

    b1 = module._cached_dsv4_attention_module("m/flash", "hca", 4, "cuda:0")
    assert len(builds) == 2
    assert list(module._MODULE_CACHE) == [("m/flash", "hca", 4, "cuda:0", "fp8")]  # key carries the KV dtype (189589d2)
    assert b1 is not a1
    module._MODULE_CACHE.clear()


def test_generation_geometry_matches_serving_invariant(monkeypatch):
    """position == past-seen/cached == persisted step (TensorRT-LLM
    pyexecutor/engine/runners/decoder/inputs.py:1616, 1625, 1663-1665): the
    decode dummy request registers seq_len+1 beam
    tokens, so all three sides of the triple are seq_len. Guards the
    off-by-one found in review (position was seq_len - 1)."""
    module = _load_module_with_torch_stub(monkeypatch)
    for seq_len in (1, 4, 512, 65536):
        geo = module.generation_request_geometry(seq_len)
        assert geo["position"] == geo["num_cached_tokens"] == geo["persisted_step"] == seq_len


# ---------------------------------------------------------------------------
# Constructor-level serving-parity capture:
# stub the tensorrt_llm surface and assert the EXACT arguments handed to the
# cache-manager constructor, add_dummy_requests and the attention Metadata
# constructor for a context and a generation case.
# ---------------------------------------------------------------------------


class _CapturingManager:
    instances: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.ctor_args = args
        self.ctor_kwargs = kwargs
        self.dummy_calls: list = []
        _CapturingManager.instances.append(self)

    def add_dummy_requests(self, request_ids, token_nums, is_gen):
        self.dummy_calls.append({"request_ids": request_ids, "token_nums": token_nums, "is_gen": is_gen})
        # KVCacheManagerV2.add_dummy_requests returns the requests, or None when the pool
        # cannot hold them.
        return [object() for _ in request_ids]

    def shutdown(self):
        pass


class _CapturingMetadata:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prepared = False
        _CapturingMetadata.instances.append(self)

    def prepare(self):
        self.prepared = True


def _install_trtllm_stubs(monkeypatch, module):
    captured = {"kv_cache_config": []}

    def _mod(name):
        m = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    tl = _mod("tensorrt_llm")
    _mod("tensorrt_llm._torch")
    _mod("tensorrt_llm._torch.attention")
    _mod("tensorrt_llm._torch.attention.backends")
    iface = _mod("tensorrt_llm._torch.attention.backends.interface")
    utils_mod = _mod("tensorrt_llm._torch.attention.backends.utils")
    bindings = _mod("tensorrt_llm.bindings")
    _mod("tensorrt_llm.bindings.internal")
    bm = _mod("tensorrt_llm.bindings.internal.batch_manager")
    llm_args = _mod("tensorrt_llm.llmapi.llm_args")
    _mod("tensorrt_llm.llmapi")
    tl.__version__ = "1.4.0rc2"

    iface.AttentionRuntimeFeatures = lambda **kw: SimpleNamespace(**kw)
    iface.KVCacheParams = lambda **kw: SimpleNamespace(**kw)
    utils_mod.get_attention_backend = lambda backend, sparse_cfg: SimpleNamespace(
        Metadata=_CapturingMetadata, __name__="FakeDSV4Backend"
    )
    bindings.DataType = SimpleNamespace(FP8="FP8")
    bm.CacheType = SimpleNamespace(SELFKONLY="SELFKONLY")

    def _kv_cache_config(**kw):
        captured["kv_cache_config"].append(kw)
        return SimpleNamespace(**kw)

    llm_args.KvCacheConfig = _kv_cache_config
    # no GPU under unit tests: pin the Hopper branch (fp8_ds_mla pool) deterministically
    monkeypatch.setattr(module, "get_sm_version", lambda: 90)
    # the create function re-imports this locally (relative-or-plain fallback)
    mla_stub = _mod("collect_mla_module")
    mla_stub.get_kv_cache_manager_cls = lambda mc, cfg: _CapturingManager

    # torch stub surface used by the create function
    torch_stub = sys.modules["torch"]
    torch_stub.int32 = "int32"
    torch_stub.int8 = "int8"
    torch_stub.cuda = SimpleNamespace(mem_get_info=lambda dev: (100 * 2**30, 128 * 2**30), empty_cache=lambda: None)
    torch_stub.tensor = lambda data, **kw: SimpleNamespace(data=data, **kw)
    torch_stub.device = lambda d: d
    return captured


def _fake_model_config():
    pretrained = SimpleNamespace(kv_lora_rank=448, qk_rope_head_dim=64, vocab_size=129280, hidden_size=7168)
    sparse = SimpleNamespace(to_sparse_metadata_params=lambda pretrained_config: {"sparse": True})
    return SimpleNamespace(
        pretrained_config=pretrained,
        sparse_attention_config=sparse,
        attn_backend="TRTLLM",
        mapping="MAPPING",
        extra_attrs={},
    )


def _run_create(module, monkeypatch, **kw):
    _CapturingManager.instances.clear()
    _CapturingMetadata.instances.clear()
    captured = _install_trtllm_stubs(monkeypatch, module)
    manager, metadata, attention_cls = module.create_dsv4_kv_cache_and_metadata(
        model_config=_fake_model_config(),
        attn_module=SimpleNamespace(indexer=None),
        device="cuda:0",
        **kw,
    )
    return manager, metadata, captured


def test_context_metadata_constructor_args_match_serving(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)
    bs, sl, prefix = 2, 64, 128
    manager, metadata, captured = _run_create(
        module, monkeypatch, batch_size=bs, seq_len=sl, is_context=True, prefix_len=prefix
    )
    # manager construction (pyexecutor/_util.py:3575-3609)
    mk = manager.ctor_kwargs
    assert mk["num_kv_heads"] == 1 and mk["num_layers"] == 1
    assert mk["head_dim"] == 448 + 64
    assert mk["tokens_per_block"] == 128
    assert mk["max_seq_len"] == 512  # engine-envelope floor (prefix+sl+1=193 -> 512)
    assert mk["vocab_size"] == 129280
    # dummy requests register the REAL request size, not the floored envelope
    dc = manager.dummy_calls[0]
    assert dc["token_nums"] == [prefix + sl] * bs and dc["is_gen"] is False
    # Metadata population (engine/metadata.py:48-90, engine/runners/decoder/inputs.py:985-1014)
    kw = metadata.kwargs
    assert kw["seq_lens"].data == [sl] * bs  # chunk-local fresh tokens
    assert kw["num_contexts"] == bs
    assert kw["kv_cache_params"].num_cached_tokens_per_seq == [prefix] * bs  # begin_compute prefix
    assert kw["prompt_lens"] == [sl] * bs  # chunk-local (SM100 cached-KV walker)
    assert kw["enable_context_mla_with_cached_kv"] is True
    assert kw["runtime_features"].cache_reuse is True
    assert kw["request_ids"] == list(range(bs))
    assert metadata.prepared


def test_generation_metadata_constructor_args_match_serving(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)
    bs, past_kv = 4, 512
    manager, metadata, captured = _run_create(
        module, monkeypatch, batch_size=bs, seq_len=past_kv, is_context=False, prefix_len=0
    )
    geo = module.generation_request_geometry(past_kv)
    dc = manager.dummy_calls[0]
    # decode dummy request registers past_kv + 1 beam tokens -> serving's
    # past_seen = max_beam_num_tokens - 1 == past_kv (inputs.py:1616)
    assert dc["token_nums"] == [past_kv + 1] * bs and dc["is_gen"] is True
    kw = metadata.kwargs
    assert kw["seq_lens"].data == [1] * bs  # one new token per decode step
    assert kw["num_contexts"] == 0
    # cached == past_seen == position triple (inputs.py:1625, 1663-1665)
    assert kw["kv_cache_params"].num_cached_tokens_per_seq == [geo["num_cached_tokens"]] * bs
    assert kw["prompt_lens"] == [past_kv] * bs
    assert kw["enable_context_mla_with_cached_kv"] is False
    assert kw["runtime_features"].cache_reuse is False
    assert metadata.prepared


def test_budget_file_moves_both_hashes(tmp_path, monkeypatch):
    """An edit to the base-op budget declaration changes (a) the collector_hash and (b) the
    generated case plan itself: a cap change re-shapes the population, so the checkpoint-derived
    case_plan_hash re-attests."""
    import shutil

    from collector import provenance

    # (a) content change -> collector_hash change (copy the module and the budget
    # file, mutate only the budget file)
    for rel in ("collector/trtllm/collect_dsv4_attn.py", "collector/cases/base_ops/dsv4_attention.yaml"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, dst)
    h_before = provenance.collector_hash(tmp_path)
    budget = tmp_path / "collector/cases/base_ops/dsv4_attention.yaml"
    budget.write_text(budget.read_text().replace("max_generation_kv_tokens: 1048576", "max_generation_kv_tokens: 1"))
    h_after = provenance.collector_hash(tmp_path)
    assert h_before != h_after

    # (b) budget change -> different generated population (plan re-attests)
    import dataclasses

    from collector import op_plans

    module = _load_module_with_torch_stub(monkeypatch)
    baseline = module.get_dsv4_csa_generation_test_cases()
    sweep = op_plans.get_dsv4_sweep()
    monkeypatch.setattr(op_plans, "get_dsv4_sweep", lambda: dataclasses.replace(sweep, generation_max_kv_tokens=65536))
    shrunk = module.get_dsv4_csa_generation_test_cases()
    assert sum(len(row[0]) for row in shrunk) < sum(len(row[0]) for row in baseline)


def _serving_oracle(bs, seq_len, prefix, is_context):
    """Independent re-derivation of the serving population (TensorRT-LLM
    1.4.0rc2) for one uniform dummy batch, written ONLY from the pinned serving
    sources (never from the collector's own code), so a collector drift FAILS
    against it. Paths are under tensorrt_llm/_torch/pyexecutor/:

    - context: chunk slicing/positions engine/runners/decoder/inputs.py:985-1001;
      past-seen prefix accounting :1012-1014; per-batch assignment :2031-2095.
    - decode: the dummy request registers token_num beam tokens and
      kv_cache/kv_cache_manager_v2.py:6052-6055 sets req.prompt_len = token_num - 1;
      past_seen = max_beam_num_tokens - 1 (inputs.py:1616); position = past_seen
      (:1625); cached = past_seen - compressed(=0) (:1663-1665); seq_lens
      appends 1 per decode request.
    """
    if is_context:
        return {
            "seq_lens": [seq_len] * bs,
            "num_cached": [prefix] * bs,
            "prompt_lens": [seq_len] * bs,
            "num_contexts": bs,
            "first_position": prefix,
            "cache_reuse": prefix > 0,
        }
    token_num = seq_len + 1  # what the collector registers per decode request
    past_seen = token_num - 1  # inputs.py:1616
    return {
        "seq_lens": [1] * bs,
        "num_cached": [past_seen] * bs,
        "prompt_lens": [token_num - 1] * bs,  # kv_cache_manager_v2.py:6052-6055
        "num_contexts": 0,
        "first_position": past_seen,  # inputs.py:1625
        "cache_reuse": False,
    }


def test_constructor_args_match_the_serving_oracle(monkeypatch):
    """S4: compare captured constructor arguments against
    the upstream-derived oracle for one context-with-prefix case and one
    ordinary decode case — the test fails when the collector diverges from
    the pinned serving semantics, not merely when its own values change."""
    module = _load_module_with_torch_stub(monkeypatch)
    for bs, sl, prefix, is_ctx in ((2, 64, 128, True), (4, 512, 0, False)):
        manager, metadata, _ = _run_create(
            module, monkeypatch, batch_size=bs, seq_len=sl, is_context=is_ctx, prefix_len=prefix
        )
        oracle = _serving_oracle(bs, sl, prefix, is_ctx)
        kw = metadata.kwargs
        assert kw["seq_lens"].data == oracle["seq_lens"]
        assert kw["kv_cache_params"].num_cached_tokens_per_seq == oracle["num_cached"]
        assert kw["prompt_lens"] == oracle["prompt_lens"]
        assert kw["num_contexts"] == oracle["num_contexts"]
        assert kw["runtime_features"].cache_reuse is oracle["cache_reuse"]
        # capacity-bound contract: max_num_tokens must bound this batch
        assert kw["max_num_tokens"] >= sum(oracle["seq_lens"])
        if not is_ctx:
            # position == past_seen (the runner builds the tensor from the
            # single-sourced geometry helper)
            assert module.generation_request_geometry(sl)["position"] == oracle["first_position"]


@pytest.mark.parametrize(
    "bs, sl, is_ctx, prefix",
    [(1, 1, True, 0), (16, 10240, True, 4096), (128, 65536, False, 0)],
)
def test_kv_quota_is_capped_by_this_batch_s_tokens(monkeypatch, bs, sl, is_ctx, prefix):
    """Per-case KV pool: the byte quota stays the serving half-of-free bound,
    AND a max_tokens cap sized to this batch (doubled, page-rounded) is handed
    to the manager's own DSV4 arithmetic — never collector-computed bytes."""
    module = _load_module_with_torch_stub(monkeypatch)
    _, _, captured = _run_create(module, monkeypatch, batch_size=bs, seq_len=sl, is_context=is_ctx, prefix_len=prefix)
    (kw,) = captured["kv_cache_config"]
    assert kw["max_gpu_total_bytes"] == 50 * 2**30  # mem_get_info stub: free=100 GiB
    tpb = kw["tokens_per_block"]
    envelope = (prefix + sl + 1) if is_ctx else (sl + 1)
    envelope = max(envelope, 512)
    needed = bs * envelope
    assert kw["max_tokens"] % tpb == 0
    floor = module._KV_POOL_MIN_TOKENS
    expected = max(2 * needed, floor)
    assert expected <= kw["max_tokens"] <= max(2 * (needed + bs * tpb) + tpb, floor)
    # the floor keeps the footer-scale pool past inductor's int64 threshold
    assert floor > 1_572_864  # H20 calibration: 1,048,576 tokens -> i32, 1,572,864 -> i64


@pytest.mark.parametrize("sm", [90, 100, 103])
def test_rows_label_the_kv_cache_with_an_sdk_enum_member_name(monkeypatch, sm):
    # engine_table_view rehydrates kv_cache_dtype through KVCacheQuantMode; "bf16" raised
    # KeyError on every SM100 DSV4 table in Full CI.
    from aisimulate_core.quantization import KVCacheQuantMode

    module = _load_module_with_torch_stub(monkeypatch)
    monkeypatch.setattr(module, "get_sm_version", lambda: sm)
    labels = {row[3] for row in module.get_dsv4_hca_generation_test_cases()}
    assert labels == {"fp8"}
    assert labels <= set(KVCacheQuantMode.__members__)


@pytest.mark.parametrize("sm, dtype", [(90, "fp8_ds_mla"), (100, None)])
def test_hopper_pool_is_the_packed_fp8_sparse_mla_cache(monkeypatch, sm, dtype):
    """Serving's DSV4 default on Hopper is kv_cache_config.dtype='fp8_ds_mla', the only pool
    DeepseekV4CacheManager accepts there; other SMs keep the checkpoint's KV dtype."""
    module = _load_module_with_torch_stub(monkeypatch)
    _CapturingManager.instances.clear()
    _CapturingMetadata.instances.clear()
    captured = _install_trtllm_stubs(monkeypatch, module)
    monkeypatch.setattr(module, "get_sm_version", lambda: sm)
    module.create_dsv4_kv_cache_and_metadata(
        model_config=_fake_model_config(),
        attn_module=SimpleNamespace(indexer=None),
        batch_size=2,
        seq_len=64,
        is_context=True,
        device="cuda:0",
    )
    (kw,) = captured["kv_cache_config"]
    assert kw.get("dtype") == dtype


def test_a_pool_that_cannot_hold_the_requests_is_a_classified_capacity_refusal(monkeypatch):
    """KVCacheManagerV2.add_dummy_requests returns None when the pool cannot hold the
    requests; the case fails as a classified capacity refusal, never with unset block tables."""
    module = _load_module_with_torch_stub(monkeypatch)
    _CapturingManager.instances.clear()
    _CapturingMetadata.instances.clear()
    _install_trtllm_stubs(monkeypatch, module)
    shutdowns = []

    class _FullManager(_CapturingManager):
        def add_dummy_requests(self, request_ids, token_nums, is_gen):
            return None

        def shutdown(self):
            shutdowns.append(self)

    sys.modules["collect_mla_module"].get_kv_cache_manager_cls = lambda mc, cfg: _FullManager
    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): does not fit"):
        module.create_dsv4_kv_cache_and_metadata(
            model_config=_fake_model_config(),
            attn_module=SimpleNamespace(indexer=None),
            batch_size=2,
            seq_len=64,
            is_context=True,
            device="cuda:0",
        )
    assert len(shutdowns) == 1, "the pool is released before the refusal propagates"
    assert _CapturingMetadata.instances == []


class _Module:
    """A weak-referenceable stand-in for a built attention module or its metadata."""


def _install_autotune_stub(monkeypatch, module, events):
    tuning = [False]

    @contextlib.contextmanager
    def autotune():
        tuning.append(True)
        yield
        tuning.append(False)

    for name in ("tensorrt_llm", "tensorrt_llm._torch"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["tensorrt_llm"].__version__ = "1.4.0rc2"
    autotuner = ModuleType("tensorrt_llm._torch.autotuner")
    autotuner.autotune = autotune
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.autotuner", autotuner)
    utils_mod = ModuleType("tensorrt_llm._torch.utils")
    utils_mod._model_extra_attrs = SimpleNamespace(attrs=None)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.utils", utils_mod)
    mla_stub = ModuleType("collect_mla_module")
    mla_stub._cleanup = lambda manager: events.append(("cleanup", manager))
    monkeypatch.setitem(sys.modules, "collect_mla_module", mla_stub)
    monkeypatch.setattr(
        module.torch,
        "cuda",
        SimpleNamespace(
            synchronize=lambda: None,
            empty_cache=lambda: None,
            set_device=lambda device: None,
            get_device_name=lambda device: "GPU",
        ),
    )
    monkeypatch.setattr(module.torch, "device", lambda device: device, raising=False)
    monkeypatch.setattr(
        module,
        "_dsv4_inputs",
        lambda model_config, batch_size, seq_len, prefix_len, is_context, device: (
            batch_size * seq_len if is_context else batch_size,
            "position_ids",
            "hidden_states",
        ),
    )
    monkeypatch.setattr(
        module,
        "_forward_dsv4",
        lambda attn_module, model_config, metadata, position_ids, hidden_states: events.append(
            ("forward", getattr(metadata, "tokens", None), tuning[-1])
        ),
    )


def test_built_module_is_tuned_on_serving_warmup_passes(monkeypatch):
    """Serving forwards the model under autotune() on a context batch of max_num_tokens
    (8192) tokens and one one-token generation request; the module has tunable ops beyond
    its projections (CuTe DSL FP8 BMM, fused indexer Q GEMM), so the module itself is
    forwarded, each pass on its own KV pool, which is released afterwards."""
    module = _load_module_with_torch_stub(monkeypatch)
    events = []
    _install_autotune_stub(monkeypatch, module, events)

    def fake_create(*, model_config, attn_module, batch_size, seq_len, is_context, device, kv_cache_dtype_str):
        events.append(("create", is_context, batch_size, seq_len, kv_cache_dtype_str))
        metadata = _Module()
        metadata.tokens = seq_len
        return ("manager", seq_len), metadata, "Backend"

    monkeypatch.setattr(module, "create_dsv4_kv_cache_and_metadata", fake_create)
    attn = _Module()
    module._autotune_dsv4_module(attn, "model_config", module._SERVING_AUTOTUNE_PASSES, "cuda:0", "fp8")
    assert events == [
        ("create", True, 1, 8192, "fp8"),
        ("forward", 8192, True),
        ("cleanup", ("manager", 8192)),
        ("create", False, 1, 1, "fp8"),
        ("forward", 1, True),
        ("cleanup", ("manager", 1)),
    ]
    assert module._TUNED_NUM_TOKENS[attn] == 8192

    tree = ast.parse((REPO_ROOT / "collector" / "trtllm" / "collect_dsv4_attn.py").read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_dsv4_attention_module")
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_autotune_dsv4_module" for n in ast.walk(build)
    ), "every built module is tuned before it is cached"


def test_a_point_past_the_tuned_count_is_tuned_at_its_own_count(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)
    events = []
    _install_autotune_stub(monkeypatch, module, events)
    attn = _Module()
    metadata = _Module()
    model_config = SimpleNamespace(extra_attrs={})
    monkeypatch.setattr(module, "_cached_dsv4_attention_module", lambda *args: (attn, model_config, {"local_heads": 8}))
    monkeypatch.setattr(
        module,
        "create_dsv4_kv_cache_and_metadata",
        lambda **kwargs: ("manager", metadata, SimpleNamespace(__name__="Backend")),
    )

    @contextlib.contextmanager
    def fake_benchmark(**kwargs):
        yield {"latency_ms": 1.0, "power_stats": None}

    monkeypatch.setattr(module, "benchmark_with_power", fake_benchmark)
    monkeypatch.setattr(module, "log_perf", lambda **kwargs: None)
    module._TUNED_NUM_TOKENS[attn] = 8192

    def run(**kwargs):
        module.run_dsv4_attn(
            tp_size=8,
            kv_cache_dtype="fp8",
            compute_dtype="bfloat16",
            gemm_type="fp8_block",
            model_path="m",
            attn_kind="csa",
            perf_filename="p",
            **kwargs,
        )

    run(seq_len=8192, batch_size=2, mode="context", dry_run=False)
    forwards = [event[2] for event in events if event[0] == "forward"]
    assert forwards == [True], "16384 new tokens > 8192 tuned: the point is forwarded under autotune()"
    assert module._TUNED_NUM_TOKENS[attn] == 16384
    run(seq_len=4096, batch_size=2, mode="context", dry_run=True)
    run(seq_len=65536, batch_size=64, mode="generation", dry_run=False)
    forwards = [event[2] for event in events if event[0] == "forward"]
    assert forwards == [True, False], "a tuned count runs only the plain dry run"
    assert [event for event in events if event[0] == "cleanup"] == [("cleanup", "manager")] * 3
