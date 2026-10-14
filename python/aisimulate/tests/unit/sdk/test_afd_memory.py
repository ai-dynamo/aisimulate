# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract of the native AFD memory-feasibility gate.

Pins the two per-pool batch compositions:

- the F pool checks at ``b_micro_total = a_replica_count * a_micro_batch`` (the F
  workers serve every A worker), while the A pool checks at ``a_micro_batch``;
- each phase's check consumes that phase's own partition weight sum
  (prefill -> context partition, decode -> generation partition).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import aisimulate.sdk.afd_memory as afd_memory
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.memory import kv_cache_budget_bytes

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pure_model(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.afd_parallel.validate_afd_model",
        lambda model: {"num_experts": 128, "num_shared_experts": 0, "num_moe_layers": 48},
    )


_CAPACITY_BYTES = 140 * (1 << 30)
_CAPACITY_GB = _CAPACITY_BYTES / (1 << 30)


@pytest.mark.parametrize("tp,ep", [(1, 1)])
def test_unsupported_topology_rejects_before_model_or_database(tp, ep, monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.afd_parallel.validate_afd_model", lambda model: pytest.fail("model metadata loaded")
    )
    monkeypatch.setattr(afd_memory, "get_database_view", lambda *a, **k: pytest.fail("database loaded"))
    with pytest.raises(NotImplementedError):
        afd_memory.check_afd_memory_fit(
            model_path="missing/model",
            backend_name="sglang",
            backend_version="test",
            isl=1024,
            osl=128,
            prefix=0,
            nextn=0,
            batch_size_per_a_replica=18,
            num_microbatches=3,
            a_weight_gb={"prefill": 0.0, "decode": 0.0},
            f_weight_gb={"prefill": 0.0, "decode": 0.0},
            a_pool={"a_tp_size": tp, "n_a_nodes": 1},
            f_pool={"f_moe_ep_size": ep, "n_f_nodes": 1},
            gpus_per_node=8,
        )


class _FakeBackend:
    """Captures the resolved per-call memory-usage kwargs (num_tokens etc.)."""

    name = SimpleNamespace(value="trtllm")

    def __init__(
        self,
        *,
        free_fraction: float | None = None,
        of_free: bool = False,
        kv_check_params: tuple[float, float] = (0.0, 0.0),
        usage: dict[str, float] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._free_fraction = free_fraction
        self._of_free = of_free
        self._kv_check_params = kv_check_params
        self._usage = usage if usage is not None else {"activations": 0.1, "kvcache": 0.1, "nccl": 0.0, "others": 0.0}

    def get_default_free_gpu_memory_fraction(self, version):
        return self._free_fraction

    def get_kv_cache_memory_check_params(self):
        return self._kv_check_params

    def memory_fraction_of_free(self):
        return self._of_free

    def get_partition_memory_usage(self, model, database, **kwargs):
        self.calls.append(kwargs)
        return dict(self._usage)


def _install(monkeypatch, **backend_kwargs) -> tuple[_FakeBackend, list[dict]]:
    backend = _FakeBackend(**backend_kwargs)
    monkeypatch.setattr(afd_memory, "get_backend", lambda name: backend)
    monkeypatch.setattr(
        afd_memory,
        "get_database_view",
        lambda *a, **k: SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}}),
    )
    monkeypatch.setattr(
        afd_memory,
        "build_model_config",
        lambda tp, pp, dp, mtp, ep, **k: ModelConfig(
            tp_size=tp, pp_size=pp, attention_dp_size=dp, moe_tp_size=mtp, moe_ep_size=ep
        ),
    )
    monkeypatch.setattr(afd_memory, "resolve_context_fmha_by_data", lambda *a, **k: None)
    monkeypatch.setattr(afd_memory, "resolve_dsv4_moe_arch", lambda *a, **k: None)
    monkeypatch.setattr(afd_memory, "resolve_nvfp4_for_system", lambda *a, **k: None)
    monkeypatch.setattr(afd_memory, "get_model", lambda path, config, backend: SimpleNamespace(config=config))

    # Spy on _pool_memory to see the phase/batch/weight selection of the
    # check loop (the real body still runs against the fake backend).
    captured: list[dict] = []
    real_pool_memory = afd_memory._pool_memory

    def spy_pool_memory(backend_arg, database, model, **kwargs):
        captured.append(dict(kwargs))
        return real_pool_memory(backend_arg, database, model, **kwargs)

    monkeypatch.setattr(afd_memory, "_pool_memory", spy_pool_memory)
    return backend, captured


@pytest.mark.parametrize("selected_roots", [False, True])
def test_custom_systems_roots_reach_fit_and_kv_budget(monkeypatch, tmp_path, selected_roots):
    _install(monkeypatch, free_fraction=0.9)
    roots = [str(tmp_path / "primary"), str(tmp_path / "secondary")]
    database_roots, version_roots = [], []

    def database(system, backend, version, **kwargs):
        database_roots.append((system, kwargs["systems_paths"]))
        return SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}})

    def latest(system, backend, **kwargs):
        version_roots.append((system, kwargs["systems_paths"]))
        return "test"

    monkeypatch.setattr(afd_memory, "get_database_view", database)
    monkeypatch.setattr(afd_memory, "get_latest_database_version", latest)
    common = dict(
        model_path="test/model",
        backend_name="sglang",
        backend_version=None,
        isl=16,
        osl=4,
        prefix=0,
        a_pool={"system": "custom_a", "a_tp_size": 1, "n_a_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=6,
        num_microbatches=3,
        a_weight_gb={"prefill": 0.0, "decode": 0.0},
        systems_paths=roots,
    )
    f_pool = {"system": "custom_f", "f_moe_ep_size": 8, "n_f_nodes": 1}
    if selected_roots:
        common["a_pool"]["systems_paths"] = roots[:1]
        f_pool["systems_paths"] = roots[1:]
    afd_memory.check_afd_memory_fit(
        **common,
        nextn=0,
        f_pool=f_pool,
        f_weight_gb={"prefill": 0.0, "decode": 0.0},
    )
    budget, per_token = afd_memory.estimate_afd_kv_budget_bytes(**common)
    assert budget > 0 and per_token > 0
    assert {system for system, _ in database_roots} == {"custom_a", "custom_f"}
    assert {system for system, _ in version_roots} == {"custom_a", "custom_f"}
    for system, actual in database_roots + version_roots:
        expected = roots[:1] if system == "custom_a" else roots[1:]
        assert actual == (expected if selected_roots else roots)


def test_f_pool_batch_scales_with_a_replica_count_and_weights_follow_phase(monkeypatch):
    backend, captured = _install(monkeypatch)

    # a_micro_batch = ceil(12 / 3) = 4; a_replica_count = 2 nodes * 8 GPUs / a_tp_size=1
    # = 16; b_micro_total = 16 * 4 = 64.
    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-30B-A3B",
        backend_name="trtllm",
        backend_version="1.3.0",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 2},
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=12,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="both",
        max_seq_len=None,
    )

    assert [(entry["phase"], entry["batch_size"], entry["partition_weight_gb"]) for entry in captured] == [
        ("prefill", 4, 1.0),  # A pool: micro batch, context-partition weights
        ("prefill", 64, 3.0),  # F pool: b_micro_total, context-partition weights
        ("decode", 4, 2.0),  # A pool: micro batch, generation-partition weights
        ("decode", 64, 4.0),  # F pool: b_micro_total, generation-partition weights
    ]
    # A-pool-only KV; the phase -> multiplier mapping happens inside
    # _pool_memory (prefill charges 1, decode num_microbatches).
    assert [entry["include_kvcache"] for entry in captured] == [True, False, True, False]
    # The phase composition inside _pool_memory: prefill charges no
    # tokens at max(isl, 1) with multiplier 1; decode charges each pool's own
    # batch at isl+osl with the microbatch multiplier.
    assert [(call["num_tokens"], call["max_seq_len"], call["kvcache_multiplier"]) for call in backend.calls] == [
        (0, 1024, 1),
        (0, 1024, 1),
        (4, 1152, 3),
        (64, 1152, 3),
    ]


def test_single_phase_role_checks_only_that_phase(monkeypatch):
    _backend, captured = _install(monkeypatch)

    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-30B-A3B",
        backend_name="trtllm",
        backend_version="1.3.0",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1},
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=18,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="decode",
        max_seq_len=None,
    )

    # a_replica_count = 1 * 8 / 1 = 8 -> b_micro_total = 8 * 6 = 48.
    assert [(entry["phase"], entry["batch_size"], entry["partition_weight_gb"]) for entry in captured] == [
        ("decode", 6, 2.0),
        ("decode", 48, 4.0),
    ]


@pytest.mark.parametrize("nextn", [0, 3])
def test_tp_replicas_and_plan_workspace_share_the_admission_budget(monkeypatch, nextn):
    backend, captured = _install(monkeypatch, free_fraction=0.9)
    monkeypatch.setattr(
        afd_memory,
        "_build_pool",
        lambda *a, **k: (SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}}), object()),
    )
    common = dict(
        model_path="Qwen/Qwen3-30B-A3B",
        backend_name="sglang",
        backend_version="0.5.14",
        isl=1024,
        osl=8,
        prefix=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 4, "n_a_nodes": 2},
        gpus_per_node=8,
        batch_size_per_a_replica=5,
        num_microbatches=2,
        a_weight_gb={"prefill": 10.0, "decode": 10.0},
        phase="decode",
        nextn=nextn,
        a_activation_gb={"decode": 2.0},
        a_workspace_gb={"decode": 3.0},
    )
    diagnostics = afd_memory.check_afd_memory_fit(
        **common,
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
        f_weight_gb={"decode": 20.0},
        f_activation_gb={"decode": 4.0},
        f_workspace_gb={"decode": 5.0},
    )
    # Two A nodes / TP4 = four replicas; ceil(5/2)=3, F receives 12,
    # not 48 tokens. Weights are resident once, independent of two microbatches.
    assert [entry["batch_size"] for entry in captured] == [3, 12]
    a = diagnostics["decode"]["attention"]["components_gib_per_gpu"]
    f = diagnostics["decode"]["ffn"]["components_gib_per_gpu"]
    assert a["weights"] == 10 and a["activations"] == 2 and a["afd_workspace"] == 3
    assert f["weights"] == 20 and f["activations"] == 4 and f["afd_workspace"] == 5
    # Native buffers already include verification width, even at MTP depth three.
    # 140*.9 - (10 weights + 2 activation + 3 in-flight workspace) = 111 GiB.
    budget, _ = afd_memory.estimate_afd_kv_budget_bytes(**common)
    assert budget == pytest.approx(111 * (1 << 30))
    backend._usage["kvcache"] = 111
    afd_memory.check_afd_memory_fit(
        **common,
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
        f_weight_gb={"decode": 0.0},
    )
    # The fake F footprint stays below hard OOM; the A KV boundary is exact.
    backend._usage["kvcache"] = 111.01
    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
        afd_memory.check_afd_memory_fit(
            **common,
            f_pool={"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
            f_weight_gb={"decode": 0.0},
        )


def _check_hetero() -> None:
    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-30B-A3B",
        backend_name="trtllm",
        backend_version=None,
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1},
        f_pool={"system": "b200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=18,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="prefill",
        max_seq_len=None,
    )


def test_hetero_pools_resolve_their_own_latest_versions(monkeypatch):
    backend, _captured = _install(monkeypatch)
    monkeypatch.setattr(
        afd_memory,
        "get_latest_database_version",
        lambda *, system, backend, systems_paths=None: {"h200_sxm": "1.3.0", "b200_sxm": "0.24.0"}[system],
    )
    views: list[tuple[str, str]] = []

    def record_view(system, backend_name, version, **kwargs):
        views.append((system, version))
        return SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}})

    monkeypatch.setattr(afd_memory, "get_database_view", record_view)
    fraction_versions: list[str] = []
    monkeypatch.setattr(
        backend,
        "get_default_free_gpu_memory_fraction",
        lambda version: fraction_versions.append(version),
    )

    _check_hetero()

    # Each pool's database view resolves its own system's latest version; the
    # A-only free-fraction check keys off the A pool's version.
    assert views == [("h200_sxm", "1.3.0"), ("b200_sxm", "0.24.0")]
    assert fraction_versions == ["1.3.0"]


def test_explicit_backend_version_applies_to_both_pools(monkeypatch):
    _backend, _captured = _install(monkeypatch)

    def no_latest(*args, **kwargs):
        raise AssertionError("get_latest_database_version must not be called")

    monkeypatch.setattr(afd_memory, "get_latest_database_version", no_latest)
    views: list[tuple[str, str]] = []

    def record_view(system, backend_name, version, **kwargs):
        views.append((system, version))
        return SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}})

    monkeypatch.setattr(afd_memory, "get_database_view", record_view)

    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-30B-A3B",
        backend_name="trtllm",
        backend_version="9.9.9",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1},
        f_pool={"system": "b200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=18,
        num_microbatches=3,
        a_weight_gb={"prefill": 1.0, "decode": 2.0},
        f_weight_gb={"prefill": 3.0, "decode": 4.0},
        phase="prefill",
        max_seq_len=None,
    )

    assert views == [("h200_sxm", "9.9.9"), ("b200_sxm", "9.9.9")]


def _check_a_pool(*, a_weight_gb: float, phase: str = "prefill") -> None:
    """Run the gate with a controllable A-pool weight sum (F pool stays tiny)."""

    afd_memory.check_afd_memory_fit(
        model_path="Qwen/Qwen3-30B-A3B",
        backend_name="sglang",
        backend_version="0.5.14",
        isl=1024,
        osl=128,
        prefix=0,
        nextn=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1},
        f_pool={"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=18,
        num_microbatches=3,
        a_weight_gb={"prefill": a_weight_gb, "decode": a_weight_gb},
        f_weight_gb={"prefill": 1.0, "decode": 1.0},
        phase=phase,
        max_seq_len=None,
    )


def _install_kv(monkeypatch, *, kvcache: float, of_free: bool, reserved: float = 0.0, tolerance: float = 0.0) -> None:
    _install(
        monkeypatch,
        free_fraction=0.9,
        of_free=of_free,
        kv_check_params=(reserved, tolerance),
        usage={"activations": 0.0, "kvcache": kvcache, "nccl": 0.0, "others": 0.0},
    )


def test_of_total_kv_budget_subtracts_non_kv(monkeypatch):
    """sglang/vLLM cap TOTAL memory, so the budget must subtract the non-KV footprint.

    A pool: weights 100 GiB + KV 30 GiB = 130 GiB total on a 140 GiB device at
    fraction 0.9. The shared formula gives ``140 * 0.9 - 100 = 26`` GiB, so the
    30 GiB KV cache is over budget; a formula that forgets ``- non_kv`` compares
    against 126 GiB and waves it through.
    """
    _install_kv(monkeypatch, kvcache=30.0, of_free=False)

    expected = kv_cache_budget_bytes(capacity=_CAPACITY_GB, non_kv=100.0, fraction=0.9, of_free=False)
    assert expected == pytest.approx(26.0)

    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
        _check_a_pool(a_weight_gb=100.0)


def test_kv_check_runs_below_the_fraction_of_total(monkeypatch):
    """The KV gate must not be gated on ``total >= capacity * fraction``.

    With tolerance 0.1 the budget is ``140 * 0.9 * 0.9 - 100 = 13.4`` GiB, so a
    20 GiB KV cache is over budget even though the 120 GiB total is still under
    ``140 * 0.9 = 126`` GiB.
    """
    _install_kv(monkeypatch, kvcache=20.0, of_free=False, tolerance=0.1)

    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
        _check_a_pool(a_weight_gb=100.0)


def test_of_free_budget_and_strict_boundary(monkeypatch):
    """TRT-LLM applies the fraction to the FREE pool and rejects only above it."""
    _install_kv(monkeypatch, kvcache=45.0, of_free=True)

    # non_kv = 135 - 45 = 90; budget = (140 - 90) * 0.9 = 45 -> 45 is not > 45.
    assert kv_cache_budget_bytes(capacity=_CAPACITY_GB, non_kv=90.0, fraction=0.9, of_free=True) == pytest.approx(45.0)
    _check_a_pool(a_weight_gb=90.0)


def test_of_free_budget_rejects_one_gib_over(monkeypatch):
    _install_kv(monkeypatch, kvcache=46.0, of_free=True)

    with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
        _check_a_pool(a_weight_gb=90.0)


@pytest.mark.parametrize(
    ("of_free", "reserved", "tolerance", "weights", "kvcache"),
    [
        (False, 0.0, 0.0, 100.0, 25.0),
        (False, 0.0, 0.0, 100.0, 30.0),
        (False, 0.05, 0.05, 90.0, 30.0),
        (False, 0.0, 0.1, 100.0, 20.0),
        (True, 0.0, 0.0, 90.0, 44.0),
        (True, 0.0, 0.0, 90.0, 45.0),
        (True, 0.0, 0.0, 90.0, 46.0),
        (True, 0.1, 0.0, 60.0, 40.0),
    ],
)
def test_kv_gate_agrees_with_the_shared_budget_formula(monkeypatch, of_free, reserved, tolerance, weights, kvcache):
    """Anti-divergence guard: the gate must decide exactly what the shared helper says.

    ``kv_cache_budget_bytes`` exists so the estimate and the sweep share one
    formula; this pins the AFD gate to it across both fraction semantics.
    """
    _install_kv(monkeypatch, kvcache=kvcache, of_free=of_free, reserved=reserved, tolerance=tolerance)

    total = weights + kvcache
    if total >= _CAPACITY_GB:
        pytest.skip("hard-OOM path raises before the KV budget check")
    budget = kv_cache_budget_bytes(
        capacity=_CAPACITY_GB,
        non_kv=total - kvcache,
        fraction=0.9,
        of_free=of_free,
        reserved=reserved,
        tolerance=tolerance,
    )
    if kvcache > budget:
        with pytest.raises(afd_memory.AFDMemoryInfeasible, match="KV cache"):
            _check_a_pool(a_weight_gb=weights)
    else:
        _check_a_pool(a_weight_gb=weights)


def test_mtp_memory_scales_only_decode_and_reduces_kv_budget():
    from aisimulate.sdk.backends.factory import get_backend

    backend = get_backend("sglang")
    common = dict(
        model_path="Qwen/Qwen3-30B-A3B",
        backend_name="sglang",
        backend_version="0.5.14",
        isl=1,
        osl=8,
        prefix=0,
        a_pool={"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=64,
        num_microbatches=1,
        a_weight_gb={"prefill": 10.0, "decode": 10.0},
    )
    usages, budgets = [], []
    for depth in [0, 3]:
        _, _, database, model = afd_memory._prepare_a_pool(
            model_path=common["model_path"],
            backend_name="sglang",
            backend_version="0.5.14",
            gpus_per_node=8,
            phase="both",
            a_pool=common["a_pool"],
            nextn=depth,
        )
        assert model.config.nextn == model._nextn == depth
        usages.append(
            [
                afd_memory._pool_memory(
                    backend,
                    database,
                    model,
                    partition_weight_gb=10.0,
                    phase=phase,
                    batch_size=64,
                    isl=1,
                    osl=8,
                    prefix=0,
                    max_seq_len=None,
                    include_kvcache=True,
                    kvcache_multiplier=1,
                )["activations"]
                for phase in ["prefill", "decode"]
            ]
        )
        budgets.append(afd_memory.estimate_afd_kv_budget_bytes(**common, nextn=depth)[0])
    # Backend activation floor is applied before the four-token verification multiplier.
    assert usages[1][0] == usages[0][0]
    assert usages[1][1] == pytest.approx(4 * usages[0][1])
    assert 0 < budgets[1] < budgets[0]


@pytest.mark.parametrize("database_mode", ["SILICON", "HYBRID", "EMPIRICAL", "SOL"])
def test_memory_keeps_explicit_source_policies(monkeypatch, database_mode):
    _install(monkeypatch, free_fraction=0.9)
    seen = []

    def database(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(system_spec={"gpu": {"mem_capacity": _CAPACITY_BYTES}})

    monkeypatch.setattr(afd_memory, "get_database_view", database)
    common = dict(
        model_path="test/model",
        backend_name="sglang",
        backend_version="test",
        isl=1,
        osl=2,
        prefix=0,
        a_pool={"system": "a", "a_tp_size": 1, "n_a_nodes": 1},
        gpus_per_node=8,
        batch_size_per_a_replica=6,
        num_microbatches=3,
        a_weight_gb={"prefill": 0.0, "decode": 0.0},
        nextn=3,
        database_mode=database_mode,
        enable_shared_layer=False,
        strict_provenance=True,
    )
    afd_memory.check_afd_memory_fit(
        **common,
        f_pool={"system": "f", "n_f_nodes": 1, "f_moe_ep_size": 8},
        f_weight_gb={"prefill": 0.0, "decode": 0.0},
    )
    afd_memory.estimate_afd_kv_budget_bytes(**common)
    assert len(seen) == 3
    assert all(row["shared_layer"] is False and row["strict_provenance"] is True for row in seen)
    assert all(row["database_mode"] == database_mode for row in seen)
    assert all(row["allow_missing_data"] is (database_mode == "SOL") for row in seen)
