# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

import aisimulate.sweeper.estimator as estimator_mod
import aisimulate.sweeper.kv_load as kv_load_mod
import aisimulate.sweeper.search as search_mod
import aisimulate.sweeper.search_space as search_space_mod
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.engine_request import (
    EngineControlTemplate,
    materialize_role_engine_request,
)
from aisimulate.sweeper.estimator import resolve_role_estimator_specs
from aisimulate.sweeper.heterogeneous import (
    DisaggBackendPair,
    DisaggRole,
    RoleEstimatorSpecs,
    RoleIdentity,
)
from aisimulate.sweeper.kv_load import resolve_kv_load
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    EstimatorSpec,
    ReplaySpec,
    RunnerCapabilities,
)
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.search_space import enumerate_branches


def _config(**search_overrides) -> SmartSearchConfig:
    search_space = {
        "model_name": "shared/model",
        "hardware_sku": "h200_sxm",
        "backend": ["vllm"],
        "deployment_mode": ["disagg"],
        "gpu_budget": 8,
        "prefill_max_num_batched_tokens": [8192],
        "prefill_max_num_seqs": [16],
        "decode_max_num_batched_tokens": [8192],
        "decode_max_num_seqs": [256],
    }
    search_space.update(search_overrides)
    return SmartSearchConfig(
        search_space=search_space,
        workload={"isl": 128, "osl": 32, "concurrency": 1, "num_request_ratio": 1},
    )


def _estimator(
    backend: str,
    *,
    model: str,
    system: str,
    version: str,
) -> EstimatorSpec:
    return EstimatorSpec(
        model_path=model,
        model_architecture="ExampleForCausalLM",
        system=system,
        backend=backend,
        backend_version=version,
        performance_data_version=version,
        database_mode="SILICON",
        transfer_policy=("xshape",),
        forward_model="op_level",
        engine_step_backend="rust",
        systems_paths=(f"/{system}/systems",),
        performance_data_root=f"/{system}/systems",
    )


def _catalog() -> RoleEstimatorSpecs:
    pair = DisaggBackendPair("sglang", "vllm")
    prefill = _estimator(
        "sglang", model="prefill/model", system="gb200_nv18", version="0.4.9"
    )
    decode = _estimator(
        "vllm", model="decode/model", system="h200_sxm", version="0.10.1"
    )
    return RoleEstimatorSpecs(
        pair=pair,
        prefill=prefill,
        decode=decode,
        identities={
            "prefill": RoleIdentity(
                DisaggRole.PREFILL,
                prefill.model_path,
                prefill.system,
                prefill.backend,
                prefill.backend_version,
            ),
            "decode": RoleIdentity(
                DisaggRole.DECODE,
                decode.model_path,
                decode.system,
                decode.backend,
                decode.backend_version,
            ),
        },
    )


def test_role_overrides_inherit_unspecified_shared_inputs() -> None:
    config = _config(
        prefill_model_name="prefill/model",
        prefill_hardware_sku="gb200_nv18",
        prefill_backend=["sglang"],
        prefill_backend_version="0.4.9",
        prefill_enable_wideep=True,
        prefill_moe_backend="deepep_moe",
        decode_backend_version="0.10.1",
    )
    ss = config.search_space

    assert ss.has_role_overrides is True
    assert ss.role_value("prefill", "model_name") == "prefill/model"
    assert ss.role_value("decode", "model_name") == "shared/model"
    assert ss.role_value("decode", "hardware_sku") == "h200_sxm"
    assert ss.role_backends("prefill") == ["sglang"]
    assert ss.role_backends("decode") == ["vllm"]
    assert ss.requested_role_backend_version("prefill", "sglang") == "0.4.9"
    assert ss.requested_role_backend_version("decode", "vllm") == "0.10.1"


def test_role_overrides_fail_closed_for_aggregated_only_and_decode_chunking() -> None:
    with pytest.raises(ValueError, match="require deployment_mode to include 'disagg'"):
        _config(deployment_mode=["agg"], prefill_model_name="prefill/model")

    with pytest.raises(
        ValueError, match="decode_enable_chunked_prefill is unsupported"
    ):
        _config(decode_enable_chunked_prefill=True)


def test_role_estimator_resolution_uses_independent_inherited_views(
    monkeypatch,
) -> None:
    config = _config(
        prefill_model_name="prefill/model",
        prefill_hardware_sku="gb200_nv18",
        prefill_backend=["sglang"],
        prefill_backend_version="0.4.9",
        decode_model_name="decode/model",
        decode_backend_version="0.10.1",
    )
    seen = []

    def fake_resolve(search_space):
        backend = search_space.backend[0]
        seen.append(
            (
                search_space.model_name,
                search_space.hardware_sku,
                backend,
                search_space.backend_version,
            )
        )
        return {
            backend: _estimator(
                backend,
                model=search_space.model_name,
                system=search_space.hardware_sku,
                version=search_space.backend_version,
            )
        }

    monkeypatch.setattr(estimator_mod, "resolve_estimator_specs", fake_resolve)

    catalogs = resolve_role_estimator_specs(config.search_space)

    assert list(catalogs) == ["prefill=sglang,decode=vllm"]
    assert seen == [
        ("prefill/model", "gb200_nv18", "sglang", "0.4.9"),
        ("decode/model", "h200_sxm", "vllm", "0.10.1"),
    ]
    assert catalogs["prefill=sglang,decode=vllm"].identities[
        "decode"
    ].inherited_fields == (
        "hardware_sku",
        "database_mode",
        "transfer_policy",
        "forward_model",
        "engine_step_backend",
        "systems_paths",
        "backend",
    )


def test_role_backend_version_mapping_records_shared_fallback_as_inherited(
    monkeypatch,
) -> None:
    config = _config(
        backend=["vllm"],
        backend_version={"vllm": "0.10.1"},
        prefill_backend=["sglang", "vllm"],
        prefill_backend_version={"sglang": "0.4.9"},
    )

    def fake_resolve(search_space):
        backend = search_space.backend[0]
        return {
            backend: _estimator(
                backend,
                model=search_space.model_name,
                system=search_space.hardware_sku,
                version=search_space.backend_version,
            )
        }

    monkeypatch.setattr(estimator_mod, "resolve_estimator_specs", fake_resolve)

    catalogs = resolve_role_estimator_specs(config.search_space)

    inherited = (
        catalogs["prefill=vllm,decode=vllm"]
        .identity_for(DisaggRole.PREFILL)
        .inherited_fields
    )
    assert "backend_version" in inherited


def test_heterogeneous_branch_pairs_role_shapes_under_shared_budget(
    monkeypatch,
) -> None:
    config = _config(
        prefill_model_name="prefill/model",
        prefill_hardware_sku="gb200_nv18",
        prefill_backend=["sglang"],
        prefill_enable_wideep=True,
        num_gpu_per_replica=[6],
    )
    catalog = _catalog()
    role_estimators = {catalog.pair.label: catalog}
    role_controls = {
        catalog.pair.label: {
            "prefill": EngineControlTemplate(
                "sglang", 32768, "EXAMPLE", True, "of_total"
            ),
            "decode": EngineControlTemplate(
                "vllm", 16384, "EXAMPLE", False, "of_total"
            ),
        }
    }
    seen = []

    def fake_parallel_configs(model, system, **kwargs):
        seen.append((model, system, kwargs["backend"], kwargs["max_seq_len"]))
        if kwargs["backend"] == "sglang":
            return [ReplicaParallelConfig(ParallelShape(4, 1, 1, 4), 1)]
        return [
            ReplicaParallelConfig(ParallelShape(2, 1, 1, 1), 1),
            ReplicaParallelConfig(ParallelShape(2, 1, 1, 1), 3),
        ]

    monkeypatch.setattr(search_space_mod, "parallel_configs_for", fake_parallel_configs)
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "disagg"), ("sglang", "disagg")),
        supported_disaggregated_backend_pairs=(("sglang", "vllm"),),
    )

    branch = enumerate_branches(
        config,
        runner_capabilities=capabilities,
        role_estimator_specs=role_estimators,
        role_engine_controls=role_controls,
    )[0]

    assert seen == [
        ("prefill/model", "gb200_nv18", "sglang", 32768),
        ("decode/model", "h200_sxm", "vllm", 16384),
    ]
    assert branch.knob_choices["backend"] == [catalog.pair.label]
    assert len(branch.parallel_configs) == 1
    assert branch.parallel_configs[0].total_gpus == 6
    assert branch.supported_backends[branch.parallel_configs[0]] == frozenset(
        {catalog.pair.label}
    )


def test_heterogeneous_execution_domains_materialize_and_match_runner_topology(
    monkeypatch,
) -> None:
    config = _config(
        gpu_budget=12,
        num_gpu_per_replica=[12],
        max_gpu_per_replica=12,
        max_prefill_workers=1,
        max_decode_workers=2,
        prefill_model_name="prefill/model",
        prefill_hardware_sku="gb200_nv18",
        prefill_backend=["sglang"],
        prefill_num_gpu_candidates=[8],
        prefill_tp_candidates=[2],
        prefill_pp_candidates=[2],
        prefill_dp_candidates=[1],
        prefill_moe_tp_candidates=[1],
        prefill_moe_ep_candidates=[1],
        prefill_cp_candidates=[2],
        prefill_num_workers_candidates=[1, 2],
        prefill_batch_size_candidates=[7],
        prefill_context_tokens_candidates=[12288],
        decode_model_name="decode/model",
        decode_num_gpu_candidates=[2],
        decode_tp_candidates=[2],
        decode_pp_candidates=[1],
        decode_dp_candidates=[1],
        decode_moe_tp_candidates=[1],
        decode_moe_ep_candidates=[1],
        decode_cp_candidates=[1],
        decode_num_workers_candidates=[1, 2],
        decode_batch_size_candidates=[19],
        decode_context_tokens_candidates=[4096],
    )
    catalog = _catalog()
    pair_label = catalog.pair.label
    role_estimators = {pair_label: catalog}
    role_controls = {
        pair_label: {
            "prefill": EngineControlTemplate(
                "sglang", 32768, "EXAMPLE", True, "of_total"
            ),
            "decode": EngineControlTemplate(
                "vllm", 16384, "EXAMPLE", False, "of_total"
            ),
        }
    }
    role_domains = {}

    def fake_parallel_configs(model, system, **kwargs):
        del model, system
        role_domains[kwargs["backend"]] = kwargs["agg_candidates"]
        if kwargs["backend"] == "sglang":
            shape = ParallelShape(2, 1, 1, 1, pp=2, cp=2)
            return [
                ReplicaParallelConfig(shape, 1),
                ReplicaParallelConfig(shape, 2),
            ]
        shape = ParallelShape(2, 1, 1, 1, pp=1, cp=1)
        return [
            ReplicaParallelConfig(shape, 1),
            ReplicaParallelConfig(shape, 2),
        ]

    monkeypatch.setattr(search_space_mod, "parallel_configs_for", fake_parallel_configs)
    capabilities = EngineReplayRunnerFactory().capabilities()

    (branch,) = enumerate_branches(
        config,
        runner_capabilities=capabilities,
        role_estimator_specs=role_estimators,
        role_engine_controls=role_controls,
    )

    assert asdict(role_domains["sglang"]) == {
        "gpus_per_worker": (8,),
        "tp": (2,),
        "pp": (2,),
        "attention_dp": (1,),
        "moe_tp": (1,),
        "moe_ep": (1,),
        "cp": (2,),
        "workers": (1, 2),
    }
    assert asdict(role_domains["vllm"])["workers"] == (1, 2)
    assert branch.enumeration_counts == {
        "considered": 1,
        "accepted": 1,
        "pruned": 0,
    }
    assert len(branch.parallel_configs) == 1
    parallel = branch.parallel_configs[0]
    assert parallel.prefill.shape.pp == 2
    assert parallel.prefill.shape.cp == 2
    assert parallel.prefill.replicas == 1
    assert parallel.decode.replicas == 2
    assert parallel.total_gpus == 12
    assert branch.knob_choices["prefill_batch_size"] == [7]
    assert branch.knob_choices["prefill_context_tokens"] == [12288]
    assert branch.knob_choices["decode_batch_size"] == [19]
    assert branch.knob_choices["decode_context_tokens"] == [4096]

    sample = unroll_sample(
        search_space=config.search_space,
        selection={
            "deployment_mode": "disagg",
            "backend": pair_label,
            "prefill_batch_size": 7,
            "prefill_context_tokens": 12288,
            "decode_batch_size": 19,
            "decode_context_tokens": 4096,
        },
        parallel_config=parallel,
        backend_pair=catalog.pair,
    )
    engine_request = materialize_role_engine_request(
        catalog.pair,
        role_controls[pair_label],
        catalog,
        config=config,
        sample=sample,
    )
    deployment = build_backend_deployment(
        sample,
        backend_version="prefill=0.4.9,decode=0.10.1",
        engine_request=engine_request,
        role_estimators={"prefill": catalog.prefill, "decode": catalog.decode},
    )

    assert deployment.num_prefill_workers == 1
    assert deployment.num_decode_workers == 2
    assert deployment.prefill_engine_args["aic_pp_size"] == 2
    assert deployment.prefill_engine_args["aic_cp_size"] == 2
    assert deployment.prefill_engine_args["max_num_seqs"] == 7
    assert deployment.prefill_engine_args["max_num_batched_tokens"] == 12288
    assert deployment.decode_engine_args["max_num_seqs"] == 19
    assert deployment.decode_engine_args["max_num_batched_tokens"] == 4096

    class RecordingRuntime:
        execution = None

        def run_replay_json(self, execution_spec_json):
            self.execution = json.loads(execution_spec_json)
            return json.dumps(
                {
                    "duration_ms": 1.0,
                    "output_throughput_tok_s": 1.0,
                    "gpu_hours": 0.0,
                    "completed_requests": 1,
                }
            )

    deployment.prefill_engine_args["num_gpu_blocks"] = 16
    deployment.decode_engine_args["num_gpu_blocks"] = 16
    runtime = RecordingRuntime()
    replay = ReplaySpec(
        backend_deployment=deployment,
        workload={"isl": 128, "osl": 32, "concurrency": 1, "num_request_ratio": 1},
        goal={"target": "throughput"},
    )
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(replay)

    assert runtime.execution["topology"]["prefill"]["initial_workers"] == 1
    assert runtime.execution["topology"]["decode"]["initial_workers"] == 2
    prefill_timing = runtime.execution["engine"]["prefill"]["rank"]["timing_model"]
    assert prefill_timing["config"]["pp"] == 2
    assert prefill_timing["config"]["cp_size"] == 2

    mismatched = replace(
        deployment,
        parallel_config={**deployment.parallel_config, "prefill_pp": 1},
    )
    with pytest.raises(ValueError, match="prefill pipeline parallel size=2"):
        EngineReplayRunnerFactory(runtime=runtime).create(0).run(
            replace(replay, backend_deployment=mismatched)
        )


def test_partial_heterogeneous_pair_pruning_retains_stable_diagnostics(
    monkeypatch,
) -> None:
    config = _config(
        prefill_backend=["sglang", "vllm"],
        decode_backend=["vllm"],
    )
    mixed = _catalog()
    homogeneous_prefill = _estimator(
        "vllm", model="prefill/model", system="gb200_nv18", version="0.10.1"
    )
    homogeneous = RoleEstimatorSpecs(
        pair=DisaggBackendPair("vllm", "vllm"),
        prefill=homogeneous_prefill,
        decode=mixed.decode,
        identities={
            "prefill": RoleIdentity(
                DisaggRole.PREFILL,
                homogeneous_prefill.model_path,
                homogeneous_prefill.system,
                homogeneous_prefill.backend,
                homogeneous_prefill.backend_version,
            ),
            "decode": mixed.identities["decode"],
        },
    )
    catalogs = {mixed.pair.label: mixed, homogeneous.pair.label: homogeneous}
    controls = {
        pair_label: {
            "prefill": EngineControlTemplate(
                estimators.prefill.backend, 32768, "EXAMPLE", True, "of_total"
            ),
            "decode": EngineControlTemplate(
                estimators.decode.backend, 16384, "EXAMPLE", False, "of_total"
            ),
        }
        for pair_label, estimators in catalogs.items()
    }

    def fake_parallel_configs(model, system, **kwargs):
        if kwargs["backend"] == "sglang":
            raise search_space_mod.NoViableParallelConfig("sglang role does not fit")
        return [ReplicaParallelConfig(ParallelShape(2, 1, 1, 1), 1)]

    monkeypatch.setattr(search_space_mod, "parallel_configs_for", fake_parallel_configs)
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "disagg"), ("sglang", "disagg")),
        supported_disaggregated_backend_pairs=(
            ("sglang", "vllm"),
            ("vllm", "vllm"),
        ),
    )

    with pytest.warns(UserWarning, match="category='no_parallel_config'"):
        (branch,) = enumerate_branches(
            config,
            runner_capabilities=capabilities,
            role_estimator_specs=catalogs,
            role_engine_controls=controls,
        )

    assert branch.knob_choices["backend"] == [homogeneous.pair.label]
    assert branch.enumeration_counts == {
        "considered": 2,
        "accepted": 1,
        "pruned": 1,
    }
    assert [diagnostic.as_dict() for diagnostic in branch.pruning_diagnostics] == [
        {
            "backend": mixed.pair.label,
            "role": "prefill",
            "category": "no_parallel_config",
            "detail": "sglang role does not fit",
        }
    ]


def test_all_heterogeneous_pairs_pruned_retain_terminal_report(
    monkeypatch,
) -> None:
    config = _config(
        prefill_backend=["sglang", "vllm"],
        decode_backend=["vllm"],
    )
    mixed = _catalog()
    homogeneous_prefill = _estimator(
        "vllm", model="prefill/model", system="gb200_nv18", version="0.10.1"
    )
    homogeneous = RoleEstimatorSpecs(
        pair=DisaggBackendPair("vllm", "vllm"),
        prefill=homogeneous_prefill,
        decode=mixed.decode,
        identities={
            "prefill": RoleIdentity(
                DisaggRole.PREFILL,
                homogeneous_prefill.model_path,
                homogeneous_prefill.system,
                homogeneous_prefill.backend,
                homogeneous_prefill.backend_version,
            ),
            "decode": mixed.identities["decode"],
        },
    )
    catalogs = {mixed.pair.label: mixed, homogeneous.pair.label: homogeneous}
    controls = {
        pair_label: {
            "prefill": EngineControlTemplate(
                estimators.prefill.backend, 32768, "EXAMPLE", True, "of_total"
            ),
            "decode": EngineControlTemplate(
                estimators.decode.backend, 16384, "EXAMPLE", False, "of_total"
            ),
        }
        for pair_label, estimators in catalogs.items()
    }

    def no_role_fits(model, system, **kwargs):
        raise search_space_mod.NoViableParallelConfig(
            f"{kwargs['backend']} role does not fit"
        )

    monkeypatch.setattr(search_space_mod, "parallel_configs_for", no_role_fits)
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("vllm", "disagg"), ("sglang", "disagg")),
        supported_disaggregated_backend_pairs=(
            ("sglang", "vllm"),
            ("vllm", "vllm"),
        ),
    )

    with pytest.warns(UserWarning) as warning_records, pytest.raises(
        search_space_mod.NoViableParallelConfig
    ) as exc_info:
        enumerate_branches(
            config,
            runner_capabilities=capabilities,
            role_estimator_specs=catalogs,
            role_engine_controls=controls,
        )

    assert any("no configured backend" in str(item.message) for item in warning_records)
    assert exc_info.value.as_dict()["enumeration_reports"] == [
        {
            "deployment_mode": "disagg",
            "counts": {"considered": 2, "accepted": 0, "pruned": 2},
            "pruning_diagnostics": [
                {
                    "backend": mixed.pair.label,
                    "role": "prefill",
                    "category": "no_parallel_config",
                    "detail": "sglang role does not fit",
                },
                {
                    "backend": homogeneous.pair.label,
                    "role": "prefill",
                    "category": "no_parallel_config",
                    "detail": "vllm role does not fit",
                },
            ],
        }
    ]


def test_deployment_materializes_role_correct_engine_and_estimator_payloads() -> None:
    config = _config(
        enable_chunked_prefill=True,
        prefill_model_name="prefill/model",
        prefill_hardware_sku="gb200_nv18",
        prefill_backend=["sglang"],
        prefill_enable_wideep=True,
        prefill_moe_backend="deepep_moe",
        prefill_free_gpu_memory_fraction=0.81,
        decode_model_name="decode/model",
        decode_gemm_quant_mode="fp8",
        decode_free_gpu_memory_fraction=0.72,
    )
    catalog = _catalog()
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(4, 1, 1, 4), 1),
        decode=ReplicaParallelConfig(ParallelShape(2, 1, 1, 1), 1),
    )
    selection = {
        "deployment_mode": "disagg",
        "backend": catalog.pair.label,
        "prefill_max_num_batched_tokens": 8192,
        "prefill_max_num_seqs": 16,
        "decode_max_num_batched_tokens": 8192,
        "decode_max_num_seqs": 256,
    }
    sample = unroll_sample(
        search_space=config.search_space,
        selection=selection,
        parallel_config=parallel,
        backend_pair=catalog.pair,
    )
    engine_request = materialize_role_engine_request(
        catalog.pair,
        {
            "prefill": EngineControlTemplate(
                "sglang", 32768, "EXAMPLE", True, "of_total"
            ),
            "decode": EngineControlTemplate(
                "vllm", 16384, "EXAMPLE", False, "of_total"
            ),
        },
        catalog,
        config=config,
        sample=sample,
    )
    sample["engine_request"] = asdict(engine_request)

    assert engine_request.role_requests["prefill"].enable_chunked_prefill is True
    assert engine_request.role_requests["decode"].enable_chunked_prefill is False

    deployment = build_backend_deployment(
        sample,
        backend_version="prefill=0.4.9,decode=0.10.1",
        engine_request=engine_request,
        role_estimators={"prefill": catalog.prefill, "decode": catalog.decode},
    )

    assert deployment.prefill_backend == "sglang"
    assert deployment.decode_backend == "vllm"
    assert deployment.prefill_backend_version == "0.4.9"
    assert deployment.decode_backend_version == "0.10.1"
    assert deployment.role_estimators == {
        "prefill": catalog.prefill,
        "decode": catalog.decode,
    }
    assert deployment.prefill_engine_args["aic_model_path"] == "prefill/model"
    assert deployment.prefill_engine_args["aic_system"] == "gb200_nv18"
    assert deployment.prefill_engine_args["engine_type"] == "sglang"
    assert deployment.prefill_engine_args["mem_fraction_static"] == 0.81
    assert deployment.prefill_engine_args["aic_enable_wideep"] is True
    assert deployment.prefill_engine_args["aic_moe_backend"] == "deepep_moe"
    assert deployment.decode_engine_args["aic_model_path"] == "decode/model"
    assert deployment.decode_engine_args["engine_type"] == "vllm"
    assert deployment.decode_engine_args["gpu_memory_utilization"] == 0.72
    assert deployment.decode_engine_args["aic_gemm_dtype"] == "fp8"
    assert deployment.prefill_engine_args["enable_chunked_prefill"] is True
    assert deployment.decode_engine_args["enable_chunked_prefill"] is False
    assert deployment.disaggregated_corrections is not None
    assert deployment.disaggregated_corrections.prefill_rate_degradation == 0.9

    with pytest.raises(ValueError, match="decode estimator backend.*sampled backend"):
        build_backend_deployment(
            sample,
            backend_version="prefill=0.4.9,decode=0.10.1",
            engine_request=engine_request,
            role_estimators={
                "prefill": catalog.prefill,
                "decode": replace(catalog.decode, backend="sglang"),
            },
        )


def test_candidate_materialization_retains_role_identity_and_provenance() -> None:
    config = _config(
        prefill_model_name="prefill/model",
        prefill_hardware_sku="gb200_nv18",
        prefill_backend=["sglang"],
        decode_model_name="decode/model",
    )
    catalog = _catalog()
    pair_label = catalog.pair.label
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(4, 1, 1, 4), 1),
        decode=ReplicaParallelConfig(ParallelShape(2, 1, 1, 1), 1),
    )
    selection = {
        "deployment_mode": "disagg",
        "backend": pair_label,
        "prefill_max_num_batched_tokens": 8192,
        "prefill_max_num_seqs": 16,
        "decode_max_num_batched_tokens": 8192,
        "decode_max_num_seqs": 256,
    }

    class Factory:
        def capabilities(self):
            return RunnerCapabilities(
                supported_backend_topologies=(
                    ("sglang", "disagg"),
                    ("vllm", "disagg"),
                ),
                supported_disaggregated_backend_pairs=(("sglang", "vllm"),),
            )

    prepared, result = search_mod._materialize_one(
        selection,
        parallel,
        config=config,
        goal=config.goal,
        providers={},
        provider_plans={},
        runner_factory=Factory(),
        estimator_specs={},
        engine_controls={},
        role_estimator_specs={pair_label: catalog},
        role_engine_controls={
            pair_label: {
                "prefill": EngineControlTemplate(
                    "sglang", 32768, "EXAMPLE", True, "of_total"
                ),
                "decode": EngineControlTemplate(
                    "vllm", 16384, "EXAMPLE", False, "of_total"
                ),
            }
        },
    )

    assert result is None
    assert prepared is not None
    assert prepared.sample["backend"] == pair_label
    assert prepared.sample["prefill_backend_version"] == "0.4.9"
    assert prepared.sample["decode_backend_version"] == "0.10.1"
    assert prepared.sample["role_identities"]["prefill"]["hardware_sku"] == "gb200_nv18"
    assert prepared.sample["role_estimators"]["decode"]["model_path"] == "decode/model"
    deployment = prepared.replay_spec.backend_deployment
    assert deployment.prefill_backend == "sglang"
    assert deployment.decode_backend == "vllm"


def test_runner_requires_explicit_heterogeneous_pair_capability() -> None:
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("sglang", "disagg"), ("vllm", "disagg"))
    )
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg",
        backend="prefill=sglang,decode=vllm",
        backend_version="prefill=0.4.9,decode=0.10.1",
        prefill_backend="sglang",
        prefill_backend_version="0.4.9",
        decode_backend="vllm",
        decode_backend_version="0.10.1",
    )

    with pytest.raises(ValueError, match="does not explicitly support.*backend pair"):
        capabilities.require_compatible(
            ReplaySpec(backend_deployment=deployment, workload={}, goal={})
        )


def test_kv_load_resolves_model_system_backend_version_and_controls_by_role(
    monkeypatch,
) -> None:
    config = _config(
        prefill_model_name="prefill/model",
        prefill_hardware_sku="gb200_nv18",
        prefill_backend=["sglang"],
        decode_model_name="decode/model",
        decode_gemm_quant_mode="fp8",
    )
    catalog = _catalog()
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(1, 1, 1, 1), 1),
        decode=ReplicaParallelConfig(ParallelShape(1, 1, 1, 1), 1),
    )
    sample = unroll_sample(
        search_space=config.search_space,
        selection={
            "deployment_mode": "disagg",
            "backend": catalog.pair.label,
            "prefill_max_num_batched_tokens": 8192,
            "prefill_max_num_seqs": 16,
            "decode_max_num_batched_tokens": 8192,
            "decode_max_num_seqs": 256,
        },
        parallel_config=parallel,
        backend_pair=catalog.pair,
    )
    sample["role_estimators"] = {
        "prefill": asdict(catalog.prefill),
        "decode": asdict(catalog.decode),
    }
    sample["engine_request"] = {
        "role_requests": {
            "prefill": {"memory_fraction": 0.81, "nextn": 0},
            "decode": {
                "memory_fraction": 0.72,
                "nextn": 0,
                "gemm_quant_mode": "fp8",
            },
        }
    }
    seen = []

    def fake_capacity(shape, **kwargs):
        seen.append(kwargs)
        return 8192

    monkeypatch.setattr(kv_load_mod, "_per_rank_capacity_tokens", fake_capacity)

    resolution = resolve_kv_load(
        sample,
        workload=config.workload,
        parallel_config=parallel,
        ratio=1.0,
        backend_version="unused-composite",
        role_backend_versions={"prefill": "0.4.9", "decode": "0.10.1"},
    )

    assert resolution.role_capacity_tokens == {"prefill": 8192, "decode": 8192}
    assert [
        (
            item["model_name"],
            item["hardware_sku"],
            item["backend"],
            item["backend_version"],
            item["memory_fraction"],
            item["gemm_quant_mode"],
        )
        for item in seen
    ] == [
        ("prefill/model", "gb200_nv18", "sglang", "0.4.9", 0.81, None),
        ("decode/model", "h200_sxm", "vllm", "0.10.1", 0.72, "fp8"),
    ]


def test_runner_capabilities_report_the_unsupported_role() -> None:
    config = _config(
        prefill_model_name="prefill/model",
        prefill_backend=["sglang"],
    )
    catalog = _catalog()
    parallel = DisaggParallelConfig(
        prefill=ReplicaParallelConfig(ParallelShape(1, 1, 1, 1), 1),
        decode=ReplicaParallelConfig(ParallelShape(1, 1, 1, 1), 1),
    )
    sample = unroll_sample(
        search_space=config.search_space,
        selection={
            "deployment_mode": "disagg",
            "backend": catalog.pair.label,
            "prefill_max_num_batched_tokens": 8192,
            "prefill_max_num_seqs": 16,
            "decode_max_num_batched_tokens": 8192,
            "decode_max_num_seqs": 256,
        },
        parallel_config=parallel,
        backend_pair=catalog.pair,
    )
    engine_request = materialize_role_engine_request(
        catalog.pair,
        {
            "prefill": EngineControlTemplate(
                "sglang", 32768, "EXAMPLE", False, "of_total"
            ),
            "decode": EngineControlTemplate(
                "vllm", 16384, "EXAMPLE", False, "of_total"
            ),
        },
        catalog,
        config=config,
        sample=sample,
    )
    deployment = build_backend_deployment(
        sample,
        backend_version="pair",
        engine_request=engine_request,
        role_estimators={"prefill": catalog.prefill, "decode": catalog.decode},
    )
    replay = ReplaySpec(backend_deployment=deployment, workload={}, goal={})
    capabilities = RunnerCapabilities(
        supported_backend_topologies=(("sglang", "disagg"),)
    )

    with pytest.raises(ValueError, match="decode backend/topology 'vllm'/'disagg'"):
        capabilities.require_compatible(replay)
