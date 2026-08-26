# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk.common import SupportResult
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import recommendation_to_sweeper
from aisimulate.runner import _materialize_engine_execution_spec
from aisimulate.support.errors import SupportWorkflowError
from aisimulate.support.fpm import run_fpm
from aisimulate.support.identity import support_cell_id
from aisimulate.support.plan import create_plan, existing_support
from aisimulate.support.schema import (
    EvidenceBundle,
    EvidenceRecord,
    ExecutionProfile,
    ExistingSupport,
    FPMProfile,
    SearchProfile,
    SloSpec,
    SupportIdentity,
    SupportRequest,
    ValidationPolicy,
    WorkloadSpec,
)
from aisimulate.support.search import bounded_topologies
from aisimulate.support.validation import validate_evidence


def _request(*, model_kind: str = "dense") -> SupportRequest:
    slo = SloSpec(ttft_ms=2_000, tpot_ms=30)
    return SupportRequest(
        identity=SupportIdentity(
            model="Qwen/Qwen3-32B",
            model_revision="0123456789abcdef0123456789abcdef01234567",
            tokenizer_revision="0123456789abcdef0123456789abcdef01234567",
            chat_template_revision="0123456789abcdef0123456789abcdef01234567",
            model_kind=model_kind,
            framework="vllm",
            framework_version="0.10.1",
            gpu="h200_sxm",
            gpu_count=8,
            node_count=1,
            gpus_per_node=8,
            interconnect="nvswitch",
            aisimulate_revision="test-revision",
        ),
        workloads=[
            WorkloadSpec(
                id="fixed-8k-1k",
                kind="synthetic",
                input_tokens=8192,
                output_tokens=1024,
                slo=slo,
            ),
            WorkloadSpec(
                id="agentx",
                kind="trace",
                trace_path="/traces/agentx.jsonl",
                trace_digest="a" * 64,
                trace_format="dynamo",
                slo=slo,
            ),
        ],
        search=SearchProfile(max_candidates=16),
        fpm=FPMProfile(),
        validation=ValidationPolicy(recommendation_uplift_min=1.5),
        execution=ExecutionProfile(),
    )


def test_bounded_search_is_deterministic_and_never_exceeds_mvp_cap() -> None:
    for model_kind in ("dense", "moe"):
        request = _request(model_kind=model_kind)
        first = bounded_topologies(request)
        second = bounded_topologies(request)
        assert first == second
        assert 4 <= len(first) <= 16
        assert len({candidate.id for candidate in first}) == len(first)
        assert all(candidate.total_gpus <= request.identity.gpu_count for candidate in first)
        if model_kind == "moe":
            assert all(
                candidate.tensor * candidate.attention_data == candidate.moe_tensor * candidate.moe_expert
                for candidate in first
            )

    large = _request().model_copy(
        update={
            "identity": _request().identity.model_copy(update={"gpu_count": 64, "node_count": 8, "gpus_per_node": 8})
        }
    )
    large_candidates = bounded_topologies(large)
    assert len(large_candidates) <= 16
    assert any(candidate.pipeline > 1 for candidate in large_candidates)


def test_plan_wires_fpm_overlay_and_brev_without_lifecycle_actions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aisimulate.support.plan.existing_support",
        lambda _request: ExistingSupport(status="unsupported"),
    )
    root = tmp_path / "support"
    plan = create_plan(_request(), root, overwrite=False)

    assert plan["search"]["candidate_count"] <= 16
    assert plan["execution"] == {"provider": "brev", "reuse_existing": True}
    assert (root / "systems" / "h200_sxm.yaml").is_file()
    commands = json.loads((root / "commands.json").read_text(encoding="utf-8"))
    assert commands["fpm_run_brev"][:3] == ["brev", "exec", "<BREV_INSTANCE>"]
    assert not any(command in commands["fpm_run_brev"] for command in ("create", "stop", "delete"))
    assert "--fpm-database-root" in commands["fpm_run_local"]

    for path in plan["outputs"]["recommendation_configs"]:
        config = CoreRecommendationConfig.from_yaml(path)
        assert config.engine.forward_model == "fpm"
        assert config.engine.systems_path == str(root / "systems")
        assert isinstance(config.engine.workers.aggregated.parallelism.preset, list)
        assert len(config.engine.workers.aggregated.parallelism.preset) <= 16
        smart = recommendation_to_sweeper(config)
        assert smart.search_space.forward_model == "fpm"
        assert smart.search_space.systems_path == str(root / "systems")


def test_support_check_rejects_architecture_only_inference(monkeypatch) -> None:
    monkeypatch.setattr(
        "aiconfigurator.sdk.common.check_support",
        lambda *_args, **_kwargs: SupportResult(
            agg_supported=True,
            disagg_supported=True,
            exact_match=False,
            architecture="Qwen3_5MoeForConditionalGeneration",
        ),
    )
    status = existing_support(_request())
    assert status.status == "unsupported"
    assert status.exact_match is False
    assert status.aggregated is False
    assert "only advisory" in (status.detail or "")


def test_plan_refuses_to_mix_exact_support_cells_on_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aisimulate.support.plan.existing_support",
        lambda _request: ExistingSupport(status="unsupported"),
    )
    root = tmp_path / "support"
    request = _request()
    create_plan(request, root, overwrite=False)
    different = request.model_copy(
        update={"identity": request.identity.model_copy(update={"model": "org/different-model"})}
    )
    with pytest.raises(SupportWorkflowError, match="refusing to mix support cells"):
        create_plan(different, root, overwrite=True)


def test_collect_preview_does_not_import_or_execute_collector(tmp_path, capsys) -> None:
    result = run_fpm(
        _request(),
        execute=False,
        smoke=False,
        limit=None,
        resume=False,
        checkpoint_dir=None,
        output_dir=str(tmp_path / "support"),
    )
    assert result == 0
    command = capsys.readouterr().out
    assert command.startswith("python3 -m collector.fpm_forward")
    assert "--plan-only" in command
    assert "--fpm-database-root" in command


def test_public_engine_config_reaches_native_fpm_timing_contract(tmp_path) -> None:
    config = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "synthetic", "input_tokens": 16, "output_tokens": 4},
                "load": {"type": "concurrency", "concurrency": 1},
                "stop": {"requests": 1},
            },
            "engine": {
                "mode": "aggregated",
                "model": "Qwen/Qwen3-32B",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.10.1",
                "forward_model": "fpm",
                "systems_path": str(tmp_path / "systems"),
                "context_length": 16384,
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "replicas": 1,
                            "tensor": 1,
                            "pipeline": 1,
                            "attention_data": 1,
                            "moe_tensor": 1,
                            "moe_expert": 1,
                        },
                        "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                    }
                },
            },
        }
    )
    replay = prediction_to_replay_spec(config)
    execution = _materialize_engine_execution_spec(
        replay,
        trace_block_size=512,
        record_per_request=False,
    )
    timing = execution["spec"]["engine"]["rank"]["timing_model"]["config"]
    assert timing["forward_model"] == "fpm"
    assert timing["systems_path"] == str(tmp_path / "systems")


def _records(request: SupportRequest) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    candidates: list[str] = []
    throughput = {"baseline": 100.0, "top1": 200.0, "top2": 160.0, "top3": 140.0}
    for workload in request.workloads:
        for role in ("baseline", "top1", "top2", "top3"):
            candidate = f"{workload.id}-{role}"
            candidates.append(candidate)
            for metric, measured in (
                ("ttft_ms", 100.0),
                ("tpot_ms", 10.0),
                ("output_throughput_tok_s", throughput[role]),
            ):
                records.append(
                    EvidenceRecord(
                        phase="e2e",
                        workload_id=workload.id,
                        config_role=role,
                        candidate_id=candidate,
                        metric=metric,
                        predicted=measured,
                        measured=measured,
                        gpu_count=request.identity.gpu_count,
                        source_run_id=f"e2e-{candidate}",
                        slo_compliant=True,
                    )
                )
    for candidate in candidates:
        role = candidate.rsplit("-", 1)[1]
        for phase in ("fpm_prefill", "fpm_decode"):
            records.append(
                EvidenceRecord(
                    phase=phase,
                    config_role=role,
                    candidate_id=candidate,
                    metric="forward_pass_ms",
                    predicted=10.0,
                    measured=10.0,
                    gpu_count=request.identity.gpu_count,
                    source_run_id=f"{phase}-{candidate}",
                    held_out=True,
                )
            )
    return records


def _write_fpm_database(request: SupportRequest, systems_root: Path) -> None:
    (systems_root / f"{request.identity.gpu}.yaml").parent.mkdir(parents=True, exist_ok=True)
    (systems_root / f"{request.identity.gpu}.yaml").write_text(
        f"data_dir: data/{request.identity.gpu}\n",
        encoding="utf-8",
    )
    version = (
        systems_root / "data" / request.identity.gpu / request.identity.framework / request.identity.framework_version
    )
    version.mkdir(parents=True)
    parquet = version / "fpm_forward_perf.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cell_id": "cell-1",
                    "model_path": request.identity.model,
                    "system": request.identity.gpu,
                    "backend": request.identity.framework,
                    "backend_version": request.identity.framework_version,
                    "workload_kind": "prefill",
                    "latency_ms": 10.0,
                    "source_plan_sha256": "plan",
                    "collector_attempt_id": "attempt",
                    "runtime_run_id": "run",
                    "runtime_grid_digest": "grid",
                }
            ]
        ),
        parquet,
    )
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "system": request.identity.gpu,
        "backend": request.identity.framework,
        "backend_version": request.identity.framework_version,
        "model_paths": [request.identity.model],
        "row_count": 1,
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
    }
    (version / "fpm_forward_perf.metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_validation_pass_requires_database_fpm_e2e_and_value_evidence(tmp_path) -> None:
    request = _request()
    systems_root = tmp_path / "systems"
    _write_fpm_database(request, systems_root)
    evidence = EvidenceBundle(
        support_cell_id=support_cell_id(request),
        records=_records(request),
    )
    result = validate_evidence(request, evidence, systems_root=systems_root)
    assert result.status == "pass"
    assert set(result.gates) == {
        "contract_completeness",
        "fpm_publication",
        "e2e_accuracy",
        "fpm_accuracy",
        "recommendation_value",
    }
    assert all(gate.status == "pass" for gate in result.gates.values())


def test_validation_blocks_when_formal_fpm_publication_is_missing(tmp_path) -> None:
    request = _request()
    evidence = EvidenceBundle(
        support_cell_id=support_cell_id(request),
        records=_records(request),
    )
    result = validate_evidence(request, evidence, systems_root=tmp_path / "missing")
    assert result.status == "blocked"
    assert result.gates["fpm_publication"].status == "blocked"


def test_validation_blocks_fpm_accuracy_until_e2e_candidates_exist(tmp_path) -> None:
    request = _request()
    evidence = EvidenceBundle(
        support_cell_id=support_cell_id(request),
        records=[],
    )
    result = validate_evidence(request, evidence, systems_root=tmp_path / "missing")
    assert result.status == "failed"
    assert result.gates["e2e_accuracy"].status == "failed"
    assert result.gates["fpm_accuracy"].status == "blocked"
