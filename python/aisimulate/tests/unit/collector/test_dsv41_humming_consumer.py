# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from collector.sglang.dsv41_contract import build_manifest, write_parquet

from aisimulate_core.sdk import common, rust_engine_step
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("system", ["h100_sxm", "h200_sxm"])
def test_native_humming_fixture_reaches_v41_prediction(tmp_path, system, monkeypatch):
    # Synthetic selection witnesses only; no GPU measurements or accuracy claim.
    source = Path(__file__).resolve().parents[3] / "src/aisimulate_core/systems"
    shutil.copy(source / f"{system}.yaml", tmp_path / f"{system}.yaml")
    version = "dev1aa0e962"
    data = tmp_path / f"data/{system}/sglang/{version}"
    data.mkdir(parents=True)
    rows = [
        dict(
            moe_dtype=name,
            num_tokens=1,
            hidden_size=5120,
            inter_size=2304,
            topk=6,
            num_experts=384,
            moe_tp_size=4,
            moe_ep_size=1,
            distribution="uniform",
            latency=value,
            kernel_source=kernel,
        )
        for name, kernel, value in [
            ("w4a16_mxfp4_humming", "sglang_mxfp4_humming_moe", 1.0),
            ("w4a16_mxfp4_cutlass", "sglang_flashinfer_cutlass_moe", 2.0),
        ]
    ]
    pq.write_table(pa.Table.from_pylist(rows), data / "moe_perf.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [dict(gemm_dtype="bfloat16", m=1, n=n, k=5120, latency=0.01, kernel_source="fixture") for n in [384, 32320]]
            + [dict(gemm_dtype="fp8_block", m=1, n=1152, k=5120, latency=0.01, kernel_source="fixture")]
        ),
        data / "gemm_perf.parquet",
    )
    import yaml

    nccl_version = yaml.safe_load((tmp_path / f"{system}.yaml").read_text())["misc"]["nccl_version"]
    comm = tmp_path / f"data/{system}/comm/nccl/{nccl_version}"
    comm.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    nccl_dtype="half",
                    op_name="all_reduce",
                    num_gpus=4,
                    message_size=size,
                    latency=0.01,
                    kernel_source="torch.distributed.nccl.all_reduce",
                )
                for size in [1, 5120, 6144, 10000000]
            ]
        ),
        comm / "nccl_perf.parquet",
    )
    manifest = build_manifest(4, False)
    modules = {}
    for entry in manifest["phases"]["generation"]:
        component, geometry = entry["component"], entry["geometry"]
        modules[(component, geometry)] = dict(
            component=component,
            geometry=geometry,
            batch_size=1,
            prefix=0,
            x=129 if component == "attention" else 1,
            latency=0.01,
            kernel_source="test.fixture",
            measurement_scope="local_compute",
            source_sha256="a" * 64,
            config_sha256=manifest["config_sha256"],
            runtime_digest="sha256:" + "b" * 64,
            used_cuda_graph=False,
            sample_count=5,
            kv_seed_regime="real_kv" if component == "attention" else "n/a",
            execution_profile="full",
        )
    write_parquet(list(modules.values()), data / "dsv41_module_perf.parquet")
    db = PerfDatabase(
        system, "sglang", version, str(tmp_path), database_mode="SILICON", shared_layer=False, strict_provenance=False
    )
    assert "w4a16_mxfp4_humming" in db.supported_quant_mode["moe"]
    from aisimulate.sdk.task_v2 import Task

    monkeypatch.setattr(Task, "_try_load_role_database", lambda self, role: db)
    task = Task(
        model_path="deepseek-ai/DeepSeek-V4.1-Flash",
        system_name=system,
        backend_name="sglang",
        backend_version=version,
        total_gpus=4,
        database_mode="SILICON",
        moe_quant_mode=common.MoEQuantMode.w4a16_mxfp4_humming,
    )
    task._validate_database_quant_modes()
    assert task.moe_quant_mode is common.MoEQuantMode.w4a16_mxfp4_humming
    latencies = []
    for mode in (common.MoEQuantMode.w4a16_mxfp4_humming, common.MoEQuantMode.w4a16_mxfp4_cutlass):
        model = get_model(
            "deepseek-ai/DeepSeek-V4.1-Flash",
            ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1, moe_quant_mode=mode),
            "sglang",
        )
        handle = rust_engine_step._cached_engine_handle(model, db)
        latencies.append(handle.predict_decode_latency(1, 128))
    assert latencies[1] - latencies[0] == pytest.approx(40.0)
