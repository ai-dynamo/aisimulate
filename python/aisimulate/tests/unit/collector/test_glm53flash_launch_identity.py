# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY public launch/trace boundary; no GPU or accepted timing evidence."""

import dataclasses
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector import collect_glm53flash as producer
from collector import glm53flash_vllm_runtime as runtime
from collector.glm53flash_contract import build_model_manifest, build_run_provenance, validate_run_identity
from collector.glm53flash_runtime_identity import VLLM_TAIL_CANDIDATE

from .test_glm53flash_vllm_runtime import Event, Tensor

pytestmark = pytest.mark.unit


def _sglang_parser_run_id(command, env, monkeypatch):
    """Run the real driver parser, stopping at TEST_ONLY ServerArgs construction."""
    import sys

    from collector.fpm_forward import sglang_driver

    captured = []

    class StopBeforeServer(Exception):
        pass

    class ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            for flag in (
                "model-path",
                "revision",
                "tp-size",
                "context-length",
                "max-running-requests",
                "chunked-prefill-size",
                "kv-cache-dtype",
                "cuda-graph-backend-decode",
                "cuda-graph-backend-prefill",
            ):
                parser.add_argument("--" + flag)
            parser.add_argument("--disable-radix-cache", action="store_true")

        @staticmethod
        def from_cli_args(args):
            captured.append(args.run_id)
            raise StopBeforeServer

    with monkeypatch.context() as scoped:
        for key, value in env.items():
            if key in ("FPM_RUN_ID", "DYN_FPM_RUN_ID"):
                scoped.setenv(key, value)
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            scoped.delenv(key, raising=False)
        scoped.setitem(sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=ServerArgs))
        scoped.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object()))
        with pytest.raises(StopBeforeServer):
            sglang_driver.main(command[3:])
    return captured[0]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_public_native_launch_writes_authoritative_identity_and_complete_points(tmp_path, monkeypatch, phase, backend):
    from aisimulate_core.sdk.utils import _load_pre_downloaded_hf_config

    model = "zai-org/GLM-5.3-Flash"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_load_pre_downloaded_hf_config(model)))
    model_paths = tmp_path / "models.json"
    model_paths.write_text(json.dumps({model: str(checkpoint)}))
    corpus = tmp_path / "TEST_ONLY_corpus.txt"
    corpus.write_text("TEST_ONLY no tokenization or GPU execution")
    monkeypatch.setenv("AISIM_GLM53_MODEL_PATHS", str(model_paths))
    monkeypatch.setenv("AISIM_GLM53_INPUT_TEXT", str(corpus))
    monkeypatch.setenv("AISIM_GLM53_RUNTIME_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setenv("ETCD_ENDPOINTS", "TEST_ONLY_no_connection")
    monkeypatch.setenv("FPM_RUN_ID", "inherited_parent_must_not_alias_new_run")
    monkeypatch.setenv("DYN_FPM_RUN_ID", "inherited_driver_must_not_alias_new_run")
    native_version = VLLM_TAIL_CANDIDATE if backend == "vllm" else "0.5.20"
    monkeypatch.setattr(producer, "version", lambda backend: native_version)
    monkeypatch.setattr(producer, "verify_sources", lambda *args: None)
    points = [{"batch_size": 1, "total_kv_read_tokens": 128}]
    if phase == "prefill":
        points[0]["total_prefill_tokens"] = 4
    launches = []

    class StopBeforeNative(Exception):
        pass

    def execute(command, output, env, backend):
        provenance = json.loads((output / "provenance.json").read_bytes())
        actual = json.loads((output / "points.json").read_bytes())
        assert actual == {"schema_version": 1, "prefill": [], "decode": [], phase: points}
        assert command[command.index("--benchmark-points-file") + 1] == str(output / "points.json")
        if backend == "vllm":
            assert validate_run_identity(provenance, env["FPM_RUN_ID"]) == output.name
        else:
            from collector.fpm_forward.sglang_driver import read_ops_provenance
            from collector.glm53flash_contract import runtime_source_pins

            assert "run_id" not in provenance
            assert env["DYN_FPM_RUN_ID"] == env["FPM_RUN_ID"] == output.name
            assert _sglang_parser_run_id(command, env, monkeypatch) == output.name
            assert (
                read_ops_provenance(
                    output / "provenance.json",
                    raw_config=_load_pre_downloaded_hf_config(model),
                    checkpoint_revision=provenance["checkpoint_revision"],
                    runtime_audit={"status": "passed", "sources": runtime_source_pins(backend, native_version)},
                )
                == provenance
            )
        assert env["FPM_RUN_ID"] != "inherited_parent_must_not_alias_new_run"
        assert provenance["backend_version"] == native_version
        launches.append(env["FPM_RUN_ID"])
        raise StopBeforeNative

    monkeypatch.setattr(producer, "_execute", execute)
    for _ in range(2):
        with pytest.raises(StopBeforeNative):
            producer.run_native(backend, model, "fp8", 2, phase, points, perf_filename=str(tmp_path / "perf.csv"))
    assert launches[0] != launches[1]


@pytest.mark.parametrize("mode", ["NONE", "FULL", "PIECEWISE"])
def test_public_render_launch_provenance_reaches_original_forward_and_trace(tmp_path, monkeypatch, mode):
    from aisimulate.generator.api import generate_config_from_input_dict, generate_from_request
    from aisimulate.generator.request import from_legacy_params
    from collector.glm53flash_graph_nodes import trace_forward_identity
    from collector.glm53flash_vllm_graph_ops import NativeVllmGraphExecution
    from collector.glm53flash_vllm_none_activity import NativeNoneExecution

    run_id = "TEST_ONLY_original_public_launch_" + mode
    provenance = build_run_provenance(
        build_model_manifest("vllm", "fp8", 2, VLLM_TAIL_CANDIDATE), "sha256:" + "a" * 64, run_id
    )
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    params = generate_config_from_input_dict(
        {
            "ServiceConfig": {
                "model_path": "/models/TEST_ONLY",
                "served_model_name": "TEST_ONLY",
                "include_frontend": False,
            },
            "DynConfig": {"mode": "agg"},
            "Workers": {
                "agg": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                    "max_batch_size": 1,
                }
            },
            "SlaConfig": {"isl": 2, "osl": 1},
            "K8sConfig": {
                "extra_env": [
                    {"name": "FPM_RUN_ID", "value": run_id},
                    {"name": "AISIM_GLM53_PROVENANCE", "value": str(provenance_path)},
                ]
            },
        },
        backend="vllm",
    )
    params["params"]["agg"]["extra_cli_args"] = ["--benchmark-mode", "decode" if mode == "FULL" else "prefill"]
    request = from_legacy_params(params, "vllm")
    request = dataclasses.replace(request, emit=dataclasses.replace(request.emit, deployment_target="fpm"))
    generate_from_request(request, output_dir=str(tmp_path / "render"))
    script = (tmp_path / "render/run.sh").read_text()
    exported = {}
    for line in script.splitlines():
        if line.startswith("export FPM_RUN_ID=") or line.startswith("export AISIM_GLM53_PROVENANCE="):
            key, value = shlex.split(line)[1].split("=", 1)
            exported[key] = value
            monkeypatch.setenv(key, value)
    original = json.loads(Path(exported["AISIM_GLM53_PROVENANCE"]).read_bytes())
    assert validate_run_identity(original, exported["FPM_RUN_ID"]) == run_id
    phase, prefix, query = ("generation", 2, 1) if mode == "FULL" else ("context", 0, 2)
    prompt = [1, 2]

    class PromptBuffer:
        def __getitem__(self, key):
            return Tensor(prompt[key[1]])

    state = runtime._TraceState.__new__(runtime._TraceState)
    state.runner = SimpleNamespace(
        max_model_len=131079,
        req_states=SimpleNamespace(
            req_id_to_index={"request": 0},
            prompt_len=SimpleNamespace(np=[2]),
            all_token_ids=SimpleNamespace(gpu=PromptBuffer()),
        ),
    )
    state.provenance, state.rank, state.counter = original, 0, 0
    state.previous = {"request": {"computed_tokens": 2, "tokens": prompt, "forward_id": "original_seed"}}
    state.matched, state.layout, state.layout_sha256 = set(), {"admitted": True}, "b" * 64
    state.observer, state.graph_execution = None, None
    state.piecewise_replay, state.serving_none = mode == "PIECEWISE", mode == "NONE"
    state.serving_none_measured, state.none_identity_sha256 = mode == "NONE", "d" * 64
    state.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: 1, Event=lambda **kwargs: Event()))
    coords = {
        "phase": phase,
        "batch_size": 1,
        "request_ids": ["request"],
        "query_lengths": [query],
        "prefix_lengths": [prefix],
        "total_new_tokens": query,
        "total_past_kv_tokens": prefix,
        "native_dispatch": {
            "descriptor": {"cg_mode": mode, "num_tokens": query, "num_reqs": 1},
            "physical_tokens": query,
            "physical_requests": 1,
        },
    }
    monkeypatch.setattr(runtime, "native_v2_coordinates", lambda *args, **kwargs: coords)
    mapping = {
        "request_set": run_id + "-TEST_ONLY_scheduler_uuid",
        "dataset_role": "calibration",
        "corpus_sha256": "c" * 64,
        "requests": {
            "request": {
                "benchmark_id": 1,
                "repetition": 4,
                "sampling_role": "warmup",
                "target_phase": phase,
                "target_query": query,
                "target_prefix": prefix,
                "target_batch_size": 1,
            }
        },
    }
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(mapping))
    monkeypatch.setenv("AISIM_GLM53_REQUEST_MANIFEST", str(requests))
    record, _ = state.before(
        object(),
        Tensor([3] if mode == "FULL" else prompt),
        None,
        native_batch=object(),
        graph_policy={"TEST_ONLY": True},
        native_descriptor=object(),
    )
    assert record["run_id"] == run_id and record["request_set"] != run_id
    profiler = SimpleNamespace(export_chrome_trace=lambda name: Path(name).write_text('{"traceEvents":[]}'))
    if mode == "NONE":
        execution = NativeNoneExecution.__new__(NativeNoneExecution)
        execution.output, execution.active = tmp_path, {"record": record}
        path, _ = execution._save_trace(profiler, [], failed=False)
    else:
        execution = NativeVllmGraphExecution.__new__(NativeVllmGraphExecution)
        execution.output, execution.rank = tmp_path, 0
        active = {"record": record, "profiler": profiler, "piecewise_registry": None}
        execution._save_trace(active, failed=False)
        path = active["path"]
    assert json.loads(path.read_bytes())["aisim_native_forward"] == trace_forward_identity(record)
    assert provenance_path.read_text() == json.dumps(provenance)


@pytest.mark.parametrize("install", [runtime.install, runtime.install_v2])
@pytest.mark.parametrize(
    "run_id,launched",
    [
        (None, "run"),
        ("", "run"),
        (True, "run"),
        (7, "run"),
        (" run", " run"),
        ("run", None),
        ("run", ""),
        ("run", "other"),
    ],
)
def test_invalid_identity_rejects_before_native_import_or_hooks(tmp_path, monkeypatch, install, run_id, launched):
    import importlib.metadata
    import sys

    path = tmp_path / "provenance.json"
    payload = {"backend_version": VLLM_TAIL_CANDIDATE}
    if run_id is not None:
        payload["run_id"] = run_id
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("AISIM_GLM53_PROVENANCE", str(path))
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", "ops")
    if launched is None:
        monkeypatch.delenv("FPM_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("FPM_RUN_ID", launched)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: VLLM_TAIL_CANDIDATE)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", None)
    with pytest.raises(ValueError, match="provenance.run_id matching FPM_RUN_ID"):
        install()
    assert json.loads(path.read_bytes()) == payload


def test_legacy_runner_graph_handoff_still_does_not_install_eager_hooks(monkeypatch):
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", "ops_graph")
    monkeypatch.delenv("AISIM_GLM53_PROVENANCE", raising=False)
    assert runtime.install() is None
