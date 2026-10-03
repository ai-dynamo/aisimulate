# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY: explicit GLM native CUDA-graph identity, without GPU claims."""

import argparse
import hashlib
import json
import shlex
from dataclasses import replace
from types import SimpleNamespace

import pytest
from collector.fpm_forward import glm53flash_validation as validation
from collector.fpm_forward import graph_policy
from collector.fpm_forward.cli import _parser
from collector.fpm_forward.config import FPMCollectionOptions, reject_fpm_arguments_without_fpm
from collector.fpm_forward.planner import build_collection_plan
from collector.fpm_forward.runner import _cell_generator_overrides, _render_cell
from collector.fpm_forward.sglang_artifact import file_receipt, validate_sglang_repetitions
from collector.fpm_forward.shards import make_shards

from tests.unit.collector import test_fpm_glm53flash_sglang_artifact as sglang_fixture
from tests.unit.collector.test_fpm_glm53flash_planning import plan
from tests.unit.collector.test_glm53flash_validation import campaign, write_plan  # noqa: F401

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"
VLLM_DEFAULT = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
SGLANG_PREFILL_BUCKETS = [
    *range(4, 33, 4),
    *range(48, 257, 16),
    *range(288, 513, 32),
    *range(576, 1025, 64),
    *range(1280, 4097, 256),
    *range(4608, 8193, 512),
]
CAPTURE = tuple(sorted(set(VLLM_DEFAULT) | set(SGLANG_PREFILL_BUCKETS)))


def rebuild(source, **options):
    return build_collection_plan(
        backend=source.backend,
        model_path=source.model_path,
        system=source.system,
        selected_ops=set(),
        options=replace(source.options, **options),
    )


def engine_argv(source, cell, directory):
    directory.mkdir()
    _render_cell(source, cell, directory, {"generator_dynamo_version": "1.3.0"})
    command = next(
        line for line in (directory / "run.sh").read_text().splitlines() if line.startswith("engine_command=(")
    )
    return shlex.split(command.removeprefix("engine_command=(").removesuffix(")"))


def test_campaign_capture_list_matches_the_qualified_62_sizes():
    assert len(SGLANG_PREFILL_BUCKETS) == 58 and SGLANG_PREFILL_BUCKETS[-1] == 8192
    assert len(CAPTURE) == 62 and CAPTURE[-1] == 8192


def test_cli_options_are_public_fpm_only_and_validated():
    args = _parser().parse_args(
        [
            "--gpu",
            "gb300",
            "--fpm-max-gpus",
            "4",
            "--vllm-cudagraph-capture-sizes",
            "1",
            "2",
            "8192",
            "--sglang-cuda-graph-backend-prefill",
            "breakable",
            "--sglang-cuda-graph-max-bs-prefill",
            "8192",
        ]
    )
    options = FPMCollectionOptions.from_args(args)
    assert options.vllm_cudagraph_capture_sizes == (1, 2, 8192)
    assert (options.sglang_cuda_graph_backend_prefill, options.sglang_cuda_graph_max_bs_prefill) == (
        "breakable",
        8192,
    )
    for name, value in (
        ("vllm_cudagraph_capture_sizes", [1, 2]),
        ("sglang_cuda_graph_backend_prefill", "breakable"),
        ("sglang_cuda_graph_max_bs_prefill", 8192),
    ):
        with pytest.raises(ValueError, match="FPM-only arguments"):
            reject_fpm_arguments_without_fpm(argparse.Namespace(ops=["gemm"], **{name: value}))
    with pytest.raises(SystemExit):
        _parser().parse_args(["--gpu", "gb300", "--sglang-cuda-graph-backend-prefill", "full"])


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"vllm_cudagraph_capture_sizes": (2, 1)}, "increasing"),
        ({"vllm_cudagraph_capture_sizes": (1, 1)}, "increasing"),
        ({"vllm_cudagraph_capture_sizes": ()}, "increasing"),
        ({"vllm_cudagraph_capture_sizes": (0, 1)}, "increasing"),
        ({"sglang_cuda_graph_backend_prefill": "breakable"}, "together"),
        ({"sglang_cuda_graph_max_bs_prefill": 8192}, "together"),
        ({"sglang_cuda_graph_backend_prefill": "full", "sglang_cuda_graph_max_bs_prefill": 8192}, "one of"),
        ({"sglang_cuda_graph_backend_prefill": "breakable", "sglang_cuda_graph_max_bs_prefill": 0}, "positive"),
    ],
)
def test_programmatic_options_fail_closed(fields, message):
    options = FPMCollectionOptions.from_args(argparse.Namespace(fpm_max_gpus=4))
    with pytest.raises(ValueError, match=message):
        replace(options, **fields)


def test_vllm_omission_preserves_plan_and_explicit_sizes_are_frozen_policy(tmp_path):
    original = plan(tmp_path, "vllm", MODEL)
    assert rebuild(original, vllm_cudagraph_capture_sizes=None).to_dict() == original.to_dict()
    assert "vllm_cudagraph_capture_sizes" not in original.to_dict()["options"]
    graph = rebuild(original, vllm_cudagraph_capture_sizes=CAPTURE, shard_token_budget=1_000_000)
    assert graph.to_dict()["options"]["vllm_cudagraph_capture_sizes"] == list(CAPTURE)
    assert not {cell.cell_id for cell in original.cells} & {cell.cell_id for cell in graph.cells}
    (policy,) = graph.backend_policies
    assert policy.policy_id == "explicit-cudagraph_capture_sizes=" + graph_policy.capture_sizes_label(CAPTURE)
    assert policy.expected_markers == {graph_policy.VLLM_CAPTURE_SIZES_MARKER: str(list(CAPTURE))}
    assert graph_policy.declared_capture_sizes(policy.expected_markers) == CAPTURE
    # Explicit capture sizes are runtime identity, not a backend identity column.
    assert policy.aic_fields == original.backend_policies[0].aic_fields
    shards = make_shards(graph)
    assert shards and all(
        shard.plan.cells[0].backend_policy.expected_markers == policy.expected_markers for shard in shards
    )
    for cell in graph.cells:
        argv = engine_argv(graph, cell, tmp_path / cell.cell_id)
        start = argv.index(graph_policy.VLLM_CAPTURE_SIZES_FLAG)
        assert argv.count(graph_policy.VLLM_CAPTURE_SIZES_FLAG) == 1
        assert argv[start + 1 : start + 1 + len(CAPTURE)] == [str(size) for size in CAPTURE]
        baseline = engine_argv(
            original,
            original.cells[0] if cell.workload_kind == "prefill" else original.cells[1],
            tmp_path / ("base-" + cell.cell_id),
        )
        assert graph_policy.VLLM_CAPTURE_SIZES_FLAG not in baseline


def test_explicit_graph_requests_are_backend_and_model_scoped(tmp_path):
    vllm = plan(tmp_path, "vllm", MODEL)
    sglang = plan(tmp_path, "sglang", MODEL)
    with pytest.raises(ValueError, match="requires backend=sglang"):
        rebuild(vllm, sglang_cuda_graph_backend_prefill="breakable", sglang_cuda_graph_max_bs_prefill=8192)
    with pytest.raises(ValueError, match="requires backend=vllm"):
        rebuild(sglang, vllm_cudagraph_capture_sizes=CAPTURE)


def test_sglang_omission_preserves_plan_and_explicit_prefill_graph_renders_once(tmp_path):
    original = plan(tmp_path, "sglang", MODEL)
    assert (
        rebuild(original, sglang_cuda_graph_backend_prefill=None, sglang_cuda_graph_max_bs_prefill=None).to_dict()
        == original.to_dict()
    )
    for cell in original.cells:
        assert "sglang_cuda_graph_backend_prefill" not in cell.to_dict()
        args = _cell_generator_overrides(original, cell, {})["params"]["agg"]["extra_cli_args"]
        assert "--cuda-graph-backend-prefill" not in args
    graph = rebuild(
        original,
        sglang_cuda_graph_backend_prefill="breakable",
        sglang_cuda_graph_max_bs_prefill=8192,
        shard_token_budget=1_000_000,
    )
    other = rebuild(original, sglang_cuda_graph_backend_prefill="breakable", sglang_cuda_graph_max_bs_prefill=4096)
    assert len({original.sha256, graph.sha256, other.sha256}) == 3
    assert not {c.cell_id for c in original.cells} & {c.cell_id for c in graph.cells}
    assert not {c.cell_id for c in other.cells} & {c.cell_id for c in graph.cells}
    options = graph.to_dict()["options"]
    assert (options["sglang_cuda_graph_backend_prefill"], options["sglang_cuda_graph_max_bs_prefill"]) == (
        "breakable",
        8192,
    )
    for shard in make_shards(graph):
        assert shard.plan.cells[0].sglang_cuda_graph_backend_prefill == "breakable"
        assert shard.plan.cells[0].sglang_cuda_graph_max_bs_prefill == 8192
    for cell in graph.cells:
        assert cell.to_dict()["sglang_cuda_graph_max_bs_prefill"] == 8192
        argv = engine_argv(graph, cell, tmp_path / cell.cell_id)
        assert argv.count("--cuda-graph-backend-prefill") == argv.count("--cuda-graph-max-bs-prefill") == 1
        assert argv[argv.index("--cuda-graph-backend-prefill") + 1] == "breakable"
        assert argv[argv.index("--cuda-graph-max-bs-prefill") + 1] == "8192"
    with pytest.raises(ValueError, match="prefill CUDA graph differs between frozen plan and cell"):
        _cell_generator_overrides(graph, original.cells[0], {})


def write_native_configs(tmp_path, payload, *, declared=None, resolved=None):
    evidence = payload["input_provenance"]["native_forward_manifest"]
    for name, update in (("declared_config", declared), ("resolved_config", resolved)):
        path = tmp_path / evidence[name]["file"]
        config = json.loads(path.read_text())
        config.update(update or {})
        path.write_text(json.dumps(config))
        evidence[name] = file_receipt(path)


BREAKABLE = {"cuda_graph_backend_prefill": "breakable", "cuda_graph_max_bs_prefill": 8192}
RESOLVED_BREAKABLE = {
    **BREAKABLE,
    "cuda_graph_config": {
        "decode": {"backend": "full", "bs": list(range(1, 33)), "max_bs": 32},
        "prefill": {"backend": "breakable", "bs": SGLANG_PREFILL_BUCKETS, "max_bs": 8192},
    },
}


def test_sglang_native_receipts_must_carry_the_requested_prefill_graph(tmp_path):
    cell, payload = sglang_fixture.artifact(tmp_path)
    # Legacy cells keep the native default, including the KDA auto-disable.
    validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")
    cell.sglang_cuda_graph_backend_prefill, cell.sglang_cuda_graph_max_bs_prefill = "breakable", 8192
    with pytest.raises(ValueError, match="declared prefill CUDA graph differs"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")
    write_native_configs(tmp_path, payload, declared=BREAKABLE, resolved=RESOLVED_BREAKABLE)
    validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize(
    "resolved, message",
    [
        ({"cuda_graph_max_bs_prefill": 4096}, "resolved prefill CUDA graph differs"),
        ({"disable_prefill_cuda_graph": True}, "disables the requested prefill"),
        (
            {"cuda_graph_config": {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}}},
            "resolved prefill CUDA graph differs from the frozen request",
        ),
        (
            {
                "cuda_graph_config": {
                    "decode": {"backend": "full"},
                    "prefill": {"backend": "breakable", "bs": [4, 8], "max_bs": 8192},
                }
            },
            "resolved prefill CUDA graph differs from the frozen request",
        ),
    ],
)
def test_sha_valid_but_different_native_prefill_graph_is_rejected(tmp_path, resolved, message):
    cell, payload = sglang_fixture.artifact(tmp_path)
    cell.sglang_cuda_graph_backend_prefill, cell.sglang_cuda_graph_max_bs_prefill = "breakable", 8192
    write_native_configs(tmp_path, payload, declared=BREAKABLE, resolved={**RESOLVED_BREAKABLE, **resolved})
    with pytest.raises(ValueError, match=message):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


def test_vllm_native_capture_block_and_resolved_marker_checks(tmp_path):
    sizes = (1, 2, 8192)
    graph_policy.validate_vllm_native_capture(None, {}, tmp_path)
    native = {"capture_sizes": [1, 2, 8192], "prefill_capture_sizes": [1, 2, 8192], "max_capture_size": 8192}
    graph_policy.validate_vllm_native_capture(sizes, {"cudagraph": native}, tmp_path)
    for broken in ({}, {**native, "prefill_capture_sizes": [1, 2]}, {**native, "max_capture_size": 64}):
        with pytest.raises(ValueError, match="CUDA-graph"):
            graph_policy.validate_vllm_native_capture(sizes, {"cudagraph": broken} if broken else {}, tmp_path)
    _, markers = graph_policy.vllm_capture_policy(sizes)
    with pytest.raises(ValueError, match="requires resolved-config evidence"):
        graph_policy.validate_resolved_markers(markers, "policy", tmp_path)
    resolved = tmp_path / "node0000" / "resolved-config.json"
    resolved.parent.mkdir()
    resolved.write_text(json.dumps({"config": {"engine_args": {"cudagraph_capture_sizes": [1, 2, 8192]}}}))
    graph_policy.validate_resolved_markers(markers, "policy", tmp_path)
    resolved.write_text(json.dumps({"config": {"engine_args": {"cudagraph_capture_sizes": VLLM_DEFAULT}}}))
    with pytest.raises(ValueError, match="backend marker mismatch"):
        graph_policy.validate_resolved_markers(markers, "policy", tmp_path)
    with pytest.raises(ValueError, match="not canonical"):
        graph_policy.declared_capture_sizes({graph_policy.VLLM_CAPTURE_SIZES_MARKER: "[1,2]"})


def frozen(tmp_path, spec, update_options=None, update_cell=None):
    path = tmp_path / spec["plan"]["path"]
    payload = json.loads(path.read_text())
    payload["options"].update(update_options or {})
    payload["cells"][0].update(update_cell or {})
    path.write_text(json.dumps(payload))
    spec["plan"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return spec


def vllm_policy(sizes):
    args, markers = graph_policy.vllm_capture_policy(sizes)
    return {
        "policy_id": "explicit-cudagraph_capture_sizes=" + graph_policy.capture_sizes_label(sizes),
        "generator_overrides": {"params": {"agg": {"extra_cli_args": args}}},
        "expected_markers": markers,
        "aic_fields": {},
        "admission_reason": "explicitly pinned backend identity",
    }


@pytest.mark.parametrize("cell_value", [None, 4096, 8192])
def test_acceptance_crossbinds_sglang_prefill_graph(tmp_path, cell_value):
    spec = write_plan(tmp_path, ("sglang", "fp8", 2, "prefill"), "holdout")
    frozen(
        tmp_path,
        spec,
        {"sglang_cuda_graph_backend_prefill": "breakable", "sglang_cuda_graph_max_bs_prefill": 8192},
        {"sglang_cuda_graph_backend_prefill": "breakable", "sglang_cuda_graph_max_bs_prefill": cell_value}
        if cell_value
        else None,
    )
    if cell_value == 8192:
        run = validation._plan_run(spec, tmp_path, "holdout")
        assert run["runtime_cell"].sglang_cuda_graph_backend_prefill == "breakable"
        assert run["runtime_cell"].sglang_cuda_graph_max_bs_prefill == 8192
    else:
        with pytest.raises(ValueError, match="prefill CUDA graph differs between frozen plan and cell"):
            validation._plan_run(spec, tmp_path, "holdout")


def test_acceptance_crossbinds_vllm_capture_policy_and_checks_holdout_markers(tmp_path, monkeypatch):
    spec = write_plan(tmp_path, validation.REQUIRED[0], "holdout")
    frozen(tmp_path, spec, {"vllm_cudagraph_capture_sizes": [1, 2]}, {"backend_policy": vllm_policy(CAPTURE)})
    with pytest.raises(ValueError, match="capture sizes differ between frozen plan and backend policy"):
        validation._plan_run(spec, tmp_path, "holdout")
    frozen(tmp_path, spec, {"vllm_cudagraph_capture_sizes": list(CAPTURE)})
    run = validation._plan_run(spec, tmp_path, "holdout")
    root = tmp_path / spec["raw_root"]
    (root / "node0000").mkdir(parents=True)
    (root / "benchmark.token-streams.jsonl").write_text(json.dumps({"requests": [{"request_id": "r"}]}) + "\n")

    def reader(cell, path, **kwargs):
        return SimpleNamespace(
            points=[SimpleNamespace(point=point, rank_wall_times=((0, 0.012),)) for point in run["points"]],
            input_provenance={
                "text_sha256": "b" * 64,
                "tokenizer_revision": validation.MODEL_REVISIONS[run["plan"]["model_path"]],
            },
            runtime_run_id="run",
            runtime_grid_digest="grid",
            backend_version="0.30.0",
        )

    monkeypatch.setattr(validation, "validate_native_collection", reader)
    with pytest.raises(ValueError, match="requires resolved-config evidence"):
        validation._native_run(run, tmp_path)
    resolved = root / "node0000" / "resolved-config.json"
    resolved.write_text(json.dumps({"config": {"engine_args": {"cudagraph_capture_sizes": VLLM_DEFAULT}}}))
    with pytest.raises(ValueError, match="backend marker mismatch"):
        validation._native_run(run, tmp_path)
    resolved.write_text(json.dumps({"config": {"engine_args": {"cudagraph_capture_sizes": list(CAPTURE)}}}))
    assert validation._native_run(run, tmp_path)["values"] == {1: 12, 2: 12, 3: 12}


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_graph_calibration_cannot_validate_default_config_holdout(campaign, tmp_path, backend):  # noqa: F811
    index = validation.REQUIRED.index((backend, "fp8", 2, "prefill"))
    calibration = campaign["entries"][index]["calibration"]
    if backend == "vllm":
        frozen(
            tmp_path,
            calibration,
            {"vllm_cudagraph_capture_sizes": list(CAPTURE)},
            {"backend_policy": vllm_policy(CAPTURE)},
        )
    else:
        graph = {"sglang_cuda_graph_backend_prefill": "breakable", "sglang_cuda_graph_max_bs_prefill": 8192}
        frozen(tmp_path, calibration, graph, graph)
    with pytest.raises(ValueError, match="calibration/holdout frozen engine/graph settings differ"):
        validation.evaluate(campaign, tmp_path)
