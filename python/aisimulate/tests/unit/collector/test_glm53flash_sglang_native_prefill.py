# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY policy and native-evidence regressions; no GPU qualification."""

import json
from types import SimpleNamespace

import pytest
from collector.collect_glm53flash import native_command
from collector.fpm_forward.sglang_driver import validate_eager_args, validate_native_prefill_scope
from collector.glm53flash_validation import load_native

from .test_glm53flash_ops_evidence import native_fixture, put, put_lines

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("representation", ["cli", "dict", "native"])
@pytest.mark.parametrize("resolved", [False, True])
def test_prefill_keeps_declared_and_resolved_decode_graphs(representation, resolved):
    config = {"decode": {"backend": "full"}, "prefill": {"backend": "disabled"}}
    server = SimpleNamespace(
        cuda_graph_config=None,
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
        resolved_dict=lambda: {"cuda_graph_config": config},
    )
    if representation == "dict":
        server.cuda_graph_config = config
    elif representation == "native":
        server.cuda_graph_config = SimpleNamespace(**{key: SimpleNamespace(**value) for key, value in config.items()})
    validate_eager_args(server, resolved=resolved, native_prefill=True)
    with pytest.raises(ValueError):
        validate_eager_args(server, resolved=resolved)
    if resolved:
        # Raw attributes do not prove the policy after native resolution.
        config["decode"]["backend"] = "disabled"
        with pytest.raises(ValueError):
            validate_eager_args(server, resolved=True, native_prefill=True)


@pytest.mark.parametrize("purpose", ["fpm", "ops", "ops_holdout", "ops_graph", "ops_graph_holdout"])
@pytest.mark.parametrize("phase", ["prefill", "decode", "both"])
def test_new_scope_cannot_enable_module_events_for_graph_or_fpm_targets(purpose, phase):
    if purpose in ("ops", "ops_holdout") and phase == "prefill":
        validate_native_prefill_scope(purpose, phase, True)
    else:
        with pytest.raises(ValueError, match="Ops prefill target"):
            validate_native_prefill_scope(purpose, phase, True)


def test_explicit_prefill_command_preserves_native_workload_and_legacy_command(tmp_path):
    args = ("sglang", "/models/TEST_ONLY", "pinned", 2, "prefill", tmp_path, tmp_path / "text")
    legacy = native_command(*args)
    actual = native_command(*args, ops_execution_mode="native_eager_prefill", sglang_mem_fraction_static=0.82)
    assert actual[actual.index("--cuda-graph-backend-decode") + 1] == "full"
    assert actual[actual.index("--cuda-graph-backend-prefill") + 1] == "disabled"
    assert actual[actual.index("--mem-fraction-static") + 1] == "0.82"
    restored = actual.copy()
    restored.remove("--ops-native-prefill")
    pos = restored.index("--mem-fraction-static")
    del restored[pos : pos + 2]
    restored[restored.index("--cuda-graph-backend-decode") + 1] = "disabled"
    assert restored == legacy
    for backend, phase in [("sglang", "decode"), ("vllm", "prefill")]:
        with pytest.raises(ValueError):
            native_command(backend, *args[1:4], phase, *args[5:], ops_execution_mode="native_eager_prefill")


def prefill_fixture(tmp_path, role):
    run, root, all_records, _, _ = native_fixture(tmp_path, "calibration" if role == "control" else "holdout")
    run["role"] = role
    run["key"] = ("sglang", "fp8", 2, "prefill")
    run["spec"]["ops_execution_mode"] = "native_eager_prefill"
    run["points"][0].update(point_type="prefill", total_prefill_tokens=1)
    requests = json.loads((root / "requests.json").read_bytes())
    for request in requests["requests"].values():
        request["target_phase"] = "context"
    put(root / "requests.json", requests)
    for rank, records in all_records.items():
        for row in records:
            row["phase"] = "context"
            row["ops_instrumented"] = False
            row["requests"][0]["prompt_token_ids"] = [4, 5, 7]
        put_lines(root / f"forward-rank-{rank}.jsonl", records)
    for name in ("sglang-declared-config.json", "sglang-resolved-config.json"):
        config = json.loads((root / name).read_bytes())
        config["cuda_graph_config"]["decode"]["backend"] = "full"
        put(root / name, config)
    return run, root, all_records


@pytest.mark.parametrize("role", ["holdout", "control"])
@pytest.mark.parametrize("defect", [None, "disabled_decode", "actual_graph", "wrong_scope"])
def test_reader_binds_real_context_history_and_whole_forward_boundary(tmp_path, role, defect):
    run, root, records = prefill_fixture(tmp_path, role)
    if defect == "disabled_decode":
        path = root / "sglang-resolved-config.json"
        config = json.loads(path.read_bytes())
        config["cuda_graph_config"]["decode"]["backend"] = "disabled"
        put(path, config)
    elif defect == "actual_graph":
        records[0][-1].update(runtime_mode="FULL", used_cuda_graph=True)
        put_lines(root / "forward-rank-0.jsonl", records[0])
    elif defect == "wrong_scope":
        run["key"] = ("sglang", "fp8", 2, "decode")
    if defect:
        with pytest.raises(ValueError):
            load_native(run, root)
    else:
        result = load_native(run, root)
        assert result["values"] == {1: 5.0}
        assert result["timing_boundary"] == "embedding_to_logits_gpu_v1"
