# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY CPU proofs; finite sampling bias has no GPU guarantee here."""

import copy
import functools
import json
from types import SimpleNamespace

import pytest

from collector import glm53flash_graph_export as graph
from collector import glm53flash_sglang_control as control
from collector import glm53flash_validation as native
from collector.glm53flash_contract import sha256_json
from collector.glm53flash_graph_callbacks import resolve_registry
from collector.glm53flash_jsonl import file_sha256, iter_records

from .test_glm53flash_graph_export import control_fixture, fixture
from .test_glm53flash_ops_evidence import put, put_lines

pytestmark = pytest.mark.unit


def producer():
    return {
        "schema": control.SUBMISSION,
        "collector_sources": dict.fromkeys(control.REQUIRED_COLLECTOR_SOURCES, "a" * 64),
        "native_sampling_sources": control.SOURCE_PINS,
    }


def worker():
    return {
        "schema": control.SUBMISSION,
        "native_sampling_sources": control.SOURCE_PINS,
        "methods": {
            name: {
                "source": source,
                "source_sha256": control.SOURCE_PINS[source],
                "qualname": name,
                "wrapped": False,
                "signature": "(TEST_ONLY)",
            }
            for name, source in control.METHOD_SOURCES.items()
        },
    }


def add_submission(root, run, reference_path=None):
    """Add authored public-call evidence to the real complete reader fixture."""
    put(root / control.PRODUCER, producer())
    reference = None
    if reference_path is not None:
        (root / control.REFERENCE).write_bytes(reference_path.read_bytes())
        reference = control.load_reference(reference_path, file_sha256(reference_path), producer())
    value = json.loads((root / "sglang-provenance.json").read_bytes())
    value["native_request_submission"] = control.SUBMISSION
    value["native_control_reference_sha256"] = file_sha256(reference_path) if reference_path else None
    put(root / "sglang-provenance.json", value)
    for rank in range(run["key"][2]):
        put(root / f"sampling-source-rank-{rank}.json", worker())
        registry_sha = None
        if run["role"] == "calibration":
            path = root / f"capture-source-nodes-rank-{rank}.jsonl"
            source = next(iter_records(path))
            source["provenance"]["native_request_submission"] = control.SUBMISSION
            source["provenance"]["native_control_reference_sha256"] = None
            put_lines(path, [source])
            callback_path = root / f"graph-clones-rank-{rank}-capture-0.json"
            registry = resolve_registry(source, json.loads(callback_path.read_bytes()))
            registry["instantiation_receipt"] = {"file": callback_path.name, "sha256": file_sha256(callback_path)}
            put_lines(root / f"capture-nodes-rank-{rank}.jsonl", [registry])
            registry_sha = graph._semantic_sha(registry)
        for stem in ("forward", "graph-forward"):
            path = root / f"{stem}-rank-{rank}.jsonl"
            rows = list(iter_records(path))
            for row in rows:
                row["native_request_submission"] = control.SUBMISSION
                row["native_control_reference_sha256"] = file_sha256(reference_path) if reference_path else None
                if stem == "graph-forward" and registry_sha is not None:
                    row["capture_registry_sha256"] = registry_sha
                if rank == 0 and stem == "forward" and row["stage"] == "measure":
                    inputs = [request["prompt_token_ids"] for request in row["requests"]]
                    params = control.sampling_parameters(reference, row["benchmark_id"], row["repetition"], inputs)
                    control.append_submission(
                        root,
                        row["benchmark_id"],
                        row["repetition"],
                        row["request_ids"],
                        inputs,
                        params,
                        file_sha256(reference_path) if reference_path else None,
                    )
            put_lines(path, rows)


def pair(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    add_submission(root, run)
    reference = tmp_path / "reference.json"
    digest = control.freeze_reference(root, run, reference)
    assert digest == file_sha256(reference)
    independent = control_fixture(tmp_path, monkeypatch)
    add_submission(independent["control_root"], independent["control_run"], reference)
    return run, root, reference, independent


def test_original_raw_reference_to_independent_control_and_published_closure(tmp_path, monkeypatch):
    run, root, reference, independent = pair(tmp_path, monkeypatch)
    original = json.loads(reference.read_bytes())
    assert len(original["targets"]) == 15
    assert set(original["source_files"]) >= {
        "retained-rank-0.jsonl",
        "forward-rank-1.jsonl",
        control.JOURNAL,
        "requests.json",
    }
    assert original["targets"][0]["requests"][0]["native_query_token_ids"] == [7]
    assert len(original["targets"][0]["source_forwards"]) == 2
    result = graph.export_graph(root, run, root / graph.BASENAME, **independent)
    assert result["rows"] == 3
    receipt = native.load_native(run, root)
    native.bind_calibration([root / graph.BASENAME], run, receipt)
    raw = next(iter_records(independent["control_root"] / control.JOURNAL))
    assert raw["sampling_params"] == [
        {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True, "logit_bias": {"7": 100.0}}
    ]
    reference.write_text(reference.read_text() + " ")
    with pytest.raises(ValueError, match="original bytes changed"):
        control.load_reference(reference, sha256_json(original), producer())


@pytest.mark.parametrize(
    "defect",
    [
        "collector_source",
        "native_source",
        "method_source",
        "method_name",
        "wrapped",
        "stripped_journal",
        "stripped_all",
        "undeclared",
    ],
)
def test_loaded_source_and_producer_declaration_cannot_be_substituted(tmp_path, monkeypatch, defect):
    run, root = fixture(tmp_path, monkeypatch)
    add_submission(root, run)
    if defect in ("collector_source", "native_source"):
        path = root / control.PRODUCER
        value = json.loads(path.read_bytes())
        if defect == "collector_source":
            value["collector_sources"].pop("glm53flash_sglang_control.py")
        else:
            value["native_sampling_sources"]["srt/layers/sampler.py"] = "f" * 64
        put(path, value)
    elif defect in ("method_source", "method_name", "wrapped"):
        path = root / "sampling-source-rank-0.json"
        value = json.loads(path.read_bytes())
        method = value["methods"]["Sampler.forward"]
        if defect == "method_source":
            method.update(source="unknown.py", source_sha256=None)
        elif defect == "method_name":
            method["qualname"] = "Substitute.forward"
        else:
            method["wrapped"] = True
        put(path, value)
    elif defect == "stripped_journal":
        (root / control.JOURNAL).unlink()
    elif defect == "stripped_all":
        for name in (control.JOURNAL, control.PRODUCER, "sampling-source-rank-0.json", "sampling-source-rank-1.json"):
            (root / name).unlink()
    else:
        # Remove the declaration from both source and current captures so the
        # graph join is otherwise internally coherent.
        path = root / "sglang-provenance.json"
        value = json.loads(path.read_bytes())
        del value["native_request_submission"]
        put(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        graph.read_graph_run(root, run)


@pytest.mark.parametrize(
    "defect",
    [
        "bias_nan",
        "bias_bool",
        "bias_changed",
        "missing_rep",
        "duplicate_rep",
        "tp_missing",
        "query",
        "prompt",
        "history",
        "slot",
        "request_reuse",
        "producer",
        "source_hash",
    ],
)
def test_reference_is_exact_complete_source_input_not_a_token_override(tmp_path, monkeypatch, defect):
    run, root = fixture(tmp_path, monkeypatch)
    add_submission(root, run)
    path = tmp_path / "reference.json"
    control.freeze_reference(root, run, path)
    value = json.loads(path.read_bytes())
    request = value["targets"][0]["requests"][0]
    if defect.startswith("bias"):
        value["logit_bias"] = {"bias_nan": float("nan"), "bias_bool": True, "bias_changed": 101.0}[defect]
    elif defect == "missing_rep":
        value["targets"].pop()
    elif defect == "duplicate_rep":
        value["targets"].append(copy.deepcopy(value["targets"][0]))
    elif defect == "tp_missing":
        value["targets"][0]["source_forwards"].pop()
    elif defect == "query":
        request["native_query_token_ids"] = [8]
    elif defect == "prompt":
        request["prompt_token_ids"] = [4, 6]
    elif defect == "history":
        request["computed_tokens_before"] = False
    elif defect == "slot":
        request["submitted_position"] = True
    elif defect == "request_reuse":
        value["targets"][1]["requests"][0]["source_request_id"] = request["source_request_id"]
    elif defect == "producer":
        value["producer"]["collector_sources"]["glm53flash_sglang_control.py"] = "b" * 64
    else:
        value["source_files"]["requests.json"] = "not-a-hash"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        control.load_reference(path, file_sha256(path), producer())


@pytest.mark.parametrize("defect", ["missing", "duplicate", "params", "reference_sha", "prompt", "request"])
def test_actual_public_submission_remains_required(tmp_path, monkeypatch, defect):
    run, root, _, independent = pair(tmp_path, monkeypatch)
    path = independent["control_root"] / control.JOURNAL
    rows = list(iter_records(path))
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "params":
        rows[0]["sampling_params"][0]["logit_bias"]["7"] = 1000.0
    elif defect == "reference_sha":
        rows[0]["control_reference_sha256"] = "f" * 64
    elif defect == "prompt":
        rows[0]["input_ids"][0][0] = 10
    else:
        rows[0]["request_ids"][0] = "invented-request"
    put_lines(path, rows)
    with pytest.raises(ValueError):
        graph.export_graph(root, run, root / graph.BASENAME, **independent)


def change_native_tokens(root, *, query=None, terminal=None):
    for rank in range(2):
        for stem in ("forward", "graph-forward"):
            path = root / f"{stem}-rank-{rank}.jsonl"
            rows = list(iter_records(path))
            for row in rows:
                for request in row["requests"]:
                    if row["stage"] == "measure":
                        if terminal is not None:
                            request["sampled_token_id"] = terminal
                        if query is not None:
                            request["native_query_token_ids"] = [query]
                            request["input_tokens_sha256"] = sha256_json(request["prompt_token_ids"] + [query])
                    elif query is not None:
                        request["sampled_token_id"] = query
            put_lines(path, rows)


def test_terminal_output_only_is_not_measured_input_but_actual_query_still_is(tmp_path, monkeypatch):
    run, root, _, independent = pair(tmp_path, monkeypatch)
    other = independent["control_root"]
    change_native_tokens(other, terminal=123)
    # All TP ranks agree and each original request history remains valid.
    native.load_native(independent["control_run"], other)
    graph.export_graph(root, run, root / graph.BASENAME, **independent)
    change_native_tokens(other, query=8)
    with pytest.raises(ValueError, match="did not reproduce the exact independent model input"):
        native.load_native(independent["control_run"], other)


def test_legacy_output_projection_preserves_actual_input_rejection(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    independent = control_fixture(tmp_path, monkeypatch)
    other = independent["control_root"]
    change_native_tokens(other, terminal=123)
    graph.profile_control(root, graph.read_graph_run(root, run), other, independent["control_run"])
    change_native_tokens(other, query=8)
    native.load_native(independent["control_run"], other)  # Valid independent native chain.
    with pytest.raises(ValueError, match="actual model inputs"):
        graph.profile_control(root, graph.read_graph_run(root, run), other, independent["control_run"])


def test_source_reference_is_rederived_against_original_bytes(tmp_path, monkeypatch):
    run, root, _, independent = pair(tmp_path, monkeypatch)
    proof = graph.read_graph_run(root, run)
    observed = graph.read_graph_run(independent["control_root"], independent["control_run"])
    control.verify_pair(root, proof, observed)
    path = root / "retained-rank-0.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="another source calibration"):
        control.verify_pair(root, proof, observed)


@pytest.mark.parametrize(
    "purpose,phase,role",
    [
        ("ops_graph", "decode", "calibration"),
        ("fpm", "decode", "calibration"),
        ("ops_graph_holdout", "prefill", "calibration"),
        ("ops_graph_holdout", "decode", "holdout"),
    ],
)
def test_control_role_does_not_change_calibration_holdout_or_fpm(purpose, phase, role):
    with pytest.raises(ValueError, match="restricted"):
        control.validate_scope(purpose, phase, role, "reference.json", "a" * 64)
    control.validate_scope(purpose, phase, role, None, None)


def test_sampling_parameters_preserve_actual_submitted_order_above_nine():
    # Each prompt and desired query is different; lexical q10/q2 sorting would
    # attach a valid token to the wrong actual request.
    items = [
        {"source_request_id": f"q{slot}", "prompt_token_ids": [slot + 1], "native_query_token_ids": [slot + 100]}
        for slot in range(12)
    ]
    reference = {"targets": [{"benchmark_id": 1, "repetition": 0, "requests": items}]}
    inputs = [row["prompt_token_ids"] for row in items]
    assert [next(iter(row["logit_bias"])) for row in control.sampling_parameters(reference, 1, 0, inputs)] == [
        str(slot + 100) for slot in range(12)
    ]
    with pytest.raises(ValueError, match="submitted prompt/order"):
        control.sampling_parameters(reference, 1, 0, sorted(inputs, key=str))


def test_loaded_callable_cannot_hide_an_interposed_wrapper(tmp_path, monkeypatch):
    # Compile authored methods at the TEST_ONLY source location; no native
    # import/GPU execution is represented by this method-chain regression.
    root = tmp_path / "sglang"
    runner_path = root / "srt/model_executor/model_runner.py"
    sampler_path = root / "srt/layers/sampler.py"
    runner_path.parent.mkdir(parents=True)
    sampler_path.parent.mkdir(parents=True)
    runner_source = "class ModelRunner:\n def sample(self): pass\n def _preprocess_logits(self): pass\n"
    sampler_source = "class Sampler:\n def forward(self): pass\n"
    runner_path.write_text(runner_source)
    sampler_path.write_text(sampler_source)
    namespace = {}
    exec(compile(runner_source, str(runner_path), "exec"), namespace)
    exec(compile(sampler_source, str(sampler_path), "exec"), namespace)
    runner = namespace["ModelRunner"]()
    runner.sampler = namespace["Sampler"]()
    monkeypatch.setattr(control, "sampling_sources", lambda: (root, control.SOURCE_PINS))
    assert control.worker_identity(runner)["methods"]["Sampler.forward"]["wrapped"] is False
    original = runner.sample

    @functools.wraps(original)
    def substitute():
        return original()

    runner.sample = substitute
    with pytest.raises(RuntimeError, match="another loaded sampling callable"):
        control.worker_identity(runner)


@pytest.mark.parametrize(
    "mode,graph_submission,biased",
    [("prefill", False, False), ("decode", False, False), ("decode", True, False), ("decode", True, True)],
)
@pytest.mark.parametrize("fail", [False, True])
def test_driver_submits_original_public_arguments_exactly_once_without_retry(
    tmp_path, mode, graph_submission, biased, fail
):
    from collector.fpm_forward.sglang_driver import generate_native_request

    calls = []
    inputs = [[4, 5], [6, 8]]
    ids = ["q2", "q10"]
    reference = (
        {
            "targets": [
                {
                    "benchmark_id": 1,
                    "repetition": 0,
                    "requests": [
                        {"prompt_token_ids": prompt, "native_query_token_ids": [token]}
                        for prompt, token in zip(inputs, [7, 9], strict=True)
                    ],
                }
            ]
        }
        if biased
        else None
    )
    expected = {"temperature": 0, "max_new_tokens": 2 if mode == "decode" else 1, "ignore_eos": True}
    if biased:
        expected = [{**expected, "logit_bias": {str(token): 100.0}} for token in (7, 9)]

    def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs == {"input_ids": inputs, "rid": ids, "sampling_params": expected}
        assert kwargs["input_ids"] is inputs and kwargs["rid"] is ids
        if graph_submission:
            rows = list(iter_records(tmp_path / control.JOURNAL))
            assert len(rows) == 1 and rows[0]["sampling_params"] == expected
        else:
            assert not (tmp_path / control.JOURNAL).exists()
        if fail:
            raise RuntimeError("TEST_ONLY original native failure")
        return "original result"

    engine = SimpleNamespace(generate=generate)

    def submit():
        return generate_native_request(
            engine,
            root=tmp_path,
            benchmark_id=1,
            repetition=0,
            mode=mode,
            request_ids=ids,
            inputs=inputs,
            graph_submission=graph_submission,
            reference=reference,
            reference_sha256="a" * 64 if biased else None,
        )

    if fail:
        with pytest.raises(RuntimeError, match="original native failure"):
            submit()
    else:
        assert submit() == "original result"
    assert len(calls) == 1


def test_declared_control_reference_cannot_be_removed_to_claim_ordinary_sampling(tmp_path, monkeypatch):
    run, root, _, independent = pair(tmp_path, monkeypatch)
    (independent["control_root"] / control.REFERENCE).unlink()
    with pytest.raises(ValueError, match="original sampling declaration"):
        graph.export_graph(root, run, root / graph.BASENAME, **independent)


def test_new_pair_cannot_borrow_unobserved_legacy_sampling_producer(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    add_submission(root, run)
    independent = control_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="actual producer/sampling identity"):
        graph.export_graph(root, run, root / graph.BASENAME, **independent)
