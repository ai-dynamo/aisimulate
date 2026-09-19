# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/cli/test_estimate_speculative.py

"""cli_estimate speculative-block wiring (agg/static paths).

Uses Qwen3-8B model metadata and checked-in h100_sxm vLLM tables with the
eagle3 scheme (chain k3) to exercise resolution, model construction, native
cost estimation, and accepted-token projection together.
"""

from __future__ import annotations

import pytest

from aisimulate.legacy_cli.api import cli_estimate

pytestmark = pytest.mark.unit

EAGLE3_CONFIG = {
    "model_type": "llama",
    "num_hidden_layers": 1,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "draft_vocab_size": 32000,
    "sliding_window": None,
    "use_sliding_window": False,
}
SPEC_BLOCK = {
    "method": "eagle3",
    "params": {"num_speculative_tokens": 3},
    "draft_config": EAGLE3_CONFIG,
    "accepted_tokens": 1.8,  # measured gsm8k chain E+1 = 2.80
}
COMMON = dict(
    model_path="Qwen/Qwen3-8B",
    system_name="h100_sxm",
    backend_name="vllm",
    backend_version="0.24.0",
    isl=64,
    osl=261,
    batch_size=8,
    gemm_quant_mode="bfloat16",
    kvcache_quant_mode="bfloat16",
    fmha_quant_mode="bfloat16",
)


@pytest.mark.parametrize("consumer", ["cli_estimate", "task"])
def test_aggregate_prices_different_draft_work_at_identical_verification_width(consumer):
    from aisimulate.sdk.task_v2 import Task
    from aisimulate_core.sdk import common, models
    from aisimulate_core.sdk.config import ModelConfig
    from aisimulate_core.sdk.perf_database import get_database_view
    from aisimulate_core.sdk.rust_engine_step import _cached_engine_handle
    from aisimulate_core.sdk.speculation import SpeculationConfig

    costs, tpots = [], []
    for tree in ([1, 1, 1], [3]):
        block = {**SPEC_BLOCK, "params": {"tree_shape": tree}}
        args = {**COMMON, "isl": 4000, "osl": 64}
        if consumer == "cli_estimate":
            tpots.append(cli_estimate(mode="agg", ctx_tokens=128, speculative=block, **args).tpot)
        else:
            task_args = {key: value for key, value in args.items() if key != "batch_size"}
            task = Task.from_yaml({"serving_mode": "agg", "speculative": block, **task_args})
            tpots.append(task.run_single_agg(tp=1, batch_size=8, ctx_tokens=128)["tpot"])
        model = models.get_model(
            COMMON["model_path"],
            ModelConfig(
                tp_size=1,
                pp_size=1,
                gemm_quant_mode=common.GEMMQuantMode.bfloat16,
                kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
                speculation=SpeculationConfig(kind="eagle3", params={"tree_shape": tree}, draft_config=EAGLE3_CONFIG),
            ),
            "vllm",
        )
        handle = _cached_engine_handle(model, get_database_view("h100_sxm", "vllm", "0.24.0"))
        indices = [i for i, op in enumerate(model.generation_ops) if op._name.startswith("draft_")]
        costs.append(sum(row[1] for row in handle.evaluate_generation_ops(indices, batch_size=28, s=4033)))
    # Target width, acceptance and prefill graph are equal; the measured
    # native draft-query difference must reach the public aggregate TPOT.
    assert costs[0] > costs[1]
    assert tpots[0] > tpots[1]
    assert tpots[0] - tpots[1] == pytest.approx((costs[0] - costs[1]) / 2.8, rel=1e-10)


class TestEstimateSpeculativeBlock:
    def test_agg_scheme_folds_acceptance(self):
        baseline = cli_estimate(mode="agg", **COMMON)
        spec = cli_estimate(mode="agg", speculative=SPEC_BLOCK, **COMMON)
        # verify-width rounds cost more than an AR step but commit 2.8
        # tokens: TPOT must land strictly between round/1 and baseline.
        assert 0 < spec.tpot < baseline.tpot
        # progress fold is 1 + accepted = 2.8: the speedup cannot exceed it.
        assert baseline.tpot / spec.tpot < 2.8
        # accepted_tokens must actually drive the projection: the same
        # scheme at zero acceptance pays the verify round for one token
        # per step and must be strictly slower.
        zero = cli_estimate(mode="agg", speculative={**SPEC_BLOCK, "accepted_tokens": 0.0}, **COMMON)
        assert spec.tpot < zero.tpot

    def test_static_gen_scheme_projection(self):
        baseline = cli_estimate(mode="static_gen", **COMMON)
        spec = cli_estimate(mode="static_gen", speculative=SPEC_BLOCK, **COMMON)
        assert 0 < spec.tpot < baseline.tpot

    def test_mtp_sugar_still_desugars_to_nextn(self):
        # mtp inside the block must be EXACTLY the legacy pair: a wrong
        # acceptance mapping or a silent AR fallback would still produce a
        # positive tpot, so compare against the legacy nextn output.
        explicit = cli_estimate(
            mode="static_gen",
            speculative={"method": "mtp", "params": {"depth": 1}, "accepted_tokens": 0.7},
            **COMMON,
        )
        legacy = cli_estimate(mode="static_gen", nextn=1, nextn_accepted=0.7, **COMMON)
        baseline = cli_estimate(mode="static_gen", **COMMON)
        assert explicit.tpot == pytest.approx(legacy.tpot)
        assert explicit.tpot != pytest.approx(baseline.tpot)  # not an AR fallback

    def test_mtp_block_still_valid_for_disagg(self):
        # The disagg rejection below covers SCHEME methods only: mtp
        # desugars to the legacy nextn pair and must keep working outside
        # agg/static — pin it against the equivalent legacy configuration.
        disagg_args = dict(
            prefill_tp_size=1,
            prefill_pp_size=1,
            prefill_batch_size=1,
            prefill_num_workers=1,
            decode_tp_size=1,
            decode_pp_size=1,
            decode_batch_size=8,
            decode_num_workers=1,
        )
        explicit = cli_estimate(
            mode="disagg",
            speculative={"method": "mtp", "params": {"depth": 1}, "accepted_tokens": 0.7},
            **disagg_args,
            **COMMON,
        )
        legacy = cli_estimate(mode="disagg", nextn=1, nextn_accepted=0.7, **disagg_args, **COMMON)
        assert explicit.tpot == pytest.approx(legacy.tpot)

    def test_scheme_rejected_for_disagg(self):
        with pytest.raises(NotImplementedError, match="agg/static"):
            cli_estimate(
                mode="disagg",
                speculative=SPEC_BLOCK,
                prefill_tp_size=1,
                prefill_pp_size=1,
                prefill_batch_size=1,
                prefill_num_workers=1,
                decode_tp_size=1,
                decode_pp_size=1,
                decode_batch_size=8,
                decode_num_workers=1,
                **COMMON,
            )


@pytest.mark.parametrize("mode", ["afd", "disagg"])
def test_unsupported_scheme_estimate_fails_before_database_lookup(monkeypatch, mode):
    import aisimulate.sdk.perf_database as perf_database

    def unexpected_database_lookup(*args, **kwargs):
        pytest.fail("unsupported speculative estimate reached a database lookup")

    monkeypatch.setattr(perf_database, "get_database_view", unexpected_database_lookup)
    with pytest.raises(NotImplementedError, match="agg/static"):
        cli_estimate(mode=mode, speculative=SPEC_BLOCK, **COMMON)


def _args(cli_parser, *extra):
    return cli_parser.parse_args(
        [
            "estimate",
            "--model-path",
            COMMON["model_path"],
            "--system",
            COMMON["system_name"],
            "--backend",
            COMMON["backend_name"],
            "--backend-version",
            COMMON["backend_version"],
            "--isl",
            "64",
            "--osl",
            "261",
            "--batch-size",
            "8",
            "--gemm-quant-mode",
            "bfloat16",
            "--kvcache-quant-mode",
            "bfloat16",
            "--fmha-quant-mode",
            "bfloat16",
            "--no-color",
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "method,token_key",
    [
        ("mtp", "depth"),
        ("ngram", "num_speculative_tokens"),
        ("draft_model", "num_speculative_tokens"),
        ("eagle3", "num_speculative_tokens"),
        ("dflash", "num_draft_tokens"),
        ("dspark", "num_draft_tokens"),
    ],
)
def test_cli_draft_count_uses_scheme_parameter(cli_parser, method, token_key):
    from aisimulate.legacy_cli.main import _speculative_block_from_args

    args = _args(cli_parser, "--spec-method", method, "--spec-num-draft-tokens", "3")
    assert _speculative_block_from_args(args)["params"] == {token_key: 3}


def test_cli_flags_reach_real_estimate(cli_parser, monkeypatch, capsys):
    import aisimulate.legacy_cli.api as cli_api
    import aisimulate.legacy_cli.main as cli_main

    results = []
    estimate = cli_api.cli_estimate

    def record_result(**kwargs):
        result = estimate(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(cli_api, "cli_estimate", record_result)
    for acceptance in ("0", "1.8"):
        cli_main._run_estimate_mode(
            _args(
                cli_parser,
                "--spec-method",
                "ngram",
                "--spec-num-draft-tokens",
                "3",
                "--spec-accepted-tokens",
                acceptance,
            )
        )
    assert 0 < results[1].tpot < results[0].tpot
    assert "Performance Estimate" in capsys.readouterr().out


def test_cli_aggregate_draft_flags_price_full_native_draft_work(cli_parser, monkeypatch, capsys):
    import math

    import aisimulate.legacy_cli.api as cli_api
    import aisimulate.legacy_cli.main as cli_main

    results, draft_costs = [], []
    estimate = cli_api.cli_estimate

    def record_result(**kwargs):
        result = estimate(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(cli_api, "cli_estimate", record_result)
    for draft_path in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
        cli_main._run_estimate_mode(
            _args(
                cli_parser,
                "--estimate-mode",
                "agg",
                "--isl",
                "4000",
                "--osl",
                "64",
                "--ctx-tokens",
                "128",
                "--spec-method",
                "draft_model",
                "--spec-num-draft-tokens",
                "3",
                "--spec-accepted-tokens",
                "1.8",
                "--spec-draft-model-path",
                draft_path,
            )
        )
        # Independent width-one checkpoint estimates: one full prefill
        # amortized over chunks, plus three forwards for seven decode requests.
        prefill = estimate(
            mode="static_ctx", **{**COMMON, "model_path": draft_path, "batch_size": 1, "isl": 4000, "osl": 2}
        )
        generation = estimate(
            mode="static_gen", **{**COMMON, "model_path": draft_path, "batch_size": 7, "isl": 4032, "osl": 2}
        )
        draft_costs.append(prefill.ttft / math.ceil(4000 / 128) + 3 * generation.tpot)
    assert results[1].tpot > results[0].tpot
    assert results[1].tpot - results[0].tpot == pytest.approx((draft_costs[1] - draft_costs[0]) / 2.8, rel=1e-10)
    assert "Performance Estimate" in capsys.readouterr().out


def test_cli_mtp_uses_block_acceptance_with_legacy_depth(cli_parser):
    from aisimulate.legacy_cli.main import _resolve_and_validate_nextn

    args = _args(
        cli_parser,
        "--nextn",
        "1",
        "--spec-method",
        "mtp",
        "--spec-num-draft-tokens",
        "1",
        "--spec-accepted-tokens",
        "0.7",
    )
    _resolve_and_validate_nextn(args)
    assert (args.nextn, args.nextn_accepted) == (1, 0.7)


def test_cli_orphan_scheme_flags_are_rejected(cli_parser):
    from aisimulate.legacy_cli.main import _run_estimate_mode

    with pytest.raises(SystemExit, match="requires --spec-method"):
        _run_estimate_mode(_args(cli_parser, "--spec-num-draft-tokens", "3"))


def test_cli_epd_rejects_scheme_instead_of_ignoring_it(cli_parser):
    from aisimulate.legacy_cli.main import _run_estimate_mode

    args = _args(
        cli_parser,
        "--enable-epd",
        "--spec-method",
        "ngram",
        "--spec-num-draft-tokens",
        "3",
        "--spec-accepted-tokens",
        "1.8",
    )
    with pytest.raises(ValueError, match="aggregated serving without EPD"):
        _run_estimate_mode(args)


@pytest.mark.parametrize("depth", [1, 3])
def test_cli_mtp_block_preserves_auto_depth(cli_parser, monkeypatch, depth):
    from aisimulate.legacy_cli.main import _resolve_and_validate_nextn

    monkeypatch.setattr("aisimulate.legacy_cli.main.resolve_nextn_auto", lambda path: depth)
    args = _args(cli_parser, "--nextn", "auto", "--spec-method", "mtp", "--spec-accepted-tokens", "0.7")
    _resolve_and_validate_nextn(args)
    assert (args.nextn, args.nextn_accepted) == (depth, 0.7)


def test_estimate_mtp_block_without_depth_preserves_resolved_auto(monkeypatch):
    monkeypatch.setattr("aisimulate.legacy_cli.api._resolve_nextn_auto", lambda path: 1)
    explicit = cli_estimate(
        mode="static_gen", nextn="auto", speculative={"method": "mtp", "accepted_tokens": 0.7}, **COMMON
    )
    legacy = cli_estimate(mode="static_gen", nextn=1, nextn_accepted=0.7, **COMMON)
    assert explicit.tpot == pytest.approx(legacy.tpot)
