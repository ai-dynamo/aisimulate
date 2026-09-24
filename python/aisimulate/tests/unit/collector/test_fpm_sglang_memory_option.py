# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY: public native memory configuration, without GPU capacity claims."""

import argparse
import hashlib
import json
import shlex
from dataclasses import replace

import pytest

from collector.fpm_forward import glm53flash_validation as validation
from collector.fpm_forward.cli import _parser
from collector.fpm_forward.config import FPMCollectionOptions, reject_fpm_arguments_without_fpm
from collector.fpm_forward.planner import build_collection_plan
from collector.fpm_forward.runner import _cell_generator_overrides, _render_cell
from collector.fpm_forward.sglang_artifact import file_receipt, validate_sglang_repetitions
from collector.fpm_forward.shards import make_shards
from tests.unit.collector.test_fpm_glm53flash_planning import plan
from tests.unit.collector.test_fpm_glm53flash_sglang_artifact import artifact
from tests.unit.collector.test_glm53flash_validation import write_plan

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"


def rebuild(campaign, **options):
    return build_collection_plan(
        backend=campaign.backend,
        model_path=campaign.model_path,
        system=campaign.system,
        selected_ops=set(),
        options=replace(campaign.options, **options),
    )


@pytest.mark.parametrize("value", ["0", "1", "-0.1", "1.1", "nan", "inf", "-inf", "invalid"])
def test_cli_rejects_invalid_fraction(value):
    with pytest.raises(SystemExit):
        _parser().parse_args(["--gpu", "gb300", f"--sglang-mem-fraction-static={value}"])


@pytest.mark.parametrize("value", [False, True, "0.82", 0, 1, float("nan"), float("inf")])
def test_programmatic_options_validate_fraction(value):
    with pytest.raises(ValueError, match="strictly between"):
        FPMCollectionOptions.from_args(argparse.Namespace(fpm_max_gpus=4, sglang_mem_fraction_static=value))


def test_option_is_public_and_fpm_only():
    args = _parser().parse_args(["--gpu", "gb300", "--fpm-max-gpus", "4", "--sglang-mem-fraction-static", "0.82"])
    assert FPMCollectionOptions.from_args(args).sglang_mem_fraction_static == 0.82
    with pytest.raises(ValueError, match="FPM-only arguments.*sglang-mem-fraction-static"):
        reject_fpm_arguments_without_fpm(argparse.Namespace(ops=["gemm"], sglang_mem_fraction_static=0.82))


def test_vllm_cannot_silently_ignore_sglang_option(tmp_path):
    campaign = plan(tmp_path, "vllm", MODEL)
    with pytest.raises(ValueError, match="requires backend=sglang"):
        rebuild(campaign, sglang_mem_fraction_static=0.82)


def test_omission_preserves_default_serialization_and_identifiers(tmp_path):
    campaign = plan(tmp_path, "sglang", MODEL)
    omitted = rebuild(campaign, sglang_mem_fraction_static=None)
    assert omitted.to_dict() == campaign.to_dict()
    assert "sglang_mem_fraction_static" not in omitted.to_dict()["options"]
    for cell in omitted.cells:
        assert "sglang_mem_fraction_static" not in cell.to_dict()
        args = _cell_generator_overrides(omitted, cell, {})["params"]["agg"]["extra_cli_args"]
        assert "--mem-fraction-static" not in args


def test_explicit_value_changes_plan_cells_and_shards_and_renders_once(tmp_path):
    original = plan(tmp_path, "sglang", MODEL)
    campaign = rebuild(original, sglang_mem_fraction_static=0.82, shard_token_budget=1_000_000)
    alternate = rebuild(original, sglang_mem_fraction_static=0.83, shard_token_budget=1_000_000)
    assert len({original.sha256, campaign.sha256, alternate.sha256}) == 3
    assert not set(cell.cell_id for cell in original.cells) & set(cell.cell_id for cell in campaign.cells)
    assert not set(cell.cell_id for cell in alternate.cells) & set(cell.cell_id for cell in campaign.cells)
    assert campaign.to_dict()["options"]["sglang_mem_fraction_static"] == 0.82
    shards = make_shards(campaign)
    assert not {s.plan.sha256 for s in shards} & {s.plan.sha256 for s in make_shards(alternate)}
    for shard in shards:
        assert shard.plan.options.sglang_mem_fraction_static == 0.82
        assert shard.plan.cells[0].sglang_mem_fraction_static == 0.82
    for cell in campaign.cells:
        assert cell.to_dict()["sglang_mem_fraction_static"] == 0.82
        directory = tmp_path / cell.cell_id
        directory.mkdir()
        _render_cell(campaign, cell, directory, {"generator_dynamo_version": "1.3.0"})
        command = next(
            line for line in (directory / "run.sh").read_text().splitlines() if line.startswith("engine_command=(")
        )
        argv = shlex.split(command.removeprefix("engine_command=(").removesuffix(")"))
        assert argv.count("--mem-fraction-static") == 1
        assert argv[argv.index("--mem-fraction-static") + 1] == "0.82"
    with pytest.raises(ValueError, match="between frozen plan and cell"):
        _cell_generator_overrides(campaign, original.cells[0], {})


@pytest.mark.parametrize("kind", ["declared_config", "resolved_config"])
@pytest.mark.parametrize("actual", [None, 0.9062999999999999, "0.82", True])
def test_sha_valid_native_receipt_must_match_frozen_requested_value(tmp_path, kind, actual):
    cell, payload = artifact(tmp_path)
    cell.sglang_mem_fraction_static = 0.82
    evidence = payload["input_provenance"]["native_forward_manifest"]
    for name in ("declared_config", "resolved_config"):
        path = tmp_path / evidence[name]["file"]
        config = json.loads(path.read_text())
        config["mem_fraction_static"] = actual if kind == name else 0.82
        path.write_text(json.dumps(config))
        evidence[name] = file_receipt(path)
    with pytest.raises(ValueError, match=f"{kind.removesuffix('_config')} memory fraction"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


def test_matching_explicit_native_setting_is_accepted(tmp_path):
    cell, payload = artifact(tmp_path)
    cell.sglang_mem_fraction_static = 0.82
    evidence = payload["input_provenance"]["native_forward_manifest"]
    for name in ("declared_config", "resolved_config"):
        path = tmp_path / evidence[name]["file"]
        config = json.loads(path.read_text())
        config["mem_fraction_static"] = 0.82
        path.write_text(json.dumps(config))
        evidence[name] = file_receipt(path)
    validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize("cell_fraction", [None, 0.82, 0.83])
def test_acceptance_reconstructs_and_crossbinds_requested_plan_value(tmp_path, cell_fraction):
    spec = write_plan(tmp_path, ("sglang", "fp8", 2, "prefill"), "holdout")
    path = tmp_path / spec["plan"]["path"]
    frozen = json.loads(path.read_text())
    frozen["options"]["sglang_mem_fraction_static"] = 0.82
    if cell_fraction is not None:
        frozen["cells"][0]["sglang_mem_fraction_static"] = cell_fraction
    path.write_text(json.dumps(frozen))
    spec["plan"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if cell_fraction == 0.82:
        run = validation._plan_run(spec, tmp_path, "holdout")
        assert run["runtime_cell"].sglang_mem_fraction_static == 0.82
    else:
        with pytest.raises(ValueError, match="between frozen plan and cell"):
            validation._plan_run(spec, tmp_path, "holdout")
