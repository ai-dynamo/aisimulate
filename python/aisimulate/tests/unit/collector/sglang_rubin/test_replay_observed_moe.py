# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed consumer identities are never rewritten to admit a changed publisher.

Set AISIM_OBSERVED_MOE_REPLAY_INPUTS to a JSON mapping with v1/v2 publication
arguments (including publisher_source), plus consumer_config, to replay the
real immutable GPU archives. Raw GPU evidence is an external artifact.
"""

import copy
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml
from collector.sglang_rubin import publish_observed_moe as v1
from collector.sglang_rubin import publish_observed_moe_v2 as v2

pytestmark = pytest.mark.unit
PUBLISHERS = {"v1": v1, "v2": v2}
EXPECTED = {
    "v1": "sha256:067ad7797474518eab028911d4f0d6f314e1dd0456b08be5bae45de267f3e332",
    "v2": "sha256:bbe3ebf2d450053f524d38b0a8ef97f0e55df000b6c6f61430b0207afc622eaf",
}


def arguments(version, tmp_path):
    keys = ("archive", "review") if version == "v1" else ("v1_archive", "v1_review", "tail_archive", "tail_review")
    return {"base_systems": tmp_path / "base", "output_systems": tmp_path / "output", **dict.fromkeys(keys, tmp_path)}


@pytest.mark.parametrize("version", PUBLISHERS)
def test_public_api_requires_explicit_qualified_source_before_writing(version, tmp_path):
    with pytest.raises(ValueError, match="publisher_source.*--publisher-source"):
        PUBLISHERS[version].publish(**arguments(version, tmp_path))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("version", PUBLISHERS)
def test_current_source_cannot_claim_fixed_historical_identity(version, tmp_path):
    from collector import provenance
    from collector.sglang_rubin import replay_observed_moe as replay

    package = Path(v1.__file__).parents[2]
    closures = provenance.load_closures(package / "collector/hash_closures.yaml")
    assert provenance.collector_hash(PUBLISHERS[version].MODULE, package, closures) != EXPECTED[version]
    source = tmp_path / "source"
    for name in replay.source_files(version):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    with pytest.raises(ValueError, match="Unapproved frozen publisher source"):
        PUBLISHERS[version].publish(publisher_source=source, **arguments(version, tmp_path))
    assert not (tmp_path / "output").exists()


def test_missing_staged_module_cannot_fall_back_to_installed_collector(tmp_path):
    from collector.sglang_rubin.replay_observed_moe import _CollectorImports

    (tmp_path / "collector").mkdir()
    finder = _CollectorImports(tmp_path)
    assert list(finder.find_spec("collector").submodule_search_locations) == [str(tmp_path / "collector")]
    with pytest.raises(ModuleNotFoundError, match="outside the staged publisher"):
        finder.find_spec("collector.helper", [str(Path(v1.__file__).parents[1])])
    with pytest.raises(ModuleNotFoundError, match="outside the staged publisher"):
        finder.find_spec("collector.helper", [str(tmp_path / "collector")])


@pytest.fixture
def real_inputs():
    path = os.environ.get("AISIM_OBSERVED_MOE_REPLAY_INPUTS")
    if not path:
        pytest.skip("Requires externally retained immutable observed-MoE evidence")
    return json.loads(Path(path).read_text())


@pytest.mark.parametrize("version", PUBLISHERS)
@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink", "tampered", "wrong_profile"])
def test_replay_rejects_changed_real_closure(version, mutation, real_inputs, tmp_path):
    source = tmp_path / "source"
    selected = ("v2" if version == "v1" else "v1") if mutation == "wrong_profile" else version
    shutil.copytree(real_inputs[selected]["publisher_source"], source)
    helper = source / "collector/helper.py"
    if mutation == "missing":
        helper.unlink()
    elif mutation == "extra":
        (source / "extra.py").write_text("raise AssertionError('must never execute')\n")
    elif mutation == "symlink":
        helper.unlink()
        helper.symlink_to(Path(real_inputs[selected]["publisher_source"]) / "collector/helper.py")
    elif mutation == "tampered":
        helper.write_text(helper.read_text() + "\nraise AssertionError('must never execute')\n")
    with pytest.raises(ValueError, match="publisher source|Symlink"):
        PUBLISHERS[version].publish(publisher_source=source, **arguments(version, tmp_path))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize(
    ("version", "argument", "kind", "error"),
    [
        ("v1", "base_systems", "input", "Unsafe source systems root"),
        ("v1", "review", "input", "Expected regular file"),
        ("v1", "output_systems", "existing", "Output systems root already exists"),
        ("v1", "output_systems", "dangling", "Output systems root already exists"),
        ("v2", "base_systems", "input", "Unsafe base systems root"),
        ("v2", "v1_review", "input", "Expected regular file"),
        ("v2", "tail_review", "input", "Expected regular file"),
        ("v2", "output_systems", "existing", "Output systems root already exists"),
        ("v2", "output_systems", "dangling", "Output systems root already exists"),
    ],
)
def test_public_replay_rejects_caller_symlinks_without_writing(
    version, argument, kind, error, relative, real_inputs, tmp_path, monkeypatch
):
    output = tmp_path / "output"
    args = dict(real_inputs[version], output_systems=output)
    target = Path(args[argument]) if kind == "input" else tmp_path / "output-target"
    if kind == "existing":
        target.mkdir()
        (target / "sentinel").write_text("unchanged")
    link = tmp_path / "caller-link"
    link.symlink_to(target, target_is_directory=argument in {"base_systems", "output_systems"})
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    args[argument] = "../caller-link" if relative else link

    with pytest.raises(RuntimeError, match=error):
        PUBLISHERS[version].publish(**args)

    assert link.is_symlink() and link.readlink() == target
    assert not output.exists() and not output.is_symlink()
    if kind == "dangling":
        assert not target.exists() and not target.is_symlink()
    elif kind == "existing":
        assert list(target.iterdir()) == [target / "sentinel"]
        assert (target / "sentinel").read_text() == "unchanged"


@pytest.mark.parametrize("version", PUBLISHERS)
@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_public_replay_preserves_parent_traversal_after_symlink(version, relative, real_inputs, tmp_path, monkeypatch):
    review_key = "review" if version == "v1" else "v1_review"
    branch = tmp_path / "branch"
    (branch / "child").mkdir(parents=True)
    shutil.copyfile(real_inputs[version][review_key], branch / "review.json")
    (tmp_path / "pivot").symlink_to(branch / "child", target_is_directory=True)
    # Collapsing pivot/.. would select this forbidden link instead of the regular file.
    (tmp_path / "review.json").symlink_to(real_inputs[version][review_key])
    monkeypatch.chdir(tmp_path)
    review = "pivot/../review.json"
    if not relative:
        review = f"{tmp_path}/{review}"
    args = dict(real_inputs[version], output_systems=tmp_path / "output", validate_only=True)
    args[review_key] = review

    result = PUBLISHERS[version].publish(**args)

    assert result["status"] == "VALIDATED_ONLY"
    assert result["replay"]["qualified_collector_hash"] == EXPECTED[version]
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("version", PUBLISHERS)
def test_real_publication_loads_through_fixed_canonical_consumer(version, real_inputs, tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    args = dict(real_inputs[version], output_systems=str(tmp_path / "published"))
    result = PUBLISHERS[version].publish(**args)
    assert result["status"] == "PUBLISHED"
    assert result["event"]["collector_hash"] == EXPECTED[version]
    assert result["replay"]["qualified_collector_hash"] == EXPECTED[version]
    assert set(result["replay"]["qualified_sources"]).isdisjoint(result["replay"]["current_support_sources"])
    identity = json.loads(PUBLISHERS[version].IDENTITY_PATH.read_text())
    table = Path(args["output_systems"]) / identity["table_relative_path"]
    metadata = yaml.safe_load((table / "collection_meta.yaml").read_text())
    assert metadata["tables"]["moe_perf"]["collections"][-1] == result["event"]
    receipt = json.loads((table / "evidence" / PUBLISHERS[version].PROFILE / "publication.json").read_text())
    assert receipt["publisher_collector_hash"] == EXPECTED[version]
    assert receipt["publisher_sources"] == result["replay"]["qualified_sources"]
    config = copy.deepcopy(real_inputs["consumer_config"])
    config["systems_paths"] = [args["output_systems"]]
    config["estimator_config"]["op_level"] = {"decode_workload_distribution": PUBLISHERS[version].PROFILE}
    model = RustForwardPassPerfModel.best_available(config)
    packaged = Path(v1.__file__).parents[2] / "src/aisimulate_core/systems"
    old_rows = pq.read_table(packaged / identity["table_relative_path"] / "moe_perf.parquet").to_pylist()
    assert result["rows"] == [row for row in old_rows if row["distribution"] == PUBLISHERS[version].PROFILE]
    config["systems_paths"] = [str(packaged)]
    baseline = RustForwardPassPerfModel.best_available(config)
    batches = [1, 8, 32] if version == "v1" else [1, 3, 8, 29, 31, 32]
    for batch in batches:
        for past_kv in [1024, 8192, 32768]:
            query = dict(batch_size=batch, input_tokens=past_kv, output_tokens=2, prefill=False)
            assert model.static_phase_latency(**query) == baseline.static_phase_latency(**query)


@pytest.mark.parametrize("version", PUBLISHERS)
def test_real_validate_only_preserves_absent_output(version, real_inputs, tmp_path):
    args = dict(real_inputs[version], output_systems=str(tmp_path / "output"), validate_only=True)
    result = PUBLISHERS[version].publish(**args)
    assert result["status"] == "VALIDATED_ONLY"
    assert result["replay"]["qualified_collector_hash"] == EXPECTED[version]
    assert not Path(args["output_systems"]).exists()


@pytest.mark.parametrize("version", PUBLISHERS)
def test_frozen_publisher_still_rejects_changed_raw_archive(version, real_inputs, tmp_path):
    args = dict(real_inputs[version], output_systems=str(tmp_path / "output"))
    archive = tmp_path / "changed.tar.gz"
    archive.write_bytes(b"not the approved GPU evidence")
    args["archive" if version == "v1" else "v1_archive"] = archive
    with pytest.raises(RuntimeError, match="Unapproved native archive"):
        PUBLISHERS[version].publish(**args)
    assert not Path(args["output_systems"]).exists()
