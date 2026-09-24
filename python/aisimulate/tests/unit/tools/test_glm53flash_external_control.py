# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY original fixtures, never accepted measurements or uploaded data."""

import copy
import json
from pathlib import Path

import pytest
from tools.glm53flash_hf import external_control as control
from tools.glm53flash_hf import raw_archive, raw_campaign

pytestmark = pytest.mark.unit


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value) + "\n").encode()
    path.write_bytes(data)
    return control.sha(data)


@pytest.fixture
def launch(tmp_path):
    source = tmp_path / "TEST_ONLY_original"
    original = Path("/original/TEST_ONLY_task")
    anchors = dict(
        launcher_manifest="launch/manifest.sha256",
        admission="launch/admission.json",
        cache_hook="prep/cache-hook/sitecustomize.py",
        source_identity="prep/source.json",
        cache_cpu_result="cpu/result.json",
    )
    hook = write(source, anchors["cache_hook"], b"# TEST_ONLY original fixture cache hook\n")
    identity = dict(source_commit="a" * 40, wheel_sha256="b" * 64, cache_hook_sha256=hook)
    frozen = {
        anchors["cache_hook"]: hook,
        anchors["source_identity"]: write(source, anchors["source_identity"], identity),
        anchors["cache_cpu_result"]: write(
            source,
            anchors["cache_cpu_result"],
            {
                "status": "passed",
                "test_only": True,
                "cache_hook_sha256": hook,
                "host_and_producer_source": identity["source_commit"],
                "runs": [{"returncode": 0}],
            },
        ),
    }
    children, runs, pairs, plans = [], [], [], {}
    for i in range(2):
        cid, plan_sha = "TEST_ONLY_child" + str(i), str(i + 1) * 64
        child = dict(child_cell_id=cid, child_plan_sha256=plan_sha, role="calibration", phase="prefill")
        children.append(child)
        env = f"inputs/native/{cid}/collector-runtime-env.sh"
        frozen[env] = write(source, env, b"export PYTHONPATH=/opt/glm53flash-cache:/opt/producer\n")
        # Identical files at distinct original paths exercise content deduplication.
        frozen[f"inputs/{cid}/same.txt"] = write(source, f"inputs/{cid}/same.txt", b"TEST_ONLY identical\n")
        raw_root = f"runs/{cid}/{700 + i}/raw/node0000"
        started = f"runs/{cid}/{700 + i}/started.json"
        prov = raw_root + "/collector-provenance.json"
        runs.append(dict(cell_id=cid, raw_root=str(original / raw_root), started=started, collector_provenance=prov))
        prov_sha = write(source, prov, dict(cell_id=cid, plan_sha256=plan_sha, attempt_id="TEST_ONLY_attempt"))
        spec = dict(
            cell_id=cid,
            raw_root=str(original / raw_root),
            attempt_id="TEST_ONLY_attempt",
            plan=dict(path=f"plans/{cid}.json", sha256="c" * 64),
        )
        pairs.append((spec, {"receipts": [{"path": "collector-provenance.json", "sha256": prov_sha}]}))
        plans[spec["plan"]["path"]] = dict(
            backend="sglang",
            sha256=plan_sha,
            options=dict(
                dataset_role="calibration",
                slurm_container_mounts=[str(original / "prep/cache-hook") + ":/opt/glm53flash-cache:ro"],
            ),
            cells=[dict(cell_id=cid, workload_kind="prefill")],
        )
    admission = dict(
        framework="sglang0.5.20",
        status="CONFIGURATION_QUALIFIED_FOR_FROZEN_FORMAL_COLLECTION",
        source_commit=identity["source_commit"],
        wheel_sha256=identity["wheel_sha256"],
        bindings=[dict(path=p, sha256=s) for p, s in frozen.items()],
        children=children,
        child_count=len(children),
    )
    admission_sha = write(source, anchors["admission"], admission)
    producer_sha = write(
        source,
        "launch/cpu-public-host-producer.json",
        {
            "state": "passed",
            "source_commit": identity["source_commit"],
            "wheel_sha256": identity["wheel_sha256"],
        },
    )
    wheel_sha = write(
        source,
        "launch/installed-wheel-source-record.json",
        {
            "head_sha": identity["source_commit"],
            "wheel_sha256": identity["wheel_sha256"],
        },
    )
    manifest_sha = write(
        source,
        anchors["launcher_manifest"],
        (
            admission_sha
            + "  admission.json\n"
            + producer_sha
            + "  cpu-public-host-producer.json\n"
            + wheel_sha
            + "  installed-wheel-source-record.json\n"
        ).encode(),
    )
    for child, run in zip(children, runs, strict=True):
        write(
            source,
            run["started"],
            dict(
                state="RUNNING",
                job=Path(run["started"]).parent.name,
                child=child,
                admission_sha256=admission_sha,
                launcher_manifest_sha256=manifest_sha,
                host_and_producer_source=identity["source_commit"],
                host_and_producer_wheel_sha256=identity["wheel_sha256"],
            ),
        )
    return dict(
        source=source,
        original=original,
        anchors=anchors,
        runs=runs,
        pairs=pairs,
        plans=plans,
        output=tmp_path / "TEST_ONLY_attachment",
    )


def prepare(fixture):
    return control.prepare(
        fixture["source"], str(fixture["original"]), fixture["anchors"], fixture["runs"], fixture["output"]
    )


def refreeze_test_launch(fixture, *, update_started=True):
    """Rebuild only explicitly TEST_ONLY fixture identities, never real receipts."""
    source, anchors = fixture["source"], fixture["anchors"]
    admission = json.loads((source / anchors["admission"]).read_bytes())
    for item in admission["bindings"]:
        item["sha256"] = raw_archive.sha_file(source / item["path"])
    admission_sha = write(source, anchors["admission"], admission)
    manifest_path = source / anchors["launcher_manifest"]
    members = control.sums(manifest_path.read_bytes(), Path(anchors["launcher_manifest"]).parent)
    manifest_sha = write(
        source,
        anchors["launcher_manifest"],
        "".join(raw_archive.sha_file(source / name) + "  " + Path(name).name + "\n" for name in members).encode(),
    )
    if update_started:
        for run in fixture["runs"]:
            started = json.loads((source / run["started"]).read_bytes())
            started.update(admission_sha256=admission_sha, launcher_manifest_sha256=manifest_sha)
            write(source, run["started"], started)


@pytest.mark.parametrize("has_wheel", [True, False])
def test_original_source_receipt_shapes_keep_actual_wheel_binding(launch, has_wheel):
    path = launch["source"] / launch["anchors"]["source_identity"]
    source = json.loads(path.read_bytes())
    if has_wheel:
        source["host_and_producer"] = "same exact installed wheel; no source overlay"
    else:
        source.pop("wheel_sha256")
        source.update(
            cell="sglang-nvfp4-tp2",
            status="PREPARATION_ONLY_NOT_ADMITTED",
            cache_cpu_qualification="609540",
            original618point_bytes_unchanged=True,
            gpu_submission="NOT_SUBMITTED",
        )
    write(launch["source"], launch["anchors"]["source_identity"], source)
    refreeze_test_launch(launch)
    document = prepare(launch)
    _, _, admission = control.validate(launch["output"], document)
    assert admission["wheel_sha256"] == "b" * 64


@pytest.mark.parametrize("value", ["f" * 64, None, ""])
def test_present_optional_source_wheel_must_match(launch, value):
    path = launch["source"] / launch["anchors"]["source_identity"]
    source = json.loads(path.read_bytes())
    source["wheel_sha256"] = value
    write(launch["source"], launch["anchors"]["source_identity"], source)
    refreeze_test_launch(launch)
    with pytest.raises(ValueError, match="source/wheel identity differs"):
        prepare(launch)


@pytest.mark.parametrize("target", ["producer", "wheel", "started"])
def test_absent_optional_wheel_cannot_weaken_original_actual_chain(launch, target):
    source, anchors = launch["source"], launch["anchors"]
    identity = json.loads((source / anchors["source_identity"]).read_bytes())
    identity.pop("wheel_sha256")
    write(source, anchors["source_identity"], identity)
    filename = {
        "producer": "launch/cpu-public-host-producer.json",
        "wheel": "launch/installed-wheel-source-record.json",
        "started": launch["runs"][0]["started"],
    }[target]
    modified = json.loads((source / filename).read_bytes())
    modified["host_and_producer_wheel_sha256" if target == "started" else "wheel_sha256"] = "f" * 64
    write(source, filename, modified)
    refreeze_test_launch(launch)
    with pytest.raises(ValueError, match="producer/wheel identity differs|started source/wheel differs"):
        prepare(launch)


def inventory(fixture):
    return [
        dict(kind="file", path=str(p.relative_to(fixture["source"] / "runs")), sha256=raw_archive.sha_file(p))
        for p in (fixture["source"] / "runs").rglob("*")
        if p.is_file()
    ]


def test_portable_attachment_exact_closure_and_dedup(launch):
    before = {str(p): raw_archive.sha_file(p) for p in launch["source"].rglob("*") if p.is_file()}
    document = prepare(launch)
    index, get, admission = control.validate(launch["output"], document)
    assert len(list((launch["output"] / "files").iterdir())) < len(index)
    control.bind_role(
        document, get, admission, launch["pairs"], launch["plans"], "/", inventory(launch), launch["original"] / "runs"
    )
    assert before == {str(p): raw_archive.sha_file(p) for p in launch["source"].rglob("*") if p.is_file()}
    # Replay needs only the portable copy, with no original absolute source path.
    assert not launch["original"].exists()
    assert all(not Path(item["path"]).is_absolute() for item in document["files"])


@pytest.mark.parametrize(
    "mutation",
    ["missing", "wrong_sha", "unfrozen_hook", "wrong_start", "wrong_child", "missing_child", "traversal", "symlink"],
)
def test_preparation_rejects_misbound_originals(launch, mutation):
    source, anchors = launch["source"], launch["anchors"]
    if mutation == "missing":
        (source / anchors["cache_cpu_result"]).unlink()
    elif mutation == "wrong_sha":
        (source / anchors["cache_hook"]).write_text("TEST_ONLY changed")
    elif mutation == "unfrozen_hook":
        anchors["cache_hook"] = "unfrozen/sitecustomize.py"
        write(source, anchors["cache_hook"], b"TEST_ONLY later source")
    elif mutation in {"wrong_start", "wrong_child"}:
        path = source / launch["runs"][0]["started"]
        data = json.loads(path.read_bytes())
        if mutation == "wrong_start":
            data["launcher_manifest_sha256"] = "f" * 64
        else:
            data["child"] = data["child"] | {"child_cell_id": "TEST_ONLY_wrong"}
        write(source, str(path.relative_to(source)), data)
    elif mutation == "missing_child":
        launch["runs"].pop()
    elif mutation == "traversal":
        anchors["cache_hook"] = "prep/../sitecustomize.py"
    elif mutation == "symlink":
        path = source / anchors["cache_hook"]
        saved = path.with_suffix(".original")
        path.rename(saved)
        path.symlink_to(saved.name)
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        prepare(launch)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_attempt", "wrong_raw", "wrong_phase", "archive_missing_start", "archive_wrong_start", "wrong_mount"],
)
def test_role_requires_archived_execution_and_accepted_native_join(launch, mutation):
    document = prepare(launch)
    _, get, admission = control.validate(launch["output"], document)
    items = inventory(launch)
    if mutation == "wrong_attempt":
        launch["pairs"][0][0]["attempt_id"] = "TEST_ONLY_other"
    elif mutation == "wrong_raw":
        launch["pairs"][0][0]["raw_root"] += "/other"
    elif mutation == "wrong_phase":
        next(iter(launch["plans"].values()))["cells"][0]["workload_kind"] = "decode"
    elif mutation == "wrong_mount":
        next(iter(launch["plans"].values()))["options"]["slurm_container_mounts"] = []
    elif mutation == "archive_missing_start":
        items = [item for item in items if not item["path"].endswith("started.json")]
    else:
        next(item for item in items if item["path"].endswith("started.json"))["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        control.bind_role(
            document, get, admission, launch["pairs"], launch["plans"], "/", items, launch["original"] / "runs"
        )


def test_attachment_rejects_later_byte_mutation_and_extra_file(launch):
    document = prepare(launch)
    modified = copy.deepcopy(document)
    modified["files"].append(dict(modified["files"][0], original_path="extra/unbound.txt"))
    with pytest.raises(ValueError, match="closure"):
        control.validate(launch["output"], modified)
    (launch["output"] / document["files"][0]["path"]).write_bytes(b"TEST_ONLY corrupt")
    with pytest.raises(ValueError, match="identity"):
        control.validate(launch["output"], document)


def test_source_change_during_copy_is_rejected(launch, monkeypatch):
    original = control.read_bytes
    reads = 0

    def changed(root, name):
        nonlocal reads
        data = original(root, name)
        if name == launch["anchors"]["cache_hook"]:
            reads += 1
            if reads == 2:
                return data + b" changed"
        return data

    monkeypatch.setattr(control, "read_bytes", changed)
    with pytest.raises(ValueError, match="source changed"):
        prepare(launch)


@pytest.mark.parametrize("trigger", ["native_receipt", "plan_mount"])
def test_hook_attachment_cannot_be_omitted(tmp_path, trigger):
    plan = {"options": {}}
    evidence = {"receipts": []}
    if trigger == "native_receipt":
        evidence["receipts"] = [{"path": "cache-setup-123.json"}]
    else:
        plan["options"]["slurm_container_mounts"] = ["/original/hook:/opt/glm53flash-cache:ro"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="requires original execution controls"):
        raw_campaign.external_controls({}, {}, tmp_path, [({}, evidence)], {"plan": path}, tmp_path)
    assert raw_campaign.external_controls({}, {}, tmp_path, [({}, {"receipts": []})], {}, tmp_path) == {}


def test_production_guard_rejects_test_attachment(launch):
    with pytest.raises(ValueError, match="not formal"):
        raw_campaign.production(prepare(launch))


def test_unknown_startup_adapter_is_rejected(launch):
    document = prepare(launch)
    document["adapter"] = "vllm_unqualified_bootstrap"
    with pytest.raises(ValueError, match="unsupported external startup"):
        control.validate(launch["output"], document)


def test_portable_role_integration_returns_all_control_bytes(launch, monkeypatch):
    document = prepare(launch)
    root = launch["output"]
    manifest = root / "external-control.json"
    inv = root / "inventory.jsonl"
    bundle = root.parent / "TEST_ONLY_archive"
    raw_archive.create_archive(
        launch["source"] / "runs",
        bundle,
        "ssh://ocijhb/lustre/TEST_ONLY/campaign.tar.gz",
        [dict(backend="sglang", weight_quantization="fp8", tp=2, phase="prefill", role="calibration")],
    )
    inv.write_bytes((bundle / raw_archive.INVENTORY).read_bytes())
    expected = dict(path="original/attachment/external-control.json", sha256=raw_archive.sha_file(manifest))
    record = dict(
        external_control=dict(raw_campaign.receipt(manifest, root), original_path=expected["path"]),
        manifest_base="/",
        source_inventory=raw_campaign.receipt(inv, root),
    )
    paths = {}
    for index, (name, plan) in enumerate(launch["plans"].items()):
        path = root / f"plan-{index}.json"
        path.write_text(json.dumps(plan))
        paths[name] = path
    monkeypatch.setattr(raw_campaign, "production", lambda _: None)  # TEST_ONLY, no production bypass.
    files = raw_campaign.external_controls(
        {"external_control": expected}, record, root, launch["pairs"], paths, launch["original"] / "runs"
    )
    assert files == {
        "external-control.json": expected["sha256"],
        **{item["path"]: item["sha256"] for item in document["files"]},
    }
    # The archive's original started bytes prevent a later self-consistent rewrite.
    row = json.loads(inv.read_text().splitlines()[0])
    assert row["path"] == ""
    inv.write_text(json.dumps(row) + "\n")
    record["source_inventory"] = raw_campaign.receipt(inv, root)
    with pytest.raises(ValueError, match="archived original"):
        raw_campaign.external_controls(
            {"external_control": expected}, record, root, launch["pairs"], paths, launch["original"] / "runs"
        )


def test_rehashed_later_launch_cannot_replace_original_started_identity(launch):
    source, anchors = launch["source"], launch["anchors"]
    later = write(source, anchors["cache_hook"], b"# TEST_ONLY later hook\n")
    identity = json.loads((source / anchors["source_identity"]).read_bytes())
    identity["cache_hook_sha256"] = later
    source_sha = write(source, anchors["source_identity"], identity)
    cpu = json.loads((source / anchors["cache_cpu_result"]).read_bytes())
    cpu["cache_hook_sha256"] = later
    cpu_sha = write(source, anchors["cache_cpu_result"], cpu)
    admission = json.loads((source / anchors["admission"]).read_bytes())
    for binding in admission["bindings"]:
        if binding["path"] == anchors["cache_hook"]:
            binding["sha256"] = later
        if binding["path"] == anchors["source_identity"]:
            binding["sha256"] = source_sha
        if binding["path"] == anchors["cache_cpu_result"]:
            binding["sha256"] = cpu_sha
    write(source, anchors["admission"], admission)
    refreeze_test_launch(launch, update_started=False)
    with pytest.raises(ValueError, match="started receipt is not bound"):
        prepare(launch)


def test_full32_binding_copies_and_rechecks_external_attachment(launch, tmp_path, request, monkeypatch):
    from tests.unit.tools.test_glm53flash_hf_publication import staged
    from tests.unit.tools.test_glm53flash_raw_campaign import hydrate_native_fixture

    # This test binds real tiny tar archives to synthetic acceptance; no data publication.
    stage_root, _ = staged.__wrapped__(tmp_path, request)
    native = tmp_path / "TEST_ONLY_NATIVE"
    hydrate_native_fixture(stage_root, native)
    old_root = launch["original"]
    launch["original"] = launch["source"]
    for run, (spec, _), plan in zip(launch["runs"], launch["pairs"], launch["plans"].values(), strict=True):
        run["raw_root"] = str(launch["source"] / Path(run["raw_root"]).relative_to(old_root))
        spec["raw_root"] = run["raw_root"]
        plan["options"]["slurm_container_mounts"] = [
            str(launch["source"] / "prep/cache-hook") + ":/opt/glm53flash-cache:ro"
        ]
        plan["cells"][0].update(topology={"tp": 2}, weight_quantization="fp8")
        plan_path = native / spec["plan"]["path"]
        write(native, spec["plan"]["path"], plan)
        spec["plan"]["sha256"] = raw_archive.sha_file(plan_path)
    prepare(launch)
    stage = raw_campaign.read(stage_root / "stage.json")
    manifest_path = stage_root / stage["input_manifest"]["path"]
    report_path = stage_root / stage["acceptance"]["path"]
    manifest, report = raw_campaign.read(manifest_path), raw_campaign.read(report_path)
    selected = next(
        i
        for i, c in enumerate(report["cells"])
        if (c["backend"], c["weight_quantization"], c["tp"], c["phase"]) == ("sglang", "fp8", 2, "prefill")
    )
    spec = manifest["entries"][selected]["calibration"]
    spec.pop("raw_root")
    spec.pop("attempt_id")
    shard_name = "plans/TEST_ONLY_external_shards.json"
    shard_sha = write(native, shard_name, {"test_only": True})
    spec.update(
        shards=[pair[0] for pair in launch["pairs"]],
        shard_manifest=dict(path=shard_name, sha256=shard_sha),
        external_control=dict(
            path=str(launch["output"] / "external-control.json"),
            sha256=raw_archive.sha_file(launch["output"] / "external-control.json"),
        ),
    )
    report["cells"][selected]["calibration_evidence"] = {
        "shards": [
            dict(
                evidence,
                child_cell_id=child["cell_id"],
                source_plan_sha256=launch["plans"][child["plan"]["path"]]["sha256"],
                runtime_run_id="TEST_ONLY_run",
                runtime_grid_digest="a" * 64,
            )
            for child, evidence in launch["pairs"]
        ]
    }
    report["input_manifest_sha256"] = raw_campaign.digest(manifest)
    write(stage_root, stage["input_manifest"]["path"], manifest)
    write(stage_root, stage["acceptance"]["path"], report)
    stage["input_manifest"]["sha256"] = stage["input_manifest_sha256"] = raw_archive.sha_file(manifest_path)
    stage["acceptance"]["sha256"] = raw_archive.sha_file(report_path)
    write(stage_root, "stage.json", stage)
    monkeypatch.setattr(raw_campaign, "production", lambda _: None)
    plan = raw_campaign.make_plan(stage_root, native)
    bundles = {}
    for job in plan["jobs"]:
        label = raw_campaign.name(job)
        job["source_root"] = str(
            launch["source"] / "runs" if label == "sglang-fp8-2-prefill-calibration" else native / "campaigns" / label
        )
        job["uri"] = "ssh://ocijhb/lustre/TEST_ONLY/" + label + "/campaign.tar.gz"
    for job in plan["jobs"]:
        label = raw_campaign.name(job)
        bundle = tmp_path / ("TEST_ONLY_archive-" + label)
        raw_campaign.archive_one(stage_root, plan, label, bundle)
        bundles[label] = str(bundle)
    output = tmp_path / "TEST_ONLY_bound"
    records = raw_campaign.bind(stage_root, plan, bundles, output)
    all_files = raw_campaign.validate(records, stage_root, output)
    assert any(path.startswith("external-controls/") for path in all_files)
    attached = next(record["external_control"] for record in records if "external_control" in record)
    document = raw_campaign.read(output / attached["path"])
    (output / Path(attached["path"]).parent / document["files"][0]["path"]).write_bytes(b"TEST_ONLY changed after copy")
    with pytest.raises(ValueError, match="content identity"):
        raw_campaign.validate(records, stage_root, output)
