# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind closed raw archives to an accepted stage; standard library, no uploads.

Run archive on the storage host. Only the small bind output belongs in the HF
import. Archive byte verification is distinct from native/accuracy acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

if __package__:
    from . import external_control
    from . import raw_archive as archive
else:
    import external_control
    import raw_archive as archive

FIELDS = ("backend", "weight_quantization", "tp", "phase", "role")
KEYS = {
    (b, q, t, p, r)
    for b in ("vllm", "sglang")
    for q in ("fp8", "nvfp4")
    for t in (2, 4)
    for p in ("prefill", "decode")
    for r in ("calibration", "holdout")
}
SCHEMA = "glm53flash_bound_raw_evidence_v1"
PLAN = "glm53flash_raw_archive_plan_v1"
require = archive.require


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(archive.canonical(value).encode()).hexdigest()


def production(value):
    if isinstance(value, dict):
        require(
            not any(value.get(k) for k in ("test_only", "test_fixture", "synthetic", "diagnostic")),
            "synthetic/test/diagnostic evidence is not formal raw evidence",
        )
        for item in value.values():
            production(item)
    elif isinstance(value, list):
        for item in value:
            production(item)
    elif isinstance(value, str):
        require("TEST_ONLY" not in value, "synthetic/test/diagnostic evidence is not formal raw evidence")


def key(value):
    return tuple(value[k] for k in FIELDS)


def name(value):
    return "-".join(str(v) for v in key(value))


def receipt(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": archive.sha_file(path), "bytes": path.stat().st_size}


def checked(root, item):
    parts = archive.relative_parts(item["path"])
    require(parts, "receipt cannot name a directory")
    path = archive.absolute_safe(Path(root).joinpath(*parts))
    require(path.is_file() and archive.sha_file(path) == item["sha256"], "raw evidence receipt SHA256 mismatch")
    if "bytes" in item:
        require(path.stat().st_size == item["bytes"], "raw evidence receipt size mismatch")
    return path


def original_path(base, value):
    path = Path(value)
    require(".." not in path.parts and "\\" not in value and "\x00" not in value, "unsafe original source path")
    require(
        Path(base).is_absolute() and ".." not in Path(base).parts, "manifest base must be an explicit absolute path"
    )
    return path if path.is_absolute() else Path(base) / path


def context(stage_root):
    stage = read(stage_root / "stage.json")
    report = read(checked(stage_root, stage["acceptance"]))
    manifest = read(checked(stage_root, stage["input_manifest"]))
    for value in (stage, report, manifest):
        production(value)
    require(
        stage.get("schema") == "glm53flash_fpm_publication_stage_v1"
        and stage.get("status") == "STAGED_NOT_PUBLISHED"
        and report.get("mode") == "fpm"
        and report.get("acceptance") == "PASSED"
        and not report.get("errors")
        and manifest.get("mode") == "fpm",
        "raw binding requires a passed formal FPM publication stage",
    )
    require(
        report.get("input_manifest_sha256") == digest(manifest)
        and stage.get("input_manifest_sha256") == stage["input_manifest"]["sha256"],
        "accepted original input manifest changed",
    )
    cells = {}
    for cell in report["cells"]:
        ident = tuple(cell[k] for k in FIELDS[:-1])
        require(
            ident not in cells and cell.get("acceptance") == "PASSED" and not cell.get("errors"),
            "duplicate or unaccepted phase cell",
        )
        cells[ident] = cell
    require(
        set(cells) == {k[:-1] for k in KEYS} and len(manifest["entries"]) == 16, "exact sixteen phase cells required"
    )
    return stage, report, manifest, cells


def control_specs(spec):
    yield spec["plan"]
    if "shards" in spec:
        yield spec["shard_manifest"]
        require(
            spec["shards"] and all("shards" not in child for child in spec["shards"]), "invalid nested/empty shards"
        )
        for child in spec["shards"]:
            yield child["plan"]


def plan_key(plan, spec, role):
    production(plan)
    cells = [c for c in plan["cells"] if c["cell_id"] == spec["cell_id"]]
    require(
        len(cells) == 1 and plan["options"].get("dataset_role", "calibration") == role, "frozen plan cell/role mismatch"
    )
    cell = cells[0]
    quant = {"fp8_block": "fp8", "fp8": "fp8", "nvfp4": "nvfp4"}.get(cell["weight_quantization"])
    ident = (plan["backend"], quant, cell["topology"]["tp"], cell["workload_kind"], role)
    require(ident in KEYS, "frozen plan is outside the required matrix")
    return ident


def native_pairs(spec, evidence):
    if "shards" not in spec:
        require("shards" not in evidence and evidence.get("receipts"), "accepted native file receipts missing")
        return [(spec, evidence)]
    children = {child["cell_id"]: child for child in spec["shards"]}
    observed = {child["child_cell_id"]: child for child in evidence["shards"]}
    require(
        len(children) == len(spec["shards"])
        and len(observed) == len(evidence["shards"])
        and children.keys() == observed.keys(),
        "accepted shard union mismatch",
    )
    return [(children[cid], observed[cid]) for cid in sorted(children)]


def external_controls(spec, record, root, pairs, paths, source):
    plans = {path: read(file) for path, file in paths.items()}
    uses_hook = any(
        Path(item["path"]).name.startswith("cache-setup-") for _, evidence in pairs for item in evidence["receipts"]
    ) or any(
        "/opt/glm53flash-cache" in mount
        for plan in plans.values()
        for mount in plan.get("options", {}).get("slurm_container_mounts", [])
    )
    expected = spec.get("external_control")
    actual = record.get("external_control")
    require(not uses_hook or expected is not None, "external cache hook requires original execution controls")
    require((expected is None) == (actual is None), "missing/unexpected external control attachment")
    if expected is None:
        return {}
    require(
        actual["original_path"] == expected["path"] and actual["sha256"] == expected["sha256"],
        "external control differs from accepted input manifest",
    )
    manifest = checked(root, actual)
    document = read(manifest)
    production(document)
    index, get, admission = external_control.validate(manifest.parent, document)
    external_control.bind_role(
        document,
        get,
        admission,
        pairs,
        plans,
        record["manifest_base"],
        archive.inventory_records(checked(root, record["source_inventory"])),
        source,
    )
    files = {actual["path"]: actual["sha256"]}
    for item in index.values():
        path = manifest.parent / item["path"]
        files[path.relative_to(root).as_posix()] = item["sha256"]
    return files


def consumer_sources(stage_root, stage, entry, cell):
    items = entry["consumer_data"]
    require(
        items and cell["prediction_provenance"]["data_receipts"] == items,
        "accepted consumer_data differs from manifest",
    )
    sources = {item["sha256"]: item for item in stage["sources"]}
    require(len(sources) == len(stage["sources"]), "duplicate source digest")
    result, seen = [], set()
    for item in items:
        origin = item["path"]
        require(origin not in seen, "duplicate original consumer path")
        seen.add(origin)
        source = sources.get(item["sha256"], {})
        require(origin in source.get("original_consumer_paths", []), "consumer source origin/SHA is absent from stage")
        checked(stage_root, source)
        result.append({"original_path": origin, "sha256": item["sha256"], "stage_path": source["path"]})
    return result


def make_plan(stage_root, manifest_base):
    stage_root = Path(stage_root)
    _, _, manifest, cells = context(stage_root)
    base = archive.absolute_safe(manifest_base)
    jobs, seen = [], set()
    for index, entry in enumerate(manifest["entries"]):
        for role in ("calibration", "holdout"):
            spec = entry[role]
            path = original_path(base, spec["plan"]["path"])
            require(archive.sha_file(archive.absolute_safe(path)) == spec["plan"]["sha256"], "original plan changed")
            ident = plan_key(read(path), spec, role)
            require(ident not in seen, "duplicate phase/role manifest entry")
            seen.add(ident)
            pairs = native_pairs(spec, cells[ident[:-1]][role + "_evidence"])
            jobs.append(
                dict(
                    zip(FIELDS, ident, strict=True),
                    entry_index=index,
                    accepted_raw_roots=[str(original_path(base, s["raw_root"])) for s, _ in pairs],
                    source_root=None,
                    uri=None,
                )
            )
    require(seen == KEYS, "manifest omits required phase/role")
    return {
        "schema": PLAN,
        "stage_sha256": archive.sha_file(stage_root / "stage.json"),
        "manifest_base": str(base),
        "jobs": jobs,
    }


def load_plan(stage_root, plan):
    expected = make_plan(stage_root, plan["manifest_base"])
    require(plan["schema"] == PLAN and plan["stage_sha256"] == expected["stage_sha256"], "archive plan stage mismatch")
    jobs = {key(j): j for j in plan["jobs"]}
    require(len(jobs) == len(plan["jobs"]) == 32 and jobs.keys() == KEYS, "exact thirty-two archive jobs required")
    uri_sources = {}
    for original in expected["jobs"]:
        job = jobs[key(original)]
        require(
            all(job[k] == v for k, v in original.items() if k not in ("source_root", "uri")),
            "archive plan changed accepted selection",
        )
        require(Path(job["source_root"]).is_absolute(), "closed source root must be absolute")
        source = archive.absolute_safe(job["source_root"])
        require(source.is_dir(), "closed campaign source must be a directory")
        archive.validate_identity(job["uri"], [{k: job[k] for k in FIELDS}])
        require(
            uri_sources.setdefault(job["uri"], source) == source,
            "one archive URI cannot identify different source roots",
        )
        for raw in job["accepted_raw_roots"]:
            require(
                archive.absolute_safe(raw).is_relative_to(source),
                "accepted raw root is outside declared closed campaign",
            )
    return jobs


def archive_one(stage_root, plan, label, output):
    jobs = load_plan(stage_root, plan)
    matches = [j for j in jobs.values() if name(j) == label]
    require(len(matches) == 1, "unknown archive label")
    job = matches[0]
    output = archive.absolute_safe(output, must_exist=False)
    immutable_roots = [Path(stage_root).absolute(), *(Path(j["source_root"]) for j in jobs.values())]
    require(
        not any(output.is_relative_to(root) for root in immutable_roots),
        "archive output is inside campaign/stage input",
    )
    labels = [
        {k: other[k] for k in FIELDS}
        for _, other in sorted(jobs.items())
        if Path(other["source_root"]) == Path(job["source_root"]) and other["uri"] == job["uri"]
    ]
    return archive.create_archive(job["source_root"], output, job["uri"], labels)


def _bindings(stage_root, ctx, record, root):
    stage, _, manifest, cells = ctx
    ident = key(record)
    index = record["entry_index"]
    require(type(index) is int and 0 <= index < len(manifest["entries"]), "invalid acceptance entry index")
    entry, cell = manifest["entries"][index], cells[ident[:-1]]
    spec = entry[ident[-1]]
    controls = record["controls"]
    expected_controls = {(c["path"], c["sha256"]) for c in control_specs(spec)}
    require(
        len(controls) == len(expected_controls)
        and {(c["original_path"], c["sha256"]) for c in controls} == expected_controls,
        "frozen control receipts differ",
    )
    paths = {c["original_path"]: checked(root, c) for c in controls}
    for path in paths.values():
        production(read(path))
    require(
        plan_key(read(paths[spec["plan"]["path"]]), spec, ident[-1]) == ident, "archive label differs from frozen plan"
    )
    pairs = native_pairs(spec, cell[ident[-1] + "_evidence"])
    archive_input = read(checked(root, record["archive_input_manifest"]))
    source = original_path("/", archive_input["source_path"])
    roots = []
    expected_files = {}
    for child, evidence in pairs:
        child_plan = read(paths[child["plan"]["path"]])
        require(plan_key(child_plan, child, ident[-1]) == ident, "shard plan differs from phase/role")
        if "shards" in spec:
            require(evidence["source_plan_sha256"] == child_plan["sha256"], "accepted shard plan identity mismatch")
        raw = original_path(record["manifest_base"], child["raw_root"])
        require(raw.is_relative_to(source), "accepted native root escapes archive source")
        prefix = raw.relative_to(source).as_posix()
        prefix = "" if prefix == "." else prefix
        archive.relative_parts(prefix)
        require(
            child.get("attempt_id") and evidence.get("runtime_run_id") and evidence.get("runtime_grid_digest"),
            "accepted native attempt/runtime identity missing",
        )
        members = {}
        for item in evidence["receipts"]:
            archive.relative_parts(item["path"])
            require(
                item["path"] and item["path"] not in members and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                "invalid or duplicate accepted native file",
            )
            members[item["path"]] = item["sha256"]
        require(members, "empty native file receipt set")
        for previous in expected_files:
            require(
                not (
                    prefix == previous
                    or prefix.startswith(previous + "/")
                    or previous.startswith(prefix + "/")
                    or prefix == ""
                    or previous == ""
                ),
                "overlapping accepted raw roots",
            )
        expected_files[prefix] = members
        roots.append(
            {
                "cell_id": child["cell_id"],
                "raw_root": child["raw_root"],
                "archive_prefix": prefix,
                "attempt_id": child["attempt_id"],
                "plan_content_sha256": child["plan"]["sha256"],
                "native_evidence_sha256": digest(evidence),
                "files": len(members),
            }
        )
    counts = {"files": 0, "directories": 0, "logical_bytes": 0}
    root_stat = None
    observed = {prefix: {} for prefix in expected_files}
    for item in archive.inventory_records(checked(root, record["source_inventory"])):
        if item["path"] == "":
            root_stat = item["stat"]
        counts["files" if item["kind"] == "file" else "directories"] += 1
        if item["kind"] != "file":
            continue
        require(
            type(item["stat"]["size"]) is int
            and item["stat"]["size"] >= 0
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
            "invalid archived file identity",
        )
        counts["logical_bytes"] += item["stat"]["size"]
        for prefix in observed:
            if prefix == "" or item["path"].startswith(prefix + "/"):
                observed[prefix][item["path"][len(prefix) + 1 :] if prefix else item["path"]] = item["sha256"]
    require(root_stat == archive_input["source_root_stat"], "archive input root stat differs from inventory")
    require(observed == expected_files, "archived accepted native file set/SHA differs from acceptance")
    external_files = external_controls(spec, record, root, pairs, paths, source)
    return roots, consumer_sources(stage_root, stage, entry, cell), counts, external_files


def validate(records, stage_root, root):
    """Portable validation of small bound receipts; large archives stay external.

    Does not fetch URIs or repeat the remote tar extraction. The archive receipt
    attests that separate operation; exact inventory/native/stage binding is
    recomputed here. Return every small file that the importer must preserve.
    """
    ctx = context(stage_root)
    stage = ctx[0]
    seen, files = set(), {}
    require(isinstance(records, list), "external raw evidence must be a list")
    archive_labels, uri_identities, archive_sidecars = {}, {}, {}
    for record in records:
        identity = (record["uri"], record["sha256"], record["bytes"])
        require(
            uri_identities.setdefault(record["uri"], identity) == identity,
            "archive URI has conflicting content identities",
        )
        archive_labels.setdefault(identity, []).append({k: record[k] for k in FIELDS})
    for record in records:
        production(record)
        ident = key(record)
        require(ident in KEYS and ident not in seen, "invalid or duplicate raw evidence role")
        seen.add(ident)
        label = {k: record[k] for k in FIELDS}
        archive.validate_identity(record["uri"], [label])
        require(
            record.get("schema") == SCHEMA
            and record.get("stage_sha256") == archive.sha_file(stage_root / "stage.json")
            and record.get("acceptance_sha256") == stage["acceptance"]["sha256"]
            and record.get("input_manifest_sha256") == stage["input_manifest"]["sha256"],
            "unbound raw evidence stage",
        )
        require(
            record.get("archive_verification") == "PASS" and record.get("external_uri_verification") == "NOT_CHECKED",
            "raw archive verification is missing",
        )
        sidecars = [record[k] for k in ("archive_receipt", "source_inventory", "archive_input_manifest")]
        identity = (record["uri"], record["sha256"], record["bytes"])
        sidecar_identity = tuple((item["sha256"], item["bytes"]) for item in sidecars)
        require(
            archive_sidecars.setdefault(identity, sidecar_identity) == sidecar_identity,
            "shared archive has conflicting receipt/inventory identities",
        )
        for item in [*sidecars, *record["controls"]]:
            checked(root, item)
            require(
                item["path"] not in files or files[item["path"]] == item["sha256"],
                "conflicting external evidence paths",
            )
            files[item["path"]] = item["sha256"]
        attestation = read(checked(root, record["archive_receipt"]))
        archive_input = read(checked(root, record["archive_input_manifest"]))
        production(attestation)
        production(archive_input)
        archive.validate_identity(attestation["external_uri"], attestation["labels"])
        labels = archive_labels[(record["uri"], record["sha256"], record["bytes"])]
        require(
            attestation.get("schema") == archive.SCHEMA
            and attestation.get("status") == "PASS"
            and attestation.get("source_recheck") == "STAT_AND_SHA256_PASS"
            and attestation.get("native_or_accuracy_acceptance") == "NOT_EVALUATED"
            and attestation.get("external_uri_verification") == "NOT_CHECKED"
            and sorted(attestation["labels"], key=key) == sorted(labels, key=key)
            and attestation["external_uri"] == record["uri"],
            "archive attestation mismatch",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            and type(record["bytes"]) is int
            and record["bytes"] > 0
            and attestation["archive"]
            == {"path": archive.ARCHIVE, "sha256": record["sha256"], "bytes": record["bytes"]},
            "archive content identity mismatch",
        )
        for field, target in (("source_inventory", "source_inventory"), ("input_manifest", "archive_input_manifest")):
            require(
                all(attestation[field][k] == record[target][k] for k in ("sha256", "bytes")),
                "archive sidecar identity mismatch",
            )
        require(
            archive_input.get("schema") == archive.SCHEMA
            and archive_input.get("native_or_accuracy_acceptance") == "NOT_EVALUATED"
            and archive_input["source_inventory"] == attestation["source_inventory"]
            and archive_input["labels"] == attestation["labels"]
            and archive_input["external_uri"] == record["uri"],
            "archive input manifest mismatch",
        )
        roots, consumers, counts, external_files = _bindings(stage_root, ctx, record, root)
        for path, digest in external_files.items():
            require(path not in files or files[path] == digest, "conflicting external control files")
            files[path] = digest
        require(
            record["native_roots"] == roots and record["consumer_sources"] == consumers,
            "raw/consumer source binding mismatch",
        )
        require(
            all(attestation["source_inventory"][k] == v for k, v in counts.items())
            and attestation["verification"]
            == dict(counts, status="PASS", method="streaming_tar_extractfile_sha256_and_exact_ordered_membership"),
            "archive verification counts/method mismatch",
        )
    require(seen == KEYS, "external raw evidence must cover exactly thirty-two phase/roles")
    return files


def _bundle_identity(bundle):
    """Observe physical files plus immutable small-file bytes around tar verification.

    The tar is streamed by verify_bundle; its device/inode/stat identity must
    stay unchanged before/after that read, each reuse and the final bind check.
    Small receipts and the complete inventory are also rehashed on every check.
    No verification cache survives this bind invocation.
    """
    bundle = archive.absolute_safe(bundle)
    root_stat = archive.stat_identity(bundle.lstat())
    result = {"root": root_stat}
    for filename in (
        archive.ARCHIVE,
        archive.INVENTORY,
        archive.INPUT_MANIFEST,
        "receipt.json",
        "external-raw-evidence.json",
    ):
        path = archive.absolute_safe(bundle / filename)
        before = archive.stat_identity(path.lstat())
        require(stat.S_ISREG(before["mode"]), "archive bundle member is not a regular file")
        with os.fdopen(os.open(path, archive.FLAGS), "rb") as stream:
            require(
                archive.stat_identity(os.fstat(stream.fileno())) == before, "archive bundle object changed before read"
            )
            content = archive.hash_stream(stream) if filename != archive.ARCHIVE else None
            require(archive.stat_identity(os.fstat(stream.fileno())) == before, "archive bundle changed during read")
        require(archive.stat_identity(path.lstat()) == before, "archive bundle path changed during read")
        result[filename] = {"stat": before, "content": content}
    require(archive.stat_identity(bundle.lstat()) == root_stat, "archive bundle directory changed")
    return result


def _verify_once(bundle, verified):
    before = _bundle_identity(bundle)
    if bundle in verified:
        require(before == verified[bundle], "verified archive bundle changed before reuse")
        return
    archive.verify_bundle(bundle)  # Full tar member extraction/hash verification.
    require(_bundle_identity(bundle) == before, "archive bundle changed during verification")
    verified[bundle] = before


def bind(stage_root, plan, bundles, destination):
    stage_root = Path(stage_root)
    ctx = context(stage_root)
    jobs = load_plan(stage_root, plan)
    require(set(bundles) == {name(job) for job in jobs.values()}, "bundle map must cover exactly thirty-two jobs")
    destination = archive.absolute_safe(destination, must_exist=False)
    require(not destination.exists(), "binding destination exists")
    inputs = [
        stage_root,
        *map(Path, bundles.values()),
        *[Path(j["source_root"]) for j in jobs.values()],
    ]
    require(
        not any(destination.is_relative_to(p.absolute()) for p in inputs),
        "binding destination is inside immutable input",
    )
    destination.mkdir(mode=0o700)
    records, verified = [], {}
    try:
        for ident, job in sorted(jobs.items()):
            bundle = archive.absolute_safe(bundles[name(job)])
            _verify_once(bundle, verified)
            attestation = read(bundle / "receipt.json")
            archive_input = read(bundle / archive.INPUT_MANIFEST)
            require(
                Path(archive_input["source_path"]) == Path(job["source_root"]),
                "bundle archived a different closed source",
            )
            target = destination / "raw-evidence" / name(job)
            target.mkdir(parents=True)
            record = dict(
                zip(FIELDS, ident, strict=True),
                schema=SCHEMA,
                stage_sha256=plan["stage_sha256"],
                acceptance_sha256=ctx[0]["acceptance"]["sha256"],
                input_manifest_sha256=ctx[0]["input_manifest"]["sha256"],
                entry_index=job["entry_index"],
                manifest_base=plan["manifest_base"],
                uri=job["uri"],
                sha256=attestation["archive"]["sha256"],
                bytes=attestation["archive"]["bytes"],
                archive_verification="PASS",
                external_uri_verification="NOT_CHECKED",
            )
            for field, filename in (
                ("archive_receipt", "receipt.json"),
                ("source_inventory", archive.INVENTORY),
                ("archive_input_manifest", archive.INPUT_MANIFEST),
            ):
                shutil.copyfile(bundle / filename, target / filename)
                record[field] = receipt(target / filename, destination)
            controls = []
            spec = ctx[2]["entries"][job["entry_index"]][ident[-1]]
            for item in control_specs(spec):
                if any(c["original_path"] == item["path"] and c["sha256"] == item["sha256"] for c in controls):
                    continue
                source = archive.absolute_safe(original_path(plan["manifest_base"], item["path"]))
                require(archive.sha_file(source) == item["sha256"], "frozen control changed")
                copied = target / (item["sha256"] + ".json")
                if not copied.exists():
                    shutil.copyfile(source, copied)
                controls.append(dict(receipt(copied, destination), original_path=item["path"]))
            record["controls"] = controls
            if "external_control" in spec:
                item = spec["external_control"]
                original = archive.absolute_safe(original_path(plan["manifest_base"], item["path"]))
                require(archive.sha_file(original) == item["sha256"], "external control manifest changed")
                document = read(original)
                index, get, _ = external_control.validate(original.parent, document)
                copied = destination / "external-controls" / item["sha256"]
                (copied / "files").mkdir(parents=True, exist_ok=True)
                for source_item in index.values():
                    content = get(source_item["original_path"])
                    target_file = copied / source_item["path"]
                    if not target_file.exists():
                        with target_file.open("xb") as stream:
                            stream.write(content)
                    require(archive.sha_file(target_file) == source_item["sha256"], "copied external control changed")
                copied_manifest = copied / "external-control.json"
                if not copied_manifest.exists():
                    with copied_manifest.open("xb") as stream:
                        stream.write(original.read_bytes())
                require(archive.sha_file(copied_manifest) == item["sha256"], "copied external manifest changed")
                record["external_control"] = dict(receipt(copied_manifest, destination), original_path=item["path"])
            roots, consumers, _, _ = _bindings(stage_root, ctx, record, destination)
            record.update(native_roots=roots, consumer_sources=consumers)
            records.append(record)
        validate(records, stage_root, destination)
        for bundle, identity in verified.items():
            require(_bundle_identity(bundle) == identity, "verified archive bundle changed before final binding")
        archive.write_json(destination / "external-raw-evidence.json", records)
        return records
    except Exception as error:
        archive.write_json(
            destination / "failure.json",
            {
                "status": "FAILED",
                "error": str(error),
                "partial_output_preserved": True,
                "completed_roles": len(records),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("--manifest-base", type=Path, required=True)
    capture = sub.add_parser("archive")
    capture.add_argument("--label", required=True)
    finish = sub.add_parser("bind")
    finish.add_argument(
        "--bundles", type=Path, required=True, help="JSON map label -> absolute archive bundle directory"
    )
    for command in (prepare, capture, finish):
        command.add_argument("--stage", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    for command in (capture, finish):
        command.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        output = archive.absolute_safe(args.output, must_exist=False)
        require(not output.is_relative_to(args.stage.absolute()), "plan output cannot modify the accepted stage")
        archive.write_json(output, make_plan(args.stage, args.manifest_base))
    elif args.command == "archive":
        archive_one(args.stage, read(args.plan), args.label, args.output)
    else:
        bind(args.stage, read(args.plan), read(args.bundles), args.output)


if __name__ == "__main__":
    main()
