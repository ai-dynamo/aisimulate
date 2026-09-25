# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure historical Slurm accounting proof; never live queue/capacity evidence.

CLI field semantics: https://slurm.schedmd.com/sacct.html. No upstream code is
copied. Installed field support and exact original command bytes are required.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

MODE = "historical-accounting"
OUTCOME = "HISTORICAL_OWNED_RUN_TERMINATED"
HISTORY = "fpm_accounting_known_step_history_v1"
CAPTURE = "fpm_accounting_termination_capture_v1"
FIELDS = (
    "Cluster,DBIndex,JobIDRaw,JobID,JobName%160,UID,User,Partition,State%80,"
    "Submit,Start,End,Restarts,ExitCode,DerivedExitCode,AllocTRES%300,NodeList%300"
)
LEGACY_FIELDS = "JobIDRaw,JobID,JobName%80,User,Partition,State%40,Start,End,ExitCode,AllocTRES%250,NodeList%200,Submit"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


def require(ok, message):
    if not ok:
        raise ValueError("historical accounting: " + message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(record):
    require(set(record) == {"sha256", "bytes", "base64"}, "embedded original fields differ")
    raw = base64.b64decode(record["base64"], validate=True)
    require(len(raw) == record["bytes"] and digest(raw) == record["sha256"], "embedded original changed")
    return json.loads(raw)


def pointer(document, parts):
    require(isinstance(parts, list) and parts, "original capture pointer missing")
    for part in parts:
        require(type(part) in (str, int), "invalid original capture pointer")
        if isinstance(document, list):
            require(type(part) is int and 0 <= part < len(document), "capture index differs")
        else:
            require(isinstance(document, dict) and isinstance(part, str) and part in document, "capture key differs")
        document = document[part]
    return document


def positive(value, name):
    require(type(value) is int and value > 0, name + " must be a positive integer")
    return value


def timestamp(value, timezone="UTC"):
    require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value), "invalid time")
    return int(dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ZoneInfo(timezone)).timestamp())


def allocation(identity):
    job = identity["allocation_job"]
    result = {"uid": positive(job.get("user_id"), "original UID")}
    for key in ("submit_time", "start_time"):
        value = job.get(key)
        require(
            isinstance(value, dict) and value.get("set") is True and value.get("infinite") is False,
            "original time missing",
        )
        result[key] = positive(value.get("number"), "original " + key)
    require(job.get("restart_cnt") == 0 and type(job.get("restart_cnt")) is int, "original restart count differs")
    require(isinstance(job.get("nodes"), str) and job["nodes"], "original node missing")
    require(isinstance(job.get("tres_alloc_str"), str) and job["tres_alloc_str"], "original resources missing")
    require(isinstance(job.get("user_name"), str) and job["user_name"], "original user missing")
    return result


def query_argv(identity):
    source = allocation(identity)
    start = dt.datetime.fromtimestamp(source["submit_time"], dt.UTC).strftime("%Y-%m-%dT00:00:00")
    return [
        "sacct",
        "--local",
        "--clusters=" + identity["cluster"],
        "--duplicates",
        "--noheader",
        "--parsable2",
        "--starttime=" + start,
        "--jobs=" + identity["job"],
        "--format=" + FIELDS,
    ]


def command_stream(command, stream):
    require(command.get("returncode") == 0 and not command.get("timeout", False), "accounting command failed")
    if stream in command:
        require(isinstance(command[stream], str), "command stream missing")
        return command[stream]
    raw = base64.b64decode(command[stream + "_b64"], validate=True)
    require(digest(raw) == command[stream + "_sha256"], "command stream hash differs")
    return raw.decode("utf-8")


def parse(command, *, expected_fields):
    require(command_stream(command, "stderr") == "", "accounting command stderr is not empty")
    fields = [re.sub(r"%\d+$", "", f) for f in expected_fields.split(",")]
    rows = []
    for line in command_stream(command, "stdout").splitlines():
        values = line.split("|")
        require(len(values) == len(fields), "malformed or truncated accounting row")
        rows.append(dict(zip(fields, values, strict=True)))
    require(rows, "empty accounting never proves termination")
    require(len({r["JobIDRaw"] for r in rows}) == len(rows), "duplicate or ambiguous job/run records")
    return rows


def validate_parent(parent, identity, timezone):
    original = allocation(identity)
    require(parent["JobIDRaw"] == parent["JobID"] == identity["job"], "parent job differs")
    require(parent["User"] == identity["allocation_job"]["user_name"], "parent user differs")
    require(timestamp(parent["Submit"], timezone) == original["submit_time"], "parent Submit/run reuse differs")
    require(timestamp(parent["Start"], timezone) == original["start_time"], "parent Start/run reuse differs")
    require(parent["NodeList"] == identity["allocation_job"]["nodes"], "parent node differs")
    require(
        sorted(parent["AllocTRES"].split(",")) == sorted(identity["allocation_job"]["tres_alloc_str"].split(",")),
        "parent resources differ",
    )


def known_history(identity):
    """Rederive membership from independent original command JSON, never a list assertion."""
    evidence = identity["accounting"]
    history = decode(evidence["history_document"])
    require(
        history.get("contract") == HISTORY and set(history) == {"contract", "captures"},
        "known history contract differs",
    )
    captures = history["captures"]
    require(
        isinstance(captures, list) and captures and len(captures) == len(evidence["capture_documents"]),
        "known captures missing",
    )
    known, latest, run_identity_captures = {}, 0, 0
    refs = set()
    for item, document in zip(captures, evidence["capture_documents"], strict=True):
        require(
            set(item) == {"reference", "command_pointer", "time_pointer", "timezone", "cluster"},
            "known capture fields differ",
        )
        ref = item["reference"]
        require((ref["path"], ref["sha256"]) not in refs, "duplicate known capture")
        refs.add((ref["path"], ref["sha256"]))
        require(all(document[k] == ref[k] for k in ("sha256", "bytes")), "known capture reference differs")
        source = decode(document)
        recorded = positive(pointer(source, item["time_pointer"]), "known capture time")
        latest = max(latest, recorded)
        require(item["cluster"] == identity["cluster"], "known cluster differs")
        command = pointer(source, item["command_pointer"])
        argv = command.get("argv")
        require(isinstance(argv, list) and argv and Path(argv[0]).name == "sacct", "known source is not accounting")
        # Positive history only. This legacy command is never reused as the fresh proof.
        require(
            not any(a in {"-X", "--allocations"} or a.startswith(("--state", "-s=")) for a in argv),
            "filtered known history",
        )
        formats = [a.split("=", 1)[1] for a in argv if a.startswith("--format=")]
        require(len(formats) == 1 and formats[0] in {FIELDS, LEGACY_FIELDS}, "unknown historical fields")
        rows = parse(command, expected_fields=formats[0])
        selected = [
            r for r in rows if r["JobIDRaw"] == identity["job"] or r["JobIDRaw"].startswith(identity["job"] + ".")
        ]
        parents = [r for r in selected if r["JobIDRaw"] == identity["job"]]
        require(len(parents) == 1, "known parent missing")
        validate_parent(parents[0], identity, item["timezone"])
        if formats[0] == FIELDS:
            parent = parents[0]
            require(
                parent["UID"] == str(allocation(identity)["uid"])
                and parent["Restarts"] == "0"
                and parent["DBIndex"].isdecimal()
                and int(parent["DBIndex"]) > 0,
                "known database run/UID/restart missing",
            )
            require(all(row["DBIndex"] == parent["DBIndex"] for row in selected), "known database run differs")
            run_identity_captures += 1
        for row in selected:
            require(row["JobIDRaw"] == row["JobID"], "historical step identity differs")
            if "Cluster" in row:
                require(row["Cluster"] == identity["cluster"], "known row cluster differs")
            key = row["JobIDRaw"]
            stable = {k: row[k] for k in ("JobIDRaw", "JobID", "JobName", "NodeList")}
            stable["Start"] = timestamp(row["Start"], item["timezone"])
            if "DBIndex" in row:
                stable["DBIndex"] = row["DBIndex"]
                if key == identity["job"]:
                    stable.update(UID=row["UID"], Restarts=row["Restarts"])
            require(
                key not in known or all(known[key][k] == stable[k] for k in known[key].keys() & stable.keys()),
                "known step/run identity changed",
            )
            known.setdefault(key, {}).update(stable)
    job, step = identity["job"], identity["owner"]["step_name"]
    require({job, job + ".batch", job + ".extern"} <= set(known), "known parent/batch/extern missing")
    require(run_identity_captures > 0, "independent database run identity capture missing")
    require(
        any(re.fullmatch(re.escape(job) + r"\.\d+", k) and r["JobName"] == step for k, r in known.items()),
        "no positive known owned step",
    )
    return known, latest


def validate_client(client):
    require(set(client) == {"executable", "config"}, "client pins differ")
    for value in client.values():
        require(set(value) == {"path", "sha256"}, "client file pin missing")
        require(
            isinstance(value["path"], str)
            and Path(value["path"]).is_absolute()
            and ".." not in Path(value["path"]).parts,
            "client path invalid",
        )
        require(re.fullmatch(r"[a-f0-9]{64}", value["sha256"]) is not None, "client digest invalid")


def validate_stat(value):
    require(set(value) == {"device", "inode", "size", "mtime_ns", "ctime_ns"}, "filesystem identity missing")
    require(
        all(type(v) is int and v >= 0 for v in value.values()) and value["inode"] > 0, "filesystem identity invalid"
    )


def validate_context(context, identity):
    require(
        set(context)
        == {"uid", "hostname", "executable", "config", "configuration", "environment_sha256", "explicit_environment"},
        "client context incomplete",
    )
    require(context["uid"] == allocation(identity)["uid"] and context["hostname"], "accounting owner/host differs")
    client = identity["accounting"]["client"]
    validate_client(client)
    for key in client:
        value = context[key]
        require(set(value) == {"path", "resolved_path", "sha256", "stat"}, "client observation incomplete")
        require(all(value[k] == client[key][k] for k in ("path", "sha256")), "client executable/config changed")
        require(Path(value["resolved_path"]).is_absolute(), "client canonical path missing")
        validate_stat(value["stat"])
    config = context["configuration"]
    require(
        set(config) == {"ClusterName", "AccountingStorageHost"}
        and config["ClusterName"] == identity["cluster"]
        and config["AccountingStorageHost"],
        "accounting configuration cluster/host differs",
    )
    require(
        context["explicit_environment"]
        == {"TZ": "UTC", "SLURM_TIME_FORMAT": "standard", "SLURM_CONF": client["config"]["path"]},
        "accounting time/config environment differs",
    )
    require(re.fullmatch(r"[a-f0-9]{64}", context["environment_sha256"]) is not None, "environment identity missing")


def validate_capture(capture, identity, canonical_cell):
    require(
        set(capture)
        == {
            "contract",
            "started_ns",
            "completed_ns",
            "context_before",
            "context_after",
            "owner_before",
            "owner_after",
            "version",
            "helpformat",
            "query",
        }
        and capture["contract"] == CAPTURE,
        "capture contract incomplete",
    )
    start = positive(capture["started_ns"], "capture start")
    end = positive(capture["completed_ns"], "capture end")
    require(start < end, "capture order invalid")
    known, recorded = known_history(identity)
    require(recorded < start, "known history is not independently recorded before this capture")
    validate_context(capture["context_before"], identity)
    validate_context(capture["context_after"], identity)
    require(capture["context_before"] == capture["context_after"], "client context changed during capture")
    expected_step = "fpm-" + digest(canonical_cell.encode())[:20]
    require(identity["owner"] == {"job_id": identity["job"], "step_name": expected_step}, "wrong original owner")
    owner = capture["owner_before"]
    require(owner == capture["owner_after"], "original owner changed during capture")
    require(set(owner) == {"path", "resolved_path", "sha256", "stat"}, "owner observation missing")
    expected_owner = Path(canonical_cell).parent / ".slurm-owners" / (expected_step + ".json")
    require(
        owner["resolved_path"] == str(expected_owner) and owner["sha256"] == identity["original_owner_sha256"],
        "owner bytes/path differ",
    )
    require(
        owner["path"] == str(Path(identity["task_root"]) / identity["owner_reference_path"]),
        "owner lexical path differs",
    )
    validate_stat(owner["stat"])
    for name, argv in (
        ("version", ["sacct", "--version"]),
        ("helpformat", ["sacct", "--helpformat"]),
        ("query", query_argv(identity)),
    ):
        command = capture[name]
        argv = [identity["accounting"]["client"]["executable"]["path"], *argv[1:]]
        require(
            set(command) == {"argv", "returncode", "stdout", "stderr", "started_ns", "completed_ns"},
            "fresh command fields differ",
        )
        require(
            command["argv"] == argv and start <= command["started_ns"] < command["completed_ns"] <= end,
            "fresh accounting argv/order differs",
        )
        require(command_stream(command, "stderr") == "", "fresh command stderr")
    require(
        capture["version"]["completed_ns"] <= capture["helpformat"]["started_ns"]
        and capture["helpformat"]["completed_ns"] <= capture["query"]["started_ns"],
        "field support must precede query",
    )
    require(
        re.fullmatch(r"slurm \d+\.\d+\.\d+\s*", command_stream(capture["version"], "stdout")) is not None,
        "installed Slurm version missing",
    )
    require(
        set(re.sub(r"%\d+", "", FIELDS).split(",")) <= set(command_stream(capture["helpformat"], "stdout").split()),
        "installed accounting fields unsupported",
    )
    rows = parse(capture["query"], expected_fields=FIELDS)
    by = {r["JobIDRaw"]: r for r in rows}
    require(set(known) <= set(by), "known original step missing")
    job = identity["job"]
    require(
        all(k == job or re.fullmatch(re.escape(job) + r"\.(?:\d+|batch|extern)", k) for k in by),
        "unexpected job/step identity",
    )
    parent = by[job]
    validate_parent(parent, identity, "UTC")
    require(parent["UID"] == str(allocation(identity)["uid"]) and parent["Restarts"] == "0", "UID/restart differs")
    require(parent["DBIndex"].isdecimal() and int(parent["DBIndex"]) > 0, "database run identity missing")
    for row in rows:
        require(
            row["Cluster"] == identity["cluster"] and row["DBIndex"] == parent["DBIndex"],
            "cluster/database run differs",
        )
        require(row["JobIDRaw"] == row["JobID"], "step ID differs")
        require(row["State"] in TERMINAL, "record nonterminal or ambiguous")
        require(
            timestamp(row["End"]) >= timestamp(row["Start"]) and timestamp(row["End"]) * 1_000_000_000 <= start,
            "terminal End missing/future/invalid",
        )
        if row["JobIDRaw"] in known:
            prior = known[row["JobIDRaw"]]
            require(
                all(row[k] == prior[k] for k in ("JobIDRaw", "JobID", "JobName", "NodeList"))
                and timestamp(row["Start"]) == prior["Start"],
                "known step identity changed",
            )
            require(
                all(row[k] == prior[k] for k in ("DBIndex", "UID", "Restarts") if k in prior),
                "independent database run/UID/restart changed",
            )
    return by


def validate(proof, identity):
    require(
        set(proof)
        == {
            "mode",
            "outcome",
            "job_id",
            "step_name",
            "canonical_cell_directory",
            "owner_sha256",
            "before",
            "after",
            "reparse_window",
        },
        "termination proof fields differ",
    )
    require(
        proof["mode"] == MODE and proof["outcome"] == OUTCOME and proof["job_id"] == identity["job"],
        "distinct historical outcome missing",
    )
    require(
        proof["step_name"] == identity["owner"]["step_name"]
        and proof["owner_sha256"] == identity["original_owner_sha256"],
        "original owner differs",
    )
    cell = proof["canonical_cell_directory"]
    require(Path(cell).is_absolute() and ".." not in Path(cell).parts, "canonical cell invalid")
    a = validate_capture(proof["before"], identity, cell)
    b = validate_capture(proof["after"], identity, cell)
    require(a == b, "accounting records changed during original reparse")
    require(
        proof["before"]["context_after"] == proof["after"]["context_before"]
        and proof["before"]["owner_after"] == proof["after"]["owner_before"],
        "owner/client changed during original reparse",
    )
    window = proof["reparse_window"]
    require(set(window) == {"started_ns", "completed_ns"}, "original reparse boundary missing")
    require(
        proof["before"]["completed_ns"]
        <= window["started_ns"]
        < window["completed_ns"]
        <= proof["after"]["started_ns"],
        "captures do not bracket original reparse",
    )
