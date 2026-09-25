# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY actual metadata files and public API, with synthetic sacct streams.

No scheduler, native model or actual original benchmark is used.
"""

import copy
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from .test_glm53flash_cleanup_reconciliation import Fixture, c, executor, ref

pytestmark = pytest.mark.unit
a = c.accounting


class AccountingFixture(Fixture):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        original = json.loads(self.paths["allocation"].read_bytes())
        job = json.loads(original["stdout"])["jobs"][0]
        job.update(
            user_id=os.getuid(),
            user_name="TEST_ONLY_user",
            nodes="TEST_ONLY_node",
            tres_alloc_str="cpu=1,mem=1G,node=1,gres/gpu=2",
            restart_cnt=0,
            submit_time={"set": True, "infinite": False, "number": 1672531200},
            start_time={"set": True, "infinite": False, "number": 1672531260},
        )
        original["stdout"] = json.dumps({"jobs": [job]})
        self.put(self.paths["allocation"], original)
        self.accounting_root = root / ("TEST_ONLY_accounting_" + self.job)
        self.accounting_root.mkdir()
        self.exe = self.accounting_root / "sacct"
        self.exe.write_text("TEST_ONLY no executable scheduler")
        self.config = self.accounting_root / "slurm.conf"
        self.config.write_text(f"ClusterName={self.cluster}\nAccountingStorageHost=TEST_ONLY_db\n")
        self.rows_accounting = []
        for suffix, name in (("", "TEST_ONLY_job"), (".batch", "batch"), (".extern", "extern"), (".0", self.step)):
            self.rows_accounting.append(
                {
                    "Cluster": self.cluster,
                    "DBIndex": "12345678901234567890",
                    "JobIDRaw": self.job + suffix,
                    "JobID": self.job + suffix,
                    "JobName": name,
                    "UID": str(os.getuid()) if not suffix else "",
                    "User": "TEST_ONLY_user" if not suffix else "",
                    "Partition": "TEST_ONLY",
                    "State": "FAILED" if not suffix else "COMPLETED",
                    "Submit": "2023-01-01T00:00:00",
                    "Start": "2023-01-01T00:01:00",
                    "End": "2023-01-01T00:03:00",
                    "Restarts": "0" if not suffix else "",
                    "ExitCode": "1:0" if not suffix else "0:0",
                    "DerivedExitCode": "0:0",
                    "AllocTRES": job["tres_alloc_str"],
                    "NodeList": job["nodes"],
                }
            )
        self.capture_path = self.accounting_root / "independent-original-capture.json"
        self.put(
            self.capture_path,
            {
                "observed_ns": time.time_ns() - 1_000_000_000,
                "rows": [
                    {
                        "argv": ["sacct", "-nP", "-j", self.job, "--format=" + a.LEGACY_FIELDS],
                        "returncode": 0,
                        "stdout": self.stdout(a.LEGACY_FIELDS),
                        "stderr": "",
                    }
                ],
            },
        )
        self.capture17_path = self.accounting_root / "independent-database-capture.json"
        self.put(
            self.capture17_path,
            {
                "observed_ns": time.time_ns() - 900_000_000,
                "rows": [
                    {
                        "argv": ["sacct", "--duplicates", "--format=" + a.FIELDS],
                        "returncode": 0,
                        "stdout": self.stdout(),
                        "stderr": "",
                    }
                ],
            },
        )
        self.history_path = self.accounting_root / "known-history.json"
        self.put(
            self.history_path,
            {
                "contract": a.HISTORY,
                "captures": [self.history_item(self.capture_path), self.history_item(self.capture17_path)],
            },
        )
        self.commands = []
        self.query_count = 0
        self.change = None

    def history_item(self, path):
        return {
            "reference": ref(self.root, path),
            "command_pointer": ["rows", 0],
            "time_pointer": ["observed_ns"],
            "timezone": "UTC",
            "cluster": self.cluster,
        }

    def stdout(self, fields=a.FIELDS):
        names = [field.split("%", 1)[0] for field in fields.split(",")]
        return "\n".join("|".join(row[n] for n in names) for row in self.rows_accounting) + "\n"

    def request(self):
        request = executor.original_request(super().original())
        request.update(
            termination_mode=a.MODE,
            accounting={
                "client": {
                    key: {"path": str(path), "sha256": c.digest(path.read_bytes())}
                    for key, path in (("executable", self.exe), ("config", self.config))
                },
                "known_history": ref(self.root, self.history_path),
            },
        )
        return request

    def original(self):
        return executor.load_original(self.request())

    def command(self, argv, **kwargs):
        self.commands.append(argv)
        assert argv[0] == str(self.exe)
        assert kwargs["env"]["TZ"] == "UTC" and kwargs["env"]["SLURM_CONF"] == str(self.config)
        assert not any(k.startswith("SACCT_") for k in kwargs["env"])
        if argv[1:] == ["--version"]:
            result = "slurm 25.11.6\n"
        elif argv[1:] == ["--helpformat"]:
            result = " ".join(field.split("%", 1)[0] for field in a.FIELDS.split(","))
        else:
            self.query_count += 1
            if self.change:
                self.change(self.query_count)
            result = self.stdout()
        return subprocess.CompletedProcess(argv, 0, result, "")

    def capture(self, identity, output):
        self.events.append("accounting")
        return executor.capture_accounting(identity, output)

    def run(self, output, **callbacks):
        with patch.object(executor.subprocess, "run", self.command):
            return executor._reconcile(
                self.original(),
                output,
                cleanup=lambda *_: pytest.fail("historical mode cannot invoke live cleanup"),
                native=callbacks.get("native", self.native),
                scan=callbacks.get("scan", self.scan),
                accounting_capture=self.capture,
            )


def test_public_historical_api_encloses_original_reparse_and_never_cancels(tmp_path, monkeypatch):
    f = AccountingFixture(tmp_path)
    monkeypatch.setenv("SACCT_STATE", "running")
    with patch.object(executor.subprocess, "run", f.command), patch.object(executor, "native_original", f.native):
        proof = executor.reconcile(f.request(), tmp_path / "output")
    assert proof["cleanup"]["outcome"] == a.OUTCOME
    assert c.verify(proof)["entry"]["status"] == "cleanup_failed"
    assert f.query_count == 2 and len(f.commands) == 6
    assert all(Path(command[0]).name == "sacct" for command in f.commands)
    assert proof["cleanup"]["before"]["query"]["stdout"] == proof["cleanup"]["after"]["query"]["stdout"]
    assert proof["cleanup"]["before"]["query"]["stdout"].splitlines()[0].split("|")[13] == "1:0"
    assert f.events == ["native"]  # Public inventory is real; no synthetic timing/model acceptance.


@pytest.mark.parametrize(
    "issue",
    [
        "cluster",
        "uid",
        "dbindex",
        "group_dbindex",
        "reuse",
        "restart",
        "missing_step",
        "missing_extern",
        "nonterminal",
        "end",
        "empty",
        "duplicate",
        "wrong_name",
    ],
)
def test_bad_accounting_stops_before_raw_read(tmp_path, issue):
    f = AccountingFixture(tmp_path)

    def change(_):
        rows = f.rows_accounting
        if issue == "cluster":
            rows[0]["Cluster"] = "other"
        elif issue == "uid":
            rows[0]["UID"] = "99999"
        elif issue == "dbindex":
            rows[-1]["DBIndex"] = "99999"
        elif issue == "group_dbindex":
            for row in rows:
                row["DBIndex"] = "99999"
        elif issue == "reuse":
            rows[0]["Submit"] = "2023-01-02T00:00:00"
        elif issue == "restart":
            rows[0]["Restarts"] = "1"
        elif issue == "missing_step":
            rows.pop()
        elif issue == "missing_extern":
            rows.pop(2)
        elif issue == "nonterminal":
            rows[-1]["State"] = "RUNNING"
        elif issue == "end":
            rows[-1]["End"] = "Unknown"
        elif issue == "empty":
            rows.clear()
        elif issue == "duplicate":
            rows.append(copy.deepcopy(rows[0]))
        elif issue == "wrong_name":
            rows[-1]["JobName"] = "another-owner"

    f.change = change
    with pytest.raises(ValueError):
        f.run(tmp_path / "output")
    assert f.events == ["accounting"]
    assert f.query_count == 1
    assert (tmp_path / "output/failure.json").is_file()
    assert not (tmp_path / "output/receipt.json").exists()


def test_changed_terminal_record_after_read_rejects_without_retry(tmp_path):
    f = AccountingFixture(tmp_path)
    f.change = lambda count: f.rows_accounting[-1].update(End="2023-01-01T00:04:00") if count == 2 else None
    with pytest.raises(ValueError, match="records changed"):
        f.run(tmp_path / "output")
    assert f.events == ["accounting", "inventory", "native", "inventory", "accounting"]
    assert f.query_count == 2 and not (tmp_path / "output/receipt.json").exists()


@pytest.mark.parametrize("change", ["owner", "metadata", "config", "history"])
def test_original_or_authority_change_during_reparse_fails(tmp_path, change):
    f = AccountingFixture(tmp_path)

    def native(identity, output):
        result = f.native(identity, output)
        path = {"owner": f.paths["owner"], "metadata": f.paths["plan"], "config": f.config, "history": f.capture_path}[
            change
        ]
        path.write_text("TEST_ONLY changed original/authority")
        return result

    with pytest.raises(ValueError):
        f.run(tmp_path / "output", native=native)
    assert not (tmp_path / "output/receipt.json").exists()


def test_native_failure_cannot_be_overridden_by_terminal_accounting(tmp_path):
    f = AccountingFixture(tmp_path)

    def native(*_):
        raise ValueError("TEST_ONLY original strict native failure")

    with pytest.raises(ValueError, match="native failure"):
        f.run(tmp_path / "output", native=native)
    assert f.query_count == 1 and not (tmp_path / "output/receipt.json").exists()


def test_watchdog_review_remains_ineligible_before_query(tmp_path):
    f = AccountingFixture(tmp_path)
    review = json.loads(f.paths["failure_review"].read_bytes())
    review["native_failures"] = ["TEST_ONLY watchdog"]
    f.put(f.paths["failure_review"], review)
    with pytest.raises(ValueError, match="native/watchdog"):
        f.run(tmp_path / "output")
    assert f.commands == [] and f.events == []


def test_offline_requires_both_ordered_captures_original_owner_and_distinct_outcome(tmp_path):
    f = AccountingFixture(tmp_path)
    proof = f.run(tmp_path / "output")
    for mutate in (
        lambda p: p["cleanup"].pop("after"),
        lambda p: p["cleanup"].update(outcome="OWNED_STEPS_ABSENT"),
        lambda p: p["cleanup"]["reparse_window"].update(started_ns=p["cleanup"]["before"]["started_ns"]),
        lambda p: p["cleanup"]["after"]["owner_before"].update(sha256="0" * 64),
        lambda p: p["original"]["accounting"].update(capture_documents=[]),
    ):
        bad = copy.deepcopy(proof)
        mutate(bad)
        with pytest.raises(ValueError):
            c.verify(bad)


def test_mutable_historical_end_is_retained_without_rewriting_original_capture(tmp_path):
    f = AccountingFixture(tmp_path)
    first = f.capture_path.read_bytes()
    f.rows_accounting[2].update(State="CANCELLED", End="2023-01-01T00:06:00", ExitCode="")
    second = f.accounting_root / "second-original.json"
    f.put(
        second,
        {
            "observed_ns": time.time_ns() - 500_000_000,
            "rows": [
                {
                    "argv": ["sacct", "--duplicates", "--format=" + a.FIELDS],
                    "returncode": 0,
                    "stdout": f.stdout(),
                    "stderr": "",
                }
            ],
        },
    )
    history = json.loads(f.history_path.read_bytes())
    history["captures"].append(f.history_item(second))
    f.put(f.history_path, history)
    proof = f.run(tmp_path / "output")
    assert len(proof["original"]["accounting"]["capture_documents"]) == 3
    assert f.capture_path.read_bytes() == first
    assert c.verify(proof)["termination_mode"] == a.MODE


def test_wrong_original_owner_rejects_before_any_raw_read(tmp_path):
    f = AccountingFixture(tmp_path)
    f.put(f.paths["owner"], {"job_id": f.job, "step_name": "fpm-wrong"})
    with pytest.raises(ValueError):
        f.run(tmp_path / "output")
    assert "inventory" not in f.events and "native" not in f.events


def test_legacy_only_history_cannot_invent_database_run_identity(tmp_path):
    f = AccountingFixture(tmp_path)
    f.put(f.history_path, {"contract": a.HISTORY, "captures": [f.history_item(f.capture_path)]})
    with pytest.raises(ValueError, match="database run identity capture missing"):
        f.run(tmp_path / "output")
    assert f.commands == [] and f.events == []


@pytest.mark.parametrize("problem", ["unsupported", "timeout", "query_stderr"])
def test_failed_native_client_preflight_or_query_never_reparses_original(tmp_path, problem):
    f = AccountingFixture(tmp_path)
    normal = f.command

    def command(argv, **kwargs):
        if problem == "timeout" and "--helpformat" in argv:
            raise subprocess.TimeoutExpired(argv, 60, output="TEST_ONLY partial", stderr="")
        result = normal(argv, **kwargs)
        if problem == "unsupported" and "--helpformat" in argv:
            result.stdout = result.stdout.replace("DBIndex", "")
        if problem == "query_stderr" and any(x.startswith("--jobs=") for x in argv):
            result.stderr = "TEST_ONLY partial accounting warning"
        return result

    f.command = command
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        f.run(tmp_path / "output")
    assert f.events == ["accounting"]
    assert f.query_count == (1 if problem == "query_stderr" else 0)
    assert not (tmp_path / "output/receipt.json").exists()


def test_owner_changed_during_query_is_rejected_before_reparse(tmp_path):
    f = AccountingFixture(tmp_path)
    f.change = lambda _: f.put(f.paths["owner"], {"job_id": "99999", "step_name": f.step})
    with pytest.raises(ValueError, match="owner changed"):
        f.run(tmp_path / "output")
    assert f.events == ["accounting"]
