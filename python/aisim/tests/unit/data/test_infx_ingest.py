# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InferenceX ingestion: retries, rate limits, incomplete pages, valid-empty, contradictions, corrections.

All HTTP is faked; the response shapes mirror the live documented endpoints
(views/options, availability, benchmarks?exact=true, workflow-info).
"""

import ast
import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aisim_core.data import e2e as e2e_module
from aisim_core.data import infx_ingest, infx_job
from aisim_core.data.e2e import evaluation_points, load_e2e, read_snapshot
from aisim_core.data.infx_ingest import (
    HttpResponse,
    HttpStatusError,
    InfxClient,
    RateLimitedError,
    RetryPolicy,
    TransientExhaustedError,
    parse_retry_after,
)
from aisim_core.data.infx_job import PromotionError, promote, run_ingestion, staleness

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
DATE = "2026-11-05"
RUN = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/987"
OPTIONS = {"models": [{"name": "Test-Model", "dbKeys": ["testmodel"]}, {"name": "Other", "dbKeys": ["other"]}]}


def row(rid=1, hardware="h100", **over):
    value = dict(
        id=str(rid), model="testmodel", hardware=hardware, framework="vllm", precision="fp8", spec_method="none",
        disagg=False, is_multinode=False, prefill_tp=1, prefill_ep=1, prefill_dp_attention=False,
        prefill_num_workers=0, decode_tp=8, decode_ep=1, decode_dp_attention=False, decode_num_workers=1,
        num_prefill_gpu=0, num_decode_gpu=8, benchmark_type="single_turn", isl=1024, osl=128, conc=32,
        date=DATE, image="img", run_url=RUN, metrics={"mean_ttft": 0.1, "mean_tpot": 0.01, "mean_e2el": 1.5},
    )  # fmt: skip
    value.update(over)
    return value


def avail(hardware="h100", model="testmodel"):
    return dict(model=model, isl=1024, osl=128, precision="fp8", hardware=hardware, framework="vllm",
                spec_method="none", disagg=False, benchmark_type="single_turn", date=DATE)  # fmt: skip


WORKFLOW = {
    "runs": [{"html_url": RUN, "run_attempt": 2, "github_run_id": "987"}],
    "runConfigs": [{"html_url": RUN, "head_sha": "a" * 40}],
    "changelogs": [],
    "configs": [],
}


class FakeInfx:
    """Scripted transport: route -> list of responses (last one repeats); records every call."""

    def __init__(self, rows=None, availability=None, workflow=WORKFLOW):
        self.rows = [row(1), row(2, "h200")] if rows is None else rows
        self.availability = [avail("h100"), avail("h200")] if availability is None else availability
        self.workflow = workflow
        self.script: dict[str, list] = {}
        self.calls: list[tuple[str, dict]] = []

    def respond(self, route, *responses):
        self.script[route] = list(responses)

    def __call__(self, url, headers, timeout):
        route = url.split("/api/v1/")[-1].split("?")[0]
        self.calls.append((url, dict(headers)))
        queue = self.script.get(route)
        if queue:
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item
        body = {"views/options": OPTIONS, "availability": self.availability, "benchmarks": self.rows,
                "workflow-info": self.workflow}[route]  # fmt: skip
        if route == "benchmarks" and isinstance(body, dict):  # per-date rows (exact=true is "as of" that date)
            body = body[url.split("date=")[1].split("&")[0]]
        etag = f'"{abs(hash(json.dumps(body, sort_keys=True))) % 10**8}"'
        if headers.get("If-None-Match") == etag:
            return HttpResponse(304, {"etag": etag}, b"")
        return HttpResponse(200, {"etag": etag}, json.dumps(body).encode())


def ok(body, **headers):
    return HttpResponse(200, {k.lower(): v for k, v in headers.items()}, json.dumps(body).encode())


def make_run(tmp_path, fake, **kw):
    sleeps = []
    result = run_ingestion(
        tmp_path / "root", tmp_path / "cache", transport=fake, sleep=sleeps.append, now=lambda: NOW,
        policy=kw.pop("policy", RetryPolicy(max_attempts=3, base_delay=1, max_retry_after=10)), **kw,
    )  # fmt: skip
    return result, sleeps


@pytest.fixture
def root(tmp_path):
    (tmp_path / "root").mkdir()
    return tmp_path / "root"


def tree(path: Path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- client


def test_retry_after_parsing():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(None) is None and parse_retry_after("soon") is None
    assert parse_retry_after("Sat, 07 Nov 2026 12:00:30 GMT", now=NOW.timestamp()) == pytest.approx(30.0)


def test_transient_5xx_and_transport_errors_are_retried_with_backoff():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(503, {}, b""), ConnectionError("reset"), ok([]))
    sleeps = []
    client = InfxClient(fake, RetryPolicy(max_attempts=4, base_delay=1), sleeps.append)
    fetched = client.get("availability")
    assert fetched.attempts == 3 and sleeps == [1, 2]


def test_rate_limit_honors_retry_after_then_succeeds():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(429, {"retry-after": "5"}, b""), ok([]))
    sleeps = []
    InfxClient(fake, RetryPolicy(max_attempts=3, max_retry_after=10), sleeps.append).get("availability")
    assert sleeps == [5.0]


def test_retry_after_beyond_cap_fails_instead_of_waiting():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(429, {"retry-after": "3600"}, b""))
    sleeps = []
    with pytest.raises(RateLimitedError):
        InfxClient(fake, RetryPolicy(max_attempts=3, max_retry_after=10), sleeps.append).get("availability")
    assert sleeps == [] and len(fake.calls) == 1


def test_retries_are_bounded_and_client_errors_are_not_retried():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(500, {}, b""))
    with pytest.raises(TransientExhaustedError):
        InfxClient(fake, RetryPolicy(max_attempts=3), lambda _: None).get("availability")
    assert len(fake.calls) == 3
    fake.calls.clear()
    fake.respond("availability", HttpResponse(400, {}, b"bad"))
    with pytest.raises(HttpStatusError):
        InfxClient(fake, RetryPolicy(max_attempts=3), lambda _: None).get("availability")
    assert len(fake.calls) == 1


def test_truncated_body_is_retried_and_request_budget_is_enforced():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(200, {"content-length": "99"}, b"[1"), ok([]))
    assert InfxClient(fake, RetryPolicy(max_attempts=2), lambda _: None).get("availability").attempts == 2
    fake.respond("availability", HttpResponse(500, {}, b""))
    with pytest.raises(RateLimitedError):
        InfxClient(fake, RetryPolicy(max_attempts=9, max_requests=2), lambda _: None).get("availability")


# --------------------------------------------------------------------------- end to end


def test_first_run_writes_candidate_with_stable_shards_and_never_qualifies(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx())
    assert result.status == "candidate" and len(result.changes["added"]) == 2
    state = infx_job.load_state(root)
    assert state["qualified"] is None and state["candidate"]["snapshot_id"] == result.snapshot_id
    snapshot, members = read_snapshot(root, state["candidate"]["snapshot_path"])
    assert snapshot.completeness == "complete" and snapshot.counts == {"observations": 2, "configurations": 2}
    assert all(m.manifest_path.startswith("e2e/Test-Model") or "test" in m.manifest_path.lower() for m in members)
    # One shared loader, producer metadata from workflow-info reached the observation row.
    dataset = load_e2e(root, members[0].manifest_path, config=None)
    assert dataset.observations[0].producer_revision == "a" * 40 and dataset.observations[0].producer_attempt == 2
    assert evaluation_points(dataset)


def test_unchanged_refresh_writes_nothing_and_uses_conditional_requests(root, tmp_path):
    make_run(tmp_path, FakeInfx())
    before = tree(root)
    fake = FakeInfx()
    result, _ = make_run(tmp_path, fake)
    assert result.status == "unchanged" and tree(root) == before
    assert any("If-None-Match" in headers for _, headers in fake.calls)


def test_candidate_is_not_promoted_by_a_later_run_and_promotion_needs_a_reviewer(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    with pytest.raises(PromotionError):
        promote(root, first.snapshot_id, "  ")
    with pytest.raises(PromotionError):
        promote(root, "0" * 64, "reviewer")
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    state = infx_job.load_state(root)
    assert state["qualified"]["snapshot_id"] == first.snapshot_id and state["candidate"] is None
    second, _ = make_run(tmp_path, FakeInfx(rows=[row(1), row(2, "h200"), row(3, "b200")],
                                            availability=[avail("h100"), avail("h200"), avail("b200")]))  # fmt: skip
    state = infx_job.load_state(root)
    assert second.status == "candidate" and state["qualified"]["snapshot_id"] == first.snapshot_id
    assert state["candidate"]["snapshot_id"] == second.snapshot_id


def test_correction_keeps_observation_id_changes_hash_and_preserves_old_snapshot(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    old_doc = (root / infx_job.snapshot_path(first.snapshot_id)).read_bytes()
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    fixed = row(2, "h200", metrics={"mean_ttft": 0.2, "mean_tpot": 0.01, "mean_e2el": 1.5})
    second, _ = make_run(tmp_path, FakeInfx(rows=[row(1), fixed]))
    assert second.status == "candidate" and second.snapshot_id != first.snapshot_id
    assert second.changes == {"added": [], "corrected": ["inferencex:2"], "removed": []}
    assert (root / infx_job.snapshot_path(first.snapshot_id)).read_bytes() == old_doc  # old pin stays readable
    read_snapshot(root, infx_job.snapshot_path(first.snapshot_id))


def test_removed_upstream_observation_is_reported(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    second, _ = make_run(tmp_path, FakeInfx(rows=[row(1)], availability=[avail("h100")]))
    assert second.changes["removed"] == ["inferencex:2"]


# --------------------------------------------------------------------------- failure keeps last qualified


@pytest.mark.parametrize(
    ("route", "response", "error"),
    [
        ("benchmarks", HttpResponse(503, {}, b""), "TransientExhaustedError"),
        ("benchmarks", HttpResponse(429, {"retry-after": "9999"}, b""), "RateLimitedError"),
        ("benchmarks", HttpResponse(200, {}, b'[{"id": 1'), "IncompleteAcquisitionError"),
        ("benchmarks", ok({"rows": []}), "IncompleteAcquisitionError"),
        ("workflow-info", HttpResponse(500, {}, b""), "TransientExhaustedError"),
    ],
)
def test_failures_write_nothing_and_retain_last_qualified(root, tmp_path, route, response, error):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    fake = FakeInfx(rows=[row(1), row(2, "h200", metrics={"mean_ttft": 9.0})])  # would otherwise be a new candidate
    fake.respond(route, response)
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and error in result.error
    assert tree(root) == before
    assert infx_job.load_state(root)["qualified"]["snapshot_id"] == first.snapshot_id


def test_transient_failure_then_success_still_produces_the_snapshot(root, tmp_path):
    fake = FakeInfx()
    fake.respond("benchmarks", HttpResponse(502, {}, b""), HttpResponse(429, {"retry-after": "2"}, b""),
                 ok(fake.rows))  # fmt: skip
    result, sleeps = make_run(tmp_path, fake)
    assert result.status == "candidate" and sleeps == [1, 2.0]


def test_incomplete_page_missing_declared_configuration_fails(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1)]))  # availability declares h200 too
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error and "absent" in result.error
    assert not (root / infx_job.STATE_FILE).exists()


def test_declared_but_empty_benchmarks_is_incomplete_not_valid_empty(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(rows=[]))
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error


def test_valid_empty_scope_is_not_a_failure_and_writes_nothing(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(availability=[]))
    assert result.status == "empty" and result.error is None and not (root / infx_job.STATE_FILE).exists()


def test_valid_empty_after_data_existed_is_a_contradiction(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    result, _ = make_run(tmp_path, FakeInfx(availability=[]))
    assert result.status == "failed" and "ContradictionError" in result.error


def test_model_filter_selecting_nothing_is_valid_empty(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(), models=["Other"])
    assert result.status == "empty"


# --------------------------------------------------------------------------- contradictions


def test_duplicate_id_with_different_bodies_is_a_contradiction(root, tmp_path):
    rows = [row(1), row(1, "h200")]
    result, _ = make_run(tmp_path, FakeInfx(rows=rows))
    assert result.status == "failed" and "ContradictionError" in result.error


def test_same_benchmark_across_as_of_dates_is_deduplicated_ignoring_curve_context(root, tmp_path):
    day2 = "2026-11-06"
    rows = {DATE: [row(1, curve_date=DATE, curve_workflow_run_id="1")],
            day2: [row(1, curve_date=day2, curve_workflow_run_id="2")]}  # fmt: skip
    availability = [avail("h100"), {**avail("h100"), "date": day2}]
    result, _ = make_run(tmp_path, FakeInfx(rows=rows, availability=availability))
    assert result.status == "candidate" and result.changes["added"] == ["inferencex:1"]
    _, members = read_snapshot(root, infx_job.load_state(root)["candidate"]["snapshot_path"])
    assert len(members) == 1


def test_row_for_a_different_model_is_a_contradiction(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1), row(2, "h200", model="other")]))
    assert result.status == "failed" and "ContradictionError" in result.error


def test_conflicting_head_sha_between_workflow_pages_is_a_contradiction(root, tmp_path):
    workflow = {
        **WORKFLOW,
        "runConfigs": [{"html_url": RUN, "head_sha": "a" * 40}, {"html_url": RUN, "head_sha": "b" * 40}],
    }
    result, _ = make_run(tmp_path, FakeInfx(workflow=workflow))
    assert result.status == "failed" and "head_sha" in result.error


def test_availability_lagging_benchmarks_only_warns(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(availability=[avail("h100")], rows=[row(1), row(2, "h200")]))
    assert result.status == "candidate" and result.warnings


def test_row_missing_required_fields_is_incomplete(root, tmp_path):
    bad = row(1)
    del bad["decode_tp"]
    result, _ = make_run(tmp_path, FakeInfx(rows=[bad, row(2, "h200")]))
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error


# --------------------------------------------------------------------------- staleness and alerts


@pytest.mark.parametrize(
    ("days", "level"), [(0, "ok"), (7.9, "ok"), (8, "warn"), (13.9, "warn"), (14, "stale"), (40, "stale")]
)
def test_staleness_thresholds_from_last_qualified(days, level):
    state = {
        "qualified": {"snapshot_id": "x", "qualified_at": (NOW - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    }
    assert staleness(state, NOW)["level"] == level
    assert staleness({"qualified": None}, NOW)["level"] == "never"


def test_alert_goes_to_slack_and_email_and_one_channel_failing_does_not_hide_the_other():
    sent = []

    def boom(_):
        raise OSError("slack down")

    failures = infx_job.alert("s", "t", slack=boom, mail=lambda subject, text: sent.append((subject, text)))
    assert sent == [("s", "t")] and failures == ["slack: slack down"]
    got = []
    assert infx_job.alert("s", "t", slack=got.append, mail=None) == ["email: not configured"] and got == ["s\nt"]


def test_cli_failure_alerts_owner_channels_and_exits_nonzero(root, tmp_path, monkeypatch):
    alerts = []
    monkeypatch.setattr(infx_job, "_env_alert", lambda subject, text: alerts.append(subject) or [])
    monkeypatch.setattr(infx_job, "run_ingestion", lambda *a, **k: infx_job.RunResult("failed", error="x"))
    assert infx_job.main(["run", "--root", str(root), "--cache-dir", str(tmp_path / "c")]) == 1
    assert alerts == ["ingestion failed"]


# --------------------------------------------------------------------------- evaluators never call InferenceX

EVALUATOR_MODULES = (
    "e2e",
    "loaders",
    "eval",
    "ensure",
    "fetch",
    "hygiene",
    "catalog",
    "validate",
    "parquet",
    "materialize",
)


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_evaluator_modules_do_not_import_the_ingestion_adapter_or_http_clients():
    data = Path(e2e_module.__file__).parent
    for name in EVALUATOR_MODULES:
        imports = _imports(data / f"{name}.py")
        assert not [i for i in imports if "infx_ingest" in i or "infx_job" in i], name
        assert not imports & {"urllib.request", "requests", "http.client", "socket"}, name


def test_evaluator_paths_make_zero_inferencex_calls(root, tmp_path, monkeypatch):
    make_run(tmp_path, FakeInfx())
    snapshot_file = infx_job.load_state(root)["candidate"]["snapshot_path"]
    calls = []
    monkeypatch.setattr(InfxClient, "get", lambda *a, **k: calls.append("get"))
    monkeypatch.setattr(infx_ingest, "urllib_transport", lambda *a, **k: calls.append("transport"))

    def no_network(*args, **kwargs):
        calls.append("socket")
        raise AssertionError("evaluator opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    _, members = read_snapshot(root, snapshot_file)
    for member in members:
        assert evaluation_points(load_e2e(root, member.manifest_path, config=None))
    assert calls == []
