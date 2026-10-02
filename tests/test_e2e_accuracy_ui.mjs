// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../pages/e2e-accuracy/app.js", import.meta.url), "utf8");
const bootstrap = "initialize().catch(showError);";
assert.ok(source.includes(bootstrap), "Application bootstrap changed; update the UI test harness before executing it.");
const published = JSON.parse(readFileSync(new URL("../pages/e2e-accuracy/summary.json", import.meta.url), "utf8"));
const historical = structuredClone(published);
// Exercise the legacy contract even after the published snapshot is refreshed.
delete historical.snapshot.evaluated_revision;
delete historical.snapshot.aic_source;
for (const model of historical.models) for (const workload of model.workloads) {
  for (const gpu of workload.gpus) delete gpu.topologies;
}
const pathFor = (key) => `branches/${key.repeat(16)}/summary.json`;
const catalog = { schema_version: 1, default_branch: "main", branches: [
  { branch: "main", status: "historical", published_from_commit: "a".repeat(40), summary_path: pathFor("a") },
  { branch: "release/0.12.0", status: "historical", published_from_commit: "a".repeat(40), summary_path: pathFor("b") },
  { branch: "release/empty", status: "unavailable", published_from_commit: "b".repeat(40), summary_path: null },
] };
const response = (data, status = 200) => ({ ok: status === 200, status, json: async () => structuredClone(data) });

function harness(fetch = async () => response(historical), url = "https://example.com/aisimulate/e2e-accuracy/") {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      innerHTML: "", textContent: "", hidden: false, dataset: {}, attributes: {},
      events: {}, classes: new Set(),
      get classList() {
        const owner = this;
        return { toggle(name, on) { owner.classes[on ? "add" : "delete"](name); } };
      },
      addEventListener(name, callback) { this.events[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; if (name === "href") delete this.href; },
      querySelectorAll() { return []; }, focus() {},
    });
    return elements.get(id);
  };
  const location = { href: url };
  const context = vm.createContext({
    document: { getElementById: element, querySelectorAll: () => [], documentElement: { dataset: {} } },
    localStorage: { setItem() {} }, location,
    history: { replaceState(_state, _title, next) { location.href = String(next); } },
    fetch, URL, Intl, console,
  });
  vm.runInContext(source.replace(bootstrap, ""), context);
  return { element, context, location, run: (code) => vm.runInContext(code, context),
    set(name, value) { context[name] = structuredClone(value); } };
}

function setup(fetch, url) {
  const app = harness(fetch, url);
  app.set("catalogFixture", catalog);
  app.set("historicalFixture", historical);
  app.run("state.catalog = validateCatalog(catalogFixture)");
  return app;
}

function withEvaluation() {
  const data = structuredClone(historical);
  data.snapshot.evaluated_revision = { branch: "main", commit_sha: "d".repeat(40) };
  data.snapshot.aic_commit_sha = "d".repeat(40);
  data.snapshot.aic_source = { repository: "https://github.com/ai-dynamo/aisimulate", ...data.snapshot.evaluated_revision };
  return data;
}

function withTopology() {
  const data = structuredClone(historical);
  const model = data.models[0];
  const workload = model.workloads[0];
  const gpu = workload.gpus[0];
  data.models = [model];
  data.totals.models = 1;
  model.workloads = [workload];
  workload.gpus = [gpu];
  for (const item of [data.totals, model, workload, gpu]) {
    item.rows = 3;
    item.aic.points = 3;
    item.aisimulate.points = 2;
    item.aisimulate.coverage_pct = 200 / 3;
    item.aisimulate.status_counts = { success: 2, unsupported: 0, failed: 1, unknown: 0 };
    item.precisions = ["fp8"];
    if ("gpu_skus" in item) item.gpu_skus = [gpu.gpu];
  }
  const point = (concurrency, success) => ({
    concurrency, status: success ? "success" : "failed",
    measured: { ttft_relative: concurrency, tpot_relative: concurrency },
    aic: { ttft_relative: concurrency * 1.2, tpot_relative: concurrency * 1.1, ttft_error_pct: 20, tpot_error_pct: 10 },
    aisimulate: { ttft_relative: success ? concurrency * 0.9 : null, tpot_relative: success ? concurrency : null,
      ttft_error_pct: success ? 10 : null, tpot_error_pct: success ? 0 : null },
  });
  gpu.topologies = [{ ...structuredClone(gpu),
    id: "0123456789abcdef", framework: "vllm", precision: "fp8", serving: "aggregated",
    spec_method: "none", parallelism: { tp_size: 8, pp_size: 1 }, points: [point(1, true), point(2, false), point(4, true)] }];
  return data;
}

test("committed, historical, and topology snapshots pass validation and initialize", async () => {
  for (const data of [published, historical, withTopology()]) {
    const app = harness(async (path) => path === "./branches.json" ? response({}, 404) : response(data));
    app.set("valid", data);
    assert.doesNotThrow(() => app.run("validateSummary(valid)"));
    await app.run("initialize()");
    assert.equal(app.run("state.data.totals.rows"), data.totals.rows);
    assert.equal(app.element("error-banner").hidden, true);
    assert.equal(app.element("download-json").href, "./summary.json");
    assert.match(app.element("summary-grid").innerHTML, /AIC \(legacy CLI\) points/);
  }
});

test("operating points link public InfX runs and allow missing historical provenance", () => {
  const data = withTopology();
  const topology = data.models[0].workloads[0].gpus[0].topologies[0];
  topology.points[0].infx_run_id = "26696231118";
  topology.points[1].infx_run_id = null;
  const app = harness(async () => response(data));
  app.set("valid", data);
  app.set("topology", topology);
  assert.doesNotThrow(() => app.run("validateSummary(valid)"));
  const html = app.run("pointTable(topology)");
  assert.match(html, /<th>InfX CI run<\/th>/);
  assert.match(html, /href="https:\/\/github.com\/SemiAnalysisAI\/InferenceX\/actions\/runs\/26696231118"/);
  assert.equal((html.match(/actions\/runs\//g) || []).length, 1);
  assert.match(html, /<td>—<\/td>/);
  for (const id of ['123" onclick="bad', "../123", "0", 123]) {
    topology.points[0].infx_run_id = id;
    app.set("valid", data);
    assert.throws(() => app.run("validateSummary(valid)"));
  }
});

test("failed AIC predictions retain silicon and AISim chart values", () => {
  const data = withTopology();
  const topology = data.models[0].workloads[0].gpus[0].topologies[0];
  const point = topology.points[0];
  point.aic_status = "failed";
  for (const key of Object.keys(point.aic)) point.aic[key] = null;
  const model = data.models[0], workload = model.workloads[0], gpu = workload.gpus[0];
  for (const item of [data.totals, model, workload, gpu, topology]) item.aic.points = 2;
  const app = harness(async () => response(data));
  app.set("valid", data);
  assert.doesNotThrow(() => app.run("validateSummary(valid)"));
  for (const item of [data.totals, model, workload, gpu, topology]) item.aic.points = 3;
  app.set("valid", data);
  assert.throws(() => app.run("validateSummary(valid)"));
  for (const item of [data.totals, model, workload, gpu, topology]) item.aic.points = 2;
  point.aic.ttft_ms = 123;
  app.set("valid", data);
  assert.throws(() => app.run("validateSummary(valid)"));
});

test("operating point failures show escaped errors and explicit missing details", () => {
  const data = withTopology();
  const topology = data.models[0].workloads[0].gpus[0].topologies[0];
  topology.points[1].aisim_error = 'RuntimeError: missing <MoE> data & KV budget';
  const app = harness();
  app.set("valid", data);
  app.set("topology", topology);
  assert.doesNotThrow(() => app.run("validateSummary(valid)"));
  const html = app.run("pointTable(topology)");
  assert.match(html, /RuntimeError: missing &lt;MoE&gt; data &amp; KV budget/);
  assert.match(html, /class="prediction-error">—<\/td>/);
  delete topology.points[1].aisim_error;
  app.set("topology", topology);
  assert.match(app.run("pointTable(topology)"), /Not recorded/);
  for (const value of [42, "", "x".repeat(2049)]) {
    topology.points[1].aisim_error = value;
    app.set("valid", data);
    assert.throws(() => app.run("validateSummary(valid)"));
  }
  delete topology.points[1].aisim_error;
  topology.points[0].aisim_error = "stale failure";
  app.set("valid", data);
  assert.throws(() => app.run("validateSummary(valid)"));
});

test("research preview retains disagg with pending AIC and explicit estimated evidence", async () => {
  const data = withTopology();
  data.snapshot.aic_commit_sha = "not-run";
  data.snapshot.research_preview = {source_commit: "d".repeat(40), estimated_points: 3, estimated_successes: 2};
  const model = data.models[0], workload = model.workloads[0], gpu = workload.gpus[0], topology = gpu.topologies[0];
  topology.serving = "disaggregated";
  for (const item of [data.totals, model, workload, gpu, topology]) {
    item.aic = {points: 0, ttft_mape_pct: null, tpot_mape_pct: null, ttft_shape_error_pct: null, tpot_shape_error_pct: null};
  }
  for (const point of topology.points) {
    point.aic_status = "pending";
    point.configuration = {configuration_quality: "estimated"};
    for (const key of Object.keys(point.aic)) point.aic[key] = null;
  }
  const app = harness(async (path) => path === "./branches.json" ? response({}, 404) : response(data));
  await app.run("initialize()");
  assert.equal(app.element("error-banner").hidden, true);
  assert.match(app.element("evidence-brief").textContent, /Local research preview.*2 successful predictions use estimated inputs.*not run/);
  assert.equal(app.run('state.data.models[0].workloads[0].gpus[0].topologies[0].serving'), 'disaggregated');
  const invalid = structuredClone(data);
  delete invalid.snapshot.research_preview;
  app.set("invalid", invalid);
  assert.throws(() => app.run("validateSummary(invalid)"));
  invalid.snapshot.research_preview = data.snapshot.research_preview;
  invalid.snapshot.evaluated_revision = {branch: "main", commit_sha: "d".repeat(40)};
  assert.throws(() => {app.set("invalid", invalid); app.run("validateSummary(invalid)");}, /local research preview provenance/);
});

test("qualified campaign shows its run and exclusions and rejects unsafe provenance", async () => {
  const data = withEvaluation();
  const revision = data.snapshot.evaluated_revision;
  data.snapshot.campaign = {
    ...revision, status: "complete", advisory: true, run_id: "123",
    wheel_sha256: "a".repeat(64), dataset_sha256: "b".repeat(64),
    selected: data.totals.rows + 3, published: data.totals.rows,
    backend_versions: ["0.10.0"], exclusion_reasons: { adapter_unsupported: 3 },
    selection_policy: "latest-complete-config-run-v1",
  };
  const app = setup(async () => response(data));
  app.set("revisionFixture", revision);
  app.run('Object.assign(state.catalog.branches[0], {status: "evaluated", evaluated_revision: revisionFixture, published_from_commit: null})');
  await app.run('loadBranch("main")');
  assert.match(app.element("provenance-content").innerHTML, /actions\/runs\/123/);
  assert.match(app.element("provenance-content").innerHTML, /Qualified e2e-accuracy-web artifact/);
  assert.doesNotMatch(app.element("provenance-content").innerHTML, /aisimulate\/blob\//);
  assert.match(app.element("provenance-content").innerHTML, /max_num_batched_tokens=8192/);
  assert.doesNotMatch(app.element("provenance-content").innerHTML, /default scheduler/);
  assert.match(app.element("provenance-content").innerHTML, /adapter_unsupported/);
  for (const change of [
    { run_id: "123/../../evil" }, { selected: 0 }, { commit_sha: "e".repeat(40) },
    { advisory: false }, { published: data.totals.rows + 1 },
    { selection_policy: undefined }, { selection_policy: "unknown-policy" },
    ...[[], { adapter_unsupported: -1 }, { adapter_unsupported: "1" }, { adapter_unsupported: true }, { unexpected: 1 }]
      .map(exclusion_reasons => ({ exclusion_reasons })),
  ]) {
    const invalid = structuredClone(data);
    Object.assign(invalid.snapshot.campaign, change);
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /campaign provenance/);
  }
  const orphan = structuredClone(data);
  delete orphan.snapshot.evaluated_revision;
  app.set("orphan", orphan);
  assert.throws(() => app.run("validateSummary(orphan)"), /campaign provenance/);
});

test("legacy summary loads with historical provenance and branch-specific download", async () => {
  const app = setup();
  await app.run('loadBranch("release/0.12.0")');
  assert.match(app.element("branch-status").textContent, /historical.*not current branch accuracy/);
  assert.equal(app.element("download-json").href, `./${pathFor("b")}`);
  assert.match(app.element("summary-grid").innerHTML, /AISim/);
  assert.match(app.element("summary-grid").innerHTML, /AIC \(legacy CLI\)/);
  assert.match(app.element("provenance-content").innerHTML, /Repository provenance was not recorded/);
});

test("branch switching updates the multi-node scope label, check, and tooltip", async () => {
  const included = structuredClone(historical);
  included.scope.multinode = "included";
  included.scope.excluded_multinode_rows = 0;
  included.scope.raw_rows = included.scope.published_rows;
  const app = setup(async (path) => response(path === `./${pathFor("b")}` ? included : historical));
  await app.run('loadBranch("main")');
  assert.equal(app.element("scope-check").hidden, true);
  assert.match(app.element("multinode-label").textContent, /Snapshot: single-node only/);

  await app.run('loadBranch("release/0.12.0")');
  assert.equal(app.element("scope-check").hidden, true);
  assert.equal(app.element("multinode-label").textContent, "Multi-node predictions included");
  assert.equal(app.element("scope-control").title, "This snapshot includes multi-node predictions.");

  await app.run('loadBranch("main")');
  assert.equal(app.element("scope-check").hidden, true);
  assert.match(app.element("multinode-label").textContent, /Snapshot: single-node only.*not exported/);
  assert.equal(app.element("scope-control").title, "This snapshot includes single-node predictions only.");
});

test("bundled AIC (legacy CLI) provenance links to AISim and rejects another repository or revision", async () => {
  const data = withEvaluation();
  const app = setup(async () => response(data));
  app.set("revisionFixture", data.snapshot.evaluated_revision);
  app.run('Object.assign(state.catalog.branches[0], {status: "evaluated", evaluated_revision: revisionFixture})');
  await app.run('loadBranch("main")');
  assert.match(app.element("provenance-content").innerHTML, /AIC \(legacy CLI\) source:.*aisimulate\/commit\/d{40}/);
  assert.match(app.element("provenance-content").innerHTML, /bundled aiconfigurator CLI/);
  for (const change of [
    { repository: "https://github.com/ai-dynamo/aiconfigurator" },
    { repository: 'javascript:alert(1)' },
    { branch: "release/0.12.0" },
    { commit_sha: "e".repeat(40) },
  ]) {
    const invalid = structuredClone(data);
    Object.assign(invalid.snapshot.aic_source, change);
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /AIC \(legacy CLI\) source/);
  }
});

test("branch switching clears the old snapshot immediately and ignores late responses", async () => {
  const pending = [];
  const app = setup(() => new Promise((resolve) => pending.push(resolve)));
  const main = app.run('loadBranch("main")');
  const release = app.run('loadBranch("release/0.12.0")');
  assert.equal(app.run("state.data"), null);
  pending[1](response(historical)); await release;
  pending[0](response({})); await main;
  assert.equal(app.run("state.branch.branch"), "release/0.12.0");
  assert.equal(app.element("error-banner").hidden, true);
  assert.equal(app.element("download-json").href, `./${pathFor("b")}`);
});

test("failed, unavailable, and unknown branches cannot retain another branch's numbers", async () => {
  let healthy = true;
  const app = setup(async () => healthy ? response(historical) : response({}, 503));
  await app.run('loadBranch("main")');
  healthy = false;
  await app.run('loadBranch("release/0.12.0")');
  assert.equal(app.run("state.data"), null);
  assert.equal(app.element("download-json").href, undefined);
  assert.match(app.element("error-banner").textContent, /HTTP 503/);
  await app.run('loadBranch("release/empty")');
  assert.match(app.element("matrix-body").innerHTML, /No published accuracy snapshot/);
  assert.equal(app.element("error-banner").hidden, true);
  await app.run('loadBranch("release/missing")');
  assert.match(app.element("error-banner").textContent, /Unknown branch/);
});

test("catalog rejects external paths, traversal, duplicates, and invalid entries", () => {
  const app = setup();
  for (const path of ["https://example.com/summary.json", "//example.com/summary.json", "../summary.json", "branches/%2e%2e/summary.json", undefined]) {
    const invalid = structuredClone(catalog); invalid.branches[0].summary_path = path;
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateCatalog(invalid)"), /invalid accuracy branch catalog/);
  }
  const duplicate = structuredClone(catalog); duplicate.branches.push(duplicate.branches[0]);
  app.set("invalid", duplicate);
  assert.throws(() => app.run("validateCatalog(invalid)"), /invalid accuracy branch catalog/);
});

test("GPU selection opens historical error and coverage details and survives sorting", async () => {
  const app = setup(); await app.run('loadBranch("main")');
  app.run('state.selection = JSON.stringify([state.data.models[0].model, state.data.models[0].workloads[0].identity, state.data.models[0].workloads[0].gpus[0].gpu]); renderDrilldown(); updateLocation()');
  assert.equal(app.element("drilldown").hidden, false);
  assert.equal(app.element("matrix-layout").classes.has("has-details"), true);
  assert.match(app.element("drilldown").innerHTML, /successful replay points/);
  assert.match(app.element("drilldown").innerHTML, /historical snapshot contains GPU aggregates only/);
  assert.match(app.location.href, /branch=main.*model=.*workload=.*gpu=/);
  app.run('state.sortKey = "aicTpot"; state.sortDirection = "desc"; renderMatrix()');
  assert.equal(app.element("drilldown").hidden, false);
  await app.run('loadBranch("release/0.12.0")');
  assert.equal(app.element("drilldown").hidden, true);
  assert.doesNotMatch(app.location.href, /model=/);
  assert.equal(app.element("matrix-layout").classes.has("has-details"), false);
});

test("closing details clears the expanded layout and shared selection", async () => {
  const app = setup(); await app.run('loadBranch("main")');
  app.run('state.selection = JSON.stringify([state.data.models[0].model, state.data.models[0].workloads[0].identity, state.data.models[0].workloads[0].gpus[0].gpu]); renderDrilldown(); updateLocation()');
  app.element("close-details").events.click();
  assert.equal(app.element("drilldown").hidden, true);
  assert.equal(app.element("matrix-layout").classes.has("has-details"), false);
  assert.doesNotMatch(app.location.href, /model=/);
});

test("topology curves retain missing-point gaps and expose normalized numeric details", async () => {
  const data = withTopology();
  const app = setup(async () => response(data)); await app.run('loadBranch("main")');
  app.run('state.selection = JSON.stringify([state.data.models[0].model, state.data.models[0].workloads[0].identity, state.data.models[0].workloads[0].gpus[0].gpu]); renderDrilldown()');
  const html = app.element("drilldown").innerHTML;
  assert.match(html, /TPOT \(relative to measured anchor\)/);
  assert.match(html, /TTFT \(relative to measured anchor\)/);
  assert.match(html, /Operating points \(3\)/);
  assert.match(html, /1\.200×/);
  assert.match(html, /<td>failed<\/td>/);
  assert.equal((html.match(/<line class="curve aisimulate"/g) ?? []).length, 0);
  assert.equal((html.match(/<circle class="point aisimulate"/g) ?? []).length, 4);
});

test("changing topology updates the rendered points and shared selection in both directions", async () => {
  const data = withTopology();
  const model = data.models[0]; const workload = model.workloads[0]; const gpu = workload.gpus[0];
  const first = gpu.topologies[0];
  const second = structuredClone(first);
  second.id = "fedcba9876543210";
  second.parallelism = { tp_size: 4, pp_size: 2 };
  for (const point of second.points) point.concurrency *= 8;
  gpu.topologies.push(second);
  for (const item of [data.totals, model, workload, gpu]) {
    item.rows *= 2;
    item.aic.points *= 2;
    item.aisimulate.points *= 2;
    for (const status of Object.keys(item.aisimulate.status_counts)) item.aisimulate.status_counts[status] *= 2;
  }
  const app = setup(async () => response(data)); await app.run('loadBranch("main")');
  app.run('state.selection = JSON.stringify([state.data.models[0].model, state.data.models[0].workloads[0].identity, state.data.models[0].workloads[0].gpus[0].gpu]); renderDrilldown(); updateLocation()');
  assert.equal(app.run("state.topologyId"), first.id);
  assert.match(app.element("drilldown").innerHTML, /<tr><td>1<\/td><td>—<\/td><td>success<\/td>/);

  for (const topology of [second, first]) {
    app.element("topology-select").events.change({ target: { value: topology.id } });
    assert.equal(app.run("state.topologyId"), topology.id);
    const html = app.element("drilldown").innerHTML;
    assert.match(html, new RegExp(`<option value="${topology.id}" selected>TP ${topology.parallelism.tp_size}${topology.parallelism.pp_size > 1 ? ` · PP ${topology.parallelism.pp_size}` : ""}`));
    assert.match(html, new RegExp(`<tr><td>${topology.points[0].concurrency}</td><td>—</td><td>success</td>`));
    const other = topology === first ? second : first;
    assert.doesNotMatch(html, new RegExp(`<tr><td>${other.points[0].concurrency}</td>`));
    assert.equal(new URL(app.location.href).searchParams.get("topology"), topology.id);
    assert.equal(app.element("detail-permalink").href, app.location.href);
  }
});

test("invalid topology values and unsafe provenance fail before rendering", () => {
  const app = setup();
  const invalid = withTopology(); invalid.models[0].workloads[0].gpus[0].topologies[0].points[0].concurrency = "<img>";
  app.set("invalid", invalid);
  assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
  invalid.models[0].workloads[0].gpus[0].topologies[0].points[0].concurrency = 1;
  invalid.snapshot.measurement_source_url = "javascript:alert(1)";
  app.set("invalid", invalid);
  assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
});

test("aggregate point counts cannot exceed rows at any hierarchy level", () => {
  const app = setup();
  for (const select of [
    (data) => data.totals,
    (data) => data.models[0],
    (data) => data.models[0].workloads[0],
    (data) => data.models[0].workloads[0].gpus[0],
    (data) => data.models[0].workloads[0].gpus[0].topologies[0],
  ]) {
    const invalid = withTopology();
    const item = select(invalid);
    item.aic.points = item.rows + 1;
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
  }
});

test("summary requires nonempty hierarchy arrays and an accurate model count", () => {
  const app = setup();
  for (const corrupt of [
    (data) => { data.models = []; data.totals.models = 0; },
    (data) => { data.totals.models += 1; },
    (data) => { data.models[0].workloads = []; },
    (data) => { data.models[0].workloads[0].gpus = []; },
    (data) => { data.models[0].workloads[0].gpus[0].topologies = []; },
  ]) {
    const invalid = withTopology(); corrupt(invalid);
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
  }
});

test("hierarchy fields reject missing and non-array values except absent historical topologies", () => {
  const app = setup();
  for (const [field, select] of [
    ["models", (data) => data],
    ["workloads", (data) => data.models[0]],
    ["gpus", (data) => data.models[0].workloads[0]],
    ["topologies", (data) => data.models[0].workloads[0].gpus[0]],
    ["points", (data) => data.models[0].workloads[0].gpus[0].topologies[0]],
  ]) {
    for (const value of [null, {}, "invalid", undefined]) {
      const data = withTopology();
      if (value === undefined) delete select(data)[field];
      else select(data)[field] = value;
      app.set("data", data);
      if (field === "topologies" && value === undefined) {
        assert.doesNotThrow(() => app.run("validateSummary(data)"));
      } else {
        assert.throws(() => app.run("validateSummary(data)"), /schema/, field);
      }
    }
  }
});

for (const [level, depth] of [["model", 1], ["workload", 2], ["GPU", 3], ["topology", 4]]) {
  test(`${level} rows must cover their parent aggregate`, () => {
    const invalid = withTopology();
    const model = invalid.models[0]; const workload = model.workloads[0]; const gpu = workload.gpus[0];
    // Keep each aggregate and all ancestors internally consistent; only this child coverage is short.
    for (const parent of [invalid.totals, model, workload, gpu].slice(0, depth)) {
      parent.rows += 1;
      parent.aisimulate.status_counts.unknown += 1;
    }
    const app = setup(); app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
  });
}

test("summary validates the metadata, dimensions, and types required by rendering", () => {
  const app = setup();
  for (const [field, corrupt] of [
    ["release tag", (data) => {
      data.snapshot.release_tag = 123;
      data.snapshot.measurement_source_url = "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/123";
    }],
    ["release URL", (data) => { data.snapshot.measurement_source_url += "-other"; }],
    ["packages", (data) => { data.snapshot.aisimulate_packages = []; }],
    ["measurement date", (data) => { data.snapshot.measurement_date_through = 123; }],
    ["completion date", (data) => { data.snapshot.aisimulate_completed_at = {}; }],
    ["scope", (data) => { data.scope = []; }],
    ["multinode scope", (data) => { data.scope.multinode = "unknown"; }],
    ["excluded rows", (data) => { data.scope.excluded_multinode_rows = -1; }],
    ["scope claim", (data) => { delete data.scope.claim; }],
    ["total GPUs", (data) => { data.totals.gpu_skus = "gpu"; }],
    ["total precisions", (data) => { data.totals.precisions = []; }],
    ["model GPUs", (data) => { data.models[0].gpu_skus = []; }],
    ["model precisions", (data) => { data.models[0].precisions = [123]; }],
    ["workload label", (data) => { delete data.models[0].workloads[0].label; }],
    ["workload GPUs", (data) => { delete data.models[0].workloads[0].gpu_skus; }],
    ["workload precisions", (data) => { data.models[0].workloads[0].precisions = null; }],
    ["GPU precisions", (data) => { data.models[0].workloads[0].gpus[0].precisions = []; }],
    ["topology ID", (data) => { data.models[0].workloads[0].gpus[0].topologies[0].id = 1234567890123456; }],
    ["topology parallelism", (data) => { data.models[0].workloads[0].gpus[0].topologies[0].parallelism = []; }],
    ["topology statuses", (data) => { data.models[0].workloads[0].gpus[0].topologies[0].aisimulate.status_counts.extra = 0; }],
  ]) {
    const invalid = withTopology(); corrupt(invalid);
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /schema/, field);
  }
});

test("invalid branch data clears rendered accuracy and disables its download", async () => {
  for (const corrupt of [
    (data) => { data.totals.aic.points += 10000; },
    (data) => { data.models = []; },
    (data) => { data.models[0].workloads[0].gpus[0].topologies = []; },
    (data) => { data.snapshot.measurement_source_url += "-other"; },
  ]) {
    const invalid = withTopology(); corrupt(invalid);
    const app = harness(async (path) => response(path === "./branches.json" ? catalog :
      path === `./${pathFor("b")}` ? invalid : historical));
    await app.run("initialize()");
    assert.match(app.element("summary-grid").innerHTML, /AIC \(legacy CLI\) points/);
    assert.equal(app.element("download-json").href, `./${pathFor("a")}`);
    await app.run('loadBranch("release/0.12.0")');
    assert.equal(app.run("state.data"), null);
    assert.equal(app.element("error-banner").hidden, false);
    assert.match(app.element("error-banner").textContent, /schema/);
    assert.match(app.element("summary-grid").innerHTML, /Accuracy data unavailable/);
    assert.match(app.element("matrix-body").innerHTML, /Accuracy data unavailable/);
    assert.equal(app.element("identity-line").textContent, "");
    assert.equal(app.element("release-label").textContent, "");
    assert.equal(app.element("drilldown").hidden, true);
    assert.equal(app.element("download-json").href, undefined);
    assert.equal(app.element("download-json").attributes["aria-disabled"], "true");
  }
});

test("aggregate replay counts must match rows and successful points at every level", () => {
  const app = setup();
  for (const select of [
    (data) => data.totals,
    (data) => data.models[0],
    (data) => data.models[0].workloads[0],
    (data) => data.models[0].workloads[0].gpus[0],
    (data) => data.models[0].workloads[0].gpus[0].topologies[0],
  ]) {
    for (const corrupt of [
      (item) => { item.aisimulate.status_counts = { success: 0, unsupported: 0, failed: 0, unknown: 0 }; },
      (item) => { item.aisimulate.points = item.aisimulate.status_counts.success + 1; },
      (item) => { item.aisimulate.status_counts.failed += 1; },
    ]) {
      const data = withTopology();
      corrupt(select(data));
      app.set("invalid", data);
      assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
    }
  }
});

test("topology point statuses must match otherwise consistent aggregate counts", async () => {
  const data = withTopology();
  const topology = data.models[0].workloads[0].gpus[0].topologies[0];
  // Keep points, successes, and rows consistent, but misclassify the failed point.
  topology.aisimulate.status_counts.failed = 0;
  topology.aisimulate.status_counts.unsupported = 1;
  const app = setup(async () => response(data));
  app.set("invalid", data);
  assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
  await app.run('loadBranch("main")');
  assert.equal(app.run("state.data"), null);
  assert.equal(app.element("error-banner").hidden, false);
  assert.match(app.element("error-banner").textContent, /schema/);
  assert.doesNotMatch(app.element("summary-grid").innerHTML, /successful replay points/);
});

test("shared links restore branch, GPU and topology; stale links fail explicitly", async () => {
  const data = withTopology();
  const model = data.models[0]; const workload = model.workloads[0]; const gpu = workload.gpus[0];
  const params = new URLSearchParams({ branch: "release/0.12.0", model: model.model, workload: workload.identity, gpu: gpu.gpu, topology: gpu.topologies[0].id });
  const app = setup(async () => response(data), `https://example.com/e2e-accuracy/?${params}`);
  await app.run('loadBranch("release/0.12.0", true)');
  assert.equal(app.element("drilldown").hidden, false);
  assert.equal(app.run("state.topologyId"), gpu.topologies[0].id);
  params.set("topology", "missing"); app.location.href = `https://example.com/e2e-accuracy/?${params}`;
  await app.run('loadBranch("release/0.12.0", true)');
  assert.match(app.element("error-banner").textContent, /linked topology is not present/);
  assert.equal(app.run("state.data"), null);
});

test("catalog 404 permits direct source preview; other errors do not silently fall back", async () => {
  const app = setup(async (path) => path === "./branches.json" ? response({}, 404) : response(historical));
  await app.run("initialize()");
  assert.equal(app.run("state.catalog.branches.length"), 1);
  assert.equal(app.element("download-json").href, "./summary.json");
  const failing = setup(async () => response({}, 500));
  await assert.rejects(failing.run("initialize()"), /HTTP 500/);
});

test("catalog initialization failure replaces loading with an unavailable control", async () => {
  for (const result of [response({}, 503), response({})]) {
    const app = harness(async () => result);
    app.element("branch-select").innerHTML = "<option>Loading…</option>";
    await app.run("initialize().catch(showError)");
    assert.match(app.element("branch-select").innerHTML, /Unavailable/);
    assert.equal(app.element("branch-select").disabled, true);
    assert.equal(app.element("error-banner").hidden, false);
  }
});

test("partial GPU links and topology-only links fail explicitly", async () => {
  for (const query of ["model=X&gpu=Y", "workload=1024%3A1024", "topology=0123456789abcdef"]) {
    const app = setup(undefined, `https://example.com/e2e-accuracy/?branch=main&${query}`);
    await app.run('loadBranch("main", true)');
    assert.match(app.element("error-banner").textContent, /missing part of the GPU selection/);
    assert.equal(app.run("state.data"), null);
  }
});

test("inherited evaluated evidence identifies its original branch", async () => {
  const data = withEvaluation();
  const app = setup(async () => response(data));
  app.set("revisionFixture", data.snapshot.evaluated_revision);
  app.run('Object.assign(state.catalog.branches[1], {status: "inherited", evaluated_revision: revisionFixture})');
  await app.run('loadBranch("release/0.12.0")');
  assert.match(app.element("branch-status").textContent, /inherited evidence from main/);
  assert.match(app.element("branch-status").textContent, /this branch has not been evaluated/);
});

test("branch options distinguish evaluated, historical, inherited, and missing evidence", async () => {
  const labeled = structuredClone(catalog);
  labeled.branches.push(
    { branch: "release/evaluated", status: "evaluated", published_from_commit: null,
      summary_path: pathFor("c"), evaluated_revision: { branch: "release/evaluated", commit_sha: "d".repeat(40) } },
    { branch: "release/inherited", status: "inherited", published_from_commit: null,
      summary_path: pathFor("d"), evaluated_revision: { branch: "main", commit_sha: "d".repeat(40) } },
  );
  const app = harness(async (path) => response(path === "./branches.json" ? labeled : historical));
  await app.run("initialize()");
  const options = app.element("branch-select").innerHTML;
  assert.match(options, /value="release\/0.12.0">release\/0.12.0 — historical only<\/option>/);
  assert.match(options, /release\/inherited — inherited evidence<\/option>/);
  assert.match(options, /release\/empty — no snapshot<\/option>/);
  assert.match(options, /value="release\/evaluated">release\/evaluated<\/option>/);
  assert.match(app.element("provenance-content").innerHTML, /Snapshot file source:.*blob\/a{40}/);
});

test("catalog metadata rejects malformed revisions and contradictory status", () => {
  const app = setup();
  for (const change of [
    { published_from_commit: "short" }, { published_from_commit: false }, { published_from_commit: undefined },
    { evaluated_revision: { branch: "main", commit_sha: "d".repeat(40) } },
    { status: "evaluated" },
    { status: "evaluated", evaluated_revision: { branch: "release/0.12.0", commit_sha: "d".repeat(40) } },
    { status: "inherited", evaluated_revision: { branch: "main", commit_sha: "d".repeat(40) } },
    { status: "evaluated", evaluated_revision: { branch: "main", commit_sha: "short" } },
    { status: "inherited", evaluated_revision: { branch: "feature/private", commit_sha: "d".repeat(40) } },
  ]) {
    const invalid = structuredClone(catalog); Object.assign(invalid.branches[0], change);
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateCatalog(invalid)"), /invalid accuracy branch catalog/);
  }
});

test("catalog requires entries, a main branch, and a main default", () => {
  const app = setup();
  for (const change of [
    { branches: [] },
    { branches: catalog.branches.filter((entry) => entry.branch !== "main") },
    { default_branch: "release/0.12.0" },
  ]) {
    app.set("invalid", { ...catalog, ...change });
    assert.throws(() => app.run("validateCatalog(invalid)"), /invalid accuracy branch catalog/);
  }
});

test("topology points cannot decrease in concurrency", () => {
  const app = setup();
  const invalid = withTopology();
  invalid.models[0].workloads[0].gpus[0].topologies[0].points.reverse();
  app.set("invalid", invalid);
  assert.throws(() => app.run("validateSummary(invalid)"), /schema/);
});

test("summary rejects evaluated branches outside the exporter contract", () => {
  const app = setup();
  for (const branch of ["", "feature/private", "release/", "release/with space",
    "release/a/", "release/0.12.0/", "release/0.13.0/rc1/", 42, null]) {
    const invalid = withEvaluation();
    invalid.snapshot.evaluated_revision.branch = branch;
    invalid.snapshot.aic_source.branch = branch;
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /invalid evaluated revision/);
  }
  for (const revision of [false, 0, "main"]) {
    const invalid = structuredClone(historical); invalid.snapshot.evaluated_revision = revision;
    app.set("invalid", invalid);
    assert.throws(() => app.run("validateSummary(invalid)"), /invalid evaluated revision/);
  }
});

test("catalog rejects trailing slashes in branch names and evaluated revisions", () => {
  const app = setup();
  for (const branch of ["release/a/", "release/0.12.0/", "release/0.13.0/rc1/"]) {
    const invalidBranch = structuredClone(catalog);
    invalidBranch.branches[1].branch = branch;
    const invalidRevision = structuredClone(catalog);
    Object.assign(invalidRevision.branches[1], {
      status: "inherited", evaluated_revision: { branch, commit_sha: "d".repeat(40) },
    });
    for (const invalid of [invalidBranch, invalidRevision]) {
      app.set("invalid", invalid);
      assert.throws(() => app.run("validateCatalog(invalid)"), /invalid accuracy branch catalog/);
    }
  }
});

test("catalog and summary retain valid release names including nested branches", () => {
  const app = setup();
  for (const branch of ["main", "release/a", "release/0.12.0", "release/0.13.0/rc1"]) {
    const data = withEvaluation();
    data.snapshot.evaluated_revision.branch = branch;
    data.snapshot.aic_source.branch = branch;
    const valid = structuredClone(catalog);
    Object.assign(valid.branches[0], { status: branch === "main" ? "evaluated" : "inherited",
      evaluated_revision: data.snapshot.evaluated_revision });
    if (branch !== "main") valid.branches[1].branch = branch;
    app.set("valid", valid);
    app.set("data", data);
    assert.doesNotThrow(() => app.run("validateCatalog(valid)"));
    assert.doesNotThrow(() => app.run("validateSummary(data)"));
  }
});

test("catalog and loaded snapshot must agree before rendering", async () => {
  const data = withEvaluation();
  const app = setup(async () => response(data));
  for (const metadata of [
    { status: "historical" },
    { status: "evaluated", evaluated_revision: { branch: "main", commit_sha: "e".repeat(40) } },
    { status: "inherited", evaluated_revision: { branch: "release/0.12.0", commit_sha: "d".repeat(40) } },
  ]) {
    app.set("metadata", metadata);
    app.run("Object.assign(state.catalog.branches[0], metadata)");
    await app.run('loadBranch("main")');
    assert.match(app.element("error-banner").textContent, /catalog and snapshot provenance disagree/);
    assert.equal(app.run("state.data"), null);
    assert.equal(app.element("download-json").href, undefined);
  }
});

test("direct source preview derives its label from the actual evaluated snapshot", async () => {
  const data = withEvaluation();
  const app = harness(async (path) => path === "./branches.json" ? response({}, 404) : response(data));
  await app.run("initialize()");
  assert.equal(app.run("state.catalog.branches[0].status"), "evaluated");
  assert.equal(app.element("branch-select").innerHTML, '<option value="main">main</option>');
  assert.equal(app.element("download-json").href, "./summary.json");
});

test("evaluated snapshots require matching legacy CLI provenance", () => {
  const app = setup();
  const data = withEvaluation();
  delete data.snapshot.aic_source;
  app.set("invalid", data);
  assert.throws(() => app.run("validateSummary(invalid)"), /AIC \(legacy CLI\) source/);
  delete data.snapshot.evaluated_revision;
  app.set("historicalOnly", data);
  assert.doesNotThrow(() => app.run("validateSummary(historicalOnly)"));
});

test("baseline entry point is validated and displayed for both layouts", async () => {
  for (const entry of ["aiconfigurator.main:main", "aisimulate.legacy_cli.entrypoint:main"]) {
    const data = withEvaluation();
    data.snapshot.aic_source.cli_entry_point = entry;
    const app = setup(async () => response(data));
    app.set("revisionFixture", data.snapshot.evaluated_revision);
    app.run('Object.assign(state.catalog.branches[0], {status: "evaluated", evaluated_revision: revisionFixture})');
    await app.run('loadBranch("main")');
    assert.ok(app.element("provenance-content").innerHTML.includes(entry));
    for (const invalid of ["foreign.main:main", "<script>alert(1)</script>", null, 42]) {
      data.snapshot.aic_source.cli_entry_point = invalid;
      app.set("invalid", data);
      assert.throws(() => app.run("validateSummary(invalid)"), /AIC \(legacy CLI\) source/);
    }
  }
});


test("tabs preserve selected topology and shareable exclusion settings", async () => {
  const app = setup(async () => response(withTopology()));
  await app.run('loadBranch("main")');
  app.run('state.tab = "details"; renderView(); state.excludeOutliers = true; updateLocation()');
  assert.equal(app.element("matrix-layout").hidden, true);
  assert.equal(app.element("details-view").hidden, false);
  assert.match(app.location.href, /tab=details/);
  assert.match(app.location.href, /outliers=1/);
  const selection = app.run('state.selection');
  app.run('state.tab = "overview"; renderView()');
  assert.equal(app.run('state.selection'), selection);
  assert.equal(app.element("matrix-layout").hidden, false);
  assert.match(app.element("details-view").innerHTML, /Measured silicon/);
});

test("outlier filtering is per predictor and metric and keeps points visible", () => {
  const app = harness();
  const t = withTopology().models[0].workloads[0].gpus[0].topologies[0];
  t.points[0].aisimulate.ttft_error_pct = 150;
  t.points[0].aisimulate.tpot_error_pct = 100;
  app.set("topologyFixture", t);
  app.run('state.excludeOutliers = true');
  const stats = app.run('aggregateTopologies([topologyFixture])');
  assert.equal(stats.aisimulate.ttft_mape_pct, t.points[2].aisimulate.ttft_error_pct);
  assert.equal(stats.aisimulate.tpot_mape_pct, (100 + t.points[2].aisimulate.tpot_error_pct) / 2);
  assert.equal(stats.aic.points, 3);
  const chart = app.run('metricChart(topologyFixture, "ttft")');
  assert.match(chart, /#ec4899/);
  assert.match(chart, /data-point="0"/);
});

test("multi-node rows are included by default and can be excluded", () => {
  const app = harness();
  const data = withTopology();
  data.models[0].workloads[0].gpus[0].topologies[0].is_multinode = true;
  app.set("multiFixture", data);
  assert.equal(app.run('filterSnapshot(multiFixture).totals.rows'), 3);
  app.run('state.excludeMultinode = true');
  assert.equal(app.run('filterSnapshot(multiFixture).totals.rows'), 0);
});

test("clearing a snapshot clears Details as well as Overview", async () => {
  const app = setup(async () => response(withTopology()));
  await app.run('loadBranch("main")');
  app.run('state.tab = "details"; renderView(); clearSnapshot("Unavailable")');
  assert.match(app.element("details-view").innerHTML, /Unavailable/);
  assert.equal(app.element("detail-filters").innerHTML, "");
  assert.equal(app.element("hardware-summary").innerHTML, "");
});

test("hardware MAPE weights individual points across models and respects exclusions", () => {
  const data = withTopology();
  const first = data.models[0];
  first.workloads[0].gpus[0].gpu = "b200";
  const second = structuredClone(first);
  second.model = "Another model";
  const topology = second.workloads[0].gpus[0].topologies[0];
  topology.points = [topology.points[0]];
  topology.points[0].aisimulate.tpot_error_pct = 60;
  topology.points[0].aisimulate.ttft_error_pct = 160;
  topology.is_multinode = true;
  data.models.push(second);
  const app = harness();
  app.set("fixture", data);
  app.run("state.data = fixture; renderSummary()");
  const table = () => app.element("hardware-summary").innerHTML;
  assert.match(table(), /B200<\/th><td>3<\/td><td>20.0%<\/td><td>60.0%<\/td>/);
  app.run("state.excludeOutliers = true; state.data = filterSnapshot(fixture); renderSummary()");
  assert.match(table(), /B200<\/th><td>3<\/td><td>20.0%<\/td><td>10.0%<\/td>/);
  app.run("state.excludeMultinode = true; state.data = filterSnapshot(fixture); renderSummary()");
  assert.match(table(), /B200<\/th><td>2<\/td><td>0.0%<\/td><td>10.0%<\/td>/);
  app.run('state.tab = "details"; renderView()');
  assert.equal(app.element("hardware-summary").hidden, true);
});
