// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const state = {
  data: null,
  rawData: null,
  tab: "overview",
  excludeOutliers: false,
  excludeAbnormal: false,
  excludeMultinode: false,
  hiddenSeries: new Set(),
  throughput: "output",
  view: "interactivity",
  catalog: null,
  branch: null,
  loadId: 0,
  selection: null,
  topologyId: null,
  sortKey: "model",
  sortDirection: "asc",
  expandedWorkloads: new Set(),
};

const summaryGrid = document.getElementById("summary-grid");
const matrixBody = document.getElementById("matrix-body");
const identityLine = document.getElementById("identity-line");
const releaseLabel = document.getElementById("release-label");
const scopeControl = document.getElementById("scope-control");
const scopeCheck = document.getElementById("scope-check");
const multinodeLabel = document.getElementById("multinode-label");
const measurementSourceLink = document.getElementById("measurement-source-link");
const scopeClaim = document.getElementById("scope-claim");
const provenanceContent = document.getElementById("provenance-content");
const errorBanner = document.getElementById("error-banner");
const themeToggle = document.getElementById("theme-toggle");
const branchSelect = document.getElementById("branch-select");
const branchStatus = document.getElementById("branch-status");
const downloadJson = document.getElementById("download-json");
const drilldown = document.getElementById("drilldown");
const matrixLayout = document.getElementById("matrix-layout");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPercent(value) {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(1)}%`;
}

function formatDate(value) {
  if (!value) return "unknown date";
  const normalized = value.includes("T") ? value : `${value}T00:00:00Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function isSafeHttpsUrl(value) {
  if (typeof value !== "string") return false;
  try {
    return new URL(value).protocol === "https:";
  } catch (_) {
    return false;
  }
}

function basicCard(label, value, accent = false) {
  return `
    <article class="summary-card">
      <div class="summary-label">${escapeHtml(label)}</div>
      <div class="summary-value${accent ? " accent" : ""}">${escapeHtml(value)}</div>
    </article>`;
}

function accuracyMetric(name, mape, shape) {
  return `
    <div class="accuracy-metric">
      <span class="accuracy-name">${escapeHtml(name)}</span>
      <div><span class="accuracy-number">${escapeHtml(formatPercent(mape))}</span><span class="accuracy-kind">MAPE</span></div>
      <div><span class="accuracy-number">${escapeHtml(formatPercent(shape))}</span><span class="accuracy-kind">shape</span></div>
    </div>`;
}

function accuracyCard(label, metrics, className) {
  return `
    <article class="summary-card accuracy-card ${escapeHtml(className)}">
      <div class="summary-label">${escapeHtml(label)}</div>
      <div class="accuracy-grid">
        ${accuracyMetric("TPOT", metrics.tpot_mape_pct, metrics.tpot_shape_error_pct)}
        ${accuracyMetric("TTFT", metrics.ttft_mape_pct, metrics.ttft_shape_error_pct)}
      </div>
      ${metrics.points === 0 ? '<p class="detail-scope">No included predictions in this snapshot or filter selection. Missing results are excluded from accuracy metrics.</p>' : ''}
    </article>`;
}

function renderSummary() {
  const totals = state.data.totals;
  summaryGrid.innerHTML = [
    basicCard("Models", String(totals.models), true),
    basicCard("AIC (legacy CLI) points", totals.aic.points.toLocaleString()),
    basicCard("AISim points", totals.aisimulate.points.toLocaleString()),
    basicCard("GPU SKUs", String(totals.gpu_skus.length)),
    accuracyCard("AISim Error · all configurations", totals.aisimulate, "aisimulate"),
    accuracyCard("AIC (legacy CLI) Error · all configurations", totals.aic, "aic"),
    ...Object.entries(totals.by_configuration_quality ?? {}).map(([quality, group]) =>
      `<article class="summary-card"><div class="summary-label">Configuration: ${escapeHtml(quality.replaceAll("_", " "))}</div>
      <p>AISim: ${group.aisimulate.points} points; TPOT / TTFT MAPE ${formatPercent(group.aisimulate.tpot_mape_pct)} / ${formatPercent(group.aisimulate.ttft_mape_pct)}</p>
      <p>AIC (legacy CLI): ${group.aic.points} points; TPOT / TTFT MAPE ${formatPercent(group.aic.tpot_mape_pct)} / ${formatPercent(group.aic.ttft_mape_pct)}</p></article>`),
  ].join("");
  const groups = new Map();
  const hardwareGroups = new Map();
  for (const model of state.data.models) for (const workload of model.workloads) for (const gpu of workload.gpus) {
    for (const topology of gpu.topologies || []) {
      if (!hardwareGroups.has(gpu.gpu)) hardwareGroups.set(gpu.gpu, []);
      hardwareGroups.get(gpu.gpu).push(topology);
      if (!groups.has(topology.serving)) groups.set(topology.serving, new Map());
      const frameworks = groups.get(topology.serving);
      if (!frameworks.has(topology.framework)) frameworks.set(topology.framework, []);
      frameworks.get(topology.framework).push(topology);
    }
  }
  const frameworkOrder = ["vllm", "sglang", "trtllm", "dynamo-vllm", "dynamo-sglang", "dynamo-trtllm"];
  document.getElementById("framework-summary").innerHTML = groups.size ? `<h2>AISim error by serving mode and framework</h2><div class="table-scroll"><table><thead><tr><th scope="col">Serving</th><th scope="col">Framework</th><th scope="col">Points</th><th scope="col">TPOT MAPE</th><th scope="col">TPOT shape</th><th scope="col">TTFT MAPE</th><th scope="col">TTFT shape</th></tr></thead>${[...groups].sort(([a], [b]) => a.localeCompare(b)).map(([serving, frameworks]) => {
    const rows = [...frameworks].sort(([a], [b]) =>
      (frameworkOrder.includes(a) ? frameworkOrder.indexOf(a) : frameworkOrder.length) -
      (frameworkOrder.includes(b) ? frameworkOrder.indexOf(b) : frameworkOrder.length) || a.localeCompare(b));
    return `<tbody>${rows.map(([framework, topologies], index) => {
      const m = aggregateTopologies(topologies).aisimulate;
      const servingLabel = {aggregated: "Agg", disaggregated: "Disagg"}[serving] || serving;
      return `<tr>${index === 0 ? `<th scope="rowgroup" rowspan="${rows.length}">${escapeHtml(servingLabel)}</th>` : ""}<th scope="row">${escapeHtml(framework.toUpperCase())}</th><td>${m.points}</td>${["tpot_mape_pct", "tpot_shape_error_pct", "ttft_mape_pct", "ttft_shape_error_pct"].map(f => `<td>${formatPercent(m[f])}</td>`).join("")}</tr>`;
    }).join("")}</tbody>`;
  }).join("")}</table></div>` : "";
  document.getElementById("hardware-summary").innerHTML = hardwareGroups.size ? `<h2>AISim error by hardware</h2><div class="table-scroll"><table><thead><tr><th scope="col">Hardware</th><th scope="col">Points</th><th scope="col">TPOT MAPE</th><th scope="col">TTFT MAPE</th></tr></thead><tbody>${[...hardwareGroups].sort(([a], [b]) => a.localeCompare(b)).map(([hardware, topologies]) => {
    const m = aggregateTopologies(topologies).aisimulate;
    return `<tr><th scope="row">${escapeHtml(hardware.toUpperCase())}</th><td>${m.points}</td><td>${formatPercent(m.tpot_mape_pct)}</td><td>${formatPercent(m.ttft_mape_pct)}</td></tr>`;
  }).join("")}</tbody></table></div>` : "";
}

function renderSnapshot() {
  const { snapshot, scope, totals } = state.data;
  if (!isSafeHttpsUrl(snapshot.measurement_source_url)) {
    throw new Error("unsafe measurement source URL");
  }
  releaseLabel.textContent = `Measurements: ${snapshot.release_tag}`;
  const includesMultinode = scope.multinode === "included";
  scopeCheck.hidden = true;
  scopeControl.title = includesMultinode
    ? "This snapshot includes multi-node predictions."
    : "This snapshot includes single-node predictions only.";
  multinodeLabel.textContent = includesMultinode
    ? "Multi-node predictions included"
    : `Snapshot: single-node only (${scope.excluded_multinode_rows.toLocaleString()} multi-node points not exported)`;
  identityLine.textContent = `GPU SKUs: ${totals.gpu_skus.join(", ")} · Precisions: ${totals.precisions.join(", ")}`;
  if (snapshot.campaign?.configuration) {
    const counts = snapshot.campaign.configuration.counts;
    identityLine.textContent += ` · Configuration: ${counts.verified ?? 0} verified, ${counts.estimated ?? 0} estimated (assumptions) · ${snapshot.campaign.selected - snapshot.campaign.published} excluded`;
  }
  measurementSourceLink.href = snapshot.measurement_source_url;
  scopeClaim.textContent = scope.claim;
  provenanceContent.innerHTML = `
    <p>
      Measurements: <a href="${escapeHtml(snapshot.measurement_source_url)}">${escapeHtml(
        snapshot.measurement_source,
      )} ${escapeHtml(snapshot.release_tag)}</a><br />
      Measured through: ${escapeHtml(formatDate(snapshot.measurement_date_through))}<br />
      AISim run completed: ${escapeHtml(formatDate(snapshot.aisimulate_completed_at))}<br />
      Packages: ${escapeHtml(
        Object.entries(snapshot.aisimulate_packages)
          .map(([name, version]) => `${name} ${version}`)
          .join(", "),
      )}
    </p>
    <p>Evaluated revision: ${snapshot.evaluated_revision
      ? `<a href="https://github.com/ai-dynamo/aisimulate/commit/${escapeHtml(snapshot.evaluated_revision.commit_sha)}">${escapeHtml(snapshot.evaluated_revision.branch)} @ ${escapeHtml(snapshot.evaluated_revision.commit_sha.slice(0, 12))}</a>`
      : "Not recorded in this historical snapshot"}</p>
    <p>AIC (legacy CLI) source: ${snapshot.aic_source
      ? `<a href="${snapshot.aic_source.repository}/commit/${snapshot.aic_source.commit_sha}">AISim ${escapeHtml(snapshot.aic_source.branch)} @ ${snapshot.aic_source.commit_sha.slice(0, 12)}</a> (${escapeHtml(snapshot.aic_source.cli_entry_point ?? "bundled aiconfigurator CLI")})`
      : "Repository provenance was not recorded in this historical snapshot"}</p>
    ${snapshot.campaign ? `<p>Accuracy campaign: <a href="https://github.com/ai-dynamo/aisimulate/actions/runs/${escapeHtml(snapshot.campaign.run_id)}">GitHub Actions run</a> (advisory)<br />
      Selected operating points: ${escapeHtml(snapshot.campaign.selected)}; published comparison points: ${escapeHtml(snapshot.campaign.published)}.<br />
      ${snapshot.campaign.configuration ? `Configuration evidence: ${snapshot.campaign.configuration.counts.verified ?? 0} verified; ${snapshot.campaign.configuration.counts.estimated ?? 0} estimated (assumptions, not verified historical settings).<br />` : ""}
      Excluded before comparison: ${escapeHtml(JSON.stringify(snapshot.campaign.exclusion_reasons))}.<br />
      Prediction database versions: ${escapeHtml(snapshot.campaign.backend_versions.join(", "))}.<br />
      Policy: ${escapeHtml(snapshot.campaign.selection_policy)}; ${snapshot.campaign.selection_policy === "gym-resolved-config-v2" ? "source-resolved serving settings and workload; independent estimate and replay outcomes" : "max_num_seqs=max(256, concurrency), max_num_batched_tokens=8192, enable_prefix_caching=False, aic_forward_model=op_level"}; unresolved recipes are excluded.</p>
      <code>Wheel SHA-256: ${escapeHtml(snapshot.campaign.wheel_sha256)}</code>
      <code>Dataset manifest SHA-256: ${escapeHtml(snapshot.campaign.dataset_sha256)}</code>` : ""}
    <p>Snapshot file source: ${state.branch.published_from_commit
      ? `<a href="https://github.com/ai-dynamo/aisimulate/blob/${state.branch.published_from_commit}/${state.branch.published_source_path || "python/aisimulate/docs/e2e-accuracy/summary.json"}">${escapeHtml(state.branch.branch)} @ ${state.branch.published_from_commit.slice(0, 12)}</a> (publication source, not an evaluated revision)`
      : snapshot.campaign
        ? "Qualified e2e-accuracy-web artifact from the campaign above"
        : "Local preview; publication commit not recorded"}</p>
    <code>Predictions SHA-256: ${escapeHtml(snapshot.predictions_sha256)}</code>
    <code>AISim evidence SHA-256: ${escapeHtml(snapshot.aisimulate_sot_sha256)}</code>`;
}

function sortValue(model) {
  switch (state.sortKey) {
    case "aicPoints":
      return model.aic.points;
    case "aisimulatePoints":
      return model.aisimulate.points;
    case "gpuSkus":
      return model.gpu_skus.length;
    case "hardware":
      return model.gpu_skus.join(",");
    case "precisions":
      return model.precisions.join(",");
    case "aisimulateTpot":
      return model.aisimulate.tpot_mape_pct;
    case "aisimulateTtft":
      return model.aisimulate.ttft_mape_pct;
    case "aicTpot":
      return model.aic.tpot_mape_pct;
    case "aicTtft":
      return model.aic.ttft_mape_pct;
    default:
      return model.model;
  }
}

function compareValues(left, right) {
  if (left == null) return right == null ? 0 : 1;
  if (right == null) return -1;
  const compared =
    typeof left === "number" && typeof right === "number"
      ? left - right
      : String(left).localeCompare(String(right), undefined, {
          numeric: true,
          sensitivity: "base",
        });
  return state.sortDirection === "asc" ? compared : -compared;
}

function metricCells(item) {
  return `
    <td class="points-cell">${escapeHtml(item.aic.points.toLocaleString())}</td>
    <td class="points-cell">${escapeHtml(item.aisimulate.points.toLocaleString())}</td>
    <td class="count-cell">${escapeHtml(item.gpu_skus.length.toLocaleString())}</td>
    <td class="mono">${escapeHtml(item.gpu_skus.join(", "))}</td>
    <td>${escapeHtml(item.precisions.join(", "))}</td>
    <td class="metric-cell">${escapeHtml(formatPercent(item.aisimulate.tpot_mape_pct))}</td>
    <td class="metric-cell">${escapeHtml(formatPercent(item.aisimulate.ttft_mape_pct))}</td>
    <td class="metric-cell">${escapeHtml(formatPercent(item.aic.tpot_mape_pct))}</td>
    <td class="metric-cell">${escapeHtml(formatPercent(item.aic.ttft_mape_pct))}</td>`;
}

function gpuRow(model, workload, gpu) {
  const item = { ...gpu, gpu_skus: [gpu.gpu] };
  const key = JSON.stringify([model.model, workload.identity, gpu.gpu]);
  const selected = state.selection === key;
  return `
    <tr class="gpu-row${selected ? " selected" : ""}">
      <td><button type="button" class="gpu-button" data-gpu-key="${escapeHtml(key)}"
        aria-expanded="${selected}" aria-controls="drilldown"
        aria-label="${selected ? "Hide" : "Show"} ${escapeHtml(model.model)} ${escapeHtml(workload.label)} ${escapeHtml(gpu.gpu)} details">
        <span aria-hidden="true">↳</span> ${escapeHtml(gpu.gpu)} <span aria-hidden="true">↗</span>
      </button></td>
      ${metricCells(item)}
    </tr>`;
}

function workloadRow(model, workload) {
  const key = JSON.stringify([model.model, workload.identity]);
  const expanded = state.expandedWorkloads.has(key);
  return `
    <tr
      class="workload-row"
      tabindex="0"
      role="button"
      aria-expanded="${String(expanded)}"
      data-workload-key="${escapeHtml(key)}"
      aria-label="${expanded ? "Collapse" : "Expand"} ${escapeHtml(model.model)} ${escapeHtml(
        workload.label,
      )} GPU rows"
    >
      <td><span class="workload-label"><span class="row-caret" aria-hidden="true">${
        expanded ? "▾" : "▸"
      }</span><span class="mono">${escapeHtml(workload.label)}</span></span></td>
      ${metricCells(workload)}
    </tr>
    ${expanded ? workload.gpus.map((gpu) => gpuRow(model, workload, gpu)).join("") : ""}`;
}

function modelRows(model) {
  return `
    <tr class="model-row">
      <td><span class="model-name">${escapeHtml(model.model)}</span></td>
      ${metricCells(model)}
    </tr>
    ${model.workloads.map((workload) => workloadRow(model, workload)).join("")}`;
}

function renderMatrix() {
  const models = [...state.data.models].sort((left, right) => {
    const compared = compareValues(sortValue(left), sortValue(right));
    return compared || left.model.localeCompare(right.model, undefined, { numeric: true });
  });
  matrixBody.innerHTML = models.length
    ? models.map(modelRows).join("")
    : '<tr><td colspan="10" class="empty-cell">No accuracy data available.</td></tr>';
}

function renderSortState() {
  document.querySelectorAll(".sort-button").forEach((button) => {
    const active = button.dataset.sort === state.sortKey;
    button.closest("th").setAttribute(
      "aria-sort",
      active ? (state.sortDirection === "asc" ? "ascending" : "descending") : "none",
    );
  });
}

function toggleWorkload(row) {
  const key = row.dataset.workloadKey;
  if (state.expandedWorkloads.has(key)) state.expandedWorkloads.delete(key);
  else state.expandedWorkloads.add(key);
  renderMatrix();
  focusData("workloadKey", key);
}

function updateThemeControl() {
  const dark = document.documentElement.dataset.theme !== "light";
  themeToggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
}

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("sm-theme", next);
  } catch (_) {
    // Keep the toggle usable when the browser blocks persistent storage.
  }
  updateThemeControl();
});

document.querySelectorAll(".sort-button").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.sort;
    if (state.sortKey === key) {
      state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = key;
      state.sortDirection = key === "model" ? "asc" : "desc";
    }
    if (state.data) renderMatrix();
    renderSortState();
  });
});

matrixBody.addEventListener("click", (event) => {
  const gpu = event.target.closest("button[data-gpu-key]");
  if (gpu) {
    const key = gpu.dataset.gpuKey;
    state.selection = state.selection === key ? null : key;
    state.topologyId = null;
    renderMatrix();
    renderDrilldown();
    updateLocation();
    focusData("gpuKey", key);
    return;
  }
  const row = event.target.closest("tr[data-workload-key]");
  if (row) toggleWorkload(row);
});

matrixBody.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest("tr[data-workload-key]");
  if (!row) return;
  event.preventDefault();
  toggleWorkload(row);
});

updateThemeControl();

initialize().catch(showError);

function focusData(property, value) {
  [...matrixBody.querySelectorAll(`[data-${property === "gpuKey" ? "gpu-key" : "workload-key"}]`)]
    .find((element) => element.dataset[property] === value)?.focus();
}

function selectedGpu() {
  if (!state.data || !state.selection) return null;
  const [modelName, workloadId, gpuName] = JSON.parse(state.selection);
  const model = state.data.models.find((item) => item.model === modelName);
  const workload = model?.workloads.find((item) => item.identity === workloadId);
  const gpu = workload?.gpus.find((item) => item.gpu === gpuName);
  return gpu ? { model, workload, gpu } : null;
}

function topologyLabel(topology) {
  const labels = {tp_size: "TP", pp_size: "PP", attention_dp_size: "DP", moe_ep_size: "EP", moe_tp_size: "MoE TP"};
  const parts = Object.entries(labels).filter(([key]) => topology.parallelism[key] != null &&
    (key === "tp_size" || topology.parallelism[key] !== 1))
    .map(([key, label]) => `${label} ${topology.parallelism[key]}`);
  if (topology.spec_method && topology.spec_method !== "none") parts.push(topology.spec_method);
  return parts.join(" · ") || "Default parallelism";
}

function topologyOptions(topologies) {
  const labels = topologies.map(topologyLabel);
  return topologies.map((t, i) => [t.id, labels.filter(label => label === labels[i]).length > 1
    ? `${labels[i]} · ${t.id.slice(0, 6)}` : labels[i]]);
}

function coverageText(item) {
  const counts = item.aisimulate.status_counts;
  return `${counts.success}/${item.rows} successful replay points · ${counts.unsupported} unsupported · ${counts.failed} failed`;
}

function errorBars(item) {
  const series = [
    ["AISim TPOT", item.aisimulate.tpot_mape_pct, "aisimulate"],
    ["AIC (legacy CLI) TPOT", item.aic.tpot_mape_pct, "aic"],
    ["AISim TTFT", item.aisimulate.ttft_mape_pct, "aisimulate"],
    ["AIC (legacy CLI) TTFT", item.aic.ttft_mape_pct, "aic"],
  ];
  const maximum = Math.max(1, ...series.map(([, value]) => value ?? 0));
  return `<div class="error-bars" aria-label="MAPE comparison">${series.map(([name, value, css]) => `
    <div class="error-bar"><span>${escapeHtml(name)}</span><span class="bar-track"><span class="bar ${css}" style="width:${(value ?? 0) / maximum * 100}%"></span></span><strong>${formatPercent(value)}</strong></div>`).join("")}</div>`;
}

function pointTable(topology) {
  const absolute = topology.points.every(p => p.measured.ttft_ms > 0 && p.measured.tpot_ms > 0);
  return `<details class="point-details" open><summary>Operating points (${topology.points.length})</summary>
    <div class="table-scroll" tabindex="0" role="region" aria-label="Operating point details"><table class="point-table">
    <caption>${absolute ? "TTFT / TPOT in milliseconds and absolute percentage errors." : "Relative TTFT / TPOT and absolute percentage errors. Ratios use measured latency at the lowest concurrency as 1×."}</caption>
    <thead><tr><th>Concurrency</th><th>Measured TTFT / TPOT</th><th>AISim TTFT / TPOT</th><th>AIC (legacy CLI) TTFT / TPOT</th><th>AISim TTFT / TPOT error</th><th>AIC (legacy CLI) TTFT / TPOT error</th><th>InfX CI run</th><th>Replay status</th><th>AISim prediction error</th></tr></thead>
    <tbody>${topology.points.map((point) => {
      const ratios = (name) => ["ttft", "tpot"].map((metric) => {
        const value = point[name][`${metric}_${absolute ? "ms" : "relative"}`];
        return value == null ? "—" : `${value.toFixed(3)}${absolute ? " ms" : "×"}`;
      }).join(" / ");
      const errors = (name) => `${formatPercent(point[name].ttft_error_pct)} / ${formatPercent(point[name].tpot_error_pct)}`;
      const run = point.infx_run_id ? `<a href="https://github.com/SemiAnalysisAI/InferenceX/actions/runs/${escapeHtml(point.infx_run_id)}" target="_blank" rel="noopener noreferrer">${escapeHtml(point.infx_run_id)}</a>` : "—";
      const failure = point.status === "success" ? "—" : point.aisim_error || "Not recorded";
      return `<tr><td>${point.concurrency}</td><td>${ratios("measured")}</td><td>${ratios("aisimulate")}</td><td>${ratios("aic")}</td><td>${errors("aisimulate")}</td><td>${errors("aic")}</td><td>${run}</td><td>${escapeHtml(point.status)}</td><td class="prediction-error">${escapeHtml(failure)}</td></tr>`;
    }).join("")}</tbody></table></div></details>`;
}

function renderDrilldown() {
  const selected = selectedGpu();
  drilldown.hidden = !selected;
  matrixLayout.classList.toggle("has-details", !!selected);
  if (!selected) { drilldown.innerHTML = ""; return; }
  const { model, workload, gpu } = selected;
  const topologies = gpu.topologies ?? [];
  const topology = topologies.find((item) => item.id === state.topologyId) ?? topologies[0];
  state.topologyId = topology?.id ?? null;
  const item = topology ?? gpu;
  drilldown.innerHTML = `
    <div class="detail-heading"><h2>${escapeHtml(model.model)} · ${escapeHtml(workload.label)} · ${escapeHtml(gpu.gpu)}</h2>
      <button type="button" id="close-details" aria-label="Close accuracy details">×</button></div>
    <p class="detail-scope">${escapeHtml(state.branch.branch)} · ${escapeHtml(state.data.snapshot.release_tag)}</p>
    <a id="detail-permalink" href="${escapeHtml(location.href)}" target="_blank" rel="noopener">Open this selection in a separate tab ↗</a>
    ${topologies.length ? `<div class="filter-toolbar">${["precision", "framework", "serving"].map(key => filterField(key, key, [...new Set(topologies.map(t => t[key]))].map(v => [v, v]), topology[key])).join("")}</div><label class="topology-control">Topology<select id="topology-select">${topologies.map((entry) => `<option value="${entry.id}"${entry.id === topology.id ? " selected" : ""}>${escapeHtml(topologyLabel(entry))}</option>`).join("")}</select></label>` : ""}
    <p class="coverage-text">${escapeHtml(coverageText(item))}</p>
    <p class="detail-scope">AISim errors cover successful replays. AIC (legacy CLI) errors cover successful baseline predictions.</p>
    ${topology ? "" : errorBars(item)}
    ${topology ? topologyContent(topology) : `<p class="detail-empty">This historical snapshot contains GPU aggregates only. Topology and concurrency details appear after its evidence is regenerated with the updated publisher.</p>`}`;
  bindCharts(drilldown, topology);
  drilldown.querySelectorAll("[data-filter]").forEach(select => select.addEventListener("change", () => {
    const key = select.dataset.filter;
    const candidates = topologies.filter(t => t[key] === select.value);
    state.topologyId = (candidates.find(t => ["precision", "framework", "serving"].every(k => k === key || t[k] === topology[k])) || candidates[0])?.id;
    refreshCharts();
  }));
  document.getElementById("close-details").addEventListener("click", () => {
    const key = state.selection;
    state.selection = null;
    state.topologyId = null;
    renderDrilldown();
    renderMatrix();
    updateLocation();
    focusData("gpuKey", key);
  });
  document.getElementById("topology-select")?.addEventListener("change", (event) => {
    state.topologyId = event.target.value;
    renderDrilldown();
    updateLocation();
    document.getElementById("topology-select").focus();
  });
}

function validBranchName(branch, preview = false) {
  return typeof branch === "string" && !branch.endsWith("/") &&
    (branch === "main" || (preview ? /^[A-Za-z0-9][A-Za-z0-9._/-]*$/ : /^release\/[A-Za-z0-9][A-Za-z0-9._/-]*$/).test(branch));
}

function validRevision(revision, preview = false) {
  return revision && typeof revision.commit_sha === "string" &&
    /^[0-9a-f]{40}$/.test(revision.commit_sha) && validBranchName(revision.branch, preview);
}

function snapshotEvidence(branch, snapshot) {
  const revision = snapshot.evaluated_revision;
  return revision
    ? { status: revision.branch === branch ? "evaluated" : "inherited", evaluated_revision: revision }
    : { status: "historical" };
}

function branchOption(entry) {
  const suffix = { historical: " — historical only", inherited: " — inherited evidence", unavailable: " — no snapshot" };
  return `<option value="${escapeHtml(entry.branch)}">${escapeHtml(entry.branch + (suffix[entry.status] || ""))}</option>`;
}

function validateSummary(data) {
  const research = data?.snapshot?.research_preview;
  const statuses = ["success", "unsupported", "failed", "unknown"];
  const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  const strings = (value) => Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === "string");
  const children = (items, parent, valid) => Array.isArray(items) && items.length > 0 && items.every(valid) &&
    items.reduce((sum, item) => sum + item.rows, 0) === parent.rows;
  const metrics = (item, rows) => object(item) && Number.isInteger(item.points) && item.points >= 0 && item.points <= rows &&
    ["ttft_mape_pct", "tpot_mape_pct", "ttft_shape_error_pct", "tpot_shape_error_pct"]
      .every((key) => item[key] === null || (Number.isFinite(item[key]) && item[key] >= 0));
  const aggregate = (item) => {
    if (!object(item) || !metrics(item.aic, item.rows) || !metrics(item.aisimulate, item.rows) ||
      !Number.isInteger(item.rows) || item.rows <= 0 || !object(item.aisimulate.status_counts)) return false;
    const counts = item.aisimulate.status_counts;
    return statuses.every((key) => Number.isInteger(counts[key]) && counts[key] >= 0) &&
      counts.success === item.aisimulate.points && Object.values(counts).reduce((sum, count) => sum + count, 0) === item.rows;
  };
  const topologyValid = (topology) => {
    if (!object(topology) || typeof topology.id !== "string" || !/^[0-9a-f]{16}$/.test(topology.id) || !aggregate(topology) ||
      !object(topology.parallelism) || !Array.isArray(topology.points) || topology.points.length !== topology.rows ||
      !["framework", "precision", "serving", "spec_method"].every((key) => typeof topology[key] === "string")) return false;
    if (topology.is_multinode !== undefined && (typeof topology.is_multinode !== "boolean" ||
      !Number.isInteger(topology.total_gpus) || topology.total_gpus <= 0)) return false;
    let previous = 0;
    let aicSuccesses = 0;
    const counts = { success: 0, unsupported: 0, failed: 0, unknown: 0 };
    return topology.points.every((point) => {
      if (!object(point) || !Number.isFinite(point.concurrency) || point.concurrency <= 0 || point.concurrency < previous ||
        !["success", "unsupported", "failed"].includes(point.status) ||
        !["success", "unsupported", "failed"].includes(point.aic_status === undefined ? "success" : point.aic_status) &&
          !(research && data.snapshot.aic_commit_sha === "not-run" && point.aic_status === "pending")) return false;
      if (point.configuration_quality !== undefined && !["verified", "estimated"].includes(point.configuration_quality)) return false;
      previous = point.concurrency;
      if (point.aisim_error != null && (point.status === "success" || typeof point.aisim_error !== "string" ||
        point.aisim_error.length === 0 || point.aisim_error.length > 2048)) return false;
      if (point.infx_run_id != null && (typeof point.infx_run_id !== "string" || !/^[1-9][0-9]*$/.test(point.infx_run_id))) return false;
      counts[point.status] += 1;
      if ((point.aic_status === undefined ? "success" : point.aic_status) === "success") aicSuccesses += 1;
      return ["measured", "aic", "aisimulate"].every((name) => object(point[name]) && ["ttft", "tpot"].every((metric) => {
        const missing = name === "aisimulate" && point.status !== "success" || name === "aic" && (point.aic_status ?? "success") !== "success";
        for (const field of ["ttft_ms", "tpot_ms", "e2e_ms", "output_per_gpu", "total_per_gpu"]) {
          const raw = point[name][field];
          if (raw !== undefined && raw !== null && (!Number.isFinite(raw) || raw <= 0 || missing)) return false;
        }
        const value = point[name]?.[`${metric}_relative`];
        const error = point[name]?.[`${metric}_error_pct`];
        return missing ? value === null && error === null :
          Number.isFinite(value) && value >= 0 && (name === "measured" || Number.isFinite(error) && error >= 0);
      }));
    }) && Object.keys(topology.aisimulate.status_counts).length === statuses.length &&
      statuses.every((key) => counts[key] === topology.aisimulate.status_counts[key]) &&
      aicSuccesses === topology.aic.points;
  };
  if (!object(data) || data.schema_version !== 1 || !object(data.snapshot) || !object(data.scope) ||
    typeof data.snapshot.release_tag !== "string" || data.snapshot.measurement_source_url !==
      `https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/${data.snapshot.release_tag}` ||
    !object(data.snapshot.aisimulate_packages) ||
    !["measurement_date_through", "aisimulate_completed_at"].every((key) =>
      data.snapshot[key] == null || typeof data.snapshot[key] === "string") ||
    !["included", "excluded"].includes(data.scope.multinode) ||
    !Number.isInteger(data.scope.excluded_multinode_rows) || data.scope.excluded_multinode_rows < 0 ||
    typeof data.scope.claim !== "string" || !aggregate(data.totals) ||
    !strings(data.totals.gpu_skus) || !strings(data.totals.precisions) ||
    !children(data.models, data.totals, (model) => aggregate(model) && typeof model.model === "string" &&
      strings(model.gpu_skus) && strings(model.precisions) &&
      children(model.workloads, model, (workload) => aggregate(workload) && typeof workload.identity === "string" &&
        typeof workload.label === "string" && strings(workload.gpu_skus) && strings(workload.precisions) &&
        children(workload.gpus, workload, (gpu) => aggregate(gpu) && typeof gpu.gpu === "string" &&
          strings(gpu.precisions) && (gpu.topologies === undefined || children(gpu.topologies, gpu, topologyValid))))) ||
    data.models.length !== data.totals.models) {
    throw new Error("unsupported accuracy summary schema");
  }
  const revision = data.snapshot.evaluated_revision;
  if (research !== undefined && (!object(research) || revision != null || data.snapshot.campaign != null ||
    !/^[0-9a-f]{40}$/.test(research.source_commit) ||
    !Number.isInteger(research.estimated_points) || research.estimated_points < 0 || research.estimated_points > data.totals.rows ||
    !Number.isInteger(research.estimated_successes) || research.estimated_successes < 0 ||
    research.estimated_successes > research.estimated_points || research.estimated_successes > data.totals.aisimulate.points)) {
    throw new Error("invalid local research preview provenance");
  }
  if (data.scope.preview === true && revision == null ||
    revision != null && (!object(revision) || !validRevision(revision, data.scope.preview === true))) {
    throw new Error("invalid evaluated revision");
  }
  const aicSource = data.snapshot.aic_source;
  if ((revision != null || aicSource !== undefined) && (!object(aicSource) ||
    aicSource.repository !== "https://github.com/ai-dynamo/aisimulate" ||
    typeof aicSource.commit_sha !== "string" || !/^[0-9a-f]{40}$/.test(aicSource.commit_sha) || typeof aicSource.branch !== "string" ||
    aicSource.commit_sha !== data.snapshot.aic_commit_sha ||
    (Object.hasOwn(aicSource, "cli_entry_point") &&
      !["aiconfigurator.main:main", "aisimulate.legacy_cli.entrypoint:main"].includes(aicSource.cli_entry_point)) ||
    (revision && (aicSource.branch !== revision.branch || aicSource.commit_sha !== revision.commit_sha)))) {
    throw new Error("invalid AIC (legacy CLI) source");
  }
  const campaign = data.snapshot.campaign;
  const exclusions = campaign?.exclusion_reasons;
  if (campaign?.configuration) {
    const configuration = campaign.configuration;
    const counts = {};
    data.models.forEach((model) => model.workloads.forEach((workload) => workload.gpus.forEach((gpu) =>
      (gpu.topologies ?? []).forEach((topology) => topology.points.forEach((point) => {
        const quality = point.configuration_quality;
        if (!["verified", "estimated"].includes(quality)) throw new Error("missing configuration quality");
        counts[quality] = (counts[quality] ?? 0) + 1;
      })))));
    if (!["verified", "coverage-experiment/1"].includes(configuration.profile) ||
      !object(configuration.counts) || Object.keys(configuration.counts).length !== Object.keys(counts).length ||
      Object.entries(counts).some(([key, count]) => configuration.counts[key] !== count) ||
      Object.values(counts).reduce((sum, count) => sum + count, 0) !== data.totals.rows ||
      (configuration.profile === "verified" && counts.estimated)) throw new Error("invalid configuration coverage");
  }

  const validExclusions = exclusions && typeof exclusions === "object" && !Array.isArray(exclusions) &&
    Object.keys(exclusions).every(key => ["recipe_required", "adapter_unsupported", "adapter_topology_mismatch", "baseline_failed", "source_unresolved", "database_unavailable"].includes(key)) &&
    Object.values(exclusions).every(value => Number.isInteger(value) && value >= 0);
  if (campaign !== undefined && (!campaign || !revision || campaign.status !== "complete" ||
    campaign.advisory !== true || !/^[0-9]+$/.test(campaign.run_id) ||
    !/^[0-9a-f]{64}$/.test(campaign.wheel_sha256) || !/^[0-9a-f]{64}$/.test(campaign.dataset_sha256) ||
    campaign.commit_sha !== revision.commit_sha || campaign.branch !== revision.branch ||
    !Number.isInteger(campaign.selected) || campaign.selected < data.totals.rows ||
    campaign.published !== data.totals.rows || !Array.isArray(campaign.backend_versions) ||
    !campaign.backend_versions.every((version) => typeof version === "string") ||
    !["latest-complete-config-run-v1", "gym-resolved-config-v2"].includes(campaign.selection_policy) ||
    !validExclusions)) {
    throw new Error("invalid accuracy campaign provenance");
  }
  const groups = data.totals.by_configuration_quality;
  if (groups !== undefined && (!object(groups) || !Object.keys(groups).length ||
    Object.entries(groups).some(([quality, group]) => !["verified", "estimated", "not_recorded"].includes(quality) ||
      !object(group) || !Number.isInteger(group.rows) || group.rows <= 0 ||
      !metrics(group.aic, group.rows) || !metrics(group.aisimulate, group.rows)) ||
    Object.values(groups).reduce((sum, group) => sum + group.rows, 0) !== data.totals.rows ||
    ["aic", "aisimulate"].some((name) => Object.values(groups).reduce((sum, group) => sum + group[name].points, 0) !== data.totals[name].points))) {
    throw new Error("invalid configuration quality metrics");
  }
  return data;
}

function validateCatalog(catalog) {
  const seen = new Set();
  const preview = catalog?.preview === true;
  if (!catalog || catalog.schema_version !== 1 ||
    (preview ? !validBranchName(catalog.default_branch, true) : catalog.default_branch !== "main") ||
    !Array.isArray(catalog.branches) || !catalog.branches.length || catalog.branches.some((entry) => {
      if (!entry || !validBranchName(entry.branch, preview) ||
        seen.has(entry.branch) || !["evaluated", "inherited", "historical", "unavailable"].includes(entry.status) ||
        (entry.summary_path !== null && !/^(summary\.json|branches\/[0-9a-f]{16}\/summary\.json)$/.test(entry.summary_path)) ||
        (entry.status === "unavailable") !== (entry.summary_path === null) ||
        !(entry.published_from_commit === null || typeof entry.published_from_commit === "string" &&
          /^[0-9a-f]{40}$/.test(entry.published_from_commit))) return true;
      if (entry.last_update && (!["success", "failed"].includes(entry.last_update.status) ||
        !/^[0-9]+$/.test(entry.last_update.run_id))) return true;
      const revision = entry.evaluated_revision;
      if (entry.published_source_path != null && !["pages/e2e-accuracy/summary.json", "python/aisimulate/docs/e2e-accuracy/summary.json"].includes(entry.published_source_path)) return true;
      if (["evaluated", "inherited"].includes(entry.status)) {
        if (!validRevision(revision, preview) || (entry.status === "evaluated") !== (revision.branch === entry.branch)) return true;
      } else if (revision != null) return true;
      seen.add(entry.branch);
      return false;
    }) || !seen.has(catalog.default_branch)) throw new Error("invalid accuracy branch catalog");
  return catalog;
}

function updateLocation() {
  const url = new URL(location.href);
  ["branch", "model", "workload", "gpu", "topology"].forEach((key) => url.searchParams.delete(key));
  if (state.branch) url.searchParams.set("branch", state.branch.branch);
  if (state.selection) {
    const [model, workload, gpu] = JSON.parse(state.selection);
    Object.entries({ model, workload, gpu, topology: state.topologyId }).forEach(([key, value]) => {
      if (value) url.searchParams.set(key, value);
    });
  }
  for (const [key, value] of Object.entries({tab: state.tab, outliers: state.excludeOutliers ? "1" : null,
    abnormal: state.excludeAbnormal ? "1" : null, multinode: state.excludeMultinode ? "1" : null,
    throughput: state.throughput, view: state.view, hidden: [...state.hiddenSeries].join(",")})) {
    if (value) url.searchParams.set(key, value); else url.searchParams.delete(key);
  }
  history.replaceState(null, "", url);
  const permalink = document.getElementById("detail-permalink");
  if (permalink) permalink.href = url.href;
}

function clearSnapshot(message) {
  document.getElementById("evidence-brief").textContent = message;
  state.rawData = null;
  state.data = null;
  state.selection = null;
  state.topologyId = null;
  state.expandedWorkloads.clear();
  document.getElementById("framework-summary").innerHTML = "";
  document.getElementById("hardware-summary").innerHTML = "";
  summaryGrid.innerHTML = `<div class="loading-card">${escapeHtml(message)}</div>`;
  matrixBody.innerHTML = `<tr><td colspan="10" class="empty-cell">${escapeHtml(message)}</td></tr>`;
  identityLine.textContent = "";
  releaseLabel.textContent = "";
  multinodeLabel.textContent = "";
  provenanceContent.textContent = "No snapshot selected.";
  scopeClaim.textContent = "No accuracy claim is available until a snapshot loads.";
  downloadJson.removeAttribute("href");
  downloadJson.setAttribute("aria-disabled", "true");
  measurementSourceLink.href = "https://github.com/SemiAnalysisAI/InferenceX-app/releases";
  errorBanner.hidden = true;
  renderDrilldown();
  document.getElementById("details-view").innerHTML = `<p role="status">${escapeHtml(message)}</p>`;
  document.getElementById("detail-filters").innerHTML = "";
}

function showError(error) {
  clearSnapshot("Accuracy data unavailable.");
  if (!state.catalog) {
    branchSelect.innerHTML = '<option value="">Unavailable</option>';
    branchSelect.disabled = true;
  }
  branchStatus.textContent = "Snapshot unavailable";
  errorBanner.hidden = false;
  errorBanner.textContent = `Could not load the published accuracy snapshot: ${error.message}`;
}

async function fetchSummary(path) {
  const response = await fetch(`./${path}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return validateSummary(await response.json());
}

async function loadBranch(branchName, restoreSelection = false, previewData = null) {
  const loadId = ++state.loadId;
  const params = new URL(location.href).searchParams;
  const entry = state.catalog.branches.find((item) => item.branch === branchName);
  clearSnapshot("Loading accuracy snapshot…");
  state.branch = entry ?? null;
  try {
    if (!entry) throw new Error(`Unknown branch: ${branchName}`);
    branchSelect.value = entry.branch;
    branchStatus.textContent = `Loading ${entry.branch}…`;
    if (!restoreSelection) updateLocation();
    if (entry.summary_path === null) {
      clearSnapshot("No published accuracy snapshot for this branch.");
      branchStatus.textContent = `${entry.branch}: no published accuracy snapshot.`;
      return;
    }
    const data = previewData || await fetchSummary(entry.summary_path);
    if (loadId !== state.loadId) return;
    if ((state.catalog.preview === true) !== (data.scope.preview === true)) {
      throw new Error("Preview and published accuracy evidence cannot be mixed");
    }
    const evidence = snapshotEvidence(entry.branch, data.snapshot);
    const revision = data.snapshot.evaluated_revision;
    if (entry.status !== evidence.status ||
      revision && ["branch", "commit_sha"].some((key) => entry.evaluated_revision?.[key] !== revision[key])) {
      throw new Error("Branch catalog and snapshot provenance disagree");
    }
    state.rawData = data;
    state.data = filterSnapshot(data);
    if (revision) {
      branchStatus.textContent = revision.branch === entry.branch
        ? `${entry.branch} · evaluated commit ${revision.commit_sha.slice(0, 12)} (snapshot results; no live rerun)`
        : `${entry.branch} · inherited evidence from ${revision.branch} @ ${revision.commit_sha.slice(0, 12)}; this branch has not been evaluated.`;
    } else {
      branchStatus.textContent = `${entry.branch} · historical package snapshot; evaluated branch and commit were not recorded. These are not current branch accuracy results.`;
    }
    if (data.scope.preview === true) branchStatus.textContent = "PR preview · " + branchStatus.textContent;
    branchStatus.textContent += ` Measurements: ${data.snapshot.release_tag}; evaluated ${formatDate(data.snapshot.aisimulate_completed_at)}.`;
    if (entry.last_update?.status === "failed") branchStatus.textContent += " Update failed; showing previous evidence. See the accuracy workflow for details.";
    const brief = document.getElementById("evidence-brief");
    brief.textContent = `${entry.branch} · ${revision ? revision.commit_sha.slice(0, 8) : "historical snapshot"} · evaluated ${formatDate(data.snapshot.aisimulate_completed_at)}` +
      (entry.status === "inherited" ? ` · inherited from ${revision.branch}` : "") +
      (entry.last_update?.status === "failed" ? " · Update failed — showing previous results" : "");
    brief.title = branchStatus.textContent;
    if (data.snapshot.research_preview) {
      const research = data.snapshot.research_preview;
      brief.textContent = `Local research preview · ${data.snapshot.release_tag} · AISim ${research.source_commit.slice(0, 8)} · ` +
        `${research.estimated_successes.toLocaleString()} successful predictions use estimated inputs · ` +
        (data.snapshot.aic_commit_sha === "not-run" ? "AIC (legacy CLI): not run · " : "") +
        "Not a qualified branch evaluation";
      branchStatus.textContent = brief.textContent;
      brief.title = brief.textContent;
    }
    downloadJson.href = `./${entry.summary_path}`;
    downloadJson.removeAttribute("aria-disabled");
    const linked = ["model", "workload", "gpu", "topology"].some((key) => params.has(key));
    if (restoreSelection && linked) {
      if (!["model", "workload", "gpu"].every((key) => params.has(key))) {
        throw new Error("The shared link is missing part of the GPU selection");
      }
      state.selection = JSON.stringify([params.get("model"), params.get("workload"), params.get("gpu")]);
      if (!selectedGpu()) throw new Error("The linked GPU selection is not present in this branch snapshot");
      const topologies = selectedGpu().gpu.topologies ?? [];
      if (params.has("topology") && !topologies.some((topology) => topology.id === params.get("topology"))) {
        throw new Error("The linked topology is not present in this branch snapshot");
      }
      state.expandedWorkloads.add(JSON.stringify([params.get("model"), params.get("workload")]));
      state.topologyId = params.get("topology");
    }
    restoreView(params);
    state.data = filterSnapshot(data);
    renderSnapshot();
    renderSummary();
    renderMatrix();
    renderSortState();
    renderDrilldown();
    renderView();
    updateLocation();
  } catch (error) {
    if (loadId === state.loadId) showError(error);
  }
}

async function initialize() {
  const response = await fetch("./branches.json");
  // Directly serving the source docs remains useful before a Pages build.
  // Only a missing catalog permits this legacy single-snapshot mode.
  const previewData = response.status === 404 ? await fetchSummary("summary.json") : null;
  const review = previewData?.scope.preview === true;
  const defaultBranch = review ? previewData.snapshot.evaluated_revision.branch : "main";
  state.catalog = validateCatalog(previewData ? {
    schema_version: 1, default_branch: defaultBranch, ...(review ? {preview: true} : {}),
    branches: [{ branch: defaultBranch, ...snapshotEvidence(defaultBranch, previewData.snapshot),
      summary_path: "summary.json", published_from_commit: null }],
  } : response.ok ? await response.json() : (() => { throw new Error(`HTTP ${response.status}`); })());
  branchSelect.innerHTML = state.catalog.branches.map(branchOption).join("");
  branchSelect.disabled = false;
  branchSelect.addEventListener("change", () => loadBranch(branchSelect.value, true));
  await loadBranch(new URL(location.href).searchParams.get("branch") || state.catalog.default_branch, true, previewData);
}

// The public dashboard operates on qualified snapshots, including historical ones.
let legendTimer;
const SERIES_NAMES = {measured: "Measured silicon", aisimulate: "AISim", aic: "AIC (legacy CLI)"};
const SERIES_COLORS = {measured: "#f59e0b", aisimulate: "#818cf8", aic: "#14b8a6"};
const average = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
const numeric = value => Number.isFinite(value) ? value.toFixed(2) : "—";

function abnormalPoint(points, index, metric) {
  const values = points.map(p => p.measured[`${metric}_relative`]);
  const v = values[index], before = values[index - 1], after = values[index + 1];
  return (before > 0 && after > 0 && ((v > before * 1.05 && v > after * 1.05) ||
    (before > v * 1.05 && after > v * 1.05))) || values.slice(index + 1).some(x => v > x * 1.05);
}

function aggregateTopologies(topologies) {
  const rows = topologies.flatMap(t => t.points);
  const result = {rows: rows.length};
  for (const name of ["aic", "aisimulate"]) {
    const acceptedPoints = new Set();
    const metrics = {};
    for (const metric of ["ttft", "tpot"]) {
      const errors = [], shapes = [];
      for (const topology of topologies) {
        const points = topology.points.filter((point, i) => {
          const error = point[name][`${metric}_error_pct`];
          return Number.isFinite(error) && (!state.excludeOutliers || error <= 100) &&
            (!state.excludeAbnormal || !abnormalPoint(topology.points, i, metric));
        });
        for (const point of points) { errors.push(point[name][`${metric}_error_pct`]); acceptedPoints.add(point); }
        const anchor = points[0];
        if (anchor) for (const point of points.slice(1)) {
          const silicon = point.measured[`${metric}_relative`] / anchor.measured[`${metric}_relative`];
          const predicted = point[name][`${metric}_relative`] / anchor[name][`${metric}_relative`];
          if (silicon > 0 && Number.isFinite(predicted)) shapes.push(Math.abs(predicted / silicon - 1) * 100);
        }
      }
      metrics[`${metric}_mape_pct`] = average(errors);
      metrics[`${metric}_shape_error_pct`] = average(shapes);
    }
    result[name] = {...metrics, points: acceptedPoints.size};
  }
  result.aisimulate.status_counts = Object.fromEntries(["success", "failed", "unsupported", "unknown"].map(
    status => [status, rows.filter(p => p.status === status).length]));
  return result;
}

function filterSnapshot(data) {
  if (!state.excludeMultinode && !state.excludeOutliers && !state.excludeAbnormal) return data;
  if (!data || !data.models.every(m => m.workloads.every(w => w.gpus.every(g => Array.isArray(g.topologies))))) return data;
  const models = [];
  const all = [];
  for (const model of data.models) {
    const workloads = [], modelTopologies = [];
    for (const workload of model.workloads) {
      const gpus = [], workloadTopologies = [];
      for (const gpu of workload.gpus) {
        const topologies = gpu.topologies.filter(t => !state.excludeMultinode || !t.is_multinode);
        if (!topologies.length) continue;
        gpus.push({...gpu, ...aggregateTopologies(topologies), topologies});
        workloadTopologies.push(...topologies);
      }
      if (!gpus.length) continue;
      workloads.push({...workload, ...aggregateTopologies(workloadTopologies), gpus, gpu_skus: gpus.map(g => g.gpu)});
      modelTopologies.push(...workloadTopologies);
    }
    if (!workloads.length) continue;
    models.push({...model, ...aggregateTopologies(modelTopologies), workloads,
      gpu_skus: [...new Set(workloads.flatMap(w => w.gpu_skus))]});
    all.push(...modelTopologies);
  }
  return {...data, models, totals: {...data.totals, ...aggregateTopologies(all), models: models.length,
    gpu_skus: [...new Set(models.flatMap(m => m.gpu_skus))]}};
}

function restoreView(params) {
  state.tab = params.get("tab") === "details" ? "details" : "overview";
  state.excludeOutliers = params.get("outliers") === "1";
  state.excludeAbnormal = params.get("abnormal") === "1";
  state.excludeMultinode = params.get("multinode") === "1";
  state.throughput = params.get("throughput") === "total" ? "total" : "output";
  state.view = ["interactivity", "e2e", "ttft"].includes(params.get("view")) ? params.get("view") : "interactivity";
  state.hiddenSeries = new Set((params.get("hidden") || "").split(",").filter(n => n in SERIES_NAMES));
}

function selectDefault() {
  if (selectedGpu()) return;
  const model = state.data?.models[0], workload = model?.workloads[0], gpu = workload?.gpus[0];
  if (gpu) {
    state.selection = JSON.stringify([model.model, workload.identity, gpu.gpu]);
    state.expandedWorkloads.add(JSON.stringify([model.model, workload.identity]));
    state.topologyId = null;
  }
}

function filterField(key, title, values, selected) {
  return `<label class="filter-${key}">${title}<select data-filter="${key}" title="${escapeHtml(values.find(([id]) => id === selected)?.[1] || title)}">${values.map(([id, label]) =>
    `<option value="${escapeHtml(id)}"${id === selected ? " selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select></label>`;
}

function renderView() {
  const details = state.tab === "details";
  document.getElementById("tab-overview").setAttribute("aria-pressed", String(!details));
  document.getElementById("tab-details").setAttribute("aria-pressed", String(details));
  matrixLayout.hidden = details;
  summaryGrid.hidden = details;
  document.getElementById("framework-summary").hidden = details;
  document.getElementById("hardware-summary").hidden = details;
  document.getElementById("details-view").hidden = !details;
  document.getElementById("detail-filters").hidden = !details;
  for (const [id, key] of [["outliers", "excludeOutliers"], ["abnormal", "excludeAbnormal"], ["multinode", "excludeMultinode"]]) {
    document.getElementById(`exclude-${id}`).checked = state[key];
  }
  if (details) { selectDefault(); renderDetails(); }
  else updateOutlierCounts(state.data?.models.flatMap(m => m.workloads.flatMap(w => w.gpus.flatMap(g => g.topologies || []))) || []);
  const historical = state.data && !state.data.models.every(m => m.workloads.every(w => w.gpus.every(g => g.topologies)));
  document.getElementById("filter-note").textContent = historical
    ? "This historical snapshot has aggregates only. Regenerate its evaluation to enable point filters and charts."
    : "Filters apply to each predictor and latency metric independently. Outliers remain visible in pink. Missing predictions remain gaps.";
  for (const id of ["outliers", "abnormal", "multinode"]) document.getElementById(`exclude-${id}`).disabled = !!historical;
}

function renderDetails() {
  const target = document.getElementById("details-view"), toolbar = document.getElementById("detail-filters");
  const selected = selectedGpu();
  if (!selected) { target.innerHTML = '<p role="status">No evaluated data matches this selection.</p>'; toolbar.innerHTML = ""; return; }
  const {model, workload, gpu} = selected;
  const topologies = gpu.topologies || [];
  const topology = topologies.find(t => t.id === state.topologyId) || [...topologies].sort((a,b) => b.points.length - a.points.length)[0];
  state.topologyId = topology?.id || null;
  updateOutlierCounts(topology ? [topology] : []);
  const options = (items, key) => items.map(v => [v[key], v[key]]);
  toolbar.innerHTML = filterField("model", "Model", options(state.data.models, "model"), model.model) +
    filterField("workload", "ISL / OSL", model.workloads.map(w => [w.identity, w.identity.replace(":", " / ")]), workload.identity) +
    filterField("gpu", "GPU", options(workload.gpus, "gpu"), gpu.gpu) +
    ["precision", "framework", "serving"].map(key => filterField(key, key[0].toUpperCase() + key.slice(1),
      [...new Set(topologies.map(t => t[key]))].sort().map(v => [v, v]), topology?.[key])).join("") +
    filterField("topology", "Parallelism", topologyOptions(topologies.filter(t => !topology ||
      ["precision", "framework", "serving"].every(k => t[k] === topology[k]))), topology?.id);
  toolbar.querySelectorAll("select").forEach(select => select.addEventListener("change", event => {
    const key = select.dataset.filter, value = event.target.value;
    let nextModel = model, nextWorkload = workload, nextGpu = gpu;
    if (key === "model") { nextModel = state.data.models.find(m => m.model === value); nextWorkload = nextModel.workloads.find(w => w.identity === workload.identity) || nextModel.workloads[0]; }
    if (key === "workload") nextWorkload = model.workloads.find(w => w.identity === value);
    nextGpu = nextWorkload.gpus.find(g => g.gpu === (key === "gpu" ? value : gpu.gpu)) || nextWorkload.gpus[0];
    state.selection = JSON.stringify([nextModel.model, nextWorkload.identity, nextGpu.gpu]);
    state.expandedWorkloads.add(JSON.stringify([nextModel.model, nextWorkload.identity]));
    if (key === "topology") state.topologyId = value;
    else if (["precision", "framework", "serving"].includes(key)) {
      const candidates = topologies.filter(t => t[key] === value);
      const compatible = candidates.find(t => ["precision", "framework", "serving"].every(k => k === key || t[k] === topology[k]));
      state.topologyId = (compatible || candidates[0])?.id;
    } else state.topologyId = null;
    renderDetails(); renderDrilldown(); renderMatrix(); updateLocation();
  }));
  target.innerHTML = topology ? `<h2>${escapeHtml(model.model)} · ${escapeHtml(gpu.gpu)}</h2>${topologyContent(topology)}` :
    '<p>This historical snapshot has no topology details. A new evaluation is required.</p>';
  bindCharts(target, topology);
}

function topologyContent(topology) {
  const stats = aggregateTopologies([topology]);
  return `<p>${escapeHtml(topologyLabel(topology))} · ${topology.total_gpus ?? "unknown"} GPUs</p>
    <p class="coverage-text">${escapeHtml(coverageText({...stats, aisimulate: {...stats.aisimulate, points: stats.aisimulate.status_counts.success}}))}</p>
    <div class="detail-cards">${accuracyCard("AISim error", stats.aisimulate, "aisimulate")}${accuracyCard("AIC (legacy CLI) error", stats.aic, "aic")}</div>
    <div class="chart-legend">${Object.entries(SERIES_NAMES).map(([key, name]) => `<button data-series="${key}" aria-pressed="${!state.hiddenSeries.has(key)}" style="color:${SERIES_COLORS[key]}"><span class="legend-line ${key}" aria-hidden="true"></span> ${name}</button>`).join("")}</div>
    <p class="detail-scope">Click a legend to hide a series; double-click to isolate it. Click a point for its configuration and values.</p>
    <div class="detail-charts">
      <section class="chart-panel" aria-label="Token latency"><h3>Token latency</h3>${metricChart(topology, "tpot")}</section>
      <section class="chart-panel" aria-label="Time to first token"><h3>Time to first token</h3>${metricChart(topology, "ttft")}</section>
      <section class="chart-panel" aria-label="Throughput"><h3>Throughput</h3>${metricChart(topology, "pareto")}
        <div class="chart-options"><label>Tokens <select data-chart="throughput">${["output", "total"].map(v => `<option${state.throughput === v ? " selected" : ""}>${v}</option>`).join("")}</select></label>
        <label>Compare against <select data-chart="view">${[["interactivity", "Interactivity"], ["e2e", "E2E latency"], ["ttft", "TTFT"]].map(([v,l]) => `<option value="${v}"${state.view === v ? " selected" : ""}>${l}</option>`).join("")}</select></label></div>
      </section>
    </div>${pointTable(topology)}`;
}

function metricChart(topology, metric) {
  const pareto = metric === "pareto";
  const absolute = topology.points.every(p => p.measured[`${metric}_ms`] > 0);
  const xLabel = !pareto ? "Concurrency" : {interactivity: "Interactivity (tok/s/user)", e2e: "E2E latency (ms)", ttft: "TTFT (ms)"}[state.view];
  const yLabel = pareto ? `${state.throughput === "total" ? "Total" : "Output"} throughput (tok/s/GPU)` : `${metric.toUpperCase()} (${absolute ? "ms" : "relative to measured anchor"})`;
  const series = Object.keys(SERIES_NAMES).filter(n => !state.hiddenSeries.has(n)).map(name => ({name,
    points: topology.points.map((p, i) => {
      const v = p[name], latencyMetric = pareto ? (state.view === "ttft" ? "ttft" : "tpot") : metric;
      const abnormal = abnormalPoint(topology.points, i, latencyMetric);
      return {i, x: !pareto ? p.concurrency : state.view === "interactivity" ? (v.tpot_ms > 0 ? 1000 / v.tpot_ms : null) : v[`${state.view}_ms`],
        y: pareto ? v[`${state.throughput}_per_gpu`] : v[`${metric}_${absolute ? "ms" : "relative"}`],
        exclude: state.excludeAbnormal && abnormal,
        outlier: v[`${latencyMetric}_error_pct`] > 100};
    })}));
  const valid = p => !p.exclude && Number.isFinite(p.x) && Number.isFinite(p.y);
  const points = series.flatMap(s => s.points.filter(valid));
  if (!points.length) return `<p class="detail-empty">${escapeHtml(yLabel)}: no values available for this snapshot or filter.</p>`;
  const maxX = Math.max(...points.map(p => p.x), 1), maxY = Math.max(...points.map(p => p.y), 0.01) * 1.08;
  const x = v => 65 + v / maxX * 465, y = v => 235 - v / maxY * 195;
  let svg = `<svg viewBox="0 0 560 290" role="img" aria-label="${escapeHtml(yLabel)} versus ${escapeHtml(xLabel)}"><text x="65" y="18" fill="currentColor">${escapeHtml(yLabel)}</text>`;
  for (let i = 0; i <= 4; i++) {
    const yy = maxY * i / 4, xx = maxX * i / 4;
    svg += `<path d="M65 ${y(yy)}H530" stroke="var(--border)"/><text x="58" y="${y(yy)+4}" text-anchor="end" fill="currentColor">${numeric(yy)}</text><text x="${x(xx)}" y="255" text-anchor="middle" fill="currentColor">${numeric(xx)}</text>`;
  }
  for (const {name, points: values} of series) {
    let path = "", connected = false;
    for (const p of values) {
      if (!valid(p)) { connected = false; continue; }
      path += `${connected ? "L" : "M"}${x(p.x)},${y(p.y)} `; connected = true;
    }
    svg += `<path d="${path}" stroke="${SERIES_COLORS[name]}" fill="none" stroke-width="2"${name === "measured" ? "" : ' stroke-dasharray="1 6" stroke-linecap="round"'}/>`;
    for (const p of values.filter(valid)) svg += `<circle class="point ${name}" cx="${x(p.x)}" cy="${y(p.y)}" r="4" fill="${p.outlier ? "#ec4899" : SERIES_COLORS[name]}" tabindex="0" role="button" data-point="${p.i}" aria-label="${SERIES_NAMES[name]}, concurrency ${topology.points[p.i].concurrency}, ${numeric(p.y)}"><title>${SERIES_NAMES[name]} · concurrency ${topology.points[p.i].concurrency} · ${numeric(p.x)}, ${numeric(p.y)}</title></circle>`;
  }
  return `<div class="metric-chart">${svg}<text x="290" y="282" text-anchor="middle" fill="currentColor">${escapeHtml(xLabel)}</text></svg></div>`;
}

function bindCharts(container, topology) {
  if (!topology) return;
  container.querySelectorAll("[data-series]").forEach(button => {
    button.addEventListener("click", () => { clearTimeout(legendTimer); legendTimer = setTimeout(() => { const n = button.dataset.series; state.hiddenSeries.has(n) ? state.hiddenSeries.delete(n) : state.hiddenSeries.add(n); refreshCharts(); }, 250); });
    button.addEventListener("dblclick", () => { clearTimeout(legendTimer); state.hiddenSeries = new Set(Object.keys(SERIES_NAMES).filter(n => n !== button.dataset.series)); refreshCharts(); });
  });
  container.querySelectorAll("[data-chart]").forEach(select => select.addEventListener("change", () => {
    state[select.dataset.chart] = select.value; refreshCharts();
  }));
  container.querySelectorAll("[data-point]").forEach(marker => {
    const open = () => {
      const point = topology.points[Number(marker.dataset.point)], selected = selectedGpu();
      document.getElementById("point-content").innerHTML = `<h2 id="point-title">Concurrency ${point.concurrency}</h2><p>${escapeHtml(topologyLabel(topology))}</p><p>${escapeHtml(selected.model.model)} · ${escapeHtml(selected.workload.identity)} · ${escapeHtml(selected.gpu.gpu)} · ${escapeHtml(point.status)}</p>
        <details open><summary>Recorded prediction configuration</summary><table><tbody>${Object.entries(point.configuration || {}).map(([key,value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value ?? "Not recorded")}</td></tr>`).join("")}</tbody></table><p>Silicon server knobs are not recorded by this campaign; prediction settings do not establish server-knob parity.</p></details>
        <table><thead><tr><th>Series</th><th>TTFT ms</th><th>TPOT ms</th><th>E2E ms</th><th>Output tok/s/GPU</th><th>Total tok/s/GPU</th></tr></thead><tbody>${Object.entries(SERIES_NAMES).map(([key,name]) => `<tr><th>${name}</th>${["ttft_ms", "tpot_ms", "e2e_ms", "output_per_gpu", "total_per_gpu"].map(f => `<td>${numeric(point[key][f])}</td>`).join("")}</tr>`).join("")}</tbody></table><p>— means this value was not recorded. Measured output throughput is unavailable when no output rate was recorded.</p>`;
      document.getElementById("point-dialog").showModal();
    };
    marker.addEventListener("click", open);
    marker.addEventListener("keydown", e => { if (["Enter", " "].includes(e.key)) {e.preventDefault(); open();} });
  });
}

function refreshCharts() { renderDrilldown(); if (state.tab === "details") renderDetails(); updateLocation(); }
for (const tab of ["overview", "details"]) document.getElementById(`tab-${tab}`).addEventListener("click", () => {
  state.tab = tab; renderView(); if (state.data) {renderMatrix(); renderDrilldown();} updateLocation();
});
for (const [id, key] of [["outliers", "excludeOutliers"], ["abnormal", "excludeAbnormal"], ["multinode", "excludeMultinode"]]) {
  document.getElementById(`exclude-${id}`).addEventListener("change", event => {
    state[key] = event.target.checked;
    if (!state.rawData) return;
    state.data = filterSnapshot(state.rawData);
    renderSummary(); renderMatrix(); renderDrilldown(); renderView(); updateLocation();
  });
}
document.getElementById("close-point").addEventListener("click", () => document.getElementById("point-dialog").close());

function updateOutlierCounts(topologies) {
  const rows = topologies.flatMap(t => t.points);
  const count = name => rows.filter(p => ["ttft", "tpot"].some(m => p[name][`${m}_error_pct`] > 100)).length;
  document.getElementById("outlier-count").textContent = `(AISim: ${count("aisimulate")}, AIC (legacy CLI): ${count("aic")})`;
}
