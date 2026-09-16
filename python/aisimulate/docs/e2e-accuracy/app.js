// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const state = {
  data: null,
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
    </article>`;
}

function renderSummary() {
  const totals = state.data.totals;
  summaryGrid.innerHTML = [
    basicCard("Models", String(totals.models), true),
    basicCard("Points (AIC CLI)", totals.aic.points.toLocaleString()),
    basicCard("Points (AISim CLI)", totals.aisimulate.points.toLocaleString()),
    basicCard("GPU SKUs", String(totals.gpu_skus.length)),
    accuracyCard("AISim CLI (new) Error", totals.aisimulate, "aisimulate"),
    accuracyCard("AIC CLI (legacy) Error", totals.aic, "aic"),
  ].join("");
}

function renderSnapshot() {
  const { snapshot, scope, totals } = state.data;
  if (!isSafeHttpsUrl(snapshot.measurement_source_url)) {
    throw new Error("unsafe measurement source URL");
  }
  releaseLabel.textContent = `Measurements: ${snapshot.release_tag}`;
  multinodeLabel.textContent = scope.multinode === "included"
    ? "Multi-node predictions included"
    : `Exclude multi-node predictions (${scope.excluded_multinode_rows.toLocaleString()} hidden)`;
  identityLine.textContent = `GPU SKUs: ${totals.gpu_skus.join(", ")} · Precisions: ${totals.precisions.join(", ")}`;
  measurementSourceLink.href = snapshot.measurement_source_url;
  scopeClaim.textContent = scope.claim;
  provenanceContent.innerHTML = `
    <p>
      Measurements: <a href="${escapeHtml(snapshot.measurement_source_url)}">${escapeHtml(
        snapshot.measurement_source,
      )} ${escapeHtml(snapshot.release_tag)}</a><br />
      Measured through: ${escapeHtml(formatDate(snapshot.measurement_date_through))}<br />
      AISimulate run completed: ${escapeHtml(formatDate(snapshot.aisimulate_completed_at))}<br />
      Packages: ${escapeHtml(
        Object.entries(snapshot.aisimulate_packages)
          .map(([name, version]) => `${name} ${version}`)
          .join(", "),
      )}
    </p>
    <p>Evaluated revision: ${snapshot.evaluated_revision
      ? `<a href="https://github.com/ai-dynamo/aisimulate/commit/${escapeHtml(snapshot.evaluated_revision.commit_sha)}">${escapeHtml(snapshot.evaluated_revision.branch)} @ ${escapeHtml(snapshot.evaluated_revision.commit_sha.slice(0, 12))}</a>`
      : "Not recorded in this historical snapshot"}</p>
    <p>Legacy AIC CLI source: ${snapshot.aic_source
      ? `<a href="${snapshot.aic_source.repository}/commit/${snapshot.aic_source.commit_sha}">AISimulate ${escapeHtml(snapshot.aic_source.branch)} @ ${snapshot.aic_source.commit_sha.slice(0, 12)}</a> (bundled aiconfigurator CLI)`
      : "Repository provenance was not recorded in this historical snapshot"}</p>
    ${snapshot.campaign ? `<p>Accuracy campaign: <a href="https://github.com/ai-dynamo/aisimulate/actions/runs/${escapeHtml(snapshot.campaign.run_id)}">GitHub Actions run</a> (advisory)<br />
      Selected operating points: ${escapeHtml(snapshot.campaign.selected)}; published comparison points: ${escapeHtml(snapshot.campaign.published)}.<br />
      Excluded before comparison: ${escapeHtml(JSON.stringify(snapshot.campaign.exclusion_reasons))}.<br />
      Prediction database versions: ${escapeHtml(snapshot.campaign.backend_versions.join(", "))}.<br />
      Policy: ${escapeHtml(snapshot.campaign.selection_policy)}; max_num_seqs=max(256, concurrency), max_num_batched_tokens=8192, enable_prefix_caching=False, aic_forward_model=op_level; unresolved recipes are excluded.</p>
      <code>Wheel SHA-256: ${escapeHtml(snapshot.campaign.wheel_sha256)}</code>
      <code>Dataset manifest SHA-256: ${escapeHtml(snapshot.campaign.dataset_sha256)}</code>` : ""}
    <p>Snapshot file source: ${state.branch.published_from_commit
      ? `<a href="https://github.com/ai-dynamo/aisimulate/blob/${state.branch.published_from_commit}/python/aisimulate/docs/e2e-accuracy/summary.json">${escapeHtml(state.branch.branch)} @ ${state.branch.published_from_commit.slice(0, 12)}</a> (publication source, not an evaluated revision)`
      : snapshot.campaign
        ? "Qualified e2e-accuracy-web artifact from the campaign above"
        : "Local preview; publication commit not recorded"}</p>
    <code>Predictions SHA-256: ${escapeHtml(snapshot.predictions_sha256)}</code>
    <code>AISimulate evidence SHA-256: ${escapeHtml(snapshot.aisimulate_sot_sha256)}</code>`;
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
  const parallelism = Object.entries(topology.parallelism)
    .filter(([, value]) => value != null)
    .map(([key, value]) => `${key.replace("_size", "").replace("attention_dp", "DP").toUpperCase()} ${value}`)
    .join(" · ");
  return `${topology.precision} · ${topology.framework} · ${topology.serving} · ${parallelism} · ${topology.spec_method} · ${topology.id.slice(0, 6)}`;
}

function coverageText(item) {
  const counts = item.aisimulate.status_counts;
  return `${item.aisimulate.points}/${item.rows} successful replay points · ${counts.unsupported} unsupported · ${counts.failed} failed`;
}

function errorBars(item) {
  const series = [
    ["AISim CLI TPOT", item.aisimulate.tpot_mape_pct, "aisimulate"],
    ["AIC CLI TPOT", item.aic.tpot_mape_pct, "aic"],
    ["AISim CLI TTFT", item.aisimulate.ttft_mape_pct, "aisimulate"],
    ["AIC CLI TTFT", item.aic.ttft_mape_pct, "aic"],
  ];
  const maximum = Math.max(1, ...series.map(([, value]) => value ?? 0));
  return `<div class="error-bars" aria-label="MAPE comparison">${series.map(([name, value, css]) => `
    <div class="error-bar"><span>${escapeHtml(name)}</span><span class="bar-track"><span class="bar ${css}" style="width:${(value ?? 0) / maximum * 100}%"></span></span><strong>${formatPercent(value)}</strong></div>`).join("")}</div>`;
}

function curveChart(topology, metric) {
  const points = topology.points;
  const names = ["measured", "aisimulate", "aic"];
  const values = points.flatMap((point) => names.map((name) => point[name][`${metric}_relative`]))
    .filter((value) => Number.isFinite(value));
  const maxY = Math.max(1, ...values) * 1.08;
  const minX = Math.log2(Math.min(...points.map((point) => point.concurrency)));
  const maxX = Math.log2(Math.max(...points.map((point) => point.concurrency)));
  const x = (concurrency) => 48 + (Math.log2(concurrency) - minX) / (maxX - minX || 1) * 330;
  const y = (value) => 172 - value / maxY * 145;
  const marks = names.map((name) => {
    let previous = null;
    return points.map((point) => {
      const value = point[name][`${metric}_relative`];
      if (!Number.isFinite(value)) { previous = null; return ""; }
      const current = [x(point.concurrency), y(value)];
      const line = previous ? `<line class="curve ${name}" x1="${previous[0]}" y1="${previous[1]}" x2="${current[0]}" y2="${current[1]}" />` : "";
      previous = current;
      return `${line}<circle class="point ${name}" cx="${current[0]}" cy="${current[1]}" r="3"><title>${name}: concurrency ${point.concurrency}, ${value.toFixed(3)}×</title></circle>`;
    }).join("");
  }).join("");
  return `<figure class="curve-chart"><figcaption>${metric.toUpperCase()} trend</figcaption>
    <svg viewBox="0 0 420 212" role="img" aria-label="${metric.toUpperCase()} normalized latency by concurrency; numeric values follow in the point table">
      <path class="axis" d="M48 22V172H390" />
      <text x="42" y="175" text-anchor="end">0</text>
      <text x="42" y="32" text-anchor="end">${maxY.toFixed(1)}×</text>
      <text x="48" y="190">${points[0].concurrency}</text>
      <text x="378" y="190" text-anchor="end">${points.at(-1).concurrency}</text>
      <text x="210" y="208" text-anchor="middle">Concurrency (log₂)</text>${marks}
    </svg></figure>`;
}

function pointTable(topology) {
  return `<details class="point-details" open><summary>Operating points (${topology.points.length})</summary>
    <div class="table-scroll" tabindex="0" role="region" aria-label="Operating point details"><table class="point-table">
    <caption>Relative TTFT / TPOT and absolute percentage errors. Ratios use measured latency at the lowest concurrency as 1×.</caption>
    <thead><tr><th>Concurrency</th><th>Replay status</th><th>Measured TTFT / TPOT</th><th>AISim CLI TTFT / TPOT</th><th>AIC CLI TTFT / TPOT</th><th>AISim CLI TTFT / TPOT error</th><th>AIC CLI TTFT / TPOT error</th></tr></thead>
    <tbody>${topology.points.map((point) => {
      const ratios = (name) => ["ttft", "tpot"].map((metric) => {
        const value = point[name][`${metric}_relative`];
        return value == null ? "—" : `${value.toFixed(3)}×`;
      }).join(" / ");
      const errors = (name) => `${formatPercent(point[name].ttft_error_pct)} / ${formatPercent(point[name].tpot_error_pct)}`;
      return `<tr><td>${point.concurrency}</td><td>${escapeHtml(point.status)}</td><td>${ratios("measured")}</td><td>${ratios("aisimulate")}</td><td>${ratios("aic")}</td><td>${errors("aisimulate")}</td><td>${errors("aic")}</td></tr>`;
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
    ${topologies.length ? `<label class="topology-control">Topology<select id="topology-select">${topologies.map((entry) => `<option value="${entry.id}"${entry.id === topology.id ? " selected" : ""}>${escapeHtml(topologyLabel(entry))}</option>`).join("")}</select></label>` : ""}
    <p class="coverage-text">${escapeHtml(coverageText(item))}</p>
    <p class="detail-scope">AISim CLI errors cover successful replays. AIC CLI errors cover all selected points.</p>
    ${errorBars(item)}
    ${topology ? `<div class="chart-legend"><span class="measured">● Measured</span><span class="aisimulate">● AISim CLI</span><span class="aic">● AIC CLI</span></div>
      <p class="detail-scope">Latency relative to the measured value at the lowest concurrency. Both predictors share that anchor; gaps indicate missing predictions.</p>
      ${curveChart(topology, "tpot")}${curveChart(topology, "ttft")}${pointTable(topology)}` : `<p class="detail-empty">This historical snapshot contains GPU aggregates only. Topology and concurrency details appear after its evidence is regenerated with the updated publisher.</p>`}`;
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

function validBranchName(branch) {
  return typeof branch === "string" && !branch.endsWith("/") &&
    (branch === "main" || /^release\/[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(branch));
}

function validRevision(revision) {
  return revision && typeof revision.commit_sha === "string" &&
    /^[0-9a-f]{40}$/.test(revision.commit_sha) && validBranchName(revision.branch);
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
  const statuses = ["success", "unsupported", "failed", "unknown"];
  const metrics = (item) => item && Number.isInteger(item.points) && item.points >= 0 &&
    ["ttft_mape_pct", "tpot_mape_pct", "ttft_shape_error_pct", "tpot_shape_error_pct"]
      .every((key) => item[key] === null || (Number.isFinite(item[key]) && item[key] >= 0));
  const aggregate = (item) => {
    if (!item || !metrics(item.aic) || !metrics(item.aisimulate) ||
      !Number.isInteger(item.rows) || item.rows <= 0 || !item.aisimulate.status_counts) return false;
    const counts = item.aisimulate.status_counts;
    return statuses.every((key) => Number.isInteger(counts[key]) && counts[key] >= 0) &&
      counts.success === item.aisimulate.points && Object.values(counts).reduce((sum, count) => sum + count, 0) === item.rows;
  };
  const topologyValid = (topology) => {
    if (!topology || !/^[0-9a-f]{16}$/.test(topology.id) || !aggregate(topology) ||
      !topology.parallelism || !Array.isArray(topology.points) || topology.points.length !== topology.rows ||
      !["framework", "precision", "serving", "spec_method"].every((key) => typeof topology[key] === "string")) return false;
    let previous = 0;
    const counts = { success: 0, unsupported: 0, failed: 0, unknown: 0 };
    return topology.points.every((point) => {
      if (!Number.isFinite(point.concurrency) || point.concurrency <= 0 || point.concurrency < previous ||
        !["success", "unsupported", "failed"].includes(point.status)) return false;
      previous = point.concurrency;
      counts[point.status] += 1;
      return ["measured", "aic", "aisimulate"].every((name) => ["ttft", "tpot"].every((metric) => {
        const value = point[name]?.[`${metric}_relative`];
        const error = point[name]?.[`${metric}_error_pct`];
        const missing = name === "aisimulate" && point.status !== "success";
        return missing ? value === null && error === null :
          Number.isFinite(value) && value >= 0 && (name === "measured" || Number.isFinite(error) && error >= 0);
      }));
    }) && statuses.every((key) => counts[key] === topology.aisimulate.status_counts[key]);
  };
  if (!data || data.schema_version !== 1 || !data.snapshot || !data.scope || !data.totals ||
    !isSafeHttpsUrl(data.snapshot.measurement_source_url) || !aggregate(data.totals) ||
    !Array.isArray(data.models) || data.models.some((model) =>
      !aggregate(model) || typeof model.model !== "string" || !Array.isArray(model.workloads) ||
      model.workloads.some((workload) => !aggregate(workload) || typeof workload.identity !== "string" ||
        !Array.isArray(workload.gpus) || workload.gpus.some((gpu) => !aggregate(gpu) || typeof gpu.gpu !== "string" ||
          (gpu.topologies !== undefined && (!Array.isArray(gpu.topologies) || !gpu.topologies.every(topologyValid))))))) {
    throw new Error("unsupported accuracy summary schema");
  }
  const revision = data.snapshot.evaluated_revision;
  if (revision != null && !validRevision(revision)) {
    throw new Error("invalid evaluated revision");
  }
  const aicSource = data.snapshot.aic_source;
  if ((revision != null || aicSource !== undefined) && (!aicSource ||
    aicSource.repository !== "https://github.com/ai-dynamo/aisimulate" ||
    !/^[0-9a-f]{40}$/.test(aicSource.commit_sha) || typeof aicSource.branch !== "string" ||
    aicSource.commit_sha !== data.snapshot.aic_commit_sha ||
    (revision && (aicSource.branch !== revision.branch || aicSource.commit_sha !== revision.commit_sha)))) {
    throw new Error("invalid legacy AIC CLI source");
  }
  const campaign = data.snapshot.campaign;
  if (campaign !== undefined && (!campaign || !revision || campaign.status !== "complete" ||
    campaign.advisory !== true || !/^[0-9]+$/.test(campaign.run_id) ||
    !/^[0-9a-f]{64}$/.test(campaign.wheel_sha256) || !/^[0-9a-f]{64}$/.test(campaign.dataset_sha256) ||
    campaign.commit_sha !== revision.commit_sha || campaign.branch !== revision.branch ||
    !Number.isInteger(campaign.selected) || campaign.selected < data.totals.rows ||
    campaign.published !== data.totals.rows || !Array.isArray(campaign.backend_versions) ||
    !campaign.backend_versions.every((version) => typeof version === "string") ||
    !campaign.exclusion_reasons || typeof campaign.exclusion_reasons !== "object")) {
    throw new Error("invalid accuracy campaign provenance");
  }
  return data;
}

function validateCatalog(catalog) {
  const seen = new Set();
  if (!catalog || catalog.schema_version !== 1 || catalog.default_branch !== "main" ||
    !Array.isArray(catalog.branches) || !catalog.branches.length || catalog.branches.some((entry) => {
      if (!entry || !validBranchName(entry.branch) ||
        seen.has(entry.branch) || !["evaluated", "inherited", "historical", "unavailable"].includes(entry.status) ||
        (entry.summary_path !== null && !/^(summary\.json|branches\/[0-9a-f]{16}\/summary\.json)$/.test(entry.summary_path)) ||
        (entry.status === "unavailable") !== (entry.summary_path === null) ||
        !(entry.published_from_commit === null || typeof entry.published_from_commit === "string" &&
          /^[0-9a-f]{40}$/.test(entry.published_from_commit))) return true;
      const revision = entry.evaluated_revision;
      if (["evaluated", "inherited"].includes(entry.status)) {
        if (!validRevision(revision) || (entry.status === "evaluated") !== (revision.branch === entry.branch)) return true;
      } else if (revision != null) return true;
      seen.add(entry.branch);
      return false;
    }) || !seen.has("main")) throw new Error("invalid accuracy branch catalog");
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
  history.replaceState(null, "", url);
  const permalink = document.getElementById("detail-permalink");
  if (permalink) permalink.href = url.href;
}

function clearSnapshot(message) {
  state.data = null;
  state.selection = null;
  state.topologyId = null;
  state.expandedWorkloads.clear();
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
    const evidence = snapshotEvidence(entry.branch, data.snapshot);
    const revision = data.snapshot.evaluated_revision;
    if (entry.status !== evidence.status ||
      revision && ["branch", "commit_sha"].some((key) => entry.evaluated_revision?.[key] !== revision[key])) {
      throw new Error("Branch catalog and snapshot provenance disagree");
    }
    state.data = data;
    if (revision) {
      branchStatus.textContent = revision.branch === entry.branch
        ? `${entry.branch} · evaluated commit ${revision.commit_sha.slice(0, 12)} (snapshot results; no live rerun)`
        : `${entry.branch} · inherited evidence from ${revision.branch} @ ${revision.commit_sha.slice(0, 12)}; this branch has not been evaluated.`;
    } else {
      branchStatus.textContent = `${entry.branch} · historical package snapshot; evaluated branch and commit were not recorded. These are not current branch accuracy results.`;
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
    renderSnapshot();
    renderSummary();
    renderMatrix();
    renderSortState();
    renderDrilldown();
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
  state.catalog = validateCatalog(previewData ? {
    schema_version: 1, default_branch: "main",
    branches: [{ branch: "main", ...snapshotEvidence("main", previewData.snapshot),
      summary_path: "summary.json", published_from_commit: null }],
  } : response.ok ? await response.json() : (() => { throw new Error(`HTTP ${response.status}`); })());
  branchSelect.innerHTML = state.catalog.branches.map(branchOption).join("");
  branchSelect.disabled = false;
  branchSelect.addEventListener("change", () => loadBranch(branchSelect.value));
  await loadBranch(new URL(location.href).searchParams.get("branch") || state.catalog.default_branch, true, previewData);
}
