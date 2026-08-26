// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const state = {
  data: null,
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
const themeIcon = document.getElementById("theme-icon");

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
    basicCard("Data Points (AIC)", totals.aic.points.toLocaleString()),
    basicCard("Data Points (AISimulate)", totals.aisimulate.points.toLocaleString()),
    basicCard("GPU SKUs", String(totals.gpu_skus.length)),
    accuracyCard("Average AISimulate Error", totals.aisimulate, "aisimulate"),
    accuracyCard("Average AIC Error", totals.aic, "aic"),
  ].join("");
}

function renderSnapshot() {
  const { snapshot, scope, totals } = state.data;
  releaseLabel.textContent = `release: ${snapshot.release_tag}`;
  multinodeLabel.textContent = `Exclude multi-node predictions (${scope.excluded_multinode_rows.toLocaleString()} hidden)`;
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

function gpuRow(gpu) {
  const item = { ...gpu, gpu_skus: [gpu.gpu] };
  return `
    <tr class="gpu-row">
      <td><span class="gpu-label"><span aria-hidden="true">↳</span><span class="mono">${escapeHtml(
        gpu.gpu,
      )}</span></span></td>
      ${metricCells(item)}
    </tr>`;
}

function workloadRow(model, workload) {
  const key = `${model.model}::${workload.identity}`;
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
    ${expanded ? workload.gpus.map(gpuRow).join("") : ""}`;
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
}

function updateThemeControl() {
  const dark = document.documentElement.dataset.theme !== "light";
  themeIcon.textContent = dark ? "☀" : "☾";
  themeToggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
}

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("aisimulate-accuracy-theme", next);
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
    renderMatrix();
    renderSortState();
  });
});

matrixBody.addEventListener("click", (event) => {
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

fetch("./summary.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((data) => {
    if (
      data.schema_version !== 1 ||
      !Array.isArray(data.models) ||
      data.models.some((model) => model.workloads.some((workload) => !Array.isArray(workload.gpus)))
    ) {
      throw new Error("unsupported accuracy summary schema");
    }
    state.data = data;
    renderSnapshot();
    renderSummary();
    renderMatrix();
    renderSortState();
  })
  .catch((error) => {
    summaryGrid.innerHTML = '<div class="loading-card">Accuracy summary unavailable.</div>';
    matrixBody.innerHTML = '<tr><td colspan="10" class="empty-cell">Accuracy data unavailable.</td></tr>';
    errorBanner.hidden = false;
    errorBanner.textContent = `Could not load the published accuracy snapshot: ${error.message}`;
    identityLine.textContent = "Snapshot unavailable";
    releaseLabel.textContent = "release: unavailable";
    provenanceContent.textContent = "Provenance unavailable";
  });
