// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

const state = {
  data: null,
  series: "aisimulate",
  sortKey: "model",
  sortDirection: "asc",
  filter: "",
};

const metricGrid = document.getElementById("metric-grid");
const matrixBody = document.getElementById("matrix-body");
const modelFilter = document.getElementById("model-filter");
const snapshotLine = document.getElementById("snapshot-line");
const provenanceContent = document.getElementById("provenance-content");
const tableNote = document.getElementById("table-note");
const errorBanner = document.getElementById("error-banner");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPercent(value) {
  return value == null || !Number.isFinite(value) ? "N/A" : `${value.toFixed(2)}%`;
}

function formatDate(value) {
  if (!value) return "unknown date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function metricCard(label, value, detail, accent = false) {
  return `
    <article class="metric-card">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value${accent ? " metric-accent" : ""}">${escapeHtml(value)}</div>
      <div class="metric-detail">${escapeHtml(detail)}</div>
    </article>`;
}

function renderSnapshot() {
  const { snapshot, scope } = state.data;
  snapshotLine.innerHTML = [
    `<strong>${escapeHtml(snapshot.release_tag)}</strong>`,
    `measured through ${escapeHtml(formatDate(scopeDate(snapshot.measurement_date_through)))}`,
    `AISimulate ${escapeHtml(snapshot.aisimulate_packages.aisimulate)}`,
  ].join("<span aria-hidden=\"true\">•</span>");

  provenanceContent.innerHTML = `
    <p>
      Measurements: <a href="${escapeHtml(snapshot.measurement_source_url)}">${escapeHtml(
        snapshot.measurement_source,
      )} ${escapeHtml(snapshot.release_tag)}</a><br />
      Measured through: ${escapeHtml(formatDate(scopeDate(snapshot.measurement_date_through)))}<br />
      AISimulate run completed: ${escapeHtml(formatDate(snapshot.aisimulate_completed_at))}<br />
      Packages: ${escapeHtml(
        Object.entries(snapshot.aisimulate_packages)
          .map(([name, version]) => `${name} ${version}`)
          .join(", "),
      )}
    </p>
    <p>Predictions input SHA-256</p>
    <code>${escapeHtml(snapshot.predictions_sha256)}</code>
    <p>AISimulate evidence SHA-256</p>
    <code>${escapeHtml(snapshot.aisimulate_sot_sha256)}</code>
  `;
}

function scopeDate(value) {
  return value && !value.includes("T") ? `${value}T00:00:00Z` : value;
}

function renderMetrics() {
  const totals = state.data.totals;
  const metrics = totals[state.series];
  const cards = [];
  if (state.series === "aisimulate") {
    const statuses = metrics.status_counts;
    cards.push(
      metricCard(
        "Evidence coverage",
        formatPercent(metrics.coverage_pct),
        `${metrics.points} successful of ${totals.rows} selected points; ${statuses.failed} failed`,
        true,
      ),
    );
  } else {
    cards.push(
      metricCard(
        "Matched points",
        String(metrics.points),
        `${totals.models} models across ${totals.gpu_skus.length} GPU SKUs`,
        true,
      ),
    );
  }
  cards.push(
    metricCard("TPOT MAPE", formatPercent(metrics.tpot_mape_pct), "Mean error across matched points"),
    metricCard("TTFT MAPE", formatPercent(metrics.ttft_mape_pct), "Mean error across matched points"),
    metricCard(
      "Curve shape",
      `${formatPercent(metrics.tpot_shape_error_pct)} TPOT`,
      `${formatPercent(metrics.ttft_shape_error_pct)} TTFT after per-topology normalization`,
    ),
  );
  metricGrid.innerHTML = cards.join("");
}

function sortValue(model) {
  const metrics = model[state.series];
  switch (state.sortKey) {
    case "coverage":
      return state.series === "aisimulate" ? metrics.coverage_pct : metrics.points / model.rows;
    case "tpot":
      return metrics.tpot_mape_pct;
    case "ttft":
      return metrics.ttft_mape_pct;
    case "tpotShape":
      return metrics.tpot_shape_error_pct;
    case "ttftShape":
      return metrics.ttft_shape_error_pct;
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
      : String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
  return state.sortDirection === "asc" ? compared : -compared;
}

function workloadCards(model) {
  return model.workloads
    .map((workload) => {
      const metrics = workload[state.series];
      const evidence =
        state.series === "aisimulate"
          ? `${metrics.points}/${workload.rows} successful (${formatPercent(metrics.coverage_pct)})`
          : `${metrics.points} matched points`;
      return `
        <div class="workload-card">
          <strong>${escapeHtml(workload.label)} · ${escapeHtml(workload.gpu_skus.join(", "))}</strong>
          <span>${escapeHtml(evidence)}</span>
          <span>TPOT MAPE ${escapeHtml(formatPercent(metrics.tpot_mape_pct))}</span>
          <span>TTFT MAPE ${escapeHtml(formatPercent(metrics.ttft_mape_pct))}</span>
        </div>`;
    })
    .join("");
}

function renderMatrix() {
  const normalizedFilter = state.filter.trim().toLowerCase();
  const models = state.data.models
    .filter((model) => model.model.toLowerCase().includes(normalizedFilter))
    .sort((left, right) => compareValues(sortValue(left), sortValue(right)));

  if (!models.length) {
    matrixBody.innerHTML = '<tr><td colspan="7" class="empty-cell">No matching models.</td></tr>';
    tableNote.textContent = "0 models shown";
    return;
  }

  matrixBody.innerHTML = models
    .map((model) => {
      const metrics = model[state.series];
      const evidence =
        state.series === "aisimulate"
          ? `${metrics.points}/${model.rows}`
          : `${metrics.points}`;
      const evidenceDetail =
        state.series === "aisimulate"
          ? `${formatPercent(metrics.coverage_pct)} successful`
          : "matched points";
      const failed = state.series === "aisimulate" ? metrics.status_counts.failed : 0;
      return `
        <tr>
          <td>
            <span class="model-name">${escapeHtml(model.model)}</span>
            <span class="model-meta">${escapeHtml(model.frameworks.join(", "))} · ${escapeHtml(
              model.precisions.join(", "),
            )}</span>
          </td>
          <td>
            <span class="evidence-value">${escapeHtml(evidence)}</span>
            <span class="evidence-meta${failed ? " status-warning" : ""}">${escapeHtml(
              `${evidenceDetail}${failed ? ` · ${failed} failed` : ""}`,
            )}</span>
          </td>
          <td><div class="tag-list">${model.gpu_skus
            .map((gpu) => `<span class="tag">${escapeHtml(gpu)}</span>`)
            .join("")}</div></td>
          <td class="metric-cell">${escapeHtml(formatPercent(metrics.tpot_mape_pct))}</td>
          <td class="metric-cell">${escapeHtml(formatPercent(metrics.ttft_mape_pct))}</td>
          <td class="metric-cell">${escapeHtml(formatPercent(metrics.tpot_shape_error_pct))}</td>
          <td class="metric-cell">${escapeHtml(formatPercent(metrics.ttft_shape_error_pct))}</td>
        </tr>
        <tr class="workload-row">
          <td colspan="7">
            <details>
              <summary>Show ${model.workloads.length} workload breakdown${
                model.workloads.length === 1 ? "" : "s"
              }</summary>
              <div class="workload-grid">${workloadCards(model)}</div>
            </details>
          </td>
        </tr>`;
    })
    .join("");
  tableNote.textContent = `${models.length} of ${state.data.models.length} models shown · ${state.data.scope.published_rows} single-node operating points in scope`;
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

function render() {
  renderMetrics();
  renderMatrix();
  renderSortState();
}

document.querySelectorAll("[data-series]").forEach((button) => {
  button.addEventListener("click", () => {
    state.series = button.dataset.series;
    document.querySelectorAll("[data-series]").forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("active", active);
      candidate.setAttribute("aria-pressed", String(active));
    });
    render();
  });
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

modelFilter.addEventListener("input", (event) => {
  state.filter = event.target.value;
  renderMatrix();
});

fetch("./summary.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((data) => {
    if (data.schema_version !== 1 || !Array.isArray(data.models)) {
      throw new Error("unsupported accuracy summary schema");
    }
    state.data = data;
    renderSnapshot();
    render();
  })
  .catch((error) => {
    matrixBody.innerHTML = '<tr><td colspan="7" class="empty-cell">Accuracy data unavailable.</td></tr>';
    errorBanner.hidden = false;
    errorBanner.textContent = `Could not load the published accuracy snapshot: ${error.message}`;
    snapshotLine.textContent = "Snapshot unavailable";
    provenanceContent.textContent = "Provenance unavailable";
  });
