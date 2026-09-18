/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from AISim FPM Gym overview. See README.md for source and modifications.
 */
(() => {
  "use strict";
  const METHODS = ["warmup", "nowarmup", "regression"];
  const labels = { warmup: "FPM (KV warmup on)", nowarmup: "FPM (KV warmup off)", regression: "Regression" };
  const body = document.getElementById("overview-body");
  const branchSelect = document.getElementById("branch");
  const collapsed = new Set();
  const integer = (value) => Number(value).toLocaleString("en-US");
  const escape = (value) => String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  const hf = (snapshot, path) => `https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/blob/${snapshot.hf_revision}/${path.split("/").map(encodeURIComponent).join("/")}`;
  const link = (url, label) => `<a href="${escape(url)}" target="_blank" rel="noopener">${escape(label)}</a>`;
  let catalog, summary, request = 0;
  let sort = { key: "model", direction: 1 };

  const themeToggle = document.getElementById("theme-toggle");
  function updateThemeControl() {
    const dark = document.documentElement.dataset.theme !== "light";
    themeToggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
  themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("sm-theme", next); } catch (_) { /* Storage may be disabled. */ }
    updateThemeControl();
  });
  updateThemeControl();

  async function load(path) {
    const response = await fetch(path, { cache: "no-cache" });
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    return response.json();
  }

  function aggregate(rows, method) {
    const result = { predicted: 0, measured: 0, errors: 0, tuning: 0, weighted: 0, mape: null };
    for (const row of rows) {
      const metric = row.results[method]?.metrics.all;
      if (!metric) continue;
      result.predicted += metric.predicted_count;
      result.measured += metric.measured_count;
      result.errors += metric.error_count;
      result.tuning += metric.tuning_error_count;
      result.weighted += (metric.mape_pct ?? 0) * metric.predicted_count;
    }
    result.mape = result.predicted ? result.weighted / result.predicted : null;
    return result;
  }

  function cells(rows) {
    return ["regression", "warmup", "nowarmup"].map((method) => {
      const metric = aggregate(rows, method);
      const value = metric.mape === null ? "—" : `${metric.mape.toFixed(2)}%`;
      const tone = metric.mape === null ? "missing" : "";
      let note = metric.measured ? `${integer(metric.predicted)}/${integer(metric.measured)} predicted · ${(100 * metric.predicted / metric.measured).toFixed(1)}% coverage` : "Unavailable";
      if (metric.errors) note += ` · ${integer(metric.errors)} errors`;
      if (metric.tuning) note += ` · ${integer(metric.tuning)} tuning errors`;
      const result = rows.length === 1 ? rows[0].results[method] : null;
      if (result?.status === "no_fpm_input") note += " · No reviewed input";
      if (result?.status === "unsupported_predictor") note += " · Unsupported by this AISim revision";
      const evidence = result?.artifact ? link(hf(summary.snapshot, result.artifact.path), "FPM input ↗") : "";
      return `<td class="overview-method-cell ${tone}" data-label="${labels[method]}"><strong>${value}</strong><span>${escape(note)}</span>${evidence}</td>`;
    }).join("");
  }

  function groups() {
    const byModel = new Map();
    for (const row of summary.rows) {
      if (!byModel.has(row.model)) byModel.set(row.model, []);
      byModel.get(row.model).push(row);
    }
    const result = [...byModel].map(([model, rows]) => ({ model, rows,
      gpu: [...new Set(rows.map((row) => row.gpu))].sort().join(", "),
      framework: [...new Set(rows.map((row) => row.framework))].sort().join(", "),
      measurements: rows.reduce((sum, row) => sum + row.measurement_count, 0),
    }));
    const value = (group) => sort.key.startsWith("method:") ? aggregate(group.rows, sort.key.slice(7)).mape : group[sort.key];
    return result.sort((a, b) => {
      const left = value(a), right = value(b);
      if (left == null || right == null) return left == null && right == null ? a.model.localeCompare(b.model) : left == null ? 1 : -1;
      return sort.direction * (typeof left === "number" ? left - right : left.localeCompare(right)) || a.model.localeCompare(b.model);
    });
  }

  function render() {
    const models = groups();
    body.innerHTML = models.map((group) => {
      const expanded = !collapsed.has(group.model);
      const model = `<tr class="overview-model-row model-row"><th scope="rowgroup"><button class="overview-model-button" data-model="${escape(group.model)}" aria-expanded="${expanded}"><span class="overview-chevron" aria-hidden="true">›</span><span>${escape(group.model)}</span><span class="overview-model-count">${group.rows.length}</span></button></th><td>${escape(group.gpu)}</td><td>${escape(group.framework)}</td><td class="overview-measurement-cell"><strong>${integer(group.measurements)}</strong><span>observations</span></td>${cells(group.rows)}</tr>`;
      return model + [...group.rows].sort((a, b) => [a.gpu, a.framework, a.framework_version, a.parallelism].join().localeCompare([b.gpu, b.framework, b.framework_version, b.parallelism].join())).map((row) => {
        const measurement = row.status === "ready" ? `${integer(row.measurement_count)} observations` : row.status.replaceAll("_", " ");
        const skipped = row.skipped_count ? `<span>${integer(row.skipped_count)} excluded or unavailable</span>` : "";
        return `<tr class="overview-config-row gpu-row" ${expanded ? "" : "hidden"}><th scope="row"><span class="overview-config-name">${escape(row.parallelism.toUpperCase())} · ${escape(row.worker_role)}</span><div class="overview-slice-tags">${link(hf(summary.snapshot, row.configuration_manifest), "Configuration ↗")}</div></th><td>${escape(row.gpu)}</td><td><strong>${escape(row.framework)}</strong><span class="overview-cell-note">${escape(row.framework_version)}</span></td><td class="overview-measurement-cell"><strong>${escape(measurement)}</strong>${skipped}${link(hf(summary.snapshot, row.measurement_manifest), "Measurements ↗")}</td>${cells([row])}</tr>`;
      }).join("");
    }).join("") || '<tr><td colspan="7" class="empty-cell">No measurements available</td></tr>';
    body.querySelectorAll("[data-model]").forEach((button) => button.addEventListener("click", () => {
      const model = button.dataset.model;
      if (collapsed.has(model)) collapsed.delete(model); else collapsed.add(model);
      render();
      [...body.querySelectorAll("[data-model]")].find((item) => item.dataset.model === model)?.focus({ preventScroll: true });
    }));
    document.getElementById("table-count").textContent = `${models.length} models · ${summary.rows.length} configurations`;
    document.querySelectorAll("[data-sort]").forEach((button) => {
      const active = button.dataset.sort === sort.key;
      button.classList.toggle("active", active);
      button.closest("th").setAttribute("aria-sort", active ? (sort.direction === 1 ? "ascending" : "descending") : "none");
    });
  }

  function clear(message) {
    summary = null;
    body.innerHTML = `<tr><td colspan="7" class="empty-cell">${escape(message)}</td></tr>`;
    ["models", "configurations", "measurement", "evaluated"].forEach((key) => { document.getElementById(`${key}-value`).textContent = "—"; });
    document.querySelector(".snapshot-value").textContent = message;
    document.getElementById("table-count").textContent = "";
    document.getElementById("freshness").textContent = "";
    document.getElementById("nav-status-text").textContent = message;
  }

  async function select(branch) {
    const current = ++request;
    collapsed.clear();
    clear("Loading evaluation…");
    const entry = catalog.branches.find((item) => item.branch === branch);
    if (!entry || entry.status !== "available") { clear("No completed evaluation"); return; }
    try {
      if (!/^branches\/[0-9a-f]{16}\/summary\.json$/.test(entry.summary_path)) throw new Error("Invalid overview data path");
      const data = await load(entry.summary_path);
      if (current !== request) return;
      if (data.schema_version !== 1 || data.snapshot?.branch !== branch || JSON.stringify(data.methods) !== JSON.stringify(METHODS) || !Array.isArray(data.rows)) throw new Error("Invalid overview data");
      const rows = data.rows.filter((row) => row.measurement_count > 0);
      summary = { ...data, rows };
      const snapshot = data.snapshot;
      const ready = rows.filter((row) => row.status === "ready").length;
      document.getElementById("models-value").textContent = integer(new Set(rows.map((row) => row.model)).size);
      document.getElementById("configurations-value").textContent = integer(rows.length);
      document.getElementById("measurement-value").textContent = `${ready} / ${rows.length}`;
      document.getElementById("evaluated-value").textContent = `${ready} / ${rows.length}`;
      document.querySelector(".snapshot-value").innerHTML = `${link(`https://github.com/ai-dynamo/aisimulate/commit/${snapshot.commit_sha}`, `AISim ${snapshot.commit_sha.slice(0, 8)}`)} · ${link(`https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/tree/${snapshot.hf_revision}`, `HF ${snapshot.hf_revision.slice(0, 8)}`)}<br>${escape(new Date(snapshot.completed_at).toISOString())}`;
      const stale = (entry.head_sha && entry.head_sha !== snapshot.commit_sha) || Date.now() - Date.parse(snapshot.completed_at) > 48 * 3600 * 1000;
      const freshness = document.getElementById("freshness");
      freshness.classList.toggle("stale", Boolean(stale));
      freshness.textContent = stale ? "Stale result — showing the latest completed evaluation" : "Latest completed evaluation";
      document.getElementById("nav-status-text").textContent = `${ready} evaluated configurations`;
      render();
    } catch (error) {
      if (current === request) clear(`Overview unavailable: ${error.message}`);
    }
  }

  document.querySelectorAll("[data-sort]").forEach((button) => button.addEventListener("click", () => {
    if (!summary) return;
    const key = button.dataset.sort;
    sort = { key, direction: sort.key === key ? -sort.direction : key === "model" ? 1 : -1 };
    render();
  }));
  branchSelect.addEventListener("change", () => {
    const url = new URL(location.href);
    url.searchParams.set("branch", branchSelect.value);
    history.replaceState(null, "", url);
    select(branchSelect.value);
  });
  load("branches.json").then((data) => {
    if (data.schema_version !== 1 || !Array.isArray(data.branches)) throw new Error("Invalid branch catalog");
    catalog = data;
    branchSelect.replaceChildren(...data.branches.map((entry) => new Option(entry.branch, entry.branch)));
    const branch = new URL(location.href).searchParams.get("branch") || data.default_branch;
    if (!data.branches.some((entry) => entry.branch === branch)) branchSelect.add(new Option(`${branch} (unavailable)`, branch));
    branchSelect.value = branch;
    branchSelect.disabled = false;
    return select(branch);
  }).catch((error) => clear(`Overview unavailable: ${error.message}`));
})();
