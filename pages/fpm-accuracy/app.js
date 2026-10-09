/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from AISim FPM Gym overview. See README.md for source and modifications.
 */
(() => {
  "use strict";
  const METHODS = ["warmup", "nowarmup", "regression"];
  const labels = { warmup: "FPM (KV warmup on)", nowarmup: "FPM (KV warmup off)", regression: "Regression" };
  const displayMethods = ["regression", "warmup", "nowarmup"];
  const comparison = document.querySelector('[data-table-view]').dataset.tableView === 'predictors';
  const columns = comparison ? 7 : 5;
  const body = document.getElementById("overview-body");
  const branchSelect = document.getElementById("branch");
  const collapsed = new Set();
  const integer = (value) => Number(value).toLocaleString("en-US");
  const escape = (value) => String(value).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
  const hf = (snapshot, path) => `https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/blob/${snapshot.hf_revision}/${path.split("/").map(encodeURIComponent).join("/")}`;
  const link = (url, label) => `<a href="${escape(url)}" target="_blank" rel="noopener">${escape(label)}</a>`;
  let catalog, summary, request = 0;
  let sort = { key: "model", direction: 1 };

  async function load(path) {
    const response = await fetch(path, { cache: "no-cache" });
    if (!response.ok) throw new Error(`Data request failed (${response.status})`);
    return response.json();
  }

  function aggregate(rows, method) {
    const result = { predicted: 0, measured: 0, errors: 0, tuning: 0, weighted: 0, mape: null };
    let weight = 0;
    for (const row of rows) {
      const metric = row.results[method]?.metrics.all;
      if (!metric) continue;
      result.predicted += metric.predicted_count;
      result.measured += metric.measured_count;
      result.errors += metric.error_count;
      result.tuning += metric.tuning_error_count;
      if (Number.isFinite(metric.mape_pct) && metric.predicted_count > 0) {
        result.weighted += metric.mape_pct * metric.predicted_count;
        weight += metric.predicted_count;
      }
    }
    result.mape = weight ? result.weighted / weight : null;
    return result;
  }

  function best(row) {
    let winner = null;
    for (const method of displayMethods) {
      const metric = row.results[method]?.metrics.all;
      if (metric?.predicted_count > 0 && Number.isFinite(metric.mape_pct)
          && (!winner || metric.mape_pct < winner.mape)) {
        winner = { ...aggregate([row], method), mape: metric.mape_pct, method };
      }
    }
    return winner;
  }

  function bestAggregate(rows) {
    const winners = rows.map(best).filter(Boolean);
    const methods = new Set(winners.map(winner => winner.method));
    return {
      mape: winners.length ? winners.reduce((sum, winner) => sum + winner.mape, 0) / winners.length : null,
      method: methods.size === 1 ? winners[0].method : null,
      label: methods.size > 1 ? 'Mixed predictors' : methods.size ? labels[winners[0].method] : 'Unavailable',
      ...Object.fromEntries(['predicted', 'measured', 'errors', 'tuning'].map(key =>
        [key, winners.reduce((sum, winner) => sum + winner[key], 0)])),
    };
  }

  function phaseLabel(row) {
    const metrics = Object.values(row.results)[0]?.metrics;
    const phases = ['prefill', 'decode', 'mixed'].filter(phase => metrics?.[phase]?.measured_count > 0);
    return phases.length ? phases.join(' + ') : 'Phase not recorded';
  }

  function cells(rows) {
    const metrics = comparison ? displayMethods.map(method => ({...aggregate(rows, method), method, label: labels[method]})) : [bestAggregate(rows)];
    return metrics.map((metric) => {
      const method = metric.method;
      const value = metric.mape === null ? "—" : `${metric.mape.toFixed(2)}%`;
      const tone = metric.mape === null ? "missing" : "";
      const notes = [];
      if (metric.errors) notes.push(`${integer(metric.errors)} errors`);
      if (metric.tuning) notes.push(`${integer(metric.tuning)} tuning errors`);
      const result = rows.length === 1 ? rows[0].results[method] : null;
      if (result?.status === "no_fpm_input") notes.push("No reviewed input");
      if (result?.status === "unsupported_predictor") notes.push("Unsupported by this AISim revision");
      const winner = comparison ? '' : `<span class="predictor-name">${escape(metric.label)}</span>`;
      const note = notes.length ? `<span>${escape(notes.join(" · "))}</span>` : '';
      return `<td class="overview-method-cell ${tone}" data-label="${escape(metric.label)}"><strong>${value}</strong>${winner}${note}</td>`;
    }).join("");
  }

  function cards(rows) {
    const averages = comparison ? displayMethods : ['overall'];
    for (const method of averages) {
      const values = rows.map(row => method === 'overall' ? best(row)?.mape : aggregate([row], method).mape).filter(Number.isFinite);
      document.getElementById(`${method}-value`).textContent = values.length
        ? `${(values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2)}%` : '—';
      document.getElementById(`${method}-count`).textContent = `${values.length} / ${rows.length} configurations with MAPE`;
    }
    if (!comparison) {
      const ready = rows.filter(row => row.status === 'ready').length;
      document.getElementById('models-value').textContent = integer(new Set(rows.map(row => row.model)).size);
      document.getElementById('configurations-value').textContent = integer(rows.length);
      document.getElementById('measurement-value').textContent = `${ready} / ${rows.length}`;
      document.getElementById('evaluated-value').textContent = `${ready} / ${rows.length}`;
    }
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
    const value = (group) => sort.key === 'best' ? bestAggregate(group.rows).mape
      : sort.key.startsWith("method:") ? aggregate(group.rows, sort.key.slice(7)).mape : group[sort.key];
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
        return `<tr class="overview-config-row gpu-row" ${expanded ? "" : "hidden"}><th scope="row"><span class="overview-config-name">${escape(row.parallelism.toUpperCase())} · ${escape(phaseLabel(row))}</span><div class="collection-note" data-configuration="${escape(row.configuration_id)}"></div><a href="evaluation-detail.html?branch=${encodeURIComponent(summary.snapshot.branch)}&amp;configuration=${encodeURIComponent(row.configuration_id)}&amp;snapshot=${encodeURIComponent(row.snapshot_id)}&amp;run=${summary.snapshot.run_id}-${summary.snapshot.run_attempt}">Details →</a> <a href="3d-visualization.html?branch=${encodeURIComponent(summary.snapshot.branch)}&amp;configuration=${encodeURIComponent(row.configuration_id)}&amp;snapshot=${encodeURIComponent(row.snapshot_id)}">3D Viz →</a><div class="overview-slice-tags">${link(hf(summary.snapshot, row.configuration_manifest), "Configuration ↗")}</div></th><td>${escape(row.gpu)}</td><td><strong>${escape(row.framework)}</strong><span class="overview-cell-note">${escape(row.framework_version)}</span></td><td class="overview-measurement-cell"><strong>${escape(measurement)}</strong>${skipped}${link(hf(summary.snapshot, row.measurement_manifest), "Measurements ↗")}</td>${cells([row])}</tr>`;
      }).join("");
    }).join("") || `<tr><td colspan="${columns}" class="empty-cell">No measurements available</td></tr>`;
    const rows = new Map(summary.rows.map(row => [row.configuration_id, row]));
    body.querySelectorAll('.collection-note').forEach(element => {
      window.fpmCollection.attach(element, rows.get(element.dataset.configuration), summary.snapshot);
    });
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
    body.innerHTML = `<tr><td colspan="${columns}" class="empty-cell">${escape(message)}</td></tr>`;
    document.querySelectorAll('.summary-value').forEach(element => { element.textContent = '—'; });
    document.querySelectorAll('.summary-note').forEach(element => { element.textContent = ''; });
    window.fpmNavigation.setSnapshot(null, message);
    document.getElementById("table-count").textContent = "";
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
      if (![1, 2].includes(data.schema_version) || data.snapshot?.branch !== branch || JSON.stringify(data.methods) !== JSON.stringify(METHODS) || !Array.isArray(data.rows)) throw new Error("Invalid overview data");
      const rows = data.rows.filter((row) => row.measurement_count > 0);
      summary = { ...data, rows };
      const snapshot = data.snapshot;
      cards(rows);
      window.fpmNavigation.setSnapshot(snapshot);
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
