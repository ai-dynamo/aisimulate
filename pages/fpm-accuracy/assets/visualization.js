/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Modified from AISim FPM Gym f934c030afc3a03cb04d8f3ff4709194f7445c98.
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Apache-2.0. See ../README.md. */
/* Measurement assets are built in CI. No HF credentials or local paths enter the browser. */
(() => {
  "use strict";
  const root = document.getElementById("gym-visualization");
  const get = id => root.querySelector("#gv-" + id);
  const base = "data/visualization/";
  const panes = ["left", "right"];
  const phase = get("phase"), scale = get("scale"), density = get("density");
  const xAxis = get("x-axis"), yAxis = get("y-axis");
  const fields = ["points", "axis_values", "rank_details", "iteration_ids"];
  const cameras = {};
  const count = n => n.toLocaleString("en-US");
  const pretty = n => n >= 1e6 ? Number((n / 1e6).toPrecision(3)) + "M"
    : n >= 1e3 ? Number((n / 1e3).toPrecision(3)) + "k" : Number(n.toPrecision(3)).toString();
  const escape = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[c]);
  let data, axes, groups, catalog, chartLibrary;
  const mutations = Object.fromEntries(panes.map(pane => [pane, Promise.resolve()]));
  let epoch = 0;

  async function json(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Could not load measurement data (HTTP " + response.status + "). Reload to retry.");
    if (url.endsWith(".gz")) {
      if (!globalThis.DecompressionStream) throw new Error("This browser cannot unpack point data. Use a current Chrome, Firefox or Safari.");
      // Explicit .gz files work on static Pages without a custom Content-Encoding header.
      return new Response(response.body.pipeThrough(new DecompressionStream("gzip"))).json();
    }
    return response.json();
  }

  function plotly() {
    if (globalThis.Plotly) return Promise.resolve();
    if (!chartLibrary) chartLibrary = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "assets/plotly.min.js";
      script.onload = () => globalThis.Plotly ? resolve() : reject(new Error("Chart library failed to initialize. Reload to retry."));
      script.onerror = () => reject(new Error("Chart library could not load. Reload to retry."));
      document.head.append(script);
    });
    return chartLibrary;
  }

  function config(pane) { return catalog.get(get(pane + "-config").value); }
  function matching(pane, all = false) {
    return (config(pane)?.groups || []).map(id => groups.get(id)).filter(g => all || phase.value === "all" || g.phase === phase.value);
  }
  function workers(pane, preferred) {
    const select = get(pane + "-worker"), previous = preferred || select.value;
    const entries = [...new Map(matching(pane).map(g => [g.worker, g])).values()];
    select.replaceChildren(...entries.map(g => new Option(g.label, g.worker)));
    select.disabled = entries.length === 0;
    if (!entries.length) select.add(new Option("No matching observations", ""));
    else select.value = entries.find(g => g.worker === previous)?.worker || entries[0].worker;
  }
  function configurations(pane, preferred) {
    const entries = data.catalog.filter(c => c.model === get(pane + "-model").value);
    get(pane + "-config").replaceChildren(...entries.map(c => new Option(c.label + (c.availability === "diagnostic" ? " · Diagnostic" : ""), c.id)));
    const chosen = entries.find(c => c.id === preferred) || entries.find(c => c.groups.length && c.snapshot === "current") || entries.find(c => c.groups.length) || entries[0];
    get(pane + "-config").value = chosen.id;
    workers(pane);
  }

  async function loadGroup(group, mode) {
    const prefix = mode === "sample" ? "sample_" : "";
    if (group[prefix + "points"]) return;
    const key = "loading_" + mode;
    if (!group[key]) group[key] = (async () => {
      const files = mode === "sample" ? [group.sample_file] : group.all_files;
      const block = Object.fromEntries(fields.map(f => [prefix + f, []]));
      for (const file of files) {
        if (!/^[a-f0-9]{64}\.json(?:\.gz)?$/.test(file)) throw new Error("Invalid point asset path.");
        const chunk = await json(base + file);
        const length = chunk[prefix + "points"]?.length;
        if (!Number.isInteger(length) || fields.some(f => chunk[prefix + f]?.length !== length)) throw new Error("Incomplete point asset.");
        for (const field of fields) for (const row of chunk[prefix + field]) block[prefix + field].push(row);
      }
      const expected = mode === "sample" ? group.sample_indices.length : group.n;
      if (block[prefix + "points"].length !== expected) throw new Error("Measurement count mismatch.");
      Object.assign(group, block);
    })().catch(error => { delete group[key]; throw error; });
    await group[key];
  }

  function notes() {
    get("workload-note").textContent = "Workload follows the native online representative rank, fixed independently of X and Y. Both axes aggregate all DP ranks; latency is their maximum.";
    get("axis-note").textContent = "X: " + axes.get(xAxis.value).definition + " · Y: " + axes.get(yAxis.value).definition;
    get("density-note").textContent = density.value === "sample"
      ? "Sample uses fixed iteration IDs plus feature and latency extremes. Changing axes keeps the same sample; point density does not represent frequency."
      : "All matching observations draws every available iteration for the selected workload and run / worker.";
    const reduced = panes.some(p => config(p)?.coordinate_scope === "reduced_record");
    get("record-note").hidden = !reduced;
    get("record-note").textContent = "Reduced measurement records use the published batch and token fields. The original DP-rank breakdown is unavailable; rank 0 denotes the normalized record, not a measured physical rank.";
  }

  function saveCamera(pane) {
    const chart = get(pane + "-chart");
    if (!chart.hidden && chart._fullLayout?.scene?._scene) cameras[pane] = chart._fullLayout.scene._scene.getCamera();
  }
  function ticks(lo, hi, log) {
    if (log) {
      const values = [];
      for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
        for (const m of [1, 2, 5]) if (m * 10 ** p >= lo && m * 10 ** p <= hi) values.push(m * 10 ** p);
      }
      return values.filter((_, i) => i % Math.max(1, Math.ceil(values.length / 6)) === 0);
    }
    if (lo === hi) return [lo];
    const raw = (hi - lo) / 4, base = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(v => v * base).find(v => v >= raw), values = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * .01; v += step) values.push(v);
    return values;
  }
  function axis(title, lo, hi, log = false) {
    const pad = Math.max((hi - lo) * .035, .3), values = ticks(lo, hi, log);
    return {title: {text: title, font: {size: 14, weight: 700, color: "#152d1e"}}, type: log ? "log" : "linear",
      range: log ? [Math.log10(lo), Math.log10(hi)] : [Math.max(0, lo - pad), hi + pad],
      tickmode: "array", tickvals: values, ticktext: values.map(pretty),
      tickfont: {size: 13, weight: 700, color: "#203b2a"}, showline: true, linecolor: "#405747", linewidth: 3,
      ticks: "outside", ticklen: 5, tickwidth: 2, tickcolor: "#405747", showbackground: true,
      backgroundcolor: "#fbfcfa", gridcolor: "#d9e1d8", zeroline: false, showspikes: false};
  }
  function empty(pane, title, message) {
    const chart = get(pane + "-chart"), element = get(pane + "-empty");
    if (chart.data) Plotly.purge(chart);
    chart.hidden = true; element.hidden = false;
    chart.parentElement.querySelector(".gv-chart-hint").hidden = true;
    const heading = document.createElement("strong"), text = document.createElement("p");
    heading.textContent = title; text.textContent = message;
    element.replaceChildren(heading, text);
    chart.dataset.visible = "0"; chart.dataset.observations = "0";
  }
  async function draw(pane, revision) {
    const selected = matching(pane).filter(g => g.worker === get(pane + "-worker").value);
    const mode = density.value;
    // Acquisition must not block newer selections; only Plotly mutations serialize.
    await Promise.all([plotly(), ...selected.map(g => loadGroup(g, mode))]);
    if (revision !== epoch) return;
    const next = mutations[pane].then(() => revision === epoch ? render(pane, revision) : undefined);
    mutations[pane] = next.catch(() => {});
    await next;
  }
  async function render(pane, revision) {
    const c = config(pane), chart = get(pane + "-chart"), info = get(pane + "-count");
    const selected = matching(pane).filter(g => g.worker === get(pane + "-worker").value);
    get(pane + "-source").href = c.source;
    Object.assign(chart.dataset, {config: c.id, phase: phase.value, density: density.value, availability: c.availability});
    if (!selected.length) {
      empty(pane, c.groups.length ? "No observations for this workload" : "Measurement points unavailable",
        c.groups.length ? "Choose another workload or All workloads." : c.reason);
      info.textContent = c.label;
      return;
    }
    const sample = density.value === "sample", mode = density.value, ax = axes.get(xAxis.value), ay = axes.get(yAxis.value);
    const log = scale.value === "log", zoom = scale.value === "linear-zoom";
    info.textContent = sample ? "Loading sample…" : "Loading all matching observations…";
    chart.hidden = true; get(pane + "-empty").hidden = true;
    await Promise.all([plotly(), ...selected.map(g => loadGroup(g, mode))]);
    if (revision !== epoch) return;
    const prefix = sample ? "sample_" : "", collect = key => selected.flatMap(g => g[prefix + key]);
    const points = collect("points"), values = collect("axis_values"), details = collect("rank_details"), ids = collect("iteration_ids");
    if (!points.length) {
      empty(pane, "No sampled points for this workload", "Choose All matching observations to display every matching iteration.");
      return;
    }
    chart.hidden = false; get(pane + "-empty").hidden = true;
    chart.parentElement.querySelector(".gv-chart-hint").hidden = false;
    const kinds = selected.flatMap(g => Array(g[prefix + "points"].length).fill(g.phase));
    const coordinates = a => points.map((p, i) => a.point_index !== undefined ? p[a.point_index] : values[i][a.feature_index]);
    const rankLabel = (ids, ranks) => ids.map(id => "r" + id + " (" + ranks.find(r => r[0] === id)[1] + ")").join(", ");
    const producer = (a, ranks) => {
      if (a.reduction === "sum") return "Sum across all " + ranks.length + " ranks";
      const scores = ranks.map(r => a.rank_columns.reduce((sum, k) => sum + r[k], 0)), max = Math.max(...scores);
      return rankLabel(ranks.filter((_, i) => scores[i] === max).map(r => r[0]), ranks);
    };
    const custom = points.map((p, i) => {
      const d = details[i], ranks = d[3];
      return p.concat([producer(ax, ranks), producer(ay, ranks), rankLabel(d[1], ranks), rankLabel([d[2]], ranks), kinds[i],
        ranks.map(r => "r" + r[0] + ": " + r[1] + " · " + Number(r[4].toPrecision(6)) + " ms").join("<br>")]);
    });
    const range = a => {
      const rs = selected.map(g => a.point_index !== undefined ? g.ranges[a.point_index] : g.axis_ranges[a.feature_index]);
      return [Math.min(...rs.map(r => r[0])), Math.max(...rs.map(r => r[1]))];
    };
    const xr = range(ax), yr = range(ay), zlo = Math.min(...selected.map(g => g.ranges[2][0]));
    const zhi = Math.max(...selected.map(g => g.ranges[2][1]));
    const zmax = zoom ? Math.max(...selected.map(g => g.zoom)) : zhi * 1.07;
    const narrow = chart.clientWidth < 420;
    const camera = cameras[pane] || {eye: {x: 1.9, y: 1.95, z: 1.3}};
    await Plotly.react(chart, [{type: "scatter3d", mode: "markers", x: coordinates(ax), y: coordinates(ay),
      z: points.map(p => p[2]), ids, customdata: custom, name: "Measured points",
      marker: {size: 3, opacity: .8, color: points.map(p => log ? Math.log10(p[2]) : p[2]),
        colorscale: [[0, "#326bb0"], [.55, "#5b8f16"], [1, "#c76528"]],
        cmin: log ? Math.log10(zlo) : 0, cmax: log ? Math.log10(zhi) : zmax, showscale: false},
      hovertemplate: "<b>" + escape(c.model) + "</b><br>" + ax.label + " = %{x:,.6~g}<br>" + ay.label + " = %{y:,.6~g}<br>Latency = %{z:,.6f} ms<br>Counter = %{customdata[3]}<br>X source: %{customdata[6]}<br>Y source: %{customdata[7]}<br>Slowest: %{customdata[8]}<br>Native representative: %{customdata[9]}<br>Workload: %{customdata[10]}<br>%{customdata[11]}<extra>" + (c.availability === "diagnostic" ? "Diagnostic" : c.coordinate_scope === "reduced_record" ? "Reduced record" : "Measured") + "</extra>"
    }], {autosize: true, paper_bgcolor: getComputedStyle(document.documentElement).getPropertyValue("--panel").trim(), margin: {l: narrow ? 24 : 12, r: narrow ? 24 : 12, t: 18, b: 24},
      showlegend: false, hoverlabel: {bgcolor: "#fff", font: {color: "#152019", size: 12}},
      scene: {xaxis: axis(narrow ? ax.short : ax.label, ...xr), yaxis: axis(narrow ? ay.short : ay.label, ...yr),
        zaxis: axis("Latency (ms)", log ? zlo * .88 : 0, zmax, log), aspectmode: "manual", aspectratio: {x: 1.1, y: 1, z: .95}, camera, dragmode: "orbit"},
      uirevision: pane
    }, {responsive: true, displaylogo: false, scrollZoom: true, displayModeBar: true,
      modeBarButtonsToRemove: ["hoverClosest3d"], toImageButtonOptions: {format: "png", filename: "fpm-" + pane, width: 1400, height: 1100, scale: 2}});
    if (revision !== epoch) return;
    const total = selected.reduce((sum, g) => sum + g.n, 0);
    const above = selected.reduce((sum, g) => sum + (g.points ? g.points.filter(p => p[2] > zmax).length : 0), 0);
    info.textContent = count(points.length) + " shown / " + count(total) + (c.availability === "diagnostic" ? " diagnostic iterations" : " observations")
      + " · " + (zoom ? "Main latency range" + (!sample ? " · " + count(above) + " above view" : "") : "Full latency range")
      + " · HF " + data.hf_revision.slice(0, 12);
    if (xr[0] === xr[1] || yr[0] === yr[1]) info.textContent += " · Constant axis; all matching observations retained.";
    Object.assign(chart.dataset, {visible: String(points.length), observations: String(total), xAxis: ax.id, yAxis: ay.id});
  }

  async function redraw() {
    const revision = ++epoch;
    try {
      root.dataset.ready = "loading"; get("loading").hidden = false; get("error").hidden = true;
      notes();
      const results = await Promise.allSettled(panes.map(pane => draw(pane, revision)));
      if (revision !== epoch) return;
      const failure = results.find(r => r.status === "rejected");
      get("loading").hidden = true;
      if (failure) {
        results.forEach((result, i) => {
          if (result.status === "rejected") empty(panes[i], "Point data could not load", result.reason.message);
        });
        fail(failure.reason); return;
      }
      root.dataset.ready = "true";
      // Retain only the selected runs, so browsing the catalog does not retain
      // every full dataset in browser memory.
      const active = new Set(panes.flatMap(p => matching(p).filter(g => g.worker === get(p + "-worker").value).map(g => g.id)));
      for (const group of groups.values()) if (!active.has(group.id)) {
        for (const field of fields) { delete group[field]; delete group["sample_" + field]; }
        delete group.loading_sample; delete group.loading_all;
      }
    } catch (error) { if (revision === epoch) fail(error); }
  }
  function fail(error) {
    root.dataset.ready = "error"; get("loading").hidden = true; get("error").hidden = false;
    get("error").textContent = error.message;
  }

  async function start() {
    data = await json(base + "catalog.json");
    if (data.schema_version !== 1 || data.policy !== "native-online-rank-unit-features-v1") throw new Error("Unsupported visualization data version.");
    axes = new Map(data.axes.map(a => [a.id, a]));
    groups = new Map(data.groups.map(g => [g.id, g])); catalog = new Map(data.catalog.map(c => [c.id, c]));
    if (!data.catalog.length) throw new Error("The pinned dataset catalog contains no configurations.");
    for (const control of [xAxis, yAxis]) control.replaceChildren(...data.axes.map(a => new Option(a.label, a.id)));
    xAxis.value = "attention"; yAxis.value = "moe";
    const models = [...new Set(data.catalog.map(c => c.model))].sort();
    get("catalog-line").textContent = count(models.length) + " models · " + count(data.catalog.length) + " snapshots · "
      + count(data.catalog.filter(c => c.groups.length).length) + " with point data · HF " + data.hf_revision.slice(0, 12);
    const statusResponse = await fetch(base + "publication.json", {cache:"no-cache"});
    if (!statusResponse.ok && statusResponse.status !== 404) throw new Error("Publication status unavailable.");
    const publication = statusResponse.ok ? await statusResponse.json() : null;
    const stale = publication && (publication.hf_revision !== publication.current_hf_revision || Date.now()-Date.parse(publication.completed_at)>48*3600000);
    document.getElementById("nav-status-text").textContent = (stale ? "Stale measurement snapshot · " : "") + "HF " + data.hf_revision.slice(0, 12)
      + (publication ? " · " + publication.completed_at : "");
    for (const pane of panes) {
      get(pane + "-model").replaceChildren(...models.map(m => new Option(m.split("/").pop(), m)));
      get(pane + "-model").value = data.defaults[pane].model;
      configurations(pane, data.defaults[pane].configuration); workers(pane, data.defaults[pane].worker);
      for (const field of ["model", "config", "worker"]) get(pane + "-" + field).addEventListener("change", () => {
        saveCamera(pane);
        if (field === "model") configurations(pane);
        if (field === "config") workers(pane);
        redraw();
      });
    }
    phase.addEventListener("change", () => { panes.forEach(p => { saveCamera(p); workers(p); }); redraw(); });
    for (const control of [scale, density, xAxis, yAxis, get("landscape")]) control.addEventListener("change", () => { panes.forEach(saveCamera); redraw(); });
    let width = root.clientWidth, timer;
    new ResizeObserver(() => {
      if (Math.abs(width - root.clientWidth) < 2) return;
      width = root.clientWidth; clearTimeout(timer);
      timer = setTimeout(() => { panes.forEach(saveCamera); redraw(); }, 180);
    }).observe(root);
    new MutationObserver(() => { panes.forEach(saveCamera); redraw(); }).observe(document.documentElement, {attributes:true, attributeFilter:["data-theme"]});
    await redraw();
  }
  start().catch(fail);
})();
