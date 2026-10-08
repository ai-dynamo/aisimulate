/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
(() => {
  "use strict";
  const cache = new Map();
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'})[c]);
  const validPath = path => typeof path === 'string' && path.startsWith('data/') && !path.split('/').some(part => !part || part === '.' || part === '..');
  const url = (revision, path, action = 'blob') => `https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/${action}/${revision}/${path.split('/').map(encodeURIComponent).join('/')}`;
  const link = (revision, path, label) => `<a href="${escape(url(revision, path))}" target="_blank" rel="noopener">${escape(label)} ↗</a>`;
  const numeric = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value.toLocaleString('en-US') : 'Not recorded';
  const fields = entries => `<dl>${entries.map(([key, value]) => `<dt>${escape(key)}</dt><dd>${escape(value)}</dd>`).join('')}</dl>`;

  function read(revision, path) {
    const key = url(revision, path, 'resolve');
    if (!cache.has(key)) cache.set(key, (async () => {
      const response = await fetch(key, {cache:'force-cache', signal:AbortSignal.timeout(15000)});
      if (!response.ok) throw new Error('Collection evidence unavailable');
      return response.text();
    })().catch(error => { cache.delete(key); throw error; }));
    return cache.get(key);
  }

  function kind(files) {
    const truth = files.filter(file => ['truth', 'derived_truth'].includes(file.role));
    const benchmark = truth.some(file => /\/benchmark[^/]*\.json(?:\.gz)?$/.test(file.path));
    const stream = truth.some(file => /(?:fpm_stream|fpm_iterations|\.csv\.gz$)/.test(file.path));
    if (benchmark && stream) return 'Benchmark + serving measurements';
    if (benchmark) return 'Self-benchmark point sweep';
    if (files.some(file => /\/agentx-job-\d+\/collection_evidence\.json$/.test(file.path))) return 'AgentX trace replay';
    return stream ? 'Serving measurements' : 'Collection method not recorded';
  }

  function jsonWorkloads(data) {
    const config = data.input_config;
    if (!config || !Array.isArray(config.phases)) return '';
    const datasets = Array.isArray(config.datasets) ? config.datasets : [];
    return config.phases.map(phase => {
      const trace = phase.timing_mode === 'agentic_replay';
      const rows = [
        ['Workload', trace ? 'Agentic trace replay' : phase.name || 'Not recorded'],
        ['ISL / OSL', trace ? 'Trace-defined (variable)' : 'Not recorded'],
        ['Concurrency', numeric(phase.concurrency)],
        ['Requested num_req', numeric(phase.request_count)],
        ['Duration (s)', numeric(phase.duration)],
      ];
      if (trace && phase.request_count == null && Number.isFinite(phase.duration)) rows[3][1] = 'Duration-based';
      if (data.completed_measured_requests != null) rows.push(['Completed requests', numeric(data.completed_measured_requests)]);
      if (data.benchmark_id) rows.push(['Benchmark ID', data.benchmark_id]);
      const names = datasets.map(dataset => dataset.dataset).filter(name => typeof name === 'string');
      if (names.length) rows.push(['Dataset', names.join(', ')]);
      return fields(rows);
    }).join('');
  }

  function windowWorkloads(text) {
    const lines = text.trim().split(/\r?\n/), headers = lines.shift().split('\t');
    // Headerless legacy window files have incompatible layouts; do not guess columns.
    if (!headers.includes('concurrency')) return '<p>Workload settings are not recorded with named columns; see source.</p>';
    const names = [['isl', 'ISL'], ['osl', 'OSL'], ['concurrency', 'Concurrency'], ['num_req', 'num_req'], ['duration_s', 'Duration (s)']];
    const rows = lines.filter(Boolean).map(line => Object.fromEntries(headers.map((key, i) => [key, line.split('\t')[i]])));
    const unique = [...new Set(rows.map(row => names.map(([key, label]) => `${label}: ${row[key] || 'Not recorded'}`).join(' · ')))];
    return unique.map(row => `<p>${escape(row)}</p>`).join('');
  }

  async function expand(content, manifest, files, revision, method) {
    const evidence = files.filter(file => /\/(?:collection_evidence|aiperf_[^/]+)\.json$/.test(file.path) || (file.role === 'window' && file.path.endsWith('.tsv')));
    content.innerHTML = '<p>Loading collection evidence…</p>';
    const results = await Promise.allSettled(evidence.map(async file => {
      // Only small, declared metadata files are read, never raw measurement streams.
      if (!Number.isFinite(file.bytes) || file.bytes > 2 * 1024 * 1024) return '';
      const raw = await read(revision, file.path);
      const details = file.path.endsWith('.tsv') ? windowWorkloads(raw) : jsonWorkloads(JSON.parse(raw));
      const job = file.path.match(/\/agentx-job-(\d+)\//)?.[1];
      const label = job ? `AgentX job ${job}` : file.path.split('/provenance/').pop();
      return `<section class="collection-run">${link(revision, file.path, label)}${details || '<p>Workload settings not recorded.</p>'}</section>`;
    }));
    const benchmark = method === 'Self-benchmark point sweep';
    const description = benchmark
      ? 'Self-benchmark sweeps forward-pass points. Batch size and token coordinates are not fixed serving ISL / OSL / concurrency / num_req.'
      : 'MAPE pools the evaluated observations in this configuration. The sources below describe collection runs; they do not have separate MAPE scores here.';
    // A manifest may contain helper runs alongside benchmark truth; label those explicitly.
    const heading = benchmark ? 'Supporting collection evidence' : 'Collection evidence';
    const runs = results.filter(result => result.status === 'fulfilled').map(result => result.value).join('');
    const failed = results.some(result => result.status === 'rejected');
    const campaign = manifest.provenance?.source_campaign_id;
    content.innerHTML = `<p>${escape(description)}</p>${campaign ? fields([['Campaign', campaign]]) : ''}`
      + `<p>${link(revision, manifest.path, 'Measurement manifest')}</p>`
      + (runs ? `<strong>${heading}</strong>${runs}` : '<p>AgentX ID and fixed ISL / OSL / concurrency / num_req are not recorded in supported collection metadata.</p>')
      + (failed ? '<p>Some evidence could not be loaded. Open the measurement manifest to inspect its source links.</p>' : '');
  }

  async function attach(element, row, snapshot) {
    const revision = snapshot.hf_revision;
    element.innerHTML = '<summary>Test set · Loading provenance…</summary><div class="collection-content"></div>';
    try {
      if (!/^[0-9a-f]{40}$/.test(revision) || !validPath(row.measurement_manifest)) throw new Error('Invalid evidence identity');
      const manifest = JSON.parse(await read(revision, row.measurement_manifest));
      if (manifest.configuration_path !== row.configuration_path || manifest.snapshot_id !== row.snapshot_id || !Array.isArray(manifest.files)) throw new Error('Mismatched evidence identity');
      const files = manifest.files.filter(file => validPath(file.path) && file.path.startsWith(row.configuration_path + '/measurements/'));
      const method = kind(files);
      element.querySelector('summary').textContent = `Test set · ${method}`;
      let loaded = false;
      const show = () => {
        if (!element.open || loaded) return;
        loaded = true;
        expand(element.querySelector('.collection-content'), {...manifest, path:row.measurement_manifest}, files, revision, method);
      };
      element.addEventListener('toggle', show);
      show();
    } catch (_) {
      element.querySelector('summary').textContent = 'Test set · Provenance unavailable';
      element.querySelector('.collection-content').innerHTML = '<p>Could not load the pinned collection metadata. See the Measurements link.</p>';
    }
  }
  window.fpmCollection = {attach};
})();
