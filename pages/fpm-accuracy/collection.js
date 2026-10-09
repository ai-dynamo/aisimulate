/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
(() => {
  'use strict';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'})[c]);
  const names = {self_benchmark:'Self-benchmark', static_serving:'Static serving', trace_replay:'Trace replay', unknown:'Unknown', not_applicable:'Not applicable', agentic_replay:'Agentic replay', pd_disaggregated:'PD disaggregated', aggregated:'Aggregated', session_trees:'session trees', sessions:'sessions', requests:'requests'};
  const label = value => value == null ? 'Unknown' : names[value] || String(value).replaceAll('_',' ');
  const number = value => value == null ? 'Unknown' : Number(value).toLocaleString('en-US', {maximumFractionDigits:2});
  const titles = {input:'Input sequence length distribution', output:'Output sequence length distribution', interactivity:'Interactivity over time', ttft:'TTFT over time'};
  const banner = (title, reason) => `<div class="workload-banner" role="status"><strong>${escape(title)}</strong><span>${escape(reason)}</span></div>`;
  function concurrencySummary(settings) {
    const units = new Map();
    for (const {value, unit} of settings || []) {
      if (!Number.isFinite(value) || value < 0 || !unit || ['unknown','not_applicable'].includes(unit)) continue;
      if (!units.has(unit)) units.set(unit, new Set());
      units.get(unit).add(value);
    }
    const count = [...units.values()].reduce((total, values)=>total+values.size,0);
    return count ? [`${count} concurrency setting${count === 1 ? '' : 's'}`, ...[...units].sort(([a],[b])=>a.localeCompare(b)).map(([unit,values])=>`${[...values].sort((a,b)=>a-b).map(number).join(' / ')} ${label(unit)}`)].join(' · ') : '';
  }
  function attach(element, row, snapshot) {
    const value = row.collection;
    const parts = value?.run_count ? [...value.types.filter(type => type && type !== 'unknown').map(label), ...value.datasets, concurrencySummary(value.concurrency_settings) || `${value.run_count} collection runs`] : [];
    const query = new URLSearchParams({branch:snapshot.branch, configuration:row.configuration_id, snapshot:row.snapshot_id, run:`${snapshot.run_id}-${snapshot.run_attempt}`});
    element.innerHTML = `<a href="evaluation-detail.html?${escape(query)}#dataset-workload">${escape(parts.join(' · ') || 'Dataset and workload')}</a>`;
  }
  function histogram(data, title, population) {
    if (!data?.count) return '<p class="workload-empty">Unavailable · No observed token counts.</p>';
    const max = Math.max(...data.bins.map(b=>b.count),1), width = 540/data.bins.length;
    const bars = data.bins.map((b,i)=>`<rect x="${50+i*width}" y="${210-160*b.count/max}" width="${Math.max(1,width-2)}" height="${160*b.count/max}" fill="var(--accent)" tabindex="0" aria-label="${escape(`${number(b.lower)}–${number(b.upper)} tokens: ${number(b.count)} requests`)}"><title>${escape(`${number(b.lower)}–${number(b.upper)} tokens · ${number(b.count)} requests`)}</title></rect>`).join('');
    return `<p>${number(data.count)} requests · ${number(population-data.count)} unavailable · P50 ${number(data.p50)} · P90 ${number(data.p90)}</p><svg viewBox="0 0 640 260" role="group" aria-label="${escape(title)}"><path d="M50 40V210H590" fill="none" stroke="currentColor"/>${bars}<text x="50" y="235">${number(data.bins[0].lower)}</text><text x="590" y="235" text-anchor="end">${number(data.bins.at(-1).upper)}</text><text x="320" y="255" text-anchor="middle">Tokens (logarithmic bins)</text><text x="45" y="40" text-anchor="end">${number(max)}</text></svg>`;
  }
  function series(data, title, unit) {
    if (!data?.count) return '<p class="workload-empty">Unavailable · No matching request timing measurements.</p>';
    const maxX = data.points.reduce((v,p)=>Math.max(v,p[0]),1), maxY = data.points.reduce((v,p)=>Math.max(v,p[1]),1);
    const x = t=>50+540*t/maxX, y=v=>210-160*v/maxY;
    const points = data.points.map(p=>`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3" fill="var(--accent)" opacity=".55"><title>${number(p[0])} s · ${number(p[1])} ${unit}</title></circle>`).join('');
    const line = data.rolling_p90.map((p,i)=>`${i?'L':'M'}${x(p[0])} ${y(p[1])}`).join(' ');
    return `<p>${number(data.count)} requests · ${number(data.excluded)} unavailable · Rolling P90 (50 requests)</p><svg viewBox="0 0 640 260" role="img" aria-label="${escape(title)}: ${number(data.count)} requests, ${escape(unit)}"><path d="M50 40V210H590" fill="none" stroke="currentColor"/>${points}<path d="${line}" fill="none" stroke="var(--text)" stroke-width="2"/><text x="45" y="40" text-anchor="end">${number(maxY)}</text><text x="50" y="235">0</text><text x="590" y="235" text-anchor="end">${number(maxX)} s</text><text x="320" y="255" text-anchor="middle">Time since collection run start · ${unit}</text></svg>`;
  }
  function render(target, collection) {
    const runs = collection?.runs || [];
    if (!runs.length) { target.innerHTML = '<h3>Dataset and workload</h3>'+banner('Workload information unavailable', 'Unknown · Normalized collection metadata is unavailable for this evaluation.'); return; }
    const url = new URL(location.href);
    const current = runs.find(r=>r.id === url.searchParams.get('collection_run')) || runs.find(r=>r.availability === 'available') || runs[0];
    const concurrency = concurrencySummary(runs.map(r=>({value:r.workload.concurrency,unit:r.workload.concurrency_unit})));
    const runLabel = run => `${Number.isFinite(run.workload.concurrency) ? number(run.workload.concurrency)+' '+label(run.workload.concurrency_unit) : 'Concurrency not recorded'} · ${label(run.collection_type)}${run.started_at ? ' · '+run.started_at.slice(0,16).replace('T',' ')+' UTC' : ''}`;
    target.innerHTML = `<h3>Dataset and workload</h3><p>${escape(concurrency || `${runs.length} contributing collection runs`)}${collection.unattributed_measurements ? ` · ${number(collection.unattributed_measurements)} observations with unknown collection` : ''}</p><label class="fpm-wide-control"><span id="collection-run-label">${concurrency ? 'Concurrency setting' : 'Collection run'}</span><select id="collection-run" aria-labelledby="collection-run-label">${runs.map(r=>`<option value="${escape(r.id)}" ${r===current?'selected':''}>${escape(runLabel(r))}</option>`).join('')}</select></label><div id="collection-settings"></div><div id="request-charts" class="request-charts"></div>`;
    function draw(run) {
      const w = run.workload;
      const length = value => value?.mode === 'fixed' ? number(value.value) : label(value?.mode);
      const dataset = [['Name',run.dataset.name],['Benchmark preset',run.benchmark_preset],['Replay mode',label(run.replay_mode)]];
      const workload = [['Concurrency',`${number(w.concurrency)} ${label(w.concurrency_unit)}`],['Input length',length(w.input_length)],['Output length',length(w.output_length)],['Duration',w.duration_s == null ? null : number(w.duration_s)+' s']];
      const collection = [['Collector',[run.collector.name,run.collector.version].filter(Boolean).join(' ') || null],['Serving layout',label(run.serving.layout)],['Completed requests',number(w.completed_requests)]];
      const settings = value => Array.isArray(value) ? value.map(settings).join(' · ') : value && typeof value === 'object' ? Object.entries(value).map(([key,item])=>`${label(key)}: ${settings(item)}`).join(' · ') : typeof value === 'number' ? number(value) : label(value);
      for (const [group,key,value] of [[dataset,'Revision',run.dataset.revision],[dataset,'Selection',run.dataset.selection],[dataset,'Transformations',run.dataset.transformations?.join(' · ')],[collection,'Worker topology',run.serving.topology],[collection,'Worker roles',run.serving.worker_roles?.map(label).join(' · ')],[workload,'Warmup',w.warmup && settings(w.warmup)],[workload,'Seed',w.seed],[collection,'Requested requests',w.requested_requests],[collection,'Failed requests',w.failed_requests],[collection,'Cancelled requests',w.cancelled_requests]]) {
        if (value != null && value !== '') group.push([key, settings(value)]);
      }
      const stages = run.charts?.stage_counts;
      const boundaries = w.stage_boundaries;
      const summary = [];
      if (boundaries) summary.push(['Profiling window', `${number(boundaries.profiling_start_s)}–${number(boundaries.profiling_end_s)} s from run start`]);
      if (stages) summary.push(['Profiling requests',number(stages.profiling || 0)],['Excluded from charts',`${number((stages.warmup || 0)+(stages.drain || 0))} warmup/drain requests`]);
      const groups = [['Dataset',dataset],['Workload',workload],['Collection',collection]];
      const rows = fields => fields.map(([k,v])=>`<div><dt>${escape(k)}</dt><dd>${escape(v ?? 'Unknown')}</dd></div>`).join('');
      const hasSettings = groups.some(([,fields])=>fields.some(([,value])=>value != null && !/^unknown(?: unknown)*$/i.test(String(value))));
      target.querySelector('#collection-settings').innerHTML = hasSettings ? `${summary.length ? `<dl class="workload-summary">${rows(summary)}</dl>` : ''}<div class="workload-groups">${groups.map(([title,fields])=>`<section class="workload-group"><h4>${title}</h4><dl class="workload-settings">${rows(fields)}</dl></section>`).join('')}</div>` : '';
      target.querySelector('#request-charts').innerHTML = run.availability !== 'available' ? banner(
        run.availability === 'not_applicable' ? 'Request charts are not applicable' : 'Request charts unavailable',
        run.reason || 'Matching request-level records are not included in this evaluation.'
      ) : Object.entries(titles).map(([key,title])=>`<section class="matrix-panel request-chart"><h4>${title}</h4>${key === 'input' || key === 'output' ? histogram(run.charts?.[key],title,run.charts.request_count) : series(run.charts?.[key],title,key === 'ttft' ? 'seconds' : 'tokens/s/user')}</section>`).join('');
    }
    target.querySelector('#collection-run').addEventListener('change', event=> {
      const run = runs.find(r=>r.id === event.target.value);
      const next = new URL(location.href);next.searchParams.set('collection_run',run.id);history.replaceState(null,'',next);draw(run);
    });
    draw(current);
  }
  window.fpmCollection = {attach, render};
})();
