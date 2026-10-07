/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
(() => {
  'use strict';
  const view = document.querySelector('[data-view]').dataset.view;
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
  const completed = timestamp => {
    const date = new Date(timestamp);
    return Number.isNaN(date.getTime()) ? timestamp : date.toISOString().slice(0,16).replace('T',' ')+' UTC';
  };
  const methods = ['regression', 'warmup', 'nowarmup'];
  const labels = {regression:'Regression', warmup:'FPM (KV warmup on)', nowarmup:'FPM (KV warmup off)'};
  const colors = ['#818cf8', '#2dd4bf', '#fb923c'];
  const phases = ['all', 'prefill', 'decode', 'mixed'];
  const params = new URLSearchParams(location.search);
  let entries = [], summaries = [], detail, selected, request = 0;
  const filters = [['model-filter','model'], ['gpu-filter','gpu'], ['framework-filter','framework']];
  async function load(path, optional = false) {
    if ((path !== 'branches.json' && !/^data\/[a-zA-Z0-9_./-]+$/.test(path)) || path.includes('..')) throw new Error('Unsafe data path');
    const response = await fetch(path, {cache:'no-cache'});
    if (optional && response.status === 404) return null;
    if (!response.ok) throw new Error(`Data unavailable (HTTP ${response.status}).`);
    return response.json();
  }
  function status(message) { $('dashboard-status').textContent = message; $('nav-status-text').textContent = message; }
  function options(element, values, all = true) {
    const prior = element.value;
    element.replaceChildren(...(all ? [new Option('All', '')] : []), ...values.map(([value, label]) => new Option(label, value)));
    if ([...element.options].some(o => o.value === prior)) element.value = prior;
  }
  function populate(rows) {
    for (const [id, key] of filters) options($(id), [...new Set(rows.map(r => r[key]))].sort().map(v => [v,v]));
  }
  function matching(rows) {
    return rows.filter(row => filters.every(([id,key]) => !$(id).value || row[key] === $(id).value)
      && (!$('search-filter')?.value || JSON.stringify([row.model,row.configuration_id,row.parallelism]).toLowerCase().includes($('search-filter').value.toLowerCase())));
  }
  function metric(rows, method, phase) {
    let predicted = 0, measured = 0, total = 0, errors = 0;
    for (const row of rows) {
      const m = row.results[method]?.metrics[phase];
      if (!m) continue;
      predicted += m.predicted_count; measured += m.measured_count;
      total += (m.mape_pct || 0) * m.predicted_count; errors += m.error_count;
    }
    return {predicted, measured, errors, mape: predicted ? total / predicted : null};
  }
  const value = m => `${m.mape == null ? '—' : m.mape.toFixed(2)+'%'} · ${m.predicted.toLocaleString()}/${m.measured.toLocaleString()} predicted · ${m.measured ? (100*m.predicted/m.measured).toFixed(1) : '0'}% coverage · ${m.errors} errors`;
  function phaseTable(rows) {
    $('phase-summary').innerHTML = '<thead><tr><th>Method</th>'+phases.map(p=>`<th>${p}</th>`).join('')+'</tr></thead><tbody>'+methods.map(method=>`<tr><th>${labels[method]}</th>${phases.map(p=>`<td>${value(metric(rows,method,p))}</td>`).join('')}</tr>`).join('')+'</tbody>';
  }
  function signature(rows, method) {
    return JSON.stringify(rows.map(r=>[r.configuration_id,r.snapshot_id,r.membership_sha256,r.parser_policy_id,r.results[method]?.artifact]).sort((a,b)=>String(a[0]).localeCompare(String(b[0]))));
  }
  let trendEvents;
  function trends() {
    trendEvents?.abort();
    trendEvents = new AbortController();
    const phase = $('phase-filter').value;
    const data = summaries.map(s=>({s,rows:matching(s.rows)})).filter(d=>d.rows.length);
    window.fpmNavigation.setRun(data.at(-1)?.s.snapshot);
    const points = [];
    const max = Math.max(1,...data.flatMap(d=>methods.map(m=>metric(d.rows,m,phase).mape || 0)));
    const x = i => (data.length === 1 ? 480 : 60 + i * 820 / Math.max(1,data.length-1)), y = v => 270 - v / max * 230;
    let chart = '<svg viewBox="0 0 960 320" role="group" aria-label="MAPE across revisions. Focus or tap a sample for metrics and evaluation details."><path d="M60 30 V270 H900" fill="none" stroke="currentColor"/>';
    for (let n=0;n<=4;n++) chart += `<text x="4" y="${y(n*max/4)}" fill="currentColor" font-size="12">${(n*max/4).toFixed(1)}%</text>`;
    data.forEach((d,i)=> {
      if (i && methods.some(m=>signature(d.rows,m) !== signature(data[i-1].rows,m))) chart += `<path d="M${(x(i)+x(i-1))/2} 30 V270" stroke="currentColor" stroke-dasharray="4 5"><title>Measurement population or FPM input changed</title></path>`;
      if (i % Math.max(1,Math.ceil(data.length/10)) === 0 || i === data.length-1) chart += `<text x="${x(i)}" y="295" text-anchor="middle" fill="currentColor" font-size="10">${d.s.snapshot.commit_sha.slice(0,7)}<tspan x="${x(i)}" dy="14">${esc(completed(d.s.snapshot.completed_at).slice(0,10))}</tspan></text>`;
    });
    methods.forEach((method,k)=>data.forEach((d,i)=> {
      const m = metric(d.rows,method,phase); if (m.mape == null) return;
      const previous = i ? metric(data[i-1].rows,method,phase) : null;
      if (previous?.mape != null && signature(d.rows,method) === signature(data[i-1].rows,method)) chart += `<path d="M${x(i-1)} ${y(previous.mape)} L${x(i)} ${y(m.mape)}" stroke="${colors[k]}" fill="none"/>`;
      const text = `${labels[method]}: ${value(m)} · AISim ${d.s.snapshot.commit_sha} · Evaluated ${completed(d.s.snapshot.completed_at)} · HF ${d.s.snapshot.hf_revision} · evaluator ${d.s.snapshot.evaluator_sha}`;
      const index = points.push({method, metric:m, snapshot:d.s.snapshot, color:colors[k]}) - 1;
      chart += `<g class="trend-point" role="img" tabindex="0" data-point="${index}" transform="translate(${x(i)} ${y(m.mape)})" style="--point-color:${colors[k]}" aria-label="${esc(text)}"><circle class="trend-hit" r="14" fill="transparent"/><circle class="trend-dot" r="5" fill="${colors[k]}"/></g>`;
    }));
    $('trend-chart').innerHTML = chart+'</svg>'+ (data.length === 1 ? '<p>One evaluation available. More daily evaluations will form the trend.</p>' : '') +'<p>'+methods.map((m,i)=>`<span style="color:${colors[i]}">● ${labels[m]}</span>`).join(' · ')+'</p>';
    const tooltip = document.createElement('div');
    tooltip.id = 'trend-tooltip'; tooltip.setAttribute('role','tooltip'); tooltip.hidden = true;
    $('trend-chart').append(tooltip);
    let active, pinned = false, timer;
    const listen = (target, event, callback, options = {}) => target.addEventListener(event, callback, {...options, signal:trendEvents.signal});
    const hide = () => {
      clearTimeout(timer);
      tooltip.hidden = true;
      active?.classList.remove('active');
      active?.removeAttribute('aria-describedby');
      active = null; pinned = false;
    };
    const leave = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (!pinned && document.activeElement !== active && !tooltip.matches(':hover') && !active?.matches(':hover')) hide();
      }, 140);
    };
    trendEvents.signal.addEventListener('abort', () => clearTimeout(timer), {once:true});
    const show = point => {
      clearTimeout(timer);
      if (active !== point) hide();
      active = point;
      const {method, metric:m, snapshot:s, color} = points[Number(point.dataset.point)];
      const rows = [
        ['Predicted / measured', `${m.predicted.toLocaleString()} / ${m.measured.toLocaleString()}`],
        ['Coverage', `${m.measured ? (100*m.predicted/m.measured).toFixed(1) : '0'}%`],
        ['Errors', m.errors.toLocaleString()],
        ['Evaluated', completed(s.completed_at)],
        ['AISim', s.commit_sha.slice(0,8)],
        ['HF dataset', s.hf_revision.slice(0,8)],
        ['Evaluator', s.evaluator_sha.slice(0,8)],
      ];
      tooltip.innerHTML = `<div class="trend-tooltip-heading" style="--point-color:${color}">${esc(labels[method])}</div><div class="trend-tooltip-mape">${m.mape.toFixed(2)}% <small>MAPE</small></div><dl>${rows.map(([label,value])=>`<dt>${label}</dt><dd>${esc(value)}</dd>`).join('')}</dl>`;
      tooltip.hidden = false;
      point.classList.add('active');
      point.setAttribute('aria-describedby', tooltip.id);
      const rect = point.getBoundingClientRect();
      const width = tooltip.offsetWidth, height = tooltip.offsetHeight;
      tooltip.style.left = Math.max(8,Math.min(rect.left+rect.width/2-width/2,innerWidth-width-8))+'px';
      const top = rect.top-height-10 >= 8 ? rect.top-height-10 : rect.bottom+10;
      tooltip.style.top = Math.max(8,Math.min(top,innerHeight-height-8))+'px';
    };
    $('trend-chart').querySelectorAll('.trend-point').forEach(point => {
      listen(point, 'pointerenter', event => { if (event.pointerType !== 'touch') show(point); });
      listen(point, 'focus', () => show(point));
      listen(point, 'pointerleave', leave);
      listen(point, 'blur', hide);
      listen(point, 'pointerdown', event => {
        if (event.pointerType !== 'touch') return;
        event.preventDefault();
        if (active === point && pinned) hide();
        else { show(point); pinned = true; }
      });
    });
    listen(tooltip, 'pointerenter', () => clearTimeout(timer));
    listen(tooltip, 'pointerleave', leave);
    listen(document, 'keydown', event => { if (event.key === 'Escape') hide(); });
    listen(document, 'pointerdown', event => {
      if (!event.target.closest('.trend-point, #trend-tooltip')) hide();
    });
    listen(window, 'resize', hide);
    listen(window, 'scroll', hide, {capture:true});
    phaseTable(data.at(-1)?.rows || []);
  }
  function heatmap(target, map, error) {
    if (!map) { target.textContent = 'No measured observations for this workload.'; return; }
    const cells = new Map(map.cells.map(c=>[`${c.x_index},${c.y_index}`,c]));
    const max = Math.max(1,...map.cells.map(c=>error ? c.mape_pct || 0 : c.measured_count));
    target.innerHTML = `<p>X: ${esc(map.x_label)} · Y: ${esc(map.y_label)}</p><table class="heatmap"><thead><tr><th>Y / X</th>${map.x_bins.map(b=>`<th>${esc(b.label)}</th>`).join('')}</tr></thead><tbody>`+map.y_bins.map((b,y)=>`<tr><th>${esc(b.label)}</th>`+map.x_bins.map((_,x)=> {
      const c = cells.get(`${x},${y}`), v = c ? (error ? c.mape_pct : c.measured_count) : null;
      return `<td style="background:color-mix(in srgb, var(--accent) ${v == null ? 0 : 10+60*v/max}%, transparent)" title="${c ? `${c.predicted_count}/${c.measured_count} predicted` : 'No observations'}">${v == null ? '—' : error ? v.toFixed(2)+'%' : v.toLocaleString()}${error && c ? `<small>${c.predicted_count}/${c.measured_count}</small>` : ''}</td>`;
    }).join('')+'</tr>').join('')+'</tbody></table>';
  }
  function maps() {
    const method = $('method-filter').value;
    const candidates = detail?.methods[method] || [];
    const winner = selected?.results[method];
    const candidate = candidates.find(c=>c.artifact?.id === winner?.artifact?.id);
    const phase = $('phase-filter').value;
    heatmap($('distribution'),detail?.workload_heatmaps[phase],false);
    const evidence = $('variant-evidence');
    evidence.replaceChildren();
    if (winner?.artifact) {
      const summary = summaries[Number($('evaluation-filter').value)];
      const link = document.createElement('a');
      link.href = `https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/blob/${summary.snapshot.hf_revision}/${winner.artifact.path.split('/').map(encodeURIComponent).join('/')}`;
      link.textContent = `FPM input: ${winner.artifact.id} ↗`;
      link.target = '_blank'; link.rel = 'noopener';
      evidence.append(link);
    }
    const unavailable = {
      no_fpm_input: 'No FPM input is available for this method.',
      unsupported_predictor: 'This predictor is not supported by the evaluated version or dependencies.',
      predictor_error: 'Predictor initialization failed.'
    };
    const reason = unavailable[candidate?.status || winner?.status];
    if (reason) { $('error-heatmap').textContent = reason; return; }
    if (!candidate) { $('error-heatmap').textContent = 'Prediction detail is unavailable for this evaluation.'; return; }
    const map = candidate._heatmaps?.[phase];
    if (map && !map.cells.some(c=>c.predicted_count > 0)) {
      $('error-heatmap').textContent = 'No successful predictions for this workload.';
      return;
    }
    heatmap($('error-heatmap'),map,true);
  }
  async function configuration() {
    const token = ++request;
    const index = Number($('evaluation-filter').value), summary = summaries[index], entry = entries[index];
    selected = summary?.rows.find(r=>r.configuration_id+'/'+r.snapshot_id === $('configuration-filter').value);
    detail = null;
    phaseTable(selected ? [selected] : []);
    $('detail-evidence').textContent = '';
    maps();
    if (!selected) return;
    const hf = `https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/blob/${summary.snapshot.hf_revision}/`;
    $('detail-evidence').innerHTML = `<span role="img" aria-label="Hugging Face" title="Hugging Face dataset evidence">🤗</span> <a target="_blank" rel="noopener" href="${hf+selected.configuration_manifest.split('/').map(encodeURIComponent).join('/')}">Pinned configuration ↗</a> · <a target="_blank" rel="noopener" href="${hf+selected.measurement_manifest.split('/').map(encodeURIComponent).join('/')}">Measurements ↗</a> · ${esc(selected.status)} · ${selected.skipped_count} excluded or unavailable`;
    if (!entry.details_path) { status('This retained evaluation has summary data only.'); return; }
    try {
      const document = await load('data/'+entry.details_path);
      if (token !== request) return;
      detail = document.rows.find(r=>r.configuration_id === selected.configuration_id && r.snapshot_id === selected.snapshot_id);
      const age = Date.now()-Date.parse(summary.snapshot.completed_at);
      status(`${age>48*3600000 ? 'Stale · ' : ''}AISim ${summary.snapshot.commit_sha.slice(0,12)} · HF ${summary.snapshot.hf_revision.slice(0,12)} · ${completed(summary.snapshot.completed_at)}`);
      if (detail && !detail.workload_heatmaps[$('phase-filter').value]) {
        const available = ['prefill','decode','mixed'].find(phase => detail.workload_heatmaps[phase]);
        if (available) $('phase-filter').value = available;
      }
      maps();
    } catch(error) { if (token === request) status(error.message); }
  }
  function configurations() {
    const rows = matching(summaries[Number($('evaluation-filter').value)]?.rows || []);
    const prior = $('configuration-filter').value;
    options($('configuration-filter'),rows.map(r=>[r.configuration_id+'/'+r.snapshot_id,`${r.model} · ${r.gpu} · ${r.parallelism} · ${r.worker_role} · ${r.snapshot_id}${r.measurement_count ? '' : ' · No measurements'}`]),false);
    if (!rows.some(r=>r.configuration_id+'/'+r.snapshot_id === prior)) {
      const measured = rows.find(r=>r.measurement_count > 0);
      if (measured) $('configuration-filter').value = measured.configuration_id+'/'+measured.snapshot_id;
    }
    if (params.get('configuration')) {
      const row = rows.find(r=>r.configuration_id === params.get('configuration') && (!params.get('snapshot') || r.snapshot_id === params.get('snapshot')));
      if (row) $('configuration-filter').value = row.configuration_id+'/'+row.snapshot_id;
      params.delete('configuration');
    }
    configuration();
  }
  function evaluation() {
    const summary = summaries[Number($('evaluation-filter').value)];
    window.fpmNavigation.setRun(summary?.snapshot);
    populate(summary?.rows || []); configurations();
  }
  async function start() {
    window.fpmNavigation.setRun(null);
    const [history, branchCatalog] = await Promise.all([load('data/history.json',true), load('branches.json')]);
    const branches = branchCatalog.branches.map(entry=>entry.branch);
    options($('branch'),branches.map(b=>[b,b]),false); $('branch').disabled = false;
    $('branch').value = branches.includes(params.get('branch')) ? params.get('branch') : 'main';
    if (view === 'trends' && $('branch').value !== 'main') { status('Trends is available for main only. Select main to view history.'); return; }
    if (!history?.entries.length) { status('No completed dashboard evaluation in the retained 90-day window.'); return; }
    entries = history.entries.filter(e=>e.snapshot.branch === $('branch').value && (view !== 'trends' || e.trend));
    if (view === 'detail') entries.sort((a,b)=>(b.revision_order || 0)-(a.revision_order || 0));
    else entries.sort((a,b)=>(a.revision_order || 0)-(b.revision_order || 0));
    summaries = await Promise.all(entries.map(e=>load('data/'+e.summary_path)));
    if (!summaries.length) { status(view === 'trends' ? 'No completed evaluation at or after the Trends baseline.' : 'No completed evaluation for this branch.'); return; }
    status(`${Date.now()-Date.parse(summaries.at(view === 'trends' ? -1 : 0).snapshot.completed_at)>48*3600000 ? 'Stale · ' : ''}${summaries.length} retained evaluations · latest HF ${summaries.at(view === 'trends' ? -1 : 0).snapshot.hf_revision.slice(0,12)}`);
    if (view === 'trends') { populate(summaries.flatMap(s=>s.rows)); trends(); }
    else {
      options($('evaluation-filter'),summaries.map((s,i)=>[String(i),`${s.snapshot.commit_sha.slice(0,12)} · HF ${s.snapshot.hf_revision.slice(0,12)} · ${completed(s.snapshot.completed_at)}`]),false);
      const index = summaries.findIndex(s=>s.snapshot.run_id+'-'+s.snapshot.run_attempt === params.get('run'));
      if (index>=0) $('evaluation-filter').value = String(index);
      evaluation();
    }
  }
  $('branch').addEventListener('change',()=> { const url = new URL(location); url.searchParams.set('branch',$('branch').value); location.assign(url); });
  if (view === 'detail') filters.push(['parallelism-filter','parallelism'],['role-filter','worker_role']);
  for (const [id] of filters) $(id).addEventListener('change',()=>view === 'trends' ? trends() : configurations());
  $('phase-filter').addEventListener('change',()=>view === 'trends' ? trends() : maps());
  if (view === 'detail') {
    $('evaluation-filter').addEventListener('change',evaluation);
    $('configuration-filter').addEventListener('change',configuration);
    $('method-filter').addEventListener('change',maps);
    $('search-filter').addEventListener('input',configurations);
  }
  start().catch(error=>status(error.message));
})();
