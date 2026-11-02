/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
"use strict";
window.AgentX = (() => {
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const avg = values => { const valid = values.filter(finite); return valid.length ? valid.reduce((a,b) => a+b,0)/valid.length : null; };
  const fmt = (v, suffix = "%") => finite(v) ? `${v.toLocaleString(undefined,{maximumFractionDigits:2})}${suffix}` : "—";
  const metrics = ["tpot_ms", "ttft_ms", "output_tokens_s"];
  let generation = 0, activeBranch = null;

  function aggregate(runs) {
    const coverage = {};
    runs.forEach(run => Object.entries(run.coverage).forEach(([key,n]) => coverage[key] = (coverage[key] || 0) + n));
    const request_errors = {};
    for (const metric of ["ttft_ms","tpot_ms"]) {
      const count = runs.reduce((n,r) => n+r.request_errors[metric].count,0);
      const sum = runs.reduce((n,r) => n+r.request_errors[metric].sum_ape,0);
      request_errors[metric] = {count, sum_ape:sum, mape:count ? sum/count : null};
    }
    return {point_errors:Object.fromEntries(metrics.map(m => [m,avg(runs.map(r => r.point_errors[m]))])),point_counts:Object.fromEntries(metrics.map(m=>[m,runs.filter(r=>finite(r.point_errors[m])).length])),request_errors,coverage};
  }

  function cells(result) {
    return metrics.map(m => `<td>${fmt(result.point_errors[m])}<small>n=${result.point_counts[m]} run${result.point_counts[m]===1?"":"s"}</small></td>`).join("") +
      ["ttft_ms","tpot_ms"].map(m => `<td title="${result.request_errors[m].count} scored requests">${fmt(result.request_errors[m].mape)}<small>n=${result.request_errors[m].count.toLocaleString()}</small></td>`).join("") +
      `<td>${(result.coverage.matched || 0).toLocaleString()} / ${(result.coverage.measured || 0).toLocaleString()}</td>`;
  }
  const headings = `<tr><th scope="col">Model / concurrency</th><th scope="col">Hardware / backend</th><th scope="col">Point TPOT error</th><th scope="col">Point TTFT error</th><th scope="col">Point output-throughput error</th><th scope="col">Request TTFT MAPE</th><th scope="col">Request TPOT MAPE</th><th scope="col">Matched / measured</th></tr>`;
  function source(run) {
    return run.category === "agentx" ? `<a href="${esc(run.benchmark_url)}" target="_blank" rel="noopener">AgentX benchmark ↗</a> · <a href="${esc(run.source_run_url)}" target="_blank" rel="noopener">Source run ↗</a>` :
      "Public dataset: coming soon"; // TODO: link the public self-collected E2E HF dataset after migration.
  }
  function pointUrl(run, branch) {
    const params = new URLSearchParams({branch,category:run.category,model:run.model,lanes:run.lanes});
    return `./agentx/point.html?${params}#view=point&stage=profile&variant=observed`;
  }
  async function json(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Evidence unavailable (HTTP ${response.status}).`);
    return response.json();
  }
  async function summary(branch, prefix) {
    const catalog = await json(prefix + "branches.json");
    const entry = catalog.branches.find(e => e.branch === branch);
    if (!entry || entry.status !== "evaluated") return null;
    if (!/^branches\/[0-9a-f]{16}\/summary.json$/.test(entry.summary_path)) throw new Error("Invalid evidence path.");
    const data = await json(prefix + entry.summary_path);
    if (data.schema !== "aisimulate.agentx.accuracy.v1" || data.snapshot.branch !== branch) throw new Error("Evidence branch mismatch.");
    return {data,base:prefix + entry.summary_path.replace(/summary.json$/, ""),entry};
  }
  async function load(branch) {
    const target = document.getElementById("agentx-content");
    if (!target || branch === activeBranch) return;
    activeBranch = branch;
    const request = ++generation;
    target.innerHTML = '<p role="status">Loading AgentX evidence…</p>';
    try {
      const loaded = await summary(branch, "./agentx/");
      if (request !== generation) return;
      if (!loaded) { target.innerHTML = '<p role="status">No AgentX evaluation is available for this branch.</p>'; return; }
      const {data,entry} = loaded;
      target.innerHTML = `<p class="evidence-brief">${esc(branch)} · evaluated ${esc(data.snapshot.commit_sha.slice(0,12))} · ${esc(data.snapshot.completed_at)}${entry.last_update?.status === "failed" ? " · Update failed; showing previous results" : ""}${data.snapshot.preview ? " · Preview" : ""}</p>` +
        [["agentx","AgentX"],["self_collected","Self-collected"]].map(([category,label]) => {
          const runs = data.runs.filter(r => r.category === category), models = [...new Set(runs.map(r=>r.model))];
          return `<section class="agentx-category"><h2>${label}</h2><div class="table-scroll"><table class="agentx-table"><thead>${headings}</thead><tbody>${models.map((model,index) => {
            const points = runs.filter(r=>r.model===model), first=points[0], id=`ax-${category}-${index}`;
            return `<tr class="agentx-model"><th><button type="button" data-agentx-expand="${id}" aria-expanded="false">▸ ${esc(model)}</button></th><td>${esc(first.hardware)} · ${esc(first.backend)}<small>${esc(first.precision)} · TP${first.tensor_parallel}</small></td>${cells(aggregate(points))}</tr>` +
              [...new Set(points.map(r=>r.lanes))].sort((a,b)=>a-b).map(lanes => {
                const matches=points.filter(r=>r.lanes===lanes), point=matches[0], failed=matches.filter(r=>r.outcome!=="completed").length;
                return `<tr data-agentx-parent="${id}" data-agentx-href="${esc(pointUrl(point,branch))}" tabindex="0" aria-label="${lanes} lanes: open details" hidden><th><a href="${esc(pointUrl(point,branch))}">${lanes} lane${lanes===1?"":"s"} ↗</a><small>${matches.length} run${matches.length===1?"":"s"}${failed?` · ${failed} unavailable`:""}</small></th><td>${source(point)}</td>${cells(aggregate(matches))}</tr>`;
              }).join("");
          }).join("")}</tbody></table></div></section>`;
        }).join("") + `<details><summary>Metrics and provenance</summary><p>Point columns average each run’s absolute percentage error equally. Throughput is output tokens/s. Request columns average individual matched request errors, with equal weight per request. Every measured request remains in coverage; preparation and warmup are excluded. Undefined errors remain unavailable. No shape metrics or timing corrections.</p><p>Silicon revision: ${esc(data.snapshot.dataset_revision)}<br>Evaluated commit: ${esc(data.snapshot.commit_sha)}</p></details>`;
      target.querySelectorAll("[data-agentx-href]").forEach(row => {
        row.addEventListener("click", event => { if (!event.target.closest("a,button")) location.href=row.dataset.agentxHref; });
        row.addEventListener("keydown", event => { if (event.key === "Enter" && event.target === row) location.href=row.dataset.agentxHref; });
      });
      target.querySelectorAll("[data-agentx-expand]").forEach(button => button.addEventListener("click", () => {
        const expanded=button.getAttribute("aria-expanded")!=="true";
        button.setAttribute("aria-expanded",String(expanded));
        button.textContent=(expanded?"▾ ":"▸ ")+button.textContent.slice(2);
        target.querySelectorAll(`[data-agentx-parent="${button.dataset.agentxExpand}"]`).forEach(row=>row.hidden=!expanded);
      }));
    } catch (error) {
      if (request === generation) target.innerHTML=`<p role="status">${esc(error.message)}</p>`;
    }
  }

  async function detail() {
    const root=document.getElementById("agentx-point"), params=new URL(location.href).searchParams;
    const branch=params.get("branch")||"main";
    try {
      const loaded=await summary(branch,"./");
      if(!loaded) throw new Error("No AgentX evidence is available for this branch.");
      const runs=loaded.data.runs.filter(r=>r.category===params.get("category") && r.model===params.get("model") && r.lanes===Number(params.get("lanes")));
      if(!runs.length) throw new Error("This concurrency was not evaluated.");
      const group=aggregate(runs), first=runs[0];
      document.getElementById("back").href=`../?${new URLSearchParams({tab:"agentx",branch})}`;
      document.getElementById("point-title").textContent=`${first.model} · ${first.lanes} lane${first.lanes===1?"":"s"}`;
      root.innerHTML=`<p>${esc(first.hardware)} · ${esc(first.backend)} · ${esc(first.precision)} · TP${first.tensor_parallel} · ${source(first)}</p><h2>Concurrency summary · all workload variants and repeats</h2><div class="table-scroll"><table class="agentx-table"><thead>${headings}</thead><tbody><tr><th>${first.lanes} lane${first.lanes===1?"":"s"}</th><td>${esc(first.hardware)} · ${esc(first.backend)}</td>${cells(group)}</tr></tbody></table></div><div id="variant-controls" class="filter-toolbar"></div><div id="selected-run"></div>`;
      const controls=document.getElementById("variant-controls");
      if(first.category==="self_collected") {
        controls.innerHTML=`<label>Initial prompt<select id="length"><option value="4096">4K</option><option value="32768">32K</option></select></label><label>Initial cache<select id="cache"><option value="cold">Cold</option><option value="warm">Warm</option></select></label><label>Repeat<select id="repeat"><option>0</option><option>1</option><option>2</option></select></label>`;
        for(const [key, fallback] of [["length","4096"],["cache","cold"],["repeat","0"]]) {
          const select=document.getElementById(key), value=params.get(key)||fallback;
          select.value=[...select.options].some(o=>o.value===value)?value:fallback;
        }
      }
      let detailGeneration=0;
      async function select() {
        const version=++detailGeneration;
        const length=document.getElementById("length")?.value, cache=document.getElementById("cache")?.value, repeat=document.getElementById("repeat")?.value;
        const run=first.category==="agentx"?first:runs.find(r=>r.initial_input_tokens===Number(length) && r.cache_state===cache && r.repeat===Number(repeat));
        if(!run) return;
        const url=new URL(location.href);
        if(length) for(const [key,value] of Object.entries({length,cache,repeat})) url.searchParams.set(key,value);
        history.replaceState(null,"",url);
        const target=document.getElementById("selected-run");
        target.innerHTML='<p role="status">Loading selected run…</p>';
        if(!/^details\/[a-z0-9-]+\.json$/.test(run.detail_path)) throw new Error("Invalid detail path.");
        const point=await json(loaded.base+run.detail_path);
        if(version!==detailGeneration)return;
        if(point.id!==run.id || point.snapshot.commit_sha!==loaded.data.snapshot.commit_sha) throw new Error("Point provenance mismatch.");
        const variant=first.category==="agentx"?"Observed outputs":`${Number(length).toLocaleString()} tokens · ${cache} · repeat ${repeat}`;
        const comparisons=metrics.map(metric=>`<tr><th>${{ttft_ms:"Mean TTFT (ms)",tpot_ms:"Mean TPOT (ms/token)",output_tokens_s:"Output tokens/s"}[metric]}</th><td>${fmt(point.silicon[metric],"")}</td><td>${fmt(point.prediction[metric],"")}</td><td>${fmt(point.point_errors[metric])}</td></tr>`).join("");
        target.innerHTML=`<h2>Selected run · ${esc(variant)}</h2><p>${esc(point.outcome)}${point.reason?` · ${esc(point.reason.replaceAll("_"," "))}`:""} · ${point.coverage.matched||0} / ${point.coverage.measured} matched measured requests</p><div class="table-scroll"><table><thead><tr><th>Metric</th><th>Silicon</th><th>AISim</th><th>Point error</th></tr></thead><tbody>${comparisons}</tbody></table></div><p>Request TTFT MAPE: ${fmt(point.request_errors.ttft_ms.mape)} (n=${point.request_errors.ttft_ms.count}) · Request TPOT MAPE: ${fmt(point.request_errors.tpot_ms.mape)} (n=${point.request_errors.tpot_ms.count})</p><details><summary>Request comparisons and coverage</summary><p>${Object.entries(point.coverage).map(([k,v])=>`${esc(k.replaceAll("_"," "))}: ${v}`).join(" · ")}</p><div class="table-scroll agentx-pairs"><table><thead><tr><th>Request</th><th>Status</th><th>Silicon TTFT</th><th>AISim TTFT</th><th>TTFT error</th><th>Silicon TPOT</th><th>AISim TPOT</th><th>TPOT error</th></tr></thead><tbody>${point.pairs.map(p=>`<tr><td>${p.request}</td><td>${esc(p.status.replaceAll("_"," "))}${p.input_length_matches===false?" · different input length":""}</td>${["silicon_ttft_ms","predicted_ttft_ms","ttft_ms_ape","silicon_tpot_ms","predicted_tpot_ms","tpot_ms_ape"].map(k=>`<td>${fmt(p[k],k.endsWith("ape")?"%":"")}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details><details><summary>Configuration, provenance, and limitations</summary><dl>${point.config_summary.map(item=>`<dt>${esc(item.name)}</dt><dd>${esc(item.value ?? "Unavailable")}</dd>`).join("")}</dl><p>Measured backend: ${esc(point.measured_backend_version)} · prediction database: ${esc(point.prediction_backend_version||"unavailable")}<br>Evaluated commit: ${esc(point.snapshot.commit_sha)}<br>Silicon revision: ${esc(point.snapshot.dataset_revision)}<br>Generated: ${esc(point.snapshot.completed_at)}</p><p>${point.boundary==="complete_finite_cohort"?"Finite measured cohort, including between-turn delays; preparation excluded.":"Successful profiling cohort; silicon and replay may progress through different requests during the fixed window."} Request comparisons require a unique successful match and equal output lengths. Matching source IDs does not imply identical prompts or cache history. Missing and zero-denominator errors remain unavailable. Accuracy is advisory; no timing correction is applied.</p></details>${point.report?'<iframe id="replay-frame" title="Replay timeline, latency, throughput, and cache details" class="agentx-replay"></iframe>':'<p>Native replay charts are unavailable for this outcome.</p>'}`;
        if(point.report) {
          const data=loaded.base.replace(/^\.\//,"")+run.detail_path;
          const state=new URLSearchParams(location.hash.slice(1));
          const stage=state.get("stage")==="preparation"?"preparation":"profile";
          const view=state.get("view")==="timeline"?"timeline":"point";
          document.getElementById("replay-frame").src=`./replay.html?${new URLSearchParams({data})}#view=${view}&stage=${stage}&variant=observed`;
        }
      }
      controls.addEventListener("change",()=>select().catch(error=>document.getElementById("selected-run").textContent=error.message));
      await select();
    } catch(error) { root.innerHTML=`<p role="alert">${esc(error.message)}</p>`; }
  }
  window.addEventListener("message", event => {
    const frame=document.getElementById("replay-frame"), state=event.data?.agentxReplayState;
    if(event.origin!==location.origin || event.source!==frame?.contentWindow || !state) return;
    if(!["preparation","profile"].includes(state.stage) || !["dashboard","timeline"].includes(state.view)) return;
    const url=new URL(location.href);
    url.hash=new URLSearchParams({view:state.view==="timeline"?"timeline":"point",stage:state.stage,variant:"observed"});
    history.replaceState(null,"",url);
  });
  return {load,detail,aggregate};
})();
