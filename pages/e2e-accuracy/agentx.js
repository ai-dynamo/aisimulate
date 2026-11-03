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
      let silicon=null,prediction=null;
      if(runs.every(r=>Array.isArray(r.pairs))) {
        const pairs=runs.flatMap(r=>r.pairs).filter(p=>finite(p[metric+"_ape"]));
        if(pairs.length!==count || pairs.some(p=>!finite(p["silicon_"+metric]) || !finite(p["predicted_"+metric]))) throw new Error("Request measurements do not reconcile with scored counts.");
        silicon=avg(pairs.map(p=>p["silicon_"+metric]));
        prediction=avg(pairs.map(p=>p["predicted_"+metric]));
      }
      request_errors[metric] = {count, sum_ape:sum, mape:count ? sum/count : null,silicon,prediction};
    }
    const values=Object.fromEntries(["silicon","prediction"].map(source=>[source,Object.fromEntries(metrics.map(m=>[m,avg(runs.filter(r=>finite(r.point_errors[m])).map(r=>r[source]?.[m]))]))]));
    return {...values,point_errors:Object.fromEntries(metrics.map(m => [m,avg(runs.map(r => r.point_errors[m]))])),point_counts:Object.fromEntries(metrics.map(m=>[m,runs.filter(r=>finite(r.point_errors[m])).length])),request_errors,coverage};
  }

  function cells(result) {
    return metrics.map(m => `<td>${fmt(result.point_errors[m])}<small>n=${result.point_counts[m]} run${result.point_counts[m]===1?"":"s"}</small></td>`).join("") +
      ["ttft_ms","tpot_ms"].map(m => `<td title="${result.request_errors[m].count} scored requests">${fmt(result.request_errors[m].mape)}<small>n=${result.request_errors[m].count.toLocaleString()}</small></td>`).join("") +
      `<td>${(result.coverage.matched || 0).toLocaleString()} / ${(result.coverage.measured || 0).toLocaleString()}</td>`;
  }
  function comparisonRows(result) {
    return [
      ["point","tpot_ms","TPOT","ms/token"],
      ["point","ttft_ms","TTFT","ms"],
      ["point","output_tokens_s","Output throughput","output tokens/s"],
      ["request","ttft_ms","TTFT","ms"],
      ["request","tpot_ms","TPOT","ms/token"],
    ].map(([scope,metric,label,unit])=>{
      const request=scope==="request",values=request?result.request_errors[metric]:null;
      const count=request?values.count:result.point_counts[metric];
      const samples=`${count.toLocaleString()} ${request?"request":"run"}${count===1?"":"s"}`;
      return `<tr data-metric="${scope}-${metric}"${request&&metric==="ttft_ms"?' class="agentx-request-start"':""}><th scope="row">${request?"Request":"Point"} ${label}<small>${unit}</small></th><td>${fmt(request?values.silicon:result.silicon[metric],"")}</td><td>${fmt(request?values.prediction:result.prediction[metric],"")}</td><td class="agentx-error"><strong>${fmt(request?values.mape:result.point_errors[metric])}</strong><small>${request?"MAPE":"Mean point error"}</small><small class="agentx-inline-count">${samples}</small></td><td class="agentx-samples">${samples}</td></tr>`;
    }).join("");
  }
  function featureChecklist(point) {
    const primary = {
      "agentx.weka":"Weka dependency graph", "agentx.seeded_snapshots":"Seeded snapshots",
      "agentx.live_warmup":"Live-trajectory warmup", "agentx.continuous_profile":"Continuous profile",
      "agentx.sequential_recycling":"Sequential lane recycling", "cache.g1_prefix":"GPU prefix caching",
      "speculation":"Speculative decoding", "cache.g2":"Host offload (G2)",
      "cache.g3":"Storage offload (G3)", "cache.kvcr":"Peer KV fetch (KVCR)",
      "routing.planner":"KV-aware routing / Planner",
    };
    const labels = {
      ...primary,
      "agentx.cache_bust":"Cache bust", "agentx.shuffle":"Random / shuffled recycling",
      "scheduling.continuous_batching":"Continuous batching", "scheduling.chunked_prefill":"Chunked prefill",
      "scheduling.async":"Asynchronous scheduling", "cache.hybrid_state":"Hybrid KV / recurrent state",
      "topology.aggregated":"Aggregated serving", "topology.tensor_parallel":"Tensor parallelism",
      "topology.expert_parallel":"Expert parallelism", "topology.attention_dp":"Attention data parallelism",
      "topology.pipeline_parallel":"Pipeline parallelism", "topology.context_parallel":"Context parallelism",
      "topology.multi_node":"Multi-node serving", "topology.pd":"Prefill/decode disaggregation",
      "precision.quantized_weights":"Quantized weights", "precision.quantized_kv":"Quantized KV",
      "workload.multimodal":"Multimodal input", "events.subagent_spawn":"Subagent spawning",
      "events.join":"Dependency joins", "events.tool_wait":"Tool waits", "events.cache_reuse":"Cache reuse",
      "events.eviction":"Cache eviction", "events.preemption":"Request preemption",
      "events.recompute":"Recomputation", "events.state_restore":"Recurrent-state restoration",
    };
    const core=Object.keys(primary), record=point.feature_metadata;
    const valid=record?.schema==="aisimulate.benchmark.features.v1" && record.benchmark_id===point.id && Array.isArray(record.features);
    const features=new Map((valid?record.features:[]).filter(f=>Object.hasOwn(labels,f.id)).map(f=>[f.id,f]));
    const statuses={enabled:["on","✓","Enabled"],disabled:["off","×","Off"],not_applicable:["na","–","N/A"]};
    const rows=ids=>ids.map(id=>{
      const feature=features.get(id), state=feature?.state;
      const [status,icon,label]=Object.hasOwn(statuses,state)?statuses[state]:["unknown","?","Not recorded"];
      const observed=feature?.observed;
      const usage=observed?.state==="exercised" && Number.isInteger(observed.count) && observed.count>0 ? `Observed · ${observed.count.toLocaleString()}` : observed?.state==="not_observed" && observed.count===0 ? "Not observed · 0" : "";
      const parameters=(feature?.parameters||[]).map(p=>`${p.name.replaceAll("_"," ")}: ${p.value??"not recorded"}`).join(" · ");
      return `<li class="agentx-feature ${status}" data-feature="${id}"><span class="agentx-feature-icon" aria-hidden="true">${icon}</span><span class="agentx-feature-name">${labels[id]}${parameters?`<small>${esc(parameters)}</small>`:""}${usage?`<small>${usage}</small>`:""}</span><span class="agentx-feature-status">${label}</span></li>`;
    }).join("");
    const more=[...features.keys()].filter(id=>!core.includes(id));
    return `<aside class="agentx-features" aria-labelledby="agentx-features-title"><div class="agentx-features-heading"><h2 id="agentx-features-title">Benchmark features</h2><a href="../../feature-support-matrix/#agentx">Support matrix ↗</a></div><p class="agentx-feature-scope">Selected run · silicon settings</p>${valid?"":'<p class="agentx-feature-missing">Feature settings were not recorded for this historical run.</p>'}<ul class="agentx-feature-list">${rows(core)}</ul>${more.length?`<details class="agentx-more-features"><summary>More features (${more.length})</summary><ul class="agentx-feature-list">${rows(more)}</ul></details>`:""}<p class="agentx-feature-note">Enabled does not imply exercised. Observed counts appear only when recorded.</p></aside>`;
  }
  const headings = `<tr><th scope="col">Model / concurrency</th><th scope="col">Hardware / backend</th><th scope="col">Point TPOT error</th><th scope="col">Point TTFT error</th><th scope="col">Point output-throughput error</th><th scope="col">Request TTFT MAPE</th><th scope="col">Request TPOT MAPE</th><th scope="col">Matched / measured</th></tr>`;
  function source(run) {
    return run.category === "agentx" ? `<a href="${esc(run.benchmark_url)}" target="_blank" rel="noopener">AgentX benchmark ↗</a> · <a href="${esc(run.source_run_url)}" target="_blank" rel="noopener">Source run ↗</a>` :
      "AISim self-collected dataset: coming soon"; // TODO: link the public self-collected E2E HF dataset after migration.
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
        `<div class="table-scroll"><table class="agentx-table"><thead>${headings}</thead><tbody>${[...new Set(data.runs.map(r=>JSON.stringify([r.category,r.model,r.hardware,r.backend,r.precision,r.tensor_parallel])))].map((key,index) => {
          const points=data.runs.filter(r=>JSON.stringify([r.category,r.model,r.hardware,r.backend,r.precision,r.tensor_parallel])===key), first=points[0], id=`ax-model-${index}`;
          return `<tr class="agentx-model"><th><button type="button" data-agentx-expand="${id}" aria-expanded="true">▾ ${esc(first.model)}</button><small class="source-tag">${first.category==="agentx"?"AgentX official":"Self-collected"}</small></th><td>${esc(first.hardware)} · ${esc(first.backend)}<small>${esc(first.precision)} · TP${first.tensor_parallel}</small></td>${cells(aggregate(points))}</tr>` +
            [...new Set(points.map(r=>r.lanes))].sort((a,b)=>a-b).map(lanes=>{
              const matches=points.filter(r=>r.lanes===lanes), point=matches[0], failed=matches.filter(r=>r.outcome!=="completed").length;
              return `<tr data-agentx-parent="${id}" data-agentx-href="${esc(pointUrl(point,branch))}" tabindex="0" aria-label="${lanes} lanes: open details"><th><a href="${esc(pointUrl(point,branch))}">${lanes} lane${lanes===1?"":"s"} ↗</a><small>${matches.length} run${matches.length===1?"":"s"}${failed?` · ${failed} unavailable`:""}</small></th><td>${source(point)}</td>${cells(aggregate(matches))}</tr>`;
            }).join("");
        }).join("")}</tbody></table></div><details class="agentx-method"><summary>Metrics and provenance</summary>${methodology()}${provenance(data.snapshot,data.runs)}</details>`;
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

  const githubIcon = `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.49c-2.23.48-2.69-.95-2.69-.95-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.8.06 1.22.82 1.22.82.71 1.22 1.87.87 2.33.67.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 4 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.74.54 1.49v2.21c0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>`;
  function methodology() {
    return `<ul class="agentx-notes"><li>Point errors average each run’s absolute percentage error equally across repeats and variants.</li><li>Request MAPE gives each valid, uniquely matched request equal weight. All measured requests remain in accounting.</li><li>Throughput is output tokens/s within the benchmark’s measured boundary, including between-turn delays.</li><li>Preparation and warmup are unscored. Missing measurements and zero denominators remain unavailable. No shape metrics or timing calibration.</li></ul>`;
  }
  function provenance(snapshot,runs) {
    return `<ul class="agentx-notes provenance"><li>${githubIcon} <strong>AISim repository commit:</strong> <a href="https://github.com/ai-dynamo/aisimulate/commit/${esc(snapshot.commit_sha)}" target="_blank" rel="noopener"><code>${esc(snapshot.commit_sha)}</code></a></li><li><span role="img" aria-label="Hugging Face">🤗</span> <strong>Silicon dataset revision:</strong> <code>${esc(snapshot.dataset_revision)}</code></li><li><strong>Measured framework:</strong> ${[...new Set(runs.map(r=>`${r.backend} ${r.measured_backend_version}`))].map(esc).join("; ")}</li><li><strong>Prediction database:</strong> ${[...new Set(runs.map(r=>`${r.backend} ${r.prediction_backend_version||"unavailable"}`))].map(esc).join("; ")}</li><li><strong>Evaluated:</strong> ${esc(snapshot.completed_at)}${snapshot.preview?" · Preview":""}</li><li><strong>Wheel SHA256:</strong> <code>${esc(snapshot.wheel_sha256)}</code></li>${[...new Map(runs.flatMap(r=>(r.op_data||[]).map(d=>[d.file,d]))).values()].map(d=>`<li>${githubIcon} <strong>Op data:</strong> <a href="https://github.com/ai-dynamo/aisimulate/blob/${esc(snapshot.commit_sha)}/python/aisimulate/src/aisimulate_core/systems/data/${esc(d.file)}">${esc(d.file)}</a><br><code>${esc(d.sha256)}</code></li>`).join("")}</ul>`;
  }
  async function detail() {
    const root=document.getElementById("agentx-point"), params=new URL(location.href).searchParams, branch=params.get("branch")||"main";
    let renderer;
    try {
      const catalog=await json("./branches.json"), branchSelect=document.getElementById("branch-select");
      branchSelect.innerHTML=catalog.branches.map(e=>`<option value="${esc(e.branch)}">${esc(e.branch)}</option>`).join("");
      branchSelect.value=branch;branchSelect.disabled=false;
      branchSelect.addEventListener("change",()=>{const url=new URL(location.href);url.searchParams.set("branch",branchSelect.value);location.href=url;});
      document.getElementById("back").href=`../?${new URLSearchParams({tab:"agentx",branch})}`;
      const loaded=await summary(branch,"./");
      if(!loaded) throw new Error("No AgentX evidence is available for this branch.");
      const runs=loaded.data.runs.filter(r=>r.category===params.get("category") && r.model===params.get("model") && r.lanes===Number(params.get("lanes")));
      if(!runs.length) throw new Error("This concurrency was not evaluated.");
      const first=runs[0];
      document.getElementById("point-title").textContent=`${first.model} · ${first.lanes} lane${first.lanes===1?"":"s"}`;
      root.innerHTML=`<p class="agentx-identity"><span class="source-tag">${first.category==="agentx"?"AgentX official":"Self-collected"}</span> ${esc(first.hardware)} · ${esc(first.backend)} · ${esc(first.precision)} · TP${first.tensor_parallel} · ${source(first)}</p><div id="variant-controls" class="filter-toolbar"></div><p id="selected-run-label" class="evidence-brief"></p><div id="selected-run"></div>`;
      const controls=document.getElementById("variant-controls");
      controls.hidden=first.category!=="self_collected";
      if(first.category==="self_collected") {
        controls.innerHTML=`<label>Initial prompt<select id="length"><option value="4096">4K</option><option value="32768">32K</option></select></label><label>Initial cache<select id="cache"><option value="cold">Cold</option><option value="warm">Warm</option></select></label><label>Repeat<select id="repeat"><option>0</option><option>1</option><option>2</option></select></label>`;
        for(const [key,fallback] of [["length","4096"],["cache","cold"],["repeat","0"]]) {
          const select=document.getElementById(key),value=params.get(key)||fallback;select.value=[...select.options].some(o=>o.value===value)?value:fallback;
        }
      }
      document.getElementById("selected-run").textContent="Loading concurrency measurements…";
      const [fragment,points]=await Promise.all([
        fetch("./dashboard-body.html").then(r=>{if(!r.ok)throw new Error("Replay assets unavailable.");return r.text();}),
        Promise.all(runs.map(async run=>{
          if(!/^details\/[a-z0-9-]+\.json$/.test(run.detail_path))throw new Error("Invalid detail path.");
          const point=await json(loaded.base+run.detail_path);
          if(point.id!==run.id||JSON.stringify(point.snapshot)!==JSON.stringify(loaded.data.snapshot)||JSON.stringify(point.request_errors)!==JSON.stringify(run.request_errors))throw new Error("Point provenance or request summary mismatch.");
          return point;
        })),
      ]);
      const group=aggregate(points);
      async function select() {
        const length=document.getElementById("length")?.value,cache=document.getElementById("cache")?.value,repeat=document.getElementById("repeat")?.value;
        const run=first.category==="agentx"?first:runs.find(r=>r.initial_input_tokens===Number(length)&&r.cache_state===cache&&r.repeat===Number(repeat));
        if(!run)throw new Error("Selected workload is unavailable.");
        const url=new URL(location.href);if(length)for(const [key,value] of Object.entries({length,cache,repeat}))url.searchParams.set(key,value);history.replaceState(null,"",url);
        const target=document.getElementById("selected-run");renderer?.destroy();target.innerHTML='<p role="status">Loading selected run…</p>';
        const point=points.find(p=>p.id===run.id);
        document.getElementById("selected-run-label").textContent=`Selected run · ${first.category==="agentx"?"Observed outputs":`${Number(length).toLocaleString()} tokens · ${cache} · repeat ${repeat}`} · ${point.outcome}${point.reason?" · "+point.reason.replaceAll("_"," "):""}`;
        const lanes=`${first.lanes} lane${first.lanes===1?"":"s"}`;
        const accuracy=`<div class="agentx-summary-layout"><section class="agentx-summary-main"><h2>Concurrency summary</h2><p>${runs.length>1?"All ":""}${runs.length} run${runs.length===1?"":"s"} · ${lanes} · ${group.coverage.matched.toLocaleString()} / ${group.coverage.measured.toLocaleString()} requests matched</p><div class="table-scroll"><table class="agentx-summary-table"><thead><tr><th scope="col">Metric</th><th scope="col"><span class="comparison-source-dot silicon" aria-hidden="true"></span>Silicon</th><th scope="col"><span class="comparison-source-dot aisim" aria-hidden="true"></span>AISim</th><th scope="col">Error</th><th scope="col" class="agentx-samples">Samples</th></tr></thead><tbody>${comparisonRows(group)}</tbody></table></div><p class="agentx-summary-note">Point rows average scored runs equally. Request rows average the individual requests scored for that metric.</p></section>${featureChecklist(point)}</div>`;
        const evidence=`<h2>Metrics and provenance</h2>${methodology()}${provenance(point.snapshot,[point])}<h2>Measured configuration</h2><ul>${point.config_summary.map(item=>`<li><strong>${esc(item.name)}:</strong> ${esc(item.value??"Unavailable")}</li>`).join("")}</ul><h2>Sources and limitations</h2><ul><li>${source(point)}</li><li>${point.boundary==="complete_finite_cohort"?"Finite measured cohort, including between-turn delays.":"Fixed profiling admission window; silicon and replay may progress through different requests."}</li><li>Request comparison requires a unique successful match and equal output lengths. Source identity does not establish identical prompts or cache history.</li><li>Curve MAPE uses covered, positive silicon observations on a one-second phase grid. Curve smoothing is described per chart. Zero and missing values are counted; preparation remains unscored.</li><li>Accuracy is advisory. No timing calibration is applied.</li></ul>`;
        target.innerHTML=`<div class="replay-dashboard embedded ${document.documentElement.dataset.theme==="light"?"light":""}">${fragment}</div>`;
        const state=new URLSearchParams(location.hash.slice(1)),view=state.get("view"),stage=state.get("stage")==="preparation"?"preparation":"profile";
        const report=point.report||{metadata:{},prediction:{},telemetry:{completion_status:point.outcome},phases:{profile:{}},requests:[],samples:[],lifecycle:[]};
        report.reference={pairs:point.pairs};
        renderer=AISimReplay.mount(target.firstElementChild,report,{accuracy,evidence,scoring:point.outcome==="completed",view:["timeline","evidence"].includes(view)?view:"dashboard",stage,
          onState:s=>{const u=new URL(location.href);u.hash=new URLSearchParams({view:s.view==="dashboard"?"point":s.view,stage:s.stage,variant:"observed"});history.replaceState(null,"",u);}});
      }
      controls.addEventListener("change",()=>select().catch(e=>document.getElementById("selected-run").textContent=e.message));
      window.addEventListener("aisim-theme",()=>renderer?.render());
      window.addEventListener("hashchange",()=>{if(!renderer)return;const hash=new URLSearchParams(location.hash.slice(1));renderer.state.view=["timeline","evidence"].includes(hash.get("view"))?hash.get("view"):"dashboard";renderer.state.stage=hash.get("stage")==="preparation"?"preparation":"profile";renderer.render();});
      await select();
    }catch(error){root.innerHTML=`<p role="alert">${esc(error.message)}</p>`;}
  }
  return {load,detail,aggregate,featureChecklist};
})();
