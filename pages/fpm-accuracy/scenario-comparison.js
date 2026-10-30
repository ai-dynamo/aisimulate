/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
(() => {
  "use strict"
  const root = document.getElementById("scenario-comparison")
  const get = id => document.getElementById("sc-" + id)
  const branchSelect = document.getElementById("branch")
  const METHODS = ["warmup", "nowarmup", "linear", "spline", "gbdt", "hybrid", "selector"]
  const SCENARIOS = ["a", "b", "c", "ac", "bc"]
  const TRAIN_PERCENTAGES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
  const methods = () => METHODS
  const splitGrid = () => data !== null
  const onlineScenarios = ["c", "ac", "bc"]
  const usesSplit = id => id === "b" || id === "bc"
  const labels = {warmup: "Interpolation · KV warmup on", nowarmup: "Interpolation · KV warmup off", linear: "Linear regression", spline: "Spline regression", gbdt: "GBDT", hybrid: "GBDT + hybrid calibration", selector: "GBDT selector"}
  const scenarios = {
    a: {code: "A", label: "A only", short: "Self-benchmark → frozen"},
    b: {code: "B", label: "B only", short: "Train / test → frozen"},
    c: {code: "C", label: "C only", short: "Strict cold start → online"},
    ac: {code: "A + C", label: "A + C", short: "Self-benchmark → online"},
    bc: {code: "B + C", label: "B + C", short: "Past data → online"},
  }
  const partitions = {
    a: "Fit the self-benchmark library, then score every eligible collected measurement with the model frozen.",
    c: "Start with no offline training data. Score every eligible measurement in chronological order, including cold-start misses. Only observations admitted at five-second ticks may train the model, including during warmup.",
    ac: "Fit the self-benchmark library, then score every eligible collected measurement in chronological order. Admit one scored observation per five-second tick for adaptation.",
  }
  const cadence = "Ticks begin five seconds after the first scored event timestamp. At each due tick, select the first observation at or after it, predict and score it, then submit only that observation. Warmup follows this same five-second schedule. Intervening observations are scored but never submitted, queued, or accumulated. A long gap consumes one tick without catch-up calls. Tied timestamps consume a tick once; later tied observations may see that update. There is no final flush."
  const chronology = "C, A+C, and B+C require declared chronology and usable recorded event timestamps. If either is missing, fitting, prediction, and updating remain unavailable; no synthetic clock is used."
  const statuses = {
    no_fpm_input: "Missing input", no_training_data: "No training data", no_test_data: "No test data",
    chronology_unavailable: "Chronology unavailable", unsupported_predictor: "Predictor unavailable",
    predictor_error: "Predictor error", no_measurements: "No measurements", unsupported_protocol: "Protocol unavailable",
    supporting_evidence_only: "Supporting data only", split_unavailable: "Split results unavailable",
    missing_result: "Result unavailable",
  }
  const metricLabels = {mape: "MAPE", coverage: "Prediction coverage", fit: "Initial training", predict: "Prediction cost", update: "Online update cost"}
  const numeric = value => typeof value === "number" && Number.isFinite(value)
  const number = value => numeric(value) ? value.toLocaleString("en-US", {maximumFractionDigits: 2}) : "Unavailable"
  const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[c])
  const object = value => value !== null && typeof value === "object" && !Array.isArray(value)
  const key = row => row.configuration_id + "::" + (row.snapshot_id || "")
  const params = new URLSearchParams(location.search)
  let scenario = SCENARIOS.includes(params.get("scenario")) ? params.get("scenario") : "a"
  let trainPercent = TRAIN_PERCENTAGES.includes(Number(params.get("train_percent"))) ? Number(params.get("train_percent")) : 50
  let selectedKey = null, evidence = null, catalog = null, data = null, request = 0
  let splitIssues = new Map()
  let sort = {method: null, direction: 1}

  function splitDescription(id) {
    return id === "b" ? trainPercent + "% training / " + (100 - trainPercent) + "% frozen testing"
      : trainPercent + "% offline B / " + (100 - trainPercent) + "% online C"
  }

  function scenarioTitle(id) {
    return scenarios[id].label + (usesSplit(id) ? " · " + splitDescription(id) : "")
  }

  function partition(id) {
    if (!usesSplit(id)) return partitions[id]
    if (id === "b") return "Within each workload kind in recorded order, allocate " + trainPercent + "% to training and the remainder to frozen testing. Training positions are evenly distributed without inspecting measured latencies, with ceil(N × " + trainPercent + " / 100) training observations per workload. At 50%, odd-numbered observations train and even-numbered observations test. Small groups can differ from the requested percentage."
    return "Train offline on the first floor(N × " + trainPercent + " / 100) observations in chronological order, then score the remainder with online adaptation. Each prediction uses the offline selection and earlier admitted test observations. The five-second schedule starts at the first test timestamp."
  }

  const unsupportedCombination = (method, id) => (["warmup", "nowarmup"].includes(method) && id !== "a")
    || (["hybrid", "selector"].includes(method) && !onlineScenarios.includes(id))
  function checkSplitGrid(row) {
    const object = value => value !== null && typeof value === "object" && !Array.isArray(value)
    const keysMatch = (value, expected) => object(value) && Object.keys(value).sort().join("|") === expected.slice().sort().join("|")
    if (!keysMatch(row.split_results, TRAIN_PERCENTAGES.map(String))) return "The complete 10%–90% split grid is missing or malformed for this configuration."
    for (const percent of TRAIN_PERCENTAGES) {
      const grid = row.split_results[String(percent)]
      if (!keysMatch(grid, ["b", "bc"])) return "The " + percent + "% grid must contain B and B+C results."
      for (const id of ["b", "bc"]) {
        if (!keysMatch(grid[id], methods()) || methods().some(method => unsupportedCombination(method, id)
          ? grid[id][method] !== null : !object(grid[id][method]) || typeof grid[id][method].status !== "string")) {
          return "The " + percent + "% " + scenarios[id].label + " method results are incomplete or malformed."
        }
      }
    }
    const canonical = value => Array.isArray(value) ? value.map(canonical) : object(value)
      ? Object.fromEntries(Object.keys(value).sort().map(name => [name, canonical(value[name])])) : value
    for (const id of ["b", "bc"]) if (JSON.stringify(canonical(row.split_results["50"][id])) !== JSON.stringify(canonical(row.results?.[id]))) {
      return "The 50% split grid disagrees with the default results for this configuration."
    }
    return null
  }


  function resultFor(row, id, method) {
    if (unsupportedCombination(method, id)) return null
    if (usesSplit(id)) {
      const problem = splitIssues.get(key(row))
      if (problem) return {status: "split_unavailable", reason: problem + " No default-ratio result is substituted."}
      return row.split_results[String(trainPercent)][id][method]
    }
    return row.results?.[id]?.[method] || {status: "missing_result", reason: "No result was published for this supported method and scenario."}
  }
  function updateURL() {
    const url = new URL(location.href)
    const selected = data?.rows.find(row => key(row) === selectedKey)
    const values = {branch: branchSelect.value, scenario, model: get("model").value, gpu: get("gpu").value, framework: get("framework").value,
      workload: get("workload").value, metric: get("metric").value, configuration: selected?.configuration_id, snapshot: selected?.snapshot_id,
      show_unavailable: get("show-unavailable").checked ? "1" : null, train_percent: splitGrid() && trainPercent !== 50 ? String(trainPercent) : null}
    for (const [name, value] of Object.entries(values)) {
      if (value && !(name === "workload" && value === "all") && !(name === "metric" && value === "mape")) url.searchParams.set(name, value)
      else url.searchParams.delete(name)
    }
    history.replaceState(null, "", url)
  }


  function explain() {
    root.dataset.methodCount = String(METHODS.length)
    get("train-percent").value = String(trainPercent)
    get("train-percent").disabled = !data || !data.rows.length || splitIssues.size === data.rows.length
    get("train-percent").setAttribute("aria-valuetext", trainPercent + "% offline training, " + (100 - trainPercent) + "% testing or online replay")
    get("split-value").textContent = trainPercent + "%"
    get("split-meaning").textContent = "B only: " + splitDescription("b") + ". B + C: " + splitDescription("bc") + "."
    get("split-note").dataset.state = splitIssues.size ? "unavailable" : "ready"
    get("split-note").textContent = !data ? "Waiting for split results."
      : splitIssues.size ? number(splitIssues.size) + " configurations have an incomplete or invalid split grid. Their B and B+C results are unavailable; show unavailable configurations to inspect the reason."
      : "Precomputed results at 10% steps. B distributes training within each workload; B+C uses a chronological prefix. A, C, and A+C stay unchanged."
    get("scenario-code").textContent = scenarios[scenario].code
    get("scenario-description").textContent = partition(scenario)
    const protocolLabel = "fpm-scenarios-v13 · Strict C · One observation per five-second tick, including warmup"
    get("protocol-label").textContent = data ? protocolLabel : "No scenario artifact loaded"
    get("protocol-version-note").textContent = data ? protocolLabel : "Strict C protocol · Waiting for a qualified evaluation artifact."
    get("gbdt-note").textContent = "GBDT uses the AISim public performance-model API. A and B freeze the offline fit. C, A+C, and B+C enable structural tree adaptation from admitted feedback; C starts with no offline data. The evidence panel records the fitted recipe and native diagnostics."
    get("hybrid-note").textContent = "Hybrid calibration uses the AISim public performance-model API with a frozen GBDT forest and adaptive correction. It applies to C, A+C, and B+C. C bootstraps only from previously scored observations admitted at five-second ticks; the same cadence applies before and after its first successful prediction."
    get("selector-note").textContent = "The causal selector uses the AISim public performance-model API to select between a stable hybrid-calibrated expert and a structurally adapting expert. C, A+C, and B+C predict with the prior choice, score, then expose only scheduled feedback. The current target can affect subsequent predictions only. Select a cell for its recorded expert and selector controls."
    for (const id of SCENARIOS) get("protocol-" + id).textContent = partition(id)
    get("cadence-note").textContent = cadence + " " + chronology
    get("update-cost-note").textContent = "Online update cost is mean host wall time per attempted public update call. Every call receives one scheduled observation, including calls used for cold-start fitting. Scored observations not selected for updating do not contribute update cost."
    document.querySelectorAll('input[name="scenario"]').forEach(input => { input.checked = input.value === scenario })
    const metric = get("metric").value
    get("matrix-heading").textContent = scenarioTitle(scenario) + " · " + metricLabels[metric]
    get("metric-note").textContent = metric === "mape" ? "Lower is better · Successful predictions only; read with coverage · Green < 10%, amber 10–30%, red > 30%"
      : metric === "coverage" ? "Higher is better · Successful predictions / measured test observations · Cold starts and errors remain in the denominator"
      : "Lower is better · Host wall time in the evaluator" + (metric === "update" ? " per attempted update call" : "") + " · All workloads; the workload filter does not change cost"
  }

  function unsupported(method) {
    return ["warmup", "nowarmup"].includes(method)
      ? "Interpolation uses a self-benchmark library and is evaluated only in A."
      : labels[method] + " is compared in C, A+C, and B+C; frozen A-only and B-only do not apply."
  }
  function metric(result) {
    if (result?.status !== "evaluated") return null
    const metric = get("metric").value, value = result.metrics?.[get("workload").value]
    if (metric === "mape") return numeric(value?.mape_pct) ? value.mape_pct : null
    if (metric === "coverage") return numeric(value?.coverage_pct) ? value.coverage_pct
      : value?.measured_count > 0 && numeric(value.predicted_count) ? 100 * value.predicted_count / value.measured_count : null
    const timing = result.cost?.[{fit: "fit_ms", predict: "predict_us", update: "update_us"}[metric]]
    return numeric(timing) ? timing : null
  }

  function counts(result, workload = get("workload").value) {
    const value = result.metrics?.[workload]
    return value && numeric(value.predicted_count) && numeric(value.measured_count)
      ? number(value.predicted_count) + " / " + number(value.measured_count) : "Test counts unavailable"
  }

  function interpolationLabel(result) {
    return result?.interpolation?.method === "sol" ? "SOL interpolation"
      : result?.interpolation?.method === "direct" ? "Direct interpolation" : ""
  }


  function onlineModelLabel(result, scenarioId, method) {
    if (!["gbdt", "hybrid", "selector"].includes(method)) return ""
    const model = result?.[onlineScenarios.includes(scenarioId) ? "online_model" : "offline_model"]
    return model?.recipe?.label || ""
  }
  function cell(row, scenarioId, method) {
    const result = resultFor(row, scenarioId, method)
    if (result === null) {
      const reason = unsupported(method, scenarioId)
      return `<td class="sc-unsupported" data-state="unsupported" title="${escape(reason)}"><span class="sr-only">Not applicable: ${escape(reason)}</span></td>`
    }
    const status = result?.status || "missing_result"
    if (status !== "evaluated" && status !== "predictor_error") {
      const reason = [statuses[status] || "Result unavailable", result?.reason || "No evaluated result was published for this combination."].join(" · ")
      const title = [labels[method], scenarioTitle(scenarioId), reason, "Select to inspect evidence"].join(" · ")
      return `<td class="sc-unavailable-cell" data-state="${escape(status)}" title="${escape(reason)}"><button type="button" class="sc-score sc-empty-score" data-row="${escape(key(row))}" data-scenario="${scenarioId}" data-method="${method}" aria-label="${escape(title)}" title="${escape(reason)}"><span class="sr-only">${escape(reason)}</span></button></td>`
    }
    const value = metric(result), name = get("metric").value
    const unit = name === "mape" || name === "coverage" ? "%" : name === "fit" ? " ms" : " µs"
    const missing = statuses[status] || (status !== "evaluated" ? "Result unavailable"
      : name === "mape" ? "No predictions" : name === "coverage" ? "No test data" : "Not timed")
    const text = numeric(value) ? number(value) + unit : missing
    const count = result ? counts(result, ["mape", "coverage"].includes(name) ? get("workload").value : "all") : "Result not published"
    const note = status !== "evaluated" && numeric(value) ? missing + " · " + count : count
    const interpolation = ["warmup", "nowarmup"].includes(method) ? interpolationLabel(result) : onlineModelLabel(result, scenarioId, method)
    const title = [labels[method], interpolation, scenarioTitle(scenarioId), text, note + " test observations", result?.reason].filter(Boolean).join(" · ")
    const tone = status === "predictor_error" ? "bad" : name === "mape" && numeric(value) ? value < 10 ? "good" : value <= 30 ? "warn" : "bad" : "neutral"
    return `<td data-state="${escape(status)}"><button type="button" class="sc-score" data-row="${escape(key(row))}" data-scenario="${scenarioId}" data-method="${method}" data-tone="${tone}" title="${escape(title)}" aria-label="${escape(title)}">${numeric(value)
      ? `<strong>${escape(text)}</strong><small>${escape(note)}</small>${status === "evaluated" && interpolation ? `<small class="sc-interpolation">${escape(interpolation)}</small>` : ""}` : `<span class="sc-unavailable">${escape(text)}</span>`}</button></td>`
  }

  function filtered() {
    return (data?.rows || []).filter(row => ["model", "gpu", "framework"].every(field => !get(field).value || row[field] === get(field).value))
  }

  function render() {
    explain()
    if (!data) return
    const matching = filtered()
    const hasResult = row => methods().some(method => ["evaluated", "predictor_error"].includes(resultFor(row, scenario, method)?.status))
    let rows = get("show-unavailable").checked ? matching : matching.filter(hasResult)
    const hidden = matching.length - rows.length
    if (sort.method) rows = rows.slice().sort((a, b) => {
      const first = metric(resultFor(a, scenario, sort.method)), second = metric(resultFor(b, scenario, sort.method))
      return first === null ? second === null ? 0 : 1 : second === null ? -1 : (first - second) * sort.direction
    })
    if (!rows.some(row => key(row) === selectedKey)) selectedKey = rows.length ? key(rows[0]) : null
    get("count").textContent = number(rows.length) + " of " + number(data.rows.length) + " configurations" + (hidden ? " · " + number(hidden) + " unavailable hidden" : "")
    get("body").innerHTML = rows.length ? rows.map(row => `<tr data-selected="${key(row) === selectedKey}"><th scope="row"><button type="button" class="sc-config" data-select="${escape(key(row))}" aria-pressed="${key(row) === selectedKey}">${escape(row.model)}<small>${escape(row.parallelism || row.configuration_id)}${row.snapshot_id ? " · " + escape(row.snapshot_id) : ""}</small></button></th><td class="sc-identity">${escape(row.gpu)}<small>${escape(row.framework)}</small></td>${methods().map(method => cell(row, scenario, method)).join("")}</tr>`).join("")
      : `<tr><td colspan="${methods().length + 2}" class="sc-empty">${matching.length ? "No evaluated results for this scenario and these filters. Show unavailable configurations to inspect the reasons." : "No configurations match these filters."}</td></tr>`
    get("table").querySelectorAll("th[data-method]").forEach(th => th.setAttribute("aria-sort", th.dataset.method === sort.method ? sort.direction === 1 ? "ascending" : "descending" : "none"))
    renderDetail()
    updateURL()
  }

  function renderDetail() {
    const row = data?.rows.find(item => key(item) === selectedKey)
    get("detail").hidden = !row
    if (!row) return
    get("detail-heading").textContent = row.model
    const ordering = {chronological: "Recorded chronological order", intentional_sweep: "Intentional sweep order", file_order_fallback: "File order; chronology unavailable"}[row.ordering] || "Ordering evidence unavailable"
    get("detail-description").textContent = [row.gpu, row.framework, row.parallelism, ordering].filter(Boolean).join(" · ")
    const url = new URL("evaluation-detail.html", location.href)
    url.searchParams.set("branch", data.snapshot.branch)
    url.searchParams.set("configuration", row.configuration_id)
    if (row.snapshot_id) url.searchParams.set("snapshot", row.snapshot_id)
    url.searchParams.set("run", data.snapshot.run_id + "-" + data.snapshot.run_attempt)
    get("dataset-link").href = url.href
    get("detail-body").innerHTML = SCENARIOS.map(id => `<tr><th scope="row">${scenarios[id].label}<small>${usesSplit(id) ? splitDescription(id) : scenarios[id].short}</small></th>${methods().map(method => cell(row, id, method)).join("")}</tr>`).join("")
    if (evidence && evidence.row === selectedKey) renderEvidence(row)
    else get("evidence").hidden = true
  }


  function renderEvidence(row) {
    const result = resultFor(row, evidence.scenario, evidence.method)
    const blockedChronology = result?.status === "chronology_unavailable"
    const online = onlineScenarios.includes(evidence.scenario)
    get("evidence").hidden = false
    get("evidence-heading").textContent = scenarioTitle(evidence.scenario) + " · " + labels[evidence.method]
    get("evidence-note").textContent = result?.reason || (result?.status === "evaluated" ? "Metrics use the selected workload; selection counts and costs cover all workloads." : "No result was published for this cell.")
    get("evidence-protocol").textContent = partition(evidence.scenario) + (online ? " " + cadence + " " + chronology : "")
    get("evidence-count-note").textContent = blockedChronology
      ? "No fitting, prediction, or update ran. Recorded training and test counts describe nominal selections for coverage accounting."
      : evidence.scenario === "c" ? "C has zero offline training inputs. Bootstrap data comes only from scored observations admitted at five-second ticks. Cold-start fitting contributes to update cost. Coverage includes all scored observations and cold misses; MAPE includes successful predictions only."
      : "Preparation counts describe selected inputs, not successfully fitted samples. Self-benchmark regression inputs repeat artifact rows for each target worker; interpolation counts artifact rows. A successful update call can be a native no-op."
    const metrics = result?.metrics?.[get("workload").value]
    const values = [
      ["Status", result?.status === "evaluated" ? "Evaluated" : statuses[result?.status] || "Unavailable"],
      [blockedChronology ? "Nominal training selection · not executed" : "Offline preparation inputs selected", number(result?.training_count)],
      [blockedChronology ? "Nominal test selection · not executed" : "Test observations selected · all workloads", number(result?.test_count)],
      ["Successful / measured · selected workload", result ? counts(result) : "Unavailable"],
      ["Prediction errors", number(metrics?.error_count)], ["Unavailable predictions", number(metrics?.unavailable_count)],
      ["Observations in failed update calls · selected workload", number(metrics?.tuning_error_count)],
      ["Offline preparation · all workloads", numeric(result?.cost?.fit_ms) ? number(result.cost.fit_ms) + " ms" : "Not timed"],
      ["Mean prediction · all workloads", numeric(result?.cost?.predict_us) ? number(result.cost.predict_us) + " µs" : "Not timed"],
      ["Mean update call · all workloads", numeric(result?.cost?.update_us) ? number(result.cost.update_us) + " µs" : "Not timed"],
      ["Cold-start misses before first prediction", number(result?.cold_start?.unavailable_before_first_prediction)],
    ]
    if (usesSplit(evidence.scenario)) values.unshift(["Requested offline training share", trainPercent + "%"], [evidence.scenario === "b" ? "Requested frozen testing share" : "Requested online C share", (100 - trainPercent) + "%"])
    if (online) {
      const schedule = result?.online_schedule
      values.push(
        ["Update interval · event time", numeric(schedule?.interval_ms) ? number(schedule.interval_ms) + " ms" : "Not recorded"],
        ["Update time basis", schedule?.time_basis === "event_time_ms" ? "Recorded event timestamps (ms)" : "Not recorded"],
        ["Cold-start warmup policy", schedule?.warmup_policy === "scheduled_only" ? "Same five-second schedule from cold start" : schedule?.warmup_policy === "none" ? "No cold-start warmup" : "Not recorded"],
        ["Scored event-time span", numeric(schedule?.elapsed_ms) ? number(schedule.elapsed_ms) + " ms" : "Unavailable"],
        ["Scheduled single-observation calls · including warmup", number(schedule?.scheduled_update_count)],
        ["Immediate warmup calls", number(schedule?.warmup_update_count)],
        ["Attempted update calls · all workloads", number(result?.cost?.update_count)],
        ["Successful update calls", number(result?.online_update_count)],
        ["Failed update calls · all workloads", number(schedule?.update_error_count)],
        ["Observations submitted across all calls", number(schedule?.submitted_observation_count)],
        ["Scored observations not selected for updates", number(schedule?.skipped_observation_count)],
        ["Update selection policy", schedule?.selection_policy === "first_observation_at_or_after_tick" ? "First observation at or after each tick; predict and score before updating" : "Not recorded"],
      )
    }
    if (["warmup", "nowarmup"].includes(evidence.method)) {
      const configuration = result?.interpolation
      const flag = value => typeof value === "boolean" ? value ? "Yes" : "No" : "Not recorded"
      values.push(["Interpolation method", interpolationLabel(result) || "Not recorded"], ["Timing-only input", flag(configuration?.timing_only)],
        ["Text-only mode", flag(configuration?.text_only)], ["Verified legacy metadata recovery", flag(configuration?.metadata_recovered)])
    }
    const model = result?.[online ? "online_model" : "offline_model"]
    if (model) {
      const recipe = model.recipe
      values.push(["Model recipe", recipe?.label || recipe?.id || "Not recorded"], ["Recipe ID", recipe?.id || "Not recorded"])
      for (const [label, value] of [["Initial training controls", recipe?.training], ["Online tree controls", recipe?.online], ["Calibration controls", recipe?.correction], ["Selector controls", recipe?.selector]]) {
        if (value) values.push([label, JSON.stringify(value)])
      }
    }
    const artifact = result?.artifact
    if (artifact) {
      values.push(["Input artifact ID", typeof artifact === "string" ? artifact : artifact.id || artifact.artifact_id || "Unavailable"])
      if (object(artifact)) values.push(["Input artifact path", artifact.path || "Unavailable"], ["Input artifact SHA-256", artifact.sha256 || "Unavailable"])
    } else values.push(["Self-benchmark artifact", ["a", "ac"].includes(evidence.scenario) ? "Unavailable" : "Not used by this scenario"])
    if (result?.training_population_sha256) values.push([blockedChronology ? "Nominal training selection SHA-256" : "Training selection SHA-256", result.training_population_sha256])
    if (result?.scoring_population_sha256) values.push([blockedChronology ? "Nominal test selection SHA-256" : "Test selection SHA-256", result.scoring_population_sha256])
    get("evidence-values").innerHTML = values.map(([label, value]) => `<div><dt>${escape(label)}</dt><dd>${escape(value)}</dd></div>`).join("")
    get("evidence").querySelector(".sc-native-evidence")?.remove()
    if (model) {
      const details = document.createElement("details")
      details.className = "sc-native-evidence"
      const summary = document.createElement("summary")
      summary.textContent = "Recorded model recipe and native diagnostics"
      const pre = document.createElement("pre")
      pre.textContent = JSON.stringify(model, null, 2)
      details.append(summary, pre)
      get("evidence").append(details)
    }
  }
  function filterOptions() {
    const current = new URLSearchParams(location.search)
    for (const field of ["model", "gpu", "framework"]) {
      const select = get(field), previous = current.get(field) || select.value
      const values = [...new Set(data.rows.map(row => row[field]).filter(Boolean))].sort()
      select.replaceChildren(new Option({model: "All models", gpu: "All GPUs", framework: "All frameworks"}[field], ""), ...values.map(value => new Option(value, value)))
      select.value = values.includes(previous) ? previous : ""
    }
  }

  function clear(message, state = "empty") {
    data = null
    explain()
    root.dataset.ready = state
    get("status").textContent = message
    get("status").dataset.state = state
    get("body").innerHTML = `<tr><td colspan="${methods().length + 2}" class="sc-empty">${escape(message)}</td></tr>`
    get("count").textContent = ""
    get("detail").hidden = true
    window.fpmNavigation.setSnapshot(null, state === "loading" ? "Loading evaluation…" : "No scenario evaluation loaded")
  }

  async function load(url, optional = false) {
    const response = await fetch(url, {cache: "no-cache"})
    if (response.status === 404 && optional) return null
    if (!response.ok) throw new Error("HTTP " + response.status)
    return response.json()
  }


  function validate(result, branch, summary) {
    const ids = entries => Array.isArray(entries) ? entries.map(entry => typeof entry === "string" ? entry : entry?.id) : null
    if (!object(result) || result.schema_version !== 13 || result.protocol_id !== "fpm-scenarios-v13") {
      throw new Error("This page requires the strict five-second protocol (v13); earlier warmup results are not comparable")
    }
    if (JSON.stringify(ids(result.methods)) !== JSON.stringify(METHODS) || JSON.stringify(ids(result.scenarios)) !== JSON.stringify(SCENARIOS)
      || !Array.isArray(result.rows)) throw new Error("Invalid scenario artifact")
    const snapshot = result.snapshot
    if (snapshot?.branch !== branch || !/^[1-9][0-9]*$/.test(snapshot.run_id) || !/^[1-9][0-9]*$/.test(snapshot.run_attempt)
      || !/^[a-f0-9]{40}$/.test(snapshot.commit_sha) || !/^[a-f0-9]{40}$/.test(snapshot.hf_revision) || !Number.isFinite(Date.parse(snapshot.completed_at))) throw new Error("Invalid scenario snapshot")
    if (["branch", "run_id", "run_attempt", "commit_sha", "hf_revision", "completed_at"].some(name => snapshot[name] !== summary?.snapshot?.[name])) {
      throw new Error("Scenario data does not match this evaluation snapshot")
    }
    const seen = new Set()
    for (const row of result.rows) {
      if (!object(row) || typeof row.configuration_id !== "string" || !row.configuration_id || !object(row.results) || seen.has(key(row))) throw new Error("Invalid scenario configuration")
      seen.add(key(row))
      for (const [id, collection] of [...Object.entries(row.results), ...Object.values(row.split_results || {}).flatMap(grid => Object.entries(grid || {}))]) {
        if (!SCENARIOS.includes(id) || !object(collection)) throw new Error("Invalid scenario results")
        for (const [method, value] of Object.entries(collection)) {
          if (!METHODS.includes(method) || (value !== null && !object(value))) throw new Error("Invalid method result")
          if (!value) continue
          if (id === "c" && value.training_count !== 0) throw new Error("Strict C cannot contain offline training data")
          if (onlineScenarios.includes(id)) {
            const schedule = value.online_schedule
            if (!schedule || schedule.interval_ms !== 5000 || schedule.time_basis !== "event_time_ms"
              || schedule.selection_policy !== "first_observation_at_or_after_tick" || schedule.warmup_policy !== (id === "c" ? "scheduled_only" : "none")
              || schedule.warmup_update_count !== 0) throw new Error("Scenario result does not use the strict five-second admission schedule")
          }
        }
      }
    }
  }
  async function select(branch) {
    const current = ++request
    clear("Loading scenario results…", "loading")
    const entry = catalog.branches.find(item => item.branch === branch)
    if (entry?.status !== "available") { clear("No completed scenario evaluation for this branch."); return }
    try {
      if (!/^branches\/[0-9a-f]{16}\/summary\.json$/.test(entry.summary_path)) throw new Error("Invalid scenario data path")
      const [result, summary] = await Promise.all([
        load(entry.summary_path.replace(/summary\.json$/, "scenarios.json"), true), load(entry.summary_path),
      ])
      if (current !== request) return
      if (!result) { clear("Scenario comparison has not been published for this evaluation yet. Existing Overview and Details results remain available."); return }
      validate(result, branch, summary)
      data = result
      splitIssues = new Map()
      if (splitGrid()) for (const row of data.rows) {
        const issue = checkSplitGrid(row)
        if (issue) splitIssues.set(key(row), issue)
      }
      else trainPercent = 50
      const currentParams = new URLSearchParams(location.search)
      const selected = data.rows.find(row => row.configuration_id === currentParams.get("configuration") && (!currentParams.get("snapshot") || row.snapshot_id === currentParams.get("snapshot")))
      selectedKey = selected ? key(selected) : null
      evidence = null
      filterOptions()
      get("status").textContent = data.rows.length ? "" : "This evaluation contains no scenario configurations."
      get("status").dataset.state = "ready"
      root.dataset.ready = "true"
      window.fpmNavigation.setSnapshot(data.snapshot)
      render()
    } catch (error) { if (current === request) clear("Scenario comparison unavailable: " + error.message + ". Reload to retry.", "error") }
  }

  document.querySelectorAll('input[name="scenario"]').forEach(input => input.addEventListener("change", () => {
    scenario = input.value; sort = {method: null, direction: 1}; explain(); render(); updateURL()
  }))
  for (const field of ["model", "gpu", "framework", "workload", "metric"]) {
    const select = get(field)
    if (["workload", "metric"].includes(field) && [...select.options].some(option => option.value === params.get(field))) select.value = params.get(field)
    select.addEventListener("change", () => { render(); updateURL() })
  }
  get("show-unavailable").checked = params.get("show_unavailable") === "1"
  get("show-unavailable").addEventListener("change", () => { render(); updateURL() })
  get("train-percent").addEventListener("input", () => {
    const value = Number(get("train-percent").value)
    if (!splitGrid() || !TRAIN_PERCENTAGES.includes(value)) return
    trainPercent = value
    render()
  })
  get("reset").addEventListener("click", () => {
    for (const field of ["model", "gpu", "framework"]) get(field).value = ""
    get("show-unavailable").checked = false
    trainPercent = 50
    get("workload").value = "all"; get("metric").value = "mape"; sort = {method: null, direction: 1}; render(); updateURL()
  })
  root.addEventListener("click", event => {
    const select = event.target.closest("[data-select]")
    if (select && data) {
      selectedKey = select.dataset.select; evidence = null; render();
      [...root.querySelectorAll("[data-select]")].find(button => button.dataset.select === selectedKey)?.focus({preventScroll: true})
    }
    const score = event.target.closest(".sc-score")
    if (score && data) {
      selectedKey = score.dataset.row
      evidence = {row: selectedKey, scenario: score.dataset.scenario, method: score.dataset.method}
      render()
      get("evidence-heading").focus({preventScroll: true})
      get("evidence").scrollIntoView({behavior: "smooth", block: "nearest"})
    }
    const button = event.target.closest("[data-sort]")
    if (button && data) { sort = {method: button.dataset.sort, direction: sort.method === button.dataset.sort ? -sort.direction : get("metric").value === "coverage" ? -1 : 1}; render() }
  })
  branchSelect.addEventListener("change", () => { updateURL(); select(branchSelect.value) })
  explain()
  load("branches.json").then(result => {
    if (result.schema_version !== 1 || !Array.isArray(result.branches)) throw new Error("Invalid branch catalog")
    catalog = result
    branchSelect.replaceChildren(...catalog.branches.map(entry => new Option(entry.branch, entry.branch)))
    const branch = params.get("branch") || catalog.default_branch
    if (!catalog.branches.some(entry => entry.branch === branch)) branchSelect.add(new Option(branch + " (unavailable)", branch))
    branchSelect.value = branch
    branchSelect.disabled = false
    return select(branch)
  }).catch(error => clear("Scenario comparison unavailable: " + error.message + ". Reload to retry.", "error"))
})()
