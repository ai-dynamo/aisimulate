/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
window.AISimReplay = { mount(root, report, options = {}) {

      const state = {
        view: options.view || "dashboard",
        stage: options.stage || (report.phases?.profile ? "profile" : "preparation"),
        group: "conversation",
        search: "",
        modes: {
          input: "distribution",
          output: "distribution",
          latency: "ttft_ms",
          cache: "aggregate",
          queue: "queue",
          throughput: "executed",
          footprint: "average",
        },
        percentiles: { interactivity: 90, latency: 90 },
      };
      const $ = (id) => root.querySelector("#" + id);
      const comparedPhase = () => report.comparison?.phases?.[state.stage];
      if (options.accuracy !== undefined) {
        $("stage-profile").textContent = "Profile";
        $("stage-preparation").textContent = "Preparation";
        $("view-dashboard").insertAdjacentHTML("afterbegin", `<section id="profile-accuracy" aria-label="Profile accuracy">${options.accuracy}</section>`);
        root.querySelector(".main-tabs").insertAdjacentHTML("beforeend", '<button role="tab" id="tab-evidence" aria-controls="view-evidence" data-view="evidence" aria-selected="false">Evidence</button>');
        root.querySelector("main").insertAdjacentHTML("beforeend", `<section id="view-evidence" role="tabpanel" aria-labelledby="tab-evidence" hidden>${options.evidence || ""}</section>`);
      }
      const finite = (value) =>
        typeof value === "number" && Number.isFinite(value);
      const seconds = (value) => (finite(value) ? value / 1000 : null);
      const fmt = (value, digits = 2) =>
        finite(value)
          ? value.toLocaleString("en-US", { maximumFractionDigits: digits })
          : "N/A";
      const esc = (value) =>
        String(value ?? "").replace(
          /[&<>"']/g,
          (char) =>
            ({
              "&": "&amp;",
              "<": "&lt;",
              ">": "&gt;",
              '"': "&quot;",
              "'": "&#39;",
            })[char],
        );
      const color = (key) =>
        getComputedStyle(root)
          .getPropertyValue("--" + key)
          .trim();
      const sourceColor = (origin) => color(origin === "silicon" ? "silicon" : "aisim");
      const phase = () => report.phases?.[state.stage] || {};
      const origin = () =>
        finite(phase().start_ms) ? phase().start_ms / 1000 : 0;
      const rows = () =>
        report.requests.filter((row) =>
          state.stage === "profile"
            ? row.phase === "profile"
            : ["primer", "warmup", "preparation", "aborted"].includes(
                row.phase,
              ),
        );
      const samples = () =>
        report.samples.filter((sample) => sample.phase === state.stage);
      const conversation = (row) =>
        row.agentic?.conversation_id || row.session_id || null;
      const conversationKey = (row) =>
        conversation(row)
          ? JSON.stringify([
              row.play_id || row.agentic?.play_id || null,
              conversation(row),
            ])
          : null;
      const arrivals = () =>
        samples().length && samples().every((sample) => finite(sample.arrivals))
          ? samples().reduce((total, sample) => total + sample.arrivals, 0)
          : null;
      const completeCohort = () =>
        phase().completion_status === "completed" &&
        !(finite(arrivals()) && arrivals() > rows().length) &&
        !(
          state.stage === "profile" &&
          finite(report.prediction.num_requests) &&
          report.prediction.num_requests > rows().length
        );
      const keyFor = (row) =>
        String(row.uuid ?? row.request_id ?? report.requests.indexOf(row));
      const requestById = new Map(
        report.requests.map((row) => [keyFor(row), row]),
      );
      const average = (values) => {
        const valid = values.filter(finite);
        return valid.length
          ? valid.reduce((a, b) => a + b, 0) / valid.length
          : null;
      };
      function quantile(values, percentile) {
        const sorted = values.filter(finite).sort((a, b) => a - b);
        return sortedQuantile(sorted, percentile);
      }
      function sortedQuantile(sorted, percentile) {
        if (!sorted.length) return null;
        const position = ((sorted.length - 1) * percentile) / 100,
          low = Math.floor(position);
        return (
          sorted[low] +
          (sorted[Math.ceil(position)] - sorted[low]) * (position - low)
        );
      }
      function cumulativeQuantiles(values, percentile) {
        const sorted = [];
        return values.map((value) => {
          let low = 0,
            high = sorted.length;
          while (low < high) {
            const middle = (low + high) >>> 1;
            if (sorted[middle] < value) low = middle + 1;
            else high = middle;
          }
          sorted.splice(low, 0, value);
          return sortedQuantile(sorted, percentile);
        });
      }
      function timeWindow(ss, index) {
        const end = ss[index].t,
          result = [];
        for (let current = index; current >= 0; current--) {
          const sample = ss[current];
          if (sample.t <= end - 50) break;
          const duration =
            Math.min(end, sample.t) - Math.max(end - 50, sample.start);
          if (duration > 0) result.push({ sample, duration });
        }
        return result;
      }
      function rolling(ss, values) {
        const observed = ss.map((sample, index) => ({
          ...sample,
          value: values[index],
        }));
        return observed.map((sample, index) => {
          if (!finite(sample.value)) return null;
          const window = timeWindow(observed, index);
          if (!window.length) return sample.value;
          if (window.some((item) => !finite(item.sample.value))) return null;
          return (
            window.reduce(
              (sum, item) => sum + item.sample.value * item.duration,
              0,
            ) / window.reduce((sum, item) => sum + item.duration, 0)
          );
        });
      }
      function cumulative(values) {
        let total = 0,
          valid = true;
        return values.map((value) => {
          if (!finite(value)) valid = false;
          else total += value;
          return valid ? total : null;
        });
      }
      const percent = (numerator, denominator) =>
        finite(numerator) && finite(denominator) && denominator > 0
          ? (100 * numerator) / denominator
          : null;
      const line = (name, x, y, shade, extra = {}) => ({
        type: "scatter",
        mode: "lines",
        name,
        x,
        y,
        connectgaps: false,
        line: { color: shade, width: 1.5 },
        hovertemplate: "%{x:.2f}s<br>%{y:,.3f}<extra>" + name + "</extra>",
        ...extra,
      });
      const plotConfig = {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        toImageButtonOptions: { format: "png", scale: 2 },
      };
      function layout(y = "Value", extra = {}) {
        return {
          paper_bgcolor: color("panel"),
          plot_bgcolor: color("panel"),
          showlegend: true,
          font: {
            family: "system-ui, sans-serif",
            size: 10,
            color: color("muted"),
          },
          margin: { l: 58, r: 15, t: 12, b: 66 },
          xaxis: {
            title: { text: "Time since phase start", font: { size: 10 } },
            gridcolor: color("grid"),
            zeroline: false,
            ticksuffix: "s",
            rangemode: "tozero",
          },
          yaxis: {
            title: { text: y, font: { size: 10 } },
            gridcolor: color("grid"),
            zeroline: false,
            rangemode: "tozero",
          },
          legend: { orientation: "h", x: 0, y: -0.28, font: { size: 9 } },
          hoverlabel: {
            bgcolor: color("raised"),
            font: { color: color("text"), size: 12 },
          },
          dragmode: "zoom",
          ...extra,
        };
      }
      const specs = [
        { id: "input", title: "Input sequence length", control: "shape" },
        { id: "output", title: "Output sequence length", control: "shape" },
        {
          id: "interactivity",
          title: "Interactivity over time",
          control: "percentile",
        },
        { id: "latency", title: "TTFT over time", control: "latency" },
        {
          id: "cache",
          title: "Resident KV cache utilization",
          control: "cache",
        },
        { id: "queue", title: "Request queue depth", control: "queue" },
        { id: "hits", title: "Prefix cache hit rate per interval" },
        { id: "throughput", title: "Token throughput", control: "throughput" },
        { id: "source", title: "Cumulative prompt token sources" },
        { id: "footprint", title: "Conversation input footprint proxy", control: "footprint" },
        { id: "acceptance", title: "Speculative acceptance length" },
        { id: "completion", title: "Request completion status" },
      ].filter(spec => options.accuracy === undefined || !["hits", "acceptance"].includes(spec.id));
      // Optional numerical references supplied by the qualified accuracy publisher.
      // Ordinary replay exports do not add comparison cards.
      if (report.reference?.pairs) specs.unshift(
        { id: "reference_ttft", title: "TTFT distribution · silicon / AISim" },
        { id: "reference_tpot", title: "TPOT distribution · silicon / AISim" },
      );
      function segments(id, options, selected, kind = "mode") {
        return (
          '<div class="segments">' +
          options
            .map(
              ([value, label, disabled = false]) =>
                `<button data-${kind}="${id}" data-value="${value}" aria-pressed="${String(selected) === String(value)}"${disabled ? " disabled" : ""}>${label}</button>`,
            )
            .join("") +
          "</div>"
        );
      }
      function controls(spec) {
        if (spec.control === "footprint") {
          const chart = comparedPhase()?.charts.find(c => c.id === "unique");
          if (!chart) return "";
          const enabled = canAverageFootprint(chart);
          return segments("footprint", [["average", "60s average", !enabled], ["raw", "Raw"]], enabled ? state.modes.footprint : "raw");
        }
        if (report.comparison && spec.control === "throughput")
          return `<select data-select-mode="throughput" aria-label="Throughput accounting">${[["executed", "Input and output"], ["completion", "Output at completion"], ["delivered", "Live output"]].map(([value,label])=>`<option value="${value}" ${state.modes.throughput===value?"selected":""}>${label}</option>`).join("")}</select>`;
        if (spec.control === "shape")
          return segments(
            spec.id,
            [
              ["distribution", "Distribution"],
              ["inflight", "In-flight avg"],
            ],
            state.modes[spec.id],
          );
        if (spec.control === "percentile")
          return segments(
            spec.id,
            [
              [75, "P75"],
              [90, "P90"],
            ],
            state.percentiles[spec.id],
            "percentile",
          );
        if (spec.control === "latency")
          return (
            segments(
              "latency",
              [
                ["ttft_ms", "TTFT"],
                ["e2e_latency_ms", "E2E"],
              ],
              state.modes.latency,
            ) +
            segments(
              "latency",
              [
                [75, "P75"],
                [90, "P90"],
              ],
              state.percentiles.latency,
              "percentile",
            )
          );
        if (spec.control === "cache")
          return (
            '<select data-select-mode="cache" aria-label="Cache pool">' +
            cacheOptions()
              .map(
                (option) =>
                  `<option value="${esc(option.key)}"${state.modes.cache === option.key ? " selected" : ""}>${esc(option.label)}</option>`,
              )
              .join("") +
            "</select>"
          );
        if (spec.control === "queue")
          return segments(
            "queue",
            [
              ["queue", "Queue"],
              ["completed", "Completed"],
              ["preemptions", "Preemptions"],
            ],
            state.modes.queue,
          );
        if (spec.control === "throughput")
          return (
            '<select id="throughput-accounting" data-select-mode="throughput" aria-label="Throughput accounting"><option value="executed"' +
            (state.modes.throughput === "executed" ? " selected" : "") +
            '>Executed / delivered</option><option value="completion"' +
            (state.modes.throughput === "completion" ? " selected" : "") +
            ">At completion</option></select>"
          );
        return "";
      }
      function activeRequestSeries(requests, observations) {
        const result = {
            input_length: Array(observations.length).fill(null),
            output_length: Array(observations.length).fill(null),
            footprint: Array(observations.length).fill(0),
          },
          events = [];
        for (const row of requests) {
          const start = row.arrival_time_ms,
            end = finite(row.terminal_time_ms)
              ? row.terminal_time_ms
              : row.terminal_status === "incomplete"
                ? Infinity
                : null;
          if (!finite(start) || end === null || end <= start) continue;
          const request = {
            end,
            input_length: row.input_length,
            output_length: row.output_length,
            group: conversationKey(row),
          };
          events.push({ time: start, delta: 1, request });
          if (finite(end)) events.push({ time: end, delta: -1, request });
        }
        events.sort((a, b) => a.time - b.time || a.delta - b.delta);
        const queries = observations
            .map((sample, index) => ({ time: sample.t * 1000, index }))
            .filter((query) => finite(query.time))
            .sort((a, b) => a.time - b.time),
          sums = { input_length: 0, output_length: 0 },
          counts = { input_length: 0, output_length: 0 },
          groups = new Map();
        let cursor = 0,
          footprint = 0,
          unknown = 0;
        for (const query of queries) {
          while (cursor < events.length && events[cursor].time <= query.time) {
            const { time, delta, request } = events[cursor++];
            for (const key of ["input_length", "output_length"])
              if (finite(request[key])) {
                sums[key] += delta * request[key];
                counts[key] += delta;
              }
            if (!request.group || !finite(request.input_length)) {
              unknown += delta;
              continue;
            }
            // A max-heap per conversation expires finished requests lazily.
            const heap = groups.get(request.group) || [],
              previous = Math.max(0, heap[0]?.input_length ?? 0);
            if (delta > 0) {
              let index = heap.length;
              heap.push(request);
              while (index > 0) {
                const parent = (index - 1) >>> 1;
                if (heap[parent].input_length >= request.input_length) break;
                heap[index] = heap[parent];
                index = parent;
              }
              heap[index] = request;
            }
            while (heap.length && heap[0].end <= time) {
              const last = heap.pop();
              if (!heap.length) break;
              let index = 0;
              while (2 * index + 1 < heap.length) {
                let child = 2 * index + 1;
                if (
                  child + 1 < heap.length &&
                  heap[child + 1].input_length > heap[child].input_length
                )
                  child++;
                if (last.input_length >= heap[child].input_length) break;
                heap[index] = heap[child];
                index = child;
              }
              heap[index] = last;
            }
            footprint += Math.max(0, heap[0]?.input_length ?? 0) - previous;
            if (heap.length) groups.set(request.group, heap);
            else groups.delete(request.group);
          }
          for (const key of ["input_length", "output_length"])
            result[key][query.index] = counts[key]
              ? sums[key] / counts[key]
              : null;
          result.footprint[query.index] = unknown ? null : footprint;
        }
        return result;
      }
      // Embedded telemetry is immutable, so phase changes and redraws reuse the sweep.
      const activeSeriesByPhase = new Map();
      function activeSeries() {
        if (!activeSeriesByPhase.has(state.stage))
          activeSeriesByPhase.set(
            state.stage,
            activeRequestSeries(rows(), samples()),
          );
        return activeSeriesByPhase.get(state.stage);
      }
      function completionOutputSamples(requests, observations) {
        const completed = [];
        for (const request of requests) {
          if (request.terminal_status !== "completed") continue;
          const time = request.terminal_time_ms;
          if (finite(time))
            completed.push({ time, output: request.output_length });
        }
        completed.sort((a, b) => a.time - b.time);
        const tokens = [0],
          missing = [0];
        for (const request of completed) {
          const valid = finite(request.output);
          tokens.push(tokens.at(-1) + (valid ? request.output : 0));
          missing.push(missing.at(-1) + (valid ? 0 : 1));
        }
        const upperBound = (time) => {
          let low = 0,
            high = completed.length;
          while (low < high) {
            const middle = (low + high) >>> 1;
            if (completed[middle].time <= time) low = middle + 1;
            else high = middle;
          }
          return low;
        };
        return observations.map((sample) => {
          // Independent bounds preserve gaps, overlaps and the native (start, end] interval.
          const start = upperBound(sample.start * 1000),
            end = upperBound(sample.t * 1000);
          return {
            ...sample,
            output_tokens:
              missing[end] !== missing[start]
                ? null
                : tokens[end] - tokens[start],
          };
        });
      }
      const completionSeriesByPhase = new Map();
      function completionSeries() {
        if (!completionSeriesByPhase.has(state.stage))
          completionSeriesByPhase.set(
            state.stage,
            completionOutputSamples(rows(), samples()),
          );
        return completionSeriesByPhase.get(state.stage);
      }
      function throughput(samples, key) {
        return samples.map((sample, index) => {
          if (!finite(sample[key])) return null;
          const window = timeWindow(samples, index);
          if (window.some((item) => !finite(item.sample[key]))) return null;
          const duration = window.reduce((sum, item) => sum + item.duration, 0);
          let instantaneous = 0;
          for (
            let current = index;
            current >= 0 && samples[current].t > sample.t - 50;
            current--
          )
            if (
              samples[current].duration === 0 &&
              finite(samples[current][key])
            )
              instantaneous += samples[current][key];
          return duration > 0
            ? (instantaneous +
                window.reduce(
                  (sum, item) =>
                    sum +
                    (item.sample[key] * item.duration) / item.sample.duration,
                  0,
                )) /
                duration
            : null;
        });
      }
      const cacheIdentity = (pool) =>
        JSON.stringify([pool.role, pool.scope, pool.worker_id, pool.dp_rank]);
      function cacheOptions() {
        const options = new Map([
          ["aggregate", { key: "aggregate", label: "All device caches" }],
        ]);
        for (const sample of samples())
          for (const [kind, pools] of [
            ["worker", sample.workers || []],
            ["host", sample.host_pools || []],
          ])
            for (const pool of pools) {
              const identity = cacheIdentity(pool),
                key = kind + ":" + identity;
              const label =
                (kind === "host" ? "Host · " : "") +
                (pool.role || "deployment shared") +
                (pool.scope ? " · " + pool.scope : "") +
                (pool.worker_id != null ? " · worker " + pool.worker_id : "") +
                (pool.dp_rank != null ? " · DP " + pool.dp_rank : "");
              options.set(key, { key, kind, identity, label });
            }
        return [...options.values()];
      }
      function comparisonKey(id) {
        if (id === "footprint") return "unique";
        if (id === "input" || id === "output") return state.modes[id] === "inflight" ? id + "_inflight" : null;
        if (id === "interactivity") return `interactivity_p${state.percentiles.interactivity}`;
        if (id === "latency") return `latency_${state.modes.latency}_p${state.percentiles.latency}`;
        if (id === "queue") return state.modes.queue;
        if (id === "cache" && state.modes.cache !== "aggregate") return null;
        if (id === "throughput") return "throughput_" + ({executed:"both",completion:"decode",delivered:"delivered"}[state.modes.throughput]);
        return id;
      }
      function scoresFor(id) {
        const chart = comparedPhase()?.charts.find(c=>c.id === comparisonKey(id));
        return chart?.scores || [{label:"Comparison",mape_percent:null,count:0,reason:state.stage !== "profile" ? "Preparation is unscored" : "No compatible silicon comparison for this view"}];
      }
      function canAverageFootprint(chart) {
        return finite(chart.score_window_end_s) && chart.score_window_end_s - chart.score_window_start_s >= 60 &&
          ["silicon", "simulated"].every(origin => chart.series.some(s => s.origin === origin));
      }
      function footprintChart(chart) {
        const sources = ["silicon", "simulated"].map(origin => chart.series.find(s => s.origin === origin));
        const maps = sources.map(s => new Map(s ? s.x.map((t, i) => [t, s.y[i]]) : []));
        const allTimes = [...new Set(sources.flatMap(s => s?.x || []))].sort((a, b) => a - b);
        // An absent timestamp is a gap, not a line across unrecorded time.
        const times = allTimes.flatMap((t, i) => i && t > allTimes[i - 1] + 1 ? [allTimes[i - 1] + 1, t] : [t]);
        const averaged = canAverageFootprint(chart) && state.modes.footprint !== "raw";
        const minutes = (allTimes.at(-1) || 0) >= 120, scale = minutes ? 60 : 1;
        const start = chart.score_window_start_s, end = chart.score_window_end_s;
        const x = [], values = [[], []], difference = [];
        if (averaged) {
          for (let left = Math.floor(start); left < Math.floor(end); left += 60) {
            const right = Math.min(left + 60, Math.floor(end)), sums = [0, 0];
            let complete = true;
            for (let t = left + 1; t <= right; t++) {
              const pair = maps.map(m => m.get(t));
              if (!pair.every(finite)) { complete = false; break; }
              pair.forEach((v, i) => sums[i] += v);
            }
            x.push((left + right) / 2 / scale);
            values.forEach((v, i) => v.push(complete ? sums[i] / (right - left) / 1000 : null));
            difference.push(complete ? values[1].at(-1) - values[0].at(-1) : null);
          }
        } else {
          for (const t of times) {
            const pair = maps.map(m => m.get(t));
            x.push(t / scale);
            values.forEach((v, i) => v.push(finite(pair[i]) ? pair[i] / 1000 : null));
            difference.push(finite(end) && t >= start && t <= end && pair.every(finite) ? (pair[1] - pair[0]) / 1000 : null);
          }
        }
        const unit = minutes ? "min" : "s";
        const traces = sources.flatMap((s, i) => s ? [{
          ...line(i === 0 ? "Silicon" : "AISim", x, values[i], sourceColor(s.origin)),
          mode: values[i].filter(finite).length === 1 ? "lines+markers" : "lines",
          line: {color: sourceColor(s.origin), width: 2.5, dash: "solid", shape: averaged ? "linear" : "hv"},
          hovertemplate: `%{x:.2f} ${unit}<br>%{y:,.2f}k tokens<extra>${i === 0 ? "Silicon" : "AISim"}</extra>`,
        }] : []);
        traces.push({
          ...line("AISim − Silicon", x, difference, color("muted")),
          xaxis: "x2", yaxis: "y2", showlegend: false,
          mode: difference.filter(finite).length === 1 ? "lines+markers" : "lines",
          line: {color: color("muted"), width: 1.7, dash: "solid", shape: averaged ? "linear" : "hv"},
          fill: "tozeroy", fillcolor: "rgba(148,163,184,0.15)",
          hovertemplate: `%{x:.2f} ${unit}<br>AISim − Silicon: %{y:+,.2f}k tokens<extra></extra>`,
        });
        const base = layout();
        return {
          traces,
          axes: layout("", {
            margin: {l: 70, r: 15, t: 34, b: 42},
            legend: {orientation: "h", x: 1, xanchor: "right", y: 1.15, font: {size: 11}},
            xaxis: {...base.xaxis, title: {text: ""}, showticklabels: false, matches: "x2", anchor: "y"},
            xaxis2: {...base.xaxis, title: {text: `${minutes ? "Minutes" : "Seconds"} since phase start`}, ticksuffix: "", anchor: "y2", range: [0, Math.max(1, allTimes.at(-1) || 1) / scale]},
            yaxis: {...base.yaxis, title: {text: "Footprint (k tokens)"}, domain: [0.42, 1]},
            yaxis2: {...base.yaxis, title: {text: "AISim − Silicon<br>(k tokens)"}, domain: [0, 0.26], zeroline: true, zerolinecolor: color("muted"), zerolinewidth: 1},
            annotations: difference.some(finite) ? [] : [{xref: "paper", yref: "paper", x: 0.5, y: 0.13, text: "No common recorded observations", showarrow: false}],
          }),
          note: averaged ? "60-second averages · common recorded coverage" : "Raw observations" + (canAverageFootprint(chart) ? "" : " · 60s averaging requires at least one minute of common coverage"),
          foot: "Positive differences mean AISim is higher; negative differences mean it is lower. " +
            (averaged ? "Averages include zeros; incomplete bins remain gaps, and the final bin may be shorter. " : "Missing observations remain gaps. ") +
            "Curve MAPE uses the original 1-second observations, independent of display mode and zoom. Request-lifetime proxy; terminal-only captures may omit unfinished work.",
          empty: values.some(v => v.some(finite)) ? null : "No footprint observations in this phase.",
        };
      }
      function scoreMarkup(id) {
        if (!report.comparison && !options.accuracy) return "";
        const metric = id === "latency" ? (state.modes.latency === "ttft_ms" ? "ttft_ms" : "e2e_ms") : id === "interactivity" ? "tpot_ms" : id === "reference_ttft" ? "ttft_ms" : id === "reference_tpot" ? "tpot_ms" : null;
        const request = comparedPhase()?.request_scores[metric];
        return scoresFor(id).map(s=>`<span class="score-badge" title="${esc(s.reason || "Full selected profile; zoom does not change the score")}">${id === "footprint" && comparedPhase() ? "Original 1s" : esc(s.label)} · Curve MAPE ${fmt(s.mape_percent)}${finite(s.mape_percent)?"%":""} · n=${fmt(s.count,0)}</span>`).join("") + (request ? `<span class="score-badge">${metric === "tpot_ms" ? "TPOT" : metric === "ttft_ms" ? "TTFT" : "E2E"} · Request MAPE ${fmt(request.mape)}${finite(request.mape)?"%":""} · n=${fmt(request.count,0)}</span>` : "");
      }
      function comparisonChart(id) {
        const c = comparedPhase()?.charts.find(c=>c.id === comparisonKey(id));
        if (!c) return null;
        if (id === "footprint") return footprintChart(c);
        const hasSilicon = c.series.some(s=>s.origin === "silicon" && s.y.some(finite));
        const traces = c.series.map(s=>({name:`${s.label} · ${s.origin === "silicon" ? "Silicon" : "AISim"}`, x:s.x,y:s.y,
          type:s.kind === "bar" ? "bar" : "scatter",mode:"lines",connectgaps:false,
          line:{color:sourceColor(s.origin),dash:id === "source" && s.metric === "miss" ? "dot" : "solid",shape:s.step?"hv":"linear"},
          marker:{color:sourceColor(s.origin)},
          hovertemplate:`%{x}<br>%{y:,.3f} ${esc(c.unit)}<extra>%{fullData.name}</extra>`}));
        return {traces,axes:layout(c.unit,{barmode:"group"}),note:(hasSilicon ? "" : "Silicon reference unavailable for this metric in the selected phase. ") + c.note,
          foot:c.foot + (hasSilicon ? " Silicon: blue. AISim: yellow." : " AISim prediction only.") + (id === "source" ? " Cache hits: solid. Cache misses: dotted." : "") + " Phase origins are recorded independently; no clock fitting.",
          empty:traces.some(t=>t.y.some(finite))?null:"No comparable observations in this phase."};
      }
      function chart(id) {
        const compared = comparisonChart(id);
        if (compared) return compared;
        const r = rows(),
          ss = samples(),
          x = ss.map((sample) => sample.t - origin());
        const blue = color("blue"),
          orange = color("accent"),
          red = color("red"),
          green = color("green"),
          purple = color("purple");
        let traces = [],
          note = "",
          foot = "",
          axes = layout(),
          empty = null;
        if (id === "reference_ttft" || id === "reference_tpot") {
          const metric = id === "reference_ttft" ? "ttft_ms" : "tpot_ms";
          const pairs = state.stage === "profile" ? report.reference.pairs.filter(p => finite(p[metric + "_ape"])) : [];
          const values = pairs.flatMap(p => [p["silicon_" + metric], p["predicted_" + metric]]);
          const maximum = values.reduce((a, b) => Math.max(a, b), 0);
          const width = (maximum || 1) / 32;
          if (state.stage !== "profile" || !pairs.length) empty = "No scored silicon comparisons in this phase.";
          traces = [["silicon", "Silicon", sourceColor("silicon")], ["predicted", "AISim", sourceColor("simulated")]].map(([key, name, color]) => {
            const counts = Array(32).fill(0);
            pairs.forEach(p => counts[Math.min(31, Math.floor(p[key + "_" + metric] / width))]++);
            return {type: "bar", name, x: counts.map((_, i) => (i + 0.5) * width), y: counts,
              marker: {color}, hovertemplate: "%{x:.2f} ms · %{y} requests<extra>%{fullData.name}</extra>"};
          });
          axes = layout("Requests", {xaxis: {title: metric === "ttft_ms" ? "TTFT (ms)" : "TPOT (ms/token)"}, barmode: "group"});
          note = "Successful, uniquely matched requests with equal recorded output lengths";
          foot = `${pairs.length} scored pairs; only valid equal-output-length pairs are shown.`;
        } else if (id === "input" || id === "output") {
          const key = id === "input" ? "input_length" : "output_length",
            values = r.map((row) => row[key]).filter(finite);
          if (state.modes[id] === "inflight") {
            if (!completeCohort())
              empty =
                "N/A · unfinished requests lack terminal records; the in-flight cohort is incomplete.";
            traces = [line("In-flight average", x, activeSeries()[key], blue)];
            axes = layout("Tokens");
            note =
              "Mean length among recorded terminal requests active at each sample";
            foot =
              id === "input"
                ? "Shared prefixes are counted per request; this is not unique-token occupancy."
                : "Delivered final output length of active requests, not tokens generated so far.";
          } else if (values.length) {
            const lo = values.reduce(
                (minimum, value) => Math.min(minimum, value),
                Infinity,
              ),
              hi = values.reduce(
                (maximum, value) => Math.max(maximum, value),
                -Infinity,
              ),
              width = (hi - lo || 1) / 32,
              bins = Array(32).fill(0);
            values.forEach(
              (value) => bins[Math.min(31, Math.floor((value - lo) / width))]++,
            );
            traces = [
              {
                type: "bar",
                name: "Requests",
                x: bins.map((_, index) => lo + (index + 0.5) * width),
                y: bins,
                width: width * 0.92,
                marker: { color: "#8b8d90" },
                showlegend: false,
                hovertemplate:
                  "%{x:,.0f} tokens<br>%{y} requests<extra></extra>",
              },
            ];
            for (const [p, shade] of [
              [50, blue],
              [75, green],
              [90, orange],
              [95, red],
            ]) {
              const q = quantile(values, p);
              traces.push(
                line(
                  `P${p} ${fmt(q, 0)}`,
                  [q, q],
                  [0, Math.max(...bins) * 1.03],
                  shade,
                  {
                    line: { color: shade, width: 1, dash: "solid" },
                    hoverinfo: "skip",
                  },
                ),
              );
            }
            axes = layout("Requests", {
              xaxis: {
                title: { text: "Tokens" },
                gridcolor: color("grid"),
                zeroline: false,
              },
            });
            note = `${fmt(values.length, 0)} requests · range ${fmt(lo, 0)}–${fmt(hi, 0)} tokens`;
            foot = `${fmt(r.length - values.length, 0)} requests without a recorded length excluded.`;
          } else empty = "No request length observations in this phase.";
        } else if (id === "interactivity" || id === "latency") {
          const key = id === "interactivity" ? "itl_ms" : state.modes.latency,
            p = state.percentiles[id];
          const valid = r
            .filter(
              (row) =>
                finite(row.arrival_time_ms) &&
                finite(row[key]) &&
                (id !== "interactivity" || row[key] > 0),
            )
            .sort((a, b) => a.arrival_time_ms - b.arrival_time_ms);
          const values = valid.map((row) => row[key]),
            xx = valid.map((row) => row.arrival_time_ms / 1000 - origin());
          const convert = (value) =>
            id === "interactivity" ? 1000 / value : value / 1000;
          const shade = id === "interactivity" ? color("cyan") : orange;
          traces = [
            line(
              id === "interactivity"
                ? `1 / rolling P${p} TPOT`
                : `Rolling P${p} (50 requests)`,
              xx,
              values.map((_, index) =>
                convert(
                  quantile(values.slice(Math.max(0, index - 49), index + 1), p),
                ),
              ),
              shade,
              { line: { color: shade, width: 1.5, shape: "hv" } },
            ),
            line(
              id === "interactivity"
                ? `1 / cumulative P${p} TPOT`
                : `Cumulative P${p}`,
              xx,
              cumulativeQuantiles(values, p).map(convert),
              red,
              { line: { color: red, width: 1.5, shape: "hv" } },
            ),
          ];
          axes = layout(
            id === "interactivity"
              ? "Tokens / second / user"
              : `${key === "ttft_ms" ? "TTFT" : "E2E"} (s)`,
          );
          note = `${fmt(valid.length, 0)} valid requests · rolling 50 and cumulative P${p}`;
          foot =
            id === "interactivity"
              ? "Percentiles apply to TPOT before inversion. Requests with fewer than two output tokens have no TPOT."
              : "Time is request arrival. Percentiles are unweighted across requests.";
          if (!valid.length)
            empty = "No valid latency observations in this phase.";
        } else if (id === "cache") {
          const option =
            cacheOptions().find((item) => item.key === state.modes.cache) ||
            cacheOptions()[0];
          const pools = ss.map((sample) =>
            (option.kind === "host" ? sample.host_pools : sample.workers)?.find(
              (pool) => cacheIdentity(pool) === option.identity,
            ),
          );
          if (option.kind === "host") {
            for (const [key, label, shade] of [
              ["resident_bytes", "Resident", blue],
              ["used_bytes", "Used incl. reservations", orange],
              ["capacity_bytes", "Capacity", purple],
            ])
              traces.push(
                line(
                  label,
                  x,
                  pools.map((pool) =>
                    finite(pool?.[key]) ? pool[key] / 2 ** 30 : null,
                  ),
                  shade,
                ),
              );
            axes = layout("Host cache (GiB)");
            note = option.label;
            foot =
              "Native physical bytes; used includes reserved transfer destinations. Shared pools appear once with their ownership scope.";
          } else {
            const values =
              option.key === "aggregate"
                ? ss.map((sample) => sample.cache_pct)
                : pools.map((pool) =>
                    finite(pool?.kv_cache_capacity_bytes) &&
                    pool.kv_cache_capacity_bytes > 0
                      ? percent(
                          pool.kv_cache_used_bytes,
                          pool.kv_cache_capacity_bytes,
                        )
                      : finite(pool?.physical_cache_usage)
                        ? 100 * pool.physical_cache_usage
                        : null,
                  );
            traces = [
              line(
                "Resident cache (trailing 50 s)",
                x,
                rolling(ss, values),
                blue,
              ),
            ];
            axes = layout("Resident KV cache (%)", {
              yaxis: {
                range: [0, 100],
                gridcolor: color("grid"),
                ticksuffix: "%",
              },
            });
            note =
              option.key === "aggregate"
                ? "Capacity-weighted resident cache / modeled rank capacity"
                : option.label;
            foot =
              "Includes retained prefixes; grouped caches use physical byte capacity. Gauges use a duration-weighted trailing 50 seconds.";
          }
        } else if (id === "queue") {
          if (state.modes.queue === "completed") {
            traces = [
              line(
                "Completed (cumulative)",
                x,
                cumulative(ss.map((sample) => sample.completed)),
                green,
              ),
            ];
            note = "Terminal completions recorded by interval counters";
          } else if (state.modes.queue === "preemptions") {
            traces = [
              line(
                "Preemptions / interval",
                x,
                ss.map((sample) => sample.preemptions),
                red,
              ),
              line(
                "Preemptions (cumulative)",
                x,
                cumulative(ss.map((sample) => sample.preemptions)),
                purple,
              ),
            ];
            note = "Native scheduler preemption counters";
          } else {
            for (const [key, label, shade] of [
              ["running", "Running", green],
              ["waiting", "Waiting", red],
              ["router_pending", "Router pending", purple],
            ])
              traces.push(
                line(
                  label + " (trailing 50 s)",
                  x,
                  rolling(
                    ss,
                    ss.map((sample) => sample[key]),
                  ),
                  shade,
                ),
              );
            traces.push(
              line(
                "Scheduler total (trailing 50 s)",
                x,
                rolling(
                  ss,
                  ss.map((sample) =>
                    finite(sample.running) && finite(sample.waiting)
                      ? sample.running + sample.waiting
                      : null,
                  ),
                ),
                blue,
              ),
            );
            note = "Scheduler gauges and router pending requests";
          }
          axes = layout("Requests");
          foot =
            "Router pending is separate from scheduler running and waiting; the scheduler total excludes it.";
        } else if (id === "hits") {
          traces = [
            line(
              "G1 prefix-cache lookup hit rate",
              x,
              ss.map((sample) =>
                percent(sample.lookup_hits, sample.lookup_queries),
              ),
              purple,
            ),
            line(
              "Scheduler prefix reuse",
              x,
              ss.map((sample) =>
                percent(sample.scheduler_hits, sample.scheduler_queries),
              ),
              blue,
            ),
          ];
          axes = layout("Hit rate (%)", {
            yaxis: {
              range: [0, 100],
              gridcolor: color("grid"),
              ticksuffix: "%",
            },
          });
          note =
            "Separate interval denominators for G1 lookups and scheduler reuse";
          foot =
            "G1 prefix-cache lookups measure device-cache matches; scheduler pass reuse and first-admission attribution remain separate.";
        } else if (id === "throughput") {
          let counters = ss,
            output = "Delivered output";
          if (state.modes.throughput === "completion") {
            counters = completionSeries();
            output = "Output at completion";
          }
          traces = [
            line(
              "Executed prefill (trailing 50 s)",
              x,
              throughput(counters, "prefill_tokens"),
              blue,
            ),
            line(
              output + " (trailing 50 s)",
              x,
              throughput(counters, "output_tokens"),
              orange,
            ),
          ];
          axes = layout("Tokens / second");
          note = "Trailing 50 seconds · partial intervals weighted by overlap";
          foot =
            state.modes.throughput === "completion"
              ? "Prefill remains executed work. Output is assigned to terminal intervals. Partial intervals assume uniform token flow; zero-duration event counters retain their tokens."
              : "Prefill excludes cache hits; output counts delivered tokens. Partial intervals assume uniform token flow; zero-duration event counters retain their tokens.";
        } else if (id === "source") {
          const total = cumulative(ss.map((sample) => sample.admission_input));
          for (const [key, label, shade] of [
            ["miss", "Compute (miss)", orange],
            ["hbm", "HBM reuse", blue],
            ["host", "Host reuse", purple],
          ])
            traces.push(
              line(
                label,
                x,
                cumulative(ss.map((sample) => sample[key])).map(
                  (value, index) => percent(value, total[index]),
                ),
                shade,
                {
                  stackgroup: "source",
                  fillcolor: shade,
                  line: { width: 2, color: shade, dash: key === "miss" ? "dot" : "solid" },
                },
              ),
            );
          axes = layout("Admitted prompt tokens (%)", {
            yaxis: {
              range: [0, 100],
              gridcolor: color("grid"),
              ticksuffix: "%",
            },
          });
          note = "Cumulative first-admission token attribution";
          foot =
            "Readmissions do not increase the first-admission denominator. Missing attribution remains N/A.";
        } else if (id === "footprint") {
          traces = [
            line(
              "Sum of largest active prompt per conversation",
              x,
              activeSeries().footprint,
              blue,
            ),
          ];
          axes = layout("Prompt tokens (proxy)");
          note =
            "Largest active prompt per play and conversation · terminal-record cohort";
          foot =
            "A conversation footprint proxy; does not measure unique tokens, generated output, or resident cache occupancy.";
          if (!completeCohort())
            empty =
              "N/A · unfinished requests lack terminal records; the footprint cohort is incomplete.";
          else if (!r.some((row) => conversation(row)))
            empty =
              "Conversation identities are unavailable; the footprint proxy is N/A.";
        } else if (id === "acceptance") {
          traces = [
            line(
              "Accepted tokens / decode forward",
              x,
              ss.map((sample) => sample.accept),
              purple,
              { mode: "lines+markers", marker: { size: 3 } },
            ),
          ];
          axes = layout("Tokens / forward");
          note = "Native decode acceptance observations";
          foot =
            "Acceptance length includes the base token. No observations remain N/A.";
        } else if (id === "completion") {
          const counts = new Map();
          r.forEach((row) => {
            const status = row.terminal_status || "unknown";
            counts.set(status, (counts.get(status) || 0) + 1);
          });
          traces = [
            {
              type: "bar",
              x: [...counts.keys()],
              y: [...counts.values()],
              marker: {
                color: [...counts.keys()].map((status) =>
                  status === "completed"
                    ? green
                    : status === "incomplete"
                      ? purple
                      : red,
                ),
              },
              text: [...counts.values()],
              textposition: "outside",
              hovertemplate: "%{x}: %{y}<extra></extra>",
            },
          ];
          axes = layout("Requests", {
            xaxis: { gridcolor: color("grid") },
            showlegend: false,
          });
          note = "Recorded terminal statuses in the selected phase";
          foot =
            "Includes zero-output and failed terminal records. Unfinished requests without terminal records are not counted.";
        }
        if (report.comparison && (id === "input" || id === "output") && state.modes[id] === "distribution")
          note += " AISim-only descriptive histogram; select In-flight for silicon versus AISim.";
        if (!empty && !traces.some((trace) => trace.y?.some(finite)))
          empty =
            "N/A · no observations for this metric in the selected phase.";
        return { traces, axes, note, foot, empty };
      }
      function title(id) {
        if (id === "latency")
          return (
            (state.modes.latency === "ttft_ms" ? "TTFT" : "E2E latency") +
            " over time"
          );
        if (id === "queue" && state.modes.queue === "completed")
          return "Completed requests";
        if (id === "queue" && state.modes.queue === "preemptions")
          return "Scheduler preemptions";
        return specs.find((spec) => spec.id === id).title;
      }
      async function draw(id, target) {
        const result = chart(id),
          node = $(target);
        node.classList.toggle("footprint-comparison", id === "footprint" && !!comparedPhase()?.charts.some(c => c.id === "unique"));
        if (result.empty) {
          Plotly.purge(node);
          node.classList.add("empty");
          node.textContent = result.empty;
        } else {
          if (node.classList.contains("empty")) node.textContent = "";
          node.classList.remove("empty");
          await Plotly.react(node, result.traces, result.axes, plotConfig);
        }
        return result;
      }
      async function renderCard(id) {
        const result = await draw(id, "plot-" + id);
        $("title-" + id).textContent = title(id);
        $("note-" + id).textContent = result.note;
        $("foot-" + id).textContent = result.foot;
        $("scores-" + id).innerHTML = scoreMarkup(id);
        $("controls-" + id).innerHTML = controls(
          specs.find((spec) => spec.id === id),
        );
      }
      async function renderExpanded(id) {
        $("expanded-title").textContent = title(id);
        if (!$("expanded-controls")) $("expanded-scores").insertAdjacentHTML("beforebegin", '<div id="expanded-controls" class="chart-controls"></div>');
        $("expanded-controls").innerHTML = controls(specs.find(spec => spec.id === id));
        const result = await draw(id, "expanded-plot");
        $("expanded-scores").innerHTML = scoreMarkup(id);
        $("expanded-note").textContent = result.note;
        $("expanded-foot").textContent = result.foot;
      }
      async function refreshChart(id) {
        await renderCard(id);
        if ($("chart-dialog").open && $("chart-dialog").dataset.chart === id) await renderExpanded(id);
      }
      async function renderGrid() {
        if (!$("chart-grid").children.length)
          $("chart-grid").innerHTML = specs
            .map(
              (spec) =>
                `<article class="chart-card" id="card-${spec.id}"><div class="chart-head"><h2 id="title-${spec.id}">${spec.title}</h2><div class="chart-controls" id="controls-${spec.id}"></div><button class="expand" data-expand="${spec.id}" aria-label="Expand ${spec.title}">⤢</button></div><div id="scores-${spec.id}" class="score-badges"></div><p class="chart-note" id="note-${spec.id}"></p><div class="plot" id="plot-${spec.id}" role="img" aria-label="${spec.title}"></div><p class="chart-foot" id="foot-${spec.id}"></p></article>`,
            )
            .join("");
        await Promise.all(specs.map((spec) => renderCard(spec.id)));
      }
      function workers(row) {
        const labels = new Set();
        (row.routing_history || []).forEach((route) => {
          if (route.logical_worker_id != null)
            labels.add(
              `${route.pool || "worker"} ${route.logical_worker_id}${route.dp_rank != null ? " · DP " + route.dp_rank : ""}`,
            );
        });
        if (!labels.size) {
          if (row.prefill_worker_idx != null)
            labels.add("Prefill " + row.prefill_worker_idx);
          if (row.decode_worker_idx != null)
            labels.add("Decode " + row.decode_worker_idx);
        }
        if (!labels.size && row.worker_id != null)
          labels.add("Worker " + row.worker_id);
        return labels.size ? [...labels] : ["Worker N/A"];
      }
      function timelineWindow() {
        const configured =
          state.stage === "profile"
            ? phase().admission_cutoff_ms
            : phase().end_ms;
        const end = finite(configured) ? configured : phase().end_ms;
        return finite(end) && end / 1000 >= origin()
          ? end / 1000 - origin()
          : null;
      }
      function renderTimeline() {
        $("timeline-pager").hidden = !comparedPhase();
        root.querySelector("#view-timeline .timeline-panel > .hint").textContent = `${comparedPhase() ? "10 minutes per screen · scroll horizontally or use < >" : "Scroll horizontally through time"} · click an interval or row label for details`;
        if (comparedPhase()) return renderComparedTimeline();
        const end = timelineWindow(),
          all = rows(),
          names = new Map();
        all.forEach((row) => {
          const identity = conversationKey(row) || "Unassigned";
          if (!names.has(identity))
            names.set(
              identity,
              conversation(row)
                ? `Conversation ${names.size + 1}`
                : "Conversation N/A",
            );
        });
        const eligible = all.filter(
          (row) =>
            finite(row.arrival_time_ms) &&
            (!finite(end) || row.arrival_time_ms / 1000 - origin() <= end),
        );
        const filtered = eligible.filter(
          (row) =>
            !state.search ||
            [
              row.request_id,
              row.uuid,
              conversation(row),
              names.get(conversationKey(row) || "Unassigned"),
              ...workers(row),
            ]
              .join(" ")
              .toLowerCase()
              .includes(state.search),
        );
        const groups = new Map();
        for (const row of filtered)
          for (const group of state.group === "conversation"
            ? [conversationKey(row) || "Unassigned"]
            : workers(row)) {
            if (!groups.has(group)) groups.set(group, []);
            groups.get(group).push(row);
          }
        $("timeline-summary").textContent =
          `${fmt(filtered.length, 0)} recorded terminal requests · ${fmt(groups.size, 0)} ${state.group === "conversation" ? "conversations" : "workers"} · all matching rows shown · ${finite(end) ? "window 0–" + fmt(end) + "s" : "window N/A"}${all.length - eligible.length ? " · " + fmt(all.length - eligible.length, 0) + " requests without a visible arrival" : ""}`;
        const node = $("timeline-plot");
        node.dataset.windowEnd = finite(end) ? String(end) : "";
        node.dataset.requestCount = String(filtered.length);
        if (!finite(end)) {
          node.innerHTML =
            '<p class="hint">N/A · the run did not record a phase window.</p>';
          return;
        }
        const scale = end > 120 ? 8 : 24,
          width = Math.max(900, Math.ceil(end * scale)),
          tick = Math.max(1, Math.ceil(end > 120 ? 30 : 5));
        const grid = `background-size:${tick * scale}px 100%`;
        let axis = "";
        for (let time = 0; time <= end; time += tick)
          axis += `<span class="timeline-tick" style="left:${time * scale}px">${fmt(time, 0)}s</span>`;
        const content = [...groups]
          .map(([group, requests]) => {
            const label =
              state.group === "conversation" ? names.get(group) : group;
            const intervals = requests
              .map((row) => {
                const start = Math.max(
                  0,
                  row.arrival_time_ms / 1000 - origin(),
                );
                const stop = finite(row.terminal_time_ms)
                  ? Math.min(end, row.terminal_time_ms / 1000 - origin())
                  : row.terminal_status === "incomplete"
                    ? end
                    : null;
                if (!finite(stop) || stop < start) return "";
                const status =
                  row.terminal_status === "completed"
                    ? "completed"
                    : row.terminal_status === "incomplete"
                      ? "incomplete"
                      : "failed";
                const label = `${row.request_id || row.uuid || "Request"} · ${row.phase} · ${row.terminal_status || "unknown"}\nTTFT ${fmt(row.ttft_ms)} ms · E2E ${fmt(row.e2e_latency_ms)} ms`;
                return `<button class="timeline-interval" data-request="${esc(keyFor(row))}" data-status="${status}" data-start-s="${start}" data-end-s="${stop}" style="left:${start * scale}px;width:${Math.max(2, (stop - start) * scale)}px" title="${esc(label)}" aria-label="${esc(label)}"></button>`;
              })
              .join("");
            return `<div class="timeline-row" data-group-key="${esc(group)}"><div class="timeline-label" title="${esc(group)}"><button data-row="${esc(group)}">${esc(label)}${state.group === "conversation" && group !== "Unassigned" ? " · …" + esc(group.slice(-8)) : ""}</button><small>${fmt(requests.length, 0)} request intervals</small></div><div class="timeline-track" style="${grid}">${intervals}</div></div>`;
          })
          .join("");
        node.innerHTML = `<div class="timeline-content" style="width:${width + 230}px"><div class="timeline-axis"><div class="timeline-label">Seconds since phase start</div><div class="timeline-track" style="${grid}">${axis}</div></div>${content || '<p class="hint">No requests match this filter.</p>'}</div>`;
        node.scrollLeft = 0;
      }
      const timelineOffsets = {};
      let timelineLayout = null;
      function comparisonTimelineGeometry() {
        const data = comparedPhase();
        const end = Math.max(data.native_duration_s, data.silicon_duration_s, 1);
        let pages = Math.ceil(end / 600);
        // Keep up to a minute of terminal drain in the final ten-minute window.
        if (pages > 1 && end % 600 > 0 && end % 600 <= 60) pages -= 1;
        const lastStart = (pages - 1) * 600, lastDuration = Math.max(600, end - lastStart);
        return {end, pages, lastStart, lastDuration};
      }
      function updateTimelinePager() {
        if (!timelineLayout || !comparedPhase() || state.view !== "timeline" || $("timeline-plot").clientWidth !== timelineLayout.viewport) return;
        const {pageWidth, pages} = timelineLayout, {lastDuration} = comparisonTimelineGeometry();
        const position = $("timeline-plot").scrollLeft / pageWidth;
        timelineOffsets[state.stage] = position;
        const page = Math.min(pages - 1, Math.floor(position + 0.001));
        const stop = position + 1 <= pages - 1 ? (position + 1) * 10 : (pages - 1) * 10 + (position + 2 - pages) * lastDuration / 60;
        $("timeline-page-label").textContent = `${fmt(position * 10, 1)}–${fmt(stop, 1)} min · ${page + 1} / ${pages}`;
        $("timeline-previous").disabled = position < 0.001;
        $("timeline-next").disabled = position >= pages - 1 - 0.001;
      }
      function sizeComparedTimeline() {
        const node = $("timeline-plot"), content = node.querySelector(".comparison-timeline");
        if (!content || !node.clientWidth || state.view !== "timeline" || !comparedPhase()) return;
        const labelWidth = parseFloat(getComputedStyle(content).getPropertyValue("--timeline-label-width"));
        const pageWidth = Math.max(1, node.clientWidth - labelWidth);
        const {pages, lastDuration} = comparisonTimelineGeometry();
        const offset = Math.min(timelineOffsets[state.stage] || 0, pages - 1);
        timelineLayout = {pageWidth, pages, viewport:node.clientWidth};
        content.style.width = `${labelWidth + pages * pageWidth}px`;
        content.style.setProperty("--timeline-minute-width", `${pageWidth / 10}px`);
        content.style.setProperty("--timeline-page-width", `${pageWidth}px`);
        content.style.setProperty("--timeline-final-minute-width", `${pageWidth * 60 / lastDuration}px`);
        node.scrollLeft = offset * pageWidth;
        updateTimelinePager();
      }
      $("timeline-plot").addEventListener("scroll", updateTimelinePager);
      const timelineResize = new ResizeObserver(sizeComparedTimeline);
      timelineResize.observe($("timeline-plot"));
      function renderComparedTimeline() {
        const data = comparedPhase(), all = data.requests;
        const workerControl = root.querySelector('[data-group="worker"]');
        workerControl.disabled = true;
        workerControl.title = "Silicon worker placement is unavailable";
        root.querySelector("#view-timeline .legend").innerHTML = '<span><i class="dot" style="background:var(--silicon)"></i>Silicon</span><span><i class="dot" style="background:var(--aisim)"></i>AISim</span><span>Failed · red outline</span>';
        const byId = new Map(all.map(r=>[r.uuid,r])), used = new Set(), pairs = [];
        for (const row of [...all.filter(r=>r.origin==="silicon"),...all.filter(r=>r.origin!=="silicon")]) {
          if (used.has(row.uuid)) continue;
          const counterpart = byId.get(row.counterpart), pair = counterpart ? [row,counterpart] : [row];
          pair.forEach(r=>used.add(r.uuid));
          if (!state.search || pair.some(r=>[r.request_id,r.uuid,r.group_id,r.origin,r.terminal_status].join(" ").toLowerCase().includes(state.search))) pairs.push(pair);
        }
        const {end, pages, lastStart, lastDuration} = comparisonTimelineGeometry();
        const span = lastStart + lastDuration;
        const percentAt = time => (time <= lastStart ? time / 600 : pages - 1 + (time - lastStart) / lastDuration) / pages * 100;
        $("timeline-summary").innerHTML = `${all.filter(r=>r.origin==="silicon").length} silicon requests · ${all.filter(r=>r.origin==="simulated").length} AISim requests · ${pairs.length} matched or unmatched rows · ${all.filter(r=>!finite(r.arrival_time_ms)||!finite(r.terminal_time_ms)).length} untimed requests (inspect by row label). Blue: silicon; yellow: AISim. Preparation is unscored.<div class="score-badges">${Object.entries(data.request_scores).map(([m,s])=>`<span class="score-badge">${m.replace("_ms", "").toUpperCase()} Request MAPE ${fmt(s.mape)}${finite(s.mape)?"%":""} · n=${s.count}</span>`).join("")}</div>`;
        $("timeline-plot").dataset.requestCount = String(all.length);
        $("timeline-plot").dataset.windowEnd = String(end);
        const axis = Array.from({length:Math.floor(span/60)+1},(_,i)=>`<span class="timeline-tick" style="left:${percentAt(i*60)}%;transform:translateX(${i===Math.floor(span/60)?-100:0}%)">${i}m</span>`).join("");
        const content = pairs.map(pair=>{
          const row = pair[0], requestId = row.request_id || row.uuid;
          const label = `Conversation ${row.group_id || "unknown"} · Request ${requestId} · ${pair.length===2?"paired":"unmatched"}`;
          const intervals = pair.map(r=>{
            if(!finite(r.arrival_time_ms)||!finite(r.terminal_time_ms))return "";
            const start=Math.max(0,r.arrival_time_ms/1000-origin()),stop=Math.min(end,r.terminal_time_ms/1000-origin());
            if(stop<start)return "";
            return `<button class="timeline-interval" data-request="${r.uuid}" data-origin="${r.origin}" data-status="${r.terminal_status}" style="left:${percentAt(start)}%;width:max(2px,${percentAt(stop)-percentAt(start)}%);top:${r.origin==="silicon"?3:12}px" title="${esc(r.origin)} · TTFT ${fmt(r.ttft_ms)} ms · TPOT ${fmt(r.itl_ms)} ms/token" aria-label="Inspect ${r.origin} request ${esc(r.request_id || r.uuid)}"></button>`;
          }).join("");
          return `<div class="timeline-row comparison-row"><div class="timeline-label"><button data-request="${row.uuid}" title="${esc(label)}" aria-label="${esc(label)}"><span class="conversation-id">${row.group_id?"…"+esc(row.group_id.slice(-8)):"Unknown"}</span> · ${esc(requestId)}</button></div><div class="timeline-track">${intervals}</div></div>`;
        }).join("");
        $("timeline-plot").innerHTML = `<div class="timeline-content comparison-timeline"><div class="timeline-axis" title="Minutes since each ${state.stage} started"><div class="timeline-label">Conversation · Request</div><div class="timeline-track">${axis}</div></div>${content}</div>`;
        sizeComparedTimeline();
      }
      const metric = (label, value) =>
        `<div class="stat"><small>${esc(label)}</small><strong>${esc(value ?? "N/A")}</strong></div>`;
      function renderSummary() {
        const r = rows(),
          complete = r.filter((row) => row.terminal_status === "completed");
        const duration =
          finite(phase().end_ms) && finite(phase().start_ms)
            ? (phase().end_ms - phase().start_ms) / 1000
            : null;
        $("stats").innerHTML =
          metric("Recorded terminal requests", fmt(r.length, 0)) +
          metric("Recorded completed", fmt(complete.length, 0)) +
          (state.stage === "profile" && finite(report.prediction.num_requests)
            ? metric("Profile requests", fmt(report.prediction.num_requests, 0))
            : "") +
          metric("Observed arrivals", fmt(arrivals(), 0)) +
          metric(
            "Mean TTFT",
            fmt(average(r.map((row) => row.ttft_ms))) + " ms",
          ) +
          metric(
            "Mean E2E",
            fmt(seconds(average(r.map((row) => row.e2e_latency_ms)))) + " s",
          ) +
          metric("Phase duration", fmt(duration) + " s") +
          metric(
            "Completion",
            phase().completion_status ||
              report.telemetry.completion_status ||
              "N/A",
          );
        $("cohort-note").textContent = completeCohort()
          ? "Request-derived charts use recorded terminal requests in the selected phase."
          : "The recorded request cohort is incomplete. Requests without terminal records are absent from request charts and the timeline; native scheduler gauges still include their work.";
        $("stage-count").textContent =
          `${fmt(r.length, 0)} requests · ${fmt(samples().length, 0)} samples`;
        $("clock-note").textContent =
          state.stage === "preparation"
            ? "Includes primers and warmup · phase-relative clock"
            : "Phase-relative clock · admission window " +
              fmt(timelineWindow()) +
              " s";
      }
      function showRequest(id) {
        const compared = comparedPhase()?.requests.find(r=>r.uuid === id);
        if (compared) {
          const other = comparedPhase().requests.find(r=>r.uuid === compared.counterpart);
          $("request-title").textContent = `${compared.origin === "silicon" ? "Silicon" : "AISim"} request`;
          const actual = compared.origin === "silicon" ? compared : other, predicted = compared.origin === "simulated" ? compared : other;
          const eligible = state.stage === "profile" && options.scoring !== false && actual?.terminal_status === "completed" && predicted?.terminal_status === "completed" && actual.output_length > 0 && actual.output_length === predicted.output_length;
          const error = key => eligible && key.endsWith("ms") && finite(actual[key]) && actual[key] > 0 && finite(predicted[key]) && predicted[key] >= 0 && (key !== "itl_ms" || actual.output_length > 1) ? 100 * Math.abs(predicted[key] - actual[key]) / actual[key] : null;
          $("request-details").innerHTML = `<p>${esc(compared.request_id)} · ${esc(compared.terminal_status)} · ${other?"Source paired":"Unmatched"}</p><table><thead><tr><th>Metric</th><th>Silicon</th><th>AISim</th><th>Request APE</th></tr></thead><tbody>${[["TTFT (ms)","ttft_ms"],["TPOT (ms/token)","itl_ms"],["E2E (ms)","e2e_latency_ms"],["Input tokens","input_length"],["Output tokens","output_length"]].map(([label,k])=>`<tr><th>${label}</th><td>${fmt(actual?.[k])}</td><td>${fmt(predicted?.[k])}</td><td>${fmt(error(k))}${finite(error(k))?"%":""}</td></tr>`).join("")}</tbody></table>${other?`<button data-request="${other.uuid}">Inspect ${other.origin === "silicon" ? "silicon" : "AISim"} counterpart</button>`:""}<p>Pairing alone does not qualify a request for accuracy: successful outcomes, equal output lengths, and valid measurements are required.</p>`;
          $("request-dialog").showModal(); return;
        }
        const row = requestById.get(id);
        if (!row) return;
        $("request-title").textContent = "Request details";
        const details = [
          ["Request ID", row.request_id || row.uuid],
          ["Phase", row.phase],
          ["Status", row.terminal_status],
          ["Input", fmt(row.input_length, 0) + " tokens"],
          ["Delivered output", fmt(row.output_length, 0) + " tokens"],
          ["Requested output", fmt(row.requested_output_length, 0) + " tokens"],
          ["Arrival", fmt(row.arrival_time_ms) + " ms"],
          ["First admission", fmt(row.first_admit_ms) + " ms"],
          ["Terminal", fmt(row.terminal_time_ms) + " ms"],
          ["TTFT", fmt(row.ttft_ms) + " ms"],
          ["TPOT", fmt(row.itl_ms) + " ms"],
          ["E2E", fmt(row.e2e_latency_ms) + " ms"],
          ["Conversation", conversation(row)],
          ["Worker placement", workers(row).join(" / ")],
          [
            "First-admission reuse",
            fmt(row.first_admission_reused_input_tokens, 0) + " tokens",
          ],
        ];
        const reference = report.reference?.pairs?.find(pair => pair.request_id === row.request_id);
        if (reference) details.push(
          ["Silicon match", reference.status],
          ["Silicon TTFT", fmt(reference.silicon_ttft_ms) + " ms"],
          ["Silicon TPOT", fmt(reference.silicon_tpot_ms) + " ms/token"],
          ["Request TTFT error", fmt(reference.ttft_ms_ape) + "%"],
          ["Request TPOT error", fmt(reference.tpot_ms_ape) + "%"],
        );
        const events = (report.lifecycle || []).filter((event) =>
          row.uuid && event.request_uuid
            ? event.request_uuid === row.uuid
            : row.agentic?.request_id &&
              event.request_id === row.agentic.request_id &&
              event.play_id === (row.play_id || row.agentic?.play_id),
        );
        $("request-details").innerHTML =
          '<div class="request-grid">' +
          details
            .map(
              ([label, value]) =>
                `<div><small>${esc(label)}</small><strong>${esc(value ?? "N/A")}</strong></div>`,
            )
            .join("") +
          `</div><p class="hint">Timestamps retain absolute virtual simulation time. Timeline intervals are clipped to the displayed admission window.</p><details><summary>Native request and lifecycle records</summary><pre>${esc(JSON.stringify({ request: row, lifecycle: events }, null, 2))}</pre></details>`;
        if (!$("request-dialog").open) $("request-dialog").showModal();
      }
      function showRow(group) {
        const selected = rows().filter((row) =>
          state.group === "conversation"
            ? (conversationKey(row) || "Unassigned") === group
            : workers(row).includes(group),
        );
        $("request-title").textContent =
          state.group === "conversation"
            ? "Conversation details"
            : "Worker details";
        $("request-details").innerHTML =
          `<p class="hint">${esc(group)}</p><div class="table-wrap"><table><thead><tr><th>Request</th><th>Status</th><th>Arrival (s)</th><th>TTFT (ms)</th><th>E2E (ms)</th><th>Input</th><th>Output</th></tr></thead><tbody>${selected.map((row) => `<tr><td><button data-request="${esc(keyFor(row))}">${esc(row.request_id || row.uuid || "Request")}</button></td><td>${esc(row.terminal_status || "unknown")}</td><td>${fmt(finite(row.arrival_time_ms) ? row.arrival_time_ms / 1000 - origin() : null)}</td><td>${fmt(row.ttft_ms)}</td><td>${fmt(row.e2e_latency_ms)}</td><td>${fmt(row.input_length, 0)}</td><td>${fmt(row.output_length, 0)}</td></tr>`).join("")}</tbody></table></div>`;
        $("request-dialog").showModal();
      }
      async function render() {
        root.dataset.ready = "false";
        root
          .querySelectorAll("[data-view]")
          .forEach((button) =>
            button.setAttribute(
              "aria-selected",
              String(button.dataset.view === state.view),
            ),
          );
        root.querySelectorAll("[data-stage]").forEach((button) => {
          button.setAttribute(
            "aria-selected",
            String(button.dataset.stage === state.stage),
          );
          button.hidden = !report.phases?.[button.dataset.stage];
        });
        root
          .querySelectorAll("[data-group]")
          .forEach((button) =>
            button.setAttribute(
              "aria-pressed",
              String(button.dataset.group === state.group),
            ),
          );
        ["dashboard", "timeline", "evidence"].forEach(view => {if ($("view-"+view)) $("view-"+view).hidden = view !== state.view;});
        if ($("profile-accuracy")) $("profile-accuracy").hidden = state.stage !== "profile";
        $("stagebar").hidden = !["dashboard","timeline"].includes(state.view);
        renderSummary();
        if (state.view === "dashboard") await renderGrid();
        if (state.view === "timeline") renderTimeline();
        if ($("chart-dialog").open) await renderExpanded($("chart-dialog").dataset.chart);
        root.dataset.ready = "true";
        options.onState?.(state);
      }
      root.addEventListener("click", async (event) => {
        const button = event.target.closest("button");
        if (!button) return;
        if (button.hasAttribute("data-close")) {
          button.closest("dialog").close();
          return;
        }
        if (button.dataset.view) {
          state.view = button.dataset.view;
          await render();
        } else if (button.dataset.stage) {
          state.stage = button.dataset.stage;
          await render();
        } else if (button.dataset.timelinePage && timelineLayout) {
          const direction = Number(button.dataset.timelinePage);
          const position = $("timeline-plot").scrollLeft / timelineLayout.pageWidth;
          const next = direction > 0 ? Math.floor(position + 0.001) + 1 : Math.ceil(position - 0.001) - 1;
          $("timeline-plot").scrollLeft = Math.max(0, Math.min(timelineLayout.pages - 1, next)) * timelineLayout.pageWidth;
          updateTimelinePager();
        } else if (button.dataset.group) {
          state.group = button.dataset.group;
          await render();
        } else if (button.dataset.mode) {
          state.modes[button.dataset.mode] = button.dataset.value;
          await refreshChart(button.dataset.mode);
        } else if (button.dataset.percentile) {
          state.percentiles[button.dataset.percentile] = Number(
            button.dataset.value,
          );
          await refreshChart(button.dataset.percentile);
        } else if (button.dataset.expand) {
          const id = button.dataset.expand;
          $("chart-dialog").dataset.chart = id;
          $("chart-dialog").showModal();
          await renderExpanded(id);
        } else if (button.dataset.request) showRequest(button.dataset.request);
        else if (button.dataset.row) showRow(button.dataset.row);
      });
      $("request-search").addEventListener("input", (event) => {
        state.search = event.target.value.trim().toLowerCase();
        renderTimeline();
      });
      root.addEventListener("change", async (event) => {
        if (event.target.dataset.selectMode) {
          state.modes[event.target.dataset.selectMode] = event.target.value;
          await refreshChart(event.target.dataset.selectMode);
        }
      });
      root.addEventListener("keydown", event => {
        const tab=event.target.closest('[role="tab"]');
        if (!tab || !["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
        const tabs=[...tab.parentElement.querySelectorAll('[role="tab"]')].filter(t=>!t.hidden), index=tabs.indexOf(tab);
        const next=event.key==="Home"?0:event.key==="End"?tabs.length-1:(index+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length;
        event.preventDefault();tabs[next].focus();tabs[next].click();
      });
      $("theme").addEventListener("click", async () => {
        const light = root.classList.toggle("light");
        $("theme").textContent = light ? "☀" : "☾";
        $("theme").setAttribute(
          "aria-label",
          light ? "Toggle dark theme" : "Toggle light theme",
        );
        await render();
      });
      const metadata = report.metadata || {};
      const heading = [
        metadata.hardware,
        metadata.model,
        metadata.framework,
      ].filter((value) => typeof value === "string" && value.length);
      $("run-title").textContent = heading.length
        ? heading.join(" · ")
        : metadata.title || "AISimulate replay";
      $("run-meta").textContent =
        report.telemetry.completion_status || "Completion N/A";
      render();
      return {render, state, destroy() { timelineResize.disconnect(); root.querySelectorAll(".plot").forEach(el => Plotly.purge(el)); }};

}};
