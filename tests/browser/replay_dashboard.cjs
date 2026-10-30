// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Usage: NODE_PATH=<existing Playwright modules> node tests/browser/replay_dashboard.cjs report.html [output-dir]
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

const finite = (value) => typeof value === "number" && Number.isFinite(value);
const conversationKey = (row) =>
  row.agentic?.conversation_id || row.session_id
    ? JSON.stringify([
        row.play_id || row.agentic?.play_id || null,
        row.agentic?.conversation_id || row.session_id,
      ])
    : "Unassigned";
const timeWindow = (samples) =>
  samples
    .map((sample) => ({
      ...sample,
      overlap: Math.max(
        0,
        Math.min(sample.t, samples.at(-1).t) -
          Math.max(sample.start, samples.at(-1).t - 50),
      ),
    }))
    .filter((sample) => sample.overlap > 0);
const mean = (values) => {
  const valid = values.filter(finite);
  return valid.length
    ? valid.reduce((sum, value) => sum + value, 0) / valid.length
    : null;
};
function quantile(values, p) {
  const sorted = values.filter(finite).sort((a, b) => a - b);
  const index = ((sorted.length - 1) * p) / 100;
  return (
    sorted[Math.floor(index)] +
    (sorted[Math.ceil(index)] - sorted[Math.floor(index)]) *
      (index - Math.floor(index))
  );
}
function close(actual, expected, message) {
  if (expected === null) assert.equal(actual, null, message);
  else
    assert.ok(
      Math.abs(actual - expected) <= Math.max(1e-8, Math.abs(expected) * 1e-10),
      `${message}: ${actual} != ${expected}`,
    );
}

function activeReference(requests, observations) {
  const result = { input_length: [], output_length: [], footprint: [] };
  for (const sample of observations) {
    const time = sample.t * 1000;
    const active = requests.filter(
      (row) =>
        finite(row.arrival_time_ms) &&
        row.arrival_time_ms <= time &&
        (finite(row.terminal_time_ms)
          ? row.terminal_time_ms > time
          : row.terminal_status === "incomplete"),
    );
    result.input_length.push(mean(active.map((row) => row.input_length)));
    result.output_length.push(mean(active.map((row) => row.output_length)));
    const maxima = new Map();
    let unknown = false;
    for (const row of active) {
      if (
        !(row.agentic?.conversation_id || row.session_id) ||
        !finite(row.input_length)
      ) {
        unknown = true;
        break;
      }
      const key = conversationKey(row);
      maxima.set(key, Math.max(maxima.get(key) ?? 0, row.input_length));
    }
    result.footprint.push(
      unknown
        ? null
        : [...maxima.values()].reduce((sum, value) => sum + value, 0),
    );
  }
  return result;
}

async function main() {
  assert.ok(
    process.argv[2],
    "Usage: node replay_dashboard.cjs report.html [output-dir]",
  );
  const input = path.resolve(process.argv[2]);
  const output = process.argv[3]
    ? path.resolve(process.argv[3])
    : await fs.mkdtemp(path.join(os.tmpdir(), "aisim-dashboard-browser-"));
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.AISIM_BROWSER_EXECUTABLE
      ? { executablePath: process.env.AISIM_BROWSER_EXECUTABLE }
      : {}),
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 },
  });
  const page = await context.newPage();
  const errors = [],
    external = [],
    checks = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await context.route(/^https?:\/\//, (route) => {
    external.push(route.request().url());
    return route.abort();
  });
  async function check(name, run) {
    try {
      await run();
      checks.push({ name, passed: true });
      console.log(`PASS ${name}`);
    } catch (error) {
      checks.push({ name, passed: false, error: error.message });
      console.error(`FAIL ${name}: ${error.message}`);
    }
  }
  const ready = () =>
    page.waitForFunction(() => document.body.dataset.ready === "true");
  const traces = (id) =>
    page.locator(`#plot-${id}`).evaluate((node) =>
      (node.data || []).map((trace) => ({
        name: trace.name,
        x: trace.x,
        y: trace.y,
        type: trace.type,
      })),
    );
  async function view(name) {
    await page.locator(`#tab-${name}`).click();
    await ready();
  }
  async function stage(name) {
    await page.locator(`#stage-${name}`).click();
    await ready();
  }
  async function mode(id, value) {
    await page.locator(`[data-mode="${id}"][data-value="${value}"]`).click();
    await page.waitForFunction(
      ({ id, value }) =>
        document
          .querySelector(`[data-mode="${id}"][data-value="${value}"]`)
          .getAttribute("aria-pressed") === "true",
      { id, value },
    );
  }
  async function percentile(id, value) {
    await page
      .locator(`[data-percentile="${id}"][data-value="${value}"]`)
      .click();
    await page.waitForFunction(
      ({ id, value }) =>
        document
          .querySelector(`[data-percentile="${id}"][data-value="${value}"]`)
          .getAttribute("aria-pressed") === "true",
      { id, value },
    );
  }
  async function load(file) {
    await page.goto(pathToFileURL(file).href);
    await ready();
  }
  async function fixture(data, name) {
    const original = await fs.readFile(input, "utf8");
    const json = JSON.stringify(data).replaceAll("<", "\\u003c");
    const html = original.replace(
      /(<script id="report-data" type="application\/json">)[\s\S]*?(<\/script>)/,
      (_, before, after) => before + json + after,
    );
    const file = path.join(output, name + ".html");
    await fs.writeFile(file, html);
    await load(file);
  }
  try {
    await load(input);
    const report = JSON.parse(await page.locator("#report-data").textContent());
    const profile = report.requests.filter((row) => row.phase === "profile");
    const samples = report.samples.filter((row) => row.phase === "profile");
    await check(
      "offline simulation-only dashboard has twelve meaningful cards",
      async () => {
        assert.equal(report.schema, "aisimulate.replay.dashboard.v1");
        assert.equal(await page.locator(".chart-card").count(), 12);
        const text = await page.locator("body").innerText();
        assert.doesNotMatch(
          text,
          /silicon|MAPE|reference benchmark|point ID|replay variant/i,
        );
        for (const card of await page.locator(".plot").all()) {
          const valid = await card.evaluate((node) =>
            node.classList.contains("empty")
              ? /N\/A|No |unavailable/i.test(node.textContent)
              : node.data?.some((trace) =>
                  trace.y?.some(
                    (value) =>
                      typeof value === "number" && Number.isFinite(value),
                  ),
                ),
          );
          assert.ok(valid, await card.getAttribute("id"));
        }
        assert.equal(external.length, 0);
      },
    );
    await check(
      "ISL and OSL distributions retain all recorded lengths",
      async () => {
        for (const [id, key] of [
          ["input", "input_length"],
          ["output", "output_length"],
        ]) {
          const plotted = await traces(id);
          assert.equal(
            plotted[0].y.reduce((a, b) => a + b, 0),
            profile.filter((row) => finite(row[key])).length,
          );
        }
      },
    );
    await check(
      "in-flight means use active requests and delivered final lengths",
      async () => {
        for (const [id, key] of [
          ["input", "input_length"],
          ["output", "output_length"],
        ]) {
          await mode(id, "inflight");
          const plotted = (await traces(id))[0];
          for (const index of [
            0,
            Math.floor(samples.length / 2),
            samples.length - 1,
          ]) {
            const time = samples[index].t * 1000;
            const values = profile
              .filter(
                (row) =>
                  row.arrival_time_ms <= time &&
                  (finite(row.terminal_time_ms)
                    ? row.terminal_time_ms > time
                    : row.terminal_status === "incomplete"),
              )
              .map((row) => row[key]);
            close(plotted.y[index], mean(values), id);
          }
          await mode(id, "distribution");
        }
      },
    );
    await check(
      "active series sweep matches the reference at boundaries and missing observations",
      async () => {
        const base = {
          arrival_time_ms: 120000,
          terminal_time_ms: 124000,
          terminal_status: "completed",
          input_length: 30,
          output_length: 3,
          play_id: "first",
          agentic: { conversation_id: "shared" },
        };
        const requests = [
          base,
          { ...base, input_length: 70, terminal_time_ms: 122000 },
          { ...base, input_length: 70, terminal_time_ms: 123000 },
          { ...base, play_id: "second", input_length: 50 },
          { ...base, arrival_time_ms: 122000, input_length: 90 },
          { ...base, arrival_time_ms: 124000, terminal_time_ms: 124000 },
          { ...base, arrival_time_ms: 125000, terminal_time_ms: 124000 },
          { ...base, arrival_time_ms: null },
          { ...base, terminal_time_ms: null },
          { ...base, terminal_time_ms: null, terminal_status: "incomplete" },
          {
            ...base,
            arrival_time_ms: 124000,
            terminal_time_ms: 125000,
            agentic: null,
          },
          {
            ...base,
            arrival_time_ms: 125000,
            terminal_time_ms: 126000,
            input_length: null,
          },
          {
            ...base,
            arrival_time_ms: 126000,
            terminal_time_ms: 127000,
            output_length: null,
          },
          {
            ...base,
            arrival_time_ms: 127000,
            terminal_time_ms: 128000,
            terminal_status: "cancelled",
          },
          {
            ...base,
            arrival_time_ms: 127000,
            terminal_time_ms: 128000,
            terminal_status: "failed",
          },
          { ...base, agentic: null, session_id: "session", input_length: -1 },
        ];
        for (let index = 0; index < 200; index++)
          requests.push({
            ...base,
            arrival_time_ms: 120000 + (index % 41) * 53,
            terminal_time_ms: 120000 + (index % 41) * 53 + (index % 19) * 97,
            input_length: (index * 61) % 107,
            output_length: index % 7 === 0 ? null : index % 13,
            play_id: index % 3,
            agentic: { conversation_id: "conversation-" + (index % 11) },
          });
        const observations = [
          ...Array.from({ length: 230 }, (_, index) => ({
            t: 119 + index / 23,
          })),
          ...[119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130].map(
            (t) => ({ t }),
          ),
        ].reverse();
        const actual = await page.evaluate(
          ({ requests, observations }) =>
            activeRequestSeries(requests, observations),
          { requests, observations },
        );
        const expected = activeReference(requests, observations);
        for (const key of Object.keys(expected))
          expected[key].forEach((value, index) =>
            close(
              actual[key][index],
              value,
              `${key} at ${observations[index].t}`,
            ),
          );
      },
    );
    await check(
      "100k requests and 3600 samples use one event sweep",
      async () => {
        const performanceResult = await page.evaluate(() => {
          let arrivalReads = 0;
          const requests = Array.from({ length: 100000 }, (_, index) => ({
            get arrival_time_ms() {
              arrivalReads++;
              return 900000 + index * 36;
            },
            terminal_time_ms: 900000 + index * 36 + 1200000,
            terminal_status: "completed",
            input_length: (index * 61) % 100003,
            output_length: index % 1024,
            play_id: "large",
            session_id: "shared",
          }));
          const observations = Array.from({ length: 3600 }, (_, index) => ({
            t: 900 + index,
          }));
          const started = performance.now();
          const result = activeRequestSeries(requests, observations);
          return {
            elapsed_ms: performance.now() - started,
            arrival_reads: arrivalReads,
            samples: result.footprint.length,
            input_at_start: result.input_length[0],
            footprint_at_start: result.footprint[0],
            last_footprint: result.footprint.at(-1),
          };
        });
        assert.equal(performanceResult.samples, 3600);
        assert.equal(performanceResult.arrival_reads, 100000);
        assert.equal(performanceResult.input_at_start, 0);
        assert.equal(performanceResult.footprint_at_start, 0);
        assert.ok(performanceResult.last_footprint > 0);
        assert.ok(
          performanceResult.elapsed_ms < 5000,
          JSON.stringify(performanceResult),
        );
        await fs.writeFile(
          path.join(output, "active-series-performance.json"),
          JSON.stringify(performanceResult, null, 2),
        );
      },
    );
    await check(
      "active series cache preserves phase clocks across theme and phase redraws",
      async () => {
        const original = await page.evaluate(() => {
          globalThis.cachedSeriesBeforeRedraw = activeSeries();
          return activeSeries();
        });
        assert.deepEqual(original, activeReference(profile, samples));
        await page.locator("#theme").click();
        await ready();
        assert.equal(
          await page.evaluate(
            () => activeSeries() === globalThis.cachedSeriesBeforeRedraw,
          ),
          true,
        );
        assert.deepEqual(await page.evaluate(() => activeSeries()), original);
        if (report.phases.preparation) {
          await stage("preparation");
          const preparation = report.requests.filter((row) =>
            ["primer", "warmup", "preparation", "aborted"].includes(row.phase),
          );
          assert.deepEqual(
            await page.evaluate(() => activeSeries()),
            activeReference(
              preparation,
              report.samples.filter((sample) => sample.phase === "preparation"),
            ),
          );
          assert.equal(await page.evaluate(() => activeSeriesByPhase.size), 2);
          await stage("profile");
          assert.deepEqual(await page.evaluate(() => activeSeries()), original);
        }
        await page.locator("#theme").click();
        await ready();
        assert.equal(
          await page.evaluate(
            () => activeSeries() === globalThis.cachedSeriesBeforeRedraw,
          ),
          true,
        );
        await page.evaluate(() => {
          delete globalThis.cachedSeriesBeforeRedraw;
        });
      },
    );
    await check(
      "interactivity takes the TPOT percentile before inversion",
      async () => {
        await percentile("interactivity", 75);
        const valid = profile
          .filter((row) => finite(row.itl_ms) && row.itl_ms > 0)
          .sort((a, b) => a.arrival_time_ms - b.arrival_time_ms);
        if (valid.length) {
          const plotted = await traces("interactivity");
          close(
            plotted[0].y.at(-1),
            1000 /
              quantile(
                valid.slice(-50).map((row) => row.itl_ms),
                75,
              ),
            "rolling TPOT",
          );
          close(
            plotted[1].y.at(-1),
            1000 /
              quantile(
                valid.map((row) => row.itl_ms),
                75,
              ),
            "cumulative TPOT",
          );
        }
        await percentile("interactivity", 90);
      },
    );
    await check(
      "TTFT and E2E expose rolling 50 and cumulative P75/P90",
      async () => {
        for (const key of ["ttft_ms", "e2e_latency_ms"])
          for (const p of [75, 90]) {
            await mode("latency", key);
            await percentile("latency", p);
            const valid = profile
              .filter((row) => finite(row[key]))
              .sort((a, b) => a.arrival_time_ms - b.arrival_time_ms);
            if (valid.length) {
              const plotted = await traces("latency");
              close(
                plotted[0].y.at(-1),
                quantile(
                  valid.slice(-50).map((row) => row[key]),
                  p,
                ) / 1000,
                "rolling latency",
              );
              close(
                plotted[1].y.at(-1),
                quantile(
                  valid.map((row) => row[key]),
                  p,
                ) / 1000,
                "cumulative latency",
              );
            }
          }
        await mode("latency", "ttft_ms");
      },
    );
    await check(
      "cumulative percentiles scale to 100,000 requests",
      async () => {
        const result = await page.evaluate(() => {
          const values = Array.from(
            { length: 100000 },
            (_, index) => (index * 7919) % 100000,
          );
          const start = performance.now();
          const cumulative = cumulativeQuantiles(values, 90);
          return {
            elapsed_ms: performance.now() - start,
            last: cumulative.at(-1),
            count: cumulative.length,
          };
        });
        assert.equal(result.count, 100000);
        close(result.last, 89999.1, "large cumulative percentile");
        console.log(
          `100,000-request percentile: ${Math.round(result.elapsed_ms)} ms`,
        );
      },
    );
    await check(
      "resident cache, queue, completion and hit rates retain native semantics",
      async () => {
        const cache = (await traces("cache"))[0];
        if (cache && finite(samples.at(-1).cache_pct))
          close(
            cache.y.at(-1),
            timeWindow(samples).reduce(
              (sum, row) => sum + row.cache_pct * row.overlap,
              0,
            ) / timeWindow(samples).reduce((sum, row) => sum + row.overlap, 0),
            "resident cache",
          );
        const hits = await traces("hits");
        for (const [name, numerator, denominator] of [
          ["Native lookup hit rate", "lookup_hits", "lookup_queries"],
          ["Scheduler prefix reuse", "scheduler_hits", "scheduler_queries"],
        ]) {
          const trace = hits.find((row) => row.name === name);
          if (trace)
            samples.forEach((sample, index) =>
              close(
                trace.y[index],
                finite(sample[numerator]) && sample[denominator] > 0
                  ? (100 * sample[numerator]) / sample[denominator]
                  : null,
                name,
              ),
            );
        }
        await mode("queue", "completed");
        if (samples.every((sample) => finite(sample.completed)))
          close(
            (await traces("queue"))[0].y.at(-1),
            samples.reduce((total, sample) => total + sample.completed, 0),
            "cumulative completions",
          );
        await mode("queue", "queue");
        assert.equal(
          (await traces("completion"))[0].y.reduce((a, b) => a + b, 0),
          profile.length,
        );
      },
    );
    await check(
      "throughput distinguishes executed work from completion accounting",
      async () => {
        const window = timeWindow(samples);
        const duration = window.reduce(
          (sum, sample) => sum + sample.overlap,
          0,
        );
        const plotted = await traces("throughput");
        for (const [index, key] of [
          [0, "prefill_tokens"],
          [1, "output_tokens"],
        ]) {
          if (duration > 0 && window.every((sample) => finite(sample[key])))
            close(
              plotted[index].y.at(-1),
              window.reduce(
                (sum, sample) =>
                  sum + (sample[key] * sample.overlap) / sample.duration,
                0,
              ) / duration,
              key,
            );
        }
        await page.locator("#throughput-accounting").selectOption("completion");
        await page.waitForFunction(() =>
          document
            .querySelector("#plot-throughput")
            .data?.[1].name.startsWith("Output at completion"),
        );
        close(
          (await traces("throughput"))[0].y.at(-1),
          plotted[0].y.at(-1),
          "executed prefill remains unchanged",
        );
        let output = 0;
        for (const sample of window)
          output +=
            (profile
              .filter(
                (row) =>
                  row.terminal_status === "completed" &&
                  row.terminal_time_ms > sample.start * 1000 &&
                  row.terminal_time_ms <= sample.t * 1000,
              )
              .reduce((sum, row) => sum + row.output_length, 0) *
              sample.overlap) /
            sample.duration;
        if (duration > 0)
          close(
            (await traces("throughput"))[1].y.at(-1),
            output / duration,
            "completion output",
          );
        await page.locator("#throughput-accounting").selectOption("executed");
      },
    );
    await check(
      "first-admission sources, footprint proxy and acceptance are meaningful",
      async () => {
        const denominator = samples.reduce(
          (sum, sample) => sum + sample.admission_input,
          0,
        );
        const source = await traces("source");
        for (const [index, key] of [
          [0, "miss"],
          [1, "hbm"],
          [2, "host"],
        ])
          if (
            denominator > 0 &&
            samples.every(
              (sample) => finite(sample[key]) && finite(sample.admission_input),
            )
          )
            close(
              source[index].y.at(-1),
              (100 * samples.reduce((sum, sample) => sum + sample[key], 0)) /
                denominator,
              key,
            );
        const footprint = (await traces("footprint"))[0];
        if (footprint) {
          const index = Math.floor(samples.length / 2),
            time = samples[index].t * 1000,
            groups = new Map();
          const active = profile.filter(
            (row) =>
              row.arrival_time_ms <= time &&
              (finite(row.terminal_time_ms)
                ? row.terminal_time_ms > time
                : row.terminal_status === "incomplete"),
          );
          if (
            active.every(
              (row) =>
                (row.agentic?.conversation_id || row.session_id) &&
                finite(row.input_length),
            )
          ) {
            for (const row of active) {
              const id = conversationKey(row);
              groups.set(id, Math.max(groups.get(id) || 0, row.input_length));
            }
            close(
              footprint.y[index],
              [...groups.values()].reduce((sum, value) => sum + value, 0),
              "conversation proxy",
            );
          }
        }
        const acceptance = (await traces("acceptance"))[0];
        if (acceptance)
          samples.forEach((sample, index) =>
            close(acceptance.y[index], sample.accept, "acceptance"),
          );
      },
    );
    await check(
      "phase switch includes primers and warmup without profiling requests",
      async () => {
        if (report.phases.preparation) {
          await stage("preparation");
          const expected = report.requests.filter((row) =>
            ["primer", "warmup", "preparation", "aborted"].includes(row.phase),
          ).length;
          assert.ok(
            (await page.locator("#stage-count").innerText()).startsWith(
              `${expected.toLocaleString("en-US")} requests`,
            ),
          );
          assert.equal(
            (await traces("completion"))[0].y.reduce((a, b) => a + b, 0),
            expected,
          );
          await page.screenshot({
            path: path.join(output, "warmup.png"),
            fullPage: true,
          });
          await stage("profile");
        } else {
          assert.ok(await page.locator("#stage-preparation").isHidden());
          assert.ok(await page.locator("#stage-profile").isVisible());
        }
      },
    );
    await page.screenshot({
      path: path.join(output, "dashboard.png"),
      fullPage: true,
    });
    await check(
      "timeline shows every conversation and interval in the configured window",
      async () => {
        await view("timeline");
        const cutoff =
          report.phases.profile.admission_cutoff_ms ??
          report.phases.profile.end_ms;
        const visible = profile.filter(
          (row) => finite(row.arrival_time_ms) && row.arrival_time_ms <= cutoff,
        );
        assert.equal(
          await page.locator(".timeline-row").count(),
          new Set(visible.map((row) => conversationKey(row))).size,
        );
        assert.equal(
          Number(
            await page
              .locator("#timeline-plot")
              .getAttribute("data-window-end"),
          ),
          (cutoff - report.phases.profile.start_ms) / 1000,
        );
        const intervals = visible.filter(
          (row) =>
            finite(row.terminal_time_ms) ||
            row.terminal_status === "incomplete",
        );
        assert.equal(
          await page.locator(".timeline-interval").count(),
          intervals.length,
        );
        if (intervals.length) {
          await page.locator(".timeline-interval").first().click();
          assert.ok(await page.locator("#request-dialog").isVisible());
          assert.match(
            await page.locator("#request-details").innerText(),
            /TTFT|TPOT/,
          );
          await page.locator("#request-dialog [data-close]").click();
          await page.locator(".timeline-label [data-row]").first().click();
          assert.ok(await page.locator("#request-details table").isVisible());
          await page.locator("#request-dialog [data-close]").click();
          await page.locator("#request-search").fill(intervals[0].uuid);
          assert.equal(
            Number(
              await page
                .locator("#timeline-plot")
                .getAttribute("data-request-count"),
            ),
            1,
          );
          await page.locator("#request-search").fill("");
        }
        const scroll = await page.locator("#timeline-plot").evaluate((node) => {
          node.scrollLeft = 500;
          return {
            left: node.scrollLeft,
            width: node.scrollWidth,
            client: node.clientWidth,
          };
        });
        assert.ok(scroll.width > scroll.client && scroll.left > 0);
        await page.locator('[data-group="worker"]').click();
        await ready();
        assert.ok((await page.locator(".timeline-row").count()) > 0);
        await page.locator('[data-group="conversation"]').click();
        await ready();
        await page.screenshot({
          path: path.join(output, "timeline.png"),
          fullPage: true,
        });
      },
    );
    await check(
      "theme and expanded chart work without help or run details",
      async () => {
        await view("dashboard");
        await page.locator("#theme").click();
        await ready();
        assert.ok(await page.locator("body.light").count());
        await page.locator('[data-expand="latency"]').click();
        assert.ok(await page.locator("#chart-dialog").isVisible());
        await page.locator("#chart-dialog [data-close]").click();
        assert.equal(await page.locator("#help, #help-dialog").count(), 0);
        assert.equal(
          await page.locator("#tab-evidence, #view-evidence").count(),
          0,
        );
        assert.equal(await page.locator("[data-view]").count(), 2);
      },
    );
    await check(
      "missing metric observations remain N/A instead of zero",
      async () => {
        const data = structuredClone(report);
        data.phases = {
          profile: {
            start_ms: 0,
            end_ms: 8000,
            admission_cutoff_ms: 8000,
            completion_status: "incomplete",
          },
        };
        data.requests = [
          {
            uuid: "unknown",
            phase: "profile",
            arrival_time_ms: 0,
            terminal_time_ms: 7000,
            terminal_status: "incomplete",
          },
        ];
        const sample = Object.fromEntries(
          Object.keys(data.samples[0]).map((key) => [key, null]),
        );
        Object.assign(sample, {
          phase: "profile",
          t: 1,
          start: 0,
          duration: 1,
          workers: [],
          host_pools: [],
        });
        data.samples = [sample];
        data.lifecycle = [];
        await fixture(data, "unknown-metrics");
        for (const id of [
          "input",
          "output",
          "interactivity",
          "latency",
          "cache",
          "queue",
          "hits",
          "throughput",
          "source",
          "footprint",
          "acceptance",
        ])
          assert.ok(await page.locator(`#plot-${id}.empty`).count(), id);
        assert.ok(await page.locator("#stage-preparation").isHidden());
        assert.ok(await page.locator("#stage-profile").isVisible());
        assert.equal((await traces("completion"))[0].y[0], 1);
      },
    );
    await check(
      "physical 50-second windows, host ownership and preemptions survive coarse sampling",
      async () => {
        const data = structuredClone(report);
        data.prediction.num_requests = 1;
        data.phases = {
          profile: {
            start_ms: 0,
            end_ms: 70000,
            admission_cutoff_ms: 70000,
            completion_status: "completed",
          },
        };
        data.requests = [
          {
            uuid: "coarse",
            phase: "profile",
            arrival_time_ms: 0,
            terminal_time_ms: 70000,
            terminal_status: "completed",
            input_length: 100,
            output_length: 1000,
            session_id: "coarse",
          },
        ];
        data.samples = [
          {
            phase: "profile",
            t: 30,
            start: 0,
            duration: 30,
            prefill_tokens: 3000,
            output_tokens: 600,
            cache_pct: 20,
            running: 2,
            waiting: 0,
            router_pending: 0,
            preemptions: 1,
            arrivals: 1,
            completed: 0,
          },
          {
            phase: "profile",
            t: 70,
            start: 30,
            duration: 40,
            prefill_tokens: 8000,
            output_tokens: 1600,
            cache_pct: 80,
            running: 8,
            waiting: 0,
            router_pending: 0,
            preemptions: 2,
            arrivals: 0,
            completed: 1,
          },
        ];
        data.samples.forEach((sample, index) => {
          sample.workers = [
            {
              role: "decode",
              worker_id: 2,
              dp_rank: 0,
              kv_cache_capacity_bytes: 4 * 2 ** 30,
              kv_cache_used_bytes: (index ? 3 : 1) * 2 ** 30,
            },
          ];
          sample.host_pools = [
            {
              role: null,
              scope: "cluster_shared",
              resident_bytes: 2 ** 30,
              used_bytes: 2 * 2 ** 30,
              capacity_bytes: 4 * 2 ** 30,
            },
            {
              role: "decode",
              scope: "local",
              worker_id: 2,
              dp_rank: 0,
              resident_bytes: 2 ** 29,
              used_bytes: 2 ** 30,
              capacity_bytes: 2 ** 31,
            },
          ];
        });
        data.samples.push({
          ...data.samples[1],
          start: 70,
          duration: 0,
          prefill_tokens: 100,
          output_tokens: 50,
          preemptions: 0,
          arrivals: 0,
          completed: 0,
        });
        data.lifecycle = [];
        await fixture(data, "coarse-sampling");
        close(
          (await traces("throughput"))[0].y.at(-1),
          182,
          "50-second prefill rate",
        );
        close(
          (await traces("throughput"))[1].y.at(-1),
          37,
          "50-second output rate",
        );
        close(
          (await traces("cache"))[0].y.at(-1),
          68,
          "duration-weighted cache gauge",
        );
        close(
          (await traces("queue"))[0].y.at(-1),
          6.8,
          "duration-weighted running gauge",
        );
        const selector = page.locator('[data-select-mode="cache"]');
        const options = await selector.locator("option").evaluateAll((nodes) =>
          nodes.map((node) => ({
            value: node.value,
            label: node.textContent,
          })),
        );
        assert.equal(options.length, 4);
        const shared = options.find((option) =>
          option.label.includes("deployment shared"),
        );
        assert.ok(shared.label.includes("cluster_shared"));
        await selector.selectOption(shared.value);
        await page.waitForFunction(
          () =>
            document.querySelector("#plot-cache").data?.[0].name === "Resident",
        );
        assert.deepEqual(
          (await traces("cache")).map((trace) => trace.y.at(-1)),
          [1, 2, 4],
        );
        await selector.selectOption(
          options.find((option) => option.value.startsWith("worker:")).value,
        );
        await page.waitForFunction(() =>
          document
            .querySelector("#plot-cache")
            .data?.[0].name.startsWith("Resident cache"),
        );
        close(
          (await traces("cache"))[0].y.at(-1),
          65,
          "grouped byte utilization",
        );
        await mode("queue", "preemptions");
        assert.deepEqual(
          (await traces("queue")).map((trace) => trace.y.at(-1)),
          [0, 3],
        );
        await page.locator("#throughput-accounting").selectOption("completion");
        await page.waitForFunction(() =>
          document
            .querySelector("#plot-throughput")
            .data?.[1].name.startsWith("Output at completion"),
        );
        assert.deepEqual(
          (await traces("throughput")).map((trace) => trace.y.at(-1)),
          [182, 20],
        );
      },
    );
    await check(
      "unfinished traffic is not presented as an empty workload",
      async () => {
        const data = structuredClone(report);
        data.prediction.num_requests = 2;
        data.phases = {
          profile: {
            start_ms: 0,
            end_ms: 1000,
            completion_status: "time_limit",
          },
        };
        data.requests = [];
        data.lifecycle = [];
        data.samples = [
          {
            phase: "profile",
            t: 1,
            start: 0,
            duration: 1,
            waiting: 2,
            running: 0,
            arrivals: 2,
            completed: 0,
          },
        ];
        await fixture(data, "unfinished-traffic");
        assert.match(
          await page.locator("#stats").innerText(),
          /Profile requests\s+2/i,
        );
        assert.match(
          await page.locator("#cohort-note").innerText(),
          /without terminal records/,
        );
        await mode("input", "inflight");
        assert.match(
          await page.locator("#plot-input").innerText(),
          /cohort is incomplete/,
        );
        assert.match(
          await page.locator("#plot-footprint").innerText(),
          /cohort is incomplete/,
        );
        const waiting = (await traces("queue")).find((trace) =>
          trace.name.startsWith("Waiting"),
        );
        assert.equal(waiting.y[0], 2);
        await view("timeline");
        assert.match(
          await page.locator("#timeline-summary").innerText(),
          /0 recorded terminal requests/,
        );
      },
    );
    await check(
      "timeline has no 400-row cap and uses the configured non-hour window",
      async () => {
        const data = structuredClone(report);
        data.phases = {
          profile: {
            start_ms: 2000,
            end_ms: 12000,
            admission_cutoff_ms: 11000,
            completion_status: "completed",
          },
        };
        data.requests = Array.from({ length: 405 }, (_, index) => ({
          uuid: "request-" + index,
          request_id: "request-" + index,
          phase: "profile",
          arrival_time_ms: 2500,
          terminal_time_ms: 11500,
          terminal_status: "completed",
          input_length: 5,
          output_length: 2,
          ttft_ms: 1,
          itl_ms: 2,
          e2e_latency_ms: 9000,
          play_id: "play-" + index,
          agentic: { conversation_id: "same-conversation" },
          decode_worker_idx: index % 3,
        }));
        data.samples = [{ phase: "profile", t: 3, start: 2, duration: 1 }];
        data.lifecycle = [];
        await fixture(data, "all-timeline-rows");
        await view("timeline");
        assert.equal(await page.locator(".timeline-row").count(), 405);
        assert.equal(await page.locator(".timeline-interval").count(), 405);
        assert.equal(
          await page.locator("#timeline-plot").getAttribute("data-window-end"),
          "9",
        );
        assert.equal(
          await page
            .locator(".timeline-interval")
            .first()
            .getAttribute("data-end-s"),
          "9",
        );
        await page.locator('[data-group="worker"]').click();
        await ready();
        assert.equal(await page.locator(".timeline-row").count(), 3);
        assert.equal(await page.locator(".timeline-interval").count(), 405);
        await page.setViewportSize({ width: 390, height: 844 });
        await view("dashboard");
        await page.waitForFunction(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        );
      },
    );
    await check("no browser errors or external network requests", async () => {
      assert.deepEqual(errors, []);
      assert.deepEqual(external, []);
    });
    await fs.writeFile(
      path.join(output, "browser-checks.json"),
      JSON.stringify({ checks, errors, external }, null, 2),
    );
    if (checks.some((check) => !check.passed)) process.exitCode = 1;
    console.log(`Evidence: ${output}`);
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
