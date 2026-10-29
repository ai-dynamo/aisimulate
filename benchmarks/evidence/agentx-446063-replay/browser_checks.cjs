// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Usage: NODE_PATH=<playwright modules> node browser_checks.cjs report.html [output-dir] [expected.json]
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

const c1Expected = {
  observed: {
    profile: 290,
    matched: 289,
    mape: 36.77528373140383,
    throughputApe: 14.77462294510577,
  },
  raw: {
    profile: 242,
    matched: 149,
    mape: 27.62074566227744,
    throughputApe: 40.40580419820861,
  },
};

async function main() {
  assert(
    process.argv[2],
    "Usage: node browser_checks.cjs report.html [output-dir] [expected.json]",
  );
  const reportPath = path.resolve(process.argv[2]);
  const outputDir = process.argv[3]
    ? path.resolve(process.argv[3])
    : await fs.mkdtemp(path.join(os.tmpdir(), "agentx-browser-"));
  await fs.mkdir(outputDir, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.AISIM_BROWSER_EXECUTABLE
      ? { executablePath: process.env.AISIM_BROWSER_EXECUTABLE }
      : {}),
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 },
    acceptDownloads: true,
  });
  const page = await context.newPage();
  const pageErrors = [];
  const externalRequests = [];
  const checks = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await context.route(/^https?:\/\//, (route) => {
    externalRequests.push(route.request().url());
    return route.abort();
  });
  async function check(name, action) {
    try {
      await action();
      checks.push({ name, passed: true });
      console.log(`PASS ${name}`);
    } catch (error) {
      checks.push({ name, passed: false, error: error.message });
      console.error(`FAIL ${name}: ${error.message}`);
    }
  }
  const plot = (id) => page.locator(`#plot-${id}`);
  const view = (id) => page.locator(`#tab-${id}`).click();
  const stage = (id) => page.locator(`#stage-${id}`).click();
  const screenshot = (name) =>
    page.screenshot({
      path: path.join(outputDir, `${name}.png`),
      fullPage: true,
    });
  async function variant(id) {
    await page.locator(`[data-variant="${id}"]`).click();
    await page.waitForFunction(
      (id) =>
        document
          .querySelector(`[data-variant="${id}"]`)
          .getAttribute("aria-pressed") === "true",
      id,
    );
  }
  async function count(value) {
    await page.waitForFunction(
      (value) =>
        document.getElementById("stage-count").textContent ===
        `${value.toLocaleString("en-US")} requests`,
      value,
    );
  }
  const close = (id) => page.locator(`#${id} [data-close]`).click();

  try {
    await page.goto(pathToFileURL(reportPath).href);
    await page.waitForFunction(
      () => document.getElementById("plot-input")?.data?.length > 0,
    );
    const initial = await page.locator("#report-data").textContent();
    const report = JSON.parse(initial);
    const primaryId = report.activeVariant;
    const primary = report.datasets.find((d) => d.id === primaryId);
    const pointId = primary.measurement.point_id ?? 446063;
    const variants = report.datasets.map((d) => d.id);
    const supplied = process.argv[4]
      ? JSON.parse(await fs.readFile(process.argv[4], "utf8"))
      : null;
    const expected =
      supplied?.datasets || (pointId === 446063 ? c1Expected : null);
    const phaseCounts = Object.fromEntries(
      report.datasets.map((d) => [
        d.id,
        {
          profile: d.requests.filter((r) => r.phase === "profile").length,
          preparation: d.requests.filter((r) => r.phase !== "profile").length,
        },
      ]),
    );

    await check(
      "benchmark identity and concurrency match the selected evidence",
      async () => {
        const expectedPoint = supplied?.point_id ?? pointId;
        const expectedConcurrency =
          supplied?.concurrency ?? primary.prediction.agentic_lanes;
        assert.equal(pointId, expectedPoint);
        assert.equal(primary.prediction.agentic_lanes, expectedConcurrency);
        assert.equal(
          await page.title(),
          `AgentX ${expectedPoint} · AISim replay`,
        );
        assert.equal(
          await page.locator("#reference-benchmark").getAttribute("href"),
          `https://inferencex.semianalysis.com/inference/agentic/${expectedPoint}`,
        );
        assert(
          (await page.locator("#point-identity").textContent()).includes(
            String(expectedPoint),
          ),
        );
        assert(
          (
            await page.locator(`[data-variant="${primaryId}"]`).textContent()
          ).includes(`c=${expectedConcurrency}`),
        );
        if (variants.length === 1) {
          assert.equal(await page.locator("#prev").isDisabled(), true);
          assert.equal(await page.locator("#next").isDisabled(), true);
        }
      },
    );

    await check(
      "embedded metrics and request counts match recorded replay evidence",
      async () => {
        if (supplied?.datasets)
          assert.deepEqual([...variants].sort(), Object.keys(expected).sort());
        if (expected) assert(variants.every((id) => id in expected));
        for (const d of report.datasets) {
          const e = expected?.[d.id];
          assert(phaseCounts[d.id].profile > 0);
          assert(phaseCounts[d.id].preparation > 0);
          const paired = d.measurement.comparisons.find((m) =>
            m.metric.includes("output-length matched"),
          );
          const throughput = d.measurement.comparisons.find(
            (m) => m.metric === "Output throughput · full profile",
          );
          assert(paired.count <= d.measurement.coverage.reference_successes);
          assert.equal(throughput.count, 1);
          if (e) {
            assert.equal(phaseCounts[d.id].profile, e.profile);
            assert.equal(phaseCounts[d.id].preparation, e.preparation ?? 11);
            assert.equal(paired.count, e.matched);
            assert(Math.abs(paired.mape_percent - e.mape) < 1e-10);
            assert(Math.abs(throughput.mape_percent - e.throughputApe) < 1e-10);
          }
        }
      },
    );
    await check(
      "each replay switches between its profile and preparation requests",
      async () => {
        for (const id of variants) {
          await variant(id);
          await stage("profile");
          await count(phaseCounts[id].profile);
          await stage("preparation");
          await count(phaseCounts[id].preparation);
          assert.match(
            await page.locator("#clock-note").textContent(),
            /Primer/,
          );
        }
        await screenshot("warmup");
        await stage("profile");
        await screenshot("desktop");
      },
    );
    await check(
      "distribution and in-flight average controls change the plotted data",
      async () => {
        for (const id of ["input", "output"]) {
          await page
            .locator(`[data-mode="${id}"][data-value="inflight"]`)
            .click();
          await page.waitForFunction(
            (id) =>
              document.getElementById(`plot-${id}`).data[0].name ===
              "In-flight avg",
            id,
          );
          await page
            .locator(`[data-mode="${id}"][data-value="distribution"]`)
            .click();
          assert.equal(await plot(id).evaluate((el) => el.data[0].type), "bar");
        }
      },
    );
    await check(
      "E2E and P75 controls update latency and interactivity traces",
      async () => {
        await page
          .locator('[data-mode="latency"][data-value="e2e_latency_ms"]')
          .click();
        assert.match(
          await page.locator("#title-latency").textContent(),
          /End-to-end/,
        );
        for (const id of ["latency", "interactivity"]) {
          await page
            .locator(`[data-percentile="${id}"][data-value="75"]`)
            .click();
          assert.match(
            await plot(id).evaluate((el) =>
              el.data.map((t) => t.name).join(" "),
            ),
            /75/,
          );
        }
      },
    );
    await check(
      "throughput and completed-request selectors change traces",
      async () => {
        const selector = page.getByLabel("Throughput series");
        await selector.selectOption("decode");
        assert.deepEqual(
          await plot("throughput").evaluate((el) => el.data.map((t) => t.name)),
          ["Decode / delivered output (avg n=50)"],
        );
        await selector.selectOption("prefill");
        assert.equal(
          await plot("throughput").evaluate((el) => el.data.length),
          1,
        );
        await selector.selectOption("both");
        assert.equal(
          await plot("throughput").evaluate((el) => el.data.length),
          3,
        );
        await page
          .locator('[data-mode="queue"][data-value="completed"]')
          .click();
        assert.match(
          await plot("queue").evaluate((el) => el.data[0].name),
          /Completed/,
        );
      },
    );
    await check(
      "isolated samples have markers and interactivity legends invert TPOT percentiles",
      async () => {
        for (const id of variants) {
          await variant(id);
          for (const metric of ["hits", "acceptance"]) {
            const traces = await plot(metric).evaluate((el) => el.data);
            assert(traces.length > 0);
            assert(
              traces.every(
                (trace) =>
                  trace.mode.includes("markers") && trace.marker.size > 0,
              ),
            );
          }
          const names = await plot("interactivity").evaluate((el) =>
            el.data.map((trace) => trace.name),
          );
          assert(
            names.every((name) =>
              /^1 \/ (rolling|cumulative) P75 TPOT$/.test(name),
            ),
          );
          assert.match(
            await page.locator("#foot-interactivity").textContent(),
            /TPOT before inversion/,
          );
        }
      },
    );
    await check(
      "final partial-interval throughput uses duration-weighted token counters",
      async () => {
        for (const id of variants) {
          await variant(id);
          const samples = report.datasets
            .find((d) => d.id === id)
            .samples.filter((s) => s.phase === "profile");
          assert(samples.at(-1).duration > 0 && samples.at(-1).duration <= 1);
          const window = samples
            .slice(-50)
            .filter((s) => s.duration > 0 && Number.isFinite(s.output_tokens));
          const expectedRate =
            window.reduce((total, sample) => total + sample.output_tokens, 0) /
            window.reduce((total, sample) => total + sample.duration, 0);
          const actualRate = await plot("throughput").evaluate((el) =>
            el.data.find((trace) => trace.name.startsWith("Decode")).y.at(-1),
          );
          assert(
            Math.abs(actualRate - expectedRate) < 1e-10,
            `${id}: ${actualRate} versus ${expectedRate} tok/s`,
          );
        }
      },
    );
    await check("expanded chart closes by button and Escape", async () => {
      for (const method of ["button", "escape"]) {
        await page.locator('[data-expand="latency"]').click();
        await page.locator("#chart-dialog[open]").waitFor();
        await page.waitForFunction(
          () => document.getElementById("expanded-plot").data?.length > 0,
        );
        if (method === "button") await close("chart-dialog");
        else await page.keyboard.press("Escape");
        assert.equal(
          await page.locator("#chart-dialog").evaluate((el) => el.open),
          false,
        );
      }
    });
    await check(
      "timeline grouping, request search, and request details work",
      async () => {
        await variant(primaryId);
        await view("timeline");
        await page.waitForFunction(
          () => document.getElementById("timeline-plot").data?.length > 0,
        );
        assert(
          (await page.locator("#timeline-summary").textContent()).includes(
            `${phaseCounts[primaryId].profile.toLocaleString("en-US")} requests`,
          ),
        );
        await page.locator('[data-group="worker"]').click();
        assert(
          (
            await page
              .locator("#timeline-plot")
              .evaluate((el) => el.data.flatMap((t) => t.y))
          ).every((v) => v.startsWith("Worker ")),
        );
        await page.locator('[data-group="conversation"]').click();
        await screenshot("timeline");
        const request = report.datasets
          .find((d) => d.id === primaryId)
          .requests.find((r) => r.phase === "profile" && r.output_length > 0);
        await page
          .getByLabel("Filter requests by ID or conversation")
          .fill(request.uuid);
        await page.waitForFunction(() =>
          document
            .getElementById("timeline-summary")
            .textContent.startsWith("1 requests"),
        );
        await page.locator('#timeline-plot[aria-busy="false"]').waitFor();
        const bars = page.locator("#timeline-plot .barlayer .point path");
        let clicked = false;
        for (const bar of await bars.all()) {
          const box = await bar.boundingBox();
          if (box && box.width > 3 && box.height > 3) {
            await bar.click({ force: true });
            clicked = true;
            break;
          }
        }
        assert(clicked, "No visible request bar available to click");
        await page.locator("#request-dialog[open]").waitFor({ timeout: 5000 });
        assert(
          (await page.locator("#request-details").textContent()).includes(
            request.request_id,
          ),
        );
        await close("request-dialog");
        await page.getByLabel("Filter requests by ID or conversation").fill("");
      },
    );
    await check(
      "timeline wheel zoom and reset restore the full range",
      async () => {
        await view("timeline");
        await page.getByLabel("Filter requests by ID or conversation").fill("");
        await page.locator('#timeline-plot[aria-busy="false"]').waitFor();
        const timeline = page.locator("#timeline-plot");
        await timeline.evaluate((el) => el.scrollIntoView({ block: "start" }));
        const bounds = await timeline.boundingBox();
        const rangeBefore = await timeline.evaluate((el) => [
          ...el._fullLayout.xaxis.range,
        ]);
        await page.mouse.move(
          bounds.x + bounds.width * 0.65,
          Math.min(
            bounds.y + bounds.height * 0.45,
            page.viewportSize().height * 0.65,
          ),
        );
        await page.mouse.wheel(0, -300);
        await page.waitForFunction(
          (before) => {
            const range =
              document.getElementById("timeline-plot")._fullLayout.xaxis.range;
            return range[1] - range[0] < before[1] - before[0];
          },
          rangeBefore,
          { timeout: 5000 },
        );
        // Plotly commits the wheel gesture after a debounce; reset follows that gesture.
        await page.waitForTimeout(350);
        await page.locator("#reset-timeline").click();
        await page.waitForFunction(
          () =>
            document.getElementById("timeline-plot")._fullLayout.xaxis
              .autorange === true,
          null,
          { timeout: 5000 },
        );
      },
    );
    await check(
      "accuracy tables preserve available variants and metric values",
      async () => {
        await view("accuracy");
        for (const id of variants) {
          await variant(id);
          const text = await page.locator("#view-accuracy").textContent();
          const d = report.datasets.find((d) => d.id === id);
          assert(
            text.includes(
              `AgentX ${pointId}, concurrency ${d.prediction.agentic_lanes}`,
            ),
          );
          for (const metric of [
            "E2E · output-length matched",
            "Output throughput · full profile",
          ]) {
            const value = d.measurement.comparisons.find(
              (m) => m.metric === metric,
            ).mape_percent;
            assert(
              text.includes(
                Number.isFinite(value)
                  ? value.toLocaleString("en-US", { maximumFractionDigits: 2 })
                  : "Unavailable",
              ),
            );
          }
        }
        await variant(primaryId);
      },
    );
    await check(
      "evidence download contains the original measurement JSON",
      async () => {
        await view("evidence");
        const downloading = page.waitForEvent("download");
        await page.locator('[data-download="measurement"]').click();
        const download = await downloading;
        assert.equal(
          download.suggestedFilename(),
          `agentx-${pointId}-measurement.json`,
        );
        const target = path.join(outputDir, download.suggestedFilename());
        await download.saveAs(target);
        assert.deepEqual(
          JSON.parse(await fs.readFile(target, "utf8")),
          report.datasets.find((d) => d.id === primaryId).measurement,
        );
      },
    );
    await check(
      "keyboard arrow, Home, and End navigation activates tabs",
      async () => {
        await page.locator("#tab-point").focus();
        await page.keyboard.press("ArrowRight");
        assert.equal(
          await page.locator("#tab-timeline").getAttribute("aria-selected"),
          "true",
        );
        await page.keyboard.press("End");
        assert.equal(
          await page.locator("#tab-evidence").getAttribute("aria-selected"),
          "true",
        );
        await page.keyboard.press("Home");
        assert.equal(
          await page.locator("#tab-point").getAttribute("aria-selected"),
          "true",
        );
        await page.locator("#stage-profile").focus();
        await page.keyboard.press("ArrowRight");
        await count(phaseCounts[primaryId].preparation);
        await page.keyboard.press("ArrowLeft");
        await count(phaseCounts[primaryId].profile);
      },
    );
    await check(
      "375px mobile layout has no page-level horizontal overflow",
      async () => {
        await page.setViewportSize({ width: 375, height: 812 });
        for (const id of ["point", "timeline", "accuracy", "evidence"]) {
          await view(id);
          await page.waitForFunction(
            () =>
              document.documentElement.scrollWidth <=
              document.documentElement.clientWidth + 1,
            null,
            { timeout: 5000 },
          );
          const size = await page.evaluate(() => ({
            client: document.documentElement.clientWidth,
            scroll: document.documentElement.scrollWidth,
          }));
          assert(
            size.scroll <= size.client + 1,
            `${id}: page width ${size.scroll}px exceeds viewport ${size.client}px`,
          );
          if (id === "point") await screenshot("mobile");
        }
      },
    );
    await check("interactions leave embedded metrics unchanged", async () => {
      assert.equal(await page.locator("#report-data").textContent(), initial);
    });
    await check(
      "no browser exceptions or external network requests",
      async () => {
        assert.deepEqual(pageErrors, []);
        assert.deepEqual(externalRequests, []);
      },
    );
  } finally {
    const result = {
      report: reportPath,
      screenshots: outputDir,
      checks,
      pageErrors,
      externalRequests,
    };
    await fs.writeFile(
      path.join(outputDir, "browser-checks.json"),
      JSON.stringify(result, null, 2) + "\n",
    );
    await browser.close();
    console.log(`Results: ${path.join(outputDir, "browser-checks.json")}`);
  }
  assert(
    checks.length > 0 && checks.every((check) => check.passed),
    "Browser checks failed; inspect browser-checks.json",
  );
}

main().catch((error) => {
  console.error(error.stack);
  process.exitCode = 1;
});
