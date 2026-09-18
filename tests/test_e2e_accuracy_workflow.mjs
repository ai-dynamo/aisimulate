// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";
import vm from "node:vm";

const workflow = readFileSync(new URL("../.github/workflows/e2e-accuracy.yml", import.meta.url), "utf8");
const source = workflow.match(/          script: \|\n((?:(?:            .*)?\n)+)/)[1];
assert.match(source, /core\.setOutput\('matrix', JSON\.stringify\(\{include: entries\}\)\);\s*$/);
const sha = "a".repeat(40);
const nightlyWorkflow = readFileSync(new URL("../.github/workflows/nightly-ci.yml", import.meta.url), "utf8");
function nightlyScript(id, workflow = nightlyWorkflow) {
  const step = workflow.split(`        id: ${id}\n`)[1]?.split(/\n      - /)[0];
  assert.ok(step, `missing nightly step ${id}`);
  const script = step.match(/          script: \|\n((?:(?:            .*)?\n)+)/);
  assert.ok(script, `missing script for nightly step ${id}`);
  return script[1];
}

test("nightly script lookup rejects missing IDs", () => {
  assert.throws(() => nightlyScript("unknown-step"), /missing nightly step unknown-step/);
});

test("nightly script lookup cannot borrow a later step's script", () => {
  const fixture = `      - name: Missing script
        id: missing
        run: echo fixture
      - name: Different step
        id: different
        with:
          script: |
            throw new Error('wrong step');
`;
  assert.throws(() => nightlyScript("missing", fixture), /missing script for nightly step missing/);
});

async function nightlyVersion(created, number, event = "schedule", attempt = 1) {
  const outputs = {};
  const sandbox = vm.createContext({
    context: { repo: { owner: "ai-dynamo", repo: "aisimulate" }, runId: 123,
      eventName: event, runAttempt: attempt },
    core: { setOutput: (name, value) => { outputs[name] = value; } },
    github: { rest: { actions: { getWorkflowRun: async args => {
      assert.equal(args.owner, "ai-dynamo");
      assert.equal(args.repo, "aisimulate");
      assert.equal(args.run_id, 123);
      return { data: { created_at: created, run_number: number } };
    } } } },
  });
  await vm.runInContext(`(async () => { ${nightlyScript("version")} })()`, sandbox);
  return outputs;
}

test("nightly versions are unique, date ordered, and stable across retries", async () => {
  for (const event of ["schedule", "workflow_dispatch"]) {
    for (const [created, number, expected] of [
      ["2026-09-17T23:59:59Z", 1234, "202609170000001234"],
      ["2026-09-17T23:59:59Z", 1235, "202609170000001235"],
      ["2026-09-18T00:00:00Z", 1236, "202609180000001236"],
    ]) {
      const outputs = await nightlyVersion(created, number, event);
      assert.deepEqual(outputs, { "dev-date": created.slice(0, 10).replaceAll("-", ""),
        "dev-version": expected });
      assert.deepEqual(await nightlyVersion(created, number, event, 2), outputs);
    }
  }
});

test("nightly versions reject invalid run metadata", async () => {
  for (const number of [0, -1, 10000000000, "invalid"]) {
    await assert.rejects(nightlyVersion("2026-09-17T00:00:00Z", number), /invalid nightly/);
  }
  await assert.rejects(nightlyVersion("invalid", 1234), /invalid nightly/);
});

async function nightlyTarget({ event = "workflow_dispatch", ref = "refs/heads/main", requested = sha,
  branches = ["main", "release/0.12.0", "feature/test"], statuses = { main: "ahead" }, apiError } = {}) {
  const outputs = {};
  const failures = [];
  const compared = [];
  let lookups = 0;
  const sandbox = vm.createContext({
    context: { eventName: event, ref, sha, repo: { owner: "ai-dynamo", repo: "aisimulate" } },
    process: { env: { REQUESTED_SHA: requested } },
    core: { notice() {}, setOutput: (name, value) => { outputs[name] = value; }, setFailed: value => failures.push(value) },
    github: {
      paginate: async () => { lookups++; return branches.map(name => ({ name })); },
      rest: { repos: {
        listBranches: "branches",
        compareCommitsWithBasehead: async ({ basehead }) => {
          const [commit, branch] = basehead.split("...");
          assert.equal(commit, requested.trim() || sha);
          compared.push(branch);
          if (apiError) throw Object.assign(new Error("API unavailable"), { status: apiError });
          return { data: { status: statuses[branch] || "diverged" } };
        },
        getCommit: async () => ({ data: { commit: { committer: { date: "2026-09-17" }, message: "target commit" } } }),
      } },
    },
  });
  await vm.runInContext(`(async () => { ${nightlyScript("target")} })()`, sandbox);
  return { outputs, failures, compared, lookups };
}

test("nightly dispatch pins main or release ancestors while cron uses its own SHA", async () => {
  const scheduled = await nightlyTarget({ event: "schedule", requested: "invalid" });
  assert.deepEqual(scheduled.outputs, { sha, ref: "refs/heads/main" });
  assert.equal(scheduled.lookups, 0);
  for (const requested of [sha, "", ` ${sha} `]) {
    assert.deepEqual((await nightlyTarget({ requested })).outputs, { sha, ref: "refs/heads/main" });
  }
  for (const status of ["ahead", "identical"]) {
    const release = await nightlyTarget({ statuses: { "release/0.12.0": status } });
    assert.deepEqual(release.outputs, { sha, ref: "refs/heads/release/0.12.0" });
    assert.deepEqual(release.compared, ["main", "release/0.12.0"]);
  }
});

test("nightly dispatch rejects wrong workflow refs, malformed SHAs, and unrelated commits", async () => {
  for (const ref of ["refs/heads/release/0.12.0", "refs/heads/feature/test", "refs/tags/v0.12.0"]) {
    const result = await nightlyTarget({ ref });
    assert.equal(result.lookups, 0);
    assert.match(result.failures[0], /Dispatch from main/);
    assert.deepEqual(result.outputs, {});
  }
  for (const requested of ["abc123", "main", "g".repeat(40)]) {
    const result = await nightlyTarget({ requested });
    assert.equal(result.lookups, 0);
    assert.match(result.failures[0], /40-hex/);
  }
  for (const statuses of [{}, { main: "behind", "feature/test": "identical" }]) {
    const result = await nightlyTarget({ statuses });
    assert.deepEqual(result.outputs, {});
    assert.match(result.failures[0], /not on main/);
    assert.ok(!result.compared.includes("feature/test"));
  }
});

test("nightly target lookup fails closed on API failures", async () => {
  for (const apiError of [403, 429, 500]) await assert.rejects(nightlyTarget({ apiError }), /API unavailable/);
  const unknown = await nightlyTarget({ apiError: 404 });
  assert.deepEqual(unknown.outputs, {});
  assert.match(unknown.failures[0], /not on main/);
});

test("only successful scheduled nightlies can suppress another scheduled build", async () => {
  for (const event of ["schedule", "workflow_dispatch"]) {
    for (const previous of [sha, "b".repeat(40)]) {
      const outputs = {};
      let queries = 0;
      const sandbox = vm.createContext({
        context: { eventName: event, sha, repo: { owner: "ai-dynamo", repo: "aisimulate" } },
        core: { notice() {}, warning() {}, setOutput: (name, value) => { outputs[name] = value; } },
        github: { rest: { actions: { listWorkflowRuns: async args => {
          queries++;
          assert.equal(args.event, "schedule");
          assert.equal(args.status, "success");
          assert.equal(args.branch, "main");
          return { data: { workflow_runs: [{ head_sha: previous, html_url: "https://example.invalid/run" }] } };
        } } } },
      });
      await vm.runInContext(`(async () => { ${nightlyScript("decide")} })()`, sandbox);
      assert.equal(queries, event === "schedule" ? 1 : 0);
      assert.equal(outputs["should-build"], event === "schedule" && previous === sha ? "false" : "true");
    }
  }
});

const key = branch => createHash("sha256").update(branch).digest("hex").slice(0, 16);
const entry = (branch, revision, nightly = "") => ({ branch, sha: revision, nightly_run: nightly, artifact_key: key(branch) });
const run = {
  id: 123, head_sha: sha, head_branch: "main", event: "schedule",
  path: ".github/workflows/nightly-ci.yml", status: "waiting", conclusion: null,
  repository: { full_name: "ai-dynamo/aisimulate" },
  head_repository: { full_name: "ai-dynamo/aisimulate" },
};

async function resolve({ event = "schedule", runs = [run], built = true, expired = false, inputs = {}, branches = [] } = {}) {
  const outputs = {};
  const actions = {
    listWorkflowRuns: async (args) => {
      assert.equal(args.head_sha, sha);
      assert.equal(args.event, "schedule");
      return { data: { workflow_runs: runs } };
    },
    listJobsForWorkflowRun: "jobs",
    listWorkflowRunArtifacts: "artifacts",
  };
  const context = vm.createContext({
    require: createRequire(import.meta.url),
    context: { eventName: event, sha, repo: { owner: "ai-dynamo", repo: "aisimulate" } },
    process: { env: inputs },
    core: { notice() {}, setOutput: (name, value) => { outputs[name] = value; } },
    github: { rest: { actions, repos: { listBranches: "branches" } }, paginate: async (method, args) => {
      assert.equal(args.per_page, 100);
      if (method === "branches") {
        assert.equal(event, "schedule");
        return branches;
      }
      assert.equal(args.run_id, 123);
      if (method === "jobs") return [
        { name: "Build artifacts (amd64)", conclusion: built ? "success" : "failure" },
        { name: "Stage artifacts", conclusion: null, status: "waiting" },
      ];
      assert.equal(method, "artifacts");
      return [{ name: "nightly-dist-amd64", expired }];
    } },
  });
  await vm.runInContext(`(async () => { ${source} })()`, context);
  return JSON.parse(outputs.matrix).include;
}

test("scheduled accuracy reuses the exact main wheel while release staging waits", async () => {
  assert.deepEqual(await resolve(), [entry("main", sha, "123")]);
});

test("missing, failed, and expired nightly builds fall back to building the scheduled SHA", async () => {
  for (const scenario of [{ runs: [] }, { built: false }, { expired: true }]) {
    assert.deepEqual(await resolve(scenario), [entry("main", sha)]);
  }
});

test("wrong-revision and foreign producer wheels are rejected", async () => {
  for (const change of [{ head_sha: "b".repeat(40) }, { head_repository: { full_name: "foreign/repo" } }]) {
    await assert.rejects(resolve({ runs: [{ ...run, ...change }] }), /Untrusted nightly producer/);
  }
});

test("schedule pins each release head, excludes feature branches, and isolates artifacts", async () => {
  const branches = [
    { name: "release/0.13.0", commit: { sha: "c".repeat(40) } },
    { name: "main", commit: { sha: "f".repeat(40) } },
    { name: "codex/feature", commit: { sha } },
    { name: "release/0.12.0", commit: { sha: "b".repeat(40) } },
  ];
  const entries = await resolve({ branches });
  assert.deepEqual(entries, [entry("main", sha, "123"), entry("release/0.12.0", "b".repeat(40)), entry("release/0.13.0", "c".repeat(40))]);
  assert.equal(new Set(entries.map(item => item.artifact_key)).size, 3);
});

test("all paginated release branches are included and excessive matrices fail explicitly", async () => {
  const branches = Array.from({ length: 101 }, (_, i) => ({ name: `release/${i}`, commit: { sha } }));
  const entries = await resolve({ branches });
  assert.equal(entries.length, 102);
  assert.ok(entries.some(item => item.branch === "release/100"));
  await assert.rejects(resolve({ branches: Array.from({ length: 256 }, (_, i) => ({ name: `release/${i}`, commit: { sha } })) }), /256-job limit/);
});

test("invalid release commits and duplicate branch artifacts fail closed", async () => {
  await assert.rejects(resolve({ branches: [{ name: "release/0.12.0", commit: { sha: "short" } }] }), /full source SHA/);
  const release = { name: "release/0.12.0", commit: { sha } };
  await assert.rejects(resolve({ branches: [release, release] }), /Duplicate accuracy branch/);
});

test("manual evaluation emits only the requested full SHA and branch", async () => {
  for (const branch of ["main", "release/0.12.0"]) {
    const inputs = { EXPECTED_SHA: "b".repeat(40), EVALUATED_BRANCH: branch };
    assert.deepEqual(await resolve({ event: "workflow_dispatch", inputs }), [entry(branch, inputs.EXPECTED_SHA)]);
    await assert.rejects(resolve({ event: "workflow_dispatch", inputs: { ...inputs, EXPECTED_SHA: "short" } }), /full source SHA/);
  }
});

test("Pages accepts failed accuracy matrices while preserving other producer gates", () => {
  const pages = readFileSync(new URL("../.github/workflows/pages.yml", import.meta.url), "utf8");
  const expression = pages.match(/  build:\n    if: >-\n([\s\S]*?)    runs-on:/)[1].trim();
  for (const name of ["E2E Accuracy Matrix", "FPM Accuracy Matrix", "FPE Support Matrix", "Main branch nightly CI", "Release branch nightly CI", "Nightly CI"]) {
    for (const conclusion of ["success", "failure", "cancelled"]) {
      for (const repository of ["ai-dynamo/aisimulate", "foreign/repo"]) {
        const github = {
          event_name: "workflow_run", ref: "refs/heads/main", repository: "ai-dynamo/aisimulate",
          event: { workflow_run: { name, conclusion, head_repository: { full_name: repository } } },
        };
        assert.equal(vm.runInNewContext(expression, { github }), repository === github.repository &&
          (conclusion === "success" || (["E2E Accuracy Matrix", "FPM Accuracy Matrix"].includes(name) && conclusion === "failure")));
      }
    }
  }
});
