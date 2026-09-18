// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";
import vm from "node:vm";

const workflow = readFileSync(new URL("../.github/workflows/fpm-accuracy.yml", import.meta.url), "utf8");
const source = workflow.match(/          script: \|\n((?:(?:            .*)?\n)+)/)[1];
assert.match(source, /core\.setOutput\('matrix', JSON\.stringify\(\{include: entries\}\)\);\s*$/);
const sha = "a".repeat(40);
const key = branch => createHash("sha256").update(branch).digest("hex").slice(0, 16);
const entry = (branch, revision, nightly = "") => ({
  branch,
  sha: revision,
  nightly_run: nightly,
  nightly_attempt: nightly ? "2" : "",
  artifact_key: key(branch),
});
const run = {
  id: 123, run_attempt: 2, head_sha: sha, head_branch: "main", event: "schedule",
  path: ".github/workflows/nightly-ci.yml", status: "waiting", conclusion: null,
  repository: { full_name: "ai-dynamo/aisimulate" },
  head_repository: { full_name: "ai-dynamo/aisimulate" },
};

async function resolve({ event = "schedule", runs = [run], built = true, expired = false, inputs = {}, branches = [], hfResponse = { ok: true, sha: "d".repeat(40) } } = {}) {
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
    fetch: async () => ({ok: hfResponse.ok, json: async () => ({sha: hfResponse.sha})}),
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
      return [{ name: "nightly-build-metadata-amd64", expired }];
    } },
  });
  await vm.runInContext(`(async () => { ${source} })()`, context);
  assert.equal(outputs.hf_revision, "d".repeat(40));
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
  const branches = Array.from({ length: 101 }, (_, i) => ({ name: `release/1.0.${i}`, commit: { sha } }));
  const entries = await resolve({ branches });
  assert.equal(entries.length, 102);
  assert.ok(entries.some(item => item.branch === "release/1.0.100"));
  await assert.rejects(resolve({ branches: Array.from({ length: 256 }, (_, i) => ({ name: `release/1.0.${i}`, commit: { sha } })) }), /256-job limit/);
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

test("FPM excludes releases below 0.12.0 and compares versions numerically", async () => {
  const branches = ["release/0.9.0", "release/0.11.9", "release/0.12.0", "release/0.100.0", "release/1.0.0", "release/0.12.0-rc1"].map(name => ({name, commit: {sha}}));
  const result = await resolve({branches});
  assert.deepEqual(new Set(result.map(item => item.branch)), new Set(["main", "release/0.12.0", "release/0.100.0", "release/1.0.0"]));
  await assert.rejects(resolve({event: "workflow_dispatch", inputs: {EXPECTED_SHA: sha, EVALUATED_BRANCH: "release/0.11.9"}}), /full source SHA/);
});


test("HF resolution rejects failed responses and invalid immutable revisions", async () => {
  await assert.rejects(resolve({hfResponse: {ok: false}}), /Cannot resolve HF/);
  for (const sha of ["", "main", "a".repeat(39), "g".repeat(40)]) {
    await assert.rejects(resolve({hfResponse: {ok: true, sha}}), /HF revision/);
  }
});
