// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const workflow = readFileSync(new URL("../.github/workflows/e2e-accuracy.yml", import.meta.url), "utf8");
const source = workflow.match(/          script: \|\n((?:            .*\n)+)/)[1];
const sha = "a".repeat(40);
const run = {
  id: 123, head_sha: sha, head_branch: "main", event: "schedule",
  path: ".github/workflows/nightly-ci.yml", status: "waiting", conclusion: null,
  repository: { full_name: "ai-dynamo/aisimulate" },
  head_repository: { full_name: "ai-dynamo/aisimulate" },
};

async function resolve({ event = "schedule", runs = [run], built = true, expired = false, inputs = {} } = {}) {
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
    context: { eventName: event, sha, repo: { owner: "ai-dynamo", repo: "aisimulate" } },
    process: { env: inputs },
    core: { notice() {}, setOutput: (key, value) => { outputs[key] = value; } },
    github: { rest: { actions }, paginate: async (method, args) => {
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
  return outputs;
}

test("scheduled accuracy reuses the exact wheel while release staging waits", async () => {
  assert.deepEqual(await resolve(), { run: "true", sha, branch: "main", "nightly-run": "123" });
});

test("missing, failed, and expired nightly builds fall back to building the scheduled SHA", async () => {
  for (const scenario of [{ runs: [] }, { built: false }, { expired: true }]) {
    assert.deepEqual(await resolve(scenario), { run: "true", sha, branch: "main", "nightly-run": "" });
  }
});

test("wrong-revision and foreign producer wheels are rejected", async () => {
  for (const change of [{ head_sha: "b".repeat(40) }, { head_repository: { full_name: "foreign/repo" } }]) {
    await assert.rejects(resolve({ runs: [{ ...run, ...change }] }), /Untrusted nightly producer/);
  }
});

test("manual evaluation preserves the requested full SHA and release branch", async () => {
  const inputs = { EXPECTED_SHA: "b".repeat(40), EVALUATED_BRANCH: "release/0.12.0" };
  assert.deepEqual(await resolve({ event: "workflow_dispatch", inputs }), {
    run: "true", sha: inputs.EXPECTED_SHA, branch: inputs.EVALUATED_BRANCH, "nightly-run": "",
  });
  await assert.rejects(resolve({ event: "workflow_dispatch", inputs: { ...inputs, EXPECTED_SHA: "short" } }), /full source SHA/);
});
