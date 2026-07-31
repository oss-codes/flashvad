import assert from "node:assert/strict";
import test from "node:test";

import { createRetryableLoader } from "../src/lib/retryable-loader.mjs";

test("a rejected runtime load can be retried", async () => {
  let attempts = 0;
  const loader = createRetryableLoader(async () => {
    attempts += 1;
    if (attempts === 1) {
      throw new Error("temporary failure");
    }
    return "ready";
  });

  await assert.rejects(loader.load(), /temporary failure/);
  assert.equal(await loader.load(), "ready");
  assert.equal(attempts, 2);
});

test("concurrent callers share one load and reset starts a new load", async () => {
  let attempts = 0;
  const loader = createRetryableLoader(async () => {
    attempts += 1;
    return attempts;
  });

  const first = loader.load();
  assert.equal(first, loader.load());
  assert.equal(await first, 1);
  loader.reset();
  assert.equal(await loader.load(), 2);
});
