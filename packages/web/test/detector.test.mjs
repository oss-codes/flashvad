import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { HysteresisDetector, validateDetectorConfig } from "../src/detector.mjs";

const fixture = JSON.parse(
  await readFile(new URL("./detector-parity.json", import.meta.url), "utf8"),
);

// Regenerate with src/flashvad/detector.py if the detector policy ever changes.
test("detector matches the Python implementation frame for frame", () => {
  const detector = new HysteresisDetector({}, 0.01);
  const events = [];
  for (const probability of fixture.probabilities) {
    events.push(...detector.update(probability));
  }

  assert.equal(events.length, fixture.events.length);
  events.forEach((actual, index) => {
    const expected = fixture.events[index];
    assert.equal(actual.kind, expected.kind);
    assert.equal(actual.frame, expected.frame);
    assert.ok(Math.abs(actual.timeSeconds - expected.timeSeconds) < 1e-9);
  });
});

test("reset restores a detector to its initial state", () => {
  const detector = new HysteresisDetector({}, 0.01);
  for (const probability of fixture.probabilities) detector.update(probability);
  detector.reset();

  const events = [];
  for (const probability of fixture.probabilities) {
    events.push(...detector.update(probability));
  }
  assert.deepEqual(
    events.map((event) => [event.kind, event.frame]),
    fixture.events.map((event) => [event.kind, event.frame]),
  );
});

test("invalid configuration is rejected", () => {
  assert.throws(() => validateDetectorConfig({ ...fixtureConfig(), stopThreshold: 0.9 }), RangeError);
  assert.throws(() => validateDetectorConfig({ ...fixtureConfig(), startFrames: 0 }), RangeError);
  assert.throws(() => new HysteresisDetector({ startThreshold: 1.5 }), RangeError);
});

test("out-of-range probabilities are rejected", () => {
  const detector = new HysteresisDetector({}, 0.01);
  assert.throws(() => detector.update(1.2), RangeError);
  assert.throws(() => detector.update(Number.NaN), RangeError);
});

function fixtureConfig() {
  return {
    startThreshold: 0.8,
    stopThreshold: 0.5,
    startFrames: 3,
    stopFrames: 4,
    preRollFrames: 3,
  };
}
