import assert from "node:assert/strict";
import test from "node:test";
import {
  detectSpeechFrames,
  formatAnalysisTime,
  speakingStateAtTime,
  summarizeTrack,
} from "../src/lib/call-analysis.mjs";

test("detector applies start and stop persistence to a probability track", () => {
  const speech = detectSpeechFrames(
    [0.1, 0.9, 0.91, 0.92, 0.8, 0.3, 0.2, 0.1, 0.05],
    0.8,
  );

  assert.deepEqual(speech, [
    false,
    false,
    false,
    true,
    true,
    true,
    true,
    true,
    false,
  ]);
});

test("timeline summarization preserves peaks and speech activity", () => {
  const probabilities = [0.1, 0.2, 0.9, 0.8];
  const speech = [false, false, true, true];
  const bins = summarizeTrack(probabilities, speech, 2);

  assert.equal(bins.length, 2);
  assert.equal(bins[0].active, false);
  assert.equal(bins[0].peak, 0.2);
  assert.equal(bins[1].active, true);
  assert.equal(bins[1].peak, 0.9);
});

test("playhead state distinguishes user, AI, overlap, and silence", () => {
  const tracks = [
    { role: "user", speechFrames: [false, true, true, false] },
    { role: "ai", speechFrames: [false, false, true, true] },
  ];

  assert.equal(speakingStateAtTime(tracks, 0), "silence");
  assert.equal(speakingStateAtTime(tracks, 10), "user");
  assert.equal(speakingStateAtTime(tracks, 20), "overlap");
  assert.equal(speakingStateAtTime(tracks, 30), "ai");
  assert.equal(formatAnalysisTime(62_340), "1:02.3");
});
