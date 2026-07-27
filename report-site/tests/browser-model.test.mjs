import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import * as ort from "onnxruntime-web";
import {
  StreamingFeatureBuffer,
  extractVadFeatures,
} from "../src/lib/vad-features.mjs";

test("browser frontend stays numerically aligned with PyTorch", () => {
  const frame = Float32Array.from(
    { length: 400 },
    (_, index) =>
      0.17 * Math.sin((2 * Math.PI * 233 * index) / 16_000) +
      0.06 * Math.cos((2 * Math.PI * 811 * index) / 16_000),
  );
  const actual = extractVadFeatures(frame);
  const expected = [
    [0, 6.7627482414245605],
    [7, 1.8659915924072266],
    [8, 1.3044509887695312],
    [16, -4.301397323608398],
    [39, -4.729711532592773],
    [40, -2.062213182449341],
    [41, -20.411218643188477],
    [42, 0.032581452280282974],
  ];

  for (const [index, value] of expected) {
    assert.ok(
      Math.abs(actual[index] - value) < 5e-4,
      `feature ${index} drifted: ${actual[index]} versus ${value}`,
    );
  }
  const melMean =
    Array.from(actual.slice(0, 40)).reduce((sum, value) => sum + value, 0) /
    40;
  assert.ok(Math.abs(melMean) < 5e-6);
});

test("streaming frontend emits one causal frame per 10 ms hop", () => {
  const streaming = new StreamingFeatureBuffer();
  const first = streaming.push(new Float32Array(159));
  const second = streaming.push(new Float32Array(321));
  assert.equal(first.length, 0);
  assert.equal(second.length, 3);
  assert.equal(second[0].length, 43);
});

test("audio worklet forwards 10 ms capture chunks", async () => {
  const source = await readFile(
    new URL("../public/vad-audio-processor.js", import.meta.url),
    "utf8",
  );
  const messages = [];
  let Processor;
  class AudioWorkletProcessor {
    constructor() {
      this.port = {
        postMessage(value) {
          messages.push(Float32Array.from(value));
        },
      };
    }
  }
  vm.runInNewContext(source, {
    AudioWorkletProcessor,
    Float32Array,
    Math,
    sampleRate: 48_000,
    registerProcessor(_name, constructor) {
      Processor = constructor;
    },
  });
  const processor = new Processor();
  for (let index = 0; index < 4; index += 1) {
    assert.equal(processor.process([[new Float32Array(128)]]), true);
  }

  assert.equal(messages.length, 1);
  assert.equal(messages[0].length, 480);
});

test("browser ONNX checkpoint loads and executes one stateful frame", async () => {
  const model = await readFile(
    new URL("../public/models/flashvad-stream.onnx", import.meta.url),
  );
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
  });
  const output = await session.run({
    feature: new ort.Tensor("float32", new Float32Array(43), [1, 1, 43]),
    recurrent: new ort.Tensor("float32", new Float32Array(64), [1, 1, 64]),
    cache_0: new ort.Tensor("float32", new Float32Array(128), [1, 64, 2]),
    cache_1: new ort.Tensor("float32", new Float32Array(256), [1, 64, 4]),
    cache_2: new ort.Tensor("float32", new Float32Array(512), [1, 64, 8]),
    cache_3: new ort.Tensor("float32", new Float32Array(1024), [1, 64, 16]),
  });

  const probability =
    1 / (1 + Math.exp(-Number(output.speech_logits.data[0])));
  assert.ok(probability > 0 && probability < 1);
  assert.deepEqual(output.next_recurrent.dims, [1, 1, 64]);
  assert.deepEqual(output.next_cache_3.dims, [1, 64, 16]);
});
