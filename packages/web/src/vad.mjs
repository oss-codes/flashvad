/**
 * Streaming FlashVAD for the browser.
 *
 * onnxruntime-web is a peer dependency the caller passes in rather than an
 * import here. Bundling it would pin a wasm build and duplicate several MB in
 * any app that already loads ORT for another model.
 */

import {
  StreamingFeatureBuffer,
  StreamingLinearResampler,
  VAD_SAMPLE_RATE,
  VAD_HOP_SAMPLES,
} from "./features.mjs";
import { HysteresisDetector } from "./detector.mjs";

const RECURRENT_DIM = 64;
const CACHE_WIDTHS = [2, 4, 8, 16];

/** Numerically stable logistic; the graph emits a logit, not a probability. */
function sigmoid(logit) {
  const exponential = Math.exp(-Math.abs(logit));
  return logit >= 0 ? 1 / (1 + exponential) : exponential / (1 + exponential);
}

function freshState() {
  return {
    recurrent: new Float32Array(RECURRENT_DIM),
    caches: CACHE_WIDTHS.map((width) => new Float32Array(RECURRENT_DIM * width)),
  };
}

export class FlashVad {
  #ort;
  #session;
  #features;
  #resampler;
  #detector;
  #state;

  constructor({ ort, session, sampleRate, detector }) {
    this.#ort = ort;
    this.#session = session;
    this.#features = new StreamingFeatureBuffer();
    this.#resampler =
      sampleRate && sampleRate !== VAD_SAMPLE_RATE
        ? new StreamingLinearResampler(sampleRate, VAD_SAMPLE_RATE)
        : null;
    this.#detector = new HysteresisDetector(detector, VAD_HOP_SAMPLES / VAD_SAMPLE_RATE);
    this.#state = freshState();
  }

  /**
   * @param {object} options
   * @param {object} options.ort   the onnxruntime-web module (`await import("onnxruntime-web/wasm")`)
   * @param {string|ArrayBuffer|Uint8Array} options.model  URL or bytes of flashvad-stream.onnx
   * @param {number} [options.sampleRate]  input rate; resampled to 16 kHz when it differs
   * @param {object} [options.detector]    detector overrides, see DEFAULT_DETECTOR_CONFIG
   */
  static async create({ ort, model, sampleRate, detector, sessionOptions } = {}) {
    if (!ort) throw new TypeError("pass the onnxruntime-web module as `ort`");
    if (!model) throw new TypeError("pass the model URL or bytes as `model`");
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      ...sessionOptions,
    });
    return new FlashVad({ ort, session, sampleRate, detector });
  }

  /** Drop all per-call state. The session and its weights are untouched. */
  reset() {
    this.#features.reset();
    this.#resampler?.reset();
    this.#detector.reset();
    this.#state = freshState();
  }

  /**
   * Push arbitrary-length mono float32 audio.
   *
   * Samples that do not complete a 10 ms hop are retained until the next call,
   * so callers may push whatever chunk size their capture path produces.
   *
   * @param {Float32Array} samples
   * @returns {Promise<{probabilities: number[], events: object[]}>}
   */
  async push(samples) {
    const audio = this.#resampler ? this.#resampler.push(samples) : samples;
    const frames = this.#features.push(audio);
    const probabilities = [];
    const events = [];

    for (const feature of frames) {
      const probability = await this.#step(feature);
      probabilities.push(probability);
      events.push(...this.#detector.update(probability));
    }
    return { probabilities, events };
  }

  async #step(feature) {
    const { Tensor } = this.#ort;
    const inputs = {
      feature: new Tensor("float32", feature, [1, 1, 43]),
      recurrent: new Tensor("float32", this.#state.recurrent, [1, 1, RECURRENT_DIM]),
    };
    CACHE_WIDTHS.forEach((width, index) => {
      inputs[`cache_${index}`] = new Tensor("float32", this.#state.caches[index], [
        1,
        RECURRENT_DIM,
        width,
      ]);
    });

    const output = await this.#session.run(inputs);
    // Copy: ORT may reuse the backing buffers on the next run.
    this.#state = {
      recurrent: Float32Array.from(output.next_recurrent.data),
      caches: CACHE_WIDTHS.map((_, index) =>
        Float32Array.from(output[`next_cache_${index}`].data),
      ),
    };
    return sigmoid(Number(output.speech_logits.data[0]));
  }
}
