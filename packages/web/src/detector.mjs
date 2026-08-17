/**
 * Hysteresis speech detector.
 *
 * Port of `src/flashvad/detector.py`. The two implementations must stay
 * frame-for-frame identical: a browser client and a Python server analysing the
 * same call are expected to emit the same events at the same frame indices.
 *
 * Defaults are the calibrated operating point from
 * `models/flashvad-v0.1/detector-calibration.json`, selected on the validation
 * split under a 10% false-alarm constraint. They are not the dataclass defaults
 * in config.py, which are pre-calibration values.
 */

export const DEFAULT_DETECTOR_CONFIG = Object.freeze({
  startThreshold: 0.8,
  stopThreshold: 0.5,
  startFrames: 3,
  stopFrames: 4,
  preRollFrames: 3,
});

export function validateDetectorConfig(config) {
  const { startThreshold, stopThreshold, startFrames, stopFrames, preRollFrames } = config;
  if (!(stopThreshold > 0 && stopThreshold < startThreshold && startThreshold < 1)) {
    throw new RangeError("expected 0 < stopThreshold < startThreshold < 1");
  }
  if (startFrames < 1 || stopFrames < 1 || preRollFrames < 0) {
    throw new RangeError("frame counts must be non-negative");
  }
  if (![startFrames, stopFrames, preRollFrames].every(Number.isInteger)) {
    throw new TypeError("frame counts must be integers");
  }
}

export class HysteresisDetector {
  /**
   * @param {object} [config] partial override of DEFAULT_DETECTOR_CONFIG
   * @param {number} [hopSeconds] seconds of audio per probability (10 ms)
   */
  constructor(config = {}, hopSeconds = 0.01) {
    this.config = { ...DEFAULT_DETECTOR_CONFIG, ...config };
    validateDetectorConfig(this.config);
    this.hopSeconds = hopSeconds;
    this.reset();
  }

  reset() {
    this.active = false;
    this.frameIndex = -1;
    this.aboveCount = 0;
    this.belowCount = 0;
  }

  /**
   * Feed one speech probability and return any events it produced.
   *
   * A speech_start is back-dated by preRollFrames so the reported frame sits
   * before the onset the detector needed several frames to confirm. Callers
   * that act on the event (sending audio to STT, say) therefore need a short
   * ring buffer of past audio; acting only from the event's arrival time will
   * clip the start of the word.
   *
   * @param {number} probability in [0, 1]
   * @returns {Array<{kind: string, frame: number, timeSeconds: number, probability: number}>}
   */
  update(probability) {
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new RangeError("probability must be finite and between 0 and 1");
    }
    this.frameIndex += 1;
    const events = [];
    const { startThreshold, stopThreshold, startFrames, stopFrames, preRollFrames } =
      this.config;

    if (!this.active) {
      this.aboveCount = probability >= startThreshold ? this.aboveCount + 1 : 0;
      if (this.aboveCount >= startFrames) {
        const frame = Math.max(0, this.frameIndex - startFrames + 1 - preRollFrames);
        this.active = true;
        this.belowCount = 0;
        events.push({
          kind: "speech_start",
          frame,
          timeSeconds: frame * this.hopSeconds,
          probability,
        });
      }
    } else {
      this.belowCount = probability <= stopThreshold ? this.belowCount + 1 : 0;
      if (this.belowCount >= stopFrames) {
        const frame = this.frameIndex - stopFrames + 1;
        this.active = false;
        this.aboveCount = 0;
        events.push({
          kind: "speech_end",
          frame,
          timeSeconds: frame * this.hopSeconds,
          probability,
        });
      }
    }
    return events;
  }
}
