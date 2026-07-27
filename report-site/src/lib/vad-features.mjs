export const VAD_SAMPLE_RATE = 16_000;
export const VAD_FRAME_SAMPLES = 400;
export const VAD_HOP_SAMPLES = 160;
export const VAD_FEATURE_DIM = 43;

const FFT_SIZE = 512;
const MEL_BANDS = 40;
const MEL_MIN_HZ = 50;
const MEL_MAX_HZ = 7_600;
const EPSILON = 1e-10;

const window = Float64Array.from(
  { length: VAD_FRAME_SAMPLES },
  (_, index) =>
    0.5 - 0.5 * Math.cos((2 * Math.PI * index) / VAD_FRAME_SAMPLES),
);

function hzToMel(frequency) {
  return 2595 * Math.log10(1 + frequency / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilterbank() {
  const minimum = hzToMel(MEL_MIN_HZ);
  const maximum = hzToMel(MEL_MAX_HZ);
  const points = Array.from({ length: MEL_BANDS + 2 }, (_, index) =>
    melToHz(minimum + ((maximum - minimum) * index) / (MEL_BANDS + 1)),
  );

  return Array.from({ length: MEL_BANDS }, (_, band) => {
    const lower = points[band];
    const center = points[band + 1];
    const upper = points[band + 2];
    const normalization = 2 / Math.max(upper - lower, 1e-8);

    const dense = Float64Array.from({ length: FFT_SIZE / 2 + 1 }, (_, bin) => {
      const frequency = (bin * VAD_SAMPLE_RATE) / FFT_SIZE;
      const rising = (frequency - lower) / Math.max(center - lower, 1e-8);
      const falling = (upper - frequency) / Math.max(upper - center, 1e-8);
      return Math.max(0, Math.min(rising, falling)) * normalization;
    });
    let start = 0;
    while (start < dense.length && dense[start] === 0) start += 1;
    let end = dense.length;
    while (end > start && dense[end - 1] === 0) end -= 1;
    return { start, weights: dense.slice(start, end) };
  });
}

const melFilterbank = buildMelFilterbank();

function fftPower(frame) {
  const real = new Float64Array(FFT_SIZE);
  const imaginary = new Float64Array(FFT_SIZE);

  for (let index = 0; index < VAD_FRAME_SAMPLES; index += 1) {
    real[index] = frame[index] * window[index];
  }

  for (let index = 1, reversed = 0; index < FFT_SIZE; index += 1) {
    let bit = FFT_SIZE >> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>= 1;
    }
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed], real[index]];
      [imaginary[index], imaginary[reversed]] = [
        imaginary[reversed],
        imaginary[index],
      ];
    }
  }

  for (let length = 2; length <= FFT_SIZE; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const rootReal = Math.cos(angle);
    const rootImaginary = Math.sin(angle);

    for (let start = 0; start < FFT_SIZE; start += length) {
      let twiddleReal = 1;
      let twiddleImaginary = 0;

      for (let offset = 0; offset < length / 2; offset += 1) {
        const even = start + offset;
        const odd = even + length / 2;
        const oddReal =
          real[odd] * twiddleReal - imaginary[odd] * twiddleImaginary;
        const oddImaginary =
          real[odd] * twiddleImaginary + imaginary[odd] * twiddleReal;

        real[odd] = real[even] - oddReal;
        imaginary[odd] = imaginary[even] - oddImaginary;
        real[even] += oddReal;
        imaginary[even] += oddImaginary;

        const nextReal =
          twiddleReal * rootReal - twiddleImaginary * rootImaginary;
        twiddleImaginary =
          twiddleReal * rootImaginary + twiddleImaginary * rootReal;
        twiddleReal = nextReal;
      }
    }
  }

  return Float64Array.from(
    { length: FFT_SIZE / 2 + 1 },
    (_, bin) => real[bin] ** 2 + imaginary[bin] ** 2,
  );
}

export function extractVadFeatures(frame) {
  if (frame.length !== VAD_FRAME_SAMPLES) {
    throw new RangeError(`Expected ${VAD_FRAME_SAMPLES} audio samples.`);
  }

  const power = fftPower(frame);
  const features = new Float32Array(VAD_FEATURE_DIM);
  let melMean = 0;

  for (let band = 0; band < MEL_BANDS; band += 1) {
    let energy = 0;
    const filter = melFilterbank[band];
    for (let index = 0; index < filter.weights.length; index += 1) {
      energy += power[filter.start + index] * filter.weights[index];
    }
    const value = Math.log(Math.max(energy, 1e-8));
    features[band] = value;
    melMean += value;
  }

  melMean /= MEL_BANDS;
  for (let band = 0; band < MEL_BANDS; band += 1) {
    features[band] -= melMean;
  }

  let squareSum = 0;
  let crossings = 0;
  for (let index = 0; index < frame.length; index += 1) {
    squareSum += frame[index] ** 2;
    if (index > 0 && frame[index] * frame[index - 1] < 0) {
      crossings += 1;
    }
  }
  const rms = Math.sqrt(Math.max(squareSum / frame.length, EPSILON));

  let logPowerSum = 0;
  let powerSum = 0;
  for (const value of power) {
    logPowerSum += Math.log(Math.max(value, EPSILON));
    powerSum += value;
  }
  const geometric = Math.exp(logPowerSum / power.length);
  const arithmetic = Math.max(powerSum / power.length, EPSILON);

  features[40] = Math.log(rms);
  features[41] = Math.log(Math.max(geometric / arithmetic, EPSILON));
  features[42] = crossings / (frame.length - 1);
  return features;
}

export class StreamingFeatureBuffer {
  constructor() {
    this.reset();
  }

  reset() {
    this.history = new Float32Array(
      VAD_FRAME_SAMPLES - VAD_HOP_SAMPLES,
    );
    this.pending = new Float32Array();
  }

  push(samples) {
    const joined = new Float32Array(this.pending.length + samples.length);
    joined.set(this.pending);
    joined.set(samples, this.pending.length);

    const features = [];
    let offset = 0;
    while (joined.length - offset >= VAD_HOP_SAMPLES) {
      const hop = joined.subarray(offset, offset + VAD_HOP_SAMPLES);
      const frame = new Float32Array(VAD_FRAME_SAMPLES);
      frame.set(this.history);
      frame.set(hop, this.history.length);
      features.push(extractVadFeatures(frame));
      this.history = frame.slice(VAD_HOP_SAMPLES);
      offset += VAD_HOP_SAMPLES;
    }

    this.pending = joined.slice(offset);
    return features;
  }
}

export function resampleLinear(samples, sourceRate, targetRate = VAD_SAMPLE_RATE) {
  if (sourceRate === targetRate) {
    return Float32Array.from(samples);
  }
  const targetLength = Math.max(
    1,
    Math.floor((samples.length * targetRate) / sourceRate),
  );
  const output = new Float32Array(targetLength);
  const scale = sourceRate / targetRate;

  for (let index = 0; index < targetLength; index += 1) {
    const position = index * scale;
    const left = Math.min(Math.floor(position), samples.length - 1);
    const right = Math.min(left + 1, samples.length - 1);
    const fraction = position - left;
    output[index] =
      samples[left] + (samples[right] - samples[left]) * fraction;
  }
  return output;
}

export class StreamingLinearResampler {
  constructor(sourceRate, targetRate = VAD_SAMPLE_RATE) {
    this.sourceRate = sourceRate;
    this.targetRate = targetRate;
    this.reset();
  }

  reset() {
    this.buffer = new Float32Array();
    this.position = 0;
  }

  push(samples) {
    if (this.sourceRate === this.targetRate) {
      return Float32Array.from(samples);
    }

    const joined = new Float32Array(this.buffer.length + samples.length);
    joined.set(this.buffer);
    joined.set(samples, this.buffer.length);
    const step = this.sourceRate / this.targetRate;
    const values = [];

    while (this.position + 1 < joined.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      values.push(
        joined[left] + (joined[left + 1] - joined[left]) * fraction,
      );
      this.position += step;
    }

    const consumed = Math.floor(this.position);
    this.buffer = joined.slice(consumed);
    this.position -= consumed;
    return Float32Array.from(values);
  }
}
