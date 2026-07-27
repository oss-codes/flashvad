"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web/wasm";
import wasmMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url&no-inline";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url&no-inline";
import {
  StreamingFeatureBuffer,
  StreamingLinearResampler,
  resampleLinear,
} from "../lib/vad-features.mjs";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
  wasm: wasmUrl,
  mjs: wasmMjsUrl,
};

const publicAsset = (path: string) => `${import.meta.env.BASE_URL}${path}`;

let wasmBinaryPromise: Promise<ArrayBuffer> | null = null;

async function loadWasmBinary() {
  if (!wasmBinaryPromise) {
    wasmBinaryPromise = fetch(wasmUrl).then((response) => {
      if (!response.ok) {
        throw new Error(`The browser runtime could not load (${response.status}).`);
      }
      return response.arrayBuffer();
    });
  }
  return wasmBinaryPromise;
}

type Phase =
  | "idle"
  | "loading"
  | "listening"
  | "analyzing"
  | "complete"
  | "error";

type FrameResult = {
  probability: number;
  latencyMs: number;
};

class BrowserVad {
  private session: ort.InferenceSession | null = null;
  private features = new StreamingFeatureBuffer();
  private recurrent = new Float32Array(64);
  private caches = [
    new Float32Array(64 * 2),
    new Float32Array(64 * 4),
    new Float32Array(64 * 8),
    new Float32Array(64 * 16),
  ];

  async load() {
    if (this.session) return;
    ort.env.wasm.wasmBinary = await loadWasmBinary();
    this.session = await ort.InferenceSession.create(
      publicAsset("models/flashvad-stream.onnx"),
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      },
    );
  }

  reset() {
    this.features.reset();
    this.recurrent = new Float32Array(64);
    this.caches = [
      new Float32Array(64 * 2),
      new Float32Array(64 * 4),
      new Float32Array(64 * 8),
      new Float32Array(64 * 16),
    ];
  }

  extract(samples: Float32Array) {
    return this.features.push(samples);
  }

  async run(
    feature: Float32Array,
    frontendLatencyMs = 0,
  ): Promise<FrameResult> {
    if (!this.session) {
      throw new Error("The browser model is not loaded.");
    }

    const started = performance.now();
    const output = await this.session.run({
      feature: new ort.Tensor("float32", feature, [1, 1, 43]),
      recurrent: new ort.Tensor("float32", this.recurrent, [1, 1, 64]),
      cache_0: new ort.Tensor("float32", this.caches[0], [1, 64, 2]),
      cache_1: new ort.Tensor("float32", this.caches[1], [1, 64, 4]),
      cache_2: new ort.Tensor("float32", this.caches[2], [1, 64, 8]),
      cache_3: new ort.Tensor("float32", this.caches[3], [1, 64, 16]),
    });
    const latencyMs = frontendLatencyMs + performance.now() - started;
    const logit = Number(output.speech_logits.data[0]);

    this.recurrent = Float32Array.from(
      output.next_recurrent.data as Float32Array,
    );
    this.caches = [
      Float32Array.from(output.next_cache_0.data as Float32Array),
      Float32Array.from(output.next_cache_1.data as Float32Array),
      Float32Array.from(output.next_cache_2.data as Float32Array),
      Float32Array.from(output.next_cache_3.data as Float32Array),
    ];

    const exponential = Math.exp(-Math.abs(logit));
    const probability =
      logit >= 0 ? 1 / (1 + exponential) : exponential / (1 + exponential);
    return { probability, latencyMs };
  }
}

const MAX_FILE_SECONDS = 30;
const TIMELINE_SIZE = 84;

export function VadPlayground() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [probability, setProbability] = useState(0);
  const [speech, setSpeech] = useState(false);
  const [threshold, setThreshold] = useState(0.8);
  const [timeline, setTimeline] = useState<number[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [speechMs, setSpeechMs] = useState(0);
  const [medianLatency, setMedianLatency] = useState(0);
  const [fileName, setFileName] = useState("");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");

  const engineRef = useRef<BrowserVad | null>(null);
  const timelineRef = useRef<number[]>([]);
  const latencyRef = useRef<number[]>([]);
  const framesRef = useRef(0);
  const speechFramesRef = useRef(0);
  const speechRef = useRef(false);
  const startCountRef = useRef(0);
  const stopCountRef = useRef(0);
  const thresholdRef = useRef(threshold);
  const queueRef = useRef(Promise.resolve());
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);

  useEffect(() => {
    thresholdRef.current = threshold;
  }, [threshold]);

  const stopMicrophone = useCallback(async () => {
    workletRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    workletRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;

    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context && context.state !== "closed") {
      await context.close();
    }
  }, []);

  useEffect(
    () => () => {
      void stopMicrophone();
    },
    [stopMicrophone],
  );

  const resetRun = useCallback(() => {
    engineRef.current?.reset();
    timelineRef.current = [];
    latencyRef.current = [];
    framesRef.current = 0;
    speechFramesRef.current = 0;
    speechRef.current = false;
    startCountRef.current = 0;
    stopCountRef.current = 0;
    setProbability(0);
    setSpeech(false);
    setTimeline([]);
    setElapsedMs(0);
    setSpeechMs(0);
    setMedianLatency(0);
    setProgress(0);
    setError("");
  }, []);

  const ensureEngine = useCallback(async () => {
    if (!engineRef.current) {
      engineRef.current = new BrowserVad();
    }
    await engineRef.current.load();
  }, []);

  const commitFrame = useCallback((result: FrameResult, force = false) => {
    framesRef.current += 1;
    const startThreshold = thresholdRef.current;
    const stopThreshold = Math.max(0.05, startThreshold - 0.3);

    if (!speechRef.current) {
      startCountRef.current =
        result.probability >= startThreshold ? startCountRef.current + 1 : 0;
      if (startCountRef.current >= 3) {
        speechRef.current = true;
        stopCountRef.current = 0;
      }
    } else {
      stopCountRef.current =
        result.probability < stopThreshold ? stopCountRef.current + 1 : 0;
      if (stopCountRef.current >= 4) {
        speechRef.current = false;
        startCountRef.current = 0;
      }
    }

    if (speechRef.current) {
      speechFramesRef.current += 1;
    }

    timelineRef.current.push(result.probability);
    if (timelineRef.current.length > TIMELINE_SIZE) {
      timelineRef.current.shift();
    }
    latencyRef.current.push(result.latencyMs);
    if (latencyRef.current.length > 120) {
      latencyRef.current.shift();
    }

    if (force || framesRef.current % 4 === 0) {
      const ordered = [...latencyRef.current].sort((left, right) => left - right);
      const median = ordered[Math.floor(ordered.length / 2)] ?? 0;
      setProbability(result.probability);
      setSpeech(speechRef.current);
      setTimeline([...timelineRef.current]);
      setElapsedMs(framesRef.current * 10);
      setSpeechMs(speechFramesRef.current * 10);
      setMedianLatency(median);
    }
  }, []);

  const processSamples = useCallback(
    async (samples: Float32Array, updateProgress?: (frames: number) => void) => {
      const engine = engineRef.current;
      if (!engine) return;
      const featureStarted = performance.now();
      const features = engine.extract(samples);
      const frontendLatencyMs =
        features.length > 0
          ? (performance.now() - featureStarted) / features.length
          : 0;

      for (let index = 0; index < features.length; index += 1) {
        const result = await engine.run(features[index], frontendLatencyMs);
        commitFrame(result, index === features.length - 1);
        updateProgress?.(framesRef.current);
        if (framesRef.current % 50 === 0) {
          await new Promise<void>((resolve) =>
            requestAnimationFrame(() => resolve()),
          );
        }
      }
    },
    [commitFrame],
  );

  const startMicrophone = useCallback(async () => {
    try {
      await stopMicrophone();
      setPhase("loading");
      resetRun();
      setFileName("");
      await ensureEngine();

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Microphone capture is unavailable in this browser.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });
      const context = new AudioContext();
      await context.audioWorklet.addModule(publicAsset("vad-audio-processor.js"));
      const source = context.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(context, "vad-audio-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
      });
      const resampler = new StreamingLinearResampler(context.sampleRate);

      worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
        const samples = resampler.push(event.data);
        queueRef.current = queueRef.current
          .then(() => processSamples(samples))
          .catch((reason: unknown) => {
            setError(
              reason instanceof Error
                ? reason.message
                : "Microphone analysis stopped unexpectedly.",
            );
            setPhase("error");
          });
      };

      streamRef.current = stream;
      audioContextRef.current = context;
      sourceRef.current = source;
      workletRef.current = worklet;
      source.connect(worklet);
      setPhase("listening");
    } catch (reason) {
      await stopMicrophone();
      setError(
        reason instanceof Error
          ? reason.message
          : "The microphone could not be started.",
      );
      setPhase("error");
    }
  }, [ensureEngine, processSamples, resetRun, stopMicrophone]);

  const finishMicrophone = useCallback(async () => {
    await stopMicrophone();
    await queueRef.current;
    setPhase(framesRef.current > 0 ? "complete" : "idle");
  }, [stopMicrophone]);

  const analyzeFile = useCallback(
    async (file: File) => {
      try {
        await stopMicrophone();
        setPhase("loading");
        resetRun();
        setFileName(file.name);
        await ensureEngine();

        const context = new AudioContext();
        const decoded = await context.decodeAudioData(await file.arrayBuffer());
        const sampleLimit = Math.min(
          decoded.length,
          Math.floor(decoded.sampleRate * MAX_FILE_SECONDS),
        );
        const mono = new Float32Array(sampleLimit);
        for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
          const source = decoded.getChannelData(channel);
          for (let index = 0; index < sampleLimit; index += 1) {
            mono[index] += source[index] / decoded.numberOfChannels;
          }
        }
        await context.close();

        const samples = resampleLinear(mono, decoded.sampleRate);
        const expectedFrames = Math.max(1, Math.floor(samples.length / 160));
        setPhase("analyzing");

        for (let offset = 0; offset < samples.length; offset += 4096) {
          await processSamples(
            samples.subarray(offset, offset + 4096),
            (frames) => setProgress(Math.min(1, frames / expectedFrames)),
          );
        }
        setProgress(1);
        setPhase("complete");
      } catch (reason) {
        setError(
          reason instanceof Error
            ? reason.message
            : "This audio file could not be analyzed.",
        );
        setPhase("error");
      }
    },
    [ensureEngine, processSamples, resetRun, stopMicrophone],
  );

  const statusLabel =
    phase === "listening"
      ? speech
        ? "Speech"
        : "Listening"
      : phase === "analyzing"
        ? "Analyzing"
        : phase === "loading"
          ? "Loading model"
          : phase === "complete"
            ? speech
              ? "Ended in speech"
              : "Analysis complete"
            : phase === "error"
              ? "Test unavailable"
              : "Ready";

  const busy = phase === "loading" || phase === "analyzing";

  return (
    <section className="section playground-section" id="try-it">
      <div className="section-heading">
        <div>
          <p className="kicker">Interactive model test</p>
          <h2>Speak—or drop in a call sample.</h2>
        </div>
        <p>
          Audio stays in your browser. The demo runs the current 46K-parameter
          research checkpoint with the same causal features and stateful model.
        </p>
      </div>

      <div className="playground-shell">
        <div className="playground-controls">
          <div className="control-intro">
            <span className="research-chip">Research checkpoint</span>
            <h3>Live speech probability</h3>
            <p>
              This checkpoint combines nine FLEURS language configurations, AMI
              conversations and MUSAN noise with teacher-guided labels. Try
              different voices and noisy recordings, but do not treat its output
              as a production accuracy result.
            </p>
          </div>

          <div className="test-actions">
            {phase === "listening" ? (
              <button className="test-button stop" onClick={finishMicrophone}>
                <span className="button-dot" aria-hidden="true" />
                Stop microphone
              </button>
            ) : (
              <button
                className="test-button mic"
                onClick={startMicrophone}
                disabled={busy}
              >
                <span className="button-dot" aria-hidden="true" />
                Test my microphone
              </button>
            )}

            <label className={`test-button upload ${busy ? "disabled" : ""}`}>
              <span aria-hidden="true">↑</span>
              Analyze audio file
              <input
                type="file"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
                disabled={busy || phase === "listening"}
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  if (file) void analyzeFile(file);
                  event.currentTarget.value = "";
                }}
              />
            </label>
          </div>

          <label className="threshold-control">
            <span>
              Start sensitivity
              <strong>{threshold.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.2"
              max="0.9"
              step="0.01"
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
              aria-label="Speech start threshold"
            />
            <small>
              Lower values detect softer speech but increase false triggers.
            </small>
          </label>

          <p className="privacy-note">
            Microphone access starts only after you press the button. No audio,
            probabilities or filenames are uploaded.
          </p>
        </div>

        <div className={`playground-monitor ${speech ? "is-speech" : ""}`}>
          <div className="monitor-header">
            <div className="monitor-status" aria-live="polite">
              <span className="monitor-light" aria-hidden="true" />
              {statusLabel}
            </div>
            <span>ONNX / WASM</span>
          </div>

          <div className="probability-readout">
            <span>speech probability</span>
            <strong>{probability.toFixed(3)}</strong>
          </div>

          <div
            className="probability-meter"
            role="meter"
            aria-label="Current speech probability"
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={probability}
          >
            <span style={{ width: `${probability * 100}%` }} />
            <i style={{ left: `${threshold * 100}%` }} aria-hidden="true" />
          </div>

          <div className="live-timeline" aria-label="Recent speech probabilities">
            {Array.from({ length: TIMELINE_SIZE }, (_, index) => {
              const value =
                timeline[index - (TIMELINE_SIZE - timeline.length)] ?? 0;
              return (
                <span
                  key={index}
                  className={value >= threshold ? "above" : ""}
                  style={{ height: `${Math.max(3, value * 100)}%` }}
                />
              );
            })}
          </div>

          {phase === "analyzing" && (
            <div className="file-progress" aria-label="File analysis progress">
              <span style={{ width: `${progress * 100}%` }} />
            </div>
          )}

          <div className="monitor-stats">
            <div>
              <span>audio tested</span>
              <strong>{(elapsedMs / 1000).toFixed(1)} s</strong>
            </div>
            <div>
              <span>speech detected</span>
              <strong>{(speechMs / 1000).toFixed(1)} s</strong>
            </div>
            <div>
              <span>full-hop median</span>
              <strong>
                {medianLatency ? `${medianLatency.toFixed(2)} ms` : "—"}
              </strong>
            </div>
          </div>

          {fileName && (
            <p className="file-name" title={fileName}>
              FILE / {fileName}
              {elapsedMs >= MAX_FILE_SECONDS * 1000
                ? ` / first ${MAX_FILE_SECONDS}s`
                : ""}
            </p>
          )}
          {error && <p className="test-error">{error}</p>}
        </div>
      </div>

      <p className="playground-caveat">
        Browser timing covers causal feature extraction plus the ONNX
        WebAssembly model step; it is not the native Accelerate benchmark. This
        tool demonstrates behavior, while the public cross-domain result remains
        exploratory evidence rather than a release test.
      </p>
    </section>
  );
}
