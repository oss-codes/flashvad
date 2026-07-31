"use client";

import {
  type CSSProperties,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import wasmMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url&no-inline";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url&no-inline";
import {
  ANALYSIS_HOP_MS,
  detectSpeechFrames,
  formatAnalysisTime,
  speakingStateAtTime,
  summarizeTrack,
} from "../lib/call-analysis.mjs";
import { createRetryableLoader } from "../lib/retryable-loader.mjs";
import {
  StreamingFeatureBuffer,
  StreamingLinearResampler,
  resampleLinear,
} from "../lib/vad-features.mjs";

type BrowserOrt = typeof import("onnxruntime-web/wasm");
type BrowserOrtSession = Awaited<
  ReturnType<BrowserOrt["InferenceSession"]["create"]>
>;

const publicAsset = (path: string) => `${import.meta.env.BASE_URL}${path}`;

const browserRuntimeLoader = createRetryableLoader(async () => {
  const runtime = await import("onnxruntime-web/wasm");
  runtime.env.wasm.numThreads = 1;
  runtime.env.wasm.wasmPaths = {
    wasm: wasmUrl,
    mjs: wasmMjsUrl,
  };
  return runtime;
});

const wasmBinaryLoader = createRetryableLoader(async () => {
  const response = await fetch(wasmUrl);
  if (!response.ok) {
    throw new Error(`The browser runtime could not load (${response.status}).`);
  }
  return response.arrayBuffer();
});

async function loadWasmBinary() {
  return wasmBinaryLoader.load();
}

const browserSessionLoader = createRetryableLoader(async () => {
  const runtime = await browserRuntimeLoader.load();
  runtime.env.wasm.wasmBinary = await loadWasmBinary();
  const session = await runtime.InferenceSession.create(
    publicAsset("models/flashvad-stream.onnx"),
    {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    },
  );
  return { runtime, session };
});

async function loadBrowserSession() {
  return browserSessionLoader.load();
}

function resetBrowserRuntime() {
  wasmBinaryLoader.reset();
  browserRuntimeLoader.reset();
  browserSessionLoader.reset();
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

type FileTrackId = "mono" | "left" | "right";
type SpeakerRole = "mixed" | "user" | "ai";

type FileProbabilityTrack = {
  id: FileTrackId;
  probabilities: number[];
};

type FileAnalysis = {
  channelCount: number;
  durationMs: number;
  tracks: FileProbabilityTrack[];
};

class BrowserVad {
  private runtime: BrowserOrt | null = null;
  private session: BrowserOrtSession | null = null;
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
    const loaded = await loadBrowserSession();
    this.runtime = loaded.runtime;
    this.session = loaded.session;
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
    if (!this.runtime || !this.session) {
      throw new Error("The browser model is not loaded.");
    }

    const started = performance.now();
    const output = await this.session.run({
      feature: new this.runtime.Tensor("float32", feature, [1, 1, 43]),
      recurrent: new this.runtime.Tensor("float32", this.recurrent, [1, 1, 64]),
      cache_0: new this.runtime.Tensor("float32", this.caches[0], [1, 64, 2]),
      cache_1: new this.runtime.Tensor("float32", this.caches[1], [1, 64, 4]),
      cache_2: new this.runtime.Tensor("float32", this.caches[2], [1, 64, 8]),
      cache_3: new this.runtime.Tensor("float32", this.caches[3], [1, 64, 16]),
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
const ANALYSIS_TIMELINE_BINS = 160;

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
  const [audioUrl, setAudioUrl] = useState("");
  const [fileAnalysis, setFileAnalysis] = useState<FileAnalysis | null>(null);
  const [playheadMs, setPlayheadMs] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [leftChannelRole, setLeftChannelRole] =
    useState<Exclude<SpeakerRole, "mixed">>("user");

  const engineRef = useRef<BrowserVad | null>(null);
  const secondaryEngineRef = useRef<BrowserVad | null>(null);
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
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef("");

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

  useEffect(() => {
    return () => {
      void stopMicrophone();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, [stopMicrophone]);

  const clearFilePlayback = useCallback(() => {
    const audio = audioElementRef.current;
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = "";
    }
    setAudioUrl("");
    setFileAnalysis(null);
    setFileName("");
    setPlayheadMs(0);
    setIsPlaying(false);
  }, []);

  const resetRun = useCallback(() => {
    engineRef.current?.reset();
    secondaryEngineRef.current?.reset();
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

  const retryBrowserRuntime = useCallback(async () => {
    await stopMicrophone();
    queueRef.current = Promise.resolve();
    resetBrowserRuntime();
    engineRef.current = null;
    secondaryEngineRef.current = null;
    resetRun();
    setPhase("idle");
  }, [resetRun, stopMicrophone]);

  const ensureEngine = useCallback(async () => {
    if (!engineRef.current) {
      engineRef.current = new BrowserVad();
    }
    await engineRef.current.load();
  }, []);

  const ensureSecondaryEngine = useCallback(async () => {
    if (!secondaryEngineRef.current) {
      secondaryEngineRef.current = new BrowserVad();
    }
    await secondaryEngineRef.current.load();
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
      clearFilePlayback();
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
  }, [
    clearFilePlayback,
    ensureEngine,
    processSamples,
    resetRun,
    stopMicrophone,
  ]);

  const finishMicrophone = useCallback(async () => {
    await stopMicrophone();
    await queueRef.current;
    setPhase(framesRef.current > 0 ? "complete" : "idle");
  }, [stopMicrophone]);

  const analyzeFile = useCallback(
    async (file: File) => {
      try {
        await stopMicrophone();
        clearFilePlayback();
        setPhase("loading");
        resetRun();
        setFileName(file.name);
        setLeftChannelRole("user");
        await ensureEngine();

        const context = new AudioContext();
        let decoded: AudioBuffer;
        try {
          decoded = await context.decodeAudioData(await file.arrayBuffer());
        } finally {
          await context.close();
        }
        const sampleLimit = Math.min(
          decoded.length,
          Math.floor(decoded.sampleRate * MAX_FILE_SECONDS),
        );
        const sourceTracks: Array<{
          id: FileTrackId;
          samples: Float32Array;
        }> =
          decoded.numberOfChannels >= 2
            ? [
                {
                  id: "left",
                  samples: Float32Array.from(
                    decoded.getChannelData(0).subarray(0, sampleLimit),
                  ),
                },
                {
                  id: "right",
                  samples: Float32Array.from(
                    decoded.getChannelData(1).subarray(0, sampleLimit),
                  ),
                },
              ]
            : [
                {
                  id: "mono",
                  samples: Float32Array.from(
                    decoded.getChannelData(0).subarray(0, sampleLimit),
                  ),
                },
              ];

        if (sourceTracks.length === 2) {
          await ensureSecondaryEngine();
        }
        const preparedTracks = sourceTracks.map((track) => ({
          id: track.id,
          samples: resampleLinear(track.samples, decoded.sampleRate),
        }));
        const expectedFrames = preparedTracks.reduce(
          (total, track) =>
            total + Math.max(1, Math.floor(track.samples.length / 160)),
          0,
        );
        let processedFrames = 0;
        const probabilityTracks: FileProbabilityTrack[] = [];
        setPhase("analyzing");

        for (
          let trackIndex = 0;
          trackIndex < preparedTracks.length;
          trackIndex += 1
        ) {
          const track = preparedTracks[trackIndex];
          const engine =
            trackIndex === 0
              ? engineRef.current
              : secondaryEngineRef.current;
          if (!engine) throw new Error("The browser model is not loaded.");
          engine.reset();
          const probabilities: number[] = [];

          for (let offset = 0; offset < track.samples.length; offset += 4096) {
            const featureStarted = performance.now();
            const features = engine.extract(
              track.samples.subarray(offset, offset + 4096),
            );
            const frontendLatencyMs =
              features.length > 0
                ? (performance.now() - featureStarted) / features.length
                : 0;

            for (let index = 0; index < features.length; index += 1) {
              const result = await engine.run(
                features[index],
                frontendLatencyMs,
              );
              probabilities.push(result.probability);
              if (trackIndex === 0) {
                commitFrame(result, index === features.length - 1);
              }
              processedFrames += 1;
              if (processedFrames % 8 === 0) {
                setProgress(Math.min(1, processedFrames / expectedFrames));
              }
              if (processedFrames % 50 === 0) {
                await new Promise<void>((resolve) =>
                  requestAnimationFrame(() => resolve()),
                );
              }
            }
          }
          probabilityTracks.push({ id: track.id, probabilities });
        }

        const durationMs =
          Math.min(
            ...probabilityTracks.map((track) => track.probabilities.length),
          ) * ANALYSIS_HOP_MS;
        if (!Number.isFinite(durationMs) || durationMs <= 0) {
          throw new Error(
            "This recording is too short to produce a 10 ms analysis frame.",
          );
        }
        const nextAudioUrl = URL.createObjectURL(file);
        audioUrlRef.current = nextAudioUrl;
        setAudioUrl(nextAudioUrl);
        setFileAnalysis({
          channelCount: decoded.numberOfChannels,
          durationMs,
          tracks: probabilityTracks,
        });
        setPlayheadMs(0);
        setIsPlaying(false);
        setProgress(1);
        setPhase("complete");
      } catch (reason) {
        clearFilePlayback();
        setError(
          reason instanceof Error
            ? reason.message
            : "This audio file could not be analyzed.",
        );
        setPhase("error");
      }
    },
    [
      clearFilePlayback,
      commitFrame,
      ensureEngine,
      ensureSecondaryEngine,
      resetRun,
      stopMicrophone,
    ],
  );

  const analyzedTracks = useMemo(() => {
    if (!fileAnalysis) return [];
    return fileAnalysis.tracks.map((track) => {
      const role: SpeakerRole =
        track.id === "mono"
          ? "mixed"
          : track.id === "left"
            ? leftChannelRole
            : leftChannelRole === "user"
              ? "ai"
              : "user";
      const speechFrames = detectSpeechFrames(
        track.probabilities,
        threshold,
      );
      return {
        ...track,
        bins: summarizeTrack(
          track.probabilities,
          speechFrames,
          ANALYSIS_TIMELINE_BINS,
        ),
        label:
          role === "mixed"
            ? "Mixed speech"
            : role === "user"
              ? "User"
              : "AI agent",
        role,
        speechFrames,
        speechMs:
          speechFrames.filter(Boolean).length * ANALYSIS_HOP_MS,
      };
    });
  }, [fileAnalysis, leftChannelRole, threshold]);

  const speakingState = useMemo(
    () => speakingStateAtTime(analyzedTracks, playheadMs),
    [analyzedTracks, playheadMs],
  );

  const speakingStateLabel =
    speakingState === "overlap"
      ? "User + AI overlap"
      : speakingState === "user"
        ? "User speaking"
        : speakingState === "ai"
          ? "AI speaking"
          : speakingState === "mixed"
            ? "Speech detected"
            : "Silence";

  const seekPlayback = useCallback(
    (milliseconds: number) => {
      if (!fileAnalysis) return;
      const bounded = Math.max(
        0,
        Math.min(milliseconds, fileAnalysis.durationMs),
      );
      setPlayheadMs(bounded);
      if (audioElementRef.current) {
        audioElementRef.current.currentTime = bounded / 1_000;
      }
    },
    [fileAnalysis],
  );

  const togglePlayback = useCallback(async () => {
    const audio = audioElementRef.current;
    if (!audio || !fileAnalysis) return;
    try {
      if (audio.paused) {
        if (audio.currentTime * 1_000 >= fileAnalysis.durationMs - 20) {
          seekPlayback(0);
        }
        await audio.play();
      } else {
        audio.pause();
      }
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Audio playback could not start.",
      );
    }
  }, [fileAnalysis, seekPlayback]);

  const updatePlaybackPosition = useCallback(() => {
    const audio = audioElementRef.current;
    if (!audio || !fileAnalysis) return;
    const nextMs = audio.currentTime * 1_000;
    if (nextMs >= fileAnalysis.durationMs) {
      audio.pause();
      audio.currentTime = fileAnalysis.durationMs / 1_000;
      setPlayheadMs(fileAnalysis.durationMs);
      setIsPlaying(false);
      return;
    }
    setPlayheadMs(nextMs);
  }, [fileAnalysis]);

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
    <section
      className="section playground-section"
      id="try-it"
      data-clarity-mask="true"
    >
      <div className="section-heading">
        <div>
          <p className="kicker">Interactive model test</p>
          <h2>Play a call with its speech analysis.</h2>
        </div>
        <p>
          Upload a mono recording for speech/silence review, or a stereo call
          with the user on one channel and the AI on the other for synchronized
          speaker tracks. Audio never leaves your browser.
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
              Analyze and play audio
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
            probabilities or filenames are uploaded. This playground is masked
            from site analytics.
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
                {medianLatency ? `${medianLatency.toFixed(2)} ms` : "N/A"}
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
          {error && (
            <div className="test-error" role="alert">
              <p>{error}</p>
              <small>
                Runtime: ONNX Runtime Web / WASM, one worker thread; audio stays
                in this browser.
              </small>
              <button
                type="button"
                onClick={() => void retryBrowserRuntime()}
              >
                Retry browser runtime
              </button>
            </div>
          )}
        </div>
      </div>

      {fileAnalysis && audioUrl && (
        <section className="call-review" aria-label="Synchronized call analysis">
          <div className="call-review-heading">
            <div>
              <p className="kicker">Synchronized playback</p>
              <h3>Hear the call and follow each speech track.</h3>
            </div>
            <div
              className={`speaker-now speaker-${speakingState}`}
              aria-live="polite"
            >
              <span>At playhead</span>
              <strong>{speakingStateLabel}</strong>
            </div>
          </div>

          <audio
            ref={audioElementRef}
            src={audioUrl}
            preload="metadata"
            aria-label="Uploaded call playback"
            onTimeUpdate={updatePlaybackPosition}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
          />

          <div className="playback-controls">
            <button type="button" onClick={() => void togglePlayback()}>
              <span aria-hidden="true">{isPlaying ? "Ⅱ" : "▶"}</span>
              {isPlaying ? "Pause" : "Play analysis"}
            </button>
            <span>{formatAnalysisTime(playheadMs)}</span>
            <input
              type="range"
              min="0"
              max={fileAnalysis.durationMs}
              step={ANALYSIS_HOP_MS}
              value={Math.min(playheadMs, fileAnalysis.durationMs)}
              onChange={(event) => seekPlayback(Number(event.target.value))}
              aria-label="Analysis playback position"
            />
            <span>{formatAnalysisTime(fileAnalysis.durationMs)}</span>
          </div>

          {fileAnalysis.tracks.length === 2 && (
            <div className="channel-mapping">
              <p>
                Stereo mapping:{" "}
                <strong>
                  left = {leftChannelRole === "user" ? "user" : "AI"}, right ={" "}
                  {leftChannelRole === "user" ? "AI" : "user"}
                </strong>
              </p>
              <button
                type="button"
                onClick={() =>
                  setLeftChannelRole((current) =>
                    current === "user" ? "ai" : "user",
                  )
                }
              >
                Swap user and AI channels
              </button>
            </div>
          )}

          <div className="speaker-tracks">
            {analyzedTracks.map((track) => (
              <div
                className={`speaker-track track-${track.role}`}
                key={track.id}
              >
                <div className="speaker-track-label">
                  <strong>{track.label}</strong>
                  <span>
                    {(track.speechMs / 1_000).toFixed(1)} s speech
                    {track.id !== "mono"
                      ? ` · ${track.id} channel`
                      : " · mono/mixed"}
                  </span>
                </div>
                <div className="speaker-track-plot">
                  <i
                    className="analysis-playhead"
                    style={{
                      left: `${Math.min(
                        100,
                        (playheadMs / fileAnalysis.durationMs) * 100,
                      )}%`,
                    }}
                    aria-hidden="true"
                  />
                  <div className="speaker-track-cells">
                    {track.bins.map((bin) => (
                      <button
                        type="button"
                        key={`${track.id}-${bin.startFrame}`}
                        className={bin.active ? "active" : ""}
                        style={
                          {
                            "--track-level": Math.max(0.06, bin.peak),
                          } as CSSProperties
                        }
                        onClick={() =>
                          seekPlayback(bin.startFrame * ANALYSIS_HOP_MS)
                        }
                        aria-label={`Seek to ${formatAnalysisTime(
                          bin.startFrame * ANALYSIS_HOP_MS,
                        )}; ${track.label} ${
                          bin.active ? "speaking" : "not speaking"
                        }`}
                        title={`${formatAnalysisTime(
                          bin.startFrame * ANALYSIS_HOP_MS,
                        )} · peak ${bin.peak.toFixed(2)}`}
                      />
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <p className="call-review-note">
            {fileAnalysis.tracks.length === 2
              ? "Speaker labels come from channel assignment, not voice identification. Use Swap if your recorder stores the channels in the opposite order."
              : "A mono or mixed recording contains no reliable user/AI identity. FlashVAD marks speech and silence; speaker diarization requires a separate model or isolated channels."}
            {fileAnalysis.channelCount > 2
              ? " Only the first two channels were analyzed."
              : ""}
          </p>
        </section>
      )}

      <p className="playground-caveat">
        Browser timing covers causal feature extraction plus the ONNX
        WebAssembly model step; it is not the native Accelerate benchmark. This
        tool demonstrates behavior, while the public cross-domain result remains
        exploratory evidence rather than a release test.
      </p>
    </section>
  );
}
