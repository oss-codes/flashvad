"use client";

import { useState } from "react";

type ModelId = "flashvad" | "silero" | "ten" | "firered";

type ModelPreview = {
  id: ModelId;
  name: string;
  status: "Live runnable here" | "Reference preview";
  summary: string;
  runtimeLabel: string;
  runtime: string;
  hop: string;
  delivery: string;
  scope: string;
  bestFit: string;
  watch: string;
  license: string;
  source: string;
};

const models: ModelPreview[] = [
  {
    id: "flashvad",
    name: "FlashVAD",
    status: "Live runnable here",
    summary:
      "Original 46K-parameter causal model optimized for small per-call state and an embedded Apple Accelerate runtime.",
    runtimeLabel: "Pinned same-run M4 Pro measurement",
    runtime: "11.33 µs median",
    hop: "10 ms",
    delivery: "261 KiB native · ONNX/WASM fallback",
    scope: "9 training configurations; India/GCC release validation open",
    bestFit: "Low-overhead browser, macOS and PSTN integration research",
    watch: "26.3% false alarms on a repeatedly consulted public set",
    license: "MIT code / CC BY 4.0 model",
    source: "#try-it",
  },
  {
    id: "silero",
    name: "Silero VAD",
    status: "Reference preview",
    summary:
      "A mature stateful ONNX/JIT baseline with broad portability, direct 8 kHz support and a large claimed training-language scope.",
    runtimeLabel: "Pinned same-run M4 Pro measurement",
    runtime: "128.17 µs / 32 ms hop",
    hop: "32 ms",
    delivery: "2.33 MB measured ONNX artifact",
    scope: "Vendor reports 6,000+ languages and varied domains",
    bestFit: "Portable general-purpose and telephony deployments",
    watch: "3.53× FlashVAD compute advantage by audio-normalized RTF; not accuracy",
    license: "MIT",
    source: "https://github.com/snakers4/silero-vad",
  },
  {
    id: "ten",
    name: "TEN VAD",
    status: "Reference preview",
    summary:
      "A compact streaming reference distributed as native libraries and WebAssembly with configurable 10/16 ms hops.",
    runtimeLabel: "Pinned same-run M4 Pro measurement",
    runtime: "161.04 µs / 10 ms hop",
    hop: "10 ms",
    delivery: "745 KB measured macOS framework",
    scope: "16 kHz input; web, mobile and desktop packages",
    bestFit: "Small cross-platform real-time integrations",
    watch: "14.2× FlashVAD compute advantage; license has additional conditions",
    license: "Custom additional conditions",
    source: "https://github.com/TEN-framework/ten-vad",
  },
  {
    id: "firered",
    name: "FireRedVAD",
    status: "Reference preview",
    summary:
      "A DFSMN-based multilingual challenger with streaming and non-streaming VAD plus optional audio-event detection.",
    runtimeLabel: "Pinned same-run M4 Pro measurement",
    runtime: "511.96 µs / 10 ms hop",
    hop: "10 ms",
    delivery: "2.28 MB measured streaming weights",
    scope: "Vendor reports speech, singing and music across 100+ languages",
    bestFit: "Multilingual server inference and teacher supervision",
    watch: "45.2× FlashVAD compute advantage; FireRed frontend was excluded",
    license: "Apache-2.0",
    source: "https://github.com/FireRedTeam/FireRedVAD",
  },
];

export function ModelPreview() {
  const [selected, setSelected] = useState<ModelId>("flashvad");
  const model = models.find((candidate) => candidate.id === selected) ?? models[0];
  const localModel = model.id === "flashvad";

  return (
    <section className="section model-preview-section" id="model-preview">
      <div className="section-heading">
        <div>
          <p className="kicker">Interactive comparison preview</p>
          <h2>Preview the model differences.</h2>
        </div>
        <p>
          Select a model to compare deployment shape, local warm runtime and
          intended fit. Only FlashVAD executes in this browser demo; competitor
          entries are reference previews from official releases.
        </p>
      </div>

      <div className="model-preview-shell">
        <div
          className="model-tabs"
          role="tablist"
          aria-label="VAD model previews"
        >
          {models.map((candidate) => (
            <button
              key={candidate.id}
              id={`model-tab-${candidate.id}`}
              className={candidate.id === selected ? "active" : ""}
              type="button"
              role="tab"
              aria-selected={candidate.id === selected}
              aria-controls="model-preview-panel"
              onClick={() => setSelected(candidate.id)}
            >
              <span>{candidate.name}</span>
              <small>{candidate.status}</small>
            </button>
          ))}
        </div>

        <article
          className={`model-preview-panel model-${model.id}`}
          id="model-preview-panel"
          role="tabpanel"
          aria-labelledby={`model-tab-${model.id}`}
        >
          <div className="model-preview-top">
            <div>
              <span className={`preview-status ${localModel ? "live" : ""}`}>
                {model.status}
              </span>
              <h3>{model.name}</h3>
            </div>
            <div className="preview-runtime">
              <span>{model.runtimeLabel}</span>
              <strong>{model.runtime}</strong>
            </div>
          </div>

          <p className="model-summary">{model.summary}</p>

          <dl className="model-preview-facts">
            <div>
              <dt>Decision hop</dt>
              <dd>{model.hop}</dd>
            </div>
            <div>
              <dt>Released shape</dt>
              <dd>{model.delivery}</dd>
            </div>
            <div>
              <dt>Language / platform scope</dt>
              <dd>{model.scope}</dd>
            </div>
            <div>
              <dt>License</dt>
              <dd>{model.license}</dd>
            </div>
          </dl>

          <div className="model-preview-notes">
            <div>
              <span>Best fit</span>
              <p>{model.bestFit}</p>
            </div>
            <div>
              <span>Important caveat</span>
              <p>{model.watch}</p>
            </div>
          </div>

          <a
            className="button primary model-preview-action"
            href={model.source}
            target={localModel ? undefined : "_blank"}
            rel={localModel ? undefined : "noreferrer"}
          >
            {localModel ? "Run the live model" : "Open the official model"}
          </a>
        </article>
      </div>

      <p className="playground-caveat">
        Runtime values are a same-machine development snapshot of different
        released stacks, not an accuracy or product ranking. Model scopes are
        vendor claims. A fair comparison requires one frozen, consented
        call-domain dataset and one predeclared operating policy.
      </p>
    </section>
  );
}
