# Multilingual data and evaluation

“Multilingual” is an empirical result, not a model description.

## Priority regions

The first production benchmark should stratify, rather than blend:

- India: Hindi, Bengali, Marathi, Gujarati, Punjabi, Tamil, Telugu, Kannada, Malayalam, Urdu,
  Indian English, and common code-switching pairs;
- GCC: Gulf Arabic varieties, Modern Standard Arabic, English, Hindi, Urdu, Bengali, Malayalam,
  Tamil, and Tagalog;
- telephony channel: G.711 mu-law/A-law, 8 kHz narrowband, packet loss, jitter concealment and
  codec cascades;
- devices: handset, wired/Bluetooth headset, speakerphone, car and far-field room audio.

## Manifest

Each JSONL record has this shape:

```json
{
  "audio": "relative/or/absolute.wav",
  "sample_rate": 16000,
  "language": "hi",
  "domain": "pstn",
  "channel": "mobile",
  "codec": "g711-ulaw",
  "device": "handset",
  "condition": "street-noise",
  "snr_db": 4.5,
  "segments": [
    {"start": 0.42, "end": 1.37, "label": "speech"}
  ]
}
```

Use BCP-47 language tags when possible. Add metadata fields freely, but never infer sensitive
demographics from voice.

Training can mix a separate, speech-free noise manifest on the fly:

```bash
uv run flashvad train \
  --config configs/base.json \
  --train-manifest /path/to/train.jsonl \
  --valid-manifest /path/to/dev.jsonl \
  --noise-manifest /path/to/noise-only.jsonl \
  --output artifacts/india-gcc
```

The noise manifest must be checked for speech leakage. It is augmentation data, not a substitute
for real noisy calls.

## Multi-event and teacher supervision

Segment labels may be `speech`, `music`, `singing`, `other_vocal`, `laughter`, or `cough`. The
binary release head treats only `speech` as target speech; the auxiliary multi-label head learns the
other event classes so they can be distinguished from foreground call speech.

Unlabeled real audio may carry aligned 10 ms teacher probabilities:

```json
{
  "audio": "unlabeled-gcc-call.wav",
  "teacher_probabilities": "unlabeled-gcc-call.teacher.npy",
  "teacher_weight": 1.0,
  "language": "ar-AE",
  "domain": "pstn",
  "segments": []
}
```

Use an ensemble of permissively licensed teachers, keep uncertainty as soft probabilities, and send
teacher disagreements plus boundary regions to human review. Do not use the frozen test set for
pseudo-labeling or hard-negative mining. TEN outputs are excluded from this workflow because its
license contains additional competitive-use restrictions.

## Labeling policy

Document whether breaths, laughter, coughs, singing, background speech and overlapping agent audio
count as speech for each product mode. Barge-in and ASR segmentation may need different targets.

At least two annotators should label the difficult boundary subset. Keep raw labels and adjudicated
labels. Measure annotator disagreement; “perfect” accuracy is undefined when humans disagree.

## Split policy

No speaker, call, room impulse response, or noise recording may cross train/dev/test splits.
Thresholds are tuned on dev only. Maintain a frozen, consented real-call test set that is never used
for hard-negative mining.

The FLEURS preparation tool preserves the dataset's upstream split boundary:
FLEURS `train` is used only for training and FLEURS `validation` only for
development. It does not download or inspect FLEURS `test`.

## Reproducible Mac training sweep

Prepare attributed multilingual speech and real non-speech negatives:

```bash
uv sync --extra data --extra training

HF_HUB_DOWNLOAD_TIMEOUT=600 uv run python scripts/prepare_fleurs_vad.py \
  --output data/fleurs-vad-mac-v4 \
  --train-samples-per-language 64 \
  --valid-samples-per-language 16

uv run python scripts/prepare_musan_negatives.py \
  --output data/musan-negatives-v2 \
  --items 120
```

The default hybrid downloader pins the current FLEURS revision, streams the
large upstream `train` Parquet shard directly, and uses the lighter rows API
for `validation`. If the rows service is unavailable, validation automatically
falls back to the same revision-pinned stream. It verifies that every rows-API
asset URL names the same revision, keeps revision-scoped source/cache paths,
and records the actual backend used for every language/split in
`PROVENANCE.json`. This avoids the datasets-server train scan limit while
preserving the upstream split boundary. The longer Hub timeout prevents
multi-gigabyte Parquet range reads from restarting on a slow link.

The default set uses all relevant configurations currently exposed by FLEURS:
Arabic (Egypt), Bengali, English (US), Gujarati, Hindi, Kannada, Malayalam,
Marathi, Punjabi, Tamil, Telugu, and Urdu. This is multilingual read-speech
development data. Arabic (Egypt) is not Gulf-dialect coverage, English (US) is
not Indian English, and none of these clips proves performance on GCC carrier
calls or code-switching.

Merge the explicit negative records into both splits, then run the committed
eight-trial loss/seed sweep:

```bash
uv run python scripts/merge_manifests.py \
  --input data/fleurs-vad-mac-v4/train.jsonl \
  --input data/musan-negatives-v2/train.jsonl \
  --output data/multilingual-mac-v4/train.jsonl

uv run python scripts/merge_manifests.py \
  --input data/fleurs-vad-mac-v4/valid.jsonl \
  --input data/musan-negatives-v2/valid.jsonl \
  --output data/multilingual-mac-v4/valid.jsonl

uv run python scripts/run_training_sweep.py \
  --sweep configs/sweeps/multilingual-mac.json \
  --train-manifest data/multilingual-mac-v4/train.jsonl \
  --valid-manifest data/multilingual-mac-v4/valid.jsonl \
  --noise-manifest data/musan-negatives-v2/train.jsonl \
  --region-profile configs/regions/india_gcc.json \
  --output artifacts/multilingual-mac-v4
```

Every trial records its exact config and checkpoint digest. A trial is eligible
only if its development detector reaches at least 0.85 F1, at most 15% false
alarms, and at most 20% misses; eligible trials are ranked by detector F1 with
lower false alarms and misses as tie-breakers. TEN, Silero, and FireRed public
benchmark sets are explicitly forbidden from sweep selection.
The runner rejects their configured identifiers in both the validation-manifest
path and record metadata. This is a guardrail against accidental reuse, not a
substitute for provenance review because a renamed dataset can evade textual
identifiers.

The completed local run trained all eight configurations for 16 epochs each.
Its selected candidate reached 0.914 development detector F1, 5.47% false
alarms, and 8.45% misses. It was not promoted because the later descriptive TEN
check traded lower false alarms for worse F1 and misses. See
`benchmarks/flashvad-multilingual-alpha/training-and-evaluation.json`.

## Release gates

Report every metric by language, channel, SNR and event type:

- PR-AUC, ROC-AUC, F1, false-alarm and miss rate;
- onset clipping and offset tail in milliseconds;
- short-utterance recall;
- false triggers per noise-only hour;
- premature end-of-turn rate;
- P50/P95/P99 compute and memory on each target device.
