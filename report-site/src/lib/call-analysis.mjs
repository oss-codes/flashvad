export const ANALYSIS_HOP_MS = 10;

export function detectSpeechFrames(
  probabilities,
  startThreshold,
  {
    startFrames = 3,
    stopFrames = 4,
    stopThreshold = Math.max(0.05, startThreshold - 0.3),
  } = {},
) {
  const speechFrames = new Array(probabilities.length).fill(false);
  let speaking = false;
  let startCount = 0;
  let stopCount = 0;

  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = Number(probabilities[index]);
    if (!speaking) {
      startCount = probability >= startThreshold ? startCount + 1 : 0;
      if (startCount >= startFrames) {
        speaking = true;
        stopCount = 0;
      }
    } else {
      stopCount = probability < stopThreshold ? stopCount + 1 : 0;
      if (stopCount >= stopFrames) {
        speaking = false;
        startCount = 0;
      }
    }
    speechFrames[index] = speaking;
  }

  return speechFrames;
}

export function summarizeTrack(probabilities, speechFrames, maximumBins = 160) {
  if (probabilities.length !== speechFrames.length) {
    throw new Error("probability and speech-frame lengths must match");
  }
  if (probabilities.length === 0) return [];

  const binCount = Math.min(maximumBins, probabilities.length);
  const framesPerBin = probabilities.length / binCount;

  return Array.from({ length: binCount }, (_, binIndex) => {
    const startFrame = Math.floor(binIndex * framesPerBin);
    const endFrame = Math.max(
      startFrame + 1,
      Math.min(probabilities.length, Math.ceil((binIndex + 1) * framesPerBin)),
    );
    let peak = 0;
    let total = 0;
    let active = false;
    for (let frameIndex = startFrame; frameIndex < endFrame; frameIndex += 1) {
      const probability = Number(probabilities[frameIndex]);
      peak = Math.max(peak, probability);
      total += probability;
      active ||= Boolean(speechFrames[frameIndex]);
    }
    return {
      active,
      average: total / (endFrame - startFrame),
      peak,
      startFrame,
      endFrame,
    };
  });
}

export function speakingStateAtTime(tracks, timeMs) {
  const frameIndex = Math.max(0, Math.floor(timeMs / ANALYSIS_HOP_MS));
  const active = tracks
    .filter((track) => Boolean(track.speechFrames[frameIndex]))
    .map((track) => track.role);

  if (active.includes("user") && active.includes("ai")) return "overlap";
  if (active.includes("user")) return "user";
  if (active.includes("ai")) return "ai";
  if (active.includes("mixed")) return "mixed";
  return "silence";
}

export function formatAnalysisTime(milliseconds) {
  const totalSeconds = Math.max(0, milliseconds) / 1_000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}
