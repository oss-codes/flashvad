import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";

const root = new URL("../dist/", import.meta.url);

async function files(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (
    await Promise.all(
      entries.map(async (entry) => {
        const url = new URL(entry.name, directory);
        return entry.isDirectory() ? files(new URL(`${entry.name}/`, directory)) : [url];
      }),
    )
  ).flat();
}

const output = await files(root);
const textFiles = output.filter((url) =>
  /\.(?:css|html|js|mjs|svg)$/.test(url.pathname),
);
const rendered = (
  await Promise.all(textFiles.map((url) => readFile(url, "utf8")))
).join("\n");

for (const asset of [
  "benchmarks/external-runtime-m4-pro.json",
  "benchmarks/onnx-provider-colab-t4.json",
  "favicon.svg",
  "models/MODEL_LICENSE.md",
  "NOTICE.txt",
  "og-image.png",
]) {
  assert.ok(
    rendered.includes(`/flashvad/${asset}`),
    `subpath build does not reference /flashvad/${asset}`,
  );
}

assert.match(rendered, /["'`]\/flashvad\/["'`]/);
assert.match(rendered, /models\/flashvad-stream\.onnx/);
assert.match(rendered, /vad-audio-processor\.js/);
assert.doesNotMatch(rendered, /\/flashvad(?:benchmarks|favicon|models|NOTICE|vad-)/);
console.log("verified /flashvad static asset URLs");
