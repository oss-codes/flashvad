import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function renderedHtml() {
  return readFile(new URL("../dist/client/index.html", import.meta.url), "utf8");
}

test("Astro renders the complete static benchmark report", async () => {
  const html = await renderedHtml();

  assert.match(html, /<title>FlashVAD — macOS benchmark report<\/title>/i);
  assert.match(html, /oss\.codes\/<\/span><span[^>]*>flashvad/i);
  assert.match(html, /Voice detection in/);
  assert.match(html, /11\.42 microseconds/);
  assert.match(html, /The native path removes framework overhead/);
  assert.match(html, /Play a call with its speech analysis\./);
  assert.match(html, /Test my microphone/);
  assert.match(html, /Analyze and play audio/);
  assert.match(html, /Audio never leaves your browser/);
  assert.match(html, /Preview the model differences/);
  assert.match(html, /Live runnable here/);
  assert.match(html, /Reference preview/);
  assert.match(html, /FlashVAD/);
  assert.match(html, /Silero VAD/);
  assert.match(html, /TEN VAD/);
  assert.match(html, /FireRedVAD/);
  assert.match(html, /14\.2×/);
  assert.match(html, /45\.2×/);
  assert.match(html, /machine-readable evidence/);
  assert.match(html, /Only FlashVAD executes in this browser demo/);
  assert.match(html, /One persistent model\. Small state per call\./);
  assert.match(html, /Market position by deployment need/);
  assert.match(html, /10 ms worklet chunks/);
  assert.match(html, /Custom VAD stream adapter/);
  assert.match(html, /Custom VADAnalyzer/);
  assert.match(
    html,
    /Training improved validation\. The external gate stopped a regression\./,
  );
  assert.match(html, /Runtime milestone: achieved\./);
  assert.match(html, /Accuracy milestone: open\./);
  assert.match(html, /href="#main"/);
  assert.match(html, /aria-label="Report sections"/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("keeps claims, methodology, and Astro islands explicit in source", async () => {
  const [
    page,
    playground,
    layout,
    styles,
    packageJson,
    astroConfig,
    wranglerConfig,
  ] =
    await Promise.all([
      readFile(new URL("../src/pages/index.astro", import.meta.url), "utf8"),
      readFile(
        new URL("../src/components/VadPlayground.tsx", import.meta.url),
        "utf8",
      ),
      readFile(
        new URL("../src/layouts/BaseLayout.astro", import.meta.url),
        "utf8",
      ),
      readFile(new URL("../src/styles/global.css", import.meta.url), "utf8"),
      readFile(new URL("../package.json", import.meta.url), "utf8"),
      readFile(new URL("../astro.config.mjs", import.meta.url), "utf8"),
      readFile(new URL("../wrangler.jsonc", import.meta.url), "utf8"),
    ]);

  assert.match(page, /These are compute measurements/);
  assert.match(page, /Pinned external rerun/);
  assert.match(page, /competitor RTF divided by FlashVAD native RTF/);
  assert.match(page, /26,243/);
  assert.match(page, /26\.3%/);
  assert.match(page, /1,056/);
  assert.match(page, /264/);
  assert.match(page, /0\.914/);
  assert.match(page, /5\.47%/);
  assert.match(page, /12\.35%/);
  assert.match(page, /was not[\s\S]*promoted/);
  assert.match(page, /0\.853–0\.910/);
  assert.match(page, /20\.2–32\.7%/);
  assert.match(page, /repeatedly on this public set/);
  assert.match(page, /<VadPlayground client:visible \/>/);
  assert.match(page, /<ModelPreview client:visible \/>/);
  assert.match(playground, /env\.wasm\.wasmBinary/);
  assert.match(playground, /onnxruntime-web\/wasm/);
  assert.match(playground, /import\.meta\.env\.BASE_URL/);
  assert.match(playground, /URL\.createObjectURL/);
  assert.match(playground, /speakingStateAtTime/);
  assert.match(playground, /leftChannelRole/);
  assert.match(playground, /aria-label="Uploaded call playback"/);
  assert.match(playground, /Synchronized playback/);
  assert.match(playground, /Hear the call and follow each speech track\./);
  assert.match(playground, /Swap user and AI channels/);
  assert.match(playground, /speaker diarization requires a separate model/);
  assert.match(layout, /FlashVAD — macOS benchmark report/);
  assert.match(styles, /color-scheme: dark/);
  assert.match(packageJson, /"astro": "7\.1\.3"/);
  assert.match(packageJson, /"@astrojs\/cloudflare": "14\.1\.5"/);
  assert.match(packageJson, /"wrangler": "4\.114\.0"/);
  assert.match(packageJson, /sync-public-assets/);
  assert.match(astroConfig, /cloudflare\(\{ imageService: "compile" \}\)/);
  assert.match(astroConfig, /output: "server"/);
  assert.match(astroConfig, /FLASHVAD_BASE/);
  assert.match(astroConfig, /trailingSlash: "always"/);
  assert.match(page, /export const prerender = true/);
  assert.match(wranglerConfig, /"name": "flashvad-report"/);
  assert.match(wranglerConfig, /"compatibility_date": "2026-07-22"/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.doesNotMatch(page, /0\.654|51\.2%/);
  assert.doesNotMatch(packageJson, /next|vinext/i);

  await assert.rejects(access(new URL("../app", import.meta.url)));
  await assert.rejects(access(new URL("../worker", import.meta.url)));
  await assert.doesNotReject(
    access(new URL("dist/client/index.html", templateRoot)),
  );
});
