import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function renderedHtml() {
  return readFile(new URL("../dist/client/index.html", import.meta.url), "utf8");
}

test("Astro renders the complete static benchmark report", async () => {
  const [html, robots, sitemapIndex, sitemap] = await Promise.all([
    renderedHtml(),
    readFile(new URL("../dist/client/robots.txt", import.meta.url), "utf8"),
    readFile(
      new URL("../dist/client/sitemap-index.xml", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../dist/client/sitemap-0.xml", import.meta.url), "utf8"),
  ]);

  assert.match(
    html,
    /<title>FlashVAD: Fast Voice Activity Detection for Voice Calls<\/title>/i,
  );
  assert.match(
    html,
    /<link rel="canonical" href="https:\/\/flash\.oss\.codes\/"/i,
  );
  assert.match(
    html,
    /<meta name="robots" content="index, follow, max-image-preview:large,/i,
  );
  assert.match(
    html,
    /<meta property="og:url" content="https:\/\/flash\.oss\.codes\/"/i,
  );
  assert.match(
    html,
    /<meta property="og:image" content="https:\/\/flash\.oss\.codes\/og-image\.png"/i,
  );
  assert.match(html, /<meta name="twitter:card" content="summary_large_image"/i);
  assert.match(html, /"@type":"SoftwareSourceCode"/);
  assert.match(html, /"name":"Himanshu Maurya"/);
  assert.equal((html.match(/xt6s4ttzdy/g) ?? []).length, 1);
  assert.equal(
    (html.match(/static\.cloudflareinsights\.com\/beacon\.min\.js/g) ?? [])
      .length,
    1,
  );
  assert.equal(
    (html.match(/6edbfa0990a946d6b30f349c4fcfd464/g) ?? []).length,
    1,
  );
  assert.match(html, /data-clarity-mask="true"/);
  assert.doesNotMatch(html, /\u2014/);
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
  assert.match(html, /A T4 helps only when enough calls are ready together\./);
  assert.match(html, /Hardened rerun required/);
  assert.match(html, /CUDA is not a single-call latency upgrade\./);
  assert.match(html, /7\.60 ms/);
  assert.match(html, /Download the preliminary T4 evidence\./);
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

  assert.match(
    robots,
    /Sitemap: https:\/\/flash\.oss\.codes\/sitemap-index\.xml/,
  );
  assert.match(
    sitemapIndex,
    /<loc>https:\/\/flash\.oss\.codes\/sitemap-0\.xml<\/loc>/,
  );
  assert.match(
    sitemap,
    /<loc>https:\/\/flash\.oss\.codes\/<\/loc>/,
  );
  await assert.doesNotReject(
    access(new URL("../dist/client/og-image.png", import.meta.url)),
  );
  await assert.doesNotReject(
    access(new URL("../dist/client/llms.txt", import.meta.url)),
  );
  const colabEvidence = JSON.parse(
    await readFile(
      new URL(
        "../dist/client/benchmarks/onnx-provider-colab-t4.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
  assert.equal(colabEvidence.hardware.gpu, "Tesla T4");
  assert.equal(colabEvidence.results.length, 12);
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
    syncScript,
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
      readFile(
        new URL("../scripts/sync-public-assets.mjs", import.meta.url),
        "utf8",
      ),
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
  assert.match(page, /const colabRows/);
  assert.match(page, /benchmark also rejects provider fallback/);
  assert.match(page, /checks the full logit[\s\S]*trace/);
  assert.match(page, /benchmarks\/onnx-provider-colab-t4\.json/);
  assert.doesNotMatch(page, /2\.85×|8\.88×/);
  assert.match(page, /<VadPlayground client:visible \/>/);
  assert.match(page, /<ModelPreview client:visible \/>/);
  assert.match(playground, /env\.wasm\.wasmBinary/);
  assert.match(playground, /wasmBinaryLoader\.reset\(\)/);
  assert.match(playground, /browserSessionLoader\.reset\(\)/);
  assert.match(playground, /Retry browser runtime/);
  assert.match(playground, /audio stays[\s\S]*this browser/);
  assert.doesNotMatch(playground, /^import \* as ort from "onnxruntime-web\/wasm";/m);
  assert.match(playground, /import\("onnxruntime-web\/wasm"\)/);
  assert.match(playground, /import\.meta\.env\.BASE_URL/);
  assert.match(playground, /URL\.createObjectURL/);
  assert.match(playground, /speakingStateAtTime/);
  assert.match(playground, /leftChannelRole/);
  assert.match(playground, /aria-label="Uploaded call playback"/);
  assert.match(playground, /Synchronized playback/);
  assert.match(playground, /Hear the call and follow each speech track\./);
  assert.match(playground, /Swap user and AI channels/);
  assert.match(playground, /speaker diarization requires a separate model/);
  assert.match(
    layout,
    /FlashVAD: Fast Voice Activity Detection for Voice Calls/,
  );
  assert.match(layout, /rel="canonical"/);
  assert.match(layout, /application\/ld\+json/);
  assert.match(layout, /xt6s4ttzdy/);
  assert.match(layout, /6edbfa0990a946d6b30f349c4fcfd464/);
  assert.match(playground, /data-clarity-mask="true"/);
  assert.match(styles, /color-scheme: dark/);
  assert.match(packageJson, /"astro": "7\.1\.3"/);
  assert.match(packageJson, /"@astrojs\/cloudflare": "14\.1\.5"/);
  assert.match(packageJson, /"@astrojs\/sitemap": "3\.7\.3"/);
  assert.match(packageJson, /"wrangler": "4\.114\.0"/);
  assert.match(packageJson, /sync-public-assets/);
  assert.match(syncScript, /onnx-provider-colab-t4\.json/);
  assert.match(astroConfig, /cloudflare\(\{ imageService: "compile" \}\)/);
  assert.match(astroConfig, /output: "server"/);
  assert.match(astroConfig, /site: "https:\/\/flash\.oss\.codes"/);
  assert.match(astroConfig, /sitemap\(/);
  assert.match(astroConfig, /FLASHVAD_BASE/);
  assert.match(astroConfig, /@astrojs\/internal-helpers > picomatch/);
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
