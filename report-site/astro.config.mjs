import cloudflare from "@astrojs/cloudflare";
import react from "@astrojs/react";
import { defineConfig } from "astro/config";

export default defineConfig({
  adapter: cloudflare({ imageService: "compile" }),
  base: process.env.FLASHVAD_BASE ?? "/",
  integrations: [react()],
  output: "server",
  trailingSlash: "always",
});
