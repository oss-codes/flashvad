import react from "@astrojs/react";
import { defineConfig } from "astro/config";

export default defineConfig({
  base: process.env.FLASHVAD_BASE ?? "/",
  integrations: [react()],
  output: "static",
  trailingSlash: "always",
});
