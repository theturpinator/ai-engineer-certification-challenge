import { defineConfig } from "@playwright/test";

// E2e harness (issue #49): runs against the local stack — compose Postgres
// must be up (`docker compose up -d` at the repo root); the API and web dev
// servers are started here (or reused if already running).
export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000, // LLM turns stream for a while
  // agent turns hit a real LLM — serial keeps runs deterministic and cheap
  workers: 1,
  use: {
    baseURL: "http://localhost:3000",
    video: "on", // every run records a demo video (ship-gate artifact)
  },
  webServer: [
    {
      command: "cd ../api && uv run uvicorn app:app --port 8000",
      url: "http://localhost:8000/health",
      reuseExistingServer: true,
      timeout: 60_000,
    },
    {
      // a prior `npm run build` leaves production chunks in .next that make
      // the dev server 404 its own bundles (page never hydrates) — start clean
      command: "rm -rf .next && npm run dev",
      url: "http://localhost:3000",
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
