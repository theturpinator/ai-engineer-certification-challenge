#!/usr/bin/env bash
# Production deploy for Ask MustangDriver (issue #24).
#
# Deploys the API first (backwards-compatible for the already-live frontend),
# health-checks it, then deploys the web app and smoke-checks it. Fails fast
# at every step. Run only after the show-first ship gate: local build +
# screenshots + owner approval.
#
#   ./deploy.sh
#
# Needs the Vercel CLI logged in and both apps linked (api/.vercel and
# web/.vercel are committed links; `vercel link` re-creates them).

set -euo pipefail
cd "$(dirname "$0")"

API_URL="https://ask-mustangdriver-api.vercel.app"
WEB_URL="https://ask-mustangdriver-web.vercel.app"

# The Vercel CLI and Next.js need Node 18+; if the shell default is older
# (this machine defaults to v16), pick up the newest nvm-installed Node.
node_major() { node --version 2>/dev/null | sed 's/^v\([0-9]*\).*/\1/'; }
if [ "$(node_major || echo 0)" -lt 18 ]; then
  newest=$(ls -d "$HOME"/.nvm/versions/node/v*/bin 2>/dev/null | sort -V | tail -1)
  [ -n "${newest:-}" ] && export PATH="$newest:$PATH"
fi
if [ "$(node_major || echo 0)" -lt 18 ]; then
  echo "error: Node 18+ required (found $(node --version 2>/dev/null || echo none))" >&2
  exit 1
fi

echo "==> Deploying API"
(cd api && vercel deploy --prod --yes)

echo "==> API health check"
curl -sf --max-time 30 "$API_URL/health" | grep -q '"ok"' \
  || { echo "error: API health check failed at $API_URL/health" >&2; exit 1; }

echo "==> Deploying web"
(cd web && vercel deploy --prod --yes)

echo "==> Web smoke check"
curl -sf --max-time 30 -o /dev/null "$WEB_URL" \
  || { echo "error: web smoke check failed at $WEB_URL" >&2; exit 1; }

echo "✓ Live: $WEB_URL  (API: $API_URL)"
