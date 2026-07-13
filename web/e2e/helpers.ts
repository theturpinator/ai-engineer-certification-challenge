import { execSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import type { Locator, Page } from "@playwright/test";

/** Run SQL against the compose Postgres (docker compose resolves the
 * repo-root file from any subdirectory); stdin sidesteps shell quoting. */
function psql(sql: string): void {
  execSync("docker compose exec -T postgres psql -U postgres -d mustang", {
    input: sql,
  });
}

/** Seed a user who has already finished onboarding. SQL instead of the API:
 * no endpoint stamps the flag, and chat-driven onboarding would burn LLM
 * turns per test run. */
export function seedOnboardedUser(): string {
  const userId = randomUUID();
  psql(`INSERT INTO garage (user_id, profile) VALUES ('${userId}', '{"onboarded": true}');`);
  return userId;
}

// 1x1 grey PNG — a placeholder portrait row so page loads never spend a
// real image-generation call.
const TINY_PNG_HEX =
  "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de0000000c49444154789c636868680000030401814bd3d2100000000049454e44ae426082";

/** Seed an onboarded user with one fully-enriched car ("car1"): stats carry
 * the exact fingerprint the API derives (so no stats LLM call fires) and a
 * placeholder portrait row exists (so no image-generation call fires). */
export function seedUserWithCar(): { userId: string; carId: string } {
  const userId = randomUUID();
  const carId = "car1";
  const stats = {
    power: 80, acceleration: 78, top_speed: 82, handling: 75, braking: 74,
    style: 70, comfort: 65, safety: 72, reliability: 75,
    hp: 435, zero_to_sixty: 4.5, top_speed_mph: 155,
    // must byte-match python's json.dumps({"identity": _car_desc(car), "v": 3})
    fp: '{"identity": "2016 S550 GT", "v": 3}',
  };
  const profile = {
    onboarded: true,
    cars: [{ id: carId, year: 2016, generation: "S550", trim: "GT",
             color: "red", stats }],
  };
  psql(
    `INSERT INTO garage (user_id, profile) VALUES ('${userId}', '${JSON.stringify(profile)}');` +
      `INSERT INTO car_images (user_id, car_id, image, prompt, content_type, user_uploaded)` +
      ` VALUES ('${userId}', '${carId}', '\\x${TINY_PNG_HEX}'::bytea, 'seed', 'image/png', FALSE);`
  );
  return { userId, carId };
}

/** Make the next page load act as this user (localStorage is read on mount). */
export async function actAsUser(page: Page, userId: string): Promise<void> {
  await page.addInitScript(
    (id) => localStorage.setItem("md_user_id", id),
    userId
  );
}

/** Send a chat message and wait for the whole agent turn: the SSE stream
 * closing is the only reliable done-signal — text visible mid-turn can be
 * narration the client wipes when a tool call fires. Returns the answer
 * bubble. */
export async function sendMessage(page: Page, text: string): Promise<Locator> {
  // Wait for a client-only render (welcome block or a message bubble): text
  // filled before Next.js hydration never reaches React state, leaving Send
  // disabled forever.
  await page.locator(".welcome, .msg").first().waitFor({ timeout: 15_000 });
  const composer = page.locator("form textarea");
  await composer.fill(text);
  const streamed = page.waitForResponse(
    (r) => r.url().endsWith("/chat") && r.request().method() === "POST",
    { timeout: 30_000 }
  );
  await page.getByRole("button", { name: "Send" }).click();
  await (await streamed).finished(); // resolves when the SSE stream closes
  return page.locator(".msg.assistant").last();
}
