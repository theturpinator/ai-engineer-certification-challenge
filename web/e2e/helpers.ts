import { execSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import type { Locator, Page } from "@playwright/test";

/** Seed a user who has already finished onboarding, straight into the
 * compose Postgres (docker compose resolves the repo-root file from any
 * subdirectory). SQL instead of the API: no endpoint stamps the flag, and
 * chat-driven onboarding would burn LLM turns per test run. */
export function seedOnboardedUser(): string {
  const userId = randomUUID();
  execSync(
    `docker compose exec -T postgres psql -U postgres -d mustang -c ` +
      `"INSERT INTO garage (user_id, profile) VALUES ('${userId}', '{\\"onboarded\\": true}')"`
  );
  return userId;
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
