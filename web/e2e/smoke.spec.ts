import { test, expect } from "@playwright/test";
import { actAsUser, seedOnboardedUser, sendMessage } from "./helpers";

// Issue #49 smoke test: load the chat, send a message, see a streamed answer.
test("chat streams an answer", async ({ page }) => {
  await actAsUser(page, seedOnboardedUser());
  await page.goto("/");

  await expect(page.getByPlaceholder("Ask about Mustangs…")).toBeVisible();
  const answer = await sendMessage(page, "What oil does a 2016 GT take?");

  // the user bubble landed…
  await expect(page.locator(".msg.user")).toHaveText("What oil does a 2016 GT take?");
  // …and the finished turn rendered a real answer, not a pending indicator
  await expect(answer.locator(".pending")).toHaveCount(0);
  expect((await answer.innerText()).length).toBeGreaterThan(40);
});
