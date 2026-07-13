import { test, expect } from "@playwright/test";
import { actAsUser, seedOnboardedUser, sendMessage } from "./helpers";

// Issue #51: live sponsor-site search + sponsor-forward answer shape.
// Uses the real live-search provider, like production.

test("niche-part question surfaces live Sponsored results", async ({ page }) => {
  await actAsUser(page, seedOnboardedUser());
  await page.goto("/");

  await sendMessage(
    page,
    "I'm restoring a 1966 Mustang and need a concours-correct taillight bezel. Where can I get one?"
  );

  const cards = page.locator(".adcard");
  await expect(cards.first()).toBeVisible();
  expect(await cards.count()).toBeLessThanOrEqual(3);
  await expect(cards.first().locator(".sponsoredtag")).toContainText("Sponsored");
});

test("upgrade-advice answer cites an archive article", async ({ page }) => {
  await actAsUser(page, seedOnboardedUser());
  await page.goto("/");

  const answer = await sendMessage(page, "What are good first mods for a 2016 Mustang GT?");

  // an inline markdown citation to the archive renders as a link
  await expect(
    answer.locator('a[href*="mustangdriver.com"]').first()
  ).toBeVisible();
});
