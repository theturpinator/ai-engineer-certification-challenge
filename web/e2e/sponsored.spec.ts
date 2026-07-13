import { test, expect } from "@playwright/test";
import { actAsUser, seedOnboardedUser, sendMessage } from "./helpers";

// Issue #50: a where-to-buy question surfaces Sponsored advertiser cards
// (capped at three) linking out to a sponsor's website.
test("where-to-buy surfaces a Sponsored card", async ({ page }) => {
  await actAsUser(page, seedOnboardedUser());
  await page.goto("/");

  await sendMessage(page, "Where should I buy a cat-back exhaust for my Mustang?");

  const cards = page.locator(".adcard");
  await expect(cards.first()).toBeVisible();
  expect(await cards.count()).toBeLessThanOrEqual(3);
  await expect(cards.first().locator(".sponsoredtag")).toContainText("Sponsored");
  expect(await cards.first().getAttribute("href")).toMatch(/^https?:\/\//);
});
