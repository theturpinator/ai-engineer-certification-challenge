import { test, expect } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { actAsUser, sendMessage } from "./helpers";

// Issue #52: a fresh user's onboarding interview follows up on upgrades
// (recorded as wishlist items), asks for a missing color once, and asks
// "any more Mustangs?" once. The buy question is skipped for this owner.
test("onboarding interviews with the new follow-ups", async ({ page }) => {
  const userId = randomUUID(); // brand-new user: the chat auto-starts the interview
  await actAsUser(page, userId);
  await page.goto("/");

  // the agent opens the interview on its own
  await expect(page.locator(".msg.assistant").first()).toContainText(/\?/, {
    timeout: 60_000,
  });

  await sendMessage(page, "Alex");
  const carTurn = await sendMessage(page, "I have a 2016 Mustang GT"); // no color
  await expect(carTurn).toContainText(/colou?r/i);

  const moreTurn = await sendMessage(page, "It's red");
  await expect(moreTurn).toContainText(/more|another|other/i);

  await sendMessage(page, "No, just the one");
  const afterUpgrades = await sendMessage(page, "Yes — a supercharger and lowering springs");
  await expect(afterUpgrades).not.toContainText(/buy/i); // owners skip the buy question

  // the named upgrades landed on the car's wishlist
  const garage = await (
    await page.request.get(`http://localhost:8000/garage/${userId}`)
  ).json();
  const wishlist = (garage.profile.cars?.[0]?.wishlist ?? []).join(" ").toLowerCase();
  expect(wishlist).toContain("supercharger");
  expect(garage.profile.cars?.[0]?.color?.toLowerCase()).toBe("red");
});
