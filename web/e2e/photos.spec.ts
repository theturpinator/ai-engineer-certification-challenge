import { test, expect, type Page } from "@playwright/test";
import { actAsUser, seedUserWithCar } from "./helpers";

// Issue #53: oversized photos compress in the browser before upload, and an
// uploaded photo can be rotated 90° (persisting across reload) and replaced.

const API = "http://localhost:8000";

/** Build a JPEG of the given size in the browser and return it as a buffer. */
async function makeJpeg(page: Page, w: number, h: number): Promise<Buffer> {
  const b64 = await page.evaluate(
    async ([w, h]) => {
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d")!;
      // per-pixel noise compresses terribly -> a reliably oversized file
      const img = ctx.createImageData(w, h);
      const px = new Uint32Array(img.data.buffer);
      for (let i = 0; i < px.length; i++) px[i] = (Math.random() * 0xffffff) | 0xff000000;
      ctx.putImageData(img, 0, 0);
      const blob: Blob = await new Promise((res) =>
        canvas.toBlob((b) => res(b!), "image/jpeg", 1)
      );
      const buf = await blob.arrayBuffer();
      let s = "";
      new Uint8Array(buf).forEach((b) => (s += String.fromCharCode(b)));
      return btoa(s);
    },
    [w, h] as const
  );
  return Buffer.from(b64, "base64");
}

/** Dimensions of the stored image, straight from the API (cache bypassed). */
async function storedDims(page: Page, uid: string, carId: string) {
  return page.evaluate(
    async (url) => {
      const blob = await (await fetch(url, { cache: "no-store" })).blob();
      const bmp = await createImageBitmap(blob);
      return { w: bmp.width, h: bmp.height };
    },
    `${API}/garage/${uid}/cars/${carId}/image?e2e=${Date.now()}`
  );
}

test("oversized upload compresses, rotates, and can be replaced", async ({ page }) => {
  const { userId, carId } = seedUserWithCar();
  await actAsUser(page, userId);
  await page.goto("/garage");

  // an oversized photo (well past the ~4.5 MB production body cap)
  const big = await makeJpeg(page, 4000, 3000);
  expect(big.byteLength).toBeGreaterThan(4_500_000);

  const putReq = page.waitForRequest(
    (r) => r.url().includes("/image") && r.method() === "PUT"
  );
  await page
    .locator('.uploadpill input[type="file"]')
    .setInputFiles({ name: "phone-photo.jpg", mimeType: "image/jpeg", buffer: big });
  const sizes = await (await putReq).sizes();
  expect(sizes.requestBodySize).toBeLessThan(4_500_000); // compressed client-side

  // the pill flips to the edit affordances once the upload lands
  await expect(page.getByText("Change photo")).toBeVisible({ timeout: 30_000 });
  let dims = await storedDims(page, userId, carId);
  expect(Math.max(dims.w, dims.h)).toBeLessThanOrEqual(2000); // downscaled
  expect(dims.w).toBeGreaterThan(dims.h); // landscape as shot

  // Rotate 90°: dimensions swap and survive a reload
  const rotated = page.waitForResponse(
    (r) => r.url().includes("/image") && r.request().method() === "PUT"
  );
  await page.getByRole("button", { name: "Rotate 90°" }).click();
  expect((await rotated).status()).toBe(204);
  await page.reload();
  await expect(page.getByText("Change photo")).toBeVisible();
  dims = await storedDims(page, userId, carId);
  expect(dims.h).toBeGreaterThan(dims.w); // portrait now — the rotate stuck

  // Change photo replaces the stored image
  const replaced = page.waitForResponse(
    (r) => r.url().includes("/image") && r.request().method() === "PUT"
  );
  await page
    .locator('.uploadpill input[type="file"]')
    .setInputFiles({
      name: "other.jpg",
      mimeType: "image/jpeg",
      buffer: await makeJpeg(page, 800, 600),
    });
  expect((await replaced).status()).toBe(204);
  dims = await storedDims(page, userId, carId);
  expect(dims).toEqual({ w: 800, h: 600 });
});
