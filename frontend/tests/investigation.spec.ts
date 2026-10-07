import { test, expect } from "@playwright/test";
test("search opens source evidence, timeline, and report", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("System ready")).toBeVisible();
  await page
    .getByRole("button", {
      name: "Who left an object at the east entrance?",
      exact: true,
    })
    .click();
  await expect(
    page.getByText("1 matching events", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Open evidence", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Person places a backpack near the east entrance",
      exact: true,
    }),
  ).toBeVisible();
  await expect
    .poll(async () =>
      page.locator("video").evaluate((v: HTMLVideoElement) => v.currentTime),
    )
    .toBeGreaterThanOrEqual(12);
  await expect(page.getByLabel("Timeline entity")).toHaveValue("P-E01");
  await expect(
    page.getByText("Ambiguous match", { exact: true }).first(),
  ).toBeVisible();
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export report", exact: true })
    .click();
  expect((await download).suggestedFilename()).toContain("VIGILIA-");
});
test("evaluation calculates metrics from fixture queries", async ({ page }) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Evaluation lab", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Run benchmark", exact: true })
    .click();
  await expect(
    page.getByText("40 queries · 35 positive · 5 negative", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("No paired timings yet.", { exact: false }),
  ).toBeVisible();
});
test("negative search preserves uncertainty and coverage", async ({ page }) => {
  await page.goto("/");
  await page
    .getByLabel("Investigation query")
    .fill("Find people at Camera 5 between 7 and 8 PM");
  await page
    .getByRole("button", { name: "Search evidence", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "No sufficiently strong match found",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Coverage assessment", exact: true }),
  ).toBeVisible();
});
