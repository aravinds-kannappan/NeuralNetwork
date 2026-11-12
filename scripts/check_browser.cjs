/* Optional end-to-end QA. Requires Playwright and an installed Chromium browser.
   node scripts/check_browser.cjs URL OUTPUT_DIR CHROMIUM_EXECUTABLE */
const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");

(async () => {
  const [url, output, executablePath] = process.argv.slice(2);
  if (!url || !output || !executablePath) throw new Error("Provide URL, output directory, and Chromium executable.");
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath, headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1080 }, reducedMotion: "reduce" });
  const page = await context.newPage();
  const errors = [], badResponses = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => { if (response.status() >= 400) badResponses.push(`${response.status()} ${response.url()}`); });
  const response = await page.goto(url, { waitUntil: "networkidle" });
  assert.equal(response.status(), 200);
  await page.waitForSelector('html[data-results-ready="true"]');
  await page.waitForSelector('#explorer-panel[aria-busy="false"]');
  assert.equal(await page.locator("#pole-select option").count(), 41);
  assert.equal(await page.locator("#earlier-contexts li").count(), 5);
  assert.equal(await page.locator("#question-list details").count(), 7);
  await page.screenshot({ path: path.join(output, "desktop-overview.png") });
  const visited = await page.evaluate(async () => {
    const data = window.REPORT_DATA ? window.REPORT_DATA["assets/explorer.json"] : await (await fetch("assets/explorer.json")).json();
    const select = document.querySelector("#pole-select"), slider = document.querySelector("#step-slider");
    let count = 0;
    for (const pole of data.neurons) {
      select.value = pole.pole;
      select.dispatchEvent(new Event("change", { bubbles: true }));
      for (let i = 0; i < 16; i++) {
        slider.value = i;
        slider.dispatchEvent(new Event("input", { bubbles: true }));
        const expected = pole.similarities.find((s) => s.checkpoint === i).semantic_to_final.toFixed(3);
        if (document.querySelector("#semantic-value").textContent !== expected) throw new Error(`Wrong cosine: ${pole.pole} checkpoint ${i}`);
        if (document.querySelector("#step-value").textContent !== data.steps[i].toLocaleString("en-US")) throw new Error("Wrong step");
        if (document.querySelector("#earlier-contexts mark").textContent !== pole.ctx[pole.steps[i].top[0][0]][1]) throw new Error("Wrong marked token");
        count++;
      }
    }
    return count;
  });
  assert.equal(visited, 656);
  await page.selectOption("#pole-select", "L03N6372+");
  await page.locator('[data-step="9"]').click();
  assert.equal(await page.locator("#step-value").textContent(), "60,000");
  await page.locator("#show-contexts").click();
  assert.equal(await page.locator("#earlier-contexts li").count(), 20);
  await page.locator("#show-contexts").click();
  await page.locator("#explorer-panel").screenshot({ path: path.join(output, "desktop-explorer.png") });
  const frame = page.frameLocator(".chart-frame iframe");
  await frame.locator("select").waitFor();
  const chartOptions = await frame.locator("select option").allTextContents();
  assert.equal(chartOptions.length, 2);
  await frame.locator("select").selectOption({ index: 1 });
  await frame.locator("select").selectOption({ index: 0 });
  await page.locator("#similarity").screenshot({ path: path.join(output, "desktop-results.png") });
  for (const details of await page.locator("#question-list details").all()) {
    if (!await details.getAttribute("open").then((x) => x !== null)) await details.locator("summary").click();
    assert.ok(await details.locator("dd").count() === 3);
    await details.locator("summary").click();
  }
  await page.locator("#step-slider").focus();
  await page.keyboard.press("ArrowRight");
  assert.equal(await page.locator("#step-value").textContent(), "100,000");
  for (const width of [390, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(overflow, false, `Horizontal document overflow at ${width}px`);
    if (width === 390) {
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({ path: path.join(output, "mobile-overview.png") });
      await page.locator("#explorer-panel").screenshot({ path: path.join(output, "mobile-explorer.png") });
      await page.locator(".chart-frame").screenshot({ path: path.join(output, "mobile-chart.png") });
      await page.locator("#question-list details").first().locator("summary").click();
      await page.locator("#questions").screenshot({ path: path.join(output, "mobile-questions.png") });
    }
  }
  const deep = new URL(url); deep.searchParams.set("pole", "L21N8785+"); deep.searchParams.set("step", "320000"); deep.hash = "explorer";
  await page.goto(deep.href, { waitUntil: "networkidle" });
  await page.waitForSelector('#explorer-panel[aria-busy="false"]');
  assert.equal(await page.locator("#pole-select").inputValue(), "L21N8785+");
  assert.equal(await page.locator("#step-value").textContent(), "320,000");
  assert.deepEqual(errors, []);
  assert.deepEqual(badResponses, []);
  const result = { url, status: "PASS", checkedPoleCheckpoints: visited, viewportWidths: [390, 768, 1024, 1440], chartViews: chartOptions, browserErrors: errors, badResponses, screenshots: fs.readdirSync(output).filter((f) => f.endsWith(".png")) };
  fs.writeFileSync(path.join(output, "browser-check.json"), JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify(result));
  await browser.close();
})().catch((error) => { console.error(error); process.exit(1); });
