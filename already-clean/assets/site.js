"use strict";

const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("en-US");
const NS = "http://www.w3.org/2000/svg";
const node = (name, cls, text) => {
  const element = document.createElement(name);
  if (cls) element.className = cls;
  if (text !== undefined) element.textContent = text;
  return element;
};
function svgNode(name, attributes, text) {
  const element = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
async function getData(path) {
  // The review report embeds its data so it also works when opened as a local file.
  if (window.REPORT_DATA && Object.hasOwn(window.REPORT_DATA, path)) return window.REPORT_DATA[path];
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Data request failed (${response.status})`);
  return response.json();
}

function drawHero(data) {
  const chart = byId("hero-curve");
  const rows = data.curves.filter((row) => row.group === "random" && row.metric === "semantic_to_final");
  const x = (v) => 43 + (v / 1413814) * 304;
  const y = (v) => 247 - v * 203;
  for (const v of [0, 0.5, 1]) {
    chart.append(svgNode("line", { x1: 43, x2: 347, y1: y(v), y2: y(v), stroke: "#d9ddd5", "stroke-width": 1 }));
    chart.append(svgNode("text", { x: 32, y: y(v) + 4, "text-anchor": "end" }, v.toFixed(1)));
  }
  for (const [v, label] of [[0, "0"], [560000, "560k"], [1413814, "1.41m"]]) {
    chart.append(svgNode("text", { x: x(v), y: 270, "text-anchor": v === 0 ? "start" : v === 1413814 ? "end" : "middle" }, label));
  }
  chart.append(svgNode("text", { x: 43, y: 26 }, "Similarity to final checkpoint"));
  chart.append(svgNode("text", { x: 347, y: 292, "text-anchor": "end" }, "Training step"));
  const upper = rows.map((r) => `${x(r.step)},${y(r.upper)}`);
  const lower = [...rows].reverse().map((r) => `${x(r.step)},${y(r.lower)}`);
  chart.append(svgNode("polygon", { points: [...upper, ...lower].join(" "), fill: "#d7e6d8" }));
  chart.append(svgNode("polyline", { points: rows.map((r) => `${x(r.step)},${y(r.mean)}`).join(" "), fill: "none", stroke: "#174f45", "stroke-width": 2.4, "stroke-linejoin": "round" }));
  for (const row of rows) chart.append(svgNode("circle", { cx: x(row.step), cy: y(row.mean), r: 2.4, fill: "#174f45", stroke: "#f7f6f1", "stroke-width": .7 }));
  const late = rows.find((r) => r.step === 560000);
  chart.append(svgNode("line", { x1: x(late.step), x2: x(late.step), y1: y(late.mean) + 8, y2: 247, stroke: "#8eac9a", "stroke-dasharray": "3 4" }));
  chart.append(svgNode("circle", { cx: x(late.step), cy: y(late.mean), r: 5, fill: "#174f45", stroke: "#f7f6f1", "stroke-width": 2 }));
  chart.append(svgNode("rect", { x: x(late.step) - 29, y: y(late.mean) - 37, width: 58, height: 25, rx: 2, fill: "#174f45" }));
  const label = svgNode("text", { x: x(late.step), y: y(late.mean) - 20, "text-anchor": "middle", style: "fill:#fffefa;font-size:12px;font-family:ui-monospace,monospace" }, late.mean.toFixed(3));
  chart.append(label);
}

function drawMasks(data) {
  const target = byId("mask-comparison");
  for (const [label, result] of [["Original text", data.similarity.tests.primary_random_same_layer], ["Insert [MASK]", data.similarity.tests.masked]]) {
    const row = node("div", "mask-row");
    row.append(node("div", "mask-label", label));
    const bars = node("div", "mask-bars");
    for (const [value, cls, name] of [[result.mean_matched, "mask-bar", "Own final"], [result.mean_mismatched, "mask-bar control", "Other same-layer final"]]) {
      const bar = node("div", cls);
      bar.style.width = `${value * 100}%`;
      bar.setAttribute("aria-label", `${name}: ${value.toFixed(3)}`);
      bar.append(node("span", null, value.toFixed(3)));
      bars.append(bar);
    }
    row.append(bars);
    const difference = node("div", "mask-gap", `+${result.mean.toFixed(3)}`);
    difference.append(node("small", null, "difference"));
    row.append(difference);
    target.append(row);
  }
}

function showQuestions(data) {
  const target = byId("question-list");
  target.replaceChildren();
  data.questions.forEach((q, i) => {
    const details = node("details", "question");
    details.id = `question-${q.id}`;
    if (i === 0) details.open = true;
    const summary = node("summary");
    summary.append(node("span", "question-number", String(i + 1).padStart(2, "0")));
    const title = node("div");
    title.append(node("span", "eyebrow", q.priority), node("h3", null, q.title));
    const plus = node("span", "plus", "+");
    plus.setAttribute("aria-hidden", "true");
    summary.append(title, plus);
    const body = node("div", "question-body");
    body.append(node("p", null, q.why));
    const dl = node("dl");
    for (const [term, value] of [["The experiment", q.test], ["What we need", q.needs], ["What would weaken the interpretation", q.failure]]) {
      dl.append(node("dt", null, term), node("dd", null, value));
    }
    body.append(dl, node("span", "question-status", q.status));
    details.append(summary, body);
    target.append(details);
  });
}

let explorerData;
let showAll = false;
let currentPole = "L03N6372+";
let currentCheckpoint = 13;

function fullText(context) { return context.join(""); }
function activationText(value) {
  if (value === 0) return "0";
  return Math.abs(value) < 0.01 || Math.abs(value) >= 1000 ? value.toPrecision(3) : Number(value.toPrecision(4)).toString();
}
function contextList(id, neuron, checkpoint, shared) {
  const list = byId(id);
  list.replaceChildren();
  const examples = neuron.steps[checkpoint].top;
  examples.slice(0, showAll ? 20 : 5).forEach(([index, activation], rank) => {
    const context = neuron.ctx[index];
    const repeated = shared.has(fullText(context));
    const li = node("li", repeated ? "repeat" : "");
    li.append(node("span", "context-rank", String(rank + 1).padStart(2, "0")));
    const text = node("span", "context-text");
    text.append(document.createTextNode(context[0]), node("mark", null, context[1]), document.createTextNode(context[2]));
    if (repeated) {
      const indication = node("span", "sr-only", " (Exact context also present in the other checkpoint.)");
      text.append(indication);
    }
    li.append(text, node("span", "context-activation", activationText(activation)));
    list.append(li);
  });
  list.scrollTop = 0;
}
function renderExplorer() {
  const n = explorerData.neurons.find((v) => v.pole === currentPole);
  const row = n.similarities.find((r) => r.checkpoint === currentCheckpoint);
  const step = explorerData.steps[currentCheckpoint];
  byId("pole-heading").textContent = n.pole;
  byId("pole-select").value = n.pole;
  byId("step-slider").value = currentCheckpoint;
  byId("step-slider").setAttribute("aria-valuetext", `${number.format(step)} training steps; checkpoint ${currentCheckpoint + 1} of 16`);
  byId("step-value").textContent = number.format(step);
  byId("checkpoint-value").textContent = `Checkpoint ${currentCheckpoint + 1} of 16`;
  byId("earlier-label").textContent = `Selected · ${number.format(step)} steps`;
  byId("semantic-value").textContent = row.semantic_to_final.toFixed(3);
  byId("overlap-value").textContent = row.context_jaccard_to_final.toFixed(3);
  byId("activation-value").textContent = activationText(n.steps[currentCheckpoint].max);
  byId("sample-note").textContent = n.group === "random"
    ? "Random viewer subset · 24 poles sampled after excluding 17 outcome-selected poles. This is not a whole-network sample."
    : n.group === "lithium"
      ? "Lithium illustration · selected using the observed outcome. Explore it as an example, not as a representative neuron."
      : "Late-pole illustration · selected using the final outcome. The primary new analysis uses the separate random subset.";
  const selectedTexts = new Set(n.steps[currentCheckpoint].top.map(([index]) => fullText(n.ctx[index])));
  const finalTexts = new Set(n.steps[15].top.map(([index]) => fullText(n.ctx[index])));
  const shared = new Set([...selectedTexts].filter((text) => finalTexts.has(text)));
  contextList("earlier-contexts", n, currentCheckpoint, shared);
  contextList("final-contexts", n, 15, shared);
  byId("show-contexts").textContent = showAll ? "Show first 5 contexts ↑" : "Show all 20 contexts ↓";
  for (const button of document.querySelectorAll("[data-step]")) {
    const active = Number(button.dataset.step) === currentCheckpoint;
    button.classList.toggle("selected", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

function initExplorer(data) {
  explorerData = data;
  const select = byId("pole-select");
  select.replaceChildren();
  for (const [group, label] of [["lithium", "Selected illustration"], ["random", "Random subset · 24 poles"], ["late", "Outcome-selected late poles"]]) {
    const optgroup = node("optgroup");
    optgroup.label = label;
    for (const n of data.neurons.filter((n) => n.group === group)) {
      const option = node("option", null, `${n.pole}${group === "lithium" ? " · lithium example" : ""}`);
      option.value = n.pole;
      optgroup.append(option);
    }
    select.append(optgroup);
  }
  const query = new URLSearchParams(location.search);
  if (data.neurons.some((n) => n.pole === query.get("pole"))) currentPole = query.get("pole");
  if (query.has("step")) {
    const index = data.steps.indexOf(Number(query.get("step")));
    if (index >= 0) currentCheckpoint = index;
  }
  select.disabled = false;
  byId("step-slider").disabled = false;
  select.addEventListener("change", () => { currentPole = select.value; renderExplorer(); });
  byId("step-slider").addEventListener("input", (event) => { currentCheckpoint = Number(event.target.value); renderExplorer(); });
  document.querySelectorAll("[data-step]").forEach((button) => button.addEventListener("click", () => {
    currentCheckpoint = Number(button.dataset.step); renderExplorer();
  }));
  byId("show-contexts").addEventListener("click", () => { showAll = !showAll; renderExplorer(); });
  byId("share-view").addEventListener("click", () => {
    const url = new URL(location.href);
    url.searchParams.set("pole", currentPole);
    url.searchParams.set("step", data.steps[currentCheckpoint]);
    url.hash = "explorer";
    history.replaceState(null, "", url);
    copyText(url.href, byId("share-view"), byId("share-status"), "Copy this view ↗");
  });
  renderExplorer();
  byId("explorer-panel").setAttribute("aria-busy", "false");
}

async function copyText(text, button, status, original) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied ✓";
    status.textContent = "Copied to clipboard.";
  } catch {
    button.textContent = "Select and copy manually";
    status.textContent = `Clipboard unavailable. ${text}`;
  }
  setTimeout(() => { button.textContent = original; }, 2200);
}
byId("copy-command").addEventListener("click", () => copyText(byId("run-command").textContent, byId("copy-command"), byId("copy-status"), "Copy command"));

let scrollQueued = false;
function updateReadingPosition() {
  const max = document.documentElement.scrollHeight - innerHeight;
  byId("progress").style.width = `${max > 0 ? Math.min(100, scrollY / max * 100) : 0}%`;
  const sections = [...document.querySelectorAll("main > section")];
  let active = sections[0].id;
  for (const section of sections) if (section.getBoundingClientRect().top < innerHeight * .3) active = section.id;
  for (const link of document.querySelectorAll(".rail nav a")) {
    const selected = link.hash === `#${active}`;
    link.classList.toggle("active", selected);
    if (selected) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
  scrollQueued = false;
}
addEventListener("scroll", () => { if (!scrollQueued) { scrollQueued = true; requestAnimationFrame(updateReadingPosition); } }, { passive: true });
addEventListener("resize", updateReadingPosition);
updateReadingPosition();

getData("assets/results.json").then((data) => {
  drawHero(data); drawMasks(data); showQuestions(data);
  document.documentElement.dataset.resultsReady = "true";
}).catch((error) => {
  byId("hero-chart-desc").textContent = "Chart data could not be loaded. All numerical results remain available in the linked report.";
  byId("question-list").firstElementChild.prepend(document.createTextNode("Interactive data could not be loaded. "));
  console.error(error);
});
getData("assets/explorer.json").then(initExplorer).catch((error) => {
  byId("explorer-error").hidden = false;
  byId("explorer-error").textContent = "The context data could not be loaded. Reload the page, or open the linked source data below.";
  byId("explorer-panel").setAttribute("aria-busy", "false");
  console.error(error);
});
