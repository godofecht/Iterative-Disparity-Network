/* Interactive research display. All experimental values originate in data/experiment.json.
 * The file is generated exclusively from verified outputs of the CI C++ training run.
 */
"use strict";

(function () {
  const NS = "http://www.w3.org/2000/svg";
  const PALETTE = { cyan: "#75ecd7", violet: "#a09bf7", amber: "#efbc74", grid: "#2b3b48", muted: "#8299a7" };
  const element = (id) => document.getElementById(id);
  const setText = (id, value) => { element(id).textContent = String(value); };
  const short = (value, digits = 3) => Number(value).toFixed(digits);
  let experiment = null;
  let selected = 0;
  let focus = 500;
  let timer = null;

  function svgNode(tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([name, value]) => node.setAttribute(name, String(value)));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function clear(host) { host.replaceChildren(); }

  function geometry(xValues, yValues, bounds) {
    const p = { left: 61, right: 16, top: 16, bottom: 38 };
    const innerW = bounds.width - p.left - p.right, innerH = bounds.height - p.top - p.bottom;
    let xMin = Math.min(...xValues), xMax = Math.max(...xValues);
    let yMin = Math.min(...yValues), yMax = Math.max(...yValues);
    if (!Number.isFinite(xMin + xMax + yMin + yMax)) throw new Error("Plot values must be finite");
    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const yPad = Math.max((yMax - yMin) * 0.1, 1e-7);
    yMin -= yPad; yMax += yPad;
    return {
      x: (v) => p.left + (v - xMin) / (xMax - xMin) * innerW,
      y: (v) => p.top + (yMax - v) / (yMax - yMin) * innerH,
      xMin, xMax, yMin, yMax, p, innerW, innerH
    };
  }

  function svgFrame(svg, geo, width, height, xLabel) {
    for (let t = 0; t < 5; t += 1) {
      const yv = geo.yMin + t * (geo.yMax - geo.yMin) / 4;
      const py = geo.y(yv);
      svg.appendChild(svgNode("line", { x1: geo.p.left, y1: py, x2: width - geo.p.right, y2: py, stroke: PALETTE.grid, "stroke-width": 1, "stroke-dasharray": "3,5" }));
      svg.appendChild(svgNode("text", { x: geo.p.left - 11, y: py + 4, fill: PALETTE.muted, "font-size": 11, "text-anchor": "end" }, Math.abs(yv) >= 100 ? yv.toFixed(0) : yv.toFixed(2)));
      const xv = geo.xMin + t * (geo.xMax - geo.xMin) / 4;
      svg.appendChild(svgNode("text", { x: geo.x(xv), y: height - 10, fill: PALETTE.muted, "font-size": 11, "text-anchor": "middle" }, xv.toFixed(0)));
    }
    svg.appendChild(svgNode("text", { x: width - geo.p.right, y: height - 9, fill: PALETTE.muted, "font-size": 10, "text-anchor": "end" }, xLabel));
  }

  function renderLines(hostId, series, opts = {}) {
    const host = element(hostId);
    clear(host);
    const width = 940, height = opts.height || 280;
    const xs = series.flatMap((item) => item.x);
    const ys = series.flatMap((item) => item.y);
    if (!xs.length || !ys.length) return;
    const geo = geometry(xs, ys, { width, height });
    const svg = svgNode("svg", { viewBox: "0 0 " + width + " " + height, width, height, preserveAspectRatio: "none", "aria-hidden": "true" });
    svg.style.height = height + "px";
    svgFrame(svg, geo, width, height, opts.xLabel || "POSITION");
    for (const item of series) {
      const n = Math.min(item.x.length, item.y.length);
      const chunks = [];
      for (let i = 0; i < n; i += 1) {
        chunks.push((i === 0 ? "M" : "L") + geo.x(item.x[i]).toFixed(2) + "," + geo.y(item.y[i]).toFixed(2));
      }
      svg.appendChild(svgNode("path", {
        d: chunks.join(" "), fill: "none", stroke: item.color,
        "stroke-width": item.width || 2.3, "stroke-linejoin": "round",
        "stroke-linecap": "round", opacity: item.opacity || 1, "vector-effect": "non-scaling-stroke"
      }));
    }
    if (Number.isFinite(opts.marker)) {
      const x = geo.x(opts.marker);
      svg.appendChild(svgNode("line", { x1: x, y1: geo.p.top, x2: x, y2: height - geo.p.bottom,
        stroke: PALETTE.amber, "stroke-width": 1.8, "stroke-dasharray": "5,6" }));
    }
    host.appendChild(svg);
  }

  function renderTopology() {
    const svg = element("hero-topology");
    clear(svg);
    const x = [38, 244, 452], inputY = Array.from({ length: 11 }, (_, i) => 14 + i * 20.5);
    const hiddenY = Array.from({ length: 10 }, (_, i) => 22 + i * 21.5);
    for (const a of inputY) for (const b of hiddenY)
      svg.appendChild(svgNode("line", { x1: x[0], y1: a, x2: x[1], y2: b, stroke: "#53878a", opacity: ".095", "stroke-width": .8 }));
    for (const b of hiddenY)
      svg.appendChild(svgNode("line", { x1: x[1], y1: b, x2: x[2], y2: 115, stroke: "#8f8cca", opacity: ".24", "stroke-width": 1.25 }));
    inputY.forEach((y) => svg.appendChild(svgNode("circle", { cx: x[0], cy: y, r: 4, fill: PALETTE.cyan, opacity: .74 })));
    hiddenY.forEach((y) => svg.appendChild(svgNode("circle", { cx: x[1], cy: y, r: 5.6, fill: "#1b313c", stroke: PALETTE.violet, "stroke-width": 1.5 })));
    svg.appendChild(svgNode("circle", { cx: x[2], cy: 115, r: 12, fill: "#224d48", stroke: PALETTE.cyan, "stroke-width": 2 }));
    svg.appendChild(svgNode("circle", { cx: x[2], cy: 115, r: 3.5, fill: PALETTE.cyan }));
  }

  function selectedCheckpoint() {
    return experiment.checkpoints[selected];
  }

  function focusRange() {
    const n = experiment.metrics.samples, size = Math.min(160, n);
    const start = Math.max(0, Math.min(n - size, focus - Math.floor(size / 2)));
    return { start, end: start + size };
  }

  function renderFocus() {
    const stride = experiment.meta.stride;
    const { start, end } = focusRange();
    const inputStart = start * stride, inputEnd = end * stride;
    const left = experiment.signals.left.slice(inputStart, inputEnd);
    const right = experiment.signals.right.slice(inputStart, inputEnd);
    const rawX = left.map((_, i) => i + inputStart);
    renderLines("signal-plot", [
      { x: rawX, y: left, color: PALETTE.cyan, opacity: .9, width: 1.6 },
      { x: rawX, y: right, color: PALETTE.violet, opacity: .9, width: 1.6 }
    ], { xLabel: "RAW PIXEL INDEX" });
    const x = Array.from({ length: end - start }, (_, i) => i + start);
    const cp = selectedCheckpoint();
    renderLines("disparity-plot", [
      { x, y: cp.predictions.slice(start, end), color: PALETTE.cyan, width: 2.4 },
      { x, y: experiment.signals.truth.slice(start, end), color: PALETTE.amber, width: 2.1, opacity: .83 }
    ], { xLabel: "SPATIAL POSITION" });
    setText("focus-label", String(focus + 1).padStart(4, "0") + " / " + experiment.metrics.samples);
    drawHeatmap();
    element("signal-plot").setAttribute("aria-label", "Raw left- and right-eye input signals from pixel " + inputStart + " to " + (inputEnd - 1));
    element("disparity-plot").setAttribute("aria-label", "Learned output and target disparity from spatial position " + start + " to " + (end - 1) + " at training epoch " + cp.epoch);
  }

  function drawHeatmap() {
    const canvas = element("hidden-map");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const width = Math.max(300, Math.round(canvas.getBoundingClientRect().width));
    const height = 190, dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0b1520"; ctx.fillRect(0, 0, width, height);
    const { start, end } = focusRange(), features = experiment.signals.final_hidden_activations;
    const n = end - start, left = 33, rowHeight = (height - 9) / features.length, cellWidth = (width - left - 3) / n;
    features.forEach((row, j) => {
      ctx.fillStyle = "#9ab5c1";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(String(j + 1).padStart(2, "0"), 5, (j + .6) * rowHeight);
      for (let i = 0; i < n; i += 1) {
        const value = Math.max(-1, Math.min(1, row[start + i]));
        const intensity = Math.round(34 + 178 * Math.abs(value));
        ctx.fillStyle = value >= 0
          ? "rgb(" + Math.round(intensity * .35) + "," + intensity + "," + Math.round(intensity * .88) + ")"
          : "rgb(" + Math.round(intensity * .74) + "," + Math.round(intensity * .55) + "," + intensity + ")";
        ctx.fillRect(left + i * cellWidth, j * rowHeight + 1, Math.ceil(cellWidth), rowHeight - 2);
      }
    });
    canvas.setAttribute("aria-label", "Final-network hidden activations for spatial positions " + start + " to " + end + ", ten rows, blue-green for positive and purple for negative");
  }

  function renderScatter() {
    const host = element("scatter-plot"); clear(host);
    const target = experiment.signals.truth, response = experiment.signals.final_prediction;
    const w = 940, h = 280, g = geometry(target, response, { width: w, height: h });
    const svg = svgNode("svg", { viewBox: "0 0 " + w + " " + h, width: w, height: h, preserveAspectRatio: "none", "aria-hidden": "true" });
    svg.style.height = h + "px";
    svgFrame(svg, g, w, h, "REFERENCE DISPARITY");
    for (let i = 0; i < target.length; i++) {
      svg.appendChild(svgNode("circle", { cx: g.x(target[i]).toFixed(2), cy: g.y(response[i]).toFixed(2),
        r: 2, fill: PALETTE.cyan, opacity: .42 }));
    }
    host.appendChild(svg);
  }

  function renderLearning() {
    const cp = selectedCheckpoint();
    const metric = experiment.history.find((item) => item.epoch === cp.epoch);
    if (!metric) throw new Error("Selected checkpoint has no matching optimization metric");
    setText("epoch-label", String(cp.epoch) + " / " + experiment.meta.completed_epochs);
    setText("selected-epoch", cp.epoch);
    setText("selected-merit", short(metric.merit, 4));
    setText("selected-corr", (cp.correlation >= 0 ? "+" : "") + short(cp.correlation, 4));
    setText("selected-abs", short(Math.abs(cp.correlation), 4));
    renderLines("training-plot", [{
      x: experiment.history.map((m) => m.epoch),
      y: experiment.history.map((m) => m.merit),
      color: PALETTE.cyan, width: 2.8
    }], { height: 320, marker: cp.epoch, xLabel: "TRAINING EPOCH" });
    element("training-plot").setAttribute("aria-label", "Objective F across " + experiment.meta.completed_epochs + " training epochs, current selected epoch " + cp.epoch + " with F equal to " + short(metric.merit, 4));
    renderFocus();
  }

  function stopPlay() {
    if (timer !== null) { window.clearInterval(timer); timer = null; }
    setText("play-button", "▶ Play evolution");
  }

  function advance() {
    if (selected >= experiment.checkpoints.length - 1) { stopPlay(); return; }
    selected += 1;
    element("epoch-slider").value = String(selected);
    renderLearning();
  }

  function bindControls() {
    const f = element("focus-slider"), e = element("epoch-slider"), play = element("play-button");
    f.max = String(experiment.metrics.samples - 1);
    f.value = String(Math.floor(experiment.metrics.samples / 2));
    f.disabled = false;
    focus = Number(f.value);
    e.max = String(experiment.checkpoints.length - 1);
    e.value = String(experiment.checkpoints.length - 1);
    selected = Number(e.value);
    e.disabled = false;
    play.disabled = false;
    f.addEventListener("input", () => { focus = Number(f.value); renderFocus(); });
    e.addEventListener("input", () => { stopPlay(); selected = Number(e.value); renderLearning(); });
    play.addEventListener("click", () => {
      if (timer !== null) { stopPlay(); return; }
      if (selected === experiment.checkpoints.length - 1) {
        selected = 0; e.value = "0"; renderLearning();
      }
      setText("play-button", "Ⅱ Pause");
      timer = window.setInterval(advance, 550);
    });
    window.addEventListener("pagehide", stopPlay);
    window.addEventListener("resize", drawHeatmap);
  }

  function applyData(data) {
    if (data.schema_version !== 1 || !data.history || !data.history.length ||
        !data.checkpoints || !data.checkpoints.length || !data.signals ||
        data.signals.truth.length !== data.metrics.samples) {
      throw new Error("Published CI result bundle is invalid");
    }
    experiment = data;
    setText("data-status", "● VERIFIED LIVE DATA  /  GitHub Actions run " + new URL(data.meta.run_url).pathname.split("/").pop() +
      "  /  SHA " + data.meta.commit.substring(0, 8));
    setText("hero-corr", short(data.metrics.absolute_correlation, 4));
    setText("metric-samples", data.metrics.samples.toLocaleString("en-GB"));
    setText("metric-initial", short(data.metrics.initial_merit, 4));
    setText("metric-final", short(data.metrics.final_merit, 4));
    setText("metric-epochs", data.meta.completed_epochs.toLocaleString("en-GB"));
    setText("commit-id", data.meta.commit.substring(0, 12));
    setText("generated-at", new Date(data.meta.generated_utc).toLocaleString("en-GB", {
      day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
      hour12: false, timeZone: "UTC"
    }) + " UTC");
    setText("experiment-seed", data.meta.seed);
    setText("left-hash", data.meta.dataset_sha256["image1.csv"]);
    setText("right-hash", data.meta.dataset_sha256["image2.csv"]);
    setText("reference-hash", data.meta.dataset_sha256["disparity.csv"]);
    const run = new URL(data.meta.run_url);
    if (run.host === "github.com" && run.pathname.startsWith("/godofecht/Iterative-Disparity-Network/actions/runs/")) {
      element("run-link").href = run.href;
    }
    renderTopology();
    bindControls();
    renderScatter();
    renderLearning();
  }

  async function load() {
    try {
      const response = await fetch("./data/experiment.json", { cache: "no-store" });
      if (!response.ok) throw new Error("HTTP " + response.status);
      applyData(await response.json());
    } catch (error) {
      const status = element("data-status");
      status.classList.add("error");
      status.textContent = "Results unavailable: the validated CI data bundle could not be loaded (" + error.message + "). View the source run instead.";
      renderTopology();
    }
  }
  load();
})();
