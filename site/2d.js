/* Paper-scale 2D observatory. Every displayed datum is loaded from CI results. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const COLORS = {cyan:"#75ecd7", amber:"#efbc74", violet:"#a09bf7",
                  grid:"#354654", text:"#a4b9c2"};
  const fmt = (value, n=3) => Number(value).toFixed(n);
  let report, scene = "train", orientationScene = "train", currentIndex = 0, timer = null;

  function text(id, value) { $(id).textContent = String(value); }
  function readyStatus(message, error=false) {
    const p = $("lab-status");
    p.textContent = message;
    p.classList.toggle("error", error);
  }
  function numbers(array, length, name) {
    if (!Array.isArray(array) || array.length !== length ||
        array.some((item) => typeof item !== "number" || !Number.isFinite(item))) {
      throw new Error("Invalid numeric array: " + name);
    }
  }
  function validate(data) {
    if (!data || data.schema_version !== 1 || !data.stereo || !data.orientation ||
        !data.provenance || !data.replications) {
      throw new Error("Missing verified two-dimensional experiment data");
    }
    if (data.stereo.meta.size !== 120 || data.stereo.meta.grid !== 40 ||
        data.orientation.meta.size !== 600 || data.orientation.meta.grid !== 120) {
      throw new Error("Experiment geometry does not match the published methodology");
    }
    const ns = 40 * 40, no = 120 * 120;
    for (const key of ["train", "test"]) {
      const s = data.stereo.scenes[key];
      numbers(s.left, 120*120, key+" left image");
      numbers(s.right, 120*120, key+" right image");
      numbers(s.truth, ns, key+" known disparity");
      numbers(s.output, ns, key+" learned output");
    }
    if (!Array.isArray(data.stereo.checkpoints) || !data.stereo.checkpoints.length)
      throw new Error("Real network checkpoints are unavailable");
    for (const state of data.stereo.checkpoints) {
      numbers(state.train, ns, "training checkpoint");
      numbers(state.test, ns, "held-out checkpoint");
    }
    if (data.stereo.checkpoints.at(-1).epoch !== data.stereo.meta.epochs_completed ||
        data.stereo.checkpoints[0].epoch !== 0)
      throw new Error("Invalid recorded learning interval");
    if (!Array.isArray(data.stereo.history) || !Array.isArray(data.orientation.history))
      throw new Error("Missing numerical training traces");
    const image = data.orientation.images;
    numbers(image.train_input, 600*600, "orientation texture");
    numbers(image.test_input, 600*600, "held-out orientation texture");
    numbers(image.train_output, no, "orientation output");
    numbers(image.test_output, no, "held-out orientation output");
    numbers(image.train_orientation, no, "orientation reference");
    numbers(image.test_orientation, no, "held-out orientation reference");
    if (!Array.isArray(image.train_valid) || image.train_valid.length !== no ||
        !Array.isArray(image.test_valid) || image.test_valid.length !== no) {
      throw new Error("Evaluation-only orientation mask is unavailable");
    }
  }

  function drawHeat(id, values, size, type="field", options={}) {
    const canvas = $(id), height = size, width = size;
    const staging = document.createElement("canvas");
    staging.width = width; staging.height = height;
    const ctx = staging.getContext("2d");
    const pixels = ctx.createImageData(width, height);
    const output = pixels.data;
    const finite = values.filter(Number.isFinite);
    const mean = finite.reduce((acc, value) => acc + value, 0) / finite.length;
    const sd = Math.sqrt(finite.reduce((acc, v) => acc + (v-mean)*(v-mean),0) / finite.length) || 1;
    const sign = options.sign || 1;
    for (let i = 0; i < values.length; i++) {
      let r, g, b;
      const value = values[i];
      if (options.mask && !options.mask[i]) {
        r=13; g=22; b=32;
      } else if (type === "grayscale") {
        const brightness = Math.round(220 * Math.max(0, Math.min(1, .5 + .21*value)));
        r = brightness; g=brightness; b=brightness;
      } else {
        const standardized = options.angle ? Math.cos(value*2) :
          Math.max(-2.8, Math.min(2.8, sign*(value - mean)/sd));
        const t = Math.max(-1, Math.min(1, standardized / (options.angle ? 1 : 2.4)));
        const strength = Math.abs(t);
        if (t >= 0) {
          r = Math.round(22 + 94*strength);
          g = Math.round(43 + 184*strength);
          b = Math.round(56 + 147*strength);
        } else {
          r = Math.round(22 + 89*strength);
          g = Math.round(43 + 88*strength);
          b = Math.round(56 + 157*strength);
        }
      }
      const offset = i*4;
      output[offset]=r; output[offset+1]=g; output[offset+2]=b; output[offset+3]=255;
    }
    ctx.putImageData(pixels, 0, 0);
    const context = canvas.getContext("2d");
    canvas.width = canvas.width; // Drop stale canvas contents before each frame.
    context.imageSmoothingEnabled = false;
    context.drawImage(staging, 0, 0, canvas.width, canvas.height);
  }

  function paintStereo() {
    const state = report.stereo.scenes[scene];
    const snapshot = report.stereo.checkpoints[currentIndex];
    const r = scene === "train"
      ? report.stereo.history.find((h) => h.epoch === snapshot.epoch).train_r
      : report.stereo.history.find((h) => h.epoch === snapshot.epoch).test_r;
    drawHeat("stereo-left", state.left, 120, "grayscale");
    drawHeat("stereo-right", state.right, 120, "grayscale");
    drawHeat("stereo-target", state.truth, 40);
    drawHeat("stereo-output", snapshot[scene], 40, "field", {sign: r < 0 ? -1 : 1});
    $("stereo-output").setAttribute("aria-label",
      "Actually computed " + scene + " network response at optimization epoch " + snapshot.epoch);
    drawHeat("hero-map", report.stereo.scenes.train.output, 40, "field",
             {sign: report.stereo.metrics.train_r < 0 ? -1 : 1});
  }

  function buttonState(activeId, inactiveId) {
    $(activeId).classList.add("active"); $(activeId).setAttribute("aria-pressed","true");
    $(inactiveId).classList.remove("active"); $(inactiveId).setAttribute("aria-pressed","false");
  }
  function paintOrientation() {
    const train = orientationScene === "train";
    const image = report.orientation.images;
    drawHeat("orientation-input", image[train ? "train_input" : "test_input"], 600, "grayscale");
    drawHeat("orientation-axes", image[train ? "train_orientation" : "test_orientation"], 120,
             "field", {angle:true, mask:image[train ? "train_valid" : "test_valid"]});
    drawHeat("orientation-output", image[train ? "train_output" : "test_output"], 120);
  }
  function plotLearning() {
    const canvas = $("stereo-learning-plot");
    const width = 1160, height = 260, p = {left:48,right:48,top:16,bottom:34};
    canvas.width=width; canvas.height=height;
    const ctx = canvas.getContext("2d");
    const hist = report.stereo.history, last = hist.at(-1).epoch;
    const mmin = Math.min(0, ...hist.map(x => x.merit));
    const mmax = Math.max(1, ...hist.map(x => x.merit));
    const x = epoch => p.left + (width-p.left-p.right)*epoch/Math.max(1,last);
    const ym = merit => p.top + (height-p.top-p.bottom)*(mmax-merit)/Math.max(1e-12,mmax-mmin);
    const yr = r => p.top + (height-p.top-p.bottom)*(1-r);
    for(let i=0;i<=4;i++){
      const py=p.top+(height-p.top-p.bottom)*i/4;
      ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;ctx.setLineDash([3,6]);
      ctx.beginPath();ctx.moveTo(p.left,py);ctx.lineTo(width-p.right,py);ctx.stroke();
      ctx.setLineDash([]);
      ctx.font="11px system-ui";ctx.fillStyle=COLORS.text;
      ctx.textAlign="right";ctx.fillText(fmt(mmax-(mmax-mmin)*i/4,2),p.left-9,py+4);
      ctx.textAlign="left";ctx.fillText(fmt(1-i/4,2),width-p.right+8,py+4);
    }
    function curve(key,color,toY,transform) {
      ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.lineJoin="round";
      hist.forEach((pt,i) => {
        const px=x(pt.epoch),py=toY(transform(pt[key]));
        if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);
      });
      ctx.stroke();
    }
    curve("merit",COLORS.cyan,ym,v=>v);
    curve("train_r",COLORS.amber,yr,Math.abs);
    curve("test_r",COLORS.violet,yr,Math.abs);
    const epoch = report.stereo.checkpoints[currentIndex].epoch;
    const px=x(epoch);
    ctx.strokeStyle="#e7edef";ctx.lineWidth=1.4;ctx.setLineDash([6,5]);
    ctx.beginPath();ctx.moveTo(px,p.top);ctx.lineTo(px,height-p.bottom);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=COLORS.text;ctx.textAlign="center";
    [0,.25,.5,.75,1].forEach(f => ctx.fillText(String(Math.round(last*f)),x(last*f),height-9));
    ctx.fillText("F",16,17);ctx.fillText("|r|",width-19,17);
    canvas.setAttribute("aria-label","Training history: merit "+fmt(hist.at(-1).merit)+
      ", final training |r| "+fmt(Math.abs(hist.at(-1).train_r))+
      ", final held-out |r| "+fmt(Math.abs(hist.at(-1).test_r))+
      ". Selected epoch "+epoch);
  }
  function paintProgress() {
    const snapshot=report.stereo.checkpoints[currentIndex];
    const metric=report.stereo.history.find(m=>m.epoch===snapshot.epoch);
    if(!metric)throw new Error("Checkpoint missing associated training metrics");
    text("epoch-current",snapshot.epoch+" / "+report.stereo.meta.epochs_completed);
    text("epoch-merit",fmt(metric.merit,4));
    text("epoch-train-r",fmt(Math.abs(metric.train_r),4));
    text("epoch-test-r",fmt(Math.abs(metric.test_r),4));
    plotLearning(); paintStereo();
  }

  function stop() {
    if(timer!==null){clearInterval(timer);timer=null;}
    text("play-2d","▶ Play learning");
  }
  function step() {
    currentIndex+=1;
    if(currentIndex>=report.stereo.checkpoints.length){currentIndex=report.stereo.checkpoints.length-1;stop();return;}
    $("stereo-epoch").value=String(currentIndex);
    paintProgress();
  }
  function seedTable(id, rows, firstKey, secondKey, stat1, stat2) {
    const host=$(id);host.replaceChildren();
    const table=document.createElement("table"),thead=document.createElement("thead"),
          tbody=document.createElement("tbody"),tfoot=document.createElement("tfoot");
    const head=document.createElement("tr");
    for(const heading of ["Seed","Training","Unseen"]){
      const th=document.createElement("th");th.scope="col";th.textContent=heading;head.appendChild(th);
    }
    thead.appendChild(head);
    for(const row of rows){
      const tr=document.createElement("tr");
      for(const cell of [String(row.seed),fmt(row[firstKey],4),fmt(row[secondKey],4)]){
        const td=document.createElement("td");td.textContent=cell;tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    const tr=document.createElement("tr");
    for(const cell of ["Mean ± SD",fmt(stat1.mean,4)+" ± "+fmt(stat1.sd,4),
                      fmt(stat2.mean,4)+" ± "+fmt(stat2.sd,4)]){
      const td=document.createElement("td");td.textContent=cell;tr.appendChild(td);
    }
    tfoot.appendChild(tr);
    table.append(thead,tbody,tfoot);host.appendChild(table);
  }
  function bind() {
    $("scene-train").addEventListener("click",()=>{
      scene="train";buttonState("scene-train","scene-test");paintStereo();
    });
    $("scene-test").addEventListener("click",()=>{
      scene="test";buttonState("scene-test","scene-train");paintStereo();
    });
    $("orientation-scene-train").addEventListener("click",()=>{
      orientationScene="train";buttonState("orientation-scene-train","orientation-scene-test");paintOrientation();
    });
    $("orientation-scene-test").addEventListener("click",()=>{
      orientationScene="test";buttonState("orientation-scene-test","orientation-scene-train");paintOrientation();
    });
    const slider=$("stereo-epoch"),play=$("play-2d");
    slider.max=report.stereo.checkpoints.length-1;
    slider.value=slider.max; currentIndex=Number(slider.value);
    slider.disabled=false;play.disabled=false;
    slider.addEventListener("input",()=>{stop();currentIndex=Number(slider.value);paintProgress();});
    play.addEventListener("click",()=>{
      if(timer!==null){stop();return;}
      if(currentIndex===report.stereo.checkpoints.length-1){
        currentIndex=0;slider.value="0";paintProgress();
      }
      text("play-2d","Ⅱ Pause");
      timer=setInterval(step,400);
    });
    window.addEventListener("pagehide",stop);
  }

  async function load() {
    try {
      const response=await fetch("./data/experiment2d.json",{cache:"no-store"});
      if(!response.ok)throw new Error("HTTP "+response.status);
      const data=await response.json();validate(data);report=data;
      const sm=data.stereo.metrics, om=data.orientation.metrics;
      text("train-r",fmt(sm.train_abs_r,4));text("test-r",fmt(sm.test_abs_r,4));
      text("orientation-train",fmt(om.training_orientation_r2,4));
      text("orientation-test",fmt(om.test_orientation_r2,4));
      text("orientation-r2-train",fmt(om.training_orientation_r2,4));
      text("orientation-r2-test",fmt(om.test_orientation_r2,4));
      seedTable("stereo-seeds",data.replications.stereo,"train_abs_r","test_abs_r",
                data.replications.summary.stereo_train_abs_r,
                data.replications.summary.stereo_test_abs_r);
      seedTable("orientation-seeds",data.replications.orientation,
                "training_orientation_r2","test_orientation_r2",
                data.replications.summary.orientation_train_r2,
                data.replications.summary.orientation_test_r2);
      const run=new URL(data.provenance.run_url);
      if(run.host==="github.com" &&
         run.pathname.startsWith("/godofecht/Iterative-Disparity-Network/actions/runs/")){
        $("2d-run-link").href=run.href;
      }
      text("2d-commit",data.provenance.commit.substring(0,12));
      readyStatus("● VERIFIED MEASUREMENT / GitHub Actions run "+
                  run.pathname.split("/").pop()+" / 3 predefined seeds / "+data.provenance.commit.substring(0,8));
      bind();paintProgress();paintOrientation();
    }catch(error){
      readyStatus("Two-dimensional CI measurements are unavailable: "+error.message+
                  ". Inspect the latest GitHub Actions run.",true);
    }
  }
  load();
})();
