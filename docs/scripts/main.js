import { unmountCharts } from "./charts/mount.js";
import { hideTooltip } from "./charts/tooltip.js";
import { mountField } from "./field-section.js";
import { reportOf } from "./sections.js";
import { StudyRuntime } from "./study-client.js";
import { readState, studyRequest, writeState } from "./url-state.js";

const form = document.querySelector("#controls");
const status = document.querySelector("#status");
const report = document.querySelector("#report");
const runButton = form.querySelector(".run-button");
const fields = form.elements;

let state = readState(window.location.search);
let study = null;

function announce(message, failed = false) {
  status.textContent = message;
  status.dataset.state = failed ? "failed" : "working";
}

function showState() {
  fields.seed.value = state.seed;
  fields.holdout.value = state.holdout;
  fields.holdoutValue.value = state.holdout;
  fields.target.value = state.target;
  fields.thorough.checked = state.thorough;
  fields.measure.value = state.measure;
}

function stateFromForm() {
  const query = new URLSearchParams({
    seed: fields.seed.value,
    holdout: fields.holdout.value,
    target: fields.target.value,
    thorough: fields.thorough.checked ? "1" : "0",
    measure: fields.measure.value,
    region: fields.region.value,
  });
  return readState(`?${query}`);
}

function rememberState() {
  const address = `${window.location.pathname}${writeState(state)}`;
  window.history.replaceState(null, "", address);
}

function offerRegions(regions) {
  const chosen = regions.includes(state.region) ? state.region : regions[0];
  const options = regions.map((region) => new Option(region, region, false, region === chosen));
  fields.region.replaceChildren(...options);
  return chosen;
}

function draw() {
  const region = offerRegions(study.regions);
  unmountCharts();
  hideTooltip();
  report.replaceChildren(...reportOf(study, { measure: state.measure, region }));
}

function setBusy(busy) {
  report.setAttribute("aria-busy", String(busy));
  runButton.disabled = busy;
}

async function runStudy(runtime) {
  setBusy(true);
  const started = performance.now();
  try {
    study = await runtime.run(studyRequest(state));
    draw();
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    announce(`Study computed in your browser in ${seconds} s.`);
  } catch (error) {
    announce(`The study could not be computed: ${error.message}`, true);
  } finally {
    setBusy(false);
  }
}

function start() {
  const runtime = new StudyRuntime((message) => announce(message));
  showState();

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    state = stateFromForm();
    showState();
    rememberState();
    announce("Running the study…");
    runStudy(runtime);
  });

  fields.holdout.addEventListener("input", () => {
    fields.holdoutValue.value = fields.holdout.value;
  });

  [fields.measure, fields.region].forEach((field) =>
    field.addEventListener("change", () => {
      state = { ...state, measure: fields.measure.value, region: fields.region.value };
      rememberState();
      if (study !== null) {
        draw();
      }
    }),
  );

  mountField(document.querySelector("#field"), runtime);
  runStudy(runtime);
}

start();
