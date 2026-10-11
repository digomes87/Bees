import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const WHEEL = "../package/bee_migration_analysis-1.0.0-py3-none-any.whl";
const PACKAGES = ["numpy", "pandas", "scikit-learn"];

function report(message) {
  self.postMessage({ kind: "status", message });
}

async function wheelBytes() {
  const response = await fetch(new URL(WHEEL, import.meta.url));
  if (!response.ok) {
    throw new Error(`Could not download the analysis package (HTTP ${response.status})`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/** Start Python and load the analysis package; resolves to its JSON functions. */
async function startPython() {
  report("Starting the Python runtime…");
  const pyodide = await loadPyodide();
  report("Loading numpy, pandas and scikit-learn…");
  await pyodide.loadPackage(PACKAGES, { messageCallback: () => {} });
  report("Loading the analysis package…");
  pyodide.unpackArchive(await wheelBytes(), "wheel");
  const study = pyodide.pyimport("bee_migration.web.study");
  const field = pyodide.pyimport("bee_migration.web.field");
  return {
    run_study: study.run_study,
    field_requests: field.field_requests,
    field_report: field.field_report,
  };
}

/** Call a Python function that takes and returns JSON text. */
function answerOf(python, argument) {
  if (argument === undefined) {
    return JSON.parse(python());
  }
  return JSON.parse(python(JSON.stringify(argument)));
}

/** The last line of a Python traceback is the message meant for people. */
function readable(error) {
  const lines = String(error?.message ?? error).trim().split("\n");
  return lines.at(-1);
}

const functions = startPython();

functions.then(
  () => self.postMessage({ kind: "ready" }),
  (error) => self.postMessage({ kind: "failed", message: readable(error) }),
);

self.addEventListener("message", async ({ data }) => {
  try {
    const python = (await functions)[data.call];
    self.postMessage({ kind: "answer", id: data.id, value: answerOf(python, data.argument) });
  } catch (error) {
    self.postMessage({ kind: "failed", id: data.id, message: readable(error) });
  }
});
