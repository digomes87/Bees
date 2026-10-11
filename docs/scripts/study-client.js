/** Runs the Python package in a background worker so the page stays responsive. */
export class StudyRuntime {
  #worker;
  #pending = new Map();
  #lastId = 0;

  /** @param {(message: string) => void} onStatus called with progress messages */
  constructor(onStatus) {
    this.#worker = new Worker(new URL("./study-worker.js", import.meta.url), { type: "module" });
    this.#worker.addEventListener("message", ({ data }) => this.#receive(data, onStatus));
    this.#worker.addEventListener("error", () => this.#failAll("The Python runtime could not start."));
  }

  /** @returns {Promise<object>} the study computed for `request` */
  run(request) {
    return this.call("run_study", request);
  }

  /**
   * Call one of the package's JSON functions.
   * @param {"run_study" | "field_requests" | "field_report"} name
   * @param {object} [argument] sent to Python as JSON
   * @returns {Promise<object>}
   */
  call(name, argument) {
    this.#lastId += 1;
    const id = this.#lastId;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, call: name, argument });
    });
  }

  #receive(data, onStatus) {
    if (data.kind === "status") {
      onStatus(data.message);
      return;
    }
    if (data.kind === "failed" && data.id === undefined) {
      this.#failAll(data.message);
      return;
    }
    this.#settle(data);
  }

  #settle(data) {
    const waiting = this.#pending.get(data.id);
    if (waiting === undefined) {
      return;
    }
    this.#pending.delete(data.id);
    if (data.kind === "failed") {
      waiting.reject(new Error(data.message));
      return;
    }
    waiting.resolve(data.value);
  }

  #failAll(message) {
    this.#pending.forEach(({ reject }) => reject(new Error(message)));
    this.#pending.clear();
  }
}
