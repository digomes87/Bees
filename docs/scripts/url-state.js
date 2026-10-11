export const TARGETS = ["bee_abundance", "species_diversity", "honey_production"];
export const MEASURES = ["temperature", "precipitation", ...TARGETS];
export const LIMITS = { seed: [0, 9999], holdout: [2, 8] };
export const DEFAULTS = Object.freeze({
  seed: 42,
  holdout: 4,
  target: "bee_abundance",
  thorough: false,
  measure: "temperature",
  region: "",
});

function boundedInteger(text, name) {
  const [lowest, highest] = LIMITS[name];
  const value = Number.parseInt(text, 10);
  if (Number.isNaN(value)) {
    return DEFAULTS[name];
  }
  return Math.min(Math.max(value, lowest), highest);
}

function oneOf(text, allowed, fallback) {
  return allowed.includes(text) ? text : fallback;
}

/** Read the study and view settings from a location search string. */
export function readState(search) {
  const query = new URLSearchParams(search);
  return {
    seed: boundedInteger(query.get("seed"), "seed"),
    holdout: boundedInteger(query.get("holdout"), "holdout"),
    target: oneOf(query.get("target"), TARGETS, DEFAULTS.target),
    thorough: query.get("thorough") === "1",
    measure: oneOf(query.get("measure"), MEASURES, DEFAULTS.measure),
    region: query.get("region") ?? DEFAULTS.region,
  };
}

function serialized(value) {
  return value === true ? "1" : String(value);
}

/** Write the settings that differ from the defaults as a search string. */
export function writeState(state) {
  const changed = Object.keys(DEFAULTS).filter((name) => state[name] !== DEFAULTS[name]);
  const query = new URLSearchParams(changed.map((name) => [name, serialized(state[name])]));
  return changed.length === 0 ? "" : `?${query}`;
}

export function studyRequest(state) {
  return {
    seed: state.seed,
    holdout_years: state.holdout,
    target: state.target,
    thorough: state.thorough,
  };
}
