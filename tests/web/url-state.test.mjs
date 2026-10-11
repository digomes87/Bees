import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULTS, readState, studyRequest, writeState } from "../../docs/scripts/url-state.js";

test("an empty address yields the default study", () => {
  assert.deepEqual(readState(""), DEFAULTS);
});

test("settings are read from the address", () => {
  const state = readState("?seed=7&holdout=6&target=honey_production&thorough=1&measure=precipitation&region=South");

  assert.deepEqual(state, {
    seed: 7,
    holdout: 6,
    target: "honey_production",
    thorough: true,
    measure: "precipitation",
    region: "South",
  });
});

test("numbers outside their limits are clamped", () => {
  const state = readState("?seed=-5&holdout=40");

  assert.equal(state.seed, 0);
  assert.equal(state.holdout, 8);
});

test("unreadable or unknown values fall back to the defaults", () => {
  const state = readState("?seed=abc&holdout=&target=wingspan&measure=altitude");

  assert.equal(state.seed, DEFAULTS.seed);
  assert.equal(state.holdout, DEFAULTS.holdout);
  assert.equal(state.target, DEFAULTS.target);
  assert.equal(state.measure, DEFAULTS.measure);
});

test("the default study writes a clean address", () => {
  assert.equal(writeState(DEFAULTS), "");
});

test("only changed settings are written, and reading them back is lossless", () => {
  const state = { ...DEFAULTS, seed: 99, thorough: true, region: "Center-West" };

  const query = writeState(state);

  assert.equal(query, "?seed=99&thorough=1&region=Center-West");
  assert.deepEqual(readState(query), state);
});

test("a study request carries only what the Python package expects", () => {
  const request = studyRequest({ ...DEFAULTS, holdout: 3, measure: "precipitation" });

  assert.deepEqual(request, {
    seed: 42,
    holdout_years: 3,
    target: "bee_abundance",
    thorough: false,
  });
});
