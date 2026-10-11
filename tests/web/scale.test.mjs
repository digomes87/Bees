import assert from "node:assert/strict";
import { test } from "node:test";

import { extent, linearScale, niceTicks } from "../../docs/scripts/charts/scale.js";

test("extent returns the smallest and largest value", () => {
  assert.deepEqual(extent([3, -1, 7, 2]), [-1, 7]);
});

test("linear scale maps the domain ends onto the range ends", () => {
  const scale = linearScale([0, 10], [100, 200]);

  assert.equal(scale(0), 100);
  assert.equal(scale(5), 150);
  assert.equal(scale(10), 200);
});

test("linear scale supports inverted ranges for vertical axes", () => {
  const scale = linearScale([0, 4], [300, 0]);

  assert.equal(scale(1), 225);
});

test("linear scale centres a domain with no spread instead of dividing by zero", () => {
  const scale = linearScale([5, 5], [0, 100]);

  assert.equal(scale(5), 0);
});

test("nice ticks use round steps that cover the whole domain", () => {
  assert.deepEqual(niceTicks([19.8, 29.4]), [18, 20, 22, 24, 26, 28, 30]);
});

test("nice ticks stay free of floating point noise", () => {
  assert.deepEqual(niceTicks([0.1, 0.72]), [0, 0.2, 0.4, 0.6, 0.8]);
});

test("nice ticks widen a flat domain so a lone value still gets an axis", () => {
  const ticks = niceTicks([4, 4]);

  assert.ok(ticks[0] <= 4);
  assert.ok(ticks.at(-1) >= 4);
  assert.ok(ticks.length >= 2);
});
