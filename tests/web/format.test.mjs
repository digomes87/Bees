import assert from "node:assert/strict";
import { test } from "node:test";

import {
  formatLatitude,
  formatMeasure,
  formatNumber,
  formatSigned,
  humanize,
} from "../../docs/scripts/format.js";

test("humanize turns column names into readable labels", () => {
  assert.equal(humanize("bee_abundance"), "Bee abundance");
  assert.equal(humanize("temperature_previous_year"), "Temperature previous year");
  assert.equal(humanize("region_Center-West"), "Region Center-West");
});

test("numbers keep the requested number of decimals", () => {
  assert.equal(formatNumber(3.14159, 2), "3.14");
  assert.equal(formatNumber(1379.4, 0), "1,379");
});

test("signed numbers always show their direction", () => {
  assert.equal(formatSigned(1.25, 1), "+1.3");
  assert.equal(formatSigned(-0.5, 1), "−0.5");
  assert.equal(formatSigned(0, 1), "0.0");
});

test("measures are formatted with their unit", () => {
  assert.equal(formatMeasure("temperature", 24.56), "24.6 °C");
  assert.equal(formatMeasure("precipitation", 1379.4), "1,379 mm");
  assert.equal(formatMeasure("bee_abundance", 61.12), "61.1");
});

test("latitudes name their hemisphere instead of carrying a sign", () => {
  assert.equal(formatLatitude(-12.34), "12.3° S");
  assert.equal(formatLatitude(4), "4.0° N");
  assert.equal(formatLatitude(null), "–");
});

test("missing values are shown as a dash instead of NaN", () => {
  assert.equal(formatNumber(null, 2), "–");
  assert.equal(formatMeasure("temperature", undefined), "–");
});
