const NICE_FACTORS = [1, 2, 5, 10];
const SIGNIFICANT_DIGITS = 12;

export function extent(values) {
  return [Math.min(...values), Math.max(...values)];
}

export function linearScale([domainStart, domainEnd], [rangeStart, rangeEnd]) {
  const span = domainEnd - domainStart || 1;
  return (value) => rangeStart + ((value - domainStart) / span) * (rangeEnd - rangeStart);
}

function withoutNoise(value) {
  return Number(value.toPrecision(SIGNIFICANT_DIGITS));
}

function niceStep(span, count) {
  const rough = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const factor = NICE_FACTORS.find((candidate) => candidate >= rough / magnitude);
  return factor * magnitude;
}

/** Round tick values that start at or below `low` and end at or above `high`. */
export function niceTicks([low, high], count = 5) {
  const step = niceStep(high - low || Math.abs(low) || 1, count);
  const first = Math.floor(low / step) * step;
  const steps = Math.max(Math.ceil(withoutNoise((high - first) / step)), 1);
  return Array.from({ length: steps + 1 }, (_, index) => withoutNoise(first + index * step));
}
