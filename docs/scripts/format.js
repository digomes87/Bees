const MISSING = "–";
const MINUS = "−";
const UNITS = {
  temperature: { suffix: " °C", decimals: 1 },
  precipitation: { suffix: " mm", decimals: 0 },
};
const PLAIN = { suffix: "", decimals: 1 };

export function humanize(name) {
  const words = name.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function formatNumber(value, decimals = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return MISSING;
  }
  return value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatSigned(value, decimals = 1) {
  const text = formatNumber(Math.abs(value), decimals);
  if (Number(text) === 0) {
    return text;
  }
  return `${value > 0 ? "+" : MINUS}${text}`;
}

export function formatLatitude(value) {
  const text = formatNumber(value, 1);
  if (text === MISSING) {
    return text;
  }
  return `${formatNumber(Math.abs(value), 1)}° ${value < 0 ? "S" : "N"}`;
}

export function unitOf(measure) {
  return (UNITS[measure] ?? PLAIN).suffix.trim();
}

export function formatMeasure(measure, value) {
  const { suffix, decimals } = UNITS[measure] ?? PLAIN;
  const text = formatNumber(value, decimals);
  return text === MISSING ? text : `${text}${suffix}`;
}
