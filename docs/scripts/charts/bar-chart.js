import { svg, text } from "./dom.js";
import { linearScale } from "./scale.js";
import { hideTooltip, showTooltip } from "./tooltip.js";

const ROW_HEIGHT = 30;
const THICKNESS = 14;
const END_RADIUS = 4;
const LABEL_SHARE = 0.38;
const WIDEST_LABEL = 190;
const VALUE_ROOM = 52;
const PADDING = 6;
const LABEL_SHARE_WHEN_NARROW = 0.46;
const NARROW = 420;
const CHARACTER_WIDTH = 6.2;
const ELLIPSIS = "…";

/** Shorten a label that cannot fit its column; the full text stays in the tooltip. */
function fitted(label, room) {
  const capacity = Math.floor(room / CHARACTER_WIDTH);
  if (label.length <= capacity) {
    return label;
  }
  return `${label.slice(0, Math.max(capacity - 1, 1))}${ELLIPSIS}`;
}

/** A bar from the baseline to `end`, rounded only at the data end. */
function barPath(baseline, end, top) {
  const direction = Math.sign(end - baseline) || 1;
  const radius = Math.min(END_RADIUS, Math.abs(end - baseline));
  const corner = end - direction * radius;
  const sweep = direction > 0 ? 1 : 0;
  return [
    `M${baseline},${top}`,
    `H${corner}`,
    `A${radius},${radius} 0 0 ${sweep} ${end},${top + radius}`,
    `V${top + THICKNESS - radius}`,
    `A${radius},${radius} 0 0 ${sweep} ${corner},${top + THICKNESS}`,
    `H${baseline}Z`,
  ].join("");
}

function rowOf(record, index, layout, title) {
  const top = PADDING + index * ROW_HEIGHT;
  const middle = top + ROW_HEIGHT / 2;
  const end = layout.x(record.value);
  const emphasis = record.emphasis ? " is-emphasis" : "";
  const strong = record.emphasis ? " is-strong" : "";
  const group = svg("g", { class: "bar-row" }, [
    text(fitted(record.label, layout.labelEnd), {
      class: `mark-label${strong}`,
      x: layout.labelEnd,
      y: middle,
      "text-anchor": "end",
      "dominant-baseline": "middle",
    }),
    svg("path", {
      class: `bar${emphasis}`,
      d: barPath(layout.x(0), end, middle - THICKNESS / 2),
    }),
    text(record.text, {
      class: `mark-label${strong}`,
      x: Math.max(end, layout.x(0)) + 6,
      y: middle,
      "dominant-baseline": "middle",
    }),
    svg("rect", { class: "hit-area", x: 0, y: top, width: layout.width, height: ROW_HEIGHT }),
  ]);
  group.addEventListener("pointermove", (event) =>
    showTooltip({ x: event.clientX, y: event.clientY }, title, [
      { name: record.label, value: record.text },
    ]),
  );
  group.addEventListener("pointerleave", hideTooltip);
  return group;
}

/**
 * Horizontal bars sharing a zero baseline, each labelled with its value.
 * @param {{ width: number, label: string, title: string,
 *   rows: { label: string, value: number, text: string, emphasis?: boolean }[] }} spec
 */
export function barChart(spec) {
  const values = spec.rows.map((row) => row.value);
  const share = spec.width < NARROW ? LABEL_SHARE_WHEN_NARROW : LABEL_SHARE;
  const labelEnd = Math.min(spec.width * share, WIDEST_LABEL);
  const layout = {
    width: spec.width,
    labelEnd,
    x: linearScale(
      [Math.min(0, ...values), Math.max(0, ...values)],
      [labelEnd + 10, spec.width - VALUE_ROOM],
    ),
  };
  const height = spec.rows.length * ROW_HEIGHT + PADDING * 2;
  const baseline = svg("line", {
    class: "axis-line",
    x1: layout.x(0),
    x2: layout.x(0),
    y1: PADDING,
    y2: height - PADDING,
  });
  const rows = spec.rows.map((row, index) => rowOf(row, index, layout, spec.title));
  return svg(
    "svg",
    { viewBox: `0 0 ${spec.width} ${height}`, role: "img", "aria-label": spec.label },
    [...rows, baseline],
  );
}
