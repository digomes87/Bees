import { svg, text } from "./dom.js";
import { hideTooltip, showTooltip } from "./tooltip.js";

const LABEL_SHARE = 0.44;
const WIDEST_LABEL = 190;
const LARGEST_CELL = 44;
const HEADER = 22;

function fillOf(value) {
  const pole = value < 0 ? "--diverging-negative" : "--diverging-positive";
  const share = Math.round(Math.abs(value) * 100);
  return `color-mix(in oklab, var(${pole}) ${share}%, var(--diverging-neutral))`;
}

function cellOf(spec, layout, row, column) {
  const value = spec.matrix[row][column];
  const cell = svg("rect", {
    class: "heat-cell",
    x: layout.left + column * layout.cell,
    y: HEADER + row * layout.cell,
    width: layout.cell,
    height: layout.cell,
  });
  cell.style.fill = fillOf(value);
  cell.addEventListener("pointermove", (event) =>
    showTooltip({ x: event.clientX, y: event.clientY }, spec.labels[row], [
      { name: spec.labels[column], value: spec.format(value) },
    ]),
  );
  cell.addEventListener("pointerleave", hideTooltip);
  return cell;
}

function rowLabel(spec, layout, row) {
  return text(`${row + 1}  ${spec.labels[row]}`, {
    class: "mark-label",
    x: layout.left - 8,
    y: HEADER + (row + 0.5) * layout.cell,
    "text-anchor": "end",
    "dominant-baseline": "middle",
  });
}

function columnLabel(layout, column) {
  return text(String(column + 1), {
    class: "tick-label",
    x: layout.left + (column + 0.5) * layout.cell,
    y: HEADER - 8,
    "text-anchor": "middle",
  });
}

/**
 * A square matrix of values between -1 and 1 on a diverging scale.
 * Columns are numbered to match the numbered row labels.
 * @param {{ width: number, label: string, labels: string[], matrix: number[][],
 *   format: (value: number) => string }} spec
 */
export function heatmap(spec) {
  const size = spec.labels.length;
  const left = Math.min(spec.width * LABEL_SHARE, WIDEST_LABEL);
  const layout = { left, cell: Math.min((spec.width - left) / size, LARGEST_CELL) };
  const indexes = Array.from({ length: size }, (_, index) => index);
  const cells = indexes.flatMap((row) =>
    indexes.map((column) => cellOf(spec, layout, row, column)),
  );
  return svg(
    "svg",
    {
      viewBox: `0 0 ${spec.width} ${HEADER + size * layout.cell}`,
      role: "img",
      "aria-label": spec.label,
    },
    [
      ...indexes.map((row) => rowLabel(spec, layout, row)),
      ...indexes.map((column) => columnLabel(layout, column)),
      ...cells,
    ],
  );
}
