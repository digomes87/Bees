import { svg, text } from "./dom.js";
import { extent, linearScale, niceTicks } from "./scale.js";
import { hideTooltip, showTooltip } from "./tooltip.js";

const MARGIN = { top: 12, right: 18, bottom: 28, left: 46 };
const HEIGHT_RATIO = 0.5;
const HEIGHT_LIMITS = [220, 340];
const RING_RADIUS = 4.5;
const DOT_RADIUS = 4;
const WIDE_ENOUGH_FOR_MORE_TICKS = 480;

function frameFor(width) {
  const [shortest, tallest] = HEIGHT_LIMITS;
  const height = Math.min(Math.max(width * HEIGHT_RATIO, shortest), tallest);
  return {
    width,
    height,
    left: MARGIN.left,
    right: width - MARGIN.right,
    top: MARGIN.top,
    bottom: height - MARGIN.bottom,
  };
}

function scalesFor(series, frame) {
  const points = series.flatMap((entry) => entry.points);
  const positions = [...new Set(points.map((point) => point.x))].sort((a, b) => a - b);
  const ticks = niceTicks(extent(points.map((point) => point.y)));
  return {
    positions,
    ticks,
    x: linearScale(extent(positions), [frame.left, frame.right]),
    y: linearScale([ticks[0], ticks.at(-1)], [frame.bottom, frame.top]),
  };
}

function valueAxis({ ticks, y }, frame, format) {
  return ticks.flatMap((tick) => [
    svg("line", { class: "grid-line", x1: frame.left, x2: frame.right, y1: y(tick), y2: y(tick) }),
    text(format(tick), {
      class: "tick-label",
      x: frame.left - 8,
      y: y(tick),
      "text-anchor": "end",
      "dominant-baseline": "middle",
    }),
  ]);
}

function positionAxis({ positions, x }, frame) {
  const count = frame.width > WIDE_ENOUGH_FOR_MORE_TICKS ? 6 : 3;
  const [first, last] = extent(positions);
  const shown = niceTicks([first, last], count).filter((tick) => tick >= first && tick <= last);
  const labels = shown.map((tick) =>
    text(String(tick), { class: "tick-label", x: x(tick), y: frame.bottom + 18, "text-anchor": "middle" }),
  );
  const baseline = svg("line", {
    class: "axis-line",
    x1: frame.left,
    x2: frame.right,
    y1: frame.bottom,
    y2: frame.bottom,
  });
  return [baseline, ...labels];
}

function pathOf(points, { x, y }) {
  return points.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.x)},${y(point.y)}`).join("");
}

function markOf(entry, scales) {
  if (!entry.ring) {
    return svg("path", { class: `series-line ${entry.slot}`, d: pathOf(entry.points, scales) });
  }
  const rings = entry.points.map((point) =>
    svg("circle", {
      class: "series-ring",
      cx: scales.x(point.x),
      cy: scales.y(point.y),
      r: RING_RADIUS,
    }),
  );
  return svg("g", { class: entry.slot }, rings);
}

function readingsAt(series, position) {
  return series
    .map((entry) => ({ entry, point: entry.points.find((point) => point.x === position) }))
    .filter(({ point }) => point !== undefined);
}

function nearestIndex(positions, x, pixel) {
  const distances = positions.map((position) => Math.abs(x(position) - pixel));
  return distances.indexOf(Math.min(...distances));
}

/** A crosshair that follows the pointer or the arrow keys and reports every series. */
function attachCrosshair(root, spec, scales, frame) {
  const rule = svg("line", { class: "crosshair", y1: frame.top, y2: frame.bottom });
  const dots = svg("g");
  const layer = svg("g", { class: "is-hidden" }, [rule, dots]);
  const hit = svg("rect", {
    class: "hit-area",
    x: frame.left,
    y: frame.top,
    width: frame.right - frame.left,
    height: frame.bottom - frame.top,
  });
  let current = scales.positions.length - 1;

  const reveal = (index, viewportPoint) => {
    current = index;
    const position = scales.positions[index];
    const readings = readingsAt(spec.series, position);
    rule.setAttribute("x1", scales.x(position));
    rule.setAttribute("x2", scales.x(position));
    dots.replaceChildren(
      ...readings.map(({ entry, point }) =>
        svg("circle", {
          class: `series-dot ${entry.slot}`,
          cx: scales.x(position),
          cy: scales.y(point.y),
          r: DOT_RADIUS,
        }),
      ),
    );
    layer.classList.remove("is-hidden");
    const rows = readings.map(({ entry, point }) => ({
      slot: entry.slot,
      name: entry.name,
      value: spec.format(point.y),
    }));
    showTooltip(viewportPoint, String(position), rows);
  };

  const conceal = () => {
    layer.classList.add("is-hidden");
    hideTooltip();
  };

  const anchorOf = (index) => {
    const box = root.getBoundingClientRect();
    const scale = box.width / frame.width;
    return { x: box.left + scales.x(scales.positions[index]) * scale, y: box.top + frame.top * scale };
  };

  hit.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const pixel = (event.clientX - box.left) * (frame.width / box.width);
    reveal(nearestIndex(scales.positions, scales.x, pixel), { x: event.clientX, y: event.clientY });
  });
  hit.addEventListener("pointerleave", conceal);
  root.addEventListener("focus", () => reveal(current, anchorOf(current)));
  root.addEventListener("blur", conceal);
  root.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
    if (step === undefined) {
      return;
    }
    event.preventDefault();
    const next = Math.min(Math.max(current + step, 0), scales.positions.length - 1);
    reveal(next, anchorOf(next));
  });

  return [layer, hit];
}

/**
 * Lines (or rings, for `ring` series) over a shared numeric axis.
 * @param {{ width: number, label: string, format: (value: number) => string,
 *   series: { name: string, slot: string, ring?: boolean, points: { x: number, y: number }[] }[] }} spec
 */
export function lineChart(spec) {
  const frame = frameFor(spec.width);
  const scales = scalesFor(spec.series, frame);
  const root = svg("svg", {
    viewBox: `0 0 ${frame.width} ${frame.height}`,
    role: "img",
    tabindex: "0",
    "aria-label": `${spec.label}. Use the arrow keys to read values.`,
  });
  root.append(
    ...valueAxis(scales, frame, spec.format),
    ...positionAxis(scales, frame),
    ...spec.series.map((entry) => markOf(entry, scales)),
    ...attachCrosshair(root, spec, scales, frame),
  );
  return root;
}
