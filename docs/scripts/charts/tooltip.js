import { html } from "./dom.js";

const OFFSET = 14;
const EDGE = 8;

const bubble = html("div", { className: "tooltip", hidden: true });
bubble.setAttribute("role", "presentation");

function row({ slot, name, value }) {
  const swatch = html("span", { className: `swatch ${slot ?? "series-plain"}` });
  return html("div", { className: "tooltip-row" }, [
    swatch,
    html("span", { textContent: name }),
    html("span", { textContent: value }),
  ]);
}

function within(position, size, limit) {
  const flipped = position + OFFSET + size > limit - EDGE;
  const placed = flipped ? position - OFFSET - size : position + OFFSET;
  return Math.max(placed, EDGE);
}

/** Show the shared tooltip next to a viewport point. */
export function showTooltip({ x, y }, title, rows) {
  if (!bubble.isConnected) {
    document.body.append(bubble);
  }
  bubble.replaceChildren(
    html("p", { className: "tooltip-title", textContent: title }),
    ...rows.map(row),
  );
  bubble.hidden = false;
  const left = within(x, bubble.offsetWidth, window.innerWidth);
  const top = within(y, bubble.offsetHeight, window.innerHeight);
  bubble.style.transform = `translate(${left}px, ${top}px)`;
}

export function hideTooltip() {
  bubble.hidden = true;
}
