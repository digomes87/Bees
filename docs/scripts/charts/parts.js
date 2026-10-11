import { html } from "./dom.js";

/** A legend naming every series next to a sample of its mark. */
export function legend(items) {
  const entries = items.map(({ name, slot, ring }) =>
    html("li", {}, [
      html("span", { className: `swatch ${slot}${ring ? " is-ring" : ""}` }),
      html("span", { textContent: name }),
    ]),
  );
  return html("ul", { className: "legend" }, entries);
}

function headerRow(columns) {
  const cells = columns.map(({ label }) => html("th", { scope: "col", textContent: label }));
  return html("thead", {}, [html("tr", {}, cells)]);
}

function bodyRow(columns, record) {
  const cells = columns.map(({ key, format = String }) =>
    html("td", { textContent: format(record[key]) }),
  );
  return html("tr", {}, cells);
}

/** The values behind a chart as a collapsible table. */
export function tableView(columns, records) {
  const body = html("tbody", {}, records.map((record) => bodyRow(columns, record)));
  const table = html("table", {}, [headerRow(columns), body]);
  return html("details", { className: "table-view" }, [
    html("summary", { textContent: "View as table" }),
    html("div", { className: "table-scroll" }, [table]),
  ]);
}

/** A titled card holding a chart, its legend and its table. */
export function chartCard({ title, note, extras = [], host, table }) {
  const caption = html("figcaption", {}, [
    html("h3", { textContent: title }),
    html("p", { textContent: note }),
  ]);
  return html("figure", { className: "chart-card" }, [caption, ...extras, host, table]);
}

export function chartHost() {
  return html("div", { className: "chart-host" });
}
