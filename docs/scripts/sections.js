import { barChart } from "./charts/bar-chart.js";
import { html } from "./charts/dom.js";
import { heatmap } from "./charts/heatmap.js";
import { lineChart } from "./charts/line-chart.js";
import { mountChart } from "./charts/mount.js";
import { chartCard, chartHost, legend, tableView } from "./charts/parts.js";
import { formatMeasure, formatNumber, formatSigned, humanize, unitOf } from "./format.js";

const YEARS_PER_DECADE = 10;

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function slotOf(study, region) {
  return `series-${study.regions.indexOf(region) + 1}`;
}

export function chapter(number, title, summary, body, split = false) {
  const heading = html("div", { className: "chapter-heading" }, [
    html("span", { className: "chapter-number", textContent: number }),
    html("h2", { textContent: title }),
    html("p", { textContent: summary }),
  ]);
  const content = html("div", { className: `chapter-body${split ? " is-split" : ""}` }, body);
  return html("section", { className: "chapter" }, [heading, content]);
}

export function tile(label, value, note, kind = "") {
  return html("div", { className: "tile" }, [
    html("span", { className: "tile-label", textContent: label }),
    html("span", { className: `tile-value ${kind}`.trim(), textContent: value }),
    html("span", { className: "tile-note", textContent: note }),
  ]);
}

function decadeChange(study, measure) {
  const change = mean(study.trends.map((trend) => trend[measure])) * YEARS_PER_DECADE;
  return `${formatSigned(change, 1)} ${unitOf(measure)}`.trim();
}

function tiles(study) {
  const target = study.request.target;
  const best = study.models.find((model) => model.model === study.best);
  const years = study.observations.map((record) => record.year);
  return html("div", { className: "tiles" }, [
    tile("Warming", decadeChange(study, "temperature"), "per decade, mean of all regions"),
    tile(humanize(target), decadeChange(study, target), "per decade, mean of all regions"),
    tile(
      "Best model",
      study.best,
      `R² ${formatNumber(best.test_r2, 2)} on ${study.request.holdout_years} held-out years`,
      "is-name",
    ),
    tile(
      "Records",
      formatNumber(study.observations.length, 0),
      `${study.regions.length} regions · ${Math.min(...years)}–${Math.max(...years)}`,
    ),
  ]);
}

function seriesByRegion(study, measure) {
  return study.regions.map((region) => ({
    name: region,
    slot: slotOf(study, region),
    points: study.observations
      .filter((record) => record.region === region)
      .map((record) => ({ x: record.year, y: record[measure] })),
  }));
}

function yearsByRegion(study, measure) {
  const years = [...new Set(study.observations.map((record) => record.year))];
  return years.map((year) => {
    const rows = study.observations.filter((record) => record.year === year);
    const cells = rows.map((record) => [record.region, record[measure]]);
    return { year, ...Object.fromEntries(cells) };
  });
}

function trendCard(study, measure) {
  const host = chartHost();
  const series = seriesByRegion(study, measure);
  const format = (value) => formatMeasure(measure, value);
  mountChart(host, (width) =>
    lineChart({ width, series, format, label: `${humanize(measure)} by region and year` }),
  );
  const columns = [
    { key: "year", label: "Year" },
    ...study.regions.map((region) => ({ key: region, label: region, format })),
  ];
  return chartCard({
    title: `${humanize(measure)} by region`,
    note: "Yearly values. Hover, or focus the chart and use the arrow keys.",
    extras: [legend(series)],
    host,
    table: tableView(columns, yearsByRegion(study, measure)),
  });
}

function correlationCard(study) {
  const host = chartHost();
  const labels = study.correlations.variables.map(humanize);
  const format = (value) => formatSigned(value, 2);
  mountChart(host, (width) =>
    heatmap({
      width,
      labels,
      matrix: study.correlations.matrix,
      format,
      label: "Correlation between every pair of variables",
    }),
  );
  const scale = html("div", { className: "scale-legend" }, [
    html("span", { textContent: "−1" }),
    html("span", { className: "scale-ramp" }),
    html("span", { textContent: "+1" }),
  ]);
  const records = study.correlations.matrix.map((row, index) => ({
    variable: labels[index],
    ...Object.fromEntries(row.map((value, column) => [column, value])),
  }));
  const columns = [
    { key: "variable", label: "Variable" },
    ...labels.map((_, column) => ({ key: column, label: String(column + 1), format })),
  ];
  return chartCard({
    title: "Correlation matrix",
    note: "Pearson correlation across all regions and years. Columns follow the row numbers.",
    extras: [scale],
    host,
    table: tableView(columns, records),
  });
}

function pairList(study) {
  const items = study.pairs.map((pair) =>
    html("li", {}, [
      html("span", { textContent: `${humanize(pair.first)} × ${humanize(pair.second)}` }),
      html("strong", { textContent: formatSigned(pair.coefficient, 2) }),
    ]),
  );
  return html("figure", { className: "chart-card" }, [
    html("figcaption", {}, [
      html("h3", { textContent: "Strongest pairs" }),
      html("p", { textContent: "Largest correlations between two different variables." }),
    ]),
    html("ul", { className: "pair-list" }, items),
  ]);
}

function modelCard(study) {
  const host = chartHost();
  const rows = study.models.map((model) => ({
    label: model.model,
    value: model.test_r2,
    text: formatNumber(model.test_r2, 2),
    emphasis: model.model === study.best,
  }));
  mountChart(host, (width) =>
    barChart({ width, rows, title: "R² on held-out years", label: "Model accuracy on held-out years" }),
  );
  const decimals = (value) => formatNumber(value, 2);
  const columns = [
    { key: "model", label: "Model" },
    { key: "test_r2", label: "Test R²", format: decimals },
    { key: "test_rmse", label: "Test RMSE", format: decimals },
    { key: "test_mae", label: "Test MAE", format: decimals },
    { key: "train_r2", label: "Train R²", format: decimals },
  ];
  return chartCard({
    title: "Model accuracy",
    note: `R² on the ${study.request.holdout_years} most recent years, which no model saw while training.`,
    host,
    table: tableView(columns, study.models),
  });
}

function forecastSeries(study, region) {
  const target = study.request.target;
  const observed = study.observations
    .filter((record) => record.region === region)
    .map((record) => ({ x: record.year, y: record[target] }));
  const predicted = study.forecasts
    .filter((row) => row.region === region)
    .map((row) => ({ x: row.year, y: row[study.best] }));
  return [
    { name: "Observed", slot: slotOf(study, region), points: observed },
    { name: `Predicted (${study.best})`, slot: "series-plain", ring: true, points: predicted },
  ];
}

function forecastCard(study, region) {
  const host = chartHost();
  const target = study.request.target;
  const series = forecastSeries(study, region);
  const format = (value) => formatMeasure(target, value);
  mountChart(host, (width) =>
    lineChart({ width, series, format, label: `Observed and predicted ${humanize(target)} in ${region}` }),
  );
  const columns = [
    { key: "year", label: "Year" },
    { key: "actual", label: "Observed", format },
    ...study.models.map((model) => ({ key: model.model, label: model.model, format })),
  ];
  return chartCard({
    title: `${humanize(target)} in ${region}`,
    note: "The line is what was observed; rings are the best model's forecasts for the held-out years.",
    extras: [legend(series)],
    host,
    table: tableView(columns, study.forecasts.filter((row) => row.region === region)),
  });
}

function importanceCard(study) {
  const host = chartHost();
  const rows = study.importance.map((entry) => ({
    label: humanize(entry.feature),
    value: entry.importance,
    text: formatNumber(entry.importance, 2),
  }));
  mountChart(host, (width) =>
    barChart({ width, rows, title: "Permutation importance", label: "Most influential features" }),
  );
  const columns = [
    { key: "feature", label: "Feature", format: humanize },
    { key: "importance", label: "Importance", format: (value) => formatNumber(value, 3) },
  ];
  return chartCard({
    title: `What ${study.best} relies on`,
    note: "Drop in R² when a feature is shuffled, measured on the training years.",
    host,
    table: tableView(columns, study.importance),
  });
}

/** Every chapter of the report for one study and the visitor's view choices. */
export function reportOf(study, view) {
  const target = humanize(study.request.target).toLowerCase();
  return [
    chapter("1", "The climate signal", "How each region changed over the study period.", [
      tiles(study),
      trendCard(study, view.measure),
    ]),
    chapter(
      "2",
      "What moves together",
      "Variables that rise and fall together across regions and years. Correlation alone does not show cause.",
      [correlationCard(study), pairList(study)],
      true,
    ),
    chapter(
      "3",
      "Can it be forecast?",
      `Four models learn ${target} from climate and from last year's value, then predict years they never saw.`,
      [modelCard(study), forecastCard(study, view.region), importanceCard(study)],
    ),
  ];
}
