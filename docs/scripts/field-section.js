import { html } from "./charts/dom.js";
import { lineChart } from "./charts/line-chart.js";
import { mountChart } from "./charts/mount.js";
import { chartCard, chartHost, legend, tableView } from "./charts/parts.js";
import { formatLatitude, formatMeasure, formatNumber, formatSigned } from "./format.js";
import { chapter, tile } from "./sections.js";

const TITLE = "What the field records say";
const SUMMARY =
  "Observed weather and real bee records, fetched live. Unlike the chapters above, nothing here is simulated.";
const CAVEAT =
  "Records show where people looked for bees as much as where bees live. A centroid that moves can mean more observers in new places, so this chart cannot confirm or rule out a migration on its own.";

const TOO_MANY_REQUESTS = 429;
const ATTEMPTS = 4;
const PAUSE_MILLISECONDS = 1500;

function pause(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/** Fetch JSON, waiting and trying again while the service asks to slow down. */
async function fetched(url, attempt = 1) {
  const response = await fetch(url);
  if (response.status === TOO_MANY_REQUESTS && attempt < ATTEMPTS) {
    await pause(PAUSE_MILLISECONDS * attempt);
    return fetched(url, attempt + 1);
  }
  if (!response.ok) {
    throw new Error(`${new URL(url).hostname} answered HTTP ${response.status}`);
  }
  return response.json();
}

/** One request at a time: GBIF rejects bursts of parallel searches. */
async function fetchedInTurn(urls) {
  const answers = [];
  for (const url of urls) {
    answers.push(await fetched(url));
  }
  return answers;
}

/** Fetch the weather and the record counts, then let the package summarise them. */
async function fieldReport(runtime) {
  const requests = await runtime.call("field_requests");
  const [climate, answers] = await Promise.all([
    fetched(requests.climate),
    fetchedInTurn(requests.bands.map((band) => band.url)),
  ]);
  const bands = requests.bands.map((band, index) => ({
    south: band.south,
    north: band.north,
    answer: answers[index],
  }));
  return runtime.call("field_report", { climate, bands });
}

function tiles(report) {
  const years = report.centroids.map((row) => row.year);
  return html("div", { className: "tiles" }, [
    tile(
      "Observed warming",
      `${formatSigned(report.warming_per_decade, 2)} °C`,
      "per decade at the five region centres (ERA5)",
    ),
    tile(
      "Record centroid",
      `${formatSigned(report.latitude_shift_per_decade, 1)}°`,
      "latitude per decade; negative is southward",
    ),
    tile(
      "Bee records",
      formatNumber(report.total_records, 0),
      `Apidae in South America, ${Math.min(...years)}–${Math.max(...years)} (GBIF)`,
    ),
  ]);
}

function climateCard(report) {
  const host = chartHost();
  const format = (value) => formatMeasure("temperature", value);
  const series = report.regions.map((region, index) => ({
    name: region,
    slot: `series-${index + 1}`,
    points: report.climate
      .filter((row) => row.region === region)
      .map((row) => ({ x: row.year, y: row.temperature })),
  }));
  mountChart(host, (width) =>
    lineChart({ width, series, format, label: "Observed mean temperature by region and year" }),
  );
  const columns = [
    { key: "year", label: "Year" },
    { key: "region", label: "Region" },
    { key: "temperature", label: "Mean temperature", format },
  ];
  return chartCard({
    title: "Observed temperature by region",
    note: "Yearly mean of daily temperature at each region's centre. Source: Open-Meteo, ERA5 reanalysis.",
    extras: [legend(series)],
    host,
    table: tableView(columns, report.climate),
  });
}

function centroidCard(report) {
  const host = chartHost();
  const series = [
    {
      name: "Mean latitude of records",
      slot: "series-plain",
      points: report.centroids.map((row) => ({ x: row.year, y: row.mean_latitude })),
    },
  ];
  mountChart(host, (width) =>
    lineChart({
      width,
      series,
      format: formatLatitude,
      label: "Mean latitude of bee records by year",
    }),
  );
  const columns = [
    { key: "year", label: "Year" },
    { key: "records", label: "Records", format: (value) => formatNumber(value, 0) },
    { key: "mean_latitude", label: "Mean latitude", format: formatLatitude },
  ];
  return chartCard({
    title: "Where bees were recorded",
    note: `Mean latitude of Apidae records in South America, from counts in 5° bands. ${CAVEAT}`,
    host,
    table: tableView(columns, report.centroids),
  });
}

function invitation(onLoad, message = "") {
  const button = html("button", {
    type: "button",
    className: "run-button",
    textContent: "Load field data",
  });
  const note = html("p", { className: "status", textContent: message });
  note.setAttribute("role", "status");
  note.dataset.state = message === "" ? "working" : "failed";
  button.addEventListener("click", () => {
    button.disabled = true;
    note.dataset.state = "working";
    note.textContent = "Fetching from GBIF and Open-Meteo…";
    onLoad();
  });
  return html("div", { className: "field-invite" }, [
    html("p", {
      textContent:
        "Loading sends about fifteen requests from your browser to api.gbif.org and archive-api.open-meteo.com.",
    }),
    button,
    note,
  ]);
}

/** Offer the field chapter in `container`, loading its data only when asked. */
export function mountField(container, runtime) {
  const show = (body) => container.replaceChildren(chapter("4", TITLE, SUMMARY, body));
  const load = async () => {
    try {
      const report = await fieldReport(runtime);
      show([tiles(report), climateCard(report), centroidCard(report)]);
    } catch (error) {
      show([invitation(load, `The field data could not be loaded: ${error.message}`)]);
    }
  };
  show([invitation(load)]);
}
