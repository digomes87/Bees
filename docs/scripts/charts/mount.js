const NARROWEST = 260;
const observers = [];

/** Draw a chart into `host` at the host's width, and again whenever it changes. */
export function mountChart(host, draw) {
  let drawnWidth = 0;
  const observer = new ResizeObserver(() => {
    const width = Math.max(Math.round(host.clientWidth), NARROWEST);
    if (width === drawnWidth) {
      return;
    }
    drawnWidth = width;
    host.replaceChildren(draw(width));
  });
  observer.observe(host);
  observers.push(observer);
}

/** Stop redrawing every mounted chart, before their hosts are discarded. */
export function unmountCharts() {
  observers.splice(0).forEach((observer) => observer.disconnect());
}
