const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/** Create an SVG element. Attributes are set as given; no markup is parsed. */
export function svg(tag, attributes = {}, children = []) {
  const node = document.createElementNS(SVG_NAMESPACE, tag);
  Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, value));
  node.append(...children);
  return node;
}

/** Create an HTML element from DOM properties such as `className` and `textContent`. */
export function html(tag, properties = {}, children = []) {
  const node = Object.assign(document.createElement(tag), properties);
  node.append(...children);
  return node;
}

export function text(content, attributes = {}) {
  const node = svg("text", attributes);
  node.textContent = content;
  return node;
}
