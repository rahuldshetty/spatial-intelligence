/* Inline placeholder art for the workflow cards — one drawing per workflow id,
plus a fallback for anything the server adds later.

`workflowArt(id, {size})` returns a `<svg class="workflow-art">` drawn with
`currentColor` and no external files, so it follows the theme and never 404s.
*/

"use strict";

const NS = "http://www.w3.org/2000/svg";

function node(tag, attrs = {}, children = []) {
  const shape = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) shape.setAttribute(key, value);
  for (const child of children) shape.append(child);
  return shape;
}

const ART = {
  blank: () => [
    node("rect", { x: 4, y: 3, width: 16, height: 18, rx: 2 }),
    node("path", { d: "M8 8h8M8 12h8M8 16h5" }),
  ],
  segmentation: () => [
    node("rect", { x: 3, y: 3, width: 18, height: 18, rx: 2 }),
    node("path", { d: "M3 9h18M3 15h18M9 3v18M15 3v18", opacity: 0.35 }),
    node("path", {
      d: "M7 8l5-1 4 5-3 4-4-1z",
      fill: "currentColor",
      "fill-opacity": 0.22,
      stroke: "none",
    }),
    node("path", { d: "M7 8l5-1 4 5-3 4-4-1z" }),
  ],
  detection: () => [
    node("rect", { x: 3, y: 3, width: 18, height: 18, rx: 2 }),
    node("rect", { x: 6, y: 7, width: 6, height: 5, rx: 1 }),
    node("rect", { x: 13, y: 13, width: 5, height: 5, rx: 1 }),
  ],
  "change-detection": () => [
    node("rect", { x: 3, y: 4, width: 8, height: 8, rx: 1 }),
    node("rect", { x: 13, y: 13, width: 8, height: 8, rx: 1 }),
    node("path", { d: "M12 7h6" }),
    node("path", { d: "M15.5 4.5L18 7l-2.5 2.5" }),
  ],
  fallback: () => [
    node("path", { d: "M12 21s7-6.3 7-11a7 7 0 10-14 0c0 4.7 7 11 7 11z" }),
    node("circle", { cx: 12, cy: 10, r: 2.5 }),
  ],
};

export function workflowArt(id, options = {}) {
  const { size = 72 } = options;
  const build = ART[id] || ART.fallback;
  return node(
    "svg",
    {
      class: "workflow-art",
      viewBox: "0 0 24 24",
      width: size,
      height: size,
      fill: "none",
      stroke: "currentColor",
      "stroke-width": 1.6,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "aria-hidden": "true",
      focusable: "false",
    },
    build()
  );
}
