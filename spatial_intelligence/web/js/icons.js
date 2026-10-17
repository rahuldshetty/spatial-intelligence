/* Inline SVG icons: one stroke style, sized in CSS pixels, colored by currentColor.

Icons are markup inside a `<span class="icon">` rather than independent SVG
nodes so the same helper works with the test DOM shim and needs no
`createElementNS`. `aria-hidden` keeps decorative glyphs out of the accessible
name; buttons carry a `title`/`aria-label` of their own.
*/

"use strict";

const PATHS = {
  logo:
    '<circle cx="12" cy="12" r="9"/>' +
    '<path d="M3 12h18"/>' +
    '<path d="M12 3c2.7 3.2 2.7 14.8 0 18"/>' +
    '<path d="M12 3c-2.7 3.2-2.7 14.8 0 18"/>' +
    '<circle cx="17.6" cy="6.6" r="2.1" fill="currentColor" stroke="none"/>',
  play: '<path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M9 5v14M15 5v14"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5" fill="currentColor" stroke="none"/>',
  "chevron-up": '<path d="m6 15 6-6 6 6"/>',
  "chevron-down": '<path d="m6 9 6 6 6-6"/>',
  "chevron-right": '<path d="m9 6 6 6-6 6"/>',
  "chevron-left": '<path d="m15 6-6 6 6 6"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  "check-circle": '<circle cx="12" cy="12" r="9"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
  circle: '<circle cx="12" cy="12" r="9"/>',
  "x-circle": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>',
  eye:
    '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/>' +
    '<circle cx="12" cy="12" r="3"/>',
  "alert-triangle":
    '<path d="M10.3 3.8 2.2 18a2 2 0 0 0 1.7 3h16.2a2 2 0 0 0 1.7-3L13.7 3.8a2 2 0 0 0-3.4 0z"/>' +
    '<path d="M12 9v4M12 17h.01"/>',
  ban: '<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',
  "trash-2":
    '<path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>' +
    '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>' +
    '<path d="M10 11v6M14 11v6"/>',
  pencil:
    '<path d="M21.2 6.8a1 1 0 0 0-4-4L3.8 16.2a2 2 0 0 0-.5.8l-1.3 4.4a.5.5 0 0 0 .6.6l4.4-1.3a2 2 0 0 0 .8-.5z"/>' +
    '<path d="m15 5 4 4"/>',
  text: '<path d="M4 6h16M4 12h10M4 18h14"/>',
  code: '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
  sparkles:
    '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/>' +
    '<path d="M19 3v4M21 5h-4M5 17v4M7 19H3"/>',
  notebook:
    '<path d="M4 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/>' +
    '<path d="M16 2v20M8 7h4M8 11h4"/>',
  database:
    '<ellipse cx="12" cy="5" rx="8" ry="3"/>' +
    '<path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>' +
    '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  folder:
    '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  "folder-plus":
    '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>' +
    '<path d="M12 11v6M9 14h6"/>',
  "folder-open":
    '<path d="M3 8V6a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v2"/>' +
    '<path d="M3 8h16.2a2 2 0 0 1 2 2.4l-1.3 6A2 2 0 0 1 18 18H5a2 2 0 0 1-2-2z"/>',
  file:
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>' +
    '<path d="M14 2v6h6"/>',
  "file-image":
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>' +
    '<path d="M14 2v6h6"/><circle cx="9" cy="13" r="1.5"/><path d="m20 18-4-4-6 6"/>',
  save:
    '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>' +
    '<path d="M7 3v6h9M7 21v-8h10v8"/>',
  settings:
    '<path d="M12.2 2h-.4a2 2 0 0 0-2 2v.2a2 2 0 0 1-1 1.7l-.4.3a2 2 0 0 1-2 0l-.2-.1a2 2 0 0 0-2.7.7l-.2.4a2 2 0 0 0 .7 2.7l.2.1a2 2 0 0 1 1 1.7v.5a2 2 0 0 1-1 1.7l-.2.1a2 2 0 0 0-.7 2.7l.2.4a2 2 0 0 0 2.7.7l.2-.1a2 2 0 0 1 2 0l.4.3a2 2 0 0 1 1 1.7v.2a2 2 0 0 0 2 2h.4a2 2 0 0 0 2-2v-.2a2 2 0 0 1 1-1.7l.4-.3a2 2 0 0 1 2 0l.2.1a2 2 0 0 0 2.7-.7l.2-.4a2 2 0 0 0-.7-2.7l-.2-.1a2 2 0 0 1-1-1.7v-.5a2 2 0 0 1 1-1.7l.2-.1a2 2 0 0 0 .7-2.7l-.2-.4a2 2 0 0 0-2.7-.7l-.2.1a2 2 0 0 1-2 0l-.4-.3a2 2 0 0 1-1-1.7V4a2 2 0 0 0-2-2z"/>' +
    '<circle cx="12" cy="12" r="3"/>',
  sun:
    '<circle cx="12" cy="12" r="4"/>' +
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2z"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
  zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9z" fill="currentColor" stroke="none"/>',
  "refresh-cw":
    '<path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/>' +
    '<path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/>' +
    '<path d="M3 21v-5h5M21 3v5h-5"/>',
  upload:
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>' +
    '<path d="m17 8-5-5-5 5M12 3v12"/>',
  download:
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>' +
    '<path d="m7 10 5 5 5-5M12 15V3"/>',
  "arrow-right": '<path d="M4 12h16M14 6l6 6-6 6"/>',
  "arrow-down-to-line": '<path d="M12 4v12M6 10l6 6 6-6M4 20h16"/>',
  wrench:
    '<path d="M14.7 6.3a4.5 4.5 0 0 0 6 6l-9.4 9.4a2.8 2.8 0 0 1-4-4z"/>' +
    '<path d="m5 5 3 3M3 8l3-3"/>',
  "list-todo":
    '<path d="m3 6 2 2 3-3M3 14l2 2 3-3M3 20l2 2 3-3"/>' +
    '<path d="M13 6h8M13 14h8M13 20h8"/>',
  power:
    '<path d="M12 2v10"/>' +
    '<path d="M18.4 6.6a9 9 0 1 1-12.8 0"/>',
  layers:
    '<path d="m12 2 9 5-9 5-9-5z"/>' +
    '<path d="m3 12 9 5 9-5M3 17l9 5 9-5"/>',
  globe:
    '<circle cx="12" cy="12" r="9"/>' +
    '<path d="M3 12h18"/>' +
    '<path d="M12 3c2.7 3.2 2.7 14.8 0 18M12 3c-2.7 3.2-2.7 14.8 0 18"/>',
  "loader-circle": '<path d="M21 12a9 9 0 1 1-6.2-8.6"/>',
};

/**
 * Build one icon node.
 *
 * `size` is CSS pixels (default 16). The span is inline-flex so an icon aligns
 * with text without baseline drift; `class` adds a hook for component CSS.
 */
export function icon(name, options = {}) {
  const size = options.size || 16;
  const node = document.createElement("span");
  node.className = "icon" + (options.class ? " " + options.class : "");
  node.setAttribute("aria-hidden", "true");
  const strokeWidth = options.strokeWidth || 2;
  node.innerHTML =
    '<svg viewBox="0 0 24 24" width="' +
    size +
    '" height="' +
    size +
    '" fill="none" stroke="currentColor" stroke-width="' +
    strokeWidth +
    '" stroke-linecap="round" stroke-linejoin="round">' +
    (PATHS[name] || PATHS.circle) +
    "</svg>";
  return node;
}

/** A menu row: icon, label, optional shortcut hint. */
export function menuLabel(name, text, shortcut = "") {
  const label = document.createElement("span");
  label.className = "menu-label";
  label.append(icon(name, { size: 15 }));
  const copy = document.createElement("span");
  copy.className = "menu-text";
  copy.textContent = text;
  label.append(copy);
  if (shortcut) {
    const hint = document.createElement("kbd");
    hint.className = "menu-shortcut";
    hint.textContent = shortcut;
    label.append(hint);
  }
  return label;
}
