/* Reusable interaction fields: build one fieldset from a server field object.

`buildField(field, {disabled, answers, requestId})` returns `{group, read}`:
`group` is a `<fieldset class="interaction-field">` carrying legend,
description, and control; `read()` returns the typed value of that control.

Markup and classes are shared with the deferred tool-input form in
`interaction.js` (radio/checkbox/text/confirmation) so both render identically.
*/

"use strict";

import { el } from "../dom.js";
import { geocodePlace, modelProfile, searchModels } from "../api.js";
import { addGeoJsonLayer, fitBounds, getDrawnFeatures, getView, removeLayer } from "../bridge.js";

export function buildField(field, options = {}) {
  const { disabled = false, answers = {}, requestId = "" } = options;
  const group = el("fieldset", { class: "interaction-field" });
  group.append(el("legend", { text: field.label + (field.required === false ? "" : " *") }));
  if (field.description) {
    group.append(el("p", { class: "interaction-description", text: field.description }));
  }

  if (field.type === "text") {
    const input = el("input", {
      type: "text",
      placeholder: field.placeholder || "",
      value:
        disabled && answers[field.id] != null
          ? String(answers[field.id])
          : field.default == null
            ? ""
            : String(field.default),
    });
    input.disabled = disabled;
    group.append(input);
    return { group, read: () => input.value.trim() };
  }

  if (field.type === "textarea") {
    const input = el("textarea", {
      rows: "4",
      placeholder: field.placeholder || "",
    });
    input.value =
      disabled && answers[field.id] != null
        ? String(answers[field.id])
        : field.default == null
          ? ""
          : String(field.default);
    input.disabled = disabled;
    group.append(input);
    return { group, read: () => input.value.trim() };
  }

  if (field.type === "confirmation") {
    const input = el("input", { type: "checkbox" });
    input.checked = disabled ? Boolean(answers[field.id]) : Boolean(field.default);
    input.disabled = disabled;
    const option = el("label", { class: "interaction-option compact" });
    option.append(input, el("span", { text: field.description || "Yes" }));
    group.append(option);
    return { group, read: () => input.checked };
  }

  if (field.type === "aoi") {
    return { group, read: buildAoi(group, field, { disabled }) };
  }

  if (field.type === "model") {
    return { group, read: buildModel(group, field, { disabled, answers }) };
  }

  const nodes = [];
  for (const option of field.options || []) {
    const input = el("input", {
      type: field.type === "radio" ? "radio" : "checkbox",
      name: "interaction-" + requestId + "-" + field.id,
      value: option.value,
    });
    const selected = disabled
      ? Array.isArray(answers[field.id])
        ? answers[field.id].includes(option.value)
        : answers[field.id] === option.value
      : Array.isArray(field.default)
        ? field.default.includes(option.value)
        : field.default === option.value || (!field.default && option.recommended);
    input.checked = selected;
    input.disabled = disabled;
    const card = el("label", {
      class: "interaction-option" + (option.thumbnail_url ? " with-thumbnail" : ""),
    });
    if (option.thumbnail_url) {
      card.append(el("img", { src: option.thumbnail_url, alt: "" }));
    }
    const copy = el("span", { class: "interaction-option-copy" });
    copy.append(
      el("strong", { text: option.label + (option.recommended ? " — Recommended" : "") })
    );
    if (option.description) copy.append(el("small", { text: option.description }));
    card.append(input, copy);
    group.append(card);
    nodes.push(input);
  }
  if (field.type === "radio") {
    return { group, read: () => (nodes.find((node) => node.checked) || {}).value };
  }
  return { group, read: () => nodes.filter((node) => node.checked).map((node) => node.value) };
}

/* -- model picker --------------------------------------------------------- */

/* A `model` field: curated cards from the server, then a Hugging Face search.
   Picking anything (card or hit) resolves a profile, whose `labels` / `promptable`
   decide which question follows — a label checklist, a "what should it find?"
   box, or both. `read()` returns `{ref, labels, target}`, or `null` while nothing
   runnable is selected so the wizard's required check blocks Next. */

function buildModel(group, field, { disabled, answers }) {
  const task = field.task || "";
  let selected = null;
  let profile = null;
  let labels = [];
  let target = "";
  let profileSeq = 0;
  let searchSeq = 0;
  let debounce = null;
  // Which of the two ways in the user is in: the models this build ships with,
  // or one they bring from the Hub. Exactly one pane is on screen at a time, so
  // the step never shows a card list, a search box and a result list at once.
  let source = "builtin";

  const root = el("div", { class: "model-field" });

  const sourceRow = el("div", { class: "model-source" });
  const sourceInputs = new Map();
  for (const [key, label] of [
    ["builtin", "Models in this build"],
    ["custom", "From Hugging Face"],
  ]) {
    const input = el("input", { type: "radio", name: "model-source-" + field.id, value: key });
    input.checked = key === source;
    input.disabled = disabled;
    input.addEventListener("change", () => {
      if (input.checked) setSource(key, true);
    });
    const pill = el("label", { class: "model-source-pill" });
    pill.append(input, el("span", { text: label }));
    sourceInputs.set(key, input);
    sourceRow.append(pill);
  }

  const cards = el("div", { class: "model-options" });
  const paneBuiltin = el("div", { class: "model-pane" });
  paneBuiltin.append(cards);

  const searchInput = el("input", {
    type: "text",
    class: "model-search-input",
    placeholder: "Search Hugging Face, or paste a repository link",
  });
  const searchStatus = el("p", { class: "model-status" });
  const results = el("div", { class: "model-results" });
  const paneCustom = el("div", { class: "model-pane" });
  const searchBox = el("div", { class: "model-search" });
  searchBox.append(searchInput, searchStatus, results);
  paneCustom.append(searchBox);

  const panel = el("div", { class: "model-profile" });
  const controls = el("div", { class: "model-controls" });
  panel.hidden = true;

  for (const option of field.options || []) {
    const meta = option.metadata || {};
    const card = el("button", { type: "button", class: "model-option", "data-ref": option.value });
    card.disabled = disabled;
    const title = el("span", { class: "model-option-title" });
    title.append(el("strong", { text: option.label || option.value }));
    if (option.recommended) title.append(badge("Recommended", "recommend"));
    if (meta.route) title.append(badge(meta.route, "route"));
    if (meta.runnable === false) title.append(badge("not runnable", "warn"));
    const copy = el("span", { class: "model-option-copy" });
    copy.append(title);
    if (option.description) copy.append(el("small", { text: option.description }));
    card.append(copy);
    card.addEventListener("click", () => pick(option.value));
    cards.append(card);
  }

  root.append(sourceRow, paneBuiltin, paneCustom, panel, controls);
  group.append(root);

  function setSource(next, clear) {
    source = next;
    for (const [key, input] of sourceInputs) input.checked = key === next;
    paneBuiltin.classList.toggle("active", next === "builtin");
    paneCustom.classList.toggle("active", next === "custom");
    if (clear) {
      searchStatus.textContent = "";
      results.replaceChildren();
      resetSelection();
    }
  }

  function resetSelection() {
    selected = null;
    profile = null;
    labels = [];
    target = "";
    panel.hidden = true;
    panel.replaceChildren();
    controls.replaceChildren();
    markSelected();
  }

  if (disabled) {
    // Only the workflow wizard builds a model field, and it never renders a
    // submitted form — show the recorded choice and stop there.
    sourceRow.hidden = true;
    paneBuiltin.hidden = true;
    paneCustom.hidden = true;
    const value = answers[field.id];
    if (value && value.ref) {
      panel.hidden = false;
      panel.append(el("p", { class: "model-status", text: "Selected: " + value.ref }));
    }
    return () => (answers[field.id] == null ? null : answers[field.id]);
  }

  function markSelected() {
    for (const node of root.querySelectorAll("[data-ref]")) {
      node.classList.toggle("selected", node.dataset.ref === selected);
    }
  }

  function pick(ref) {
    selected = ref;
    profile = null;
    labels = [];
    target = "";
    markSelected();
    loadProfile(ref);
  }

  async function loadProfile(ref) {
    const token = ++profileSeq;
    panel.hidden = false;
    panel.replaceChildren(el("p", { class: "model-status", text: "Resolving " + ref + "…" }));
    controls.replaceChildren();
    try {
      const resolved = await modelProfile(ref, task);
      if (token !== profileSeq) return;
      profile = resolved || null;
    } catch (error) {
      if (token !== profileSeq) return;
      profile = { ref, supported: false, reason: error.message || String(error) };
    }
    renderProfile();
    renderControls();
  }

  function renderProfile() {
    panel.replaceChildren();
    if (!profile) return;
    const head = el("div", { class: "model-profile-head" });
    head.append(el("strong", { text: profile.ref || selected }));
    const wrongTask = profile.supported === false;
    const cannotRun = !wrongTask && profile.runnable === false;
    const unusable = wrongTask || cannotRun;
    head.append(
      el("span", {
        class: "model-verdict " + (unusable ? "bad" : "good"),
        text: wrongTask
          ? "cannot do " + (task || "this task")
          : cannotRun
            ? "cannot run here yet"
            : "can run " + (task || "this task"),
      })
    );
    panel.append(head);

    const meta = el("div", { class: "model-badges" });
    if (profile.route) meta.append(badge(profile.route, "route"));
    if (profile.revision) meta.append(badge("rev " + profile.revision));
    if (profile.source) meta.append(badge(profile.source));
    if (Array.isArray(profile.tasks) && profile.tasks.length) {
      meta.append(badge(profile.tasks.join(", ")));
    }
    if (meta.children.length) panel.append(meta);

    if (unusable) {
      panel.append(
        el("p", {
          class: "model-error",
          text:
            profile.reason ||
            (wrongTask
              ? "This model cannot do the " + (task || "required") + " task. Pick another one."
              : "This model cannot run on this machine yet."),
        })
      );
    } else if (profile.reason) {
      panel.append(el("p", { class: "model-status", text: profile.reason }));
    }
  }

  /** Whether the chosen model still cannot be used, either way. */
  function blocked(profile) {
    return !profile || profile.supported === false || profile.runnable === false;
  }

  function renderControls() {
    controls.replaceChildren();
    if (blocked(profile)) return;
    const hasLabels = Array.isArray(profile.labels) && profile.labels.length > 0;
    if (hasLabels) controls.append(buildLabels(profile.labels));
    // The free-text box is always offered: with a label list it is how someone
    // asks for a class the model may know nothing about, which gets a warning.
    controls.append(buildTarget({ optional: hasLabels, labels: hasLabels ? profile.labels : [] }));
  }

  function buildLabels(choices) {
    const box = el("div", { class: "model-labels" });
    const head = el("div", { class: "model-labels-head" });
    head.append(el("span", { class: "wizard-label", text: "Labels" }));
    const all = el("button", { type: "button", class: "aoi-button", text: "All" });
    const none = el("button", { type: "button", class: "aoi-button", text: "None" });
    head.append(all, none);
    const list = el("div", { class: "model-label-list" });
    const inputs = [];
    for (const label of choices) {
      const input = el("input", { type: "checkbox", value: label });
      input.addEventListener("change", sync);
      const item = el("label", { class: "model-label" });
      item.append(input, el("span", { text: label }));
      inputs.push(input);
      list.append(item);
    }
    function sync() {
      labels = inputs.filter((input) => input.checked).map((input) => input.value);
    }
    all.addEventListener("click", () => {
      for (const input of inputs) input.checked = true;
      sync();
    });
    none.addEventListener("click", () => {
      for (const input of inputs) input.checked = false;
      sync();
    });
    box.append(head, list);
    return box;
  }

  function buildTarget({ optional, labels: known }) {
    const box = el("div", { class: "model-target" });
    const input = el("input", {
      type: "text",
      value: target,
      placeholder: optional ? "a class this model may predict" : "What should it find?",
    });
    box.append(
      el("label", {
        class: "wizard-label",
        text: optional ? "Or type a class" : "What should it find?",
      }),
      input
    );
    // A class the model cannot produce is worth saying out loud at selection
    // time; typing nothing is fine — the model's own output is the target then.
    const warning = el("p", { class: "model-warning" });
    warning.hidden = true;
    if (known.length) box.append(warning);
    input.addEventListener("input", () => {
      target = input.value.trim();
      if (!known.length) return;
      const problem = unknownLabel(target, known);
      warning.textContent = problem;
      warning.hidden = !problem;
    });
    return box;
  }

  searchInput.addEventListener("input", () => {
    clearTimeout(debounce);
    const query = searchInput.value.trim();
    results.replaceChildren();
    if (!query) {
      searchStatus.textContent = "";
      return;
    }
    // A pasted link resolves straight through the profile endpoint; keyword
    // search is only for everything else.
    if (/^https?:\/\//i.test(query)) {
      searchStatus.textContent = "";
      pick(query);
      return;
    }
    searchStatus.textContent = "Searching…";
    debounce = setTimeout(() => runSearch(query), 400);
  });

  async function runSearch(query) {
    const token = ++searchSeq;
    results.replaceChildren();
    try {
      const payload = await searchModels(query, task, 20);
      if (token !== searchSeq) return;
      const hits = (payload && payload.results) || [];
      searchStatus.textContent = hits.length ? "" : "No models found.";
      for (const hit of hits) results.append(buildResult(hit));
      markSelected();
    } catch (error) {
      if (token !== searchSeq) return;
      searchStatus.textContent = error.message || String(error);
    }
  }

  function buildResult(hit) {
    const row = el("button", { type: "button", class: "model-result", "data-ref": hit.ref });
    const copy = el("span", { class: "model-result-copy" });
    copy.append(el("strong", { text: hit.repo || hit.ref }));
    const facts = [];
    if (hit.downloads != null) facts.push(count(hit.downloads) + " downloads");
    if (hit.likes != null) facts.push(count(hit.likes) + " likes");
    if (hit.library) facts.push(hit.library);
    if (facts.length) copy.append(el("small", { text: facts.join(" · ") }));
    const badges = el("span", { class: "model-badges" });
    const meta = hit.profile || {};
    if (hit.gated) badges.append(badge("gated", "warn"));
    if (Array.isArray(meta.labels) && meta.labels.length) {
      badges.append(badge(meta.labels.length + " labels"));
    }
    if (meta.promptable) badges.append(badge("promptable"));
    row.append(copy, badges);
    row.addEventListener("click", () => pick(hit.ref));
    return row;
  }

  // Out-of-the-box default: a curated option named by the server resolves its
  // profile on render so the wizard opens ready to submit.
  setSource(source, false);
  if (field.default) {
    const match = (field.options || []).find((option) => option.value === field.default);
    if (match) pick(field.default);
  }

  return () => {
    if (blocked(profile)) return null;
    return { ref: selected, labels: labels.slice(), target };
  };
}

/** Lower-case a label and drop a trailing plural, so "Cars" matches "car". */
function normalizeLabel(text) {
  const cleaned = String(text == null ? "" : text).trim().toLowerCase();
  if (cleaned.endsWith("s") && !cleaned.endsWith("ss") && cleaned.length > 3) {
    return cleaned.slice(0, -1);
  }
  return cleaned;
}

/** The warning for a typed class the model does not predict, or "". */
const MAX_NAMES_IN_WARNING = 8;

function unknownLabel(text, known) {
  const wanted = normalizeLabel(text);
  if (!wanted) return "";
  if (known.some((label) => normalizeLabel(label) === wanted)) return "";
  const names = known.slice(0, MAX_NAMES_IN_WARNING).join(", ");
  const rest = known.length > MAX_NAMES_IN_WARNING ? ", …" : "";
  return (
    "“" + text + "” is not one of this model's classes — it predicts " + names + rest + "."
  );
}

function badge(text, kind) {
  return el("span", { class: "model-badge" + (kind ? " " + kind : ""), text });
}

function count(value) {
  const number = Number(value) || 0;
  if (number >= 1000000) return (number / 1000000).toFixed(1) + "M";
  if (number >= 1000) return (number / 1000).toFixed(1) + "k";
  return String(number);
}

/* -- area of interest ----------------------------------------------------- */

function buildAoi(group, field, { disabled }) {
  // Resolved value: the contract's AOI object. `null` until a mode produces a
  // bbox, which is what the required-field check looks at.
  let resolved = null;

  const modes = el("div", { class: "aoi-modes" });
  const panePlace = el("div", { class: "aoi-pane" });
  const paneBbox = el("div", { class: "aoi-pane" });
  const paneMap = el("div", { class: "aoi-pane" });
  const panes = { place: panePlace, bbox: paneBbox, view: paneMap };
  const radios = [];
  let pane = "place";

  for (const [value, label] of [
    ["place", "Place name"],
    ["bbox", "Bounding box"],
    ["view", "From the map"],
  ]) {
    const input = el("input", {
      type: "radio",
      name: "aoi-" + (field.id || "aoi"),
      value,
    });
    input.checked = value === pane;
    input.disabled = disabled;
    input.addEventListener("change", () => {
      if (input.checked) select(value);
    });
    const badge = el("label", { class: "aoi-mode" });
    badge.append(input, el("span", { text: label }));
    radios.push(input);
    modes.append(badge);
  }

  function select(next) {
    pane = next;
    for (const [key, node] of Object.entries(panes)) node.hidden = key !== next;
  }

  // place
  const placeInput = el("input", {
    type: "text",
    class: "aoi-input",
    placeholder: "e.g. Berlin, Germany",
  });
  const placeStatus = el("p", { class: "aoi-status" });
  const placeBtn = el("button", { type: "button", class: "aoi-button", text: "Look up" });
  placeBtn.disabled = disabled;
  placeBtn.addEventListener("click", async () => {
    const q = placeInput.value.trim();
    if (!q) return;
    placeStatus.textContent = "Looking up…";
    try {
      const hit = await geocodePlace(q);
      resolved = {
        mode: "place",
        bbox: hit.bbox,
        geojson: null,
        label: hit.label || hit.place || q,
      };
      placeStatus.textContent = resolved.label + " — " + formatBbox(hit.bbox);
      // A name the user typed is a place they want to see: frame it on the map
      // so they can draw over it. Failing is not worth a message — the map may
      // not be up yet.
      fitBounds(hit.bbox).catch(() => {});
    } catch (error) {
      resolved = null;
      placeStatus.textContent = error.message || String(error);
    }
  });
  const placeRow = el("div", { class: "aoi-row" });
  placeRow.append(placeInput, placeBtn);
  panePlace.append(placeRow, placeStatus);
  panePlace.dataset.mode = "place";

  // bbox
  const bboxInput = el("input", {
    type: "text",
    class: "aoi-input",
    placeholder: "w,s,e,n  e.g. 13.0,52.3,13.8,52.7",
  });
  bboxInput.disabled = disabled;
  const bboxStatus = el("p", { class: "aoi-status" });
  const readBbox = () => {
    const parts = bboxInput.value.split(/[\s,]+/).filter(Boolean).map(Number);
    if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n))) {
      resolved = null;
      bboxStatus.textContent = "Enter four numbers: west, south, east, north.";
      return;
    }
    const [w, s, e, n] = parts;
    if (w >= e || s >= n || s < -90 || n > 90) {
      resolved = null;
      bboxStatus.textContent = "West < east and south < north required.";
      return;
    }
    resolved = { mode: "bbox", bbox: [w, s, e, n], geojson: null, label: null };
    bboxStatus.textContent = formatBbox([w, s, e, n]);
  };
  bboxInput.addEventListener("input", readBbox);
  paneBbox.append(bboxInput, bboxStatus);
  paneBbox.dataset.mode = "bbox";

  // map
  const mapStatus = el("p", { class: "aoi-status" });
  const viewBtn = el("button", { type: "button", class: "aoi-button", text: "Use map view" });
  const drawnBtn = el("button", { type: "button", class: "aoi-button", text: "Use drawn shape" });
  const showBtn = el("button", { type: "button", class: "aoi-button", text: "Show on the map" });
  viewBtn.disabled = disabled;
  drawnBtn.disabled = disabled;
  showBtn.disabled = disabled;
  // The map's own GeoEditor is the drawing tool; the bridge cannot open it (no
  // such command), so the wizard points at it instead of pretending to.
  const mapHelp = el("div", { class: "aoi-map-help" });
  mapHelp.append(
    el("strong", { text: "Draw the area on the map" }),
    el("ol", {}, [
      el("li", { text: "Pick the GeoEditor tool in the map toolbar." }),
      el("li", { text: "Sketch the area, then press Save there." }),
      el("li", { text: "Return here and press Use drawn shape." }),
    ]),
    el("small", { text: "The map stays live beside this panel while you choose." })
  );
  //: Id of the preview layer this widget pushed onto the map, so the next
  //: preview replaces it instead of stacking layers.
  let previewLayer = "";

  /** Show the resolved area on the map: frame it and outline it. */
  async function showOnMap() {
    if (!resolved || !resolved.bbox) {
      mapStatus.textContent = "Pick an area first — a place, a bounding box, the view, or a shape.";
      return;
    }
    const [w, s, e, n] = resolved.bbox;
    const ring = [
      [w, s],
      [e, s],
      [e, n],
      [w, n],
      [w, s],
    ];
    const geojson =
      resolved.geojson ||
      {
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            properties: {},
            geometry: { type: "Polygon", coordinates: [ring] },
          },
        ],
      };
    try {
      await fitBounds(resolved.bbox);
      if (previewLayer) await removeLayer(previewLayer).catch(() => {});
      previewLayer = (await addGeoJsonLayer("Workflow AOI", geojson)) || "";
      mapStatus.textContent = "Shown on the map: " + formatBbox(resolved.bbox);
    } catch (error) {
      mapStatus.textContent = error.message || String(error);
    }
  }

  viewBtn.addEventListener("click", async () => {
    mapStatus.textContent = "Reading the map view…";
    try {
      const view = await getView();
      if (!view || !view.bbox) {
        resolved = null;
        mapStatus.textContent = "The map did not report a view.";
        return;
      }
      resolved = { mode: "view", bbox: view.bbox, geojson: null, label: null };
      mapStatus.textContent = formatBbox(view.bbox);
    } catch (error) {
      resolved = null;
      mapStatus.textContent = error.message || String(error);
    }
  });
  drawnBtn.addEventListener("click", async () => {
    mapStatus.textContent = "Reading drawn shapes…";
    try {
      const features = await getDrawnFeatures();
      const shape = (features || []).find((feature) => {
        const type = feature && feature.geometry && feature.geometry.type;
        return type === "Polygon" || type === "MultiPolygon";
      });
      if (!shape) {
        resolved = null;
        mapStatus.textContent =
          "Nothing drawn yet — draw a shape with the map's GeoEditor tool, then press again.";
        return;
      }
      const bbox = geometryBbox(shape.geometry);
      resolved = {
        mode: "drawn",
        bbox,
        geojson: { type: "FeatureCollection", features: [shape] },
        label: null,
      };
      mapStatus.textContent = formatBbox(bbox);
    } catch (error) {
      resolved = null;
      mapStatus.textContent = error.message || String(error);
    }
  });
  const mapRow = el("div", { class: "aoi-row" });
  mapRow.append(viewBtn, drawnBtn, showBtn);
  showBtn.addEventListener("click", showOnMap);
  paneMap.append(mapRow, mapStatus, mapHelp);
  paneMap.dataset.mode = "view";

  select(pane);
  group.append(
    el("p", {
      class: "aoi-note",
      text:
        "The area bounds the imagery the workflow searches and clips. Pick whichever input is easiest; you can refine it later.",
    }),
    modes,
    panePlace,
    paneBbox,
    paneMap
  );
  return () => resolved;
}

function formatBbox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return "";
  return "[" + bbox.map((n) => Number(n).toFixed(4)).join(", ") + "]";
}

function geometryBbox(geometry) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const walk = (node) => {
    if (!Array.isArray(node)) return;
    if (typeof node[0] === "number" && typeof node[1] === "number") {
      minX = Math.min(minX, node[0]);
      minY = Math.min(minY, node[1]);
      maxX = Math.max(maxX, node[0]);
      maxY = Math.max(maxY, node[1]);
      return;
    }
    for (const child of node) walk(child);
  };
  walk(geometry && geometry.coordinates);
  if (!Number.isFinite(minX)) return null;
  return [minX, minY, maxX, maxY];
}
