/* GeoLibre map panel: one long-lived iframe plus the postMessage bridge.

The iframe is created a single time per workspace and never rebuilt for cell,
trace, job, or file updates — only a workspace switch restarts the embedded
app so no legend/colorbar/plugin state leaks across workspaces.
*/

"use strict";

import { setMapProject, reportBridge } from "../api.js";
import { el } from "../dom.js";
import { state } from "../store.js";

const BRIDGE_METHODS = [
  "project.load",
  "project.request_state",
  "toImage",
  "getView",
  "getDrawnFeatures",
];

let panel = null;
let iframe = null;
let iframeSrc = null;
let iframeWorkspace = null;
let mapReady = false;
let lastRemoteProject = null;
let mapSeq = 0;
let listening = false;

export function mountMap() {
  if (!panel) {
    panel = el("div", { id: "map-panel" });
    panel.append(el("div", { id: "map-error" }));
    attachBridge();
  }
  syncMap();
  return panel;
}

export function syncMap() {
  if (!panel || !state.map_app_url) return;

  if (iframeWorkspace !== state.active_workspace) {
    // Workspace switched (or first mount): drop the old app so GeoLibre boots
    // fresh. `geolibre:ready` re-posts the project for the new workspace.
    if (iframe && iframe.isConnected) iframe.remove();
    iframe = null;
    iframeSrc = null;
    mapReady = false;
    lastRemoteProject = null;
    iframeWorkspace = state.active_workspace;
  }

  const src = mapSrc();
  if (iframeSrc !== src) {
    // A different theme (or map app URL) is a different document, so the
    // embedded app must reload to pick it up.
    if (src !== null) {
      if (!iframe) iframe = el("iframe", { allow: "fullscreen" });
      iframe.setAttribute("src", src);
      iframeSrc = src;
      mapReady = false;
      lastRemoteProject = null;
    }
  }
  if (iframe && !iframe.isConnected) panel.append(iframe);
  postProject();
}

function mapSrc() {
  const base = state.map_app_url;
  if (!base) return null;
  const theme = (state.settings && state.settings.theme) === "dark" ? "dark" : "light";
  return base + "index.html?embed=1&theme=" + theme + "&layout=embed";
}

function mapOrigin() {
  if (!state.map_app_url) return null;
  try {
    return new URL(state.map_app_url).origin;
  } catch (_) {
    return null;
  }
}

function postProject() {
  if (!iframe || !iframe.contentWindow || !mapReady) return;
  if (!state.map_project) return;
  const target = mapOrigin();
  if (!target) return;
  if (lastRemoteProject && sameProject(state.map_project, lastRemoteProject)) return;
  mapSeq += 1;
  iframe.contentWindow.postMessage(
    { type: "geolibre:load-project", seq: mapSeq, project: state.map_project },
    target
  );
}

function attachBridge() {
  if (listening) return;
  listening = true;
  window.addEventListener("message", (event) => {
    const origin = mapOrigin();
    if (!origin || event.origin !== origin) return;
    if (!iframe || event.source !== iframe.contentWindow) return;
    const data = event.data;
    if (!data || typeof data !== "object") return;

    if (data.type === "geolibre:ready") {
      mapReady = true;
      reportBridge({ version: data.version || null, methods: BRIDGE_METHODS }).catch(() => {});
      postProject();
    } else if (data.type === "geolibre:state") {
      lastRemoteProject = data.project;
      // Track app-initiated edits (pan/zoom/layer edits) in the local project
      // too. Otherwise the next shell render posts the stale project back and
      // snaps the map to its previous view.
      if (data.project) {
        state.map_project = data.project;
        setMapProject(data.project).catch(() => {});
      }
    } else if (data.type === "geolibre:error") {
      const error = document.getElementById("map-error");
      if (error) {
        error.style.display = "block";
        error.textContent = "Map error: " + (data.message || "");
      }
    }
  });
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value).sort()) out[key] = canonical(value[key]);
    return out;
  }
  return value;
}

function sameProject(a, b) {
  if (a == null || b == null) return a === b;
  return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
}
