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
let lastSentSeq = 0;
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
    mapSeq = 0;
    lastSentSeq = 0;
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
  lastSentSeq = mapSeq;
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
    // A message from the iframe of the workspace we just left must not touch
    // the new workspace's map while syncMap is replacing it.
    if (iframeWorkspace !== state.active_workspace) return;
    const data = event.data;
    if (!data || typeof data !== "object") return;

    if (data.type === "geolibre:ready") {
      mapReady = true;
      reportBridge({ version: data.version || null, methods: BRIDGE_METHODS }).catch(() => {});
      postProject();
    } else if (data.type === "geolibre:state") {
      // The app echoes the `seq` of the last load-project it applied. A lower
      // seq is a state captured before our latest project reached it; posting
      // that back would overwrite the newer server project with stale layers.
      if (lastSentSeq === 0) return;
      const seq = typeof data.seq === "number" ? data.seq : null;
      if (seq !== null && seq < lastSentSeq) return;
      lastRemoteProject = data.project;
      // Track app-initiated edits (pan/zoom/layer edits) in the local project
      // too. Otherwise the next shell render posts the stale project back and
      // snaps the map to its previous view.
      if (data.project) {
        state.map_project = data.project;
        const requestedWorkspace = iframeWorkspace;
        setMapProject(data.project, state.map_revision, requestedWorkspace)
          .then((result) => {
            if (!result || result.workspace_mismatch) return;
            // The tab may have switched workspaces while this push was in
            // flight; its answer belongs to the previous map.
            if (requestedWorkspace !== state.active_workspace) return;
            if (result.workspace != null && result.workspace !== state.active_workspace) {
              return;
            }
            if (result.revision != null) state.map_revision = result.revision;
            if (result.accepted === false && result.project) {
              // Another tab or a tool changed the map first. Adopt the
              // authoritative project and let syncMap push it to the iframe.
              state.map_project = result.project;
              syncMap();
            }
          })
          .catch(() => {});
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
