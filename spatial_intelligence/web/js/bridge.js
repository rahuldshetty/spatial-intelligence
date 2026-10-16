/* GeoLibre postMessage bridge: request/response RPC into the embedded map app.

The map is a cross-origin iframe; every request is `{type:"geolibre:command",
requestId, method, params}` posted to its window, and the app answers
`{type:"geolibre:result", requestId, ok, value}`. This module owns its own
message listener so `pages/map.js` stays untouched.
*/

"use strict";

import { state } from "./store.js";

const TIMEOUT_MS = 8000;

let listening = false;
let seq = 0;
const pending = new Map();

function mapFrame() {
  return document.querySelector("#map-panel iframe");
}

function mapOrigin() {
  if (!state.map_app_url) return null;
  try {
    return new URL(state.map_app_url).origin;
  } catch (_) {
    return null;
  }
}

function attach() {
  if (listening) return;
  listening = true;
  window.addEventListener("message", (event) => {
    if (event.origin !== mapOrigin()) return;
    const data = event.data;
    if (!data || typeof data !== "object" || data.type !== "geolibre:result") return;
    const entry = pending.get(data.requestId);
    if (!entry) return;
    pending.delete(data.requestId);
    clearTimeout(entry.timer);
    if (data.ok) entry.resolve(data.value);
    else entry.reject(new Error(String(data.value == null ? "map command failed" : data.value)));
  });
}

/** Send one command to the map app and resolve with its `value`. */
export function bridgeRequest(method, params = {}) {
  attach();
  const iframe = mapFrame();
  const origin = mapOrigin();
  if (!iframe || !iframe.contentWindow || !origin) {
    return Promise.reject(new Error("The map is not ready yet."));
  }
  const requestId = "geoai-" + Date.now() + "-" + (seq += 1);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(requestId);
      reject(new Error("The map did not answer `" + method + "`."));
    }, TIMEOUT_MS);
    pending.set(requestId, { resolve, reject, timer });
    iframe.contentWindow.postMessage(
      { type: "geolibre:command", requestId, method, params },
      origin
    );
  });
}

/** GeoJSON features the user drew with the map's own GeoEditor tool. */
export function getDrawnFeatures() {
  return bridgeRequest("getDrawnFeatures", {});
}

/** Current view: `{center, zoom, bearing, pitch, bbox}`. */
export function getView() {
  return bridgeRequest("getView", {});
}

/** Move the map to `[west, south, east, north]` so the user sees the area. */
export function fitBounds(bounds) {
  return bridgeRequest("fitBounds", { bounds });
}

/** Add a layer from GeoJSON; resolves with the new layer's id. */
export function addGeoJsonLayer(name, geojson) {
  return bridgeRequest("addGeoJsonLayer", { name, geojson });
}

/** Remove a layer by the id `addGeoJsonLayer` returned. */
export function removeLayer(layerId) {
  return bridgeRequest("removeLayer", { layerId });
}
