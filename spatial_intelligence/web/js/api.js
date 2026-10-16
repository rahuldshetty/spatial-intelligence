/* Server API: one async function per endpoint. Every call returns the parsed
JSON body; failures raise an Error carrying the server's `detail` message. */

"use strict";

import { state } from "./store.js";

export async function request(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      if (payload.detail) {
        detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch (_) {
      /* non-JSON error body — keep the status text */
    }
    throw new Error(detail);
  }
  return response.json();
}

export function getState() {
  return request("GET", "/api/state");
}

export function updateSettings(patch) {
  return request("PUT", "/api/settings", patch);
}

export function listWorkflows() {
  return request("GET", "/api/workflows");
}

export function searchModels(q, task, limit = 20) {
  const params = new URLSearchParams({ q, task: task || "" });
  if (limit != null) params.set("limit", String(limit));
  return request("GET", "/api/models/search?" + params.toString());
}

export function modelProfile(model, task) {
  return request("POST", "/api/models/profile", { model, task: task || "" });
}

export function geocodePlace(q) {
  return request("GET", "/api/geocode?q=" + encodeURIComponent(q));
}

export function createWorkspace(name, workflow = "", params = {}) {
  return request("POST", "/api/workspace/new", { name, workflow, params });
}

export function openWorkspace(name) {
  return request("POST", "/api/workspace/open", { name });
}

export function saveWorkspace() {
  return request("POST", "/api/workspace/save");
}

export function closeWorkspace() {
  return request("POST", "/api/workspace/close");
}

export function addCell(kind) {
  return request("POST", "/api/cells", { kind, source: "", index: null });
}

export function updateCell(cellId, source) {
  return request("PUT", "/api/cells/" + cellId, { source });
}

export function deleteCell(cellId) {
  return request("DELETE", "/api/cells/" + cellId);
}

export async function moveCell(cellId, direction) {
  const index = state.cells.findIndex((cell) => cell.id === cellId);
  if (index < 0) return null;
  const target = index + direction;
  if (target < 0 || target >= state.cells.length) return null;
  return request("POST", "/api/cells/" + cellId + "/move", { index: target });
}

export function runCell(cellId) {
  return request("POST", "/api/cells/" + cellId + "/run");
}

export function stopCell(cellId) {
  return request("POST", "/api/cells/" + cellId + "/stop");
}

export function runAll() {
  return request("POST", "/api/run-all");
}

export function respondInteraction(cellId, interactionId, answers) {
  return request("POST", "/api/cells/" + cellId + "/interaction", {
    interaction_id: interactionId,
    answers,
  });
}

export function cancelInteraction(cellId, interactionId) {
  return request("DELETE", "/api/cells/" + cellId + "/interaction/" + interactionId);
}

export async function importFiles(fileList) {
  const form = new FormData();
  for (const file of fileList) {
    form.append("files", file, file.webkitRelativePath || file.name);
  }
  const response = await fetch("/api/import/local", { method: "POST", body: form });
  if (!response.ok) {
    let detail = "import failed (" + response.status + ")";
    try {
      const payload = await response.json();
      if (payload.detail) detail = payload.detail;
    } catch (_) {
      /* keep the status message */
    }
    throw new Error(detail);
  }
  return response.json();
}

export function importUrl(url) {
  return request("POST", "/api/import/url", { url, filename: null });
}

export function setMapProject(project) {
  return request("POST", "/api/map/project", { project });
}

export function reportBridge(payload) {
  return request("POST", "/api/geolibre/bridge", payload);
}
