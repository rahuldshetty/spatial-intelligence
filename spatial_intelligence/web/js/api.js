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

/**
 * Upload files with byte progress.
 *
 * The regular :func:`importFiles` is fine when nothing is watching; a wizard
 * that reports per-item progress needs the upload's loaded/total counters,
 * which only ``XMLHttpRequest`` exposes.
 */
export function importFilesWithProgress(fileList, onProgress) {
  const form = new FormData();
  for (const file of fileList) {
    form.append("files", file, file.webkitRelativePath || file.name);
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/import/local");
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", (event) => {
      if (!onProgress || !event.lengthComputable || !event.total) return;
      onProgress(Math.min(1, event.loaded / event.total));
    });
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response || {});
        return;
      }
      const payload = xhr.response;
      reject(
        new Error(
          (payload && payload.detail) || "import failed (" + xhr.status + ")"
        )
      );
    };
    xhr.onerror = () => reject(new Error("upload failed"));
    xhr.onabort = () => reject(new Error("upload cancelled"));
    xhr.send(form);
  });
}

export function importUrl(url, filename = null) {
  return request("POST", "/api/import/url", { url, filename });
}

export function detectDroneFiles(paths) {
  return request("POST", "/api/drone/detect", { paths });
}

export function importDroneFiles(paths, orthorectify = false) {
  return request("POST", "/api/drone/import", { paths, orthorectify });
}

export function deleteWorkspaceFile(path) {
  return request("POST", "/api/files/delete", { path });
}

export function renameWorkspaceFile(path, name) {
  return request("POST", "/api/files/rename", { path, name });
}

/** Ask the server whether a URL can be downloaded before creating a workspace. */
export function checkImportUrl(url) {
  return request("POST", "/api/import/check", { url });
}

export function setMapProject(project, revision = null, workspace = null) {
  const body = { project };
  if (revision != null) body.revision = revision;
  if (workspace != null) body.workspace = workspace;
  return request("POST", "/api/map/project", body);
}

/** Resolve a pending JSON map query the server sent over SSE. */
export function sendMapBridge(requestId, token, result, error = null) {
  return request("POST", "/api/map/bridge/" + encodeURIComponent(requestId), {
    token,
    result,
    error,
  });
}

/** Resolve a pending capture request the server sent over SSE. */
export function sendMapCapture(captureId, token, image, view, error = null) {
  return request("POST", "/api/map/capture/" + encodeURIComponent(captureId), {
    token,
    image,
    view,
    error,
  });
}

export function reportBridge(payload) {
  return request("POST", "/api/geolibre/bridge", payload);
}
