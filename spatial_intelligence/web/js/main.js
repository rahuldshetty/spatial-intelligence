/* Boot: mount the shell, wire SSE, load the initial snapshot. */

"use strict";

import {
  getState,
  runAll,
  runCell,
  saveWorkspace,
  sendMapBridge,
  sendMapCapture,
} from "./api.js";
import { bridgeRequest } from "./bridge.js";
import { closeAllMenus } from "./components/menubar.js";
import { revealInteraction } from "./components/interaction.js";
import { updateJobNode } from "./components/progress.js";
import { refreshStatusBar, setConnection } from "./components/status-bar.js";
import { openSettingsDialog } from "./components/settings-dialog.js";
import { closeDialog, toast } from "./dom.js";
import { applyTrace, flushPendingTrace, repaintTrace } from "./components/trace.js";
import { connectEvents } from "./events.js";
import { renderCellsOnly } from "./pages/cells.js";
import { renderDataOnly } from "./pages/data.js";
import { mountShell, renderShell } from "./pages/shell.js";
import { syncMap } from "./pages/map.js";
import { jump } from "./scroll.js";
import { applySnapshot, isGeneratedCell, normalizeCells, setFiles, setState, state, subscribe, upsertJob } from "./store.js";

export function boot() {
  mountShell(document.getElementById("app"));
  subscribe(() => renderShell());
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".menu")) closeAllMenus();
  });
  document.addEventListener("keydown", handleShortcut);
  connectEvents({
    onCell: handleCell,
    onCellDeleted: handleCellDeleted,
    onTrace: handleTrace,
    onJob: handleJob,
    onJobs: handleJobs,
    onMap: handleMap,
    onFiles: handleFiles,
    onSettings: handleSettings,
    onVision: handleVision,
    onRunning: handleRunning,
    onMapCapture: handleMapCapture,
    onMapBridge: handleMapBridge,
    onStatus: setConnection,
  });
  loadState();
}

function handleShortcut(event) {
  if (event.key === "Escape") {
    closeAllMenus();
    closeDialog();
    return;
  }
  if (!(event.metaKey || event.ctrlKey)) return;
  const key = event.key.toLowerCase();
  if (key === "s") {
    event.preventDefault();
    if (!state.active_workspace) {
      toast("No workspace open", "warning");
      return;
    }
    saveWorkspace()
      .then(() => toast("Workspace saved", "success"))
      .catch((error) => toast(error.message || String(error), "error"));
  } else if (event.key === ",") {
    event.preventDefault();
    openSettingsDialog();
  } else if (event.key === "Enter") {
    event.preventDefault();
    if (event.shiftKey) {
      runAll()
        .then(jump)
        .catch((error) => toast(error.message || String(error), "error"));
      return;
    }
    const cell = document.activeElement && document.activeElement.closest(".cell");
    const cellId = cell && cell.getAttribute("data-cell-id");
    if (cellId) {
      runCell(cellId)
        .then(jump)
        .catch((error) => toast(error.message || String(error), "error"));
    }
  }
}

async function loadState() {
  applySnapshot(await getState());
  // Steps streamed while the snapshot was in flight are not part of it.
  flushPendingTrace();
  const pending = state.cells.find(
    (cell) => cell.status === "waiting_for_input" && cell.interaction
  );
  if (pending) revealInteraction(pending.id);
}

function handleCell(cell) {
  if (!cell || isGeneratedCell(cell)) return;
  const normalized = normalizeCells([cell])[0];
  if (!normalized) return;
  const index = state.cells.findIndex((item) => item.id === normalized.id);
  if (index === -1) {
    if (normalized.kind) state.cells.push(normalized);
  } else {
    state.cells[index] = { ...state.cells[index], ...normalized };
  }
  renderCellsOnly();
  refreshStatusBar();
  if (normalized.status === "waiting_for_input" && normalized.interaction) {
    revealInteraction(normalized.id);
  }
}

function handleCellDeleted(data) {
  const cellId = data && data.id;
  if (!cellId) return;
  const cells = state.cells.filter((cell) => cell.id !== cellId);
  if (cells.length === state.cells.length) return;
  const jobs = (state.jobs || []).filter((job) => job.parent_id !== cellId);
  setState({ cells, jobs });
}

function handleTrace(data) {
  if (!data) return;
  applyTrace(data);
}

function handleVision(data) {
  if (!data) return;
  state.vision = data;
  refreshStatusBar();
}

function handleRunning(data) {
  state.running_cell = (data && data.cell_id) || null;
  refreshStatusBar();
}

/** Map queries the app is allowed to answer for the server. */
const MAP_QUERY_METHODS = new Set(["getDrawnFeatures"]);

/**
 * Whether this tab owns the workspace a map request was sent for.
 *
 * Capture and query requests are broadcast over SSE to every open tab. A tab
 * on another workspace must stay silent: answering with an error resolves the
 * server's pending request and poisons it for the tab that can actually
 * answer. The server times out with a useful message if no tab owns it.
 */
function ownsMapWorkspace(workspace) {
  return !workspace || workspace === state.active_workspace;
}

/** Answer a server JSON map query with the live map's data. */
async function handleMapBridge(data) {
  if (!data || typeof data.id !== "string" || typeof data.token !== "string") return;
  if (!ownsMapWorkspace(data.workspace)) return;
  try {
    if (!MAP_QUERY_METHODS.has(data.method)) {
      throw new Error("Unsupported map query: " + String(data.method));
    }
    const result = await bridgeRequest(data.method, data.params || {});
    await sendMapBridge(data.id, data.token, result);
  } catch (error) {
    await sendMapBridge(
      data.id,
      data.token,
      null,
      error && error.message ? error.message : String(error)
    ).catch(() => {});
  }
}

/** Answer a server capture request with the current map canvas. */
async function handleMapCapture(data) {
  if (!data || typeof data.id !== "string" || typeof data.token !== "string") return;
  if (!ownsMapWorkspace(data.workspace)) return;
  try {
    const [image, view] = await Promise.all([
      bridgeRequest("toImage", {}),
      bridgeRequest("getView", {}),
    ]);
    await sendMapCapture(data.id, data.token, image, view);
  } catch (error) {
    await sendMapCapture(
      data.id,
      data.token,
      null,
      null,
      error && error.message ? error.message : String(error)
    ).catch(() => {});
  }
}

function handleJob(event) {
  if (!event) return;
  const job = upsertJob(event);
  if (!job) return;
  const node = document.querySelector('[data-job-id="' + job.job_id + '"]');
  if (node) {
    updateJobNode(node, job);
    return;
  }
  // A job the trace does not show yet: repaint the strip it belongs to, so its
  // card lands at the step that opened it instead of at the end of the trace.
  const parent = (state.cells || []).find((cell) => cell.id === job.parent_id);
  if (parent) repaintTrace(parent);
  else renderCellsOnly();
}

/** Replace the browser's job list (the session forgets jobs on re-run/delete). */
function handleJobs(payload) {
  if (!payload) return;
  setState({ jobs: payload.jobs || [] });
  renderCellsOnly();
}

function handleMap(payload) {
  if (!payload) return;
  state.map_project = payload.project != null ? payload.project : null;
  if (payload.revision != null) state.map_revision = payload.revision;
  syncMap();
}

function handleFiles(payload) {
  if (!payload) return;
  setFiles(payload.files || []);
  renderDataOnly();
}

function handleSettings(payload) {
  if (!payload || !payload.settings) return;
  setState({ settings: payload.settings });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
