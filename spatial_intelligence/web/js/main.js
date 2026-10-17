/* Boot: mount the shell, wire SSE, load the initial snapshot. */

"use strict";

import { getState, runAll, runCell, saveWorkspace } from "./api.js";
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
    onTrace: handleTrace,
    onJob: handleJob,
    onJobs: handleJobs,
    onMap: handleMap,
    onFiles: handleFiles,
    onSettings: handleSettings,
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
      runAll().catch((error) => toast(error.message || String(error), "error"));
      return;
    }
    const cell = document.activeElement && document.activeElement.closest(".cell");
    const cellId = cell && cell.getAttribute("data-cell-id");
    if (cellId) runCell(cellId).catch((error) => toast(error.message || String(error), "error"));
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

function handleTrace(data) {
  if (!data) return;
  applyTrace(data);
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
