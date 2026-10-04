/* Cells tab: the cell list, standalone jobs, and the cell actions. */

import { el, toast } from "../dom.js";
import { addCell as createCell, updateCell } from "../api.js";
import { hold, isFollowing, scrollTo } from "../scroll.js";
import { jobsFor, normalizeCells, setState, state } from "../store.js";
import { renderCell } from "../components/cell.js";
import { renderJob } from "../components/progress.js";

export function renderAddCellRow() {
  const row = el("div", { class: "add-cell-row" });
  row.append(
    el("button", { text: "+ Markdown", onclick: () => addCell("markdown") }),
    el("button", { text: "+ Python", onclick: () => addCell("python") }),
    el("button", { text: "+ Prompt", onclick: () => addCell("prompt") })
  );
  return row;
}

export function renderCellsTab() {
  const wrap = el("div", {});

  if (!state.active_workspace) {
    wrap.append(
      el("div", { class: "empty-hint", text: "No workspace open — File → New or Open" })
    );
    return wrap;
  }

  if (!state.cells.length && !(state.jobs || []).length) {
    wrap.append(el("div", { class: "empty-hint", text: "No cells yet — add a cell above." }));
    return wrap;
  }

  const attached = new Set();
  for (const cell of state.cells) {
    const jobs = jobsFor(cell.id);
    wrap.append(renderCell(cell, jobs));
    for (const job of jobs) attached.add(job.job_id);
  }
  for (const job of state.jobs || []) {
    if (!attached.has(job.job_id)) wrap.append(renderJob(job));
  }
  return wrap;
}

/** Repaint the cells tab in place: follow new content, or hold the reader's place. */
export function renderCellsOnly() {
  const content = document.getElementById("tab-content");
  if (!content || state.selected_tab !== "Cells") return;

  const previous = content.scrollTop;
  content.replaceChildren(renderCellsTab());
  if (isFollowing()) {
    hold();
    return;
  }
  // Reading history: keep the reader where they were, clamped to the new height.
  const limit = Math.max(0, content.scrollHeight - content.clientHeight);
  window.requestAnimationFrame(() => {
    scrollTo(content, Math.min(previous, limit));
  });
}

/** Swap a markdown cell's rendered body for a textarea that saves on blur. */
export function editMarkdown(cell) {
  const box = document.querySelector('.cell[data-cell-id="' + cell.id + '"]');
  if (!box) return;
  const body = box.querySelector(".cell-body");
  if (!body) return;
  const source = cell.source == null ? "" : String(cell.source);
  const ta = el("textarea", { rows: Math.min(12, Math.max(2, source.split("\n").length)) });
  ta.value = source;
  ta.addEventListener("blur", async () => {
    const idx = state.cells.findIndex((c) => c.id === cell.id);
    if (idx >= 0) state.cells[idx].source = ta.value;
    try {
      await updateCell(cell.id, ta.value);
    } catch (e) {
      toast(e.message || String(e));
    }
    renderCellsOnly();
  });
  body.replaceChildren(ta);
  ta.focus();
}

export function focusCell(cell) {
  window.requestAnimationFrame(() => {
    const box = document.querySelector('.cell[data-cell-id="' + cell.id + '"]');
    if (!box) return;
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
    if (cell.kind === "markdown") {
      editMarkdown(cell);
    } else {
      const ta = box.querySelector("textarea");
      if (ta) ta.focus();
    }
  });
}

export async function addCell(kind) {
  if (!state.active_workspace) {
    toast("No workspace open");
    return;
  }
  const before = new Set(state.cells.map((cell) => cell.id));
  try {
    const snap = await createCell(kind);
    const patch = {};
    if (Array.isArray(snap.cells)) patch.cells = normalizeCells(snap.cells);
    if (Array.isArray(snap.jobs)) patch.jobs = snap.jobs;
    setState(patch);
    const cell = state.cells.find((candidate) => !before.has(candidate.id));
    if (cell) {
      focusCell(cell);
      // Adding a cell is an explicit action, so follow it even if the reader
      // was somewhere else in the history.
      const follow = panelFollower();
      if (follow) follow.jump();
    }
  } catch (e) {
    toast(e.message || String(e));
  }
}
