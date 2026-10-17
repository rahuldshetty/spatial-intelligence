/* Cells tab: the cell list, standalone jobs, and the cell actions. */

import { el, toast } from "../dom.js";
import { addCell as createCell, deleteCell as deleteCellRequest, updateCell } from "../api.js";
import { icon } from "../icons.js";
import { hold, isFollowing, scrollTo } from "../scroll.js";
import { jobsFor, normalizeCells, setState, state } from "../store.js";
import { renderCell } from "../components/cell.js";
import { renderJob } from "../components/progress.js";

function addButton(kind, label, iconName) {
  const button = el("button", { type: "button", onclick: () => addCell(kind) });
  button.append(icon(iconName, { size: 14 }), el("span", { text: label }));
  return button;
}

export function renderAddCellRow() {
  const row = el("div", { class: "add-cell-row" });
  row.append(
    addButton("markdown", "+ Markdown", "text"),
    addButton("python", "+ Python", "code"),
    addButton("prompt", "+ Prompt", "sparkles")
  );
  return row;
}

function emptyState(iconName, title, detail) {
  const box = el("div", { class: "empty-hint" });
  box.append(
    icon(iconName, { size: 26, class: "empty-icon" }),
    el("strong", { class: "empty-title", text: title }),
    el("span", { class: "empty-detail", text: detail })
  );
  return box;
}

export function renderCellsTab() {
  const wrap = el("div", {});

  if (!state.active_workspace) {
    wrap.append(
      emptyState(
        "globe",
        "No workspace open — File → New or Open",
        "A workspace is the folder the agent reads, writes, and maps."
      )
    );
    return wrap;
  }

  if (!state.cells.length && !(state.jobs || []).length) {
    wrap.append(
      emptyState(
        "sparkles",
        "No cells yet — add a cell above.",
        "Add a Prompt cell and ask a question in plain language; or write Python and Markdown yourself."
      )
    );
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
  ta.addEventListener("blur", async (event) => {
    if (deletingCells.has(cell.id)) return;
    const target = event && event.relatedTarget;
    if (
      target &&
      typeof target.closest === "function" &&
      target.closest(".cell")?.getAttribute("data-cell-id") === cell.id &&
      target.closest('button[title="Delete"]')
    ) {
      return;
    }
    const idx = state.cells.findIndex((c) => c.id === cell.id);
    if (idx >= 0) state.cells[idx].source = ta.value;
    try {
      await updateCell(cell.id, ta.value);
    } catch (e) {
      toast(e.message || String(e), "error");
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

const deletingCells = new Set();

/** Whether a delete is in flight for ``cellId`` (its blur save must not run). */
export function isCellDeleted(cellId) {
  return deletingCells.has(cellId);
}

/** Delete a cell and apply the returned snapshot immediately. */
export async function removeCell(cellId) {
  deletingCells.add(cellId);
  try {
    const snap = await deleteCellRequest(cellId);
    const patch = {};
    if (Array.isArray(snap.cells)) patch.cells = normalizeCells(snap.cells);
    if (Array.isArray(snap.jobs)) patch.jobs = snap.jobs;
    setState(patch);
  } catch (error) {
    deletingCells.delete(cellId);
    throw error;
  }
}

export async function addCell(kind) {
  if (!state.active_workspace) {
    toast("No workspace open", "warning");
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
      // If the reader already chose Follow, pin the new cell at the bottom.
      // `hold` is a no-op when following is off, so their position is kept.
      hold();
    }
  } catch (e) {
    toast(e.message || String(e), "error");
  }
}
