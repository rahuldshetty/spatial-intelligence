/* One cell: header actions, source, output, trace, and its job strip. */

import { el, expandableContent, toast, usageLabel, usageTitle } from "../dom.js";
import { moveCell, runCell, stopCell, updateCell } from "../api.js";
import { icon } from "../icons.js";
import { state } from "../store.js";
import { cellOutputText, renderMarkdown } from "./markdown.js";
import { renderTrace } from "./trace.js";
import { editMarkdown, isCellDeleted, removeCell } from "../pages/cells.js";

/** Fire a cell action, surfacing its failure the way the old app did. */
function act(action) {
  Promise.resolve()
    .then(action)
    .catch((error) => toast(error.message || String(error), "error"));
}

function cellBadge(kind, geoai) {
  if (kind === "tool") return "agent tool";
  if (geoai.kind === "response") return "agent response";
  if (geoai.kind === "interaction_response") return "user choice";
  if (kind === "markdown") return "md";
  if (kind === "prompt") return "prompt";
  return "py";
}

function cellStatusText(status) {
  if (status === "waiting_for_input") return "waiting";
  if (status === "running") return "running";
  if (status === "error") return "failed";
  if (status === "done") return "done";
  return "";
}

function headButton(name, title, onclick, className = "icon") {
  const button = el("button", {
    class: className,
    type: "button",
    title,
    "aria-label": title,
    onclick,
  });
  button.append(icon(name, { size: 14 }));
  return button;
}

function renderHeader(cell, kind, geoai, generated) {
  const header = el("div", { class: "cell-header" });
  if (kind !== "markdown" && kind !== "tool" && cell.status !== "waiting_for_input") {
    if (cell.status === "running" && kind === "prompt") {
      header.append(
        headButton("stop", "Stop", () => act(() => stopCell(cell.id)), "run-btn stop")
      );
    } else {
      header.append(headButton("play", "Run", () => act(() => runCell(cell.id)), "run-btn"));
    }
  } else {
    // Keep the header's leading gutter stable across cell kinds and states.
    header.append(el("span", { class: "run-slot" }));
  }
  header.append(
    el("span", {
      class: "counter",
      text: "In[" + (cell.execution_count != null ? cell.execution_count : " ") + "]",
    }),
    el("span", { class: "badge badge-" + kind, text: cellBadge(kind, geoai) })
  );
  if (kind === "tool") {
    header.append(
      icon("wrench", { size: 13, class: "tool-icon" }),
      el("span", { class: "agent-action-name", text: geoai.tool_name || "tool" }),
      el("span", {
        class: "agent-action-status status-" + (cell.status || "idle"),
        text: cellStatusText(cell.status),
      })
    );
  }
  header.append(el("span", { class: "spacer" }));

  const actions = el("div", { class: "cell-actions" });
  if (kind === "markdown") {
    if (!generated) {
      actions.append(
        headButton("pencil", "Edit", () => editMarkdown(cell)),
        headButton("trash-2", "Delete", () => act(() => removeCell(cell.id)))
      );
    }
  } else if (kind === "tool") {
    actions.append(headButton("trash-2", "Delete", () => act(() => removeCell(cell.id))));
  } else {
    actions.append(
      headButton("chevron-up", "Move up", () => act(() => moveCell(cell.id, -1))),
      headButton("chevron-down", "Move down", () => act(() => moveCell(cell.id, 1))),
      headButton("trash-2", "Delete", () => act(() => removeCell(cell.id)))
    );
  }
  header.append(actions);
  return header;
}

function renderBody(cell, kind, geoai, source, generated) {
  const body = el("div", { class: "cell-body" });
  if (kind === "markdown") {
    const markdown = el("div", { class: "markdown" });
    markdown.innerHTML = renderMarkdown(source);
    body.append(markdown);
  } else if (kind === "tool") {
    body.append(
      expandableContent("Input", geoai.args === undefined ? source : JSON.stringify(geoai.args, null, 2))
    );
  } else {
    const ta = el("textarea", {
      rows: Math.min(12, Math.max(2, source.split("\n").length)),
      spellcheck: "false",
    });
    ta.value = source;
    if (generated) {
      ta.readOnly = true;
    } else {
      ta.addEventListener("input", () => {
        const idx = state.cells.findIndex((c) => c.id === cell.id);
        if (idx >= 0) state.cells[idx].source = ta.value;
        autoGrow(ta);
      });
      ta.addEventListener("blur", (event) => {
        if (isCellDeleted(cell.id)) return;
        const target = event && event.relatedTarget;
        if (
          target &&
          typeof target.closest === "function" &&
          target.closest(".cell")?.getAttribute("data-cell-id") === cell.id &&
          target.closest('button[title="Delete"]')
        ) {
          return;
        }
        act(() => updateCell(cell.id, ta.value));
      });
    }
    body.append(ta);
    window.requestAnimationFrame(() => autoGrow(ta));
  }
  return body;
}

/** Grow a textarea to its content, capped so one cell cannot own the panel. */
function autoGrow(ta) {
  if (!ta || !ta.style) return;
  ta.style.height = "auto";
  const height = Number(ta.scrollHeight) || 0;
  if (height) ta.style.height = Math.min(height, 360) + "px";
}

function renderOutputRow(cell, kind) {
  const row = el("div", { class: "cell-out" });
  row.append(
    el("span", {
      class: "counter",
      text: "Out[" + (cell.execution_count != null ? cell.execution_count : " ") + "]",
    })
  );
  if (cell.status === "running") {
    row.append(el("span", { class: "running", text: "running…" }));
  } else if (cell.status === "waiting_for_input") {
    row.append(el("span", { class: "running waiting", text: "waiting for input" }));
  } else if (cell.status === "stopped") {
    row.append(el("span", { class: "running stopped", text: "stopped" }));
  }
  if (kind === "prompt" && cell.usage) {
    row.append(
      el("span", { class: "usage", text: usageLabel(cell.usage), title: usageTitle(cell.usage) })
    );
  }
  return row;
}

function renderOutputBlock(cell) {
  const block = el("div", { class: "cell-out-block" });
  const text = cellOutputText(cell);
  if (text) {
    if (cell.status === "error") {
      block.append(el("pre", { class: "error", text }));
    } else {
      const markdown = el("div", { class: "markdown" });
      markdown.innerHTML = renderMarkdown(text);
      block.append(markdown);
    }
  }
  return block;
}

function renderToolOutput(cell) {
  const text = cellOutputText(cell);
  if (!text) return null;
  const output = el("div", { class: "agent-action-output" });
  output.append(
    expandableContent(
      cell.status === "error" ? "Error" : "Output",
      text,
      cell.status === "error" ? "error" : ""
    )
  );
  return output;
}

/**
 * Build one cell.
 *
 * `jobs` is jobsFor(cell.id): progress jobs owned by this cell. They render in
 * the cell's job strip whatever the cell kind, so a job parented to a python or
 * tool cell is visible instead of being dropped.
 */
export function renderCell(cell, jobs = []) {
  const kind = cell.kind;
  const geoai = (cell.metadata && cell.metadata.geoai) || {};
  const source = cell.source == null ? "" : String(cell.source);
  const generated = Boolean(geoai.generated);
  const box = el("div", {
    class: "cell" + (generated ? " generated-cell" : ""),
    "data-cell-id": cell.id,
  });

  box.append(renderHeader(cell, kind, geoai, generated));
  box.append(renderBody(cell, kind, geoai, source, generated));

  if (kind !== "markdown" && kind !== "tool") {
    box.append(renderOutputRow(cell, kind));
  } else if (kind === "tool") {
    const output = renderToolOutput(cell);
    if (output) box.append(output);
  }

  const traceNodes = renderTrace(cell, jobs);
  // A prompt cell always mounts its trace container, even while it is empty:
  // live steps stream in before the first one is part of the cell snapshot, and
  // appending needs the mount point to already exist. Other kinds mount one
  // only when they have something to show (a job strip, or a replayed trace).
  if (kind === "prompt" || traceNodes.length) {
    const trace = el("div", { class: "trace" });
    trace.append(...traceNodes);
    box.append(trace);
  }

  if (kind !== "markdown" && kind !== "tool") {
    box.append(renderOutputBlock(cell));
  }

  return box;
}
