/* Full-screen trace review: every step plus request-level token usage. */

import { el, formatCost, formatTokens, openDialog } from "../dom.js";
import { renderTrace } from "./trace.js";

function stat(label, value) {
  const box = el("div", { class: "trace-stat" });
  box.append(
    el("span", { class: "trace-stat-label", text: label }),
    el("span", { class: "trace-stat-value", text: String(value) })
  );
  return box;
}

function requestRows(cell) {
  const rows = [];
  for (const step of cell.trace || []) {
    if (step.type === "usage" && Array.isArray(step.requests)) {
      rows.push(...step.requests);
    }
  }
  return rows;
}

function requestTable(rows) {
  const table = el("table", { class: "trace-usage-table" });
  const head = el("thead");
  const headRow = el("tr");
  for (const label of [
    "Request",
    "Model",
    "Input",
    "Output",
    "Cached",
    "Cost",
    "Tools",
  ]) {
    headRow.append(el("th", { text: label }));
  }
  head.append(headRow);
  const body = el("tbody");
  rows.forEach((row, index) => {
    const tr = el("tr");
    tr.append(
      el("td", { text: String(index + 1) }),
      el("td", { text: row.model || "—" }),
      el("td", { text: formatTokens(row.input_tokens) }),
      el("td", { text: formatTokens(row.output_tokens) }),
      el("td", { text: formatTokens(row.cache_read_tokens) }),
      el("td", { text: formatCost(row.cost) || "—" }),
      el("td", { text: (row.tools || []).join(", ") || "—" })
    );
    body.append(tr);
  });
  table.append(head, body);
  return table;
}

function stepsSection(cell) {
  const section = el("section", { class: "trace-dialog-section" });
  section.append(el("h4", { class: "trace-modal-heading", text: "Steps" }));
  const nodes = renderTrace({ ...cell, status: "idle" }, []);
  if (!nodes.length) {
    section.append(el("p", { class: "trace-note", text: "No trace steps recorded." }));
    return section;
  }
  const trace = el("div", { class: "trace trace-modal-trace" });
  trace.append(...nodes);
  trace.querySelectorAll("details").forEach((details) => {
    details.open = true;
  });
  trace.querySelectorAll(".trace-more").forEach((button) => {
    button.click();
  });
  section.append(trace);
  return section;
}

export function openTraceModal(cell) {
  openDialog((dialog) => {
    dialog.classList.add("trace-dialog");
    const usageStep = [...(cell.trace || [])]
      .reverse()
      .find((step) => step.type === "usage");
    const usage = cell.usage || (usageStep && usageStep.usage) || {};
    const rows = requestRows(cell);
    const stepCount = (cell.trace || []).filter(
      (step) => step.type !== "text_delta"
    ).length;

    const heading = el("div", { class: "dialog-heading" });
    heading.append(
      el("h3", { text: "Prompt run details" }),
      el("p", {
        class: "dialog-subtitle",
        text:
          stepCount + " steps · " + rows.length + " recorded model request(s)",
      })
    );
    dialog.append(heading);

    const body = el("div", { class: "trace-dialog-body" });

    const totals = el("section", { class: "trace-dialog-section" });
    totals.append(
      el("h4", { class: "trace-modal-heading", text: "Run totals" })
    );
    const input = Number(usage.input_tokens) || 0;
    const output = Number(usage.output_tokens) || 0;
    const cached = Number(usage.cache_read_tokens) || 0;
    const cacheRate = input ? Math.round((cached / input) * 100) : 0;
    const stats = el("div", { class: "trace-stats" });
    stats.append(
      stat("Requests", usage.requests ?? "—"),
      stat("Input", formatTokens(input)),
      stat("Output", formatTokens(output)),
      stat("Total", formatTokens(usage.total_tokens || input + output)),
      stat("Cached", input ? cacheRate + "%" : "—"),
      stat("Cost", formatCost(usage.cost) || "—"),
      stat("Tool calls", usage.tool_calls ?? "—")
    );
    totals.append(stats);
    body.append(totals);

    const requests = el("section", { class: "trace-dialog-section" });
    requests.append(
      el("h4", {
        class: "trace-modal-heading",
        text: "Token usage by model request",
      })
    );
    if (rows.length) requests.append(requestTable(rows));
    else
      requests.append(
        el("p", {
          class: "trace-note",
          text: "This run predates per-request usage recording; totals above are from the run aggregate.",
        })
      );
    body.append(requests);
    body.append(stepsSection(cell));
    dialog.append(body);
  });
}
