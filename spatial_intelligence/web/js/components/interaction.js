/* Deferred tool input: the form a request_user_input tool call waits on. */

import { el } from "../dom.js";
import { cancelInteraction, getState, respondInteraction } from "../api.js";
import { normalizeCells, setState, state } from "../store.js";
import { renderCellsOnly } from "../pages/cells.js";
import { buildField } from "./fields.js";

export function renderInteraction(cell, interaction = cell.interaction) {
  const request = interaction;
  const submitted = request.submitted === true;
  const answers = request.answers || {};
  const form = el(submitted ? "div" : "form", {
    class: "interaction-form" + (submitted ? " submitted" : ""),
  });
  form.append(
    el("h4", { text: request.title || "Input required" }),
    el("p", { class: "interaction-prompt", text: request.prompt || "" })
  );
  const controls = new Map();
  for (const field of request.fields || []) {
    const { group, read } = buildField(field, {
      disabled: submitted,
      answers,
      requestId: request.id,
    });
    controls.set(field.id, { field, read });
    form.append(group);
  }
  if (submitted) {
    form.append(el("div", { class: "interaction-submitted", text: "Selection submitted" }));
    return form;
  }
  const error = el("div", { class: "interaction-error" });
  const submit = el("button", {
    type: "submit",
    class: "primary",
    text: request.submit_label || "Continue",
  });
  const actions = el("div", { class: "interaction-actions" });
  if (request.allow_cancel !== false) {
    actions.append(
      el("button", {
        type: "button",
        text: "Cancel",
        onclick: async () => {
          try {
            await cancelInteraction(cell.id, request.id);
            const current = state.cells.find((item) => item.id === cell.id) || cell;
            current.status = "stopped";
            current.interaction = null;
            renderCellsOnly();
          } catch (e) {
            error.textContent = e.message || String(e);
          }
        },
      })
    );
  }
  actions.append(submit);
  form.append(error, actions);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = {};
    for (const [id, control] of controls) {
      const { field, read } = control;
      values[id] = read();
      const empty =
        values[id] == null ||
        values[id] === "" ||
        (Array.isArray(values[id]) && !values[id].length);
      if (field.required !== false && empty) {
        error.textContent = "Please complete " + field.label + ".";
        return;
      }
    }
    submit.disabled = true;
    try {
      await respondInteraction(cell.id, request.id, values);
      const completed = { ...request, answers: values, submitted: true };
      const history = cell.interaction_history || [];
      if (!history.some((item) => item.id === completed.id)) {
        cell.interaction_history = [...history, completed];
      }
      cell.status = "running";
      cell.interaction = null;
      renderCellsOnly();
    } catch (e) {
      submit.disabled = false;
      error.textContent = e.message || String(e);
    }
  });
  return form;
}

export function revealInteraction(cellId) {
  window.requestAnimationFrame(() => {
    const form = document.querySelector(
      '.cell[data-cell-id="' + cellId + '"] .interaction-form'
    );
    if (form) form.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

export async function reconcilePendingInteraction(cellId, attempt = 0) {
  try {
    const snap = await getState();
    const serverCell = (snap.cells || []).find((cell) => cell.id === cellId);
    if (serverCell) {
      setState({ cells: normalizeCells(snap.cells || []) });
      renderCellsOnly();
      if (serverCell.status === "waiting_for_input" && serverCell.interaction) {
        revealInteraction(cellId);
        return;
      }
      if (serverCell.status !== "running") return;
    }
  } catch (_) {
    // SSE remains the primary update path; retry briefly while the deferred
    // tool transitions from a streamed call into a waiting prompt.
  }
  if (attempt < 20) {
    window.setTimeout(() => reconcilePendingInteraction(cellId, attempt + 1), 250);
  }
}
