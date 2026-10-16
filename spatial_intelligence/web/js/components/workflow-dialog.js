/* New-workspace wizard: pick a type, fill its params, review, create.

Steps come from the workflow's params — Type, then one step per group (Model,
Area of interest, Details), then Review. The chip indicator stays hidden on
Type and from there names the real steps. Inline errors block Next; Back keeps
everything typed. Opened passthrough over a transparent overlay so the map
stays visible and usable while an `aoi` field is resolved from the view or a
drawn shape.
*/

"use strict";

import { createWorkspace, listWorkflows } from "../api.js";
import { closeDialog, el, openDialog, toast } from "../dom.js";
// `state` is the wizard's own draft inside this module, so the store comes in
// under a name that cannot be shadowed by it.
import { applySnapshot, state as appState } from "../store.js";
import { buildField } from "./fields.js";
import { workflowArt } from "./workflow-art.js";

const BLANK = {
  id: "",
  label: "Blank",
  summary: "An empty notebook.",
  params: [],
};

export function openWorkflowDialog() {
  openDialog(
    (dialog) => {
      dialog.classList.add("wizard-dialog");
      dialog.append(el("h3", { text: "New workspace" }));

      const indicator = el("ol", { class: "wizard-steps" });
      indicator.hidden = true;
      dialog.append(indicator);

      const typeStep = el("section", { class: "wizard-step" });
      const reviewStep = el("section", { class: "wizard-step" });
      reviewStep.hidden = true;
      const nodes = { type: typeStep, review: reviewStep };
      const readers = {};
      const errors = {};
      dialog.append(typeStep, reviewStep);

      // One footer for every step: the chrome (title, progress, actions) stays
      // put while the body scrolls, and the primary button means whatever the
      // step it is on needs — Next, or Create on the summary.
      const cancelBtn = el("button", { text: "Cancel", onclick: closeDialog });
      const backBtn = el("button", { text: "Back" });
      backBtn.hidden = true;
      const primaryBtn = el("button", { class: "primary", text: "Next" });
      primaryBtn.disabled = true;
      const footer = el("div", { class: "wizard-footer" });
      const footerLeft = el("div", { class: "wizard-footer-group" });
      const footerRight = el("div", { class: "wizard-footer-group" });
      footerLeft.append(cancelBtn);
      footerRight.append(backBtn, primaryBtn);
      footer.append(footerLeft, footerRight);
      dialog.append(footer);

      const state = { name: "", workflow: BLANK, params: {} };
      let current = "type";
      let order = [
        { key: "type", label: "Type" },
        { key: "review", label: "Review" },
      ];
      let chips = [];
      let builtFor = null;

      // Steps derived from the workflow's params, in a fixed group order.
      function planSteps(workflow) {
        const fields = workflow.params || [];
        return [
          ["model", "Model", (field) => field.type === "model"],
          ["aoi", "Area of interest", (field) => field.type === "aoi"],
          ["details", "Details", (field) => field.type !== "model" && field.type !== "aoi"],
        ]
          .map(([key, label, match]) => ({ key, label, fields: fields.filter(match) }))
          .filter((step) => step.fields.length);
      }

      function renderIndicator() {
        indicator.replaceChildren();
        chips = order.map((step, index) => {
          const chip = el("li", { class: "wizard-step-chip" });
          chip.append(
            el("span", { class: "wizard-step-no", text: String(index + 1) }),
            el("span", { text: step.label })
          );
          indicator.append(chip);
          return chip;
        });
      }

      function show(key) {
        current = key;
        for (const [name, node] of Object.entries(nodes)) {
          const active = name === key;
          // The class is what lays the step out; `hidden` keeps it out of the
          // accessibility tree and of find-in-page.
          node.classList.toggle("active", active);
          node.hidden = !active;
        }
        backBtn.hidden = key === "type";
        primaryBtn.textContent = key === "review" ? "Create" : "Next";
        // The area-of-interest step reads the live map, so the backdrop lifts to
        // a hint of tint there and stays dimmed everywhere else.
        const overlay = document.getElementById("overlay");
        if (overlay) overlay.classList.toggle("map-live", key === "aoi");
        const index = order.findIndex((step) => step.key === key);
        indicator.hidden = key === "type";
        chips.forEach((chip, i) => {
          chip.classList.toggle("active", i === index);
          chip.classList.toggle("done", i < index);
        });
      }

      function disposeParamSteps() {
        for (const key of Object.keys(nodes)) {
          if (key === "type" || key === "review") continue;
          nodes[key].remove();
          delete nodes[key];
          delete readers[key];
        }
      }

      function heading(workflow) {
        const row = el("div", { class: "wizard-heading-row" });
        const copy = el("div", { class: "wizard-heading-copy" });
        copy.append(el("strong", { text: workflow.label }));
        if (workflow.summary) copy.append(el("small", { class: "wizard-hint", text: workflow.summary }));
        row.append(workflowArt(workflow.id, { size: 44 }), copy);
        return row;
      }

      // Name rules mirror the server's: a name is a directory, so it may not be
      // empty, may not hold a path separator, and may not collide with one that
      // is already there ("New" must never open an old workspace).
      function nameProblem(name) {
        if (!name) return "Please enter a workspace name.";
        if (name.length > 64) return "A workspace name can be at most 64 characters.";
        if (name.startsWith(".")) return "A workspace name cannot start with a dot.";
        const reserved = Array.from(new Set(name.match(/[\/\\:*?"<>|]/g) || []));
        if (reserved.length) {
          return "A workspace name cannot contain " + reserved.join(" ") + ".";
        }
        if ((appState.workspaces || []).includes(name)) {
          return (
            "A workspace named “" + name + "” already exists — open it, or pick another name."
          );
        }
        return "";
      }
      // -- step 1: name + type -------------------------------------------
      const typeError = el("div", { class: "interaction-error" });
      errors.type = typeError;
      const nameInput = el("input", {
        type: "text",
        placeholder: "workspace name",
        // A name the browser remembers is how an old workspace gets recreated
        // by accident: the field must start empty every time.
        autocomplete: "off",
        spellcheck: "false",
      });
      nameInput.addEventListener("input", () => {
        typeError.textContent = nameProblem(nameInput.value.trim());
      });
      const cards = el("div", { class: "wizard-cards" });
      cards.append(el("p", { class: "wizard-hint", text: "Loading workflows…" }));

      typeStep.append(
        el("div", { class: "wizard-field" }, [
          el("label", { class: "wizard-label", text: "Name" }),
          nameInput,
        ]),
        el("div", { class: "wizard-field" }, [
          el("div", { class: "wizard-section-title", text: "Workspace type" }),
          cards,
        ]),
        typeError
      );

      let workflows = [];

      function renderCards() {
        cards.replaceChildren();
        for (const workflow of workflows) {
          const input = el("input", {
            type: "radio",
            name: "workflow-type",
            value: workflow.id,
          });
          input.checked = workflow.id === state.workflow.id;
          const copy = el("span", { class: "wizard-card-copy" });
          copy.append(el("strong", { text: workflow.label }));
          if (workflow.summary) copy.append(el("small", { text: workflow.summary }));
          const card = el("label", {
            class: "wizard-card" + (input.checked ? " selected" : ""),
          });
          card.append(input, workflowArt(workflow.id, { size: 72 }), copy);
          input.addEventListener("change", () => {
            if (!input.checked) return;
            state.workflow = workflow;
            state.params = {};
            builtFor = null;
            for (const node of cards.querySelectorAll(".wizard-card")) {
              node.classList.toggle("selected", node === card);
            }
          });
          cards.append(card);
        }
      }

      listWorkflows()
        .then((payload) => {
          const list = (payload && payload.workflows) || [];
          // The server always lists its own blank workflow first; only fall back
          // to a local one when the request itself failed.
          workflows = list.length ? list : [BLANK];
          state.workflow = workflows.find((workflow) => workflow.id === "blank") || workflows[0];
          renderCards();
          primaryBtn.disabled = false;
        })
        .catch((error) => {
          workflows = [BLANK];
          state.workflow = BLANK;
          renderCards();
          primaryBtn.disabled = false;
          typeError.textContent = "Could not load workflows: " + (error.message || String(error));
        });

      /** Move to the next step, or render the summary when there is none. */
      function advance(fromKey) {
        const index = order.findIndex((step) => step.key === fromKey);
        const nextKey = order[index + 1] ? order[index + 1].key : "";
        if (!nextKey || nextKey === "review") {
          showReview();
          return;
        }
        show(nextKey);
      }

      /** Validate the step we are on; true when it may be left. */
      function validate(key) {
        const error = errors[key];
        if (error) error.textContent = "";
        if (key === "type") {
          const name = nameInput.value.trim();
          const problem = nameProblem(name);
          if (problem) {
            if (error) error.textContent = problem;
            return null;
          }
          state.name = name;
          return {};
        }
        const params = {};
        for (const { field, read } of readers[key] || []) {
          const value = read();
          params[field.id] = value;
          if (field.required !== false && isEmpty(value)) {
            if (error) error.textContent = "Please complete " + field.label + ".";
            return null;
          }
        }
        return params;
      }

      backBtn.addEventListener("click", () => {
        const index = order.findIndex((step) => step.key === current);
        if (index > 0) show(order[index - 1].key);
      });

      primaryBtn.addEventListener("click", () => {
        if (current === "review") {
          if (errors.review) errors.review.textContent = "";
          submit(errors.review);
          return;
        }
        const params = validate(current);
        if (params === null) return;
        if (current === "type") ensureParamSteps();
        else Object.assign(state.params, params);
        advance(current);
      });

      // -- param steps: one section per group, each validating its own fields
      function buildParamSteps() {
        disposeParamSteps();
        order = [
          { key: "type", label: "Type" },
          ...planSteps(state.workflow),
          { key: "review", label: "Review" },
        ];
        for (const step of planSteps(state.workflow)) {
          const node = el("section", { class: "wizard-step" });
          node.hidden = true;
          const error = el("div", { class: "interaction-error" });
          const stepReaders = [];
          node.append(heading(state.workflow));
          for (const field of step.fields) {
            const { group, read } = buildField(field, { requestId: "new-workspace" });
            stepReaders.push({ field, read });
            node.append(group);
          }
          node.append(error);
          nodes[step.key] = node;
          readers[step.key] = stepReaders;
          errors[step.key] = error;
          // Keep Review last in the document: the sections are hidden and shown
          // rather than rebuilt, so their order is only for the reading order.
          dialog.insertBefore(node, reviewStep);
        }
        renderIndicator();
      }

      /** Make sure the DOM holds one section per step of the current plan. */
      function ensureParamSteps() {
        const planned = planSteps(state.workflow).map((step) => step.key);
        const built = Object.keys(nodes).filter((key) => key !== "type" && key !== "review");
        if (builtFor === state.workflow.id && built.length === planned.length) return;
        state.params = {};
        buildParamSteps();
        builtFor = state.workflow.id;
      }

      // -- review ---------------------------------------------------------
      function showReview() {
        reviewStep.replaceChildren();
        const error = el("div", { class: "interaction-error" });
        errors.review = error;
        const list = el("dl", { class: "wizard-review" });
        addRow(list, "Name", state.name);
        addRow(list, "Type", state.workflow.label);
        for (const field of state.workflow.params || []) {
          addRow(list, field.label, formatValue(state.params[field.id]));
        }
        reviewStep.append(heading(state.workflow), list, error);
        show("review");
      }

      async function submit(error) {
        try {
          const snapshot = await createWorkspace(state.name, state.workflow.id, state.params);
          closeDialog();
          applySnapshot(snapshot);
          toast("Workspace " + state.name + " created");
        } catch (err) {
          const message = err.message || String(err);
          error.textContent = message;
          toast(message);
        }
      }

      show("type");
      nameInput.focus();
    },
    { passthrough: true }
  );
}

function isEmpty(value) {
  return value == null || value === "" || (Array.isArray(value) && !value.length);
}

function addRow(list, label, value) {
  list.append(el("dt", { text: label }), el("dd", { text: value }));
}

function formatValue(value) {
  if (value == null || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "object") {
    if (value.ref) {
      const parts = [value.ref];
      if (value.labels && value.labels.length) parts.push("labels: " + value.labels.join(", "));
      if (value.target) parts.push('"' + value.target + '"');
      return parts.join(" · ");
    }
    if (value.bbox) {
      const bbox = value.bbox.map((n) => Number(n).toFixed(3)).join(", ");
      return (value.label ? value.label + " — " : "") + "bbox " + bbox;
    }
    return JSON.stringify(value);
  }
  return String(value);
}
