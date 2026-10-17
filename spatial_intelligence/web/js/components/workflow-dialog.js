/* New-workspace wizard: pick a type, fill its params, review, create.

Steps come from the workflow's params — Type, then one step per group (Model,
Area of interest, Details), then Review. The chip indicator stays hidden on
Type and from there names the real steps. Inline errors block Next; Back keeps
everything typed. Opened passthrough over a transparent overlay so the map
stays visible and usable while an `aoi` field is resolved from the view or a
drawn shape.
*/

"use strict";

import {
  checkImportUrl,
  createWorkspace,
  getState,
  importFilesWithProgress,
  importUrl as postImportUrl,
  listWorkflows,
} from "../api.js";
import { closeDialog, el, formatBytes, openDialog, toast } from "../dom.js";
// `state` is the wizard's own draft inside this module, so the store comes in
// under a name that cannot be shadowed by it.
import { applySnapshot, state as appState } from "../store.js";
import { icon } from "../icons.js";
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

      const header = el("div", { class: "wizard-header" });
      const mark = el("span", { class: "wizard-mark" });
      mark.append(icon("sparkles", { size: 22 }));
      const headerCopy = el("div", { class: "wizard-header-copy" });
      headerCopy.append(
        el("h3", { text: "New workspace" }),
        el("p", {
          class: "wizard-subtitle",
          text: "Choose a workspace type, set its inputs, and review before creating it.",
        })
      );
      header.append(mark, headerCopy);
      dialog.append(header);

      const body = el("div", { class: "wizard-body" });
      const indicator = el("ol", { class: "wizard-steps" });
      indicator.hidden = true;

      const typeStep = el("section", { class: "wizard-step" });
      const reviewStep = el("section", { class: "wizard-step" });
      reviewStep.hidden = true;
      const nodes = { type: typeStep, review: reviewStep };
      const readers = {};
      const errors = {};
      body.append(indicator, typeStep, reviewStep);
      dialog.append(body);

      // One footer for every step: the chrome (title, progress, actions) stays
      // put while the body scrolls, and the primary button means whatever the
      // step it is on needs — Next, or Create on the summary.
      const cancelBtn = el("button", {
        class: "secondary",
        type: "button",
        text: "Cancel",
        onclick: closeDialog,
      });
      const stepStatus = el("span", { class: "wizard-step-status" });
      const backBtn = el("button", { class: "secondary", type: "button" });
      backBtn.append(icon("chevron-left", { size: 14 }), el("span", { text: "Back" }));
      backBtn.hidden = true;
      const primaryBtn = el("button", { class: "primary", type: "button" });
      primaryBtn.disabled = true;
      const setPrimary = (label, iconName, spinning = false) => {
        primaryBtn.replaceChildren(
          icon(iconName, { size: 14, class: spinning ? "spin" : "" }),
          el("span", { text: label })
        );
      };
      setPrimary("Next", "arrow-right");
      const footer = el("div", { class: "wizard-footer" });
      const footerLeft = el("div", { class: "wizard-footer-group" });
      const footerRight = el("div", { class: "wizard-footer-group" });
      footerLeft.append(cancelBtn, stepStatus);
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
      let attachmentsCard = null;
      let attachmentsList = null;
      let creating = false;
      let retryMode = false;
      let workspaceCreated = false;
      const attachments = [];

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

      /** The full step list for a workflow, used before its sections exist. */
      function orderFor(workflow) {
        return [
          { key: "type", label: "Type" },
          ...planSteps(workflow).map(({ key, label }) => ({ key, label })),
          { key: "review", label: "Review" },
        ];
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
        setPrimary(key === "review" ? "Create" : "Next", key === "review" ? "check-circle" : "arrow-right");
        // The area-of-interest step reads the live map, so the backdrop lifts to
        // a hint of tint there and stays dimmed everywhere else.
        const overlay = document.getElementById("overlay");
        if (overlay) overlay.classList.toggle("map-live", key === "aoi");
        const index = order.findIndex((step) => step.key === key);
        indicator.hidden = false;
        chips.forEach((chip, i) => {
          chip.classList.toggle("active", i === index);
          chip.classList.toggle("done", i < index);
        });
        stepStatus.textContent =
          index >= 0 ? "Step " + (index + 1) + " of " + order.length : "";
        const scroller = dialog.querySelector(".wizard-body");
        if (scroller) scroller.scrollTop = 0;
      }

      function disposeParamSteps() {
        for (const key of Object.keys(nodes)) {
          if (key === "type" || key === "review") continue;
          nodes[key].remove();
          delete nodes[key];
          delete readers[key];
        }
      }

      // -- optional workspace files --------------------------------------
      function attachmentsSection() {
        if (attachmentsCard) return attachmentsCard;
        attachmentsCard = el("div", { class: "wizard-field wizard-attachments" });
        const actions = el("div", { class: "attachment-actions" });
        const filesInput = el("input", {
          type: "file",
          multiple: "multiple",
          style: "display:none",
        });
        const folderInput = el("input", {
          type: "file",
          webkitdirectory: "",
          style: "display:none",
        });
        filesInput.addEventListener("change", () => {
          addLocalFiles(filesInput.files, false);
          filesInput.value = "";
        });
        folderInput.addEventListener("change", () => {
          addLocalFiles(folderInput.files, true);
          folderInput.value = "";
        });
        actions.append(
          filesInput,
          folderInput,
          attachmentButton("upload", "Add files", () => filesInput.click()),
          attachmentButton("folder-open", "Add folder", () => folderInput.click())
        );

        const urlRow = el("div", { class: "attachment-url-row" });
        const urlInput = el("input", {
          type: "url",
          class: "attachment-url-input",
          placeholder: "https://example.com/file.tif",
          "aria-label": "URL to download into the workspace",
        });
        const urlButton = el("button", {
          type: "button",
          class: "secondary",
          text: "Add link",
        });
        const urlStatus = el("span", { class: "attachment-url-status" });
        const addUrl = () => addAttachmentUrl(urlInput, urlStatus, urlButton);
        urlButton.addEventListener("click", addUrl);
        urlInput.addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            addUrl();
          }
        });
        urlRow.append(urlInput, urlButton, urlStatus);

        attachmentsList = el("ul", { class: "attachment-list" });
        attachmentsCard.append(
          el("div", { class: "wizard-section-title", text: "Workspace files" }),
          el("span", {
            class: "wizard-hint",
            text:
              "Optional. Copy files or a folder, or download a link, into data/. A short note tells the agent what each input is.",
          }),
          actions,
          urlRow,
          el("p", { class: "attachment-empty", text: "No files attached yet." }),
          attachmentsList
        );
        renderAttachments();
        return attachmentsCard;
      }

      function attachmentButton(name, label, onclick) {
        const button = el("button", { type: "button", class: "secondary", onclick });
        button.append(icon(name, { size: 14 }), el("span", { text: label }));
        return button;
      }

      function addLocalFiles(fileList, isFolder) {
        if (creating) return;
        const files = Array.from(fileList || []);
        if (!files.length) return;
        if (isFolder) {
          const first = files[0].webkitRelativePath || files[0].name;
          const folder = first.split("/")[0] || "folder";
          const path = "data/" + folder + "/";
          if (!attachments.some((item) => item.path === path)) {
            attachments.push({
              kind: "folder",
              files,
              label: folder,
              path,
              description: "",
              state: "ready",
              meta: files.length + (files.length === 1 ? " file" : " files"),
            });
          }
        } else {
          for (const file of files) {
            const relative = file.webkitRelativePath || file.name;
            const path = "data/" + relative;
            if (attachments.some((item) => item.path === path)) continue;
            attachments.push({
              kind: "file",
              file,
              label: file.name,
              path,
              description: "",
              state: "ready",
              meta: formatBytes(file.size),
            });
          }
        }
        renderAttachments();
      }

      async function addAttachmentUrl(input, status, button) {
        if (creating) return;
        const url = input.value.trim();
        if (!url) {
          status.className = "attachment-url-status error";
          status.textContent = "Enter a link first.";
          return;
        }
        button.disabled = true;
        status.className = "attachment-url-status";
        status.textContent = "Checking reachability…";
        try {
          const check = await checkImportUrl(url);
          if (!check.ok) {
            status.className = "attachment-url-status error";
            status.textContent = "Unreachable: " + (check.error || "no response");
            return;
          }
          const filename = check.filename || "download";
          const path = "data/" + filename;
          if (!attachments.some((item) => item.path === path)) {
            attachments.push({
              kind: "url",
              url,
              label: filename,
              path,
              description: "",
              state: "ready",
              meta: check.content_length ? formatBytes(check.content_length) : "link",
            });
            status.className = "attachment-url-status success";
            status.textContent = "Added " + filename;
            input.value = "";
            renderAttachments();
          } else {
            status.className = "attachment-url-status";
            status.textContent = filename + " is already attached.";
          }
        } catch (error) {
          status.className = "attachment-url-status error";
          status.textContent = error.message || String(error);
        } finally {
          button.disabled = false;
        }
      }

      function attachmentKindIcon(item) {
        if (item.kind === "folder") return "folder";
        if (item.kind === "url") return "download";
        return "file";
      }

      function attachmentStatus(item) {
        if (item.state === "working") {
          if (item.kind === "url") return "Downloading…";
          return "Uploading " + Math.round((item.progress || 0) * 100) + "%";
        }
        if (item.state === "done") return item.kind === "url" ? "Downloaded" : "Copied";
        if (item.state === "error") return "Failed";
        return item.meta || "Ready";
      }

      function renderAttachments() {
        if (!attachmentsCard) return;
        const empty = attachmentsCard.querySelector(".attachment-empty");
        if (empty) empty.hidden = attachments.length > 0;
        attachmentsList.replaceChildren();
        for (const item of attachments) {
          const row = el("li", { class: "attachment-item state-" + item.state });
          const mark = el("span", { class: "attachment-icon" });
          mark.append(icon(attachmentKindIcon(item), { size: 16 }));
          const main = el("div", { class: "attachment-main" });
          const title = el("div", { class: "attachment-title" });
          title.append(
            el("strong", { class: "attachment-name", text: item.label, title: item.path }),
            el("span", { class: "attachment-meta", text: item.path })
          );
          const note = el("input", {
            type: "text",
            class: "attachment-note",
            placeholder: "What is this file about? (optional)",
            value: item.description || "",
          });
          note.addEventListener("input", () => {
            item.description = note.value;
          });
          const progress = el("div", { class: "attachment-progress" });
          const fill = el("span", { class: "attachment-progress-fill" });
          progress.append(fill);
          main.append(title, note, progress);
          const status = el("span", { class: "attachment-status", text: attachmentStatus(item) });
          const remove = el("button", {
            type: "button",
            class: "attachment-remove",
            title: "Remove",
            "aria-label": "Remove " + item.label,
            onclick: () => {
              if (creating) return;
              const index = attachments.indexOf(item);
              if (index >= 0) attachments.splice(index, 1);
              renderAttachments();
            },
          });
          remove.append(icon("x", { size: 13 }));
          row.append(mark, main, status, remove);
          item.ui = { row, status, fill, progress, note, remove };
          updateAttachmentRow(item);
          attachmentsList.append(row);
        }
      }

      function updateAttachmentRow(item) {
        if (!item.ui) return;
        const { row, status, fill, progress, note, remove } = item.ui;
        row.className = "attachment-item state-" + item.state;
        status.textContent = attachmentStatus(item);
        status.title = item.state === "error" ? item.error || "Failed" : "";
        status.className = "attachment-status" + (item.state === "error" ? " error" : "");
        const active = item.state === "working";
        progress.hidden = !active;
        if (active) fill.style.width = Math.round((item.progress || 0) * 100) + "%";
        note.disabled = creating || retryMode || workspaceCreated || item.state === "done";
        remove.disabled = creating || workspaceCreated;
      }

      function attachmentParams() {
        return attachments
          .filter((item) => item.path)
          .map((item) => ({
            path: item.path,
            description: (item.description || "").trim(),
          }));
      }

      async function transferAttachments() {
        for (const item of attachments) {
          if (item.state === "done") continue;
          item.state = "working";
          item.progress = 0;
          item.error = "";
          updateAttachmentRow(item);
          try {
            if (item.kind === "url") {
              await postImportUrl(item.url, item.label);
            } else {
              const files = item.kind === "folder" ? item.files : [item.file];
              await importFilesWithProgress(files, (fraction) => {
                item.progress = fraction;
                updateAttachmentRow(item);
              });
            }
            item.state = "done";
            item.progress = 1;
          } catch (error) {
            item.state = "error";
            item.error = error.message || String(error);
          }
          updateAttachmentRow(item);
        }
        return attachments.some((item) => item.state === "error");
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
          el("label", { class: "wizard-label", text: "Workspace name" }),
          el("span", {
            class: "wizard-hint",
            text: "This becomes the folder name. Letters, numbers, spaces, dots, dashes, and underscores.",
          }),
          nameInput,
        ]),
        el("div", { class: "wizard-field" }, [
          el("div", { class: "wizard-section-title", text: "Workspace type" }),
          el("span", {
            class: "wizard-hint",
            text:
              "A workflow adds a prompt cell already set up for the task. You can edit and run it like any other cell.",
          }),
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
          card.append(input, workflowArt(workflow.id, { size: 64 }), copy);
          input.addEventListener("change", () => {
            if (!input.checked) return;
            state.workflow = workflow;
            state.params = {};
            builtFor = null;
            for (const node of cards.querySelectorAll(".wizard-card")) {
              node.classList.toggle("selected", node === card);
            }
            // Show the shape of the flow the choice creates before Next.
            order = orderFor(workflow);
            renderIndicator();
            show("type");
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
        if (creating) return;
        if (current === "review") {
          if (retryMode) {
            retryFailed(errors.review);
            return;
          }
          if (errors.review) errors.review.textContent = "";
          runSetup(errors.review);
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
        order = orderFor(state.workflow);
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
          if (step.key === "details") node.append(attachmentsSection());
          node.append(error);
          nodes[step.key] = node;
          readers[step.key] = stepReaders;
          errors[step.key] = error;
          // Keep Review last in the document: the sections are hidden and shown
          // rather than rebuilt, so their order is only for the reading order.
          body.insertBefore(node, reviewStep);
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
        const attached = attachmentParams();
        if (attached.length) {
          addRow(
            list,
            "Files",
            attached
              .map((item) => item.path + (item.description ? " — " + item.description : ""))
              .join("; ")
          );
        }
        reviewStep.append(heading(state.workflow), list, error);
        show("review");
      }

      function lockSetup(locked) {
        cancelBtn.disabled = locked;
        backBtn.disabled = locked;
        primaryBtn.disabled = locked;
        renderAttachments();
      }

      async function runSetup(error) {
        creating = true;
        retryMode = false;
        error.textContent = "";
        lockSetup(true);
        setPrimary("Creating workspace…", "loader-circle", true);
        stepStatus.textContent = "Creating workspace";
        try {
          const params = { ...state.params, attachments: attachmentParams() };
          const snapshot = await createWorkspace(state.name, state.workflow.id, params);
          workspaceCreated = true;
          applySnapshot(snapshot);
          const failed = await transferAttachments();
          applySnapshot(await getState());
          creating = false;
          if (!failed) {
            closeDialog();
            toast("Workspace " + state.name + " created", "success");
            return;
          }
          retryMode = true;
          error.textContent =
            "The workspace was created, but some files could not be transferred. " +
            "Retry them here, or close this dialog and add them from the Data tab.";
          lockSetup(false);
          backBtn.disabled = true;
          setPrimary("Retry failed files", "refresh-cw");
          stepStatus.textContent = "Workspace created";
          renderAttachments();
        } catch (err) {
          creating = false;
          lockSetup(false);
          setPrimary("Create", "check-circle");
          const message = err.message || String(err);
          error.textContent = message;
          toast(message, "error");
        }
      }

      async function retryFailed(error) {
        if (creating) return;
        creating = true;
        error.textContent = "";
        lockSetup(true);
        setPrimary("Retrying files…", "loader-circle", true);
        try {
          const failed = await transferAttachments();
          applySnapshot(await getState());
          creating = false;
          if (!failed) {
            closeDialog();
            toast("Workspace " + state.name + " created", "success");
            return;
          }
          retryMode = true;
          lockSetup(false);
          backBtn.disabled = true;
          setPrimary("Retry failed files", "refresh-cw");
          renderAttachments();
        } catch (err) {
          creating = false;
          lockSetup(false);
          backBtn.disabled = true;
          setPrimary("Retry failed files", "refresh-cw");
          error.textContent = err.message || String(err);
        }
      }

      renderIndicator();
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
