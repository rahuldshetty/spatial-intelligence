/* Open-workspace picker: the recent workspace list, shared by the File menu
   and the map's empty state. */

"use strict";

import { openWorkspace } from "../api.js";
import { closeDialog, el, openDialog, toast } from "../dom.js";
import { icon } from "../icons.js";
import { applySnapshot, state } from "../store.js";
import { openWorkflowDialog } from "./workflow-dialog.js";

export function openWorkspaceDialog() {
  openDialog((dialog) => {
    const head = el("div", { class: "dialog-heading" });
    head.append(
      el("h3", { text: "Open workspace" }),
      el("p", {
        class: "dialog-subtitle",
        text: "Pick a workspace to continue where you left off.",
      })
    );
    dialog.append(head);

    if (!(state.workspaces || []).length) {
      const empty = el("div", { class: "dialog-empty" });
      empty.append(
        icon("folder-open", { size: 28 }),
        el("strong", { text: "No workspaces yet" }),
        el("span", { text: "Create one from File → New workspace." })
      );
      dialog.append(empty);
      return;
    }

    const list = el("ul", { id: "ws-list", class: "workspace-list" });
    for (const name of state.workspaces) {
      const item = el("li", {});
      const button = el("button", { type: "button", class: "workspace-row" });
      button.append(
        icon("folder", { size: 16 }),
        el("span", { class: "workspace-row-name", text: name }),
        icon("chevron-right", { size: 14, class: "workspace-row-arrow" })
      );
      button.addEventListener("click", async () => {
        if (button.disabled) return;
        // Binding a workspace builds its runtime and agent, which can take a
        // moment: keep the dialog up and say so instead of closing on nothing.
        const arrow = button.querySelector(".workspace-row-arrow");
        button.disabled = true;
        button.setAttribute("aria-busy", "true");
        if (arrow) arrow.replaceWith(icon("loader-circle", { size: 14, class: "spin" }));
        try {
          const snapshot = await openWorkspace(name);
          closeDialog();
          applySnapshot(snapshot);
        } catch (error) {
          button.disabled = false;
          button.removeAttribute("aria-busy");
          const busy = button.querySelector(".spin");
          if (busy) busy.replaceWith(icon("chevron-right", { size: 14, class: "workspace-row-arrow" }));
          toast(error.message || String(error), "error");
        }
      });
      item.append(button);
      list.append(item);
    }
    dialog.append(list);

    const actions = el("div", { class: "dialog-actions" });
    actions.append(
      el("button", {
        class: "primary",
        text: "New workspace",
        onclick: () => {
          closeDialog();
          openWorkflowDialog();
        },
      })
    );
    dialog.append(actions);
  });
}
