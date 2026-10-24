/* Tool Configuration: what the agent may call and what run_python may import.

The dialog is a draft editor. Nothing is applied while a checkbox changes and
nothing is sent for Cancel, so closing it always leaves the session on its
previous configuration. Save persists the selected scope: a workspace save
writes that workspace's `tools.json`, a global save writes the `tools.json` in
the workspaces root.
*/

"use strict";

import { getToolConfig, saveToolConfig } from "../api.js";
import { closeDialog, el, openDialog, toast } from "../dom.js";
import { icon } from "../icons.js";
import { setState, state } from "../store.js";

const TABS = [
  { id: "tools", label: "Tools" },
  { id: "python", label: "Python runtime" },
];

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function uniqueSorted(values) {
  return Array.from(new Set(values || [])).sort();
}

function draftPython(draft) {
  if (!draft.python || typeof draft.python !== "object") {
    draft.python = { disabled_packages: [], disabled_methods: [] };
  }
  draft.python.disabled_packages = draft.python.disabled_packages || [];
  draft.python.disabled_methods = draft.python.disabled_methods || [];
  return draft.python;
}

function setDisabled(draft, key, name, disabled) {
  const current = new Set(draft[key] || []);
  if (disabled) current.add(name);
  else current.delete(name);
  draft[key] = uniqueSorted(current);
}

function setPythonDisabled(draft, key, name, disabled) {
  const python = draftPython(draft);
  const current = new Set(python[key] || []);
  if (disabled) current.add(name);
  else current.delete(name);
  python[key] = uniqueSorted(current);
}

export function openToolConfigDialog() {
  openDialog((dialog) => {
    dialog.classList.add("tool-config-dialog");

    const heading = el("div", { class: "dialog-heading" });
    heading.append(
      el("h3", { text: "Tool Configuration" }),
      el("p", {
        class: "dialog-subtitle",
        text:
          "Choose which tools and capabilities the agent can use, and what run_python " +
          "can import. Required tools stay enabled because the system prompt depends on them.",
      })
    );

    const scopeBar = el("div", { class: "tool-config-scope" });
    const scopeSwitch = el("div", {
      class: "scope-switch",
      role: "group",
      "aria-label": "Configuration scope",
    });
    const scopeButtons = {};
    for (const [id, label] of [
      ["global", "Global"],
      ["workspace", "This workspace"],
    ]) {
      const button = el("button", {
        type: "button",
        text: label,
        "aria-pressed": "false",
        disabled: "disabled",
        onclick: () => selectScope(id),
      });
      scopeButtons[id] = button;
      scopeSwitch.append(button);
    }
    const scopeNote = el("p", { class: "tool-config-scope-note" });
    scopeBar.append(scopeSwitch, scopeNote);

    const tabBar = el("div", {
      class: "tool-config-tabs",
      role: "tablist",
      "aria-label": "Tool configuration sections",
    });
    const tabButtons = {};
    for (const tab of TABS) {
      const button = el("button", {
        type: "button",
        role: "tab",
        text: tab.label,
        "aria-selected": "false",
        onclick: () => selectTab(tab.id),
      });
      tabButtons[tab.id] = button;
      tabBar.append(button);
    }

    const body = el("div", { class: "tool-config-body" });
    const status = el("p", { class: "dialog-status" });

    const actions = el("div", { class: "dialog-actions" });
    const hint = el("span", {
      class: "dialog-footer-hint",
      text: "Cancel discards unsaved changes.",
    });
    const cancel = el("button", { type: "button", text: "Cancel", onclick: closeDialog });
    const save = el("button", {
      class: "primary",
      type: "button",
      text: "Save changes",
      disabled: "disabled",
    });
    actions.append(hint, cancel, save);

    dialog.append(heading, scopeBar, tabBar, body, status, actions);

    let payload = null;
    let scope = "global";
    let tab = "tools";
    let filterText = "";
    const drafts = {};

    body.append(el("p", { class: "tool-config-empty", text: "Loading configuration…" }));
    getToolConfig()
      .then((data) => {
        payload = data;
        scope = data.source === "workspace" ? "workspace" : "global";
        if (scope === "workspace" && !state.active_workspace) scope = "global";
        drafts.global = clone(data.global || {});
        if (data.workspace) drafts.workspace = clone(data.workspace);
        save.disabled = false;
        render();
      })
      .catch((error) => {
        body.replaceChildren(
          el("p", { class: "tool-config-empty", text: "Could not load tool configuration." })
        );
        status.className = "dialog-status error";
        status.textContent = error.message || String(error);
      });

    function workspaceAvailable() {
      return Boolean(state.active_workspace);
    }

    function draftFor(target) {
      if (!drafts[target]) {
        const base =
          target === "workspace"
            ? payload.workspace || drafts.global || payload.global
            : payload.global;
        drafts[target] = clone(base || {});
      }
      draftForEnsure(drafts[target]);
      return drafts[target];
    }

    function draftForEnsure(draft) {
      draft.disabled_tools = draft.disabled_tools || [];
      draftPython(draft);
    }

    function selectScope(next) {
      if (!payload || next === scope) return;
      if (next === "workspace" && !workspaceAvailable()) return;
      scope = next;
      render();
    }

    function selectTab(next) {
      if (next === tab) return;
      tab = next;
      render();
    }

    function markDirty() {
      status.className = "dialog-status";
      status.textContent = "Unsaved changes";
    }

    function render() {
      renderScope();
      renderTabs();
      renderBody();
    }

    function renderScope() {
      for (const [id, button] of Object.entries(scopeButtons)) {
        const active = id === scope;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", String(active));
        button.disabled = id === "workspace" && !workspaceAvailable();
      }
      const hasOverride = Boolean(payload && payload.workspace);
      if (scope === "global") {
        scopeNote.textContent =
          "Saved to workspaces/tools.json. Workspaces with their own tools.json ignore this configuration.";
      } else if (!workspaceAvailable()) {
        scopeNote.textContent = "Open a workspace to save a workspace configuration.";
      } else if (hasOverride) {
        scopeNote.textContent =
          "This workspace has its own tools.json; it overrides the global configuration.";
      } else {
        scopeNote.textContent =
          "This workspace follows the global configuration. Saving creates tools.json in this workspace.";
      }
      if (payload && payload.source === "workspace" && scope === "global") {
        scopeNote.textContent += " The open workspace is currently using its own override.";
      }
    }

    function renderTabs() {
      for (const [id, button] of Object.entries(tabButtons)) {
        const active = id === tab;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      }
    }

    function renderBody(keepScroll = false) {
      if (!payload) return;
      const scrollTop = keepScroll ? body.scrollTop || 0 : 0;
      body.replaceChildren();
      if (tab === "tools") renderTools();
      else renderPython();
      body.scrollTop = scrollTop;
    }

    function renderTools() {
      const intro = el("p", {
        class: "tool-config-intro",
        text:
          "Category switches enable or disable every configurable tool in that group. " +
          "Locked tools and harness capabilities are always available.",
      });
      const search = el("input", {
        type: "text",
        class: "tool-config-search",
        placeholder: "Filter tools…",
        autocomplete: "off",
        spellcheck: "false",
        "aria-label": "Filter tools",
      });
      search.value = filterText;
      const list = el("div", { class: "tool-config-list" });
      const draft = draftFor(scope);

      const draw = () => {
        list.replaceChildren();
        const query = filterText.trim().toLowerCase();
        for (const category of payload.categories) {
          const entries = category.tools.filter((tool) => {
            if (!query) return true;
            return (
              tool.name.toLowerCase().includes(query) ||
              tool.summary.toLowerCase().includes(query) ||
              category.label.toLowerCase().includes(query)
            );
          });
          if (!entries.length) continue;
          list.append(categoryBlock(category, entries, draft));
        }
        if (!list.children.length) {
          list.append(
            el("p", { class: "tool-config-empty", text: "No tools match this filter." })
          );
        }
      };

      search.addEventListener("input", () => {
        filterText = search.value || "";
        draw();
      });
      body.append(intro, search, list);
      draw();
    }

    function categoryBlock(category, entries, draft) {
      const disabled = new Set(draft.disabled_tools || []);
      const configurable = category.tools.filter((tool) => tool.configurable);
      const enabledCount = configurable.filter((tool) => !disabled.has(tool.name)).length;
      const block = el("section", { class: "tool-category" });
      const head = el("div", { class: "tool-category-head" });
      const copy = el("div", { class: "tool-category-copy" });
      copy.append(
        el("span", { class: "tool-category-name", text: category.label }),
        el("span", {
          class: "tool-category-meta",
          text: category.configurable_count
            ? `${enabledCount} of ${category.configurable_count} enabled`
            : "Always on",
        })
      );
      const master = el("input", {
        type: "checkbox",
        class: "tool-category-toggle",
        "aria-label": `Enable all ${category.label} tools`,
      });
      master.disabled = !configurable.length || !entries.some((tool) => tool.configurable);
      master.checked = configurable.length > 0 && enabledCount === configurable.length;
      master.indeterminate = enabledCount > 0 && enabledCount < configurable.length;
      master.addEventListener("change", () => {
        for (const tool of configurable) {
          setDisabled(draft, "disabled_tools", tool.name, !master.checked);
        }
        markDirty();
        renderBody(true);
      });
      if (category.description) {
        copy.append(el("span", { class: "tool-category-hint", text: category.description }));
      }
      head.append(copy, master);
      block.append(head);

      for (const tool of entries) block.append(toolRow(tool, draft));
      return block;
    }

    function toolRow(tool, draft) {
      const disabled = new Set(draft.disabled_tools || []);
      const enabled = tool.required || !tool.configurable || !disabled.has(tool.name);
      const row = el("label", {
        class: "tool-row" + (enabled ? "" : " disabled") + (tool.available ? "" : " unavailable"),
      });
      const check = el("input", { type: "checkbox", "aria-label": tool.name });
      check.checked = enabled;
      check.disabled = !tool.configurable;
      check.addEventListener("change", () => {
        setDisabled(draft, "disabled_tools", tool.name, !check.checked);
        markDirty();
        renderBody(true);
      });

      const copy = el("span", { class: "tool-row-copy" });
      const title = el("span", { class: "tool-row-title" });
      const name = el("span", { class: "tool-row-name", text: tool.name });
      title.append(name);
      if (tool.required) {
        const badge = badgeWithIcon("shield", "Required", tool.required_reason);
        title.append(badge);
      } else if (!tool.configurable) {
        title.append(
          badgeWithIcon("lock", "Harness", "Registered by a harness capability; always on.")
        );
      }
      if (tool.model_specific) title.append(badge("Model-specific", ""));
      if (!tool.available) {
        title.append(badge("Unavailable", "The configured model cannot serve this tool."));
      }
      copy.append(title);
      if (tool.summary) copy.append(el("span", { class: "tool-row-summary", text: tool.summary }));
      row.append(check, copy);
      return row;
    }

    function badge(text, title) {
      const span = el("span", { class: "tool-badge", text });
      if (title) span.setAttribute("title", title);
      return span;
    }

    function badgeWithIcon(name, text, title) {
      const span = el("span", { class: "tool-badge locked", title: title || "" });
      span.append(icon(name, { size: 11 }), el("span", { text }));
      return span;
    }

    function renderPython() {
      const draft = draftFor(scope);
      const python = draftPython(draft);
      body.append(
        el("p", {
          class: "tool-config-intro",
          text:
            "These control the run_python sandbox only; tool implementations keep their " +
            "own dependencies. Dangerous mode still bypasses the import guard.",
        }),
        el("p", {
          class: "tool-config-note",
          text:
            "Disabling a package removes it and its alias from the snippet namespace. " +
            "Disabling every method in a helper removes that helper.",
        })
      );

      for (const group of payload.python.package_groups) {
        const rows = payload.python.packages.filter((item) => item.group === group);
        if (!rows.length) continue;
        body.append(
          pythonGroup(
            group,
            rows.map((item) => packageRow(item, python)),
            "Packages"
          )
        );
      }
      for (const group of payload.python.method_groups) {
        const rows = payload.python.methods.filter((item) => item.group === group);
        if (!rows.length) continue;
        body.append(
          pythonGroup(
            group,
            rows.map((item) => methodRow(item, python)),
            "Helper methods"
          )
        );
      }
    }

    function pythonGroup(label, rows, kind) {
      const section = el("section", { class: "tool-category" });
      const head = el("div", { class: "tool-category-head" });
      const copy = el("div", { class: "tool-category-copy" });
      copy.append(
        el("span", { class: "tool-category-name", text: label }),
        el("span", { class: "tool-category-meta", text: kind })
      );
      head.append(copy);
      section.append(head);
      for (const row of rows) section.append(row);
      return section;
    }

    function packageRow(item, python) {
      const disabled = new Set(python.disabled_packages || []);
      const enabled = !disabled.has(item.name);
      const row = el("label", {
        class: "tool-row" + (enabled ? "" : " disabled") + (item.installed ? "" : " unavailable"),
      });
      const check = el("input", { type: "checkbox", "aria-label": item.name });
      check.checked = enabled;
      check.disabled = !item.installed;
      check.addEventListener("change", () => {
        setPythonDisabled(draftFor(scope), "disabled_packages", item.name, !check.checked);
        markDirty();
        renderBody(true);
      });
      const copy = el("span", { class: "tool-row-copy" });
      const title = el("span", { class: "tool-row-title" });
      title.append(el("span", { class: "tool-row-name", text: item.name }));
      if (item.optional) title.append(badge("Optional", ""));
      if (!item.installed) title.append(badge("Not installed", "This package is not in this interpreter."));
      else if (item.version) title.append(badge(item.version, "Installed version"));
      copy.append(title);
      if (item.purpose) copy.append(el("span", { class: "tool-row-summary", text: item.purpose }));
      row.append(check, copy);
      return row;
    }

    function methodRow(item, python) {
      const disabled = new Set(python.disabled_methods || []);
      const enabled = !disabled.has(item.name);
      const row = el("label", { class: "tool-row" + (enabled ? "" : " disabled") });
      const check = el("input", { type: "checkbox", "aria-label": item.name });
      check.checked = enabled;
      check.addEventListener("change", () => {
        setPythonDisabled(draftFor(scope), "disabled_methods", item.name, !check.checked);
        markDirty();
        renderBody(true);
      });
      const copy = el("span", { class: "tool-row-copy" });
      const title = el("span", { class: "tool-row-title" });
      title.append(el("span", { class: "tool-row-name", text: item.name }));
      copy.append(title);
      if (item.purpose) copy.append(el("span", { class: "tool-row-summary", text: item.purpose }));
      row.append(check, copy);
      return row;
    }

    save.addEventListener("click", async () => {
      if (!payload) return;
      const draft = clone(draftFor(scope));
      draft.disabled_tools = uniqueSorted(draft.disabled_tools || []);
      draftPython(draft);
      draft.python.disabled_packages = uniqueSorted(draft.python.disabled_packages);
      draft.python.disabled_methods = uniqueSorted(draft.python.disabled_methods);
      save.disabled = true;
      cancel.disabled = true;
      status.className = "dialog-status";
      status.textContent = "Saving…";
      try {
        const response = await saveToolConfig(scope, draft);
        payload = response;
        setState({
          tool_config: response.effective,
          tool_config_source: response.source,
        });
        toast(
          scope === "global" && response.source === "workspace"
            ? "Global tool configuration saved; this workspace keeps its override"
            : "Tool configuration saved",
          "success"
        );
        closeDialog();
      } catch (error) {
        status.className = "dialog-status error";
        status.textContent = error.message || String(error);
        save.disabled = false;
        cancel.disabled = false;
      }
    });
  });
}
