/* Shared DOM construction, dialog, toast, and formatting helpers. */

"use strict";

import { icon } from "./icons.js";

let toastTimer = null;
let dialogCleanup = null;
let dialogCleanupFocus = null;

const TOAST_ICONS = {
  info: "circle",
  success: "check-circle",
  warning: "alert-triangle",
  error: "x-circle",
};

export function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2), value);
      } else if (value !== null && value !== undefined) {
        node.setAttribute(key, value);
      }
    }
  }
  if (children != null) {
    for (const child of Array.isArray(children) ? children : [children]) {
      if (child == null) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
  }
  return node;
}

export function appendChildren(node, ...children) {
  node.append(...children.filter((child) => child != null));
}

export function toast(message, kind = "info") {
  const variant = TOAST_ICONS[kind] ? kind : "info";
  let node = document.getElementById("toast");
  if (!node) {
    node = el("div", { id: "toast" });
    document.body.append(node);
  }
  node.className = "toast toast-" + variant;
  node.setAttribute("role", variant === "error" ? "alert" : "status");
  node.setAttribute("aria-live", variant === "error" ? "assertive" : "polite");
  node.replaceChildren(
    icon(TOAST_ICONS[variant], { size: 15 }),
    el("span", { class: "toast-text", text: message })
  );
  node.style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.style.display = "none";
  }, variant === "error" ? 7000 : 4000);
}

function dismissMenus() {
  document.querySelectorAll(".menu.open").forEach((menu) => menu.classList.remove("open"));
}

export function openDialog(buildContent, options) {
  dismissMenus();
  const passthrough = Boolean(options && options.passthrough);
  const overlay = el("div", {
    id: "overlay",
    class: "open" + (passthrough ? " passthrough" : ""),
  });
  const dialog = el("div", { id: "dialog", role: "dialog", "aria-modal": "true" });
  overlay.append(dialog);
  document.body.append(overlay);
  dialogCleanupFocus = document.activeElement;
  dialogCleanup = () => {
    overlay.remove();
    const previous = dialogCleanupFocus;
    dialogCleanupFocus = null;
    if (previous && typeof previous.focus === "function") previous.focus();
  };
  buildContent(dialog);
  if (!dialog.querySelector(".dialog-close")) {
    const close = el("button", {
      class: "dialog-close",
      type: "button",
      title: "Close",
      "aria-label": "Close dialog",
      onclick: closeDialog,
    });
    close.append(icon("x", { size: 16 }));
    dialog.append(close);
  }
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeDialog();
  });
  overlay.addEventListener("keydown", trapDialogKeys);
  window.requestAnimationFrame(() => {
    const target =
      dialog.querySelector("input:not([type=hidden]), select, textarea") ||
      dialog.querySelector(".primary") ||
      dialog.querySelector(".dialog-close");
    if (target && typeof target.focus === "function") target.focus();
  });
}

function trapDialogKeys(event) {
  if (event.key === "Escape") {
    event.preventDefault();
    closeDialog();
    return;
  }
  if (event.key !== "Tab") return;
  const focusable = Array.from(
    event.currentTarget.querySelectorAll(
      "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])"
    )
  );
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

export function closeDialog() {
  if (dialogCleanup) {
    dialogCleanup();
    dialogCleanup = null;
  }
}

export function dialogActions(dialog, primary, onConfirm) {
  const actions = el("div", { class: "dialog-actions" });
  actions.append(el("button", { text: "Cancel", onclick: closeDialog }));
  if (primary) {
    actions.append(el("button", { class: "primary", text: primary, onclick: onConfirm }));
  }
  dialog.append(actions);
}

export function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  if (bytes < 1024) return bytes + " B";
  const units = ["KB", "MB", "GB", "TB"];
  let amount = bytes;
  let unit = -1;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return amount.toFixed(amount >= 10 || unit === 0 ? 0 : 1) + " " + units[unit];
}

export function formatTokens(value) {
  const count = Number(value) || 0;
  if (count >= 1000000) return (count / 1000000).toFixed(1) + "M";
  if (count >= 1000) return (count / 1000).toFixed(1) + "k";
  return String(count);
}

export function formatCost(value) {
  if (value == null) return null;
  const cost = Number(value);
  if (!isFinite(cost) || cost === 0) return null;
  return "$" + (cost < 0.001 ? cost.toExponential(1) : cost.toFixed(4));
}

export function usageLabel(usage) {
  const parts = [];
  if (usage.input_tokens != null) parts.push("↑" + formatTokens(usage.input_tokens));
  if (usage.output_tokens != null) parts.push("↓" + formatTokens(usage.output_tokens));
  const cost = formatCost(usage.cost);
  if (cost) parts.push(cost);
  return parts.join(" ");
}

export function usageTitle(usage) {
  const parts = [];
  if (usage.requests != null) {
    parts.push(usage.requests + " request" + (usage.requests === 1 ? "" : "s"));
  }
  if (usage.tool_calls != null) {
    parts.push(usage.tool_calls + " tool call" + (usage.tool_calls === 1 ? "" : "s"));
  }
  if (usage.total_tokens != null) parts.push(usage.total_tokens + " total tokens");
  if (usage.cache_read_tokens) {
    parts.push(usage.cache_read_tokens + " cached read tokens");
  }
  return parts.join(" · ");
}

export function truncate(value, limit) {
  const text = String(value == null ? "" : value);
  return text.length > limit ? text.slice(0, limit) + "…" : text;
}

export function prettyValue(value) {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (
      (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
      (trimmed.startsWith("[") && trimmed.endsWith("]"))
    ) {
      try {
        return JSON.stringify(JSON.parse(value), null, 2);
      } catch (_) {
        /* not valid JSON — fall through to raw */
      }
    }
    return value;
  }
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch (_) {
    return String(value);
  }
}

export function compactPreview(value) {
  let text;
  if (typeof value === "string") text = value;
  else if (value == null) text = "";
  else {
    try {
      text = JSON.stringify(value);
    } catch (_) {
      text = String(value);
    }
  }
  return truncate(String(text).replace(/\s+/g, " ").trim(), 100);
}

export function codeBlock(content) {
  const pre = el("pre", { class: "trace-code" });
  pre.append(el("code", { text: content == null ? "" : String(content) }));
  return pre;
}

export function expandableContent(label, content, extraClass = "") {
  const text = String(content == null ? "" : content);
  const details = el("details", {
    class: "agent-action-content" + (extraClass ? " " + extraClass : ""),
  });
  const summary = el("summary", {});
  summary.append(
    el("strong", { text: label }),
    el("span", { class: "agent-action-preview", text: compactPreview(text) })
  );
  details.append(summary, el("pre", { text }));
  return details;
}
