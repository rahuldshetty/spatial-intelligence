/* Shared DOM construction, dialog, toast, and formatting helpers. */

"use strict";

let toastTimer = null;
let dialogCleanup = null;

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

export function toast(message) {
  let node = document.getElementById("toast");
  if (!node) {
    node = el("div", { id: "toast" });
    document.body.append(node);
  }
  node.textContent = message;
  node.style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.style.display = "none";
  }, 4000);
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
  const dialog = el("div", { id: "dialog" });
  overlay.append(dialog);
  document.body.append(overlay);
  dialogCleanup = () => overlay.remove();
  buildContent(dialog);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeDialog();
  });
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
