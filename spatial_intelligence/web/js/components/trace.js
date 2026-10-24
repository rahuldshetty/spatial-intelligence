/* Agent trace: one ordered renderer, plus a live fast path for streamed text. */

import { codeBlock, compactPreview, el, prettyValue } from "../dom.js";
import { icon } from "../icons.js";
import { jobsFor, state, stepKey } from "../store.js";
import { cellOutputText, renderMarkdown, visibleTraceGroups } from "./markdown.js";
import { planFromTrace, renderPlan } from "./plan.js";
import { renderJob } from "./progress.js";
import { renderInteraction, reconcilePendingInteraction } from "./interaction.js";

/* Steps published before the snapshot that introduces their cell.

The event stream opens before ``/api/state`` is applied, so a step can arrive
while ``state.cells`` is still empty. The snapshot landing next already carries
everything published up to the moment it was built, and that overlap is not
knowable from here: a buffered step is replayed only when the snapshot does not
already hold one like it.
*/
const pendingTrace = new Map();
let awaitingSnapshot = true;

/* Long reasoning is folded by default; the reader can open it once. */
const expandedText = new Set();
const TEXT_LINE_LIMIT = 12;
const TEXT_CHAR_LIMIT = 1200;

function isCodeTool(name) {
  return name === "run_python";
}

function extractCode(args) {
  let obj = args;
  if (typeof obj === "string") {
    try {
      obj = JSON.parse(obj);
    } catch (_) {
      return obj;
    }
  }
  if (obj && typeof obj === "object") {
    if (obj.code != null) return obj.code;
    if (obj.source != null) return obj.source;
  }
  return obj;
}

function parsePlanItems(args) {
  let obj = args;
  if (typeof obj === "string") {
    try {
      obj = JSON.parse(obj);
    } catch (_) {
      return [];
    }
  }
  if (obj && typeof obj === "object" && Array.isArray(obj.items)) return obj.items;
  return [];
}

function statusOf(item) {
  return (item && item.status) || "pending";
}

function toolPreview(name, call, result, isCode) {
  if (name === "write_plan" && call) {
    const items = parsePlanItems(call.args);
    const done = items.filter((item) => {
      const status = statusOf(item);
      return status === "completed" || status === "cancelled";
    }).length;
    const active = items.filter((item) => statusOf(item) === "in_progress").length;
    return items.length + " steps · " + done + " done" + (active ? " · " + active + " active" : "");
  }
  if (call) return compactPreview(isCode ? extractCode(call.args) : call.args);
  if (result) return compactPreview(result.content);
  return "";
}

function toolStepNode(group) {
  const call = group.call;
  const result = group.result;
  const name = (call && call.name) || (result && result.name) || "";
  const isCode = isCodeTool(name);
  const details = el("details", {
    class: "trace-step tool-call" + (result ? "" : " pending"),
    // A repaint rebuilds this node; the key lets it stay open if it was open.
    "data-trace-key": "call:" + ((call && call.tool_call_id) || name),
  });
  const summary = el("summary", {});
  const mark = el("span", { class: "trace-icon" });
  mark.append(icon("arrow-right", { size: 13 }));
  summary.append(mark, el("span", { class: "trace-name", text: name || "" }));
  const preview = toolPreview(name, call, result, isCode);
  if (preview) summary.append(el("span", { class: "trace-preview", text: preview }));
  details.append(summary);
  const body = el("div", { class: "trace-body" });
  if (call) {
    body.append(el("div", { class: "trace-io-label", text: "Input" }));
    body.append(codeBlock(isCode ? extractCode(call.args) : prettyValue(call.args)));
  }
  if (result) {
    body.append(el("div", { class: "trace-io-label", text: "Output" }));
    body.append(codeBlock(prettyValue(result.content)));
  }
  details.append(body);
  return details;
}

function recallNode(step) {
  const results = Array.isArray(step.results) ? step.results : [];
  const terms = Array.isArray(step.terms) ? step.terms : [];
  const details = el("details", {
    class: "trace-step trace-recall",
    "data-trace-key": "recall:" + (step.id || step.query || ""),
  });
  const summary = el("summary", {});
  const mark = el("span", { class: "trace-icon" });
  mark.append(icon("database", { size: 13 }));
  const count = results.length + (results.length === 1 ? " hit" : " hits");
  summary.append(
    mark,
    el("span", { class: "trace-name", text: "history recall" }),
    el("span", {
      class: "trace-preview",
      text: (step.query || "") + " · " + count,
    })
  );
  details.append(summary);

  const body = el("div", { class: "trace-body recall-body" });
  if (terms.length) {
    const row = el("div", { class: "recall-terms" });
    row.append(el("span", { class: "recall-terms-label", text: "Terms" }));
    for (const term of terms) {
      row.append(el("span", { class: "recall-term", text: term }));
    }
    body.append(row);
  }
  if (!results.length) {
    body.append(el("div", { class: "recall-empty", text: "No earlier prompt matched." }));
  } else {
    const list = el("div", { class: "recall-hits" });
    for (const hit of results) {
      const item = el("div", { class: "recall-hit" });
      const head = el("div", { class: "recall-hit-head" });
      const kind = hit.tool_name ? hit.kind + " · " + hit.tool_name : hit.kind || "match";
      head.append(el("span", { class: "recall-hit-kind", text: kind }));
      const meta = [
        hit.prompt_id ? "prompt " + hit.prompt_id : "",
        hit.step_index != null ? "step " + hit.step_index : "",
      ]
        .filter(Boolean)
        .join(" · ");
      if (meta) head.append(el("span", { class: "recall-hit-meta", text: meta }));
      item.append(head);
      item.append(el("div", { class: "recall-hit-hint", text: hit.hint || "" }));
      if (hit.prompt) {
        item.append(el("div", { class: "recall-hit-prompt", text: "Prompt: " + hit.prompt }));
      }
      list.append(item);
    }
    body.append(list);
  }
  details.append(body);
  return details;
}

function traceTextNode(content, cellId, index) {
  const node = el("div", { class: "trace-step trace-text markdown" });
  node.dataset.source = content || "";
  const body = el("div", { class: "trace-text-body" });
  body.innerHTML = renderMarkdown(node.dataset.source);
  node.append(body);

  const source = node.dataset.source;
  const lines = source.split("\n").length;
  if (lines <= TEXT_LINE_LIMIT && source.length <= TEXT_CHAR_LIMIT) return node;

  const key = String(cellId || "") + ":" + String(index);
  const open = expandedText.has(key);
  node.classList.toggle("collapsed", !open);
  const more = el("button", {
    type: "button",
    class: "trace-more",
    text: open ? "Show less" : "View more",
    "aria-expanded": String(open),
  });
  more.addEventListener("click", () => {
    const collapsed = node.classList.toggle("collapsed");
    expandedText[collapsed ? "delete" : "add"](key);
    more.textContent = collapsed ? "View more" : "Show less";
    more.setAttribute("aria-expanded", String(!collapsed));
  });
  node.append(more);
  return node;
}

/**
 * Group trace steps for display, keeping the trace index each group starts at.
 *
 * The index is what places a progress card: a card belongs next to the steps
 * that were already published when its job opened.
 */
export function groupTraceSteps(trace) {
  const groups = [];
  let textBuf = null;
  let textIndex = 0;
  const flush = () => {
    if (textBuf !== null) {
      groups.push({ type: "text", content: textBuf, index: textIndex });
      textBuf = null;
    }
  };
  const pending = new Map();
  let fallbackSeq = 0;

  const steps = trace || [];
  for (let index = 0; index < steps.length; index += 1) {
    const step = steps[index];
    if (step.type === "text") {
      flush();
      textBuf = step.content || "";
      textIndex = index;
    } else if (step.type === "text_delta") {
      if (textBuf === null) {
        textBuf = "";
        textIndex = index;
      }
      textBuf += step.content || "";
    } else if (step.type === "tool_call") {
      flush();
      const group = { type: "tool", call: step, result: null, index };
      groups.push(group);
      pending.set(step.tool_call_id || "seq:" + fallbackSeq++, group);
    } else if (step.type === "tool_result") {
      flush();
      let group = null;
      if (step.tool_call_id && pending.has(step.tool_call_id)) {
        group = pending.get(step.tool_call_id);
        pending.delete(step.tool_call_id);
      } else {
        for (const [key, candidate] of pending) {
          if ((candidate.call && candidate.call.name) === (step.name || "")) {
            group = candidate;
            pending.delete(key);
            break;
          }
        }
      }
      if (group) group.result = step;
      else groups.push({ type: "tool", call: null, result: step, index });
    } else if (step.type === "usage") {
      flush();
      groups.push({ type: "usage", usage: step.usage, index });
    } else if (step.type === "recall") {
      flush();
      groups.push({ type: "recall", step, index });
    } else {
      flush();
    }
  }
  flush();
  return groups;
}

/**
 * One cell's trace steps, each entry carrying the trace index it starts at.
 *
 * Entries are what the ordered renderer places progress cards between; an
 * interaction form is part of the entry it belongs to, or trails the steps.
 */
function traceEntries(trace, cell) {
  const entries = [];
  let interactionRendered = false;
  const renderedInteractionIds = new Set();
  const history = (cell && cell.interaction_history) || [];
  const groups = visibleTraceGroups(
    groupTraceSteps(trace || []),
    cell && cell.status,
    Boolean(cell && cellOutputText(cell).trim())
  );
  for (const group of groups) {
    if (group.type === "text") {
      entries.push({
        index: group.index,
        nodes: [traceTextNode(group.content, cell && cell.id, group.index)],
      });
    } else if (group.type === "tool") {
      const toolNode = toolStepNode(group);
      const nodes = [toolNode];
      const call = group.call;
      const completed =
        call && history.find((item) => item.tool_call_id === call.tool_call_id);
      const pending =
        cell &&
        cell.status === "waiting_for_input" &&
        cell.interaction &&
        call &&
        call.tool_call_id === cell.interaction.tool_call_id;
      if (completed || pending) {
        toolNode.classList.remove("pending");
        if (pending) toolNode.classList.add("waiting");
        const summary = toolNode.querySelector("summary");
        if (summary) {
          summary.append(
            el("span", {
              class: "trace-input-required",
              text: pending ? "input required" : "input provided",
            })
          );
        }
        nodes.push(renderInteraction(cell, completed || cell.interaction));
        if (completed) renderedInteractionIds.add(completed.id);
        else interactionRendered = true;
      }
      entries.push({ index: group.index, nodes });
    } else if (group.type === "recall") {
      entries.push({ index: group.index, nodes: [recallNode(group.step)] });
    }
  }
  const trailing = [];
  if (cell && cell.status === "waiting_for_input" && cell.interaction && !interactionRendered) {
    trailing.push(renderInteraction(cell));
  }
  for (const completed of history) {
    if (!renderedInteractionIds.has(completed.id)) {
      trailing.push(renderInteraction(cell, completed));
    }
  }
  if (trailing.length) {
    entries.push({ index: Number.MAX_SAFE_INTEGER, nodes: trailing });
  }
  return entries;
}

/**
 * Render a cell's trace contents: the plan, its streamed steps, and the
 * progress cards of ``jobs``, each card at the trace index its job opened at.
 *
 * This is the one place the trace's document order is decided. Streaming draws
 * in the same order (see :func:`storeStep`), so a repaint — switching tabs,
 * refreshing the workspace, the end of a run — cannot move a card past the
 * steps that came after it.
 */
export function renderTrace(cell, jobs = []) {
  const trace = (cell && cell.trace) || [];
  const plan = planFromTrace(trace);
  const nodes = plan ? [renderPlan(plan)] : [];
  const cards = (jobs || [])
    .filter(Boolean)
    .map((job) => ({ anchor: jobAnchor(job), node: renderJob(job, "trace") }))
    .sort((left, right) => left.anchor - right.anchor);
  let next = 0;
  const drain = (limit) => {
    while (next < cards.length && cards[next].anchor <= limit) {
      nodes.push(cards[next].node);
      next += 1;
    }
  };
  for (const entry of traceEntries(trace, cell)) {
    drain(entry.index);
    nodes.push(...entry.nodes);
  }
  drain(Number.MAX_SAFE_INTEGER);
  return nodes;
}

/**
 * Where a job's card belongs: the number of steps its cell had published when
 * the job opened. A job from a server that did not report one goes last.
 */
function jobAnchor(job) {
  const anchor = Number(job && job.anchor);
  return Number.isFinite(anchor) ? anchor : Number.MAX_SAFE_INTEGER;
}

/**
 * Draw one freshly streamed step.
 *
 * A text delta that continues the rendered text node is the one update worth
 * doing in place: any other step can move a card, the plan, or an interaction,
 * so the container is rebuilt from the cell's own trace instead.
 */
function storeStep(cell, step) {
  if (!Array.isArray(cell.trace)) cell.trace = [];
  cell.trace.push(step);
  if (step.type === "tool_call" && step.name === "request_user_input") {
    window.setTimeout(() => reconcilePendingInteraction(cell.id), 150);
  }
  const container = document.querySelector('.cell[data-cell-id="' + cell.id + '"] .trace');
  if (!container) return;
  const last = container.lastElementChild;
  if (step.type === "text_delta" && last && last.classList.contains("trace-text")) {
    last.dataset.source = (last.dataset.source || "") + (step.content || "");
    last.innerHTML = renderMarkdown(last.dataset.source);
    container.scrollTop = container.scrollHeight;
    return;
  }
  repaintTrace(cell);
}

/**
 * Repaint a cell's trace container from its trace and its progress jobs.
 *
 * Details the reader opened are reopened by key, so a repaint triggered by the
 * next step does not collapse what is being read.
 */
export function repaintTrace(cell) {
  if (!cell) return null;
  const container = document.querySelector('.cell[data-cell-id="' + cell.id + '"] .trace');
  if (!container) return null;
  const open = new Set(
    [...container.querySelectorAll("details[data-trace-key][open]")].map((node) =>
      node.getAttribute("data-trace-key")
    )
  );
  container.replaceChildren(...renderTrace(cell, jobsFor(cell.id)));
  for (const node of container.querySelectorAll("details[data-trace-key]")) {
    if (open.has(node.getAttribute("data-trace-key"))) node.setAttribute("open", "open");
  }
  container.scrollTop = container.scrollHeight;
  return container;
}

/** Store a streamed trace step on its cell and draw it. */
export function applyTrace(data) {
  if (!data || !data.step) return;
  const cell = state.cells.find((item) => item.id === data.id);
  if (cell) {
    storeStep(cell, data.step);
    return;
  }
  if (!awaitingSnapshot) return;
  const buffered = pendingTrace.get(data.id);
  if (buffered) buffered.push(data.step);
  else pendingTrace.set(data.id, [data.step]);
}

/**
 * Replay the steps buffered before the first snapshot; called once it lands.
 *
 * Steps the snapshot already carries are dropped by key, so a step published
 * during the boot window is neither lost nor rendered twice.
 */
export function flushPendingTrace() {
  awaitingSnapshot = false;
  const buffered = [...pendingTrace];
  pendingTrace.clear();
  for (const [cellId, steps] of buffered) {
    const cell = state.cells.find((item) => item.id === cellId);
    if (!cell) continue;
    const seen = new Map();
    for (const step of cell.trace || []) {
      const key = stepKey(step);
      seen.set(key, (seen.get(key) || 0) + 1);
    }
    for (const step of steps) {
      const key = stepKey(step);
      const remaining = seen.get(key) || 0;
      if (remaining > 0) {
        seen.set(key, remaining - 1);
        continue;
      }
      storeStep(cell, step);
    }
  }
}
