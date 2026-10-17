/* Job progress: one component for every job kind, mounted standalone or in a cell. */

import { appendChildren, el, formatBytes } from "../dom.js";
import { icon } from "../icons.js";

function jobStatusLabel(job) {
  if (job.status === "done") return "complete";
  if (job.status === "error") return "failed";
  if (job.status === "cancelled") return "stopped";
  return job.status || "running";
}

function isTerminal(job) {
  return job.status === "done" || job.status === "error" || job.status === "cancelled";
}

function jobAmount(value, unit) {
  if (unit === "bytes") return formatBytes(value);
  const amount = Number(value) || 0;
  return unit ? amount + " " + unit : String(amount);
}

function jobPercent(job) {
  const total = job.total == null ? null : Number(job.total);
  if (total == null || !Number.isFinite(total) || total <= 0) return null;
  const completed = Number(job.completed) || 0;
  return Math.min(100, Math.round((completed / total) * 100));
}

function jobProgressLabel(job) {
  const completed = Number(job.completed) || 0;
  const percent = jobPercent(job);
  const amount = jobAmount(completed, job.unit);
  if (job.status === "done") return "100% · " + amount;
  if (percent != null) {
    return percent + "% · " + amount + " / " + jobAmount(job.total, job.unit);
  }
  return isTerminal(job) ? amount : amount + " · " + jobStatusLabel(job);
}

function jobDescription(job) {
  const saved = job.artifact ? "Saved to " + job.artifact : "";
  if (job.detail && saved) return job.detail + " · " + saved;
  return job.detail || saved;
}

function jobBody(job, className) {
  const body = el("div", { class: className });
  body.append(
    el("div", { class: "download-description", text: jobDescription(job) })
  );
  const progress = el("div", { class: "download-progress" });
  progress.append(el("div", { class: "download-progress-fill" }));
  body.append(progress, el("div", { class: "download-progress-label" }));
  if (job.error) body.append(el("div", { class: "download-error", text: job.error }));
  return body;
}

function jobStatusNode(job) {
  return el("span", {
    class: "agent-action-status download-status status-" + (job.status || "running"),
    text: jobStatusLabel(job),
  });
}

/**
 * Build the progress node for one job.
 *
 * `variant` defaults to "cell", the standalone cell used for a job whose
 * parent_id matches no visible cell; "trace" is the compact node mounted in a
 * cell's job strip.
 */
export function renderJob(job, variant = "cell") {
  const node =
    variant === "trace"
      ? el("div", {
          class: "trace-step download-trace download-progress-node",
          "data-job-id": job.job_id,
        })
      : el("div", { class: "cell download-cell", "data-job-id": job.job_id });

  if (variant === "trace") {
    const summary = el("div", { class: "download-trace-summary" });
    const mark = el("span", { class: "trace-icon" });
    mark.append(icon("download", { size: 13 }));
    appendChildren(
      summary,
      mark,
      el("span", { class: "trace-name", text: job.kind || "job" }),
      el("span", {
        class: "trace-preview download-filename",
        text: job.label || "",
      }),
      jobStatusNode(job)
    );
    node.append(summary, jobBody(job, "trace-body download-body"));
  } else {
    const header = el("div", { class: "cell-header" });
    appendChildren(
      header,
      el("span", { class: "counter", text: "In[ ]" }),
      el("span", { class: "badge", text: job.kind || "job" }),
      el("span", { class: "download-filename", text: job.label || "" }),
      jobStatusNode(job)
    );
    node.append(header, jobBody(job, "download-body"));
  }

  updateJob(node, job);
  return node;
}

/** Refresh a job node in place after a progress event. */
export function updateJob(node, job) {
  const percent = jobPercent(job);
  const indeterminate = percent == null && !isTerminal(job);
  const fill = node.querySelector(".download-progress-fill");
  const progress = node.querySelector(".download-progress");
  const label = node.querySelector(".download-progress-label");
  const description = node.querySelector(".download-description");
  const status = node.querySelector(".download-status");

  if (progress) progress.classList.toggle("indeterminate", indeterminate);
  if (fill) {
    if (indeterminate) fill.style.width = "35%";
    else if (percent != null) fill.style.width = percent + "%";
    else fill.style.width = job.status === "done" ? "100%" : "0";
  }
  if (label) label.textContent = jobProgressLabel(job);
  if (description) description.textContent = jobDescription(job);
  if (status) {
    status.textContent = jobStatusLabel(job);
    status.className =
      "agent-action-status download-status status-" + (job.status || "running");
  }
  if (job.error) {
    let error = node.querySelector(".download-error");
    if (!error) {
      error = el("div", { class: "download-error" });
      const body = node.querySelector(".download-body");
      if (body) body.append(error);
    }
    error.textContent = job.error;
  }
  return node;
}

export { updateJob as updateJobNode };
