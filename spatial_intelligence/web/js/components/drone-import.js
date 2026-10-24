/* Drone imagery prompt: list what an import detected, then add it to the map. */

import {
  cancelDroneImport,
  detectDroneFiles,
  getState,
  importDroneFiles,
} from "../api.js";
import { closeDialog, el, formatBytes, openDialog, toast } from "../dom.js";
import { icon } from "../icons.js";
import { applySnapshot, state } from "../store.js";

/**
 * Inspect `paths` and, when drone files are found, open the import dialog.
 *
 * Returns `true` when a dialog was opened. Callers that actively asked for an
 * import (the file tree) pass `notifyIfEmpty` so an empty selection is not
 * silent.
 */
export async function offerDroneImport(paths, { notifyIfEmpty = false } = {}) {
  const selected = (paths || []).filter((path) => typeof path === "string" && path);
  if (!selected.length) {
    if (notifyIfEmpty) toast("Select a file or folder first", "warning");
    return false;
  }

  let result;
  try {
    result = await detectDroneFiles(selected);
  } catch (error) {
    if (notifyIfEmpty) toast(error.message || String(error), "error");
    return false;
  }

  const files = result.files || [];
  if (!files.length) {
    if (notifyIfEmpty) {
      toast(
        "No drone imagery found. Supported: GeoTIFF/COG and geotagged JPEG/TIFF/PNG/WebP.",
        "warning"
      );
    }
    return false;
  }

  openDialog((dialog) => {
    const heading = el("div", { class: "dialog-heading" });
    heading.append(
      el("h3", { text: "Imagery detected" }),
      el("p", {
        class: "dialog-subtitle",
        text:
          "These files can be added to the live GeoLibre map. Nothing is added until you choose Import.",
      })
    );

    const counts = [];
    if (result.photos) counts.push(result.photos + " photo(s)");
    if (result.rasters) counts.push(result.rasters + " raster(s)");
    if (result.skipped) counts.push(result.skipped + " skipped");
    if (result.truncated) counts.push("scan truncated");
    const summary = el("p", {
      class: "drone-detect-summary",
      text: counts.length ? counts.join(" · ") : files.length + " file(s)",
    });

    const list = el("div", { class: "drone-detect-list" });
    for (const file of files) {
      const row = el("div", { class: "drone-detect-row" });
      const raster = file.kind === "raster";
      row.append(
        icon(raster ? "layers" : "globe", {
          size: 17,
          class: "drone-detect-icon " + (raster ? "kind-raster" : "kind-photo"),
        })
      );
      const copy = el("div", { class: "drone-detect-copy" });
      const name = String(file.path || "").split("/").pop() || file.path || "file";
      copy.append(
        el("span", { class: "drone-detect-name", text: name }),
        el("span", { class: "drone-detect-detail", text: file.detail || file.kind })
      );
      row.append(copy);
      row.append(
        el("span", {
          class: "drone-detect-size",
          text: formatBytes(file.size),
        })
      );
      list.append(row);
    }

    const ortho = result.orthorectify || {};
    const option = el("div", { class: "drone-ortho-option" });
    let orthoBox = null;
    if (ortho.available) {
      const label = el("label", { class: "drone-ortho-label" });
      orthoBox = el("input", { type: "checkbox" });
      orthoBox.checked = false;
      const copy = el("span", { class: "drone-ortho-copy" });
      copy.append(
        el("strong", {
          text: "Build an orthomosaic from " + ortho.photos + " photo(s)",
        }),
        el("small", {
          text:
            "Orthority orthorectifies each frame and merges them. DEM: " +
            String(ortho.dem || "").split("/").pop() +
            ". The raw frames are also added as points. This can take a minute.",
        })
      );
      label.append(orthoBox, copy);
      option.append(label);
    } else if (ortho.reason && ortho.photos) {
      option.append(
        el("p", {
          class: "drone-ortho-hint",
          text: "Orthorectification unavailable: " + ortho.reason,
        })
      );
    }

    const status = el("p", { class: "dialog-status" });
    let running = false;
    let wantsOrtho = false;
    let stopJobId = null;
    let watcher = null;
    const stopWatching = () => {
      clearInterval(watcher);
      watcher = null;
    };

    const cancel = el("button", { type: "button", text: "Cancel" });
    cancel.addEventListener("click", () => {
      if (!running) {
        closeDialog();
        return;
      }
      if (!wantsOrtho) return;
      if (!stopJobId) {
        status.textContent = "Starting orthorectification…";
        return;
      }
      status.className = "dialog-status";
      status.textContent = "Stopping…";
      cancelDroneImport(stopJobId).catch(() => {});
    });

    const add = el("button", {
      class: "primary",
      type: "button",
      text: "Import to GeoLibre",
    });
    const actions = el("div", { class: "dialog-actions" });
    actions.append(cancel, add);

    const updateAddLabel = () => {
      add.textContent = orthoBox && orthoBox.checked
        ? "Import & orthorectify"
        : "Import to GeoLibre";
    };
    if (orthoBox) {
      orthoBox.addEventListener("change", updateAddLabel);
      updateAddLabel();
    }

    add.addEventListener("click", async () => {
      wantsOrtho = Boolean(orthoBox && orthoBox.checked);
      running = true;
      add.disabled = true;
      cancel.disabled = false;
      status.className = "dialog-status";
      status.textContent = wantsOrtho
        ? "Starting orthorectification…"
        : "Adding layers to the map…";
      if (wantsOrtho) {
        watcher = setInterval(() => {
          if (!status.isConnected) {
            stopWatching();
            return;
          }
          const job = (state.jobs || []).find(
            (item) => item.kind === "drone.orthorectify" && item.status === "running"
          );
          if (!job) return;
          stopJobId = job.job_id;
          const total = Number(job.total) || 0;
          const done = Number(job.completed) || 0;
          status.textContent = total
            ? "Orthorectifying " + done + "/" + total + " frames…"
            : "Orthorectifying…";
        }, 500);
      }
      try {
        const imported = await importDroneFiles(
          files.map((file) => file.path),
          wantsOrtho
        );
        applySnapshot(await getState());
        stopWatching();
        running = false;
        closeDialog();
        const added = imported.added != null
          ? imported.added
          : (imported.layers || []).length;
        const reused = imported.reused || 0;
        const parts = [];
        if (added) {
          parts.push(added === 1 ? "Added 1 layer" : "Added " + added + " layers");
        }
        if (reused) parts.push(reused + " already on the map");
        const built = imported.orthorectified
          ? " Built an orthomosaic from " + imported.orthorectified.frames + " photo(s)."
          : "";
        toast((parts.join("; ") || "Nothing new to add") + "." + built, "success");
      } catch (error) {
        stopWatching();
        running = false;
        stopJobId = null;
        status.className = "dialog-status error";
        status.textContent = error.message || String(error);
        add.disabled = false;
        cancel.disabled = false;
      }
    });

    dialog.append(heading, summary, list, option, status, actions);
  });
  return true;
}
