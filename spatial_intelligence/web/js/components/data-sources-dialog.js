/* Data Sources: the registry of available catalogs, their filters, and results.

The modal is two panes: sources on the left, grouped by category, and one source
on the right (search, drill down, then add/download). Sources are enabled by
default and the modal only browses them; the backend still keeps the
enable/disable configuration for a future opt-in UI.
*/

"use strict";

import {
  addDataSourceDataset,
  downloadDataSourceDataset,
  getDataSources,
  searchDataSource,
} from "../api.js";
import { closeDialog, el, openDialog, toast } from "../dom.js";
import { icon } from "../icons.js";
import { state } from "../store.js";

const KIND_LABELS = {
  raster: "Raster",
  vector: "Vector",
  tiles: "Tiles",
  table: "Table",
  collection: "Collection",
};

const CAPABILITY_LABELS = {
  search: "Search",
  spatial: "Area",
  temporal: "Date",
  download: "Download",
  map: "Map",
};

function text(value) {
  return value == null ? "" : String(value);
}

export function openDataSourcesDialog() {
  openDialog((dialog) => {
    dialog.classList.add("data-sources-dialog");

    const heading = el("div", { class: "dialog-heading" });
    heading.append(
      el("h3", { text: "Data Sources" }),
      el("p", {
        class: "dialog-subtitle",
        text:
          "Search the available data sources and add a dataset to the map. " +
          "Search terms are optional; Save to workspace writes only the selected " +
          "dataset under data/.",
      })
    );
    dialog.append(heading);

    const layout = el("div", { class: "data-sources-layout" });
    const sidebar = el("div", { class: "data-sources-sidebar" });
    const main = el("div", { class: "data-sources-main" });
    layout.append(sidebar, main);
    dialog.append(layout);

    const status = el("p", { class: "dialog-status" });
    dialog.append(status);

    const actions = el("div", { class: "dialog-actions" });
    actions.append(
      el("button", { type: "button", text: "Close", onclick: closeDialog })
    );
    dialog.append(actions);

    let payload = null;
    let selectedId = null;
    let filterText = "";
    let mode = "search";
    let results = null;
    let lastQuery = null;
    let busy = false;
    let controls = [];
    let controlsByName = new Map();
    let currentForm = null;
    let resultsNode = null;
    let drilldown = null;

    main.append(el("p", { class: "data-sources-empty", text: "Loading sources…" }));

    getDataSources()
      .then((data) => {
        payload = data;
        render();
      })
      .catch((error) => {
        main.replaceChildren(
          el("p", { class: "data-sources-empty", text: "Could not load data sources." })
        );
        status.className = "dialog-status error";
        status.textContent = error.message || String(error);
      });

    function workspaceAvailable() {
      return Boolean(state.active_workspace);
    }

    function selectedSource() {
      return payload && selectedId
        ? payload.sources.find((source) => source.id === selectedId) || null
        : null;
    }

    function conditions(source) {
      return new Set(source.capabilities || []);
    }

    function setStatus(message, kind = "") {
      status.className = "dialog-status" + (kind ? " " + kind : "");
      status.textContent = message || "";
    }

    function setFormBusy(active) {
      if (!currentForm) return;
      for (const control of currentForm.node.querySelectorAll("input, select, button")) {
        control.disabled = active;
      }
      currentForm.submit.classList.toggle("loading", active);
      if (active) {
        currentForm.submit.replaceChildren(
          el("span", { class: "data-sources-spinner" }),
          document.createTextNode("Searching…")
        );
      } else {
        currentForm.submit.replaceChildren(document.createTextNode("Search"));
      }
    }

    function render() {
      renderSidebar();
      renderMain();
    }

    function renderSidebar() {
      sidebar.replaceChildren();
      const search = el("input", {
        type: "search",
        class: "data-sources-filter",
        placeholder: "Filter sources…",
        autocomplete: "off",
        value: filterText,
        "aria-label": "Filter data sources",
      });
      search.addEventListener("input", () => {
        filterText = search.value || "";
        drawSources();
      });
      const list = el("div", { class: "data-sources-list" });
      sidebar.append(search, list);

      const drawSources = () => {
        list.replaceChildren();
        const query = filterText.trim().toLowerCase();
        const groups = payload.categories || [];
        const seen = new Set();
        for (const category of groups) {
          const rows = payload.sources.filter((source) => {
            if (source.category !== category.id) return false;
            if (!query) return true;
            const haystack = [
              source.id,
              source.name,
              source.summary,
              source.attribution,
            ]
              .join(" ")
              .toLowerCase();
            return haystack.includes(query);
          });
          if (!rows.length) continue;
          seen.add(category.id);
          list.append(sourceGroup(category.label, rows));
        }
        const rest = payload.sources.filter(
          (source) => !seen.has(source.category) && (!query || source.id.includes(query))
        );
        if (rest.length) list.append(sourceGroup("Other", rest));
        if (!list.children.length) {
          list.append(
            el("p", { class: "data-sources-empty", text: "No sources match this filter." })
          );
        }
      };

      drawSources();
    }

    function sourceGroup(label, rows) {
      const section = el("section", { class: "data-sources-group" });
      const head = el("div", { class: "data-sources-group-head" });
      head.append(el("span", { class: "data-sources-group-name", text: label }));
      section.append(head);
      for (const source of rows) section.append(sourceRow(source));
      return section;
    }

    function sourceRow(source) {
      const row = el("div", {
        class:
          "data-sources-row" + (source.id === selectedId ? " selected" : ""),
      });
      const button = el("button", {
        type: "button",
        class: "data-sources-select",
        onclick: () => selectSource(source.id),
      });
      const title = el("span", { class: "data-sources-row-title" });
      title.append(el("span", { text: source.name }));
      if (source.requires_key) title.append(badge("Key", "Requires a key or sign-in."));
      if (!source.available) title.append(badge("Unavailable", source.status?.reason || ""));
      button.append(
        title,
        el("span", { class: "data-sources-row-summary", text: source.summary })
      );
      row.append(button);
      return row;
    }

    function selectSource(id) {
      selectedId = id;
      const source = selectedSource();
      const caps = source ? conditions(source) : new Set();
      mode = caps.has("search") ? "search" : "about";
      results = null;
      lastQuery = null;
      drilldown = null;
      renderSidebar();
      renderMain();
    }

    function renderMain() {
      main.replaceChildren();
      resultsNode = null;
      currentForm = null;
      controls = [];
      controlsByName = new Map();
      const source = selectedSource();
      if (!source) {
        main.append(
          el("p", {
            class: "data-sources-empty",
            text: "Select a source to see what it provides and search it.",
          })
        );
        return;
      }
      const caps = conditions(source);
      const head = el("div", { class: "data-sources-detail-head" });
      const copy = el("div", { class: "data-sources-detail-copy" });
      copy.append(
        el("h4", { text: source.name }),
        el("p", { class: "data-sources-detail-summary", text: source.summary })
      );
      const badges = el("div", { class: "data-sources-badges" });
      badges.append(badge(source.category));
      if (!source.available) badges.append(badge("Unavailable"));
      for (const capability of source.capabilities || []) {
        if (capability === "spatial") continue;
        if (CAPABILITY_LABELS[capability]) badges.append(badge(CAPABILITY_LABELS[capability]));
      }
      copy.append(badges);
      head.append(copy);
      main.append(head);

      const tabs = el("div", { class: "data-sources-tabs", role: "tablist" });
      for (const [id, label, visible] of [
        ["search", "Explore", caps.has("search")],
        ["about", "About", true],
      ]) {
        if (!visible) continue;
        const button = el("button", {
          type: "button",
          role: "tab",
          text: label,
          "aria-selected": String(mode === id),
          onclick: () => {
            if (mode === id) return;
            mode = id;
            results = null;
            lastQuery = null;
            renderMain();
          },
        });
        if (mode === id) button.classList.add("active");
        tabs.append(button);
      }
      main.append(tabs);

      const body = el("div", { class: "data-sources-body" });
      main.append(body);
      if (mode === "about") renderAbout(body, source);
      else renderSearch(body, source);
    }

    function renderAbout(body, source) {
      const list = el("dl", { class: "data-sources-about" });
      addAbout(list, "Source id", source.id);
      addAbout(list, "Category", source.category);
      addAbout(list, "License", source.license || "Provider-specific; check each dataset");
      addAbout(list, "Attribution", source.attribution || "See the provider");
      if (source.homepage) {
        const row = el("dd");
        const link = el("a", {
          href: source.homepage,
          target: "_blank",
          rel: "noopener noreferrer",
          text: source.homepage,
        });
        row.append(link);
        list.append(el("dt", { text: "Homepage" }), row);
      }
      if (source.docs_url) {
        const row = el("dd");
        row.append(
          el("a", {
            href: source.docs_url,
            target: "_blank",
            rel: "noopener noreferrer",
            text: source.docs_url,
          })
        );
        list.append(el("dt", { text: "Docs" }), row);
      }
      addAbout(
        list,
        "Capabilities",
        (source.capabilities || []).map((item) => CAPABILITY_LABELS[item] || item).join(", ")
      );
      if (source.filters?.length) {
        const row = el("dd");
        row.append(
          el("ul", { class: "data-sources-filter-list" }, [
            ...source.filters
              .filter((filter) => !filter.hidden)
              .map((filter) =>
                el("li", { text: filter.label + " (" + filter.kind + ")" })
              ),
          ])
        );
        list.append(el("dt", { text: "Search filters" }), row);
      }
      body.append(list);
    }

    function addAbout(list, label, value) {
      list.append(el("dt", { text: label }), el("dd", { text: text(value) }));
    }

    function renderSearch(body, source) {
      const form = buildForm(source);
      body.append(form.node);
      resultsNode = el("div", { class: "data-sources-results" });
      body.append(resultsNode);
      form.submit.addEventListener("click", () => runSearch(source));
      form.node.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          runSearch(source);
        }
      });
      if (results) drawResults(source, results);
      else resultsNode.append(emptyResults("Run a search to see datasets."));
    }

    function buildForm(source) {
      const node = el("div", { class: "data-sources-form" });
      const caps = conditions(source);
      const bboxMode = source.bbox || "hidden";
      const common = el("div", { class: "data-sources-fields" });
      node.append(common);

      const textInput = el("input", {
        type: "search",
        placeholder: "Search terms (optional)",
        autocomplete: "off",
        "aria-label": "Search terms",
      });
      common.append(field("Search", textInput, "Optional for every source."));

      const limitInput = el("input", {
        type: "number",
        min: "1",
        max: "100",
        step: "1",
        value: "20",
        "aria-label": "Result limit",
      });
      common.append(field("Limit", limitInput, ""));

      for (const spec of source.filters || []) {
        const control = buildFilter(spec);
        if (!control) continue;
        controls.push(control);
        controlsByName.set(spec.name, control);
        if (!spec.hidden) {
          common.append(field(spec.label, control.node, spec.help || ""));
        }
      }

      let startInput = null;
      let endInput = null;
      const hasDateFilter = (source.filters || []).some((filter) => filter.name === "date");
      if (caps.has("temporal") && !hasDateFilter) {
        startInput = el("input", { type: "date", "aria-label": "Start date" });
        endInput = el("input", { type: "date", "aria-label": "End date" });
        const dates = el("div", { class: "data-sources-dates" });
        dates.append(startInput, endInput);
        common.append(field("Dates", dates, "Optional; YYYY-MM-DD."));
      }

      let bboxInputs = null;
      if (bboxMode !== "hidden") {
        bboxInputs = {};
        const bboxRow = el("div", { class: "data-sources-bbox" });
        for (const [key, label, placeholder] of [
          ["west", "W", "-90.1"],
          ["south", "S", "29.9"],
          ["east", "E", "-90.0"],
          ["north", "N", "30.0"],
        ]) {
          const input = el("input", {
            type: "number",
            step: "any",
            placeholder,
            "aria-label": "Bounding box " + label,
          });
          bboxInputs[key] = input;
          bboxRow.append(el("label", { class: "data-sources-bbox-cell" }, [
            el("span", { text: label }),
            input,
          ]));
        }
        common.append(
          field(
            "Area (W, S, E, N)",
            bboxRow,
            bboxMode === "required"
              ? "Required for this source."
              : "All four values or leave blank."
          )
        );
      }

      const submit = el("button", {
        class: "primary",
        type: "button",
        text: "Search",
      });
      const actions = el("div", { class: "data-sources-form-actions" });
      actions.append(submit);
      node.append(actions);

      const form = {
        node,
        submit,
        collect() {
          const body = {
            text: textInput.value.trim(),
            limit: Math.max(1, Math.min(100, Number(limitInput.value) || 20)),
            filters: {},
          };
          if (bboxInputs) {
            const values = ["west", "south", "east", "north"].map((key) =>
              bboxInputs[key].value.trim()
            );
            if (bboxMode === "required" && !values.every(Boolean)) {
              throw new Error("This source needs all four area values.");
            }
            if (values.some(Boolean) && !values.every(Boolean)) {
              throw new Error("Enter all four bbox values or leave them blank.");
            }
            if (values.every(Boolean)) body.bbox = values.map(Number);
          }
          if (startInput && startInput.value) body.start_date = startInput.value;
          if (endInput && endInput.value) body.end_date = endInput.value;
          for (const control of controls) {
            const value = control.get();
            if (value === undefined || value === "" || value === null) continue;
            if (Array.isArray(value) && value.length === 0) continue;
            body.filters[control.spec.name] = value;
          }
          return body;
        },
      };
      currentForm = form;
      return form;
    }

    function buildFilter(spec) {
      const name = spec.name;
      if (spec.kind === "select") {
        const select = el("select", { "aria-label": spec.label });
        select.append(el("option", { value: "", text: "Any" }));
        for (const choice of spec.choices || []) {
          select.append(el("option", { value: choice.value, text: choice.label }));
        }
        if (spec.default != null) select.value = String(spec.default);
        return { spec, node: select, get: () => select.value, set: (value) => (select.value = value) };
      }
      if (spec.kind === "multiselect") {
        if (!spec.choices?.length) {
          const input = el("input", {
            type: "text",
            placeholder: "One or more ids, comma-separated",
            value: Array.isArray(spec.default) ? spec.default.join(", ") : "",
            "aria-label": spec.label,
          });
          return {
            spec,
            node: input,
            get: () =>
              input.value
                .split(/[,\s]+/)
                .map((value) => value.trim())
                .filter(Boolean),
            set: (values) => {
              input.value = Array.isArray(values) ? values.join(", ") : String(values || "");
            },
          };
        }
        const select = el("select", {
          multiple: "multiple",
          size: "4",
          "aria-label": spec.label,
        });
        for (const choice of spec.choices || []) {
          select.append(el("option", { value: choice.value, text: choice.label }));
        }
        const defaults = Array.isArray(spec.default) ? spec.default : [];
        for (const option of select.options) option.selected = defaults.includes(option.value);
        return {
          spec,
          node: select,
          get: () => Array.from(select.selectedOptions).map((option) => option.value),
          set: (values) => {
            const wanted = new Set(values || []);
            for (const option of select.options) option.selected = wanted.has(option.value);
          },
        };
      }
      if (spec.kind === "number") {
        const input = el("input", {
          type: "number",
          step: "any",
          value: spec.default != null ? String(spec.default) : "",
          "aria-label": spec.label,
        });
        return {
          spec,
          node: input,
          get: () => (input.value === "" ? undefined : Number(input.value)),
          set: (value) => {
            input.value = value == null ? "" : String(value);
          },
        };
      }
      if (spec.kind === "date") {
        const input = el("input", {
          type: "date",
          value: spec.default || "",
          "aria-label": spec.label,
        });
        return {
          spec,
          node: input,
          get: () => input.value,
          set: (value) => {
            input.value = value == null ? "" : String(value);
          },
        };
      }
      if (spec.kind === "boolean") {
        const input = el("input", { type: "checkbox", "aria-label": spec.label });
        input.checked = Boolean(spec.default);
        return {
          spec,
          node: input,
          get: () => input.checked,
          set: (value) => {
            input.checked = Boolean(value);
          },
        };
      }
      if (spec.kind === "daterange") {
        const start = el("input", { type: "date", "aria-label": spec.label + " start" });
        const end = el("input", { type: "date", "aria-label": spec.label + " end" });
        const row = el("div", { class: "data-sources-dates" }, [start, end]);
        return {
          spec,
          node: row,
          get: () => [start.value, end.value].filter(Boolean),
          set: (values) => {
            start.value = Array.isArray(values) ? values[0] || "" : "";
            end.value = Array.isArray(values) ? values[1] || "" : "";
          },
        };
      }
      if (spec.kind === "bbox") {
        const input = el("input", {
          type: "text",
          placeholder: "west,south,east,north",
          "aria-label": spec.label,
        });
        return {
          spec,
          node: input,
          get: () => {
            if (!input.value.trim()) return undefined;
            return input.value.split(",").map((value) => Number(value.trim()));
          },
          set: (values) => {
            input.value = Array.isArray(values) ? values.join(",") : String(values || "");
          },
        };
      }
      const input = el("input", {
        type: "text",
        value: spec.default || "",
        "aria-label": spec.label,
      });
      return {
        spec,
        node: input,
        get: () => input.value.trim(),
        set: (value) => {
          input.value = Array.isArray(value) ? value.join(", ") : String(value || "");
        },
      };
    }

    function field(label, control, hint) {
      const wrapper = el("label", { class: "data-sources-field" });
      wrapper.append(el("span", { class: "data-sources-field-label", text: label }), control);
      if (hint) wrapper.append(el("span", { class: "data-sources-field-hint", text: hint }));
      return wrapper;
    }

    async function runSearch(source) {
      if (busy || !currentForm) return;
      let body;
      try {
        body = currentForm.collect();
      } catch (error) {
        setStatus(error.message || String(error), "error");
        return;
      }
      busy = true;
      setFormBusy(true);
      setStatus("Searching " + source.name + "…");
      try {
        const page = await searchDataSource(source.id, body);
        results = page;
        lastQuery = body;
        const drilled =
          Array.isArray(body.filters.collections) && body.filters.collections.length
            ? "collections"
            : body.filters.event_id
              ? "event_id"
              : null;
        drilldown = drilled ? { filter: drilled, value: body.filters[drilled] } : null;
        drawResults(source, page);
        scrollResultsIntoView();
        setStatus(resultLine(page), "");
      } catch (error) {
        setStatus(error.message || String(error), "error");
        if (resultsNode) resultsNode.replaceChildren(emptyResults("Search failed."));
      } finally {
        busy = false;
        setFormBusy(false);
      }
    }

    function resultLine(page) {
      const count = page.datasets?.length || 0;
      if (page.total != null) return count + " shown • " + page.total + " total";
      return count + " result" + (count === 1 ? "" : "s");
    }

    function emptyResults(message) {
      return el("p", { class: "data-sources-empty", text: message });
    }

    function scrollResultsIntoView() {
      if (!resultsNode) return;
      const delta =
        resultsNode.getBoundingClientRect().top -
        main.getBoundingClientRect().top -
        8;
      main.scrollTop = Math.max(0, main.scrollTop + delta);
    }

    function drawResults(source, page) {
      if (!resultsNode) return;
      resultsNode.replaceChildren();
      if (drilldown) {
        const value = Array.isArray(drilldown.value)
          ? drilldown.value.join(", ")
          : drilldown.value;
        const bar = el("div", { class: "data-sources-drilldown" });
        bar.append(
          el("span", {
            class: "data-sources-drilldown-label",
            text:
              drilldown.filter === "event_id"
                ? "Event " + value
                : "Collection " + value,
          }),
          el("button", {
            type: "button",
            class: "data-sources-action",
            text: "Back",
            onclick: () => clearDrilldown(source),
          })
        );
        resultsNode.append(bar);
      }
      const datasets = page.datasets || [];
      if (!datasets.length) {
        resultsNode.append(emptyResults("No datasets matched."));
        return;
      }
      const list = el("div", { class: "data-sources-result-list" });
      for (const dataset of datasets) list.append(resultCard(source, dataset));
      resultsNode.append(list);
      if (page.next_cursor) {
        const more = el("button", {
          type: "button",
          class: "data-sources-more",
          text: "Load more",
          onclick: () => loadMore(source),
        });
        resultsNode.append(more);
      }
    }

    function clearDrilldown(source) {
      if (!drilldown) return;
      const control = controlsByName.get(drilldown.filter);
      if (control && control.set) {
        control.set(drilldown.filter === "collections" ? [] : "");
      }
      drilldown = null;
      runSearch(source);
    }

    function resultCard(source, dataset) {
      const card = el("article", { class: "data-sources-card" });
      if (dataset.thumbnail_url) {
        const image = el("img", {
          class: "data-sources-thumb",
          src: dataset.thumbnail_url,
          alt: "",
          loading: "lazy",
        });
        image.addEventListener("error", () => image.remove());
        card.append(image);
      } else {
        card.append(
          el("span", { class: "data-sources-thumb placeholder" }, [
            icon(dataset.kind === "collection" ? "database" : "layers", { size: 18 }),
          ])
        );
      }
      const body = el("div", { class: "data-sources-card-body" });
      const title = el("div", { class: "data-sources-card-title" });
      title.append(el("span", { text: dataset.title }));
      title.append(badge(KIND_LABELS[dataset.kind] || dataset.kind));
      body.append(title);
      const meta = [
        dataset.datetime ? String(dataset.datetime).slice(0, 10) : "",
        dataset.attribution || "",
        dataset.license || "",
      ].filter(Boolean);
      if (meta.length) {
        body.append(el("p", { class: "data-sources-card-meta", text: meta.join(" • ") }));
      }
      if (dataset.description) {
        body.append(
          el("p", {
            class: "data-sources-card-description",
            text: dataset.description.slice(0, 260),
          })
        );
      }
      const actions = el("div", { class: "data-sources-card-actions" });
      if (dataset.kind === "collection") {
        const collectionId =
          dataset.properties?.collection || dataset.properties?.event_id;
        const filterName = dataset.properties?.collection ? "collections" : "event_id";
        if (collectionId) {
          actions.append(
            el("button", {
              type: "button",
              class: "data-sources-action primary",
              text: "Open",
              onclick: () => openCollection(source, collectionId, filterName),
            })
          );
        }
      } else {
        const canAdd = conditions(source).has("map") && workspaceAvailable();
        const addButton = el("button", {
          type: "button",
          class: "data-sources-action primary",
          text: "Add to map",
          disabled: canAdd ? null : "disabled",
          title: canAdd ? "" : "Open a workspace to add layers.",
          onclick: () => addDataset(source, dataset, addButton),
        });
        actions.append(addButton);
        if (conditions(source).has("download") && dataset.kind !== "tiles") {
          const downloadButton = el("button", {
            type: "button",
            class: "data-sources-action",
            text: "Save to workspace",
            disabled: workspaceAvailable() ? null : "disabled",
            onclick: () => downloadDataset(source, dataset, downloadButton),
          });
          actions.append(downloadButton);
        }
      }
      body.append(actions);
      card.append(body);
      return card;
    }

    function openCollection(source, collectionId, filterName) {
      const control = controlsByName.get(filterName);
      if (!control || !control.set) {
        setStatus("This source does not expose a filter for " + filterName + ".", "error");
        return;
      }
      control.set(filterName === "collections" ? [collectionId] : collectionId);
      runSearch(source);
    }

    async function addDataset(source, dataset, button) {
      if (busy) return;
      busy = true;
      const previous = button.textContent;
      button.disabled = true;
      button.textContent = "Adding…";
      setStatus("Adding " + dataset.title + "…");
      try {
        const response = await addDataSourceDataset(source.id, dataset.id, {
          download: true,
        });
        const count = response.layers?.length || 0;
        setStatus(
          "Added " + count + " layer" + (count === 1 ? "" : "s") + " to the map.",
          "success"
        );
        toast("Added to the map", "success");
      } catch (error) {
        setStatus(error.message || String(error), "error");
        toast(error.message || String(error), "error");
      } finally {
        busy = false;
        button.disabled = false;
        button.textContent = previous;
      }
    }

    async function downloadDataset(source, dataset, button) {
      if (busy) return;
      busy = true;
      const previous = button.textContent;
      button.disabled = true;
      button.textContent = "Saving…";
      setStatus("Saving " + dataset.title + " to the workspace…");
      try {
        const response = await downloadDataSourceDataset(source.id, dataset.id, {});
        const files = response.files || [];
        setStatus(
          "Saved " + (files[0]?.path || "the dataset") + ".",
          "success"
        );
        toast("Saved to workspace", "success");
      } catch (error) {
        setStatus(error.message || String(error), "error");
        toast(error.message || String(error), "error");
      } finally {
        busy = false;
        button.disabled = false;
        button.textContent = previous;
      }
    }

    async function loadMore(source) {
      if (busy || !lastQuery || !results?.next_cursor) return;
      busy = true;
      setStatus("Loading more…");
      try {
        const body = { ...lastQuery, cursor: results.next_cursor };
        const page = await searchDataSource(source.id, body);
        results = {
          ...page,
          datasets: [...(results.datasets || []), ...(page.datasets || [])],
        };
        drawResults(source, results);
        setStatus(resultLine(results), "");
      } catch (error) {
        setStatus(error.message || String(error), "error");
      } finally {
        busy = false;
      }
    }

    function badge(label, title) {
      const node = el("span", { class: "data-sources-badge", text: label });
      if (title) node.setAttribute("title", title);
      return node;
    }

  });
}
