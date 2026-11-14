# Spatial Intelligence — development and release targets.

VENV   := .venv
DIST   := dist
PYTHON ?= python3
UV     ?= uv

ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
PY  := $(BIN)/python.exe
else
BIN := $(VENV)/bin
PY  := $(BIN)/python
endif

# The repo root must be importable even before `make venv` has installed the
# package, and it should win over an installed copy while editing.
export PYTHONPATH := .

.PHONY: help venv dev run test lint check release appimage clean

help:
	@echo "Targets:"
	@echo "  make venv         create $(VENV) if missing, then install the package (editable)"
	@echo "  make dev          run the app (python -m spatial_intelligence.server)"
	@echo "  make run          alias for make dev"
	@echo "  make test         run the pytest suite (what CI runs)"
	@echo "  make lint         run ruff over the repo"
	@echo "  make check        alias for make lint test"
	@echo "  make release      build the AppImage into $(DIST)/ (requires docker)"
	@echo "  make appimage     alias for make release"
	@echo "  make clean        remove $(VENV), caches, and build artifacts"

# Installs prefer uv: this checkout's .venv is uv-managed and contains no pip.
# A plain `python -m venv` falls back to pip. The `dev` dependency group brings
# pytest and ruff, which `make test` and `make lint` need.
venv:
	@if [ ! -x "$(PY)" ]; then \
		if command -v $(UV) >/dev/null 2>&1; then \
			echo "Creating virtualenv with uv"; \
			$(UV) venv $(VENV); \
		else \
			echo "Creating virtualenv with $(PYTHON)"; \
			$(PYTHON) -m venv $(VENV); \
		fi; \
	fi
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) pip install -e . --group dev; \
	else \
		"$(PY)" -m pip install --upgrade pip && "$(PY)" -m pip install -e . pytest ruff; \
	fi

# The interpreter doubles as the "environment is ready" marker, so `make dev`
# sets up once and never reinstalls on later runs.
$(PY):
	@$(MAKE) --no-print-directory venv

dev: $(PY)
	"$(PY)" -m spatial_intelligence.server

run: dev

# pytest is the CI runner; the suite itself is unittest-based, so this is the
# same set of tests `python -m unittest discover` collects.
test: $(PY)
	"$(PY)" -m pytest -q

lint: $(PY)
	"$(PY)" -m ruff check .

check: lint test

# Linux AppImage of this app (see packaging/README.md). Requires docker.
release:
	./packaging/build-appimage.sh $(DIST)

appimage: release

clean:
	rm -rf $(VENV) $(DIST) build build-pyi
	-find spatial_intelligence tests -name __pycache__ -type d -prune -exec rm -rf {} +
