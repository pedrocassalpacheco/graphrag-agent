.PHONY: setup lint format test run clean

PROJECT_NAME = graphrag_agent
VENV_NAME = .venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
UV = $(shell command -v uv 2> /dev/null)
SRC_DIR = $(shell pwd)/src



setup:
	@echo "Using uv for virtual environment creation and dependency management..."
	uv venv $(VENV_NAME)
	uv sync

lint:
	@echo "Running linters..."
	$(PYTHON) -m flake8 ./agent ./tests
	$(PYTHON) -m black --check ./agent ./tests

format:
	@echo "Formatting code..."
	$(PYTHON) -m black ./agent ./tests

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest -v ./tests

run:
	@echo "Running the AI agent..."
	$(UV) venv
	$(UV) sync        
	PYTHONPATH=$(SRC_DIR) $(PYTHON) src/graphrag_agent/main.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME) __pycache__ *.pyc

.PHONY: clean
