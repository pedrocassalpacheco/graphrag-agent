.PHONY: setup lint format test run clean test-coverage test-html clean-pyc clean-test test-all

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

test-file:
	@echo "Running single testtests..."
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -m pytest -v --pdb --log-cli-level=DEBUG ./src/tests/unit/$(FILE
	)

unit-test:
	@echo "Running tests..."
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -m pytest -v --pdb --log-cli-level=DEBUG ./src/tests/unit

integration-test:
	@echo "Running integration tests..."
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -m pytest -v --pdb --log-cli-level=DEBUG ./src/tests/integration

tests: unit-test integration-test
	@echo "All tests completed."

run:
	@echo "Running the AI agent..."
	$(UV) venv
	$(UV) sync        
	PYTHONPATH=$(SRC_DIR) $(PYTHON) src/graphrag_agent/main.py

debug:
	@echo "Starting the debugger..."
	PYTHONPATH=$(SRC_DIR) $(PYTHON) -m pdb src/graphrag_agent/main.py

PROFILE_OUTPUT=profile_results.prof

profile:
	PYTHONPATH=$(SRC_DIR) python -m cProfile -o $(PROFILE_OUTPUT) src/graphrag_agent/main.py

view-profile:
	PYTHONPATH=$(SRC_DIR) python -m pstats $(PROFILE_OUTPUT)

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME) __pycache__ *.pyc

# Run tests with coverage
test-coverage:
	python -m pytest --cov=src/graphrag_agent --cov-report=term-missing tests/

# Generate HTML coverage report and open it
test-html:
	python -m pytest --cov=src/graphrag_agent --cov-report=html:coverage_reports/html tests/
	open coverage_reports/html/index.html

# Clean Python cache files
clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +

# Clean test and coverage artifacts
clean-test:
	rm -rf .coverage
	rm -rf coverage_reports/
	rm -rf .pytest_cache/
	rm -rf tests/tools/test_output/

# Run all cleanup and tests
test-all: clean-pyc clean-test test-coverage
