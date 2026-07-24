.PHONY: run test lint format clean load-seed save-seed

# Run the Streamlit dashboard application
run:
	streamlit run app/streamlit_app.py

# Run the test suite
test:
	pytest

# Run code formatters and linters check
lint:
	ruff check .
	black --check .
	isort --check-only --profile black .

# Auto-format codebase
format:
	black .
	isort --profile black .
	ruff check --fix .

# Clean temporary Python, testing, and IDE cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +

# Load pre-populated seed data
load-seed:
	python scripts/manage_seed.py load

# Save current databases as seed data
save-seed:
	python scripts/manage_seed.py save
