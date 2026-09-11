.PHONY: start test typecheck lint format format-check lint-fix build clean

# Run main.py (uv run python main.py)
start:
	uv run python main.py

# Run all tests (uv run pytest)
test:
	uv run pytest

# Type-check all package sources (matches CI: uv run mypy packages/*/src).
# Not `mypy .`: that walks every package's tests/conftest.py, which mypy sees as
# duplicate top-level `conftest` modules and aborts before checking anything.
typecheck:
	uv run mypy packages/*/src

# Check for lint errors
lint:
	uv run ruff check .

# Format all files
format:
	uv run ruff format .

# Check formatting without writing
format-check:
	uv run ruff format --check .

# Auto-fix lint errors and format
lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

# Build all packages
build:
	@for pkg in packages/*/; do \
		echo "Building $$pkg..."; \
		(cd $$pkg && uv build --out-dir dist); \
	done

# Clean build artifacts
clean:
	rm -rf packages/*/dist
