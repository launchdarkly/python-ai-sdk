.PHONY: start test typecheck lint format format-check lint-fix build clean

# Run main.py (uv run python main.py)
start:
	uv run python main.py

# Run all tests (uv run pytest)
test:
	uv run pytest

# Type-check all packages (uv run mypy)
# Not bare `mypy .`: the per-package `tests/conftest.py` files all resolve to the module name
# `conftest`, and mypy aborts on that clash (`Duplicate module named "conftest"`) before checking
# anything, so the command gives no signal at all. CI type-checks package sources only — both
# `.github/workflows/ci.yml` and `.github/actions/ci/action.yml` — and this mirrors that exactly so
# local and CI agree. The test suites are not currently mypy-clean under `strict`.
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
