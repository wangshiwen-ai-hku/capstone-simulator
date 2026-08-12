.PHONY: test lint backend frontend-build

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check backend evals mars scripts tests
	python3 -m compileall -q backend evals mars scripts tests

backend:
	python3 -m uvicorn backend.app.main:app --reload --port 8000

frontend-build:
	cd frontend && npm run build
