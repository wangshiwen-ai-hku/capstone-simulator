.PHONY: test backend frontend-build

test:
	python3 -m pytest -q

backend:
	python3 -m uvicorn backend.app.main:app --reload --port 8000

frontend-build:
	cd frontend && npm run build
