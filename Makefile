.PHONY: test backend frontend-build

test:
	python3 -m unittest discover -s tests -p 'test_*.py' -v

backend:
	python3 -m uvicorn backend.app.main:app --reload --port 8000

frontend-build:
	cd frontend && npm run build
