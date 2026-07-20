.PHONY: test backend frontend-build

test:
	PYTHONPATH=backend python3 -m unittest discover -s backend -p 'test_*.py' -v

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend-build:
	cd frontend && npm run build
