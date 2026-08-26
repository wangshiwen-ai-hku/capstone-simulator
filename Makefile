.PHONY: test lint proto backend frontend-build

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check agent backend evals mars scripts tests
	python3 -m compileall -q agent backend evals interfaces mars scripts tests

proto:
	python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. interfaces/proto/mars/v1/*.proto

backend:
	python3 -m uvicorn backend.app.main:app --reload --port 8000

frontend-build:
	cd frontend && npm run build
