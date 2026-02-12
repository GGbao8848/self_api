.PHONY: run test lint

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8666 --reload

test:
	pytest

lint:
	python -m compileall app tests
