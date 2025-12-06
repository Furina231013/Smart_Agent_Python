.PHONY: install dev-install test format lint clean run demo

# Installation
install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -e .

# Development
test:
	pytest tests/ -v

format:
	black src/ tests/

lint:
	mypy src/
	flake8 src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/

# Running
run:
	python -m src.main

demo:
	python -m src.main

# Docker (optional)
docker-build:
	docker build -t python-llm-mcp-rag .

docker-run:
	docker run --rm -it --env-file .env python-llm-mcp-rag

# Development helpers
install-pre-commit:
	pre-commit install

check-all: format lint test