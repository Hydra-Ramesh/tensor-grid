.PHONY: test lint format up down build

# Testing
test:
	pytest tests/unit/ -v --asyncio-mode=auto

# Code Quality
lint:
	flake8 src tests
	mypy src tests --ignore-missing-imports
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

# Docker commands
up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build
