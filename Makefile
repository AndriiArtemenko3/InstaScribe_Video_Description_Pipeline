# InstaScribe developer tasks. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help demo dev server install install-web install-py test lint

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

demo: install-web  ## Zero-key demo: build the committed-fixture web app and serve it (no API key, no backend)
	cd App && npm run demo

dev: install-web  ## Run the web app in dev mode against a local backend (start `make server` in another shell)
	cd App && npm run dev

server:  ## Run the Flask API + single-origin server on :8765
	python modular_pipeline/server.py

install: install-web install-py  ## Install both the web and Python dependencies

install-web:  ## Install frontend dependencies
	cd App && npm install

install-py:  ## Create .venv and install the full pipeline
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:  ## Run the Python and web test suites
	pytest -q
	cd App && npm test

lint:  ## Ruff (Python) + ESLint (web)
	ruff check .
	cd App && npm run lint
