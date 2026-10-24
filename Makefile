.PHONY: install train app test lint check

install:
	pip install -r requirements.txt -r requirements-dev.txt

train:           ## train on the file named in config.yaml
	python -m churn.train --config config.yaml

app:             ## open http://127.0.0.1:8000
	python -m app

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

check: lint test
