.PHONY: install data train eval serve test lint clean docker

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

train:
	python scripts/train.py --data data/creditcard.csv --out artifacts/model.pkl

eval:
	python scripts/eval.py --data data/creditcard.csv --model artifacts/model.pkl

serve:
	MODEL_PATH=artifacts/model.pkl uvicorn src.score_api:app --port 8000 --reload

test:
	pytest

lint:
	ruff check src/ tests/ scripts/

docker:
	docker build -t fraud-detection:latest .

clean:
	rm -rf artifacts/*.pkl __pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete
