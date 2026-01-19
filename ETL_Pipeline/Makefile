.PHONY: help venv deps docker up down etl train predict shap export streamlit

help:
	@echo "Makefile targets: venv deps docker up down etl train predict shap export streamlit"

venv:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip

deps: venv
	.venv/bin/pip install -r requirements.txt

docker:
	docker-compose build --pull

up: docker
	docker-compose up -d

down:
	docker-compose down

etl:
	.venv/bin/python src/etl.py --run --out data/processed/processed.csv

train:
	.venv/bin/python src/models/train.py --input data/processed/processed.csv

predict:
	.venv/bin/python src/models/predict.py --input data/raw/new_customers.csv --out predictions.csv

shap:
	.venv/bin/python notebooks/eda_shap.py

export:
	.venv/bin/python src/export_powerbi.py


streamlit:
	# Installe streamlit si nécessaire: .venv/bin/pip install streamlit
	.venv/bin/streamlit run app.py --server.port 8501