# Ignore this, I'm using this to build using poetry. 
# I will remove it when we submit.
include ../build_tools/poetry.mk

test:
	pytest tests/ -vv

run:
	python src/main.py

venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install $(shell cat requirements.txt | tr '\n' ' ')

clean:
	rm -rf venv
