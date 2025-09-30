# Ignore this, I'm using this to build using poetry. 
# I will remove it when we submit.
include ../build_tools/poetry.mk

test:
	pytest tests/ -vv -s

run:
	python src/main.py

venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install $(shell cat requirements.txt | tr '\n' ' ')

clean:
	rm -rf venv

data/parking:
	mkdir -p data
	wget https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip -O data/parking.zip
	unzip data/parking.zip -d data
