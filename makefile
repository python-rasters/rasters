.PHONY: clean test build twine-upload dist environment install uninstall reinstall test

clean:
	rm -rf *.o *.out *.log
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	pytest

build:
	python -m build

twine-upload:
	twine upload dist/*

dist:
	make clean
	make build
	make twine-upload

environment:
	mamba create -y -n rasters python=3.11 jupyter

remove-environment:
	mamba env remove -y -n rasters

install:
	pip install -e .[dev]

uninstall:
	pip uninstall -y rasters

reinstall:
	make uninstall
	make install
