clean:
	rm -rf build/ dist/ *.egg-info/

build:
	python -m build

publish:
	make clean
	make build
	twine upload dist/*
