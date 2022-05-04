# A makefile to simplify speatative steps

package:
	python setup.py bdist_wheel
	python setup.py sdist

pypi-upload:
	twine upload --verbose dist/*
