PHONY: docs

documentation:
	@rm -r docs/;\
  pipenv run pdoc3 --html rlmodels/;\
  mv html/rlmodels/ docs/ && rm -r html/


pypi-release:
	rm -r dist/
	pipenv run python setup.py sdist
	pipenv run twine check dist/* && pipenv run twine upload dist/*