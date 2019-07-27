rm -r dist/
pipenv run python setup.py sdist
pipenv run twine check dist/* && pipenv run twine upload dist/*