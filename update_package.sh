#!/bin/bash
rm -r docs/
pipenv run pip install .
pipenv run pdoc --html rlmodels/
mv html/rlmodels/ docs/ && rm -r html/