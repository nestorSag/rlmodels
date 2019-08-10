#!/bin/bash
rm -r docs/
pipenv run pdoc --html rlmodels/
mv html/rlmodels/ docs/ && rm -r html/
