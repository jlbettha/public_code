#!/bin/bash

pre-commit run --all-files;
# ruff check --fix *.py;
# pycodestyle find . -name "utils/*.py";

find ./ -type f -wholename "./data_compression/*.py" -exec pycodestyle "{}" \;
find ./ -type f -wholename "./optical_flow/*.py" -exec pycodestyle "{}" \;