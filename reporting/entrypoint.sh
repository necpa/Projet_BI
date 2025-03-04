#!/bin/sh

python project.py
python -m http.server 8082 --directory /app/reporting

