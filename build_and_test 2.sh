#!/usr/bin/env bash
source bin/activate
pip install -e .
pytest tests/ -k "not slow"