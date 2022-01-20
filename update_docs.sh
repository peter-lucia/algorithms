#!/bin/bash
jupyter nbconvert --execute --to html *.ipynb --output-dir=docs
