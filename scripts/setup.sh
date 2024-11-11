#!/bin/bash

# setup python env
python3.11 -m venv venv

# activate python env
source venv/bin/activate

# install dependencies
pip install -r requirements.txt