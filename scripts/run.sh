#!/bin/bash

# make sure that we are using the right python
# if not run the setup script
if [ ! -d "venv" ]; then
    ./scripts/setup.sh
fi

# run the app
python main.py
