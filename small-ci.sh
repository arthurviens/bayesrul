#!/bin/bash

echo "Testing local build"
echo "--- Setup Python"
source /opt/intelpython3/etc/profile.d/conda.sh
conda create -n smallci python=3.8 -y
conda activate smallci 

echo "--- Install dependencies"
python -m pip install --upgrade pip
python -m pip install poetry pytest
python -m poetry install

echo "Tests with pytest"
python -m poetry run python -m pytest -v tests


echo "Cleaning"
conda deactivate 
conda env remove -n smallci