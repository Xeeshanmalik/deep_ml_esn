#!/usr/bin/env bash

# Python Version
cd python_version

cd python
conda env update
source activate ML-ESN
python main.py ../../data/MackeyGlass_t17.txt
source deactivate

cd ../..
# Tensorflow Version
cd tensorflow_version
conda env update
source activate TENSLOWFLOW-MLESN
python main.py ../data/MackeyGlass_t17.txt
source deactivate
cd ..