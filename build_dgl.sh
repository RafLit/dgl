#!/bin/bash

cd build
#rm -rf *
#cmake .. -DCMAKE_CXX_COMPILER=/home/pablo/intel/oneapi/compiler/latest/linux/bin/icpx
make -j24


cd ..
cd python
pip uninstall dgl
pip install -e .


cd ..

./run.sh

