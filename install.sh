#!/bin/bash
cd ./3rdparty/densecrf
mkdir build
cd build
cmake .. 
make

cd ../../../
mkdir build
cd build
cmake .. 
make 
make install

cd ../python/utils
python ./setup.py build_ext --inplace
cd ../../
