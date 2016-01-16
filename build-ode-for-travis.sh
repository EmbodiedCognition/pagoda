#!/bin/bash

cd ode
svn checkout -r 1939 https://svn.code.sf.net/p/opende/code/trunk opende

cd opende
patch -p0 < ../ode-r1939.patch
./bootstrap
./configure --enable-double-precision --enable-shared --prefix=$HOME/ode
make -j
make install

cd bindings/python
LD_LIBRARY_PATH=$HOME/ode/lib:$LD_LIBRARY_PATH \
    python setup.py build_ext -L$HOME/ode/lib -I$HOME/ode/include
python setup.py install
