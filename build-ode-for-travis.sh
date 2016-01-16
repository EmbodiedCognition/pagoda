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
PKG_CONFIG_PATH=$HOME/ode/lib:$PKG_CONFIG_PATH \
    python setup.py build_ext -L$HOME/ode/lib -I$HOME/ode/include
python setup.py install
