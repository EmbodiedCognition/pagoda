#!/bin/bash

cd ode
svn checkout -r 1939 https://svn.code.sf.net/p/opende/code/trunk opende >/dev/null 2>&1

cd opende
patch -p0 < ../ode-r1939.patch
./bootstrap >/dev/null 2>&1
./configure --enable-double-precision --enable-shared --prefix=$HOME/ode >/dev/null 2>&1
make -j >/dev/null 2>&1
make install

export PKG_CONFIG_PATH=$HOME/ode/lib/pkgconfig:$HOME/ode/lib:$PKG_CONFIG_PATH

cd bindings/python
python setup.py build_ext -L$HOME/ode/lib -I$HOME/ode/include
python setup.py install
