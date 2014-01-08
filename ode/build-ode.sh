#!/bin/bash

svn checkout -r 1939 http://svn.code.sf.net/p/opende/code/trunk opende

(
    cd opende
    patch -p0 < ../ode-r1939.patch
    ./bootstrap
    ./configure --enable-double-precision --enable-shared
    make -j4
    make install
)

(
    cd opende/bindings/python
    python setup.py install
)
