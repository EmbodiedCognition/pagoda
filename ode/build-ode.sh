#!/bin/bash

if [[ -z "$VIRTUAL_ENV" ]]
then
    echo 'this script can only be run inside a virtualenv!'
    exit 1
fi

svn checkout -r 1939 https://svn.code.sf.net/p/opende/code/trunk opende
patch -dopende -p0 < ode-r1939.patch

(
    cd opende
    ./bootstrap
    ./configure --enable-double-precision --enable-shared --prefix=$VIRTUAL_ENV
    make -j
    make install
)

if [[ -z "$(which cython)" ]]
then pip install cython
fi

(
    export PKG_CONFIG_PATH=$VIRTUAL_ENV/lib/pkgconfig:$PKG_CONFIG_PATH
    cd opende/bindings/python
    python setup.py install
)
