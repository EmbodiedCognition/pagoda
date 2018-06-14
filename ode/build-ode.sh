#!/bin/bash

if [[ -z "$VIRTUAL_ENV" ]]
then
    echo 'this script can only be run inside a virtualenv!'
    exit 1
fi

curl -L https://bitbucket.org/odedevs/ode/downloads/ode-0.14.tar.gz | tar xz

(
    cd ode-0.14
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
