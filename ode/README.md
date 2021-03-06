This file contains notes on how to build ODE for use in our physics simulator.

First, check out the version of ODE that corresponds to the patch revision we
have:

    svn checkout -r 1939 https://svn.code.sf.net/p/opende/code/trunk opende

Then, apply the patch to your checked-out copy of ODE:

    cd opende
    patch -p0 < ../ode-r1939.patch

Generate the configuration scripts, configure, make, and make install:

    ./bootstrap
    ./configure --enable-double-precision --enable-shared
    make -j4
    make install

(Assuming you have permissions to install to /usr/local. If not, either sudo
make install on that last line, or add --prefix /path/to/my/workspace to the
configure invocation.)

Finally, build and install the Python bindings:

    cd bindings/python
    python setup.py install

