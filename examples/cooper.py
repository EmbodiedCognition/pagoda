#!/usr/bin/env python

# Copyright (c) 2013 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import lmj.cli
import lmj.sim as ls
import os
import sys


@lmj.cli.annotate(
    frame_rate=('frame rate of the simulation', 'option', None, float),
    friction=('coefficient of friction', 'option', None, float),
    elasticity=('elasticity constant for collisions', 'option', None, float),
    )
def main(frame_rate=60., friction=5000, elasticity=0.1):
    w = ls.cooper.World(dt=1. / frame_rate, friction=friction, elasticity=elasticity)
    w.create_from_file(os.path.join(os.path.dirname(__file__), 'cooper.txt'))
    ls.viewer.GL(w, paused=True).run()


if __name__ == '__main__':
    lmj.cli.call(main)
