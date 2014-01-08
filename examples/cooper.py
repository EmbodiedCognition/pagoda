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
import lmj.sim
import lmj.sim.cooper
import os
import sys


def main():
    w = lmj.sim.cooper.World(dt=1. / 120)
    w.load_skeleton(os.path.join(os.path.dirname(__file__), 'cooper-skeleton.txt'))
    w.load_markers(os.path.join(os.path.dirname(__file__), 'cooper-motion.c3d'),
                   os.path.join(os.path.dirname(__file__), 'cooper-markers.txt'))
    lmj.sim.viewer.Physics(w).run()


if __name__ == '__main__':
    lmj.cli.call(main)
