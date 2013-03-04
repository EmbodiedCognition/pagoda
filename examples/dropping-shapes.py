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

import lmj.sim
import numpy as np
import numpy.random as rng
import ode
import sys


class World(lmj.sim.World):
    def reset(self):
        for b in self.bodies:
            b.position = np.array([0, 0, 10]) + 3 * rng.randn(3)
            b.quaternion = self.make_quaternion(np.pi * rng.rand(), 0, 1, 1)


@lmj.sim.args(
    n=('number of bodies in the simulation', 'option', None, int),
    frame_rate=('frame rate of the simulation', 'option', None, float),
    friction=('coefficient of friction', 'option', None, float),
    elasticity=('elasticity constant for collisions', 'option', None, float),
    )
def main(n=10, frame_rate=60., friction=5000, elasticity=0.1):
    w = World(dt=1. / frame_rate, friction=friction, elasticity=elasticity)
    for _ in range(n):
        s, kw = sorted(dict(
            box=dict(lengths=rng.gamma(3, 0.1, size=3)),
            capsule=dict(radius=rng.gamma(3, 0.1), length=rng.gamma(7, 0.1)),
            cylinder=dict(radius=rng.gamma(2, 0.1), length=rng.gamma(7, 0.1)),
            sphere=dict(radius=rng.gamma(2, 0.1)),
            ).iteritems())[rng.randint(4)]
        w.create_body(s, **kw)
    w.reset()
    lmj.sim.Viewer(w).show()


if __name__ == '__main__':
    lmj.sim.call(main)
