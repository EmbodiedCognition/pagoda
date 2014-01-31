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

import climate
import lmj.sim
import lmj.sim.viewer
import numpy as np
import numpy.random as rng


class World(lmj.sim.physics.World):
    def reset(self):
        for b in self.bodies:
            b.position = np.array([0, 0, 10]) + 3 * rng.randn(3)
            b.quaternion = lmj.sim.physics.make_quaternion(
                np.pi * rng.rand(), 0, 1, 1)


@climate.annotate(
    n=('number of bodies in the simulation', 'option', None, int),
    )
def main(n=20):
    w = World()
    # set the cfm parameter below for a trampoline-like floor !
    #w.cfm = 1e-3
    g = lambda n, k=0.1, size=1: np.clip(rng.gamma(n, k, size=size), 0.5, 1000)
    for _ in range(n):
        s, kw = sorted(dict(
            box=dict(lengths=g(8, size=3)),
            capsule=dict(radius=g(3), length=g(10)),
            cylinder=dict(radius=g(2), length=g(10)),
            sphere=dict(radius=g(2)),
            ).iteritems())[rng.randint(4)]
        kw['color'] = tuple(rng.uniform(0, 1, size=3)) + (0.9, )
        w.create_body(s, **kw)
    w.reset()
    lmj.sim.viewer.Physics(w).run()


if __name__ == '__main__':
    climate.call(main)
