#!/usr/bin/env python

import lmj.sim
import numpy as np
import numpy.random as rng
import ode
import sys


class World(lmj.sim.physics.World):
    def reset(self):
        for b in self.bodies:
            b.position = np.array([0, 20, 0]) + 3 * rng.randn(3)
            b.quaternion = lmj.sim.physics.make_quaternion(np.pi * rng.rand(), 0, 1, 0)

    def draw(self):
        for b in self.bodies:
            b.draw()


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
