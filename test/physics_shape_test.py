from __future__ import division

import numpy as np
import pagoda


def test_sphere(world):
    b = pagoda.physics.Sphere('sph', world, radius=3)
    assert b.radius == 3
    assert b.volume == 4/3 * np.pi * 27
    assert tuple(b.dimensions) == (6, 6, 6)


def test_cap(world):
    b = pagoda.physics.Capsule('cap', world, radius=3, length=2)
    assert b.radius == 3
    assert b.length == 2
    assert b.volume == np.pi * (9 * 2 + 4/3 * 27)
    assert tuple(b.dimensions) == (6, 6, 8), b.dimensions


def test_box(world):
    b = pagoda.physics.Box('box', world, lengths=(3, 4, 5))
    assert b.lengths == (3, 4, 5)
    assert b.volume == 3 * 4 * 5
    assert tuple(b.dimensions) == (3, 4, 5)


def test_cylinder(world):
    b = pagoda.physics.Cylinder('cyl', world, radius=3, length=2)
    assert b.radius == 3
    assert b.length == 2
    assert b.volume == np.pi * 9 * 2
    assert tuple(b.dimensions) == (6, 6, 2)
