==========
``PAGODA``
==========

The ``pagoda`` package contains Python glue that binds together the `Open
Dynamics Engine (ODE)`_, a physics simulator, with Pyglet_, a package for
relatively painless graphics. This package provides some additional classes that
make it even easier to construct, simulate, and visualize physical worlds.

.. _Open Dynamics Engine: http://ode.org
.. _Pyglet: http://pyglet.org

The package defines Pythonic wrappers for all of the Body and most of the Joint
types provided in ODE. It also defines a visualization base class that defines
several helper methods to make it easy to draw common geometric objects.

Example Code
============

Normally in ``pagoda`` you create a new world, add some bodies to the world, and
then run a visualization tool to see what happens::

  import climate
  import pagoda
  import pagoda.viewer
  import numpy as np
  import numpy.random as rng

  # define a new physics world with a custom "reset" method.
  class World(pagoda.physics.World):
      def reset(self):
          for b in self.bodies:
              b.position = np.array([0, 0, 10]) + 3 * rng.randn(3)
              b.quaternion = pagoda.physics.make_quaternion(
                  np.pi * rng.rand(), 0, 1, 1)

  # helper to create gamma-distributed random values.
  def gam(n, k=0.1, size=1):
      return np.clip(rng.gamma(n, k, size=size), 0.5, 1000)

  # create the simulation world.
  w = World()

  # add 20 random bodies to the world.
  for _ in range(20):
      s, kw = (
          ('box', dict(lengths=gam(8, size=3))),
          ('capsule', dict(radius=gam(3), length=gam(10))),
          ('cylinder', dict(radius=gam(2), length=gam(10))),
          ('sphere', dict(radius=gam(2))),
          )[rng.randint(4)]
      kw['color'] = tuple(rng.random(3)) + (0.9, )
      w.create_body(s, **kw)

  # reset the positions of all the bodies.
  w.reset()

  # run the pyglet visualizer!
  pagoda.viewer.Viewer(w).run()

At the moment most of the documentation lives in the API reference, but
hopefully much of the API is relatively simple to understand. Have fun!

Documentation
=============

.. toctree::
   :maxdepth: 2

   reference
