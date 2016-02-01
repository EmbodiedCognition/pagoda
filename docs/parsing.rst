This parser class reads a text configuration of a pagoda simulation.

The format for the configuration file follows basic Unix conventions. A
preprocessor discards any text on a line following the pound (#) character.
Blank lines are skipped. The remainder of the file is chunked into sections
based on keywords; each keyword in the file indicates the start of a new
section. The recognized keywords are:

- body -- indicates the definition of a new rigid body;
- join -- indicates that two bodies will be joined together.

Bodies
======

A body specification must contain the following information:

- An identifier for the body. This cannot contain spaces.

- A shape specification. This must be one of box (rectangular prism), sph or
  sphere (sphere), cyl or cylinder (cylinder), or cap or capsule (capped
  cylinder).

  The shape indicates the canonical geometry that will be used to (a) specify
  the size of the body, and (b) detect collisions.

- One or more dimension parameters. Each of these consists of a shape keyword
  followed by one or more sizes (in meters). Different dimension keywords are
  required based on the shape of the body:

  - box: X Y Z
  - sph: R@
  - cyl: L R@ or R@ L
  - cap: L R@ or R@ L

Additionally, a body definition may be followed by extra quantities specifying
the position, orientation, and other quantities of the body.

- position X Y Z -- specifies the absolute location of the body in world
  coordinates (in meters). Typically this is only given for one object. By
  default bodies are all created at the origin.

- quaternion W X Y Z -- specifies the angle (in degrees) and axis of rotation
  for the body. By default bodies are created with a 0 1 0 0 quaternion.

- density -- specifies the density of the body (kg/m^3).

- mass -- specifies the total mass of the body (kg)

- root -- indicates that this body is the root in the skeleton.

Joints
======

A joint specification must contain the following information:

- A shape specification. This must be one of fixed, slider, hinge, piston,
  universal, or ball. The corresponding ODE joint type will be used to join
  together two bodies.

- Two attachment specifications. These specify the bodies to be joined, as well
  as the body-relative offsets (on each body) where the joint will be anchored.

Additionally, joints may specify information about their axes, rotation limits,
etc.

- axes <X Y Z> [<X Y Z> [<X Y Z>]] -- specifies the axes for the joint. By
  default, the axes are <1 0 0> <0 1 0> <0 0 1>.

- stops A..A' [B..B' [C..C']] -- specifies the range of permitted rotations (in
  degrees) for this joint along its respective axes.

- stop_cfms N [N [N]] -- set the stop CFMs for DOFs on this joint.

- stop_erps N [N [N]] -- set the stop ERPs for DOFs on this joint.

Example
=======

A simple example is probably sufficient to get started; also see the
``examples/cooper-skeleton.txt`` file distributed with the pagoda source for a
more complete example.

.. code-block:: shell

  foo box 0.5 0.3 0.2  position 0 0 2
  bar sph 0.1@
  baz cap 0.1@ 0.3     root

This section of the example causes three bodies to be created:

- A box identified as "foo" that is 50cm x 30cm x 20cm along the X, Y, and Z
  axes, respectively. foo will be positioned at world coordinates (0, 0, 2).

- A sphere identified as "bar" that is 10cm in radius.

- A capped cylinder identified as "baz" that is 10cm in radius and 30cm long
  (the total length of this geometry is actually 50cm due to the hemispherical
  cylinder end caps). This body is the root of the skeleton.

.. code-block:: shell

  hinge foo(1 0 0) <- bar(0 -1 0) stops -10..20
  ball bar <> baz(0 0 1)

This section joins the three bodies together:

- The center of the positive-X side of foo-body is joined to the point on the
  face of the bar sphere with the smallest Y coordinate, using a hinge joint (1
  DOF). This joint is constrained to rotate between -10 and +20 degrees.

- The center of bar (offset (0 0 0) is used by default) is joined to the point
  on baz with the largest Z coordinate using an unconstrained ball joint.
