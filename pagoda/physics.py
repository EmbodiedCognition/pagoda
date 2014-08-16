# Copyright (c) 2013 Leif Johnson <leif@cs.utexas.edu>
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

'''Convenience wrappers for ODE objects.'''

from __future__ import division, print_function

import numpy as np
import ode

from . import base


class Body:
    '''This class wraps things that participate in the ODE physics simulation.

    The primary attribute of this class is "ode_body" -- a PyODE Body object. In
    addition, there is a PyODE Geom object (for detecting collisions -- not sure
    if this is really necessary to keep around though).

    This class also provides lots of Python-specific properties that call the
    equivalent ODE getters and setters for things like position, rotation, etc.
    '''

    def __init__(self, name, world, space, color=(0.3, 0.6, 0.9, 1), density=1000., **shape):
        self.name = name
        self.shape = shape
        self.color = color

        m = ode.Mass()
        self.init_mass(m, density)
        self.ode_body = ode.Body(world)
        self.ode_body.setMass(m)
        self.ode_geom = getattr(ode, 'Geom%s' % self.__class__.__name__)(space, **shape)
        self.ode_geom.setBody(self.ode_body)

    def __str__(self):
        return '{0.__class__.__name__} {0.name} at {1}'.format(
            self, np.array(self.position).round(3))

    @property
    def mass(self):
        return self.ode_body.getMass()

    @property
    def position(self):
        return self.ode_body.getPosition()

    @position.setter
    def position(self, position):
        self.ode_body.setPosition(position)

    @property
    def rotation(self):
        return self.ode_body.getRotation()

    @rotation.setter
    def rotation(self, rotation):
        self.ode_body.setRotation(rotation)

    @property
    def quaternion(self):
        return self.ode_body.getQuaternion()

    @quaternion.setter
    def quaternion(self, quaternion):
        self.ode_body.setQuaternion(quaternion)

    @property
    def linear_velocity(self):
        return self.ode_body.getLinearVel()

    @linear_velocity.setter
    def linear_velocity(self, velocity):
        self.ode_body.setLinearVel(velocity)

    @property
    def angular_velocity(self):
        return self.ode_body.getAngularVel()

    @angular_velocity.setter
    def angular_velocity(self, velocity):
        self.ode_body.setAngularVel(velocity)

    @property
    def force(self):
        return self.ode_body.getForce()

    @force.setter
    def force(self, force):
        self.ode_body.setForce(force)

    @property
    def torque(self):
        return self.ode_body.getTorque()

    @torque.setter
    def torque(self, torque):
        self.ode_body.setTorque(torque)

    @property
    def is_kinematic(self):
        return self.ode_body.isKinematic()

    @is_kinematic.setter
    def is_kinematic(self, is_kinematic):
        if is_kinematic:
            self.ode_body.setKinematic()
        else:
            self.ode_body.setDynamic()

    @property
    def follows_gravity(self):
        return self.ode_body.getGravityMode()

    @follows_gravity.setter
    def follows_gravity(self, follows_gravity):
        self.ode_body.setGravity(follows_gravity)

    def rotate_to_body(self, x):
        return np.dot(x, np.array(self.rotation).reshape((3, 3)))

    def body_to_world(self, position):
        return self.ode_body.getRelPointPos(position)

    def world_to_body(self, position):
        return self.ode_body.getPosRelPoint(position)

    def add_force(self, force, relative=False, position=None, relative_position=None):
        b = self.ode_body
        if relative_position is not None:
            op = b.addRelForceAtRelPos if relative else b.addForceAtRelPos
            op(force, relative_position)
        elif position is not None:
            op = b.addRelForceAtPos if relative else b.addForceAtPos
            op(force, position)
        else:
            op = b.addRelForce if relative else b.addForce
            op(force)

    def add_torque(self, torque, relative=False):
        op = self.ode_body.addRelTorque if relative else self.ode_body.addTorque
        op(torque)


class Box(Body):
    @property
    def lengths(self):
        return self.shape['lengths']

    @property
    def dimensions(self):
        return np.array(self.lengths).squeeze()

    def init_mass(self, m, density):
        m.setBox(density, *self.lengths)


class Sphere(Body):
    @property
    def radius(self):
        return self.shape['radius']

    @property
    def dimensions(self):
        d = 2 * self.radius
        return np.array([d, d, d]).squeeze()

    def init_mass(self, m, density):
        m.setSphere(density, self.radius)


class Cylinder(Body):
    @property
    def radius(self):
        return self.shape['radius']

    @property
    def length(self):
        return self.shape['length']

    @property
    def dimensions(self):
        d = self.radius
        return np.array([d, d, self.length]).squeeze()

    def init_mass(self, m, density):
        m.setCylinder(density, 3, self.radius, self.length)


class Capsule(Body):
    @property
    def radius(self):
        return self.shape['radius']

    @property
    def length(self):
        return self.shape['length']

    @property
    def dimensions(self):
        d = 2 * self.radius
        return np.array([d, d, d + self.length]).squeeze()

    def init_mass(self, m, density):
        m.setCapsule(density, 3, self.radius, self.length)


# Create a lookup table for things derived from the Body class.
BODIES = {}
for cls in Body.__subclasses__():
    name = cls.__name__.lower()
    for i in range(3, len(name) + 1):
        BODIES[name[:i]] = cls


def _get_params(target, param, dof):
    '''Get the given param from each of the DOFs for a joint.'''
    return [target.getParam(getattr(ode, 'Param{}{}'.format(param, s)))
            for s in ['', '2', '3'][:dof]]

def _set_params(target, param, values, dof):
    '''Set the given param for each of the DOFs for a joint.'''
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = [values] * dof
    assert dof == len(values)
    for s, value in zip(['', '2', '3'][:dof], values):
        target.setParam(getattr(ode, 'Param{}{}'.format(param, s)), value)


class Motor:
    '''This class wraps an ODE motor -- either an LMotor or an AMotor.

    The class has read-write properties for :

    - the axes of rotation or motion for the motor (`axes`),
    - the low and high stops of the motor's motion (`lo_stops` and `hi_stops`),
    - the max force that is allowed to be applied (`max_forces`),
    - the target value for the motor (`velocities`), and
    - the constraint force mixing parameter (`cfms`).

    There are also read-only properties for :

    - the feedback through the motor (`feedback`),
    - the number of degrees of freedom (`dof`), and
    - the current state of the motor (`angles`) as well as the derivative
      (`angle_rates`).

    All of these properties except `dof` return sequences of values. The setters
    can be applied using a scalar value, which will be applied to all degrees of
    freedom, or with a sequence whose length must match the number of DOFs in
    the motor.
    '''

    def __init__(self, name, world, body_a, body_b=None, feedback=False, dof=3, **kwargs):
        self.name = name
        if isinstance(world, World):
            world = world.ode_world
        self.ode_motor = self.MOTOR_FACTORY(world)
        self.ode_motor.attach(body_a.ode_body, body_b.ode_body if body_b else None)
        self.ode_motor.setFeedback(feedback)
        self.ode_motor.setNumAxes(dof)

    @property
    def feedback(self):
        return self.ode_motor.getFeedback()

    @property
    def dof(self):
        return self.ode_motor.getNumAxes()

    @property
    def angles(self):
        return [self.ode_motor.getAngle(i) for i in range(self.dof)]

    @property
    def angle_rates(self):
        return [self.ode_motor.getAngleRate(i) for i in range(self.dof)]

    @property
    def axes(self):
        return [dict(rel=self.ode_motor.getAxisRel(i), axis=self.ode_motor.getAxis(i))
                for i in range(self.dof)]

    @axes.setter
    def axes(self, axes):
        assert self.dof == len(axes)
        for i, axis in enumerate(axes):
            rel = 0
            if isinstance(axis, dict):
                rel = axis.get('rel', 0)
                axis = axis.get('axis')
            if axis is not None:
                self.ode_motor.setAxis(i, rel, axis)

    @property
    def lo_stops(self):
        return _get_params(self.ode_motor, 'LoStop', self.dof)

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        _set_params(self.ode_motor, 'LoStop', lo_stops, self.dof)

    @property
    def hi_stops(self):
        return _get_params(self.ode_motor, 'HiStop', self.dof)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        _set_params(self.ode_motor, 'HiStop', hi_stops, self.dof)

    @property
    def velocities(self):
        return _get_params(self.ode_motor, 'Vel', self.dof)

    @velocities.setter
    def velocities(self, velocities):
        _set_params(self.ode_motor, 'Vel', velocities, self.dof)

    @property
    def max_forces(self):
        return _get_params(self.ode_motor, 'FMax', self.dof)

    @max_forces.setter
    def max_forces(self, max_forces):
        _set_params(self.ode_motor, 'FMax', max_forces, self.dof)

    @property
    def cfms(self):
        return _get_params(self.ode_motor, 'CFM', self.dof)

    @cfms.setter
    def cfms(self, cfms):
        _set_params(self.ode_motor, 'CFM', cfms, self.dof)

    @property
    def stop_cfms(self):
        return _get_params(self.ode_joint, 'StopCFM', self.ADOF)

    @stop_cfms.setter
    def stop_cfms(self, stop_cfms):
        _set_params(self.ode_joint, 'StopCFM', stop_cfms, self.ADOF)

    @property
    def stop_erps(self):
        return _get_params(self.ode_joint, 'StopERP', self.ADOF)

    @stop_erps.setter
    def stop_erps(self, stop_erps):
        _set_params(self.ode_joint, 'StopERP', stop_erps, self.ADOF)


class AMotor(Motor):
    '''An AMotor applies forces to change an angle in the physics world.

    AMotors can be created in "user" mode---in which case the user must supply
    all axis and angle values---or, for 3 DOF motors, in "euler" mode---in which
    case the first and last axes must be specified, and ODE computes the middle
    axis automatically.
    '''

    MOTOR_FACTORY = ode.AMotor

    def __init__(self, *args, **kwargs):
        super(AMotor, self).__init__(*args, **kwargs)
        mode = kwargs.get('mode', 'user')
        if isinstance(mode, str):
            mode = ode.AMotorEuler if mode.lower().startswith('e') else ode.AMotorUser
        self.ode_motor.setMode(mode)

    @property
    def amotor(self):
        return self

    @property
    def ADOF(self):
        return self.ode_motor.getNumAxes()

    def add_torques(self, torques):
        self.ode_motor.addTorques(*torques)


class LMotor(Motor):
    '''An LMotor applies forces to change a position in the physics world.
    '''

    MOTOR_FACTORY = ode.LMotor

    @property
    def LDOF(self):
        return self.ode_motor.getNumAxes()


class Joint:
    '''This class wraps the ODE Joint class with some Python properties.

    The class has read-write properties for :

    - the axes of rotation or motion for the joint (`axes`),
    - the low and high stops of the joint's motion (`lo_stops` and `hi_stops`),
    - the maximum force that is allowed to be applied (`max_forces`),
    - the target velocities for the joint (`velocities`), and
    - the constraint force mixing parameter (`cfms`).

    There are also read-only properties for :

    - the feedback that the joint is experiencing (`feedback`),
    - the anchor of the joint in world coordinates (`anchor`),
    - the anchor of the joint on body 2 in world coordinates (`anchor2`),
    - the current angular configuration of the joint (`angles`) as well as the
      derivative (`angle_rates`),
    - the current linear configuration of the joint (`position`) as well as the
      derivative (`position_rate`),

    All of these properties except the position ones return sequences of values.
    The setters can be applied using a scalar value, which will be applied to
    all degrees of freedom, or with a sequence whose length must match the
    number of angular DOFs in the joint. (All joints in ODE have at most one
    linear axis of displacement, so the linear properties are scalars.)
    '''

    def __init__(self, name, world, body_a, body_b=None,
                 anchor=None, feedback=False, jointgroup=None):
        '''Create a new joint connecting two bodies in the world.
        '''
        self.name = name
        if isinstance(world, World):
            world = world.ode_world
        self.ode_joint = getattr(ode, '{}Joint'.format(self.__class__.__name__))(
            world, jointgroup=jointgroup)
        self.ode_joint.attach(body_a.ode_body, body_b.ode_body if body_b else None)
        self.ode_joint.setAnchor(anchor)
        self.ode_joint.setFeedback(feedback)
        self.ode_joint.setParam(ode.ParamCFM, 0)

        # we augment angular joints with a motor that allows us to monitor the
        # necessary joint forces, independent of the kinematic state.
        self.amotor = None
        if self.ADOF > 0:
            self.amotor = AMotor(
                name + ':amotor', world, body_a,
                body_b=body_b, dof=self.ADOF, mode='user',
                feedback=feedback, jointgroup=jointgroup)

    def __str__(self):
        return self.name

    @property
    def feedback(self):
        return self.ode_joint.getFeedback()

    @property
    def anchor(self):
        return self.ode_joint.getAnchor()

    @property
    def anchor2(self):
        return self.ode_joint.getAnchor2()

    @property
    def angles(self):
        return (self.ode_joint.getAngle(), )

    @property
    def angle_rates(self):
        return (self.ode_joint.getAngleRate(), )

    @property
    def position(self):
        return self.ode_joint.getPosition()

    @property
    def position_rate(self):
        return self.ode_joint.getPositionRate()

    @property
    def axes(self):
        return (self.ode_joint.getAxis(), )

    @axes.setter
    def axes(self, axes):
        self.amotor.axes = (dict(rel=1, axis=axes[0]), )
        self.ode_joint.setAxis(axes[0])

    @property
    def velocities(self):
        return self.amotor.velocities if self.ADOF > 0 else ()

    @velocities.setter
    def velocities(self, velocities):
        if self.ADOF > 0: self.amotor.velocities = velocities

    @property
    def max_forces(self):
        return self.amotor.max_forces if self.ADOF > 0 else ()

    @max_forces.setter
    def max_forces(self, max_forces):
        if self.ADOF > 0: self.amotor.max_forces = max_forces

    @property
    def cfms(self):
        return self.amotor.cfms if self.ADOF > 0 else ()

    @cfms.setter
    def cfms(self, cfms):
        if self.ADOF > 0: self.amotor.cfms = cfms

    @property
    def lo_stops(self):
        return _get_params(self.ode_joint, 'LoStop', self.ADOF)

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        _set_params(self.ode_joint, 'LoStop', lo_stops, self.ADOF)

    @property
    def hi_stops(self):
        return _get_params(self.ode_joint, 'HiStop', self.ADOF)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        _set_params(self.ode_joint, 'HiStop', hi_stops, self.ADOF)

    @property
    def stop_cfms(self):
        return _get_params(self.ode_joint, 'StopCFM', self.ADOF)

    @stop_cfms.setter
    def stop_cfms(self, stop_cfms):
        _set_params(self.ode_joint, 'StopCFM', stop_cfms, self.ADOF)

    @property
    def stop_erps(self):
        return _get_params(self.ode_joint, 'StopERP', self.ADOF)

    @stop_erps.setter
    def stop_erps(self, stop_erps):
        _set_params(self.ode_joint, 'StopERP', stop_erps, self.ADOF)

    def add_torques(self, torques):
        self.amotor.add_torques(torques)


class Fixed(Joint):
    ADOF = 0
    LDOF = 0


class Slider(Joint):
    ADOF = 0
    LDOF = 1


class Hinge(Joint):
    ADOF = 1
    LDOF = 0


class Piston(Joint):
    ADOF = 1
    LDOF = 1


class Universal(Joint):
    ADOF = 2
    LDOF = 0

    @property
    def axes(self):
        return (self.ode_joint.getAxis1(), self.ode_joint.getAxis2())

    @axes.setter
    def axes(self, axes):
        self.amotor.axes = dict(rel=1, axis=axes[0]), dict(rel=2, axis=axes[1])
        setters = [self.ode_joint.setAxis1, self.ode_joint.setAxis2]
        for axis, setter in zip(axes, setters):
            if axis is not None:
                setter(axis)

    @property
    def angles(self):
        return (self.ode_joint.getAngle1(), self.ode_joint.getAngle2())

    @property
    def angle_rates(self):
        return (self.ode_joint.getAngle1Rate(), self.ode_joint.getAngle2Rate())


class Ball(Joint):
    ADOF = 3
    LDOF = 0

    def __init__(self, *args, **kwargs):
        super(Ball, self).__init__(*args, **kwargs)

        # we augment ball joints with an additional motor that allows us to set
        # rotation limits.
        kw = {k: v for k, v in kwargs.items() if k != 'anchor'}
        self.alimit = AMotor(args[0] + ':alimit', *args[1:],
                             dof=self.ADOF, mode='euler', **kw)

    @property
    def angles(self):
        return self.alimit.angles

    @property
    def angle_rates(self):
        return self.alimit.angle_rates

    @property
    def axes(self):
        return self.alimit.axes

    @axes.setter
    def axes(self, axes):
        # always set axes in euler mode.
        axes = dict(rel=1, axis=axes[0]), None, dict(rel=2, axis=axes[1])
        self.amotor.axes = axes
        self.alimit.axes = axes

    @property
    def lo_stops(self):
        return self.alimit.lo_stops

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        self.alimit.lo_stops = lo_stops

    @property
    def hi_stops(self):
        return self.alimit.hi_stops

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        self.alimit.hi_stops = hi_stops


# Create a lookup table for things derived from the Joint class.
JOINTS = {}
for cls in Joint.__subclasses__():
    name = cls.__name__.lower()
    for i in range(3, len(name) + 1):
        JOINTS[name[:i]] = cls


def make_quaternion(theta, *axis):
    '''Given an angle and an axis, create a quaternion.'''
    x, y, z = axis
    r = np.sqrt(x * x + y * y + z * z)
    st = np.sin(theta / 2.)
    ct = np.cos(theta / 2.)
    return [x * st / r, y * st / r, z * st / r, ct]


def center_of_mass(bodies):
    '''Given a set of bodies, compute their center of mass in world coordinates.
    '''
    x = np.zeros(3.)
    t = 0.
    for b in bodies:
        m = b.mass
        x += np.asarray(b.body_to_world(m.c)) * m.mass
        t += m.mass
    return x / t


class World(base.World):
    '''A wrapper for an ODE World object, for running in a simulator.'''

    def __init__(self, dt=1. / 60, max_angular_speed=20):
        self.ode_world = ode.World()
        self.ode_world.setMaxAngularSpeed(max_angular_speed)
        self.ode_space = ode.QuadTreeSpace((0, 0, 0), (100, 100, 20), 10)
        self.ode_floor = ode.GeomPlane(self.ode_space, (0, 0, 1), 0)
        self.ode_contactgroup = ode.JointGroup()

        self.frame_no = 0
        self.dt = dt
        self.elasticity = 0.1
        self.friction = 2000
        self.gravity = 0, 0, -9.81
        self.cfm = 1e-8
        self.erp = 0.5

        self._bodies = {}
        self._joints = {}

    @property
    def gravity(self):
        return self.ode_world.getGravity()

    @gravity.setter
    def gravity(self, gravity):
        return self.ode_world.setGravity(gravity)

    @property
    def cfm(self):
        return self.ode_world.getCFM()

    @cfm.setter
    def cfm(self, cfm):
        return self.ode_world.setCFM(cfm)

    @property
    def erp(self):
        return self.ode_world.getERP()

    @erp.setter
    def erp(self, erp):
        return self.ode_world.setERP(erp)

    @property
    def bodies(self):
        for k in sorted(self._bodies):
            yield self._bodies[k]

    @property
    def joints(self):
        for k in sorted(self._joints):
            yield self._joints[k]

    def get_body(self, name):
        return self._bodies[name]

    def get_joint(self, name):
        return self._joints[name]

    def create_body(self, shape, name=None, **kwargs):
        '''Create a new body.'''
        shape = shape.lower()
        if name is None:
            for i in range(1 + len(self._bodies)):
                name = '{}{}'.format(shape, i)
                if name not in self._bodies:
                    break
        body = BODIES[shape](name, self.ode_world, self.ode_space, **kwargs)
        self._bodies[name] = body
        return body

    def join(self, shape, body_a, body_b=None, name=None, **kwargs):
        '''Create a new joint that connects two bodies together.'''
        ba = body_a
        if isinstance(body_a, str):
            ba = self.get_body(body_a)
        bb = body_b
        if isinstance(body_b, str):
            bb = self.get_body(body_b)
        shape = shape.lower()
        if name is None:
            name = '{}^{}^{}'.format(ba.name, shape, bb.name if bb else '')
        joint = JOINTS[shape](name, self.ode_world, ba, bb, **kwargs)
        self._joints[name] = joint
        return joint

    def move_next_to(self, body_a, body_b, offset_a, offset_b):
        '''Move body_b to be near body_a.

        After moving, offset_a on body_a will be in the same place as offset_b
        on body_b.

        Returns the location of the shared point, which is often useful to use
        as a joint anchor.
        '''
        ba = self.get_body(body_a)
        bb = self.get_body(body_b)
        anchor = ba.body_to_world(offset_a * ba.dimensions / 2)
        bb.position = (
            np.asarray(bb.position) + anchor -
            bb.body_to_world(offset_b * bb.dimensions / 2))
        return anchor

    def get_body_states(self):
        '''Return a list of the states of all bodies in the world.'''
        return [(b.name,
                 b.position,
                 b.quaternion,
                 b.linear_velocity,
                 b.angular_velocity) for b in self.bodies]

    def set_body_states(self, states):
        '''Set the states of all bodies in the world.'''
        for name, pos, rot, lin, ang in states:
            body = self.get_body(name)
            body.position = pos
            body.quaternion = rot
            body.linear_velocity = lin
            body.angular_velocity = ang

    def step(self, substeps=2):
        '''Step the world forward by one frame.'''
        self.frame_no += 1
        dt = self.dt / substeps
        for _ in range(substeps):
            self.ode_contactgroup.empty()
            self.ode_space.collide(None, self.on_collision)
            self.ode_world.step(dt)

    def are_connected(self, body_a, body_b):
        '''Return True iff the given bodies are currently connected.'''
        ba = body_a
        if isinstance(body_a, str):
            ba = self.get_body(body_a)
        bb = body_b
        if isinstance(body_b, str):
            bb = self.get_body(body_b)
        return bool(ode.areConnected(ba.ode_body, bb.ode_body))

    def on_collision(self, args, geom_a, geom_b):
        '''Callback function for the collide() method.'''
        body_a = geom_a.getBody()
        body_b = geom_b.getBody()
        if (ode.areConnected(body_a, body_b) or
            (body_a and body_a.isKinematic()) or
            (body_b and body_b.isKinematic())):
            return
        for c in ode.collide(geom_a, geom_b):
            c.setBounce(self.elasticity)
            c.setMu(self.friction)
            ode.ContactJoint(self.ode_world, self.ode_contactgroup, c).attach(
                geom_a.getBody(), geom_b.getBody())
