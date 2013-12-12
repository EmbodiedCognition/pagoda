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

'''Convenience wrappers for ODE objects.'''

from __future__ import division, print_function

import numpy as np
import numpy.random as rng
import ode
import OpenGL.GL as gl
import OpenGL.GLUT as glut

from . import base

TAU = 2 * np.pi


class Body(object):
    '''This class wraps things that participate in the ODE physics simulation.

    The primary attribute of this class is "body" -- a PyODE Body object. In
    addition, there is a PyODE Geom object (for detecting collisions -- not sure
    if this is really necessary to keep around though). This class also provides
    lots of Python-specific properties that call the equivalent ODE getters and
    setters for things like position, rotation, etc.
    '''

    def __init__(self, name, world, space, feedback=True, density=1000., **shape):
        self.name = name
        self.shape = shape
        self.feedback = feedback

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

    @property
    def rotation(self):
        return self.ode_body.getRotation()

    @property
    def quaternion(self):
        return self.ode_body.getQuaternion()

    @property
    def linear_velocity(self):
        return self.ode_body.getLinearVel()

    @property
    def angular_velocity(self):
        return self.ode_body.getAngularVel()

    @property
    def force(self):
        return self.ode_body.getForce()

    @property
    def torque(self):
        return self.ode_body.getTorque()

    @position.setter
    def position(self, position):
        self.ode_body.setPosition(position)

    @rotation.setter
    def rotation(self, rotation):
        self.ode_body.setRotation(rotation)

    @quaternion.setter
    def quaternion(self, quaternion):
        self.ode_body.setQuaternion(quaternion)

    @linear_velocity.setter
    def linear_velocity(self, velocity):
        self.ode_body.setLinearVel(velocity)

    @angular_velocity.setter
    def angular_velocity(self, velocity):
        self.ode_body.setAngularVel(velocity)

    @force.setter
    def force(self, force):
        self.ode_body.setForce(force)

    @torque.setter
    def torque(self, torque):
        self.ode_body.setTorque(torque)

    def trace(self):
        if not self.feedback:
            return ''
        x, y, z = self.position
        a, b, c, d = self.quaternion
        lx, ly, lz = self.linear_velocity
        ax, ay, az = self.angular_velocity
        return '{} p {} {} {} q {} {} {} {} lv {} {} {} av {} {} {}'.format(
            self.name, x, y, z, a, b, c, d, lx, ly, lz, ax, ay, az)

    def rotate_to_body(self, x):
        return np.dot(np.asarray(self.rotation).reshape((3, 3)), x)

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
        return np.asarray(self.lengths)

    def init_mass(self, m, density):
        m.setBox(density, *self.lengths)


class Sphere(Body):
    @property
    def radius(self):
        return self.shape['radius']

    @property
    def dimensions(self):
        d = 2 * self.radius
        return np.asarray([d, d, d])

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
        return np.asarray([d, d, self.length])

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
        return np.asarray([d, d, d + self.length])

    def init_mass(self, m, density):
        m.setCappedCylinder(density, 3, self.radius, self.length)


# Create a lookup table for things derived from the Body class.
BODIES = {}
for cls in Body.__subclasses__():
    name = cls.__name__.lower()
    BODIES[name] = BODIES[name[:3]] = cls


class Joint(object):
    '''This class wraps the ODE Joint class with some Python properties.'''

    def __init__(self, name, world, body_a, body_b=None, anchor=None, feedback=True):
        '''
        '''
        self.name = name
        self.world = world
        self.body_a = body_a
        self.body_b = body_b

        self.ode_joint = getattr(ode, '{}Joint'.format(self.__class__.__name__))(world)
        self.ode_joint.attach(body_a.ode_body, body_b.ode_body if body_b else None)
        self.ode_joint.setAnchor(anchor)
        self.ode_joint.setFeedback(feedback)

    def __str__(self):
        return self.name

    def _get_params(self, target, param):
        '''Get the given param from each of the DOFs for this joint.'''
        return [target.getParam(getattr(ode, 'Param{}{}'.format(param, s)))
                for s in ['', '2', '3'][:self.ADOF]]

    def _set_params(self, target, param, values):
        '''Set the given param for each of the DOFs for this joint.'''
        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [values] * self.ADOF
        assert self.ADOF == len(values)
        for s, value in zip(['', '2', '3'][:self.ADOF], values):
            target.setParam(getattr(ode, 'Param{}{}'.format(param, s)), value)

    @property
    def feedback(self):
        return self.ode_joint.getFeedback()

    @property
    def axes(self):
        return self.ode_joint.getAxis(),

    @property
    def anchor(self):
        return self.ode_joint.getAnchor()

    @property
    def velocities(self):
        return self._get_params(self.ode_joint, 'Vel')

    @property
    def max_forces(self):
        return self._get_params(self.ode_joint, 'FMax')

    @property
    def cfms(self):
        return self._get_params(self.ode_joint, 'CFM')

    @property
    def lo_stops(self):
        return self._get_params(self.ode_joint, 'LoStop')

    @property
    def hi_stops(self):
        return self._get_params(self.ode_joint, 'HiStop')

    @property
    def angles(self):
        return self.ode_joint.getAngle(),

    @property
    def angle_rates(self):
        return self.ode_joint.getAngleRate(),

    @property
    def position(self):
        return self.ode_joint.getPosition()

    @property
    def position_rate(self):
        return self.ode_joint.getPositionRate()

    @axes.setter
    def axes(self, axes):
        axis = axes
        if isinstance(axes, (list, tuple)):
            axis = axes[0]
        if isinstance(axis, dict):
            axis = axis.get('axis')
        if axis is not None:
            self.ode_joint.setAxis(axis)

    @velocities.setter
    def velocities(self, velocities):
        self._set_params(self.ode_joint, 'Vel', velocities)

    @max_forces.setter
    def max_forces(self, forces):
        self._set_params(self.ode_joint, 'FMax', forces)

    @cfms.setter
    def cfms(self, cfms):
        self._set_params(self.ode_joint, 'CFM', cfms)

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        self._set_params(self.ode_joint, 'LoStop', lo_stops)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        self._set_params(self.ode_joint, 'HiStop', hi_stops)

    def trace(self):
        feedback = self.feedback
        if not feedback:
            return ''
        parts = [self.name]
        for n, (x, y, z) in zip(('f1', 't1', 'f2', 't2'), feedback):
            parts.append('{} {} {} {}'.format(n, x, y, z))
        return ' '.join(parts)


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
        return self.ode_joint.getAxis1(), self.ode_joint.getAxis2()

    @property
    def angles(self):
        return self.ode_joint.getAngle1(), self.ode_joint.getAngle2()

    @property
    def angle_rates(self):
        return self.ode_joint.getAngle1Rate(), self.ode_joint.getAngle2Rate()

    @axes.setter
    def axes(self, axes):
        setters = [self.ode_joint.setAxis1, self.ode_joint.setAxis2]
        for axis, setter in zip(axes, setters):
            if isinstance(axis, dict):
                axis = axis.get('axis')
            if axis is not None:
                setter(axis)


class Ball(Joint):
    ADOF = 3
    LDOF = 0

    def __init__(self, name, world, body_a, body_b=None, **kwargs):
        # attach auxiliary amotors to apply forces to the joint and enforce
        # kinematic rotation limits.
        self.amotor = AMotor(name + '.motor', world, body_a, body_b)
        self.alimits = AMotor(name + '.limits', world, body_a, body_b)
        super(Ball, self).__init__(name, world, body_a, body_b=body_b, **kwargs)

    @property
    def axes(self):
        return self.amotor.axes

    @property
    def lo_stops(self):
        return self.alimits.lo_stops

    @property
    def hi_stops(self):
        return self.alimits.hi_stops

    @property
    def velocities(self, velocities):
        return self.amotor.velocities

    @property
    def max_forces(self, forces):
        return self.amotor.max_forces

    @property
    def cfms(self, cfms):
        return self.amotor.cfms

    @property
    def angles(self):
        return self.alimits.angles

    @property
    def angle_rates(self):
        return self.alimits.angle_rates

    @axes.setter
    def axes(self, axes):
        self.amotor.axes = axes
        self.alimits.axes = axes

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        self.alimits.lo_stops = lo_stops

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        self.alimits.hi_stops = hi_stops

    @velocities.setter
    def velocities(self, velocities):
        self.amotor.velocities = velocities

    @max_forces.setter
    def max_forces(self, forces):
        self.amotor.max_forces = forces

    @cfms.setter
    def cfms(self, cfms):
        self.amotor.cfms = cfms


class AMotor(Joint):
    LDOF = 0

    def __init__(self, name, world, body_a, body_b=None,
                 axes=(dict(rel=1, axis=(1, 0, 0)),
                       dict(rel=1, axis=(0, 1, 0)),
                       dict(rel=2, axis=(0, 0, 1))),
                 feedback=True,
                 mode=ode.AMotorUser):
        self.name = name
        self.world = world
        self.body_a = body_a
        self.body_b = body_b

        self.amotor = ode.AMotor(world)
        self.amotor.attach(body_a.ode_body, body_b.ode_body if body_b else None)
        self.amotor.setMode(mode)
        self.amotor.setFeedback(feedback)
        self.axes = axes

    @property
    def axes(self):
        return [dict(rel=self.amotor.getAxisRel(i), axis=self.amotor.getAxis(i))
                for i in range(self.ADOF)]

    @property
    def lo_stops(self):
        return self._get_params(self.amotor, 'LoStop')

    @property
    def hi_stops(self):
        return self._get_params(self.amotor, 'HiStop')

    @property
    def velocities(self):
        return self._get_params(self.amotor, 'Vel')

    @property
    def max_forces(self):
        return self._get_params(self.amotor, 'FMax')

    @property
    def cfms(self):
        return self._get_params(self.amotor, 'CFM')

    @property
    def angles(self):
        return [self.amotor.getAngle(i) for i in range(self.ADOF)]

    @property
    def angle_rates(self):
        return [self.amotor.getAngleRate(i) for i in range(self.ADOF)]

    @axes.setter
    def axes(self, axes):
        self.ADOF = len(axes)
        self.amotor.setNumAxes(len(axes))
        for i, axis in enumerate(axes):
            rel = 1
            if isinstance(axis, dict):
                rel = axis.get('rel', 1)
                axis = axis.get('axis')
            if axis is not None:
                self.amotor.setAxis(i, rel, axis)

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        self._set_params(self.amotor, 'LoStop', lo_stops)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        self._set_params(self.amotor, 'HiStop', hi_stops)

    @velocities.setter
    def velocities(self, velocities):
        self._set_params(self.amotor, 'Vel', velocities)

    @max_forces.setter
    def max_forces(self, forces):
        self._set_params(self.amotor, 'FMax', forces)

    @cfms.setter
    def cfms(self, cfms):
        self._set_params(self.amotor, 'CFM', cfms)


class LMotor(Joint):
    ADOF = 0

    def __init__(self, name, world, body_a, body_b=None,
                 axes=(dict(rel=1, axis=(1, 0, 0)),
                       dict(rel=1, axis=(0, 1, 0)),
                       dict(rel=2, axis=(0, 0, 1))),
                 feedback=True):
        self.name = name
        self.world = world
        self.body_a = body_a
        self.body_b = body_b

        self.lmotor = ode.LMotor(world)
        self.lmotor.attach(body_a.ode_body, body_b.ode_body if body_b else None)
        self.lmotor.setFeedback(feedback)
        self.axes = axes

    @property
    def axes(self):
        return [self.lmotor.getAxis(i) for i in range(self.LDOF)]

    @property
    def lo_stops(self):
        return self._get_params(self.lmotor, 'LoStop')

    @property
    def hi_stops(self):
        return self._get_params(self.lmotor, 'HiStop')

    @property
    def velocities(self):
        return self._get_params(self.lmotor, 'Vel')

    @property
    def max_forces(self):
        return self._get_params(self.lmotor, 'FMax')

    @property
    def cfms(self):
        return self._get_params(self.lmotor, 'CFM')

    @axes.setter
    def axes(self, axes):
        self.LDOF = len(axes)
        self.lmotor.setNumAxes(len(axes))
        for i, axis in enumerate(axes):
            rel = 1
            if isinstance(axis, dict):
                rel = axis.get('rel', 1)
                axis = axis.get('axis')
            if axis is not None:
                self.lmotor.setAxis(i, rel, axis)

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        self._set_params(self.lmotor, 'LoStop', lo_stops)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        self._set_params(self.lmotor, 'HiStop', hi_stops)

    @velocities.setter
    def velocities(self, velocities):
        self._set_params(self.lmotor, 'Vel', velocities)

    @max_forces.setter
    def max_forces(self, forces):
        self._set_params(self.lmotor, 'FMax', forces)

    @cfms.setter
    def cfms(self, cfms):
        self._set_params(self.lmotor, 'CFM', cfms)


# Create a lookup table for things derived from the Body class. Should probably
# do this using a metaclass, but this is less head-warpy.
JOINTS = {}
for cls in Joint.__subclasses__():
    name = cls.__name__.lower()
    JOINTS[name] = JOINTS[name[:3]] = cls


def make_quaternion(theta, *axis):
    '''Given an angle and an axis, create a quaternion.'''
    x, y, z = axis
    r = np.sqrt(x * x + y * y + z * z)
    st = np.sin(theta / 2.)
    ct = np.cos(theta / 2.)
    return [x * st / r, y * st / r, z * st / r, ct]


class World(base.World):
    '''A wrapper for an ODE World object, for running in a simulator.'''

    def __init__(self,
                 dt=1. / 60,
                 elasticity=0.2,
                 friction=200,
                 gravity=(0, 0, -9.81),
                 erp=0.8,
                 cfm=1e-5,
                 max_angular_speed=20):
        self.frame = 0
        self.dt = dt
        self.elasticity = elasticity
        self.friction = friction

        self.world = ode.World()
        self.world.setGravity(gravity)
        self.world.setERP(erp)
        self.world.setCFM(cfm)
        self.world.setMaxAngularSpeed(max_angular_speed)

        self.space = ode.QuadTreeSpace((0, 0, 0), (100, 100, 20), 10)

        self.floor = ode.GeomPlane(self.space, (0, 0, 1), 0)
        self.contactgroup = ode.JointGroup()
        self._colors = {}
        self._bodies = {}
        self._joints = {}

    @property
    def bodies(self):
        for k in sorted(self._bodies):
            yield self._bodies[k]

    @property
    def joints(self):
        for k in sorted(self._joints):
            yield self._joints[k]

    @property
    def center_of_mass(self):
        x = np.zeros(3.)
        t = 0.
        for b in self.bodies:
            m = b.mass
            x += np.asarray(b.body_to_world(m.c)) * m.mass
            t += m.mass
        return x / t

    def get_body(self, name):
        return self._bodies[name]

    def get_joint(self, name):
        return self._joints[name]

    def create_body(self, shape, name=None, color=None, **kwargs):
        '''Create a new body.'''
        shape = shape.lower()
        if name is None:
            for i in range(1 + len(self._bodies)):
                name = '{}{}'.format(shape, i)
                if name not in self._bodies:
                    break
        body = BODIES[shape](name, self.world, self.space, **kwargs)
        self._colors[name] = color
        if color is None:
            self._colors[name] = tuple(rng.random(3)) + (0.5, )
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
            name = '{}:{}:{}'.format(ba.name, shape, bb.name if bb else '')
        joint = JOINTS[shape](name, self.world, ba, bb, **kwargs)
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
        self.frame += 1
        dt = self.dt / substeps
        for _ in range(substeps):
            self.contactgroup.empty()
            self.space.collide(None, self.on_collision)
            self.world.step(dt)
        return True

    def trace(self, handle):
        '''Trace world bodies and joints.'''
        bodies = ' '.join(b.trace() for b in self.bodies)
        joints = ' '.join(j.trace() for j in self.joints)
        print('{} {} {}'.format(self.frame, bodies, joints), file=handle)

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
        if ode.areConnected(geom_a.getBody(), geom_b.getBody()):
            return
        for c in ode.collide(geom_a, geom_b):
            c.setBounce(self.elasticity)
            c.setMu(self.friction)
            ode.ContactJoint(self.world, self.contactgroup, c).attach(
                geom_a.getBody(), geom_b.getBody())

    def draw(self, color=None, n=59):
        '''Draw all bodies in the world.'''
        for name, body in self._bodies.iteritems():
            gl.glColor(*(color or self._colors[name]))
            x, y, z = body.position
            r = body.rotation
            gl.glPushMatrix()
            gl.glMultMatrixf([r[0], r[3], r[6], 0.,
                              r[1], r[4], r[7], 0.,
                              r[2], r[5], r[8], 0.,
                              x, y, z, 1.])
            if isinstance(body, Box):
                gl.glScale(*body.lengths)
                glut.glutSolidCube(1)
            if isinstance(body, Sphere):
                glut.glutSolidSphere(body.radius, n, n)
            if isinstance(body, Cylinder):
                l = body.length
                gl.glTranslate(0, 0, -l / 2.)
                glut.glutSolidCylinder(body.radius, l, n, n)
            if isinstance(body, Capsule):
                r = body.radius
                l = body.length
                gl.glTranslate(0, 0, -l / 2.)
                glut.glutSolidCylinder(r, l, n, n)
                glut.glutSolidSphere(r, n, n)
                gl.glTranslate(0, 0, l)
                glut.glutSolidSphere(r, n, n)
            gl.glPopMatrix()
        return
        for name, joint in self._joints.iteritems():
            l = 0.3
            x, y, z = joint.anchor
            for i, (rx, ry, rz) in enumerate(joint.axes):
                r = joint.body_a.rotation
                gl.glColor((i+1) / 3., 0, 0)
                gl.glPushMatrix()
                gl.glMultMatrixf([r[0], r[3], r[6], 0.,
                                  r[1], r[4], r[7], 0.,
                                  r[2], r[5], r[8], 0.,
                                  x, y, z, 1.])
                # http://thjsmith.com/40/cylinder-between-two-points-opengl-c
                # http://en.wikipedia.org/wiki/Cross_product
                # (rx, ry, rz) x (0, 0, 1) = (ry, -rx, 0)
                # theta = acos((rx, ry, rz) * (0, 0, 1)) / |(rx, ry, rz)|
                rl = np.sqrt(rx * rx + ry * ry + rz * rz)
                gl.glRotate(np.arccos(rz / rl) * 360 / TAU, ry, -rx, 0)
                gl.glTranslate(0, 0, -l / 2.)
                glut.glutSolidCylinder(l / 20, l, n, n)
                gl.glPopMatrix()
