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
        self.body = ode.Body(world)
        self.body.setMass(m)

        self.geom = getattr(ode, 'Geom%s' % self.__class__.__name__)(space, **shape)
        self.geom.setBody(self.body)

    def __str__(self):
        return '%s: %s at %s' % (self.name, self.__class__.__name__, self.position)

    @property
    def mass(self):
        return self.body.getMass()

    @property
    def position(self):
        return self.body.getPosition()

    @property
    def rotation(self):
        return self.body.getRotation()

    @property
    def quaternion(self):
        return self.body.getQuaternion()

    @property
    def linear_velocity(self):
        return self.body.getLinearVel()

    @property
    def angular_velocity(self):
        return self.body.getAngularVel()

    @property
    def force(self):
        return self.body.getForce()

    @property
    def torque(self):
        return self.body.getTorque()

    @position.setter
    def position(self, position):
        self.body.setPosition(position)

    @rotation.setter
    def rotation(self, rotation):
        self.body.setRotation(rotation)

    @quaternion.setter
    def quaternion(self, quaternion):
        self.body.setQuaternion(quaternion)

    @linear_velocity.setter
    def linear_velocity(self, velocity):
        self.body.setLinearVel(velocity)

    @angular_velocity.setter
    def angular_velocity(self, velocity):
        self.body.setAngularVel(velocity)

    @force.setter
    def force(self, force):
        self.body.setForce(force)

    @torque.setter
    def torque(self, torque):
        self.body.setTorque(torque)

    def trace(self):
        if not self.feedback:
            return ''
        x, y, z = self.position
        a, b, c, d = self.quaternion
        lx, ly, lz = self.linear_velocity
        ax, ay, az = self.angular_velocity
        return '%s p %f %f %f q %f %f %f %f lv %f %f %f av %f %f %f' % (
            self.name, x, y, z, a, b, c, d, lx, ly, lz, ax, ay, az)

    def body_to_world(self, position):
        return self.body.getRelPointPos(position)

    def world_to_body(self, position):
        return self.body.getPosRelPoint(position)

    def add_force(self, force, relative=False, position=None, relative_position=None):
        if relative_position is not None:
            op = self.body.addRelForceAtRelPos if relative else self.body.addForceAtRelPos
            op(force, relative_position)
        elif position is not None:
            op = self.body.addRelForceAtPos if relative else self.body.addForceAtPos
            op(force, position)
        else:
            op = self.body.addRelForce if relative else self.body.addForce
            op(force)

    def add_torque(self, torque, relative=False):
        op = self.body.addRelTorque if relative else self.body.addTorque
        op(torque)


class Box(Body):
    @property
    def lengths(self):
        return self.shape['lengths']

    @property
    def dimensions(self):
        return np.array(self.lengths)

    def init_mass(self, m, density):
        m.setBox(density, *self.lengths)


class Sphere(Body):
    @property
    def radius(self):
        return self.shape['radius']

    @property
    def dimensions(self):
        d = 2 * self.radius
        return np.array([d, d, d])

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
        d = 2 * self.radius
        return np.array([d, d, self.length])

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
        return np.array([d, d, d + self.length])

    def init_mass(self, m, density):
        m.setCappedCylinder(density, 3, self.radius, self.length)


# Create a lookup table for things derived from the Body class. Should probably
# do this using a metaclass, but this is less head-warpy.
BODIES = {}
for name, cls in globals().iteritems():
    if issubclass(cls, Body):
        name = name.lower()
        BODIES[name] = BODIES[name[:3]] = cls


class Joint(object):
    '''This class wraps the ODE Joint class with some Python properties.'''

    def __init__(self, name, world, body_a, body_b=None, feedback=False, **kwargs):
        self.name = name
        self.body_a = body_a
        self.body_b = body_b

        # attach an ode joint to these two bodies to constrain their movements
        # and measure forces experienced at the joint.
        self.joint = getattr(ode, '%sJoint' % self.__class__.__name__)(world)
        self.joint.attach(body_a.body, body_b.body if body_b else None)
        if feedback:
            self.joint.setFeedback(True)

        # attach an amotor to apply angular forces to the joint.
        self.amotor = None
        if self.ADOF > 0:
            self.amotor = ode.AMotor(world)
            self.amotor.attach(body_a.body, body_b.body if body_b else None)
            self.amotor.setMode(ode.AMotorEuler)
            self.amotor.setNumAxes(self.ADOF)
            for i in range(self.ADOF):
                ax = kwargs.get('angular_axis{}'.format(i+1))
                if ax is not None:
                    mode = kwargs.get('angular_axis{}_mode'.format(i+1), 1)
                    self.amotor.setAxis(i, mode, ax)

        # attach an lmotor to apply linear forces to the joint.
        self.lmotor = None
        if self.LDOF > 0:
            self.lmotor = ode.LMotor(world)
            self.lmotor.attach(body_a.body, body_b.body if body_b else None)
            self.lmotor.setNumAxes(self.LDOF)
            for i in range(self.LDOF):
                ax = kwargs.get('linear_axis{}'.format(i+1))
                if ax is not None:
                    mode = kwargs.get('linear_axis{}_mode'.format(i+1), 1)
                    self.lmotor.setAxis(i, mode, ax)

        for k, v in kwargs.iteritems():
            if k.startswith('angular_axis') or k.startswith('linear_axis'):
                continue
            setattr(self, k, v)

    def __str__(self):
        return '%s: %s (%s <-> %s) at %s' % (
            self.name, self.__class__.__name__,
            self.body_a.name, self.body_b.name if self.body_b else None,
            self.anchor)

    def _get_params(self, target, param):
        '''Get the given param from each of the DOFs for this joint.'''
        for s in self.axis_suffixes:
            yield target.getParam(getattr(ode, 'Param%s%s' % (param, s)))

    def _set_params(self, target, param, values):
        '''Set the given param for each of the DOFs for this joint.'''
        if not isinstance(values, (list, tuple)):
            values = (values, )
        assert self.ADOF == len(values)
        for s, value in zip(self.axis_suffixes, values):
            target.setParam(getattr(ode, 'Param%s%s' % (param, s)), value)

    @property
    def axis_suffixes(self):
        return ['', '2', '3'][:self.ADOF]

    @property
    def feedback(self):
        return self.joint.getFeedback()

    @property
    def anchor(self):
        return self.joint.getAnchor()

    @property
    def target_velocities(self):
        return self._get_params(self.amotor, 'Vel')

    @property
    def max_forces(self):
        return self._get_params(self.amotor, 'FMax')

    @property
    def lo_stops(self):
        return self._get_params(self.joint, 'LoStop')

    @property
    def hi_stops(self):
        return self._get_params(self.joint, 'HiStop')

    @property
    def angles(self):
        return map(self.amotor.getAngle, range(self.ADOF))

    @property
    def angle_rates(self):
        return map(self.amotor.getAngleRate, range(self.ADOF))

    @property
    def position(self):
        return self.joint.getPosition()

    @property
    def position_rate(self):
        return self.joint.getPositionRate()

    @target_velocities.setter
    def target_velocities(self, velocities):
        self._set_params(self.amotor, 'Vel', velocities)

    @max_forces.setter
    def max_forces(self, forces):
        self._set_params(self.amotor, 'FMax', forces)

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        self._set_params(self.joint, 'LoStop', lo_stops)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        self._set_params(self.joint, 'HiStop', hi_stops)

    def trace(self):
        feedback = self.feedback
        if not feedback:
            return ''
        parts = [self.name]
        for n, (x, y, z) in zip(('f1', 't1', 'f2', 't2'), feedback):
            parts.append('%s %f %f %f' % (n, x, y, z))
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

    @property
    def axes(self):
        return self.joint.getAxis(),


class Piston(Joint):
    ADOF = 1
    LDOF = 1

    @property
    def axes(self):
        return self.joint.getAxis(),


class Universal(Joint):
    ADOF = 2
    LDOF = 0

    @property
    def axes(self):
        return self.joint.getAxis1(), self.joint.getAxis2()


class Ball(Joint):
    ADOF = 3
    LDOF = 0

    def __init__(self, name, world, body_a, body_b=None, feedback=False, **kwargs):
        super(Ball, self).__init__(
            name, world, body_a, body_b=body_b, feedback=feedback, **kwargs)

        # attach an auxiliary amotor to apply angular joint limits.
        self.alimits = ode.AMotor(world)
        self.alimits.attach(body_a.body, body_b.body if body_b else None)
        self.alimits.setMode(ode.AMotorEuler)
        self.alimits.setNumAxes(self.ADOF)
        self.alimits.setAxis(0, 1, self.amotor.getAxis(0))
        self.alimits.setAxis(1, 1, self.amotor.getAxis(1))
        self.alimits.setAxis(2, 2, self.amotor.getAxis(2))

    @property
    def axes(self):
        return (self.amotor.getAxis(0),
                self.amotor.getAxis(1),
                self.amotor.getAxis(2))

    @property
    def lo_stops(self):
        return self._get_params(self.alimits, 'LoStop')

    @property
    def hi_stops(self):
        return self._get_params(self.alimits, 'HiStop')

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        self._set_params(self.alimits, 'LoStop', lo_stops)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        self._set_params(self.alimits, 'HiStop', hi_stops)


# Create a lookup table for things derived from the Body class. Should probably
# do this using a metaclass, but this is less head-warpy.
JOINTS = {}
for name, cls in globals().iteritems():
    if issubclass(cls, Joint):
        name = name.lower()
        JOINTS[name] = JOINTS[name[:3]] = cls


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

        # TODO: not yet in pyode :(
        #self.world.setMaxAngularSpeed(max_angular_speed)

        self.space = ode.HashSpace()

        self.floor = ode.GeomPlane(self.space, (0, 0, 1), 0)
        self.contactgroup = ode.JointGroup()
        self._colors = {}
        self._bodies = {}
        self._joints = {}

    @staticmethod
    def make_quaternion(theta, *axis):
        x, y, z = axis
        r = np.sqrt(x * x + y * y + z * z)
        st = np.sin(theta / 2.)
        ct = np.cos(theta / 2.)
        return [x * st / r, y * st / r, z * st / r, ct]

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

    def create_body(self, shape, name=None, color=None, feedback=True, **kwargs):
        '''Create a new body.'''
        shape = shape.lower()
        if name is None:
            for i in range(1 + len(self._bodies)):
                name = '%s%d' % (shape, i)
                if name not in self._bodies:
                    break
        body = BODIES[shape](name, self.world, self.space, feedback=feedback, **kwargs)
        self._colors[name] = color if color is not None else rng.random(3)
        self._bodies[name] = body
        return body

    def join(self, shape, body_a, body_b=None, name=None, feedback=True, **kwargs):
        '''Create a new joint that connects two bodies together.'''
        ba = body_a
        if isinstance(body_a, str):
            ba = self.get_body(body_a)
        bb = body_b
        if isinstance(body_b, str):
            bb = self.get_body(body_b)
        shape = shape.lower()
        if name is None:
            name = '%s:%s:%s' % (ba.name, shape, bb.name if bb else '')
        joint = JOINTS[shape](name, self.world, ba, bb, feedback=feedback, **kwargs)
        return self._joints[name] = joint
        return joint

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
        print('%d %s %s' % (self.frame, bodies, joints), file=handle)

    def are_connected(self, body_a, body_b):
        '''Return True iff the given bodies are currently connected.'''
        ba = body_a
        if isinstance(body_a, str):
            ba = self.get_body(body_a)
        bb = body_b
        if isinstance(body_b, str):
            bb = self.get_body(body_b)
        return bool(ode.areConnected(ba.body, bb.body))

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
