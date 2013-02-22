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

import numpy as np
import numpy.random as rng
import ode
import OpenGL.GL as gl
import OpenGL.GLUT as glut

from . import world

def make_quaternion(theta, *axis):
    x, y, z = axis
    r = np.sqrt(x * x + y * y + z * z)
    st = np.sin(theta / 2.)
    ct = np.cos(theta / 2.)
    return [x * st / r, y * st / r, z * st / r, ct]


class Body(object):
    '''This class wraps things that participate in the ODE physics simulation.

    The primary attribute of this class is "body" -- an actual PyODE Body
    object. In addition, there is a color (for drawing the object), a PyODE Geom
    object (for detecting collisions -- not sure if this is really necessary to
    keep around though), and several utility methods for doing things
    like drawing. This class also provides lots of Python-specific properties
    (which call the equivalent ODE getters and setters) for things like
    position, rotation, etc.
    '''

    def __init__(self, world, space, color=None, density=1., **shape):
        self.density = density
        self.shape = shape

        self.color = color or rng.rand(3)

        m = ode.Mass()
        self.init_mass(m)
        self.body = ode.Body(world)
        self.body.setMass(m)

        self.geom = getattr(ode, 'Geom%s' % self.__class__.__name__)(space, **shape)
        self.geom.setBody(self.body)

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
        try:
            self.body.setRotation(rotation)
        except:
            ct = np.cos(rotation)
            st = np.sin(rotation)
            self.body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])

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

    def body_to_world(self, position):
        return self.body.getRelPointPos(position)

    def world_to_body(self, position):
        return self.body.getPosRelPoint(position)

    def add_force(self, force, relative=False, position=None, relative_position=None):
        if relative_position is not None:
            if relative:
                self.body.addRelForceAtRelPos(force, relative_position)
            else:
                self.body.addForceAtRelPos(force, relative_position)
        elif position is not None:
            if relative:
                self.body.addRelForceAtPos(force, position)
            else:
                self.body.addForceAtPos(force, position)
        else:
            if relative:
                self.body.addRelForce(force)
            else:
                self.body.addForce(force)

    def add_torque(self, torque, relative=False):
        if relative:
            self.body.addRelTorque(torque)
        else:
            self.body.addTorque(torque)

    def draw(self):
        gl.glColor(*self.color)
        x, y, z = self.position
        R = self.rotation
        gl.glPushMatrix()
        gl.glMultMatrixf([R[0], R[3], R[6], 0.,
                          R[1], R[4], R[7], 0.,
                          R[2], R[5], R[8], 0.,
                          x, y, z, 1.])
        self._draw()
        gl.glPopMatrix()


class Box(Body):
    def init_mass(self, m):
        m.setBox(self.density, *self.shape['lengths'])

    def _draw(self):
        gl.glScale(*self.shape['lengths'])
        glut.glutSolidCube(1)


class Sphere(Body):
    def init_mass(self, m):
        m.setSphere(self.density, self.shape['radius'])

    def _draw(self):
        r = self.shape['radius']
        gl.glScale(r, r, r)
        glut.glutSolidSphere(1, 31, 31)


class Cylinder(Body):
    def init_mass(self, m):
        m.setCylinder(self.density, 3, self.shape['radius'], self.shape['length'])

    def _draw(self):
        r = self.shape['radius']
        gl.glScale(r, r, self.shape['length'])
        gl.glTranslate(0, 0, -0.5)
        glut.glutSolidCylinder(1, 1, 31, 31)


class Capsule(Body):
    def init_mass(self, m):
        m.setCappedCylinder(self.density, 3, self.shape['radius'], self.shape['length'])

    def _draw(self):
        r = self.shape['radius']
        l = self.shape['length']

        gl.glTranslate(0, 0, -l / 2.)
        glut.glutSolidCylinder(r, l, 31, 31)
        glut.glutSolidSphere(r, 31, 31)
        gl.glTranslate(0, 0, l)
        glut.glutSolidSphere(r, 31, 31)


class Joint(object):
    '''This class wraps the ODE Joint class with some Python properties.'''

    def __init__(self, world, body_a, body_b=None):
        self.joint = getattr(ode, self.__class__.__name__)(world)
        self.joint.attach(body_a, body_b)

    @property
    def feedback(self):
        return self.joint.getFeedback()

    @property
    def anchor(self):
        return self.joint.getAnchor()

    @property
    def axis(self):
        return self.joint.getAxis()

    @property
    def velocity(self):
        return self.joint.getParam(ode.ParamVel)

    @property
    def max_force(self):
        return self.joint.getParam(ode.ParamFMax)

    @property
    def lo_stop(self):
        return self.joint.getParam(ode.ParamLoStop)

    @property
    def hi_stop(self):
        return self.joint.getParam(ode.ParamHiStop)

    @anchor.setter
    def anchor(self, anchor):
        return self.joint.setAnchor(anchor)

    @axis.setter
    def axis(self, axis):
        return self.joint.setAxis(axis)

    @velocity.setter
    def velocity(self, velocity):
        return self.joint.setParam(ode.ParamVel, velocity)

    @max_force.setter
    def max_force(self, force):
        return self.joint.setParam(ode.ParamFMax, force)

    @lo_stop.setter
    def lo_stop(self, lo_stop):
        return self.joint.setParam(ode.ParamLoStop, lo_stop)

    @hi_stop.setter
    def hi_stop(self, hi_stop):
        return self.joint.setParam(ode.ParamHiStop, hi_stop)


class Fixed(Joint):
    pass


class Hinge(Joint):
    pass


class Slider(Joint):
    @property
    def position(self):
        return self.joint.getPosition()

    @property
    def position_rate(self):
        return self.joint.getPositionRate()


class Universal(Joint):
    @property
    def axes(self):
        return self.joint.getAxis1(), self.joint.getAxis2()

    @axes.setter
    def axes(self, axis1, axis2):
        self.joint.setAxis1(axis1)
        self.joint.setAxis2(axis2)

    @property
    def velocity_2(self):
        return self.joint.getParam(ode.ParamVel2)

    @property
    def max_force_2(self):
        return self.joint.getParam(ode.ParamFMax2)

    @velocity_2.setter
    def velocity_2(self, velocity):
        return self.joint.setParam(ode.ParamVel2, velocity)

    @max_force_2.setter
    def max_force_2(self, force):
        return self.joint.setParam(ode.ParamFMax2, force)


class Ball(Joint):
    @property
    def axes(self):
        return self.joint.getAxis1(), self.joint.getAxis2(), self.joint.getAxis3()

    @property
    def velocity_2(self):
        return self.joint.getParam(ode.ParamVel2)

    @property
    def velocity_3(self):
        return self.joint.getParam(ode.ParamVel3)

    @property
    def max_force_2(self):
        return self.joint.getParam(ode.ParamFMax2)

    @property
    def max_force_3(self):
        return self.joint.getParam(ode.ParamFMax3)

    @axes.setter
    def axes(self, axis1, axis2, axis3):
        self.joint.setAxis1(axis1)
        self.joint.setAxis2(axis2)
        self.joint.setAxis2(axis3)

    @velocity_2.setter
    def velocity_2(self, velocity):
        return self.joint.setParam(ode.ParamVel2, velocity)

    @velocity_3.setter
    def velocity_3(self, velocity):
        return self.joint.setParam(ode.ParamVel3, velocity)

    @max_force_2.setter
    def max_force_2(self, force):
        return self.joint.setParam(ode.ParamFMax2, force)

    @max_force_3.setter
    def max_force_3(self, force):
        return self.joint.setParam(ode.ParamFMax3, force)


class World(world.World):
    '''This class wraps the ODE World class with some convenience methods.'''

    def __init__(self,
                 dt=1. / 60,
                 elasticity=0.2,
                 friction=5000,
                 gravity=(0, -9.81, 0),
                 erp=0.8,
                 cfm=1e-5,
                 max_angular_speed=20):
        super(World, self).__init__(dt)

        self.elasticity = elasticity
        self.friction = friction

        self.world = ode.World()
        self.world.setGravity(gravity)
        self.world.setERP(erp)
        self.world.setCFM(cfm)

        # TODO: not yet in pyode :(
        #self.world.setMaxAngularSpeed(max_angular_speed)

        self.space = ode.Space()

        self.floor = ode.GeomPlane(self.space, (0, 1, 0), 0)
        self.contactgroup = ode.JointGroup()
        self.bodies = []

    @property
    def center_of_mass(self):
        x = np.zeros(3.)
        t = 0.
        for b in self.bodies:
            m = b.mass
            x += np.asarray(b.body_to_world(m.c)) * m.mass
            t += m.mass
        return x / t

    def create_body(self, shape, **kwargs):
        '''Create a new body.'''
        b = globals()[shape.capitalize()](self.world, self.space, **kwargs)
        self.bodies.append(b)
        return b

    def join(self, joint, body_a, body_b=None):
        '''Create a new joint that connects two bodies together.'''
        return globals()[joint.capitalize()](self.world, body_a, body_b)

    def step(self, substeps=2):
        '''Step the world forward by one frame.'''
        dt = self.dt / substeps
        for _ in range(substeps):
            self.space.collide(None, self.on_collision)
            self.world.step(dt)
            self.contactgroup.empty()

    def on_collision(self, args, geom_a, geom_b):
        '''Callback function for the collide() method.'''
        for c in ode.collide(geom_a, geom_b):
            c.setBounce(self.elasticity)
            c.setMu(self.friction)
            ode.ContactJoint(self.world, self.contactgroup, c).attach(
                geom_a.getBody(), geom_b.getBody())
