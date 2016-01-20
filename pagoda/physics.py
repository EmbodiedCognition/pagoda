'''This module contains convenience wrappers for ODE objects.'''

from __future__ import division, print_function

import collections
import numpy as np
import ode

from . import base


BodyState = collections.namedtuple(
    'BodyState', 'name position quaternion linear_velocity angular_velocity')


class Body(object):
    '''This class wraps things that participate in the ODE physics simulation.

    This class basically provides lots of Python-specific properties that call
    the equivalent ODE getters and setters for things like position, rotation,
    etc.
    '''

    def __init__(self, name, world, density=1000., **shape):
        self.name = name
        self.world = world
        self.shape = shape

        m = ode.Mass()
        self.init_mass(m, density)
        self.ode_body = ode.Body(world.ode_world)
        self.ode_body.setMass(m)
        self.ode_geom = getattr(ode, 'Geom%s' % self.__class__.__name__)(
            world.ode_space, **shape)
        self.ode_geom.setBody(self.ode_body)

    def __str__(self):
        return '{0.__class__.__name__} {0.name} at {1}'.format(
            self, np.array(self.position).round(3))

    @property
    def mass(self):
        '''The ODE mass object for this body.'''
        return self.ode_body.getMass()

    @property
    def state(self):
        '''The state of this body includes:

            - name of the body (str)
            - position (3-tuple)
            - quaternion (4-tuple)
            - linear velocity (3-tuple)
            - angular velocity (3-tuple)
        '''
        return BodyState(self.name,
                         self.position,
                         self.quaternion,
                         self.linear_velocity,
                         self.angular_velocity)

    @state.setter
    def state(self, state):
        '''Set the state of this body.

        Parameters
        ----------
        state : BodyState tuple
            The desired state of the body.
        '''
        assert self.name == state.name, \
            'state name "{}" != body name "{}"'.format(state.name, self.name)
        self.position = state.position
        self.quaternion = state.quaternion
        self.linear_velocity = state.linear_velocity
        self.angular_velocity = state.angular_velocity

    @property
    def position(self):
        '''The (x, y, z) coordinates of the center of this body.'''
        return self.ode_body.getPosition()

    @position.setter
    def position(self, position):
        '''Set the (x, y, z) coordinates of the center of this body.

        Parameters
        ----------
        position : 3-tuple of float
            The coordinates of the desired center of this body.
        '''
        self.ode_body.setPosition(position)

    @property
    def rotation(self):
        '''The rotation matrix for this body.'''
        return self.ode_body.getRotation()

    @rotation.setter
    def rotation(self, rotation):
        '''Set the rotation of this body using a rotation matrix.

        Parameters
        ----------
        rotation : sequence of 9 floats
            The desired rotation matrix for this body.
        '''
        self.ode_body.setRotation(rotation)

    @property
    def quaternion(self):
        '''The (w, x, y, z) rotation quaternion for this body.'''
        return self.ode_body.getQuaternion()

    @quaternion.setter
    def quaternion(self, quaternion):
        self.ode_body.setQuaternion(quaternion)

    @property
    def linear_velocity(self):
        '''Current linear velocity of this body (in world coordinates).'''
        return self.ode_body.getLinearVel()

    @linear_velocity.setter
    def linear_velocity(self, velocity):
        '''Set the linear velocity for this body.

        Parameters
        ----------
        velocity : 3-tuple of float
            The desired velocity for this body, in world coordinates.
        '''
        self.ode_body.setLinearVel(velocity)

    @property
    def angular_velocity(self):
        '''Current angular velocity of this body (in world coordinates).'''
        return self.ode_body.getAngularVel()

    @angular_velocity.setter
    def angular_velocity(self, velocity):
        '''Set the angular velocity for this body.

        Parameters
        ----------
        velocity : 3-tuple of float
            The desired angular velocity for this body, in world coordinates.
        '''
        self.ode_body.setAngularVel(velocity)

    @property
    def force(self):
        '''Current net force acting on this body (in world coordinates).'''
        return self.ode_body.getForce()

    @force.setter
    def force(self, force):
        '''Set the force acting on this body.

        Parameters
        ----------
        force : 3-tuple of float
            The desired force acting on this body, in world coordinates.
        '''
        self.ode_body.setForce(force)

    @property
    def torque(self):
        '''Current net torque acting on this body (in world coordinates).'''
        return self.ode_body.getTorque()

    @torque.setter
    def torque(self, torque):
        '''Set the torque acting on this body.

        Parameters
        ----------
        torque : 3-tuple of float
            The desired torque acting on this body, in world coordinates.
        '''
        self.ode_body.setTorque(torque)

    @property
    def is_kinematic(self):
        '''True iff this body is kinematic.'''
        return self.ode_body.isKinematic()

    @is_kinematic.setter
    def is_kinematic(self, is_kinematic):
        '''Set the kinematic/dynamic attribute for this body.

        In pagoda, kinematic bodies have infinite mass and do interact with
        other bodies via collisions.

        Parameters
        ----------
        is_kinematic : bool
            If True, this body will be set to kinematic. If False, it will be
            set to dynamic.
        '''
        if is_kinematic:
            self.ode_body.setKinematic()
        else:
            self.ode_body.setDynamic()

    @property
    def follows_gravity(self):
        '''True iff this body follows gravity.'''
        return self.ode_body.getGravityMode()

    @follows_gravity.setter
    def follows_gravity(self, follows_gravity):
        '''Set whether this body follows gravity.

        Parameters
        ----------
        follows_gravity : bool
            This body will follow gravity iff this parameter is True.
        '''
        self.ode_body.setGravityMode(follows_gravity)

    def rotate_to_body(self, x):
        '''Rotate the given vector to the same orientation as this body.

        Parameters
        ----------
        x : 3-tuple of float
            A point in three dimensions.

        Returns
        -------
        xrot : 3-tuple of float
            The same point after rotation into the orientation of this body.
        '''
        return np.dot(x, np.array(self.rotation).reshape((3, 3)))

    def body_to_world(self, position):
        '''Convert a body-relative offset to world coordinates.

        Parameters
        ----------
        position : 3-tuple of float
            A tuple giving body-relative offsets.

        Returns
        -------
        position : 3-tuple of float
            A tuple giving the world coordinates of the given offset.
        '''
        return self.ode_body.getRelPointPos(position)

    def world_to_body(self, position):
        '''Convert a point in world coordinates to a body-relative offset.

        Parameters
        ----------
        position : 3-tuple of float
            A world coordinates position.

        Returns
        -------
        offset : 3-tuple of float
            A tuple giving the body-relative offset of the given position.
        '''
        return self.ode_body.getPosRelPoint(position)

    def relative_offset_to_world(self, offset):
        '''Convert a relative body offset to world coordinates.

        Parameters
        ----------
        offset : 3-tuple of float
            The offset of the desired point, given as a relative fraction of the
            size of this body. For example, offset (0, 0, 0) is the center of
            the body, while (0.5, -0.2, 0.1) describes a point halfway from the
            center towards the maximum x-extent of the body, 20% of the way from
            the center towards the minimum y-extent, and 10% of the way from the
            center towards the maximum z-extent.

        Returns
        -------
        position : 3-tuple of float
            A position in world coordinates of the given body offset.
        '''
        return self.body_to_world(offset * self.dimensions / 2)

    def add_force(self, force, relative=False, position=None, relative_position=None):
        '''Add a force to this body.

        Parameters
        ----------
        force : 3-tuple of float
            A vector giving the forces along each world or body coordinate axis.
        relative : bool, optional
            If False, the force values are assumed to be given in the world
            coordinate frame. If True, they are assumed to be given in the
            body-relative coordinate frame. Defaults to False.
        position : 3-tuple of float, optional
            If given, apply the force at this location in world coordinates.
            Defaults to the current position of the body.
        relative_position : 3-tuple of float, optional
            If given, apply the force at this relative location on the body. If
            given, this method ignores the ``position`` parameter.
        '''
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
        '''Add a torque to this body.

        Parameters
        ----------
        force : 3-tuple of float
            A vector giving the torque along each world or body coordinate axis.
        relative : bool, optional
            If False, the torque values are assumed to be given in the world
            coordinate frame. If True, they are assumed to be given in the
            body-relative coordinate frame. Defaults to False.
        '''
        op = self.ode_body.addRelTorque if relative else self.ode_body.addTorque
        op(torque)

    def join_to(self, joint, other_body=None, **kwargs):
        '''Connect this body to another one using a joint.

        This method creates a joint to fasten this body to the other one. See
        :func:`World.join`.

        Parameters
        ----------
        joint : str
            The type of joint to use when connecting these bodies.
        other_body : :class:`Body` or str, optional
            The other body to join with this one. If not given, connects this
            body to the world.
        '''
        self.world.join(joint, self, other_body, **kwargs)

    def connect_to(self, joint, other_body, offset=(0, 0, 0), other_offset=(0, 0, 0),
                   **kwargs):
        '''Move another body next to this one and join them together.

        This method will move the ``other_body`` so that the anchor points for
        the joint coincide. It then creates a joint to fasten the two bodies
        together. See :func:`World.move_next_to` and :func:`World.join`.

        Parameters
        ----------
        joint : str
            The type of joint to use when connecting these bodies.
        other_body : :class:`Body` or str
            The other body to join with this one.
        offset : 3-tuple of float, optional
            The body-relative offset where the anchor for the joint should be
            placed. Defaults to (0, 0, 0). See :func:`World.move_next_to` for a
            description of how offsets are specified.
        other_offset : 3-tuple of float, optional
            The offset on the second body where the joint anchor should be
            placed. Defaults to (0, 0, 0). Like ``offset``, this is given as an
            offset relative to the size and shape of ``other_body``.
        '''
        anchor = self.world.move_next_to(self, other_body, offset, other_offset)
        self.world.join(joint, self, other_body, anchor=anchor, **kwargs)


class Box(Body):
    @property
    def lengths(self):
        return self.shape['lengths']

    @property
    def dimensions(self):
        return np.array(self.lengths).squeeze()

    @property
    def volume(self):
        return np.prod(self.lengths)

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

    @property
    def volume(self):
        return 4 / 3 * np.pi * self.radius ** 3

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
        return np.array([d, d, self.length]).squeeze()

    @property
    def volume(self):
        return self.length * np.pi * self.radius ** 2

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

    @property
    def volume(self):
        return 4 / 3 * np.pi * self.radius ** 3 + \
            self.length * np.pi * self.radius ** 2

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


class Constraints(object):
    '''This class wraps an ODE entity that constrains body movement.

    In ODE, :class:`Body` objects represent mass/inertia properties, while
    :class:`Joint` and :class:`Motor` objects represent mathematical constraints
    that govern how specific pairs of bodies interact. For example, a
    :class:`BallJoint` that connects two bodies will force the anchor point for
    those two bodies to remain in the same location in space -- any linear force
    that displaces one of the bodies will also cause a force to be applied to
    the second body, because of the constraint imposed by the ball joint. As
    another example, a :class:`Slider` that connects two bodies allows those two
    bodies to displace relative to one another along a single axis, but not to
    rotate with respect to one another -- any torque applied to one body will
    also cause a torque to be applied to the other body.

    Constraints can be applied to angular degrees of freedom (e.g.,
    :class:`AMotor`), linear degrees of freedom (e.g., :class:`BallJoint`,
    :class:`LMotor`), or both (e.g., :class:`PistonJoint`).

    Both joints and motors apply constraints to pairs of bodies, but they are
    quite different in many ways and so are represented using specific
    subclasses. This superclass is just a mixin to avoid repeating the getters
    and setters that are common between motors and joints.
    '''

    ADOF = 0
    LDOF = 0

    @property
    def feedback(self):
        '''Feedback buffer (list of 3-tuples) for this ODE motor/joint.'''
        return self.ode_obj.getFeedback()

    @property
    def positions(self):
        '''List of positions for linear degrees of freedom.'''
        return [self.ode_obj.getPosition(i) for i in range(self.LDOF)]

    @property
    def position_rates(self):
        '''List of position rates for linear degrees of freedom.'''
        return [self.ode_obj.getPositionRate(i) for i in range(self.LDOF)]

    @property
    def angles(self):
        '''List of angles for rotational degrees of freedom.'''
        return [self.ode_obj.getAngle(i) for i in range(self.ADOF)]

    @property
    def angle_rates(self):
        '''List of angle rates for rotational degrees of freedom.'''
        return [self.ode_obj.getAngleRate(i) for i in range(self.ADOF)]

    @property
    def axes(self):
        '''List of axes for this object's degrees of freedom.'''
        def ax(i):
            return dict(rel=self.ode_obj.getAxisRel(i),
                        axis=self.ode_obj.getAxis(i))
        return [ax(i) for i in range(self.ADOF)]

    @axes.setter
    def axes(self, axes):
        '''Set the axes for this object's degrees of freedom.

        Parameters
        ----------
        axes : list of None, 3 floats, or dict
            A list of axis values to set. This list must have the same number of
            elements as the degrees of freedom of the underlying ODE object.
            Each element can be

            (a) None, which has no effect on the corresponding axis,
            (b) three floats specifying the axis to set, or
            (c) a dictionary with an "axis" key specifying the axis to set and a
                "rel" key specifying the relative body to set the axis on.
        '''
        assert self.ADOF == len(axes)
        for i, axis in enumerate(axes):
            rel = 0
            if isinstance(axis, dict):
                rel = axis.get('rel', 0)
                axis = axis.get('axis')
            if axis is not None:
                self.ode_obj.setAxis(i, rel, axis)

    @property
    def lo_stops(self):
        '''List of lo stop values for this object's degrees of freedom.'''
        return _get_params(self.ode_obj, 'LoStop', self.ADOF)

    @lo_stops.setter
    def lo_stops(self, lo_stops):
        '''Set the lo stop values for this object's degrees of freedom.

        Parameters
        ----------
        lo_stops : float or sequence of float
            A lo stop value to set on all degrees of freedom, or a list
            containing one such value for each degree of freedom. For rotational
            degrees of freedom, these values must be in radians.
        '''
        _set_params(self.ode_obj, 'LoStop', lo_stops, self.ADOF)

    @property
    def hi_stops(self):
        '''List of hi stop values for this object's degrees of freedom.'''
        return _get_params(self.ode_obj, 'HiStop', self.ADOF)

    @hi_stops.setter
    def hi_stops(self, hi_stops):
        '''Set the hi stop values for this object's degrees of freedom.

        Parameters
        ----------
        hi_stops : float or sequence of float
            A hi stop value to set on all degrees of freedom, or a list
            containing one such value for each degree of freedom. For rotational
            degrees of freedom, these values must be in radians.
        '''
        _set_params(self.ode_obj, 'HiStop', hi_stops, self.ADOF)

    @property
    def velocities(self):
        '''List of target velocity values for rotational degrees of freedom.'''
        return _get_params(self.ode_obj, 'Vel', self.ADOF)

    @velocities.setter
    def velocities(self, velocities):
        '''Set the target velocities for this object's degrees of freedom.

        Parameters
        ----------
        velocities : float or sequence of float
            A target velocity value to set on all degrees of freedom, or a list
            containing one such value for each degree of freedom. For rotational
            degrees of freedom, these values must be in radians / second.
        '''
        _set_params(self.ode_obj, 'Vel', velocities, self.ADOF)

    @property
    def max_forces(self):
        '''List of max force values for rotational degrees of freedom.'''
        return _get_params(self.ode_obj, 'FMax', self.ADOF)

    @max_forces.setter
    def max_forces(self, max_forces):
        '''Set the maximum forces for this object's degrees of freedom.

        Parameters
        ----------
        max_forces : float or sequence of float
            A maximum force value to set on all degrees of freedom, or a list
            containing one such value for each degree of freedom.
        '''
        _set_params(self.ode_obj, 'FMax', max_forces, self.ADOF)

    @property
    def cfms(self):
        '''List of CFM values for this object's degrees of freedom.'''
        return _get_params(self.ode_obj, 'CFM', self.ADOF)

    @cfms.setter
    def cfms(self, cfms):
        '''Set the CFM values for this object's degrees of freedom.

        Parameters
        ----------
        cfms : float or sequence of float
            A CFM value to set on all degrees of freedom, or a list
            containing one such value for each degree of freedom.
        '''
        _set_params(self.ode_obj, 'CFM', cfms, self.ADOF)

    @property
    def stop_cfms(self):
        '''List of lo/hi stop CFM values.'''
        return _get_params(self.ode_obj, 'StopCFM', self.ADOF)

    @stop_cfms.setter
    def stop_cfms(self, stop_cfms):
        '''Set the CFM values for this object's DOF limits.

        Parameters
        ----------
        stop_cfms : float or sequence of float
            A CFM value to set on all degrees of freedom limits, or a list
            containing one such value for each degree of freedom limit.
        '''
        _set_params(self.ode_obj, 'StopCFM', stop_cfms, self.ADOF)

    @property
    def stop_erps(self):
        '''List of lo/hi stop ERP values.'''
        return _get_params(self.ode_obj, 'StopERP', self.ADOF)

    @stop_erps.setter
    def stop_erps(self, stop_erps):
        '''Set the ERP values for this object's DOF limits.

        Parameters
        ----------
        stop_erps : float or sequence of float
            An ERP value to set on all degrees of freedom limits, or a list
            containing one such value for each degree of freedom limit.
        '''
        _set_params(self.ode_obj, 'StopERP', stop_erps, self.ADOF)

    def enable_feedback(self):
        '''Enable feedback on this ODE object.'''
        self.ode_obj.setFeedback(True)

    def disable_feedback(self):
        '''Disable feedback on this ODE object.'''
        self.ode_obj.setFeedback(False)


class Motor(Constraints):
    '''This class wraps an ODE motor -- either an LMotor or an AMotor.

    Parameters
    ----------
    name : str
        A name for this object in the world.
    world : :class:`World`
        A world object to which this motor belongs.
    body_a : :class:`Body`
        A first body connected to this joint.
    body_b : :class:`Body`, optional
        A second body connected to this joint. If not given, the joint will
        connect the first body to the world.
    feedback : bool, optional
        Feedback will be enabled on this motor iff this is True. Defaults to
        False.
    dof : int, optional
        Number of degrees of freedom in this motor. Defaults to 3.
    jointgroup : ode.JointGroup, optional
        A joint group to which this motor belongs. Defaults to the default joint
        group in the world.
    '''

    def __init__(self, name, world, body_a, body_b=None, feedback=False, dof=3,
                 jointgroup=None):
        self.name = name
        self.ode_obj = self.MOTOR_FACTORY(world.ode_world, jointgroup=jointgroup)
        self.ode_obj.attach(body_a.ode_body, body_b.ode_body if body_b else None)
        self.ode_obj.setNumAxes(dof)
        self.cfms = 1e-8
        if feedback:
            self.enable_feedback()
        else:
            self.disable_feedback()


class AMotor(Motor):
    '''An angular motor applies torques to change an angle in the physics world.

    AMotors can be created in "user" mode---in which case the user must supply
    all axis and angle values---or, for 3-DOF motors, in "euler" mode---in which
    case the first and last axes must be specified, and ODE computes the middle
    axis automatically.
    '''

    MOTOR_FACTORY = ode.AMotor

    def __init__(self, *args, **kwargs):
        mode = kwargs.pop('mode', 'user')
        super(AMotor, self).__init__(*args, **kwargs)
        if isinstance(mode, str):
            if self.ADOF == 3 and mode.lower().startswith('e'):
                mode = ode.AMotorEuler
            else:
                mode = ode.AMotorUser
        self.ode_obj.setMode(mode)

    @property
    def ADOF(self):
        '''Number of angular degrees of freedom for this motor.'''
        return self.ode_obj.getNumAxes()

    def add_torques(self, torques):
        '''Add the given torques along this motor's axes.

        Parameters
        ----------
        torques : sequence of float
            A sequence of torque values to apply to this motor's axes.
        '''
        self.ode_obj.addTorques(*torques)


class LMotor(Motor):
    '''An LMotor applies forces to change a position in the physics world.'''

    MOTOR_FACTORY = ode.LMotor

    @property
    def LDOF(self):
        '''Number of linear degrees of freedom for this motor.'''
        return self.ode_obj.getNumAxes()


class Joint(Constraints):
    '''This class wraps the ODE Joint class with some Python properties.

    Parameters
    ----------
    name : str
        Name of the joint to create. This is only to make the joint discoverable
        in the world.
    world : :class:`World`
        Wrapper for the world in which this joint exists.
    body_a : :class:`Body`
        Wrapper for the first body that this joint connects.
    body_b : :class:`Body`, optional
        Wrapper for the second body that this joint connects. If this is None,
        the joint will connect ``body_a`` to the ``world``.
    anchor : 3-tuple of floats, optional
        Anchor in world coordinates for the joint. Optional for :class:`Fixed`
        joint.
    feedback : bool, optional
        If this is True, a force feedback structure will be enabled for this
        joint, which will make it possible to record the forces that this joint
        exerts on its two bodies. By default, no structure will be allocated.
    jointgroup : ODE joint group, optional
        Add the joint to this group. Defaults to the default world joint group.
    '''

    def __init__(self, name, world, body_a, body_b=None, anchor=None, feedback=False,
                 jointgroup=None):
        self.name = name

        build = getattr(ode, '{}Joint'.format(self.__class__.__name__))
        self.ode_obj = build(world.ode_world, jointgroup=jointgroup)
        self.ode_obj.attach(body_a.ode_body, body_b.ode_body if body_b else None)
        if anchor is not None:
            self.ode_obj.setAnchor(anchor)
            self.ode_obj.setParam(ode.ParamCFM, 0)

        self.amotor = None
        if self.ADOF > 0:
            self.amotor = AMotor(name=name + ':amotor',
                                 world=world,
                                 body_a=body_a,
                                 body_b=body_b,
                                 feedback=feedback,
                                 jointgroup=jointgroup,
                                 dof=self.ADOF,
                                 mode='euler' if self.ADOF == 3 else 'user')

        self.lmotor = None
        if self.LDOF > 0:
            self.lmotor = LMotor(name=name + ':lmotor',
                                 world=world,
                                 body_a=body_a,
                                 body_b=body_b,
                                 feedback=feedback,
                                 jointgroup=jointgroup,
                                 dof=self.LDOF)

        if feedback:
            self.enable_feedback()
        else:
            self.disable_feedback()

    def __str__(self):
        return self.name

    @property
    def anchor(self):
        '''3-tuple specifying location of this joint's anchor.'''
        return self.ode_obj.getAnchor()

    @property
    def anchor2(self):
        '''3-tuple specifying location of the anchor on the second body.'''
        return self.ode_obj.getAnchor2()

    def add_torques(self, *torques):
        '''Add the given torques along this joint's axes.

        Parameters
        ----------
        torques : sequence of float
            A sequence of torque values to apply to this motor's axes.
        '''
        self.amotor.add_torques(*torques)


class Fixed(Joint):
    ADOF = 0
    LDOF = 0


class Slider(Joint):
    ADOF = 0
    LDOF = 1

    @property
    def positions(self):
        '''List of positions for this joint's linear degrees of freedom.'''
        return [self.ode_obj.getPosition()]

    @property
    def position_rates(self):
        '''List of position rates for this joint's degrees of freedom.'''
        return [self.ode_obj.getPositionRate()]

    @property
    def axes(self):
        '''Axis of displacement for this joint.'''
        return [self.ode_obj.getAxis()]

    @axes.setter
    def axes(self, axes):
        '''Set the linear axis of displacement for this joint.

        Parameters
        ----------
        axes : list containing one 3-tuple of floats
            A list of the axes for this joint. For a slider joint, which has one
            degree of freedom, this must contain one 3-tuple specifying the X,
            Y, and Z axis for the joint.
        '''
        self.lmotor.axes = [dict(rel=1, axis=axes[0])]
        self.ode_obj.setAxis(axes[0])


class Hinge(Joint):
    ADOF = 1
    LDOF = 0

    @property
    def angles(self):
        '''List of angles for this joint's rotational degrees of freedom.'''
        return [self.ode_obj.getAngle()]

    @property
    def angle_rates(self):
        '''List of angle rates for this joint's degrees of freedom.'''
        return [self.ode_obj.getAngleRate()]

    @property
    def axes(self):
        '''Axis of rotation for this joint.'''
        return [self.ode_obj.getAxis()]

    @axes.setter
    def axes(self, axes):
        '''Set the angular axis of rotation for this joint.

        Parameters
        ----------
        axes : list containing one 3-tuple of floats
            A list of the axes for this joint. For a hinge joint, which has one
            degree of freedom, this must contain one 3-tuple specifying the X,
            Y, and Z axis for the joint.
        '''
        self.amotor.axes = [dict(rel=1, axis=axes[0])]
        self.ode_obj.setAxis(axes[0])


class Piston(Joint):
    ADOF = 1
    LDOF = 1

    @property
    def axes(self):
        '''Axis of rotation and displacement for this joint.'''
        return [self.ode_obj.getAxis()]

    @axes.setter
    def axes(self, axes):
        self.amotor.axes = [dict(rel=1, axis=axes[0])]
        self.lmotor.axes = [dict(rel=1, axis=axes[0])]
        self.ode_obj.setAxis(axes[0])


class Universal(Joint):
    ADOF = 2
    LDOF = 0

    @property
    def axes(self):
        '''A list of axes of rotation for this joint.'''
        return [self.ode_obj.getAxis1(), self.ode_obj.getAxis2()]

    @axes.setter
    def axes(self, axes):
        self.amotor.axes = dict(rel=1, axis=axes[0]), dict(rel=2, axis=axes[1])
        setters = [self.ode_obj.setAxis1, self.ode_obj.setAxis2]
        for axis, setter in zip(axes, setters):
            if axis is not None:
                setter(axis)

    @property
    def angles(self):
        '''A list of two angles for this joint's degrees of freedom.'''
        return [self.ode_obj.getAngle1(), self.ode_obj.getAngle2()]

    @property
    def angle_rates(self):
        '''A list of two angle rates for this joint's degrees of freedom.'''
        return [self.ode_obj.getAngle1Rate(), self.ode_obj.getAngle2Rate()]


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
        self.cfm = 1e-6
        self.erp = 0.7

        self._bodies = {}
        self._joints = {}

    @property
    def gravity(self):
        '''Current gravity vector in the world.'''
        return self.ode_world.getGravity()

    @gravity.setter
    def gravity(self, gravity):
        '''Set the gravity vector in the world.

        Parameters
        ----------
        gravity : 3-tuple of float
            The vector where gravity should point.
        '''
        return self.ode_world.setGravity(gravity)

    @property
    def cfm(self):
        '''Current global CFM value.'''
        return self.ode_world.getCFM()

    @cfm.setter
    def cfm(self, cfm):
        '''Set the global CFM value.

        Parameters
        ----------
        cfm : float
            The desired global CFM value.
        '''
        return self.ode_world.setCFM(cfm)

    @property
    def erp(self):
        '''Current global ERP value.'''
        return self.ode_world.getERP()

    @erp.setter
    def erp(self, erp):
        '''Set the global ERP value.

        Parameters
        ----------
        erp : float
            The desired global ERP value.
        '''
        return self.ode_world.setERP(erp)

    @property
    def bodies(self):
        '''Sequence of all bodies in the world, sorted by name.'''
        for k in sorted(self._bodies):
            yield self._bodies[k]

    @property
    def joints(self):
        '''Sequence of all joints in the world, sorted by name.'''
        for k in sorted(self._joints):
            yield self._joints[k]

    def get_body(self, key):
        '''Get a body by key.

        Parameters
        ----------
        key : str, None, or :class:`Body`
            The key for looking up a body. If this is None or a :class:`Body`
            instance, the key itself will be returned.

        Returns
        -------
        body : :class:`Body`
            The body in the world with the given key.
        '''
        return self._bodies.get(key, key)

    def get_joint(self, key):
        '''Get a joint by key.

        Parameters
        ----------
        key : str
            The key for a joint to look up.

        Returns
        -------
        joint : :class:`Joint`
            The joint in the world with the given key, or None if there is no
            such joint.
        '''
        return self._joints.get(key, None)

    def create_body(self, shape, name=None, **kwargs):
        '''Create a new body.

        Parameters
        ----------
        shape : str
            The "shape" of the body to be created. This should name a type of
            body object, e.g., "box" or "cap".
        name : str, optional
            The name to use for this body. If not given, a default name will be
            constructed of the form "{shape}{# of objects in the world}".

        Returns
        -------
        body : :class:`Body`
            The created body object.
        '''
        shape = shape.lower()
        if name is None:
            for i in range(1 + len(self._bodies)):
                name = '{}{}'.format(shape, i)
                if name not in self._bodies:
                    break
        self._bodies[name] = BODIES[shape](name, self, **kwargs)
        return self._bodies[name]

    def join(self, shape, body_a, body_b=None, name=None, **kwargs):
        '''Create a new joint that connects two bodies together.

        Parameters
        ----------
        shape : str
            The "shape" of the joint to use for joining together two bodies.
            This should name a type of joint, such as "ball" or "piston".
        body_a : str or :class:`Body`
            The first body to join together with this joint. If a string is
            given, it will be used as the name of a body to look up in the
            world.
        body_b : str or :class:`Body`, optional
            If given, identifies the second body to join together with
            ``body_a``. If not given, ``body_a`` is joined to the world.
        name : str, optional
            If given, use this name for the created joint. If not given, a name
            will be constructed of the form
            "{body_a.name}^{shape}^{body_b.name}".

        Returns
        -------
        joint : :class:`Joint`
            The joint object that was created.
        '''
        ba = self.get_body(body_a)
        bb = self.get_body(body_b)
        shape = shape.lower()
        if name is None:
            name = '{}^{}^{}'.format(ba.name, shape, bb.name if bb else '')
        self._joints[name] = JOINTS[shape](name, self, body_a=ba, body_b=bb, **kwargs)
        return self._joints[name]

    def move_next_to(self, body_a, body_b, offset_a, offset_b):
        '''Move one body to be near another one.

        After moving, the location described by ``offset_a`` on ``body_a`` will
        be coincident with the location described by ``offset_b`` on ``body_b``.

        Parameters
        ----------
        body_a : str or :class:`Body`
            The body to use as a reference for moving the other body. If this is
            a string, it is treated as the name of a body to look up in the
            world.
        body_b : str or :class:`Body`
            The body to move next to ``body_a``. If this is a string, it is
            treated as the name of a body to look up in the world.
        offset_a : 3-tuple of float
            The offset of the anchor point, given as a relative fraction of the
            size of ``body_a``. See :func:`Body.relative_offset_to_world`.
        offset_b : 3-tuple of float
            The offset of the anchor point, given as a relative fraction of the
            size of ``body_b``.

        Returns
        -------
        anchor : 3-tuple of float
            The location of the shared point, which is often useful to use as a
            joint anchor.
        '''
        ba = self.get_body(body_a)
        bb = self.get_body(body_b)
        if ba is None:
            return bb.relative_offset_to_world(offset_b)
        if bb is None:
            return ba.relative_offset_to_world(offset_a)
        anchor = ba.relative_offset_to_world(offset_a)
        offset = bb.relative_offset_to_world(offset_b)
        bb.position = np.asarray(bb.position) + anchor - offset
        return anchor

    def get_body_states(self):
        '''Return the complete state of all bodies in the world.

        Returns
        -------
        states : list of state information tuples
            A list of body state information for each body in the world. See
            :func:`Body.state`.
        '''
        return [b.state for b in self.bodies]

    def set_body_states(self, states):
        '''Set the states of some bodies in the world.

        Parameters
        ----------
        states : sequence of states
            A complete state tuple for one or more bodies in the world. See
            :func:`get_body_states`.
        '''
        for state in states:
            self.get_body(state.name).state = state

    def step(self, substeps=2):
        '''Step the world forward by one frame.

        Parameters
        ----------
        substeps : int, optional
            Split the step into this many sub-steps. This helps to prevent the
            time delta for an update from being too large.
        '''
        self.frame_no += 1
        dt = self.dt / substeps
        for _ in range(substeps):
            self.ode_contactgroup.empty()
            self.ode_space.collide(None, self.on_collision)
            self.ode_world.step(dt)

    def are_connected(self, body_a, body_b):
        '''Determine whether the given bodies are currently connected.

        Parameters
        ----------
        body_a : str or :class:`Body`
            One body to test for connectedness. If this is a string, it is
            treated as the name of a body to look up.
        body_b : str or :class:`Body`
            One body to test for connectedness. If this is a string, it is
            treated as the name of a body to look up.

        Returns
        -------
        connected : bool
            Return True iff the two bodies are connected.
        '''
        return bool(ode.areConnected(
            self.get_body(body_a).ode_body,
            self.get_body(body_b).ode_body))

    def on_collision(self, args, geom_a, geom_b):
        '''Callback function for the collide() method.

        Parameters
        ----------
        args : None
            Arguments passed when the callback was registered. Not used.
        geom_a : ODE geometry
            The geometry object of one of the bodies that has collided.
        geom_b : ODE geometry
            The geometry object of one of the bodies that has collided.
        '''
        body_a = geom_a.getBody()
        body_b = geom_b.getBody()
        if ode.areConnected(body_a, body_b) or \
           (body_a and body_a.isKinematic()) or \
           (body_b and body_b.isKinematic()):
            return
        for c in ode.collide(geom_a, geom_b):
            c.setBounce(self.elasticity)
            c.setMu(self.friction)
            ode.ContactJoint(self.ode_world, self.ode_contactgroup, c).attach(
                geom_a.getBody(), geom_b.getBody())
