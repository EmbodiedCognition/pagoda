import climate
import numpy as np
import ode

from . import parser

logging = climate.get_logger(__name__)


def pid(kp=0., ki=0., kd=0., smooth=0.1):
    '''Create a callable that implements a PID controller.

    A PID controller returns a control signal u(t) given a history of error
    measurements e(0) ... e(t), using proportional (P), integral (I), and
    derivative (D) terms, according to:

    u(t) = kp * e(t) + ki * \int_{s=0}^t e(s) ds + kd * \frac{de(s)}{ds}(t)

    The proportional term is just the current error, the integral term is the
    sum of all error measurements, and the derivative term is the instantaneous
    derivative of the error measurement.

    Parameters
    ----------
    kp : float
        The weight associated with the proportional term of the PID controller.
    ki : float
        The weight associated with the integral term of the PID controller.
    kd : float
        The weight associated with the derivative term of the PID controller.
    smooth : float in [0, 1]
        Derivative values will be smoothed with this exponential average. A
        value of 1 never incorporates new derivative information, a value of 0.5
        uses the mean of the historic and new information, and a value of 0
        discards historic information (i.e., the derivative in this case will be
        unsmoothed). The default is 0.1.

    Returns
    -------
    (error, dt) -> control
        Returns a function that accepts an error measurement and a delta-time
        value since the previous measurement, and returns a control signal.
    '''
    state = dict(p=0, i=0, d=0)
    def control(error, dt=1):
        state['d'] = smooth * state['d'] + (1 - smooth) * (error - state['p']) / dt
        state['i'] += error * dt
        state['p'] = error
        return kp * state['p'] + ki * state['i'] + kd * state['d']
    return control


class Skeleton:
    '''A skeleton is a group of rigid bodies connected with articulated joints.

    Commonly, the skeleton is used to represent an articulated body that is
    capable of mimicking the motion of the human body.
    '''

    def __init__(self, world, filename, pid_params=None):
        '''
        '''
        self.world = world
        self.filename = filename
        self.jointgroup = ode.JointGroup()

        p = parser.Parser(world, filename, self.jointgroup)
        self.roots = [world.get_body(r) for r in p.create()]
        self.bodies = p.bodies
        self.joints = p.joints

        # we add some additional attributes for controlling skeleton joints
        kwargs = dict(kp=1. / world.dt)
        kwargs.update(**(pid_params or {}))
        for joint in self.joints:
            joint.target_angles = [None] * joint.ADOF
            joint.controllers = [pid(**kwargs) for i in range(joint.ADOF)]

    @property
    def num_dofs(self):
        return sum(j.ADOF for j in self.joints)

    @property
    def max_force(self):
        for joint in self.joints:
            return joint.max_forces[0]

    @max_force.setter
    def max_force(self, max_force):
        for joint in self.joints:
            joint.max_forces = max_force

    @property
    def cfm(self):
        for joint in self.joints:
            return joint.cfms[0]

    @cfm.setter
    def cfm(self, cfm):
        for joint in self.joints:
            joint.cfms = cfm

    @property
    def feedback(self):
        for joint in self.joints:
            return joint.amotor.ode_motor.getFeedback()

    @feedback.setter
    def feedback(self, feedback):
        for joint in self.joints:
            joint.amotor.ode_motor.setFeedback(feedback)

    @property
    def velocities(self):
        values = []
        for joint in self.joints:
            values.extend(joint.velocities)
        return values

    @property
    def angles(self):
        values = []
        for joint in self.joints:
            values.extend(joint.angles)
        return values

    @property
    def torques(self):
        values = []
        for joint in self.joints:
            values.extend(joint.amotor.feedback[-1][:joint.ADOF])
        return values

    @property
    def body_positions(self):
        values = []
        for body in self.bodies:
            values.extend(body.position)
        return values

    @property
    def body_linear_velocities(self):
        values = []
        for body in self.bodies:
            values.extend(body.linear_velocity)
        return values

    def indices_for(self, name):
        j = 0
        for joint in self.joints:
            if joint.name == name:
                return list(range(j, j + joint.ADOF))
            j += joint.ADOF
        return []

    def rmse(self):
        deltas = []
        for joint in self.joints:
            delta = np.array(joint.anchor) - joint.anchor2
            deltas.append((delta * delta).sum())
        return np.sqrt(np.mean(deltas))

    def get_body_states(self):
        '''Return a list of the states of all bodies in the skeleton.'''
        return [(b.name,
                 b.position,
                 b.quaternion,
                 b.linear_velocity,
                 b.angular_velocity) for b in self.bodies]

    def set_body_states(self, states):
        '''Set the states of all bodies in the skeleton.'''
        for name, pos, rot, lin, ang in states:
            body = self.world.get_body(name)
            body.position = pos
            body.quaternion = rot
            body.linear_velocity = lin
            body.angular_velocity = ang

    def reset_velocities(self, target=0):
        for joint in self.joints:
            joint.velocities = target

    def enable_motors(self, max_force):
        self.max_force = max_force
        self.feedback = True

    def disable_motors(self):
        self.max_force = 0
        self.feedback = False

    def set_angles(self, angles):
        '''Move each joint toward a target angle.'''
        j = 0
        for joint in self.joints:
            velocities = [
                ctrl(tgt - cur, self.world.dt) for cur, tgt, ctrl in
                zip(joint.angles, angles[j:j+joint.ADOF], joint.controllers)]
            joint.velocities = velocities
            j += joint.ADOF

    def add_torques(self, torques):
        j = 0
        for joint in self.joints:
            joint.add_torques(
                list(torques[j:j+joint.ADOF]) + [0] * (3 - joint.ADOF))
            j += joint.ADOF


