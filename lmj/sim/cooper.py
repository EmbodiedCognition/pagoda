'''Python implementation of forward-dynamics solver by Joseph Cooper.'''

import lmj.c3d
import lmj.cli
import lmj.pid
import numpy as np
import numpy.random as rng
import ode
import re

from . import physics

logging = lmj.cli.get_logger(__name__)


class DataSeries(object):
    def __init__(self, filename=None, num_frames=0, num_dofs=0):
        self.data = None
        if filename:
            self.load(filename)
        elif num_frames:
            self.data = np.zeros((num_frames, num_dofs), float)
        self.num_frames = len(self.data)
        self.num_dofs = len(self.data[0])

    def load(self, filename):
        if filename.endswith('.c3d'):
            reader = lmj.c3d.Reader(filename)
            raise NotImplementedError
        elif filename.endswith('.npy'):
            self.data = np.load(filename)
        else:
            self.data = np.loadtxt(filename)

    def save(self, filename):
        if filename.endswith('.npy'):
            np.save(filename, self.data)
        else:
            np.savetxt(filename, self.data)

    @classmethod
    def like(cls, dataset):
        return cls(num_frames=dataset.num_frames, num_dofs=dataset.num_dofs)


class Parser(object):
    def __init__(self, world, filename):
        self.world = world
        self.filename = filename

        self.config = []
        self.index = 0
        self.root = None

        with open(filename) as handle:
            for i, line in enumerate(handle):
                for j, token in enumerate(line.split('#')[0].strip().split()):
                    if token.strip():
                        self.config.append((i, j, token))

    def error(self, msg):
        lineno, tokenno, token = self.config[self.index - 1]
        logging.fatal('%s:%d:%d: error parsing "%s": %s',
                      self.filename, lineno+1, tokenno+1, token, msg)

    def next_token(self, expect=None, lower=True, dtype=None):
        if self.index < len(self.config):
            _, _, token = self.config[self.index]
            self.index += 1
            if lower:
                token = token.lower()
            if expect and re.match(expect, token) is None:
                return self.error('expected {}'.format(expect))
            if callable(dtype):
                token = dtype(token)
            return token
        return None

    def peek_token(self):
        if self.index < len(self.config):
            return self.config[self.index][-1]
        return None

    def next_float(self):
        return self.next_token(expect=r'^-?\d+(\.\d*)?$', dtype=float)

    def array(self, n=3):
        return np.array([self.next_float() for _ in range(n)])

    def handle_body(self, namespace):
        shape = self.next_token(expect='^({})$'.format('|'.join(physics.BODIES)))
        name = namespace + self.next_token(lower=False)

        kwargs = {}
        quaternion = None
        position = None
        token = self.next_token()
        while token:
            if token in ('body', 'join'):
                break
            if token == 'lengths':
                kwargs[token] = self.array()
            if token == 'radius':
                kwargs[token] = self.next_float()
            if token == 'length':
                kwargs[token] = self.next_float()
            if token == 'quaternion':
                theta, x, y, z = self.array(4)
                quaternion = physics.make_quaternion(physics.TAU * theta / 360, x, y, z)
            if token == 'position':
                position = self.array()
            if token == 'root':
                logging.info('"%s" will be used as the root', name)
                self.root = name
            token = self.next_token()

        logging.info('creating %s %s %s', shape, name, kwargs)

        body = self.world.create_body(shape, name, **kwargs)

        if quaternion is not None:
            logging.info('setting rotation %s', quaternion)
            body.quaternion = quaternion

        if position is not None:
            logging.info('setting position %s', position)
            body.position = position

        return token

    def handle_joint(self, namespace):
        shape = self.next_token(expect='^({})$'.format('|'.join(physics.JOINTS)))
        body1 = namespace + self.next_token(lower=False)
        offset1 = self.array()
        body2 = namespace + self.next_token(lower=False)
        offset2 = self.array()

        anchor = self.world.move_next_to(body1, body2, offset1, offset2)

        token = self.next_token()
        axes = [dict(rel=1, axis=(1, 0, 0)),
                dict(rel=2, axis=(0, 1, 0)),
                dict(rel=2, axis=(0, 0, 1))]
        lo_stops = None
        hi_stops = None
        while token:
            if token in ('body', 'join'):
                break
            if token.startswith('axis'):
                ax = axes[int(token.replace('axis', ''))]
                ax['axis'] = self.array()
                if self.peek_token() == 'rel':
                    rel = self.next_token()
                    ax['rel'] = self.next_token(expect='^\d$', dtype=int)
            if token == 'lo_stops':
                lo_stops = physics.TAU * self.array(physics.JOINTS[shape].ADOF) / 360
            if token == 'hi_stops':
                hi_stops = physics.TAU * self.array(physics.JOINTS[shape].ADOF) / 360
            token = self.next_token()

        logging.info('joining %s %s %s', shape, body1, body2)

        joint = self.world.join(shape, body1, body2, anchor=anchor)
        joint.axes = axes
        if lo_stops is not None:
            joint.lo_stops = lo_stops
        if hi_stops is not None:
            joint.hi_stops = hi_stops

        # we add some additional attributes for controlling this joint
        joint.target_angles = [None] * joint.ADOF
        joint.controllers = [lmj.pid.Controller(kp=0.9) for i in range(joint.ADOF)]

        return token

    def create(self, namespace=''):
        token = self.next_token(expect='^(body|joint)$')
        while token is not None:
            if token == 'body':
                token = self.handle_body(namespace)
            elif token == 'join':
                token = self.handle_joint(namespace)
            else:
                self.error('unexpected token')
        return self.root


def create_skeleton(world, filename, namespace='', cfm=1e-10, max_force=250):
    if namespace and namespace[0] not in '.:-':
        namespace += '.'

    p = Parser(world, filename)

    root = world.get_body(p.create(namespace))

    lm = world.join(namespace + 'lmotor', root)
    am = world.join(namespace + 'amotor', root)
    al = world.join(namespace + 'amotor', root)

    for joint in world.joints:
        if joint.name.startswith(namespace):
            joint.cfms = cfm
            joint.max_forces = max_force

    lm.velocities = 0
    lm.axes = [dict(rel=0, axis=(1, 0, 0)),
               dict(rel=0, axis=(0, 1, 0)),
               dict(rel=0, axis=(0, 0, 1))]
    am.velocities = 0
    am.axes = [dict(rel=0, axis=(1, 0, 0)),
               dict(rel=0, axis=(0, 1, 0)),
               dict(rel=0, axis=(0, 0, 1))]
    al.lo_stops = -2 * physics.TAU / 9
    al.hi_stops = 2 * physics.TAU / 9


class World(physics.World):
    @property
    def num_dofs(self):
        return sum(j.ADOF + j.LDOF for j in self.joints)

    def set_random_forces(self):
        for body in self.bodies:
            body.add_force(self.FMAX * rng.randn(3))

    def reset(self):
        for b in self.bodies:
            b.position = b.position + np.array([0, 0, 2])

    def load_marker_attachments(self, filename):
        pass

    def markers_to_angles(self, markers):
        '''Follow a set of marker data.'''
        if isinstance(markers, str):
            markers = Dataset(markers)
        state_sequence = []
        smoothed_markers = Dataset.like(markers)
        angles = Dataset(num_frames=markers.num_frames, num_dofs=self.num_dofs)
        for i, frame in enumerate(markers):
            self.contactgroup.empty()
            self.space.collide(None, self.on_collision)
            states_sequence.append(self.get_body_states())
            self.set_body_states(state_sequence[-1])

            # update the positions and velocities of the markers.
            markers # XXX

            # update the ode world.
            self.world.step(self.dt)

            # record the marker positions on the body.
            smoothed_markers[i] = 0 # XXX

            # recocrd the angles of each joint in the body.
            j = 0
            for joint in self.joints:
                angles[i, j:j+joint.ADOF] = joint.angles
                j += joint.ADOF

        return state_sequence, smoothed_markers, angles

    def angles_to_torques(self, angles):
        '''Follow a set of angle data.'''
        state_sequence = []
        torques = Dataset.like(angles)
        for i, frame in enumerate(angles):
            self.contactgroup.empty()
            self.space.collide(None, self.on_collision)
            state_sequence.append(self.get_body_states())
            self.set_body_states(state_sequence[-1])

            # move toward target angles for each joint.
            j = 0
            for joint in self.joints:
                joint.velocities = [
                    ctrl(tgt - cur, self.dt) for cur, tgt, ctrl in
                    zip(joint.angles, frame[j:j+joint.ADOF], joint.controllers)]
                j += joint.ADOF

            # update the ode world.
            self.world.step(self.dt)

            # record the torques that the joints experienced.
            torques[i] = 0 # XXX

        return state_sequence, torques

    def follow_torques(self, torques):
        '''Move the body according to a set of torque data.'''
        state_sequence = []
        for i, frame in enumerate(torques):
            self.contactgroup.empty()
            self.space.collide(None, self.on_collision)
            state_sequence.append(self.get_body_states())
            self.set_body_states(state_sequence[-1])

            self.set_body_states(state_sequence[-1])
            for joint in self.joints:
                joint.max_forces = [0] * joint.ADOF

            j = 0
            for joint in self.joints:
                joint.add_torque(frame[j:j+joint.ADOF])
                j += joint.ADOF

            # update the ode world.
            self.world.step(self.dt)

            #body->restoreControl()  # XXX

        return state_sequence
