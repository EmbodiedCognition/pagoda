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


def deg(z):
    return 360 * np.asarray(z) / physics.TAU

def rad(z):
    return physics.TAU * np.asarray(z) / 360


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

    def floats(self, n=3):
        return [self.next_float() for _ in range(n)]

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
                kwargs[token] = self.floats()
            if token == 'radius':
                kwargs[token] = self.next_float()
            if token == 'length':
                kwargs[token] = self.next_float()
            if token == 'quaternion':
                theta, x, y, z = self.floats(4)
                quaternion = physics.make_quaternion(physics.TAU * theta / 360, x, y, z)
            if token == 'position':
                position = self.floats()
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

        # store marker attachment points in a list on each body.
        body.markers = []

        return token

    def handle_joint(self, namespace):
        shape = self.next_token(expect='^({})$'.format('|'.join(physics.JOINTS)))
        body1 = namespace + self.next_token(lower=False)
        offset1 = self.floats()
        body2 = namespace + self.next_token(lower=False)
        offset2 = self.floats()

        anchor = self.world.move_next_to(body1, body2, offset1, offset2)

        token = self.next_token()
        axes = [(1, 0, 0)]
        if shape.startswith('bal'):
            axes = [dict(rel=1, axis=(1, 0, 0)),
                    None,
                    dict(rel=2, axis=(0, 1, 0))]
        if shape.startswith('uni'):
            axes = [(1, 0, 0), (0, 1, 0)]
        lo_stops = None
        hi_stops = None
        while token:
            if token in ('body', 'join'):
                break
            if token.startswith('axis'):
                i = int(token.replace('axis', ''))
                if isinstance(axes[i], dict):
                    axes[i]['axis'] = self.floats()
                else:
                    axes[i] = self.floats()
                if self.peek_token() == 'rel':
                    _ = self.next_token()
                    axes[i]['rel'] = self.next_token(expect='^\d$', dtype=int)
            if token == 'lo_stops':
                lo_stops = rad(self.floats(physics.JOINTS[shape].ADOF))
            if token == 'hi_stops':
                hi_stops = rad(self.floats(physics.JOINTS[shape].ADOF))
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

    def handle_marker(self, namespace):
        index = label = self.next_token()
        if index.count(':') == 1:
            index, label = index.split(':')
        index = int(index)
        body = self.world.get_body(namespace + self.next_token(lower=False))
        offset = self.floats()

        logging.info('attaching %s %s <-> marker %d:%s',
                     body, offset, index, label)

        marker = self.world.create_body(
            'sphere', radius=0.01, color=(1, 0, 0),
            name='{}marker:{}'.format(namespace, label))

        joint = ode.BallJoint(self.world.ode_world, group)
        joint.attach(marker.ode_body, body.ode_body)
        joint.setAnchor(body.body_to_world(offset))
        joint.setParam(ode.ParamCFM, 0.0001)
        joint.setParam(ode.ParamERP, 0.2)

        # we store the marker attachments for a body on the body object.
        body.markers.append((marker, joint))

        return token

    def create(self, namespace=''):
        token = self.next_token(expect='^(body|joint|marker)$')
        while token is not None:
            try:
                if token == 'body':
                    token = self.handle_body(namespace)
                elif token == 'join':
                    token = self.handle_joint(namespace)
                elif token == 'marker':
                    token = self.handle_marker(namespace)
                else:
                    self.error('unexpected token')
            except:
                self.error('internal error')
                raise
        return self.root


def create_skeleton(world, filename, namespace='', cfm=1e-10, max_force=250):
    if namespace and namespace[0] not in '.:-':
        namespace += '.'

    root = world.get_body(Parser(world, filename).create(namespace))

    lm = physics.LMotor(namespace + 'lmotor', world, root, dof=3)
    lm.velocities = 0
    lm.cfms = cfm
    lm.max_forces = max_force
    lm.axes = (dict(rel=0, axis=(1, 0, 0)),
               dict(rel=0, axis=(0, 1, 0)),
               dict(rel=0, axis=(0, 0, 1)))

    am = physics.AMotor(namespace + 'amotor', world, root, mode='euler', dof=3)
    am.velocities = 0
    am.cfms = cfm
    am.max_forces = max_force
    am.axes = (dict(rel=1, axis=(1, 0, 0)),
               None,
               dict(rel=2, axis=(0, 0, 1)))

    al = physics.AMotor(namespace + 'alimit', world, root, mode='euler', dof=3)
    al.lo_stops = -physics.TAU / 5
    al.hi_stops = physics.TAU / 5
    al.axes = (dict(rel=1, axis=(1, 0, 0)),
               None,
               dict(rel=2, axis=(0, 0, 1)))

    for joint in world.joints:
        if joint.name.startswith(namespace):
            joint.velocities = 0
            joint.cfms = cfm
            joint.max_forces = max_force


class Frames(object):
    def __init__(self, filename=None, num_frames=0, num_dofs=0):
        self.data = None
        if filename:
            self.load(filename)
        elif num_frames:
            self.data = np.zeros((num_frames, num_dofs), float)
        self.num_dofs = len(self.data[0])

    def __len__(self):
        return len(self.data)

    def load(self, filename):
        if filename.endswith('.c3d'):
            reader = lmj.c3d.Reader(filename)
            param = reader.group('POINT').params['LABELS']
            length, count = param.dimensions
            labels = [param.bytes[i*length:(i+1)*length] for i in range(count)]
            frames = list(reader.read_frames())
            self.data = np.zeros((len(frames), len(labels)), float)
            for i, (frame, _) in enumerate(frames):
                for j, (x, y, z, c) in enumerate(frame):
                    if j > 0 and not 1 < c < 10:
                        x, y, z = self.data[i-1, j*3:(j+1)*3]
                    self.data[i, j*3:(j+1)*3] = x, y, z
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
    def like(cls, frames):
        return cls(num_frames=len(frames), num_dofs=frames.num_dofs)


class World(physics.World):
    @property
    def num_dofs(self):
        return sum(j.ADOF + j.LDOF for j in self.joints)

    def reset(self):
        for b in self.bodies:
            b.position = b.position + np.array([0, 0, 2])

    def markers_to_angles(self, markers):
        '''Follow a set of marker data.'''
        if isinstance(markers, str):
            markers = Frames(markers)
        state_sequence = []
        smoothed_markers = Frames.like(markers)
        angles = Frames(num_frames=markers.num_frames, num_dofs=self.num_dofs)
        for i, frame in enumerate(markers):
            self.contactgroup.empty()
            self.space.collide(None, self.on_collision)
            states_sequence.append(self.get_body_states())
            self.set_body_states(state_sequence[-1])

            # update the positions and velocities of the markers.
            markers # XXX

            # update the ode world.
            self.ode_world.step(self.dt)

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
        torques = Frames.like(angles)
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
            self.ode_world.step(self.dt)

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
            self.ode_world.step(self.dt)

            #body->restoreControl()  # XXX

        return state_sequence
