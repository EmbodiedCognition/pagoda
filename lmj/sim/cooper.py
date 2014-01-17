'''Python implementation of forward-dynamics solver by Joseph Cooper.'''

import c3d
import climate
import itertools
import lmj.pid
import numpy as np
import ode
import os
import re

from . import physics

logging = climate.get_logger(__name__)


class Parser(object):
    def __init__(self, world, source, namespace, jointgroup=None):
        self.world = world
        self.filename = source
        self.namespace = namespace
        self.jointgroup = jointgroup

        self.config = []
        self.index = 0
        self.root = None

        if isinstance(source, str):
            source = open(source)
        else:
            self.filename = '(file)'
        for i, line in enumerate(source):
            for j, token in enumerate(line.split('#')[0].strip().split()):
                if token.strip():
                    self.config.append((i, j, token))
        source.close()

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

    def handle_body(self):
        shape = self.next_token(expect='^({})$'.format('|'.join(physics.BODIES)))
        name = self.namespace + self.next_token(lower=False)

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

        return token

    def handle_joint(self):
        shape = self.next_token(expect='^({})$'.format('|'.join(physics.JOINTS)))
        body1 = self.namespace + self.next_token(lower=False)
        offset1 = self.floats()
        body2 = self.namespace + self.next_token(lower=False)
        offset2 = self.floats()

        anchor = self.world.move_next_to(body1, body2, offset1, offset2)

        token = self.next_token()
        axes = [(1, 0, 0), (0, 1, 0)]
        lo_stops = None
        hi_stops = None
        while token:
            if token in ('body', 'join'):
                break
            if token.startswith('axis'):
                axes[int(token.replace('axis', ''))] = self.floats()
            if token == 'lo_stops':
                lo_stops = np.deg2rad(self.floats(physics.JOINTS[shape].ADOF))
            if token == 'hi_stops':
                hi_stops = np.deg2rad(self.floats(physics.JOINTS[shape].ADOF))
            token = self.next_token()

        logging.info('joining %s %s %s', shape, body1, body2)

        joint = self.world.join(
            shape, body1, body2, anchor=anchor, jointgroup=self.jointgroup)
        joint.axes = axes[:joint.ADOF]
        if lo_stops is not None:
            joint.lo_stops = lo_stops
        if hi_stops is not None:
            joint.hi_stops = hi_stops

        # we add some additional attributes for controlling this joint
        joint.target_angles = [None] * joint.ADOF
        joint.controllers = [lmj.pid.Controller(kp=1) for i in range(joint.ADOF)]

        return token

    def create(self):
        token = self.next_token(expect='^(body|joint)$')
        while token is not None:
            try:
                if token == 'body':
                    token = self.handle_body()
                elif token == 'join':
                    token = self.handle_joint()
                else:
                    self.error('unexpected token')
            except:
                self.error('internal error')
                raise
        return self.root


class Skeleton(object):
    '''
    '''

    def __init__(self, world, filename, namespace=None):
        '''
        '''
        self.world = world
        self.filename = filename
        self.jointgroup = ode.JointGroup()

        if namespace is None:
            base = os.path.basename(filename).lower()
            namespace, _ = os.path.splitext(base)
        if namespace[-1] not in '.:-':
            namespace += ':'
        self.namespace = namespace

        parser = Parser(world, filename, namespace, self.jointgroup)
        self.root = world.get_body(parser.create())

    @property
    def num_dofs(self):
        return sum(j.ADOF + j.LDOF for j in self.joints)

    @property
    def bodies(self):
        for body in self.world.bodies:
            if body.name.startswith(self.namespace):
                yield body

    @property
    def joints(self):
        for joint in self.world.joints:
            if joint.name.startswith(self.namespace):
                yield joint

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
    def velocities(self):
        values = []
        for joint in self.joints:
            values.extend(joint.velocities)
        return values

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

    def add_torques(self, torques):
        j = 0
        for joint in self.joints:
            joint.add_torque(torques[j:j+joint.ADOF])
            j += joint.ADOF

    def set_target_angles(self, angles):
        j = 0
        for joint in self.joints:
            # move toward target angles for each joint.
            joint.velocities = [
                ctrl(tgt - cur, self.world.dt) for cur, tgt, ctrl in
                zip(joint.angles, angles[j:j+joint.ADOF], joint.controllers)]
            j += joint.ADOF


class Frames(object):
    def __init__(self, filename=None, num_frames=0, num_dofs=0):
        self.data = None
        if filename:
            self.load(filename)
        elif num_frames:
            self.data = np.zeros((num_frames, num_dofs), float)

    @property
    def num_frames(self):
        return self.data.shape[0]

    @property
    def num_dofs(self):
        return self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load(self, filename):
        if filename.lower().endswith('.npy'):
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


class Markers(Frames):
    '''
    '''

    def __init__(self, world, filename, channels=None):
        self.world = world
        self.jointgroup = ode.JointGroup()
        self.joints = []
        self.cfm = 1e-4
        self.erp = 0.7

        self.skeleton = None
        self.marker_bodies = {}
        self.attach_bodies = {}
        self.attach_offsets = {}

        self.channels = self._interpret_channels(channels)
        self.load(filename)

    @property
    def num_markers(self):
        return self.data.shape[1]

    def _interpret_channels(self, channels):
        if isinstance(channels, str):
            channels = channels.strip().split()
        if isinstance(channels, (tuple, list)):
            return dict((c, i) for i, c in enumerate(channels))
        return channels or {}

    def load(self, filename, channels=None):
        if not filename.lower().endswith('.c3d'):
            self.channels = self._interpret_channels(channels or self.channels)
            super(Markers, self).load(filename)
            self.create_bodies()
            return

        with open(filename, 'rb') as handle:
            reader = c3d.Reader(handle)

            # make sure the c3d file's frame rate matches our world.
            assert self.world.dt == 1. / reader.frame_rate()

            # set up a map from marker label to index in the data stream.
            param = reader.get('POINT.LABELS')
            l, n = param.dimensions
            labels = [param.bytes[i*l:(i+1)*l].strip() for i in range(n)]
            logging.info('%s: loaded marker labels %s', filename, labels)
            self.channels = self._interpret_channels(labels)

            # read the actual c3d data into a numpy array.
            self.data = np.asarray(
                [frame / 1000 for frame, _ in reader.read_frames()])

        logging.info('%s: loaded marker data %s', filename, self.data.shape)
        self.create_bodies()

    def create_bodies(self):
        self.marker_bodies = {}
        for label in self.channels:
            body = self.world.create_body(
                'sphere',
                name='marker:{}'.format(label),
                radius=0.02,
                color=(1, 1, 1, 0.5))
            body.is_kinematic = True
            self.marker_bodies[label] = body

    def load_attachments(self, source, skeleton):
        self.skeleton = skeleton

        self.attach_bodies = {}
        self.attach_offsets = {}

        filename = source
        if isinstance(source, str):
            source = open(source)
        else:
            filename = '(stringio)'

        for i, line in enumerate(source):
            tokens = line.split('#')[0].strip().split()
            if not tokens:
                continue
            label = tokens.pop(0)
            if label not in self.channels:
                logging.info('%s:%d: unknown marker %s', filename, i, label)
                continue
            if not tokens:
                continue
            name = tokens.pop(0)
            s = '{}{}'.format(skeleton.namespace, name)
            bodies = [b for b in skeleton.bodies if b.name == s]
            if len(bodies) != 1:
                logging.info('%s:%d: %d skeleton bodies match %s',
                             filename, i, len(bodies), name)
                continue
            b = self.attach_bodies[label] = bodies[0]
            o = self.attach_offsets[label] = \
                np.array(map(float, tokens)) * b.dimensions / 2
            logging.info('%s <--> %s, offset %s', label, b.name, o)

    def detach(self):
        self.jointgroup.empty()
        self.joints = []

    def attach(self, frame_no):
        for label, j in self.channels.iteritems():
            if not 1 < self.data[frame_no, j, 3] < 100:
                continue
            joint = ode.BallJoint(self.world.ode_world, self.jointgroup)
            joint.attach(self.marker_bodies[label].ode_body,
                         self.attach_bodies[label].ode_body)
            joint.setAnchor1Rel([0, 0, 0])
            joint.setAnchor2Rel(self.attach_offsets[label])
            joint.setParam(ode.ParamCFM, self.cfm)
            joint.setParam(ode.ParamERP, self.erp)
            self.joints.append(joint)

    def reposition(self, frame_no):
        frame = self.data[frame_no, :, :3]
        delta = np.zeros_like(frame)
        if 0 < frame_no < self.num_frames - 1:
            delta = (self.data[frame_no + 1, :, :3] -
                     self.data[frame_no - 1, :, :3]) / (2 * self.world.dt)
        for label, j in self.channels.iteritems():
            body = self.marker_bodies[label]
            body.position = frame[j]
            body.linear_velocity = delta[j]

    @classmethod
    def like(cls, markers):
        new = cls(markers.world)
        new.data = np.zeros_like(markers.data)
        return new


class World(physics.World):
    def load_skeleton(self, filename, namespace=None):
        return Skeleton(self, filename, namespace)

    def load_markers(self, filename, attachments, skeleton):
        markers = Markers(self, filename=filename)
        markers.load_attachments(attachments, skeleton)
        return markers

    def step(self, substeps=2):
        # by default we step by following our loaded marker data.
        try:
            next(self.follower)
        except:
            self.follower = iter(self.follow(self.markers))

    def follow(self, markers):
        '''Iterate over a set of marker data, dragging its skeleton along.'''
        for i, frame in enumerate(markers):
            # update the positions and velocities of the markers.
            markers.detach()
            markers.reposition(i)
            markers.attach(i)

            self.ode_space.collide(None, self.on_collision)

            states = markers.skeleton.get_body_states()
            markers.skeleton.set_body_states(states)

            # update the ode world.
            self.ode_world.step(self.dt)

            # yield the current simulation state to our caller.
            yield frame, states

            # clear out contact joints to prepare for the next frame.
            self.ode_contactgroup.empty()

    def inverse_kinematics(self, markers):
        '''Follow a set of marker data, yielding kinematic joint angles.'''
        for i, (frame, states) in enumerate(self.follow(markers)):
            # record the smoothed marker positions on the body.
            smoothed = (j.anchor2 + (np.pi, ) for j in markers.joints)

            # record the angles of each joint in the body.
            angles = itertools.chain.from_iterable(
                j.angles for j in markers.skeleton.joints)

            # yield the smoothed markers and angles to our caller.
            yield list(smoothed), list(angles)

    def inverse_dynamics(self, markers, angles):
        '''Follow a set of angle data, yielding dynamic joint torques.'''
        for i, frame in enumerate(angles):
            # update the positions and velocities of the markers.
            markers.detach()
            markers.reposition(i)
            markers.attach(i)

            self.ode_space.collide(None, self.on_collision)

            states = markers.skeleton.get_body_states()
            markers.skeleton.set_body_states(states)

            # set the target angles for each joint.
            markers.skeleton.set_target_angles(frame)

            # update the ode world.
            self.ode_world.step(self.dt)

            # record the joint torques.
            torques = itertools.chain.from_iterable(
                j.amotor.feedback[-1][:j.ADOF]
                for j in markers.skeleton.joints)

            # yield the computed torques to our caller.
            yield torques

            # reset the markers and skeleton to the start of the step.
            markers.detach()
            markers.reposition(i)
            markers.attach(i)
            markers.skeleton.set_body_states(states)

            # reset the torques for the skeleton, and step again.
            markers.skeleton.max_force = 0
            markers.skeleton.add_torques(torques)
            self.ode_world.step(self.dt)
            markers.skeleton.max_force = 9999

            self.ode_contactgroup.empty()

    def forward_dynamics(self, markers, torques):
        '''Move the body according to a set of torque data.'''
        for i, (_, states) in enumerate(self.follow(markers)):
            markers.skeleton.add_torques(torques[i])
