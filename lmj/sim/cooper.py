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

TAU = 2 * np.pi


class Parser:
    def __init__(self, world, source, jointgroup=None, pid_params=None):
        self.world = world
        self.filename = source
        self.jointgroup = jointgroup

        self.joints = []
        self.bodies = []

        self.pid_params = dict(kp=1. / self.world.dt)
        self.pid_params.update(**(pid_params or {}))

        self.config = []
        self.index = 0
        self.roots = []

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
        name = self.next_token(lower=False)

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
                quaternion = physics.make_quaternion(TAU * theta / 360, x, y, z)
            if token == 'position':
                position = self.floats()
            if token == 'root':
                logging.info('"%s" will be used as a root', name)
                self.roots.append(name)
            token = self.next_token()

        logging.info('creating %s %s %s', shape, name, kwargs)

        body = self.world.create_body(shape, name, **kwargs)
        if quaternion is not None:
            logging.info('setting rotation %s', quaternion)
            body.quaternion = quaternion
        if position is not None:
            logging.info('setting position %s', position)
            body.position = position

        self.bodies.append(body)

        return token

    def handle_joint(self):
        shape = self.next_token(expect='^({})$'.format('|'.join(physics.JOINTS)))
        body1 = self.next_token(lower=False)
        offset1 = self.floats()
        body2 = self.next_token(lower=False)
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

        self.joints.append(joint)

        # we add some additional attributes for controlling this joint
        joint.target_angles = [None] * joint.ADOF
        joint.controllers = [lmj.pid.Controller(**self.pid_params) for i in range(joint.ADOF)]

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
        return self.roots


class Skeleton:
    '''
    '''

    def __init__(self, world, filename, pid_params=None):
        '''
        '''
        self.world = world
        self.filename = filename
        self.jointgroup = ode.JointGroup()

        parser = Parser(world, filename, self.jointgroup, pid_params=pid_params)
        self.roots = [world.get_body(r) for r in parser.create()]
        self.bodies = parser.bodies
        self.joints = parser.joints

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
    def body_velocities(self):
        values = []
        for body in self.bodies:
            values.extend(body.velocity)
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


class Markers:
    '''
    '''

    DEFAULT_CFM = 1e-4
    DEFAULT_ERP = 0.3

    def __init__(self, world):
        self.world = world
        self.jointgroup = ode.JointGroup()
        self.joints = []

        self.cfm = Markers.DEFAULT_CFM
        self.erp = Markers.DEFAULT_ERP
        self.root_attachment_factor = 1.

        self.marker_bodies = {}
        self.attach_bodies = {}
        self.attach_offsets = {}
        self.channels = {}

    @property
    def num_frames(self):
        return self.data.shape[0]

    @property
    def num_markers(self):
        return self.data.shape[1]

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _interpret_channels(self, channels):
        if isinstance(channels, str):
            channels = channels.strip().split()
        if isinstance(channels, (tuple, list)):
            return dict((c, i) for i, c in enumerate(channels))
        return channels or {}

    def load(self, filename, channels=None, max_frames=1e100):
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
            labels = [s.strip() for s in reader.point_labels()]
            logging.info('%s: loaded marker labels %s', filename, labels)
            self.channels = self._interpret_channels(labels)

            # read the actual c3d data into a numpy array.
            data = []
            for _, frame, _ in reader.read_frames():
                data.append(frame)
                if len(data) > max_frames:
                    break
            self.data = np.array(data)

            # scale the data to meters -- mm is a very common C3D unit.
            if reader['POINT:UNITS'].string_value.strip().lower() == 'mm':
                logging.info('scaling point data from mm to m')
                self.data[:, :, :4] /= 1000.

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
        self.roots = skeleton.roots

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
            bodies = [b for b in skeleton.bodies if b.name == name]
            if len(bodies) != 1:
                logging.info('%s:%d: %d skeleton bodies match %s',
                             filename, i, len(bodies), name)
                continue
            b = self.attach_bodies[label] = bodies[0]
            o = self.attach_offsets[label] = \
                np.array(list(map(float, tokens))) * b.dimensions / 2
            logging.info('%s <--> %s, offset %s', label, b.name, o)

    def detach(self):
        self.jointgroup.empty()
        self.joints = []

    def attach(self, frame_no):
        for label, j in self.channels.items():
            target = self.attach_bodies.get(label)
            if target is None:
                continue
            if self.data[frame_no, j, 4] < 0:
                continue
            f = self.root_attachment_factor if target in self.roots else 1.
            joint = ode.BallJoint(self.world.ode_world, self.jointgroup)
            joint.attach(self.marker_bodies[label].ode_body, target.ode_body)
            joint.setAnchor1Rel([0, 0, 0])
            joint.setAnchor2Rel(self.attach_offsets[label])
            joint.setParam(ode.ParamCFM, self.cfm / f)
            joint.setParam(ode.ParamERP, self.erp)
            self.joints.append(joint)

    def reposition(self, frame_no):
        frame = self.data[frame_no, :, :3]
        delta = np.zeros_like(frame)
        if 0 < frame_no < self.num_frames - 1:
            prev = self.data[frame_no - 1]
            next = self.data[frame_no + 1]
            for c in range(self.num_markers):
                if prev[c, 4] > -1 and next[c, 4] > -1:
                    delta[c] = (next[c, :3] - prev[c, :3]) / (2 * self.world.dt)
        for label, j in self.channels.items():
            body = self.marker_bodies[label]
            body.position = frame[j]
            body.linear_velocity = delta[j]

    def rmse(self):
        deltas = []
        for joint in self.joints:
            delta = np.array(joint.getAnchor()) - joint.getAnchor2()
            deltas.append((delta * delta).sum())
        return np.sqrt(np.mean(deltas))

    @classmethod
    def like(cls, markers):
        new = cls(markers.world)
        new.data = np.zeros_like(markers.data)
        return new


class World(physics.World):
    def load_skeleton(self, filename, pid_params=None):
        self.skeleton = Skeleton(self, filename, pid_params)

    def load_markers(self, filename, attachments, max_frames=1e100):
        self.markers = Markers(self)
        self.markers.load(filename, max_frames=max_frames)
        self.markers.load_attachments(attachments, self.skeleton)

    def step(self, substeps=2):
        # by default we step by following our loaded marker data.
        try:
            next(self.follower)
        except (AttributeError, StopIteration) as err:
            self.reset()

    def reset(self):
        self.follower = self.follow()

    def settle(self, min_frame=0, max_frame=0, max_rmse=0.06, pose=None):
        self.markers.cfm = Markers.DEFAULT_CFM
        self.markers.erp = Markers.DEFAULT_ERP
        frame_no = states = rmse = None
        for frame_no, states in enumerate(self.follow(0, None)):
            rmse = self.markers.rmse()
            logging.debug('settling at frame %d: marker rmse %.3f',
                          frame_no, rmse)
            if frame_no < min_frame:
                if pose is not None:
                    self.skeleton.set_body_states(pose)
                continue
            if frame_no > max_frame > 0 or max_rmse > rmse:
                break
        logging.info('settled to markers at frame %d with rmse %.3f',
                     frame_no, rmse)
        return frame_no, states

    def follow(self, start=0, states=None):
        '''Iterate over a set of marker data, dragging its skeleton along.'''
        if states is not None:
            self.skeleton.set_body_states(states)

        for frame_no, frame in enumerate(self.markers):
            if frame_no < start:
                continue

            # update the positions and velocities of the markers.
            self.markers.detach()
            self.markers.reposition(frame_no)
            self.markers.attach(frame_no)

            # detect collisions
            self.ode_space.collide(None, self.on_collision)

            # record the state of each skeleton body.
            states = self.skeleton.get_body_states()
            self.skeleton.set_body_states(states)

            # yield the current simulation state to our caller.
            yield states

            # update the ode world.
            self.ode_world.step(self.dt)

            # clear out contact joints to prepare for the next frame.
            self.ode_contactgroup.empty()

    def inverse_kinematics(self, start=0, states=None, max_force=100):
        '''Follow a set of marker data, yielding kinematic joint angles.'''
        zeros = None
        if max_force > 0:
            self.skeleton.enable_motors(max_force)
            zeros = np.zeros(self.skeleton.num_dofs)
        for _ in self.follow(start, states):
            if zeros is not None:
                self.skeleton.set_angles(zeros)
            yield self.skeleton.angles

    def inverse_dynamics(self, angles, start=0, states=None, max_force=300):
        '''Follow a set of angle data, yielding dynamic joint torques.'''
        for i, states in enumerate(self.follow(start, states)):
            # joseph's stability fix: step to compute torques, then reset the
            # skeleton to the start of the step, and then step using computed
            # torques. thus any numerical errors between the body states after
            # stepping using angle constraints will be removed, because we
            # will be stepping the model using the computed torques.

            self.skeleton.enable_motors(max_force)
            self.skeleton.set_angles(angles[i])

            self.ode_world.step(self.dt)

            torques = self.skeleton.torques
            self.skeleton.disable_motors()
            self.skeleton.set_body_states(states)
            self.skeleton.add_torques(torques)
            yield torques

    def forward_dynamics(self, torques, start=0, states=None):
        '''Move the body according to a set of torque data.'''
        for i, _ in enumerate(self.follow(start, states)):
            self.skeleton.add_torques(torques[i])
