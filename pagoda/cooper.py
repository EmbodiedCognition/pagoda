'''Python implementation of forward-dynamics solver by Joseph Cooper.'''

import c3d
import climate
import numpy as np
import ode

from . import parser
from . import physics
from . import skeleton

logging = climate.get_logger(__name__)


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
        self.skeleton = skeleton.Skeleton(self, filename, pid_params)

    def load_markers(self, filename, attachments, max_frames=1e100):
        self.markers = Markers(self)
        self.markers.load(filename, max_frames=max_frames)
        self.markers.load_attachments(attachments, self.skeleton)

    def step(self, substeps=2):
        # by default we step by following our loaded marker data.
        self.frame_no += 1
        try:
            next(self.follower)
        except (AttributeError, StopIteration) as err:
            self.reset()

    def reset(self):
        self.follower = self.follow()

    def settle(self, frame_no=0, max_rmse=0.10, pose=None):
        self.markers.cfm = Markers.DEFAULT_CFM
        self.markers.erp = Markers.DEFAULT_ERP
        if pose is not None:
            self.skeleton.set_body_states(pose)
        while True:
            for states in self._step_to_frame(frame_no):
                pass
            rmse = self.markers.rmse()
            logging.info('settling at frame %d: marker rmse %.3f', frame_no, rmse)
            if rmse < max_rmse:
                return states

    def follow(self, start=0, end=1e100, states=None):
        '''Iterate over a set of marker data, dragging its skeleton along.'''
        if states is not None:
            self.skeleton.set_body_states(states)
        for frame_no, frame in enumerate(self.markers):
            if start <= frame_no < end:
                # TODO: replace with "yield from" for full py3k goodness
                for states in self._step_to_frame(frame_no):
                    yield states

    def _step_to_frame(self, frame_no):
        '''
        '''
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

    def inverse_kinematics(self, start=0, end=1e100, states=None, max_force=100):
        '''Follow a set of marker data, yielding kinematic joint angles.'''
        zeros = None
        if max_force > 0:
            self.skeleton.enable_motors(max_force)
            zeros = np.zeros(self.skeleton.num_dofs)
        for _ in self.follow(start, end, states):
            if zeros is not None:
                self.skeleton.set_angles(zeros)
            yield self.skeleton.angles

    def inverse_dynamics(self, angles, start=0, states=None, max_force=300):
        '''Follow a set of angle data, yielding dynamic joint torques.'''
        for i, states in enumerate(self.follow(start, start + len(angles), states)):
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
