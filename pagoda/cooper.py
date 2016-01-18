# -*- coding: utf-8 -*-

'''This module contains a Python implementation of a forward-dynamics solver.

For detailed information about the solver, its raison d'être, and how it works,
please see the documentation for the :class:`World` class.

Further comments and documentation are available in this source file. Eventually
I hope to integrate these comments into some sort of online documentation for
the package as a whole.
'''

import climate
import numpy as np
import ode
import re

from . import physics
from . import skeleton

logging = climate.get_logger(__name__)


class Markers:
    '''
    '''

    DEFAULT_CFM = 1e-6
    DEFAULT_ERP = 0.3

    def __init__(self, world):
        self.world = world
        self.jointgroup = ode.JointGroup()
        self.joints = []

        self.erp = Markers.DEFAULT_ERP

        self.marker_bodies = {}
        self.attach_bodies = {}
        self.attach_offsets = {}
        self.channels = {}

        self.data = None
        self.cfms = None

        # these arrays are derived from the data array.
        self.visibility = None
        self.positions = None
        self.velocities = None

    @property
    def num_frames(self):
        '''Return the number of frames of marker data.'''
        return self.data.shape[0]

    @property
    def num_markers(self):
        '''Return the number of markers in each frame of data.'''
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

    def load_csv(self, filename, start_frame=10, max_frames=int(1e300)):
        '''Load marker data from a CSV file.

        The file will be imported using Pandas, which must be installed to use
        this method. (``pip install pandas``)

        The first line of the CSV file will be used for header information. The
        "time" column will be used as the index for the data frame. There must
        be columns named 'markerAB-foo-x','markerAB-foo-y','markerAB-foo-z', and
        'markerAB-foo-c' for marker 'foo' to be included in the model.

        Parameters
        ----------
        filename : str
            Name of the CSV file to load.
        '''
        import pandas as pd

        compression = None
        if filename.endswith('.gz'):
            compression = 'gzip'
        df = pd.read_csv(filename, compression=compression).set_index('time').fillna(-1)

        # make sure the data frame's time index matches our world.
        assert self.world.dt == pd.Series(df.index).diff().mean()

        markers = []
        for c in df.columns:
            m = re.match(r'^marker\d\d-(.*)-c$', c)
            if m:
                markers.append(m.group(1))
        self.channels = self._interpret_channels(markers)

        cols = [c for c in df.columns if re.match(r'^marker\d\d-.*-[xyzc]$', c)]
        self.data = df[cols].values.reshape((len(df), len(markers), 4))[start_frame:]
        self.data[:, :, [1, 2]] = self.data[:, :, [2, 1]]

        logging.info('%s: loaded marker data %s', filename, self.data.shape)
        self.process_data()
        self.create_bodies()

    def load_c3d(self, filename, start_frame=0, max_frames=int(1e300)):
        '''Load marker data from a C3D file.

        The file will be imported using the c3d module, which must be installed
        to use this method. (``pip install c3d``)

        Parameters
        ----------
        filename : str
            Name of the C3D file to load.
        start_frame : int, optional
            Discard the first N frames. Defaults to 0.
        max_frames : int, optional
            Maximum number of frames to load. Defaults to loading all frames.
        '''
        import c3d

        with open(filename, 'rb') as handle:
            reader = c3d.Reader(handle)

            # set up a map from marker label to index in the data stream.
            labels = [s.strip() for s in reader.point_labels]
            logging.info('%s: loaded marker labels %s', filename, labels)
            self.channels = self._interpret_channels(labels)

            # read the actual c3d data into a numpy array.
            data = []
            for i, (_, frame, _) in enumerate(reader.read_frames()):
                if i >= start_frame:
                    data.append(frame[:, [0, 1, 2, 4]])
                if len(data) > max_frames:
                    break
            self.data = np.array(data)

            # scale the data to meters -- mm is a very common C3D unit.
            if reader.get('POINT:UNITS').string_value.strip().lower() == 'mm':
                logging.info('scaling point data from mm to m')
                self.data[:, :, :3] /= 1000.

        logging.info('%s: loaded marker data %s', filename, self.data.shape)
        self.process_data()
        self.create_bodies()

    def process_data(self):
        '''Process data to produce velocity and dropout information.'''
        self.visibility = self.data[:, :, 3]
        self.positions = self.data[:, :, :3]
        self.velocities = np.zeros_like(self.positions) + 1000
        for frame_no in range(1, len(self.data) - 1):
            prev = self.data[frame_no - 1]
            next = self.data[frame_no + 1]
            for c in range(self.num_markers):
                if -1 < prev[c, 3] < 100 and -1 < next[c, 3] < 100:
                    self.velocities[frame_no, c] = (
                        next[c, :3] - prev[c, :3]) / (2 * self.world.dt)
        self.cfms = np.zeros_like(self.visibility) + self.DEFAULT_CFM

    def create_bodies(self):
        '''Create physics bodies corresponding to each marker in our data.'''
        self.marker_bodies = {}
        for label in self.channels:
            body = self.world.create_body(
                'sphere', name='marker:{}'.format(label), radius=0.02)
            body.is_kinematic = True
            body.color = 0.9, 0.1, 0.1, 0.5
            self.marker_bodies[label] = body

    def load_attachments(self, source, skeleton):
        '''Load attachment configuration from the given text source.

        The attachment configuration file has a simple format. After discarding
        Unix-style comments (any part of a line that starts with the pound (#)
        character), each line in the file is then expected to have the following
        format::

            marker-name body-name X Y Z

        The marker name must correspond to an existing "channel" in our marker
        data. The body name must correspond to a rigid body in the skeleton. The
        X, Y, and Z coordinates specify the body-relative offsets where the
        marker should be attached: 0 corresponds to the center of the body along
        the given axis, while -1 and 1 correspond to the minimal (maximal,
        respectively) extent of the body's bounding box along the corresponding
        dimension.

        Parameters
        ----------
        source : str or file-like
            A filename or file-like object that we can use to obtain text
            configuration that describes how markers are attached to skeleton
            bodies.

        skeleton : :class:`pagoda.skeleton.Skeleton`
            The skeleton to attach our marker data to.
        '''
        self.attach_bodies = {}
        self.attach_offsets = {}

        filename = source
        if isinstance(source, str):
            source = open(source)
        else:
            filename = '(file-{})'.format(id(source))

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
        '''Detach all marker bodies from their associated skeleton bodies.'''
        self.jointgroup.empty()
        self.joints = []

    def attach(self, frame_no):
        '''Attach marker bodies to the corresponding skeleton bodies.

        Attachments are only made for markers that are not in a dropout state in
        the given frame.

        Parameters
        ----------
        frame_no : int
            The frame of data we will use for attaching marker bodies.
        '''
        for label, j in self.channels.items():
            target = self.attach_bodies.get(label)
            if target is None:
                continue
            if self.visibility[frame_no, j] < 0:
                continue
            if np.linalg.norm(self.velocities[frame_no, j]) > 10:
                continue
            joint = ode.BallJoint(self.world.ode_world, self.jointgroup)
            joint.attach(self.marker_bodies[label].ode_body, target.ode_body)
            joint.setAnchor1Rel([0, 0, 0])
            joint.setAnchor2Rel(self.attach_offsets[label])
            joint.setParam(ode.ParamCFM, self.cfms[frame_no, j])
            joint.setParam(ode.ParamERP, self.erp)
            joint.name = label
            self.joints.append(joint)

    def reposition(self, frame_no):
        '''Reposition markers to a specific frame of data.

        Parameters
        ----------
        frame_no : int
            The frame of data where we should reposition marker bodies. Markers
            will be positioned in the appropriate places in world coordinates.
            In addition, linear velocities of the markers will be set according
            to the data as long as there are no dropouts in neighboring frames.
        '''
        for label, j in self.channels.items():
            body = self.marker_bodies[label]
            body.position = self.positions[frame_no, j]
            body.linear_velocity = self.velocities[frame_no, j]

    def distances(self):
        '''Get a list of the distances between markers and their attachments.

        Returns
        -------
        distances : list of float
            List of distances for each marker joint in our attachment setup.
        '''
        deltas = []
        for label, _ in self.channels.items():
            joints = [j for j in self.joints if j.name == label]
            if joints:
                joint = joints[0]
                delta = np.array(joint.getAnchor()) - joint.getAnchor2()
                deltas.append(np.sqrt((delta * delta).sum()))
            else:
                deltas.append(0)
        return deltas

    def forces(self):
        '''Return an array of the forces exerted by marker springs.

        Returns
        -------
        forces : list of float
            A list of the force exerted by each marker spring attachment.
        '''
        return [j.getFeedback()[-1] for j in self.joints]


class World(physics.World):
    '''Simulate a physics world that includes an articulated skeleton model.

    The "cooper" method, originally described by Cooper & Ballard (2012 Proc.
    Motion in Games), uses a forward physics simulator (here, the Open Dynamics
    Engine; ODE) to compute inverse motion quantities like angles and torques
    using motion-capture data and a structured, articulated model of the human
    skeleton. The prerequisites for this method are:

    - Record some motion-capture data from a human. This is expected to result
      in the locations, in world coordinates, of several motion-capture markers
      at regularly-spaced intervals over time.

    - Construct a simulated skeleton that matches the size and shape of the
      human to some reasonable degree of accuracy. The more accurate the
      skeleton, the more accurate the resulting measurements.

    In broad strokes, the cooper method proceeds in two stages:

    1. :func:`Inverse Kinematics <inverse_kinematics>`. The motion-capture data
       are attached to the simulated skeleton using ball joints. These ball
       joints are configured so that their constraints (namely, placing both
       anchor points of the joint at the same location in space) are allowed to
       slip; ODE implements this slippage using a spring dynamics, which
       provides a natural mechanism for the articulated skeleton to interpolate
       the marker data as well as possible.

       At each frame during the first pass, the motion-capture markers are
       placed at the appropriate location in space, and the attached articulated
       skeleton "snaps" toward the markers using its inertia (from the motion in
       preceding frames) as well as the spring constraints provided by the
       marker joint slippage.

       At each frame of this process, the articulated skeleton can be queried to
       obtain joint angles for each degree of freedom. In addition, the markers
       can be queried to find their constraint slippage.

    2. :func:`Inverse Dynamics <inverse_dynamics>`. The marker constraints are
       removed, and the joint angles computed in the first pass are used to
       constrain the skeleton's movements.

       At each frame during the second pass, the joints in the skeleton attempt
       to follow the angles computed in the first pass; a PID controller is used
       to convert the angular error value into a target angular velocity for
       each joint.

       The torques that ODE computes to solve this forward angle-following
       problem are returned as a result of the second pass.

    In general, the cooper model is a useful way of getting a physics simulator,
    a model of a human skeleton, and some motion-capture data to interact
    smoothly. Particularly useful for almost any simulations of human motion are
    the :func:`settle_to_markers` and :func:`follow_markers` methods.
    '''

    def load_skeleton(self, filename, pid_params=None):
        '''Create and configure a skeleton in our model.

        Parameters
        ----------
        filename : str
            The name of a file containing skeleton configuration data.
        pid_params : dict, optional
            If given, use this dictionary to set the PID controller
            parameters on each joint in the skeleton. See
            :func:`pagoda.skeleton.pid` for more information.
        '''
        self.skeleton = skeleton.Skeleton(self)
        self.skeleton.load(filename)
        if pid_params:
            self.skeleton.set_pid_params(**pid_params)
        self.skeleton.erp = 0.1
        self.skeleton.cfm = 0
        self.skeleton.color = 0.3, 0.5, 0.9, 0.8

    def load_markers(self, filename, attachments, max_frames=1e100):
        '''Load marker data and attachment preferences into the model.

        Parameters
        ----------
        filename : str
            The name of a file containing marker data. This currently needs to
            be either a .C3D or a .CSV file. CSV files must adhere to a fairly
            strict column naming convention; see :func:`Markers.load_csv` for
            more information.
        attachments : str
            The name of a text file specifying how markers are attached to
            skeleton bodies.
        max_frames : number, optional
            Only read in this many frames of marker data. By default, the entire
            data file is read into memory.

        Returns
        -------
        markers : :class:`Markers`
            Returns a markers object containing loaded marker data as well as
            skeleton attachment configuration.
        '''
        self.markers = Markers(self)
        fn = filename.lower()
        if fn.endswith('.c3d'):
            self.markers.load_c3d(filename, max_frames=max_frames)
        elif fn.endswith('.csv') or fn.endswith('.csv.gz'):
            self.markers.load_csv(filename, max_frames=max_frames)
        else:
            logging.fatal('%s: not sure how to load markers!', filename)
        self.markers.load_attachments(attachments, self.skeleton)

    def step(self, substeps=2):
        '''Advance the physics world by one step.

        Typically this is called as part of a :class:`pagoda.viewer.Viewer`, but
        it can also be called manually (or some other stepping mechanism
        entirely can be used).
        '''
        # by default we step by following our loaded marker data.
        self.frame_no += 1
        try:
            next(self.follower)
        except (AttributeError, StopIteration) as err:
            self.reset()

    def reset(self):
        '''Reset the automatic process that gets called by :func:`step`.

        By default this follows whatever marker data is loaded into our model.

        Provide an override for this method to customize the default behavior of
        the :func:`step` method.
        '''
        self.follower = self.follow_markers()

    def settle_to_markers(self, frame_no=0, max_distance=0.1, max_iters=300, states=None):
        '''Settle the skeleton to our marker data at a specific frame.

        Parameters
        ----------
        frame_no : int, optional
            Settle the skeleton to marker data at this frame. Defaults to 0.
        max_distance : float, optional
            The settling process will stop when the mean marker distance falls
            below this threshold. Defaults to 0.1m (10cm). Setting this too
            small prevents the settling process from finishing (it will loop
            indefinitely), and setting it too large prevents the skeleton from
            settling to a stable state near the markers.
        max_iters : int, optional
            Attempt to settle markers for at most this many iterations. Defaults
            to 1000.
        states : list of body states, optional
            If given, set the bodies in our skeleton to these kinematic states
            before starting the settling process.
        '''
        if states is not None:
            self.skeleton.set_body_states(states)
        dist = None
        for _ in range(max_iters):
            for states in self._step_to_marker_frame(frame_no):
                pass
            dist = np.nanmean(self.markers.distances())
            logging.info('settling to frame %d: marker distance %.3f', frame_no, dist)
            if dist < max_distance:
                return states
        return states

    def follow_markers(self, start=0, end=1e100, states=None):
        '''Iterate over a set of marker data, dragging its skeleton along.

        Parameters
        ----------
        start : int, optional
            Start following marker data after this frame. Defaults to 0.
        end : int, optional
            Stop following marker data after this frame. Defaults to the end of
            the marker data.
        states : list of body states, optional
            If given, set the states of the skeleton bodies to these values
            before starting to follow the marker data.
        '''
        if states is not None:
            self.skeleton.set_body_states(states)
        for frame_no, frame in enumerate(self.markers):
            if frame_no < start:
                continue
            if frame_no >= end:
                break
            for states in self._step_to_marker_frame(frame_no):
                yield states

    def _step_to_marker_frame(self, frame_no, dt=None):
        '''Update the simulator to a specific frame of marker data.

        This method returns a generator of body states for the skeleton! This
        generator must be exhausted (e.g., by consuming this call in a for loop)
        for the simulator to work properly.

        This process involves the following steps:

        - Move the markers to their new location:
          - Detach from the skeleton
          - Update marker locations
          - Reattach to the skeleton
        - Detect ODE collisions
        - Yield the states of the bodies in the skeleton
        - Advance the ODE world one step

        Parameters
        ----------
        frame_no : int
            Step to this frame of marker data.
        dt : float, optional
            Step with this time duration. Defaults to ``self.dt``.

        Returns
        -------
        states : sequence of state tuples
            A generator of a sequence of one body state for the skeleton. This
            generator must be exhausted for the simulation to work properly.
        '''
        # update the positions and velocities of the markers.
        self.markers.detach()
        self.markers.reposition(frame_no)
        self.markers.attach(frame_no)

        # detect collisions.
        self.ode_space.collide(None, self.on_collision)

        # record the state of each skeleton body.
        states = self.skeleton.get_body_states()
        self.skeleton.set_body_states(states)

        # yield the current simulation state to our caller.
        yield states

        # update the ode world.
        self.ode_world.step(dt or self.dt)

        # clear out contact joints to prepare for the next frame.
        self.ode_contactgroup.empty()

    def inverse_kinematics(self, start=0, end=1e100, states=None, max_force=20):
        '''Follow a set of marker data, yielding kinematic joint angles.

        Parameters
        ----------
        start : int, optional
            Start following marker data after this frame. Defaults to 0.
        end : int, optional
            Stop following marker data after this frame. Defaults to the end of
            the marker data.
        states : list of body states, optional
            If given, set the states of the skeleton bodies to these values
            before starting to follow the marker data.
        max_force : float, optional
            Allow each degree of freedom in the skeleton to exert at most this
            force when attempting to maintain its equilibrium position. This
            defaults to 20N. Set this value higher to simulate a stiff skeleton
            while following marker data.

        Returns
        -------
        angles : sequence of angle frames
            Returns a generator of joint angle data for the skeleton. One set of
            joint angles will be generated for each frame of marker data between
            `start` and `end`.
        '''
        zeros = None
        if max_force > 0:
            self.skeleton.enable_motors(max_force)
            zeros = np.zeros(self.skeleton.num_dofs)
        for _ in self.follow_markers(start, end, states):
            if zeros is not None:
                self.skeleton.set_target_angles(zeros)
            yield self.skeleton.joint_angles

    def inverse_dynamics(self, angles, start=0, end=1e100, states=None, max_force=100):
        '''Follow a set of angle data, yielding dynamic joint torques.

        Parameters
        ----------
        angles : ndarray (num-frames x num-dofs)
            Follow angle data provided by this array of angle values.
        start : int, optional
            Start following angle data after this frame. Defaults to the start
            of the angle data.
        end : int, optional
            Stop following angle data after this frame. Defaults to the end of
            the angle data.
        states : list of body states, optional
            If given, set the states of the skeleton bodies to these values
            before starting to follow the marker data.
        max_force : float, optional
            Allow each degree of freedom in the skeleton to exert at most this
            force when attempting to follow the given joint angles. Defaults to
            100N. Setting this value to be large results in more accurate
            following but can cause oscillations in the PID controllers,
            resulting in noisy torques.

        Returns
        -------
        torques : sequence of torque frames
            Returns a generator of joint torque data for the skeleton. One set
            of joint torques will be generated for each frame of angle data
            between `start` and `end`.
        '''
        self.skeleton.enable_motors(max_force)
        for i, states in enumerate(self.follow_markers(start, end, states)):
            # joseph's stability fix: step to compute torques, then reset the
            # skeleton to the start of the step, and then step using computed
            # torques. thus any numerical errors between the body states after
            # stepping using angle constraints will be removed, because we
            # will be stepping the model using the computed torques.

            self.skeleton.enable_motors(max_force)
            self.skeleton.set_target_angles(angles[start + i])

            self.ode_world.step(self.dt)

            torques = self.skeleton.joint_torques
            self.skeleton.disable_motors()
            self.skeleton.set_body_states(states)
            self.skeleton.add_torques(torques)
            yield torques

    def forward_dynamics(self, torques, start=0, states=None):
        '''Move the body according to a set of torque data.'''
        for i, _ in enumerate(self.follow_markers(start, states)):
            self.skeleton.add_torques(torques[i])
