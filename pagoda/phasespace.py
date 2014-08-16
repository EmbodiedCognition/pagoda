'''Classes for fetching data from phasespace.'''

import collections
import contextlib
import logging
import OWL
import threading

ERROR_MAP = {
    OWL.OWL_NO_ERROR: 'No Error',
    OWL.OWL_INVALID_VALUE: 'Invalid Value',
    OWL.OWL_INVALID_ENUM: 'Invalid Enum',
    OWL.OWL_INVALID_OPERATION: 'Invalid Operation',
    }


class Base(object):
    '''This class has some common methods for handling errors.'''

    def __init__(self, name):
        self.name = name

    def check_error(self, msg):
        '''Check for an error, and log it if there was one.'''
        err = OWL.owlGetError()
        if err != OWL.OWL_NO_ERROR:
            logging.error('%s: OWL error %s while %s',
                self.name, ERROR_MAP.get(err, '0x%x' % err), msg)
            return True
        return False

    def critical(self, msg):
        '''Check for an error; if there was one, log it and exit.'''
        err = OWL.owlGetError()
        if err != OWL.OWL_NO_ERROR:
            logging.critical('%s: OWL error %s while %s',
                self.name, ERROR_MAP.get(err, '0x%x' % err), msg)
            sys.exit(err)


class Tracker(Base):
    '''Base class for managing OWL point and rigid body trackers.

    The primary feature of this class is that it defines a method, `track()`,
    which can be used as a Python context manager to ensure clean state
    manipulation of the phasespace server::

    >>> with tracker.track():
    ...    # interact with OWL server, tracker will
    ...    # be disabled automatically when finished.

    Attributes
    ----------
    value : any
        Read from this attribute to get the current state of the tracker.
    '''

    def __init__(self, index, name):
        super(Tracker, self).__init__(name)

        self._lock = threading.Lock()
        self._value = None

        self.index = index

        OWL.owlTrackeri(index, OWL.OWL_CREATE, self.TRACKER_TYPE)

    def __del__(self):
        OWL.owlTrackeri(self.index, OWL.OWL_DESTROY)

    @property
    def value(self):
        '''Get the current value stored by this object.'''
        with self._lock:
            return self._value

    @contextlib.contextmanager
    def track(self):
        OWL.owlTracker(self.index, OWL.OWL_ENABLE)
        yield
        OWL.owlTracker(self.index, OWL.OWL_DISABLE)

    def check_tracker(self):
        with self.track():
            if not OWL.owlGetStatus():
                self.critical('setting up tracker')


class PointTracker(Base):
    '''Track a set of markers on the phasespace server.

    Attributes
    ----------
    value : {int: [float, float, float, float]}
        Read from this attribute to get the current state of each marker in the
        tracker. The current value is stored as a dictionary that maps marker
        ids onto (x, y, z, condition) tuples.
    '''

    TRACKER_TYPE = OWL.OWL_POINT_TRACKER

    def __init__(self, index, name, marker_ids):
        super(PointTracker, self).__init__(index, name)

        if isinstance(marker_ids, int):
            marker_ids = list(range(marker_ids))
        self.marker_ids = marker_ids

        for i, marker_id in enumerate(self.marker_ids):
            OWL.owlMarkeri(OWL.MARKER(index, i), OWL.OWL_SET_LED, marker_id)

        self.check_tracker()

    def update(self):
        '''Update our current point data from the phasespace server.'''
        markers = []
        with self.track():
            received = OWL.owlGetMarkers(markers, len(self.marker_ids))
            if self.check_error('getting marker data'):
                return
        if received != len(self.marker_ids):
            logging.info('expected %d markers, received %d',
                         len(self.marker_ids), received)
        else:
            with self._lock:
                self._value = dict(
                    (i, (m.x, m.y, m.z, m.cond))
                    for i, m in zip(self.marker_ids, markers))


Pose = collections.NamedTuple('Pose', 'pos quat cond')

class RigidTracker(Base):
    '''Track a rigid body on the phasespace server.

    Attributes
    ----------
    value : Pose
        Read from this attribute to get the current pose of the rigid body. The
        pose is stored as a named 3-tuple containing ".pos" (three floating
        point position values), ".quat" (four floating point quaternion values),
        and ".cond" (one floating point confidence estimate).
    '''

    TRACKER_TYPE = OWL.OWL_RIGID_TRACKER

    def __init__(self, index, name, marker_map):
        super(RigidTracker, self).__init__(index, name)

        self.marker_map = sorted(marker_map.iteritems())

        for marker_id, marker_pos in self.marker_map:
            m = OWL.MARKER(index, i)
            OWL.owlMarkeri(m, OWL.OWL_SET_LED, marker_id)
            OWL.owlMarkerfv(m, OWL.OWL_SET_POSITION, marker_pos)

        self.check_tracker()

    def update(self):
        '''Update our current rigid body data from the phasespace server.'''
        rigids = []
        markers = []
        with self.track():
            received = OWL.owlGetRigids(rigids, 1)
            OWL.owlGetMarkers(markers, len(self.marker_map))
            if self.check_error('getting marker data'):
                return
        if received != 1:
            logging.info('expected 1 rigid body, received %d', received)
        if rigids:
            r = rigids[0]
            with self._lock:
                self._value = Pose(pos=r.pose[0:3], quat=r.pose[3:7], cond=r.cond)
                self._markers = dict(
                    (i, (m.x, m.y, m.z))
                    for i, m in zip(self.marker_map, markers))

    def write(self, filename):
        '''Write the pose for this rigid body out to a .rb file.'''
        with self._lock, open(filename, 'r') as handle:
            for i in sorted(self._markers):
                handle.write('{}, {} {} {}'.format(i, *self._markers[i]))


class Phasespace(Base):
    '''Wrap the details of getting data from phasespace.'''

    def __init__(self, server_name, init_flags=0):
        '''Initialize a connection to the phasespace server.

        server_name: The hostname for the phasespace server.
        init_flags: Server flags for phasespace.
        '''
        if OWL.owlInit(server_name, init_flags) < 0:
            self.error('initializing')

        self.point_trackers = []
        self.rigid_trackers = []

        self._thread = threading.Thread(target=self._listen)

    def __del__(self):
        '''Clean up our connection to the phasespace server.'''
        self.stop()
        OWL.owlDone()

    def _listen(self):
        '''Continually update our trackers with data from phasespace.'''
        while True:
            with self._lock:
                for pt in self.point_trackers:
                    pt.update()
            with self._lock:
                for rt in self.rigid_trackers:
                    rt.update()

    def start(self):
        '''Start tracking Phasespace data in our worker thread.'''
        logging.info('enabling phasespace streaming...')
        OWL.owlSetFloat(OWL.OWL_FREQUENCY, OWL.OWL_MAX_FREQUENCY)
        OWL.owlSetInteger(OWL.OWL_STREAMING, OWL.OWL_ENABLE)
        logging.info('starting phasespace worker thread...')
        self._thread.start()

    def stop(self):
        '''Stop tracking Phasespace data in our worker thread.'''
        logging.info('stopping phasespace worker thread...')
        self._thread.stop()
        logging.info('disabling phasespace streaming...')
        OWL.owlSetInteger(OWL.OWL_STREAMING, OWL.OWL_DISABLE)

    def add_point_tracker(self, markers, index=None, name=None):
        '''Add a point tracker to the phasespace workload.

        Parameters
        ----------
        markers : int or sequence of int
            If this is an integer, specifies the number of markers we ought to
            track. If this is a sequence of integers, it specifies the IDs of
            the markers to track.
        index : int
            If given, use this tracker index. We will make a unique index if
            this is None.
        name : str
            A name for the tracker. We'll use a default name if this is None.

        Returns
        -------
        PointTracker :
            The tracker object that was created for this set of markers. Use
            its `.value` attribute to retrieve current point data.
        '''
        if index is None:
            index = max(t.index for t in self.point_trackers) + 1
        if name is None:
            name = 'point-{}'.format(index)
        tracker = PointTracker(index, name, marker_count)
        with self._lock:
            self.point_trackers.append(tracker)
        return tracker

    def add_rigid_tracker(self, marker_map=None, filename=None, index=None, name=None):
        '''Add a rigid-body tracker to the phasespace workload.

        Parameters
        ----------
        marker_map : {int: [float, float, float]}
            A dictionary that maps marker index values onto positions for those
            markers. The positions are used to define the rigid body.
        filename : str
            If this is not None, `marker_map` will be loaded from a rigid body
            definition file. (This is often a text file with an .rb extension.)
        index : int
            If given, use this tracker index. We will make a unique index if
            this is None.
        name : str
            A name for the tracker. We'll use a default name if this is None.

        Returns
        -------
        RigidTracker :
            The rigid tracker object that was created for this rigid body. Use
            its `.value` attribute to retrieve rigid current body data.
        '''
        if index is None:
            index = max(t.index for t in self.rigid_trackers) + 1
        if filename is not None:
            marker_map = {}
            with open(filename) as handle:
                for line in handle:
                    id, x, y, z = line.replace(',', '').strip().split()
                    marker_map[int(id)] = float(x), float(y), float(z)
            if name is None:
                name = 'rigid-{}'.format(
                    os.path.splitext(os.path.basename(filename))[0])
        if name is None:
            name = 'rigid-{}'.format(index)
        tracker = RigidTracker(index, name, marker_map)
        with self._lock:
            self.rigid_trackers.append(tracker)
        return tracker
