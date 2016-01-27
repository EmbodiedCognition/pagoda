import pagoda.cooper


def fn(s):
    return 'examples/cooper-{}'.format(s)


class Base(object):
    def setUp(self):
        self.world = pagoda.cooper.World()


class TestMarkers(Base):
    def setUp(self):
        super(TestMarkers, self).setUp()
        self.markers = pagoda.cooper.Markers(self.world)

    def test_c3d(self):
        self.markers.load_c3d(fn('motion.c3d'))
        assert self.markers.num_frames == 343
        assert len(self.markers.bodies) == 41
        assert len(self.markers.targets) == 0
        assert len(self.markers.offsets) == 0
        assert len(self.markers.channels) == 41

    def test_csv(self):
        return  # TODO
        self.markers.load_csv(fn('motion.csv'))
        assert self.markers.num_frames == 343
        assert len(self.markers.bodies) == 41
        assert len(self.markers.targets) == 0
        assert len(self.markers.offsets) == 0
        assert len(self.markers.channels) == 41

    def test_load_attachments(self):
        skel = pagoda.skeleton.Skeleton(self.world)
        skel.load(fn('skeleton.txt'))
        self.markers.load_c3d(fn('motion.c3d'))
        self.markers.load_attachments(fn('markers.txt'), skel)
        assert len(self.markers.targets) == 41
        assert len(self.markers.offsets) == 41


class TestWorld(Base):
    def setUp(self):
        super(TestWorld, self).setUp()
        self.world.load_skeleton(fn('skeleton.txt'))
        self.world.load_markers(fn('motion.c3d'), fn('markers.txt'))

    def tearDown(self):
        self.world.skeleton.disable_motors()

    def test_load_skeleton(self):
        assert self.world.skeleton is not None

    def test_load_markers(self):
        assert self.world.markers is not None

    def test_follow_markers(self):
        left = self.world.markers.num_frames
        for states in self.world.follow_markers():
            assert left > 0
            left -= 1
        assert left == 0

    def test_settle_to_markers(self):
        st000 = self.world.skeleton.get_body_states()
        st100 = self.world.settle_to_markers(100)
        assert st000 != st100
        st200 = self.world.settle_to_markers(200)
        assert st100 != st200

    def test_inverse_kinematics(self):
        angles = list(self.world.inverse_kinematics(10))
        assert len(angles) == self.world.markers.num_frames - 10

    def test_inverse_dynamics(self):
        angles = list(self.world.inverse_kinematics(10))
        torques = list(self.world.inverse_dynamics(angles))
        assert len(torques) == len(angles)
