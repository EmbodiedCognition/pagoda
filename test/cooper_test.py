import pagoda.cooper


class Base(object):
    def setUp(self):
        self.world = pagoda.cooper.World()


class TestMarkers(Base):
    def setUp(self):
        super(TestMarkers, self).setUp()
        self.markers = pagoda.cooper.Markers(self.world)

    def test_c3d(self):
        self.markers.load_c3d('examples/cooper-motion.c3d')
        assert self.markers.num_frames == 343
        assert len(self.markers.bodies) == 41
        assert len(self.markers.targets) == 0
        assert len(self.markers.offsets) == 0
        assert len(self.markers.channels) == 41

    def test_csv(self):
        return  # TODO
        self.markers.load_csv('examples/cooper-motion.csv')
        assert self.markers.num_frames == 343
        assert len(self.markers.bodies) == 41
        assert len(self.markers.targets) == 0
        assert len(self.markers.offsets) == 0
        assert len(self.markers.channels) == 41
