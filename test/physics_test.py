import pagoda.physics


class TestBody:
    def setUp(self):
        self.world = pagoda.physics.World()

    def test_sphere(self):
        sph = pagoda.physics.Sphere('sph', self.world, radius=3)
        assert sph.radius == 3
