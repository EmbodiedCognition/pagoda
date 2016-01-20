import numpy as np
import ode
import pagoda.physics


class Base(object):
    def setUp(self):
        self.world = pagoda.physics.World()
        self.box = self.world.create_body('box', lengths=(1, 2, 3))
        self.cap = self.world.create_body('cap', radius=1, length=2)
        self.cap.position = 1, 0, 0


class TestShapes(Base):
    def test_sphere(self):
        b = pagoda.physics.Sphere('sph', self.world, radius=3)
        assert b.radius == 3
        assert b.volume == 4/3 * np.pi * 27
        assert tuple(b.dimensions) == (6, 6, 6)

    def test_cap(self):
        b = pagoda.physics.Capsule('cap', self.world, radius=3, length=2)
        assert b.radius == 3
        assert b.length == 2
        assert b.volume == np.pi * (9 * 2 + 4/3 * 27)
        assert tuple(b.dimensions) == (6, 6, 8), b.dimensions

    def test_box(self):
        b = pagoda.physics.Box('box', self.world, lengths=(3, 4, 5))
        assert b.lengths == (3, 4, 5)
        assert b.volume == 3 * 4 * 5
        assert tuple(b.dimensions) == (3, 4, 5)

    def test_cylinder(self):
        b = pagoda.physics.Cylinder('cyl', self.world, radius=3, length=2)
        assert b.radius == 3
        assert b.length == 2
        assert b.volume == np.pi * 9 * 2
        assert tuple(b.dimensions) == (6, 6, 2)


class TestBody(Base):
    def test_state(self):
        st = self.box.state
        assert st == ('box0', (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))
        self.box.state = st

    def test_position(self):
        assert self.box.position == (0, 0, 0)
        self.box.position = 1, 2, 3
        assert self.box.position == (1, 2, 3)

    def test_rotation(self):
        assert self.box.rotation == (1, 0, 0, 0, 1, 0, 0, 0, 1)
        self.box.rotation = 1, 2, 3, 0, 1, 0, 2, 1, 3
        assert self.box.rotation == (
            0.2672612419124244, 0.5345224838248488, 0.8017837257372732,
            -0.16903085094570336, 0.8451542547285166, -0.50709255283711,
            -0.9486832980505139, -2.7755575615628914e-17, 0.316227766016838)

    def test_quaternion(self):
        assert self.box.quaternion == (1, 0, 0, 0)
        self.box.quaternion = 0, 1, 0, 0
        assert self.box.quaternion == (0, 1, 0, 0)
        self.box.quaternion = 0.5, 0.5, 0.4, 0.2
        assert self.box.quaternion == (0.5976143046671968,
                                       0.5976143046671968,
                                       0.47809144373375745,
                                       0.23904572186687872)

    def test_linear_velocity(self):
        assert self.box.linear_velocity == (0, 0, 0)
        self.box.linear_velocity = 1, 2, 3
        assert self.box.linear_velocity == (1, 2, 3)

    def test_angular_velocity(self):
        assert self.box.angular_velocity == (0, 0, 0)
        self.box.angular_velocity = 1, 2, 3
        assert self.box.angular_velocity == (1, 2, 3)

    def test_force(self):
        assert self.box.force == (0, 0, 0)
        assert self.box.torque == (0, 0, 0)

        self.box.force = 1, 2, 3
        assert self.box.force == (1, 2, 3)
        assert self.box.torque == (0, 0, 0)

        self.box.add_force((2, 0, 0))
        assert self.box.force == (3, 2, 3)
        assert self.box.torque == (0, 0, 0)

        self.box.add_force((2, 0, 0), relative=True)
        assert self.box.force == (5, 2, 3)
        assert self.box.torque == (0, 0, 0)

        self.box.add_force((2, 0, 0), position=(0, 1, 2))
        assert self.box.force == (7, 2, 3)
        assert self.box.torque == (0, 4, -2)

        self.box.add_force((2, 0, 0), relative_position=(0.5, 1, -1))
        assert self.box.force == (9, 2, 3)
        assert self.box.torque == (0, 2, -4)

    def test_torque(self):
        assert self.box.torque == (0, 0, 0)

        self.box.torque = 1, 2, 3
        assert self.box.torque == (1, 2, 3)

        self.box.add_torque((2, 0, 0))
        assert self.box.torque == (3, 2, 3)

        self.box.add_torque((2, 0, 0), relative=True)
        assert self.box.torque == (5, 2, 3)

    def test_is_kinematic(self):
        assert not self.box.is_kinematic
        self.box.is_kinematic = True
        assert self.box.is_kinematic
        self.box.is_kinematic = False
        assert not self.box.is_kinematic

    def test_follows_gravity(self):
        assert self.box.follows_gravity
        self.box.follows_gravity = False
        assert not self.box.follows_gravity
        self.box.follows_gravity = True
        assert self.box.follows_gravity

    def test_rotate_to_body(self):
        assert np.allclose(self.box.rotate_to_body((1, 0, 0)), (1, 0, 0))
        self.box.quaternion = 0, 1, 0, 1
        assert np.allclose(self.box.rotate_to_body((1, 0, 0)), (0, 0, 1))
        self.box.quaternion = 0, 1, 0, 0.3
        assert np.allclose(self.box.rotate_to_body((1, 0, 0)), (0.83486, 0, 0.55046))

    def test_body_to_world(self):
        assert self.box.body_to_world((1, 2, 3)) == (1, 2, 3)
        self.box.quaternion = 0, 1, 0, 1
        assert np.allclose(self.box.body_to_world((1, 2, 3)), (3, -2, 1))

    def test_world_to_body(self):
        assert self.box.world_to_body((1, 2, 3)) == (1, 2, 3)
        self.box.quaternion = 0, 1, 0, 1
        assert np.allclose(self.box.world_to_body((3, -2, 1)), (1, 2, 3))


class TestMotor(Base):
    def test_amotor(self):
        pass

    def test_lmotor(self):
        pass


class TestJoint(Base):
    def test_fixed(self):
        j = pagoda.physics.Fixed('fix', self.world, self.box)
        assert j is not None

    def test_slider(self):
        j = pagoda.physics.Slider('sli', self.world, self.box)
        assert j.positions == [0]
        assert j.position_rates == [0]

    def test_hinge(self):
        j = pagoda.physics.Hinge('hin', self.world, self.box, anchor=(0, 0, 0))
        assert j.axes == [(1, 0, 0)]
        assert j.angles == [0]
        assert j.angle_rates == [0]
        j.axes = [(0, 1, 0)]
        assert j.axes == [(0, 1, 0)]

    def test_universal(self):
        j = pagoda.physics.Universal('uni', self.world, self.box, anchor=(0, 0, 0))
        assert j.axes == [(1, 0, 0), (0, 1, 0)]
        assert j.angles == [0, 0]
        assert j.angle_rates == [0, 0]
        j.axes = [(0, 1, 0), (0, 0, 1)]
        assert j.axes == [(0, 1, 0), (0, 0, 1)]

    def test_ball(self):
        j = pagoda.physics.Ball('bal', self.world, self.box, anchor=(0, 0, 0))

    def test_join_to(self):
        b = pagoda.physics.Box('b', self.world, lengths=(1, 2, 3))
        assert not ode.areConnected(self.box.ode_body, b.ode_body)
        self.box.join_to('hinge', b)
        assert ode.areConnected(self.box.ode_body, b.ode_body)

    def test_connect_to(self):
        b = pagoda.physics.Box('b', self.world, lengths=(1, 2, 3))
        assert not ode.areConnected(self.box.ode_body, b.ode_body)
        assert self.box.position == (0, 0, 0)
        assert b.position == (0, 0, 0)

        self.box.connect_to('hinge', b, (1, 0, 0), (-1, 0, 0), name='j')
        assert ode.areConnected(self.box.ode_body, b.ode_body)
        assert self.box.position == (0, 0, 0)
        assert b.position == (1, 0, 0)

        j = self.world.get_joint('j')
        assert j.anchor == (0.5, 0, 0)


class TestWorld(Base):
    def test_gravity(self):
        assert self.world.gravity == (0, 0, -9.81)
        self.world.gravity = 0, 1, 0
        assert self.world.gravity == (0, 1, 0)

    def test_cfm(self):
        self.world.cfm = 0.1
        assert self.world.cfm == 0.1

    def test_erp(self):
        self.world.erp = 0.1
        assert self.world.erp == 0.1

    def test_create_body(self):
        s = self.world.create_body('sphere', 'foo', radius=3)
        assert self.world.get_body('foo') is s

    def test_join(self):
        j = self.world.join('hinge', self.box, self.cap, name='foo', anchor=(0, 0, 0))
        assert self.world.get_joint('foo') is j

    def test_body_states(self):
        states = self.world.get_body_states()
        assert states == [('box0', (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0)),
                          ('cap0', (1, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))]
        self.world.set_body_states(states)
        assert states == self.world.get_body_states()

    def test_are_connected(self):
        assert not self.world.are_connected('box0', 'cap0')
        self.world.join('hinge', 'box0', 'cap0')
        assert self.world.are_connected('box0', 'cap0')

    def test_on_collision(self):
        assert not self.world.are_connected('box0', 'cap0')
        self.world.on_collision(None, self.box.ode_geom, self.cap.ode_geom)
        assert self.world.are_connected('box0', 'cap0')
