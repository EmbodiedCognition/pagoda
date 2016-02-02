import numpy as np
import pagoda
import pytest


@pytest.fixture
def box():
    world = pagoda.physics.World()
    return world.create_body('box', lengths=(1, 2, 3))


def test_state(box):
    st = box.state
    assert st == ('box0', (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    box.state = st
    assert st == box.state


def test_position(box):
    assert box.position == (0, 0, 0)
    box.position = 1, 2, 3
    assert box.position == (1, 2, 3)


def test_rotation(box):
    assert box.rotation == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    box.rotation = 1, 2, 3, 0, 1, 0, 2, 1, 3
    assert box.rotation == (
        0.2672612419124244, 0.5345224838248488, 0.8017837257372732,
        -0.16903085094570336, 0.8451542547285166, -0.50709255283711,
        -0.9486832980505139, -2.7755575615628914e-17, 0.316227766016838)


def test_quaternion(box):
    assert box.quaternion == (1, 0, 0, 0)
    box.quaternion = 0, 1, 0, 0
    assert box.quaternion == (0, 1, 0, 0)
    box.quaternion = 0.5, 0.5, 0.4, 0.2
    assert box.quaternion == (0.5976143046671968,
                              0.5976143046671968,
                              0.47809144373375745,
                              0.23904572186687872)


def test_linear_velocity(box):
    assert box.linear_velocity == (0, 0, 0)
    box.linear_velocity = 1, 2, 3
    assert box.linear_velocity == (1, 2, 3)


def test_angular_velocity(box):
    assert box.angular_velocity == (0, 0, 0)
    box.angular_velocity = 1, 2, 3
    assert box.angular_velocity == (1, 2, 3)


def test_force(box):
    assert box.force == (0, 0, 0)
    assert box.torque == (0, 0, 0)

    box.force = 1, 2, 3
    assert box.force == (1, 2, 3)
    assert box.torque == (0, 0, 0)

    box.add_force((2, 0, 0))
    assert box.force == (3, 2, 3)
    assert box.torque == (0, 0, 0)

    box.add_force((2, 0, 0), relative=True)
    assert box.force == (5, 2, 3)
    assert box.torque == (0, 0, 0)

    box.add_force((2, 0, 0), position=(0, 1, 2))
    assert box.force == (7, 2, 3)
    assert box.torque == (0, 4, -2)

    box.add_force((2, 0, 0), relative_position=(0.5, 1, -1))
    assert box.force == (9, 2, 3)
    assert box.torque == (0, 2, -4)


def test_torque(box):
    assert box.torque == (0, 0, 0)

    box.torque = 1, 2, 3
    assert box.torque == (1, 2, 3)

    box.add_torque((2, 0, 0))
    assert box.torque == (3, 2, 3)

    box.add_torque((2, 0, 0), relative=True)
    assert box.torque == (5, 2, 3)


def test_is_kinematic(box):
    assert not box.is_kinematic
    box.is_kinematic = True
    assert box.is_kinematic
    box.is_kinematic = False
    assert not box.is_kinematic


def test_follows_gravity(box):
    assert box.follows_gravity
    box.follows_gravity = False
    assert not box.follows_gravity
    box.follows_gravity = True
    assert box.follows_gravity


def test_rotate_to_body(box):
    assert np.allclose(box.rotate_to_body((1, 0, 0)), (1, 0, 0))
    box.quaternion = 0, 1, 0, 1
    assert np.allclose(box.rotate_to_body((1, 0, 0)), (0, 0, 1))
    box.quaternion = 0, 1, 0, 0.3
    assert np.allclose(box.rotate_to_body((1, 0, 0)), (0.83486, 0, 0.55046))


def test_body_to_world(box):
    assert box.body_to_world((1, 2, 3)) == (1, 2, 3)
    box.quaternion = 0, 1, 0, 1
    assert np.allclose(box.body_to_world((1, 2, 3)), (3, -2, 1))


def test_world_to_body(box):
    assert box.world_to_body((1, 2, 3)) == (1, 2, 3)
    box.quaternion = 0, 1, 0, 1
    assert np.allclose(box.world_to_body((3, -2, 1)), (1, 2, 3))
