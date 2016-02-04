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
    assert np.allclose(box.position, (0, 0, 0))
    box.position = 1, 2, 3
    assert np.allclose(box.position, (1, 2, 3))


def test_rotation(box):
    assert np.allclose(box.rotation, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    box.rotation = np.array([[1, 2, 3], [0, 1, 0], [2, 1, 3]])
    assert np.allclose(box.rotation, [[0.2672612, 0.5345225, 0.8017837],
                                      [-0.1690309, 0.8451543, -0.5070926],
                                      [-0.9486833, -2.775558e-17, 0.3162278]])


def test_quaternion(box):
    assert np.allclose(box.quaternion, (1, 0, 0, 0))
    box.quaternion = 0, 1, 0, 0
    assert np.allclose(box.quaternion, (0, 1, 0, 0))
    box.quaternion = 0.5, 0.5, 0.4, 0.2
    assert np.allclose(box.quaternion,
                       (0.597614, 0.597614, 0.478091, 0.239045))


def test_linear_velocity(box):
    assert np.allclose(box.linear_velocity, (0, 0, 0))
    box.linear_velocity = 1, 2, 3
    assert np.allclose(box.linear_velocity, (1, 2, 3))


def test_angular_velocity(box):
    assert np.allclose(box.angular_velocity, (0, 0, 0))
    box.angular_velocity = 1, 2, 3
    assert np.allclose(box.angular_velocity, (1, 2, 3))


def test_force(box):
    assert np.allclose(box.force, (0, 0, 0))
    assert np.allclose(box.torque, (0, 0, 0))

    box.force = 1, 2, 3
    assert np.allclose(box.force, (1, 2, 3))
    assert np.allclose(box.torque, (0, 0, 0))

    box.add_force((2, 0, 0))
    assert np.allclose(box.force, (3, 2, 3))
    assert np.allclose(box.torque, (0, 0, 0))

    box.add_force((2, 0, 0), relative=True)
    assert np.allclose(box.force, (5, 2, 3))
    assert np.allclose(box.torque, (0, 0, 0))

    box.add_force((2, 0, 0), position=(0, 1, 2))
    assert np.allclose(box.force, (7, 2, 3))
    assert np.allclose(box.torque, (0, 4, -2))

    box.add_force((2, 0, 0), relative_position=(0.5, 1, -1))
    assert np.allclose(box.force, (9, 2, 3))
    assert np.allclose(box.torque, (0, 2, -4))


def test_torque(box):
    assert np.allclose(box.torque, (0, 0, 0))

    box.torque = 1, 2, 3
    assert np.allclose(box.torque, (1, 2, 3))

    box.add_torque((2, 0, 0))
    assert np.allclose(box.torque, (3, 2, 3))

    box.add_torque((2, 0, 0), relative=True)
    assert np.allclose(box.torque, (5, 2, 3))


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
    assert np.allclose(box.body_to_world((1, 2, 3)), (1, 2, 3))
    box.quaternion = 0, 1, 0, 1
    assert np.allclose(box.body_to_world((1, 2, 3)), (3, -2, 1))


def test_world_to_body(box):
    assert np.allclose(box.world_to_body((1, 2, 3)), (1, 2, 3))
    box.quaternion = 0, 1, 0, 1
    assert np.allclose(box.world_to_body((3, -2, 1)), (1, 2, 3))
