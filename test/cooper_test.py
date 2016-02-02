import pagoda
import pytest


def test_load_skeleton(cooper):
    assert cooper.skeleton is not None


def test_load_markers(cooper):
    assert cooper.markers is not None


def test_follow_markers(cooper):
    left = cooper.markers.num_frames
    for states in cooper.follow_markers():
        assert left > 0
        left -= 1
    assert left == 0


def test_settle_to_markers(cooper):
    st000 = cooper.skeleton.get_body_states()
    st100 = cooper.settle_to_markers(100)
    assert st000 != st100
    st200 = cooper.settle_to_markers(200)
    assert st100 != st200


def test_inverse_kinematics(cooper):
    angles = list(cooper.inverse_kinematics(10))
    assert len(angles) == cooper.markers.num_frames - 10


def test_inverse_dynamics(cooper):
    angles = list(cooper.inverse_kinematics(10))
    torques = list(cooper.inverse_dynamics(angles))
    assert len(torques) == len(angles)
