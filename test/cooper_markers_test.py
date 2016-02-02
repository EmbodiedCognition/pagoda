from conftest import fn
import pagoda
import pytest


@pytest.fixture
def markers():
    world = pagoda.cooper.World()
    return pagoda.cooper.Markers(world)


def test_c3d(markers):
    markers.load_c3d(fn('cooper-motion.c3d'))
    assert markers.num_frames == 343
    assert len(markers.bodies) == 41
    assert len(markers.targets) == 0
    assert len(markers.offsets) == 0
    assert len(markers.channels) == 41


def test_csv(markers):
    return  # TODO
    markers.load_csv(fn('cooper-motion.csv'))
    assert markers.num_frames == 343
    assert len(markers.bodies) == 41
    assert len(markers.targets) == 0
    assert len(markers.offsets) == 0
    assert len(markers.channels) == 41


def test_load_attachments(world):
    skel = pagoda.skeleton.Skeleton(world)
    skel.load(fn('cooper-skeleton.txt'))

    markers = pagoda.cooper.Markers(world)
    markers.load_c3d(fn('cooper-motion.c3d'))
    markers.load_attachments(fn('cooper-markers.txt'), skel)

    assert len(markers.targets) == 41
    assert len(markers.offsets) == 41
