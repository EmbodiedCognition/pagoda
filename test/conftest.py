import pagoda
import pytest


def fn(s):
    return 'examples/{}'.format(s)


@pytest.fixture
def world():
    return pagoda.physics.World()


@pytest.fixture
def box(world):
    return world.create_body('box', lengths=(1, 2, 3))


@pytest.fixture
def cooper(request):
    world = pagoda.cooper.World()
    world.load_skeleton(fn('cooper-skeleton.txt'))
    world.load_markers(fn('cooper-motion.c3d'), fn('cooper-markers.txt'))
    request.addfinalizer(lambda: world.skeleton.disable_motors())
    return world
