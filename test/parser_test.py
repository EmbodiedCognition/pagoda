import pagoda

from conftest import fn


def test_cooper(world):
    with open(fn('cooper-skeleton.txt')) as h:
        visitor = pagoda.parser.parse(h, world, density=12, color='foo')
    assert len(visitor.bodies) == 21
    assert visitor.bodies[0].color == 'foo'


def test_hinge(world):
    with open(fn('hinge-limits.txt')) as h:
        visitor = pagoda.parser.parse(h, world, color='foo')
    assert len(visitor.bodies) == 2
    assert len(visitor.joints) == 2
    assert visitor.bodies[0].color == (0.9, 0.3, 0.1, 0.8)


def test_135(world):
    with open(fn('135.asf')) as h:
        visitor = pagoda.parser.parse_asf(h, world, color='foo')
    assert len(visitor.bodies) == 0
    assert len(visitor.joints) == 0
    assert len(visitor.bones) == 30
    assert len(visitor.hierarchy) == 24
    visitor.create_bodies()
    assert len(visitor.bodies) == 30
    assert len(visitor.joints) == 0
    assert visitor.bodies[0].color == 'foo'
    visitor.create_joints()
    assert len(visitor.joints) == 27
