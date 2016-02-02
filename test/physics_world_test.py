import pagoda
import pytest


def test_gravity(world):
    assert world.gravity == (0, 0, -9.81)
    world.gravity = 0, 1, 0
    assert world.gravity == (0, 1, 0)


def test_cfm(world):
    world.cfm = 0.1
    assert world.cfm == 0.1


def test_erp(world):
    world.erp = 0.1
    assert world.erp == 0.1


def test_create_body(world):
    assert len(list(world.bodies)) == 0
    s1 = world.create_body('sphere', 'foo', radius=3)
    assert world.get_body('foo') is s1
    assert len(list(world.bodies)) == 1
    s2 = world.create_body('sphere', radius=2)
    assert world.get_body('sphere0') is s2
    assert len(list(world.bodies)) == 2


def test_join(world):
    box = world.create_body('box', lengths=(1, 1, 1))
    cap = world.create_body('cap', length=1, radius=0.1)
    cap.position = 0, 0, 1
    j = world.join('hinge', box, cap, name='foo', anchor=(0, 0, 0))
    assert world.get_joint('foo') is j


def test_body_states(world):
    assert world.get_body_states() == []
    box = world.create_body('box', lengths=(1, 1, 1))
    assert world.get_body_states() == [
        ('box0', (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))]
    BS = pagoda.physics.BodyState
    world.set_body_states([
        BS('box0', (1, 2, 3), (1, 0, 0, 0), (3, -1, 2), (0, 0, 0))])
    assert world.get_body_states() == [
        ('box0', (1, 2, 3), (1, 0, 0, 0), (3, -1, 2), (0, 0, 0))]


def test_are_connected(world):
    box = world.create_body('box', lengths=(1, 1, 1))
    cap = world.create_body('cap', length=1, radius=0.1)
    cap.position = 0, 0, 1
    assert not world.are_connected('box0', 'cap0')
    world.join('hinge', 'box0', 'cap0')
    assert world.are_connected('box0', 'cap0')


def test_on_collision(world):
    box = world.create_body('box', lengths=(1, 1, 1))
    cap = world.create_body('cap', length=1, radius=0.1)
    cap.position = 0, 0, 1
    assert not world.are_connected('box0', 'cap0')
    world.on_collision(None, box.ode_geom, cap.ode_geom)
    assert world.are_connected('box0', 'cap0')
