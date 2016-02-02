import numpy as np
import ode
import pagoda
import pytest


def test_fixed(box):
    j = pagoda.physics.Fixed('fix', box.world, box)
    assert j is not None


def test_slider(box):
    j = pagoda.physics.Slider('sli', box.world, box)
    assert j.positions == [0]
    assert j.position_rates == [0]


def test_hinge(box):
    j = pagoda.physics.Hinge('hin', box.world, box, anchor=(0, 0, 0))
    assert j.axes == [(1, 0, 0)]
    assert j.angles == [0]
    assert j.angle_rates == [0]
    j.axes = [(0, 1, 0)]
    assert j.axes == [(0, 1, 0)]


def test_universal(box):
    j = pagoda.physics.Universal('uni', box.world, box, anchor=(0, 0, 0))
    assert j.axes == [(1, 0, 0), (0, 1, 0)]
    assert j.angles == [0, 0]
    assert j.angle_rates == [0, 0]
    j.axes = [(0, 2, 1), (0, 0, 1)]
    assert j.axes == [(0, 2 / np.sqrt(5), 1 / np.sqrt(5)), (0, 0, 1)]


def test_ball(box):
    j = pagoda.physics.Ball('bal', box.world, box, anchor=(0, 0, 0))
    j.axes = [(1, 0, 0), (0, 0, 1)]
    assert j.axes == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert j.angles == [0, 0, 0]
    assert j.angle_rates == [0, 0, 0]


def test_join_to(box):
    b = pagoda.physics.Box('b', box.world, lengths=(1, 2, 3))
    assert not ode.areConnected(box.ode_body, b.ode_body)
    box.join_to('hinge', b)
    assert ode.areConnected(box.ode_body, b.ode_body)


def test_connect_to(box):
    b = pagoda.physics.Box('b', box.world, lengths=(1, 2, 3))
    assert not ode.areConnected(box.ode_body, b.ode_body)
    assert box.position == (0, 0, 0)
    assert b.position == (0, 0, 0)

    box.connect_to('hinge', b, (1, 0, 0), (-1, 0, 0), name='j')
    assert ode.areConnected(box.ode_body, b.ode_body)
    assert box.position == (0, 0, 0)
    assert b.position == (1, 0, 0)

    j = box.world.get_joint('j')
    assert j.anchor == (0.5, 0, 0)
