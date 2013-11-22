'''Python implementation of forward-dynamics solver by Joseph Cooper.'''

import numpy as np
import numpy.random as rng
import ode

from . import physics

# CapBody
class World(physics.World):

    FMAX = 250
    INTERNAL_CFM = 0

    def __init__(self, *args, **kwargs):
        super(World, self).__init__(*args, **kwargs)

        self._create_bodies()
        self._create_joints()
        #self._create_root_joints()

    def set_random_forces(self):
        for body in self._bodies.values():
            body.add_force(30 * rng.randn(3))

    def _create_bodies(self):
        # We rotate some capsules when placing them
        # So we build quaternions representing that
        # rotation.
        tl = physics.make_quaternion(physics.TAU / 4, 0, 1, 0)
        tr = physics.make_quaternion(-physics.TAU / 4, 0, 1, 0)

        # We build all of the body parts first with the desired dimensions
        # (should probaly be loaded from file)
        self.create_body('sph', 'head', radius=0.13)
        self.create_body('box', 'neck', lengths=(0.08, 0.08, 0.08))
        self.create_body('cap', 'u-torso', radius=0.14, length=0.20).quaternion = tl
        self.create_body('box', 'l-torso', lengths=(0.35, 0.15, 0.20))
        self.create_body('cap', 'waist', radius=0.07, length=0.20).quaternion = tl

        self.create_body('cap', 'r-collar', radius=0.06, length=0.09).quaternion = tl
        self.create_body('cap', 'ru-arm', radius=0.05, length=0.27)
        self.create_body('cap', 'rl-arm', radius=0.04, length=0.17)
        self.create_body('sph', 'r-hand', radius=0.045)

        self.create_body('cap', 'l-collar', radius=0.06, length=0.09).quaternion = tr
        self.create_body('cap', 'lu-arm', radius=0.05, length=0.27)
        self.create_body('cap', 'll-arm', radius=0.04, length=0.17)
        self.create_body('sph', 'l-hand', radius=0.045)

        self.create_body('cap', 'ru-leg', radius=0.10, length=0.30)
        self.create_body('cap', 'rl-leg', radius=0.08, length=0.28)
        self.create_body('sph', 'r-heel', radius=0.07)
        self.create_body('cap', 'r-tarsal', radius=0.03, length=0.04).quaternion = tl
        #self.create_body('cap', 'r-toe', radius=0.03, length=0.03).quaternion = tl

        self.create_body('cap', 'lu-leg', radius=0.10, length=0.30)
        self.create_body('cap', 'll-leg', radius=0.08, length=0.28)
        self.create_body('sph', 'l-heel', radius=0.07)
        self.create_body('cap', 'l-tarsal', radius=0.03, length=0.04).quaternion = tr
        #self.create_body('cap', 'l-toe', radius=0.03, length=0.03).quaternion = tr

    def _create_joints(self):
        # Now the body parts are built, we move the head and then begin creating
        # joints and attaching them from the head down
        self.get_body('head').position = 0, 0, 2

        def reposition(body1, body2, offset1, offset2):
            body1 = self.get_body(body1)
            body2 = self.get_body(body2)
            pp1 = body1.body_to_world(offset1 * body1.dimensions / 2)
            pp2 = body2.body_to_world(offset2 * body2.dimensions / 2)
            body2.position = tuple(np.asarray(body2.position) + pp1 - pp2)
            return pp1

        def create_ball(body1, body2,
                        offset1, offset2,
                        lo_stops, hi_stops,
                        axis1=(1, 0, 0),
                        axis2=(0, 1, 0),
                        axis3=(0, 0, 1)):
            anchor = reposition(body1, body2, offset1, offset2)
            joint = self.join('ball', body1, body2, anchor=anchor,
                angular_mode=ode.AMotorEuler,
                angular_axis1=axis1, angular_axis1_frame=1,
                angular_axis2=axis2, angular_axis2_frame=1, #?
                angular_axis3=axis3, angular_axis3_frame=2)
            joint.lo_stops = -physics.TAU * np.asarray(lo_stops)
            joint.hi_stops = physics.TAU * np.asarray(hi_stops)
            #joint.max_forces = self.FMAX
            joint.cfm = self.INTERNAL_CFM

        def create_uni(body1, body2,
                       offset1, offset2,
                       lo_stops, hi_stops,
                       axis1=(1, 0, 0),
                       axis2=(0, 1, 0)):
            anchor = reposition(body1, body2, offset1, offset2)
            joint = self.join('universal', body1, body2, anchor=anchor,
                angular_axis1=axis1, angular_axis1_frame=1,
                angular_axis2=axis2, angular_axis2_frame=2)
            joint.lo_stops = -physics.TAU * np.asarray(lo_stops)
            joint.hi_stops = physics.TAU * np.asarray(hi_stops)
            joint.max_forces = self.FMAX
            joint.cfm = self.INTERNAL_CFM

        def create_hinge(body1, body2,
                         offset1, offset2,
                         lo_stops, hi_stops,
                         axis1=(1, 0, 0)):
            anchor = reposition(body1, body2, offset1, offset2)
            joint = self.join('hinge', body1, body2, anchor=anchor,
                angular_axis1=axis1, angular_axis1_frame=1)
            joint.lo_stops = -physics.TAU * np.asarray(lo_stops)
            joint.hi_stops = physics.TAU * np.asarray(hi_stops)
            joint.max_forces = self.FMAX
            joint.cfm = self.INTERNAL_CFM

        create_ball('head', 'neck',
                    offset1=(0, -0.2, -0.85),
                    offset2=(0, 0, 0.95),
                    lo_stops=(1./6, 1./8, 1./10),
                    hi_stops=(1./8, 1./8, 1./10))
        create_ball('neck', 'u-torso',
                    offset1=(0, 0, -0.95),
                    offset2=(0.9, -0.2, 0),
                    lo_stops=(1./8, 1./8, 1./18),
                    hi_stops=(1./18, 1./8, 1./18))
        create_ball('u-torso', 'l-torso',
                    offset1=(-0.95, -0.1, 0),
                    offset2=(0, 0, 0.90),
                    lo_stops=(1./8, 1./8, 1./12),
                    hi_stops=(1./12, 1./8, 1./12))
        create_ball('l-torso', 'waist',
                    offset1=(0, 0, -0.9),
                    offset2=(0.5, 0, 0),
                    lo_stops=(1./8, 1./6, 1./12),
                    hi_stops=(1./12, 1./6, 1./12))

        create_ball('u-torso', 'r-collar',
                    offset1=(0.75, 0, -0.35),
                    offset2=(0, 0, 1),
                    lo_stops=(1./12, 1./8, 1./12),
                    hi_stops=(1./12, 1./8, 1./8))
        create_ball('r-collar', 'ru-arm',
                    offset1=(0, 0, -0.9),
                    offset2=(0, 0, 0.85),
                    lo_stops=(5./18, 1./6, 1./8),
                    hi_stops=(5./18, 1./6, 1./3))
        create_uni('ru-arm', 'rl-arm',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 0.95),
                   axis2=(0, 0, 1),
                   lo_stops=(2./5, 1./4),
                   hi_stops=(0.01, 1./4))
        create_uni('rl-arm', 'r-hand',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 1),
                   lo_stops=(1./10, 1./4),
                   hi_stops=(1./10, 1./4))

        create_ball('u-torso', 'l-collar',
                    offset1=(0.75, 0, 0.35),
                    offset2=(0, 0, 1),
                    lo_stops=(1./12, 1./8, 1./8),
                    hi_stops=(1./12, 1./8, 1./12))
        create_ball('l-collar', 'lu-arm',
                    offset1=(0, 0, -0.9),
                    offset2=(0, 0, 0.85),
                    lo_stops=(5./18, 1./6, 1./3),
                    hi_stops=(5./18, 1./6, 1./8))
        create_uni('lu-arm', 'll-arm',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 0.95),
                   axis2=(0, 0, 1),
                   lo_stops=(2./5, 1./4),
                   hi_stops=(0.01, 1./4))
        create_uni('ll-arm', 'l-hand',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 1),
                   lo_stops=(1./10, 1./4),
                   hi_stops=(1./10, 1./4))

        create_ball('waist', 'ru-leg',
                    offset1=(0, 0, -0.6),
                    offset2=(0,0, 0.95),
                    lo_stops=(1./3, 1./6, 1./12),
                    hi_stops=(1./6, 1./6, 1./3))
        create_uni('ru-leg', 'rl-leg',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 0.95),
                   axis2=(0, 0, 1),
                   lo_stops=(-0.01, 1./10),
                   hi_stops=(2./5, 1./10))
        create_uni('rl-leg', 'r-heel',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 1),
                   lo_stops=(1./6, 1./6),
                   hi_stops=(1./6, 1./6))
        create_hinge('r-heel', 'r-tarsal',
                     offset1=(0, 1.5, -1),
                     offset2=(-1, 0, 0),
                     lo_stops=(1./32, ),
                     hi_stops=(1./32, ))
        #create_hinge('r-tarsal', 'r-toe',
        #    0,0,0,
        #    0,-2,0,
        #    1,0,0,
        #    -M_PI/8,M_PI/8)

        create_ball('waist', 'lu-leg',
                    offset1=(0, 0, 0.6),
                    offset2=(0, 0, 0.95),
                    lo_stops=(1./3, 1./6, 1./3),
                    hi_stops=(1./6, 1./6, 1./12))
        create_uni('lu-leg', 'll-leg',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 0.95),
                   axis2=(0, 0, 1),
                   lo_stops=(-0.01, 1./10),
                   hi_stops=(2./5, 1./10))
        create_uni('ll-leg', 'l-heel',
                   offset1=(0, 0, -0.95),
                   offset2=(0, 0, 1),
                   lo_stops=(1./6, 1./6),
                   hi_stops=(1./6, 1./6))
        create_hinge('l-heel', 'l-tarsal',
                     offset1=(0, 1.5, -1),
                     offset2=(1, 0, 0),
                     lo_stops=(1./32, ),
                     hi_stops=(1./32, ))
        #create_hinge('l-tarsal', 'l-toe',
        #    0,0,0,
        #    0,-2,0,
        #    1,0,0,
        #    -M_PI/8,M_PI/8)

    def _create_root_joints(self):
        root = self.get_body('waist')

        lm = self._root_lmotor = ode.LMotor(self.world)
        lm.attach(root, None)
        lm.setNumAxes(3)
        lm.setAxis(0, 0, (1, 0, 0))
        lm.setAxis(1, 0, (0, 1, 0))
        lm.setAxis(2, 0, (0, 0, 1))
        lm.setParam(ode.ParamVel1, 0)
        lm.setParam(ode.ParamVel2, 0)
        lm.setParam(ode.ParamVel3, 0)
        lm.setParam(ode.ParamCFM1, 1e-10)
        lm.setParam(ode.ParamCFM2, 1e-10)
        lm.setParam(ode.ParamCFM3, 1e-10)
        lm.setParam(ode.ParamFMax1, self.FMAX)
        lm.setParam(ode.ParamFMax2, self.FMAX)
        lm.setParam(ode.ParamFMax3, self.FMAX)
        lm.setFeedback(True)

        am = self._root_alimit = ode.AMotor(self.world)
        am.attach(root, None)
        am.setNumAxes(3)
        am.setMode(ode.AMotorEuler)
        am.setAxis(0, 1, 1, 0, 0)
        am.setAxis(2, 2, 0, 0, 1)
        am.setParam(ode.ParamLoStop1, -2 * physics.TAU / 9)
        am.setParam(ode.ParamHiStop1, 2 * physics.TAU / 9)
        am.setFeedback(True)

        am = self._root_amotor = ode.AMotor(self.world)
        am.attach(root, None)
        am.setNumAxes(3)
        am.setMode(ode.AMotorEuler)
        am.setAxis(0, 1, 1, 0, 0)
        am.setAxis(2, 2, 0, 0, 1)
        am.setParam(ode.ParamVel1, 0)
        am.setParam(ode.ParamVel2, 0)
        am.setParam(ode.ParamVel3, 0)
        am.setParam(ode.ParamCFM1, 1e-10)
        am.setParam(ode.ParamCFM2, 1e-10)
        am.setParam(ode.ParamCFM3, 1e-10)
        am.setParam(ode.ParamFMax1, self.FMAX)
        am.setParam(ode.ParamFMax2, self.FMAX)
        am.setParam(ode.ParamFMax3, self.FMAX)
        am.setFeedback(True)
