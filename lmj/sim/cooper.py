'''Python implementation of forward-dynamics solver by Joseph Cooper.'''

from . import physics

# CapBody
class World(physics.World):
    def __init__(self, *args, **kwargs):
        super(World, self).__init__(*args, **kwargs)

        self._create_bodies()
        self._create_joints()

    def _create_bodies(self):
        # We rotate some capsules when placing them
        # So we build quaternions representing that
        # rotation.
        turn_left = self.make_quaternion(-physics.TAU / 4, 0, 1, 0)

        # We build all of the body parts first with the desired dimensions
        # (should probaly be loaded from file)
        self.create_body('sphere', 'head', radius=0.13)
        self.create_body('box', 'neck', lengths=(0.08, 0.08, 0.08))
        b = self.create_body('cap', 'u-torso', radius=0.14, length=0.20)
        b.quaternion = turn_left
        self.create_body('box', 'l-torso', lengths=(0.35, 0.15, 0.20))
        b = self.create_body('cap', 'waist', radius=0.07, length=0.20)
        b.quaternion = turn_left

        b = self.create_body('cap', 'r-collar', radius=0.06, length=0.09)
        b.quaternion = turn_left
        self.create_body('cap', 'ru-arm', radius=0.05, length=0.27)
        self.create_body('cap', 'rl-arm', radius=0.04, length=0.17)
        self.create_body('sphere', 'r-hand', radius=0.045)

        b = self.create_body('cap', 'l-collar', radius=0.06, length=0.09)
        b.quaternion = turn_left
        self.create_body('cap', 'lu-arm', radius=0.05, length=0.27)
        self.create_body('cap', 'll-arm', radius=0.04, length=0.17)
        self.create_body('sphere', 'l-hand', radius=0.045)

        self.create_body('cap', 'ru-leg', radius=0.10, length=0.30)
        self.create_body('cap', 'rl-leg', radius=0.08, length=0.28)
        self.create_body('sphere', 'r-heel', radius=0.07)
        b = self.create_body('cap', 'r-tarsal', radius=0.03, length=0.04)
        b.quaternion = turn_left
        #b = self.create_body('cap', 'r-toe', radius=0.03, length=0.03)
        #b.quaternion = turn_left

        self.create_body('cap', 'lu-leg', radius=0.10, length=0.30)
        self.create_body('cap', 'll-leg', radius=0.08, length=0.28)
        self.create_body('sphere', 'l-heel', radius=0.07)
        b = self.create_body('cap', 'l-tarsal', radius=0.03, length=0.04)
        b.quaternion = turn_left
        #b = self.create_body('cap', 'l-toe', radius=0.03, length=0.03)
        #b.quaternion = turn_left

    def _create_joints(self):
        # Now the body parts are built, we move the head and then begin creating
        # joints and attaching them from the head down
        self.get_body('head').position = 0, 0, 2

        def reposition(body1, body2, offset1, offset2):
            body1 = self.get_body(body1)
            body2 = self.get_body(body2)
            pp1 = body1.body_to_world(offset1 * body1.dimensions / 2)
            pp2 = body2.body_to_world(offset2 * body2.dimensions / 2)
            body2.position = np.array(pp1) - pp2 + body2.position
            return pp1

        def create_ball(body1, body2,
                        offset1, offset2,
                        lo_stops, hi_stops,
                        axis1=(1, 0, 0),
                        axis3=(0, 1, 0)):
            anchor = reposition(body1, body2, offset1, offset2)
            joint = self.join('ball', body1, body2, anchor=anchor,
                angular_axis1=axis1, angular_axis1_mode=1,
                angular_axis3=axis3, angular_axis3_mode=2)
            joint.lo_stops = -TAU * np.array(lo_stops)
            joint.hi_stops = TAU * np.array(hi_stops)

        def create_uni(body1, body2,
                       offset1, offset2,
                       lo_stops, hi_stops,
                       axis1=(1, 0, 0),
                       axis2=(0, 1, 0)):
            anchor = reposition(body1, body2, offset1, offset2)
            joint = self.join('ball', body1, body2, anchor=anchor,
                angular_axis1=axis1, angular_axis2=axis2)
            joint.lo_stops = -TAU * np.array(lo_stops)
            joint.hi_stops = TAU * np.array(hi_stops)

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
                    lo_stops=(1./8, 1./8, 1./6),
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

  createBall( L_HIP_JOINT,WAIST_BODY,LUP_LEG_BODY,
    0,0,.6,
    0,0,.95,
    1,0,0,
    0,1,0,
    -2*M_PI/3,M_PI/3,
    -M_PI/3,M_PI/3,
    -2*M_PI/3,M_PI/6)

  createUni( L_KNEE_JOINT,LUP_LEG_BODY,LLO_LEG_BODY,
    0,0,-.95,
    0,0,.95,
    1,0,0,
    0,0,1,
    -0.01,4*M_PI/5,
    -M_PI/5,M_PI/5)

  createUni( L_ANKLE_JOINT,LLO_LEG_BODY,L_HEEL_BODY,
            0,0,-.95,
            0,0,1,
            1,0,0,
            0,1,0,
            -M_PI/3,M_PI/3,
            -M_PI/3,M_PI/3)

  createHinge(L_FOOT_JOINT,L_HEEL_BODY,L_TARSAL_BODY,
    0,1.5,-1,
    1,0,0,
    0,1,0,
    -M_PI/16,M_PI/16)

#  createHinge(L_TOE_JOINT,L_TARSAL_BODY,L_TOE_BODY,
#    0,0,0,
#    0,-2,0,
#    1,0,0,
#    -M_PI/8,M_PI/8)

  BodyType rootBody = HEAD_BODY


  motors[ROOT_LINMOTOR_JOINT] =
  joints[ROOT_LINMOTOR_JOINT] = dJointCreateLMotor(world,0)
  dJointAttach(joints[ROOT_LINMOTOR_JOINT],bodies[rootBody],0)
  dJointSetLMotorNumAxes(joints[ROOT_LINMOTOR_JOINT],3)
  dJointSetLMotorAxis(joints[ROOT_LINMOTOR_JOINT],0,0,1,0,0)
  dJointSetLMotorAxis(joints[ROOT_LINMOTOR_JOINT],1,0,0,1,0)
  dJointSetLMotorAxis(joints[ROOT_LINMOTOR_JOINT],2,0,0,0,1)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamVel1,0)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamVel2,0)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamVel3,0)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamCFM1,1e-10)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamCFM2,1e-10)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamCFM3,1e-10)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamFMax1,FMAX)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamFMax2,FMAX)
  dJointSetLMotorParam(joints[ROOT_LINMOTOR_JOINT],dParamFMax3,FMAX)
  jointInfo[ROOT_LINMOTOR_JOINT].link[0].id=rootBody
  jointInfo[ROOT_LINMOTOR_JOINT].link[1].id=-1

  limits[ROOT_ANGMOTOR_JOINT] =
  joints[ROOT_ANGMOTOR_JOINT] = dJointCreateAMotor(world,0)
  dJointAttach(joints[ROOT_ANGMOTOR_JOINT],bodies[rootBody],0)
  dJointSetAMotorNumAxes(joints[ROOT_ANGMOTOR_JOINT],3)
  dJointSetAMotorMode(joints[ROOT_ANGMOTOR_JOINT],dAMotorEuler)
  dJointSetAMotorAxis(joints[ROOT_ANGMOTOR_JOINT],0,1,1,0,0)
  dJointSetAMotorAxis(joints[ROOT_ANGMOTOR_JOINT],2,2,0,0,1)
  dJointSetAMotorParam(joints[ROOT_ANGMOTOR_JOINT],dParamLoStop1,-4*M_PI/9)
  dJointSetAMotorParam(joints[ROOT_ANGMOTOR_JOINT],dParamHiStop1, 4*M_PI/9)


  motors[ROOT_ANGMOTOR_JOINT] = dJointCreateAMotor(world,0)
  dJointAttach(motors[ROOT_ANGMOTOR_JOINT],bodies[rootBody],0)
  dJointSetAMotorNumAxes(motors[ROOT_ANGMOTOR_JOINT],3)
  dJointSetAMotorMode(motors[ROOT_ANGMOTOR_JOINT],dAMotorEuler)
  dJointSetAMotorAxis(motors[ROOT_ANGMOTOR_JOINT],0,1,1,0,0)
  dJointSetAMotorAxis(motors[ROOT_ANGMOTOR_JOINT],2,2,0,0,1)
  dJointSetAMotorParam(motors[ROOT_ANGMOTOR_JOINT],dParamVel1,0)
  dJointSetAMotorParam(motors[ROOT_ANGMOTOR_JOINT],dParamVel2,0)
  dJointSetAMotorParam(motors[ROOT_ANGMOTOR_JOINT],dParamVel3,0)
  dJointSetAMotorParam(joints[ROOT_ANGMOTOR_JOINT],dParamCFM1,1e-10)
  dJointSetAMotorParam(joints[ROOT_ANGMOTOR_JOINT],dParamCFM2,1e-10)
  dJointSetAMotorParam(joints[ROOT_ANGMOTOR_JOINT],dParamCFM3,1e-10)
  dJointSetAMotorParam(motors[ROOT_ANGMOTOR_JOINT],dParamFMax1,FMAX)
  dJointSetAMotorParam(motors[ROOT_ANGMOTOR_JOINT],dParamFMax2,FMAX)
  dJointSetAMotorParam(motors[ROOT_ANGMOTOR_JOINT],dParamFMax3,FMAX)




  jointInfo[ROOT_ANGMOTOR_JOINT].link[0].id=rootBody
  jointInfo[ROOT_ANGMOTOR_JOINT].link[1].id=-1

  for (int ii=0ii<JOINT_COUNT++ii) {
    dJointSetFeedback(motors[ii],&(feedback[ii]))
  }

  for (int ii=0ii<BODY_COUNT++ii) {
    getBody(ii)
    dBodySetData(bodies[ii],(void*)ii)
  }
}
