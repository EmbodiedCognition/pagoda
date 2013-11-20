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

        body1 = self.get_body('body')
        body2 = self.get_body('neck')
        pos2 = body2.position
        pp1 = body1.relative_position(0, -0.2, -0.85, )
        pp2 = body2.relative_position(0, 0, 0.95, )
        body2.position = pp1 - pp2 + pos2

        joint = self.join('ball', 'head', 'neck', anchor=pp1,
                          angular_axis1=(1, 0, 0),
                          angular_axis1_mode=1,
                          angular_axis3=(0, 1, 0),
                          angular_axis3_mode=2,
        )
        joint.lo_stops = -TAU / 6, -TAU / 8, -TAU / 10
        joint.hi_stops = TAU / 8, TAU / 8, TAU / 10

  createBall( THROAT_JOINT,NECK_BODY,UP_TORSO_BODY,
    0,0,-.95,
    .9,-.2, 0,
    1,0,0,
    0,1,0,
    -M_PI/4,M_PI/9,
    -M_PI/4,M_PI/4,
    -M_PI/9,M_PI/9)

  createBall( R_COLLAR_JOINT,UP_TORSO_BODY,R_COLLAR_BODY,
    .75,0,-.35,
    0,0,1,
    1,0,0,
    0,1,0,
    -M_PI/6,M_PI/6,
    -M_PI/4,M_PI/4,
    -M_PI/6,M_PI/4)

  createBall( R_SHOULDER_JOINT,R_COLLAR_BODY,RUP_ARM_BODY,
    0,0,-.9,
    0,0,.85,
    1,0,0,
    0,1,0,
    -5*M_PI/9,5*M_PI/9,
    -M_PI/3,M_PI/3,
    -M_PI/4,2*M_PI/3)

  createUni( R_ELBOW_JOINT,RUP_ARM_BODY,RLO_ARM_BODY,
    0,0,-.95,
    0,0,.95,
    1,0,0,
    0,0,1,
    -4*M_PI/5,0.01,
    -M_PI/2,M_PI/2)
  createUni( R_WRIST_JOINT,RLO_ARM_BODY,R_HAND_BODY,
    0,0,-.95,
    0,0,1,
    1,0,0,
    0,1,0,
    -M_PI/5,M_PI/5,
    -M_PI/2,M_PI/2)

  createBall( L_COLLAR_JOINT,UP_TORSO_BODY,L_COLLAR_BODY,
    .75,0,.35,
    0,0,1,
    1,0,0,
    0,1,0,
    -M_PI/6,M_PI/6,
    -M_PI/4,M_PI/4,
    -M_PI/4,M_PI/6)

  createBall( L_SHOULDER_JOINT,L_COLLAR_BODY,LUP_ARM_BODY,
    0,0,-.9,
    0,0,.85,
    1,0,0,
    0,1,0,
    -5*M_PI/9,5*M_PI/9,
    -M_PI/3,M_PI/3,
    -2*M_PI/3,M_PI/4)

  createUni( L_ELBOW_JOINT,LUP_ARM_BODY,LLO_ARM_BODY,
    0,0,-.95,
    0,0,.95,
    1,0,0,
    0,0,1,
    -4*M_PI/5,0.01,
    -M_PI/2,M_PI/2)
  createUni( L_WRIST_JOINT,LLO_ARM_BODY,L_HAND_BODY,
    0,0,-.95,
    0,0,1,
    1,0,0,
    0,1,0,
    -M_PI/5,M_PI/5,
    -M_PI/2,M_PI/2)

  createBall( SPINE_JOINT,UP_TORSO_BODY,LO_TORSO_BODY,
    -.95,-.1,0,
    0,0,.90,
    1,0,0,
    0,1,0,
    - M_PI/4, M_PI/6,
    - M_PI/4, M_PI/4,
    - M_PI/6, M_PI/6)

  createBall( WAIST_JOINT,LO_TORSO_BODY,WAIST_BODY,
    0,0,-.9,
    0.5,0,0,
    1,0,0,
    0,1,0,
    - M_PI/4, M_PI/6,
    - M_PI/3, M_PI/3,
    - M_PI/6, M_PI/6)

  createBall( R_HIP_JOINT,WAIST_BODY,RUP_LEG_BODY,
    0,0,-.6,
    0,0,.95,
    1,0,0,
    0,1,0,
    -2*M_PI/3,M_PI/3,
    -M_PI/3,M_PI/3,
    -M_PI/6,2*M_PI/3)

  createUni( R_KNEE_JOINT,RUP_LEG_BODY,RLO_LEG_BODY,
    0,0,-.95,
    0,0,.95,
    1,0,0,
    0,0,1,
    -0.01,4*M_PI/5,
    -M_PI/5,M_PI/5)

  createUni( R_ANKLE_JOINT,RLO_LEG_BODY,R_HEEL_BODY,
            0,0,-.95,
            0,0,1,
            1,0,0,
            0,1,0,
            -M_PI/3,M_PI/3,
            -M_PI/3,M_PI/3)

  createHinge(R_FOOT_JOINT,R_HEEL_BODY,R_TARSAL_BODY,
    0,1.5,-1,
    -1,0,0,
    0,1,0,
    -M_PI/16,M_PI/16)

#  createHinge(R_TOE_JOINT,R_TARSAL_BODY,R_TOE_BODY,
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
