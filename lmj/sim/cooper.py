'''Python implementation of forward-dynamics solver by Joseph Cooper.'''

import lmj.cli
import numpy as np
import numpy.random as rng
import ode
import re

from . import physics

logging = lmj.cli.get_logger(__name__)


class World(physics.World):
    FMAX = 250
    CFM = 1e-10
    INTERNAL_CFM = 0

    _joint_index = 0
    _axis_index = 0
    _step_index = 0
    _steps = np.linspace(0, physics.TAU, 31)

    def set_random_forces(self):
        for body in self._bodies.values():
            body.add_force(self.FMAX * rng.randn(3))

    def reset(self):
        for b in self._bodies.itervalues():
            b.position = b.position + np.array([0, 0, 2])

    def create_from_file(self, filename):
        config = []
        index = [0]
        with open(filename) as handle:
            for i, line in enumerate(handle):
                for j, token in enumerate(line.split('#')[0].strip().split()):
                    if token.strip():
                        config.append((i, j, token))

        def next_token(expect=None, lower=True, dtype=None):
            if index[0] < len(config):
                _, _, token = config[index[0]]
                index[0] += 1
                if lower:
                    token = token.lower()
                if expect and re.match(expect, token) is None:
                    return error('expected {}'.format(expect))
                if callable(dtype):
                    token = dtype(token)
                return token
            return None

        def error(msg):
            lineno, tokenno, token = config[index[0]-1]
            logging.fatal('%s:%d:%d: error parsing "%s": %s',
                          filename, lineno+1, tokenno+1, token, msg)

        def array(n=3, dtype=float):
            return np.array([next_token(expect=r'^-?\d+(\.\d*)?$', dtype=dtype) for _ in range(n)])

        tl = physics.make_quaternion(physics.TAU / 4, 0, 1, 0)
        tr = physics.make_quaternion(-physics.TAU / 4, 0, 1, 0)
        turns = {'turn-left': tl, 'tl': tl, 'turn-right': tr, 'tr': tr}

        def handle_body():
            shape = next_token(expect='^({})$'.format('|'.join(physics.BODIES)))
            name = next_token(lower=False)
            token = next_token()
            kwargs = {}
            while '=' in token:
                k, v = token.split('=', 1)
                kwargs[k.strip()] = float(v.strip())
                token = next_token()
            logging.info('creating %s %s %s', shape, name, kwargs)
            body = self.create_body(shape, name, **kwargs)
            if name == 'head':
                body.position = 0, 0, 2
            turn = turns.get(token)
            if turn is None:
                return token
            body.quaternion = turn
            return next_token()

        def handle_joint():
            shape = next_token(expect='^({})$'.format('|'.join(physics.JOINTS)))
            body1 = next_token(lower=False)
            offset1 = array()
            body2 = next_token(lower=False)
            offset2 = array()
            kwargs = dict(
                anchor=self.move_next_to(body1, body2, offset1, offset2),
                angular_axis1=(1, 0, 0),
                angular_axis1_frame=1,
                angular_axis2=(0, 1, 0),
                angular_axis2_frame=1,
                angular_axis3=(0, 0, 1),
                angular_axis3_frame=2,
            )
            if shape == 'ball':
                kwargs['amotor_mode'] = ode.AMotorEuler
            token = next_token()
            while token:
                if token in ('body', 'join'):
                    break
                key = token
                value = None
                if key.startswith('angular_axis'):
                    value = array()
                if key == 'lo_stops' or key == 'hi_stops':
                    value = physics.TAU * array(physics.JOINTS[shape].ADOF) / 360
                token = next_token()
            logging.info('joining %s %s %s\n%s', shape, body1, body2,
                         '\n'.join('{} = {}'.format(k, kwargs[k]) for k in sorted(kwargs)))
            joint = self.join(shape, body1, body2, **kwargs)
            joint.motor_cfms = self.CFM
            #joint.max_forces = self.FMAX
            return token

        token = next_token(expect='^(body|joint)$')
        while token is not None:
            if token == 'body':
                token = handle_body()
            elif token == 'join':
                token = handle_joint()
            else:
                error('unexpected token')

        root = self.get_body('head')

        lm = self._root_lmotor = ode.LMotor(self.world)
        lm.attach(root.ode_body, None)
        lm.setNumAxes(3)
        lm.setAxis(0, 0, (1, 0, 0))
        lm.setAxis(1, 0, (0, 1, 0))
        lm.setAxis(2, 0, (0, 0, 1))
        lm.setParam(ode.ParamVel, 0)
        lm.setParam(ode.ParamVel2, 0)
        lm.setParam(ode.ParamVel3, 0)
        lm.setParam(ode.ParamCFM, self.CFM)
        lm.setParam(ode.ParamCFM2, self.CFM)
        lm.setParam(ode.ParamCFM3, self.CFM)
        lm.setParam(ode.ParamFMax, 100 * self.FMAX)
        lm.setParam(ode.ParamFMax2, 100 * self.FMAX)
        lm.setParam(ode.ParamFMax3, 100 * self.FMAX)
        lm.setFeedback(True)

        am = self._root_alimit = ode.AMotor(self.world)
        am.attach(root.ode_body, None)
        am.setNumAxes(3)
        am.setMode(ode.AMotorEuler)
        am.setAxis(0, 1, (1, 0, 0))
        am.setAxis(2, 2, (0, 0, 1))
        am.setParam(ode.ParamLoStop, -2 * physics.TAU / 9)
        am.setParam(ode.ParamHiStop, 2 * physics.TAU / 9)
        am.setFeedback(True)

        am = self._root_amotor = ode.AMotor(self.world)
        am.attach(root.ode_body, None)
        am.setNumAxes(3)
        am.setMode(ode.AMotorEuler)
        am.setAxis(0, 1, (1, 0, 0))
        am.setAxis(2, 2, (0, 0, 1))
        am.setParam(ode.ParamVel, 0)
        am.setParam(ode.ParamVel2, 0)
        am.setParam(ode.ParamVel3, 0)
        am.setParam(ode.ParamCFM, self.CFM)
        am.setParam(ode.ParamCFM2, self.CFM)
        am.setParam(ode.ParamCFM3, self.CFM)
        am.setParam(ode.ParamFMax, self.FMAX)
        am.setParam(ode.ParamFMax2, self.FMAX)
        am.setParam(ode.ParamFMax3, self.FMAX)
        am.setFeedback(True)

    def step(self, substeps=2):
        joint = list(self.joints)[self._joint_index]

        lo_stop = list(joint.lo_stops)[self._axis_index]
        hi_stop = list(joint.hi_stops)[self._axis_index]

        angles = list(joint.angles)
        angles[self._axis_index] = lo_stop + self._steps[self._step_index] * (hi_stop - lo_stop)
        joint.angles = angles

        logging.info('joint %s, axis %s, step %s: %s %s',
                     self._joint_index, self._axis_index, self._step_index,
                     joint, joint.angles)

        self._step_index += 1
        if self._step_index == len(self._steps):
            self._step_index = 0
            self._axis_index += 1
            if self._axis_index == joint.ADOF:
                self._axis_index = 0
                self._joint_index = (self._joint_index + 1) % len(self._joints)

        for _ in range(substeps):
            self.world.step(self.dt / substeps)

        return True
