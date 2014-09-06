'''Parser for configuring objects and joints in a pagoda simulation.'''

import climate
import numpy as np
import re

from . import physics

logging = climate.get_logger(__name__)

TAU = 2 * np.pi


class Parser:
    '''This parser class reads a text configuration of a pagoda simulation.

    The format for the configuration file follows basic Unix conventions. A
    preprocessor discards any text on a line following the pound (#) character.
    Blank lines are skipped. The remainder of the file is chunked into sections
    based on keywords; each keyword in the file indicates the start of a new
    section. The recognized keywords are:

    - body -- indicates the definition of a new rigid body;
    - join -- indicates that two bodies will be joined together.

    Bodies
    ------

    A "body" section must contain the following information:

    - A shape specification. This must be one of box (rectangular prism), sph or
      sphere (sphere), cyl or cylinder (cylinder), or cap (capped cylinder).

      The shape indicates the canonical geometry that will be used to (a)
      specify the size of the body, and (b) detect collisions.

    - An identifier for the body. This cannot contain spaces.

    - One or more dimension parameters. Each of these consists of a shape
      keyword followed by one or more size in meters. Different dimension
      keywords are required based on the shape of the body:

      - box: length X Y Z
      - sph: radius R
      - cyl: length L radius R
      - cap: length L radius R

    Additionally, a body definition may be followed by extra quantities
    specifying the position, orientation, and other quantities of the body.

    - position X Y Z -- specifies the absolute location of the body in world
      coordinates. Typically this is only given for one object. By default
      bodies are all created at the origin.

    - quaternion W X Y Z -- specifies the angle (in degrees) and axis of
      rotation for the body. By default bodies are created without any rotation
      (i.e., with a 0 1 0 0 quaternion).

    - root -- indicates that this body is a root in the skeleton.

    Joints
    ------

    A "join" section must contain the following information:

    - A shape specification. This must be one of fixed, slider, hinge, piston,
      universal, or ball. The corresponding ODE joint type will be used to join
      together two bodies.

    - Two attachment specifications. These specify the bodies to be joined, as
      well as the body-relative offsets (on each body) where the joint will be
      anchored.

    Additionally, joints may specify information about their axes, rotation
    limits, etc.

    - axisN X [Y [Z]] -- specifies that axis number N (either 0, 1, or 2, and
      depending on the type of joint) points along the vector X Y Z. By default,
      axis0 points along 1 0 0, axis1 along 0 1 0, and axis2 along 0 0 1.

    - lo_stops A [B [C]] -- specifies the minimum permitted rotation (in
      degrees) for this joint along its respective axes.

    - hi_stops A [B [C]] -- specifies the maximum permitted rotation (in
      degrees) for this joint along its respective axes.

    - stop_cfm N -- set the lo- and hi-stop CFM for DOFs on this joint.

    - stop_erp N -- set the lo- and hi-stop ERP for DOFs on this joint.

    - passive -- indicates that this joint's motor remains disabled.

    Example
    -------

    A simple example is probably sufficient to get started; also see the
    ``cooper-skeleton.txt`` file distributed with the pagoda source for a more
    complete example.

    | body box foo-body lengths 0.5 0.3 0.2    position 0 0 2
    | body sph bar      radius 0.1
    | body cap baz      radius 0.1 length 0.3  root

    This section of the example causes three bodies to be created:

    - A box identified as "foo-body" that is 50cm x 30cm x 20cm along the X, Y,
      and Z axes, respectively. foo-body will be positioned at world coordinates
      (0, 0, 2).

    - A sphere identified as "bar" that is 10cm in radius.

    - A capped cylinder identified as "baz" that is 10cm in radius and 30cm long
      (the total length of this geometry is actually 50cm due to the
      hemispherical cylinder end caps). This body is a root of the skeleton.

    | join hinge  foo-body 1 0 0  bar 0 -1 0
    | lo_stops -10
    | hi_stops  20
    | join ball   bar      0 0 0  baz 0  0 1

    This section joins the three bodies together:

    - The center of the positive-X side of foo-body is joined to the point on
      the face of the bar sphere with the smallest Y coordinate, using a hinge
      joint (1 DOF). This joint is constrained to rotate between -10 and +20
      degrees.

    - The center of bar is joined to the point on baz with the largest Z
      coordinate using an unconstrained ball joint.

    '''

    def __init__(self, world, jointgroup=None):
        self.world = world
        self.jointgroup = jointgroup

        self.joints = []
        self.bodies = []
        self.roots = []

        self.filename = None
        self.tokens = []
        self.index = 0

    def load(self, source):
        '''Load body information from a file-like source.

        Parameters
        ----------
        source : str or file
            A filename or file-like object that contains text information
            describing a skeleton.
        '''
        if isinstance(source, str):
            self.filename = source
            source = open(source)
        else:
            self.filename = '(file-{:r})'.format(source)
        for i, line in enumerate(source):
            for j, token in enumerate(line.split('#')[0].strip().split()):
                if token.strip():
                    self.tokens.append((i, j, token))
        source.close()

    def _error(self, msg):
        '''Log a parsing error of some sort.

        Parameters
        ----------
        msg : str
            The specific error message to log.
        '''
        lineno, tokenno, token = self.tokens[self.index - 1]
        logging.fatal('%s:%d:%d: error parsing "%s": %s',
                      self.filename, lineno+1, tokenno+1, token, msg)
        raise RuntimeError

    def _next_token(self, expect=None, lower=True, dtype=None):
        '''Get the next token in our parsing state, and update the state.

        Parameters
        ----------
        expect : regexp, optional
            If given, require the next token to match the given expression.
        lower : bool, optional
            Convert the token to lowercase. True by default.
        dtype : callable, optional
            If provided, call this method with the given token. Used to convert
            tokens to int or float automatically.

        Return
        ------
        str or number :
            The next token in the file being parsed.
        '''
        if self.index < len(self.tokens):
            _, _, token = self.tokens[self.index]
            self.index += 1
            if lower:
                token = token.lower()
            if expect and re.match(expect, token) is None:
                return self._error('expected {}'.format(expect))
            if callable(dtype):
                token = dtype(token)
            return token
        return None

    def _peek_token(self):
        '''Get the next token in our parsing state, but do not update the state.

        Return
        ------
        str :
            The next token in the file being parsed.
        '''
        if self.index < len(self.tokens):
            return self.tokens[self.index][-1]
        return None

    def _next_float(self):
        '''Get a floating-point token and update the parsing state.

        Return
        ------
        float :
            The floating-point value of the next token in the source.

        Raise
        -----
            Logs an error if the next token does not match a float regexp.
        '''
        return self._next_token(expect=r'^[-+]?\d+(\.\d*)?([efgEFG][-+]?\d+(\.\d*)?)?$', dtype=float)

    def _floats(self, n=3):
        '''Get a number of floats and update the parsing state.

        Parameters
        ----------
        n : int
            The number of floating-point values to retrieve and return.

        Return
        ------
        list of float :
            The floating-point value of the next token in the source.

        Raise
        -----
            Logs an error if the next n tokens do not match a float regexp.
        '''
        return [self._next_float() for _ in range(n)]

    def _handle_body(self):
        '''Parse the entirety of a "body" section in the source.'''
        shape = self._next_token(expect='^({})$'.format('|'.join(physics.BODIES)))
        name = self._next_token(lower=False)

        kwargs = {}
        quaternion = None
        position = None
        token = self._next_token()
        while token:
            if token in ('body', 'join'):
                break
            if token == 'lengths':
                kwargs[token] = self._floats()
            if token == 'radius':
                kwargs[token] = self._next_float()
            if token == 'length':
                kwargs[token] = self._next_float()
            if token == 'quaternion':
                theta, x, y, z = self._floats(4)
                quaternion = physics.make_quaternion(TAU * theta / 360, x, y, z)
            if token == 'position':
                position = self._floats()
            if token == 'root':
                logging.info('"%s" will be used as a root', name)
                self.roots.append(name)
            token = self._next_token()

        logging.info('creating %s %s %s', shape, name, kwargs)

        body = self.world.create_body(shape, name, **kwargs)
        if quaternion is not None:
            logging.info('setting rotation %s', quaternion)
            body.quaternion = quaternion
        if position is not None:
            logging.info('setting position %s', position)
            body.position = position

        self.bodies.append(body)

        return token

    def _handle_joint(self):
        '''Parse the entirety of a "join" section in the source.'''
        shape = self._next_token(expect='^({})$'.format('|'.join(physics.JOINTS)))
        body1 = self._next_token(lower=False)
        offset1 = self._floats()
        body2 = self._next_token(lower=False)
        offset2 = self._floats()

        anchor = self.world.move_next_to(body1, body2, offset1, offset2)

        token = self._next_token()
        axes = [(1, 0, 0), (0, 1, 0)]
        lo_stops = hi_stops = stop_cfm = stop_erp = None
        is_passive = False
        while token:
            if token in ('body', 'join'):
                break
            if token.startswith('axis'):
                axes[int(token.replace('axis', ''))] = self._floats()
            if token == 'lo_stops':
                lo_stops = np.deg2rad(self._floats(physics.JOINTS[shape].ADOF))
            if token == 'hi_stops':
                hi_stops = np.deg2rad(self._floats(physics.JOINTS[shape].ADOF))
            if token == 'passive':
                is_passive = True
            if token == 'stop_cfm':
                stop_cfm = self._next_float()
            if token == 'stop_erp':
                stop_erp = self._next_float()
            token = self._next_token()

        logging.info('joining %s %s %s', shape, body1, body2)

        joint = self.world.join(
            shape, body1, body2, anchor=anchor, jointgroup=self.jointgroup)
        joint.axes = axes[:joint.ADOF]
        joint.is_passive = is_passive
        if lo_stops is not None:
            joint.lo_stops = lo_stops
        if hi_stops is not None:
            joint.hi_stops = hi_stops
        if stop_cfm is not None:
            joint.stop_cfms = stop_cfm
        if stop_erp is not None:
            joint.stop_erps = stop_erp

        self.joints.append(joint)

        return token

    def parse(self, source):
        '''Load and parse a source file.

        Parameters
        ----------
        source : str or file
            A filename or file-like object that contains text information
            describing a skeleton.
        '''
        self.load(source)
        token = self._next_token(expect='^(body|joint)$')
        while token is not None:
            if token == 'body':
                token = self._handle_body()
            elif token == 'join':
                token = self._handle_joint()
            else:
                self._error('unexpected token')
