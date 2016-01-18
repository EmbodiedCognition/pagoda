'''Parser for configuring objects and joints in a pagoda simulation.'''

import climate
import numpy as np
import re

from . import physics

logging = climate.get_logger(__name__)

FLOAT_RE = r'^[-+]?\d+(\.\d*)?([efgEFG][-+]?\d+(\.\d*)?)?$'


class Parser(object):
    '''Base class for skeleton parsers of various sorts.
    '''

    def __init__(self, world, jointgroup=None):
        self.world = world
        self.jointgroup = jointgroup

        self.joints = []
        self.bodies = []
        self.root = None

        self._filename = None
        self._tokens = []
        self._index = 0

    def load(self, source):
        '''Load body information from a file-like source.

        Parameters
        ----------
        source : str or file
            A filename or file-like object that contains text information
            describing a skeleton.
        '''
        if isinstance(source, str):
            self._filename = source
            source = open(source)
        else:
            self._filename = '(raw)'
        for lineno, line in enumerate(source):
            for tokenno, token in enumerate(line.split('#')[0].strip().split()):
                if token.strip():
                    self._tokens.append((lineno, tokenno, token))
        source.close()

    def _error(self, msg):
        '''Log a parsing error of some sort.

        Parameters
        ----------
        msg : str
            The specific error message to log.
        '''
        lineno, tokenno, token = self._tokens[self._index - 1]
        logging.fatal('%s:%d:%d: error parsing "%s": %s',
                      self._filename, lineno+1, tokenno+1, token, msg)
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
        if self._index < len(self._tokens):
            _, _, token = self._tokens[self._index]
            self._index += 1
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
        if self._index < len(self._tokens):
            return self._tokens[self._index][-1]
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
        return self._next_token(expect=FLOAT_RE, dtype=float)

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

    def parse(self, source):
        '''Load and parse a source file.

        Parameters
        ----------
        source : str or file
            A filename or file-like object that contains text information
            describing a skeleton.
        '''
        raise NotImplementedError


class BodyParser(Parser):
    '''This parser class reads a text configuration of a pagoda simulation.

    The format for the configuration file follows basic Unix conventions. A
    preprocessor discards any text on a line following the pound (#) character.
    Blank lines are skipped. The remainder of the file is chunked into sections
    based on keywords; each keyword in the file indicates the start of a new
    section. The recognized keywords are:

    - body -- indicates the definition of a new rigid body;
    - join -- indicates that two bodies will be joined together.

    **Bodies**

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

    - root -- indicates that this body is the root in the skeleton.

    **Joints**

    A "join" section must contain the following information:

    - A shape specification. This must be one of fixed, slider, hinge, piston,
      universal, or ball. The corresponding ODE joint type will be used to join
      together two bodies.

    - Two attachment specifications. These specify the bodies to be joined, as
      well as the body-relative offsets (on each body) where the joint will be
      anchored.

    Additionally, joints may specify information about their axes, rotation
    limits, etc.

    - axisN X Y Z -- specifies that axis number N (either 0, 1, or 2, and
      depending on the type of joint) points along the vector X Y Z. By default,
      axis0 points along 1 0 0, axis1 along 0 1 0, and axis2 along 0 0 1.

    - lo_stops A [B [C]] -- specifies the minimum permitted rotation (in
      degrees) for this joint along its respective axes.

    - hi_stops A [B [C]] -- specifies the maximum permitted rotation (in
      degrees) for this joint along its respective axes.

    - stop_cfm N -- set the lo- and hi-stop CFM for DOFs on this joint.

    - stop_erp N -- set the lo- and hi-stop ERP for DOFs on this joint.

    **Example**

    A simple example is probably sufficient to get started; also see the
    ``cooper-skeleton.txt`` file distributed with the pagoda source for a more
    complete example.

    .. code-block:: shell

      body box foo-body lengths 0.5 0.3 0.2    position 0 0 2
      body sph bar      radius 0.1
      body cap baz      radius 0.1 length 0.3  root

    This section of the example causes three bodies to be created:

    - A box identified as "foo-body" that is 50cm x 30cm x 20cm along the X, Y,
      and Z axes, respectively. foo-body will be positioned at world coordinates
      (0, 0, 2).

    - A sphere identified as "bar" that is 10cm in radius.

    - A capped cylinder identified as "baz" that is 10cm in radius and 30cm long
      (the total length of this geometry is actually 50cm due to the
      hemispherical cylinder end caps). This body is the root of the skeleton.

    .. code-block:: shell

      join hinge  foo-body 1 0 0  bar 0 -1 0
      lo_stops -10
      hi_stops  20
      join ball   bar      0 0 0  baz 0  0 1

    This section joins the three bodies together:

    - The center of the positive-X side of foo-body is joined to the point on
      the face of the bar sphere with the smallest Y coordinate, using a hinge
      joint (1 DOF). This joint is constrained to rotate between -10 and +20
      degrees.

    - The center of bar is joined to the point on baz with the largest Z
      coordinate using an unconstrained ball joint.
    '''

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
        mass = sum(b.mass.mass for b in self.bodies)
        vol = sum(b.volume for b in self.bodies)
        logging.info('%.1f mass / %f volume = %.1f overall density',
                     mass, vol, mass / vol)

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
            if token == 'density':
                kwargs[token] = self._next_float()
            if token == 'quaternion':
                theta, x, y, z = self._floats(4)
                quaternion = physics.make_quaternion(np.deg2rad(theta), x, y, z)
            if token == 'position':
                position = self._floats()
            if token == 'root':
                logging.info('"%s" will be used as a root', name)
                assert self.root is None, 'more than one root!'
                self.root = name
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
        offset1 = 0, 0, 0
        if body1 == 'world':
            body1 = None
        else:
            offset1 = self._floats()

        body2 = self._next_token(lower=False)
        offset2 = 0, 0, 0
        if body2 == 'world':
            body2 = None
        else:
            offset2 = self._floats()

        anchor = self.world.move_next_to(body1, body2, offset1, offset2)
        if shape.startswith('fix'):
            anchor = None

        token = self._next_token()
        axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        lo_stops = hi_stops = stop_cfm = stop_erp = None
        while token:
            if token in ('body', 'join'):
                break
            if token.startswith('axis'):
                axes[int(token.replace('axis', ''))] = self._floats()
            if token == 'lo_stops':
                lo_stops = np.deg2rad(self._floats(physics.JOINTS[shape].ADOF))
            if token == 'hi_stops':
                hi_stops = np.deg2rad(self._floats(physics.JOINTS[shape].ADOF))
            if token == 'stop_cfm':
                stop_cfm = self._next_float()
            if token == 'stop_erp':
                stop_erp = self._next_float()
            token = self._next_token()

        logging.info('joining %s %s %s', shape, body1, body2)

        joint = self.world.join(
            shape, body1, body2, anchor=anchor, jointgroup=self.jointgroup)

        if joint.ADOF or joint.LDOF:
            joint.axes = axes[:max(joint.ADOF, joint.LDOF)]

        if joint.ADOF and lo_stops is not None:
            joint.lo_stops = lo_stops
        if joint.ADOF and hi_stops is not None:
            joint.hi_stops = hi_stops
        if joint.ADOF and stop_cfm is not None:
            joint.stop_cfms = stop_cfm
        if joint.ADOF and stop_erp is not None:
            joint.stop_erps = stop_erp

        self.joints.append(joint)

        return token


class AsfParser(Parser):
    '''Parse a skeleton definition in ASF format.

    For a description of the file format, see
    http://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html

    The format describes only the configuration of joints in an articulated
    body. It notably omits the dimensions and masses of the rigid bodies that
    connect the joints together, so we need additional information to construct
    a skeleton in the physics simulator.
    '''

    TOPLEVEL_TOKENS = 'version name units documentation root bonedata hierarchy'.split()
    TOPLEVEL_TOKEN_RE = r'^:({})$'.format('|'.join(TOPLEVEL_TOKENS))

    def parse(self, source):
        '''Load and parse a source file.

        Parameters
        ----------
        source : str or file
            A filename or file-like object that contains text information
            describing a skeleton.
        '''
        toplevel = AsfParser.TOPLEVEL_TOKEN_RE
        self.load(source)
        token = self._next_token(expect=toplevel)
        while token is not None:
            if token == ':version':
                self.version = self._next_token()
                token = self._next_token(expect=toplevel)
            elif token == ':name':
                self.name = self._next_token()
                token = self._next_token(expect=toplevel)
            elif token == ':units':
                token = self._handle_units()
            elif token == ':documentation':
                token = self._handle_documentation()
            elif token == ':root':
                token = self._handle_root()
            elif token == ':bonedata':
                token = self._handle_bonedata()
            elif token == ':hierarchy':
                token = self._handle_hierarchy()
            else:
                self._error('unexpected token')

    def _handle_units(self):
        units = {}
        token = self._next_token()
        while token:
            value = self._next_token()
            if re.match(FLOAT_RE, value):
                value = float(value)
            units[token] = value
            token = self._next_token()
            if re.match(AsfParser.TOPLEVEL_TOKEN_RE, token):
                break
        self.units = units
        return token

    def _handle_documentation(self):
        token = self._next_token()
        tokens = []
        while token:
            if re.match(AsfParser.TOPLEVEL_TOKEN_RE, token):
                break
            tokens.append(token)
            token = self._next_token()
        self.documentation = ' '.join(tokens)
        return token

    def _handle_root(self):
        raise NotImplementedError

    def _handle_bonedata(self):
        raise NotImplementedError

    def _handle_bone(self):
        raise NotImplementedError

    def _handle_hierarchy(self):
        raise NotImplementedError
