'''Parser for configuring objects and joints in a pagoda simulation.'''

from __future__ import division, absolute_import, unicode_literals

import climate
import numpy as np
import os
import parsimonious
import re

from . import physics

logging = climate.get_logger(__name__)


def parse(source, world, jointgroup=None, density=1000, color=None):
    '''Load and parse a source file.

    Parameters
    ----------
    source : file
        A file-like object that contains text information describing bodies and
        joints to add to the world.
    world : :class:`pagoda.physics.World`
        The world to add objects and joints to.
    jointgroup : ode.JointGroup, optional
        If provided, add all joints from this parse to the given group. The
        default behavior adds joints to the world without an explicit group.
    density : float, optional
        Default density for bodies. This is overridden if the source provides a
        density or mass value for a given body.
    color : tuple of floats, optional
        Default color for bodies from this source. Defaults to None, which does
        not assign a color to parsed bodies.
    '''
    visitor = Visitor(world, jointgroup, density, color)
    visitor.parse(source.read())
    return visitor


class Visitor(parsimonious.nodes.NodeVisitor):
    '''
    '''

    grammar = parsimonious.grammar.Grammar(
        open(os.path.join(os.path.dirname(__file__), 'pagoda.peg')).read())

    def __init__(self, world, jointgroup=None, density=1000, color=None,
                 *args, **kwargs):
        super(Visitor, self).__init__(*args, **kwargs)

        self.world = world
        self.jointgroup = jointgroup
        self.density = density
        self.color = color

        self.bodies = []
        self.joints = []

        self.attrs = {}

    def generic_visit(self, node, children):
        return node.text

    def visit_(self, node, children):
        return children

    def visit_number(self, node, children):
        return float(node.text)

    def visit_float(self, node, children):
        return float(node.text)

    def visit_int(self, node, children):
        return int(node.text)

    def visit_tuplecolor(self, node, children):
        r, _, g, _, b, rest = tuple(children)
        a = rest[0][1] if rest else 1
        return r, g, b, a

    def visit_hexcolor(self, node, children):
        r = g = b = a = 1
        if 3 <= len(node.text) <= 4:
            r = int(node.text[0], 16) / 16
            g = int(node.text[1], 16) / 16
            b = int(node.text[2], 16) / 16
            if len(node.text) == 4:
                a = int(node.text[3], 16) / 16
        else:
            r = int(node.text[0:2], 16) / 256
            g = int(node.text[2:4], 16) / 256
            b = int(node.text[4:6], 16) / 256
            if len(node.text) == 8:
                a = int(node.text[6:8], 16) / 256
        return r, g, b, a

    def visit_color(self, node, children):
        self.attrs['color'] = children[-1][0]

    def visit_mass(self, node, children):
        self.attrs['mass'] = children[-1]

    def visit_density(self, node, children):
        self.attrs['density'] = children[-1]

    def visit_position(self, node, children):
        _, _, x, _, y, _, z = children
        self.attrs['position'] = x, y, z

    def visit_quaternion(self, node, children):
        _, _, theta, _, x, _, y, _, z = children
        self.attrs['quaternion'] = physics.make_quaternion(
            np.deg2rad(theta), x, y, z)

    def visit_handofgod(self, node, children):
        self.attrs['handofgod'] = True

    def visit_radius(self, node, children):
        self.attrs['radius'] = children[0]

    def visit_radlen(self, node, children):
        self.attrs['length'] = [x for x in children[0] if isinstance(x, float)][0]

    def visit_lengths(self, node, children):
        x, _, y, _, z = children
        self.attrs['lengths'] = x, y, z

    def visit_body(self, node, children):
        name, _, shape = children[:3]

        if 'mass' not in self.attrs and 'density' not in self.attrs:
            self.attrs['density'] = self.density

        logging.info('creating %s %s %s', shape, name, self.attrs)

        handofgod = self.attrs.pop('handofgod', None)
        position = self.attrs.pop('position', None)
        quaternion = self.attrs.pop('quaternion', None)
        color = self.attrs.pop('color', self.color)

        body = self.world.create_body(shape, name, **self.attrs)

        if quaternion:
            body.quaternion = quaternion
        if position:
            body.position = position
        if color:
            body.color = color
        if handofgod:
            logging.info('adding hand-of-god forces to %s', name)
            hog = physics.AMotor(
                '{}:hog'.format(name), self.world, body, mode='euler',
                jointgroup=self.jointgroup)
            hog.axes = [dict(rel=1, axis=(1, 0, 0)), None,
                        dict(rel=2, axis=(0, 0, 1))]
            self.joints.append(hog)

        self.bodies.append(body)

        self.attrs = {}

    def visit_attach(self, node, children):
        name, offset = children
        return name, offset[0][1] if offset else (0, 0, 0)

    def visit_offset(self, node, children):
        _, _, x, _, y, _, z, _, _ = children
        return x, y, z

    def visit_axis(self, node, children):
        _, _, x, _, y, _, z, _, _ = children
        return x, y, z

    def visit_range(self, node, children):
        lo, _, _, _, hi = children
        return lo, hi

    def visit_fmax(self, node, children):
        _, values = children
        self.attrs['fmax'] = [x for _, x in values]

    def visit_cfms(self, node, children):
        _, values = children
        self.attrs['cfms'] = [x for _, x in values]

    def visit_erps(self, node, children):
        _, values = children
        self.attrs['erps'] = [x for _, x in values]

    def visit_stops(self, node, children):
        _, ranges = children
        self.attrs['lo_stops'] = [a for _, (a, b) in ranges]
        self.attrs['hi_stops'] = [b for _, (a, b) in ranges]

    def visit_stop_cfms(self, node, children):
        _, values = children
        self.attrs['stop_cfms'] = [x for _, x in values]

    def visit_stop_erps(self, node, children):
        _, values = children
        self.attrs['stop_erps'] = [x for _, x in values]

    def visit_axes(self, node, children):
        _, axes = children
        self.attrs['axes'] = [ax for _, ax in axes]

    def visit_joint(self, node, children):
        shape, _, (body1, offset1), _, motion, _, (body2, offset2) = children[:7]

        if body1.lower() == 'world':
            body1 = None
        if body2.lower() == 'world':
            body2 = None

        anchor = None
        if motion == '<-':
            anchor = self.world.move_next_to(body1, body2, offset1, offset2)
        if motion == '->':
            anchor = self.world.move_next_to(body2, body1, offset2, offset1)
        if motion == '<>':
            anchor = self.world.get_body(body1).position  # TODO
        if shape.startswith('fix') or shape.startswith('sli'):
            anchor = None

        logging.info('joining %s <-%s-> %s at %s', body1, shape, body2,
                     [round(a, 3) for a in anchor or []])

        joint = self.world.join(shape, body1, body2, anchor=anchor,
                                jointgroup=self.jointgroup)

        default_axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        joint.axes = self.attrs.get('axes', default_axes[:joint.ADOF])

        if 'erps' in self.attrs:
            joint.erps = self.attrs['erps']
        if 'cfms' in self.attrs:
            joint.cfms = self.attrs['cfms']
        if 'fmax' in self.attrs:
            joint.fmax = self.attrs['fmax']
        if 'lo_stops' in self.attrs:
            if not shape.startswith('sli'):
                self.attrs['lo_stops'] = np.deg2rad(self.attrs['lo_stops'])
                self.attrs['hi_stops'] = np.deg2rad(self.attrs['hi_stops'])
            joint.lo_stops = self.attrs['lo_stops']
            joint.hi_stops = self.attrs['hi_stops']
            if 'stop_cfms' in self.attrs:
                joint.stop_cfms = self.attrs['stop_cfms']
            if 'stop_erps' in self.attrs:
                joint.stop_erps = self.attrs['stop_erps']

        self.joints.append(joint)

        self.attrs = {}

    def visit_top(self, node, children):
        mass = sum(b.mass.mass for b in self.bodies)
        vol = sum(b.volume for b in self.bodies)
        logging.info('%.1f mass / %f volume = %.1f overall density',
                     mass, vol, mass / vol)


class AsfParser(object):
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
